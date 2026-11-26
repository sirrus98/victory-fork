//! Title memory as the card's own memory: a host heap opened on memory
//! the title already has, and a buffer placed in it, so that shaders read
//! and write the title's bytes themselves and no host copy stands between
//! (the console's one memory). This file holds the two openings Direct3D
//! has, a probe that says what this PC's driver makes of them
//! (`VICTORY_INPLACE_PROBE`), and a census of which title memory could be
//! opened.
//!
//! What the Direct3D runtime and the kernel answer, measured on this PC
//! with the software adapter (the rules are theirs, not the adapter's;
//! what the card's driver adds is what the probe is for):
//!
//! - `OpenExistingHeapFromAddress` takes the base address of a whole
//!   `VirtualAlloc` allocation and nothing inside it; every page must be
//!   committed and have one protection. Opening locks every page into the
//!   PC's memory at once (256 MiB never touched: 70 ms and 256 MB of
//!   working set).
//! - `OpenExistingHeapFromAddress1` (`ID3D12Device13`) takes any committed
//!   part of an allocation, with the same rule for its pages; parts may
//!   overlap, and the whole may be opened beside them. It also takes a
//!   committed view of a section.
//! - `OpenExistingHeapFromFileMapping` takes a section made with
//!   `SEC_COMMIT` and refuses one made with `SEC_RESERVE` (how `xmem`
//!   makes the title's page pool today). The heap is the section, not a
//!   view of it: views may be unmapped, mapped elsewhere or made
//!   read-only, and the section's handle closed, with the heap alive.
//!   Opening it locks nothing (256 MiB: 0.2 ms, no working set).
//! - The heap is a custom heap in the PC's memory (write-back pages, pool
//!   L0), shared and cross-adapter shared, 64 KiB aligned. A buffer in it
//!   must have `ALLOW_CROSS_ADAPTER`; unordered access beside it is taken.
//!   A texture must be row-major with one level. Tile mappings into such
//!   a heap are refused and REMOVE THE DEVICE: never call them.
//! - While a heap opened from an address stands, the title cannot
//!   decommit, release or protect the pages in it (`VirtualFree` and
//!   `VirtualProtect` fail with error 87), nor unmap a view in it. Pages
//!   outside the opened part are free to change.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};
use windows_sys::Win32::System::Memory as mem;

/// The unit heaps and placed buffers are cut on, and the console's page.
pub(crate) const K64: usize = 64 << 10;
/// The host's write watching (not in windows-sys' memory module).
pub(crate) const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// `VICTORY_INPLACE_PROBE`: 0 (unset) nothing; 1 the probe on memory of
/// its own at the first sweep (with Stage 0's small probes, P2-P6 and R9,
/// `crate::inplace_probes`, on a thread of its own), and the census 30 and
/// 150 seconds later; 2 also opens one part of the title's own memory for
/// a moment at the first census; 3 also runs the probes that lock or
/// commit gigabytes for a moment (P1, P2's 4 GiB, P7). The same probes
/// without the title: `victory inplace-probe`.
pub(crate) fn probe_level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| std::env::var("VICTORY_INPLACE_PROBE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// `VICTORY_INPLACE`: 0 (unset) off. 1: a window of title memory outside
/// buffers that lies in private memory committed whole is not copied: a
/// part of the title's allocation is opened as a host heap and the window
/// is a buffer placed over it ([`place_for`]). 2: title buffers too,
/// except the big ones in the card's memory. 3: as 2, but each is a view
/// into a shared arena (`crate::arena`: one heap per region of private
/// memory the runtime reported committed, parts of at most 4 GiB - 64 KiB
/// every 2 GiB; and one per committed section of physical pages, for view
/// memory that is one run of a section), not a heap of its own; the
/// budget is the arenas' bytes ([`crate::arena::most`]). 4: as 3, and the
/// big title buffers that got a copy in the card's memory are residents
/// where their memory lies in an arena (`crate::resident`: a buffer in the
/// card's memory fed from the arena and merged back into it by the card,
/// with no copying, hashing or copy-back on the CPU).
///
/// What stage 1 covers is private memory (`XMemVirtualAlloc`'s). View
/// memory (the page pool) is turned down here: its section is made
/// reserved, which the host refuses, and what a view shows changes when
/// the title maps pages.
///
/// What it assumes, for 2: nothing reads title memory that a submission
/// not yet run writes through such a buffer, except what the flights'
/// rules hold back. Shader writes in place are declared as for copies
/// (`Shadow::gpu_writes`; `host::in_place_writes` puts them in the flights'
/// `written`), so late wait and the relay hold back a CPU read of them (a
/// title buffer copy, a fill, a texture, a step of ours, a value wait)
/// until the flight has run, exactly as for a copy's declared ranges. That
/// makes 2 with `VICTORY_LATE_WAIT`/`VICTORY_RELAY` as safe as copies are:
/// the same assumption (shaders write only what is declared) and the same
/// shortcut (another host copy of the same bytes is not told). Before, the
/// writes were not declared and the two were not to be combined.
///
/// shortcut: while a window is in place the title cannot decommit, free
/// or protect its pages (the call fails with error 87; the probe shows
/// it). The title's heaps are allocated once and kept, and nothing of the
/// kind was seen; [`title_memory_freeing`] is there for the runtime to
/// call before such a call once it takes them over.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| std::env::var("VICTORY_INPLACE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// The most title memory in place at once, in bytes
/// (`VICTORY_INPLACE_MB`, 4096): opening locks the pages into the PC's
/// memory, also those the title never touched.
fn most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("VICTORY_INPLACE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4096) << 20)
}

/// Title buffers smaller than this many bytes are not put in place: they
/// stay copies, as without `VICTORY_INPLACE` (`VICTORY_INPLACE_MIN_KB`,
/// 1024 by default; 0: no floor). Windows are not affected.
///
/// Shaders reach memory in place across the bus. Three compute passes that
/// run once a frame went 1-6 ms -> 19-67 ms of card time with the tiny
/// buffers they write with atomics (52 B, 480 B, 56 KB) in place; kept as
/// copies, title lists took 100-129 ms of card time a frame against
/// 222-244 ms, and the frame 234.6 ms against 495 ms (the control: 240 ms).
/// Such buffers are 42 MB in all: copying them costs little.
pub(crate) fn min_bytes() -> u64 {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| min_from(std::env::var("VICTORY_INPLACE_MIN_KB").ok().as_deref()))
}

/// The floor in bytes for a `VICTORY_INPLACE_MIN_KB` setting in KiB.
fn min_from(kb: Option<&str>) -> u64 {
    kb.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1024).saturating_mul(1 << 10)
}

#[cfg(test)]
mod min_tests {
    use super::min_from;

    #[test]
    fn floor_defaults_to_one_mib_and_zero_turns_it_off() {
        assert_eq!(min_from(None), 1 << 20);
        assert_eq!(min_from(Some("0")), 0);
        assert_eq!(min_from(Some(" 64 ")), 64 << 10);
        assert_eq!(min_from(Some("nonsense")), 1 << 20);
    }
}

/// Title buffers kept as copies by [`min_bytes`], since the start: count, bytes.
pub(crate) static BELOW_MIN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// In place now: windows and their bytes, title buffers and their bytes;
/// then, since the start: asks turned down because the memory is a view
/// or not private memory committed whole, because the host refused,
/// because of [`most`]; and what was let go because its memory changed.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A buffer placed over title memory, and its heap.
pub(crate) struct Placed {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) heap: crate::arena::Heap,
    /// The host GPU address of the title memory's first byte.
    pub(crate) gpu: u64,
    /// Bytes from the buffer's first byte to the title memory's first.
    pub(crate) bias: u64,
    /// View memory in a section's arena: the section (`bias` is then the
    /// title memory's offset in it).
    pub(crate) section: Option<usize>,
}

/// Opens the 64 KiB pages that hold title memory `[start, start + len)`
/// as a heap of their own and places one buffer over all of them. `Err`
/// with nothing if the memory is not private memory committed whole
/// (nothing was asked of the host), with the host's error if it refused.
pub(crate) fn place_part(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, Option<windows::core::Error>> {
    let at = start & !(K64 as u64 - 1);
    let bytes = (start + len.max(1)).next_multiple_of(K64 as u64) - at;
    if !private_whole(at as usize, bytes as usize) {
        return Err(None);
    }
    let heap = open(device, From::Part(at as usize, bytes as usize)).map_err(Some)?;
    let buffer = place(device, &heap, 0, bytes).map_err(Some)?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    Ok(Placed { buffer, heap: heap.into(), gpu, bias: start - at, section: None })
}

/// Title memory `[start, start + len)` in place, for a window outside
/// buffers or a title buffer, if [`level`] asks for it and the memory can
/// be opened; `None` leaves the caller to make a copy, as before.
pub(crate) fn place_for(start: u64, len: u64, window: bool) -> Option<Placed> {
    if level() < if window { 1 } else { 2 } {
        return None;
    }
    if level() >= 3 {
        // A view into the arena that holds it (`crate::arena`): private
        // memory the runtime reported committed, or a run of a committed
        // section; no heap of its own.
        let Some(r) = crate::arena::resolve(start, len).filter(|r| r.left >= len) else {
            STATS[4].fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let kind = if r.kind == crate::arena::Kind::Section { 5 } else { 4 };
        crate::arena::ARENA_STATS[kind].fetch_add(len, Ordering::Relaxed);
        counted(window, len);
        return Some(Placed { buffer: r.resource, heap: r.heap, gpu: r.gpu, bias: r.offset, section: r.section });
    }
    // View memory: known from the runtime without asking Windows.
    if cover(start as usize, len as usize) != Cover::None {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if STATS[1].load(Ordering::Relaxed) + STATS[3].load(Ordering::Relaxed) + len > most() {
        STATS[6].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let host = crate::host::host()?;
    match place_part(&host.device, start, len) {
        Ok(placed) => {
            counted(window, len);
            Some(placed)
        }
        Err(None) => {
            STATS[4].fetch_add(1, Ordering::Relaxed);
            None
        }
        Err(Some(e)) => {
            if STATS[5].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(start = format_args!("{start:#x}"), len, error = %e, "title memory not opened in place: a copy instead");
            }
            None
        }
    }
}

/// Things in place let go because their memory changed, counted.
pub(crate) fn let_go_counted(n: u64) {
    STATS[7].fetch_add(n, Ordering::Relaxed);
}

/// The bytes of a section the runtime made committed, if it did.
pub(crate) fn section_bytes(section: usize) -> Option<u64> {
    views().sections.iter().find(|s| s.0 == section && s.2).map(|s| s.1)
}

/// What is in place now counts one more (undone by [`gone`]).
pub(crate) fn counted(window: bool, bytes: u64) {
    let at = if window { 0 } else { 2 };
    STATS[at].fetch_add(1, Ordering::Relaxed);
    STATS[at + 1].fetch_add(bytes, Ordering::Relaxed);
}

/// What [`place_for`] gave is gone.
pub(crate) fn gone(window: bool, bytes: u64) {
    let at = if window { 0 } else { 2 };
    STATS[at].fetch_sub(1, Ordering::Relaxed);
    STATS[at + 1].fetch_sub(bytes, Ordering::Relaxed);
}

/// Title memory `[at, at + len)` is about to show something else, or the
/// title is about to decommit, free or protect it: what is in place there
/// is let go (the tables forget it; its heap goes when the last list and
/// slot that hold it do). Called for memory the runtime maps anew; for
/// the runtime to call before the title's own `VirtualFree` once it takes
/// that call over.
pub fn title_memory_freeing(at: usize, len: usize) {
    if STATS[0].load(Ordering::Relaxed) + STATS[2].load(Ordering::Relaxed) == 0 {
        return;
    }
    let gone = crate::shadow::in_place_let_go(at as u64, len as u64, true);
    STATS[7].fetch_add(gone, Ordering::Relaxed);
}

/// Whether `[at, at + len)` is still one run of `section` from `offset`
/// (a window in place there stays good across the title's remapping).
pub(crate) fn still_run(at: u64, len: u64, section: usize, offset: u64) -> bool {
    cover(at as usize, len.max(1) as usize) == Cover::Run(section, offset)
}

/// The line of what is in place, every ten seconds while [`level`] is on.
fn say() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 10_000 || LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    if level() >= 3 {
        let (parts, sections, locked) = crate::arena::census();
        let a: Vec<u64> = crate::arena::ARENA_STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
        tracing::info!(
            "title memory in place, arenas: {parts} parts of private memory and {sections} sections open, {:.2} GiB locked (budget {:.1} GiB); \
             since the start {} opened ({} MB), {} refused, {} retired, {} ranges resolved; buffers and windows placed in them: {} MB of private memory, {} MB of view memory; turned down: {}; \
             remapped under a placement in a section: {} kept (still the same run), {} let go and placed again",
            locked as f64 / f64::from(1u32 << 30),
            crate::arena::most() as f64 / f64::from(1u32 << 30),
            a[0],
            a[3] >> 20,
            a[1],
            a[2],
            a[6],
            a[4] >> 20,
            a[5] >> 20,
            crate::arena::turned_down(),
            crate::shadow::REMAPS[0].load(Ordering::Relaxed),
            crate::shadow::REMAPS[1].load(Ordering::Relaxed)
        );
    }
    tracing::info!(
        "title memory in place: {} windows {} MB, {} title buffers {} MB (no copy, no refresh, no copy-back); turned down: {} view memory or not private memory committed whole, \
         {} refused by the host, {} over VICTORY_INPLACE_MB; {} let go because their memory changed",
        s[0],
        s[1] >> 20,
        s[2],
        s[3] >> 20,
        s[4],
        s[5],
        s[6],
        s[7]
    );
}

/// What a heap is opened on.
pub(crate) enum From {
    /// A whole `VirtualAlloc` allocation, by its base address.
    Region(usize),
    /// A committed part of one: address and bytes.
    Part(usize, usize),
    /// A section made with `SEC_COMMIT`, by its handle. Refused unless
    /// `VICTORY_ALLOW_SECTION_HEAPS=1` ([`section_heaps_allowed`]).
    Section(usize),
}

/// Whether a heap may be opened on a section by its handle
/// (`OpenExistingHeapFromFileMapping`): only with
/// `VICTORY_ALLOW_SECTION_HEAPS=1`. A section opened so keeps its commit
/// until the PC restarts, however the heap and the section are let go, on
/// the card and on the software adapter alike
/// (`tests::section_heap_commit_comes_back`); a view of it opened by its
/// address does not ([`From::Part`], what the arenas do).
pub(crate) fn section_heaps_allowed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALLOW_SECTION_HEAPS").is_some_and(|v| v == "1"))
}

/// Opens a heap on memory the process has.
pub(crate) fn open(device: &ID3D12Device, from: From) -> windows::core::Result<ID3D12Heap> {
    if matches!(from, From::Section(_)) && !section_heaps_allowed() {
        return Err(windows::core::Error::new(
            windows::Win32::Foundation::E_ACCESSDENIED,
            "not opened: a section opened as a heap by its handle keeps its commit until the PC restarts (VICTORY_ALLOW_SECTION_HEAPS=1 to try)",
        ));
    }
    // SAFETY: the calls take an address or a handle and check both
    // themselves; a refusal is an error code.
    unsafe {
        match from {
            From::Region(at) => device.cast::<ID3D12Device3>()?.OpenExistingHeapFromAddress(at as *const c_void),
            From::Part(at, bytes) => device.cast::<ID3D12Device13>()?.OpenExistingHeapFromAddress1(at as *const c_void, bytes),
            From::Section(handle) => device.cast::<ID3D12Device3>()?.OpenExistingHeapFromFileMapping(HANDLE(handle as *mut c_void)),
        }
    }
}

pub(crate) fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// A buffer of `size` bytes at `offset` (a multiple of 64 KiB) of an
/// opened heap, which shaders may read and write, in the common state.
pub(crate) fn place(device: &ID3D12Device, heap: &ID3D12Heap, offset: u64, size: u64) -> windows::core::Result<ID3D12Resource> {
    let mut placed: Option<ID3D12Resource> = None;
    // The heap is cross-adapter shared, so the buffer must say so too.
    let flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    // SAFETY: plain creation in a live heap.
    unsafe { device.CreatePlacedResource(heap, offset, &buffer_desc(size, flags), D3D12_RESOURCE_STATE_COMMON, None, &mut placed)? };
    placed.ok_or_else(windows::core::Error::empty)
}

// The title's views of its physical pages, as the runtime maps them: the
// census needs them to say which view memory is one run of a section (one
// placed buffer could show it) and which is scattered pages.

struct Views {
    /// By address: bytes, the section's handle, the offset in it.
    views: BTreeMap<usize, (usize, usize, u64)>,
    /// By section and byte offset of a 64 KiB page: the address of that
    /// page in every view that shows it. A write through one view of a
    /// page is a write of the bytes every other view shows
    /// (`crate::watch` guards them all).
    pages: BTreeMap<(usize, u64), Vec<usize>>,
    /// Sections by handle: bytes, and whether made committed.
    sections: Vec<(usize, u64, bool)>,
}

static VIEWS: Mutex<Views> =
    Mutex::new(Views { views: BTreeMap::new(), pages: BTreeMap::new(), sections: Vec::new() });

fn views() -> std::sync::MutexGuard<'static, Views> {
    VIEWS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The runtime made a section of `bytes` for physical pages; `committed`
/// if with `SEC_COMMIT` (only such a section can be opened as a heap).
pub fn title_memory_section(section: usize, bytes: u64, committed: bool) {
    {
        let mut v = views();
        v.sections.retain(|s| s.0 != section);
        v.sections.push((section, bytes, committed));
    }
    // A view of our own made on an earlier section that had this handle's
    // number shows that section's pages, not this one's: it is forgotten
    // (left mapped: nothing reads it any more) and the next store through
    // a view of ours maps this section.
    let gone = OWN.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|own| own.remove(&section)).flatten();
    if gone.is_some() {
        tracing::info!(section, "a section handle's number came round again: our own view of the earlier section is forgotten");
    }
}

/// The runtime mapped `[at, at + len)` as a view of `section` from
/// `offset` on. Call after [`crate::title_memory_remapped`] of the range.
pub fn title_memory_view(at: usize, len: usize, section: usize, offset: u64) {
    {
        let mut v = views();
        v.views.insert(at, (len, section, offset));
        for k in 0..len.div_ceil(K64) {
            v.pages.entry((section, offset + (k * K64) as u64)).or_default().push(at + k * K64);
        }
    }
    // This view is writable from now on: a guard over the same bytes at
    // another view must look at them again, and the pages are noted written
    // there (`crate::watch`).
    crate::watch::views_mapped(section, offset, len);
}

/// What `[at, at + len)` showed is gone ([`crate::title_memory_remapped`]):
/// the views that touch it are forgotten (the runtime tells again those it
/// maps anew).
pub(crate) fn views_gone(at: usize, len: usize) {
    let mut v = views();
    if v.views.is_empty() {
        return;
    }
    let end = at.saturating_add(len);
    let first = v.views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at).map(|(base, _)| *base);
    let gone: Vec<usize> = first.into_iter().chain(v.views.range(at.saturating_add(1)..end).map(|(base, _)| *base)).collect();
    for base in gone {
        let Some((bytes, section, offset)) = v.views.remove(&base) else {
            continue;
        };
        for k in 0..bytes.div_ceil(K64) {
            let key = (section, offset + (k * K64) as u64);
            if let Some(pages) = v.pages.get_mut(&key) {
                let page = base + k * K64;
                pages.retain(|&shown| shown != page);
                if pages.is_empty() {
                    v.pages.remove(&key);
                }
            }
        }
    }
}

/// The title views that overlap `[start, end)`, as `(address, bytes)`.
pub(crate) fn views_between(start: usize, end: usize) -> Vec<(usize, usize)> {
    if start >= end {
        return Vec::new();
    }
    let v = views();
    let mut out = Vec::new();
    if let Some((&base, &(bytes, _, _))) = v.views.range(..=start).next_back()
        && base + bytes > start
    {
        out.push((base, bytes));
    }
    for (&base, &(bytes, _, _)) in v.views.range(start.saturating_add(1)..end) {
        out.push((base, bytes));
    }
    out
}

/// The address of the 64 KiB page shown at `page` in every title view that
/// shows it, `page`'s own address included; empty if `page` is not a view.
pub(crate) fn pages_of(page: usize) -> Vec<usize> {
    let v = views();
    let Some((&base, &(bytes, section, offset))) = v.views.range(..=page).next_back() else {
        return Vec::new();
    };
    let within = page.wrapping_sub(base);
    if within >= bytes || !within.is_multiple_of(K64) {
        return Vec::new();
    }
    v.pages.get(&(section, offset + within as u64)).cloned().unwrap_or_default()
}

/// The title views that show any of the physical bytes `[offset, offset +
/// len)` of `section`, as `(address, bytes)` ranges in address order; not
/// those of a page shown in very many views (`MANY_VIEWS`).
pub(crate) fn views_of_section(section: usize, offset: u64, len: usize) -> Vec<(usize, usize)> {
    if len == 0 {
        return Vec::new();
    }
    let v = views();
    let end = offset.saturating_add(len as u64);
    let mut pages: Vec<usize> = Vec::new();
    let mut at = offset - offset % K64 as u64;
    while at < end {
        if let Some(shown) = v.pages.get(&(section, at))
            && shown.len() <= crate::watch::MANY_VIEWS
        {
            pages.extend(shown.iter().copied());
        }
        match at.checked_add(K64 as u64) {
            Some(next) => at = next,
            None => break,
        }
    }
    pages.sort_unstable();
    pages.dedup();
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(pages.len());
    for page in pages {
        match runs.last_mut() {
            Some((first, bytes)) if *first + *bytes == page => *bytes += K64,
            _ => runs.push((page, K64)),
        }
    }
    runs
}

/// Whether what shaders wrote goes back into title view memory through a
/// view of our own (`VICTORY_BACK_OWN_VIEW=1`; off by default).
///
/// The copy-back writes title memory at the title's address, so it opens
/// the guard on the 64 KiB block first (and on every other view of the
/// page), and the block then counts as written by the title: the host
/// copy that just put the bytes there copies all 64 KiB of it back in at
/// its next refresh, and sends them up to the card again if it lives
/// there. In runs 218-222 the bytes copied in went with the bytes copied
/// back, 0.83 MB for each MB (78 readings), some 50 MB of 142 MB a frame.
///
/// With the switch the bytes are stored through a view of our own on the
/// same section, which no guard covers: the title's views stay read-only
/// (a title write still faults and is seen), the write-watch journal is
/// told the page was written (`crate::watch::wrote_own`) so that every
/// other copy of it takes it again, and the copy that wrote it notes the
/// page current (`put_back_own` in `crate::shadow`).
///
/// `VICTORY_BACK_OWN_VIEW=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log, for measuring it in one scene.
pub(crate) fn back_own() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_BACK_OWN_VIEW").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "copy-back through views of our own from now (VICTORY_BACK_OWN_VIEW=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// A view of our own on a whole section of the title's physical pages:
/// where it is, its bytes, and which of its 64 KiB blocks we have asked
/// to be committed through it (a bit each).
struct Own {
    base: usize,
    bytes: u64,
    committed: Vec<u64>,
}

/// Our own views by section handle; `None` in a section's place if the
/// host refused the view (not asked again).
static OWN: Mutex<Option<std::collections::HashMap<usize, Option<Own>>>> = Mutex::new(None);

/// Stores `bytes` at title address `at` through a view of our own
/// ([`back_own`]), if `[at, at + bytes.len())` lies in one 64 KiB page of
/// a title view of a section. Gives the addresses of that page in every
/// title view that shows it (the page of `at` among them), or `None` with
/// nothing written: not view memory, a page of very many views, the
/// runtime changing mappings at this moment, or the host refusing.
pub(crate) fn write_own(at: usize, bytes: &[u8]) -> Option<Vec<usize>> {
    if bytes.is_empty() || at / K64 != (at + bytes.len() - 1) / K64 {
        return None;
    }
    // No mapping changes until the bytes are in: the view found below
    // shows these pages for as long as this is held.
    let _still = crate::watch::title_memory_reading_now()?;
    let (section, offset, shown, section_bytes) = {
        let v = views();
        let (&base, &(len, section, offset)) = v.views.range(..=at).next_back()?;
        if at + bytes.len() > base + len {
            return None;
        }
        let offset = offset + (at - base) as u64;
        let shown = v.pages.get(&(section, offset - offset % K64 as u64))?.clone();
        if shown.len() > crate::watch::MANY_VIEWS {
            return None;
        }
        let section_bytes = v.sections.iter().find(|s| s.0 == section)?.1;
        (section, offset, shown, section_bytes)
    };
    if offset + bytes.len() as u64 > section_bytes {
        return None;
    }
    let mut own = OWN.lock().unwrap_or_else(|e| e.into_inner());
    let view = own
        .get_or_insert_with(Default::default)
        .entry(section)
        .or_insert_with(|| {
            // SAFETY: a section handle the runtime gave us, alive as long
            // as the title's views of it are; the view is ours and stays.
            let at = unsafe { mem::MapViewOfFile(section as *mut c_void, mem::FILE_MAP_WRITE, 0, 0, 0) };
            if at.Value.is_null() {
                tracing::warn!(
                    section,
                    error = %std::io::Error::last_os_error(),
                    "no view of our own on a section of title pages: what shaders wrote goes back the old way there"
                );
                return None;
            }
            tracing::info!(
                section,
                mb = section_bytes >> 20,
                "a view of our own on a section of title pages, for what shaders wrote (VICTORY_BACK_OWN_VIEW=1)"
            );
            Some(Own {
                base: at.Value as usize,
                bytes: section_bytes,
                committed: vec![0; (section_bytes as usize).div_ceil(K64).div_ceil(64)],
            })
        })
        .as_mut()?;
    if offset + bytes.len() as u64 > view.bytes {
        return None;
    }
    let block = offset as usize / K64;
    if (view.committed[block / 64] & (1u64 << (block % 64))) == 0 {
        // The title's view committed these pages of the section; asked
        // once more through ours, which changes nothing if they are.
        // SAFETY: a block inside our own view.
        let there = unsafe {
            mem::VirtualAlloc((view.base + block * K64) as *const c_void, K64, mem::MEM_COMMIT, mem::PAGE_READWRITE)
        };
        if there.is_null() {
            return None;
        }
        view.committed[block / 64] |= 1u64 << (block % 64);
    }
    // SAFETY: committed bytes inside our own view; the source is the
    // caller's slice.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), (view.base + offset as usize) as *mut u8, bytes.len()) };
    Some(shown)
}

/// How the views cover `[at, at + len)`.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Cover {
    /// No view holds its first byte.
    None,
    /// Views hold a part of it only.
    Holes,
    /// Views hold all of it, from more than one place.
    Scattered,
    /// All of it is one run of one section: the section and the offset of
    /// `at` in it.
    Run(usize, u64),
}

fn cover_in(views: &BTreeMap<usize, (usize, usize, u64)>, at: usize, len: usize) -> Cover {
    let end = at + len.max(1);
    let Some((&base, &(bytes, section, offset))) = views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at) else {
        return Cover::None;
    };
    let first = (section, offset + (at - base) as u64);
    let (mut reached, mut one) = (base + bytes, true);
    while reached < end {
        let Some(&(bytes, s, o)) = views.get(&reached) else {
            return Cover::Holes;
        };
        one &= s == first.0 && o == first.1 + (reached - at) as u64;
        reached += bytes;
    }
    if one { Cover::Run(first.0, first.1) } else { Cover::Scattered }
}

/// Where the run of one section that holds `at` ends, at most at
/// `most`: views that continue each other in the section are one run.
/// `None` if no view holds `at`.
fn run_end_in(views: &BTreeMap<usize, (usize, usize, u64)>, at: usize, most: usize) -> Option<usize> {
    let (&base, &(bytes, section, offset)) = views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at)?;
    let mut reached = base + bytes;
    while reached < most {
        match views.get(&reached) {
            Some(&(n, s, o)) if s == section && o == offset + (reached - base) as u64 => reached += n,
            _ => break,
        }
    }
    Some(reached.min(most))
}

/// For a window outside buffers over view memory (`VICTORY_INPLACE=3`):
/// the range `[start, start + len)` of whole 64 KiB pages around the ask
/// `[at, end)` that is one run of one section, if the ask lies in one;
/// for an ask without a length (it reads to the window's end), the run
/// from the ask's page to where the run ends in its 4 MiB cell.
pub(crate) fn run_window(at: u64, end: u64, unsized_ask: bool) -> Option<(u64, u64)> {
    const CELL: u64 = 4 << 20;
    let start = at & !(K64 as u64 - 1);
    let most = if unsized_ask { (at & !(CELL - 1)) + CELL } else { end.next_multiple_of(K64 as u64) };
    let reached = run_end_in(&views().views, start as usize, most as usize)? as u64;
    (reached >= end.max(at + 1)).then_some((start, reached - start))
}

/// [`Cover`] of `[at, at + len)` by the title's views as they are now.
pub(crate) fn cover(at: usize, len: usize) -> Cover {
    cover_in(&views().views, at, len)
}

/// Whether `[at, at + len)` is private memory, all committed and
/// read-write: what [`From::Part`] takes. One question to Windows (it
/// walks the page tables to the end of the pages that share the first
/// one's state: a millisecond in the title's big blocks, so ask when a
/// copy is made, not when it is used).
pub(crate) fn private_whole(at: usize, len: usize) -> bool {
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == mem::MEM_COMMIT
        && info.Type == mem::MEM_PRIVATE
        && info.Protect == mem::PAGE_READWRITE
        && (info.BaseAddress as usize).saturating_add(info.RegionSize) >= at.saturating_add(len)
}

pub(crate) fn hr<T>(result: &windows::core::Result<T>) -> String {
    match result {
        Ok(_) => "ok".to_owned(),
        Err(e) => format!("REFUSED {:#010x} ({})", e.code().0, e.message().trim()),
    }
}

/// A queue, a list and a fence of the probe's own on `device`: nothing of
/// the host's submissions is touched.
pub(crate) struct Rig {
    pub(crate) device: ID3D12Device,
    pub(crate) queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: u64,
    event: HANDLE,
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

impl Rig {
    /// Why the device was removed, if it was.
    pub(crate) fn removed(&self) -> Option<String> {
        // SAFETY: a plain query.
        let reason = unsafe { self.device.GetDeviceRemovedReason() };
        reason.err().map(|e| format!("{:#010x} {}", e.code().0, e.message().trim()))
    }

    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Rig { device: device.clone(), queue, allocator, list, fence, done: 0, event })
        }
    }

    /// Records with `record`, runs it and waits (ten seconds at most).
    pub(crate) fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> windows::core::Result<()> {
        // SAFETY: our own list and queue, one run at a time.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            record(&self.list);
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.done += 1;
            self.queue.Signal(&self.fence, self.done)?;
            self.fence.SetEventOnCompletion(self.done, self.event)?;
            if WaitForSingleObject(self.event, 10_000).0 != 0 {
                return Err(windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL));
            }
            self.device.GetDeviceRemovedReason()
        }
    }

    fn committed(&self, kind: D3D12_HEAP_TYPE, size: u64, state: D3D12_RESOURCE_STATES) -> windows::core::Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            self.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE), state, None, &mut buffer)?
        };
        buffer.ok_or_else(windows::core::Error::empty)
    }

    /// What the GPU reads of `size` bytes of `resource` from `offset`.
    pub(crate) fn fetch(&mut self, resource: &ID3D12Resource, offset: u64, size: usize) -> windows::core::Result<Vec<u8>> {
        let back = self.committed(D3D12_HEAP_TYPE_READBACK, size as u64, D3D12_RESOURCE_STATE_COPY_DEST)?;
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(&back, 0, resource, offset, size as u64) })?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds `size` bytes until unmapped.
        unsafe {
            back.Map(0, None, Some(&mut data))?;
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size).to_vec();
            back.Unmap(0, None);
            Ok(bytes)
        }
    }

    /// The GPU writes `bytes` at `offset` of `resource` (a copy).
    pub(crate) fn store(&mut self, resource: &ID3D12Resource, offset: u64, bytes: &[u8]) -> windows::core::Result<()> {
        let up = self.committed(D3D12_HEAP_TYPE_UPLOAD, bytes.len() as u64, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds the buffer's bytes until unmapped.
        unsafe {
            up.Map(0, None, Some(&mut data))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(resource, offset, &up, 0, bytes.len() as u64) })
    }

    /// A compute shader stores, through a root unordered-access view at
    /// `gpu`, the sum of the buffer's first two words in its third.
    pub(crate) fn shade(&mut self, gpu: u64) -> windows::core::Result<()> {
        const SOURCE: &str = "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n[RootSignature(RS)]\n[numthreads(1, 1, 1)]\nvoid main() { b.Store(8, b.Load(0) + b.Load(4)); }\n";
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; then the
        // blob's own bytes, which hold the root signature too.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("inplace"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)?;
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            (root, pso)
        };
        // SAFETY: an open list of ours; the view names a live buffer.
        self.run(|list| unsafe {
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRootUnorderedAccessView(0, gpu);
            list.Dispatch(1, 1, 1);
        })
    }

    /// Milliseconds the card takes to copy `bytes` of `from` into its own
    /// memory, the best of three.
    fn copy_ms(&mut self, from: &ID3D12Resource, bytes: u64) -> windows::core::Result<f64> {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut to: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            self.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, None, &mut to)?
        };
        let to = to.ok_or_else(windows::core::Error::empty)?;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            // SAFETY: two live buffers of `bytes` bytes or more.
            self.run(|list| unsafe { list.CopyBufferRegion(&to, 0, from, 0, bytes) })?;
            best = best.min(started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(best)
    }
}

pub(crate) fn fill(at: *mut u8, len: usize, seed: u8) {
    for i in 0..len {
        // SAFETY: the caller's own committed memory of `len` bytes.
        unsafe { at.add(i).write((i as u8).wrapping_mul(31).wrapping_add(seed)) };
    }
}

pub(crate) fn same(at: *const u8, bytes: &[u8]) -> bool {
    // SAFETY: the caller's own committed memory of that length.
    unsafe { std::slice::from_raw_parts(at, bytes.len()) == bytes }
}

/// The process's working set in MB.
pub(crate) fn working_set_mb() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    let mut counters = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut counters, counters.cb) };
    counters.WorkingSetSize as u64 >> 20
}

/// What the probe found. The lines are the whole account; the flags are
/// what the stages build on.
#[derive(Default)]
pub(crate) struct Probe {
    pub(crate) lines: Vec<String>,
    /// A part of private, write-watched memory opens, a buffer in it shows
    /// the memory's bytes to the GPU, also those the CPU wrote afterwards.
    pub(crate) part_reads: bool,
    /// What the GPU writes there (a copy, and a shader through unordered
    /// access) the CPU reads at the memory's address.
    pub(crate) part_writes: bool,
    /// The same two for a committed section, through a view of it.
    pub(crate) section_reads: bool,
    pub(crate) section_writes: bool,
    /// The section's heap outlives the view: the GPU's write with no view
    /// mapped shows in a view mapped afterwards.
    pub(crate) section_outlives_views: bool,
    /// A section made with `SEC_RESERVE` (as `xmem` makes the page pool)
    /// was refused.
    pub(crate) reserved_section_refused: bool,
}

/// Tries the openings on memory of the probe's own, made the way `xmem`
/// makes title memory, with a queue of its own on `device`; `memory`
/// gives the card's memory account (`Host::video_memory`) where there is
/// one. Nothing of the title's is touched. `mb` is the size of the cost
/// measurement (0 leaves it out).
#[cfg(test)]
pub(crate) fn probe(device: &ID3D12Device, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize) -> Probe {
    probe_with(device, memory, mb, false, false)
}

/// [`probe`], then Stage 0's probes (`crate::inplace_probes`): with `big`
/// also those that lock or commit gigabytes, with `tiles` the tile mapping
/// that removes the device (only on a device of the probe's own).
pub(crate) fn probe_with(device: &ID3D12Device, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize, big: bool, tiles: bool) -> Probe {
    let mut p = probe_first(device, memory, mb);
    if let Ok(mut rig) = Rig::new(device) {
        crate::inplace_probes::stage0(&mut p.lines, &mut rig, big, tiles);
    }
    p
}

fn probe_first(device: &ID3D12Device, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize) -> Probe {
    let mut p = Probe::default();
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => {
            p.lines.push(format!("no queue of its own: {e}"));
            return p;
        }
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: a plain query that fills our structure.
    let _ = unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, (&raw mut options).cast(), size_of_val(&options) as u32) };
    p.lines.push(format!(
        "the device: ID3D12Device3 {}, ID3D12Device13 {} (parts of an allocation); cross-adapter row-major textures {}; heap tier {}",
        device.cast::<ID3D12Device3>().is_ok(),
        device.cast::<ID3D12Device13>().is_ok(),
        options.CrossAdapterRowMajorTextureSupported.as_bool(),
        options.ResourceHeapTier.0
    ));
    // SAFETY: memory of the probe's own, made and released here; every
    // address used lies inside it.
    unsafe {
        private(&mut p, &mut rig);
        section(&mut p, &mut rig);
        if mb > 0 {
            cost(&mut p, &mut rig, memory, mb);
        }
    }
    p.lines.push(format!(
        "VERDICT: private memory in place: reads {}, writes {}; the page pool's section in place: reads {}, writes {}, whatever the views do {}; \
         a reserved section is refused {} (the pool must be made committed to be opened)",
        p.part_reads, p.part_writes, p.section_reads, p.section_writes, p.section_outlives_views, p.reserved_section_refused
    ));
    p
}

fn describe(heap: &ID3D12Heap) -> String {
    // SAFETY: a plain query.
    let d = unsafe { heap.GetDesc() };
    format!(
        "heap of {} KiB: type {} (4 = custom), CPU pages {} (3 = write-back), pool {} (1 = the PC's memory), alignment {}, flags {:#x} (0x1 shared, 0x20 cross-adapter, 0x200 write watch, 0x400 shader atomics)",
        d.SizeInBytes >> 10,
        d.Properties.Type.0,
        d.Properties.CPUPageProperty.0,
        d.Properties.MemoryPoolPreference.0,
        d.Alignment,
        d.Flags.0
    )
}

/// Private memory, reserved as `xmem` reserves the title's heaps
/// (committed, read-write, write-watched): a part of it opened.
unsafe fn private(p: &mut Probe, rig: &mut Rig) {
    const SIZE: usize = 64 * K64;
    // SAFETY (the whole function): the allocation below, `SIZE` bytes.
    unsafe {
        let base = mem::VirtualAlloc(std::ptr::null(), SIZE, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE).cast::<u8>();
        if base.is_null() {
            p.lines.push("private memory: none to try with".to_owned());
            return;
        }
        // The part: 16 pages from the 16th on.
        let (at, bytes) = (base.add(16 * K64), 16 * K64);
        let whole = open(&rig.device, From::Region(base as usize));
        p.lines.push(format!("private memory, the whole allocation (OpenExistingHeapFromAddress): {}", hr(&whole)));
        drop(whole);
        let opened = open(&rig.device, From::Part(at as usize, bytes));
        p.lines.push(format!("private memory, a part of it (OpenExistingHeapFromAddress1): {}", hr(&opened)));
        if let Ok(heap) = &opened {
            p.lines.push(format!("  {}", describe(heap)));
            let placed = place(&rig.device, heap, K64 as u64, 4 * K64 as u64);
            p.lines.push(format!("  a buffer placed in it (cross-adapter, unordered access): {}", hr(&placed)));
            if let Ok(buffer) = &placed {
                let memory = at.add(K64);
                fill(memory, 4 * K64, 1);
                let first = rig.fetch(buffer, 0, 4 * K64).is_ok_and(|got| same(memory, &got));
                fill(memory, 4 * K64, 77);
                let second = rig.fetch(buffer, 0, 4 * K64).is_ok_and(|got| same(memory, &got));
                p.part_reads = first && second;
                p.lines.push(format!("  the GPU reads the memory's bytes: {first}; and those the CPU wrote afterwards, with no call in between: {second}"));
                mem::ResetWriteWatch(base.cast(), SIZE);
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x5a).collect();
                let copied = rig.store(buffer, 4096, &new).is_ok() && same(memory.add(4096), &new);
                let (mut found, mut count, mut granularity) = ([0usize; 64], 64usize, 0u32);
                let watched = mem::GetWriteWatch(0, base.cast(), SIZE, found.as_mut_ptr().cast(), &mut count, &mut granularity) == 0;
                memory.cast::<[u32; 3]>().write([40, 2, 0]);
                let shaded = rig.shade(buffer.GetGPUVirtualAddress());
                let shader = shaded.is_ok() && memory.cast::<[u32; 3]>().read()[2] == 42;
                p.part_writes = copied && shader;
                p.lines.push(format!(
                    "  the CPU reads at the memory's address what the GPU wrote: by a copy {copied}; by a shader through unordered access {shader} ({}); \
                     write watching saw {} of the copy's 16 pages{}",
                    hr(&shaded),
                    if watched { count } else { 0 },
                    if watched { "" } else { " (the asking failed)" }
                ));
            }
            // What the title can still do with the pages while the heap stands.
            let mut old = 0;
            let protect_in = mem::VirtualProtect(at.add(8 * K64).cast(), K64, mem::PAGE_READONLY, &mut old) != 0;
            if protect_in {
                mem::VirtualProtect(at.add(8 * K64).cast(), K64, mem::PAGE_READWRITE, &mut old);
            }
            let decommit_in = mem::VirtualFree(at.add(9 * K64).cast(), K64, mem::MEM_DECOMMIT) != 0;
            let decommit_out = mem::VirtualFree(base.add(40 * K64).cast(), K64, mem::MEM_DECOMMIT) != 0;
            p.lines.push(format!(
                "  with the heap standing, the title's own calls: read-only inside the part {protect_in}, decommit inside {decommit_in}, decommit outside {decommit_out} \
                 (false inside means the heap must go before the title frees or guards those pages)"
            ));
            drop(placed);
        }
        drop(opened);
        let freed = mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE) != 0;
        p.lines.push(format!("  released after the heap went: {freed}"));
    }
}

/// A section and a view of it in a placeholder, as `xmem` maps physical
/// pages: reserved (as it is made today) and committed.
unsafe fn section(p: &mut Probe, rig: &mut Rig) {
    const SIZE: usize = 32 * K64;
    // SAFETY (the whole function): the sections, placeholder and views
    // made below.
    unsafe {
        let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
        let invalid = windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        let reserved = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_RESERVE, 0, SIZE as u32, std::ptr::null());
        if !reserved.is_null() {
            // As the arenas open a section: a view of all of it, by its
            // address.
            let whole = mem::MapViewOfFile(reserved, mem::FILE_MAP_WRITE, 0, 0, SIZE);
            let opened = if whole.Value.is_null() { Err(windows::core::Error::from_thread()) } else { open(&rig.device, From::Part(whole.Value as usize, SIZE)) };
            p.reserved_section_refused = opened.is_err();
            p.lines.push(format!("a section made reserved, as the page pool is without VICTORY_INPLACE_POOL (a view of it, by its address): {}", hr(&opened)));
            drop(opened);
            if !whole.Value.is_null() {
                mem::UnmapViewOfFile(whole);
            }
            windows_sys::Win32::Foundation::CloseHandle(reserved);
        }
        let handle = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, SIZE as u32, std::ptr::null());
        let hole = mem::VirtualAlloc2(process, std::ptr::null(), SIZE, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0).cast::<u8>();
        if handle.is_null() || hole.is_null() {
            p.lines.push("a committed section: none to try with".to_owned());
            return;
        }
        // Pages 4..12 of the section at the placeholder's start.
        mem::VirtualFree(hole.cast(), 8 * K64, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
        let view = mem::MapViewOfFile3(handle, process, hole.cast(), (4 * K64) as u64, 8 * K64, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
        // As the arenas open a section: a view of all of it of their own, by
        // its address; the title's view below shows the same pages.
        let whole = mem::MapViewOfFile(handle, mem::FILE_MAP_WRITE, 0, 0, SIZE);
        let opened = if whole.Value.is_null() { Err(windows::core::Error::from_thread()) } else { open(&rig.device, From::Part(whole.Value as usize, SIZE)) };
        p.lines.push(format!("a section made committed (a view of all of it, by its address): {}", hr(&opened)));
        let mut unmapped = false;
        if let (Ok(heap), false) = (&opened, view.Value.is_null()) {
            p.lines.push(format!("  {}", describe(heap)));
            // The section's page 5 is the view's second page.
            let placed = place(&rig.device, heap, 5 * K64 as u64, 2 * K64 as u64);
            p.lines.push(format!("  a buffer placed at the section's offset of a view's pages: {}", hr(&placed)));
            if let Ok(buffer) = &placed {
                let memory = hole.add(K64);
                fill(memory, 2 * K64, 3);
                p.section_reads = rig.fetch(buffer, 0, 2 * K64).is_ok_and(|got| same(memory, &got));
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x33).collect();
                p.section_writes = rig.store(buffer, 0, &new).is_ok() && same(memory, &new);
                // The guard of `watch.rs`: the view read-only.
                let mut old = 0;
                let guarded = mem::VirtualProtect(memory.cast(), K64, mem::PAGE_READONLY, &mut old) != 0;
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x44).collect();
                let through = rig.store(buffer, 0, &new).is_ok() && same(memory, &new);
                // The view goes; the GPU writes; another view shows it.
                unmapped = mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER) != 0;
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x55).collect();
                let stored = rig.store(buffer, 0, &new).is_ok();
                let other = mem::MapViewOfFile(handle, mem::FILE_MAP_WRITE, 0, (5 * K64) as u32, K64);
                p.section_outlives_views = unmapped && stored && !other.Value.is_null() && same(other.Value.cast(), &new);
                p.lines.push(format!(
                    "  the GPU reads the view's bytes {}; the CPU reads through the view what the GPU wrote {}; the view made read-only under the heap {guarded}, \
                     and the GPU writes it all the same {through}; the view unmapped under the heap {unmapped}, the GPU's write then shows in a view mapped afterwards {}",
                    p.section_reads, p.section_writes, p.section_outlives_views
                ));
                if !other.Value.is_null() {
                    mem::UnmapViewOfFile(other);
                }
            }
        }
        drop(opened);
        if !unmapped && !view.Value.is_null() {
            mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER);
        }
        if !whole.Value.is_null() {
            mem::UnmapViewOfFile(whole);
        }
        windows_sys::Win32::Foundation::CloseHandle(handle);
        // The placeholder was split in two: each half is released.
        mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
        mem::VirtualFree(hole.add(8 * K64).cast(), 0, mem::MEM_RELEASE);
    }
}

/// What opening costs, on `mb` MB each of private memory never touched
/// and of a committed section never touched: the time, the working set,
/// the card's account of the PC's memory shared with it; and how fast the
/// card copies out of such a heap against a shared buffer of today's kind.
unsafe fn cost(p: &mut Probe, rig: &mut Rig, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize) {
    let bytes = mb << 20;
    let shared = || memory().map_or(0, |m| m[3] >> 20);
    let ms = |speed: Option<f64>| speed.map_or("?".to_owned(), |ms| format!("{ms:.1}"));
    // SAFETY (the whole function): the allocation and the section below.
    unsafe {
        let base = mem::VirtualAlloc(std::ptr::null(), bytes, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE).cast::<u8>();
        if !base.is_null() {
            let (set, account, started) = (working_set_mb(), shared(), std::time::Instant::now());
            let opened = open(&rig.device, From::Part(base as usize, bytes));
            let took = started.elapsed().as_secs_f64() * 1000.0;
            let placed = opened.as_ref().ok().and_then(|heap| place(&rig.device, heap, 0, bytes as u64).ok());
            let speed = placed.as_ref().and_then(|buffer| rig.copy_ms(buffer, bytes as u64).ok());
            p.lines.push(format!(
                "cost, {mb} MB of private memory never touched: {}; opened in {took:.1} ms; working set {set} -> {} MB; shared with the card {account} -> {} MB; \
                 the card copies it out in {} ms",
                hr(&opened),
                working_set_mb(),
                shared(),
                ms(speed)
            ));
            drop((placed, opened));
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
        let invalid = windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        let handle = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, (bytes as u64 >> 32) as u32, bytes as u32, std::ptr::null());
        if !handle.is_null() {
            let (set, account, started) = (working_set_mb(), shared(), std::time::Instant::now());
            let opened = open(&rig.device, From::Section(handle as usize));
            let took = started.elapsed().as_secs_f64() * 1000.0;
            let (set_opened, account_opened) = (working_set_mb(), shared());
            let placed = opened.as_ref().ok().and_then(|heap| place(&rig.device, heap, 0, bytes as u64).ok());
            let speed = placed.as_ref().and_then(|buffer| rig.copy_ms(buffer, bytes as u64).ok());
            p.lines.push(format!(
                "cost, {mb} MB of a committed section never touched: {}; opened in {took:.1} ms; working set {set} -> {set_opened} -> after the card read all of it {} MB; \
                 shared with the card {account} -> {account_opened} -> {} MB; the card copies it out in {} ms",
                hr(&opened),
                working_set_mb(),
                shared(),
                ms(speed)
            ));
            drop((placed, opened));
            windows_sys::Win32::Foundation::CloseHandle(handle);
        }
        // Today's kind of host copy, for the comparison.
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let made = rig.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        );
        if let (Ok(()), Some(buffer)) = (made, buffer) {
            let speed = rig.copy_ms(&buffer, bytes as u64).ok();
            p.lines.push(format!("cost, for comparison: the card copies {mb} MB out of a shared buffer of today's kind in {} ms", ms(speed)));
        }
    }
}

// The census: which of the title memory that has host copies today could
// be opened, and by which opening.

/// Counts and bytes by kind, in the order of [`KINDS`].
#[derive(Default, Clone, Copy)]
struct Tally([(u64, u64); 6]);

const KINDS: [&str; 6] = [
    "private, committed whole (a part can be opened)",
    "view memory, one run of a section (one placed buffer)",
    "view memory, scattered pages",
    "view memory with unmapped pages",
    "private but not committed whole, or guarded",
    "other",
];

fn kind_of(at: usize, len: usize, known: &mut Vec<(usize, usize)>) -> usize {
    match cover(at, len) {
        Cover::Run(..) => return 1,
        Cover::Scattered => return 2,
        Cover::Holes => return 3,
        Cover::None => {}
    }
    let end = at + len;
    if known.iter().any(|&(from, to)| from <= at && end <= to) {
        return 0;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    if n == 0 {
        return 5;
    }
    let reach = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    if info.Type == mem::MEM_PRIVATE && info.State == mem::MEM_COMMIT && info.Protect == mem::PAGE_READWRITE {
        known.push((info.BaseAddress as usize, reach));
        return if reach >= end { 0 } else { 4 };
    }
    match info.Type {
        mem::MEM_PRIVATE => 4,
        mem::MEM_MAPPED => 3,
        _ => 5,
    }
}

/// The census lines for `spans` (start, bytes, whether a window outside
/// buffers).
fn census_of(spans: &[(u64, u64, bool)]) -> Vec<String> {
    let mut known = Vec::new();
    let (mut buffers, mut windows) = (Tally::default(), Tally::default());
    for &(start, len, window) in spans {
        let kind = kind_of(start as usize, len as usize, &mut known);
        let tally = if window { &mut windows } else { &mut buffers };
        tally.0[kind].0 += 1;
        tally.0[kind].1 += len;
    }
    let mut lines = Vec::new();
    for (name, tally) in [("title buffers", buffers), ("windows outside buffers", windows)] {
        let all = tally.0.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        lines.push(format!("census, {name}: {} of {} MB", all.0, all.1 >> 20));
        for (kind, &(n, bytes)) in KINDS.iter().zip(&tally.0) {
            if n > 0 {
                lines.push(format!("census,   {kind}: {n} of {} MB", bytes >> 20));
            }
        }
    }
    let v = views();
    let (count, bytes) = (v.views.len(), v.views.values().map(|view| view.0 as u64).sum::<u64>());
    let pages = v.views.values().filter(|view| view.0 == K64).count();
    let big = v.views.values().map(|view| view.0).max().unwrap_or(0);
    lines.push(format!(
        "census, the title's views of physical pages: {count} of {} MB; {pages} of them one 64 KiB page; the longest {} MB",
        bytes >> 20,
        big >> 20
    ));
    for &(_, bytes, committed) in &v.sections {
        lines.push(format!(
            "census, a section of {} MB, {}",
            bytes >> 20,
            if committed { "committed: it can be opened as one heap" } else { "reserved: refused as a heap until it is made with SEC_COMMIT" }
        ));
    }
    lines
}

/// With [`probe_level`] 2: one part of the title's own private memory
/// (the start of the biggest such title buffer, 4 MiB at most) is opened
/// for a moment, a buffer placed, 64 KiB read by the GPU and compared
/// with what the CPU reads there (the title may be writing it meanwhile:
/// the count of equal bytes is given), and all of it released. While the
/// heap stands (milliseconds) a `VirtualFree` of the title's on those
/// pages would fail.
fn real_part(spans: &[(u64, u64, bool)]) -> Vec<String> {
    let Some(host) = crate::host::host() else {
        return Vec::new();
    };
    let mut known = Vec::new();
    let found = spans
        .iter()
        .filter(|s| !s.2 && s.1 >= 2 * K64 as u64)
        .filter(|s| kind_of(s.0 as usize, s.1 as usize, &mut known) == 0)
        .max_by_key(|s| s.1);
    let Some(&(start, len, _)) = found else {
        return vec!["the title's memory: no title buffer in private, committed memory to try".to_owned()];
    };
    let at = (start as usize).next_multiple_of(K64);
    let bytes = ((start + len) as usize).saturating_sub(at).min(4 << 20) & !(K64 - 1);
    if bytes == 0 || !private_whole(at, bytes) {
        return vec!["the title's memory: the buffer found holds no whole 64 KiB page that is still committed".to_owned()];
    }
    let mut lines = Vec::new();
    let (set, started) = (working_set_mb(), std::time::Instant::now());
    let opened = open(&host.device, From::Part(at, bytes));
    lines.push(format!(
        "the title's memory, {} KiB at {at:#x} (of the title buffer {start:#x}+{len:#x}): {}; in {:.1} ms; working set {set} -> {} MB",
        bytes >> 10,
        hr(&opened),
        started.elapsed().as_secs_f64() * 1000.0,
        working_set_mb()
    ));
    if let Ok(heap) = &opened {
        let placed = place(&host.device, heap, 0, bytes as u64);
        lines.push(format!("  a buffer placed over it: {}", hr(&placed)));
        if let (Ok(buffer), Ok(mut rig)) = (&placed, Rig::new(&host.device)) {
            let mut before = vec![0u8; K64];
            crate::host::copy_mapped(before.as_mut_ptr(), at as *const u8, K64, false);
            let got = rig.fetch(buffer, 0, K64);
            let mut after = vec![0u8; K64];
            crate::host::copy_mapped(after.as_mut_ptr(), at as *const u8, K64, false);
            match got {
                Ok(got) => {
                    let equal = (0..K64).filter(|&i| got[i] == before[i] || got[i] == after[i]).count();
                    let zeros = got.iter().filter(|&&b| b == 0).count();
                    lines.push(format!(
                        "  the GPU read 64 KiB of the title's own bytes: {equal} of {K64} equal to what the CPU read just before or just after ({zeros} of them zero)"
                    ));
                }
                Err(e) => lines.push(format!("  the GPU's read failed: {e}")),
            }
        }
    }
    lines
}

/// Called once a second by the thread that submits, before a submission
/// (`host::sweep`): with [`probe_level`], the probe at the first call, the
/// census (on a thread of its own: it asks Windows about every copy) 30
/// and 150 seconds later.
pub(crate) fn tick() {
    if level() > 0 {
        say();
    }
    if probe_level() == 0 {
        return;
    }
    static STARTED: AtomicU64 = AtomicU64::new(0);
    static DONE: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms().max(1);
    let started = match STARTED.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => {
            let Some(host) = crate::host::host() else {
                return;
            };
            let mb = std::env::var("VICTORY_INPLACE_PROBE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64usize);
            // On a thread of its own (a queue of its own too): Stage 0's
            // probes take seconds; with level 3 they also lock and commit
            // gigabytes for a moment (P1, P2's 4 GiB, P7).
            let device = host.device.clone();
            let _ = std::thread::Builder::new().name("victory-inplace-probe".into()).spawn(move || {
                let took = std::time::Instant::now();
                let probe = probe_with(&device, &|| host.video_memory(), mb.min(1024), probe_level() >= 3, false);
                for line in &probe.lines {
                    tracing::info!("in place probe: {line}");
                }
                tracing::info!("in place probe: took {:.0} ms", took.elapsed().as_secs_f64() * 1000.0);
            });
            now
        }
        Err(started) => started,
    };
    let due = match DONE.load(Ordering::Relaxed) {
        0 => now >= started + 30_000,
        1 => now >= started + 150_000,
        _ => false,
    };
    if due {
        let which = DONE.fetch_add(1, Ordering::Relaxed);
        std::thread::spawn(move || {
            let spans = crate::shadow::spans();
            for line in census_of(&spans) {
                tracing::info!("in place {line}");
            }
            if which == 0 && probe_level() >= 2 {
                for line in real_part(&spans) {
                    tracing::info!("in place probe: {line}");
                }
            }
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Views as the runtime tells them: a range inside one view, or over
    /// views that continue each other in one section, is one run; over
    /// views from elsewhere it is scattered; over a gap it has holes.
    #[test]
    fn views_say_what_one_placed_buffer_could_show() {
        let mut views = BTreeMap::new();
        let at = 0x7_0000_0000usize;
        views.insert(at, (4 * K64, 9, 16 * K64 as u64));
        // Continues the first in the section.
        views.insert(at + 4 * K64, (2 * K64, 9, 20 * K64 as u64));
        // Follows in address, from elsewhere in the section.
        views.insert(at + 6 * K64, (K64, 9, 3 * K64 as u64));
        // After a gap of one page.
        views.insert(at + 8 * K64, (K64, 9, 4 * K64 as u64));
        assert_eq!(cover_in(&views, at + K64, 2 * K64), Cover::Run(9, 17 * K64 as u64));
        assert_eq!(cover_in(&views, at + K64, 5 * K64), Cover::Run(9, 17 * K64 as u64));
        assert_eq!(cover_in(&views, at, 7 * K64), Cover::Scattered);
        assert_eq!(cover_in(&views, at + 6 * K64, 3 * K64), Cover::Holes);
        assert_eq!(cover_in(&views, at + 7 * K64, K64), Cover::None);
        assert_eq!(cover_in(&views, at - K64, 2 * K64), Cover::None);
    }

    /// A window over view memory with arenas is the run of one section
    /// around the ask: as far as the views continue each other, within the
    /// ask (with a length) or the ask's 4 MiB cell (without).
    #[test]
    fn a_window_over_views_is_the_run_around_the_ask() {
        let mut views = BTreeMap::new();
        let at = 0x7_c000_0000usize;
        // Pages 0..3 continue each other in section 4; page 3 is from
        // elsewhere; page 5 is unmapped.
        for k in 0..3 {
            views.insert(at + k * K64, (K64, 4, ((10 + k) * K64) as u64));
        }
        views.insert(at + 3 * K64, (K64, 4, 2 * K64 as u64));
        views.insert(at + 4 * K64, (K64, 4, 3 * K64 as u64));
        assert_eq!(run_end_in(&views, at, at + (4 << 20)), Some(at + 3 * K64));
        assert_eq!(run_end_in(&views, at + 3 * K64, at + (4 << 20)), Some(at + 5 * K64));
        assert_eq!(run_end_in(&views, at + K64, at + 2 * K64), Some(at + 2 * K64), "at most what is asked");
        assert_eq!(run_end_in(&views, at + 5 * K64, at + (4 << 20)), None);
        let base = 0x7_d000_0000usize;
        for k in 0..3 {
            title_memory_view(base + k * K64, K64, 4, ((10 + k) * K64) as u64);
        }
        let b = base as u64;
        assert_eq!(run_window(b + 0x10, b + 0x20, false), Some((b, K64 as u64)));
        assert_eq!(run_window(b + 0x10, b + K64 as u64 + 0x20, false), Some((b, 2 * K64 as u64)));
        assert_eq!(run_window(b + K64 as u64 + 8, 0, true), Some((b + K64 as u64, 2 * K64 as u64)), "to the run's end");
        assert_eq!(run_window(b + 0x10, b + 4 * K64 as u64, false), None, "past the run");
        views_gone(base, 3 * K64);
    }

    /// What the runtime forgets when memory is mapped anew.
    #[test]
    fn remapped_memory_forgets_its_views() {
        let at = 0x7_4000_0000usize;
        title_memory_view(at, 4 * K64, 77, 0);
        title_memory_view(at + 4 * K64, K64, 77, 4 * K64 as u64);
        assert_eq!(cover(at, 5 * K64), Cover::Run(77, 0));
        // One page inside the first view goes: the whole view is forgotten
        // (the runtime maps and tells again what stays of it).
        views_gone(at + K64, K64);
        assert_eq!(cover(at, K64), Cover::None);
        assert_eq!(cover(at + 4 * K64, K64), Cover::Run(77, 4 * K64 as u64));
        views_gone(at + 4 * K64, K64);
        assert_eq!(cover(at + 4 * K64, K64), Cover::None);
    }

    /// Only private memory committed whole, read-write, is taken for a part.
    #[test]
    fn only_private_memory_committed_whole_is_a_part() {
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 * K64, mem::MEM_RESERVE, mem::PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert!(!private_whole(base as usize, K64), "reserved only");
            mem::VirtualAlloc(base.cast(), 4 * K64, mem::MEM_COMMIT, mem::PAGE_READWRITE);
            assert!(private_whole(base as usize, 4 * K64));
            assert!(private_whole(base as usize + K64, 2 * K64));
            assert!(!private_whole(base as usize, 5 * K64), "reaches into the reserved half");
            let mut old = 0;
            mem::VirtualProtect(base.cast(), K64, mem::PAGE_READONLY, &mut old);
            assert!(!private_whole(base as usize, 2 * K64), "a guarded page");
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    pub(crate) fn warp() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        // SAFETY: plain DXGI queries and device creation on the software
        // adapter; a card is never opened.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                return None;
            }
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
            device
        }
    }

    /// The probe, whole, on the software adapter: the same calls the card
    /// gets. The runtime's and the kernel's rules hold there; the card's
    /// driver is what a run with `VICTORY_INPLACE_PROBE=1` adds.
    #[test]
    fn warp_probe_opens_title_like_memory_and_the_gpu_sees_its_bytes() {
        let Some(device) = warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            eprintln!("no ID3D12Device13 here: parts of an allocation cannot be opened");
            return;
        }
        let probe = probe(&device, &|| None, 4);
        for line in &probe.lines {
            eprintln!("in place probe: {line}");
        }
        assert!(probe.part_reads, "the GPU reads private memory in place");
        assert!(probe.part_writes, "GPU writes land in private memory");
        assert!(probe.section_reads && probe.section_writes, "a committed section in place");
        assert!(probe.section_outlives_views, "the section's heap does not depend on the title's views");
        assert!(probe.reserved_section_refused, "a reserved section is refused");
    }

    /// Stage 0's small probes on the software adapter: every one gives its
    /// line and none removes the device.
    #[test]
    fn warp_stage0_probes_give_their_lines() {
        let Some(device) = warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            return;
        }
        let mut rig = Rig::new(&device).unwrap();
        let mut lines = Vec::new();
        crate::inplace_probes::stage0(&mut lines, &mut rig, false, false);
        for line in &lines {
            eprintln!("in place probe: {line}");
        }
        let text = lines.join("\n");
        for name in ["P1 GPU virtual", "P2 private", "P3 private part", "P3 committed section", "P4 ", "P5 512", "P5 two", "P6 vertex", "P6 root", "P6 ExecuteIndirect", "P6 CopyTextureRegion", "P6 ResolveQueryData", "R9 "] {
            assert!(text.contains(name), "no line {name}: {text}");
        }
        assert!(!text.contains("REMOVED"), "{text}");
        assert!(text.contains("InterlockedOr true, InterlockedAnd true"), "{text}");
    }

    /// All of Stage 0 on this PC's card (`cargo test -p victory-gpu --
    /// --ignored probe_on_hardware --nocapture`); `victory inplace-probe`
    /// is the same with a choice of what runs.
    #[test]
    #[ignore = "locks and commits gigabytes on the card's device"]
    fn probe_on_hardware() {
        for line in crate::inplace_probes::run_alone(1024, true, false, false) {
            eprintln!("in place probe: {line}");
        }
    }

    /// The system's shared commit (pagefile-backed sections), in MB.
    pub(crate) fn shared_commit_mb() -> u64 {
        #[link(name = "ntdll")]
        unsafe extern "system" {
            fn NtQuerySystemInformation(class: u32, info: *mut u8, len: u32, returned: *mut u32) -> i32;
        }
        let mut buf = [0u8; 512];
        let mut returned = 0;
        // SAFETY: SystemPerformanceInformation into a buffer big enough.
        let status = unsafe { NtQuerySystemInformation(2, buf.as_mut_ptr(), buf.len() as u32, &mut returned) };
        assert_eq!(status, 0, "NtQuerySystemInformation");
        // SharedCommittedPages, in pages of 4 KiB.
        (u64::from(u32::from_le_bytes(buf[336..340].try_into().unwrap())) * 4096) >> 20
    }

    fn card() -> ID3D12Device {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        // SAFETY: plain DXGI queries and device creation.
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter1 = factory.EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE).unwrap();
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).unwrap();
            device.unwrap()
        }
    }

    const LEAK_MB: usize = 256;

    /// A committed section of [`LEAK_MB`] opened as a heap on the card, a
    /// buffer placed over it and read by the card once.
    fn section_heap(rig: &mut Rig) -> (HANDLE, ID3D12Heap, ID3D12Resource) {
        let bytes = LEAK_MB << 20;
        // SAFETY: plain section creation.
        let handle = unsafe {
            mem::CreateFileMappingW(windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, bytes as u32, std::ptr::null())
        };
        assert!(!handle.is_null());
        let heap = open(&rig.device, From::Section(handle as usize)).unwrap();
        let buffer = place(&rig.device, &heap, 0, bytes as u64).unwrap();
        rig.copy_ms(&buffer, bytes as u64).unwrap();
        (HANDLE(handle), heap, buffer)
    }

    /// Whether the shared commit of a committed section opened as a heap on
    /// the card comes back: released in the process, and in a child that is
    /// killed holding it, killed after letting the heap go, or exits without
    /// letting it go (`VICTORY_ALLOW_SECTION_HEAPS=1 cargo test -p
    /// victory-gpu --release -- --ignored section_heap_commit_comes_back
    /// --nocapture`); with `VICTORY_LEAK_VIEW=1`, a view of it opened by its
    /// address instead. On the RTX 4090 (2026-11-24) every case by the
    /// handle kept the 256 MB until a restart; the view kept none. A case
    /// that leaks keeps [`LEAK_MB`] until the PC restarts.
    #[test]
    #[ignore = "may leak shared commit on the card's driver until a restart"]
    fn section_heap_commit_comes_back() {
        use std::io::BufRead;
        if std::env::var_os("VICTORY_LEAK_VIEW").is_none() && !section_heaps_allowed() {
            eprintln!("VICTORY_ALLOW_SECTION_HEAPS=1 to open sections by their handle on the card");
            return;
        }
        if let Ok(mode) = std::env::var("VICTORY_LEAK_CHILD") {
            let mut rig = Rig::new(&card()).unwrap();
            let (handle, heap, buffer) = section_heap(&mut rig);
            if mode == "release" {
                drop((buffer, heap));
                rig.run(|_| {}).unwrap();
                println!("READY");
            } else if mode == "exit" {
                println!("READY");
                std::process::exit(0);
            } else {
                println!("READY");
                std::mem::forget((buffer, heap));
            }
            let _ = handle;
            std::thread::sleep(std::time::Duration::from_secs(120));
            return;
        }
        let settle = |before: u64| {
            let mut now = shared_commit_mb();
            for _ in 0..20 {
                if now <= before + 32 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
                now = shared_commit_mb();
            }
            now
        };
        // A0, A1: opened only; opened and placed, the card never touching it.
        let opens: &[bool] = if std::env::var_os("VICTORY_LEAK_VIEW").is_some() { &[] } else { &[false, true] };
        for &place_it in opens {
            let before = shared_commit_mb();
            let rig = Rig::new(&card()).unwrap();
            let bytes = LEAK_MB << 20;
            // SAFETY: plain section creation.
            let handle = unsafe {
                mem::CreateFileMappingW(windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, bytes as u32, std::ptr::null())
            };
            let heap = open(&rig.device, From::Section(handle as usize)).unwrap();
            let buffer = place_it.then(|| place(&rig.device, &heap, 0, bytes as u64).unwrap());
            drop((buffer, heap, rig));
            // SAFETY: our handle.
            unsafe { windows_sys::Win32::Foundation::CloseHandle(handle) };
            let after = settle(before);
            eprintln!("leak case A{} (opened{}, never used): shared commit {before} -> after {after} MB (leaked {} MB)", place_it as u8, if place_it { " and placed" } else { "" }, after as i64 - before as i64);
        }
        // V: a view of the section, opened by its address.
        if std::env::var_os("VICTORY_LEAK_VIEW").is_some() {
            let before = shared_commit_mb();
            let mut rig = Rig::new(&card()).unwrap();
            let bytes = LEAK_MB << 20;
            // SAFETY: plain section creation and a view of all of it.
            let (handle, view) = unsafe {
                let handle = mem::CreateFileMappingW(windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, bytes as u32, std::ptr::null());
                (handle, mem::MapViewOfFile(handle, mem::FILE_MAP_WRITE, 0, 0, bytes))
            };
            fill(view.Value.cast(), 4096, 7);
            let opened = open(&rig.device, From::Part(view.Value as usize, bytes));
            let mut line = format!("open by address of a view: {}", hr(&opened));
            if let Ok(heap) = &opened {
                let buffer = place(&rig.device, heap, 0, bytes as u64).unwrap();
                let seen = rig.fetch(&buffer, 0, 4096).unwrap();
                line += &format!("; the card reads the view's bytes {}", same(view.Value.cast(), &seen));
                rig.store(&buffer, 8192, &[0xAB; 64]).unwrap();
                // SAFETY: inside the view.
                line += &format!("; its write lands in the view {}", unsafe { *view.Value.cast::<u8>().add(8192 + 63) } == 0xAB);
                drop(buffer);
            }
            drop((opened, rig));
            // SAFETY: our view and handle.
            unsafe {
                mem::UnmapViewOfFile(view);
                windows_sys::Win32::Foundation::CloseHandle(handle);
            }
            let after = settle(before);
            eprintln!("leak case V ({line}): shared commit {before} -> after {after} MB (leaked {} MB)", after as i64 - before as i64);
            return;
        }
        if std::env::var_os("VICTORY_LEAK_ONLY_OPEN").is_some() {
            return;
        }
        // A: in the process, everything let go in order.
        {
            let before = shared_commit_mb();
            let mut rig = Rig::new(&card()).unwrap();
            let (handle, heap, buffer) = section_heap(&mut rig);
            let held = shared_commit_mb();
            drop((buffer, heap));
            rig.run(|_| {}).unwrap();
            // SAFETY: our handle.
            unsafe { windows::Win32::Foundation::CloseHandle(handle).unwrap() };
            drop(rig);
            let after = settle(before);
            eprintln!("leak case A (released in the process): shared commit {before} -> held {held} -> after {after} MB (leaked {} MB)", after as i64 - before as i64);
        }
        for mode in ["kill", "release", "exit"] {
            let before = shared_commit_mb();
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "inplace::tests::section_heap_commit_comes_back", "--ignored", "--nocapture", "--test-threads=1"])
                .env("VICTORY_LEAK_CHILD", mode)
                .stdout(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            let out = std::io::BufReader::new(child.stdout.take().unwrap());
            let ready = out.lines().map_while(Result::ok).any(|l| l.contains("READY"));
            let held = shared_commit_mb();
            if mode != "exit" {
                child.kill().unwrap();
            }
            let status = child.wait().unwrap();
            let after = settle(before);
            eprintln!(
                "leak case {mode} (child {}, {status}): shared commit {before} -> held {held} -> after {after} MB (leaked {} MB)",
                if ready { "ready" } else { "NOT ready" },
                after as i64 - before as i64
            );
        }
    }

    /// Stage 1's own call on the software adapter: a title buffer that
    /// begins and ends off the 64 KiB lines is placed over its pages, the
    /// GPU reads the buffer's bytes at the offset the views use, and its
    /// write lands at the title's address. Memory that is not committed
    /// whole is turned down without asking the host.
    #[test]
    fn warp_a_title_buffer_in_place_shows_its_bytes_at_its_offset() {
        let Some(device) = warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            return;
        }
        check_buffer_at_offset(device);
    }

    /// Exercise the same small allocation and fenced copies through the physical adapter.
    #[test]
    #[ignore = "requires a physical adapter with ID3D12Device13 and host memory import"]
    fn physical_in_place_buffer_shows_bytes_at_offset() {
        let device = card();
        device.cast::<ID3D12Device13>().expect("physical adapter supports ID3D12Device13");
        check_buffer_at_offset(device);
    }

    fn check_buffer_at_offset(device: ID3D12Device) {
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 * K64, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE).cast::<u8>();
            let (start, len) = (base as u64 + K64 as u64 + 0x1230, 2 * K64 as u64 + 5);
            let placed = place_part(&device, start, len).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(placed.bias, 0x1230);
            assert_eq!(placed.gpu, placed.buffer.GetGPUVirtualAddress() + 0x1230);
            assert_eq!(placed.buffer.GetDesc().Width, 3 * K64 as u64, "the pages that hold the buffer");
            let mut rig = Rig::new(&device).unwrap();
            fill(start as *mut u8, len as usize, 7);
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert!(same(start as *const u8, &got), "the GPU reads the title buffer's bytes");
            rig.store(&placed.buffer, placed.bias + 16, &[0xa5; 64]).unwrap();
            assert!(same((start + 16) as *const u8, &[0xa5; 64]), "the GPU's write is at the title's address");
            drop(placed);
            // Half of it decommitted: turned down, the host not asked.
            assert_ne!(mem::VirtualFree(base.add(2 * K64).cast(), K64, mem::MEM_DECOMMIT), 0);
            assert!(matches!(place_part(&device, start, len), Err(None)));
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    /// The census names each kind of memory.
    #[test]
    fn census_tells_private_memory_from_views() {
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 * K64, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) as u64;
            let view = 0x7_8000_0000u64;
            title_memory_view(view as usize, 2 * K64, 5, 0);
            title_memory_view(view as usize + 2 * K64, K64, 5, 9 * K64 as u64);
            let lines = census_of(&[(base, 2 * K64 as u64, false), (base + 4 * K64 as u64, K64 as u64, true), (view, K64 as u64, true), (view, 3 * K64 as u64, false)]);
            let text = lines.join("\n");
            assert!(text.contains("census, title buffers: 2 of"), "{text}");
            assert!(text.contains("census,   private, committed whole (a part can be opened): 1 of"), "{text}");
            assert!(text.contains("census,   view memory, scattered pages: 1 of"), "{text}");
            assert!(text.contains("census,   view memory, one run of a section (one placed buffer): 1 of"), "{text}");
            views_gone(view as usize, 3 * K64);
            mem::VirtualFree(base as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }
}
