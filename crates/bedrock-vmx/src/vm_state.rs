// SPDX-License-Identifier: GPL-2.0

//! `VmState`: all VM state except guest memory, shared by `RootVm` and `ForkedVm`.

#[cfg(not(feature = "cargo"))]
use super::prelude::*;
#[cfg(feature = "cargo")]
use crate::prelude::*;

type DeviceStatesBox = HeapBox<DeviceStates>;
type ExitStatsBox = HeapBox<AllExitStats>;

/// Unbounded set of registered feedback buffers. Each ~2KB entry is boxed
/// separately so the vector only holds pointers and never needs a large
/// contiguous allocation. Append-only (no unregister), so a buffer's slot index
/// is its stable position in this vector.
pub type FeedbackBuffers = HeapVec<HeapBox<FeedbackBufferInfo>>;

/// PEBS VM-entry MSR-load list entries, in fixed order; `pebs_pre_vm_entry`
/// writes indexes and values.
///
/// - Entry 0 (`IA32_A_PMC0`): while armed this page replaces the instruction
///   counter's (IC, on `IA32_PMC0`) entry-load page, so it must reload the IC's
///   saved value. The full-width alias is required: plain `IA32_PMC0` writes
///   truncate to 32 bits and sign-extend (SDM Vol 3B 21.2.8). The exit-store
///   list still points at the IC's page, which stays the source of truth.
/// - Entries 1 and 8 (`IA32_PERF_GLOBAL_CTRL`): the MSR-load area runs after
///   the VMCS global-ctrl load (SDM Vol 3C 28.3.2, 28.4), so the reconfig in
///   entries 2–4 would hit a running counter, which disqualifies PDist (SDM
///   Vol 3B 21.9.6). Disable/re-enable brackets it.
/// - `IA32_PERF_GLOBAL_STATUS_RESET` must precede `IA32_PEBS_ENABLE`: stale
///   overflow bits otherwise make PEBS fire on nearly every VM-entry.
pub const PEBS_ENTRY_MSR_INDEXES: [u32; 9] = [
    msr::IA32_A_PMC0,
    msr::IA32_PERF_GLOBAL_CTRL,
    msr::IA32_FIXED_CTR0,
    msr::IA32_FIXED_CTR_CTRL,
    msr::MSR_PEBS_DATA_CFG,
    msr::IA32_DS_AREA,
    msr::IA32_PERF_GLOBAL_STATUS_RESET,
    msr::IA32_PEBS_ENABLE,
    msr::IA32_PERF_GLOBAL_CTRL,
];

/// `IA32_PERF_CAPABILITIES` bit 12: `IA32_DEBUGCTL.FREEZE_WHILE_SMM` is
/// supported (SDM Vol 3B 19.4.1).
pub const PERF_CAPABILITIES_FREEZE_WHILE_SMM: u64 = 1 << 12;

/// `IA32_DEBUGCTL` bit 14: SMI delivery clears every `IA32_PERF_GLOBAL_CTRL`
/// enable and RSM restores them, so SMM code is not counted. Without it, an
/// SMI during guest execution adds the SMM handler's instructions (about
/// 2.3M on a Kaby Lake desktop) to the instruction counter, i.e. to the
/// emulated TSC (SDM Vol 3B 19.4.1).
pub const DEBUGCTL_FREEZE_WHILE_SMM: u64 = 1 << 14;

/// PMI-mode counterpart of `PEBS_ENTRY_MSR_INDEXES`, minus the PEBS-only
/// MSRs (which may not exist on hosts that fall back to PMI).
pub const PMI_ENTRY_MSR_INDEXES: [u32; 6] = [
    msr::IA32_A_PMC0,
    msr::IA32_PERF_GLOBAL_CTRL,
    msr::IA32_FIXED_CTR0,
    msr::IA32_FIXED_CTR_CTRL,
    msr::IA32_PERF_GLOBAL_STATUS_RESET,
    msr::IA32_PERF_GLOBAL_CTRL,
];

/// Create an empty feedback-buffers vector.
fn feedback_buffers_new() -> FeedbackBuffers {
    heap_vec_with_capacity(0).expect("Failed to allocate feedback buffers")
}

/// Deep-copy the parent's feedback buffers for a forked VM.
fn feedback_buffers_from(parent: &FeedbackBuffers) -> FeedbackBuffers {
    let mut v = heap_vec_with_capacity(parent.len()).expect("Failed to allocate feedback buffers");
    for fb in parent.iter() {
        // Copy heap-to-heap: a ~2KB stack temporary here would blow the 8KB
        // kernel stack on the deep fork-creation call chain.
        let cloned = heap_box_copy_from(&**fb).expect("Failed to clone feedback buffer");
        heap_vec_push(&mut v, cloned).expect("Failed to clone feedback buffer");
    }
    v
}

/// Size of the I/O channel shared page (one 4KB page).
pub const IO_CHANNEL_BUF_SIZE: usize = 4096;

/// Heap-allocated 4KB buffer. Kept off `VmState` inline storage, which would
/// blow the 8KB kernel stack during `VmState::new`.
pub type IoPageBufBox = VmallocBox<[u8; IO_CHANNEL_BUF_SIZE]>;

/// Allocate a zeroed I/O channel page buffer directly on the heap.
#[cfg(feature = "cargo")]
fn box_io_page_buf() -> IoPageBufBox {
    extern crate alloc;
    let v = alloc::vec![0u8; IO_CHANNEL_BUF_SIZE];
    let boxed_slice = v.into_boxed_slice();
    let ptr = alloc::boxed::Box::into_raw(boxed_slice) as *mut [u8; IO_CHANNEL_BUF_SIZE];
    // SAFETY: `boxed_slice` has exactly `IO_CHANNEL_BUF_SIZE` elements, so its
    // pointer can be reinterpreted as a pointer to a fixed-size array of the
    // same length.
    unsafe { alloc::boxed::Box::from_raw(ptr) }
}

#[cfg(not(feature = "cargo"))]
fn box_io_page_buf() -> IoPageBufBox {
    let mut boxed: kernel::alloc::KVBox<core::mem::MaybeUninit<[u8; IO_CHANNEL_BUF_SIZE]>> =
        kernel::alloc::KVBox::new_uninit(kernel::alloc::flags::GFP_KERNEL)
            .expect("Failed to allocate I/O channel page buffer");
    // SAFETY: we zero-fill the entire allocation before `assume_init`, and 0 is
    // a valid `u8`.
    unsafe {
        let ptr = boxed.as_mut_ptr().cast::<u8>();
        core::ptr::write_bytes(ptr, 0, IO_CHANNEL_BUF_SIZE);
        boxed.assume_init()
    }
}

/// Heap-allocated early-boot serial line accumulator. Boxed for the same stack
/// reason as [`IoPageBufBox`]: `VmState` is built by value on the stack.
pub type SerialLineBufBox = VmallocBox<[u8; SERIAL_LINE_ACC_SIZE]>;

/// Allocate a zeroed serial line accumulator directly on the heap.
#[cfg(feature = "cargo")]
fn box_serial_line_buf() -> SerialLineBufBox {
    extern crate alloc;
    let v = alloc::vec![0u8; SERIAL_LINE_ACC_SIZE];
    let boxed_slice = v.into_boxed_slice();
    let ptr = alloc::boxed::Box::into_raw(boxed_slice) as *mut [u8; SERIAL_LINE_ACC_SIZE];
    // SAFETY: `boxed_slice` has exactly `SERIAL_LINE_ACC_SIZE` elements, so its
    // pointer can be reinterpreted as a pointer to a fixed-size array of the
    // same length.
    unsafe { alloc::boxed::Box::from_raw(ptr) }
}

#[cfg(not(feature = "cargo"))]
fn box_serial_line_buf() -> SerialLineBufBox {
    let mut boxed: kernel::alloc::KVBox<core::mem::MaybeUninit<[u8; SERIAL_LINE_ACC_SIZE]>> =
        kernel::alloc::KVBox::new_uninit(kernel::alloc::flags::GFP_KERNEL)
            .expect("Failed to allocate serial line accumulator");
    // SAFETY: freshly allocated; we zero-fill the entire region before calling
    // `assume_init`, so every byte is initialized to a valid `u8` (0).
    unsafe {
        let ptr = boxed.as_mut_ptr().cast::<u8>();
        core::ptr::write_bytes(ptr, 0, SERIAL_LINE_ACC_SIZE);
        boxed.assume_init()
    }
}

/// Maximum I/O channel requests queued behind the in-flight slot. Bounds heap
/// use to ~`PENDING_IO_QUEUE_CAP * IO_CHANNEL_BUF_SIZE`.
pub const PENDING_IO_QUEUE_CAP: usize = 256;

/// One pending I/O action waiting for the in-flight slot. `data` is sized
/// exactly (not a 4KB box) since requests are usually short commands.
pub struct PendingIoAction {
    /// Earliest emulated TSC at which this action may fire.
    pub target_tsc: u64,
    /// Request bytes, exactly as the guest sees them on its shared page.
    pub data: HeapVec<u8>,
}

/// Size of the paravirtual-console shared page; the max bytes per
/// `HYPERCALL_SERIAL_WRITE`. Equals the `IoPageBufBox` capacity, so a clamped
/// write always fits in `pending_buf`.
pub const SERIAL_CONSOLE_PAGE_SIZE: usize = PAGE_SIZE;

/// State for the paravirtual batch console. The guest's `bedrock-console.ko`
/// registers a shared page (`HYPERCALL_SERIAL_REGISTER_PAGE`) and sends whole
/// printk records via `HYPERCALL_SERIAL_WRITE`, each emitted as one `Serial`
/// event. Excluded from the determinism state hash: nothing here is
/// guest-visible.
pub struct SerialConsoleState {
    /// GPA of the registered console page; 0 = unregistered (writes fail).
    pub page_gpa: u64,
    /// Staging buffer for a `HYPERCALL_SERIAL_WRITE` record before emission.
    pub pending_buf: IoPageBufBox,
}

impl Default for SerialConsoleState {
    fn default() -> Self {
        Self::new()
    }
}

impl SerialConsoleState {
    /// Create unregistered console state.
    pub fn new() -> Self {
        Self {
            page_gpa: 0,
            pending_buf: box_io_page_buf(),
        }
    }

    /// Clone state for a forked child. The page registration is inherited
    /// (same GPA in CoW memory); pending bytes belong to the parent and are
    /// dropped.
    pub fn clone_for_fork(parent: &Self) -> Self {
        Self {
            page_gpa: parent.page_gpa,
            pending_buf: box_io_page_buf(),
        }
    }
}

/// State for the deterministic hypervisor↔guest I/O channel.
///
/// - `HYPERCALL_IO_REGISTER_PAGE` sets `page_gpa`.
/// - `BEDROCK_VM_QUEUE_IO_ACTION` queues/promotes a request.
/// - `check_io_channel` sets `request_delivered` once the IRR bit is set.
/// - `HYPERCALL_IO_GET_REQUEST` copies the request to the shared page.
/// - `HYPERCALL_IO_PUT_RESPONSE` fills `response_buf` and exits to userspace.
pub struct IoChannelState {
    /// GPA of the registered shared page; 0 = unregistered, so IRQ injection
    /// must be held off.
    pub page_gpa: u64,
    /// Length of the in-flight request; 0 = slot free.
    pub request_len: usize,
    /// IRQ already raised for the in-flight request (don't re-set IRR).
    pub request_delivered: bool,
    /// Earliest emulated TSC for delivering the in-flight request; 0 = as soon
    /// as the guest is interruptible. Non-zero arms PEBS for a precise exit at
    /// this target, and `check_io_channel` defers IRR until it is reached.
    pub request_target_tsc: u64,
    /// Length of `response_buf`, valid from the `VmcallIoResponse` exit until
    /// userspace drains it.
    pub response_len: usize,
    /// In-flight request bytes.
    pub request_buf: IoPageBufBox,
    /// Latest response, drained via `BEDROCK_VM_DRAIN_IO_RESPONSE`.
    pub response_buf: IoPageBufBox,
    /// FIFO of requests waiting for the in-flight slot. Each GET_REQUEST
    /// promotes the next one without waiting for PUT_RESPONSE, since the guest
    /// may handle requests in parallel.
    pub pending: HeapVec<PendingIoAction>,
}

impl Default for IoChannelState {
    fn default() -> Self {
        Self::new()
    }
}

impl IoChannelState {
    /// Create unregistered I/O channel state.
    pub fn new() -> Self {
        Self {
            page_gpa: 0,
            request_len: 0,
            request_delivered: false,
            request_target_tsc: 0,
            response_len: 0,
            request_buf: box_io_page_buf(),
            response_buf: box_io_page_buf(),
            pending: heap_vec_with_capacity(0).expect("Failed to allocate pending queue"),
        }
    }

    /// Clone state for a forked child. Only the page registration is inherited
    /// (same GPA in CoW memory); no I/O is in flight at a fork point.
    pub fn clone_for_fork(parent: &Self) -> Self {
        let _ = parent;
        Self {
            page_gpa: parent.page_gpa,
            request_len: 0,
            request_delivered: false,
            request_target_tsc: 0,
            response_len: 0,
            request_buf: box_io_page_buf(),
            response_buf: box_io_page_buf(),
            pending: heap_vec_with_capacity(0).expect("Failed to allocate pending queue"),
        }
    }

    /// Promote the front of `pending` into the in-flight slot if it is free.
    pub fn promote_next_pending(&mut self) {
        if self.request_len != 0 {
            return;
        }
        // O(n) shift; negligible at these queue depths.
        let next = match heap_vec_remove_front(&mut self.pending) {
            Some(n) => n,
            None => return,
        };
        let len = next.data.len().min(IO_CHANNEL_BUF_SIZE);
        self.request_buf[..len].copy_from_slice(&next.data[..len]);
        self.request_len = len;
        self.request_target_tsc = next.target_tsc;
        self.request_delivered = false;
        self.response_len = 0;
    }

    /// Append a request to the pending queue. Does not promote.
    pub fn enqueue_pending(&mut self, action: PendingIoAction) -> EnqueueResult {
        if self.pending.len() >= PENDING_IO_QUEUE_CAP {
            return EnqueueResult::Full;
        }
        match heap_vec_push(&mut self.pending, action) {
            Ok(()) => EnqueueResult::Queued,
            Err(_) => EnqueueResult::OutOfMemory,
        }
    }
}

/// Outcome of `IoChannelState::enqueue_pending`. `Full` maps to `-EBUSY`,
/// `OutOfMemory` to `-ENOMEM`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueResult {
    Queued,
    Full,
    OutOfMemory,
}

/// Boxed VmState type alias - used by RootVm and ForkedVm to reduce stack usage.
pub type VmStateBox<V, I> = HeapBox<VmState<V, I>>;

/// Box a VmState for heap allocation.
pub fn box_vm_state<V: VirtualMachineControlStructure, I: InstructionCounter>(
    state: VmState<V, I>,
) -> VmStateBox<V, I> {
    heap_box(state)
}

const PAGE_SIZE: usize = 4096;

/// Maximum pages in a single feedback buffer (1MB). The buffer count is
/// unbounded (see [`FeedbackBuffers`]).
pub const FEEDBACK_BUFFER_MAX_PAGES: usize = 256;

/// mmap stride of one feedback-buffer slot (1MB): buffer `i` lives at
/// `base + i * FEEDBACK_BUFFER_SLOT_SIZE`. Shared by the kernel mmap handlers
/// and the userspace mapper.
pub const FEEDBACK_BUFFER_SLOT_SIZE: u64 = FEEDBACK_BUFFER_MAX_PAGES as u64 * PAGE_SIZE as u64;

/// Fixed mmap file offset of the unified event buffer (64 TiB), above the
/// guest-memory and unbounded feedback-buffer regions for any realistic
/// configuration. Shared by the kernel mmap handlers and the userspace mapper.
pub const EVENT_BUFFER_MMAP_OFFSET: u64 = 1 << 46;

/// Maximum length, in bytes, of a feedback-buffer identifier. Sized to fit a
/// SHA-256 hex digest with room for a colon-separated suffix.
pub const FEEDBACK_BUFFER_ID_MAX_LEN: usize = 128;

/// A feedback buffer (e.g. coverage bitmap) registered via
/// `HYPERCALL_REGISTER_FEEDBACK_BUFFER`. IDs (e.g. a `--build-id`) need not be
/// unique; duplicates are instances of the same domain, merged by the host.
#[derive(Clone, Copy)]
pub struct FeedbackBufferInfo {
    /// Original guest virtual address.
    pub gva: u64,
    /// Size in bytes.
    pub size: u64,
    /// Number of pages.
    pub num_pages: usize,
    /// Page-aligned GPAs that make up the buffer.
    pub gpas: [u64; FEEDBACK_BUFFER_MAX_PAGES],
    /// Identifier bytes; the first `id_len` are meaningful, the rest zero.
    pub id: [u8; FEEDBACK_BUFFER_ID_MAX_LEN],
    /// Identifier length, `<= FEEDBACK_BUFFER_ID_MAX_LEN`.
    pub id_len: u32,
}

impl Default for FeedbackBufferInfo {
    fn default() -> Self {
        Self {
            gva: 0,
            size: 0,
            num_pages: 0,
            gpas: [0u64; FEEDBACK_BUFFER_MAX_PAGES],
            id: [0u8; FEEDBACK_BUFFER_ID_MAX_LEN],
            id_len: 0,
        }
    }
}

impl FeedbackBufferInfo {
    /// The identifier as a byte slice.
    pub fn id_bytes(&self) -> &[u8] {
        &self.id[..self.id_len as usize]
    }
}

/// Clear the read and write intercept bits for `msr` (enable passthrough).
///
/// MSR bitmap layout (SDM Vol 3C 26.6.9): read-low at 0, read-high at 1024,
/// write-low at 2048, write-high at 3072. Low = 0..0x1FFF,
/// high = 0xC0000000..0xC0001FFF.
///
/// # Safety
/// `bitmap` must point to a valid 4KB MSR bitmap page.
#[inline]
fn msr_bitmap_clear_intercept(bitmap: *mut u8, msr: u32) {
    let (read_base, write_base, index) = if msr < 0x2000 {
        (0usize, 2048usize, msr as usize)
    } else if (0xC000_0000..0xC000_2000).contains(&msr) {
        (1024usize, 3072usize, (msr - 0xC000_0000) as usize)
    } else {
        // Outside the bitmap: always exits.
        return;
    };

    let byte_offset = index / 8;
    let bit_mask = !(1u8 << (index % 8));

    // Safety: caller guarantees bitmap points to valid 4KB page
    unsafe {
        let read_ptr = bitmap.add(read_base + byte_offset);
        *read_ptr &= bit_mask;

        let write_ptr = bitmap.add(write_base + byte_offset);
        *write_ptr &= bit_mask;
    }
}

/// Default IA32_PAT value after reset.
/// PAT0=WB(6), PAT1=WT(4), PAT2=UC-(7), PAT3=UC(0),
/// PAT4=WB(6), PAT5=WT(4), PAT6=UC-(7), PAT7=UC(0)
pub const PAT_DEFAULT: u64 = 0x0007_0406_0007_0406;

/// Default TSC frequency (2995.2 MHz) for deterministic time emulation.
pub const DEFAULT_TSC_FREQUENCY: u64 = 2_995_200_000;

/// Logging mode for deterministic exit capture.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExitTrigger {
    /// No logging.
    #[default]
    Disabled = 0,
    /// Log every exit.
    AllExits = 1,
    /// Log once when TSC >= `exit_target_tsc`, hashing full memory (bisection).
    AtTsc = 2,
    /// Log once at vmcall shutdown, hashing full memory.
    AtShutdown = 3,
    /// Log every `exit_target_tsc` ticks, without the memory hash.
    Checkpoints = 4,
    /// Log exits within `single_step_tsc_range` (used with single-stepping).
    TscRange = 5,
}

/// Synthetic (non-VMX) exit reason marking checkpoint entries.
pub const EXIT_REASON_CHECKPOINT: u32 = 0xFFFFFFFF;

/// Where a just-emitted `Exit` record's payload lives, so its deferred
/// `memory_hash` field can be patched after the guest's memory stabilizes.
#[derive(Clone, Copy)]
pub enum ExitLoc {
    /// Byte offset of the payload within the event buffer.
    Buffer(usize),
    /// The record was staged pending (buffer was full at emit); its payload is
    /// at offset 0 of `event_pending_buf`.
    Pending,
}

/// Per-exit-type count and handling cycles (host RDTSC).
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct ExitStats {
    /// Number of exits of this type.
    pub count: u64,
    /// Total CPU cycles spent handling this exit type (via RDTSC).
    pub cycles: u64,
}

impl ExitStats {
    /// Record an exit with the given cycle count.
    #[inline]
    pub fn record(&mut self, cycles: u64) {
        self.count += 1;
        self.cycles += cycles;
    }

    /// Get the average cycles per exit, or 0 if no exits occurred.
    #[inline]
    pub fn avg_cycles(&self) -> u64 {
        self.cycles.checked_div(self.count).unwrap_or(0)
    }
}

/// COW fault statistics, used to judge whether pre-allocating adjacent pages
/// would help.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct CowStats {
    /// Total number of COW faults handled.
    pub total_faults: u64,
    /// Number of COW faults where an adjacent page (±1) was already COW'd.
    pub adjacent_1: u64,
    /// Number of COW faults where a page within ±2 pages was already COW'd.
    pub adjacent_2: u64,
    /// Number of COW faults where a page within ±4 pages was already COW'd.
    pub adjacent_4: u64,
    /// Number of COW faults where a page within ±8 pages was already COW'd.
    pub adjacent_8: u64,
    /// EPT violations on already-COW'd pages, i.e. stale EPT TLB entries.
    pub stale_tlb_faults: u64,
}

impl CowStats {
    /// Record a COW fault. `min_distance` is the distance in pages to the
    /// nearest already-COW'd page, if any.
    #[inline]
    pub fn record(&mut self, min_distance: Option<u64>) {
        self.total_faults += 1;
        if let Some(dist) = min_distance {
            if dist <= 1 {
                self.adjacent_1 += 1;
            }
            if dist <= 2 {
                self.adjacent_2 += 1;
            }
            if dist <= 4 {
                self.adjacent_4 += 1;
            }
            if dist <= 8 {
                self.adjacent_8 += 1;
            }
        }
    }
}

/// Exit statistics for all exit types.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct AllExitStats {
    /// CPUID instruction exits.
    pub cpuid: ExitStats,
    /// MSR read (RDMSR) exits.
    pub msr_read: ExitStats,
    /// MSR write (WRMSR) exits.
    pub msr_write: ExitStats,
    /// Control register access exits.
    pub cr_access: ExitStats,
    /// I/O instruction exits.
    pub io_instruction: ExitStats,
    /// EPT violation exits.
    pub ept_violation: ExitStats,
    /// External interrupt exits.
    pub external_interrupt: ExitStats,
    /// RDTSC instruction exits.
    pub rdtsc: ExitStats,
    /// RDTSCP instruction exits.
    pub rdtscp: ExitStats,
    /// RDPMC instruction exits.
    pub rdpmc: ExitStats,
    /// MWAIT instruction exits.
    pub mwait: ExitStats,
    /// VMCALL hypercall exits.
    pub vmcall: ExitStats,
    /// APIC access exits.
    pub apic_access: ExitStats,
    /// Monitor trap flag (MTF) exits.
    pub mtf: ExitStats,
    /// XSETBV instruction exits.
    pub xsetbv: ExitStats,
    /// RDRAND instruction exits.
    pub rdrand: ExitStats,
    /// RDSEED instruction exits.
    pub rdseed: ExitStats,
    /// Exception/NMI exits.
    pub exception_nmi: ExitStats,
    /// All other exit types combined.
    pub other: ExitStats,
    /// Total cycles in VM run loop (including guest time).
    pub total_run_cycles: u64,
    /// Total cycles in guest mode (actual VMX non-root execution).
    pub guest_cycles: u64,
    /// Cycles spent in run loop setup before VM entry (VMCS updates, GPR sync).
    pub vmentry_overhead_cycles: u64,
    /// Cycles spent after VM exit before exit handler (GPR sync, LFENCE, etc),
    /// excluding time in the IRQ window.
    pub vmexit_overhead_cycles: u64,
    /// Cycles spent in the IRQ window between VM exits (host interrupt servicing
    /// and perf counter read).
    pub irq_window_cycles: u64,
    /// Copy-on-write page allocation statistics.
    pub cow: CowStats,
    /// MTF exits inside the PEBS margin window. Kept separate from `mtf`
    /// because the count depends on PEBS skid and varies across runs.
    pub pebs_margin_steps: u64,
    /// `arm_precise_exit` returned `BelowMinDelta` (MTF stepping lands it).
    pub pebs_arm_below_min_delta: u64,
    /// `arm_precise_exit` returned `AlreadyPast` — `target_tsc < current_tsc`.
    pub pebs_arm_already_past: u64,
    /// Iterations entered with PEBS armed that exited for another reason.
    pub pebs_armed_iter_no_fire: u64,
    /// `check_apic_timer` fired with `emulated_tsc > deadline`, i.e. the
    /// precise PEBS+MTF path was missed and the timer was delivered late.
    pub apic_timer_late_inject: u64,
    /// Largest PEBS skid this run (`pebs_exit_tsc - armed_target_tsc`); the
    /// minimum safe `exits::pebs::margin_for_host_cpu` for this host.
    pub max_pebs_skid: i64,
}

impl AllExitStats {
    /// Record an exit of the given type with the specified cycle count.
    #[inline]
    pub fn record(&mut self, reason: ExitReason, cycles: u64) {
        match reason {
            ExitReason::Cpuid => self.cpuid.record(cycles),
            ExitReason::MsrRead => self.msr_read.record(cycles),
            ExitReason::MsrWrite => self.msr_write.record(cycles),
            ExitReason::CrAccess => self.cr_access.record(cycles),
            ExitReason::IoInstruction => self.io_instruction.record(cycles),
            ExitReason::EptViolation => self.ept_violation.record(cycles),
            ExitReason::ExternalInterrupt => self.external_interrupt.record(cycles),
            ExitReason::Rdtsc => self.rdtsc.record(cycles),
            ExitReason::Rdtscp => self.rdtscp.record(cycles),
            ExitReason::Rdpmc => self.rdpmc.record(cycles),
            ExitReason::Mwait => self.mwait.record(cycles),
            ExitReason::Vmcall | ExitReason::VmcallShutdown => self.vmcall.record(cycles),
            ExitReason::ApicAccess | ExitReason::ApicWrite => self.apic_access.record(cycles),
            ExitReason::MonitorTrapFlag => self.mtf.record(cycles),
            ExitReason::Xsetbv => self.xsetbv.record(cycles),
            ExitReason::Rdrand => self.rdrand.record(cycles),
            ExitReason::Rdseed => self.rdseed.record(cycles),
            ExitReason::ExceptionNmi => self.exception_nmi.record(cycles),
            _ => self.other.record(cycles),
        }
    }

    /// Get total exit count across all types.
    pub fn total_exit_count(&self) -> u64 {
        self.cpuid.count
            + self.msr_read.count
            + self.msr_write.count
            + self.cr_access.count
            + self.io_instruction.count
            + self.ept_violation.count
            + self.external_interrupt.count
            + self.rdtsc.count
            + self.rdtscp.count
            + self.rdpmc.count
            + self.mwait.count
            + self.vmcall.count
            + self.apic_access.count
            + self.mtf.count
            + self.xsetbv.count
            + self.rdrand.count
            + self.rdseed.count
            + self.exception_nmi.count
            + self.other.count
    }

    /// Get total exit handling cycles across all types.
    pub fn total_exit_cycles(&self) -> u64 {
        self.cpuid.cycles
            + self.msr_read.cycles
            + self.msr_write.cycles
            + self.cr_access.cycles
            + self.io_instruction.cycles
            + self.ept_violation.cycles
            + self.external_interrupt.cycles
            + self.rdtsc.cycles
            + self.rdtscp.cycles
            + self.rdpmc.cycles
            + self.mwait.cycles
            + self.vmcall.cycles
            + self.apic_access.cycles
            + self.mtf.cycles
            + self.xsetbv.cycles
            + self.rdrand.cycles
            + self.rdseed.cycles
            + self.exception_nmi.cycles
            + self.other.cycles
    }

    /// Reset all statistics to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Guest SYSCALL/SYSRET MSR state.
#[derive(Clone, Copy, Debug, Default)]
pub struct SyscallMsrs {
    /// IA32_STAR (0xC0000081) - SYSCALL segment selectors.
    pub star: Star,
    /// IA32_LSTAR (0xC0000082) - SYSCALL 64-bit entry point.
    pub lstar: Lstar,
    /// IA32_CSTAR (0xC0000083) - SYSCALL compatibility mode entry point.
    pub cstar: Cstar,
    /// IA32_FMASK (0xC0000084) - SYSCALL RFLAGS mask.
    pub fmask: Fmask,
}

impl SyscallMsrs {
    /// Capture SYSCALL MSRs from the current CPU.
    pub fn capture<M: MsrAccess>(msr_access: &M) -> Self {
        Self {
            star: Star::new(msr_access.read_msr(msr::IA32_STAR).unwrap_or(0)),
            lstar: Lstar::new(msr_access.read_msr(msr::IA32_LSTAR).unwrap_or(0)),
            cstar: Cstar::new(msr_access.read_msr(msr::IA32_CSTAR).unwrap_or(0)),
            fmask: Fmask::new(msr_access.read_msr(msr::IA32_FMASK).unwrap_or(0)),
        }
    }

    /// Write these SYSCALL MSRs to hardware (guest values before VM entry).
    pub fn load<M: MsrAccess>(&self, msr_access: &M) {
        let _ = msr_access.write_msr(msr::IA32_STAR, self.star.bits());
        let _ = msr_access.write_msr(msr::IA32_LSTAR, self.lstar.bits());
        let _ = msr_access.write_msr(msr::IA32_CSTAR, self.cstar.bits());
        let _ = msr_access.write_msr(msr::IA32_FMASK, self.fmask.bits());
    }
}

/// Size of the early-boot serial line accumulator. `OUT 0x3F8` bytes are
/// buffered and emitted as one `Serial` event per line (on `\n` or when full)
/// rather than one event header per character.
pub const SERIAL_LINE_ACC_SIZE: usize = 256;

/// All VM state except guest memory, shared by RootVm and ForkedVm.
#[repr(C)]
pub struct VmState<V: VirtualMachineControlStructure, I: InstructionCounter> {
    /// The Virtual Machine Control Structure.
    pub vmcs: V,
    /// Guest/host GPRs and launch state for VM entry/exit.
    pub vmx_ctx: VmxContext,
    /// Exit handlers' GPR view, synced with `vmx_ctx` around entry/exit.
    pub gprs: GeneralPurposeRegisters,
    /// EPT page table (GPA -> HPA).
    pub ept: EptPageTable<V::P>,
    /// MSR bitmap page (4KB, controls MSR access interception).
    pub msr_bitmap: V::P,
    /// VM-exit MSR-load page holding `IA32_PEBS_ENABLE = 0`, so PEBS is
    /// disabled atomically on exit and a skidding record can't fault on the
    /// host's stale `IA32_DS_AREA`. Count stays 0 until `register_pebs_page`.
    pub pebs_exit_msr_load_page: V::P,
    /// VM-entry MSR-load page for per-arming PEBS values (see
    /// `PEBS_ENTRY_MSR_INDEXES`). Loading them with VM-entry rather than WRMSR
    /// avoids re-enabling PEBS in host mode with a guest `IA32_DS_AREA`, which
    /// SMAP-faults.
    pub pebs_entry_msr_load_page: V::P,
    /// Early-boot serial line accumulator. See `SERIAL_LINE_ACC_SIZE`.
    pub serial_line_buf: SerialLineBufBox,
    /// Number of valid bytes in `serial_line_buf`.
    pub serial_line_len: usize,
    /// Emulated TSC at the line's first byte; stamps the emitted event.
    pub serial_line_tsc: u64,
    /// Host (raw RDTSC) captured at the first byte of the in-progress line.
    pub serial_line_real_tsc: u64,
    /// Guest XSAVE area page (4KB) for extended state (FPU/SSE/AVX) save/restore.
    pub guest_xsave_page: V::P,
    /// Host XSAVE area page (4KB).
    pub host_xsave_page: V::P,
    /// XCR0 mask for XSAVE/XRSTOR; 0 disables XSAVE.
    pub xcr0_mask: u64,
    /// Last exit qualification (saved when the run loop returns to userspace).
    pub last_exit_qualification: u64,
    /// Last guest physical address (saved during VM exit for EPT violations).
    pub last_guest_physical_addr: u64,
    /// Emulated device states (APIC, serial, IOAPIC, RTC, MTRR, RDRAND).
    pub devices: DeviceStatesBox,
    /// Host state captured at VM initialization (for guest MSR emulation).
    pub host_state: HostState,
    /// Grouped guest MSR state (PAT, TSC_AUX, SYSCALL MSRs).
    pub msr_state: GuestMsrState,
    /// IA32_KERNEL_GS_BASE (0xC0000102) - kernel GS base for SWAPGS.
    pub kernel_gs_base: u64,
    /// Instruction counter for deterministic execution.
    pub instruction_counter: I,
    /// Last instruction count read after VM exit.
    pub last_instruction_count: u64,
    /// Emulated TSC: `last_instruction_count + tsc_offset`.
    pub emulated_tsc: u64,
    /// Added to the instruction count; grows when HLT/MWAIT skips to a deadline.
    pub tsc_offset: u64,
    /// Configured TSC frequency in Hz.
    pub tsc_frequency: u64,
    /// Logging mode for deterministic exit capture.
    pub exit_trigger: ExitTrigger,
    /// AtTsc: target TSC. Checkpoints: interval.
    pub exit_target_tsc: u64,
    /// No logging in any mode until `emulated_tsc >= exit_start_tsc`; 0 = off.
    pub exit_start_tsc: u64,
    /// Single-point modes (AtTsc/AtShutdown) already logged.
    pub exit_captured: bool,

    // --- Unified event stream (see `crate::events`). ---
    /// 1 MB kernel-allocated event buffer, mmap'd to userspace. `None` until
    /// attached.
    pub event_buffer_ptr: Option<*mut u8>,
    /// Write cursor; always a multiple of 8 so each `EventHeader` is aligned.
    pub event_len: usize,
    /// Record sequence number, never reset on drain so userspace can detect
    /// gaps.
    pub event_seq: u64,
    /// Enabled categories (via ioctl). Empty by default; a disabled category
    /// costs one bit test at emit time.
    pub event_categories: EventCategories,
    /// One event staged because the buffer was full; `event_clear()`
    /// re-appends it after the drain.
    pub event_pending: Option<EventKind>,
    /// Payload bytes of the pending event.
    pub event_pending_buf: IoPageBufBox,
    /// Number of valid bytes in `event_pending_buf`.
    pub event_pending_len: usize,
    /// Header flags of the pending event.
    pub event_pending_flags: u16,
    /// Emulated TSC the pending event was originally stamped with.
    pub event_pending_tsc: u64,
    /// Host (raw RDTSC) the pending event was originally stamped with.
    pub event_pending_real_tsc: u64,
    /// Location of the last `Exit` record awaiting
    /// `finalize_exit_memory_hash`, if any.
    pub pending_exit_loc: Option<ExitLoc>,
    /// When true, skip memory hashing in exit records (memory_hash stays 0).
    pub skip_memory_hash: bool,
    /// TSC range for single-stepping (start, end). None means disabled.
    pub single_step_tsc_range: Option<(u64, u64)>,
    /// Whether MTF is currently enabled in VMCS.
    pub mtf_enabled: bool,
    /// Stop VM when emulated_tsc reaches this value. None means disabled.
    pub stop_at_tsc: Option<u64>,
    /// Exit handler performance statistics.
    pub exit_stats: ExitStatsBox,
    /// Last checkpoint index logged (Checkpoints mode).
    pub last_checkpoint_idx: u64,
    /// Last exit was deterministic, so `emulated_tsc` is current. Interrupt
    /// injection is skipped otherwise to avoid acting on a stale TSC.
    pub last_exit_deterministic: bool,
    /// PEBS diagnostics for the last PEBS-induced exit, copied into the next
    /// exit record and then reset to 0 by `write_exit_record`.
    ///
    /// Skid past the programmed firing point, in retired instructions.
    pub last_pebs_skid: i64,
    /// INST_RETIRED gain from arming to fire.
    pub last_pebs_inst_delta: i64,
    /// `tsc_offset` gain from arming to fire; expected 0.
    pub last_pebs_tsc_offset_delta: i64,
    /// Run-loop iterations the firing arming persisted across.
    pub last_pebs_iters_since_arm: u32,
    /// Firing target minus current TSC at arming time.
    pub last_pebs_arm_delta: u64,
    /// Guest-registered feedback buffers; the slot index is returned in RAX.
    pub feedback_buffers: FeedbackBuffers,
    /// VPID for this VM; 0 = none (VPID disabled or cargo/test mode).
    pub vpid: u16,
    /// Intercept guest #PF (logged and reinjected) for determinism analysis.
    pub intercept_pf: bool,
    /// PEBS state for precise exits; `None` until registered or when
    /// unsupported. See `exits/pebs.rs` and SDM Vol 3B 21.9.5.
    pub pebs_state: Option<HeapBox<PebsState>>,
    /// Host supports EPT-friendly PEBS (`PEBS_BASELINE` and `PEBS_FMT >= 4`).
    /// Cached because reading `IA32_PERF_CAPABILITIES` may `#GP`.
    pub pebs_supported: bool,
    /// Host can fall back to PMI-delivered precise exits. See
    /// `exits::pebs::host_supports_pmi_precise_exit`.
    pub pmi_supported: bool,
    /// Host supports `IA32_DEBUGCTL.FREEZE_WHILE_SMM` (`IA32_PERF_CAPABILITIES`
    /// bit 12), so the guest `DEBUGCTL` can freeze the counters across SMIs.
    pub freeze_perfmon_in_smm: bool,
    /// Hypervisor↔guest I/O channel. See `IoChannelState`.
    pub io_channel: IoChannelState,
    /// Paravirtual batch console. See `SerialConsoleState`.
    pub serial_console: SerialConsoleState,
    /// Logical CPU this VM last ran on; `run()` issues INVEPT when it changes.
    pub last_cpu: Option<u32>,
}

/// Error type for VmState creation.
#[derive(Debug)]
pub enum VmStateError<E> {
    /// EPT page table creation failed.
    EptCreation(E),
    /// MSR bitmap allocation failed.
    MsrBitmapAlloc,
    /// PEBS VM-exit MSR-load page allocation failed.
    PebsExitMsrLoadAlloc,
    /// XSAVE area page allocation failed.
    XsavePageAlloc,
    /// VMCS setup failed.
    VmcsSetup(VmcsSetupError),
    /// Guest state copy failed.
    GuestStateCopy,
    /// INVEPT failed during fork (EPT TLB invalidation).
    InveptFailed,
}

impl<V: VirtualMachineControlStructure, I: InstructionCounter> VmState<V, I> {
    /// Create a VmState: allocates the MSR bitmap, MSR-load and XSAVE pages,
    /// captures host state, and sets up the VMCS. `exit_handler_rip` becomes
    /// HOST_RIP.
    #[inline(never)]
    pub fn new<A: FrameAllocator<Frame = V::P>>(
        vmcs: V,
        ept: EptPageTable<V::P>,
        machine: &V::M,
        exit_handler_rip: u64,
        instruction_counter: I,
        tsc_frequency: u64,
    ) -> Result<Self, VmStateError<A::Error>> {
        let msr_bitmap = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::MsrBitmapAlloc)?;

        // Intercept all MSR accesses by default (SDM Vol 3C 26.6.9).
        let ptr = msr_bitmap.virtual_address().as_u64() as *mut u8;
        // SAFETY: ptr points to a freshly-allocated zeroed 4KB page; writing PAGE_SIZE bytes is within bounds.
        unsafe {
            core::ptr::write_bytes(ptr, 0xFF, PAGE_SIZE);
        }

        // PEBS VM-exit MSR-load page: one entry, `IA32_PEBS_ENABLE = 0`.
        // Dormant until PEBS registration sets the count (SDM Vol 3C 26.7.2).
        let pebs_exit_msr_load_page = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::PebsExitMsrLoadAlloc)?;
        let entry_ptr = pebs_exit_msr_load_page.virtual_address().as_u64() as *mut u32;
        // SAFETY: page is freshly allocated, zero-initialized, page-aligned;
        // writing 16 bytes at offset 0 is within bounds.
        unsafe {
            core::ptr::write(entry_ptr, msr::IA32_PEBS_ENABLE);
        }

        let pebs_entry_msr_load_page = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::PebsExitMsrLoadAlloc)?;

        // Passthrough MSRs that are saved/restored either by VMCS guest-state
        // fields or manually around entry/exit.
        msr_bitmap_clear_intercept(ptr, msr::IA32_FS_BASE); // FS_BASE
        msr_bitmap_clear_intercept(ptr, msr::IA32_GS_BASE); // GS_BASE

        // No VMCS field; saved/restored manually.
        msr_bitmap_clear_intercept(ptr, msr::IA32_KERNEL_GS_BASE);
        msr_bitmap_clear_intercept(ptr, msr::IA32_EFER); // IA32_EFER

        // SYSCALL MSRs: saved/restored manually.
        msr_bitmap_clear_intercept(ptr, msr::IA32_STAR);
        msr_bitmap_clear_intercept(ptr, msr::IA32_LSTAR);
        msr_bitmap_clear_intercept(ptr, msr::IA32_CSTAR);
        msr_bitmap_clear_intercept(ptr, msr::IA32_FMASK);

        msr_bitmap_clear_intercept(ptr, msr::IA32_SYSENTER_CS);
        msr_bitmap_clear_intercept(ptr, msr::IA32_SYSENTER_ESP);
        msr_bitmap_clear_intercept(ptr, msr::IA32_SYSENTER_EIP);

        let guest_xsave_page = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::XsavePageAlloc)?;
        let host_xsave_page = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::XsavePageAlloc)?;

        // Deterministic initial FPU/SSE state so FXSAVE/XSAVE results match.
        // SAFETY: guest_xsave_page is valid and 4KB aligned
        unsafe {
            let xsave_ptr = guest_xsave_page.virtual_address().as_u64() as *mut u8;

            // FCW at offset 0: FINIT default.
            let fcw: u16 = 0x037F;
            core::ptr::copy_nonoverlapping(fcw.to_le_bytes().as_ptr(), xsave_ptr, 2);

            // MXCSR at offset 24: reset default.
            let mxcsr: u32 = 0x1F80;
            core::ptr::copy_nonoverlapping(mxcsr.to_le_bytes().as_ptr(), xsave_ptr.add(24), 4);

            // XSTATE_BV at offset 512: components XRSTOR restores.
            let xstate_bv: u64 = xcr0::SSE_AVX;
            core::ptr::copy_nonoverlapping(xstate_bv.to_le_bytes().as_ptr(), xsave_ptr.add(512), 8);
        }

        let host_state = HostState::capture(
            machine.cr_access(),
            machine.msr_access(),
            machine.descriptor_table_access(),
            exit_handler_rip,
            // RSP for exit handler - set dynamically before VM entry
            0,
        );

        // A `#GP` on the read means unsupported (SDM Vol 3B 21.8).
        let pebs_supported = machine
            .msr_access()
            .read_msr(msr::IA32_PERF_CAPABILITIES)
            .is_ok_and(host_has_ept_friendly_pebs);
        let pmi_supported = host_supports_pmi_precise_exit(machine.msr_access());
        let freeze_perfmon_in_smm = machine
            .msr_access()
            .read_msr(msr::IA32_PERF_CAPABILITIES)
            .is_ok_and(|caps| caps & PERF_CAPABILITIES_FREEZE_WHILE_SMM != 0);

        vmcs.setup(ept.eptp(), Some(msr_bitmap.physical_address()), &host_state)
            .map_err(VmStateError::VmcsSetup)?;

        let vpid = vmcs.read16(VmcsField16::VirtualProcessorId).unwrap_or(0);

        // Flush stale translations from a previous VM that may have used the
        // same EPT root address.
        <V::M as Machine>::V::invept_single_context(ept.eptp())
            .map_err(|_| VmStateError::InveptFailed)?;

        Ok(Self {
            vmcs,
            vmx_ctx: VmxContext::new(),
            gprs: GeneralPurposeRegisters::default(),
            ept,
            msr_bitmap,
            pebs_exit_msr_load_page,
            pebs_entry_msr_load_page,
            serial_line_buf: box_serial_line_buf(),
            serial_line_len: 0,
            serial_line_tsc: 0,
            serial_line_real_tsc: 0,
            guest_xsave_page,
            host_xsave_page,
            xcr0_mask: xcr0::SSE_AVX,
            last_exit_qualification: 0,
            last_guest_physical_addr: 0,
            devices: heap_box(DeviceStates::default()),
            host_state,
            msr_state: GuestMsrState::new(),
            kernel_gs_base: 0,
            instruction_counter,
            last_instruction_count: 0,
            emulated_tsc: 0,
            tsc_offset: 0,
            tsc_frequency,
            exit_trigger: ExitTrigger::Disabled,
            exit_target_tsc: 0,
            exit_start_tsc: 0,
            exit_captured: false,
            event_buffer_ptr: None,
            event_len: 0,
            event_seq: 0,
            event_categories: EventCategories::empty(),
            event_pending: None,
            event_pending_buf: box_io_page_buf(),
            event_pending_len: 0,
            event_pending_flags: 0,
            event_pending_tsc: 0,
            event_pending_real_tsc: 0,
            pending_exit_loc: None,
            skip_memory_hash: false,
            single_step_tsc_range: None,
            mtf_enabled: false,
            stop_at_tsc: None,
            exit_stats: heap_box(AllExitStats::default()),
            last_checkpoint_idx: 0,
            last_exit_deterministic: true,
            last_pebs_skid: 0,
            last_pebs_inst_delta: 0,
            last_pebs_tsc_offset_delta: 0,
            last_pebs_iters_since_arm: 0,
            last_pebs_arm_delta: 0,
            feedback_buffers: feedback_buffers_new(),
            vpid,
            intercept_pf: false,
            pebs_state: None,
            pebs_supported,
            pmi_supported,
            freeze_perfmon_in_smm,
            io_channel: IoChannelState::new(),
            serial_console: SerialConsoleState::new(),
            last_cpu: None,
        })
    }

    // --- Unified event stream (see `crate::events`). ---

    /// Attach the kernel-allocated event buffer (1 MB, mmap'd to userspace).
    pub fn set_event_buffer(&mut self, ptr: *mut u8) {
        self.event_buffer_ptr = Some(ptr);
    }

    /// Detach the event buffer and reset the cursor (e.g. on disable).
    pub fn clear_event_buffer_ptr(&mut self) {
        self.event_buffer_ptr = None;
        self.event_len = 0;
        self.event_pending = None;
        self.event_pending_len = 0;
    }

    /// Number of valid bytes in the event buffer.
    pub fn event_buffer_len(&self) -> usize {
        self.event_len
    }

    /// True if an event was staged because the buffer filled (forces a drain).
    pub fn event_buffer_full(&self) -> bool {
        self.event_pending.is_some()
    }

    /// Returns the event buffer virtual address for mmap.
    pub fn event_buffer_ptr(&self) -> Option<*mut u8> {
        self.event_buffer_ptr
    }

    /// Set the userspace category include/exclude mask (from ioctl).
    pub fn set_event_categories(&mut self, categories: EventCategories) {
        self.event_categories = categories;
    }

    /// Current category mask.
    pub fn event_categories(&self) -> EventCategories {
        self.event_categories
    }

    /// True if `kind`'s category is enabled.
    pub fn event_category_enabled(&self, kind: EventKind) -> bool {
        self.event_categories.contains(kind.category())
    }

    /// Reset the cursor after a drain and re-append the staged event, if any.
    /// Called at the start of every RUN ioctl.
    pub fn event_clear(&mut self) {
        self.event_len = 0;
        if let Some(kind) = self.event_pending.take() {
            let len = self.event_pending_len;
            let flags = self.event_pending_flags;
            let tsc = self.event_pending_tsc;
            let real_tsc = self.event_pending_real_tsc;
            // `event_pending_buf` is a separate allocation from the event buffer.
            let src = self.event_pending_buf.as_ptr();
            // SAFETY: `src` is valid for `len` bytes (set when the event was
            // staged); the now-empty buffer has room for it.
            unsafe {
                self.event_write(kind, flags, tsc, real_tsc, src, len, core::ptr::null(), 0);
            }
            // Relocate a still-pending memory_hash patch to the buffer.
            if kind == EventKind::Exit && matches!(self.pending_exit_loc, Some(ExitLoc::Pending)) {
                self.pending_exit_loc = Some(ExitLoc::Buffer(self.event_len_at_last_record()));
            }
            self.event_pending_len = 0;
        }
    }

    /// Payload offset of the record just re-appended by `event_clear`.
    fn event_len_at_last_record(&self) -> usize {
        let total = EVENT_HEADER_SIZE + align_up(self.event_pending_len, 8);
        self.event_len - total + EVENT_HEADER_SIZE
    }

    /// Append one event stamped with the current emulated/host TSC.
    ///
    /// Returns `false` if the buffer was full: the payload is staged and the
    /// caller must advance RIP and exit to userspace to drain.
    pub fn event_append(&mut self, kind: EventKind, payload: &[u8]) -> bool {
        // Skip the host TSC read when filtered (the common case).
        if !self.event_categories.contains(kind.category()) {
            return true;
        }
        let tsc = self.emulated_tsc;
        let real_tsc = rdtsc();
        // SAFETY: `payload` is a valid slice that never aliases the event buffer.
        unsafe {
            self.event_write(
                kind,
                kind.default_flags(),
                tsc,
                real_tsc,
                payload.as_ptr(),
                payload.len(),
                core::ptr::null(),
                0,
            )
        }
    }

    /// Append one event stamped with an explicit TSC.
    pub fn event_append_at(
        &mut self,
        kind: EventKind,
        payload: &[u8],
        tsc: u64,
        real_tsc: u64,
    ) -> bool {
        // SAFETY: as in `event_append`. `event_write` applies the category filter.
        unsafe {
            self.event_write(
                kind,
                kind.default_flags(),
                tsc,
                real_tsc,
                payload.as_ptr(),
                payload.len(),
                core::ptr::null(),
                0,
            )
        }
    }

    /// Write an [`EventHeader`] then the `head` and `tail` payload parts
    /// contiguously, padded to 8 bytes.
    ///
    /// # Safety
    ///
    /// `head_ptr`/`tail_ptr` must be valid for `head_len`/`tail_len` reads and
    /// must not alias the event buffer. `tail_ptr` may be null when
    /// `tail_len == 0`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn event_write(
        &mut self,
        kind: EventKind,
        flags: u16,
        tsc: u64,
        real_tsc: u64,
        head_ptr: *const u8,
        head_len: usize,
        tail_ptr: *const u8,
        tail_len: usize,
    ) -> bool {
        if !self.event_categories.contains(kind.category()) {
            return true;
        }
        let base = match self.event_buffer_ptr {
            Some(p) => p,
            None => return true, // no buffer attached: drop silently
        };
        let len = head_len + tail_len;
        let need = EVENT_HEADER_SIZE + align_up(len, 8);
        if self.event_len + need > EVENT_BUFFER_SIZE {
            // Stage the payload for re-append after the drain. If one is already
            // staged, drop this record rather than clobber it.
            if self.event_pending.is_none() {
                let cap = IO_CHANNEL_BUF_SIZE;
                let head_copy = head_len.min(cap);
                let tail_copy = tail_len.min(cap - head_copy);
                let dst = self.event_pending_buf.as_mut_ptr().cast::<u8>();
                // SAFETY: sources are valid for `head_copy`/`tail_copy` bytes;
                // `dst` is a distinct buffer with room for `head_copy + tail_copy
                // <= cap`. The tail copy is skipped when `tail_ptr` may be null.
                unsafe {
                    core::ptr::copy_nonoverlapping(head_ptr, dst, head_copy);
                    if tail_copy > 0 {
                        core::ptr::copy_nonoverlapping(tail_ptr, dst.add(head_copy), tail_copy);
                    }
                }
                self.event_pending = Some(kind);
                self.event_pending_len = head_copy + tail_copy;
                self.event_pending_flags = flags;
                self.event_pending_tsc = tsc;
                self.event_pending_real_tsc = real_tsc;
            }
            return false;
        }
        let header = EventHeader {
            seq: self.event_seq,
            tsc,
            real_tsc,
            kind: kind.as_u16(),
            flags,
            len: len as u32,
        };
        let padded = align_up(len, 8);
        // SAFETY: `event_len + need <= EVENT_BUFFER_SIZE` (checked above), so
        // header, payload and padding fit. `base` is page-aligned and `event_len`
        // a multiple of 8, so the header is aligned. Sources are valid and
        // distinct from the buffer per this function's contract.
        unsafe {
            let rec = base.add(self.event_len);
            core::ptr::write(rec.cast::<EventHeader>(), header);
            core::ptr::copy_nonoverlapping(head_ptr, rec.add(EVENT_HEADER_SIZE), head_len);
            if tail_len > 0 {
                core::ptr::copy_nonoverlapping(
                    tail_ptr,
                    rec.add(EVENT_HEADER_SIZE + head_len),
                    tail_len,
                );
            }
            // Zero padding so it never leaks host memory or varies across runs.
            core::ptr::write_bytes(rec.add(EVENT_HEADER_SIZE + len), 0, padded - len);
        }
        self.event_seq = self.event_seq.wrapping_add(1);
        self.event_len += need;
        true
    }

    /// Accumulate one early-boot serial byte, emitting a `Serial` event stamped
    /// with the line-start TSC on newline or when full. No-op when `Serial` is
    /// disabled. Returns `false` if the event buffer filled.
    pub fn event_serial_byte(&mut self, byte: u8) -> bool {
        if !self.event_categories.contains(EventCategories::SERIAL) {
            return true;
        }
        if self.serial_line_len == 0 {
            self.serial_line_tsc = self.emulated_tsc;
            self.serial_line_real_tsc = rdtsc();
        }
        if self.serial_line_len < SERIAL_LINE_ACC_SIZE {
            self.serial_line_buf[self.serial_line_len] = byte;
            self.serial_line_len += 1;
        }
        if byte == b'\n' || self.serial_line_len == SERIAL_LINE_ACC_SIZE {
            return self.event_flush_serial_line();
        }
        true
    }

    /// Emit and reset the accumulated early-boot serial line, if any (also
    /// called before paravirt writes and at shutdown so partial lines survive).
    /// Returns `false` if the event buffer filled.
    pub fn event_flush_serial_line(&mut self) -> bool {
        if self.serial_line_len == 0 {
            return true;
        }
        let len = self.serial_line_len;
        let tsc = self.serial_line_tsc;
        let real_tsc = self.serial_line_real_tsc;
        // Stack copy so the payload does not borrow `self`.
        let mut tmp = [0u8; SERIAL_LINE_ACC_SIZE];
        tmp[..len].copy_from_slice(&self.serial_line_buf[..len]);
        self.serial_line_len = 0;
        self.event_append_at(EventKind::Serial, &tmp[..len], tsc, real_tsc)
    }

    /// Emit `serial_console.pending_buf[..len]` as one `Serial` event. Returns
    /// `false` if the event buffer filled.
    pub fn event_emit_console(&mut self, len: usize) -> bool {
        if !self.event_categories.contains(EventCategories::SERIAL) {
            return true;
        }
        let tsc = self.emulated_tsc;
        let real_tsc = rdtsc();
        let src = self.serial_console.pending_buf.as_ptr();
        let len = len.min(SERIAL_CONSOLE_PAGE_SIZE);
        // SAFETY: `src` is valid for `len` <= SERIAL_CONSOLE_PAGE_SIZE bytes and
        // does not alias the event buffer.
        unsafe {
            self.event_write(
                EventKind::Serial,
                EventKind::Serial.default_flags(),
                tsc,
                real_tsc,
                src,
                len,
                core::ptr::null(),
                0,
            )
        }
    }

    /// Emit one `IoChannel` event: [`IoChannelPayload`] followed by the request
    /// or response bytes. Only `Request` records are flagged deterministic;
    /// response bytes are host-derived. Returns `false` if the buffer filled.
    pub fn event_emit_io_channel(&mut self, payload: &IoChannelPayload) -> bool {
        if !self.event_categories.contains(EventCategories::IO_CHANNEL) {
            return true;
        }
        let is_response = payload.phase == IoChannelPhase::Response as u8;
        let (data_ptr, data_len) = if is_response {
            (
                self.io_channel.response_buf.as_ptr(),
                self.io_channel.response_len,
            )
        } else {
            (
                self.io_channel.request_buf.as_ptr(),
                self.io_channel.request_len,
            )
        };
        let flags = if is_response {
            0
        } else {
            EVENT_FLAG_DETERMINISTIC
        };
        let tsc = self.emulated_tsc;
        let real_tsc = rdtsc();
        let data_len = data_len.min(IO_CHANNEL_BUF_SIZE);
        // SAFETY: `payload` is caller-owned; `data_ptr` points into a distinct
        // heap buffer valid for `data_len` bytes; neither aliases the event buffer.
        unsafe {
            self.event_write(
                EventKind::IoChannel,
                flags,
                tsc,
                real_tsc,
                payload.as_bytes().as_ptr(),
                core::mem::size_of::<IoChannelPayload>(),
                data_ptr,
                data_len,
            )
        }
    }

    /// Emit one `Exit` event and remember where it landed for
    /// `finalize_exit_memory_hash`.
    ///
    /// # Safety
    ///
    /// `payload_ptr` must point to a valid `ExitRecord` of `len` bytes that does
    /// not alias the event buffer.
    unsafe fn emit_exit_event(&mut self, payload_ptr: *const u8, len: usize, deterministic: bool) {
        self.pending_exit_loc = None;
        if self.event_buffer_ptr.is_none() || !self.event_categories.contains(EventCategories::EXIT)
        {
            return;
        }
        let flags = if deterministic {
            EVENT_FLAG_DETERMINISTIC
        } else {
            0
        };
        let tsc = self.emulated_tsc;
        let real_tsc = rdtsc();
        let payload_off = self.event_len + EVENT_HEADER_SIZE;
        // SAFETY: forwarded from this function's contract.
        let fit = unsafe {
            self.event_write(
                EventKind::Exit,
                flags,
                tsc,
                real_tsc,
                payload_ptr,
                len,
                core::ptr::null(),
                0,
            )
        };
        self.pending_exit_loc = Some(if fit {
            ExitLoc::Buffer(payload_off)
        } else {
            ExitLoc::Pending
        });
    }

    /// Patch `memory_hash` and `cow_page_count` into the last emitted `Exit`
    /// record, once guest memory has stabilized. No-op if none is pending.
    pub fn finalize_exit_memory_hash(&mut self, memory_hash: u64, cow_page_count: u32) {
        let loc = match self.pending_exit_loc.take() {
            Some(l) => l,
            None => return,
        };
        let mh_off = core::mem::offset_of!(ExitRecord, memory_hash);
        let cc_off = core::mem::offset_of!(ExitRecord, cow_page_count);
        let payload_base: *mut u8 = match loc {
            ExitLoc::Buffer(payload_off) => match self.event_buffer_ptr {
                // SAFETY: `payload_off` is within the 1 MB buffer (the record
                // was written there).
                Some(base) => unsafe { base.add(payload_off) },
                None => return,
            },
            // Patch the staged copy so the re-appended record carries the hash.
            ExitLoc::Pending => self.event_pending_buf.as_mut_ptr().cast::<u8>(),
        };
        // SAFETY: both fields lie within the 512-byte payload, and
        // `payload_base` is 8-aligned (aligned record + 32-byte header).
        unsafe {
            core::ptr::write(payload_base.add(mh_off).cast::<u64>(), memory_hash);
            core::ptr::write(payload_base.add(cc_off).cast::<u32>(), cow_page_count);
        }
    }

    /// Check if logging is enabled (any mode except Disabled).
    pub fn exit_capture_enabled(&self) -> bool {
        self.exit_trigger != ExitTrigger::Disabled
    }

    /// Enable deterministic logging in AllExits mode.
    pub fn enable_exit_capture(&mut self) {
        self.exit_trigger = ExitTrigger::AllExits;
        self.exit_captured = false;
    }

    /// Disable deterministic logging.
    pub fn disable_exit_capture(&mut self) {
        self.exit_trigger = ExitTrigger::Disabled;
        self.exit_captured = false;
    }

    /// Set the logging mode and `exit_target_tsc` (see [`ExitTrigger`]).
    pub fn set_exit_trigger(&mut self, mode: ExitTrigger, target_tsc: u64) {
        self.exit_trigger = mode;
        self.exit_target_tsc = target_tsc;
        self.exit_captured = false;
    }

    /// Get the current logging mode.
    pub fn exit_trigger(&self) -> ExitTrigger {
        self.exit_trigger
    }

    /// Set the logging start threshold for all modes (0 = log from start).
    pub fn set_exit_start_tsc(&mut self, start_tsc: u64) {
        self.exit_start_tsc = start_tsc;
    }

    /// Set the #PF interception flag; `apply_intercept_pf()` writes it to the
    /// VMCS once loaded.
    pub fn set_intercept_pf(&mut self, enable: bool) {
        self.intercept_pf = enable;
    }

    /// Apply the #PF interception flag to the exception bitmap. Must be called
    /// after `vmcs.load()`.
    pub fn apply_intercept_pf(&self) {
        let bitmap = self.vmcs.read32(VmcsField32::ExceptionBitmap).unwrap_or(0);
        let new_bitmap = if self.intercept_pf {
            bitmap | (1 << 14)
        } else {
            bitmap & !(1 << 14)
        };
        let _ = self.vmcs.write32(VmcsField32::ExceptionBitmap, new_bitmap);
    }

    /// Record this VM exit if the current [`ExitTrigger`] selects it. AtTsc and
    /// Checkpoints only log deterministic exits; AtShutdown is handled by
    /// `capture_exit_at_shutdown`.
    pub fn capture_exit(
        &mut self,
        exit_reason: ExitReason,
        exit_qualification: u64,
        deterministic: bool,
    ) {
        if self.exit_start_tsc > 0 && self.emulated_tsc < self.exit_start_tsc {
            return;
        }

        match self.exit_trigger {
            ExitTrigger::Disabled => return,
            ExitTrigger::AtShutdown => return, // Handled by capture_exit_at_shutdown()
            ExitTrigger::Checkpoints => {
                if !deterministic {
                    return;
                }
                let interval = self.exit_target_tsc;
                if interval == 0 {
                    return;
                }

                let checkpoint_idx = self.emulated_tsc / interval;
                if checkpoint_idx > self.last_checkpoint_idx {
                    self.last_checkpoint_idx = checkpoint_idx;
                } else {
                    return; // Not yet reached next checkpoint
                }
            }
            ExitTrigger::AtTsc => {
                if !deterministic {
                    return;
                }
                if self.exit_captured || self.emulated_tsc < self.exit_target_tsc {
                    return;
                }
            }
            ExitTrigger::AllExits => {}
            ExitTrigger::TscRange => {
                // Non-deterministic exits are included: they are essential for
                // diagnosing divergences.
                if let Some((start, end)) = self.single_step_tsc_range {
                    if self.emulated_tsc < start || self.emulated_tsc >= end {
                        return;
                    }
                } else {
                    return; // No range configured
                }
            }
        }

        let flags = if deterministic {
            EXIT_RECORD_FLAG_DETERMINISTIC
        } else {
            0
        };
        self.write_exit_record(exit_reason, exit_qualification, flags);

        if self.exit_trigger == ExitTrigger::AtTsc {
            self.exit_captured = true;
        }
    }

    /// Record final state at vmcall shutdown (AtShutdown mode, once).
    pub fn capture_exit_at_shutdown(&mut self) {
        if self.exit_trigger != ExitTrigger::AtShutdown || self.exit_captured {
            return;
        }

        if self.exit_start_tsc > 0 && self.emulated_tsc < self.exit_start_tsc {
            return;
        }

        self.write_exit_record(
            ExitReason::VmcallShutdown,
            0,
            EXIT_RECORD_FLAG_DETERMINISTIC,
        );
        self.exit_captured = true;
    }

    /// Record state for a snapshot hypercall (no-op when logging is disabled).
    pub fn capture_exit_at_snapshot(&mut self) {
        if self.exit_start_tsc > 0 && self.emulated_tsc < self.exit_start_tsc {
            return;
        }

        if self.exit_trigger == ExitTrigger::Disabled {
            return;
        }

        self.write_exit_record(
            ExitReason::VmcallSnapshot,
            0,
            EXIT_RECORD_FLAG_DETERMINISTIC,
        );
    }

    /// Build an `ExitRecord` (registers, device hashes, diagnostics) and emit
    /// it as an `Exit` event. `memory_hash` is patched later by
    /// `finalize_exit_memory_hash`.
    fn write_exit_record(&mut self, exit_reason: ExitReason, exit_qualification: u64, flags: u32) {
        if self.event_buffer_ptr.is_none() || !self.event_categories.contains(EventCategories::EXIT)
        {
            return;
        }

        let rip = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestRip)
            .unwrap_or(0);
        let rflags = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestRflags)
            .unwrap_or(0);
        let fs_base = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestFsBase)
            .unwrap_or(0);
        let gs_base = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestGsBase)
            .unwrap_or(0);
        let cr3 = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestCr3)
            .unwrap_or(0);
        let cs_base = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestCsBase)
            .unwrap_or(0);
        let ds_base = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestDsBase)
            .unwrap_or(0);
        let es_base = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestEsBase)
            .unwrap_or(0);
        let ss_base = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestSsBase)
            .unwrap_or(0);
        let pending_dbg_exceptions = self
            .vmcs
            .read_natural(VmcsFieldNatural::GuestPendingDebugExceptions)
            .unwrap_or(0);
        let interruptibility_state = self
            .vmcs
            .read32(VmcsField32::GuestInterruptibilityState)
            .unwrap_or(0);

        let apic_hash = self.devices.apic.state_hash();
        let serial_hash = self.devices.serial.state_hash();
        let ioapic_hash = self.devices.ioapic.state_hash();
        let rtc_hash = self.devices.rtc.state_hash();
        let mtrr_hash = self.devices.mtrr.state_hash();
        // Randomness device (RDRAND/RDSEED + GET_RANDOM): mode, PRNG position,
        // staged value.
        let rdrand_hash = self.devices.random.state_hash();

        // Patched later by `finalize_exit_memory_hash`.
        let memory_hash = 0;

        let entry = ExitRecord {
            tsc: self.emulated_tsc,
            exit_reason: exit_reason as u32,
            flags,
            exit_qualification,
            rax: self.gprs.rax,
            rcx: self.gprs.rcx,
            rdx: self.gprs.rdx,
            rbx: self.gprs.rbx,
            rsp: self.gprs.rsp,
            rbp: self.gprs.rbp,
            rsi: self.gprs.rsi,
            rdi: self.gprs.rdi,
            r8: self.gprs.r8,
            r9: self.gprs.r9,
            r10: self.gprs.r10,
            r11: self.gprs.r11,
            r12: self.gprs.r12,
            r13: self.gprs.r13,
            r14: self.gprs.r14,
            r15: self.gprs.r15,
            rip,
            rflags,
            apic_hash,
            serial_hash,
            ioapic_hash,
            rtc_hash,
            mtrr_hash,
            rdrand_hash,
            memory_hash,
            fs_base,
            gs_base,
            kernel_gs_base: self.kernel_gs_base,
            cr3,
            cs_base,
            ds_base,
            es_base,
            ss_base,
            pending_dbg_exceptions,
            interruptibility_state,
            cow_page_count: 0,
            pebs_skid: self.last_pebs_skid,
            pebs_inst_delta: self.last_pebs_inst_delta,
            pebs_tsc_offset_delta: self.last_pebs_tsc_offset_delta,
            pebs_iters_since_arm: self.last_pebs_iters_since_arm,
            pebs_arm_delta: self.last_pebs_arm_delta,
            last_instruction_count: self.last_instruction_count,
            apic_timer_deadline: self.devices.apic.timer_deadline,
            io_channel_target_tsc: self.io_channel.request_target_tsc,
            pebs_armed_target_tsc: self
                .pebs_state
                .as_deref()
                .map(|p| p.armed_target_tsc)
                .unwrap_or(0),
            vmx_state_flags: u64::from(self.mtf_enabled)
                | (u64::from(self.last_exit_deterministic) << 1),
            _padding: [0; 16],
        };
        // PEBS diagnostics belong only to the record that captured them.
        self.last_pebs_skid = 0;
        self.last_pebs_inst_delta = 0;
        self.last_pebs_tsc_offset_delta = 0;
        self.last_pebs_iters_since_arm = 0;
        self.last_pebs_arm_delta = 0;

        // The header's determinism bit mirrors `ExitRecord.flags`.
        let deterministic = flags & EXIT_RECORD_FLAG_DETERMINISTIC != 0;
        // SAFETY: `entry` is a stack-local `ExitRecord` not aliasing the buffer.
        unsafe {
            self.emit_exit_event(
                core::ptr::from_ref(&entry).cast::<u8>(),
                core::mem::size_of::<ExitRecord>(),
                deterministic,
            );
        }
    }

    /// Create a minimally initialized VmState for tests (empty EPT, zeroed
    /// pages, default device and MSR state).
    #[cfg(test)]
    pub fn new_mock<A: FrameAllocator<Frame = V::P>, K: Kernel<P = V::P>>(
        vmcs: V,
        allocator: &mut A,
        kernel: &K,
        instruction_counter: I,
    ) -> Result<Self, &'static str> {
        let ept: EptPageTable<V::P> =
            EptPageTable::new(allocator).map_err(|_| "EPT creation failed")?;

        let msr_bitmap = kernel
            .alloc_zeroed_page()
            .ok_or("MSR bitmap alloc failed")?;
        let guest_xsave_page = kernel
            .alloc_zeroed_page()
            .ok_or("Guest XSAVE alloc failed")?;
        let host_xsave_page = kernel
            .alloc_zeroed_page()
            .ok_or("Host XSAVE alloc failed")?;
        let pebs_exit_msr_load_page = kernel
            .alloc_zeroed_page()
            .ok_or("PEBS exit MSR load page alloc failed")?;
        let pebs_entry_msr_load_page = kernel
            .alloc_zeroed_page()
            .ok_or("PEBS entry MSR load page alloc failed")?;

        Ok(Self {
            vmcs,
            vmx_ctx: VmxContext::new(),
            gprs: GeneralPurposeRegisters::default(),
            ept,
            msr_bitmap,
            pebs_exit_msr_load_page,
            pebs_entry_msr_load_page,
            serial_line_buf: box_serial_line_buf(),
            serial_line_len: 0,
            serial_line_tsc: 0,
            serial_line_real_tsc: 0,
            guest_xsave_page,
            host_xsave_page,
            xcr0_mask: 0x7, // x87 + SSE + AVX
            last_exit_qualification: 0,
            last_guest_physical_addr: 0,
            devices: heap_box(DeviceStates::default()),
            host_state: HostState::default(),
            msr_state: GuestMsrState::new(),
            kernel_gs_base: 0,
            instruction_counter,
            last_instruction_count: 0,
            emulated_tsc: 0,
            tsc_offset: 0,
            tsc_frequency: DEFAULT_TSC_FREQUENCY,
            exit_trigger: ExitTrigger::Disabled,
            exit_target_tsc: 0,
            exit_start_tsc: 0,
            exit_captured: false,
            event_buffer_ptr: None,
            event_len: 0,
            event_seq: 0,
            event_categories: EventCategories::empty(),
            event_pending: None,
            event_pending_buf: box_io_page_buf(),
            event_pending_len: 0,
            event_pending_flags: 0,
            event_pending_tsc: 0,
            event_pending_real_tsc: 0,
            pending_exit_loc: None,
            skip_memory_hash: false,
            single_step_tsc_range: None,
            mtf_enabled: false,
            stop_at_tsc: None,
            exit_stats: heap_box(AllExitStats::default()),
            last_checkpoint_idx: 0,
            last_exit_deterministic: true,
            last_pebs_skid: 0,
            last_pebs_inst_delta: 0,
            last_pebs_tsc_offset_delta: 0,
            last_pebs_iters_since_arm: 0,
            last_pebs_arm_delta: 0,
            feedback_buffers: feedback_buffers_new(),
            vpid: 0, // Tests don't use VPID
            intercept_pf: false,
            pebs_state: None,
            pebs_supported: false,
            pmi_supported: false,
            freeze_perfmon_in_smm: false,
            io_channel: IoChannelState::new(),
            serial_console: SerialConsoleState::new(),
            last_cpu: None,
        })
    }

    /// Create a forked VM's state from `parent_state`. The VMCS region is
    /// memcpy'd, which assumes parent and child share the same VMCS revision
    /// and (implementation-specific) format; per-child fields are then
    /// rewritten. `ept` is the already-cloned R+X EPT.
    #[inline(never)]
    pub fn new_for_fork<A: FrameAllocator<Frame = V::P>, I2: InstructionCounter>(
        vmcs: V,
        ept: EptPageTable<V::P>,
        parent_state: &VmState<V, I2>,
        machine: &V::M,
        _exit_handler_rip: u64,
        instruction_counter: I,
    ) -> Result<Self, VmStateError<A::Error>>
    where
        V::M: Machine,
    {
        let msr_bitmap = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::MsrBitmapAlloc)?;

        let parent_bitmap_ptr = parent_state.msr_bitmap.virtual_address().as_u64() as *const u8;
        let bitmap_ptr = msr_bitmap.virtual_address().as_u64() as *mut u8;
        // SAFETY: Both pointers refer to valid PAGE_SIZE allocations and do not overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(parent_bitmap_ptr, bitmap_ptr, PAGE_SIZE);
        }

        // Child-owned PEBS exit-load page; the copied VMCS is repointed at it
        // below.
        let pebs_exit_msr_load_page = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::PebsExitMsrLoadAlloc)?;
        let entry_ptr = pebs_exit_msr_load_page.virtual_address().as_u64() as *mut u32;
        // SAFETY: page is freshly allocated and zero-initialized; writing 4 bytes at
        // offset 0 stays within the page.
        unsafe {
            core::ptr::write(entry_ptr, msr::IA32_PEBS_ENABLE);
        }
        let pebs_entry_msr_load_page = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::PebsExitMsrLoadAlloc)?;

        let guest_xsave_page = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::XsavePageAlloc)?;
        let host_xsave_page = machine
            .kernel()
            .alloc_zeroed_page()
            .ok_or(VmStateError::XsavePageAlloc)?;

        let parent_xsave_ptr =
            parent_state.guest_xsave_page.virtual_address().as_u64() as *const u8;
        let guest_xsave_ptr = guest_xsave_page.virtual_address().as_u64() as *mut u8;
        // SAFETY: Both pointers refer to valid PAGE_SIZE allocations and do not overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(parent_xsave_ptr, guest_xsave_ptr, PAGE_SIZE);
        }

        #[allow(unused_mut)] // mut needed in kernel mode but not cargo mode
        let mut allocated_vpid: u16 = 0;

        // Mock VMCSes (cargo) are HashMaps, so the region copy is kernel-only.
        #[cfg(not(feature = "cargo"))]
        {
            // VMCLEAR flushes the parent's VMCS data to memory.
            parent_state
                .vmcs
                .clear()
                .map_err(|_| VmStateError::GuestStateCopy)?;

            // Parent's vmx_ctx.launched is not reset: the parent must not run
            // while forks are active.

            // SAFETY: Both VMCS region pointers are valid PAGE_SIZE allocations.
            // Parent VMCS was cleared (flushed to memory) above, so the copy is coherent.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    parent_state.vmcs.vmcs_region_ptr(),
                    vmcs.vmcs_region_ptr(),
                    PAGE_SIZE,
                );
            }

            vmcs.load().map_err(|_| VmStateError::GuestStateCopy)?;

            vmcs.write64(VmcsField64::EptPointer, ept.eptp())
                .map_err(|_| VmStateError::GuestStateCopy)?;
            vmcs.write64(
                VmcsField64::MsrBitmapAddr,
                msr_bitmap.physical_address().as_u64(),
            )
            .map_err(|_| VmStateError::GuestStateCopy)?;
            // Don't inherit the parent's partially counted-down timer.
            vmcs.write32(VmcsField32::VmxPreemptionTimerValue, 0x100000)
                .map_err(|_| VmStateError::GuestStateCopy)?;

            // The copied exit-load address points at the parent's page (the
            // parent may free it first), and `register_pebs_page` won't run
            // again. Entry-load fields are rewritten every iteration.
            // PMI mode uses no exit-load list.
            if parent_state
                .pebs_state
                .as_deref()
                .is_some_and(|p| p.mode == PreciseExitMode::Pebs)
            {
                vmcs.write64(
                    VmcsField64::VmExitMsrLoadAddr,
                    pebs_exit_msr_load_page.physical_address().as_u64(),
                )
                .map_err(|_| VmStateError::GuestStateCopy)?;
                vmcs.write32(VmcsField32::VmExitMsrLoadCount, 1)
                    .map_err(|_| VmStateError::GuestStateCopy)?;
            }

            // A fresh VPID, so the child doesn't share the parent's tagged TLB
            // entries.
            let current_exec2 = vmcs
                .read32(VmcsField32::SecondaryProcBasedVmExecControls)
                .unwrap_or(0);
            allocated_vpid = if current_exec2 & secondary_exec::ENABLE_VPID != 0 {
                let vpid = allocate_vpid();
                vmcs.write16(VmcsField16::VirtualProcessorId, vpid)
                    .map_err(|_| VmStateError::GuestStateCopy)?;

                // Flush entries from a previous user of this VPID.
                let _ = <V::M as Machine>::V::invvpid_single_context(vpid);

                log_info!("Forked VM allocated VPID={}\n", vpid);
                vpid
            } else {
                0
            };

            // Stale cached translations could let the child read parent pages
            // or miss CoW exits.
            <V::M as Machine>::V::invept_single_context(ept.eptp())
                .map_err(|_| VmStateError::InveptFailed)?;

            // The copied launch state is "launched"; VMLAUNCH needs "clear".
            vmcs.clear().map_err(|_| VmStateError::GuestStateCopy)?;
        }

        log_info!(
            "Forked VM created parent_tsc={} (offset={}, instrs={})\n",
            parent_state.emulated_tsc,
            parent_state.tsc_offset,
            parent_state.last_instruction_count,
        );

        Ok(Self {
            vmcs,
            vmx_ctx: VmxContext::new(),
            gprs: parent_state.gprs, // Copy GPRs from parent
            ept,
            msr_bitmap,
            pebs_exit_msr_load_page,
            pebs_entry_msr_load_page,
            serial_line_buf: box_serial_line_buf(),
            serial_line_len: 0,
            serial_line_tsc: 0,
            serial_line_real_tsc: 0,
            guest_xsave_page,
            host_xsave_page,
            xcr0_mask: parent_state.xcr0_mask,
            last_exit_qualification: 0,
            last_guest_physical_addr: 0,
            devices: heap_box((*parent_state.devices).clone()),
            host_state: parent_state.host_state.clone(), // Copy host state from parent
            msr_state: parent_state.msr_state,           // Copy MSR state
            kernel_gs_base: parent_state.kernel_gs_base,
            instruction_counter,
            last_instruction_count: 0, // Child's counter starts from 0
            emulated_tsc: parent_state.emulated_tsc,
            tsc_offset: parent_state.emulated_tsc,
            tsc_frequency: parent_state.tsc_frequency,
            exit_trigger: ExitTrigger::Disabled, // Forked VMs start with logging disabled
            exit_target_tsc: 0,
            exit_start_tsc: 0,
            exit_captured: false,
            event_buffer_ptr: None,
            event_len: 0,
            event_seq: 0,
            event_categories: EventCategories::empty(),
            event_pending: None,
            event_pending_buf: box_io_page_buf(),
            event_pending_len: 0,
            event_pending_flags: 0,
            event_pending_tsc: 0,
            event_pending_real_tsc: 0,
            pending_exit_loc: None,
            skip_memory_hash: false,
            single_step_tsc_range: None,
            mtf_enabled: false,
            stop_at_tsc: None,
            exit_stats: heap_box(AllExitStats::default()), // Forked VMs start with fresh stats
            last_checkpoint_idx: 0, // Forked VMs start checkpoint tracking fresh
            last_exit_deterministic: true,
            last_pebs_skid: 0,
            last_pebs_inst_delta: 0,
            last_pebs_tsc_offset_delta: 0,
            last_pebs_iters_since_arm: 0,
            last_pebs_arm_delta: 0,
            feedback_buffers: feedback_buffers_from(&parent_state.feedback_buffers), // Deep-copy parent's feedback buffers
            vpid: allocated_vpid,
            intercept_pf: false,
            // Inherit PEBS registration: the child never re-issues
            // `HYPERCALL_REGISTER_PEBS_PAGE`. Runtime fields are reset.
            pebs_state: parent_state
                .pebs_state
                .as_deref()
                .map(|p| heap_box(p.clone_for_fork())),
            pebs_supported: parent_state.pebs_supported,
            pmi_supported: parent_state.pmi_supported,
            freeze_perfmon_in_smm: parent_state.freeze_perfmon_in_smm,
            io_channel: IoChannelState::clone_for_fork(&parent_state.io_channel),
            serial_console: SerialConsoleState::clone_for_fork(&parent_state.serial_console),
            last_cpu: None,
        })
    }
}

#[cfg(test)]
#[path = "vm_state_event_tests.rs"]
mod event_producer_tests;
