use super::Page;

#[cfg(not(feature = "cargo"))]
use crate::memory::{HostPhysAddr, VirtAddr};
#[cfg(feature = "cargo")]
use memory::{HostPhysAddr, VirtAddr};

/// Guest memory (e.g. `vmalloc_user`), mappable into kernel and userspace.
/// Virtually contiguous but physically scattered; freed on drop.
pub trait GuestMemory: Sized {
    /// Size in bytes.
    fn size(&self) -> usize;

    /// Kernel virtual address of the start of the region.
    fn virt_addr(&self) -> VirtAddr;

    fn as_ptr(&self) -> *const u8 {
        self.virt_addr().as_u64() as *const u8
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.virt_addr().as_u64() as *mut u8
    }

    /// Host physical address of the page at page-aligned `page_offset` (for EPT
    /// mapping), or `None` if out of range.
    fn page_phys_addr(&self, page_offset: usize) -> Option<HostPhysAddr>;
}

/// Trait representing low-level kernel operations.
pub trait Kernel {
    type P: Page;
    type G: GuestMemory;

    fn alloc_zeroed_page(&self) -> Option<Self::P>;

    /// Allocate zeroed, userspace-mappable guest memory (`vmalloc_user` in the
    /// kernel); `size` is rounded up to a page.
    fn alloc_guest_memory(&self, size: usize) -> Option<Self::G>;

    fn phys_to_virt(&self, phys: HostPhysAddr) -> *mut u8;

    /// Run `func` on all online CPUs and wait. It runs in interrupt context and
    /// must not sleep. Returns the first error.
    fn call_on_all_cpus_with_data<F, T, E>(&self, data: &T, func: F) -> Result<(), E>
    where
        F: Fn(&T) -> Result<(), E> + Sync + Send,
        T: Sync,
        E: Send;

    fn current_cpu_id(&self) -> usize;

    /// Whether the run loop should yield to the scheduler (TIF_NEED_RESCHED in
    /// the kernel; always false in tests).
    fn need_resched(&self) -> bool;

    fn local_irq_enable(&self);

    fn local_irq_disable(&self);

    /// Mark this CPU as running a PMI-mode precise-exit loop, so the host NMI
    /// handler claims our `IA32_FIXED_CTR0` overflow PMIs that land in host
    /// context (and re-unmasks LVTPC, which Linux perf only does while it has
    /// active events). Deactivating returns how many such PMIs were claimed
    /// since activation. No-op outside the kernel.
    fn set_precise_pmi_active(&self, _active: bool) -> u64 {
        0
    }
}

/// RAII guard that marks the current CPU as running a PMI-mode precise-exit
/// loop (see [`Kernel::set_precise_pmi_active`]) while held. `None` inside
/// when PMI mode is not in use, so the guard is a no-op.
pub struct PrecisePmiGuard<'a, K: Kernel> {
    kernel: Option<&'a K>,
}

impl<'a, K: Kernel> PrecisePmiGuard<'a, K> {
    #[inline]
    pub fn new(kernel: &'a K, active: bool) -> Self {
        if active {
            kernel.set_precise_pmi_active(true);
        }
        Self {
            kernel: active.then_some(kernel),
        }
    }

    /// Deactivate and return the host-context PMI claims since activation.
    #[inline]
    pub fn finish(mut self) -> u64 {
        self.kernel
            .take()
            .map_or(0, |k| k.set_precise_pmi_active(false))
    }
}

impl<K: Kernel> Drop for PrecisePmiGuard<'_, K> {
    #[inline]
    fn drop(&mut self) {
        if let Some(k) = self.kernel.take() {
            k.set_precise_pmi_active(false);
        }
    }
}

/// RAII guard that disables local interrupts while held.
///
/// Protects the XCR0 switch around VM entry/exit: XCR0 holds the guest value
/// (possibly without AVX-512) before VMRESUME, and an interrupt handler using
/// AVX-512 in that window would crash.
pub struct IrqGuard<'a, K: Kernel> {
    kernel: &'a K,
}

impl<'a, K: Kernel> IrqGuard<'a, K> {
    #[inline]
    pub fn new(kernel: &'a K) -> Self {
        kernel.local_irq_disable();
        Self { kernel }
    }
}

impl<K: Kernel> Drop for IrqGuard<'_, K> {
    #[inline]
    fn drop(&mut self) {
        self.kernel.local_irq_enable();
    }
}

/// RAII guard that enables local interrupts while held.
///
/// Inverse of [`IrqGuard`]; used inside one to open a brief window for pending
/// host interrupts (timer ticks, IPIs).
pub struct ReverseIrqGuard<'a, K: Kernel> {
    kernel: &'a K,
}

impl<'a, K: Kernel> ReverseIrqGuard<'a, K> {
    #[inline]
    pub fn new(kernel: &'a K) -> Self {
        kernel.local_irq_enable();
        Self { kernel }
    }
}

impl<K: Kernel> Drop for ReverseIrqGuard<'_, K> {
    #[inline]
    fn drop(&mut self) {
        self.kernel.local_irq_disable();
    }
}
