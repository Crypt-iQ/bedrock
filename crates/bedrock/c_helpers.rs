// SPDX-License-Identifier: GPL-2.0

//! Bindings to local C helper functions in helpers.c

use kernel::bindings::{page, phys_addr_t, smp_call_func_t};

/// XXH64 streaming state; matches the kernel's `struct xxh64_state`.
#[repr(C)]
pub(crate) struct Xxh64State {
    pub total_len: u64,
    pub v1: u64,
    pub v2: u64,
    pub v3: u64,
    pub v4: u64,
    pub mem64: [u64; 4],
    pub memsize: u32,
}

/// Info passed to callbacks by bedrock_for_each_cpu (C `bedrock_cpu_call_info`).
#[repr(C)]
pub(crate) struct BedrockCpuCallInfo {
    pub(crate) info: *mut core::ffi::c_void,
    pub(crate) error: i32,
}

/// VMX capabilities; matches the C `struct bedrock_vmx_caps` layout.
#[repr(C)]
pub(crate) struct BedrockVmxCaps {
    pub(crate) pin_based_exec_ctrl: u32,
    pub(crate) cpu_based_exec_ctrl: u32,
    pub(crate) cpu_based_exec_ctrl2: u32,
    pub(crate) vmexit_ctrl: u32,
    pub(crate) vmentry_ctrl: u32,
    pub(crate) cr0_fixed0: u64,
    pub(crate) cr0_fixed1: u64,
    pub(crate) cr4_fixed0: u64,
    pub(crate) cr4_fixed1: u64,
    pub(crate) has_ept: bool,
    pub(crate) has_vpid: bool,
    pub(crate) pebs_format: u8,
    pub(crate) pebs_baseline: bool,
    pub(crate) pebs_trap: bool,
}

#[allow(improper_ctypes)]
extern "C" {
    /// Convert a struct page pointer to its physical address.
    pub(crate) fn bedrock_page_to_phys(page: *mut page) -> phys_addr_t;

    /// Get the kernel virtual address for a page.
    pub(crate) fn bedrock_page_address(page: *mut page) -> *mut core::ffi::c_void;

    /// Run a function on each online CPU sequentially. Returns 0 or the first
    /// error, with `failed_cpu` set to the CPU that failed.
    pub(crate) fn bedrock_for_each_cpu(
        func: smp_call_func_t,
        info: *mut core::ffi::c_void,
        failed_cpu: *mut i32,
    ) -> i32;

    /// Read an MSR while handling the #GP raised by an unavailable address.
    pub(crate) fn bedrock_rdmsr_safe(msr: u32, value: *mut u64) -> core::ffi::c_int;

    /// Write an MSR while handling the #GP raised by an unavailable address or value.
    pub(crate) fn bedrock_wrmsr_safe(msr: u32, value: u64) -> core::ffi::c_int;

    /// Register the host NMI handler that claims PMI-mode overflow PMIs
    /// delivered in host context. Returns 0 or a negative errno.
    pub(crate) fn bedrock_pmi_nmi_register() -> core::ffi::c_int;

    /// Unregister the handler registered by `bedrock_pmi_nmi_register`.
    pub(crate) fn bedrock_pmi_nmi_unregister();

    /// Mark the current CPU as running a PMI-mode loop (interrupts must be
    /// disabled). Deactivating returns the PMIs claimed since activation.
    pub(crate) fn bedrock_pmi_set_active(active: bool) -> u64;

    /// Allocate zeroed memory that can be mapped to userspace.
    pub(crate) fn bedrock_vmalloc_user(size: core::ffi::c_ulong) -> *mut core::ffi::c_void;

    /// Free memory allocated with bedrock_vmalloc_user.
    pub(crate) fn bedrock_vfree(addr: *mut core::ffi::c_void);

    /// Physical address of a page within vmalloc memory, or 0 if not vmalloc.
    pub(crate) fn bedrock_vmalloc_to_phys(addr: *mut core::ffi::c_void) -> phys_addr_t;

    /// Convert any kernel virtual address (vmalloc or direct-mapped) to its
    /// physical address, or 0 if invalid.
    pub(crate) fn bedrock_kva_to_phys(addr: *mut core::ffi::c_void) -> phys_addr_t;

    /// Convert a physical address to a kernel virtual address.
    pub(crate) fn bedrock_phys_to_virt(phys: phys_addr_t) -> *mut core::ffi::c_void;

    /// Create an anonymous inode FD with the given fops; `priv_` is stored in
    /// file->private_data. Returns the FD or a negative error code.
    pub(crate) fn bedrock_anon_inode_getfd(
        name: *const core::ffi::c_char,
        fops: *const kernel::bindings::file_operations,
        priv_: *mut core::ffi::c_void,
        flags: core::ffi::c_int,
    ) -> core::ffi::c_int;

    /// Copy from userspace. Returns the number of bytes NOT copied.
    pub(crate) fn bedrock_copy_from_user(
        to: *mut core::ffi::c_void,
        from: *const core::ffi::c_void,
        n: core::ffi::c_ulong,
    ) -> core::ffi::c_ulong;

    /// Copy to userspace. Returns the number of bytes NOT copied.
    pub(crate) fn bedrock_copy_to_user(
        to: *mut core::ffi::c_void,
        from: *const core::ffi::c_void,
        n: core::ffi::c_ulong,
    ) -> core::ffi::c_ulong;

    /// Map vmalloc_user() memory into a userspace VMA. Returns 0 or -errno.
    pub(crate) fn bedrock_remap_vmalloc_range(
        vma: *mut kernel::bindings::vm_area_struct,
        addr: *mut core::ffi::c_void,
        pgoff: core::ffi::c_ulong,
    ) -> core::ffi::c_int;

    /// Map page-aligned, possibly non-contiguous HPAs into a userspace VMA whose
    /// size must equal num_pages * PAGE_SIZE. Returns 0 or -errno.
    pub(crate) fn bedrock_remap_pages(
        vma: *mut kernel::bindings::vm_area_struct,
        hpas: *const u64,
        num_pages: core::ffi::c_int,
    ) -> core::ffi::c_int;

    /// Get VMA start address.
    pub(crate) fn bedrock_vma_start(
        vma: *mut kernel::bindings::vm_area_struct,
    ) -> core::ffi::c_ulong;

    /// Get VMA end address.
    pub(crate) fn bedrock_vma_end(vma: *mut kernel::bindings::vm_area_struct)
        -> core::ffi::c_ulong;

    /// Get VMA page offset.
    pub(crate) fn bedrock_vma_pgoff(
        vma: *mut kernel::bindings::vm_area_struct,
    ) -> core::ffi::c_ulong;

    /// Disable preemption on the current CPU.
    pub(crate) fn bedrock_preempt_disable();

    /// Enable preemption on the current CPU.
    pub(crate) fn bedrock_preempt_enable();

    /// Enable local interrupts (sets IF flag in RFLAGS).
    pub(crate) fn bedrock_local_irq_enable();

    /// Disable local interrupts (clears IF flag in RFLAGS).
    pub(crate) fn bedrock_local_irq_disable();

    /// Non-zero if TIF_NEED_RESCHED is set (wraps need_resched()).
    pub(crate) fn bedrock_need_resched() -> core::ffi::c_int;

    /// One-shot XXH64 hash.
    pub(crate) fn bedrock_xxh64(input: *const core::ffi::c_void, length: usize, seed: u64) -> u64;

    /// Reset XXH64 state for streaming hashing.
    pub(crate) fn bedrock_xxh64_reset(state: *mut Xxh64State, seed: u64);

    /// Update XXH64 state with more data.
    pub(crate) fn bedrock_xxh64_update(
        state: *mut Xxh64State,
        input: *const core::ffi::c_void,
        length: usize,
    );

    /// Finalize and return the XXH64 hash.
    pub(crate) fn bedrock_xxh64_digest(state: *const Xxh64State) -> u64;

    /// Check if VMX is enabled on the current CPU.
    /// Must be called with preemption disabled.
    pub(crate) fn bedrock_vcpu_is_vmxon() -> bool;

    /// Set VMX enabled state on the current CPU.
    /// Must be called with preemption disabled.
    pub(crate) fn bedrock_vcpu_set_vmxon(enabled: bool);

    /// Get VMX capabilities for the current CPU.
    /// Returns a pointer that is valid while preemption is disabled.
    pub(crate) fn bedrock_vcpu_get_capabilities() -> *const BedrockVmxCaps;

    /// Set VMX capabilities for the current CPU.
    /// Must be called with preemption disabled.
    pub(crate) fn bedrock_vcpu_set_capabilities(
        pin_based: u32,
        cpu_based: u32,
        cpu_based2: u32,
        vmexit: u32,
        vmentry: u32,
        cr0_fixed0: u64,
        cr0_fixed1: u64,
        cr4_fixed0: u64,
        cr4_fixed1: u64,
        has_ept: bool,
        has_vpid: bool,
        pebs_format: u8,
        pebs_baseline: bool,
        pebs_trap: bool,
    );

    /// Set VMXON region for the current CPU.
    /// Must be called with preemption disabled.
    pub(crate) fn bedrock_vcpu_set_vmxon_region(phys: u64, virt: u64);

    /// Set CR4.VMXE via cr4_set_bits() so the kernel's CR4 shadow stays in sync.
    pub(crate) fn bedrock_cr4_set_vmxe();

    /// Clear CR4.VMXE via cr4_clear_bits(). Only call after VMXOFF.
    pub(crate) fn bedrock_cr4_clear_vmxe();
}

/// RAII guard that keeps the current thread on this CPU by disabling preemption.
pub(crate) struct PreemptionGuard {
    _marker: core::marker::PhantomData<*mut ()>,
}

impl PreemptionGuard {
    /// Disable preemption until the guard is dropped.
    #[inline]
    pub(crate) fn new() -> Self {
        // SAFETY: This is a valid kernel call that disables preemption.
        unsafe { bedrock_preempt_disable() };
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl Drop for PreemptionGuard {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: Preemption was disabled in new(), so it's safe to re-enable.
        unsafe { bedrock_preempt_enable() };
    }
}

/// Enable local interrupts.
#[inline]
pub(crate) fn local_irq_enable() {
    // SAFETY: Enabling interrupts is always safe.
    unsafe { bedrock_local_irq_enable() };
}

/// Disable local interrupts.
#[inline]
pub(crate) fn local_irq_disable() {
    // SAFETY: Disabling interrupts is always safe.
    unsafe { bedrock_local_irq_disable() };
}
