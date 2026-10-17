// SPDX-License-Identifier: GPL-2.0

//! Ioctl handlers shared by root and forked VMs via the `VmFileOps` trait.

use core::mem::size_of;
use core::sync::atomic::{AtomicBool, Ordering};

use kernel::bindings;

use super::super::c_helpers::{bedrock_copy_from_user, bedrock_copy_to_user, PreemptionGuard};
use super::super::factory::KernelFrameAllocator;
use super::super::machine::MACHINE;
use super::super::page::{EventBuffer, PagePool};
use super::super::vmx::registers::GuestRegisters;
use super::super::vmx::traits::{
    CowAllocator, InstructionCounterError, Machine, VmContext, VmRunError,
};
use super::super::vmx::ExitReason;
use super::super::vmx::{EventCategories, ExitTrigger, RdrandMode};
use super::super::vmx_asm::RealVmRunner;
use super::structs::*;

/// VM file operations common to root and forked VMs.
pub(crate) trait VmFileOps {
    type Vm: VmContext;

    fn vm(&self) -> &Self::Vm;

    fn vm_mut(&mut self) -> &mut Self::Vm;

    fn vm_id(&self) -> u64;

    fn running(&self) -> &AtomicBool;

    fn event_buffer(&self) -> Option<&EventBuffer>;

    fn event_buffer_mut(&mut self) -> &mut Option<EventBuffer>;

    /// False if the VM has children (forkable VMs).
    fn can_run(&self) -> bool;

    /// Children count (for error messages).
    fn children_count(&self) -> usize;

    /// Split-borrow the VM and page pool (disjoint fields).
    fn vm_and_pool(&mut self) -> (&mut Self::Vm, &mut PagePool);
}

/// Handle GET_REGS ioctl - copy all VM registers to userspace.
pub(crate) fn handle_get_regs<F: VmFileOps>(vm_file: &F, arg: usize) -> isize {
    let vm = vm_file.vm();

    // Stay on this CPU for the VMCS load/read/clear.
    let _preempt_guard = PreemptionGuard::new();

    let guest_regs = match vm.get_registers_guarded() {
        Ok(regs) => regs,
        Err(e) => {
            log_err!("GET_REGS failed: {:?}\n", e);
            return -(bindings::EINVAL as isize);
        }
    };

    let regs = BedrockRegs {
        gprs: guest_regs.gprs,
        control_regs: guest_regs.control_regs,
        debug_regs: guest_regs.debug_regs,
        segment_regs: guest_regs.segment_regs,
        descriptor_tables: guest_regs.descriptor_tables,
        extended_control: guest_regs.extended_control_regs,
        rip: guest_regs.rip,
        rflags: guest_regs.rflags,
    };

    // SAFETY: Bounded copy of the stack-local `regs` to the user pointer `arg`.
    let not_copied = unsafe {
        bedrock_copy_to_user(
            arg as *mut core::ffi::c_void,
            core::ptr::from_ref(&regs).cast::<core::ffi::c_void>(),
            size_of::<BedrockRegs>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    0
}

/// Handle SET_REGS ioctl - copy all VM registers from userspace.
pub(crate) fn handle_set_regs<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize {
    let mut regs = core::mem::MaybeUninit::<BedrockRegs>::uninit();

    // SAFETY: Bounded copy from the user pointer `arg` into writable `regs`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            regs.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<BedrockRegs>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    // SAFETY: The copy returned 0, so `regs` is fully initialized.
    let regs = unsafe { regs.assume_init() };

    // Stay on this CPU for the VMCS load/write/clear.
    let _preempt_guard = PreemptionGuard::new();

    let vm = vm_file.vm_mut();
    let guest_regs = GuestRegisters {
        gprs: regs.gprs,
        control_regs: regs.control_regs,
        debug_regs: regs.debug_regs,
        segment_regs: regs.segment_regs,
        descriptor_tables: regs.descriptor_tables,
        extended_control_regs: regs.extended_control,
        rip: regs.rip,
        rflags: regs.rflags,
    };
    match vm.set_registers_guarded(&guest_regs) {
        Ok(()) => 0,
        Err(e) => {
            log_err!("SET_REGS failed: {:?}\n", e);
            -(bindings::EINVAL as isize)
        }
    }
}

use super::super::vmcs::RealVmcs;

/// Handle RUN ioctl - run the VM until it exits to userspace.
///
/// Before each entry the page pool is refilled in sleepable context; a
/// PoolExhausted exit drops back here to refill and re-enter.
pub(crate) fn handle_run<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize
where
    F::Vm: VmContext<Vmcs = RealVmcs>,
    for<'a> KernelFrameAllocator<'a>: CowAllocator<<F::Vm as VmContext>::CowPage>,
{
    // Raw pointer avoids borrow conflicts with vm_file below.
    let running_ptr = core::ptr::from_ref(vm_file.running());

    // SAFETY: running_ptr points into vm_file, which outlives this function.
    if unsafe { &*running_ptr }
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        log_err!(
            "CONCURRENT ACCESS DETECTED: VM {} RUN called while already running!\n",
            vm_file.vm_id()
        );
        return -(bindings::EBUSY as isize);
    }

    if !vm_file.can_run() {
        log_err!(
            "VM {} has {} children and cannot be run\n",
            vm_file.vm_id(),
            vm_file.children_count()
        );
        // SAFETY: running_ptr is valid.
        unsafe { &*running_ptr }.store(false, Ordering::Release);
        return -(bindings::EBUSY as isize);
    }

    // Clears the running flag on all exit paths.
    struct RunningGuard(*const AtomicBool);
    impl Drop for RunningGuard {
        fn drop(&mut self) {
            // SAFETY: The pointer stays valid until the function returns.
            unsafe { &*self.0 }.store(false, Ordering::Release);
        }
    }
    let _running_guard = RunningGuard(running_ptr);

    let mut first_iteration = true;
    // Only refill after PoolExhausted, so refills stay deterministic.
    let mut pool_exhausted = false;
    let exit_reason = loop {
        if pool_exhausted {
            let (_, pool) = vm_file.vm_and_pool();
            if !pool.refill() {
                return -(bindings::ENOMEM as isize);
            }
        }

        let _preempt_guard = PreemptionGuard::new();
        let (vm, pool) = vm_file.vm_and_pool();

        if first_iteration {
            // Clear only on first entry (userspace has drained the previous
            // run); `event_clear` re-appends any event staged when the buffer
            // filled mid-run.
            vm.state_mut().event_clear();
            first_iteration = false;
        }

        let mut runner = RealVmRunner::new();
        let mut allocator = KernelFrameAllocator::new_with_pool(MACHINE.kernel(), pool);

        // SAFETY: Preemption is disabled so the VMCS stays on this CPU; runner,
        // machine, and allocator are valid for the call.
        match unsafe { vm.run(&mut runner, &MACHINE, &mut allocator) } {
            Ok(ExitReason::PoolExhausted) => {
                // Guard dropped: back in sleepable context; refill at loop top.
                pool_exhausted = true;
                continue;
            }
            Ok(reason) => break reason,
            Err(e) => {
                log_err!("VM run failed: {:?}\n", e);
                return match e {
                    VmRunError::InstructionCounter(InstructionCounterError::Unavailable) => {
                        -(bindings::EOPNOTSUPP as isize)
                    }
                    _ => -(bindings::EIO as isize),
                };
            }
        }
    };

    let vm = vm_file.vm();
    let exit_qualification = vm.state().last_exit_qualification;
    let guest_physical_addr = vm.state().last_guest_physical_addr;
    let event_len = vm.state().event_buffer_len();
    let emulated_tsc = vm.state().emulated_tsc;
    let tsc_frequency = vm.state().tsc_frequency;

    let exit_info = BedrockVmExit {
        exit_reason: exit_reason as u32,
        _reserved: 0,
        exit_qualification,
        guest_physical_addr,
        event_len: event_len as u32,
        _pad: 0,
        emulated_tsc,
        tsc_frequency,
    };

    // SAFETY: Bounded copy of the stack-local `exit_info` to the user pointer `arg`.
    let not_copied = unsafe {
        bedrock_copy_to_user(
            arg as *mut core::ffi::c_void,
            core::ptr::from_ref(&exit_info).cast::<core::ffi::c_void>(),
            size_of::<BedrockVmExit>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    0
}

/// Handle GET_VM_ID ioctl - return the VM's unique identifier.
pub(crate) fn handle_get_vm_id<F: VmFileOps>(vm_file: &F, arg: usize) -> isize {
    let vm_id = vm_file.vm_id();

    // SAFETY: Bounded copy of the stack-local `vm_id` to the user pointer `arg`.
    let not_copied = unsafe {
        bedrock_copy_to_user(
            arg as *mut core::ffi::c_void,
            core::ptr::from_ref(&vm_id).cast::<core::ffi::c_void>(),
            size_of::<u64>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    0
}

/// Handle SET_RDRAND_CONFIG ioctl - configure RDRAND emulation mode.
pub(crate) fn handle_set_rdrand_config<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize {
    let mut config = core::mem::MaybeUninit::<BedrockRdrandConfig>::uninit();

    // SAFETY: Bounded copy from the user pointer `arg` into writable `config`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            config.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<BedrockRdrandConfig>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    // SAFETY: The copy returned 0, so `config` is fully initialized.
    let config = unsafe { config.assume_init() };

    let mode = match config.mode {
        0 => RdrandMode::SeededRng,
        1 => RdrandMode::ExitToUserspace,
        _ => {
            log_err!("SET_RDRAND_CONFIG: invalid mode {}\n", config.mode);
            return -(bindings::EINVAL as isize);
        }
    };

    // One mode and seed shared by RDRAND, RDSEED and HYPERCALL_GET_RANDOM.
    let vm = vm_file.vm_mut();
    vm.state_mut().devices.random.configure(mode, config.value);

    log_info!(
        "SET_RDRAND_CONFIG: mode={:?}, value=0x{:x}\n",
        mode,
        config.value
    );
    0
}

/// Handle GET_RANDOM_REQUEST ioctl - return the pending `HYPERCALL_GET_RANDOM`
/// request (PID + byte count).
pub(crate) fn handle_get_random_request<F: VmFileOps>(vm_file: &F, arg: usize) -> isize {
    let req = {
        let r = &vm_file.vm().state().devices.random;
        BedrockRandomRequest {
            pid: r.pid,
            len: r.req_len,
        }
    };

    // SAFETY: Bounded copy of the stack-local `req` to the user pointer `arg`.
    let not_copied = unsafe {
        bedrock_copy_to_user(
            arg as *mut core::ffi::c_void,
            core::ptr::from_ref(&req).cast::<core::ffi::c_void>(),
            size_of::<BedrockRandomRequest>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    0
}

/// Handle SET_RANDOM_BYTES ioctl - stage reply bytes for the pending
/// `HYPERCALL_GET_RANDOM` request.
pub(crate) fn handle_set_random_bytes<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize {
    let mut payload = core::mem::MaybeUninit::<BedrockRandomBytes>::uninit();

    // SAFETY: Bounded copy from the user pointer `arg` into writable `payload`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            payload.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<BedrockRandomBytes>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    // SAFETY: The copy returned 0, so `payload` is fully initialized.
    let payload = unsafe { payload.assume_init() };
    let len = (payload.len as usize).min(BEDROCK_RANDOM_REPLY_MAX);

    let vm = vm_file.vm_mut();
    vm.state_mut()
        .devices
        .random
        .stage_reply(&payload.data[..len]);

    0
}

/// Handle SET_RDRAND_VALUE ioctl - set pending RDRAND value.
pub(crate) fn handle_set_rdrand_value<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize {
    let mut value: u64 = 0;

    // SAFETY: Bounded copy from the user pointer `arg` into the stack-local `value`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            core::ptr::from_mut(&mut value).cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<u64>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    let vm = vm_file.vm_mut();
    vm.state_mut().devices.random.set_pending_value(value);

    0
}

/// Handle SET_EVENT_CONFIG ioctl - enable/disable the event stream (allocating
/// or freeing the 1MB buffer), set the category mask and `Exit`-record trigger.
pub(crate) fn handle_set_event_config<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize {
    let mut config = core::mem::MaybeUninit::<BedrockEventConfig>::uninit();

    // SAFETY: Bounded copy from the user pointer `arg` into writable `config`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            config.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<BedrockEventConfig>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    // SAFETY: The copy returned 0, so `config` is fully initialized.
    let config = unsafe { config.assume_init() };

    let was_enabled = vm_file.event_buffer().is_some();
    let want_enabled = config.enabled != 0;

    if want_enabled && !was_enabled {
        let buffer = match EventBuffer::new() {
            Some(b) => b,
            None => {
                log_err!("SET_EVENT_CONFIG: failed to allocate event buffer\n");
                return -(bindings::ENOMEM as isize);
            }
        };

        // vm_file.event_buffer keeps the buffer alive until disabled or close.
        vm_file
            .vm_mut()
            .state_mut()
            .set_event_buffer(buffer.as_ptr());
        *vm_file.event_buffer_mut() = Some(buffer);
    } else if !want_enabled && was_enabled {
        vm_file.vm_mut().state_mut().clear_event_buffer_ptr();
        *vm_file.event_buffer_mut() = None;
    }

    // Applied regardless of the enable transition, so categories can change
    // while the stream stays enabled.
    vm_file
        .vm_mut()
        .state_mut()
        .set_event_categories(EventCategories(config.categories));

    // Capturing exits needs the stream enabled, the EXIT category, and a
    // non-Disabled trigger.
    let trigger = match config.exit_trigger {
        0 => ExitTrigger::Disabled,
        1 => ExitTrigger::AllExits,
        2 => ExitTrigger::AtTsc,
        3 => ExitTrigger::AtShutdown,
        4 => ExitTrigger::Checkpoints,
        5 => ExitTrigger::TscRange,
        _ => {
            log_err!(
                "SET_EVENT_CONFIG: invalid exit_trigger {}\n",
                config.exit_trigger
            );
            return -(bindings::EINVAL as isize);
        }
    };
    let trigger = if want_enabled {
        trigger
    } else {
        ExitTrigger::Disabled
    };
    let state = vm_file.vm_mut().state_mut();
    state.set_exit_trigger(trigger, config.exit_target_tsc);
    state.set_exit_start_tsc(config.exit_start_tsc);
    state.skip_memory_hash = (config.exit_flags & 1) != 0;
    state.set_intercept_pf((config.exit_flags & 2) != 0);

    log_info!(
        "SET_EVENT_CONFIG: enabled={}, categories={:#x}, exit_trigger={:?}, exit_flags={:#x} for VM {}\n",
        want_enabled,
        config.categories,
        trigger,
        config.exit_flags,
        vm_file.vm_id()
    );
    0
}

/// Handle GET_EXIT_STATS ioctl - retrieve exit handler performance statistics.
pub(crate) fn handle_get_exit_stats<F: VmFileOps>(vm_file: &F, arg: usize) -> isize {
    let stats = &vm_file.vm().state().exit_stats;

    let convert = |s: &super::super::vmx::vm_state::ExitStats| BedrockExitStatEntry {
        count: s.count,
        cycles: s.cycles,
    };

    let exit_stats = BedrockExitStats {
        cpuid: convert(&stats.cpuid),
        msr_read: convert(&stats.msr_read),
        msr_write: convert(&stats.msr_write),
        cr_access: convert(&stats.cr_access),
        io_instruction: convert(&stats.io_instruction),
        ept_violation: convert(&stats.ept_violation),
        external_interrupt: convert(&stats.external_interrupt),
        rdtsc: convert(&stats.rdtsc),
        rdtscp: convert(&stats.rdtscp),
        rdpmc: convert(&stats.rdpmc),
        mwait: convert(&stats.mwait),
        vmcall: convert(&stats.vmcall),
        apic_access: convert(&stats.apic_access),
        mtf: convert(&stats.mtf),
        xsetbv: convert(&stats.xsetbv),
        rdrand: convert(&stats.rdrand),
        rdseed: convert(&stats.rdseed),
        exception_nmi: convert(&stats.exception_nmi),
        other: convert(&stats.other),
        total_run_cycles: stats.total_run_cycles,
        guest_cycles: stats.guest_cycles,
        vmentry_overhead_cycles: stats.vmentry_overhead_cycles,
        vmexit_overhead_cycles: stats.vmexit_overhead_cycles,
        irq_window_cycles: stats.irq_window_cycles,
        pebs_arm_below_min_delta: stats.pebs_arm_below_min_delta,
        pebs_arm_already_past: stats.pebs_arm_already_past,
        pebs_armed_iter_no_fire: stats.pebs_armed_iter_no_fire,
        apic_timer_late_inject: stats.apic_timer_late_inject,
        max_pebs_skid: stats.max_pebs_skid,
        pmi_lvtpc_unmasked: stats.pmi_lvtpc_unmasked,
        pmi_nmi_forwarded: stats.pmi_nmi_forwarded,
        pmi_host_claims: stats.pmi_host_claims,
    };

    // SAFETY: Bounded copy of the stack-local `exit_stats` to the user pointer `arg`.
    let not_copied = unsafe {
        bedrock_copy_to_user(
            arg as *mut core::ffi::c_void,
            core::ptr::from_ref(&exit_stats).cast::<core::ffi::c_void>(),
            size_of::<BedrockExitStats>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    0
}

/// Handle SET_STOP_TSC ioctl - set TSC value at which VM should stop.
pub(crate) fn handle_set_stop_tsc<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize {
    let mut value = core::mem::MaybeUninit::<u64>::uninit();

    // SAFETY: Bounded copy from the user pointer `arg` into writable `value`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            value.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<u64>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    // SAFETY: The copy returned 0, so `value` is fully initialized.
    let value = unsafe { value.assume_init() };

    // 0 disables.
    let vm = vm_file.vm_mut();
    if value == 0 {
        vm.state_mut().stop_at_tsc = None;
        log_info!("SET_STOP_TSC: disabled for VM {}\n", vm_file.vm_id());
    } else {
        vm.state_mut().stop_at_tsc = Some(value);
        log_info!(
            "SET_STOP_TSC: set to {} for VM {}\n",
            value,
            vm_file.vm_id()
        );
    }

    0
}

/// Handle SET_SINGLE_STEP ioctl - configure MTF single-stepping.
pub(crate) fn handle_set_single_step<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize {
    let mut config = core::mem::MaybeUninit::<BedrockSingleStepConfig>::uninit();

    // SAFETY: Bounded copy from the user pointer `arg` into writable `config`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            config.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<BedrockSingleStepConfig>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    // SAFETY: The copy returned 0, so `config` is fully initialized.
    let config = unsafe { config.assume_init() };
    let vm = vm_file.vm_mut();

    if config.enabled != 0 {
        vm.state_mut().single_step_tsc_range = Some((config.tsc_start, config.tsc_end));
        log_info!(
            "SET_SINGLE_STEP: enabled for TSC range [{}, {}) for VM {}\n",
            config.tsc_start,
            config.tsc_end,
            vm_file.vm_id()
        );
    } else {
        vm.state_mut().single_step_tsc_range = None;
        vm.state_mut().mtf_enabled = false;
        log_info!("SET_SINGLE_STEP: disabled for VM {}\n", vm_file.vm_id());
    }

    0
}

/// Handle GET_FEEDBACK_BUFFER_INFO ioctl - return info for the requested index.
pub(crate) fn handle_get_feedback_buffer_info<F: VmFileOps>(vm_file: &F, arg: usize) -> isize {
    let mut request = core::mem::MaybeUninit::<BedrockFeedbackBufferInfoRequest>::uninit();

    // SAFETY: Bounded copy from the user pointer `arg` into writable `request`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            request.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<BedrockFeedbackBufferInfoRequest>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    // SAFETY: The copy returned 0, so `request` is fully initialized.
    let request = unsafe { request.assume_init() };
    let index = request.index as usize;

    let vm = vm_file.vm();
    // Unregistered/out-of-range indices report registered = 0, so userspace
    // can enumerate until the first gap.
    let info = match vm.state().feedback_buffers.get(index) {
        Some(buffer) => BedrockFeedbackBufferInfo {
            gva: buffer.gva,
            size: buffer.size,
            num_pages: buffer.num_pages as u64,
            registered: 1,
            index: index as u32,
            id_len: buffer.id_len,
            _reserved: 0,
            id: buffer.id,
        },
        None => BedrockFeedbackBufferInfo {
            gva: 0,
            size: 0,
            num_pages: 0,
            registered: 0,
            index: index as u32,
            id_len: 0,
            _reserved: 0,
            id: [0u8; super::structs::FEEDBACK_BUFFER_ID_MAX_LEN],
        },
    };

    // SAFETY: Bounded copy of the stack-local `info` to the user pointer `arg`.
    let not_copied = unsafe {
        bedrock_copy_to_user(
            arg as *mut core::ffi::c_void,
            core::ptr::from_ref(&info).cast::<core::ffi::c_void>(),
            size_of::<BedrockFeedbackBufferInfo>() as core::ffi::c_ulong,
        )
    };

    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    0
}

/// Offset of the data payload past the header in a `BedrockIoActionPayload`.
const IO_ACTION_DATA_OFFSET: usize = size_of::<BedrockIoActionHeader>();

/// Handle `BEDROCK_VM_QUEUE_IO_ACTION` ioctl - push a request onto the pending
/// queue and return immediately. `HYPERCALL_IO_GET_REQUEST` promotes the next
/// entry once the previous one is consumed (independent of
/// `HYPERCALL_IO_PUT_RESPONSE`), so the guest can run commands in parallel.
///
/// Only the 16-byte header is staged on the stack; data is copied straight into
/// a per-request `HeapVec<u8>`. Returns `EBUSY` if the queue is at
/// `PENDING_IO_QUEUE_CAP`.
pub(crate) fn handle_queue_io_action<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize {
    let mut header = core::mem::MaybeUninit::<BedrockIoActionHeader>::uninit();

    // SAFETY: Bounded copy from the user pointer `arg` into writable `header`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            header.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<BedrockIoActionHeader>() as core::ffi::c_ulong,
        )
    };
    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }
    // SAFETY: The copy returned 0, so `header` is fully initialized.
    let header = unsafe { header.assume_init() };

    let len = header.len as usize;
    if len > BEDROCK_IO_CHANNEL_BUF_SIZE {
        return -(bindings::EINVAL as isize);
    }

    // Sized to `len` so heap use is proportional to the request.
    let mut data = match super::super::vmx::heap_vec_with_capacity::<u8>(len) {
        Ok(v) => v,
        Err(_) => return -(bindings::ENOMEM as isize),
    };
    if len > 0 {
        // Zero-fill, then overwrite via bedrock_copy_from_user. Capacity is
        // reserved, so push failing means a real allocation failure.
        for _ in 0..len {
            if super::super::vmx::heap_vec_push(&mut data, 0u8).is_err() {
                return -(bindings::ENOMEM as isize);
            }
        }
        // SAFETY: `data` has exactly `len` writable bytes; `arg +
        // IO_ACTION_DATA_OFFSET` is the user data past the header.
        let not_copied = unsafe {
            bedrock_copy_from_user(
                data.as_mut_ptr().cast::<core::ffi::c_void>(),
                (arg + IO_ACTION_DATA_OFFSET) as *const core::ffi::c_void,
                len as core::ffi::c_ulong,
            )
        };
        if not_copied != 0 {
            return -(bindings::EFAULT as isize);
        }
    }

    let action = super::super::vmx::PendingIoAction {
        target_tsc: header.target_tsc,
        data,
    };

    let chan = &mut vm_file.vm_mut().state_mut().io_channel;
    match chan.enqueue_pending(action) {
        super::super::vmx::EnqueueResult::Queued => {}
        super::super::vmx::EnqueueResult::Full => return -(bindings::EBUSY as isize),
        super::super::vmx::EnqueueResult::OutOfMemory => return -(bindings::ENOMEM as isize),
    }
    // If the in-flight slot is free, promote now so the next
    // `inject_pending_interrupt` can fire without another VM exit.
    chan.promote_next_pending();

    0
}

/// Handle `BEDROCK_VM_DRAIN_IO_RESPONSE` ioctl - single-shot consume of the
/// last `HYPERCALL_IO_PUT_RESPONSE` response; `response_len` is reset to 0 so a
/// second drain returns an empty payload.
pub(crate) fn handle_drain_io_response<F: VmFileOps>(vm_file: &mut F, arg: usize) -> isize {
    let mut header = core::mem::MaybeUninit::<BedrockIoActionHeader>::uninit();
    // SAFETY: Bounded copy from the user pointer `arg` into writable `header`.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            header.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<BedrockIoActionHeader>() as core::ffi::c_ulong,
        )
    };
    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }
    // SAFETY: The copy returned 0, so `header` is fully initialized.
    let header = unsafe { header.assume_init() };

    let user_capacity = header.len as usize;
    if user_capacity > BEDROCK_IO_CHANNEL_BUF_SIZE {
        return -(bindings::EINVAL as isize);
    }

    let chan = &mut vm_file.vm_mut().state_mut().io_channel;
    let response_len = chan.response_len.min(user_capacity);

    if response_len > 0 {
        // SAFETY: `response_buf` is a `Box<[u8; IO_CHANNEL_BUF_SIZE]>`, valid for
        // `response_len` bytes; bedrock_copy_to_user validates the user pointer.
        let not_copied = unsafe {
            bedrock_copy_to_user(
                (arg + IO_ACTION_DATA_OFFSET) as *mut core::ffi::c_void,
                chan.response_buf.as_ptr().cast::<core::ffi::c_void>(),
                response_len as core::ffi::c_ulong,
            )
        };
        if not_copied != 0 {
            return -(bindings::EFAULT as isize);
        }
    }

    let out_header = BedrockIoActionHeader {
        len: response_len as u32,
        _reserved: 0,
        target_tsc: 0,
    };
    // SAFETY: Overwrites only the header bytes at the start of the user's
    // `BedrockIoActionPayload`.
    let not_copied = unsafe {
        bedrock_copy_to_user(
            arg as *mut core::ffi::c_void,
            core::ptr::from_ref(&out_header).cast::<core::ffi::c_void>(),
            size_of::<BedrockIoActionHeader>() as core::ffi::c_ulong,
        )
    };
    if not_copied != 0 {
        return -(bindings::EFAULT as isize);
    }

    chan.response_len = 0;
    0
}
