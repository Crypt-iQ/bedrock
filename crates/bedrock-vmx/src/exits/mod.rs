// SPDX-License-Identifier: GPL-2.0

//! VM exit dispatch and per-reason handlers, written against `VmContext` so
//! they can be unit-tested with mocks.

mod apic;
mod cpuid;
mod cr;
mod ept;
mod helpers;
mod interrupts;
mod io;
mod misc;
mod msr;
mod pebs;
mod qualifications;
mod rdrand;
mod reasons;
mod time;
mod vmcall;

pub use apic::{APIC_BASE, APIC_SIZE, IOAPIC_BASE, IOAPIC_SIZE};
pub use helpers::{ExitError, ExitHandlerResult};
pub use interrupts::{
    check_io_channel, inject_pending_interrupt, reinject_vectored_event, IO_CHANNEL_IRQ,
};
pub use pebs::{
    arm_for_next_iteration, arm_precise_exit, disarm_precise_exit, get_pebs_margin,
    host_has_ept_friendly_pebs, host_supports_pmi_precise_exit, min_arm_delta, pebs_post_vm_exit,
    pebs_pre_vm_entry, ArmResult, DsManagementArea, PebsAction, PebsHostMsrs, PebsState,
    PreciseExitMode, PEBS_MIN_DELTA, PERF_GLOBAL_CTRL_FIXED_CTR0,
};
pub use qualifications::{
    CrAccessQualification, EptViolationQualification, IoQualification, RdrandInstructionInfo,
    RdrandOperandSize,
};
pub use reasons::ExitReason;
pub use vmcall::{
    FB_ERR_BAD_ID_LEN, FB_ERR_BAD_SIZE, FB_ERR_BUFFER_NOT_RESIDENT, FB_ERR_ID_NOT_RESIDENT,
    FB_ERR_NO_SLOTS,
};

use cpuid::handle_cpuid;
use cr::handle_cr_access;
use ept::handle_ept_violation;
use helpers::{advance_rip, read_exit_qualification, read_exit_reason, ExitError as EE};
use interrupts::{disable_interrupt_window_exiting, handle_external_interrupt};
use io::handle_io;
use misc::{dump_triple_fault_state, handle_exception_nmi, handle_xsetbv};
use msr::{handle_msr_read, handle_msr_write};
use rdrand::{handle_rdrand, handle_rdseed};
use time::{handle_idle, handle_rdpmc, handle_rdtsc, handle_rdtscp};
use vmcall::handle_vmcall;

#[cfg(not(feature = "cargo"))]
use super::prelude::*;
#[cfg(feature = "cargo")]
use crate::prelude::*;

/// Retired-instruction count at which the APIC timer fires, or `None` if the
/// timer is disarmed, the APIC software-disabled, or the LVT masked.
fn next_timer_exit_count<C: VmContext>(ctx: &C) -> Option<u64> {
    let state = ctx.state();
    let apic = &state.devices.apic;
    if apic.timer_deadline == 0 {
        return None;
    }
    if (apic.svr & (1 << 8)) == 0 {
        return None;
    }
    if (apic.lvt_timer & (1 << 16)) != 0 {
        return None;
    }
    Some(apic.timer_deadline.saturating_sub(state.tsc_offset))
}

/// Emulated TSC at which the pending I/O channel request fires, or `None` if
/// the page is unregistered, no undelivered request is queued, the target is
/// 0 ("ASAP", handled by the normal IRR path), or the guest's IOAPIC entry is
/// not yet set up. Single readiness predicate for both PEBS arming and MTF.
pub(super) fn next_io_channel_target_tsc<C: VmContext>(ctx: &C) -> Option<u64> {
    let chan = &ctx.state().io_channel;
    if chan.page_gpa == 0 {
        return None;
    }
    if chan.request_len == 0 || chan.request_delivered {
        return None;
    }
    if chan.request_target_tsc == 0 {
        return None;
    }
    let entry = ctx.state().devices.ioapic.redtbl[interrupts::IO_CHANNEL_IRQ as usize];
    if (entry >> 16) & 1 != 0 || (entry & 0xFF) < 16 {
        return None;
    }
    Some(chan.request_target_tsc)
}

/// Instruction-count form of `next_io_channel_target_tsc`.
pub(super) fn next_io_channel_exit_count<C: VmContext>(ctx: &C) -> Option<u64> {
    next_io_channel_target_tsc(ctx).map(|t| t.saturating_sub(ctx.state().tsc_offset))
}

/// Emulated TSC at which single-stepping should begin, or `None` if no range
/// is configured or it has been entered.
///
/// Enabling MTF lazily on whichever exit first crosses `start` would let a
/// non-deterministic host-interrupt exit pick the start point, so forks would
/// log from different counts. Treating `start` as a precise-exit target (PEBS
/// plus MTF margin) lands the first step exactly on it.
pub(super) fn next_single_step_start_tsc<C: VmContext>(ctx: &C) -> Option<u64> {
    let (start, _end) = ctx.state().single_step_tsc_range?;
    let current = ctx.state().last_instruction_count + ctx.state().tsc_offset;
    (current < start).then_some(start)
}

/// Instruction-count form of `next_single_step_start_tsc`.
fn next_single_step_start_count<C: VmContext>(ctx: &C) -> Option<u64> {
    next_single_step_start_tsc(ctx).map(|t| t.saturating_sub(ctx.state().tsc_offset))
}

/// Width of the MTF window before a precise-exit target. Covers both the
/// margin and the `BelowMinDelta` case, where the counter can't arm and MTF must step
/// the whole remaining distance; a narrower window would let such a target
/// fire at a run-dependent later exit.
fn get_mtf_margin(mode: PreciseExitMode) -> u64 {
    min_arm_delta(mode) + get_pebs_margin(mode)
}

/// Enable MTF inside the configured single-step range, or (when PEBS is
/// registered) within `get_mtf_margin()` before any precise-exit target, so
/// the boundary step lands exactly on it.
///
/// The PEBS-registered gate matters for determinism: without PEBS arming the
/// margin would only engage when a non-deterministic exit happened to land in
/// it, so runs would disagree. Unregistered, all runs take the late-inject path.
///
/// Uses `last_instruction_count + tsc_offset` since `emulated_tsc` is stale on
/// non-deterministic exits.
pub fn update_mtf_state<C: VmContext>(ctx: &mut C) -> Result<(), ExitError> {
    let count = ctx.state().last_instruction_count;
    let tsc = count + ctx.state().tsc_offset;
    let range = ctx.state().single_step_tsc_range;
    let currently_enabled = ctx.state().mtf_enabled;
    let pebs_mode = ctx.state().pebs_state.as_deref().map(|p| p.mode);
    let pebs_registered = pebs_mode.is_some();

    let in_single_step = match range {
        Some((start, end)) => tsc >= start && tsc < end,
        None => false,
    };

    // Any target's window counts, not just the one PEBS armed for (PEBS has
    // a single counter).
    let mtf_margin = get_mtf_margin(pebs_mode.unwrap_or(PreciseExitMode::Pebs));
    let in_margin = |target_opt: Option<u64>| match target_opt {
        Some(target) => count >= target.saturating_sub(mtf_margin) && count < target,
        None => false,
    };
    let stop_at_count = ctx
        .state()
        .stop_at_tsc
        .map(|t| t.saturating_sub(ctx.state().tsc_offset));
    let in_pebs_margin = pebs_registered
        && (in_margin(next_timer_exit_count(ctx))
            || in_margin(next_io_channel_exit_count(ctx))
            || in_margin(stop_at_count)
            || in_margin(next_single_step_start_count(ctx)));

    let should_enable = in_single_step || in_pebs_margin;

    if should_enable != currently_enabled {
        let mut controls = ctx
            .state()
            .vmcs
            .read32(VmcsField32::PrimaryProcBasedVmExecControls)
            .map_err(|_| EE::Fatal("Failed to read primary controls for MTF"))?;

        if should_enable {
            controls |= cpu_based::MONITOR_TRAP_FLAG;
        } else {
            controls &= !cpu_based::MONITOR_TRAP_FLAG;
        }

        ctx.state()
            .vmcs
            .write32(VmcsField32::PrimaryProcBasedVmExecControls, controls)
            .map_err(|_| EE::Fatal("Failed to write primary controls for MTF"))?;

        ctx.state_mut().mtf_enabled = should_enable;
    }

    Ok(())
}

/// Handle a VM exit: classify determinism, dispatch, then run MTF, stop-at-TSC
/// and event-capture bookkeeping.
pub fn handle_exit<C: VmContext, K: Kernel, A: CowAllocator<C::CowPage>>(
    ctx: &mut C,
    kernel: &K,
    allocator: &mut A,
) -> ExitHandlerResult {
    let start_tsc = rdtsc();

    let reason = match read_exit_reason(ctx) {
        Ok(r) => r,
        Err(e) => return ExitHandlerResult::Error(e),
    };

    let qual = match read_exit_qualification(ctx) {
        Ok(q) => q,
        Err(e) => return ExitHandlerResult::Error(e),
    };

    let non_deterministic_exit = match reason {
        ExitReason::ExternalInterrupt
        | ExitReason::VmxPreemptionTimer
        | ExitReason::ExceptionNmi => true,
        // Only APIC/IOAPIC MMIO violations are deterministic. PEBS exits can
        // skid; CoW faults, stale TLB hits etc. depend on host state.
        ExitReason::EptViolation => {
            let ept_qual = EptViolationQualification::from(qual);
            if ept_qual.asynchronous && ept_qual.write {
                true
            } else {
                let gpa = ctx
                    .state()
                    .vmcs
                    .read64(VmcsField64::GuestPhysicalAddr)
                    .unwrap_or(0);
                !((APIC_BASE..APIC_BASE + APIC_SIZE).contains(&gpa)
                    || (IOAPIC_BASE..IOAPIC_BASE + IOAPIC_SIZE).contains(&gpa))
            }
        }
        // Deterministic only on a target boundary or inside the single-step
        // range; margin steps depend on PEBS skid.
        ExitReason::MonitorTrapFlag => {
            let count = ctx.state().last_instruction_count;
            let tsc = count + ctx.state().tsc_offset;
            let on_target = |t: Option<u64>| matches!(t, Some(target) if count == target);
            let stop_at_count = ctx
                .state()
                .stop_at_tsc
                .map(|t| t.saturating_sub(ctx.state().tsc_offset));
            let on_boundary = on_target(next_timer_exit_count(ctx))
                || on_target(next_io_channel_exit_count(ctx))
                || on_target(stop_at_count);
            let in_single_step_range = match ctx.state().single_step_tsc_range {
                Some((start, end)) => tsc >= start && tsc < end,
                None => false,
            };
            !(on_boundary || in_single_step_range)
        }
        _ => false,
    };

    ctx.state_mut().last_exit_deterministic = !non_deterministic_exit;

    // tsc_offset is advanced by HLT/MWAIT.
    if !non_deterministic_exit {
        let tsc = ctx.state().last_instruction_count + ctx.state().tsc_offset;
        ctx.state_mut().emulated_tsc = tsc;
    }

    // Handle before logging/threshold checks so device state is complete if we
    // return to userspace (e.g. for forking).
    let result = match reason {
        ExitReason::Cpuid => handle_cpuid(ctx),
        ExitReason::MsrRead => handle_msr_read(ctx),
        ExitReason::MsrWrite => handle_msr_write(ctx),
        ExitReason::CrAccess => handle_cr_access(ctx, CrAccessQualification::from(qual)),
        ExitReason::IoInstruction => handle_io(ctx, IoQualification::from(qual)),
        ExitReason::EptViolation => {
            handle_ept_violation(ctx, EptViolationQualification::from(qual), allocator)
        }
        ExitReason::ExceptionNmi => handle_exception_nmi(ctx),
        ExitReason::Xsetbv => handle_xsetbv(ctx),

        ExitReason::Rdtsc => handle_rdtsc(ctx),
        ExitReason::Rdtscp => handle_rdtscp(ctx),
        ExitReason::Rdpmc => handle_rdpmc(ctx),

        ExitReason::Rdrand => handle_rdrand(ctx),
        ExitReason::Rdseed => handle_rdseed(ctx),

        ExitReason::MonitorTrapFlag => ExitHandlerResult::Continue,

        ExitReason::Hlt => handle_idle(ctx),

        ExitReason::Mwait => handle_idle(ctx),

        ExitReason::Monitor => {
            // Never arm the monitor hardware, so MWAIT's qualification is
            // always 0 regardless of host timing. MWAIT wakes via the timer
            // deadline, not memory stores.
            if let Err(e) = advance_rip(ctx) {
                return ExitHandlerResult::Error(e);
            }
            ExitHandlerResult::Continue
        }

        ExitReason::TripleFault => {
            dump_triple_fault_state(ctx);
            ExitHandlerResult::Error(EE::TripleFault)
        }

        ExitReason::InvalidGuestState => ExitHandlerResult::Error(EE::InvalidGuestState),

        ExitReason::Vmcall => handle_vmcall(ctx, allocator),

        // No nested VMX.
        ExitReason::Vmclear
        | ExitReason::Vmlaunch
        | ExitReason::Vmptrld
        | ExitReason::Vmptrst
        | ExitReason::Vmread
        | ExitReason::Vmresume
        | ExitReason::Vmwrite
        | ExitReason::Vmxoff
        | ExitReason::Vmxon => {
            // Could inject #UD instead.
            ExitHandlerResult::ExitToUserspace(reason)
        }

        // Userspace heartbeat (serial output, signals); it just calls RUN again.
        ExitReason::VmxPreemptionTimer => {
            // ~10ms
            if ctx
                .state()
                .vmcs
                .write32(VmcsField32::VmxPreemptionTimerValue, 0x100000)
                .is_err()
            {
                return ExitHandlerResult::Error(EE::Fatal("Failed to reset preemption timer"));
            }
            ExitHandlerResult::ExitToUserspace(reason)
        }

        // Delivered through the host IDT by briefly enabling interrupts.
        ExitReason::ExternalInterrupt => {
            handle_external_interrupt(kernel);
            ExitHandlerResult::Continue
        }

        // inject_pending_interrupt() injects on the next VM entry.
        ExitReason::InterruptWindow => {
            if let Err(e) = disable_interrupt_window_exiting(ctx) {
                return ExitHandlerResult::Error(e);
            }
            ExitHandlerResult::Continue
        }

        ExitReason::Init
        | ExitReason::Sipi
        | ExitReason::NmiWindow
        | ExitReason::TprBelowThreshold
        | ExitReason::ApicAccess
        | ExitReason::ApicWrite => ExitHandlerResult::ExitToUserspace(reason),

        _ => ExitHandlerResult::ExitToUserspace(reason),
    };

    // Margin-window MTF steps get a separate bucket so `mtf.count` stays
    // reproducible (the determinism harness compares it).
    let end_tsc = rdtsc();
    let cycles = end_tsc.saturating_sub(start_tsc);
    if reason == ExitReason::MonitorTrapFlag && non_deterministic_exit {
        ctx.state_mut().exit_stats.pebs_margin_steps += 1;
    } else {
        ctx.state_mut().exit_stats.record(reason, cycles);
    }

    // Unconditional: the margin window must engage on the (non-deterministic)
    // PEBS exit and persist through the margin steps.
    if let Err(e) = update_mtf_state(ctx) {
        return ExitHandlerResult::Error(e);
    }

    if !non_deterministic_exit {
        if let Some(stop_tsc) = ctx.state().stop_at_tsc {
            if ctx.state().emulated_tsc >= stop_tsc {
                let apic = &ctx.state().devices.apic;
                log_err!(
                    "STOP-AT-TSC: exit={:?}, tsc={}, deadline={}, initial={}, lvt_timer={:#x}, irr[7]={:#x}, isr[7]={:#x}\n",
                    reason,
                    ctx.state().emulated_tsc,
                    apic.timer_deadline,
                    apic.timer_initial,
                    apic.lvt_timer,
                    apic.irr[7],
                    apic.isr[7]
                );
                ctx.state_mut().capture_exit_at_shutdown();
                return ExitHandlerResult::ExitToUserspace(ExitReason::StopTscReached);
            }
        }
    }

    // After the stop check: a buffer-drain round-trip re-entering the guest
    // first would make StopTscReached non-deterministic.
    if ctx.state().exit_capture_enabled() {
        ctx.state_mut()
            .capture_exit(reason, qual, !non_deterministic_exit);
    }

    // Force a drain round-trip if the buffer filled and we would re-enter; the
    // staged record is re-appended on the next RUN via `event_clear()`. Guest
    // state is untouched, so this stays deterministic.
    if ctx.state().event_buffer_full() && matches!(result, ExitHandlerResult::Continue) {
        return ExitHandlerResult::ExitToUserspace(ExitReason::EventBufferFull);
    }

    result
}

#[cfg(test)]
mod single_step_target_tests {
    use super::*;
    use crate::tests::MockVmContext;

    /// The window start is a precise-exit target only while before the window.
    #[test]
    fn single_step_start_armed_before_window_only() {
        let mut ctx = MockVmContext::new();
        ctx.state_mut().single_step_tsc_range = Some((10_000, 20_000));
        ctx.state_mut().tsc_offset = 1_000;

        ctx.state_mut().last_instruction_count = 5_000; // emulated_tsc = 6_000
        assert_eq!(next_single_step_start_tsc(&ctx), Some(10_000));
        assert_eq!(next_single_step_start_count(&ctx), Some(9_000));

        ctx.state_mut().last_instruction_count = 9_000; // emulated_tsc = 10_000
        assert_eq!(next_single_step_start_tsc(&ctx), None);
        assert_eq!(next_single_step_start_count(&ctx), None);

        ctx.state_mut().last_instruction_count = 14_000; // emulated_tsc = 15_000
        assert_eq!(next_single_step_start_tsc(&ctx), None);

        ctx.state_mut().single_step_tsc_range = None;
        ctx.state_mut().last_instruction_count = 5_000;
        assert_eq!(next_single_step_start_tsc(&ctx), None);
        assert_eq!(next_single_step_start_count(&ctx), None);
    }
}
