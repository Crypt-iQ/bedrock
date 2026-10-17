// SPDX-License-Identifier: GPL-2.0

//! Interrupt injection and APIC timer handling.

use core::arch::asm;

use super::helpers::{inject_exception, ExitError};
use super::pebs::arm_for_next_iteration;
use super::qualifications::InterruptionInfo;

/// IOAPIC pin for the hypervisor-guest I/O channel. Advertised in the MP table
/// so `bedrock-io.ko` can `request_irq()` it; delivered via
/// [`ioapic_deliver_irq`]. ISA IRQ 9 is normally ACPI, which bedrock guests
/// lack, and no emulated device uses it.
pub const IO_CHANNEL_IRQ: u8 = 9;

#[cfg(not(feature = "cargo"))]
use super::super::prelude::*;
#[cfg(feature = "cargo")]
use crate::prelude::*;

/// Set the timer vector in IRR if the APIC timer has expired (per emulated TSC).
pub fn check_apic_timer<C: VmContext>(ctx: &mut C) {
    let current_tsc = ctx.state().emulated_tsc;
    let timer_deadline = ctx.state().devices.apic.timer_deadline;
    let svr = ctx.state().devices.apic.svr;
    let lvt_timer_init = ctx.state().devices.apic.lvt_timer;

    if timer_deadline == 0 {
        return;
    }
    if current_tsc < timer_deadline {
        return;
    }
    if (svr & (1 << 8)) == 0 {
        return;
    }
    if (lvt_timer_init & (1 << 16)) != 0 {
        return;
    }

    // The precise PEBS+MTF path lands exactly on the deadline; later means
    // the timer is delivered late on some subsequent deterministic exit.
    if current_tsc > timer_deadline {
        ctx.state_mut().exit_stats.apic_timer_late_inject += 1;
    }

    let apic = &mut ctx.state_mut().devices.apic;

    let vector = (apic.lvt_timer & 0xFF) as u8;

    let irr_index = (vector / 32) as usize;
    let irr_bit = 1u32 << (vector % 32);
    apic.irr[irr_index] |= irr_bit;

    // Bit 17: periodic.
    if (apic.lvt_timer & (1 << 17)) != 0 {
        let divisor = apic_timer_divisor(apic.timer_divide);
        let ticks = u64::from(apic.timer_initial) * u64::from(divisor);
        apic.timer_deadline = current_tsc.wrapping_add(ticks);
    } else {
        apic.timer_deadline = 0;
    }

    // `target_tsc` vs the header's emit tsc exposes scheduled-vs-delivered
    // drift. Buffer-full is handled by the exit dispatcher.
    let payload = InjectPayload {
        vector,
        source: InjectSource::Timer as u8,
        _pad: [0; 6],
        target_tsc: timer_deadline,
    };
    let _ = ctx
        .state_mut()
        .event_append(EventKind::Inject, payload.as_bytes());

    // After the inject record, so the dump can never take its staging slot.
    if current_tsc > timer_deadline {
        super::pebs::pmi_trace_dump(ctx, timer_deadline, current_tsc);
    }
}

/// Raise the I/O channel IRQ for a queued, undelivered request once the guest
/// has registered the page and unmasked `redtbl[IO_CHANNEL_IRQ]`. Until then
/// the request stays pending; `request_delivered` is set only after IRR is
/// actually raised so early requests aren't dropped.
pub fn check_io_channel<C: VmContext>(ctx: &mut C) {
    let chan = &ctx.state().io_channel;
    if chan.page_gpa == 0 {
        return;
    }
    if chan.request_len == 0 {
        return;
    }
    if chan.request_delivered {
        return;
    }
    // Wait for the target TSC. PEBS+MTF normally lands exactly on it; any
    // later exit is the fallback when arming wasn't possible.
    if chan.request_target_tsc != 0 && ctx.state().emulated_tsc < chan.request_target_tsc {
        return;
    }

    let entry = ctx.state().devices.ioapic.redtbl[IO_CHANNEL_IRQ as usize];
    // Masked or vector < 16: guest hasn't wired up the IRQ yet.
    if (entry >> 16) & 1 != 0 || (entry & 0xFF) < 16 {
        return;
    }

    ioapic_deliver_irq(ctx, IO_CHANNEL_IRQ);
    ctx.state_mut().io_channel.request_delivered = true;

    // Own event category so request firing is capturable without full `Exit`
    // capture. Buffer-full is handled centrally.
    let target_tsc = ctx.state().io_channel.request_target_tsc;
    let payload = IoChannelPayload {
        phase: IoChannelPhase::Request as u8,
        _pad: [0; 7],
        target_tsc,
    };
    let _ = ctx.state_mut().event_emit_io_channel(&payload);
}

/// Calculate APIC timer divisor from the Divide Configuration Register (DCR).
fn apic_timer_divisor(dcr: u32) -> u32 {
    let encoded = ((dcr >> 1) & 0b100) | (dcr & 0b11);
    match encoded {
        0b000 => 2,
        0b001 => 4,
        0b010 => 8,
        0b011 => 16,
        0b100 => 32,
        0b101 => 64,
        0b110 => 128,
        0b111 => 1,
        _ => 1,
    }
}

/// Highest-priority vector pending in the APIC IRR.
fn apic_pending_vector<C: VmContext>(ctx: &C) -> Option<u8> {
    let apic = &ctx.state().devices.apic;

    // SVR bit 8: APIC enabled.
    if (apic.svr & (1 << 8)) == 0 {
        return None;
    }

    for i in (0..8).rev() {
        if apic.irr[i] != 0 {
            let bit = 31 - apic.irr[i].leading_zeros();
            return Some((i * 32 + bit as usize) as u8);
        }
    }
    None
}

/// Enable interrupt-window exiting so we get a VM exit when the guest becomes interruptible.
pub fn enable_interrupt_window_exiting<C: VmContext>(ctx: &mut C) -> Result<(), ExitError> {
    let controls = ctx
        .state()
        .vmcs
        .read32(VmcsField32::PrimaryProcBasedVmExecControls)
        .map_err(ExitError::VmcsReadError)?;
    if controls & cpu_based::INTR_WINDOW_EXITING == 0 {
        ctx.state()
            .vmcs
            .write32(
                VmcsField32::PrimaryProcBasedVmExecControls,
                controls | cpu_based::INTR_WINDOW_EXITING,
            )
            .map_err(ExitError::VmcsWriteError)?;
    }
    Ok(())
}

/// Disable interrupt-window exiting.
pub fn disable_interrupt_window_exiting<C: VmContext>(ctx: &mut C) -> Result<(), ExitError> {
    let controls = ctx
        .state()
        .vmcs
        .read32(VmcsField32::PrimaryProcBasedVmExecControls)
        .map_err(ExitError::VmcsReadError)?;
    if controls & cpu_based::INTR_WINDOW_EXITING != 0 {
        ctx.state()
            .vmcs
            .write32(
                VmcsField32::PrimaryProcBasedVmExecControls,
                controls & !cpu_based::INTR_WINDOW_EXITING,
            )
            .map_err(ExitError::VmcsWriteError)?;
    }
    Ok(())
}

/// Re-inject an event whose IDT delivery was interrupted by the VM exit (e.g.
/// EPT violation pushing the frame), per SDM Vol 3C 29.2.4. Returns whether
/// an event was re-injected.
pub fn reinject_vectored_event<C: VmContext>(ctx: &mut C) -> Result<bool, ExitError> {
    let idt_info = ctx
        .state()
        .vmcs
        .read32(VmcsField32::IdtVectoringInfo)
        .map_err(ExitError::VmcsReadError)?;

    if idt_info & (1 << 31) == 0 {
        return Ok(false);
    }

    let vector = (idt_info & 0xFF) as u8;
    let int_type = (idt_info >> 8) & 0x7;

    log_info!(
        "IDT-vectoring: re-injecting interrupted event vector={} type={}\n",
        vector,
        int_type
    );

    // Identical formats (SDM Tables 26-18, 26-21).
    ctx.state()
        .vmcs
        .write32(VmcsField32::VmEntryInterruptionInfo, idt_info)
        .map_err(ExitError::VmcsWriteError)?;

    // Bit 11: error code valid.
    if idt_info & (1 << 11) != 0 {
        let error_code = ctx
            .state()
            .vmcs
            .read32(VmcsField32::IdtVectoringErrorCode)
            .map_err(ExitError::VmcsReadError)?;
        ctx.state()
            .vmcs
            .write32(VmcsField32::VmEntryExceptionErrorCode, error_code)
            .map_err(ExitError::VmcsWriteError)?;
    }

    Ok(true)
}

/// Inject any pending interrupt before VMLAUNCH/VMRESUME, and re-arm PEBS.
pub fn inject_pending_interrupt<C: VmContext>(ctx: &mut C) -> Result<(), ExitError> {
    // Skip new injection when:
    //   - an interrupted event was re-injected (SDM Vol 3C 29.2.4; it must
    //     complete first);
    //   - the last exit was non-deterministic: setting IRR there happens at a
    //     host-dependent point (e.g. a host NMI absorbing an interrupt-window
    //     exit);
    //   - an event is already pending in `VmEntryInterruptionInfo`.
    //
    // `check_apic_timer` must run before the re-arm below, which reads the
    // (possibly reloaded) periodic deadline.
    let inject_eligible =
        !reinject_vectored_event(ctx)? && ctx.state().last_exit_deterministic && {
            let pending = ctx
                .state()
                .vmcs
                .read32(VmcsField32::VmEntryInterruptionInfo)
                .unwrap_or(0);
            if pending & (1 << 31) != 0 {
                false
            } else {
                check_apic_timer(ctx);
                // After the timer, so the (usually higher-vector) timer wins
                // and ours waits in IRR.
                check_io_channel(ctx);
                true
            }
        };

    // Always re-arm: FIXED_CTR0 is reloaded with `counter_reload` on every
    // entry, so a stale value would overshoot by the instructions retired in
    // an interrupted iteration.
    arm_for_next_iteration(ctx);

    if !inject_eligible {
        return Ok(());
    }

    let vector = match apic_pending_vector(ctx) {
        Some(v) => v,
        None => return Ok(()),
    };

    let rflags = ctx
        .state()
        .vmcs
        .read_natural(VmcsFieldNatural::GuestRflags)
        .map_err(ExitError::VmcsReadError)?;
    if (rflags & (1 << 9)) == 0 {
        enable_interrupt_window_exiting(ctx)?;
        return Ok(());
    }

    let interruptibility = ctx
        .state()
        .vmcs
        .read32(VmcsField32::GuestInterruptibilityState)
        .map_err(ExitError::VmcsReadError)?;
    // Blocked by STI or MOV SS.
    if (interruptibility & 0x3) != 0 {
        enable_interrupt_window_exiting(ctx)?;
        return Ok(());
    }

    let info = InterruptionInfo::external_interrupt(vector);
    inject_exception(ctx, info, None)?;

    {
        let apic = &mut ctx.state_mut().devices.apic;
        let irr_index = (vector / 32) as usize;
        let bit = 1u32 << (vector % 32);
        apic.irr[irr_index] &= !bit;
        apic.isr[irr_index] |= bit;
    }

    disable_interrupt_window_exiting(ctx)?;

    Ok(())
}

/// Deliver an IOAPIC pin to the local APIC IRR unless masked.
pub fn ioapic_deliver_irq<C: VmContext>(ctx: &mut C, irq: u8) {
    if irq as usize >= IOAPIC_NUM_PINS {
        return;
    }

    let entry = ctx.state().devices.ioapic.redtbl[irq as usize];

    if (entry >> 16) & 1 != 0 {
        return;
    }

    let vector = (entry & 0xFF) as u8;
    if vector < 16 {
        return;
    }

    let irr_idx = (vector / 32) as usize;
    let irr_bit = 1u32 << (vector % 32);

    if irr_idx < 8 {
        ctx.state_mut().devices.apic.irr[irr_idx] |= irr_bit;
    }
}

/// Let a pending host interrupt be delivered through the IDT by briefly
/// enabling interrupts.
#[inline]
pub fn handle_external_interrupt<K: Kernel>(kernel: &K) {
    let _irq_window = ReverseIrqGuard::new(kernel);
    // SAFETY: NOP is a safe instruction; the IRQ window opened by ReverseIrqGuard
    // allows pending host interrupts to be delivered through the IDT.
    unsafe {
        asm!("nop", options(nomem, nostack, preserves_flags));
    }
}
