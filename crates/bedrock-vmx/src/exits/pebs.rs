// SPDX-License-Identifier: GPL-2.0

//! Precise VM exits via EPT-friendly PEBS, or a PMI fallback.
//!
//! PEBS on `IA32_FIXED_CTR0` overflows shortly before the target instruction;
//! the PEBS Buffer page is R+E in EPT, so the record write traps as an EPT
//! violation with the asynchronous bit set. The MTF margin path in
//! `exits/mod.rs` single-steps the last few instructions.
//!
//! Without EPT-friendly PEBS, `PreciseExitMode::Pmi` instead sets the PMI bit
//! of `IA32_FIXED_CTR0`: the overflow raises a PMI, which Linux routes through
//! LVTPC as an NMI, so it arrives as an NMI-exiting VM exit. PMI skid is far
//! larger than PDist's, so the margin (and thus MTF stepping) is wider.
//!
//! See SDM Vol 3B 21.9 (PEBS), 21.9.4 (Reduced Skid), 21.9.5 (EPT-Friendly);
//! Vol 3C Table 29-7 (exit qualification bit 16).

use super::ept::translate_gva_to_gpa;
use super::helpers::ExitHandlerResult;
use super::qualifications::EptViolationQualification;
use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(not(feature = "cargo"))]
use super::super::prelude::*;
#[cfg(feature = "cargo")]
use crate::prelude::*;

/// Offset of the PEBS Buffer within the scratch page; the DS Management Area
/// occupies offset 0 (~160 bytes).
pub const PEBS_BUFFER_OFFSET: u64 = 0x100;
/// PEBS Buffer size. Only the first record is ever written (it traps).
pub const PEBS_BUFFER_SIZE: u64 = 0x800;

/// Linux x86_64 direct-map base with KASLR disabled (required for
/// determinism, see `boot/constants.rs`). `IA32_DS_AREA` is resolved through
/// the *current* CR3, so it must be a kernel-half alias valid in every
/// process, not the registering process's userspace VA (which would #PF once
/// the guest schedules another process).
pub const GUEST_LINUX_PAGE_OFFSET: u64 = 0xffff_8880_0000_0000;

/// DS Management Area layout for adaptive (format >= 4) PEBS. Accessed via
/// linear addresses (guest paging, then EPT). See SDM Vol 3B Figure 21-69.
#[repr(C)]
#[derive(Default)]
pub struct DsManagementArea {
    pub bts_buffer_base: u64,
    pub bts_index: u64,
    pub bts_absolute_maximum: u64,
    pub bts_interrupt_threshold: u64,
    /// Linear address of the first byte of the PEBS Buffer.
    pub pebs_buffer_base: u64,
    /// Linear address of the next PEBS record (updated by hardware).
    pub pebs_index: u64,
    /// Linear address one past the end of the PEBS Buffer.
    pub pebs_absolute_maximum: u64,
    /// PEBS index threshold at which the PEBS PMI is signalled.
    pub pebs_interrupt_threshold: u64,
    /// PEBS record reload values for IA32_PMC0..IA32_PMC7 (offsets 0x40..0x78).
    pub pebs_gp_counter_reset: [u64; 8],
    /// PEBS record reload values for IA32_FIXED_CTR0..IA32_FIXED_CTR3
    /// (offsets 0x80..0x98).
    pub pebs_fixed_counter_reset: [u64; 4],
}

/// Action taken when the PEBS-induced EPT violation fires.
#[derive(Debug, Clone, Copy)]
pub enum PebsAction {
    /// Inject the given external interrupt vector and resume the guest.
    InjectInterrupt(u8),
}

/// How precise-exit counter overflows reach the hypervisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreciseExitMode {
    /// EPT-friendly PEBS + PDist: the record write traps as an EPT violation.
    Pebs,
    /// Counter-overflow PMI, delivered as a host NMI (NMI exiting).
    Pmi,
}

/// Host PMU MSRs saved by `pebs_pre_vm_entry`, restored by `pebs_post_vm_exit`.
#[derive(Default, Clone, Copy)]
pub struct PebsHostMsrs {
    pub pebs_enable: u64,
    pub ds_area: u64,
    pub fixed_ctr_ctrl: u64,
    pub fixed_ctr0: u64,
    pub pebs_data_cfg: u64,
}

/// Per-VM PEBS state for precise exits. `Some` in `VmState` once
/// `HYPERCALL_REGISTER_PEBS_PAGE` has installed a scratch page.
pub struct PebsState {
    /// Overflow delivery mechanism, fixed at registration.
    pub mode: PreciseExitMode,
    /// GPA of the page holding the DS Management Area.
    pub ds_management_gpa: u64,
    /// GPA of the PEBS Buffer; same page as `ds_management_gpa`, R+E in EPT.
    pub pebs_buffer_gpa: u64,
    /// Guest linear address loaded into `IA32_DS_AREA`.
    pub ds_area_va: u64,
    /// Action for the next PEBS-induced EPT violation; `None` = disarmed.
    pub armed_action: Option<PebsAction>,
    /// PEBS firing point of the last arming (`target_tsc - margin`, not the
    /// final deadline). Skid is measured against this.
    pub armed_target_tsc: u64,
    /// `last_instruction_count` at the last arming; separates hardware skid
    /// from HLT/MWAIT time-warp.
    pub armed_inst_count: u64,
    /// `tsc_offset` at the last arming; a fire-time diff reveals HLT/MWAIT
    /// clamps between arming and firing.
    pub armed_tsc_offset: u64,
    /// Run-loop iterations entered with an arming still live since the last
    /// successful arming. Non-zero at fire time means a stale arming fired.
    pub iters_since_arm: u32,
    /// `IA32_FIXED_CTR0` reload value for the next entry.
    pub counter_reload: u64,
    /// `IA32_FIXED_CTR_CTRL` loaded on armed entry (`FIXED_CTR_CTRL_FC0_OS_USR`).
    pub fixed_ctr_ctrl: u64,
    /// `IA32_PEBS_ENABLE` loaded on armed entry (bit 32: `IA32_FIXED_CTR0`).
    pub pebs_enable: u64,
    /// `MSR_PEBS_DATA_CFG`; 0 = Basic Info record only (records are never read).
    pub pebs_data_cfg: u64,
    /// Host PMU MSRs saved at the last armed VM-entry.
    pub host_msrs: PebsHostMsrs,
    /// Gates the one-time "first arm" log line per VM.
    pub logged_first_arm: bool,
    /// PMI mode: the last NMI exit was our `IA32_FIXED_CTR0` overflow (set by
    /// `pebs_post_vm_exit`, consumed by `handle_exception_nmi`).
    pub pmi_fired: bool,
    /// PMI mode: with `pmi_fired`, other counters had overflowed too, so the
    /// NMI must still reach the host handler (e.g. the NMI watchdog's sample).
    pub pmi_forward: bool,
}

impl PebsState {
    /// Inherit PEBS registration into a forked child, which never re-issues
    /// `HYPERCALL_REGISTER_PEBS_PAGE`. Registration constants are copied;
    /// runtime fields are reset so the child arms fresh.
    ///
    /// The scratch page is shared with the parent via the EPT clone (R+E leaf
    /// preserved); safe because parent and child never run concurrently.
    pub fn clone_for_fork(&self) -> Self {
        Self {
            mode: self.mode,
            ds_management_gpa: self.ds_management_gpa,
            pebs_buffer_gpa: self.pebs_buffer_gpa,
            ds_area_va: self.ds_area_va,
            fixed_ctr_ctrl: self.fixed_ctr_ctrl,
            pebs_enable: self.pebs_enable,
            pebs_data_cfg: self.pebs_data_cfg,
            armed_action: None,
            armed_target_tsc: 0,
            armed_inst_count: 0,
            armed_tsc_offset: 0,
            iters_since_arm: 0,
            counter_reload: 0,
            host_msrs: PebsHostMsrs::default(),
            logged_first_arm: false,
            pmi_fired: false,
            pmi_forward: false,
        }
    }
}

/// `IA32_FIXED_CTR_CTRL` enabling only `IA32_FIXED_CTR0` in OS+USR (no
/// AnyThread/PMI/Adaptive_Record); the instruction counter uses `IA32_PMC0`.
/// FIXED_CTR0 is used for PEBS because SPR lists its PDist as
/// instruction-granularity (SDM Vol 3B Table 21-51, Figure 21-2).
const FIXED_CTR_CTRL_FC0_OS_USR: u64 = 0b11;

/// `FIXED_CTR_CTRL_FC0_OS_USR` plus the FC0 PMI-on-overflow bit (bit 3).
const FIXED_CTR_CTRL_FC0_OS_USR_PMI: u64 = 0b1011;

/// `IA32_PERF_GLOBAL_STATUS` overflow bit for `IA32_FIXED_CTR0`.
const GLOBAL_STATUS_FIXED_CTR0_OVF: u64 = 1 << 32;

/// Retired instructions by which the PMI exit is aimed before the target.
/// PMI delivery (LAPIC NMI) skids far more than PDist: up to 239 seen on Ice
/// Lake-SP. `max_pebs_skid` reports the observed worst case for tuning.
const PMI_MARGIN: u64 = 320;

/// Smallest target distance worth arming for: `PEBS_MIN_DELTA` for PDist; a
/// PMI has no such floor, so its MTF window is only the margin wide.
pub fn min_arm_delta(mode: PreciseExitMode) -> u64 {
    match mode {
        PreciseExitMode::Pebs => PEBS_MIN_DELTA,
        PreciseExitMode::Pmi => PEBS_MIN_DELTA, // TODO: Changeme
    }
}

/// Bit 32 of `IA32_PERF_GLOBAL_CTRL` enables `IA32_FIXED_CTR0`.
pub const PERF_GLOBAL_CTRL_FIXED_CTR0: u64 = 1 << 32;

/// Performance counter width; reload values are masked to it (SDM Vol 3B
/// 21.2.8).
const PMC_COUNTER_WIDTH_BITS: u32 = 48;
const PMC_COUNTER_MASK: u64 = (1u64 << PMC_COUNTER_WIDTH_BITS) - 1;

/// Minimum encoded delta (reload distance from overflow) for PDist. The SDM
/// minimum is 256 (Vol 3B 21.9.6); 257 keeps the borderline case on the MTF
/// path (`update_mtf_state`).
pub const PEBS_MIN_DELTA: u64 = 257;

/// Retired instructions by which the precise exit fires *before* the target;
/// `update_mtf_state` single-steps the rest. Absorbs the occasional skid of
/// the asynchronous record write (PEBS, model-specific, resolved from host
/// CPUID once and cached) or of PMI delivery.
pub fn get_pebs_margin(mode: PreciseExitMode) -> u64 {
    if mode == PreciseExitMode::Pmi {
        return PMI_MARGIN;
    }
    // `u64::MAX` = not yet resolved.
    let cached = PEBS_MARGIN_CACHE.load(Ordering::Relaxed);
    if cached != u64::MAX {
        return cached;
    }
    // Racing callers compute the same value, so the race is benign.
    let margin = margin_for_host_cpu();
    PEBS_MARGIN_CACHE.store(margin, Ordering::Relaxed);
    margin
}

/// Cache backing [`get_pebs_margin`].
static PEBS_MARGIN_CACHE: AtomicU64 = AtomicU64::new(u64::MAX);

fn margin_for_host_cpu() -> u64 {
    let (family, model) = host_family_model();
    // Tuned on the Bitcoin workload (0 late timer injects). Model numbers
    // from Linux `intel-family.h`.
    match (family, model) {
        (0x6, 0x8F) => 3,  // Sapphire Rapids-SP (ex: Xeon Gold 5412U)
        (0x6, 0x6A) => 12, // Ice Lake-SP (ex: Xeon Silver 4310); skid up to 9 seen
        _ => 8,            // default for untested models
    }
}

/// Whether the host has EPT-friendly PEBS, given `IA32_PERF_CAPABILITIES`.
/// Needs PEBS_BASELINE (bit 14) plus either PEBS_FMT (bits 11:8) >= 5 or a
/// format-4 part listed as EPT-friendly: Ice Lake server and later, and 12th
/// gen Core and later (SDM Vol 3B 21.9.5, Tables 21-51/21-53). Other
/// format-4 parts (client Ice Lake, Tiger Lake, Rocket Lake, Tremont) are
/// not: DS Area accesses there must never fault or exit (Vol 3B 19.4.9),
/// which the R+E scratch page relies on. Linux `x86_pmu.pebs_ept` is similar
/// but omits the client hybrid parts.
pub fn host_has_ept_friendly_pebs(perf_capabilities: u64) -> bool {
    let baseline = (perf_capabilities >> 14) & 1 != 0;
    let format = (perf_capabilities >> 8) & 0xF;
    if !baseline || format < 4 {
        return false;
    }
    if format >= 5 {
        return true;
    }
    matches!(
        host_family_model(),
        (0x6, 0x6A)     // Ice Lake-SP
            | (0x6, 0x6C) // Ice Lake-D
            | (0x6, 0x8F) // Sapphire Rapids-SP
            | (0x6, 0xCF) // Emerald Rapids
            | (0x6, 0xAD) // Granite Rapids-SP
            | (0x6, 0xAE) // Granite Rapids-D
            | (0x6, 0x97) // Alder Lake
            | (0x6, 0x9A) // Alder Lake-L
            | (0x6, 0xBE) // Alder Lake-N
            | (0x6, 0xB7) // Raptor Lake
            | (0x6, 0xBA) // Raptor Lake-P
            | (0x6, 0xBF) // Raptor Lake-S
            | (0x6, 0xAA) // Meteor Lake-L
            | (0x6, 0xAC) // Meteor Lake
    )
}

/// Host family/model from CPUID.01H:EAX (SDM Vol 1 21.3).
fn host_family_model() -> (u32, u32) {
    let (eax, _, _, _) = super::cpuid::cpuid(1, 0);

    let base_model = (eax >> 4) & 0xF;
    let base_family = (eax >> 8) & 0xF;
    let ext_model = (eax >> 16) & 0xF;
    let ext_family = (eax >> 20) & 0xFF;

    // Extended model applies for family 06H/0FH.
    let model = if base_family == 0x6 || base_family == 0xF {
        (ext_model << 4) | base_model
    } else {
        base_model
    };

    // Extended family applies for family 0FH.
    let family = if base_family == 0xF {
        base_family + ext_family
    } else {
        base_family
    };

    (family, model)
}

/// Outcome of arming a precise exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmResult {
    Armed,
    /// `target_tsc < current_tsc`.
    AlreadyPast,
    /// Delta below `min_arm_delta + margin`; left to `update_mtf_state`.
    BelowMinDelta,
    /// No `PebsState` is installed (no scratch page registered).
    NotRegistered,
}

/// Arm a precise exit at `target_tsc - margin`; `update_mtf_state` steps the
/// rest. The emulated TSC ticks once per retired instruction, so TSC deltas
/// are instruction deltas.
pub fn arm_precise_exit(
    pebs: &mut PebsState,
    current_tsc: u64,
    target_tsc: u64,
    action: PebsAction,
    inst_count_now: u64,
    tsc_offset_now: u64,
) -> ArmResult {
    if target_tsc < current_tsc {
        return ArmResult::AlreadyPast;
    }
    let pebs_margin = get_pebs_margin(pebs.mode);
    let delta = target_tsc - current_tsc;
    if delta < min_arm_delta(pebs.mode) + pebs_margin {
        return ArmResult::BelowMinDelta;
    }
    let encoded_delta = delta - pebs_margin;
    // Overflows after `encoded_delta` increments; PDist records on the
    // overflowing instruction.
    pebs.counter_reload = encoded_delta.wrapping_neg() & PMC_COUNTER_MASK;
    pebs.armed_target_tsc = target_tsc - pebs_margin;
    pebs.armed_inst_count = inst_count_now;
    pebs.armed_tsc_offset = tsc_offset_now;
    pebs.iters_since_arm = 0;
    pebs.armed_action = Some(action);
    if !pebs.logged_first_arm {
        pebs.logged_first_arm = true;
        log_info!(
            "pebs: first arm: fixed_ctr_ctrl=0x{:x} pebs_enable=0x{:x} pebs_data_cfg=0x{:x} reload=0x{:x} delta={}\n",
            pebs.fixed_ctr_ctrl,
            pebs.pebs_enable,
            pebs.pebs_data_cfg,
            pebs.counter_reload,
            encoded_delta,
        );
    }
    ArmResult::Armed
}

/// Disarm any pending precise exit. Idempotent.
pub fn disarm_precise_exit(pebs: &mut PebsState) {
    pebs.armed_action = None;
}

/// Pre-VM-entry: arm PEBS for the earliest pending deadline (APIC timer, I/O
/// channel, `stop_at_tsc`, single-step start), or disarm if none. Arming for
/// `stop_at_tsc` makes stops exact; the dispatcher's coarse check alone only
/// fires on the next deterministic exit past it. Called from
/// `inject_pending_interrupt`.
pub fn arm_for_next_iteration<C: VmContext>(ctx: &mut C) {
    // Not `emulated_tsc`: it is only refreshed on deterministic exits and
    // lags after a non-deterministic one. `last_instruction_count` is updated
    // on every exit and `tsc_offset` only by (deterministic) HLT/MWAIT clamps.
    let inst_count = ctx.state().last_instruction_count;
    let tsc_offset = ctx.state().tsc_offset;
    let current = inst_count + tsc_offset;
    let apic_deadline = ctx.state().devices.apic.timer_deadline;
    let apic_vector = (ctx.state().devices.apic.lvt_timer & 0xFF) as u8;

    // Same readiness predicate as the MTF margin path.
    let io_channel_deadline = super::next_io_channel_target_tsc(ctx);

    let stop_deadline = ctx.state().stop_at_tsc;

    // Precise start keeps forks from beginning single-step logging on
    // different (possibly non-deterministic) exits.
    let single_step_start = super::next_single_step_start_tsc(ctx);

    // APIC deadline 0 = unset. The action's vector is informational (the
    // pre-entry path does the real injection), so the APIC vector is reused
    // for every target kind.
    let chosen_target = [
        Some(apic_deadline).filter(|&t| t != 0),
        io_channel_deadline,
        stop_deadline,
        single_step_start,
    ]
    .into_iter()
    .flatten()
    .min()
    .unwrap_or(0);

    let arm_result = {
        let pebs = match ctx.state_mut().pebs_state.as_deref_mut() {
            Some(p) => p,
            None => return,
        };

        // A successful re-arm below resets this to 0.
        let prev_armed = pebs.armed_action.is_some();
        if prev_armed {
            pebs.iters_since_arm = pebs.iters_since_arm.saturating_add(1);
        }

        let result = if chosen_target != 0 {
            let r = arm_precise_exit(
                pebs,
                current,
                chosen_target,
                PebsAction::InjectInterrupt(apic_vector),
                inst_count,
                tsc_offset,
            );
            // Otherwise the prior arming's stale counter_reload would be
            // reloaded and PEBS would fire far past the missed deadline.
            // The IRR-set path delivers the interrupt instead.
            if !matches!(r, ArmResult::Armed) {
                disarm_precise_exit(pebs);
            }
            Some(r)
        } else {
            disarm_precise_exit(pebs);
            None
        };
        (prev_armed, result)
    };

    let (prev_armed, arm_result) = arm_result;

    let stats = &mut ctx.state_mut().exit_stats;
    if prev_armed {
        stats.pebs_armed_iter_no_fire += 1;
    }
    match arm_result {
        Some(ArmResult::AlreadyPast) => stats.pebs_arm_already_past += 1,
        Some(ArmResult::BelowMinDelta) => stats.pebs_arm_below_min_delta += 1,
        _ => {}
    }
}

/// Snapshot host PMU MSRs and stage guest PEBS values in the VM-entry
/// MSR-load list, so they are written atomically with VM-entry. Loading a
/// guest `IA32_DS_AREA` in host context caused SMAP faults on bare metal
/// (a buffered record flushed into user memory on PEBS re-enable).
///
/// Repoints the MSR-load list from the instruction counter's page to the PEBS
/// page (entry 0 preserves `IA32_PMC0`); `pebs_post_vm_exit` switches back.
/// Called only when armed.
pub fn pebs_pre_vm_entry<C: VmContext, M: MsrAccess>(ctx: &mut C, msr: &M) {
    // The VM-exit MSR-load list already zeroed IA32_PEBS_ENABLE, so these
    // reads see clean host values.
    let pebs_mode = ctx
        .state()
        .pebs_state
        .as_deref()
        .expect("pebs_pre_vm_entry called without registered pebs_state")
        .mode;
    let is_pebs = pebs_mode == PreciseExitMode::Pebs;
    // PEBS-only MSRs may not exist in PMI mode.
    let read_pebs_msr = |index| {
        if is_pebs {
            msr.read_msr(index).unwrap_or(0)
        } else {
            0
        }
    };
    let host_pebs_enable = read_pebs_msr(msr::IA32_PEBS_ENABLE);
    let host_ds_area = read_pebs_msr(msr::IA32_DS_AREA);
    let host_fixed_ctr_ctrl = msr.read_msr(msr::IA32_FIXED_CTR_CTRL).unwrap_or(0);
    let host_fixed_ctr0 = msr.read_msr(msr::IA32_FIXED_CTR0).unwrap_or(0);
    let host_pebs_data_cfg = read_pebs_msr(msr::MSR_PEBS_DATA_CFG);

    let pmc0_value = ctx.state().instruction_counter.read();

    // Re-applied as the last MSR-load entry after reconfiguring the counters.
    let guest_perf_global_ctrl = ctx
        .state()
        .vmcs
        .read64(VmcsField64::GuestIa32PerfGlobalCtrl)
        .unwrap_or(0);

    let entry_load_va = ctx
        .state()
        .pebs_entry_msr_load_page
        .virtual_address()
        .as_u64();
    let entry_load_pa = ctx.state().pebs_entry_msr_load_page.physical_address();
    let pebs = ctx
        .state_mut()
        .pebs_state
        .as_deref_mut()
        .expect("pebs_pre_vm_entry called without registered pebs_state");
    pebs.host_msrs.pebs_enable = host_pebs_enable;
    pebs.host_msrs.ds_area = host_ds_area;
    pebs.host_msrs.fixed_ctr_ctrl = host_fixed_ctr_ctrl;
    pebs.host_msrs.fixed_ctr0 = host_fixed_ctr0;
    pebs.host_msrs.pebs_data_cfg = host_pebs_data_cfg;

    // Same order as `PEBS_ENTRY_MSR_INDEXES` in vm_state.rs:
    //   0 IA32_PMC0                  — preserve instruction counter
    //   1 IA32_PERF_GLOBAL_CTRL = 0  — disable counters before reconfig
    //   2 IA32_FIXED_CTR0            — counter reload value
    //   3 IA32_FIXED_CTR_CTRL        — enable FC0 in OS+USR
    //   4 MSR_PEBS_DATA_CFG          — 0 (basic record)
    //   5 IA32_DS_AREA
    //   6 IA32_PERF_GLOBAL_STATUS_RESET — clear overflow bits
    //   7 IA32_PEBS_ENABLE           — bit 32 (PEBS on FIXED_CTR0)
    //   8 IA32_PERF_GLOBAL_CTRL = guest — re-enable counters after reconfig
    //
    // Clearing overflow status (6) prevents a stale buffered record from
    // flushing on re-enable.
    let pebs_values = [
        pmc0_value,
        0,
        pebs.counter_reload,
        pebs.fixed_ctr_ctrl,
        pebs.pebs_data_cfg,
        pebs.ds_area_va,
        pebs.pebs_enable,
        pebs.pebs_enable,
        guest_perf_global_ctrl,
    ];
    // Same order as `PMI_ENTRY_MSR_INDEXES`: the PEBS list without its
    // PEBS-only MSRs; status reset keeps a stale overflow from raising a PMI
    // on entry.
    let pmi_values = [
        pmc0_value,
        0,
        pebs.counter_reload,
        pebs.fixed_ctr_ctrl,
        GLOBAL_STATUS_FIXED_CTR0_OVF,
        guest_perf_global_ctrl,
    ];
    let (indexes, values): (&[u32], &[u64]) = if is_pebs {
        (&PEBS_ENTRY_MSR_INDEXES, &pebs_values)
    } else {
        (&PMI_ENTRY_MSR_INDEXES, &pmi_values)
    };

    // 16-byte entries: u32 index, u32 reserved, u64 value (SDM Vol 3C Table
    // 26-16). Indexes are rewritten each time since the list depends on mode.
    // SAFETY: page is 4KB, page-aligned; we touch bytes within the at most
    // 9-entry MSR-load area (9 * 16 = 144), well within bounds.
    unsafe {
        for (i, (index, value)) in indexes.iter().zip(values).enumerate() {
            let entry_ptr = (entry_load_va as *mut u8).add(i * 16);
            core::ptr::write(entry_ptr.cast::<u32>(), *index);
            core::ptr::write(entry_ptr.add(8).cast::<u64>(), *value);
        }
    }

    let _ = ctx
        .state()
        .vmcs
        .write64(VmcsField64::VmEntryMsrLoadAddr, entry_load_pa.as_u64());
    let _ = ctx
        .state()
        .vmcs
        .write32(VmcsField32::VmEntryMsrLoadCount, values.len() as u32);

    // The LAPIC masks LVTPC on every PMI it delivers. `claim_pmi` and the
    // host NMI handler re-unmask it for our own PMIs, but any other PMI path
    // relies on Linux perf, which leaves it masked when it has no active
    // events; every later overflow would then be silently dropped. x2APIC
    // MSRs cannot go in the VM-entry MSR-load list (SDM Vol 3C 28.4), so
    // check here, last thing before entry.
    if !is_pebs && unmask_lvtpc(msr) {
        ctx.state_mut().exit_stats.pmi_lvtpc_unmasked += 1;
    }
}

/// Clear the LVTPC mask bit if set; returns whether it was masked. x2APIC is
/// a precondition of PMI mode (`host_supports_pmi_precise_exit`).
fn unmask_lvtpc<M: MsrAccess>(msr: &M) -> bool {
    match msr.read_msr(msr::X2APIC_LVTPC) {
        Ok(lvtpc) if lvtpc & APIC_LVT_MASKED != 0 => {
            let _ = msr.write_msr(msr::X2APIC_LVTPC, lvtpc & !APIC_LVT_MASKED);
            true
        }
        _ => false,
    }
}

/// Undo `pebs_pre_vm_entry`: hand the MSR-load list back to the instruction
/// counter and restore host PMU MSRs. Safe because the VM-exit MSR-load list
/// already zeroed `IA32_PEBS_ENABLE` (PEBS mode); in PMI mode, the host
/// `IA32_PERF_GLOBAL_CTRL` loaded at exit has FC0 cleared (see `run_loop`), so
/// FC0 stopped counting at the exit.
///
/// PMI mode also claims our overflow PMI here, where MSR access is at hand.
pub fn pebs_post_vm_exit<C: VmContext, M: MsrAccess>(ctx: &mut C, msr: &M) {
    if ctx.state().pebs_state.as_deref().map(|p| p.mode) == Some(PreciseExitMode::Pmi) {
        let claim = claim_pmi(ctx, msr);
        if let Some(p) = ctx.state_mut().pebs_state.as_deref_mut() {
            p.pmi_fired = claim != PmiClaim::NotOurs;
            p.pmi_forward = claim == PmiClaim::OursAndOthers;
        }
    }

    // Null instruction counter: no page, so load nothing.
    if let Some(ic_phys) = ctx.state().instruction_counter.msr_save_load_entry_phys() {
        let _ = ctx
            .state()
            .vmcs
            .write64(VmcsField64::VmEntryMsrLoadAddr, ic_phys);
        let _ = ctx
            .state()
            .vmcs
            .write32(VmcsField32::VmEntryMsrLoadCount, 1);
    } else {
        let _ = ctx
            .state()
            .vmcs
            .write32(VmcsField32::VmEntryMsrLoadCount, 0);
    }

    let pebs = ctx
        .state()
        .pebs_state
        .as_deref()
        .expect("pebs_post_vm_exit called without registered pebs_state");
    let _ = msr.write_msr(msr::IA32_FIXED_CTR_CTRL, pebs.host_msrs.fixed_ctr_ctrl);
    let _ = msr.write_msr(msr::IA32_FIXED_CTR0, pebs.host_msrs.fixed_ctr0);
    if pebs.mode == PreciseExitMode::Pebs {
        let _ = msr.write_msr(msr::IA32_DS_AREA, pebs.host_msrs.ds_area);
        let _ = msr.write_msr(msr::MSR_PEBS_DATA_CFG, pebs.host_msrs.pebs_data_cfg);
        let _ = msr.write_msr(msr::IA32_PEBS_ENABLE, pebs.host_msrs.pebs_enable);
    }
}

/// Outcome of `claim_pmi`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PmiClaim {
    /// Not an NMI exit, or `IA32_FIXED_CTR0` did not overflow.
    NotOurs,
    /// Our PMI alone: acked and LVTPC unmasked; the NMI can be swallowed.
    Ours,
    /// Our PMI coalesced with other counters' overflows: ours is acked, but
    /// the NMI must still go to the host handler, which acks the rest and
    /// unmasks LVTPC.
    OursAndOthers,
}

/// PMI mode: if this exit is an NMI and `IA32_FIXED_CTR0` overflowed, it is
/// our PMI. Ack the overflow and unmask LVTPC (LAPIC masks it on delivery;
/// perf's handler would normally unmask it), so `handle_exception_nmi` can
/// swallow the NMI instead of forwarding it to perf as an unknown NMI.
///
/// One NMI carries every counter's overflow, so if others are pending too
/// (e.g. the NMI watchdog's), LVTPC is left masked and the NMI forwarded;
/// unmasking with overflow bits still set would only re-raise the PMI.
///
/// An overflow that raced a different exit reason is left alone: its NMI is
/// still pending and the host NMI handler claims it. Either way the next
/// arming reloads the counter and the MTF margin covers the target.
fn claim_pmi<C: VmContext, M: MsrAccess>(ctx: &C, msr: &M) -> PmiClaim {
    let status = msr.read_msr(msr::IA32_PERF_GLOBAL_STATUS).unwrap_or(0);
    if status & GLOBAL_STATUS_FIXED_CTR0_OVF == 0 {
        return PmiClaim::NotOurs;
    }
    let vmcs = &ctx.state().vmcs;
    let reason = vmcs.read32(VmcsField32::VmExitReason).unwrap_or(u32::MAX) & 0xFFFF;
    let info = vmcs
        .read32(VmcsField32::VmExitInterruptionInfo)
        .unwrap_or(0);
    // Exit reason 0 with a valid NMI-type interruption (SDM Vol 3C 25.9.2).
    let is_nmi = reason == ExitReason::ExceptionNmi as u32
        && info & (1 << 31) != 0
        && (info >> 8) & 0x7 == 2;
    if !is_nmi {
        return PmiClaim::NotOurs;
    }
    let _ = msr.write_msr(
        msr::IA32_PERF_GLOBAL_STATUS_RESET,
        GLOBAL_STATUS_FIXED_CTR0_OVF,
    );
    if status & !GLOBAL_STATUS_FIXED_CTR0_OVF != 0 {
        return PmiClaim::OursAndOthers;
    }
    unmask_lvtpc(msr);
    PmiClaim::Ours
}

/// LVT mask bit (SDM Vol 3A Figure 13-8).
const APIC_LVT_MASKED: u64 = 1 << 16;

/// Whether the host can deliver precise exits by PMI: architectural PMU v2+
/// with at least one fixed counter (CPUID.0AH, SDM Vol 3B 21.2.2), and the
/// LAPIC in x2APIC mode so `claim_pmi` can unmask LVTPC with an MSR write.
pub fn host_supports_pmi_precise_exit<M: MsrAccess>(msr: &M) -> bool {
    let (eax, _, _, edx) = super::cpuid::cpuid(0xA, 0);
    let version = eax & 0xFF;
    let fixed_counters = edx & 0x1F;
    // IA32_APIC_BASE: bit 11 = enable, bit 10 = x2APIC.
    let x2apic = msr
        .read_msr(msr::IA32_APIC_BASE)
        .is_ok_and(|v| v & (0b11 << 10) == (0b11 << 10));
    version >= 2 && fixed_counters >= 1 && x2apic
}

/// True if the EPT violation is a PEBS record write. Qualification bit 16
/// (asynchronous) also covers Intel PT and user-interrupt delivery (SDM Vol 3C
/// Table 29-7), but only PEBS is in use here.
pub fn is_pebs_induced(qual: &EptViolationQualification) -> bool {
    qual.asynchronous && qual.write
}

/// `HYPERCALL_REGISTER_PEBS_PAGE` result, returned in guest RAX.
#[repr(u64)]
pub enum RegisterPebsPageResult {
    Success = 0,
    /// Host lacks both EPT-friendly PEBS (`PEBS_BASELINE` clear,
    /// `PEBS_FMT < 4`, or `IA32_PERF_CAPABILITIES` absent) and the PMI
    /// fallback (`host_supports_pmi_precise_exit`).
    Unsupported = u64::MAX,
    /// Guest virtual address is not 4KB-aligned.
    Misaligned = u64::MAX - 1,
    /// Guest page-table walk failed — page not currently mapped.
    Untranslatable = u64::MAX - 2,
    /// GPA not yet in EPT; the guest must touch/`mlock` the page first.
    NotEptMapped = u64::MAX - 3,
    /// `PebsState` was already registered for this VM.
    AlreadyRegistered = u64::MAX - 4,
}

/// Build the DS Management Area for the single scratch page (buffer on the
/// same page). The interrupt threshold equals the buffer base so the first
/// record crosses it; the PMI is never taken (the write traps first), but a
/// pending PMI may shorten the async record-write latency.
fn build_ds_management_area(page_va: u64) -> DsManagementArea {
    let buffer_base = page_va + PEBS_BUFFER_OFFSET;
    let buffer_max = buffer_base + PEBS_BUFFER_SIZE;
    DsManagementArea {
        bts_buffer_base: 0,
        bts_index: 0,
        bts_absolute_maximum: 0,
        bts_interrupt_threshold: 0,
        pebs_buffer_base: buffer_base,
        pebs_index: buffer_base,
        pebs_absolute_maximum: buffer_max,
        pebs_interrupt_threshold: buffer_base,
        pebs_gp_counter_reset: [0; 8],
        pebs_fixed_counter_reset: [0; 4],
    }
}

/// Serialize a `DsManagementArea` to its on-page byte layout.
fn ds_management_area_bytes(
    area: &DsManagementArea,
) -> [u8; core::mem::size_of::<DsManagementArea>()] {
    // SAFETY: `DsManagementArea` is `#[repr(C)]` with only u64 fields and no
    // padding; reinterpreting it as a byte array is well-defined.
    unsafe {
        core::mem::transmute::<DsManagementArea, [u8; core::mem::size_of::<DsManagementArea>()]>(
            DsManagementArea {
                bts_buffer_base: area.bts_buffer_base,
                bts_index: area.bts_index,
                bts_absolute_maximum: area.bts_absolute_maximum,
                bts_interrupt_threshold: area.bts_interrupt_threshold,
                pebs_buffer_base: area.pebs_buffer_base,
                pebs_index: area.pebs_index,
                pebs_absolute_maximum: area.pebs_absolute_maximum,
                pebs_interrupt_threshold: area.pebs_interrupt_threshold,
                pebs_gp_counter_reset: area.pebs_gp_counter_reset,
                pebs_fixed_counter_reset: area.pebs_fixed_counter_reset,
            },
        )
    }
}

/// Process `HYPERCALL_REGISTER_PEBS_PAGE` for the guest's 4KB scratch page at
/// `page_va` (must be mapped/`mlock`'d): write the DS Management Area, map the
/// page R+E in EPT so PEBS record writes trap, and install `PebsState`.
pub fn register_pebs_page<C: VmContext, A: CowAllocator<C::CowPage>>(
    ctx: &mut C,
    allocator: &mut A,
    page_va: u64,
) -> RegisterPebsPageResult {
    let mode = if ctx.state().pebs_supported {
        PreciseExitMode::Pebs
    } else if ctx.state().pmi_supported {
        PreciseExitMode::Pmi
    } else {
        // E.g. nested under KVM. Once registered, every VM-entry would touch
        // PMU MSRs and #GP.
        return RegisterPebsPageResult::Unsupported;
    };
    if page_va & 0xFFF != 0 {
        return RegisterPebsPageResult::Misaligned;
    }
    if ctx.state().pebs_state.is_some() {
        return RegisterPebsPageResult::AlreadyRegistered;
    }
    if mode == PreciseExitMode::Pmi {
        // No scratch page, EPT remap or VM-exit MSR-load list needed.
        enable_guest_fixed_ctr0(ctx);
        ctx.state_mut().pebs_state = Some(heap_box(PebsState {
            mode,
            ds_management_gpa: 0,
            pebs_buffer_gpa: 0,
            ds_area_va: 0,
            armed_action: None,
            armed_target_tsc: 0,
            armed_inst_count: 0,
            armed_tsc_offset: 0,
            iters_since_arm: 0,
            counter_reload: 0,
            fixed_ctr_ctrl: FIXED_CTR_CTRL_FC0_OS_USR_PMI,
            pebs_enable: 0,
            pebs_data_cfg: 0,
            host_msrs: PebsHostMsrs::default(),
            logged_first_arm: false,
            pmi_fired: false,
            pmi_forward: false,
        }));
        log_info!("HYPERCALL_REGISTER_PEBS_PAGE: no EPT-friendly PEBS, using PMI precise exits\n");
        return RegisterPebsPageResult::Success;
    }

    let gpa = match translate_gva_to_gpa(ctx, page_va) {
        Ok(g) => g,
        Err(()) => return RegisterPebsPageResult::Untranslatable,
    };
    let gpa_page = gpa.as_u64() & !0xFFF;

    if ctx
        .state()
        .ept
        .lookup(allocator, GuestPhysAddr::new(gpa_page))
        .is_none()
    {
        return RegisterPebsPageResult::NotEptMapped;
    }

    // Direct-map alias; see `GUEST_LINUX_PAGE_OFFSET`.
    let kernel_va = GUEST_LINUX_PAGE_OFFSET + gpa_page;

    // Host writes bypass EPT permissions.
    let area = build_ds_management_area(kernel_va);
    let bytes = ds_management_area_bytes(&area);
    if ctx
        .write_guest_memory(GuestPhysAddr::new(gpa_page), &bytes)
        .is_err()
    {
        return RegisterPebsPageResult::NotEptMapped;
    }

    // R+E so PEBS record writes trap.
    let host_phys = match ctx
        .state()
        .ept
        .lookup(allocator, GuestPhysAddr::new(gpa_page))
    {
        Some((hp, _)) => hp,
        None => return RegisterPebsPageResult::NotEptMapped,
    };
    if ctx
        .state_mut()
        .ept
        .remap_4k(
            allocator,
            GuestPhysAddr::new(gpa_page),
            host_phys,
            EptPermissions::READ_EXECUTE,
            EptMemoryType::WriteBack,
        )
        .is_err()
    {
        return RegisterPebsPageResult::NotEptMapped;
    }

    enable_guest_fixed_ctr0(ctx);

    // Zero IA32_PEBS_ENABLE atomically on every VM-exit. A record can still
    // be pending after the eventing instruction (SDM Vol 3B 21.9.4/21.9.5);
    // written in host context with the guest DS_AREA it would SMAP-fault.
    // Disabling at exit drops it (Vol 3C 26.7.2, 29.6).
    let exit_load_pa = ctx.state().pebs_exit_msr_load_page.physical_address();
    let _ = ctx
        .state()
        .vmcs
        .write64(VmcsField64::VmExitMsrLoadAddr, exit_load_pa.as_u64());
    let _ = ctx.state().vmcs.write32(VmcsField32::VmExitMsrLoadCount, 1);
    // The VM-entry MSR-load list is switched per-iteration by
    // `pebs_pre_vm_entry` / `pebs_post_vm_exit`.

    ctx.state_mut().pebs_state = Some(heap_box(PebsState {
        mode,
        ds_management_gpa: gpa_page,
        pebs_buffer_gpa: gpa_page,
        ds_area_va: kernel_va,
        armed_action: None,
        armed_target_tsc: 0,
        armed_inst_count: 0,
        armed_tsc_offset: 0,
        iters_since_arm: 0,
        counter_reload: 0,
        fixed_ctr_ctrl: FIXED_CTR_CTRL_FC0_OS_USR,
        // PEBS for IA32_FIXED_CTR0 (SDM Vol 3B Figure 21-68).
        pebs_enable: 1u64 << 32,
        pebs_data_cfg: 0,
        host_msrs: PebsHostMsrs::default(),
        logged_first_arm: false,
        pmi_fired: false,
        pmi_forward: false,
    }));

    log_info!(
        "HYPERCALL_REGISTER_PEBS_PAGE: user_va={:#x} kernel_va={:#x} gpa={:#x}\n",
        page_va,
        kernel_va,
        gpa_page
    );

    RegisterPebsPageResult::Success
}

/// Enable FIXED_CTR0 alongside the instruction counter's PMC0, in guest
/// context only (SDM Vol 3B 21.4.2).
fn enable_guest_fixed_ctr0<C: VmContext>(ctx: &mut C) {
    if let Ok(prev) = ctx
        .state()
        .vmcs
        .read64(VmcsField64::GuestIa32PerfGlobalCtrl)
    {
        let _ = ctx.state().vmcs.write64(
            VmcsField64::GuestIa32PerfGlobalCtrl,
            prev | PERF_GLOBAL_CTRL_FIXED_CTR0,
        );
    }
}

#[cfg(all(test, feature = "cargo"))]
mod tests {
    use super::*;

    fn make_pebs_state() -> PebsState {
        PebsState {
            mode: PreciseExitMode::Pebs,
            ds_management_gpa: 0x1000,
            pebs_buffer_gpa: 0x1000,
            ds_area_va: 0xffff_8000_0000_1000,
            armed_action: None,
            armed_target_tsc: 0,
            armed_inst_count: 0,
            armed_tsc_offset: 0,
            iters_since_arm: 0,
            counter_reload: 0,
            fixed_ctr_ctrl: FIXED_CTR_CTRL_FC0_OS_USR,
            pebs_enable: 1u64 << 32,
            pebs_data_cfg: 0,
            host_msrs: PebsHostMsrs::default(),
            logged_first_arm: false,
            pmi_fired: false,
            pmi_forward: false,
        }
    }

    #[test]
    fn arm_already_past_returns_already_past_and_does_not_arm() {
        let mut p = make_pebs_state();
        let r = arm_precise_exit(&mut p, 100, 50, PebsAction::InjectInterrupt(0x20), 0, 0);
        assert_eq!(r, ArmResult::AlreadyPast);
        assert!(p.armed_action.is_none());
    }

    #[test]
    fn arm_below_min_delta_returns_below_min_delta() {
        let mut p = make_pebs_state();
        let target = 100 + PEBS_MIN_DELTA + get_pebs_margin(PreciseExitMode::Pebs) - 1;
        let r = arm_precise_exit(&mut p, 100, target, PebsAction::InjectInterrupt(0x20), 0, 0);
        assert_eq!(r, ArmResult::BelowMinDelta);
        assert!(p.armed_action.is_none());
    }

    #[test]
    fn arm_minimum_delta_writes_correct_counter_reload() {
        let mut p = make_pebs_state();
        // Smallest delta that arms.
        let delta = PEBS_MIN_DELTA + get_pebs_margin(PreciseExitMode::Pebs);
        let r = arm_precise_exit(
            &mut p,
            100,
            100 + delta,
            PebsAction::InjectInterrupt(0x20),
            0,
            0,
        );
        assert_eq!(r, ArmResult::Armed);
        let encoded = delta - get_pebs_margin(PreciseExitMode::Pebs);
        assert_eq!(p.counter_reload, encoded.wrapping_neg() & PMC_COUNTER_MASK);
        // Tracks the firing point, not the requested target.
        assert_eq!(
            p.armed_target_tsc,
            100 + delta - get_pebs_margin(PreciseExitMode::Pebs)
        );
        assert!(matches!(
            p.armed_action,
            Some(PebsAction::InjectInterrupt(0x20))
        ));
    }

    #[test]
    fn arm_large_delta_writes_correct_counter_reload() {
        let mut p = make_pebs_state();
        let delta: u64 = 1_000_000;
        let r = arm_precise_exit(&mut p, 0, delta, PebsAction::InjectInterrupt(0x20), 0, 0);
        assert_eq!(r, ArmResult::Armed);
        let encoded = delta - get_pebs_margin(PreciseExitMode::Pebs);
        let expected = encoded.wrapping_neg() & PMC_COUNTER_MASK;
        assert_eq!(p.counter_reload, expected);
        let wrapped = p.counter_reload.wrapping_add(encoded) & PMC_COUNTER_MASK;
        assert_eq!(wrapped, 0);
    }

    #[test]
    fn arm_pmi_mode_uses_pmi_margin() {
        let mut p = make_pebs_state();
        p.mode = PreciseExitMode::Pmi;
        let delta: u64 = 1_000_000;
        let r = arm_precise_exit(&mut p, 0, delta, PebsAction::InjectInterrupt(0x20), 0, 0);
        assert_eq!(r, ArmResult::Armed);
        assert_eq!(p.armed_target_tsc, delta - PMI_MARGIN);
        let below = PMI_MARGIN;
        let r = arm_precise_exit(&mut p, 0, below, PebsAction::InjectInterrupt(0x20), 0, 0);
        assert_eq!(r, ArmResult::BelowMinDelta);
    }

    #[test]
    fn arm_overwrites_previous_action() {
        let mut p = make_pebs_state();
        let _ = arm_precise_exit(&mut p, 0, 1000, PebsAction::InjectInterrupt(0x20), 0, 0);
        let _ = arm_precise_exit(&mut p, 0, 1000, PebsAction::InjectInterrupt(0x30), 0, 0);
        assert!(matches!(
            p.armed_action,
            Some(PebsAction::InjectInterrupt(0x30))
        ));
    }

    #[test]
    fn disarm_clears_armed_action_idempotent() {
        let mut p = make_pebs_state();
        let _ = arm_precise_exit(&mut p, 0, 1000, PebsAction::InjectInterrupt(0x20), 0, 0);
        disarm_precise_exit(&mut p);
        assert!(p.armed_action.is_none());
        disarm_precise_exit(&mut p);
        assert!(p.armed_action.is_none());
    }

    fn ept_qual(write: bool, asynchronous: bool) -> EptViolationQualification {
        EptViolationQualification {
            read: false,
            write,
            execute: false,
            readable: true,
            writable: false,
            executable: true,
            guest_linear_valid: false,
            asynchronous,
        }
    }

    #[test]
    fn is_pebs_induced_requires_async_and_write() {
        assert!(is_pebs_induced(&ept_qual(true, true)));
        // Async read: Intel PT, user-interrupt delivery.
        assert!(!is_pebs_induced(&ept_qual(false, true)));
        // Sync write: CoW / MMIO.
        assert!(!is_pebs_induced(&ept_qual(true, false)));
        assert!(!is_pebs_induced(&ept_qual(false, false)));
    }
}

/// Handle a PEBS-induced EPT violation: record skid diagnostics and consume
/// (disarm) the one-shot armed action.
pub fn handle_pebs_precise_exit<C: VmContext>(ctx: &mut C) -> ExitHandlerResult {
    // One-shot log; distinguishes hangs before vs after the first PEBS exit.
    #[cfg(not(feature = "cargo"))]
    {
        use core::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            log_info!(
                "PEBS first-exit: tsc={} action={:?}\n",
                ctx.state().emulated_tsc,
                ctx.state()
                    .pebs_state
                    .as_deref()
                    .and_then(|p| p.armed_action),
            );
        }
    }
    // Skid diagnostics for the (non-deterministic) EPT_VIOLATION_PEBS record,
    // separating hardware imprecision from hypervisor bugs: instruction
    // delta, tsc_offset delta (HLT/MWAIT clamps; expected 0), and iterations
    // the arming survived. Uses `inst_count + tsc_offset` because
    // `emulated_tsc` is not refreshed on non-deterministic exits.
    let inst_count_now = ctx.state().last_instruction_count;
    let tsc_offset_now = ctx.state().tsc_offset;
    let current_tsc = inst_count_now.saturating_add(tsc_offset_now);
    let (armed_target, armed_inst, armed_offset, iters) = ctx
        .state()
        .pebs_state
        .as_deref()
        .map(|p| {
            (
                p.armed_target_tsc,
                p.armed_inst_count,
                p.armed_tsc_offset,
                p.iters_since_arm,
            )
        })
        .unwrap_or((0, 0, 0, 0));
    if armed_target != 0 {
        // Arm-time distance, to correlate skid outliers with arming regime.
        let arm_current = armed_inst.saturating_add(armed_offset);
        let arm_delta = armed_target.saturating_sub(arm_current);
        let state = ctx.state_mut();
        let skid = (current_tsc as i64) - (armed_target as i64);
        state.last_pebs_skid = skid;
        state.exit_stats.max_pebs_skid = state.exit_stats.max_pebs_skid.max(skid);
        state.last_pebs_inst_delta = (inst_count_now as i64) - (armed_inst as i64);
        state.last_pebs_tsc_offset_delta = (tsc_offset_now as i64) - (armed_offset as i64);
        state.last_pebs_iters_since_arm = iters;
        state.last_pebs_arm_delta = arm_delta;
    }

    // `take()` also disarms.
    let action = ctx
        .state_mut()
        .pebs_state
        .as_deref_mut()
        .and_then(|p| p.armed_action.take());
    match action {
        Some(PebsAction::InjectInterrupt(_vector)) => {
            // The normal pre-entry path MTF-steps the margin, injects at the
            // deadline, and re-arms.
            ExitHandlerResult::Continue
        }
        None => {
            // Possible if a record was pending across a disarm.
            log_warn!("unexpected PEBS-induced EPT violation with no armed action\n");
            ExitHandlerResult::Continue
        }
    }
}
