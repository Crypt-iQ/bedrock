// SPDX-License-Identifier: GPL-2.0

//! MSR read/write exit handlers.

use super::helpers::{advance_rip, ExitHandlerResult};
use super::reasons::ExitReason;

#[cfg(not(feature = "cargo"))]
use super::super::prelude::*;
#[cfg(feature = "cargo")]
use crate::prelude::*;

/// Fake microcode version to report to guest (matches KVM's default).
const FAKE_MICROCODE_VERSION: u64 = 0x1_0000_0000;

/// Handle MSR read (RDMSR) exit.
pub fn handle_msr_read<C: VmContext>(ctx: &mut C) -> ExitHandlerResult {
    let msr_num = ctx.state().gprs.rcx as u32;

    let value: u64 = match msr_num {
        msr::IA32_MISC_ENABLE => {
            // Host value with BTS/PEBS unavailable and MWAIT cleared.
            let host_val = ctx.state().host_state.misc_enable;
            MiscEnable::for_guest(host_val).bits()
        }
        msr::IA32_BIOS_SIGN_ID => FAKE_MICROCODE_VERSION,
        msr::IA32_MCG_CAP | msr::IA32_MCG_STATUS => {
            // No machine check support (like bhyve).
            0
        }
        msr::IA32_SPEC_CTRL => 0,
        msr::IA32_PRED_CMD => {
            // Write-only.
            0
        }
        msr::IA32_TSC_ADJUST => {
            // TSC adjustment unsupported.
            0
        }
        msr::IA32_TSC_DEADLINE => {
            // TSC-deadline mode is hidden in CPUID.
            0
        }
        msr::IA32_MTRRCAP => {
            // Like bhyve: WC (bit 10), FIX (bit 8), VCNT = MTRR_VAR_MAX.
            (1 << 10) | (1 << 8) | (MTRR_VAR_MAX as u64)
        }
        msr::IA32_MTRR_DEF_TYPE => ctx.state().devices.mtrr.def_type,
        msr::IA32_MTRR_PHYSBASE0..=msr::IA32_MTRR_PHYSMASK9 => {
            let offset = (msr_num - msr::IA32_MTRR_PHYSBASE0) as usize;
            let index = offset / 2;
            if index < MTRR_VAR_MAX {
                let (base, mask) = ctx.state().devices.mtrr.var[index];
                if offset.is_multiple_of(2) {
                    base
                } else {
                    mask
                }
            } else {
                0
            }
        }
        msr::IA32_MTRR_FIX64K_00000 => ctx.state().devices.mtrr.fixed_64k,
        msr::IA32_MTRR_FIX16K_80000 | msr::IA32_MTRR_FIX16K_A0000 => {
            let index = (msr_num - msr::IA32_MTRR_FIX16K_80000) as usize;
            ctx.state().devices.mtrr.fixed_16k[index]
        }
        msr::IA32_MTRR_FIX4K_C0000..=msr::IA32_MTRR_FIX4K_F8000 => {
            let index = (msr_num - msr::IA32_MTRR_FIX4K_C0000) as usize;
            if index < 8 {
                ctx.state().devices.mtrr.fixed_4k[index]
            } else {
                0
            }
        }
        msr::IA32_PAT => ctx.state().msr_state.pat,
        msr::IA32_APIC_BASE => ctx.state().devices.apic.base,
        msr::IA32_TSC_AUX => ctx.state().msr_state.tsc_aux,
        msr::IA32_FEATURE_CONTROL => {
            // Locked (bit 0), VMX disabled: no nested virt.
            0x1
        }
        msr::IA32_PLATFORM_INFO => {
            // Host value, like bhyve (max non-turbo ratio in bits 15:8).
            ctx.state().host_state.platform_info
        }
        msr::IA32_THERM_INTERRUPT
        | msr::IA32_THERM_STATUS
        | msr::IA32_PACKAGE_THERM_STATUS
        | msr::IA32_PACKAGE_THERM_INTERRUPT => 0,
        msr::IA32_PPIN_CTL => 0,
        msr::IA32_PERF_CAPABILITIES => 0,
        msr::IA32_LBR_TOS => 0,
        msr::IA32_OFFCORE_RSP_0 | msr::IA32_OFFCORE_RSP_1 => 0,
        msr::IA32_PEBS_LD_LAT_THRESHOLD | msr::IA32_PEBS_FRONTEND => 0,
        msr::IA32_PERFEVTSEL0
        | msr::IA32_PERFEVTSEL1
        | msr::IA32_PERFEVTSEL2
        | msr::IA32_PERFEVTSEL3
        | msr::IA32_PERFEVTSEL4
        | msr::IA32_PERFEVTSEL5
        | msr::IA32_PERFEVTSEL6
        | msr::IA32_PERFEVTSEL7
        | msr::IA32_FIXED_CTR_CTRL
        | msr::IA32_PMC0
        | msr::IA32_PMC1
        | msr::IA32_PMC2
        | msr::IA32_PMC3
        | msr::IA32_PMC4
        | msr::IA32_PMC5
        | msr::IA32_PMC6
        | msr::IA32_PMC7
        | msr::IA32_MPERF
        | msr::IA32_APERF => 0,
        msr::MSR_ATOM_CORE_RATIOS | msr::MSR_ATOM_CORE_VIDS | msr::MSR_ATOM_CORE_TURBO_RATIOS => 0,
        msr::IA32_MISC_FEATURES_ENABLES => {
            // CPUID faulting not enabled (like bhyve).
            0
        }
        msr::IA32_RTIT_CTL => 0,
        msr::MSR_RAPL_POWER_UNIT => {
            // Fixed RAPL units for determinism:
            //   Power units  (bits 3:0)  = 0x3 → 1/8 Watts
            //   Energy units (bits 12:8) = 0x10 → 1/65536 Joules
            //   Time units   (bits 19:16) = 0xA → 1/1024 seconds
            0x000A_1003
        }
        msr::MSR_PKG_ENERGY_STATUS
        | msr::MSR_DRAM_ENERGY_STATUS
        | msr::MSR_PP0_ENERGY_STATUS
        | msr::MSR_PP1_ENERGY_STATUS => 0,
        msr::MSR_PKG_POWER_LIMIT
        | msr::MSR_PKG_POWER_INFO
        | msr::MSR_DRAM_POWER_LIMIT
        | msr::MSR_DRAM_POWER_INFO
        | msr::MSR_PP0_POWER_LIMIT
        | msr::MSR_PP1_POWER_LIMIT => 0,
        msr::MSR_PKG_CST_CONFIG_CONTROL
        | msr::MSR_POWER_CTL
        | msr::MSR_PPERF
        | msr::MSR_MISC_PWR_MGMT
        | msr::IA32_PM_ENABLE
        | msr::IA32_HWP_CAPABILITIES
        | msr::IA32_HWP_INTERRUPT
        | msr::IA32_HWP_REQUEST
        | msr::IA32_HWP_STATUS
        | msr::IA32_PERF_STATUS
        | msr::IA32_PERF_CTL
        | msr::IA32_ENERGY_PERF_BIAS
        | msr::MSR_OC_MAILBOX => 0,
        msr::MSR_SMI_COUNT => 0,
        msr::IA32_DEBUGCTL => {
            // No branch tracing or block stepping.
            0
        }
        msr::MSR_AMD64_DE_CFG => {
            // AMD MSR probed by Linux on Intel.
            0
        }
        msr::IA32_MKTME_KEYID_PARTITIONING => 0,
        // SYSCALL MSRs (STAR, LSTAR, CSTAR, FMASK) are passthrough.
        _ => {
            if let Err(e) = advance_rip(ctx) {
                return ExitHandlerResult::Error(e);
            }
            return ExitHandlerResult::ExitToUserspace(ExitReason::MsrRead);
        }
    };

    let gprs = &mut ctx.state_mut().gprs;
    gprs.rax = value & 0xFFFF_FFFF;
    gprs.rdx = value >> 32;

    if let Err(e) = advance_rip(ctx) {
        return ExitHandlerResult::Error(e);
    }
    ExitHandlerResult::Continue
}

/// Handle MSR write (WRMSR) exit.
pub fn handle_msr_write<C: VmContext>(ctx: &mut C) -> ExitHandlerResult {
    let msr_num = ctx.state().gprs.rcx as u32;
    let value = ((ctx.state().gprs.rdx & 0xFFFF_FFFF) << 32) | (ctx.state().gprs.rax & 0xFFFF_FFFF);

    match msr_num {
        msr::IA32_MISC_ENABLE => {
            // Ignored.
        }
        msr::IA32_BIOS_SIGN_ID => {
            // Ignored; reads return a fixed fake version.
        }
        msr::IA32_MCG_CAP | msr::IA32_MCG_STATUS => {
            // Ignored.
        }
        msr::IA32_SPEC_CTRL => {
            // Ignored.
        }
        msr::IA32_PRED_CMD => {
            // IBPB: no-op.
        }
        msr::IA32_TSC_ADJUST => {
            // Ignored.
        }
        msr::IA32_TSC_DEADLINE => {
            // Ignored; TSC-deadline mode is hidden in CPUID.
        }
        msr::IA32_MTRRCAP => {
            // Read-only; ignored (bhyve returns an error).
        }
        msr::IA32_MTRR_DEF_TYPE => {
            // Reserved bits are not validated (bhyve does).
            ctx.state_mut().devices.mtrr.def_type = value;
        }
        msr::IA32_MTRR_PHYSBASE0..=msr::IA32_MTRR_PHYSMASK9 => {
            let offset = (msr_num - msr::IA32_MTRR_PHYSBASE0) as usize;
            let index = offset / 2;
            if index < MTRR_VAR_MAX {
                if offset.is_multiple_of(2) {
                    ctx.state_mut().devices.mtrr.var[index].0 = value; // base
                } else {
                    ctx.state_mut().devices.mtrr.var[index].1 = value; // mask
                }
            }
        }
        msr::IA32_MTRR_FIX64K_00000 => {
            ctx.state_mut().devices.mtrr.fixed_64k = value;
        }
        msr::IA32_MTRR_FIX16K_80000 | msr::IA32_MTRR_FIX16K_A0000 => {
            let index = (msr_num - msr::IA32_MTRR_FIX16K_80000) as usize;
            ctx.state_mut().devices.mtrr.fixed_16k[index] = value;
        }
        msr::IA32_MTRR_FIX4K_C0000..=msr::IA32_MTRR_FIX4K_F8000 => {
            let index = (msr_num - msr::IA32_MTRR_FIX4K_C0000) as usize;
            if index < 8 {
                ctx.state_mut().devices.mtrr.fixed_4k[index] = value;
            }
        }
        msr::IA32_PAT => {
            ctx.state_mut().msr_state.pat = value;
        }
        msr::IA32_APIC_BASE => {
            // Like bhyve, reject changes: APIC MMIO is hardcoded to 0xFEE00000.
            if ctx.state().devices.apic.base != value {
                if let Err(e) = advance_rip(ctx) {
                    return ExitHandlerResult::Error(e);
                }
                return ExitHandlerResult::ExitToUserspace(ExitReason::MsrWrite);
            }
        }
        msr::IA32_TSC_AUX => {
            ctx.state_mut().msr_state.tsc_aux = value;
        }
        msr::IA32_FEATURE_CONTROL => {
            // Ignored; the MSR is locked.
        }
        msr::IA32_PLATFORM_INFO => {
            // Read-only.
        }
        msr::IA32_THERM_INTERRUPT
        | msr::IA32_THERM_STATUS
        | msr::IA32_PACKAGE_THERM_STATUS
        | msr::IA32_PACKAGE_THERM_INTERRUPT => {
            // Ignored.
        }
        msr::IA32_PPIN_CTL
        | msr::IA32_PERF_CAPABILITIES
        | msr::IA32_LBR_TOS
        | msr::IA32_OFFCORE_RSP_0
        | msr::IA32_OFFCORE_RSP_1
        | msr::IA32_PEBS_LD_LAT_THRESHOLD
        | msr::IA32_PEBS_FRONTEND
        | msr::IA32_PERFEVTSEL0
        | msr::IA32_PERFEVTSEL1
        | msr::IA32_PERFEVTSEL2
        | msr::IA32_PERFEVTSEL3
        | msr::IA32_PERFEVTSEL4
        | msr::IA32_PERFEVTSEL5
        | msr::IA32_PERFEVTSEL6
        | msr::IA32_PERFEVTSEL7
        | msr::IA32_FIXED_CTR_CTRL
        | msr::IA32_PMC0
        | msr::IA32_PMC1
        | msr::IA32_PMC2
        | msr::IA32_PMC3
        | msr::IA32_PMC4
        | msr::IA32_PMC5
        | msr::IA32_PMC6
        | msr::IA32_PMC7
        | msr::IA32_MPERF
        | msr::IA32_APERF
        | msr::MSR_ATOM_CORE_RATIOS
        | msr::MSR_ATOM_CORE_VIDS
        | msr::MSR_ATOM_CORE_TURBO_RATIOS => {
            // Ignored.
        }
        msr::IA32_MISC_FEATURES_ENABLES => {
            // Ignored; no CPUID faulting (like bhyve).
        }
        msr::IA32_RTIT_CTL => {
            // Ignored.
        }
        msr::MSR_RAPL_POWER_UNIT
        | msr::MSR_PKG_POWER_LIMIT
        | msr::MSR_PKG_ENERGY_STATUS
        | msr::MSR_PKG_POWER_INFO
        | msr::MSR_DRAM_POWER_LIMIT
        | msr::MSR_DRAM_ENERGY_STATUS
        | msr::MSR_DRAM_POWER_INFO
        | msr::MSR_PP0_POWER_LIMIT
        | msr::MSR_PP0_ENERGY_STATUS
        | msr::MSR_PP1_POWER_LIMIT
        | msr::MSR_PP1_ENERGY_STATUS => {
            // Ignored.
        }
        msr::MSR_PKG_CST_CONFIG_CONTROL
        | msr::MSR_POWER_CTL
        | msr::MSR_PPERF
        | msr::MSR_MISC_PWR_MGMT
        | msr::IA32_PM_ENABLE
        | msr::IA32_HWP_CAPABILITIES
        | msr::IA32_HWP_INTERRUPT
        | msr::IA32_HWP_REQUEST
        | msr::IA32_HWP_STATUS
        | msr::IA32_PERF_STATUS
        | msr::IA32_PERF_CTL
        | msr::IA32_ENERGY_PERF_BIAS
        | msr::MSR_OC_MAILBOX => {
            // Ignored.
        }
        msr::IA32_DEBUGCTL => {
            // Ignored; BTF unsupported.
        }
        // SYSCALL MSRs (STAR, LSTAR, CSTAR, FMASK) are passthrough.
        _ => {
            if let Err(e) = advance_rip(ctx) {
                return ExitHandlerResult::Error(e);
            }
            return ExitHandlerResult::ExitToUserspace(ExitReason::MsrWrite);
        }
    }

    if let Err(e) = advance_rip(ctx) {
        return ExitHandlerResult::Error(e);
    }
    ExitHandlerResult::Continue
}
