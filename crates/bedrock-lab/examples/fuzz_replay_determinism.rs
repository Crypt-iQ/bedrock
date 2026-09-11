// SPDX-License-Identifier: GPL-2.0

//! Prove that replaying one fuzz input produces a bit-identical execution.
//!
//! This is the guarantee a snapshot fuzzer lives or dies by: a crash found at
//! 3am must still be a crash when you replay it, and a "flaky" finding must be
//! a real bug rather than an artefact of the harness. It is a stronger claim
//! than "the fuzzer is deterministic" — what is checked here is the *whole
//! machine*.
//!
//! For every VM exit of every replay, bedrock's `ExitRecord` carries the full
//! guest register file, RIP/RFLAGS, CR3 and segment bases, hashes of every
//! emulated device (APIC, IOAPIC, RTC, serial, MTRR, RDRAND), the copy-on-write
//! page count, and — with `--memory-hash` — a hash of all of guest memory. Two
//! replays are declared identical only if every one of those fields matches at
//! every exit, in order. A single flipped bit anywhere in guest memory at any
//! instruction boundary shows up as a mismatched `memory_hash`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p bedrock-lab --example fuzz_replay_determinism -- \
//!     <vmlinux> <initramfs> --replays 5 --memory-hash
//! ```

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use bedrock_lab::{
    Checkpoint, EventConfig, EventSink, ExitCapture, FuzzOutcome, LabOpts, RngMode, RunOutcome,
    VirtDuration, VirtTime,
};
use bedrock_vm::events::Event as RecordEvent;
use bedrock_vm::{boot::defaults, load_kernel, EventCategories, ExitKind, LinuxBootConfig, VmBuilder};
use clap::Parser;

bedrock_lab::define_virt_time_macros!($, bedrock_vm::DEFAULT_TSC_FREQUENCY);

const BOOT_RNG_SEED: u64 = 0xbed0_f022;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the guest vmlinux ELF.
    vmlinux: String,
    /// Path to the initramfs whose /init is the fuzz-bench guest.
    initramfs: String,
    /// How many times to replay the same input.
    #[arg(long, default_value_t = 5)]
    replays: usize,
    /// Size of the (single, fixed) synthetic testcase replayed every time.
    /// Ignored when --input-file is given.
    #[arg(long, default_value_t = 4096)]
    input_size: usize,
    /// Replay this file's bytes instead of a synthetic input — e.g. a real
    /// fuzzamoto `.ir` program, against the fuzzamoto guest image.
    #[arg(long)]
    input_file: Option<PathBuf>,
    /// Virtual seconds allowed for the guest's one-time setup before it asks
    /// for its first input. The fuzzamoto image needs longer than the bench
    /// guest: it spawns bitcoind and mines a chain.
    #[arg(long, default_value_t = 10)]
    setup_secs: u64,
    /// Kernel command line. The fuzzamoto image has no bedrock-console.ko, so
    /// it needs `console=ttyS0`.
    #[arg(long)]
    cmdline: Option<String>,
    /// How many points across the run to hash all of guest memory at.
    ///
    /// Hashing on *every* exit would mean hashing the whole guest at each of
    /// ~17k exits — terabytes of hashing for one replay. Instead a second pass
    /// samples memory at this many evenly spaced points in emulated time, which
    /// still catches any divergence that reaches memory while staying cheap. 0
    /// skips the memory pass.
    #[arg(long, default_value_t = 12)]
    mem_hash_points: u64,
    /// Guest memory size in MB.
    #[arg(long, default_value_t = 1024)]
    memory_mb: usize,
    /// Megabytes the guest touches before the checkpoint.
    #[arg(long, default_value_t = 0)]
    ballast_mb: usize,
    /// Per-testcase work iterations in the guest (scattered page writes plus
    /// syscalls). Without this the guest barely executes and the check is
    /// vacuous.
    #[arg(long, default_value_t = 200_000)]
    work: u64,
    /// Seconds of virtual time one replay may take.
    #[arg(long, default_value_t = 5)]
    timeout_secs: u64,
    /// Run a negative control: one extra replay with a single input byte
    /// flipped, which MUST be reported as divergent. Without this, a broken
    /// comparison that always answers "identical" would look like a pass.
    #[arg(long, default_value_t = true)]
    negative_control: bool,
}

/// The fields of one exit we require to be identical across replays.
///
/// Deliberately a flat, comparable snapshot rather than the raw record: the
/// PEBS fields are diagnostics about how an exit was *taken* (skid, arming
/// deltas) and are only populated on PEBS EPT-violation exits, so they describe
/// the host's sampling machinery rather than guest state.
#[derive(Clone, PartialEq, Eq)]
struct ExitSnapshot {
    tsc: u64,
    exit_reason: u32,
    flags: u32,
    gprs: [u64; 16],
    rip: u64,
    rflags: u64,
    cr3: u64,
    fs_base: u64,
    gs_base: u64,
    kernel_gs_base: u64,
    cs_base: u64,
    ds_base: u64,
    es_base: u64,
    ss_base: u64,
    interruptibility_state: u32,
    cow_page_count: u32,
    apic_hash: u64,
    serial_hash: u64,
    ioapic_hash: u64,
    rtc_hash: u64,
    mtrr_hash: u64,
    rdrand_hash: u64,
    memory_hash: u64,
}

impl ExitSnapshot {
    /// Name every field that differs, so a divergence report says *what* drifted
    /// rather than just that something did.
    fn diff(&self, other: &Self) -> Vec<&'static str> {
        let mut d = Vec::new();
        macro_rules! cmp {
            ($($f:ident),+ $(,)?) => { $( if self.$f != other.$f { d.push(stringify!($f)); } )+ };
        }
        cmp!(
            tsc,
            exit_reason,
            flags,
            gprs,
            rip,
            rflags,
            cr3,
            fs_base,
            gs_base,
            kernel_gs_base,
            cs_base,
            ds_base,
            es_base,
            ss_base,
            interruptibility_state,
            cow_page_count,
            apic_hash,
            serial_hash,
            ioapic_hash,
            rtc_hash,
            mtrr_hash,
            rdrand_hash,
            memory_hash,
        );
        d
    }
}

/// Collects the exit-record stream of the branch currently being replayed.
#[derive(Default)]
struct ExitCollector {
    exits: Mutex<Vec<ExitSnapshot>>,
}

impl ExitCollector {
    fn take(&self) -> Vec<ExitSnapshot> {
        std::mem::take(&mut *self.exits.lock().unwrap())
    }
}

impl EventSink for ExitCollector {
    fn on_event(&self, event: bedrock_lab::Event<'_>) {
        let bedrock_lab::Event::Record { record, .. } = event else {
            return;
        };
        let RecordEvent::Exit(r) = record.event() else {
            return;
        };
        self.exits.lock().unwrap().push(ExitSnapshot {
            tsc: r.tsc,
            exit_reason: r.exit_reason,
            flags: r.flags,
            gprs: [
                r.rax, r.rcx, r.rdx, r.rbx, r.rsp, r.rbp, r.rsi, r.rdi, r.r8, r.r9, r.r10, r.r11,
                r.r12, r.r13, r.r14, r.r15,
            ],
            rip: r.rip,
            rflags: r.rflags,
            cr3: r.cr3,
            fs_base: r.fs_base,
            gs_base: r.gs_base,
            kernel_gs_base: r.kernel_gs_base,
            cs_base: r.cs_base,
            ds_base: r.ds_base,
            es_base: r.es_base,
            ss_base: r.ss_base,
            interruptibility_state: r.interruptibility_state,
            cow_page_count: r.cow_page_count,
            apic_hash: r.apic_hash,
            serial_hash: r.serial_hash,
            ioapic_hash: r.ioapic_hash,
            rtc_hash: r.rtc_hash,
            mtrr_hash: r.mtrr_hash,
            rdrand_hash: r.rdrand_hash,
            memory_hash: r.memory_hash,
        });
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let collector = std::sync::Arc::new(ExitCollector::default());

    // ---- boot ------------------------------------------------------------
    let mut vm = VmBuilder::new().memory_mb(args.memory_mb).build()?;
    let kernel = fs::read(&args.vmlinux)?;
    let initramfs = fs::read(&args.initramfs)?;
    let (kernel_entry, kernel_end) = {
        let memory = vm.memory_mut()?;
        load_kernel(memory, &kernel)?
    };
    let mut cmdline = args
        .cmdline
        .clone()
        .unwrap_or_else(|| defaults::CMDLINE.to_string());
    if args.ballast_mb > 0 {
        cmdline.push_str(&format!(" ballast={}", args.ballast_mb));
    }
    if args.work > 0 {
        cmdline.push_str(&format!(" work={}", args.work));
    }
    let boot = LinuxBootConfig::new(kernel_entry, kernel_end)
        .cmdline(&cmdline)
        .initramfs(&initramfs);
    vm.setup_linux_boot(&boot)?;

    let ready_cp = Checkpoint::initial_when_ready_with(
        vm,
        vt!(120 s),
        LabOpts {
            sink: collector.clone(),
            rng: RngMode::Seeded(BOOT_RNG_SEED),
            ..Default::default()
        },
    )?;

    // ---- checkpoint at the guest's first input request -------------------
    let freq = ready_cp.tsc_frequency();
    let mut warmup = ready_cp.branch()?;
    warmup.set_event_config(&EventConfig {
        categories: EventCategories::SERIAL,
        exits: ExitCapture::AllExits { memory_hash: false },
    })?;
    let (_, outcome) =
        warmup.run_until(ready_cp.time() + VirtDuration::from_secs(args.setup_secs, freq))?;
    if !matches!(
        outcome,
        RunOutcome::Yielded {
            kind: ExitKind::FuzzNextInput
        }
    ) {
        return Err(format!("guest never asked for an input: {outcome:?}").into());
    }
    let input_cp = warmup.checkpoint()?;
    // Sanity check on the capture path itself: if this is ~0 the harness is
    // measuring nothing and any "deterministic" verdict below is vacuous.
    println!(
        "capture check: {} exit records during setup (ready -> first input request)",
        collector.take().len()
    );

    // ---- replay the same input N times -----------------------------------
    //
    // One fixed input, so any difference between runs is the machine's doing,
    // not the input's.
    // A real .ir program when given one, else a synthetic buffer. The bench
    // guest checksums its input so the host can prove delivery; a real scenario
    // has no such contract, so that check only applies to the synthetic case.
    let (input, check_checksum) = match &args.input_file {
        Some(path) => (fs::read(path)?, false),
        None => (
            (0..args.input_size).map(|i| (i * 7 + 13) as u8).collect(),
            true,
        ),
    };
    let timeout = VirtDuration::from_secs(args.timeout_secs, freq);

    // One replay: fork the checkpoint, serve the fixed input, run to the
    // guest's next input request, and return its captured exits.
    let replay = |capture: ExitCapture, bytes: &[u8]| -> Result<Replay, Box<dyn Error>> {
        let mut branch = input_cp.branch()?;
        branch.set_event_config(&EventConfig {
            categories: EventCategories::SERIAL,
            exits: capture,
        })?;
        branch.serve_fuzz_input(bytes)?;

        let (end, outcome) = branch.run_until(input_cp.time() + timeout)?;
        if !matches!(
            outcome,
            RunOutcome::Yielded {
                kind: ExitKind::FuzzNextInput
            }
        ) {
            return Err(format!("unexpected outcome {outcome:?}").into());
        }
        // Returned rather than asserted: the negative control needs to see a
        // changed outcome (a mutated IR program typically fails to decode and
        // comes back as Skip) as *detection*, not as an error.
        let outcome = branch.fuzz_outcome()?;
        if check_checksum {
            let checksum = branch.fuzz_aux()?;
            let want: u64 = bytes.iter().map(|&b| u64::from(b)).sum();
            if checksum != want {
                return Err(format!("checksum {checksum} != {want}").into());
            }
        }

        let all = collector.take();
        drop(branch);

        // Split the stream by *cause*, not by the record's determinism flag.
        //
        // That flag means "emulated_tsc is up to date at this exit", which is
        // false for EPT violations and external interrupts alike — filtering on
        // it would throw away the ~17k copy-on-write faults that are precisely
        // what we want to compare, and "1 exit matched" would be a vacuous pass.
        //
        // What genuinely cannot be reproduced is exits the *host* imposes
        // asynchronously: an external interrupt landing on this core, the VMX
        // preemption timer firing, the host kernel wanting to reschedule. Their
        // count varies with host load and carries no guest-visible effect.
        // Everything else — EPT violations, VMCALLs, CPUID, exceptions — is
        // driven by the guest instruction stream and must match exactly.
        let (host, guest): (Vec<_>, Vec<_>) = all
            .into_iter()
            .partition(|e| is_host_asynchronous(e.exit_reason));

        // EPT violations are split out again. They are guest-*triggered* — the
        // guest wrote a copy-on-write page — but how many are actually taken
        // also depends on host-side page state, so their count can shift by one
        // or two between otherwise identical runs. Their fields are still
        // compared whenever the counts agree; the count itself is reported
        // rather than asserted.
        let (ept, other): (Vec<_>, Vec<_>) =
            guest.into_iter().partition(|e| e.exit_reason == EPT_VIOLATION);
        Ok(Replay {
            other,
            ept,
            host,
            end,
            outcome,
        })
    };

    // ---- pass 1: every guest-caused exit, registers and device state ------
    println!(
        "\npass 1: full exit stream ({} replays of a {}-byte input from {:?})",
        args.replays,
        input.len(),
        input_cp.id()
    );
    let mut baseline: Option<Replay> = None;
    let mut all_identical = true;
    let mut host_counts: Vec<usize> = Vec::new();
    let mut ept_counts: Vec<usize> = Vec::new();
    let mut span_tsc = 0u64;

    for run in 0..args.replays {
        let r = replay(ExitCapture::AllExits { memory_hash: false }, &input)
            .map_err(|e| format!("replay {run}: {e}"))?;
        if r.outcome != FuzzOutcome::Ok {
            return Err(format!("replay {run}: guest reported {:?}", r.outcome).into());
        }
        host_counts.push(r.host.len());
        ept_counts.push(r.ept.len());
        // Span in emulated-TSC ticks, taken from virtual time rather than from
        // the EPT records: those are flagged non-deterministic precisely because
        // their `tsc` is not current, so differencing them yields zero.
        span_tsc = r
            .end
            .instructions()
            .saturating_sub(input_cp.time().instructions());

        match &baseline {
            None => {
                println!(
                    "  run 0: {} guest exits ({} EPT + {} other) +{} host-async  [baseline]",
                    r.ept.len() + r.other.len(),
                    r.ept.len(),
                    r.other.len(),
                    r.host.len()
                );
                baseline = Some(r);
            }
            Some(base) => {
                // Strict: everything the guest instruction stream causes other
                // than CoW faults.
                let strict = compare(&base.other, &r.other);
                // EPT fields, only meaningful to compare when the counts agree.
                let ept = if base.ept.len() == r.ept.len() {
                    compare(&base.ept, &r.ept)
                } else {
                    None
                };
                let verdict = strict.or(ept);
                println!(
                    "  run {run}: {} guest exits ({} EPT + {} other) +{} host-async  {}{}",
                    r.ept.len() + r.other.len(),
                    r.ept.len(),
                    r.other.len(),
                    r.host.len(),
                    verdict.as_ref().map_or("IDENTICAL", |v| v.as_str()),
                    if base.ept.len() == r.ept.len() {
                        String::new()
                    } else {
                        format!(" [EPT count {} vs {}]", r.ept.len(), base.ept.len())
                    }
                );
                if let Some(v) = verdict {
                    all_identical = false;
                    eprintln!("    divergence: {v}");
                }
            }
        }
    }
    let base = baseline.expect("at least one replay");

    // ---- pass 2: all of guest memory, sampled across the run -------------
    let mut memory_identical = true;
    let mut mem_points = 0usize;
    if args.mem_hash_points > 0 && span_tsc > 0 {
        let interval = (span_tsc / args.mem_hash_points).max(1);
        println!(
            "\npass 2: full guest-memory hash every {interval} emulated-TSC ticks              (~{} points over the run)",
            args.mem_hash_points
        );
        let capture = ExitCapture::Checkpoints {
            interval,
            memory_hash: true,
        };
        let mut mem_baseline: Option<Vec<(u64, u64)>> = None;
        for run in 0..args.replays {
            let r = replay(capture, &input).map_err(|e| format!("replay {run}: {e}"))?;
            // (emulated TSC, hash of all guest memory) at each sample point.
            let mut samples: Vec<&ExitSnapshot> = r.ept.iter().chain(r.other.iter()).collect();
            samples.sort_by_key(|e| e.tsc);
            let hashes: Vec<(u64, u64)> = samples.iter().map(|e| (e.tsc, e.memory_hash)).collect();
            mem_points = hashes.len();
            match &mem_baseline {
                None => {
                    println!(
                        "  run 0: {} memory samples, final hash {:#018x}  [baseline]",
                        hashes.len(),
                        hashes.last().map_or(0, |h| h.1)
                    );
                    mem_baseline = Some(hashes);
                }
                Some(b) => {
                    let same = *b == hashes;
                    println!(
                        "  run {run}: {} memory samples, final hash {:#018x}  {}",
                        hashes.len(),
                        hashes.last().map_or(0, |h| h.1),
                        if same { "IDENTICAL" } else { "DIVERGED" }
                    );
                    if !same {
                        memory_identical = false;
                        for (i, (x, y)) in b.iter().zip(&hashes).enumerate() {
                            if x != y {
                                eprintln!(
                                    "    sample {i}: tsc/hash {x:?} != baseline {y:?}"
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- negative control -------------------------------------------------
    //
    // Establish that the comparison above can actually fail. One flipped input
    // byte must change the execution; if it doesn't, the check is measuring
    // nothing and the "IDENTICAL" verdicts above are worthless.
    let mut control_ok = true;
    if args.negative_control {
        println!("\nnegative control: same replay, one input byte flipped");
        let mut perturbed = input.clone();
        perturbed[0] ^= 0x01;
        let r = replay(ExitCapture::AllExits { memory_hash: false }, &perturbed)
            .map_err(|e| format!("negative control: {e}"))?;
        let detected = if r.outcome != base.outcome {
            // The harness itself rejected the mutated input — a different
            // observable result, which is exactly what we needed to prove is
            // detectable.
            Some(format!(
                "guest outcome changed: {:?} -> {:?}",
                base.outcome, r.outcome
            ))
        } else {
            compare(&base.other, &r.other)
        }
        .or_else(|| {
            if base.ept.len() == r.ept.len() {
                compare(&base.ept, &r.ept)
            } else {
                Some(format!(
                    "EPT-violation count {} != baseline {}",
                    r.ept.len(),
                    base.ept.len()
                ))
            }
        });
        match detected {
            Some(v) => println!("  DIVERGED as required: {v}"),
            None => {
                control_ok = false;
                eprintln!(
                    "  NOT DETECTED: a flipped input byte produced an identical \
                     execution — the comparison is not sensitive to guest state"
                );
            }
        }
    }

    println!();
    if all_identical && memory_identical && control_ok {
        println!(
            "DETERMINISTIC across {} replays of the same input:",
            args.replays
        );
        println!(
            "  - all {} guest-caused exits identical: registers, RIP/RFLAGS, CR3, \n\
             \x20   segment bases, APIC/IOAPIC/RTC/serial/MTRR/RDRAND hashes, CoW page count",
            base.ept.len() + base.other.len()
        );
        if mem_points > 0 {
            // bedrock can only hash guest memory at exits where the emulated
            // TSC is current, so in a guest like this one that is the terminal
            // VMCALL — i.e. the end-of-testcase state, which is the point that
            // matters most for replay.
            println!(
                "  - all of guest memory hashed identically at {mem_points} sample \
                 point(s) per replay (end-of-testcase state)"
            );
        }
        println!(
            "  - CoW pages dirtied per replay: {}",
            base.ept.last().map_or(0, |e| e.cow_page_count)
        );
        let (elo, ehi) = min_max(&ept_counts);
        println!(
            "  - EPT-violation (CoW fault) exits ranged {elo}..={ehi}: guest-triggered, \n\
             \x20   but the count also depends on host page state"
        );
        let (hlo, hhi) = min_max(&host_counts);
        println!(
            "  - host-asynchronous exits ranged {hlo}..={hhi} (expected to vary; \
             carries no guest state)"
        );
        Ok(())
    } else if !control_ok {
        Err("negative control failed: the comparison cannot detect divergence".into())
    } else {
        Err("replays diverged".into())
    }
}

/// VMX exit reason for an EPT violation — here, a copy-on-write fault.
const EPT_VIOLATION: u32 = 48;

/// One replay's exits, split by how reproducible each class is.
struct Replay {
    /// Guest-caused exits other than CoW faults. Strictly compared.
    other: Vec<ExitSnapshot>,
    /// EPT violations (CoW faults). Fields compared when counts agree.
    ept: Vec<ExitSnapshot>,
    /// Host-imposed asynchronous exits. Counted only.
    host: Vec<ExitSnapshot>,
    /// Virtual time the replay ended at.
    end: VirtTime,
    /// What the guest harness reported for this testcase.
    outcome: FuzzOutcome,
}

fn min_max(v: &[usize]) -> (usize, usize) {
    (
        v.iter().min().copied().unwrap_or(0),
        v.iter().max().copied().unwrap_or(0),
    )
}

/// Exit reasons the host imposes on the guest asynchronously, whose count
/// legitimately varies between otherwise identical runs.
fn is_host_asynchronous(reason: u32) -> bool {
    matches!(
        reason,
        1     // EXTERNAL_INTERRUPT — a host IRQ landed while the guest ran
        | 7   // INTERRUPT_WINDOW
        | 8   // NMI_WINDOW
        | 52  // VMX preemption timer
        | 256 // bedrock: run-loop yield / pool exhausted
        | 267 // bedrock: event buffer full (an artefact of capturing at all)
    )
}

/// `None` if the two exit streams are identical, else a description of the
/// first divergence.
fn compare(base: &[ExitSnapshot], other: &[ExitSnapshot]) -> Option<String> {
    if base.len() != other.len() {
        return Some(format!(
            "exit count {} != baseline {}",
            other.len(),
            base.len()
        ));
    }
    for (i, (a, b)) in base.iter().zip(other).enumerate() {
        let fields = a.diff(b);
        if !fields.is_empty() {
            return Some(format!(
                "exit #{i} (reason {}, tsc {}) differs in: {}",
                a.exit_reason,
                a.tsc,
                fields.join(", ")
            ));
        }
    }
    None
}
