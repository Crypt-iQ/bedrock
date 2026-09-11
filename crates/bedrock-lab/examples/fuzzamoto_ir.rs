// SPDX-License-Identifier: GPL-2.0

//! Drive fuzzamoto's IR scenario against a real bitcoind, one fork per testcase.
//!
//! This is the fuzzamoto bedrock backend end to end: boot a guest containing
//! `scenario-ir` and a coverage-instrumented `bitcoind`, let the scenario do its
//! expensive setup (spawn the node, mine a chain, open P2P connections) exactly
//! once, checkpoint the VM at the moment it asks for its first testcase, then
//! fork that checkpoint once per IR program.
//!
//! Coverage comes back through the same mechanism as the input: `bitcoind` is
//! built with `-fsanitize-coverage=trace-pc-guard` and linked against bedrock's
//! `libpcguard.c`/`libfeedback.c`, which register an edge-hitcount buffer under
//! `cov-<build-id>`. Each fork has its own copy-on-write view of that buffer, so
//! what we read back after a run is the coverage of that testcase alone.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p bedrock-lab --example fuzzamoto_ir -- \
//!     <vmlinux> <initramfs> --corpus <dir>
//! ```

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use bedrock_lab::{
    Checkpoint, Event, EventSink, FuzzOutcome, LabOpts, RngMode, RunOutcome, VirtDuration,
};
use bedrock_vm::{load_kernel, ExitKind, LinuxBootConfig, VmBuilder};
use clap::Parser;

bedrock_lab::define_virt_time_macros!($, bedrock_vm::DEFAULT_TSC_FREQUENCY);

const BOOT_RNG_SEED: u64 = 0xbed0_f022;

/// The guest needs room for the initramfs, bitcoind, and a tmpfs datadir.
const DEFAULT_MEMORY_MB: usize = 8192;

/// `console=ttyS0` because this image carries no `bedrock-console.ko`; the
/// emulated UART is slower but needs nothing in the guest.
const CMDLINE: &str = "console=ttyS0 nopti nokaslr mitigations=off audit=0";

#[derive(Parser, Debug)]
struct Args {
    /// Path to the guest vmlinux ELF.
    vmlinux: String,
    /// Path to the fuzzamoto guest initramfs.
    initramfs: String,
    /// Directory of `.ir` programs to execute.
    #[arg(long)]
    corpus: PathBuf,
    /// Guest memory size in MB.
    #[arg(long, default_value_t = DEFAULT_MEMORY_MB)]
    memory_mb: usize,
    /// Virtual seconds allowed for the scenario's one-time setup.
    #[arg(long, default_value_t = 120)]
    setup_secs: u64,
    /// Virtual seconds allowed per testcase.
    #[arg(long, default_value_t = 30)]
    timeout_secs: u64,
    /// Print guest console output.
    #[arg(long)]
    verbose: bool,
}

/// Forwards guest console lines when asked; the scenario's own logging is the
/// only window into what the harness is doing.
struct Console {
    verbose: bool,
}

impl EventSink for Console {
    fn on_event(&self, event: Event<'_>) {
        if let Event::SerialLine { line, .. } = event {
            if self.verbose {
                println!("  guest| {}", String::from_utf8_lossy(line));
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // ---- load the corpus -------------------------------------------------
    let mut programs: Vec<(String, Vec<u8>)> = fs::read_dir(&args.corpus)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "ir"))
        .map(|p| {
            let name = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            fs::read(&p).map(|b| (name, b))
        })
        .collect::<Result<_, _>>()?;
    programs.sort_by(|a, b| a.0.cmp(&b.0));
    if programs.is_empty() {
        return Err(format!("no .ir programs in {}", args.corpus.display()).into());
    }
    println!("corpus: {} programs from {}", programs.len(), args.corpus.display());

    // ---- boot ------------------------------------------------------------
    let mut vm = VmBuilder::new().memory_mb(args.memory_mb).build()?;
    let kernel = fs::read(&args.vmlinux)?;
    let initramfs = fs::read(&args.initramfs)?;
    let (kernel_entry, kernel_end) = {
        let memory = vm.memory_mut()?;
        load_kernel(memory, &kernel)?
    };
    let boot = LinuxBootConfig::new(kernel_entry, kernel_end)
        .cmdline(CMDLINE)
        .initramfs(&initramfs);
    vm.setup_linux_boot(&boot)?;

    let t0 = Instant::now();
    let ready_cp = Checkpoint::initial_when_ready_with(
        vm,
        vt!(120 s),
        LabOpts {
            sink: std::sync::Arc::new(Console { verbose: args.verbose }),
            rng: RngMode::Seeded(BOOT_RNG_SEED),
            ..Default::default()
        },
    )?;
    println!(
        "agent ready at {:.3}s virtual ({:.1}s wall) — buffer registered, node not yet spawned",
        ready_cp.time().as_secs_f64(),
        t0.elapsed().as_secs_f64()
    );

    // ---- run the scenario's setup once, and checkpoint after it ----------
    //
    // The agent signals ready before the scenario spawns bitcoind, so the ready
    // checkpoint is too early to fork from. Running on to the first input
    // request puts the guest past node startup, chain mining and connection
    // setup — the work worth paying for only once.
    let freq = ready_cp.tsc_frequency();
    let t1 = Instant::now();
    let mut setup = ready_cp.branch()?;
    let (at, outcome) =
        setup.run_until(ready_cp.time() + VirtDuration::from_secs(args.setup_secs, freq))?;
    match outcome {
        RunOutcome::Yielded { kind: ExitKind::FuzzNextInput } => {}
        other => {
            return Err(format!(
                "scenario never requested an input (got {other:?}); run with --verbose"
            )
            .into())
        }
    }
    let capacity = setup.fuzz_input_capacity()?;
    let cov_ids: Vec<String> = setup
        .feedback_buffer_ids()?
        .iter()
        .map(|id| String::from_utf8_lossy(id).into_owned())
        .collect();
    let input_cp = setup.checkpoint()?;
    println!(
        "scenario set up at {:.3}s virtual ({:.1}s wall); input buffer {capacity} bytes",
        at.as_secs_f64(),
        t1.elapsed().as_secs_f64()
    );
    println!("registered feedback buffers: {}", cov_ids.join(", "));

    // The coverage buffer is whichever registration isn't our input channel.
    let cov_id = cov_ids
        .iter()
        .find(|id| id.starts_with("cov-"))
        .cloned();
    match &cov_id {
        Some(id) => println!("coverage buffer: {id}"),
        None => println!("WARNING: no cov-* buffer — bitcoind coverage is not being collected"),
    }

    // ---- one fork per program --------------------------------------------
    let timeout = VirtDuration::from_secs(args.timeout_secs, freq);
    let mut ok = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    let mut timed_out = 0usize;
    let mut union: Vec<u8> = Vec::new();
    let t2 = Instant::now();

    for (name, program) in &programs {
        if program.len() > capacity {
            println!("  {name}: SKIP (larger than the guest's input buffer)");
            skipped += 1;
            continue;
        }
        let mut branch = input_cp.branch()?;
        branch.serve_fuzz_input(program)?;

        let (_, outcome) = branch.run_until(input_cp.time() + timeout)?;
        match outcome {
            RunOutcome::Yielded { kind: ExitKind::FuzzNextInput } => {
                match branch.fuzz_outcome()? {
                    FuzzOutcome::Ok => ok += 1,
                    FuzzOutcome::Skip => skipped += 1,
                    FuzzOutcome::Fail(msg) => {
                        failed += 1;
                        println!("  {name}: BUG REPORTED: {msg}");
                    }
                    FuzzOutcome::Unknown(s) => {
                        failed += 1;
                        println!("  {name}: unknown status word {s}");
                    }
                }
            }
            RunOutcome::ReachedTime => {
                timed_out += 1;
                println!("  {name}: TIMEOUT after {}s virtual", args.timeout_secs);
            }
            other => {
                failed += 1;
                println!("  {name}: unexpected outcome {other:?}");
            }
        }

        // Fold this testcase's edge hitcounts into a running union, so the
        // total says how much of bitcoind the corpus reached.
        if let Some(id) = &cov_id {
            for buf in branch.feedback_buffers_to_vec(id.as_bytes())? {
                if union.len() < buf.len() {
                    union.resize(buf.len(), 0);
                }
                for (u, b) in union.iter_mut().zip(&buf) {
                    *u |= *b;
                }
            }
        }
    }

    let elapsed = t2.elapsed();
    let executed = ok + skipped + failed + timed_out;
    println!();
    println!(
        "{executed} testcases in {:.1}s = {:.1} execs/s",
        elapsed.as_secs_f64(),
        executed as f64 / elapsed.as_secs_f64()
    );
    println!("  ok {ok}, skipped {skipped}, bugs {failed}, timeouts {timed_out}");
    if !union.is_empty() {
        let hit = union.iter().filter(|&&b| b != 0).count();
        println!(
            "  coverage: {hit} of {} edge counters hit across the corpus ({:.2}%)",
            union.len(),
            100.0 * hit as f64 / union.len() as f64
        );
    }
    Ok(())
}
