// SPDX-License-Identifier: GPL-2.0

//! Measure fork-per-testcase throughput over the fuzz-input channel.
//!
//! This is the go/no-go benchmark for snapshot fuzzing on bedrock: it answers
//! "how many testcases per second can we push through a forked VM?" with the
//! target application taken out of the picture entirely. The guest
//! (`guest/fuzz-bench.c`) does nothing but sum the bytes of each input, so what
//! is being timed is fork + input delivery + run + teardown — the floor under
//! any real harness.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p bedrock-lab --example fuzz_fork_bench -- \
//!     <vmlinux> <initramfs> --iterations 500
//! ```
//!
//! where `<initramfs>` is a cpio.gz whose `/init` is `guest/fuzz-bench.c` built
//! static:
//!
//! ```text
//! cc -O2 -static -Iguest -o init guest/fuzz-bench.c
//! echo init | cpio -o -H newc | gzip -9 > initramfs.cpio.gz
//! ```

use std::error::Error;
use std::fs;
use std::time::Instant;

use bedrock_lab::{Checkpoint, FuzzOutcome, LabOpts, RngMode, RunOutcome, VirtDuration};
use bedrock_vm::{boot::defaults, load_kernel, ExitKind, LinuxBootConfig, VmBuilder};
use clap::Parser;

bedrock_lab::define_virt_time_macros!($, bedrock_vm::DEFAULT_TSC_FREQUENCY);

const BOOT_RNG_SEED: u64 = 0xbed0_f022;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the guest vmlinux ELF.
    vmlinux: String,
    /// Path to the initramfs whose /init is the fuzz-bench guest.
    initramfs: String,
    /// Number of testcases to execute.
    #[arg(long, default_value_t = 500)]
    iterations: usize,
    /// Size of each synthetic testcase, in bytes.
    #[arg(long, default_value_t = 4096)]
    input_size: usize,
    /// Seconds of virtual time one testcase may take before we call it a hang.
    #[arg(long, default_value_t = 5)]
    timeout_secs: u64,
    /// Guest memory size in MB.
    #[arg(long, default_value_t = 1024)]
    memory_mb: usize,
    /// Megabytes the guest touches before the checkpoint, to stand in for a
    /// real target's resident set. Fork is copy-on-write, so this is what
    /// per-testcase cost actually scales with.
    #[arg(long, default_value_t = 0)]
    ballast_mb: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // ---- boot the root VM -------------------------------------------------
    let mut vm = VmBuilder::new().memory_mb(args.memory_mb).build()?;
    let kernel = fs::read(&args.vmlinux)?;
    let initramfs = fs::read(&args.initramfs)?;
    let (kernel_entry, kernel_end) = {
        let memory = vm.memory_mut()?;
        load_kernel(memory, &kernel)?
    };
    // The guest reads `ballast=` off its command line.
    let cmdline = if args.ballast_mb > 0 {
        format!("{} ballast={}", defaults::CMDLINE, args.ballast_mb)
    } else {
        defaults::CMDLINE.to_string()
    };
    let boot = LinuxBootConfig::new(kernel_entry, kernel_end)
        .cmdline(&cmdline)
        .initramfs(&initramfs);
    vm.setup_linux_boot(&boot)?;

    let boot_started = Instant::now();
    let ready_cp = Checkpoint::initial_when_ready_with(
        vm,
        vt!(120 s),
        LabOpts {
            rng: RngMode::Seeded(BOOT_RNG_SEED),
            ..Default::default()
        },
    )?;
    println!(
        "boot->ready: {:.2}s wall, {:.3}s virtual",
        boot_started.elapsed().as_secs_f64(),
        ready_cp.time().as_secs_f64()
    );

    // ---- advance to the guest's first input request and checkpoint there ---
    //
    // This is the snapshot every testcase forks from. The guest is parked
    // inside HYPERCALL_FUZZ_NEXT_INPUT with all of its setup already done, so
    // that setup is paid once here rather than once per testcase.
    let freq = ready_cp.tsc_frequency();
    let mut warmup = ready_cp.branch()?;
    let (at, outcome) = warmup.run_until(ready_cp.time() + VirtDuration::from_secs(10, freq))?;
    match outcome {
        RunOutcome::Yielded {
            kind: ExitKind::FuzzNextInput,
        } => {}
        other => return Err(format!("guest never asked for an input: {other:?}").into()),
    }
    let capacity = warmup.fuzz_input_capacity()?;
    let input_cp = warmup.checkpoint()?;
    println!(
        "input checkpoint at {:.3}s virtual, guest buffer holds {capacity} bytes",
        at.as_secs_f64()
    );

    if args.input_size > capacity {
        return Err(format!(
            "--input-size {} exceeds the guest's {capacity}-byte buffer",
            args.input_size
        )
        .into());
    }

    // ---- baseline: fork and immediately discard ---------------------------
    //
    // Separates "what does a fork cost" from "what does an execution cost".
    let t0 = Instant::now();
    for _ in 0..args.iterations {
        let branch = input_cp.branch()?;
        drop(branch);
    }
    let fork_only = t0.elapsed();
    println!(
        "fork+drop only:   {:>8.1} forks/s  ({:.3} ms each)",
        args.iterations as f64 / fork_only.as_secs_f64(),
        fork_only.as_secs_f64() * 1000.0 / args.iterations as f64
    );

    // ---- the real loop ----------------------------------------------------
    let timeout = VirtDuration::from_secs(args.timeout_secs, freq);
    let mut executed = 0usize;
    let mut verified = 0usize;
    let t0 = Instant::now();

    for i in 0..args.iterations {
        // A distinct input per iteration, so a fork that wrongly inherited a
        // sibling's bytes shows up as a checksum mismatch rather than passing
        // silently.
        let input = synthetic_input(i, args.input_size);
        let expected: u64 = input.iter().map(|&b| u64::from(b)).sum();

        let mut branch = input_cp.branch()?;
        branch.serve_fuzz_input(&input)?;

        let (_, outcome) = branch.run_until(input_cp.time() + timeout)?;
        match outcome {
            // The guest asking for its *next* input is how it says "done".
            RunOutcome::Yielded {
                kind: ExitKind::FuzzNextInput,
            } => executed += 1,
            RunOutcome::ReachedTime => {
                eprintln!("iteration {i}: timed out");
                continue;
            }
            other => {
                eprintln!("iteration {i}: unexpected outcome {other:?}");
                continue;
            }
        }

        // The guest reports status + a checksum of the bytes it actually saw.
        match branch.fuzz_outcome()? {
            FuzzOutcome::Ok => {}
            other => {
                eprintln!("iteration {i}: guest reported {other:?}");
                continue;
            }
        }
        let got = branch.fuzz_aux()?;
        if got == expected {
            verified += 1;
        } else {
            eprintln!("iteration {i}: checksum {got} != expected {expected}");
        }
    }

    let elapsed = t0.elapsed();
    println!(
        "fork+serve+run:   {:>8.1} execs/s ({:.3} ms each)",
        executed as f64 / elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / executed.max(1) as f64
    );
    println!(
        "{executed}/{} executed, {verified}/{executed} checksums verified",
        args.iterations
    );

    if verified != args.iterations {
        return Err("not every testcase was executed and verified".into());
    }
    Ok(())
}

/// A testcase whose bytes depend on `seed`, so no two iterations agree.
fn synthetic_input(seed: usize, len: usize) -> Vec<u8> {
    let mut state = (seed as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}
