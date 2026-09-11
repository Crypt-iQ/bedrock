// SPDX-License-Identifier: GPL-2.0

//! Fuzz-input channel — the host half of the `HYPERCALL_FUZZ_NEXT_INPUT` ABI.
//!
//! This is how a guest harness receives testcases from a host-side fuzzer. It
//! is deliberately the same shape as [`crate::file_xfer`]: the guest registers
//! one feedback buffer under [`FUZZ_INPUT_BUFFER_ID`], and the host writes into
//! that buffer's (copy-on-write) pages before resuming the VM.
//!
//! 1. During its one-time setup the guest registers a buffer under the id
//!    `fuzzamoto-input` (`HYPERCALL_REGISTER_FEEDBACK_BUFFER`), sized for the
//!    largest testcase it expects plus [`FUZZ_INPUT_HEADER_LEN`].
//! 2. The guest issues `HYPERCALL_FUZZ_NEXT_INPUT`, which exits to userspace as
//!    [`ExitKind::FuzzNextInput`](crate::ExitKind::FuzzNextInput).
//! 3. The host ([`InputServer::serve`]) overwrites the buffer with a response
//!    header (`result` = input length, or [`FUZZ_INPUT_RESULT_EOF`]) followed by
//!    the input bytes, and runs the VM again.
//! 4. The guest reads `result` back out of the buffer and executes that input.
//!
//! ## Fork-per-testcase
//!
//! The reason this is worth a hypercall of its own rather than a file fetch is
//! where the *checkpoint* lands. The host runs the guest once, up to the first
//! `FuzzNextInput` exit, and checkpoints there — with the guest parked just
//! after the VMCALL, all of its expensive setup already done. Every subsequent
//! testcase is a fork of that checkpoint: map the fork's buffer, write a
//! different input into it, resume. Setup is paid once for the whole campaign,
//! and no two testcases can contaminate each other, because each one runs in a
//! VM that is bit-identical at the moment it receives its input.
//!
//! The hypervisor treats the buffer as opaque — it only advances RIP and exits
//! to userspace — so this module is the single host-side definition of the
//! framing. Keep it in sync with the guest's `libvmcall.h`.

use std::io;

use crate::Vm;

/// Identifier the guest registers its fuzz-input buffer under
/// (`HYPERCALL_REGISTER_FEEDBACK_BUFFER`). The host finds the buffer by this id.
pub const FUZZ_INPUT_BUFFER_ID: &[u8] = b"fuzzamoto-input";

/// Bytes reserved at the start of the shared buffer for the header. Payload
/// begins at this offset.
///
/// The header carries both directions of the exchange, because both happen at
/// the same hypercall:
///
/// | range      | writer | meaning                                          |
/// |------------|--------|--------------------------------------------------|
/// | `[0..8)`   | host   | `i64` result: input length, or [`FUZZ_INPUT_RESULT_EOF`] |
/// | `[8..16)`  | guest  | `u64` status of the *previous* testcase ([`FuzzOutcome`]) |
/// | `[16..24)` | guest  | `u64` aux: message length when status is `Fail`  |
/// | `[24..32)` | —      | reserved (zero)                                  |
///
/// The payload after the header is the input bytes (host→guest) or, when the
/// guest reports `Fail`, its message (guest→host).
pub const FUZZ_INPUT_HEADER_LEN: usize = 32;

/// Response `result` sentinel: no further inputs — the guest should shut down.
pub const FUZZ_INPUT_RESULT_EOF: i64 = -1;

/// What the guest reported about the testcase it just finished.
///
/// The guest writes this into the header immediately before asking for its next
/// input, so the single `HYPERCALL_FUZZ_NEXT_INPUT` exit means both "I am done
/// with the last one, here is how it went" and "give me another".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuzzOutcome {
    /// The testcase ran to completion without the harness detecting a bug.
    Ok,
    /// The harness could not use this testcase (e.g. it failed to decode).
    /// The fuzzer should not count it as an execution.
    Skip,
    /// The harness detected a bug, with this message.
    Fail(String),
    /// The guest wrote a status word this host build does not know.
    Unknown(u64),
}

/// Status word values the guest writes into `[8..16)`.
const STATUS_OK: u64 = 0;
const STATUS_SKIP: u64 = 1;
const STATUS_FAIL: u64 = 2;

/// Writes fuzzer testcases into a guest's registered input buffer.
///
/// One server drives one guest. Construct it once and call
/// [`serve`](Self::serve) for every
/// [`ExitKind::FuzzNextInput`](crate::ExitKind::FuzzNextInput) exit.
///
/// The resolved slot index is cached, but the *mapping* is not: a forked VM has
/// its own copy-on-write pages, so each fork must be mapped separately. That
/// makes an `InputServer` cheap to create per branch, which is the expected
/// usage in a fork-per-testcase loop.
#[derive(Debug, Default)]
pub struct InputServer {
    /// Cached feedback-buffer slot, resolved on first use. Registration happens
    /// during guest setup, before the first `FuzzNextInput` exit.
    slot: Option<usize>,
}

impl InputServer {
    /// Build a server for one guest.
    #[must_use]
    pub fn new() -> Self {
        Self { slot: None }
    }

    /// The usable input capacity of the guest's buffer, in bytes: its total
    /// size less the response header.
    ///
    /// # Errors
    ///
    /// Returns an error if the guest registered no `fuzzamoto-input` buffer, or
    /// if mapping it fails.
    pub fn capacity(&mut self, vm: &mut Vm) -> io::Result<usize> {
        let buf = self.buffer(vm)?;
        Ok(buf.len() - FUZZ_INPUT_HEADER_LEN)
    }

    /// Write one testcase into the guest's input buffer.
    ///
    /// Call this on a [`ExitKind::FuzzNextInput`](crate::ExitKind::FuzzNextInput)
    /// exit, then run the VM again.
    ///
    /// # Errors
    ///
    /// Returns an error if the guest registered no `fuzzamoto-input` buffer, if
    /// mapping it fails, or if `input` does not fit in it. An over-large input
    /// is the caller's bug (the fuzzer should respect
    /// [`capacity`](Self::capacity)), so it is a hard error rather than a
    /// truncation.
    pub fn serve(&mut self, vm: &mut Vm, input: &[u8]) -> io::Result<()> {
        let buf = self.buffer(vm)?;
        let cap = buf.len() - FUZZ_INPUT_HEADER_LEN;
        if input.len() > cap {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "fuzz input of {} bytes exceeds the guest's {cap}-byte input buffer",
                    input.len()
                ),
            ));
        }

        write_result(buf, i64::try_from(input.len()).expect("len fits in i64"));
        buf[FUZZ_INPUT_HEADER_LEN..FUZZ_INPUT_HEADER_LEN + input.len()].copy_from_slice(input);
        Ok(())
        // Note: write_result zeroes the guest-written words, so the outcome
        // read after this run is always the guest's report on *this* input and
        // never a leftover from the checkpoint it forked from.
    }

    /// Tell the guest there are no more inputs, so it should shut down.
    ///
    /// # Errors
    ///
    /// As [`serve`](Self::serve), minus the capacity check.
    pub fn serve_eof(&mut self, vm: &mut Vm) -> io::Result<()> {
        let buf = self.buffer(vm)?;
        write_result(buf, FUZZ_INPUT_RESULT_EOF);
        Ok(())
    }

    /// Read back what the guest reported about the testcase it just finished.
    ///
    /// Valid after a run that ended in
    /// [`ExitKind::FuzzNextInput`](crate::ExitKind::FuzzNextInput). Note that
    /// the very first such exit follows the guest's *setup*, not a testcase, so
    /// its status is meaningless — that is the exit a fuzzer checkpoints at
    /// rather than scores.
    ///
    /// # Errors
    ///
    /// As [`serve`](Self::serve).
    pub fn outcome(&mut self, vm: &mut Vm) -> io::Result<FuzzOutcome> {
        let buf = self.buffer(vm)?;
        let status = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        Ok(match status {
            STATUS_OK => FuzzOutcome::Ok,
            STATUS_SKIP => FuzzOutcome::Skip,
            STATUS_FAIL => {
                let len = u64::from_le_bytes(buf[16..24].try_into().unwrap()) as usize;
                // The guest is the untrusted side of this boundary: clamp
                // rather than trust its length, and accept lossy UTF-8.
                let end = FUZZ_INPUT_HEADER_LEN.saturating_add(len).min(buf.len());
                let msg = String::from_utf8_lossy(&buf[FUZZ_INPUT_HEADER_LEN..end]).into_owned();
                FuzzOutcome::Fail(msg)
            }
            other => FuzzOutcome::Unknown(other),
        })
    }

    /// The raw `aux` header word the guest wrote (`[16..24)`).
    ///
    /// Only meaningful to a guest and host that agree on what it holds; the
    /// harness ABI uses it for a `Fail` message length.
    ///
    /// # Errors
    ///
    /// As [`serve`](Self::serve).
    pub fn aux(&mut self, vm: &mut Vm) -> io::Result<u64> {
        let buf = self.buffer(vm)?;
        Ok(u64::from_le_bytes(buf[16..24].try_into().unwrap()))
    }

    /// Resolve, map and return the guest's input buffer as a mutable slice.
    fn buffer<'a>(&mut self, vm: &'a mut Vm) -> io::Result<&'a mut [u8]> {
        let slot = self.resolve_slot(vm)?;

        // Ensure the buffer is mapped read-write so writes land in the guest's
        // pages. Map it once per VM; subsequent serves reuse the mapping.
        if vm.feedback_buffer_mut_at(slot).is_none() {
            vm.map_feedback_buffer_mut_at(slot)?;
        }
        let buf = vm
            .feedback_buffer_mut_at(slot)
            .ok_or_else(|| io::Error::other("fuzz-input buffer disappeared after mapping"))?;

        if buf.len() < FUZZ_INPUT_HEADER_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "fuzz-input buffer smaller than header",
            ));
        }
        Ok(buf)
    }

    /// Resolve (and cache) the feedback-buffer slot the guest registered the
    /// input buffer under.
    fn resolve_slot(&mut self, vm: &Vm) -> io::Result<usize> {
        if let Some(slot) = self.slot {
            return Ok(slot);
        }
        let slots = vm.feedback_buffer_slots_for_id(FUZZ_INPUT_BUFFER_ID)?;
        let slot = slots.first().copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "guest issued HYPERCALL_FUZZ_NEXT_INPUT but registered no fuzzamoto-input buffer",
            )
        })?;
        self.slot = Some(slot);
        Ok(slot)
    }
}

/// Write the response `result` word into the buffer header and zero the rest of
/// the header — including the guest-written status and aux words, so each run
/// starts from a clean slate.
fn write_result(buf: &mut [u8], result: i64) {
    buf[0..8].copy_from_slice(&result.to_le_bytes());
    buf[8..FUZZ_INPUT_HEADER_LEN].fill(0);
}
