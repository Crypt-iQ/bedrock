// SPDX-License-Identifier: GPL-2.0

//! Host half of the `HYPERCALL_FILE_STORE` hypercall.
//!
//! FileWriter reads a file chunk from the guest via the registered shared buffer
//! and writes it to a host file.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;

use crate::Vm;

/// Identifier the guest registers its file-storing buffer under
/// (`HYPERCALL_REGISTER_FEEDBACK_BUFFER`). The host finds the buffer by this id.
pub const FILE_STORE_BUFFER_ID: &[u8] = b"bedrock-file-store";

/// Bytes reserved at the start of the shared buffer for the request/response
/// header. Data begins at this offset. The request header is
/// `u32 name_len | u32 chunk_len | u64 reserved`; the response header is
/// `i64 result | u64 reserved`. See guest/libvmcall.h for
/// VMCALL_FILE_STORE_HEADER_LEN.
pub const FILE_STORE_HEADER_LEN: usize = 16;

/// An error occurred while writing a chunk.
pub const FILE_STORE_IO_ERROR: i64 = -1;

/// Reads chunks of guest files and appends them to host files.
///
/// The guest chooses the file name. By default it is used as given, relative to
/// the process's current directory — so a guest may name an absolute host path,
/// which is how `bedrock-file-store` is normally driven (`bedrock-file-store
/// <guest-path> <host-path>`). [`with_base_dir`](FileWriter::with_base_dir)
/// redirects *relative* names into a chosen directory; an absolute name still
/// wins, as `Path::join` semantics imply.
pub struct FileWriter {
    handles: HashMap<String, File>,
    slot: Option<usize>,
    /// Directory guest files are written into. `None` means the process's
    /// current directory, preserving the original behaviour.
    base_dir: Option<PathBuf>,
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FileWriter {
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
            slot: None,
            base_dir: None,
        }
    }

    /// Write guest files into `dir` instead of the process's current directory.
    ///
    /// The directory is created if missing on first write.
    #[must_use]
    pub fn with_base_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    /// Write the chunk the guest sent to a file on the host.
    /// Returns the bytes accepted or 0 if the host disallows the file name.
    pub fn write(&mut self, vm: &mut Vm) -> io::Result<usize> {
        let slot = self.resolve_slot(vm)?;

        // Ensure the buffer is mapped read-write so the response lands in the
        // guest's pages. Map it once; subsequent fetches reuse the mapping.
        if vm.feedback_buffer_mut_at(slot).is_none() {
            vm.map_feedback_buffer_mut_at(slot)?;
        }
        let buf = vm
            .feedback_buffer_mut_at(slot)
            .ok_or_else(|| io::Error::other("file-store buffer disappeared after mapping"))?;

        if buf.len() < FILE_STORE_HEADER_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file-store buffer smaller than header",
            ));
        }

        // Parse the request header.
        let name_len = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let chunk_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;

        let name_end = FILE_STORE_HEADER_LEN + name_len;
        let chunk_end = name_end + chunk_len;
        if chunk_end > buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file-store overflows buffer, name_len {}, chunk_len {}",
                    name_len, chunk_len
                ),
            ));
        }
        let name =
            String::from_utf8_lossy(&buf[FILE_STORE_HEADER_LEN..FILE_STORE_HEADER_LEN + name_len])
                .into_owned();

        let base_dir = self.base_dir.clone();

        // Open and truncate the file if not already cached from a prior chunked write.
        let file = match self.handles.entry(name) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let path = match &base_dir {
                    Some(dir) => {
                        std::fs::create_dir_all(dir)?;
                        dir.join(e.key())
                    }
                    None => PathBuf::from(e.key()),
                };
                let handle = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(path)?;
                e.insert(handle)
            }
        };

        match file.write_all(&buf[name_end..chunk_end]) {
            Ok(()) => {
                write_result(buf, chunk_len as i64);
                Ok(chunk_len)
            }
            Err(e) => {
                write_result(buf, FILE_STORE_IO_ERROR);
                Err(e)
            }
        }
    }

    /// Resolve and cache the feedback-buffer slot the guest registered.
    fn resolve_slot(&mut self, vm: &Vm) -> io::Result<usize> {
        if let Some(slot) = self.slot {
            return Ok(slot);
        }
        let slots = vm.feedback_buffer_slots_for_id(FILE_STORE_BUFFER_ID)?;
        let slot = slots.first().copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "guest issued HYPERCALL_FILE_STORE but registered no bedrock-file-store buffer",
            )
        })?;
        self.slot = Some(slot);
        Ok(slot)
    }
}

/// Write the response `result` into the buffer header and zero the reserved part of
/// the header.
fn write_result(buf: &mut [u8], result: i64) {
    buf[0..8].copy_from_slice(&result.to_le_bytes());
    buf[8..16].fill(0);
}

