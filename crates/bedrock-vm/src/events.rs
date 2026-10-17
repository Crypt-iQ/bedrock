// SPDX-License-Identifier: GPL-2.0

//! Userspace reader for the unified event stream.
//!
//! Wire-format types live in `no_std` `bedrock_vmx::events`; this adds a
//! zero-copy TLV [`Iterator`] and `serde` JSON output (serialize-only: the byte
//! stream is canonical).

use std::borrow::Cow;
use std::io::{self, Write};
use std::mem::size_of;

use serde::Serialize;
use zerocopy::FromBytes;

use crate::ExitRecord;
pub use bedrock_vmx::events::{
    EventCategories, EventHeader, EventKind, InjectPayload, IoChannelPayload, IoChannelPhase,
    RandomPayload, RandomSource, EVENT_BUFFER_SIZE, EVENT_FLAG_DETERMINISTIC, EVENT_HEADER_SIZE,
};

/// A decoded view of one record's payload.
pub enum Event<'a> {
    Exit(&'a ExitRecord),
    Serial(&'a [u8]),
    Inject(&'a InjectPayload),
    /// Header plus the served bytes for `GetRandom` (empty for RDRAND/RDSEED,
    /// whose value is inline in the header).
    Randomness(&'a RandomPayload, &'a [u8]),
    /// Metadata plus the request command or response bytes.
    IoChannel(&'a IoChannelPayload, &'a [u8]),
    Unknown {
        kind: u16,
        payload: &'a [u8],
    },
    /// The payload was too short for the kind's fixed struct.
    Malformed,
}

/// A borrowed view over one TLV record: its header plus its payload bytes.
#[derive(Clone, Copy, Debug)]
pub struct EventRecord<'a> {
    pub header: &'a EventHeader,
    pub payload: &'a [u8],
}

impl<'a> EventRecord<'a> {
    pub fn seq(&self) -> u64 {
        self.header.seq
    }

    /// Emulated (deterministic) TSC at emit time.
    pub fn tsc(&self) -> u64 {
        self.header.tsc
    }

    /// Host (non-deterministic) TSC at emit time.
    pub fn real_tsc(&self) -> u64 {
        self.header.real_tsc
    }

    pub fn kind(&self) -> u16 {
        self.header.kind
    }

    /// True if the record participates in run-vs-run comparison.
    pub fn is_deterministic(&self) -> bool {
        self.header.flags & EVENT_FLAG_DETERMINISTIC != 0
    }

    /// Decode the payload according to `kind`.
    pub fn event(&self) -> Event<'a> {
        match self.header.kind {
            k if k == EventKind::Exit.as_u16() => match ExitRecord::ref_from_prefix(self.payload) {
                Ok((p, _)) => Event::Exit(p),
                Err(_) => Event::Malformed,
            },
            k if k == EventKind::Serial.as_u16() => Event::Serial(self.payload),
            k if k == EventKind::Inject.as_u16() => {
                match InjectPayload::ref_from_prefix(self.payload) {
                    Ok((p, _)) => Event::Inject(p),
                    Err(_) => Event::Malformed,
                }
            }
            k if k == EventKind::Randomness.as_u16() => {
                match RandomPayload::ref_from_prefix(self.payload) {
                    Ok((p, bytes)) => Event::Randomness(p, bytes),
                    Err(_) => Event::Malformed,
                }
            }
            k if k == EventKind::IoChannel.as_u16() => {
                match IoChannelPayload::ref_from_prefix(self.payload) {
                    Ok((p, data)) => Event::IoChannel(p, data),
                    Err(_) => Event::Malformed,
                }
            }
            kind => Event::Unknown {
                kind,
                payload: self.payload,
            },
        }
    }

    pub fn to_json(&self) -> EventJson<'a> {
        let body = match self.event() {
            Event::Exit(p) => EventBody::Exit(p),
            Event::Serial(bytes) => EventBody::Serial(String::from_utf8_lossy(bytes)),
            Event::Inject(p) => EventBody::Inject(p),
            Event::Randomness(p, bytes) => EventBody::Randomness {
                source: p.source,
                width: p.width,
                value: p.value,
                pid: p.pid,
                len: bytes.len(),
            },
            Event::IoChannel(p, data) => io_channel_body(p, data),
            Event::Unknown { kind, payload } if kind == EventKind::Diagnostic.as_u16() => {
                EventBody::Diagnostic {
                    len: payload.len(),
                    hex: payload.iter().map(|b| format!("{b:02x}")).collect(),
                }
            }
            Event::Unknown { kind, payload } => EventBody::Unknown {
                kind,
                len: payload.len(),
            },
            Event::Malformed => EventBody::Unknown {
                kind: self.header.kind,
                len: self.payload.len(),
            },
        };
        EventJson {
            seq: self.header.seq,
            tsc: self.header.tsc,
            real_tsc: self.header.real_tsc,
            deterministic: self.is_deterministic(),
            body,
        }
    }
}

/// Iterator over a drained event buffer (`buf[0..event_len]`); stops (without
/// panicking) at a truncated tail.
pub struct EventStream<'a> {
    buf: &'a [u8],
    off: usize,
}

impl<'a> EventStream<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, off: 0 }
    }
}

impl<'a> Iterator for EventStream<'a> {
    type Item = EventRecord<'a>;

    fn next(&mut self) -> Option<EventRecord<'a>> {
        let rest = self.buf.get(self.off..)?;
        let (header, after) = EventHeader::ref_from_prefix(rest).ok()?;
        let payload = after.get(..header.len as usize)?;
        self.off += size_of::<EventHeader>() + header.len as usize;
        // Records are padded up to an 8-byte boundary.
        self.off = (self.off + 7) & !7;
        Some(EventRecord { header, payload })
    }
}

// ============================================================================
// serde / JSONL output
// ============================================================================

/// Serialize a `u64` as a `0x…` hex string.
fn hex<S: serde::Serializer>(v: &u64, s: S) -> Result<S::Ok, S::Error> {
    s.collect_str(&format_args!("{:#x}", v))
}

fn io_channel_body<'a>(meta: &'a IoChannelPayload, data: &'a [u8]) -> EventBody<'a> {
    use crate::io_channel::{self, IoTarget};
    let is_request = meta.phase == IoChannelPhase::Request as u8;

    let mut target = None;
    let mut command = None;
    let mut record_output = None;
    let mut status = None;
    let mut exit_code = None;
    let mut output_len = None;
    let mut text = None;

    if let Some(req) = io_channel::decode_request(data) {
        target = Some(match req.target {
            IoTarget::Host => Cow::Borrowed("host"),
            IoTarget::Container(name) => name,
        });
        command = Some(req.command);
        record_output = Some(req.record_output);
    } else if let Some(resp) = io_channel::decode_response(data) {
        status = Some(resp.status);
        exit_code = Some(resp.exit_code);
        output_len = Some(resp.output_len);
    } else if !data.is_empty() {
        text = Some(String::from_utf8_lossy(data));
    }

    EventBody::IoChannel {
        phase: if is_request { "request" } else { "response" },
        target_tsc: is_request.then_some(meta.target_tsc),
        target,
        command,
        record_output,
        status,
        exit_code,
        output_len,
        text,
    }
}

/// A human-friendly serializable view of a record body (no padding, hex
/// randomness, utf8-lossy serial).
///
/// Adjacent tagging is required because `Serial` is a string, not a map.
#[derive(Serialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum EventBody<'a> {
    Exit(&'a ExitRecord),
    Serial(Cow<'a, str>),
    Inject(&'a InjectPayload),
    /// `width`/`value` apply to RDRAND/RDSEED; `pid`/`len` to GET_RANDOM (the
    /// served bytes are omitted).
    Randomness {
        /// 0 = RDRAND, 1 = RDSEED, 2 = GET_RANDOM.
        source: u8,
        width: u8,
        #[serde(serialize_with = "hex")]
        value: u64,
        pid: u32,
        len: usize,
    },
    /// Requests fill `target_tsc`/`target`/`command`/`record_output`; responses
    /// `status`/`exit_code`/`output_len`; undecodable payloads fall back to `text`.
    IoChannel {
        /// `"request"` or `"response"`.
        phase: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        target_tsc: Option<u64>,
        /// `"host"` or the container name.
        #[serde(skip_serializing_if = "Option::is_none")]
        target: Option<Cow<'a, str>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        command: Option<Cow<'a, str>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        record_output: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_len: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<Cow<'a, str>>,
    },
    /// Host-derived debugging payload (e.g. the PMI flight recorder), as hex.
    Diagnostic {
        len: usize,
        hex: String,
    },
    Unknown {
        kind: u16,
        len: usize,
    },
}

/// A flat JSON object for one record: header fields plus the flattened body.
#[derive(Serialize)]
pub struct EventJson<'a> {
    pub seq: u64,
    /// Emulated (deterministic) TSC.
    pub tsc: u64,
    /// Host (non-deterministic) TSC.
    pub real_tsc: u64,
    pub deterministic: bool,
    #[serde(flatten)]
    pub body: EventBody<'a>,
}

/// Write every record in a drained buffer as JSONL; returns the record count.
pub fn write_jsonl<W: Write>(writer: &mut W, drained: &[u8]) -> io::Result<usize> {
    let mut n = 0;
    for rec in EventStream::new(drained) {
        serde_json::to_writer(&mut *writer, &rec.to_json()).map_err(io::Error::other)?;
        writeln!(writer)?;
        n += 1;
    }
    Ok(n)
}

/// [`write_jsonl`] restricted to records whose kind is in `categories`.
pub fn write_jsonl_filtered<W: Write>(
    writer: &mut W,
    drained: &[u8],
    categories: EventCategories,
) -> io::Result<usize> {
    let mut n = 0;
    for rec in EventStream::new(drained) {
        if !categories.contains(category_of(rec.header.kind)) {
            continue;
        }
        serde_json::to_writer(&mut *writer, &rec.to_json()).map_err(io::Error::other)?;
        writeln!(writer)?;
        n += 1;
    }
    Ok(n)
}

/// Map a raw `kind` to its category bit (empty for unknown kinds).
pub fn category_of(kind: u16) -> EventCategories {
    match kind {
        k if k == EventKind::Exit.as_u16() => EventCategories::EXIT,
        k if k == EventKind::Serial.as_u16() => EventCategories::SERIAL,
        k if k == EventKind::Inject.as_u16() => EventCategories::INJECT,
        k if k == EventKind::Randomness.as_u16() => EventCategories::RANDOMNESS,
        k if k == EventKind::IoChannel.as_u16() => EventCategories::IO_CHANNEL,
        k if k == EventKind::Diagnostic.as_u16() => EventCategories::DIAGNOSTIC,
        _ => EventCategories::empty(),
    }
}

#[cfg(test)]
#[path = "events_tests.rs"]
mod tests;
