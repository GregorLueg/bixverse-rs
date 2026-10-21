//! The v4 chunk encoding and the v3 -> v4 converters.
//!
//! v4 keeps the v3 container (per-chunk length prefix, offsets, index map,
//! library-size index) and changes what a chunk holds. Per chunk:
//!
//! * gene (or cell) indices as delta bit-packed blocks of 128 instead of
//!   absolute `u32`, which lz4 cannot compress;
//! * raw counts bit-packed the same way instead of `u16` / `u32`;
//! * the normalised layer dropped when it recomputes bit for bit from the raw
//!   counts, library size and target size, stored verbatim otherwise;
//! * an uncompressed fixed header, so nnz and library size are readable
//!   without decompressing, and an optional byte codec over the body.
//!
//! Gene-based files recompute the norm of an entry from the library size of
//! its cell, so they carry a per-cell library-size table in the tail, pointed
//! at by the file header's `library_index_offset`.
//!
//! Chunk layout (all little endian):
//!
//! ```text
//! [0..4)   nnz            u32
//! [4..8)   body_len       u32   decoded body length
//! [8..16)  aux            u64   cell: library size; gene: avg_exp f16 bits
//! [16..24) original_index u64
//! [24]     to_keep        u8
//! [25]     flags          u8    bit 0: norm stored, bit 1: raw stored minus one
//! [26]     codec          u8    0 none, 1 lz4, 2 zstd
//! [27..32) reserved
//! ```
//!
//! The body is the packed indices, the packed raw counts and, when flagged,
//! the f16 norm layer, then compressed with `codec`.

use lz4_flex::block::{compress as lz4_compress, decompress_into as lz4_decompress_into};
use rayon::prelude::*;
use std::time::Instant;
use thousands::Separable;

use crate::data_io::*;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Length of the fixed v4 chunk header, see the module docs.
pub(crate) const V4_HEADER_LEN: usize = 32;

/// Values per bit-packed block. Each block carries its own width, so one large
/// delta only widens its own block, and decoding keeps a short fixed trip
/// count.
const PACK_BLOCK: usize = 128;

/// Header flag: the norm layer is stored verbatim after the raw counts.
const FLAG_NORM_STORED: u8 = 1;

/// Header flag: raw counts are stored as `count - 1`. Set when every count is
/// at least one, which holds for any non-zero entry and saves a bit per value.
const FLAG_RAW_MINUS_ONE: u8 = 2;

/// Largest count the per-cell norm lookup table covers. Counts above it fall
/// back to one `ln_1p` each; they are rare in UMI data.
const NORM_LUT_MAX: usize = 255;

/// Counts covered by the per-cell norm table of a gene file, see
/// [`gene_norm_table`]. Larger counts fall back to one `ln_1p`. Not tuned: it
/// trades table size (two bytes per cell per count) against how often the
/// fallback runs.
pub(crate) const GENE_NORM_LUT_COUNTS: usize = 4;

/// Zstd level for [`ChunkCodec::Zstd`]. Low on purpose: decode speed barely
/// moves with level, so higher levels only cost encode time for little gain on
/// payloads this small.
const ZSTD_LEVEL: i32 = 1;

/// Cell chunks converted per parallel batch. Bounds the decoded v3 chunks and
/// the compressed v4 payloads held at once.
const CONVERT_CELL_BATCH: usize = 4_096;

/// Gene chunks converted per parallel batch. Genes are far larger than cells
/// (one entry per expressing cell), so the batch is much smaller.
const CONVERT_GENE_BATCH: usize = 64;

///////////
// Codec //
///////////

/// Byte codec applied to a v4 chunk body.
///
/// Recorded per chunk, so a chunk that does not shrink is stored as
/// [`ChunkCodec::None`] whatever the file was asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkCodec {
    /// Body stored as is.
    None,
    /// lz4 block. Fastest to decode.
    Lz4,
    /// zstd at a low level. Smaller than lz4, slower to decode.
    Zstd,
}

impl ChunkCodec {
    /// The byte stored in the chunk header.
    ///
    /// ### Returns
    ///
    /// 0 for none, 1 for lz4, 2 for zstd.
    fn to_u8(self) -> u8 {
        match self {
            ChunkCodec::None => 0,
            ChunkCodec::Lz4 => 1,
            ChunkCodec::Zstd => 2,
        }
    }

    /// Parse the byte stored in a chunk header.
    ///
    /// ### Params
    ///
    /// * `b` - The header byte
    ///
    /// ### Returns
    ///
    /// The codec, or [`ScDataErrors::V4ChunkCorrupt`] for an unknown byte.
    fn from_u8(b: u8) -> Result<Self, ScDataErrors> {
        match b {
            0 => Ok(ChunkCodec::None),
            1 => Ok(ChunkCodec::Lz4),
            2 => Ok(ChunkCodec::Zstd),
            other => Err(ScDataErrors::V4ChunkCorrupt(format!(
                "unknown codec byte {other}"
            ))),
        }
    }
}

/////////////////
// Bit packing //
/////////////////

/// Bit-pack values in blocks of [`PACK_BLOCK`]: per block one width byte, then
/// the values at that width, padded to a whole byte.
///
/// ### Params
///
/// * `vals` - The values
/// * `out` - Destination, appended to
fn pack(vals: &[u32], out: &mut Vec<u8>) {
    for block in vals.chunks(PACK_BLOCK) {
        let max = block.iter().copied().max().unwrap_or(0);
        let width = (32 - max.leading_zeros()) as usize;
        out.push(width as u8);
        let (mut acc, mut nbits) = (0u64, 0usize);
        for &v in block {
            acc |= (v as u64) << nbits;
            nbits += width;
            while nbits >= 8 {
                out.push(acc as u8);
                acc >>= 8;
                nbits -= 8;
            }
        }
        if nbits > 0 {
            out.push(acc as u8);
        }
    }
}

/// Read eight little-endian bytes at `at`, zero padded past the end.
///
/// ### Params
///
/// * `b` - The block bytes
/// * `at` - Start offset, at most `b.len()`
///
/// ### Returns
///
/// The word.
#[inline(always)]
fn read_word(b: &[u8], at: usize) -> u64 {
    match b.get(at..at + 8) {
        Some(s) => u64::from_le_bytes(s.try_into().expect("8-byte slice")),
        None => {
            let mut tmp = [0u8; 8];
            let rest = &b[at..];
            tmp[..rest.len()].copy_from_slice(rest);
            u64::from_le_bytes(tmp)
        }
    }
}

/// Unpack one full block of [`PACK_BLOCK`] values at the compile-time width
/// `W`.
///
/// Eight values at width `W` occupy exactly `W` bytes, so group `g` starts at
/// byte `g * W` and every shift and offset inside a group is a constant. That
/// lets the compiler unroll and vectorise what the generic loop cannot.
///
/// ### Params
///
/// * `src` - At least `16 * W + 8` bytes starting at the block's packed data;
///   the slack lets every value be read as one unaligned word
/// * `out` - The 128 decoded values
#[inline(always)]
fn unpack_full<const W: usize>(src: &[u8], out: &mut [u32; PACK_BLOCK]) {
    let mask = if W >= 32 { u32::MAX } else { (1u32 << W) - 1 };
    for g in 0..PACK_BLOCK / 8 {
        for j in 0..8 {
            let bit = j * W;
            let at = g * W + (bit >> 3);
            let word = u64::from_le_bytes(src[at..at + 8].try_into().expect("8-byte slice"));
            out[g * 8 + j] = ((word >> (bit & 7)) as u32) & mask;
        }
    }
}

/// Dispatch [`unpack_full`] on a runtime width in `1..=32`.
///
/// ### Params
///
/// * `width` - Bit width of the block
/// * `src` - See [`unpack_full`]
/// * `out` - The 128 decoded values
#[inline(always)]
fn unpack_full_dispatch(width: usize, src: &[u8], out: &mut [u32; PACK_BLOCK]) {
    macro_rules! by_width {
        ($($w:literal),*) => {
            match width {
                $($w => unpack_full::<$w>(src, out),)*
                _ => unreachable!("width checked to be 1..=32"),
            }
        };
    }
    by_width!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32
    );
}

/// Unpack `n` values written by [`pack`].
///
/// Full blocks with slack behind them go through the width-specialised
/// [`unpack_full`]; the last partial block, and a full block that ends the
/// buffer, take the generic path.
///
/// ### Params
///
/// * `buf` - Bytes starting at the first block
/// * `n` - Number of values
/// * `out` - Destination, appended to
///
/// ### Returns
///
/// The bytes consumed, or [`ScDataErrors::V4ChunkCorrupt`] for a bad width or
/// a buffer that ends early.
fn unpack(buf: &[u8], n: usize, out: &mut Vec<u32>) -> Result<usize, ScDataErrors> {
    out.reserve(n);
    let mut pos = 0usize;
    let mut left = n;
    while left > 0 {
        let take = left.min(PACK_BLOCK);
        let width = *buf.get(pos).ok_or_else(truncated)? as usize;
        if width > 32 {
            return Err(ScDataErrors::V4ChunkCorrupt(format!(
                "bit width {width} above 32"
            )));
        }
        pos += 1;
        let need = (take * width).div_ceil(8);
        let block = buf.get(pos..pos + need).ok_or_else(truncated)?;

        if width == 0 {
            out.resize(out.len() + take, 0);
        } else if take == PACK_BLOCK && buf.len() >= pos + need + 8 {
            let at = out.len();
            out.resize(at + PACK_BLOCK, 0);
            let dst: &mut [u32; PACK_BLOCK] = (&mut out[at..at + PACK_BLOCK])
                .try_into()
                .expect("128-value slice");
            unpack_full_dispatch(width, &buf[pos..], dst);
        } else {
            let mask = if width == 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            for k in 0..take {
                let bit = k * width;
                out.push(((read_word(block, bit >> 3) >> (bit & 7)) as u32) & mask);
            }
        }
        pos += need;
        left -= take;
    }
    Ok(pos)
}

/// Bytes [`pack`] used for `n` values, found by walking the block widths.
///
/// ### Params
///
/// * `buf` - Bytes starting at the first block
/// * `n` - Number of values
///
/// ### Returns
///
/// The packed length, or [`ScDataErrors::V4ChunkCorrupt`] for a bad width or a
/// buffer that ends early.
fn packed_len(buf: &[u8], n: usize) -> Result<usize, ScDataErrors> {
    let mut pos = 0usize;
    let mut left = n;
    while left > 0 {
        let take = left.min(PACK_BLOCK);
        let width = *buf.get(pos).ok_or_else(truncated)? as usize;
        if width > 32 {
            return Err(ScDataErrors::V4ChunkCorrupt(format!(
                "bit width {width} above 32"
            )));
        }
        pos += 1 + (take * width).div_ceil(8);
        left -= take;
    }
    if pos > buf.len() {
        return Err(truncated());
    }
    Ok(pos)
}

/// The error for a body that ends before its own header says it should.
///
/// ### Returns
///
/// [`ScDataErrors::V4ChunkCorrupt`].
fn truncated() -> ScDataErrors {
    ScDataErrors::V4ChunkCorrupt("body ends early".to_string())
}

////////////
// Header //
////////////

/// Parsed fixed header of a v4 chunk.
#[derive(Clone, Copy, Debug)]
pub(crate) struct V4Header {
    /// Entries in the chunk.
    pub nnz: usize,
    /// Decoded body length in bytes.
    body_len: usize,
    /// Cell chunks: library size. Gene chunks: f16 bits of the average
    /// expression.
    pub aux: u64,
    /// Original index of the cell or gene.
    pub original_index: usize,
    /// Whether the cell or gene is flagged to keep.
    pub to_keep: bool,
    /// [`FLAG_NORM_STORED`] and [`FLAG_RAW_MINUS_ONE`].
    flags: u8,
    /// Codec of the body.
    codec: ChunkCodec,
}

impl V4Header {
    /// Parse the header at the start of a chunk.
    ///
    /// ### Params
    ///
    /// * `chunk` - The chunk bytes, header first
    ///
    /// ### Returns
    ///
    /// The header, [`ScDataErrors::ChunkBufferTooSmall`] for a chunk shorter
    /// than the header, or [`ScDataErrors::V4ChunkCorrupt`] for a bad codec.
    pub(crate) fn parse(chunk: &[u8]) -> Result<Self, ScDataErrors> {
        if chunk.len() < V4_HEADER_LEN {
            return Err(ScDataErrors::ChunkBufferTooSmall {
                expected: V4_HEADER_LEN,
                found: chunk.len(),
            });
        }
        let u32_at = |at: usize| {
            u32::from_le_bytes(chunk[at..at + 4].try_into().expect("4-byte slice")) as usize
        };
        let u64_at =
            |at: usize| u64::from_le_bytes(chunk[at..at + 8].try_into().expect("8-byte slice"));
        Ok(Self {
            nnz: u32_at(0),
            body_len: u32_at(4),
            aux: u64_at(8),
            original_index: u64_at(16) as usize,
            to_keep: chunk[24] != 0,
            flags: chunk[25],
            codec: ChunkCodec::from_u8(chunk[26])?,
        })
    }
}

////////////
// Encode //
////////////

/// Everything [`encode_chunk`] needs about one cell or gene.
struct EncodeIn<'a> {
    /// Gene (cell chunks) or cell (gene chunks) indices, ascending.
    indices: &'a [u32],
    /// Raw counts.
    raw: &'a [u32],
    /// Stored normalised layer, written only when it does not recompute.
    norm: &'a [F16],
    /// The header `aux` field.
    aux: u64,
    /// Original index of the cell or gene.
    original_index: usize,
    /// Keep flag.
    to_keep: bool,
    /// Whether the norm layer recomputes bit for bit and can be dropped.
    norm_derivable: bool,
}

/// Serialise one chunk into the v4 layout.
///
/// Index deltas wrap, so unsorted or duplicated indices stay lossless; they
/// only cost width.
///
/// ### Params
///
/// * `p` - The chunk contents
/// * `codec` - Requested body codec; falls back to none if it does not shrink
///   the body
///
/// ### Returns
///
/// Header plus body.
fn encode_chunk(p: &EncodeIn, codec: ChunkCodec) -> Vec<u8> {
    let n = p.indices.len();

    let mut deltas = Vec::with_capacity(n);
    let mut prev = 0u32;
    for (k, &x) in p.indices.iter().enumerate() {
        deltas.push(if k == 0 {
            x
        } else {
            x.wrapping_sub(prev).wrapping_sub(1)
        });
        prev = x;
    }

    let mut flags = 0u8;
    let mut body = Vec::with_capacity(n * 2);
    pack(&deltas, &mut body);

    if p.raw.iter().all(|&x| x >= 1) {
        flags |= FLAG_RAW_MINUS_ONE;
        let minus_one: Vec<u32> = p.raw.iter().map(|&x| x - 1).collect();
        pack(&minus_one, &mut body);
    } else {
        pack(p.raw, &mut body);
    }

    if !p.norm_derivable {
        flags |= FLAG_NORM_STORED;
        for v in p.norm {
            body.extend_from_slice(&v.to_le_bytes());
        }
    }

    let (used, payload) = match codec {
        ChunkCodec::None => (ChunkCodec::None, None),
        ChunkCodec::Lz4 => (ChunkCodec::Lz4, Some(lz4_compress(&body))),
        ChunkCodec::Zstd => (
            ChunkCodec::Zstd,
            zstd::bulk::compress(&body, ZSTD_LEVEL).ok(),
        ),
    };
    let (used, payload) = match payload {
        Some(c) if c.len() < body.len() => (used, c),
        _ => (ChunkCodec::None, body.clone()),
    };

    let mut out = Vec::with_capacity(V4_HEADER_LEN + payload.len());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&p.aux.to_le_bytes());
    out.extend_from_slice(&(p.original_index as u64).to_le_bytes());
    out.extend_from_slice(&[p.to_keep as u8, flags, used.to_u8(), 0, 0, 0, 0, 0]);
    out.extend_from_slice(&payload);
    out
}

impl CsrCellChunk {
    /// Serialise the cell into a v4 chunk.
    ///
    /// The norm layer is dropped when every entry recomputes bit for bit from
    /// the raw count, the library size and `target_size`.
    ///
    /// ### Params
    ///
    /// * `target_size` - Target size of the file the cell goes into, `0.0` if
    ///   unknown, which stores the norm
    /// * `codec` - Body codec
    ///
    /// ### Returns
    ///
    /// The chunk bytes, or [`ScDataErrors::V4ChunkCorrupt`] for layers of
    /// unequal length.
    pub fn to_v4_bytes(
        &self,
        target_size: f32,
        codec: ChunkCodec,
    ) -> Result<Vec<u8>, ScDataErrors> {
        let n = self.indices.len();
        if self.data_raw.len() != n || self.data_norm.len() != n {
            return Err(ScDataErrors::V4ChunkCorrupt(format!(
                "cell {} has layers of unequal length",
                self.original_index
            )));
        }
        let raw: Vec<u32> = self.data_raw.iter().collect();
        let lib = self.library_size as f32;
        let norm_derivable = target_size != 0.0
            && raw
                .iter()
                .zip(&self.data_norm)
                .all(|(&x, v)| norm_value(x as f32, lib, target_size).to_bits() == v.to_bits());

        Ok(encode_chunk(
            &EncodeIn {
                indices: &self.indices,
                raw: &raw,
                norm: &self.data_norm,
                aux: self.library_size as u64,
                original_index: self.original_index,
                to_keep: self.to_keep,
                norm_derivable,
            },
            codec,
        ))
    }
}

impl CscGeneChunk {
    /// Serialise the gene into a v4 chunk.
    ///
    /// The norm layer is dropped when every entry recomputes bit for bit from
    /// the raw count, the library size of its cell and `target_size`.
    ///
    /// ### Params
    ///
    /// * `cell_lib` - Library size of every cell as `f32`, indexed by the cell
    ///   indices this chunk stores
    /// * `target_size` - Target size of the file the gene goes into, `0.0` if
    ///   unknown, which stores the norm
    /// * `codec` - Body codec
    ///
    /// ### Returns
    ///
    /// The chunk bytes, or [`ScDataErrors::V4ChunkCorrupt`] for layers of
    /// unequal length.
    pub fn to_v4_bytes(
        &self,
        cell_lib: &[f32],
        target_size: f32,
        codec: ChunkCodec,
    ) -> Result<Vec<u8>, ScDataErrors> {
        let n = self.indices.len();
        if self.data_raw.len() != n || self.data_norm.len() != n {
            return Err(ScDataErrors::V4ChunkCorrupt(format!(
                "gene {} has layers of unequal length",
                self.original_index
            )));
        }
        let raw: Vec<u32> = self.data_raw.iter().collect();
        let norm_derivable = target_size != 0.0
            && self
                .indices
                .iter()
                .zip(&raw)
                .zip(&self.data_norm)
                .all(|((&cell, &x), v)| {
                    cell_lib.get(cell as usize).is_some_and(|&lib| {
                        norm_value(x as f32, lib, target_size).to_bits() == v.to_bits()
                    })
                });

        Ok(encode_chunk(
            &EncodeIn {
                indices: &self.indices,
                raw: &raw,
                norm: &self.data_norm,
                aux: self.avg_exp.to_bits() as u64,
                original_index: self.original_index,
                to_keep: self.to_keep,
                norm_derivable,
            },
            codec,
        ))
    }
}

////////////
// Decode //
////////////

/// Reusable buffers for decoding v4 chunks, one per worker.
///
/// `bytes` also serves the v3 path as its decompression scratch, so the fused
/// readers carry one scratch type for both versions.
#[derive(Default)]
pub(crate) struct V4Decoded {
    /// Decompressed body (v4) or decompressed chunk (v3).
    pub bytes: Vec<u8>,
    /// Decoded indices.
    pub idx: Vec<u32>,
    /// Decoded raw counts.
    pub raw: Vec<u32>,
    /// Decoded normalised layer.
    pub norm: Vec<f32>,
}

/// What the decoder needs to know about the file a chunk came from.
pub(crate) struct V4Ctx<'a> {
    /// Target size from the file header, `0.0` if unknown.
    pub target_size: f32,
    /// Per-cell library sizes as `f32`, present on gene-based v4 files.
    pub cell_lib: Option<&'a [f32]>,
    /// Normalised value of small counts per cell, see [`gene_norm_table`].
    pub cell_norm: Option<&'a [u16]>,
    /// Whether the chunk is a cell (CSR) chunk.
    pub is_cell: bool,
}

/// Decode one v4 chunk into `s`.
///
/// Indices are always decoded. Raw counts are decoded when asked for, or when
/// the norm has to be recomputed from them. The norm is only produced when
/// asked for.
///
/// ### Params
///
/// * `chunk` - Header plus body
/// * `ctx` - File context
/// * `want_raw` - Whether `s.raw` is needed
/// * `want_norm` - Whether `s.norm` is needed
/// * `s` - Scratch the layers are decoded into
///
/// ### Returns
///
/// The parsed header, or the first corruption error.
pub(crate) fn decode_v4(
    chunk: &[u8],
    ctx: &V4Ctx,
    want_raw: bool,
    want_norm: bool,
    s: &mut V4Decoded,
) -> Result<V4Header, ScDataErrors> {
    let V4Decoded {
        bytes,
        idx,
        raw,
        norm,
    } = s;
    idx.clear();
    norm.clear();
    decode_v4_into(
        chunk,
        ctx,
        want_raw,
        bytes,
        raw,
        idx,
        want_norm.then_some(norm),
    )
}

/// Decode one v4 chunk, appending indices and norm to caller-owned vectors.
///
/// The fused readers hand in the vectors of their output fragment, so an
/// unfiltered run is decoded straight into place without an intermediate copy.
/// Raw counts land in `raw`, which is overwritten: they still need a typed
/// conversion by the caller.
///
/// ### Params
///
/// * `chunk` - Header plus body
/// * `ctx` - File context
/// * `want_raw` - Whether `raw` is needed
/// * `bytes` - Scratch for the decompressed body
/// * `raw` - Receives the raw counts, cleared first
/// * `idx` - Indices are appended
/// * `norm` - The normalised layer is appended when `Some`
///
/// ### Returns
///
/// The parsed header, or the first corruption error. On error `idx` and
/// `norm` may hold a partial run.
pub(crate) fn decode_v4_into(
    chunk: &[u8],
    ctx: &V4Ctx,
    want_raw: bool,
    bytes: &mut Vec<u8>,
    raw: &mut Vec<u32>,
    idx: &mut Vec<u32>,
    norm: Option<&mut Vec<f32>>,
) -> Result<V4Header, ScDataErrors> {
    let h = V4Header::parse(chunk)?;

    let stored = &chunk[V4_HEADER_LEN..];
    let body: &[u8] = match h.codec {
        ChunkCodec::None => stored,
        ChunkCodec::Lz4 => {
            bytes.resize(h.body_len, 0);
            let n = lz4_decompress_into(stored, bytes)
                .map_err(|e| ScDataErrors::V4ChunkCorrupt(format!("lz4: {e}")))?;
            &bytes[..n]
        }
        ChunkCodec::Zstd => {
            bytes.resize(h.body_len, 0);
            let n = zstd::bulk::decompress_to_buffer(stored, &mut bytes[..])
                .map_err(|e| ScDataErrors::V4ChunkCorrupt(format!("zstd: {e}")))?;
            &bytes[..n]
        }
    };

    let n = h.nnz;
    let start = idx.len();
    let used = unpack(body, n, idx)?;
    let mut prev = 0u32;
    for (k, v) in idx[start..].iter_mut().enumerate() {
        if k > 0 {
            *v = prev.wrapping_add(*v).wrapping_add(1);
        }
        prev = *v;
    }

    let norm_stored = h.flags & FLAG_NORM_STORED != 0;
    let want_norm = norm.is_some();
    let need_raw = want_raw || (want_norm && !norm_stored);
    let rest = &body[used..];

    let raw_len = if need_raw {
        raw.clear();
        let len = unpack(rest, n, raw)?;
        if h.flags & FLAG_RAW_MINUS_ONE != 0 {
            raw.iter_mut().for_each(|x| *x = x.wrapping_add(1));
        }
        len
    } else if want_norm {
        packed_len(rest, n)?
    } else {
        0
    };

    let Some(norm) = norm else {
        return Ok(h);
    };

    norm.reserve(n);
    if norm_stored {
        let stored_norm = rest.get(raw_len..raw_len + n * 2).ok_or_else(truncated)?;
        norm.extend(
            stored_norm
                .chunks_exact(2)
                .map(|c| F16::from_le_bytes([c[0], c[1]]).to_f32()),
        );
    } else if ctx.is_cell {
        let lib = h.aux as f32;
        let top = raw.iter().copied().max().unwrap_or(0) as usize;
        let lut: Vec<f32> = (0..=top.min(NORM_LUT_MAX))
            .map(|x| norm_value(x as f32, lib, ctx.target_size).to_f32())
            .collect();
        norm.extend(raw.iter().map(|&x| match lut.get(x as usize) {
            Some(&v) => v,
            None => norm_value(x as f32, lib, ctx.target_size).to_f32(),
        }));
    } else {
        // an empty run has nothing to recompute and needs no table
        let cell_lib = match ctx.cell_lib {
            Some(table) => table,
            None if n == 0 => &[],
            None => return Err(ScDataErrors::V4MissingCellLibraryTable),
        };
        // an out-of-table cell is flagged and reported once after the loop,
        // which keeps the hot closure free of an early return
        let mut outside = None;
        let table = ctx.cell_norm.unwrap_or(&[]);
        norm.extend(idx[start..].iter().zip(raw.iter()).map(|(&cell, &x)| {
            let slot = (x as usize).wrapping_sub(1);
            if slot < GENE_NORM_LUT_COUNTS
                && let Some(&bits) = table.get(cell as usize * GENE_NORM_LUT_COUNTS + slot)
            {
                return F16::from_bits(bits).to_f32();
            }
            match cell_lib.get(cell as usize) {
                Some(&lib) => norm_value(x as f32, lib, ctx.target_size).to_f32(),
                None => {
                    outside = Some(cell);
                    0.0
                }
            }
        }));
        if let Some(cell) = outside {
            return Err(ScDataErrors::V4ChunkCorrupt(format!(
                "cell {cell} outside the library table"
            )));
        }
    }

    Ok(h)
}

/// Per-cell table of the normalised value of counts `1..=GENE_NORM_LUT_COUNTS`.
///
/// Gene runs normalise each entry against a different cell, so there is no
/// per-run lookup table as in the cell view. Nearly every UMI count is small,
/// so a flat table indexed by `cell * GENE_NORM_LUT_COUNTS + count - 1` turns
/// the per-entry `ln_1p` into a load. Built once when the file is opened, with
/// the same [`norm_value`] the encoder verified against, so values stay bit
/// identical.
///
/// ### Params
///
/// * `cell_lib` - Library size per cell as `f32`
/// * `target_size` - Target size from the file header
///
/// ### Returns
///
/// f16 bits, `cell_lib.len() * GENE_NORM_LUT_COUNTS` long.
pub(crate) fn gene_norm_table(cell_lib: &[f32], target_size: f32) -> Vec<u16> {
    let mut table = vec![0u16; cell_lib.len() * GENE_NORM_LUT_COUNTS];
    table
        .par_chunks_mut(GENE_NORM_LUT_COUNTS)
        .zip(cell_lib.par_iter())
        .for_each(|(row, &lib)| {
            for (k, v) in row.iter_mut().enumerate() {
                *v = norm_value((k + 1) as f32, lib, target_size).to_bits();
            }
        });
    table
}

/// Borrowed view of a decoded v4 run, for the fused readers.
pub(crate) struct V4Run<'a> {
    /// Decoded indices.
    idx: &'a [u32],
    /// Decoded raw counts, stale when raw was not requested.
    raw: &'a [u32],
    /// Decoded normalised layer, stale when norm was not requested.
    norm: &'a [f32],
}

impl<'a> V4Run<'a> {
    /// View the layers a [`decode_v4`] call left in `s`.
    ///
    /// ### Params
    ///
    /// * `s` - The scratch
    ///
    /// ### Returns
    ///
    /// The view.
    pub(crate) fn new(s: &'a V4Decoded) -> Self {
        Self {
            idx: &s.idx,
            raw: &s.raw,
            norm: &s.norm,
        }
    }
}

impl RunSource for V4Run<'_> {
    #[inline]
    fn nnz(&self) -> usize {
        self.idx.len()
    }

    #[inline(always)]
    fn index_at(&self, i: usize) -> u32 {
        self.idx[i]
    }

    #[inline(always)]
    fn raw_at(&self, i: usize) -> u32 {
        self.raw[i]
    }

    #[inline(always)]
    fn norm_at(&self, i: usize) -> f32 {
        self.norm[i]
    }

    fn extend_indices(&self, out: &mut Vec<u32>) {
        out.extend_from_slice(self.idx);
    }

    fn extend_raw<T: num_traits::FromPrimitive>(
        &self,
        out: &mut Vec<T>,
    ) -> Result<(), ScDataErrors> {
        extend_raw_checked(out, self.raw[..self.idx.len()].iter().copied())
    }

    fn extend_norm(&self, out: &mut Vec<f32>) {
        out.extend_from_slice(&self.norm[..self.idx.len()]);
    }
}

/// Decode a v4 cell chunk into an owned [`CsrCellChunk`].
///
/// ### Params
///
/// * `chunk` - Header plus body
/// * `ctx` - File context
/// * `s` - Scratch
///
/// ### Returns
///
/// The cell, with both layers.
pub(crate) fn v4_cell_chunk(
    chunk: &[u8],
    ctx: &V4Ctx,
    s: &mut V4Decoded,
) -> Result<CsrCellChunk, ScDataErrors> {
    let h = decode_v4(chunk, ctx, true, true, s)?;
    Ok(CsrCellChunk {
        data_raw: RawCounts::from_u32_auto(&s.raw),
        data_norm: s.norm.iter().map(|&x| F16::from_f32(x)).collect(),
        library_size: h.aux as usize,
        indices: s.idx.clone(),
        original_index: h.original_index,
        to_keep: h.to_keep,
    })
}

/// Decode a v4 gene chunk into an owned [`CscGeneChunk`].
///
/// ### Params
///
/// * `chunk` - Header plus body
/// * `ctx` - File context
/// * `s` - Scratch
///
/// ### Returns
///
/// The gene, with both layers.
pub(crate) fn v4_gene_chunk(
    chunk: &[u8],
    ctx: &V4Ctx,
    s: &mut V4Decoded,
) -> Result<CscGeneChunk, ScDataErrors> {
    let h = decode_v4(chunk, ctx, true, true, s)?;
    Ok(CscGeneChunk {
        data_raw: RawCounts::from_u32_auto(&s.raw),
        data_norm: s.norm.iter().map(|&x| F16::from_f32(x)).collect(),
        avg_exp: F16::from_bits(h.aux as u16),
        nnz: h.nnz,
        indices: s.idx.clone(),
        original_index: h.original_index,
        to_keep: h.to_keep,
    })
}

/////////////////
// ChunkFormat //
/////////////////

/// How a writer encodes the chunks it appends.
///
/// Copied out of the writer so parallel payload builders can encode without
/// borrowing it.
#[derive(Clone, Copy, Debug)]
pub struct ChunkFormat {
    /// File version: [`SC_FILE_VERSION`] or [`SC_FILE_VERSION_V3`].
    pub version: u32,
    /// Body codec of v4 chunks. Unused by v3.
    pub codec: ChunkCodec,
    /// Target size in the file header, which readers recompute the norm layer
    /// with. `0.0` stores the norm layer verbatim.
    pub target_size: f32,
}

impl ChunkFormat {
    /// Whether this format is the current (v4) one.
    ///
    /// ### Returns
    ///
    /// `true` for v4.
    pub fn is_v4(&self) -> bool {
        self.version == SC_FILE_VERSION
    }

    /// Encode a cell chunk for a file of this format.
    ///
    /// ### Params
    ///
    /// * `chunk` - The cell
    ///
    /// ### Returns
    ///
    /// The payload the writer's `write_compressed_cell_chunks` expects.
    pub fn encode_cell(&self, chunk: &CsrCellChunk) -> Result<Vec<u8>, ScDataErrors> {
        if self.is_v4() {
            chunk.to_v4_bytes(self.target_size, self.codec)
        } else {
            Ok(chunk.to_compressed_bytes()?)
        }
    }

    /// Encode a gene chunk for a file of this format.
    ///
    /// ### Params
    ///
    /// * `chunk` - The gene
    /// * `cell_lib` - Library size per cell as `f32`, the table the file will
    ///   carry. Empty stores the norm layer verbatim.
    ///
    /// ### Returns
    ///
    /// The payload the writer's `write_compressed_gene_chunks` expects.
    pub fn encode_gene(
        &self,
        chunk: &CscGeneChunk,
        cell_lib: &[f32],
    ) -> Result<Vec<u8>, ScDataErrors> {
        if self.is_v4() {
            chunk.to_v4_bytes(cell_lib, self.target_size, self.codec)
        } else {
            Ok(chunk.to_compressed_bytes()?)
        }
    }
}

////////////////
// Converters //
////////////////

/// Inverse of the index map: original index per chunk, in file order.
///
/// ### Params
///
/// * `header` - The store header
///
/// ### Returns
///
/// `order[chunk] = original index`, or [`ScDataErrors::V4ChunkCorrupt`] for a
/// map that points outside the file.
fn chunk_order(header: &SparseDataHeader) -> Result<Vec<usize>, ScDataErrors> {
    let mut order = vec![usize::MAX; header.no_chunks];
    for (&orig, &chunk) in &header.index_map {
        *order.get_mut(chunk).ok_or_else(|| {
            ScDataErrors::V4ChunkCorrupt(format!("chunk {chunk} outside the file"))
        })? = orig;
    }
    Ok(order)
}

/// Refuse anything but a v3 input.
///
/// ### Params
///
/// * `reader` - The opened input
///
/// ### Returns
///
/// `Ok(())` for v3, [`ScDataErrors::FileVersionMismatch`] otherwise.
fn require_v3(reader: &ParallelSparseReader) -> Result<(), ScDataErrors> {
    if reader.file_version() != SC_FILE_VERSION_V3 {
        return Err(ScDataErrors::FileVersionMismatch {
            expected: SC_FILE_VERSION_V3,
            found: reader.file_version(),
        });
    }
    Ok(())
}

/// Convert a v3 cell-based file to v4.
///
/// Streams in batches of [`CONVERT_CELL_BATCH`] chunks, in file order, so the
/// index map of the output equals the input's.
///
/// ### Params
///
/// * `cell_in` - Path to the v3 cell-based file
/// * `cell_out` - Path of the v4 file to write
/// * `codec` - Body codec of the output
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Library sizes indexed by original cell index, `total_cells` long at least,
/// which [`convert_gene_file_to_v4`] needs to drop the gene-file norm.
pub fn convert_cell_file_to_v4(
    cell_in: &str,
    cell_out: &str,
    codec: ChunkCodec,
    verbose: bool,
) -> Result<Vec<u64>, ScDataErrors> {
    let start = Instant::now();
    let reader = ParallelSparseReader::new(cell_in)?;
    require_v3(&reader)?;
    if !reader.is_cell_based() {
        return Err(ScDataErrors::ReaderModeMismatch {
            actual: "gene-based",
            requested: "cell-based",
        });
    }
    let header = reader.get_header();
    let target_size = reader.target_size().unwrap_or(0.0);
    let order = chunk_order(&header)?;

    let mut writer = CellGeneSparseWriter::new(
        cell_out,
        true,
        header.total_cells,
        header.total_genes,
        target_size,
    )?;

    let table_len = header
        .total_cells
        .max(order.iter().copied().max().map_or(0, |m| m + 1));
    let mut lib_table = vec![0u64; table_len];
    let mut stored_norm = 0usize;

    for batch in order.chunks(CONVERT_CELL_BATCH) {
        let cells = reader.read_cells_parallel(batch)?;
        let payloads = cells
            .par_iter()
            .map(|cell| {
                Ok(CellPayload {
                    original_index: cell.original_index,
                    library_size: cell.library_size,
                    bytes: cell.to_v4_bytes(target_size, codec)?,
                })
            })
            .collect::<Result<Vec<_>, ScDataErrors>>()?;
        for p in &payloads {
            lib_table[p.original_index] = p.library_size as u64;
            stored_norm += (V4Header::parse(&p.bytes)?.flags & FLAG_NORM_STORED != 0) as usize;
        }
        writer.write_compressed_cell_chunks(&payloads)?;
    }
    writer.finalise()?;

    if verbose {
        println!(
            "  Converted {} cells to v4 ({} with stored norm) in {:.2?}.",
            header.no_chunks.separate_with_underscores(),
            stored_norm.separate_with_underscores(),
            start.elapsed()
        );
    }

    Ok(lib_table)
}

/// Convert a v3 gene-based file to v4.
///
/// Streams in batches of [`CONVERT_GENE_BATCH`] chunks, in file order. The
/// output carries `cell_lib` as its library-size table.
///
/// ### Params
///
/// * `gene_in` - Path to the v3 gene-based file
/// * `gene_out` - Path of the v4 file to write
/// * `cell_lib` - Library sizes indexed by original cell index, as returned by
///   [`convert_cell_file_to_v4`]
/// * `codec` - Body codec of the output
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once the output is finalised.
pub fn convert_gene_file_to_v4(
    gene_in: &str,
    gene_out: &str,
    cell_lib: &[u64],
    codec: ChunkCodec,
    verbose: bool,
) -> Result<(), ScDataErrors> {
    let start = Instant::now();
    let reader = ParallelSparseReader::new(gene_in)?;
    require_v3(&reader)?;
    if !reader.is_gene_based() {
        return Err(ScDataErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }
    let header = reader.get_header();
    let target_size = reader.target_size().unwrap_or(0.0);
    let order = chunk_order(&header)?;
    let lib_f32: Vec<f32> = cell_lib.iter().map(|&l| l as f32).collect();

    let mut writer = CellGeneSparseWriter::new(
        gene_out,
        false,
        header.total_cells,
        header.total_genes,
        target_size,
    )?;

    let mut stored_norm = 0usize;
    for batch in order.chunks(CONVERT_GENE_BATCH) {
        let genes = reader.read_gene_parallel(batch)?;
        let payloads = genes
            .par_iter()
            .map(|gene| {
                Ok((
                    gene.original_index,
                    gene.to_v4_bytes(&lib_f32, target_size, codec)?,
                ))
            })
            .collect::<Result<Vec<_>, ScDataErrors>>()?;
        for (_, bytes) in &payloads {
            stored_norm += (V4Header::parse(bytes)?.flags & FLAG_NORM_STORED != 0) as usize;
        }
        writer.write_compressed_gene_chunks(&payloads)?;
    }
    writer.set_cell_library_table(cell_lib.to_vec())?;
    writer.finalise()?;

    if verbose {
        println!(
            "  Converted {} genes to v4 ({} with stored norm) in {:.2?}.",
            header.no_chunks.separate_with_underscores(),
            stored_norm.separate_with_underscores(),
            start.elapsed()
        );
    }

    Ok(())
}

/// Convert a v3 store to v4: the cell file, and the gene file when given.
///
/// Inputs are never modified. Neither output may equal an input path.
///
/// ### Params
///
/// * `cell_in` - Path to the v3 cell-based file
/// * `gene_in` - Path to the v3 gene-based file, if any
/// * `cell_out` - Path of the v4 cell file
/// * `gene_out` - Path of the v4 gene file, required with `gene_in`
/// * `codec` - Body codec of both outputs
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once both outputs are finalised.
pub fn convert_store_v3_to_v4(
    cell_in: &str,
    gene_in: Option<&str>,
    cell_out: &str,
    gene_out: Option<&str>,
    codec: ChunkCodec,
    verbose: bool,
) -> Result<(), ScDataErrors> {
    let lib = convert_cell_file_to_v4(cell_in, cell_out, codec, verbose)?;
    if let (Some(gene_in), Some(gene_out)) = (gene_in, gene_out) {
        convert_gene_file_to_v4(gene_in, gene_out, &lib, codec, verbose)?;
    }
    Ok(())
}

/// Replace a v3 store by its v4 conversion, in place.
///
/// Converts into `<path>.v4.tmp` next to each file first and only touches the
/// originals once every conversion has succeeded, so a failure leaves the
/// store as it was. Each original is then renamed to `<path>.v3.bak` when
/// `keep_backup` is set, and deleted otherwise.
///
/// ### Params
///
/// * `cell_path` - Path to the v3 cell-based file
/// * `gene_path` - Path to the v3 gene-based file, if any
/// * `codec` - Body codec of the converted files
/// * `keep_backup` - Keep the v3 originals as `<path>.v3.bak`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once the converted files are in place.
pub fn upgrade_store_to_v4(
    cell_path: &str,
    gene_path: Option<&str>,
    codec: ChunkCodec,
    keep_backup: bool,
    verbose: bool,
) -> Result<(), ScDataErrors> {
    let tmp = |p: &str| format!("{p}.v4.tmp");
    let cleanup = |paths: &[Option<&str>]| {
        for p in paths.iter().flatten() {
            let _ = std::fs::remove_file(tmp(p));
        }
    };

    let result = (|| {
        let cell_tmp = tmp(cell_path);
        let lib = convert_cell_file_to_v4(cell_path, &cell_tmp, codec, verbose)?;
        if let Some(gene_path) = gene_path {
            convert_gene_file_to_v4(gene_path, &tmp(gene_path), &lib, codec, verbose)?;
        }
        // the converted files must open and agree with the originals
        for p in std::iter::once(cell_path).chain(gene_path) {
            let (old, new) = (
                ParallelSparseReader::new(p)?.get_header(),
                ParallelSparseReader::new(&tmp(p))?.get_header(),
            );
            if (old.total_cells, old.total_genes, old.no_chunks)
                != (new.total_cells, new.total_genes, new.no_chunks)
            {
                return Err(ScDataErrors::V4ChunkCorrupt(format!(
                    "converted {p} does not match the original"
                )));
            }
        }
        Ok(())
    })();
    if let Err(e) = result {
        cleanup(&[Some(cell_path), gene_path]);
        return Err(e);
    }

    for p in std::iter::once(cell_path).chain(gene_path) {
        if keep_backup {
            std::fs::rename(p, format!("{p}.v3.bak"))?;
        }
        std::fs::rename(tmp(p), p)?;
    }
    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexSet;

    #[test]
    fn test_pack_unpack_round_trip_widths() {
        for &(n, max) in &[
            (0usize, 0u32),
            (1, 5),
            (127, 1),
            (128, 255),
            (129, 70_000),
            (300, 0),
            (500, u32::MAX),
        ] {
            let vals: Vec<u32> = (0..n)
                .map(|i| ((i as u64 * 2_654_435_761) % (max as u64 + 1)) as u32)
                .collect();
            let mut buf = Vec::new();
            pack(&vals, &mut buf);
            let mut out = Vec::new();
            let used = unpack(&buf, n, &mut out).unwrap();
            assert_eq!(out, vals, "n {n} max {max}");
            assert_eq!(used, buf.len());
            assert_eq!(packed_len(&buf, n).unwrap(), buf.len());
        }
    }

    #[test]
    fn test_unpack_every_width_through_the_specialised_path() {
        for width in 1..=32u32 {
            let max = if width == 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            let vals: Vec<u32> = (0..PACK_BLOCK * 3 + 5)
                .map(|i| {
                    if i % 128 == 0 {
                        max
                    } else {
                        (i as u32).wrapping_mul(2_654_435_761) & max
                    }
                })
                .collect();
            let mut buf = Vec::new();
            pack(&vals, &mut buf);
            let mut out = Vec::new();
            assert_eq!(unpack(&buf, vals.len(), &mut out).unwrap(), buf.len());
            assert_eq!(out, vals, "width {width}");
        }
    }

    #[test]
    fn test_unpack_rejects_truncated_and_wide() {
        let mut buf = Vec::new();
        pack(&[1, 2, 3, 400], &mut buf);
        let mut out = Vec::new();
        assert!(unpack(&buf[..buf.len() - 1], 4, &mut out).is_err());
        assert!(unpack(&[33], 1, &mut out).is_err());
    }

    /// Removes the files of a test on drop.
    struct Scratch(Vec<std::path::PathBuf>);

    impl Scratch {
        fn new(tag: &str, n: usize) -> Self {
            let dir = std::env::temp_dir();
            Self(
                (0..n)
                    .map(|i| dir.join(format!("bx_v4_{tag}_{}_{i}.bin", std::process::id())))
                    .collect(),
            )
        }

        fn path(&self, i: usize) -> &str {
            self.0[i].to_str().unwrap()
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            for p in &self.0 {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    const N_CELLS: usize = 300;
    const N_GENES: usize = 40;
    const TARGET: f32 = 1e4;

    /// Cells with a spread of counts, one empty cell, one with a count above
    /// every small width, and one whose stored norm does not recompute.
    fn synthetic_cells() -> Vec<CsrCellChunk> {
        let mut state = 12345u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        (0..N_CELLS)
            .map(|c| {
                let mut genes = Vec::new();
                let mut counts = Vec::new();
                if c != 7 {
                    for g in 0..N_GENES as u32 {
                        if next() % 3 == 0 {
                            genes.push(g);
                            counts.push(1 + next() % 4 + if next() % 50 == 0 { 300 } else { 0 });
                        }
                    }
                }
                if c == 11 {
                    match genes.last() {
                        Some(&g) if g == N_GENES as u32 - 1 => *counts.last_mut().unwrap() = 70_000,
                        _ => {
                            genes.push(N_GENES as u32 - 1);
                            counts.push(70_000);
                        }
                    }
                }
                let mut chunk = CsrCellChunk::from_data(&counts, &genes, c, TARGET, true);
                if c == 13 && !chunk.data_norm.is_empty() {
                    chunk.data_norm[0] = F16::from_f32(chunk.data_norm[0].to_f32() + 0.5);
                }
                chunk
            })
            .collect()
    }

    fn write_v3_store(cell_path: &str, gene_path: &str, target: f32) {
        let mut writer =
            CellGeneSparseWriter::new_v3(cell_path, true, N_CELLS, N_GENES, target).unwrap();
        for chunk in synthetic_cells() {
            writer.write_cell_chunk(chunk).unwrap();
        }
        writer.finalise().unwrap();
        crate::gene_file_io::write_gene_file_v3(cell_path, gene_path, Some(500), false).unwrap();
    }

    fn assert_cells_equal(a: &CsrCellChunk, b: &CsrCellChunk) {
        assert_eq!(a.original_index, b.original_index);
        assert_eq!(a.library_size, b.library_size);
        assert_eq!(a.to_keep, b.to_keep);
        assert_eq!(a.indices, b.indices);
        assert_eq!(
            a.data_raw.iter().collect::<Vec<_>>(),
            b.data_raw.iter().collect::<Vec<_>>()
        );
        let bits = |c: &CsrCellChunk| c.data_norm.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(a), bits(b));
    }

    fn assert_genes_equal(a: &CscGeneChunk, b: &CscGeneChunk) {
        assert_eq!(a.original_index, b.original_index);
        assert_eq!(a.nnz, b.nnz);
        assert_eq!(a.avg_exp.to_bits(), b.avg_exp.to_bits());
        assert_eq!(a.indices, b.indices);
        assert_eq!(
            a.data_raw.iter().collect::<Vec<_>>(),
            b.data_raw.iter().collect::<Vec<_>>()
        );
        let bits = |c: &CscGeneChunk| c.data_norm.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(a), bits(b));
    }

    fn assert_sparse_equal(
        a: &CompressedSparseData2<f32, f32>,
        b: &CompressedSparseData2<f32, f32>,
    ) {
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
        assert_eq!(a.data_2, b.data_2);
    }

    /// Every read path of a converted store matches the v3 original, for each
    /// codec and with a zero target size (norm always stored).
    fn check_conversion(codec: ChunkCodec, target: f32) {
        let tag = format!("{codec:?}_{}", target as u32);
        let s = Scratch::new(&tag, 4);
        write_v3_store(s.path(0), s.path(1), target);
        convert_store_v3_to_v4(
            s.path(0),
            Some(s.path(1)),
            s.path(2),
            Some(s.path(3)),
            codec,
            false,
        )
        .unwrap();

        let (cells3, cells4) = (
            ParallelSparseReader::new(s.path(0)).unwrap(),
            ParallelSparseReader::new(s.path(2)).unwrap(),
        );
        let (genes3, genes4) = (
            ParallelSparseReader::new(s.path(1)).unwrap(),
            ParallelSparseReader::new(s.path(3)).unwrap(),
        );
        assert_eq!(cells3.file_version(), SC_FILE_VERSION_V3);
        assert_eq!(cells4.file_version(), SC_FILE_VERSION);
        assert_eq!(genes4.file_version(), SC_FILE_VERSION);
        assert_eq!(cells4.target_size(), cells3.target_size());

        let all_cells: Vec<usize> = (0..N_CELLS).collect();
        let all_genes: Vec<usize> = (0..N_GENES).collect();
        for (a, b) in cells3
            .read_cells_parallel(&all_cells)
            .unwrap()
            .iter()
            .zip(&cells4.read_cells_parallel(&all_cells).unwrap())
        {
            assert_cells_equal(a, b);
        }
        for (a, b) in genes3
            .read_gene_parallel(&all_genes)
            .unwrap()
            .iter()
            .zip(&genes4.read_gene_parallel(&all_genes).unwrap())
        {
            assert_genes_equal(a, b);
        }

        let shuffled: Vec<usize> = (0..N_CELLS).rev().step_by(3).collect();
        let keep: IndexSet<u32> = (0..N_CELLS as u32).filter(|c| c % 4 != 1).collect();
        for layer in [
            DataLayerReturn::Raw,
            DataLayerReturn::Norm,
            DataLayerReturn::BothLayers,
        ] {
            assert_sparse_equal(
                &cells3.read_cells_sparse::<f32>(&shuffled, &layer).unwrap(),
                &cells4.read_cells_sparse::<f32>(&shuffled, &layer).unwrap(),
            );
            for subset in [None, Some(&keep)] {
                assert_sparse_equal(
                    &genes3
                        .read_genes_sparse::<f32>(&all_genes, subset, &layer)
                        .unwrap(),
                    &genes4
                        .read_genes_sparse::<f32>(&all_genes, subset, &layer)
                        .unwrap(),
                );
            }
        }

        for (a, b) in genes3
            .read_gene_parallel_filtered(&all_genes, &keep)
            .unwrap()
            .iter()
            .zip(
                &genes4
                    .read_gene_parallel_filtered(&all_genes, &keep)
                    .unwrap(),
            )
        {
            assert_genes_equal(a, b);
        }
        assert_eq!(
            cells3.read_cell_library_sizes(&all_cells).unwrap(),
            cells4.read_cell_library_sizes(&all_cells).unwrap()
        );
        assert_eq!(
            genes3.read_gene_nnz(&all_genes).unwrap(),
            genes4.read_gene_nnz(&all_genes).unwrap()
        );
        assert_eq!(
            cells3.get_header().chunk_offsets.len(),
            cells4.get_header().chunk_offsets.len()
        );
    }

    #[test]
    fn test_v3_to_v4_round_trip_none() {
        check_conversion(ChunkCodec::None, TARGET);
    }

    #[test]
    fn test_v3_to_v4_round_trip_lz4() {
        check_conversion(ChunkCodec::Lz4, TARGET);
    }

    #[test]
    fn test_v3_to_v4_round_trip_zstd() {
        check_conversion(ChunkCodec::Zstd, TARGET);
    }

    #[test]
    fn test_v3_to_v4_round_trip_unknown_target_stores_norm() {
        check_conversion(ChunkCodec::Lz4, 0.0);
    }

    #[test]
    fn test_v4_drops_norm_and_shrinks() {
        let s = Scratch::new("shrink", 4);
        write_v3_store(s.path(0), s.path(1), TARGET);
        convert_store_v3_to_v4(
            s.path(0),
            Some(s.path(1)),
            s.path(2),
            Some(s.path(3)),
            ChunkCodec::Lz4,
            false,
        )
        .unwrap();
        let size = |i: usize| std::fs::metadata(s.path(i)).unwrap().len();
        assert!(size(2) < size(0), "cell file {} vs {}", size(2), size(0));
        assert!(size(3) < size(1), "gene file {} vs {}", size(3), size(1));
    }

    #[test]
    fn test_v4_chunk_keeps_unsorted_and_duplicate_indices() {
        let chunk = CsrCellChunk::from_data(&[3u32, 1, 2, 5], &[9u32, 2, 2, 400], 5, TARGET, true);
        let bytes = chunk.to_v4_bytes(TARGET, ChunkCodec::Lz4).unwrap();
        let ctx = V4Ctx {
            target_size: TARGET,
            cell_lib: None,
            cell_norm: None,
            is_cell: true,
        };
        let back = v4_cell_chunk(&bytes, &ctx, &mut V4Decoded::default()).unwrap();
        assert_cells_equal(&chunk, &back);
    }

    #[test]
    fn test_default_writer_emits_v4_and_converter_refuses_it() {
        let s = Scratch::new("guards", 2);
        let mut writer = CellGeneSparseWriter::new(s.path(0), true, 1, 3, TARGET).unwrap();
        writer
            .write_cell_chunk(CsrCellChunk::from_data(&[1u32], &[0u32], 0, TARGET, true))
            .unwrap();
        writer.finalise().unwrap();
        assert_eq!(peek_file_version(s.path(0)).unwrap(), SC_FILE_VERSION);
        assert!(matches!(
            convert_cell_file_to_v4(s.path(0), s.path(1), ChunkCodec::Lz4, false),
            Err(ScDataErrors::FileVersionMismatch { .. })
        ));
    }

    #[test]
    fn test_peek_file_version_distinguishes_v3_and_v4() {
        let s = Scratch::new("peek", 2);
        let chunk = || CsrCellChunk::from_data(&[1u32, 2], &[0u32, 1], 0, TARGET, true);
        let mut v3 = CellGeneSparseWriter::new_v3(s.path(0), true, 1, 3, TARGET).unwrap();
        v3.write_cell_chunk(chunk()).unwrap();
        v3.finalise().unwrap();
        let mut v4 = CellGeneSparseWriter::new(s.path(1), true, 1, 3, TARGET).unwrap();
        v4.write_cell_chunk(chunk()).unwrap();
        v4.finalise().unwrap();
        assert_eq!(peek_file_version(s.path(0)).unwrap(), SC_FILE_VERSION_V3);
        assert_eq!(peek_file_version(s.path(1)).unwrap(), SC_FILE_VERSION);
    }

    #[test]
    fn test_default_gene_file_is_v4_and_matches_the_v3_one() {
        let s = Scratch::new("genefile", 3);
        write_v3_store(s.path(0), s.path(1), TARGET);
        crate::gene_file_io::write_gene_file(s.path(0), s.path(2), Some(500), false).unwrap();
        assert_eq!(peek_file_version(s.path(2)).unwrap(), SC_FILE_VERSION);

        let (v3, v4) = (
            ParallelSparseReader::new(s.path(1)).unwrap(),
            ParallelSparseReader::new(s.path(2)).unwrap(),
        );
        let all: Vec<usize> = (0..N_GENES).collect();
        for (a, b) in v3
            .read_gene_parallel(&all)
            .unwrap()
            .iter()
            .zip(&v4.read_gene_parallel(&all).unwrap())
        {
            assert_genes_equal(a, b);
        }
    }

    #[test]
    fn test_upgrade_store_replaces_files_and_keeps_backups() {
        let s = Scratch::new("upgrade", 4);
        write_v3_store(s.path(0), s.path(1), TARGET);
        let cells_before = ParallelSparseReader::new(s.path(0))
            .unwrap()
            .get_all_cells()
            .unwrap();
        upgrade_store_to_v4(s.path(0), Some(s.path(1)), ChunkCodec::Lz4, true, false).unwrap();

        assert_eq!(peek_file_version(s.path(0)).unwrap(), SC_FILE_VERSION);
        assert_eq!(peek_file_version(s.path(1)).unwrap(), SC_FILE_VERSION);
        let backup = format!("{}.v3.bak", s.path(0));
        assert_eq!(peek_file_version(&backup).unwrap(), SC_FILE_VERSION_V3);
        for (a, b) in cells_before.iter().zip(
            &ParallelSparseReader::new(s.path(0))
                .unwrap()
                .get_all_cells()
                .unwrap(),
        ) {
            assert_cells_equal(a, b);
        }
        for p in [backup, format!("{}.v3.bak", s.path(1))] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn test_archive_round_trips_a_v4_cell_file() {
        let s = Scratch::new("archive", 6);
        write_v3_store(s.path(0), s.path(1), TARGET);
        convert_cell_file_to_v4(s.path(0), s.path(2), ChunkCodec::Lz4, false).unwrap();
        crate::archive_io::archive_cell_file(s.path(2), s.path(3), 3, false).unwrap();
        crate::archive_io::restore_archive(s.path(3), s.path(4), s.path(5), Some(500), false)
            .unwrap();

        let original = ParallelSparseReader::new(s.path(0)).unwrap();
        let restored = ParallelSparseReader::new(s.path(4)).unwrap();
        let all: Vec<usize> = (0..N_CELLS).collect();
        for (a, b) in original
            .read_cells_parallel(&all)
            .unwrap()
            .iter()
            .zip(&restored.read_cells_parallel(&all).unwrap())
        {
            assert_cells_equal(a, b);
        }
    }
}
