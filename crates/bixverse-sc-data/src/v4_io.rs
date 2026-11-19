//! The v4 chunk encoding, shared by the v4 and v5 containers, and the
//! converters from v3 and v4 to v5.
//!
//! Per chunk:
//!
//! * gene (or cell) indices as delta bit-packed blocks of 128 instead of
//!   absolute `u32`, which lz4 cannot compress;
//! * each raw layer bit-packed the same way (v4 has exactly one, v5 up to
//!   `MAX_LAYERS`);
//! * the normalised layer dropped when it recomputes bit for bit from the raw
//!   counts, library size and target size, stored verbatim otherwise (f16 in
//!   v4, f32 in v5);
//! * an uncompressed fixed header, so nnz and library size are readable
//!   without decompressing, and an optional byte codec over the body.
//!
//! Gene-based files recompute the norm of an entry from the library size of
//! its cell, so they carry a per-cell library-size table in the tail.
//!
//! Chunk layout (all little endian):
//!
//! ```text
//! [0..4)   nnz            u32
//! [4..8)   body_len       u32   decoded body length
//! [8..16)  aux            u64   cell: library size; gene: v4 avg_exp f16
//!                               bits, v5 sum of raw layer 0
//! [16..24) original_index u64
//! [24]     to_keep        u8
//! [25]     flags          u8    bit 0: norm stored, bit 1: layer 0 stored
//!                               minus one
//! [26]     codec          u8    0 none, 1 lz4, 2 zstd
//! [27]     minus_one      u8    bit k - 1: layer k stored minus one (v5)
//! [28..32) reserved
//! ```
//!
//! The body is the packed indices, the packed raw layers in order and, when
//! flagged, the norm layer, then compressed with `codec`. With one layer the
//! body is the same in v4 and v5 unless the norm is stored.

use lz4_flex::block::{compress as lz4_compress, decompress_into as lz4_decompress_into};
use rayon::prelude::*;
use std::time::Instant;
use thousands::Separable;

use crate::data_io::*;
use crate::prelude::*;
use crate::v5_io::{MAX_LAYERS, NormKind};

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
/// trades table size (four bytes per cell per count) against how often the
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

/// Parse a codec name
///
/// ### Params
///
/// * `s` - One of `"lz4"`, `"zstd"` or `"none"`
///
/// ### Returns
///
/// The [ChunkCodec], or `None` for anything else.
pub fn parse_chunk_codec(s: &str) -> Option<ChunkCodec> {
    match s {
        "lz4" => Some(ChunkCodec::Lz4),
        "zstd" => Some(ChunkCodec::Zstd),
        "none" => Some(ChunkCodec::None),
        _ => None,
    }
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
    /// Bit `k`: raw layer `k` is stored minus one. Bit 0 mirrors
    /// [`FLAG_RAW_MINUS_ONE`].
    minus_one: u16,
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
            minus_one: ((chunk[25] & FLAG_RAW_MINUS_ONE != 0) as u16) | ((chunk[27] as u16) << 1),
        })
    }

    /// Whether the chunk stores its normalised layer verbatim, i.e. the layer
    /// did not recompute bit for bit from the raw counts.
    ///
    /// ### Returns
    ///
    /// `true` if [`FLAG_NORM_STORED`] is set.
    pub(crate) fn norm_stored(&self) -> bool {
        self.flags & FLAG_NORM_STORED != 0
    }
}

////////////
// Encode //
////////////

/// Replace an f16 norm layer by its f32 recompute when every entry agrees at
/// f16 precision.
///
/// v3 holds the norm as f16, which never passes the bit-exact f32 check the
/// encoder applies. A v3 layer that is the f16 rounding of [`norm_value`]
/// becomes the f32 value, so the converted chunk drops it. A layer with any
/// other entry is left alone and stored verbatim.
///
/// ### Params
///
/// * `norm` - The layer, as read from v3
/// * `recompute` - [`norm_value`] of entry `k`
fn snap_v3_norm(norm: &mut [f32], recompute: impl Fn(usize) -> f32) {
    let same =
        |k: usize, v: f32| F16::from_f32(recompute(k)).to_bits() == F16::from_f32(v).to_bits();
    if norm.iter().enumerate().all(|(k, &v)| same(k, v)) {
        norm.iter_mut()
            .enumerate()
            .for_each(|(k, v)| *v = recompute(k));
    }
}

/// Everything [`encode_chunk`] needs about one cell or gene.
struct EncodeIn<'a> {
    /// Gene (cell chunks) or cell (gene chunks) indices, ascending.
    indices: &'a [u32],
    /// Raw layers, each as long as `indices`. At least one.
    layers: &'a [&'a [u32]],
    /// Normalised layer, written only when it does not recompute.
    norm: &'a [f32],
    /// The header `aux` field.
    aux: u64,
    /// Original index of the cell or gene.
    original_index: usize,
    /// Keep flag.
    to_keep: bool,
    /// Whether the norm layer recomputes bit for bit and can be dropped.
    norm_derivable: bool,
    /// Store a verbatim norm as f32 (v5) rather than f16 (v4).
    norm_f32: bool,
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
    debug_assert!((1..=MAX_LAYERS).contains(&p.layers.len()));
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
    let mut mask = 0u8;
    let mut body = Vec::with_capacity(n * (1 + p.layers.len()));
    pack(&deltas, &mut body);

    for (l, layer) in p.layers.iter().enumerate() {
        if layer.iter().all(|&x| x >= 1) {
            if l == 0 {
                flags |= FLAG_RAW_MINUS_ONE;
            } else {
                mask |= 1 << (l - 1);
            }
            let minus_one: Vec<u32> = layer.iter().map(|&x| x - 1).collect();
            pack(&minus_one, &mut body);
        } else {
            pack(layer, &mut body);
        }
    }

    if !p.norm_derivable {
        flags |= FLAG_NORM_STORED;
        for &v in p.norm {
            if p.norm_f32 {
                body.extend_from_slice(&v.to_le_bytes());
            } else {
                body.extend_from_slice(&F16::from_f32(v).to_le_bytes());
            }
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
    out.extend_from_slice(&[p.to_keep as u8, flags, used.to_u8(), mask, 0, 0, 0, 0]);
    out.extend_from_slice(&payload);
    out
}

impl CsrCellChunk {
    /// Serialise the cell into a v4-codec chunk.
    ///
    /// The norm layer is dropped when every entry recomputes bit for bit from
    /// the raw count, the library size and the format's target size, and
    /// stored verbatim otherwise.
    ///
    /// ### Params
    ///
    /// * `format` - Format of the file the cell goes into. A target size of
    ///   `0.0` stores the norm.
    ///
    /// ### Returns
    ///
    /// The chunk bytes, or [`ScDataErrors::V4ChunkCorrupt`] for layers of
    /// unequal length.
    pub fn to_v4_bytes(&self, format: &ChunkFormat) -> Result<Vec<u8>, ScDataErrors> {
        let n = self.indices.len();
        if self.data_raw.len() != n || self.data_norm.len() != n {
            return Err(ScDataErrors::V4ChunkCorrupt(format!(
                "cell {} has layers of unequal length",
                self.original_index
            )));
        }
        let raw: Vec<u32> = self.data_raw.iter().collect();
        let lib = self.library_size as f32;
        let target_size = format.target_size;
        let norm_derivable = target_size != 0.0
            && raw
                .iter()
                .zip(&self.data_norm)
                .all(|(&x, &v)| norm_value(x as f32, lib, target_size).to_bits() == v.to_bits());

        Ok(encode_chunk(
            &EncodeIn {
                indices: &self.indices,
                layers: &[&raw],
                norm: &self.data_norm,
                aux: self.library_size as u64,
                original_index: self.original_index,
                to_keep: self.to_keep,
                norm_derivable,
                norm_f32: format.norm_f32(),
            },
            format.codec,
        ))
    }
}

impl CscGeneChunk {
    /// Serialise the gene into a v4-codec chunk.
    ///
    /// The norm layer is dropped when every entry recomputes bit for bit from
    /// the raw count, the library size of its cell and the format's target
    /// size, and stored verbatim otherwise.
    ///
    /// ### Params
    ///
    /// * `cell_lib` - Library size of every cell as `f32`, indexed by the cell
    ///   indices this chunk stores
    /// * `format` - Format of the file the gene goes into. A target size of
    ///   `0.0` stores the norm.
    ///
    /// ### Returns
    ///
    /// The chunk bytes, or [`ScDataErrors::V4ChunkCorrupt`] for layers of
    /// unequal length.
    pub fn to_v4_bytes(
        &self,
        cell_lib: &[f32],
        format: &ChunkFormat,
    ) -> Result<Vec<u8>, ScDataErrors> {
        let n = self.indices.len();
        if self.data_raw.len() != n || self.data_norm.len() != n {
            return Err(ScDataErrors::V4ChunkCorrupt(format!(
                "gene {} has layers of unequal length",
                self.original_index
            )));
        }
        let raw: Vec<u32> = self.data_raw.iter().collect();
        let target_size = format.target_size;
        let norm_derivable = target_size != 0.0
            && self
                .indices
                .iter()
                .zip(&raw)
                .zip(&self.data_norm)
                .all(|((&cell, &x), &v)| {
                    cell_lib.get(cell as usize).is_some_and(|&lib| {
                        norm_value(x as f32, lib, target_size).to_bits() == v.to_bits()
                    })
                });

        Ok(encode_chunk(
            &EncodeIn {
                indices: &self.indices,
                layers: &[&raw],
                norm: &self.data_norm,
                aux: format.gene_aux(self.avg_exp, &raw),
                original_index: self.original_index,
                to_keep: self.to_keep,
                norm_derivable,
                norm_f32: format.norm_f32(),
            },
            format.codec,
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
    /// Per-cell library sizes as `f32`, present on gene-based files.
    pub cell_lib: Option<&'a [f32]>,
    /// Normalised value of small counts per cell, see [`gene_norm_table`].
    /// Only valid without `weights`.
    pub cell_norm: Option<&'a [f32]>,
    /// Whether the chunk is a cell (CSR) chunk.
    pub is_cell: bool,
    /// Raw layers per chunk, 1 for v4.
    pub n_layers: usize,
    /// The raw layer reads return, and the norm is derived from.
    pub layer: usize,
    /// Whether a verbatim norm is f32 (v5) rather than f16 (v4).
    pub norm_f32: bool,
    /// How the norm derives from the raw counts.
    pub norm_kind: NormKind,
    /// Per-feature weight `w_f` of the norm (the IDF of [`NormKind::TfIdf`]),
    /// `None` for `w_f = 1`.
    pub weights: Option<&'a [f32]>,
}

impl V4Ctx<'_> {
    /// Whether the store has a normalised layer to derive.
    ///
    /// ### Returns
    ///
    /// `false` for [`NormKind::None`] and for TF-IDF without an IDF table.
    pub(crate) fn norm_available(&self) -> bool {
        match self.norm_kind {
            NormKind::None => false,
            NormKind::LogCpm => true,
            NormKind::TfIdf => self.weights.is_some(),
        }
    }
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
    if want_norm && !norm_stored {
        match (ctx.norm_kind, ctx.weights) {
            (NormKind::None, _) => {
                return Err(ScDataErrors::NormNotAvailable("norm kind none".into()));
            }
            (NormKind::TfIdf, None) => {
                return Err(ScDataErrors::NormNotAvailable(
                    "tf_idf without an IDF table".into(),
                ));
            }
            _ => {}
        }
    }
    let need_raw = want_raw || (want_norm && !norm_stored);
    let rest = &body[used..];

    // walk the layers: decode the selected one, skip the ones before it, and
    // the ones after it only when the stored norm behind them is wanted
    let mut pos = 0usize;
    for l in 0..ctx.n_layers {
        let from = rest.get(pos..).ok_or_else(truncated)?;
        if need_raw && l == ctx.layer {
            raw.clear();
            pos += unpack(from, n, raw)?;
            if (h.minus_one >> l) & 1 == 1 {
                raw.iter_mut().for_each(|x| *x = x.wrapping_add(1));
            }
        } else if (need_raw && l < ctx.layer) || (want_norm && norm_stored) {
            pos += packed_len(from, n)?;
        } else {
            break;
        }
    }

    let Some(norm) = norm else {
        return Ok(h);
    };

    norm.reserve(n);
    if norm_stored {
        if ctx.norm_f32 {
            let stored_norm = rest.get(pos..pos + n * 4).ok_or_else(truncated)?;
            norm.extend(
                stored_norm
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| f32::from_le_bytes(c)),
            );
        } else {
            let stored_norm = rest.get(pos..pos + n * 2).ok_or_else(truncated)?;
            norm.extend(
                stored_norm
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&c| F16::from_le_bytes(c).to_f32()),
            );
        }
    } else if ctx.is_cell {
        let lib = h.aux as f32;
        if let Some(w) = ctx.weights {
            // TF-IDF: the target varies per gene, so no per-count table
            let mut outside = None;
            norm.extend(idx[start..].iter().zip(raw.iter()).map(
                |(&g, &x)| match w.get(g as usize) {
                    Some(&wg) => norm_value(x as f32, lib, ctx.target_size * wg),
                    None => {
                        outside = Some(g);
                        0.0
                    }
                },
            ));
            if let Some(g) = outside {
                return Err(ScDataErrors::V4ChunkCorrupt(format!(
                    "feature {g} outside the IDF table"
                )));
            }
        } else {
            let top = raw.iter().copied().max().unwrap_or(0) as usize;
            let lut: Vec<f32> = (0..=top.min(NORM_LUT_MAX))
                .map(|x| norm_value(x as f32, lib, ctx.target_size))
                .collect();
            norm.extend(raw.iter().map(|&x| match lut.get(x as usize) {
                Some(&v) => v,
                None => norm_value(x as f32, lib, ctx.target_size),
            }));
        }
    } else if let Some(w) = ctx.weights {
        let cell_lib = match ctx.cell_lib {
            Some(table) => table,
            None if n == 0 => &[],
            None => return Err(ScDataErrors::V4MissingCellLibraryTable),
        };
        let target = w
            .get(h.original_index)
            .map(|&wg| ctx.target_size * wg)
            .ok_or_else(|| {
                ScDataErrors::V4ChunkCorrupt(format!(
                    "feature {} outside the IDF table",
                    h.original_index
                ))
            })?;
        let mut outside = None;
        norm.extend(idx[start..].iter().zip(raw.iter()).map(|(&cell, &x)| {
            match cell_lib.get(cell as usize) {
                Some(&lib) => norm_value(x as f32, lib, target),
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
                && let Some(&v) = table.get(cell as usize * GENE_NORM_LUT_COUNTS + slot)
            {
                return v;
            }
            match cell_lib.get(cell as usize) {
                Some(&lib) => norm_value(x as f32, lib, ctx.target_size),
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
/// the same [`norm_value`] the fallback calls, so values stay bit identical.
///
/// ### Params
///
/// * `cell_lib` - Library size per cell as `f32`
/// * `target_size` - Target size from the file header
///
/// ### Returns
///
/// The values, `cell_lib.len() * GENE_NORM_LUT_COUNTS` long.
pub(crate) fn gene_norm_table(cell_lib: &[f32], target_size: f32) -> Vec<f32> {
    let mut table = vec![0.0f32; cell_lib.len() * GENE_NORM_LUT_COUNTS];
    table
        .par_chunks_mut(GENE_NORM_LUT_COUNTS)
        .zip(cell_lib.par_iter())
        .for_each(|(row, &lib)| {
            for (k, v) in row.iter_mut().enumerate() {
                *v = norm_value((k + 1) as f32, lib, target_size);
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
/// The cell, with the selected raw layer and the norm, or an empty norm on a
/// store without one (see [`V4Ctx::norm_available`]).
pub(crate) fn v4_cell_chunk(
    chunk: &[u8],
    ctx: &V4Ctx,
    s: &mut V4Decoded,
) -> Result<CsrCellChunk, ScDataErrors> {
    let h = decode_v4(chunk, ctx, true, ctx.norm_available(), s)?;
    Ok(CsrCellChunk {
        data_raw: RawCounts::from_u32_auto(&s.raw),
        data_norm: s.norm.clone(),
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
/// The gene, with the selected raw layer and the norm, or an empty norm on a
/// store without one. v5 keeps the feature total in the header, so `avg_exp`
/// is summed from the decoded norm there.
pub(crate) fn v4_gene_chunk(
    chunk: &[u8],
    ctx: &V4Ctx,
    s: &mut V4Decoded,
) -> Result<CscGeneChunk, ScDataErrors> {
    let h = decode_v4(chunk, ctx, true, ctx.norm_available(), s)?;
    let avg_exp = if ctx.norm_f32 {
        F16::from_f32(s.norm.iter().sum::<f32>())
    } else {
        F16::from_bits(h.aux as u16)
    };
    Ok(CscGeneChunk {
        data_raw: RawCounts::from_u32_auto(&s.raw),
        data_norm: s.norm.clone(),
        avg_exp,
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
    /// File version: [`SC_FILE_VERSION`], [`SC_FILE_VERSION_V4`] or
    /// [`SC_FILE_VERSION_V3`].
    pub version: u32,
    /// Body codec of v4-codec chunks. Unused by v3.
    pub codec: ChunkCodec,
    /// Target size in the file header, which readers recompute the norm layer
    /// with. `0.0` stores the norm layer verbatim.
    pub target_size: f32,
}

impl ChunkFormat {
    /// Whether chunks use the v4 codec, i.e. the file is v4 or v5.
    ///
    /// ### Returns
    ///
    /// `false` only for v3.
    pub fn uses_v4_codec(&self) -> bool {
        self.version >= SC_FILE_VERSION_V4
    }

    /// Whether a verbatim norm is stored as f32 (v5) rather than f16 (v4).
    ///
    /// ### Returns
    ///
    /// `true` for v5.
    pub(crate) fn norm_f32(&self) -> bool {
        self.version >= SC_FILE_VERSION
    }

    /// The `aux` header field of a gene chunk.
    ///
    /// ### Params
    ///
    /// * `avg_exp` - Sum of the normalised layer, what v4 stores
    /// * `raw` - Raw layer 0, whose sum v5 stores
    ///
    /// ### Returns
    ///
    /// The field.
    pub(crate) fn gene_aux(&self, avg_exp: F16, raw: &[u32]) -> u64 {
        if self.norm_f32() {
            raw.iter().map(|&x| x as u64).sum()
        } else {
            avg_exp.to_bits() as u64
        }
    }

    /// Encode a chunk with several raw layers and a derived (or no) norm.
    ///
    /// v5 only. `aux` is the sum of layer 0: the library size of a cell, the
    /// feature total of a gene.
    ///
    /// ### Params
    ///
    /// * `indices` - Gene (cell chunk) or cell (gene chunk) indices, ascending
    /// * `layers` - Raw layers, each as long as `indices`, 1 to
    ///   [`MAX_LAYERS`]
    /// * `original_index` - Index of the cell or gene
    /// * `to_keep` - Keep flag
    ///
    /// ### Returns
    ///
    /// The payload, or [`ScDataErrors::V4ChunkCorrupt`] for a layer count out
    /// of range, layers of unequal length or a pre-v5 format.
    pub fn encode_layers(
        &self,
        indices: &[u32],
        layers: &[&[u32]],
        original_index: usize,
        to_keep: bool,
    ) -> Result<Vec<u8>, ScDataErrors> {
        if !self.norm_f32() {
            return Err(ScDataErrors::V4ChunkCorrupt(
                "raw layers need a v5 store".into(),
            ));
        }
        if !(1..=MAX_LAYERS).contains(&layers.len())
            || layers.iter().any(|l| l.len() != indices.len())
        {
            return Err(ScDataErrors::V4ChunkCorrupt(format!(
                "chunk {original_index}: {} layers, or of unequal length",
                layers.len()
            )));
        }
        Ok(encode_chunk(
            &EncodeIn {
                indices,
                layers,
                norm: &[],
                aux: layers[0].iter().map(|&x| x as u64).sum(),
                original_index,
                to_keep,
                norm_derivable: true,
                norm_f32: true,
            },
            self.codec,
        ))
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
        if self.uses_v4_codec() {
            chunk.to_v4_bytes(self)
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
        if self.uses_v4_codec() {
            chunk.to_v4_bytes(cell_lib, self)
        } else {
            Ok(chunk.to_compressed_bytes()?)
        }
    }

    /// Encode a v4 gene chunk whose norm layer is known to recompute.
    ///
    /// Same bytes as [`Self::encode_gene`] on a kept gene whose every entry
    /// passes the derivability check, without carrying or re-checking the
    /// layer. The caller guarantees that, e.g. by deriving `avg_exp` from the
    /// same [`norm_value`] the reader recomputes with.
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices, ascending
    /// * `raw` - Raw counts, same length
    /// * `avg_exp` - Sum of the normalised layer, the header `aux`
    /// * `original_index` - Gene index
    ///
    /// ### Returns
    ///
    /// The payload the writer's `write_compressed_gene_chunks` expects.
    pub(crate) fn encode_gene_derivable(
        &self,
        indices: &[u32],
        raw: &[u32],
        avg_exp: F16,
        original_index: usize,
    ) -> Vec<u8> {
        debug_assert!(self.uses_v4_codec() && indices.len() == raw.len());
        encode_chunk(
            &EncodeIn {
                indices,
                layers: &[raw],
                norm: &[],
                aux: self.gene_aux(avg_exp, raw),
                original_index,
                to_keep: true,
                norm_derivable: true,
                norm_f32: self.norm_f32(),
            },
            self.codec,
        )
    }
}

////////////////
// Converters //
////////////////

/// Refuse anything but a v3 or v4 input.
///
/// ### Params
///
/// * `reader` - The opened input
///
/// ### Returns
///
/// `Ok(())` for v3 and v4, [`ScDataErrors::FileVersionMismatch`] otherwise.
fn require_pre_v5(reader: &ParallelSparseReader) -> Result<(), ScDataErrors> {
    match reader.file_version() {
        SC_FILE_VERSION_V3 | SC_FILE_VERSION_V4 => Ok(()),
        found => Err(ScDataErrors::FileVersionMismatch {
            expected: SC_FILE_VERSION_V4,
            found,
        }),
    }
}

/// Convert a v3 or v4 cell-based file to v5.
///
/// Streams in batches of [`CONVERT_CELL_BATCH`] chunks, in file order, so the
/// original index of every chunk is kept. A v3 norm that is the f16 rounding
/// of the recompute is dropped (see [`snap_v3_norm`]); a verbatim v4 norm
/// widens to f32 exactly.
///
/// ### Params
///
/// * `cell_in` - Path to the v3 or v4 cell-based file
/// * `cell_out` - Path of the v5 file to write
/// * `codec` - Body codec of the output
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Library sizes indexed by original cell index, `total_cells` long at least,
/// which [`convert_gene_file_to_v5`] needs to drop the gene-file norm.
pub fn convert_cell_file_to_v5(
    cell_in: &str,
    cell_out: &str,
    codec: ChunkCodec,
    verbose: bool,
) -> Result<Vec<u64>, ScDataErrors> {
    let start = Instant::now();
    let reader = ParallelSparseReader::open_any(cell_in)?;
    require_pre_v5(&reader)?;
    if !reader.is_cell_based() {
        return Err(ScDataErrors::ReaderModeMismatch {
            actual: "gene-based",
            requested: "cell-based",
        });
    }
    let dims = reader.dims();
    let from_v3 = reader.file_version() == SC_FILE_VERSION_V3;
    let target_size = reader.target_size().unwrap_or(0.0);
    let order = reader.chunk_order()?;

    let writer = CellGeneSparseWriter::new(
        cell_out,
        true,
        dims.total_cells,
        dims.total_genes,
        target_size,
    )?;
    let mut writer = writer.with_codec(codec);
    let format = writer.format();

    let table_len = dims
        .total_cells
        .max(order.iter().copied().max().map_or(0, |m| m + 1));
    let mut lib_table = vec![0u64; table_len];
    let mut stored_norm = 0usize;

    for batch in order.chunks(CONVERT_CELL_BATCH) {
        let cells = reader.read_cells_parallel(batch)?;
        let payloads = cells
            .into_par_iter()
            .map(|mut cell| {
                if from_v3 {
                    let lib = cell.library_size as f32;
                    let raw: Vec<u32> = cell.data_raw.iter().collect();
                    snap_v3_norm(&mut cell.data_norm, |k| {
                        norm_value(raw[k] as f32, lib, target_size)
                    });
                }
                Ok(CellPayload {
                    original_index: cell.original_index,
                    library_size: cell.library_size,
                    bytes: cell.to_v4_bytes(&format)?,
                })
            })
            .collect::<Result<Vec<_>, ScDataErrors>>()?;
        for p in &payloads {
            lib_table[p.original_index] = p.library_size as u64;
            stored_norm += V4Header::parse(&p.bytes)?.norm_stored() as usize;
        }
        writer.write_compressed_cell_chunks(&payloads)?;
    }
    writer.finalise()?;

    if verbose {
        println!(
            "  Converted {} cells to v5 ({} with stored norm) in {:.2?}.",
            order.len().separate_with_underscores(),
            stored_norm.separate_with_underscores(),
            start.elapsed()
        );
    }

    Ok(lib_table)
}

/// Convert a v3 or v4 gene-based file to v5.
///
/// Streams in batches of [`CONVERT_GENE_BATCH`] chunks, in file order. The
/// output carries `cell_lib` as its library-size table.
///
/// ### Params
///
/// * `gene_in` - Path to the v3 or v4 gene-based file
/// * `gene_out` - Path of the v5 file to write
/// * `cell_lib` - Library sizes indexed by original cell index, as returned by
///   [`convert_cell_file_to_v5`]
/// * `codec` - Body codec of the output
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once the output is finalised.
pub fn convert_gene_file_to_v5(
    gene_in: &str,
    gene_out: &str,
    cell_lib: &[u64],
    codec: ChunkCodec,
    verbose: bool,
) -> Result<(), ScDataErrors> {
    let start = Instant::now();
    let reader = ParallelSparseReader::open_any(gene_in)?;
    require_pre_v5(&reader)?;
    if !reader.is_gene_based() {
        return Err(ScDataErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }
    let dims = reader.dims();
    let from_v3 = reader.file_version() == SC_FILE_VERSION_V3;
    let target_size = reader.target_size().unwrap_or(0.0);
    let order = reader.chunk_order()?;
    let lib_f32: Vec<f32> = cell_lib.iter().map(|&l| l as f32).collect();

    let writer = CellGeneSparseWriter::new(
        gene_out,
        false,
        dims.total_cells,
        dims.total_genes,
        target_size,
    )?;
    let mut writer = writer.with_codec(codec);
    let format = writer.format();

    let mut stored_norm = 0usize;
    for batch in order.chunks(CONVERT_GENE_BATCH) {
        let genes = reader.read_gene_parallel(batch)?;
        let payloads = genes
            .into_par_iter()
            .map(|mut gene| {
                if from_v3 {
                    let raw: Vec<u32> = gene.data_raw.iter().collect();
                    // a cell outside the table yields NaN, which never matches,
                    // and `to_v4_bytes` stores the layer
                    snap_v3_norm(&mut gene.data_norm, |k| {
                        let lib = lib_f32.get(gene.indices[k] as usize).copied();
                        lib.map_or(f32::NAN, |l| norm_value(raw[k] as f32, l, target_size))
                    });
                }
                Ok((gene.original_index, gene.to_v4_bytes(&lib_f32, &format)?))
            })
            .collect::<Result<Vec<_>, ScDataErrors>>()?;
        for (_, bytes) in &payloads {
            stored_norm += V4Header::parse(bytes)?.norm_stored() as usize;
        }
        writer.write_compressed_gene_chunks(&payloads)?;
    }
    writer.set_cell_library_table(cell_lib.to_vec())?;
    writer.finalise()?;

    if verbose {
        println!(
            "  Converted {} genes to v5 ({} with stored norm) in {:.2?}.",
            order.len().separate_with_underscores(),
            stored_norm.separate_with_underscores(),
            start.elapsed()
        );
    }

    Ok(())
}

/// Convert a v3 or v4 store to v5: the cell file, and the gene file when
/// given.
///
/// Inputs are never modified. Neither output may equal an input path.
///
/// ### Params
///
/// * `cell_in` - Path to the cell-based file
/// * `gene_in` - Path to the gene-based file, if any
/// * `cell_out` - Path of the v5 cell file
/// * `gene_out` - Path of the v5 gene file, required with `gene_in`
/// * `codec` - Body codec of both outputs
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once both outputs are finalised.
pub fn convert_store_to_v5(
    cell_in: &str,
    gene_in: Option<&str>,
    cell_out: &str,
    gene_out: Option<&str>,
    codec: ChunkCodec,
    verbose: bool,
) -> Result<(), ScDataErrors> {
    let lib = convert_cell_file_to_v5(cell_in, cell_out, codec, verbose)?;
    if let (Some(gene_in), Some(gene_out)) = (gene_in, gene_out) {
        convert_gene_file_to_v5(gene_in, gene_out, &lib, codec, verbose)?;
    }
    Ok(())
}

/// Replace a v3 or v4 store by its v5 conversion, in place.
///
/// Converts into `<path>.v5.tmp` next to each file first and only touches the
/// originals once every conversion has succeeded, so a failure leaves the
/// store as it was. Each original is then renamed to `<path>.v<n>.bak` (`n`
/// its version) when `keep_backup` is set, and deleted otherwise.
///
/// ### Params
///
/// * `cell_path` - Path to the cell-based file
/// * `gene_path` - Path to the gene-based file, if any
/// * `codec` - Body codec of the converted files
/// * `keep_backup` - Keep the originals as `<path>.v<n>.bak`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once the converted files are in place.
pub fn upgrade_store_to_v5(
    cell_path: &str,
    gene_path: Option<&str>,
    codec: ChunkCodec,
    keep_backup: bool,
    verbose: bool,
) -> Result<(), ScDataErrors> {
    let tmp = |p: &str| format!("{p}.v5.tmp");
    let cleanup = |paths: &[Option<&str>]| {
        for p in paths.iter().flatten() {
            let _ = std::fs::remove_file(tmp(p));
        }
    };

    let mut versions = Vec::new();
    let result = (|| {
        let cell_tmp = tmp(cell_path);
        let lib = convert_cell_file_to_v5(cell_path, &cell_tmp, codec, verbose)?;
        if let Some(gene_path) = gene_path {
            convert_gene_file_to_v5(gene_path, &tmp(gene_path), &lib, codec, verbose)?;
        }
        // the converted files must open and agree with the originals
        for p in std::iter::once(cell_path).chain(gene_path) {
            let old = ParallelSparseReader::open_any(p)?;
            let new = ParallelSparseReader::new(&tmp(p))?;
            if (old.dims(), old.n_chunks()) != (new.dims(), new.n_chunks()) {
                return Err(ScDataErrors::V4ChunkCorrupt(format!(
                    "converted {p} does not match the original"
                )));
            }
            versions.push(old.file_version());
        }
        Ok(())
    })();
    if let Err(e) = result {
        cleanup(&[Some(cell_path), gene_path]);
        return Err(e);
    }

    for (p, version) in std::iter::once(cell_path).chain(gene_path).zip(versions) {
        if keep_backup {
            std::fs::rename(p, format!("{p}.v{version}.bak"))?;
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
                    chunk.data_norm[0] += 0.5;
                }
                chunk
            })
            .collect()
    }

    /// Write the synthetic store in a legacy version: v3 or v4.
    fn write_legacy_store(version: u32, cell_path: &str, gene_path: &str, target: f32) {
        let ctor = if version == SC_FILE_VERSION_V3 {
            CellGeneSparseWriter::new_v3
        } else {
            CellGeneSparseWriter::new_v4
        };
        let mut writer = ctor(cell_path, true, N_CELLS, N_GENES, target).unwrap();
        for chunk in synthetic_cells() {
            writer.write_cell_chunk(chunk).unwrap();
        }
        writer.finalise().unwrap();
        if version == SC_FILE_VERSION_V3 {
            crate::gene_file_io::write_gene_file_v3(cell_path, gene_path, Some(500), false)
                .unwrap();
        } else {
            crate::gene_file_io::write_gene_file(cell_path, gene_path, Some(500), false).unwrap();
        }
    }

    /// The norm as compared across a conversion: at the f16 precision v3
    /// holds, exactly from v4 (whose recompute and verbatim values both carry
    /// over to v5 unchanged).
    fn norm_bits(norm: &[f32], exact: bool) -> Vec<u32> {
        norm.iter()
            .map(|&v| {
                if exact {
                    v.to_bits()
                } else {
                    F16::from_f32(v).to_bits() as u32
                }
            })
            .collect()
    }

    fn assert_cells_equal(a: &CsrCellChunk, b: &CsrCellChunk, exact: bool) {
        assert_eq!(a.original_index, b.original_index);
        assert_eq!(a.library_size, b.library_size);
        assert_eq!(a.to_keep, b.to_keep);
        assert_eq!(a.indices, b.indices);
        assert_eq!(
            a.data_raw.iter().collect::<Vec<_>>(),
            b.data_raw.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            norm_bits(&a.data_norm, exact),
            norm_bits(&b.data_norm, exact)
        );
    }

    fn assert_genes_equal(a: &CscGeneChunk, b: &CscGeneChunk, exact: bool) {
        assert_eq!(a.original_index, b.original_index);
        assert_eq!(a.nnz, b.nnz);
        assert_eq!(a.indices, b.indices);
        assert_eq!(
            a.data_raw.iter().collect::<Vec<_>>(),
            b.data_raw.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            norm_bits(&a.data_norm, exact),
            norm_bits(&b.data_norm, exact)
        );
    }

    fn assert_sparse_equal(
        a: &CompressedSparseData2<f32, f32>,
        b: &CompressedSparseData2<f32, f32>,
        exact: bool,
    ) {
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
        assert_eq!(
            a.data_2.as_deref().map(|n| norm_bits(n, exact)),
            b.data_2.as_deref().map(|n| norm_bits(n, exact))
        );
    }

    /// Every read path of a converted store matches the v3 or v4 original, for
    /// each codec and with a zero target size (norm always stored).
    fn check_conversion(version: u32, codec: ChunkCodec, target: f32) {
        let exact = version == SC_FILE_VERSION_V4;
        let tag = format!("v{version}_{codec:?}_{}", target as u32);
        let s = Scratch::new(&tag, 4);
        write_legacy_store(version, s.path(0), s.path(1), target);
        convert_store_to_v5(
            s.path(0),
            Some(s.path(1)),
            s.path(2),
            Some(s.path(3)),
            codec,
            false,
        )
        .unwrap();

        let (cells_old, cells_new) = (
            ParallelSparseReader::open_any(s.path(0)).unwrap(),
            ParallelSparseReader::new(s.path(2)).unwrap(),
        );
        let (genes_old, genes_new) = (
            ParallelSparseReader::open_any(s.path(1)).unwrap(),
            ParallelSparseReader::new(s.path(3)).unwrap(),
        );
        assert_eq!(cells_old.file_version(), version);
        assert_eq!(cells_new.file_version(), SC_FILE_VERSION);
        assert_eq!(genes_new.file_version(), SC_FILE_VERSION);
        assert_eq!(cells_new.target_size(), cells_old.target_size());

        let all_cells: Vec<usize> = (0..N_CELLS).collect();
        let all_genes: Vec<usize> = (0..N_GENES).collect();
        for (a, b) in cells_old
            .read_cells_parallel(&all_cells)
            .unwrap()
            .iter()
            .zip(&cells_new.read_cells_parallel(&all_cells).unwrap())
        {
            assert_cells_equal(a, b, exact);
        }
        for (a, b) in genes_old
            .read_gene_parallel(&all_genes)
            .unwrap()
            .iter()
            .zip(&genes_new.read_gene_parallel(&all_genes).unwrap())
        {
            assert_genes_equal(a, b, exact);
        }

        let shuffled: Vec<usize> = (0..N_CELLS).rev().step_by(3).collect();
        let keep: IndexSet<u32> = (0..N_CELLS as u32).filter(|c| c % 4 != 1).collect();
        for layer in [
            DataLayerReturn::Raw,
            DataLayerReturn::Norm,
            DataLayerReturn::BothLayers,
        ] {
            assert_sparse_equal(
                &cells_old
                    .read_cells_sparse::<f32>(&shuffled, &layer)
                    .unwrap(),
                &cells_new
                    .read_cells_sparse::<f32>(&shuffled, &layer)
                    .unwrap(),
                exact,
            );
            for subset in [None, Some(&keep)] {
                assert_sparse_equal(
                    &genes_old
                        .read_genes_sparse::<f32>(&all_genes, subset, &layer)
                        .unwrap(),
                    &genes_new
                        .read_genes_sparse::<f32>(&all_genes, subset, &layer)
                        .unwrap(),
                    exact,
                );
            }
        }

        for (a, b) in genes_old
            .read_gene_parallel_filtered(&all_genes, &keep)
            .unwrap()
            .iter()
            .zip(
                &genes_new
                    .read_gene_parallel_filtered(&all_genes, &keep)
                    .unwrap(),
            )
        {
            assert_genes_equal(a, b, exact);
        }
        assert_eq!(
            cells_old.read_cell_library_sizes(&all_cells).unwrap(),
            cells_new.read_cell_library_sizes(&all_cells).unwrap()
        );
        assert_eq!(
            genes_old.read_gene_nnz(&all_genes).unwrap(),
            genes_new.read_gene_nnz(&all_genes).unwrap()
        );
        assert_eq!(cells_old.n_chunks(), cells_new.n_chunks());
    }

    #[test]
    fn test_v3_to_v5_round_trip_none() {
        check_conversion(SC_FILE_VERSION_V3, ChunkCodec::None, TARGET);
    }

    #[test]
    fn test_v3_to_v5_round_trip_lz4() {
        check_conversion(SC_FILE_VERSION_V3, ChunkCodec::Lz4, TARGET);
    }

    #[test]
    fn test_v3_to_v5_round_trip_zstd() {
        check_conversion(SC_FILE_VERSION_V3, ChunkCodec::Zstd, TARGET);
    }

    #[test]
    fn test_v3_to_v5_round_trip_unknown_target_stores_norm() {
        check_conversion(SC_FILE_VERSION_V3, ChunkCodec::Lz4, 0.0);
    }

    #[test]
    fn test_v4_to_v5_round_trip_is_exact() {
        check_conversion(SC_FILE_VERSION_V4, ChunkCodec::Lz4, TARGET);
        check_conversion(SC_FILE_VERSION_V4, ChunkCodec::Zstd, 0.0);
    }

    #[test]
    fn test_v5_drops_norm_and_shrinks() {
        let s = Scratch::new("shrink", 4);
        write_legacy_store(SC_FILE_VERSION_V3, s.path(0), s.path(1), TARGET);
        convert_store_to_v5(
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

    /// The context a single-layer log-CPM cell chunk decodes with.
    fn cell_ctx(norm_f32: bool) -> V4Ctx<'static> {
        V4Ctx {
            target_size: TARGET,
            cell_lib: None,
            cell_norm: None,
            is_cell: true,
            n_layers: 1,
            layer: 0,
            norm_f32,
            norm_kind: NormKind::LogCpm,
            weights: None,
        }
    }

    /// The chunk format of a version.
    fn format(version: u32) -> ChunkFormat {
        ChunkFormat {
            version,
            codec: ChunkCodec::Lz4,
            target_size: TARGET,
        }
    }

    #[test]
    fn test_v4_chunk_keeps_unsorted_and_duplicate_indices() {
        let chunk = CsrCellChunk::from_data(&[3u32, 1, 2, 5], &[9u32, 2, 2, 400], 5, TARGET, true);
        let bytes = chunk.to_v4_bytes(&format(SC_FILE_VERSION)).unwrap();
        let back = v4_cell_chunk(&bytes, &cell_ctx(true), &mut V4Decoded::default()).unwrap();
        assert_cells_equal(&chunk, &back, true);
    }

    /// With one layer and a derived norm a cell chunk is the same bytes in v4
    /// and v5.
    #[test]
    fn test_single_layer_cell_chunk_is_byte_identical_in_v4_and_v5() {
        for chunk in synthetic_cells().iter().filter(|c| c.original_index != 13) {
            assert_eq!(
                chunk.to_v4_bytes(&format(SC_FILE_VERSION_V4)).unwrap(),
                chunk.to_v4_bytes(&format(SC_FILE_VERSION)).unwrap()
            );
        }
    }

    /// v5 keeps a verbatim norm at f32; v4 narrows it to f16.
    #[test]
    fn test_verbatim_norm_is_f32_in_v5_and_f16_in_v4() {
        let mut chunk = CsrCellChunk::from_data(&[1u32, 2, 7], &[0u32, 4, 9], 0, TARGET, true);
        chunk.data_norm[1] = 0.123_456_7;
        let v5 = chunk.to_v4_bytes(&format(SC_FILE_VERSION)).unwrap();
        let v4 = chunk.to_v4_bytes(&format(SC_FILE_VERSION_V4)).unwrap();
        let mut s = V4Decoded::default();
        let back5 = v4_cell_chunk(&v5, &cell_ctx(true), &mut s).unwrap();
        let back4 = v4_cell_chunk(&v4, &cell_ctx(false), &mut s).unwrap();
        assert_eq!(
            norm_bits(&back5.data_norm, true),
            norm_bits(&chunk.data_norm, true)
        );
        assert_ne!(back4.data_norm[1], chunk.data_norm[1]);
        assert_eq!(
            norm_bits(&back4.data_norm, false),
            norm_bits(&chunk.data_norm, false)
        );
    }

    #[test]
    fn test_default_writer_emits_v5_and_converter_refuses_it() {
        let s = Scratch::new("guards", 2);
        let mut writer = CellGeneSparseWriter::new(s.path(0), true, 1, 3, TARGET).unwrap();
        writer
            .write_cell_chunk(CsrCellChunk::from_data(&[1u32], &[0u32], 0, TARGET, true))
            .unwrap();
        writer.finalise().unwrap();
        assert_eq!(peek_file_version(s.path(0)).unwrap(), SC_FILE_VERSION);
        assert!(matches!(
            convert_cell_file_to_v5(s.path(0), s.path(1), ChunkCodec::Lz4, false),
            Err(ScDataErrors::FileVersionMismatch { .. })
        ));
    }

    #[test]
    fn test_peek_file_version_distinguishes_v3_v4_and_v5() {
        let s = Scratch::new("peek", 3);
        let chunk = || CsrCellChunk::from_data(&[1u32, 2], &[0u32, 1], 0, TARGET, true);
        let ctors = [
            CellGeneSparseWriter::new_v3,
            CellGeneSparseWriter::new_v4,
            CellGeneSparseWriter::new,
        ];
        for (i, ctor) in ctors.into_iter().enumerate() {
            let mut w = ctor(s.path(i), true, 1, 3, TARGET).unwrap();
            w.write_cell_chunk(chunk()).unwrap();
            w.finalise().unwrap();
        }
        assert_eq!(peek_file_version(s.path(0)).unwrap(), SC_FILE_VERSION_V3);
        assert_eq!(peek_file_version(s.path(1)).unwrap(), SC_FILE_VERSION_V4);
        assert_eq!(peek_file_version(s.path(2)).unwrap(), SC_FILE_VERSION);
        assert_eq!(
            crate::data_io::peek_target_size(s.path(1)).unwrap(),
            Some(TARGET)
        );
        assert_eq!(
            crate::data_io::peek_target_size(s.path(2)).unwrap(),
            Some(TARGET)
        );
    }

    /// v3 opens for the converters only.
    #[test]
    fn test_reader_refuses_v3_with_the_converter_hint() {
        let s = Scratch::new("v3refuse", 2);
        write_legacy_store(SC_FILE_VERSION_V3, s.path(0), s.path(1), TARGET);
        assert!(matches!(
            ParallelSparseReader::new(s.path(0)),
            Err(ScDataErrors::V3NeedsConversion(_))
        ));
        assert!(ParallelSparseReader::open_any(s.path(0)).is_ok());
    }

    #[test]
    fn test_default_gene_file_is_v5_and_matches_the_v3_one() {
        let s = Scratch::new("genefile", 4);
        write_legacy_store(SC_FILE_VERSION_V3, s.path(0), s.path(1), TARGET);
        convert_cell_file_to_v5(s.path(0), s.path(2), ChunkCodec::Lz4, false).unwrap();
        crate::gene_file_io::write_gene_file(s.path(2), s.path(3), Some(500), false).unwrap();
        assert_eq!(peek_file_version(s.path(3)).unwrap(), SC_FILE_VERSION);

        let (v3, v5) = (
            ParallelSparseReader::open_any(s.path(1)).unwrap(),
            ParallelSparseReader::new(s.path(3)).unwrap(),
        );
        let all: Vec<usize> = (0..N_GENES).collect();
        for (a, b) in v3
            .read_gene_parallel(&all)
            .unwrap()
            .iter()
            .zip(&v5.read_gene_parallel(&all).unwrap())
        {
            assert_genes_equal(a, b, false);
        }
    }

    #[test]
    fn test_upgrade_store_replaces_files_and_keeps_backups() {
        for version in [SC_FILE_VERSION_V3, SC_FILE_VERSION_V4] {
            let s = Scratch::new(&format!("upgrade{version}"), 2);
            write_legacy_store(version, s.path(0), s.path(1), TARGET);
            let cells_before = ParallelSparseReader::open_any(s.path(0))
                .unwrap()
                .get_all_cells()
                .unwrap();
            upgrade_store_to_v5(s.path(0), Some(s.path(1)), ChunkCodec::Lz4, true, false).unwrap();

            assert_eq!(peek_file_version(s.path(0)).unwrap(), SC_FILE_VERSION);
            assert_eq!(peek_file_version(s.path(1)).unwrap(), SC_FILE_VERSION);
            let backup = format!("{}.v{version}.bak", s.path(0));
            assert_eq!(peek_file_version(&backup).unwrap(), version);
            for (a, b) in cells_before.iter().zip(
                &ParallelSparseReader::new(s.path(0))
                    .unwrap()
                    .get_all_cells()
                    .unwrap(),
            ) {
                assert_cells_equal(a, b, version == SC_FILE_VERSION_V4);
            }
            for p in [backup, format!("{}.v{version}.bak", s.path(1))] {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    #[test]
    fn test_archive_round_trips_a_v5_cell_file() {
        let s = Scratch::new("archive", 6);
        write_legacy_store(SC_FILE_VERSION_V3, s.path(0), s.path(1), TARGET);
        convert_cell_file_to_v5(s.path(0), s.path(2), ChunkCodec::Lz4, false).unwrap();
        crate::archive_io::archive_cell_file(s.path(2), s.path(3), 3, false).unwrap();
        crate::archive_io::restore_archive(s.path(3), s.path(4), s.path(5), Some(500), false)
            .unwrap();

        let original = ParallelSparseReader::new(s.path(2)).unwrap();
        let restored = ParallelSparseReader::new(s.path(4)).unwrap();
        let all: Vec<usize> = (0..N_CELLS).collect();
        for (a, b) in original
            .read_cells_parallel(&all)
            .unwrap()
            .iter()
            .zip(&restored.read_cells_parallel(&all).unwrap())
        {
            assert_cells_equal(a, b, false);
        }
    }

    ////////
    // v5 //
    ////////

    /// Dense `[cell][gene]` layers for the multi-layer tests: DP-like depth,
    /// AD-like alt counts with zeros where depth is not, and a binarised
    /// layer of ones.
    fn allele_layers(n_cells: usize, n_genes: usize) -> [Vec<Vec<u32>>; 3] {
        let mut state = 99u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut dp = vec![vec![0u32; n_genes]; n_cells];
        let mut ad = vec![vec![0u32; n_genes]; n_cells];
        let mut bin = vec![vec![0u32; n_genes]; n_cells];
        for c in 0..n_cells {
            for g in 0..n_genes {
                if next() % 4 == 0 {
                    dp[c][g] = 1 + next() % 30;
                    ad[c][g] = next() % (dp[c][g] + 1);
                    bin[c][g] = 1;
                }
            }
        }
        [dp, ad, bin]
    }

    /// Write `layers` as a v5 cell file, then its gene file.
    fn write_layer_store(cell_path: &str, gene_path: &str, layers: &[Vec<Vec<u32>>]) {
        let (n_cells, n_genes) = (layers[0].len(), layers[0][0].len());
        let mut w = CellGeneSparseWriter::new(cell_path, true, n_cells, n_genes, 0.0)
            .unwrap()
            .with_kinds(crate::v5_io::FeatureKind::Snp, NormKind::None)
            .unwrap()
            .with_layers(layers.len())
            .unwrap();
        for c in 0..n_cells {
            let idx: Vec<u32> = (0..n_genes as u32)
                .filter(|&g| layers[0][c][g as usize] > 0)
                .collect();
            let vals: Vec<Vec<u32>> = layers
                .iter()
                .map(|l| idx.iter().map(|&g| l[c][g as usize]).collect())
                .collect();
            let refs: Vec<&[u32]> = vals.iter().map(Vec::as_slice).collect();
            w.write_cell_layers(c, &idx, &refs, true).unwrap();
        }
        w.finalise().unwrap();
        crate::gene_file_io::write_gene_file(cell_path, gene_path, Some(37), false).unwrap();
    }

    /// Every layer reads back from both orientations, the norm is refused and
    /// the kinds carry over to the gene file.
    #[test]
    fn test_v5_layers_round_trip_both_orientations() {
        let (n_cells, n_genes) = (57, 23);
        let layers = allele_layers(n_cells, n_genes);
        let s = Scratch::new("layers", 2);
        write_layer_store(s.path(0), s.path(1), &layers);

        let all_cells: Vec<usize> = (0..n_cells).collect();
        let all_genes: Vec<usize> = (0..n_genes).collect();
        for (l, dense) in layers.iter().enumerate() {
            let cells = ParallelSparseReader::new(s.path(0))
                .unwrap()
                .with_layer(l)
                .unwrap();
            let genes = ParallelSparseReader::new(s.path(1))
                .unwrap()
                .with_layer(l)
                .unwrap();
            assert_eq!(genes.n_layers(), 3);
            assert_eq!(genes.norm_kind(), NormKind::None);
            assert_eq!(genes.feature_kind(), crate::v5_io::FeatureKind::Snp);

            let csr = cells
                .read_cells_sparse::<u32>(&all_cells, &DataLayerReturn::Raw)
                .unwrap();
            let csc = genes
                .read_genes_sparse::<u32>(&all_genes, None, &DataLayerReturn::Raw)
                .unwrap();
            for c in 0..n_cells {
                for k in csr.indptr[c] as usize..csr.indptr[c + 1] as usize {
                    assert_eq!(csr.data[k], dense[c][csr.indices[k] as usize]);
                }
            }
            for g in 0..n_genes {
                for k in csc.indptr[g] as usize..csc.indptr[g + 1] as usize {
                    assert_eq!(csc.data[k], dense[csc.indices[k] as usize][g]);
                }
            }
            assert!(matches!(
                cells.read_cells_sparse::<u32>(&all_cells, &DataLayerReturn::Norm),
                Err(ScDataErrors::NormNotAvailable(_))
            ));
        }
        assert!(matches!(
            ParallelSparseReader::new(s.path(0)).unwrap().with_layer(3),
            Err(ScDataErrors::LayerOutOfRange { .. })
        ));
    }

    /// A binarised layer packs to one width byte per block of 128.
    #[test]
    fn test_binarised_layer_packs_to_width_zero() {
        let idx: Vec<u32> = (0..1024).map(|i| i * 3).collect();
        let ones = vec![1u32; idx.len()];
        let fmt = ChunkFormat {
            version: SC_FILE_VERSION,
            codec: ChunkCodec::None,
            target_size: 0.0,
        };
        let one = fmt.encode_layers(&idx, &[&ones], 0, true).unwrap();
        let two = fmt.encode_layers(&idx, &[&ones, &ones], 0, true).unwrap();
        assert_eq!(two.len() - one.len(), idx.len() / PACK_BLOCK);
    }

    /// Chunks written out of index order read back by original index through
    /// the sorted lookup section, and in file order through `chunk_order`.
    #[test]
    fn test_v5_out_of_order_chunks_resolve_by_original_index() {
        let s = Scratch::new("lookup", 1);
        let order = [4usize, 0, 9, 2, 7];
        let mut w = CellGeneSparseWriter::new(s.path(0), true, 10, 5, TARGET).unwrap();
        for &o in &order {
            let counts = [o as u32 + 1, 2];
            w.write_cell_chunk(CsrCellChunk::from_data(
                &counts,
                &[0u32, 3],
                o,
                TARGET,
                true,
            ))
            .unwrap();
        }
        w.finalise().unwrap();

        let r = ParallelSparseReader::new(s.path(0)).unwrap();
        assert_eq!(r.chunk_order().unwrap(), order.to_vec());
        let back = r.read_cells_parallel(&[9, 0, 7]).unwrap();
        for (cell, want) in back.iter().zip([9usize, 0, 7]) {
            assert_eq!(cell.original_index, want);
            assert_eq!(cell.data_raw.get(0), want as u32 + 1);
        }
        assert_eq!(r.read_cell_library_sizes(&[2, 4]).unwrap(), vec![5, 7]);
        assert!(matches!(
            r.read_cells_parallel(&[1]),
            Err(ScDataErrors::ChunkIndexNotFound(1))
        ));
    }

    /// Overwrite bytes of a finished file.
    fn patch_bytes(path: &str, at: u64, bytes: &[u8]) {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        f.seek(SeekFrom::Start(at)).unwrap();
        f.write_all(bytes).unwrap();
    }

    /// A small v5 cell file.
    fn small_v5(path: &str) {
        let mut w = CellGeneSparseWriter::new(path, true, 2, 4, TARGET).unwrap();
        for c in 0..2 {
            w.write_cell_chunk(CsrCellChunk::from_data(
                &[1u32, 3],
                &[0u32, 2],
                c,
                TARGET,
                true,
            ))
            .unwrap();
        }
        w.finalise().unwrap();
    }

    /// Unknown incompat bits, unknown kinds and broken sections refuse the
    /// file; an unknown optional section is skipped.
    #[test]
    fn test_v5_refuses_what_it_does_not_understand() {
        let s = Scratch::new("refuse", 1);
        let invalid = |p: &str| {
            matches!(
                ParallelSparseReader::new(p),
                Err(ScDataErrors::V5HeaderInvalid(_))
            )
        };

        small_v5(s.path(0));
        patch_bytes(s.path(0), 12, &1u32.to_le_bytes());
        assert!(invalid(s.path(0)));

        small_v5(s.path(0));
        patch_bytes(s.path(0), 18, &[9]);
        assert!(invalid(s.path(0)));

        small_v5(s.path(0));
        patch_bytes(s.path(0), 19, &[0]);
        assert!(invalid(s.path(0)));

        // the first directory entry is the offsets: make it an unknown kind
        let dir = |p: &str| {
            let b = std::fs::read(p).unwrap();
            crate::v5_io::le_u64(&b, 48)
        };
        small_v5(s.path(0));
        patch_bytes(s.path(0), dir(s.path(0)), &99u32.to_le_bytes());
        assert!(invalid(s.path(0)), "required unknown section");
        patch_bytes(s.path(0), dir(s.path(0)) + 4, &0u32.to_le_bytes());
        // optional now, so skipped, and the offsets are missing
        assert!(invalid(s.path(0)), "offsets missing");

        small_v5(s.path(0));
        patch_bytes(s.path(0), dir(s.path(0)) + 8, &u64::MAX.to_le_bytes());
        assert!(invalid(s.path(0)), "section outside the file");

        small_v5(s.path(0));
        patch_bytes(s.path(0), 48, &u64::MAX.to_le_bytes());
        assert!(invalid(s.path(0)), "directory outside the file");
    }

    /// The cell-ID hash is written, set after the fact and compared.
    #[test]
    fn test_v5_cell_ids_align_stores() {
        let s = Scratch::new("cellids", 2);
        let ids = ["AAAC-1", "AAAG-1"];
        let mut w = CellGeneSparseWriter::new(s.path(0), true, 2, 4, TARGET).unwrap();
        w.set_cell_ids(&ids).unwrap();
        for c in 0..2 {
            w.write_cell_chunk(CsrCellChunk::from_data(&[1u32], &[1u32], c, TARGET, true))
                .unwrap();
        }
        w.finalise().unwrap();
        small_v5(s.path(1));

        let (a, b) = (
            ParallelSparseReader::new(s.path(0)).unwrap(),
            ParallelSparseReader::new(s.path(1)).unwrap(),
        );
        assert_eq!(a.cell_id_hash(), Some(crate::v5_io::cell_id_hash(&ids)));
        assert_eq!(b.cell_id_hash(), None);
        assert!(crate::v5_io::check_same_cells(&[&a, &b]).is_err());

        crate::v5_io::set_cell_ids(&[s.path(1)], &ids).unwrap();
        let b = ParallelSparseReader::new(s.path(1)).unwrap();
        crate::v5_io::check_same_cells(&[&a, &b]).unwrap();
        assert_eq!(b.read_cells_parallel(&[0, 1]).unwrap().len(), 2);
        assert!(crate::v5_io::set_cell_ids(&[s.path(1)], &ids[..1]).is_err());
    }

    /// TF-IDF reads `ln1p(x / lib * scale * n / total_f)` in both orientations.
    #[test]
    fn test_v5_tf_idf_matches_the_formula() {
        let s = Scratch::new("tfidf", 2);
        let (n_cells, n_genes) = (40, 12);
        let [dp, _, _] = allele_layers(n_cells, n_genes);
        let mut w = CellGeneSparseWriter::new(s.path(0), true, n_cells, n_genes, 0.0)
            .unwrap()
            .with_kinds(crate::v5_io::FeatureKind::Peak, NormKind::LogCpm)
            .unwrap();
        for (c, row) in dp.iter().enumerate() {
            let idx: Vec<u32> = (0..n_genes as u32)
                .filter(|&g| row[g as usize] > 0)
                .collect();
            let vals: Vec<u32> = idx.iter().map(|&g| row[g as usize]).collect();
            w.write_cell_layers(c, &idx, &[&vals], true).unwrap();
        }
        w.finalise().unwrap();
        crate::gene_file_io::write_gene_file(s.path(0), s.path(1), None, false).unwrap();

        let genes = ParallelSparseReader::new(s.path(1)).unwrap();
        let idf = crate::v5_io::tf_idf_weights(&genes, None).unwrap();
        crate::v5_io::set_tf_idf(&[s.path(0), s.path(1)], &idf, 1e4).unwrap();

        let lib: Vec<f32> = dp.iter().map(|r| r.iter().sum::<u32>() as f32).collect();
        let total: Vec<f64> = (0..n_genes)
            .map(|g| dp.iter().map(|r| r[g] as f64).sum())
            .collect();
        let want = |c: usize, g: usize| {
            let w = (n_cells as f64 / total[g]) as f32;
            (dp[c][g] as f32 / lib[c] * (1e4 * w)).ln_1p()
        };

        let cells = ParallelSparseReader::new(s.path(0)).unwrap();
        let genes = ParallelSparseReader::new(s.path(1)).unwrap();
        assert_eq!(cells.norm_kind(), NormKind::TfIdf);
        let all_cells: Vec<usize> = (0..n_cells).collect();
        let all_genes: Vec<usize> = (0..n_genes).collect();
        let csr = cells
            .read_cells_sparse::<u32>(&all_cells, &DataLayerReturn::Norm)
            .unwrap();
        let csc = genes
            .read_genes_sparse::<u32>(&all_genes, None, &DataLayerReturn::Norm)
            .unwrap();
        let (n1, n2) = (csr.data_2.unwrap(), csc.data_2.unwrap());
        for c in 0..n_cells {
            for k in csr.indptr[c] as usize..csr.indptr[c + 1] as usize {
                let g = csr.indices[k] as usize;
                assert_eq!(n1[k].to_bits(), want(c, g).to_bits(), "cell {c} gene {g}");
            }
        }
        for g in 0..n_genes {
            for k in csc.indptr[g] as usize..csc.indptr[g + 1] as usize {
                let c = csc.indices[k] as usize;
                assert_eq!(n2[k].to_bits(), want(c, g).to_bits(), "gene {g} cell {c}");
            }
        }
    }
}
