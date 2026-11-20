//! The v4 chunk encoding, shared by the v4 and v5 containers.
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
//! [8..16)  aux            u64   cell: library size; gene: v5 sum of raw
//!                               layer 0, v4 unused (written as zero)
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

use crate::prelude::*;
use crate::store::chunk::*;
use crate::store::v5::{MAX_LAYERS, NormKind};

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
        out.reserve((block.len() * width).div_ceil(8));
        // at most 31 bits carry over, plus 32 new: fits the u64
        let (mut acc, mut nbits) = (0u64, 0usize);
        for &v in block {
            acc |= (v as u64) << nbits;
            nbits += width;
            if nbits >= 32 {
                out.extend_from_slice(&(acc as u32).to_le_bytes());
                acc >>= 32;
                nbits -= 32;
            }
        }
        out.extend_from_slice(&acc.to_le_bytes()[..nbits.div_ceil(8)]);
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
    /// Cell chunks: library size. Gene chunks: sum of raw layer 0 on v5,
    /// unused on v4.
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
        // a chunk-level feature this build does not know must not decode as
        // if it were absent
        let unknown_flags = chunk[25] & !(FLAG_NORM_STORED | FLAG_RAW_MINUS_ONE);
        if unknown_flags != 0 || chunk[28..32] != [0; 4] {
            return Err(ScDataErrors::V4ChunkCorrupt(format!(
                "unknown chunk flags {unknown_flags:#x} or reserved bytes set, written by a \
                 newer bixverse or corrupt"
            )));
        }
        Ok(Self {
            nnz: le_u32(chunk, 0) as usize,
            body_len: le_u32(chunk, 4) as usize,
            aux: le_u64(chunk, 8),
            original_index: le_u64(chunk, 16) as usize,
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
pub(crate) fn snap_v3_norm(norm: &mut [f32], recompute: impl Fn(usize) -> f32) {
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
/// Header plus body, or [`ScDataErrors::V4ChunkCorrupt`] for a chunk whose
/// entry count or body does not fit the header's `u32` fields.
fn encode_chunk(p: &EncodeIn, codec: ChunkCodec) -> Result<Vec<u8>, ScDataErrors> {
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

    let too_big = || {
        ScDataErrors::V4ChunkCorrupt(format!(
            "chunk {} holds {n} entries in {} bytes, more than a chunk header can record",
            p.original_index,
            body.len()
        ))
    };
    let n32 = u32::try_from(n).map_err(|_| too_big())?;
    let body_len = u32::try_from(body.len()).map_err(|_| too_big())?;
    let mut out = Vec::with_capacity(V4_HEADER_LEN + payload.len());
    out.extend_from_slice(&n32.to_le_bytes());
    out.extend_from_slice(&body_len.to_le_bytes());
    out.extend_from_slice(&p.aux.to_le_bytes());
    out.extend_from_slice(&(p.original_index as u64).to_le_bytes());
    out.extend_from_slice(&[p.to_keep as u8, flags, used.to_u8(), mask, 0, 0, 0, 0]);
    out.extend_from_slice(&payload);
    Ok(out)
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

        encode_chunk(
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
        )
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

        encode_chunk(
            &EncodeIn {
                indices: &self.indices,
                layers: &[&raw],
                norm: &self.data_norm,
                aux: format.gene_aux(&raw),
                original_index: self.original_index,
                to_keep: self.to_keep,
                norm_derivable,
                norm_f32: format.norm_f32(),
            },
            format.codec,
        )
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
/// store without one.
pub(crate) fn v4_gene_chunk(
    chunk: &[u8],
    ctx: &V4Ctx,
    s: &mut V4Decoded,
) -> Result<CscGeneChunk, ScDataErrors> {
    let h = decode_v4(chunk, ctx, true, ctx.norm_available(), s)?;
    Ok(CscGeneChunk {
        data_raw: RawCounts::from_u32_auto(&s.raw),
        data_norm: s.norm.clone(),
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
    /// * `raw` - Raw layer 0
    ///
    /// ### Returns
    ///
    /// The sum of `raw` on v5. Zero on v4, whose field held a norm sum
    /// nothing reads.
    pub(crate) fn gene_aux(&self, raw: &[u32]) -> u64 {
        if self.norm_f32() {
            raw.iter().map(|&x| x as u64).sum()
        } else {
            0
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
        encode_chunk(
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
        )
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

    /// Encode a cell chunk straight from its integer counts.
    ///
    /// Same bytes as [`Self::encode_cell`] on the chunk
    /// [`CsrCellChunk::from_data`] builds from these counts. A sequential f32
    /// sum of integers is a whole number, so the stored library size converts
    /// back to the same f32 and the norm recomputes bit for bit: it is never
    /// built.
    ///
    /// ### Params
    ///
    /// * `genes` - Gene indices, ascending
    /// * `counts` - Raw count per gene
    /// * `original_index` - Index the cell is stored under
    ///
    /// ### Returns
    ///
    /// `Ok(Some((library size, payload)))`, `Ok(None)` where the norm would be
    /// stored (v3, or a target size of `0.0`; build the chunk then), or an
    /// error for a chunk too large for its header.
    pub fn encode_cell_counts(
        &self,
        genes: &[u32],
        counts: &[u32],
        original_index: usize,
    ) -> Result<Option<(usize, Vec<u8>)>, ScDataErrors> {
        if !self.uses_v4_codec() || self.target_size == 0.0 {
            return Ok(None);
        }
        // must stay the sum `CsrCellChunk::from_data` takes
        let library_size = counts.iter().map(|&x| x as f32).sum::<f32>() as usize;
        let bytes = encode_chunk(
            &EncodeIn {
                indices: genes,
                layers: &[counts],
                norm: &[],
                aux: library_size as u64,
                original_index,
                to_keep: true,
                norm_derivable: true,
                norm_f32: self.norm_f32(),
            },
            self.codec,
        )?;
        Ok(Some((library_size, bytes)))
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
    /// layer. The caller guarantees that, e.g. because every cell chunk of
    /// the input recomputes its norm.
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices, ascending
    /// * `raw` - Raw counts, same length
    /// * `original_index` - Gene index
    ///
    /// ### Returns
    ///
    /// The payload the writer's `write_compressed_gene_chunks` expects.
    pub(crate) fn encode_gene_derivable(
        &self,
        indices: &[u32],
        raw: &[u32],
        original_index: usize,
    ) -> Result<Vec<u8>, ScDataErrors> {
        debug_assert!(self.uses_v4_codec() && indices.len() == raw.len());
        encode_chunk(
            &EncodeIn {
                indices,
                layers: &[raw],
                norm: &[],
                aux: self.gene_aux(raw),
                original_index,
                to_keep: true,
                norm_derivable: true,
                norm_f32: self.norm_f32(),
            },
            self.codec,
        )
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::store::reader::peek_file_version;
    use crate::store::writer::CellGeneSparseWriter;
    use tempfile::TempDir;

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

    /// `n` scratch file paths in a fresh temporary directory.
    ///
    /// ### Params
    ///
    /// * `n` - Number of paths
    ///
    /// ### Returns
    ///
    /// The directory, which removes everything in it on drop, and the paths.
    pub(crate) fn scratch(n: usize) -> (TempDir, Vec<String>) {
        let dir = TempDir::new().unwrap();
        let paths = (0..n)
            .map(|i| {
                dir.path()
                    .join(format!("{i}.bin"))
                    .to_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        (dir, paths)
    }

    /// Cells in the synthetic store.
    pub(crate) const N_CELLS: usize = 300;
    /// Genes in the synthetic store.
    pub(crate) const N_GENES: usize = 40;
    /// Target size of the synthetic store.
    pub(crate) const TARGET: f32 = 1e4;

    /// Cells with a spread of counts, one empty cell, one with a count above
    /// every small width, and one whose stored norm does not recompute.
    ///
    /// ### Returns
    ///
    /// The cells, in index order.
    pub(crate) fn synthetic_cells() -> Vec<CsrCellChunk> {
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
    ///
    /// ### Params
    ///
    /// * `version` - [`SC_FILE_VERSION_V3`] or [`SC_FILE_VERSION_V4`]
    /// * `cell_path` - Cell file
    /// * `gene_path` - Gene file
    /// * `target` - Target size
    pub(crate) fn write_legacy_store(version: u32, cell_path: &str, gene_path: &str, target: f32) {
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
            crate::store::gene_file::write_gene_file_v3(cell_path, gene_path, Some(500), false)
                .unwrap();
        } else {
            crate::store::gene_file::write_gene_file(cell_path, gene_path, Some(500), false)
                .unwrap();
        }
    }

    /// The norm bits to compare: exact, or at the f16 precision v3 and verbatim
    /// v4 norms hold (a conversion snaps those to the f32 recompute).
    ///
    /// ### Params
    ///
    /// * `norm` - The norm layer
    /// * `exact` - Keep f32 bits rather than f16
    ///
    /// ### Returns
    ///
    /// The bits to compare.
    pub(crate) fn norm_bits(norm: &[f32], exact: bool) -> Vec<u32> {
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

    /// Assert two cells agree, the norm at the precision `exact` asks for.
    ///
    /// ### Params
    ///
    /// * `a` - First cell
    /// * `b` - Second cell
    /// * `exact` - Compare the norm bit for bit rather than at f16
    pub(crate) fn assert_cells_equal(a: &CsrCellChunk, b: &CsrCellChunk, exact: bool) {
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

    /// Assert two genes agree, the norm at the precision `exact` asks for.
    ///
    /// ### Params
    ///
    /// * `a` - First gene
    /// * `b` - Second gene
    /// * `exact` - Compare the norm bit for bit rather than at f16
    pub(crate) fn assert_genes_equal(a: &CscGeneChunk, b: &CscGeneChunk, exact: bool) {
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

    /// The context a single-layer log-CPM cell chunk decodes with.
    ///
    /// ### Params
    ///
    /// * `norm_f32` - Whether a verbatim norm is f32 (v5)
    ///
    /// ### Returns
    ///
    /// The context.
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
    ///
    /// ### Params
    ///
    /// * `version` - File version
    ///
    /// ### Returns
    ///
    /// The format, lz4 at the synthetic target size.
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
    fn test_peek_file_version_distinguishes_v3_v4_and_v5() {
        let (_dir, s) = scratch(3);
        let chunk = || CsrCellChunk::from_data(&[1u32, 2], &[0u32, 1], 0, TARGET, true);
        let ctors = [
            CellGeneSparseWriter::new_v3,
            CellGeneSparseWriter::new_v4,
            CellGeneSparseWriter::new,
        ];
        for (i, ctor) in ctors.into_iter().enumerate() {
            let mut w = ctor(&s[i], true, 1, 3, TARGET).unwrap();
            w.write_cell_chunk(chunk()).unwrap();
            w.finalise().unwrap();
        }
        assert_eq!(peek_file_version(&s[0]).unwrap(), SC_FILE_VERSION_V3);
        assert_eq!(peek_file_version(&s[1]).unwrap(), SC_FILE_VERSION_V4);
        assert_eq!(peek_file_version(&s[2]).unwrap(), SC_FILE_VERSION);
        assert_eq!(
            crate::store::reader::peek_target_size(&s[1]).unwrap(),
            Some(TARGET)
        );
        assert_eq!(
            crate::store::reader::peek_target_size(&s[2]).unwrap(),
            Some(TARGET)
        );
    }

    /// v3 opens for the converters only.
    #[test]
    fn test_reader_refuses_v3_with_the_converter_hint() {
        let (_dir, s) = scratch(2);
        write_legacy_store(SC_FILE_VERSION_V3, &s[0], &s[1], TARGET);
        assert!(matches!(
            ParallelSparseReader::new(&s[0]),
            Err(ScDataErrors::V3NeedsConversion(_))
        ));
        assert!(ParallelSparseReader::open_any(&s[0]).is_ok());
    }

    ////////
    // v5 //
    ////////

    /// Dense `[cell][gene]` layers for the multi-layer tests: DP-like depth,
    /// AD-like alt counts with zeros where depth is not, and a binarised
    /// layer of ones.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Cells
    /// * `n_genes` - Genes
    ///
    /// ### Returns
    ///
    /// `[depth, alt, binarised]`.
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
    ///
    /// ### Params
    ///
    /// * `cell_path` - Cell file
    /// * `gene_path` - Gene file
    /// * `layers` - Dense `[cell][gene]` layers, layer 0 first
    fn write_layer_store(cell_path: &str, gene_path: &str, layers: &[Vec<Vec<u32>>]) {
        let (n_cells, n_genes) = (layers[0].len(), layers[0][0].len());
        let mut w = CellGeneSparseWriter::new(cell_path, true, n_cells, n_genes, 0.0)
            .unwrap()
            .with_kinds(crate::store::v5::FeatureKind::Snp, NormKind::None)
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
        crate::store::gene_file::write_gene_file(cell_path, gene_path, Some(37), false).unwrap();
    }

    /// Every layer reads back from both orientations, the norm is refused and
    /// the kinds carry over to the gene file.
    #[test]
    fn test_v5_layers_round_trip_both_orientations() {
        let (n_cells, n_genes) = (57, 23);
        let layers = allele_layers(n_cells, n_genes);
        let (_dir, s) = scratch(2);
        write_layer_store(&s[0], &s[1], &layers);

        let all_cells: Vec<usize> = (0..n_cells).collect();
        let all_genes: Vec<usize> = (0..n_genes).collect();
        for (l, dense) in layers.iter().enumerate() {
            let cells = ParallelSparseReader::new(&s[0])
                .unwrap()
                .with_layer(l)
                .unwrap();
            let genes = ParallelSparseReader::new(&s[1])
                .unwrap()
                .with_layer(l)
                .unwrap();
            assert_eq!(genes.n_layers(), 3);
            assert_eq!(genes.norm_kind(), NormKind::None);
            assert_eq!(genes.feature_kind(), crate::store::v5::FeatureKind::Snp);

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
            ParallelSparseReader::new(&s[0]).unwrap().with_layer(3),
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
        let (_dir, s) = scratch(1);
        let order = [4usize, 0, 9, 2, 7];
        let mut w = CellGeneSparseWriter::new(&s[0], true, 10, 5, TARGET).unwrap();
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

        let r = ParallelSparseReader::new(&s[0]).unwrap();
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
    ///
    /// ### Params
    ///
    /// * `path` - The file
    /// * `at` - Byte offset
    /// * `bytes` - Replacement bytes
    fn patch_bytes(path: &str, at: u64, bytes: &[u8]) {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        f.seek(SeekFrom::Start(at)).unwrap();
        f.write_all(bytes).unwrap();
    }

    /// A small v5 cell file.
    ///
    /// ### Params
    ///
    /// * `path` - Output file
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
        let (_dir, s) = scratch(1);
        let invalid = |p: &str| {
            matches!(
                ParallelSparseReader::new(p),
                Err(ScDataErrors::V5HeaderInvalid(_))
            )
        };

        small_v5(&s[0]);
        patch_bytes(&s[0], 12, &1u32.to_le_bytes());
        assert!(invalid(&s[0]));

        small_v5(&s[0]);
        patch_bytes(&s[0], 18, &[9]);
        assert!(invalid(&s[0]));

        small_v5(&s[0]);
        patch_bytes(&s[0], 19, &[0]);
        assert!(invalid(&s[0]));

        // the first directory entry is the offsets: make it an unknown kind
        let dir = |p: &str| {
            let b = std::fs::read(p).unwrap();
            le_u64(&b, 48)
        };
        small_v5(&s[0]);
        patch_bytes(&s[0], dir(&s[0]), &99u32.to_le_bytes());
        assert!(invalid(&s[0]), "required unknown section");
        patch_bytes(&s[0], dir(&s[0]) + 4, &0u32.to_le_bytes());
        // optional now, so skipped, and the offsets are missing
        assert!(invalid(&s[0]), "offsets missing");

        small_v5(&s[0]);
        patch_bytes(&s[0], dir(&s[0]) + 8, &u64::MAX.to_le_bytes());
        assert!(invalid(&s[0]), "section outside the file");

        small_v5(&s[0]);
        patch_bytes(&s[0], 48, &u64::MAX.to_le_bytes());
        assert!(invalid(&s[0]), "directory outside the file");
    }

    /// The cell-ID hash is written, set after the fact and compared.
    #[test]
    fn test_v5_cell_ids_align_stores() {
        let (_dir, s) = scratch(2);
        let ids = ["AAAC-1", "AAAG-1"];
        let mut w = CellGeneSparseWriter::new(&s[0], true, 2, 4, TARGET).unwrap();
        w.set_cell_ids(&ids).unwrap();
        for c in 0..2 {
            w.write_cell_chunk(CsrCellChunk::from_data(&[1u32], &[1u32], c, TARGET, true))
                .unwrap();
        }
        w.finalise().unwrap();
        small_v5(&s[1]);

        let (a, b) = (
            ParallelSparseReader::new(&s[0]).unwrap(),
            ParallelSparseReader::new(&s[1]).unwrap(),
        );
        assert_eq!(a.cell_id_hash(), Some(crate::store::v5::cell_id_hash(&ids)));
        assert_eq!(b.cell_id_hash(), None);
        assert!(crate::store::v5::check_same_cells(&[&a, &b]).is_err());

        crate::store::v5::set_cell_ids(&[&s[1]], &ids).unwrap();
        let b = ParallelSparseReader::new(&s[1]).unwrap();
        crate::store::v5::check_same_cells(&[&a, &b]).unwrap();
        assert_eq!(b.read_cells_parallel(&[0, 1]).unwrap().len(), 2);
        assert!(crate::store::v5::set_cell_ids(&[&s[1]], &ids[..1]).is_err());
    }

    /// TF-IDF reads `ln1p(x / lib * scale * n / total_f)` in both orientations.
    #[test]
    fn test_v5_tf_idf_matches_the_formula() {
        let (_dir, s) = scratch(2);
        let (n_cells, n_genes) = (40, 12);
        let [dp, _, _] = allele_layers(n_cells, n_genes);
        let mut w = CellGeneSparseWriter::new(&s[0], true, n_cells, n_genes, 0.0)
            .unwrap()
            .with_kinds(crate::store::v5::FeatureKind::Peak, NormKind::LogCpm)
            .unwrap();
        for (c, row) in dp.iter().enumerate() {
            let idx: Vec<u32> = (0..n_genes as u32)
                .filter(|&g| row[g as usize] > 0)
                .collect();
            let vals: Vec<u32> = idx.iter().map(|&g| row[g as usize]).collect();
            w.write_cell_layers(c, &idx, &[&vals], true).unwrap();
        }
        w.finalise().unwrap();
        crate::store::gene_file::write_gene_file(&s[0], &s[1], None, false).unwrap();

        let genes = ParallelSparseReader::new(&s[1]).unwrap();
        let idf = crate::store::v5::tf_idf_weights(&genes, None).unwrap();
        crate::store::v5::set_tf_idf(&[&s[0], &s[1]], &idf, 1e4).unwrap();

        let lib: Vec<f32> = dp.iter().map(|r| r.iter().sum::<u32>() as f32).collect();
        let total: Vec<f64> = (0..n_genes)
            .map(|g| dp.iter().map(|r| r[g] as f64).sum())
            .collect();
        let want = |c: usize, g: usize| {
            let w = (n_cells as f64 / total[g]) as f32;
            libm::log1p((dp[c][g] as f32 / lib[c] * (1e4 * w)) as f64) as f32
        };

        let cells = ParallelSparseReader::new(&s[0]).unwrap();
        let genes = ParallelSparseReader::new(&s[1]).unwrap();
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
