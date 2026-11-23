//! On-disk chunks of the binary format: raw count storage, the CSR cell and
//! CSC gene chunks with their codecs, the file headers, and assembly of
//! chunks into sparse matrices.

use bincode::{Decode, Encode};
use half::f16;
use half::slice::HalfFloatSliceExt;
use indexmap::IndexSet;
use lz4_flex::compress_prepend_size;
use num_traits::FromPrimitive;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::{io::Write, marker::Sync};

use crate::prelude::*;

//////////////////////
// Dense index maps //
//////////////////////

/// Marker in a dense index map for a cell or gene that is dropped.
pub const INDEX_DROPPED: u32 = u32::MAX;

/// Dense old-to-new map from the kept cell or gene indices.
///
/// Replaces a hash lookup per stored value with an index.
///
/// ### Params
///
/// * `kept` - Old indices to keep, in new-index order
///
/// ### Returns
///
/// `map[old] = new`, [`INDEX_DROPPED`] for indices not kept. Sized to the
/// largest kept index, so callers treat out-of-range as dropped.
pub fn dense_index_map(kept: &[usize]) -> Vec<u32> {
    let mut map = vec![INDEX_DROPPED; kept.iter().max().map_or(0, |&g| g + 1)];
    for (new, &old) in kept.iter().enumerate() {
        map[old] = new as u32;
    }
    map
}

/// Checked dense map for a gene selection over every gene of a store.
///
/// ### Params
///
/// * `genes` - Store genes to keep, in output column order
/// * `n_genes` - Genes in the store
///
/// ### Returns
///
/// `map[gene] = column`, [`INDEX_DROPPED`] for genes not kept, or
/// [`ScDataErrors::InvalidArgument`] naming the first out-of-range or
/// repeated gene.
pub fn gene_subset_map(genes: &[usize], n_genes: usize) -> Result<Vec<u32>, ScDataErrors> {
    let mut map = vec![INDEX_DROPPED; n_genes];
    for (col, &g) in genes.iter().enumerate() {
        let slot = map.get_mut(g).ok_or_else(|| {
            ScDataErrors::InvalidArgument(format!(
                "gene index {g} is out of range for a store of {n_genes} genes"
            ))
        })?;
        if *slot != INDEX_DROPPED {
            return Err(ScDataErrors::InvalidArgument(format!(
                "gene index {g} is repeated"
            )));
        }
        *slot = col as u32;
    }
    Ok(map)
}

/// [`dense_index_map`] over a keep set, rows in the set's iteration order.
///
/// ### Params
///
/// * `cells_to_keep` - Cells to keep, in output order
///
/// ### Returns
///
/// `map[cell] = row`, [`INDEX_DROPPED`] for cells not kept.
pub fn keep_set_index_map(cells_to_keep: &IndexSet<u32>) -> Vec<u32> {
    let kept: Vec<usize> = cells_to_keep.iter().map(|&c| c as usize).collect();
    dense_index_map(&kept)
}

/// Dense gene map from a per-file `local -> final` mapping.
///
/// ### Params
///
/// * `mapping` - `Some(final)` for kept local genes, `None` otherwise
///
/// ### Returns
///
/// `map[local] = final`, [`INDEX_DROPPED`] for dropped genes.
pub fn dense_gene_map_from_options(mapping: &[Option<usize>]) -> Vec<u32> {
    mapping
        .iter()
        .map(|g| g.map_or(INDEX_DROPPED, |g| g as u32))
        .collect()
}

/// Look up an index in a dense index map.
///
/// ### Params
///
/// * `map` - Dense map from [`dense_index_map`] or a sibling
/// * `i` - Old index
///
/// ### Returns
///
/// The new index, or `None` if `i` is dropped or past the end of the map.
#[inline]
pub fn remap(map: &[u32], i: usize) -> Option<u32> {
    match map.get(i) {
        Some(&m) if m != INDEX_DROPPED => Some(m),
        _ => None,
    }
}

///////////////////
// Count wrapper //
///////////////////

/// Wrapper enum for raw counts that supports both u16 and u32 storage.
///
/// Most single cell data sets have counts well within the u16 range (0-65,535).
/// The u32 variant exists for edge cases such as mixed-omics data where counts
/// can exceed this limit. The variant is selected automatically at construction
/// time based on the data, and is encoded as a discriminant byte in the binary
/// chunk format.
#[derive(Encode, Decode, Serialize, Deserialize, Debug, Clone)]
pub enum RawCounts {
    /// Stores u16 values
    U16(Vec<u16>),
    /// Stores u32 values (for the rare case where a single gene exceeds the
    /// counts here)
    U32(Vec<u32>),
}

/// Raw count element size discriminant for the binary format.
/// Stored as a single byte in chunk headers.
pub(crate) const RAW_ELEM_U16: u8 = 2;
/// Raw count element size discriminant for u32 counts.
pub(crate) const RAW_ELEM_U32: u8 = 4;

/// Header length of a CSR cell chunk: 4+4+4+8+8+4 bytes. See
/// [`CsrCellChunk::write_to_bytes`] for the field layout.
const CSR_CHUNK_HEADER_LEN: usize = 32;

/// Header length of a CSC gene chunk: 4+4+4+2+2+8+8+4 bytes. See
/// [`CscGeneChunk::write_to_bytes`] for the field layout.
const CSC_CHUNK_HEADER_LEN: usize = 36;

impl RawCounts {
    /// Get a single value as u32
    ///
    /// ### Returns
    ///
    /// The count as u32
    ///
    /// ### Params
    ///
    /// * `i` - Position
    pub fn get(&self, i: usize) -> u32 {
        match self {
            RawCounts::U16(v) => v[i] as u32,
            RawCounts::U32(v) => v[i],
        }
    }

    /// Total number of elements
    ///
    /// ### Returns
    ///
    /// The length of elements
    pub fn len(&self) -> usize {
        match self {
            RawCounts::U16(v) => v.len(),
            RawCounts::U32(v) => v.len(),
        }
    }

    /// Whether the vector is empty
    ///
    /// ### Returns
    ///
    /// Boolean if the data is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterator that yields all values as u32
    ///
    /// ### Returns
    ///
    /// An iterator
    pub fn iter(&self) -> RawCountsIter<'_> {
        RawCountsIter {
            inner: self,
            pos: 0,
        }
    }

    /// Element size in bytes (2 for u16, 4 for u32)
    ///
    /// ### Returns
    ///
    /// The element size
    pub fn elem_size(&self) -> u8 {
        match self {
            RawCounts::U16(_) => RAW_ELEM_U16,
            RawCounts::U32(_) => RAW_ELEM_U32,
        }
    }

    /// Write raw bytes to a writer
    ///
    /// ### Params
    ///
    /// * `writer` - The writer to use
    ///
    /// ### Returns
    ///
    /// The I/O result
    pub fn write_bytes(&self, writer: &mut impl Write) -> std::io::Result<()> {
        // Little-endian throughout, matching every header field in the format.
        match self {
            RawCounts::U16(v) => {
                let mut bytes = Vec::with_capacity(v.len() * 2);
                for x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                writer.write_all(&bytes)
            }
            RawCounts::U32(v) => {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for x in v {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                writer.write_all(&bytes)
            }
        }
    }

    /// Read raw counts from a buffer given element size and count
    ///
    /// ### Params
    ///
    /// * `buffer` - The byte buffer to read from (must be positioned at the
    ///   start of the raw count data)
    /// * `count` - Number of elements to read
    /// * `elem_size` - Element size in bytes (2 for u16, 4 for u32)
    ///
    /// ### Returns
    ///
    /// The `RawCounts`, or [`ScDataErrors::RawElemSizeInvalid`] if the
    /// discriminant is unrecognised, or [`ScDataErrors::ChunkPayloadTruncated`]
    /// if the buffer is shorter than `count * elem_size`.
    pub fn read_from_buffer(
        buffer: &[u8],
        count: usize,
        elem_size: u8,
    ) -> Result<Self, ScDataErrors> {
        // Zero covers legacy files where the discriminant byte was padding.
        let width = match elem_size {
            RAW_ELEM_U32 => 4_usize,
            RAW_ELEM_U16 | 0 => 2_usize,
            other => return Err(ScDataErrors::RawElemSizeInvalid(other)),
        };

        let needed = count * width;
        if buffer.len() < needed {
            return Err(ScDataErrors::ChunkPayloadTruncated {
                expected: needed,
                found: buffer.len(),
            });
        }

        Ok(if width == 4 {
            RawCounts::U32(
                buffer[..needed]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| u32::from_le_bytes(c))
                    .collect(),
            )
        } else {
            RawCounts::U16(
                buffer[..needed]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&c| u16::from_le_bytes(c))
                    .collect(),
            )
        })
    }

    /// Build RawCounts from a u32 slice, auto-selecting the variant.
    ///
    /// If all values fit in u16, stores as U16 for better compression.
    /// Otherwise stores as U32.
    ///
    /// ### Params
    ///
    /// * `data` - The counts
    ///
    /// ### Returns
    ///
    /// The counts as U16 when they all fit, U32 otherwise.
    pub fn from_u32_auto(data: &[u32]) -> Self {
        if data.iter().all(|&v| v <= u16::MAX as u32) {
            RawCounts::U16(data.iter().map(|&v| v as u16).collect())
        } else {
            RawCounts::U32(data.to_vec())
        }
    }
}

/// Structure of the RawCountsIterator
pub struct RawCountsIter<'a> {
    /// The reference to the row counts
    inner: &'a RawCounts,
    /// The position
    pos: usize,
}

impl<'a> Iterator for RawCountsIter<'a> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.pos < self.inner.len() {
            let val = self.inner.get(self.pos);
            self.pos += 1;
            Some(val)
        } else {
            None
        }
    }
}

//////////////////
// CsrCellChunk //
//////////////////

/// The `data_norm` value of one count: `ln1p(x / lib_size * target_size)`.
///
/// The single definition of the normalised layer. v4 and the archive drop the
/// norm and recompute it with this on read, so any change here changes what
/// every such store returns.
///
/// ### Params
///
/// * `x` - Raw count
/// * `lib_size` - Library size of the cell
/// * `target_size` - Target size of the normalisation
///
/// ### Returns
///
/// The normalised value.
pub fn norm_value(x: f32, lib_size: f32, target_size: f32) -> f32 {
    // libm in f64, not the platform `ln_1p`: the layer is derived on read, so
    // every OS must compute the same bits, and rounding the f64 result gives
    // the correctly rounded f32 in all but the rarest cases
    libm::log1p((x / lib_size * target_size) as f64) as f32
}

//////////////////
// ChunkPayload //
//////////////////

/// Borrowed view over the three payload regions of one decompressed chunk.
///
/// Cell and gene chunks differ only in their headers; the payload is the same
/// raw / norm / index run in both. This is the single parser for that run: the
/// owned chunk types build their `Vec`s from it, and the fused batch readers
/// walk it in place without allocating per chunk. Reads are byte-wise rather
/// than pointer casts: the buffer is an lz4-decompressed `Vec<u8>`, so no
/// region is guaranteed to be aligned.
pub(crate) struct ChunkPayload<'a> {
    /// Raw count bytes, at `raw_width` bytes per element.
    data_raw: &'a [u8],
    /// Number of raw count elements.
    data_raw_len: usize,
    /// Element width discriminant for `data_raw`, as stored in the header.
    raw_elem_size: u8,
    /// Bytes per raw count element, 2 or 4.
    raw_width: usize,
    /// Normalised count bytes, two per element.
    data_norm: &'a [u8],
    /// Index bytes, four per element.
    indices: &'a [u8],
}

impl<'a> ChunkPayload<'a> {
    /// Slice the payload regions that follow a chunk header.
    ///
    /// ### Params
    ///
    /// * `buffer` - One decompressed chunk, header included
    /// * `header_len` - Length of the header the payload follows
    /// * `lens` - The `(raw, norm, index)` element counts from the header
    /// * `raw_elem_size` - Raw element width discriminant from the header
    ///
    /// ### Returns
    ///
    /// The payload, [`ScDataErrors::RawElemSizeInvalid`] for an unknown
    /// width, or [`ScDataErrors::ChunkPayloadTruncated`] for a chunk shorter
    /// than its own header claims.
    fn slice(
        buffer: &'a [u8],
        header_len: usize,
        lens: (usize, usize, usize),
        raw_elem_size: u8,
    ) -> Result<Self, ScDataErrors> {
        let (data_raw_len, data_norm_len, indices_len) = lens;

        // Zero covers legacy files where the discriminant byte was padding.
        let raw_width = match raw_elem_size {
            RAW_ELEM_U32 => 4,
            RAW_ELEM_U16 | 0 => 2,
            other => return Err(ScDataErrors::RawElemSizeInvalid(other)),
        };

        let data_end = header_len + data_raw_len * raw_width;
        let norm_end = data_end + data_norm_len * 2;
        let total_end = norm_end + indices_len * 4;

        // The three length fields are untrusted, so validate the total payload
        // before taking any slice.
        if buffer.len() < total_end {
            return Err(ScDataErrors::ChunkPayloadTruncated {
                expected: total_end,
                found: buffer.len(),
            });
        }

        Ok(Self {
            data_raw: &buffer[header_len..data_end],
            data_raw_len,
            raw_elem_size,
            raw_width,
            data_norm: &buffer[data_end..norm_end],
            indices: &buffer[norm_end..total_end],
        })
    }

    /// Iterate the normalised layer.
    ///
    /// ### Returns
    ///
    /// An iterator over the stored f16 values, widened to f32.
    fn norm_iter(&self) -> impl Iterator<Item = f32> + '_ {
        self.data_norm
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| F16::from_le_bytes(c).to_f32())
    }

    /// Iterate the indices.
    ///
    /// ### Returns
    ///
    /// An iterator over the stored indices.
    fn index_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.indices
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
    }

    /// Copy the raw layer out as owned [`RawCounts`].
    ///
    /// ### Returns
    ///
    /// The raw counts at their stored width.
    fn raw_counts(&self) -> Result<RawCounts, ScDataErrors> {
        RawCounts::read_from_buffer(self.data_raw, self.data_raw_len, self.raw_elem_size)
    }
}

///////////////
// RunSource //
///////////////

/// Entries per stack block when widening f16 to f32 in bulk. Big enough for
/// `half`'s slice conversion to run its vector path, small enough to stay on
/// the stack.
const F16_BLOCK: usize = 64;

/// One stored run (a cell row or a gene column) as the fused readers consume
/// it: random access for the filtered path, bulk extends for the unfiltered
/// one.
pub(crate) trait RunSource {
    /// Number of stored entries.
    ///
    /// ### Returns
    ///
    /// The run length.
    fn nnz(&self) -> usize;

    /// Stored index of entry `i`.
    ///
    /// ### Params
    ///
    /// * `i` - Entry position
    ///
    /// ### Returns
    ///
    /// The gene (cell run) or cell (gene run) index.
    fn index_at(&self, i: usize) -> u32;

    /// Raw count of entry `i`.
    ///
    /// ### Params
    ///
    /// * `i` - Entry position
    ///
    /// ### Returns
    ///
    /// The count.
    fn raw_at(&self, i: usize) -> u32;

    /// Normalised value of entry `i`.
    ///
    /// ### Params
    ///
    /// * `i` - Entry position
    ///
    /// ### Returns
    ///
    /// The value as f32.
    fn norm_at(&self, i: usize) -> f32;

    /// Append every stored index.
    ///
    /// ### Params
    ///
    /// * `out` - Destination
    fn extend_indices(&self, out: &mut Vec<u32>);

    /// Append every raw count, converted to `T`.
    ///
    /// ### Params
    ///
    /// * `out` - Destination
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::RawCountOverflow`] for a count that does
    /// not fit `T`. Nothing is appended on error.
    fn extend_raw<T: FromPrimitive>(&self, out: &mut Vec<T>) -> Result<(), ScDataErrors>;

    /// Append every normalised value.
    ///
    /// ### Params
    ///
    /// * `out` - Destination
    fn extend_norm(&self, out: &mut Vec<f32>);
}

/// Convert one raw count exactly into `T`.
///
/// ### Params
///
/// * `val` - The stored count
///
/// ### Returns
///
/// The count as `T`, or [`ScDataErrors::RawCountOverflow`] when it does not
/// fit.
#[inline(always)]
pub(crate) fn raw_to<T: FromPrimitive>(val: u32) -> Result<T, ScDataErrors> {
    T::from_u32(val).ok_or(ScDataErrors::RawCountOverflow {
        value: val,
        target_type: std::any::type_name::<T>(),
    })
}

/// Append raw counts converted to `T`, checking overflow once on the maximum.
///
/// Every `FromPrimitive` target used for counts (u16, u32, f32, f64) is
/// monotone over `u32`, so if the maximum fits, every value does. That keeps
/// the branch out of the copy loop.
///
/// ### Params
///
/// * `out` - Destination
/// * `vals` - The counts, iterated twice
///
/// ### Returns
///
/// `Ok(())`, or [`ScDataErrors::RawCountOverflow`] for the maximum.
#[inline]
pub(crate) fn extend_raw_checked<T, I>(out: &mut Vec<T>, vals: I) -> Result<(), ScDataErrors>
where
    T: FromPrimitive,
    I: Iterator<Item = u32> + Clone,
{
    raw_to::<T>(vals.clone().max().unwrap_or(0))?;
    out.extend(vals.map(|v| T::from_u32(v).expect("the maximum fits, so every value does")));
    Ok(())
}

impl RunSource for ChunkPayload<'_> {
    #[inline]
    fn nnz(&self) -> usize {
        self.indices.len() / 4
    }

    #[inline(always)]
    fn index_at(&self, i: usize) -> u32 {
        let b = &self.indices[i * 4..i * 4 + 4];
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    #[inline(always)]
    fn raw_at(&self, i: usize) -> u32 {
        if self.raw_width == 4 {
            let b = &self.data_raw[i * 4..i * 4 + 4];
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            let b = &self.data_raw[i * 2..i * 2 + 2];
            u16::from_le_bytes([b[0], b[1]]) as u32
        }
    }

    #[inline(always)]
    fn norm_at(&self, i: usize) -> f32 {
        let b = &self.data_norm[i * 2..i * 2 + 2];
        F16::from_le_bytes([b[0], b[1]]).to_f32()
    }

    fn extend_indices(&self, out: &mut Vec<u32>) {
        out.extend(self.index_iter());
    }

    fn extend_raw<T: FromPrimitive>(&self, out: &mut Vec<T>) -> Result<(), ScDataErrors> {
        if self.raw_width == 4 {
            extend_raw_checked(
                out,
                self.data_raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| u32::from_le_bytes(c)),
            )
        } else {
            extend_raw_checked(
                out,
                self.data_raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&c| u16::from_le_bytes(c) as u32),
            )
        }
    }

    fn extend_norm(&self, out: &mut Vec<f32>) {
        let start = out.len();
        out.resize(start + self.data_norm.len() / 2, 0.0);
        let mut block = [f16::ZERO; F16_BLOCK];
        for (src, dst) in self
            .data_norm
            .chunks(2 * F16_BLOCK)
            .zip(out[start..].chunks_mut(F16_BLOCK))
        {
            for (b, &c) in block.iter_mut().zip(src.as_chunks::<2>().0) {
                *b = f16::from_le_bytes(c);
            }
            block[..dst.len()].convert_to_f32_slice(dst);
        }
    }
}

//////////////////
// CsrChunkView //
//////////////////

/// Borrowed view over one decompressed CSR cell chunk.
///
/// The cell header fields plus the shared [`ChunkPayload`]. Every payload
/// slice is bounds-checked against the buffer before it is taken, so the three
/// untrusted length fields in the header cannot index out of range.
struct CsrChunkView<'a> {
    /// The raw / norm / gene index regions.
    payload: ChunkPayload<'a>,
    /// Total library size of the cell.
    library_size: usize,
    /// Original index of the cell in the experiment.
    original_index: usize,
    /// Whether the cell passed QC.
    to_keep: bool,
}

impl<'a> CsrChunkView<'a> {
    /// Parse the chunk header and slice out the three payload regions.
    ///
    /// ### Params
    ///
    /// * `buffer` - One decompressed chunk
    ///
    /// ### Returns
    ///
    /// The view, or [`ScDataErrors::ChunkBufferTooSmall`] /
    /// [`ScDataErrors::ChunkPayloadTruncated`] for a chunk shorter than its
    /// own header claims.
    fn parse(buffer: &'a [u8]) -> Result<Self, ScDataErrors> {
        if buffer.len() < CSR_CHUNK_HEADER_LEN {
            return Err(ScDataErrors::ChunkBufferTooSmall {
                expected: CSR_CHUNK_HEADER_LEN,
                found: buffer.len(),
            });
        }

        let header = &buffer[0..CSR_CHUNK_HEADER_LEN];
        let lens = (
            le_u32(header, 0) as usize,
            le_u32(header, 4) as usize,
            le_u32(header, 8) as usize,
        );

        Ok(Self {
            payload: ChunkPayload::slice(buffer, CSR_CHUNK_HEADER_LEN, lens, header[29])?,
            library_size: le_u64(header, 12) as usize,
            original_index: le_u64(header, 20) as usize,
            to_keep: header[28] != 0,
        })
    }
}

/// Parse a CSR cell chunk down to its payload.
///
/// ### Params
///
/// * `buffer` - One decompressed cell chunk
///
/// ### Returns
///
/// The payload, or the errors of [`CsrChunkView::parse`].
pub(crate) fn csr_payload(buffer: &[u8]) -> Result<ChunkPayload<'_>, ScDataErrors> {
    CsrChunkView::parse(buffer).map(|view| view.payload)
}

/// Parse a CSC gene chunk down to its payload.
///
/// The gene header fields (`nnz`, `original_index`, `to_keep`) are
/// not needed by the fused readers, so they are skipped rather than decoded.
///
/// ### Params
///
/// * `buffer` - One decompressed gene chunk
///
/// ### Returns
///
/// The payload, or [`ScDataErrors::ChunkBufferTooSmall`] /
/// [`ScDataErrors::ChunkPayloadTruncated`] for a chunk shorter than its own
/// header claims.
pub(crate) fn csc_payload(buffer: &[u8]) -> Result<ChunkPayload<'_>, ScDataErrors> {
    if buffer.len() < CSC_CHUNK_HEADER_LEN {
        return Err(ScDataErrors::ChunkBufferTooSmall {
            expected: CSC_CHUNK_HEADER_LEN,
            found: buffer.len(),
        });
    }

    let lens = (
        le_u32(buffer, 0) as usize,
        le_u32(buffer, 4) as usize,
        le_u32(buffer, 8) as usize,
    );

    ChunkPayload::slice(buffer, CSC_CHUNK_HEADER_LEN, lens, buffer[33])
}

/// CsrCellChunk
///
/// This structure is designed to store the data of a single cell in a
/// CSR-like format optimised for rapid access on disk.
///
/// Raw counts are stored as a `RawCounts` enum which transparently handles
/// both u16 and u32 storage. Gene indices are stored as u32 to support
/// data sets with more than 65,535 features.
#[derive(Debug)]
pub struct CsrCellChunk {
    /// Raw counts for this cell, stored as either u16 or u32 depending on
    /// the data range
    pub data_raw: RawCounts,
    /// Normalised counts of this cell.
    pub data_norm: Vec<f32>,
    /// Total library size/UMI counts of the cell.
    pub library_size: usize,
    /// Index positions of the genes (u32 to support >65k features)
    pub indices: Vec<u32>,
    /// Original index in the data
    pub original_index: usize,
    /// Flag if the cell should be kept. (Not used at the moment.)
    pub to_keep: bool,
}

impl CsrCellChunk {
    /// Function to generate the chunk from R data
    ///
    /// Assumes columns = genes, rows = cells.
    ///
    /// ### Params
    ///
    /// * `data` - The raw counts present in this cell
    /// * `col_idx` - The column indices where the gene is expressed.
    /// * `original_index` - Original row index in the matrix
    /// * `size_factor` - To which size to normalise to. 1e6 -> CPM normalisation.
    /// * `to_keep` - Whether the cell passes QC
    ///
    /// ### Returns
    ///
    /// The `CsrCellChunk` for this cell.
    pub fn from_data<T, U>(
        data: &[T],
        col_idx: &[U],
        original_index: usize,
        size_factor: f32,
        to_keep: bool,
    ) -> Self
    where
        T: FloatAndUInt,
        U: FloatAndUInt,
    {
        // one sequential f32 accumulator: the sum feeds the normalisation, so
        // its rounding is part of the stored values
        let sum = data.iter().map(|&x| x.to_f32()).sum::<f32>();
        let data_norm: Vec<f32> = data
            .iter()
            .map(|&x| norm_value(x.to_f32(), sum, size_factor))
            .collect();

        let data_raw = if data.iter().all(|&x| x.to_u32() <= u16::MAX as u32) {
            RawCounts::U16(data.iter().map(|&x| x.to_u32() as u16).collect())
        } else {
            RawCounts::U32(data.iter().map(|&x| x.to_u32()).collect())
        };

        Self {
            data_raw,
            data_norm,
            library_size: sum as usize,
            indices: col_idx.iter().map(|&x| x.to_u32()).collect::<Vec<u32>>(),
            original_index,
            to_keep,
        }
    }

    /// Write directly to bytes on disk
    ///
    /// Binary layout (header = 32 bytes):
    ///
    /// ```text
    /// [0..4)   data_raw_len    u32
    /// [4..8)   data_norm_len   u32
    /// [8..12)  indices_len     u32
    /// [12..20) library_size    u64
    /// [20..28) original_index  u64
    /// [28]     to_keep         u8
    /// [29]     raw_elem_size   u8  (2 = u16, 4 = u32)
    /// [30..32) padding         2 bytes
    /// ```
    ///
    /// Followed by:
    /// - data_raw: `data_raw_len * raw_elem_size` bytes
    /// - data_norm: `data_norm_len * 2` bytes
    /// - indices: `indices_len * 4` bytes (u32)
    ///
    /// ### Params
    ///
    /// * `writer` - Destination
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the I/O error.
    pub fn write_to_bytes(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.to_bytes())
    }

    /// Serialise into one exactly sized buffer, in the layout of
    /// [`Self::write_to_bytes`].
    ///
    /// ### Returns
    ///
    /// The serialised chunk.
    fn to_bytes(&self) -> Vec<u8> {
        let raw_bytes = self.data_raw.len() * self.data_raw.elem_size() as usize;
        let mut buffer = Vec::with_capacity(
            CSR_CHUNK_HEADER_LEN + raw_bytes + self.data_norm.len() * 2 + self.indices.len() * 4,
        );

        buffer.extend_from_slice(&(self.data_raw.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.data_norm.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.indices.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.library_size as u64).to_le_bytes());
        buffer.extend_from_slice(&(self.original_index as u64).to_le_bytes());
        buffer.extend_from_slice(&[self.to_keep as u8, self.data_raw.elem_size(), 0, 0]);

        match &self.data_raw {
            RawCounts::U16(v) => {
                for x in v {
                    buffer.extend_from_slice(&x.to_le_bytes());
                }
            }
            RawCounts::U32(v) => {
                for x in v {
                    buffer.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        // v3 keeps the norm as f16 on disk
        for &v in &self.data_norm {
            buffer.extend_from_slice(&F16::from_f32(v).to_le_bytes());
        }
        for i in &self.indices {
            buffer.extend_from_slice(&i.to_le_bytes());
        }

        buffer
    }

    /// Serialise and lz4-compress the chunk into a v3 payload.
    ///
    /// Only a v3 writer takes this; v4 and v5 writers take
    /// [`crate::store::v4::ChunkFormat::encode_cell`], which dispatches here for v3.
    ///
    /// ### Returns
    ///
    /// The compressed payload, size-prepended as the v3 reader expects.
    pub(crate) fn to_compressed_bytes(&self) -> std::io::Result<Vec<u8>> {
        Ok(compress_prepend_size(&self.to_bytes()))
    }

    /// Read data from buffer
    ///
    /// ### Params
    ///
    /// * `buffer` - A slice of u8's representing the buffer
    ///
    /// ### Returns
    ///
    /// The `CsrCellChunk`
    pub fn read_from_buffer(buffer: &[u8]) -> Result<Self, ScDataErrors> {
        let view = CsrChunkView::parse(buffer)?;

        Ok(Self {
            data_raw: view.payload.raw_counts()?,
            data_norm: view.payload.norm_iter().collect(),
            library_size: view.library_size,
            indices: view.payload.index_iter().collect(),
            original_index: view.original_index,
            to_keep: view.to_keep,
        })
    }

    /// Generate a vector of Chunks from CompressedSparseData2
    ///
    /// ### Params
    ///
    /// * `sparse_data` - The `CompressedSparseData2` (in CSR format!)
    /// * `cell_qc` - Minimum QC thresholds
    ///
    /// ### Returns
    ///
    /// A tuple of the `Vec<CsrCellChunk>` and if the cell should be kept.
    pub fn generate_chunks_sparse_data<T, U>(
        sparse_data: CompressedSparseData2<T, U>,
        cell_qc: MinCellQuality,
    ) -> (Vec<CsrCellChunk>, CellQuality)
    where
        T: Clone + Default + Into<u32> + Sync,
        U: Clone + Default,
    {
        let n_cells = sparse_data.indptr.len() - 1;
        let n_genes = sparse_data.shape.1;

        // count how many cells express each gene
        let mut no_cells_exp_gene = vec![0usize; n_genes];
        for i in 0..n_cells {
            let start_i = sparse_data.indptr[i] as usize;
            let end_i = sparse_data.indptr[i + 1] as usize;
            for &gene_idx in &sparse_data.indices[start_i..end_i] {
                no_cells_exp_gene[gene_idx as usize] += 1;
            }
        }

        // filter genes
        let genes_to_keep: Vec<usize> = (0..n_genes)
            .filter(|&i| no_cells_exp_gene[i] >= cell_qc.min_cells)
            .collect();

        let gene_map = dense_index_map(&genes_to_keep);

        // Process cells
        let results: Vec<_> = (0..n_cells)
            .into_par_iter()
            .map(|i| {
                let start_i = sparse_data.indptr[i] as usize;
                let end_i = sparse_data.indptr[i + 1] as usize;

                let mut filtered_data = Vec::new();
                let mut filtered_indices = Vec::new();

                for idx in start_i..end_i {
                    if let Some(new_gene_idx) = remap(&gene_map, sparse_data.indices[idx] as usize)
                    {
                        filtered_data.push(sparse_data.data[idx].clone().into());
                        filtered_indices.push(new_gene_idx);
                    }
                }

                let sum_data_i = filtered_data.iter().sum::<u32>() as usize;
                let nnz_i = filtered_indices.len();

                let to_keep_i =
                    (nnz_i >= cell_qc.min_unique_genes) && (sum_data_i >= cell_qc.min_lib_size);

                let chunk_i = CsrCellChunk::from_data(
                    &filtered_data,
                    &filtered_indices,
                    i,
                    cell_qc.target_size,
                    to_keep_i,
                );

                (chunk_i, sum_data_i, nnz_i, to_keep_i, i)
            })
            .collect();

        let mut res = Vec::with_capacity(n_cells);
        let mut lib_size = Vec::new();
        let mut nnz = Vec::new();
        let mut cells_to_keep = Vec::new();

        for (chunk, lib, n, keep, idx) in results {
            res.push(chunk);
            if keep {
                lib_size.push(lib);
                nnz.push(n);
                cells_to_keep.push(idx);
            }
        }

        let qc_data = CellQuality {
            cell_indices: cells_to_keep,
            gene_indices: genes_to_keep,
            lib_size,
            nnz,
        };

        (res, qc_data)
    }

    /// Helper function to get QC parameters for this cell
    ///
    /// ### Returns
    ///
    /// A tuple of `(no_genes, library_size)`
    pub fn get_qc_info(&self) -> (usize, usize) {
        (self.indices.len(), self.library_size)
    }

    /// Filter to keep only specified genes
    ///
    /// This will remove the genes that are not set to `true` in the boolean
    /// vector. This assumes that the indices of the genes and the booleans
    /// are the same!
    ///
    /// ### Params
    ///
    /// * `genes_to_keep` - Keep only the genes within this boolean vectors.
    pub fn filter_genes(&mut self, genes_to_keep: &[bool]) {
        let kept: Vec<usize> = (0..genes_to_keep.len())
            .filter(|&i| genes_to_keep[i])
            .collect();
        let gene_map = dense_index_map(&kept);

        let mut new_indices = Vec::new();
        let mut new_norm = Vec::new();
        let mut keep_positions = Vec::new();

        for (i, &gene_idx) in self.indices.iter().enumerate() {
            if let Some(new_gene_idx) = remap(&gene_map, gene_idx as usize) {
                new_indices.push(new_gene_idx);
                new_norm.push(self.data_norm[i]);
                keep_positions.push(i);
            }
        }

        let new_raw = match &self.data_raw {
            RawCounts::U16(v) => RawCounts::U16(keep_positions.iter().map(|&p| v[p]).collect()),
            RawCounts::U32(v) => RawCounts::U32(keep_positions.iter().map(|&p| v[p]).collect()),
        };

        self.indices = new_indices;
        self.data_raw = new_raw;
        self.data_norm = new_norm;
    }
}

//////////////////
// CscGeneChunk //
//////////////////

/// CscGeneChunk
///
/// This structure is designed to store the data of a single gene in a
/// CSC-like format optimised for rapid access on disk.
///
/// Raw counts are stored as a `RawCounts` enum (u16 or u32). Cell indices
/// are stored as u32.
#[derive(Encode, Decode, Serialize, Deserialize, Debug)]
pub struct CscGeneChunk {
    /// Raw counts per cell/spot for this gene, stored as either u16 or u32
    pub data_raw: RawCounts,
    /// Normalised counts per cell/spot for this gene.
    pub data_norm: Vec<f32>,
    /// Number of cells expressing this gene
    pub nnz: usize,
    /// Indices of the cells expressing this gene
    pub indices: Vec<u32>,
    /// Original index from the data
    pub original_index: usize,
    /// Flag to indicate if gene shall be kept. Not in use at the moment.
    pub to_keep: bool,
}

impl CscGeneChunk {
    /// Helper function to generate the CscGeneChunk from converted data
    ///
    /// ### Params
    ///
    /// * `data_raw` - The raw counts for this gene
    /// * `data_norm` - The normalised counts for this gene.
    /// * `col_idx` - The column indices for which cells this gene is expressed.
    /// * `original_index` - Original row index
    /// * `to_keep` - Shall the gene be included in later analysis.
    ///
    /// ### Returns
    ///
    /// The `CscGeneChunk` for this gene.
    pub fn from_conversion(
        data_raw: RawCounts,
        data_norm: &[f32],
        col_idx: &[usize],
        original_index: usize,
        to_keep: bool,
    ) -> Self {
        let nnz = data_raw.len();

        Self {
            data_raw,
            data_norm: data_norm.to_vec(),
            nnz,
            indices: col_idx.iter().map(|x| *x as u32).collect::<Vec<u32>>(),
            original_index,
            to_keep,
        }
    }

    /// Write directly to bytes on disk
    ///
    /// Binary layout (header = 36 bytes):
    ///
    /// ```text
    /// [0..4)   data_raw_len    u32
    /// [4..8)   data_norm_len   u32
    /// [8..12)  indices_len     u32
    /// [12..16) unused          4 bytes, zero (v3 kept a norm sum here)
    /// [16..24) nnz             u64
    /// [24..32) original_index  u64
    /// [32]     to_keep         u8
    /// [33]     raw_elem_size   u8  (2 = u16, 4 = u32)
    /// [34..36) padding         2 bytes
    /// ```
    ///
    /// Followed by:
    /// - data_raw: `data_raw_len * raw_elem_size` bytes
    /// - data_norm: `data_norm_len * 2` bytes
    /// - indices: `indices_len * 4` bytes (u32)
    ///
    /// ### Params
    ///
    /// * `writer` - Destination
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the I/O error.
    pub fn write_to_bytes(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&(self.data_raw.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.data_norm.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.indices.len() as u32).to_le_bytes())?;
        writer.write_all(&[0; 4])?;
        writer.write_all(&(self.nnz as u64).to_le_bytes())?;
        writer.write_all(&(self.original_index as u64).to_le_bytes())?;
        writer.write_all(&[self.to_keep as u8, self.data_raw.elem_size(), 0, 0])?;

        self.data_raw.write_bytes(writer)?;

        // v3 keeps the norm as f16 on disk
        let mut data_norm_bytes = Vec::with_capacity(self.data_norm.len() * 2);
        for &v in &self.data_norm {
            data_norm_bytes.extend_from_slice(&F16::from_f32(v).to_le_bytes());
        }
        writer.write_all(&data_norm_bytes)?;

        let mut row_indices_bytes = Vec::with_capacity(self.indices.len() * 4);
        for i in &self.indices {
            row_indices_bytes.extend_from_slice(&i.to_le_bytes());
        }
        writer.write_all(&row_indices_bytes)?;

        Ok(())
    }

    /// Serialise and lz4-compress the chunk into a v3 payload.
    ///
    /// Only a v3 writer takes this; v4 and v5 writers take
    /// [`crate::store::v4::ChunkFormat::encode_gene`], which dispatches here for v3.
    ///
    /// ### Returns
    ///
    /// The compressed payload, size-prepended as the v3 reader expects.
    pub(crate) fn to_compressed_bytes(&self) -> std::io::Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.write_to_bytes(&mut buffer)?;
        Ok(compress_prepend_size(&buffer))
    }

    /// Read data from buffer
    ///
    /// ### Params
    ///
    /// * `buffer` - A slice of u8's representing the buffer
    ///
    /// ### Returns
    ///
    /// The `CscGeneChunk`
    pub fn read_from_buffer(buffer: &[u8]) -> Result<Self, ScDataErrors> {
        let payload = csc_payload(buffer)?;

        Ok(Self {
            data_raw: payload.raw_counts()?,
            data_norm: payload.norm_iter().collect(),
            nnz: le_u64(buffer, 16) as usize,
            indices: payload.index_iter().collect(),
            original_index: le_u64(buffer, 24) as usize,
            to_keep: buffer[32] != 0,
        })
    }

    /// Helper function that allows to filter out cells
    ///
    /// Kept cells are renumbered to their position in `cells_to_keep`, and the
    /// output is ordered by that position (PCA and NEBULA rely on it).
    ///
    /// ### Params
    ///
    /// * `cells_to_keep` - IndexSet with cell index positions to keep.
    pub fn filter_selected_cells(&mut self, cells_to_keep: &IndexSet<u32>) {
        self.filter_cells_by(|cell| cells_to_keep.get_index_of(&cell).map(|i| i as u32));
    }

    /// Filter to the cells a dense index map keeps.
    ///
    /// Same result as [`Self::filter_selected_cells`], but each stored entry
    /// costs an array load rather than a hash probe. Build the map once per
    /// read with [`dense_index_map`] and share it across genes.
    ///
    /// ### Params
    ///
    /// * `cell_map` - `cell_map[cell] = output row`, [`INDEX_DROPPED`] or out
    ///   of range for cells not kept.
    pub fn filter_selected_cells_dense(&mut self, cell_map: &[u32]) {
        self.filter_cells_by(|cell| remap(cell_map, cell as usize));
    }

    /// Keep the entries whose cell maps to an output row, renumbered to it.
    ///
    /// One walk over this gene's non-zeros. Stored cell indices ascend, so the
    /// kept rows only need sorting when the keep set itself is not ascending.
    ///
    /// ### Params
    ///
    /// * `row_of` - Output row for a cell, `None` to drop it.
    #[inline]
    fn filter_cells_by(&mut self, row_of: impl Fn(u32) -> Option<u32>) {
        let mut kept: Vec<(u32, u32)> = self
            .indices
            .iter()
            .enumerate()
            .filter_map(|(pos, &cell)| row_of(cell).map(|row| (row, pos as u32)))
            .collect();

        if !kept.is_sorted_by_key(|&(row, _)| row) {
            kept.sort_unstable_by_key(|&(row, _)| row);
        }

        self.data_norm = kept
            .iter()
            .map(|&(_, p)| self.data_norm[p as usize])
            .collect();
        self.data_raw = match &self.data_raw {
            RawCounts::U16(v) => RawCounts::U16(kept.iter().map(|&(_, p)| v[p as usize]).collect()),
            RawCounts::U32(v) => RawCounts::U32(kept.iter().map(|&(_, p)| v[p as usize]).collect()),
        };
        self.indices = kept.iter().map(|&(row, _)| row).collect();
        self.nnz = kept.len();
    }

    /// Calculate the average gene expression given a set of cells
    ///
    /// ### Params
    ///
    /// * `cells_to_keep` - IndexSet with cell index positions to keep.
    ///
    /// ### Returns
    ///
    /// Tuple of `(gene_index, average expression)`.
    pub fn calculate_avg_exp(&self, cells_to_keep: &IndexSet<u32>) -> (usize, f32) {
        let sum: f32 = self
            .indices
            .iter()
            .zip(self.data_norm.iter())
            .filter(|(cell_idx, _)| cells_to_keep.contains(*cell_idx))
            .map(|(_, &val)| val)
            .sum();
        let avg = sum / cells_to_keep.len() as f32;

        (self.original_index, avg)
    }
}

/// SparseDataHeader
///
/// Stores the information in terms of total cells, total genes, number of
/// chunks in terms of cells and genes and the offset vectors
#[derive(Encode, Decode, Serialize, Deserialize, Clone)]
pub struct SparseDataHeader {
    /// Total number of cells in the experiment.
    pub total_cells: usize,
    /// Total number of genes in the experiemnt.
    pub total_genes: usize,
    /// Is the file written in a way for fast cell data retrieval (set to true)
    /// or for fast gene retrieval (set to false)
    pub cell_based: bool,
    /// Number of chunks in this file storing either the cell or gene data
    pub no_chunks: usize,
    /// Offset vector for reading in the data
    pub chunk_offsets: Vec<u64>,
    /// FxHashMap with the original index -> chunk info
    pub index_map: FxHashMap<usize, usize>,
}

/// Leading bytes identifying a bixverse single-cell binary. Checked on open so
/// a foreign file is refused rather than parsed as one of ours.
pub(crate) const FILE_MAGIC: &[u8; 8] = b"SCRNASEQ";

/// The little-endian `u32` at `at`.
///
/// ### Params
///
/// * `b` - The bytes, at least `at + 4` long
/// * `at` - Offset
///
/// ### Returns
///
/// The value.
#[inline]
pub(crate) fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4-byte slice"))
}

/// The little-endian `u64` at `at`.
///
/// ### Params
///
/// * `b` - The bytes, at least `at + 8` long
/// * `at` - Offset
///
/// ### Returns
///
/// The value.
#[inline]
pub(crate) fn le_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8-byte slice"))
}

/// Fixed-size file header that points to the main header location
#[repr(C)]
#[derive(Encode, Decode, Serialize, Deserialize)]
pub(crate) struct FileHeader {
    /// Magic string as bytes to recognise the file
    pub(crate) magic: [u8; 8],
    /// Version of the file
    pub(crate) version: u32,
    /// Offset of the main header, i.e., 64 bytes
    pub(crate) main_header_offset: u64,
    /// Is cell-based (CSR format)
    pub(crate) cell_based: bool,
    /// Library size the `data_norm` layer was normalised against. Carved out
    /// of the reserved block, so files predating this field decode it as
    /// `0.0`, which is the "unknown" sentinel (a real target size is never 0).
    pub(crate) target_size: f32,
    /// Offset of the library-size index (cell-based files). Carved out of the
    /// reserved block like `target_size`, so older files read `0`: no index,
    /// and readers fall back to the chunk headers.
    pub(crate) library_index_offset: u64,
    /// 20 additional reserved bytes for the future
    _reserved_1: [u8; 20],
    /// 3 additional reserved bytes for the future
    _reserved_2: [u8; 3],
}

impl FileHeader {
    /// Generate a new header
    ///
    /// ### Params
    ///
    /// * `cell_based` - Is the data stored for fast cell retrieval.
    /// * `target_size` - Library size the normalised layer was scaled to. Pass
    ///   `0.0` on raw-only paths where no normalisation was applied.
    ///
    /// ### Returns
    ///
    /// A new object of `FileHeader`
    pub(crate) fn new(cell_based: bool, target_size: f32) -> Self {
        Self {
            magic: *FILE_MAGIC,
            version: SC_FILE_VERSION_V4,
            main_header_offset: 0,
            cell_based,
            target_size,
            library_index_offset: 0,
            _reserved_1: [0; 20],
            _reserved_2: [0; 3],
        }
    }
}

//////////////////////
// Chunks to sparse //
//////////////////////

#[derive(Clone, Debug, Copy)]
/// Which of the data layers to return from a given chunk when transforming
/// into a [CompressedSparseData2]
pub enum DataLayerReturn {
    /// Return both layers, i.e., raw and
    BothLayers,
    /// Return only the raw layer
    Raw,
    /// Return only the norm layer
    Norm,
}

/// Append a chunk's raw counts to `data`, converted exactly into `T`.
///
/// The `RawCounts` variant is matched once per chunk rather than per value.
///
/// ### Params
///
/// * `data` - Destination vector
/// * `raw` - The chunk's raw counts
///
/// ### Returns
///
/// `Ok(())`, or [`ScDataErrors::RawCountOverflow`] for a count that does not
/// fit `T`.
fn extend_raw_counts<T>(data: &mut Vec<T>, raw: &RawCounts) -> Result<(), ScDataErrors>
where
    T: BixverseNumeric + FromPrimitive,
{
    let convert = |val: u32| {
        T::from_u32(val).ok_or(ScDataErrors::RawCountOverflow {
            value: val,
            target_type: std::any::type_name::<T>(),
        })
    };
    match raw {
        RawCounts::U16(v) => {
            for &x in v {
                data.push(convert(x as u32)?);
            }
        }
        RawCounts::U32(v) => {
            for &x in v {
                data.push(convert(x)?);
            }
        }
    }
    Ok(())
}

/// Converts a slice of gene chunks into a CSC sparse matrix
///
/// Constructs a cells x genes compressed sparse column matrix from individual
/// gene chunks. Depending on `data_layer`, the raw counts and/or the normalised
/// counts (f32) are populated.
///
/// Raw counts are stored on disk as `u32` and are converted into `T` exactly:
/// a count that does not fit returns [`ScDataErrors::RawCountOverflow`]
/// rather than saturating. Note that `T = f32` starts rounding above 2^24,
/// which is far beyond any plausible per-gene count.
///
/// ### Params
///
/// * `chunks` - Vector of gene chunks to convert
/// * `data_layer` - Which data layer(s) to populate in the output
/// * `n_cells` - Total number of cells in the dataset
///
/// ### Returns
///
/// A CSC-formatted sparse matrix with the requested layer(s) populated.
pub fn from_gene_chunks<T>(
    chunks: Vec<CscGeneChunk>,
    data_layer: &DataLayerReturn,
    n_cells: usize,
) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
where
    T: BixverseNumeric + FromPrimitive,
{
    let n_genes = chunks.len();
    let keep_raw = matches!(
        data_layer,
        DataLayerReturn::BothLayers | DataLayerReturn::Raw
    );
    let keep_norm = matches!(
        data_layer,
        DataLayerReturn::BothLayers | DataLayerReturn::Norm
    );

    let total_nnz: usize = chunks.iter().map(|c| c.indices.len()).sum();
    let mut data: Vec<T> = Vec::with_capacity(if keep_raw { total_nnz } else { 0 });
    let mut data_2: Vec<f32> = Vec::with_capacity(if keep_norm { total_nnz } else { 0 });
    let mut indices: Vec<u32> = Vec::with_capacity(total_nnz);
    let mut indptr: Vec<usize> = Vec::with_capacity(n_genes + 1);
    indptr.push(0);

    for chunk in chunks {
        if keep_raw {
            extend_raw_counts(&mut data, &chunk.data_raw)?;
        }
        if keep_norm {
            data_2.extend_from_slice(&chunk.data_norm);
        }
        indices.extend_from_slice(&chunk.indices);
        indptr.push(indices.len());
    }

    Ok(CompressedSparseData2 {
        data,
        indices,
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csc,
        data_2: if keep_norm { Some(data_2) } else { None },
        shape: (n_cells, n_genes),
    })
}

/// Converts a slice of cell chunks into a CSR sparse matrix
///
/// Constructs a cells x genes compressed sparse row matrix from individual
/// cell chunks. Same exact-conversion behaviour as [`from_gene_chunks`]: a raw
/// count that does not fit `T` returns [`ScDataErrors::RawCountOverflow`]
/// rather than saturating.
///
/// ### Params
///
/// * `chunks` - Slice of cell chunks to convert
/// * `data_layer` - Which data layer(s) to populate in the output
/// * `n_genes` - Total number of genes in the dataset
///
/// ### Returns
///
/// A CSR-formatted sparse matrix with the requested layer(s) populated.
pub fn from_cell_chunks<T>(
    chunks: &[CsrCellChunk],
    data_layer: &DataLayerReturn,
    n_genes: usize,
) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
where
    T: BixverseNumeric + FromPrimitive,
{
    let n_cells = chunks.len();
    let keep_raw = matches!(
        data_layer,
        DataLayerReturn::BothLayers | DataLayerReturn::Raw
    );
    let keep_norm = matches!(
        data_layer,
        DataLayerReturn::BothLayers | DataLayerReturn::Norm
    );

    let total_nnz: usize = chunks.iter().map(|c| c.indices.len()).sum();
    let mut data: Vec<T> = Vec::with_capacity(if keep_raw { total_nnz } else { 0 });
    let mut data_2: Vec<f32> = Vec::with_capacity(if keep_norm { total_nnz } else { 0 });
    let mut indices: Vec<u32> = Vec::with_capacity(total_nnz);
    let mut indptr: Vec<usize> = Vec::with_capacity(n_cells + 1);
    indptr.push(0);

    for chunk in chunks {
        if keep_raw {
            extend_raw_counts(&mut data, &chunk.data_raw)?;
        }
        if keep_norm {
            data_2.extend_from_slice(&chunk.data_norm);
        }
        indices.extend_from_slice(&chunk.indices);
        indptr.push(indices.len());
    }

    Ok(CompressedSparseData2 {
        data,
        indices,
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: if keep_norm { Some(data_2) } else { None },
        shape: (n_cells, n_genes),
    })
}

/////////////////////////////
// Chunk transforms for sc //
/////////////////////////////

impl CscGeneChunk {
    /// Convert the normalised counts from the `log1p(u * sf)` scale to the
    /// `log1p(u)` scale, enabling downstream use of the shifted CLR (PFlog1pPF)
    /// transformation. Mutates `data_norm` in place.
    ///
    /// ### Params
    ///
    /// * `size_factor` - The size factor used in the original normalisation
    ///   (e.g. 1e4 for CP10k).
    pub fn transform_to_clr(&mut self, size_factor: f32) {
        let sf_inv = 1.0_f32 / size_factor;
        for v in self.data_norm.iter_mut() {
            *v = (v.exp_m1() * sf_inv).ln_1p();
        }
    }
}

impl CsrCellChunk {
    /// Combine two cells into a synthetic doublet for doublet simulation.
    ///
    /// Filters both cells to HVG genes, sums their raw counts, normalises
    /// to `target_size` using the combined HVG library size, and optionally
    /// log-transforms.
    ///
    /// ### Params
    ///
    /// * `cell1` - First parent cell.
    /// * `cell2` - Second parent cell.
    /// * `hvg_indices` - Set of HVG gene indices to retain.
    /// * `hvg_lib_size_combined` - Sum of parent HVG library sizes.
    /// * `target_size` - Normalisation target (e.g. mean HVG library size).
    /// * `log_transform` - Whether to apply `ln(1 + x)` after normalisation.
    /// * `doublet_index` - Index to assign to the new chunk.
    ///
    /// ### Returns
    ///
    /// A new `CsrCellChunk` representing the synthetic doublet.
    #[allow(clippy::too_many_arguments)]
    pub fn add_cells_scrublet(
        cell1: &CsrCellChunk,
        cell2: &CsrCellChunk,
        hvg_indices: &FxHashSet<usize>,
        hvg_lib_size_combined: usize,
        target_size: f32,
        log_transform: bool,
        doublet_index: usize,
    ) -> Self {
        let mut gene_counts: FxHashMap<u32, u32> = FxHashMap::default();

        for i in 0..cell1.indices.len() {
            if hvg_indices.contains(&(cell1.indices[i] as usize)) {
                *gene_counts.entry(cell1.indices[i]).or_insert(0) += cell1.data_raw.get(i);
            }
        }
        for i in 0..cell2.indices.len() {
            if hvg_indices.contains(&(cell2.indices[i] as usize)) {
                *gene_counts.entry(cell2.indices[i]).or_insert(0) += cell2.data_raw.get(i);
            }
        }

        let mut gene_vec: Vec<(u32, u32)> = gene_counts.into_iter().collect();
        gene_vec.sort_unstable_by_key(|&(gene, _)| gene);

        let mut data_raw: Vec<u32> = Vec::with_capacity(gene_vec.len());
        let mut data_norm = Vec::with_capacity(gene_vec.len());
        let mut indices = Vec::with_capacity(gene_vec.len());

        let norm_factor = target_size / hvg_lib_size_combined as f32;

        for (gene, count) in gene_vec {
            data_raw.push(count);

            data_norm.push(if log_transform {
                (count as f32 * norm_factor).ln_1p()
            } else {
                count as f32 * norm_factor
            });
            indices.push(gene);
        }

        Self {
            data_raw: RawCounts::from_u32_auto(&data_raw),
            data_norm,
            library_size: hvg_lib_size_combined,
            indices,
            original_index: doublet_index,
            to_keep: true,
        }
    }
}

impl CsrCellChunk {
    /// Construct a chunk from pre-computed doublet simulation data.
    ///
    /// Unlike `from_data`, this does not recompute normalisation! The caller
    /// has already applied the correct normalisation (with noise-injected raw
    /// counts and the effective library size).
    ///
    /// ### Params
    ///
    /// * `indices` - Selected genes-remapped gene positions (sparse, sorted).
    /// * `raw_counts` - Raw counts at each position (post-noise).
    /// * `norm_values` - Pre-computed normalised values (same length
    ///   as `indices`).
    /// * `effective_lib_size` - Sum of raw counts after all noise
    ///   treatments.
    /// * `doublet_index` - Index of this doublet in the simulation batch (used
    ///   as `original_index`).
    ///
    /// ### Returns
    ///
    /// A `CsrCellChunk` ready for downstream PCA and feature
    /// engineering.
    pub fn from_doublet_simulation(
        indices: Vec<u32>,
        raw_counts: Vec<u32>,
        norm_values: Vec<f32>,
        effective_lib_size: usize,
        doublet_index: usize,
    ) -> Self {
        debug_assert_eq!(indices.len(), raw_counts.len());
        debug_assert_eq!(indices.len(), norm_values.len());

        Self {
            data_raw: RawCounts::from_u32_auto(&raw_counts),
            data_norm: norm_values,
            library_size: effective_lib_size,
            indices,
            original_index: doublet_index,
            to_keep: true,
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use bincode::{config, decode_from_slice, serde::encode_to_vec};

    /// Build a cell chunk from raw counts and gene indices.
    ///
    /// ### Params
    ///
    /// * `raw` - Raw counts, converted via [`RawCounts::from_u32_auto`].
    /// * `indices` - Gene indices, same length as `raw`.
    /// * `original_index` - Index the chunk is stored under.
    ///
    /// ### Returns
    ///
    /// The `CsrCellChunk`.
    pub(crate) fn cell_chunk(raw: &[u32], indices: &[u32], original_index: usize) -> CsrCellChunk {
        CsrCellChunk {
            data_raw: RawCounts::from_u32_auto(raw),
            data_norm: raw.iter().map(|&v| v as f32).collect(),
            library_size: raw.iter().map(|&v| v as usize).sum(),
            indices: indices.to_vec(),
            original_index,
            to_keep: true,
        }
    }

    /// Build a gene chunk from raw counts and cell indices.
    ///
    /// ### Params
    ///
    /// * `raw` - Raw counts, converted via [`RawCounts::from_u32_auto`].
    /// * `indices` - Cell indices, same length as `raw`.
    /// * `original_index` - Index the chunk is stored under.
    ///
    /// ### Returns
    ///
    /// The `CscGeneChunk`.
    pub(crate) fn gene_chunk(raw: &[u32], indices: &[u32], original_index: usize) -> CscGeneChunk {
        CscGeneChunk {
            data_raw: RawCounts::from_u32_auto(raw),
            data_norm: raw.iter().map(|&v| v as f32).collect(),
            nnz: raw.len(),
            indices: indices.to_vec(),
            original_index,
            to_keep: true,
        }
    }

    ////////////////////
    // Cell filtering //
    ////////////////////

    /// Kept cells are renumbered to their keep-set position and ordered by it,
    /// also when the keep set is not ascending; both filter paths agree.
    #[test]
    fn test_filter_selected_cells_follows_keep_order() {
        let keep: IndexSet<u32> = [7_u32, 1, 4, 900].into_iter().collect();

        let mut by_set = gene_chunk(&[10, 30, 50, 70], &[1, 3, 5, 7], 0);
        by_set.filter_selected_cells(&keep);
        assert_eq!(by_set.indices, vec![0, 1]);
        assert_eq!(by_set.data_raw.iter().collect::<Vec<u32>>(), vec![70, 10]);
        assert_eq!(by_set.data_norm[0], 70.0);
        assert_eq!(by_set.nnz, 2);

        // cell 900 sizes the map; cells past it must drop, not index out
        let mut by_map = gene_chunk(&[10, 30, 50, 70, 90], &[1, 3, 5, 7, 2_000], 0);
        by_map.filter_selected_cells_dense(&keep_set_index_map(&keep));
        assert_eq!(by_map.indices, by_set.indices);
        assert_eq!(
            by_map.data_raw.iter().collect::<Vec<u32>>(),
            by_set.data_raw.iter().collect::<Vec<u32>>()
        );
        assert_eq!(by_map.data_norm, by_set.data_norm);
    }

    ////////////////////
    // Chunk codec //
    ////////////////////

    /// Pins the v3 CSR chunk codec: every field returns bit-identical, the
    /// norm at the f16 precision v3 stores.
    #[test]
    fn test_cell_chunk_round_trip_preserves_fields() {
        let raw = [1_u32, 7, 42, 65_535, 3];
        let indices = [0_u32, 5, 9, 100_000, 300_000];
        let chunk = cell_chunk(&raw, &indices, 17);

        let mut buffer = Vec::new();
        chunk.write_to_bytes(&mut buffer).expect("serialise");
        let back = CsrCellChunk::read_from_buffer(&buffer).expect("deserialise");

        assert_eq!(back.data_raw.iter().collect::<Vec<_>>(), raw.to_vec());
        assert_eq!(back.indices, indices.to_vec());
        assert_eq!(back.library_size, chunk.library_size);
        assert_eq!(back.original_index, 17);
        assert!(back.to_keep);
        let bits: Vec<u16> = back
            .data_norm
            .iter()
            .map(|&v| F16::from_f32(v).to_bits())
            .collect();
        let want: Vec<u16> = chunk
            .data_norm
            .iter()
            .map(|&v| F16::from_f32(v).to_bits())
            .collect();
        assert_eq!(bits, want);
    }

    /// Same pin for the CSC codec, which carries `nnz` too.
    #[test]
    fn test_gene_chunk_round_trip_preserves_fields() {
        let raw = [2_u32, 0, 99, 70_000];
        let indices = [1_u32, 4, 8, 12];
        let chunk = gene_chunk(&raw, &indices, 3);

        let mut buffer = Vec::new();
        chunk.write_to_bytes(&mut buffer).expect("serialise");
        let back = CscGeneChunk::read_from_buffer(&buffer).expect("deserialise");

        assert_eq!(back.data_raw.iter().collect::<Vec<_>>(), raw.to_vec());
        assert_eq!(back.indices, indices.to_vec());
        assert_eq!(back.nnz, 4);
        assert_eq!(back.original_index, 3);
    }

    /// Regression: the CSC guard used to check 32 bytes then slice 36.
    #[test]
    fn test_gene_chunk_short_buffer_errors_instead_of_panicking() {
        for len in CSR_CHUNK_HEADER_LEN..CSC_CHUNK_HEADER_LEN {
            let buffer = vec![0_u8; len];
            match CscGeneChunk::read_from_buffer(&buffer) {
                Err(ScDataErrors::ChunkBufferTooSmall { expected, found }) => {
                    assert_eq!(expected, CSC_CHUNK_HEADER_LEN);
                    assert_eq!(found, len);
                }
                other => panic!("expected ChunkBufferTooSmall for len {len}, got {other:?}"),
            }
        }
    }

    /// Regression: payload lengths came from untrusted header fields and were
    /// never validated against the buffer.
    #[test]
    fn test_chunk_payload_truncation_is_caught() {
        let chunk = cell_chunk(&[1, 2, 3], &[0, 1, 2], 0);
        let mut buffer = Vec::new();
        chunk.write_to_bytes(&mut buffer).expect("serialise");

        // Claim four times as many raw counts as were written.
        buffer[0..4].copy_from_slice(&12_u32.to_le_bytes());

        assert!(matches!(
            CsrCellChunk::read_from_buffer(&buffer),
            Err(ScDataErrors::ChunkPayloadTruncated { .. })
        ));

        // Same for a gene chunk, via the indices length.
        let gene = gene_chunk(&[1, 2, 3], &[0, 1, 2], 0);
        let mut buffer = Vec::new();
        gene.write_to_bytes(&mut buffer).expect("serialise");
        buffer[8..12].copy_from_slice(&64_u32.to_le_bytes());

        assert!(matches!(
            CscGeneChunk::read_from_buffer(&buffer),
            Err(ScDataErrors::ChunkPayloadTruncated { .. })
        ));
    }

    /// Regression: with `elem_size == 2` and an odd raw/norm element count the
    /// old `*const u32` read landed on a 2-byte-aligned address.
    #[test]
    fn test_chunk_round_trip_with_unaligned_index_offset() {
        // 3 raw u16 + 3 norm u16 puts the indices at offset 32 + 6 + 6 = 44,
        // which is 4-byte misaligned.
        let raw = [1_u32, 2, 3];
        let indices = [7_u32, 8, 9];
        let chunk = cell_chunk(&raw, &indices, 0);
        assert_eq!(chunk.data_raw.elem_size(), RAW_ELEM_U16);

        let mut buffer = Vec::new();
        chunk.write_to_bytes(&mut buffer).expect("serialise");
        assert_eq!(buffer.len() % 4, 0);

        let back = CsrCellChunk::read_from_buffer(&buffer).expect("deserialise");
        assert_eq!(back.indices, indices.to_vec());
        assert_eq!(back.data_raw.iter().collect::<Vec<_>>(), raw.to_vec());
    }

    /// A zero discriminant still means u16; an unknown width must error.
    #[test]
    fn test_raw_counts_element_size_discriminant() {
        let payload = [1_u8, 0, 2, 0];

        // Legacy files left the discriminant byte as zero padding.
        let legacy = RawCounts::read_from_buffer(&payload, 2, 0).expect("legacy parses as u16");
        assert_eq!(legacy.iter().collect::<Vec<_>>(), vec![1, 2]);

        let explicit =
            RawCounts::read_from_buffer(&payload, 2, RAW_ELEM_U16).expect("explicit u16 parses");
        assert_eq!(explicit.iter().collect::<Vec<_>>(), vec![1, 2]);

        assert!(matches!(
            RawCounts::read_from_buffer(&payload, 2, 3),
            Err(ScDataErrors::RawElemSizeInvalid(3))
        ));
    }

    /////////////////////
    // Count widening //
    /////////////////////

    /// Regression: counts above `u16::MAX` used to saturate silently.
    #[test]
    fn test_large_raw_count_survives_round_trip() {
        let raw = [70_000_u32, 1];
        let chunk = cell_chunk(&raw, &[0, 1], 0);
        assert_eq!(chunk.data_raw.elem_size(), RAW_ELEM_U32);

        let mut buffer = Vec::new();
        chunk.write_to_bytes(&mut buffer).expect("serialise");
        let back = CsrCellChunk::read_from_buffer(&buffer).expect("deserialise");

        let csr = from_cell_chunks::<u32>(&[back], &DataLayerReturn::Raw, 2).expect("assemble");
        assert_eq!(csr.data, vec![70_000_u32, 1]);
    }

    /// A count above `u16::MAX` must be reported, not wrapped or saturated.
    #[test]
    fn test_narrow_target_type_reports_overflow() {
        let chunk = gene_chunk(&[70_000, 1], &[0, 1], 0);

        match from_gene_chunks::<u16>(vec![chunk], &DataLayerReturn::Raw, 2) {
            Err(ScDataErrors::RawCountOverflow { value, .. }) => assert_eq!(value, 70_000),
            other => panic!("expected RawCountOverflow, got {:?}", other.map(|_| ())),
        }
    }

    /////////////////
    // File header //
    /////////////////

    /// The v3 header layout before `target_size` was carved out of the
    /// reserved block. Used to prove the swap is byte-neutral in both
    /// directions.
    #[derive(Encode, Decode, Serialize, Deserialize)]
    struct LegacyFileHeader {
        /// Magic string as bytes to recognise the file
        magic: [u8; 8],
        /// Version of the file
        version: u32,
        /// Offset of the main header
        main_header_offset: u64,
        /// Is cell-based (CSR format)
        cell_based: bool,
        /// 32 reserved bytes, all zero in files written by the old writer
        _reserved_1: [u8; 32],
        /// 3 further reserved bytes
        _reserved_2: [u8; 3],
    }

    /// Encode a header into the fixed, zero-padded 64-byte slot.
    ///
    /// ### Params
    ///
    /// * `header` - Anything the bincode standard config can encode.
    ///
    /// ### Returns
    ///
    /// Exactly 64 bytes.
    fn encode_slot<T: Serialize>(header: &T) -> [u8; 64] {
        let encoded = encode_to_vec(header, config::standard()).expect("encode");
        assert!(encoded.len() <= 64, "header must fit the fixed slot");
        let mut slot = [0_u8; 64];
        slot[..encoded.len()].copy_from_slice(&encoded);
        slot
    }

    /// Backwards: pre-`target_size` header bytes must still decode, reading 0.
    #[test]
    fn test_legacy_header_decodes_with_zero_target_size() {
        let legacy = LegacyFileHeader {
            magic: *b"SCRNASEQ",
            version: SC_FILE_VERSION_V3,
            main_header_offset: 123_456,
            cell_based: true,
            _reserved_1: [0; 32],
            _reserved_2: [0; 3],
        };

        let slot = encode_slot(&legacy);
        let (decoded, _) =
            decode_from_slice::<FileHeader, _>(&slot, config::standard()).expect("decode as v3");

        assert_eq!(decoded.magic, *b"SCRNASEQ");
        assert_eq!(decoded.version, SC_FILE_VERSION_V3);
        assert_eq!(decoded.main_header_offset, 123_456);
        assert!(decoded.cell_based);
        assert_eq!(decoded.target_size, 0.0);
    }

    /// Forwards: today's header must still decode under the old layout.
    #[test]
    fn test_new_header_still_decodes_on_the_old_layout() {
        let mut header = FileHeader::new(false, 1e4);
        header.main_header_offset = 987_654_321;

        let slot = encode_slot(&header);
        let (decoded, _) = decode_from_slice::<LegacyFileHeader, _>(&slot, config::standard())
            .expect("decode as legacy");

        assert_eq!(decoded.magic, *b"SCRNASEQ");
        assert_eq!(decoded.version, SC_FILE_VERSION_V4);
        assert_eq!(decoded.main_header_offset, 987_654_321);
        assert!(!decoded.cell_based);
    }

    /// The new field survives the fixed, zero-padded 64-byte slot.
    #[test]
    fn test_target_size_survives_header_round_trip() {
        let slot = encode_slot(&FileHeader::new(true, 1e5));
        let (decoded, _) =
            decode_from_slice::<FileHeader, _>(&slot, config::standard()).expect("decode");
        assert_eq!(decoded.target_size, 1e5);
    }

    ////////////////////////
    // Library-size index //
    ////////////////////////

    /// The `FileHeader` layout before `library_index_offset` was carved out
    /// of the reserved block (28 reserved bytes, all zero when written).
    #[derive(Encode, Decode, Serialize, Deserialize)]
    struct TargetSizeFileHeader {
        /// File magic.
        magic: [u8; 8],
        /// File version.
        version: u32,
        /// Offset of the main header.
        main_header_offset: u64,
        /// Cell- or gene-based.
        cell_based: bool,
        /// Target size of the norm.
        target_size: f32,
        /// Reserved, zero.
        _reserved_1: [u8; 28],
        /// Padding.
        _reserved_2: [u8; 3],
    }

    /// Backwards: header bytes from before the index decode with offset 0,
    /// which readers treat as "no index".
    #[test]
    fn test_pre_index_header_decodes_with_no_library_index() {
        let old = TargetSizeFileHeader {
            magic: *b"SCRNASEQ",
            version: SC_FILE_VERSION_V3,
            main_header_offset: 123_456,
            cell_based: true,
            target_size: 1e4,
            _reserved_1: [0; 28],
            _reserved_2: [0; 3],
        };

        let slot = encode_slot(&old);
        let (decoded, _) =
            decode_from_slice::<FileHeader, _>(&slot, config::standard()).expect("decode");

        assert_eq!(decoded.main_header_offset, 123_456);
        assert_eq!(decoded.target_size, 1e4);
        assert_eq!(decoded.library_index_offset, 0);
    }

    /// Forwards: a header carrying the index still decodes under the layout
    /// before it, so readers built before this change open new files.
    #[test]
    fn test_header_with_library_index_decodes_on_the_pre_index_layout() {
        let mut header = FileHeader::new(true, 1e4);
        header.main_header_offset = 987_654_321;
        header.library_index_offset = u64::MAX;

        let slot = encode_slot(&header);
        let (decoded, _) = decode_from_slice::<TargetSizeFileHeader, _>(&slot, config::standard())
            .expect("decode as pre-index");

        assert_eq!(decoded.magic, *b"SCRNASEQ");
        assert_eq!(decoded.version, SC_FILE_VERSION_V4);
        assert_eq!(decoded.main_header_offset, 987_654_321);
        assert!(decoded.cell_based);
        assert_eq!(decoded.target_size, 1e4);
    }
}
