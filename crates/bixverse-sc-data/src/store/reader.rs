//! Readers for the binary format: the `SingleCellReading` trait, the
//! memory-mapped `ParallelSparseReader`, page cache prefetch and fused
//! batch reads.

use bincode::{config, decode_from_slice};
use indexmap::IndexSet;
use lz4_flex::block::{decompress_into, uncompressed_size};
use lz4_flex::decompress_size_prepended;
use memmap2::MmapOptions;
use num_traits::FromPrimitive;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{fs::File, marker::Sync, path::Path, sync::Arc};

use crate::prelude::*;
use crate::store::chunk::{
    FILE_MAGIC, FileHeader, RunSource, SparseDataHeader, csc_payload, csr_payload,
    extend_raw_checked, gene_subset_map, keep_set_index_map, le_u64, raw_to, remap,
};
use crate::store::interval_index::GenomeIntervals;
use crate::store::v4::{
    V4Ctx, V4Decoded, V4Header, V4Run, decode_v4, decode_v4_into, gene_norm_table, v4_cell_chunk,
    v4_gene_chunk,
};
use crate::store::v5::{
    FeatureKind, NormKind, Section, SectionKind, V5_HEADER_LEN, V5FileHeader, check_header_hash,
    expect_len, find_section, parse_key_values, parse_sections,
};

///////////////////////
// SingleCellReading //
///////////////////////

/// Dimensions and orientation of a count store.
///
/// What consumers of [`SingleCellReading`] need to know about a store without
/// touching its offset table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreDims {
    /// Total number of cells.
    pub total_cells: usize,
    /// Total number of genes (features).
    pub total_genes: usize,
    /// Whether the store is cell-based (CSR) rather than gene-based (CSC).
    pub cell_based: bool,
}

/// Read access to a chunked single cell count store.
///
/// Abstracts over where the data physically lives. The only implementor today
/// is [`ParallelSparseReader`], which memory-maps the bixverse binary format
/// from disk, but the trait exposes nothing file-specific so that other
/// backends can be added later without touching consumers.
///
/// Whether a store is cell-based (CSR, fast per-cell access) or gene-based
/// (CSC, fast per-gene access) is a runtime property of the underlying data,
/// not of the type. Asking a gene-based store for cells returns
/// [`ScDataErrors::ReaderModeMismatch`].
///
/// Implementors need only supply `read_cells_parallel`, `read_gene_parallel`,
/// `dims` and `is_cell_based`. The rest have default bodies written in
/// terms of those four. Backends that can do better should override: the
/// defaults for `read_cell_library_sizes` and `read_gene_nnz` parse whole
/// chunks, whereas a backend able to peek at chunk headers can avoid that.
///
/// `Send + Sync` is required: readers are shared across `rayon` threads.
pub trait SingleCellReading: Send + Sync {
    /// Read cells by index in a multi-threaded manner
    ///
    /// ### Params
    ///
    /// * `indices` - Index positions of the cells to retrieve
    ///
    /// ### Returns
    ///
    /// The `CsrCellChunk`s in the order given by `indices`.
    fn read_cells_parallel(&self, indices: &[usize]) -> Result<Vec<CsrCellChunk>, ScDataErrors>;

    /// Read genes by index in a multi-threaded manner
    ///
    /// ### Params
    ///
    /// * `indices` - Index positions of the genes to retrieve
    ///
    /// ### Returns
    ///
    /// The `CscGeneChunk`s in the order given by `indices`.
    fn read_gene_parallel(&self, indices: &[usize]) -> Result<Vec<CscGeneChunk>, ScDataErrors>;

    /// Read genes for consumers of the raw counts and indices only.
    ///
    /// `data_norm` may come back empty, so only use it when the consumer never
    /// touches that layer. The default reads both layers.
    ///
    /// ### Params
    ///
    /// * `indices` - Index positions of the genes to retrieve
    ///
    /// ### Returns
    ///
    /// The `CscGeneChunk`s in the order given by `indices`.
    fn read_gene_parallel_raw(&self, indices: &[usize]) -> Result<Vec<CscGeneChunk>, ScDataErrors> {
        self.read_gene_parallel(indices)
    }

    /// Read a batch of cells straight into one CSR matrix.
    ///
    /// Same output as [`Self::read_cells_parallel`] followed by
    /// [`from_cell_chunks`], without materialising a chunk per cell. Prefer it
    /// whenever the consumer only needs the counts and indices.
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices, in output row order
    /// * `data_layer` - Which layer(s) to materialise
    ///
    /// ### Returns
    ///
    /// CSR of shape `(indices.len(), total_genes)`, raw counts in `data` (empty
    /// when not requested) and normalised counts in `data_2`, or
    /// [`ScDataErrors::ReaderModeMismatch`] on a gene-major store and
    /// [`ScDataErrors::RawCountOverflow`] for a count that does not fit `T`.
    fn read_cells_sparse<T>(
        &self,
        indices: &[usize],
        data_layer: &DataLayerReturn,
    ) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
    where
        T: BixverseNumeric + FromPrimitive;

    /// Read a batch of genes straight into one CSC matrix, optionally keeping
    /// only a cell subset.
    ///
    /// Same output as [`Self::read_gene_parallel_filtered`] (or
    /// [`Self::read_gene_parallel`] without a subset) followed by
    /// [`from_gene_chunks`], without materialising a chunk per gene. Kept
    /// cells are renumbered to their position in `cells_to_keep`.
    ///
    /// ### Params
    ///
    /// * `indices` - Gene indices, in output column order
    /// * `cells_to_keep` - Cells to keep, in output row order, or `None` for
    ///   all cells
    /// * `data_layer` - Which layer(s) to materialise
    ///
    /// ### Returns
    ///
    /// CSC of shape `(cells, indices.len())`, raw counts in `data` (empty when
    /// not requested) and normalised counts in `data_2`, or
    /// [`ScDataErrors::ReaderModeMismatch`] on a cell-major store and
    /// [`ScDataErrors::RawCountOverflow`] for a count that does not fit `T`.
    fn read_genes_sparse<T>(
        &self,
        indices: &[usize],
        cells_to_keep: Option<&IndexSet<u32>>,
        data_layer: &DataLayerReturn,
    ) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
    where
        T: BixverseNumeric + FromPrimitive;

    /// Read genes from a gene-major store straight into one CSR matrix.
    ///
    /// Same matrix as [`Self::read_genes_sparse`] followed by a transpose,
    /// without ever holding the CSC next to the CSR. Genes are read in batches
    /// of [`CSR_GENE_BATCH`]; each batch is transposed locally into
    /// [`CSR_ROW_BLOCKS_PER_THREAD`] row blocks per thread, so the scatter
    /// stays cache sized and no count pass is needed. The final CSR is then
    /// assembled block by block, freeing each block as it is copied. The
    /// output is zero-allocated and only commits as it is written, so the peak
    /// stays near one copy plus one batch. That relies on the allocator
    /// returning the freed blocks to the OS, which large allocations do.
    ///
    /// ### Params
    ///
    /// * `indices` - Gene indices, in output column order
    /// * `cells_to_keep` - Cells to keep, in output row order, or `None` for
    ///   all cells
    /// * `data_layer` - Which layer(s) to materialise
    ///
    /// ### Returns
    ///
    /// CSR of shape `(cells, indices.len())`, columns ascending within each
    /// row, or whatever [`Self::read_genes_sparse`] raises.
    fn read_genes_sparse_csr<T>(
        &self,
        indices: &[usize],
        cells_to_keep: Option<&IndexSet<u32>>,
        data_layer: &DataLayerReturn,
    ) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
    where
        T: BixverseNumeric + FromPrimitive,
    {
        genes_sparse_to_csr(self, indices, cells_to_keep, data_layer, CSR_GENE_BATCH)
    }

    /// Dimensions and orientation of the store.
    ///
    /// Cheap: a copy of a few scalars, never the offset table.
    ///
    /// ### Returns
    ///
    /// The [`StoreDims`].
    fn dims(&self) -> StoreDims;

    /// Is the store laid out for fast cell retrieval?
    ///
    /// ### Returns
    ///
    /// `true` for a cell-based (CSR) store.
    fn is_cell_based(&self) -> bool;

    /// Is the store laid out for fast gene retrieval?
    ///
    /// ### Returns
    ///
    /// `true` for a gene-based (CSC) store.
    fn is_gene_based(&self) -> bool {
        !self.is_cell_based()
    }

    /// Library size the `data_norm` layer of this store was scaled to.
    ///
    /// Needed by anything that has to undo the `ln1p(count / lib_size *
    /// target_size)` normalisation, such as `CscGeneChunk::transform_to_clr`.
    /// Defaults to `None` for stores that do not record it.
    ///
    /// ### Returns
    ///
    /// `Some(target_size)`, or `None` when the store cannot report one.
    fn target_size(&self) -> Option<f32> {
        None
    }

    /// Read a single cell by index
    ///
    /// ### Params
    ///
    /// * `index` - Cell index
    ///
    /// ### Returns
    ///
    /// The `CsrCellChunk` of this cell.
    fn read_cell(&self, index: usize) -> Result<CsrCellChunk, ScDataErrors> {
        self.read_cells_parallel(&[index])?
            .into_iter()
            .next()
            .ok_or(ScDataErrors::ChunkIndexNotFound(index))
    }

    /// Read a single gene by index
    ///
    /// ### Params
    ///
    /// * `index` - Gene index
    ///
    /// ### Returns
    ///
    /// The `CscGeneChunk` of this gene.
    fn read_gene(&self, index: usize) -> Result<CscGeneChunk, ScDataErrors> {
        self.read_gene_parallel(&[index])?
            .into_iter()
            .next()
            .ok_or(ScDataErrors::ChunkIndexNotFound(index))
    }

    /// Read a contiguous range of cells
    ///
    /// Helper for memory-bounded gene generation.
    ///
    /// ### Params
    ///
    /// * `start` - First cell index (inclusive)
    /// * `end` - Last cell index (exclusive)
    ///
    /// ### Returns
    ///
    /// The `CsrCellChunk`s in that range.
    fn read_cells_range(
        &self,
        start: usize,
        end: usize,
    ) -> Result<Vec<CsrCellChunk>, ScDataErrors> {
        let indices: Vec<usize> = (start..end).collect();
        self.read_cells_parallel(&indices)
    }

    /// Return all cells in the store
    ///
    /// ### Returns
    ///
    /// Every `CsrCellChunk`.
    fn get_all_cells(&self) -> Result<Vec<CsrCellChunk>, ScDataErrors> {
        let indices: Vec<usize> = (0..self.dims().total_cells).collect();
        self.read_cells_parallel(&indices)
    }

    /// Return all genes in the store
    ///
    /// ### Returns
    ///
    /// Every `CscGeneChunk`.
    fn get_all_genes(&self) -> Result<Vec<CscGeneChunk>, ScDataErrors> {
        let indices: Vec<usize> = (0..self.dims().total_genes).collect();
        self.read_gene_parallel(&indices)
    }

    /// Read genes by index and filter them to a set of cells
    ///
    /// ### Params
    ///
    /// * `indices` - Index positions of the genes to retrieve
    /// * `cells_to_keep` - Cell index positions to keep, in output order
    ///
    /// ### Returns
    ///
    /// The filtered `CscGeneChunk`s.
    fn read_gene_parallel_filtered(
        &self,
        indices: &[usize],
        cells_to_keep: &IndexSet<u32>,
    ) -> Result<Vec<CscGeneChunk>, ScDataErrors> {
        let mut chunks = self.read_gene_parallel(indices)?;
        let cell_map = keep_set_index_map(cells_to_keep);
        chunks
            .par_iter_mut()
            .for_each(|chunk| chunk.filter_selected_cells_dense(&cell_map));
        Ok(chunks)
    }

    /// Read only the library sizes for the given cells
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices
    ///
    /// ### Returns
    ///
    /// Vector of library sizes in the order given by `indices`.
    fn read_cell_library_sizes(&self, indices: &[usize]) -> Result<Vec<usize>, ScDataErrors> {
        Ok(self
            .read_cells_parallel(indices)?
            .iter()
            .map(|chunk| chunk.library_size)
            .collect())
    }

    /// Read the NNZ for the given genes
    ///
    /// ### Params
    ///
    /// * `indices` - Gene indices
    ///
    /// ### Returns
    ///
    /// Vector of NNZ counts in the order given by `indices`.
    fn read_gene_nnz(&self, indices: &[usize]) -> Result<Vec<usize>, ScDataErrors> {
        Ok(self
            .read_gene_parallel(indices)?
            .iter()
            .map(|chunk| chunk.nnz)
            .collect())
    }

    /// Return NNZ for all genes
    ///
    /// ### Returns
    ///
    /// Vector of NNZ values for all genes.
    fn get_all_gene_nnz(&self) -> Result<Vec<usize>, ScDataErrors> {
        let indices: Vec<usize> = (0..self.dims().total_genes).collect();
        self.read_gene_nnz(&indices)
    }

    /// Compute the shifted CLR per-cell offsets freshly from raw counts.
    ///
    /// The offset for cell i is `m_i = (1/D) * sum_j log1p(x_ij / s_i)`, where
    /// D is the total gene count and s_i is the cell's library size. Size
    /// factor is fixed at 1, independent of any size factor used during the
    /// original normalisation. Cells are read in batches to bound peak memory.
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices to compute offsets for
    /// * `batch_size` - Number of cells to load per batch (defaults to
    ///   [`CELL_BATCH_SIZE`])
    ///
    /// ### Returns
    ///
    /// Vector of CLR offsets in the same order as `indices`.
    fn get_clr_offsets(
        &self,
        indices: &[usize],
        batch_size: Option<usize>,
    ) -> Result<Vec<f64>, ScDataErrors> {
        if !self.is_cell_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "gene-based",
                requested: "cell-based",
            });
        }
        let batch_size = batch_size.unwrap_or(CELL_BATCH_SIZE);

        let n_genes = self.dims().total_genes as f64;
        let mut offsets = Vec::with_capacity(indices.len());

        for batch in indices.chunks(batch_size) {
            let cells = self.read_cells_parallel(batch)?;
            let batch_offsets: Vec<f64> = cells
                .par_iter()
                .map(|chunk| {
                    let inv_s = 1.0_f64 / chunk.library_size as f64;
                    let sum: f64 = chunk
                        .data_raw
                        .iter()
                        .map(|c| (c as f64 * inv_s).ln_1p())
                        .sum();
                    sum / n_genes
                })
                .collect();
            offsets.extend_from_slice(&batch_offsets);
        }

        Ok(offsets)
    }
}

//////////////////////
// Streaming reader //
//////////////////////

/// Read the `target_size` out of a bin file without mapping the whole file.
///
/// Only the fixed file header is touched, so this is cheap enough to run over
/// every input of a merge before deciding what to write.
///
/// ### Params
///
/// * `f_path` - Path to the .bin file.
///
/// ### Returns
///
/// `Some(target_size)`, or `None` for raw-only files and files written before
/// the field entered the header.
pub fn peek_target_size<P: AsRef<Path>>(f_path: P) -> Result<Option<f32>, ScDataErrors> {
    use std::io::Read;

    let mut buffer = [0u8; V5_HEADER_LEN];
    let mut file = File::open(f_path)?;
    file.read_exact(&mut buffer[..64])?;

    let target_size = if buffer[8] as u32 == SC_FILE_VERSION {
        file.read_exact(&mut buffer[64..])?;
        V5FileHeader::parse(&buffer)?.target_size
    } else {
        let (file_header, _) =
            decode_from_slice::<FileHeader, _>(&buffer[..64], config::standard())
                .map_err(|_| ScDataErrors::HeaderDecodeFailed)?;
        file_header.target_size
    };

    Ok((target_size != 0.0).then_some(target_size))
}

/// Read the format version out of a bin file without mapping the whole file.
///
/// Reads the magic and byte 8, which holds the version in the v5 layout and in
/// the bincode header of v3 and v4 alike, so callers can check a store of any
/// size before opening it and decide whether it needs `upgrade_store_to_v5`.
///
/// ### Params
///
/// * `f_path` - Path to the .bin file.
///
/// ### Returns
///
/// The version, or [`ScDataErrors::HeaderDecodeFailed`] for a file that is not
/// a bixverse store.
pub fn peek_file_version<P: AsRef<Path>>(f_path: P) -> Result<u32, ScDataErrors> {
    use std::io::Read;

    let mut buffer = [0u8; 9];
    File::open(f_path)?.read_exact(&mut buffer)?;
    if &buffer[0..8] != FILE_MAGIC {
        return Err(ScDataErrors::HeaderDecodeFailed);
    }
    Ok(buffer[8] as u32)
}

/// Where the chunk offsets of an open store live.
enum ChunkOffsets {
    /// Decoded from the bincode header of a v3 or v4 file.
    Owned(Vec<u64>),
    /// Read through the memory map: byte offset of the v5 offset section,
    /// which holds `n_chunks + 1` u64.
    Mapped(usize),
}

/// How an original index resolves to a chunk.
enum ChunkLookup {
    /// Chunk `i` holds original index `i`.
    Identity,
    /// The hash map of a v3 or v4 header whose chunks are out of order.
    Hash(FxHashMap<usize, usize>),
    /// The v5 original-index section: byte offset of `(original, chunk)` u64
    /// pairs sorted by original index, and the pair count.
    Sorted(usize, usize),
}

/// ParallelSparseReader
pub struct ParallelSparseReader {
    /// Cells, genes and orientation.
    dims: StoreDims,
    /// Chunks in the file.
    n_chunks: usize,
    /// Where the chunk offsets live.
    offsets: ChunkOffsets,
    /// How an original index resolves to a chunk.
    lookup: ChunkLookup,
    /// Reference to the memory map for safe sharing across threads
    mmap: Arc<memmap2::Mmap>,
    /// The open file, for the explicit reads of [`Self::prefetch`].
    file: File,
    /// Start position of the chunks after the header file
    chunks_start: u64,
    /// Log-CPM target size, as recorded in the file header. `0.0` when
    /// unknown.
    target_size: f32,
    /// TF-IDF scale, `0.0` unless the store is TF-IDF.
    tf_idf_scale: f32,
    /// Byte offset in the mmap of the first library-size index entry, or
    /// `None` for files written without the index.
    library_index: Option<usize>,
    /// Format version of the file.
    version: u32,
    /// Library size per original cell index as `f32`, on gene-based files
    /// that carry the table. Gene chunks recompute their norm from it.
    cell_lib: Option<Arc<Vec<f32>>>,
    /// Normalised value of small counts per cell, built from `cell_lib` when
    /// the file has a target size and log-CPM norm. See [`gene_norm_table`].
    cell_norm: Option<Arc<Vec<f32>>>,
    /// What the features are. Genes for v3 and v4.
    feature_kind: FeatureKind,
    /// How the norm derives. Log-CPM for v3 and v4.
    norm_kind: NormKind,
    /// Raw layers per chunk. One for v3 and v4.
    n_layers: usize,
    /// The raw layer reads return, see [`Self::with_layer`].
    layer: usize,
    /// Hash of the cell IDs, all zero when unset or pre-v5.
    cell_id_hash: [u8; 16],
    /// Per-feature IDF of a TF-IDF store.
    idf: Option<Arc<Vec<f32>>>,
    /// Byte range of the v5 interval-index section, decoded on request.
    intervals: Option<std::ops::Range<usize>>,
    /// Known sections of a v5 file; empty for v3 and v4.
    sections: Vec<Section>,
}

impl ParallelSparseReader {
    /// Generate a new parallelised streaming reader
    ///
    /// Opens v5 and v4 stores. A v3 store is refused with a pointer to the
    /// converter.
    ///
    /// ### Params
    ///
    /// * `f_path` - Path to the file
    ///
    /// ### Returns
    ///
    /// Initialised `ParallelSparseReader`, or
    /// [`ScDataErrors::V3NeedsConversion`] for a v3 store.
    pub fn new(f_path: &str) -> Result<Self, ScDataErrors> {
        let reader = Self::open_any(f_path)?;
        if reader.version == SC_FILE_VERSION_V3 {
            return Err(ScDataErrors::V3NeedsConversion(f_path.to_string()));
        }
        Ok(reader)
    }

    /// Open a store of any version this build reads, v3 included.
    ///
    /// For the converters, the only readers of v3.
    ///
    /// ### Params
    ///
    /// * `f_path` - Path to the file
    ///
    /// ### Returns
    ///
    /// The reader.
    pub(crate) fn open_any(f_path: &str) -> Result<Self, ScDataErrors> {
        let file = File::open(f_path)?;
        let file_size = file.metadata()?.len();

        let mmap = unsafe {
            let mut opts = MmapOptions::new();
            if file_size <= 8 * 1024 * 1024 * 1024 {
                opts.populate();
            }
            opts.map(&file)?
        };

        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Random)?;

        // Everything below indexes the mmap directly, so each offset is checked
        // before it is used. A corrupt or foreign file must return an error,
        // not panic on a slice bound or parse as if it were ours.
        if mmap.len() < 64 || &mmap[0..8] != FILE_MAGIC {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }
        match mmap[8] as u32 {
            SC_FILE_VERSION => Self::open_v5(mmap, file),
            SC_FILE_VERSION_V3 | SC_FILE_VERSION_V4 => Self::open_legacy(mmap, file),
            found => Err(ScDataErrors::FileVersionMismatch {
                expected: SC_FILE_VERSION,
                found,
            }),
        }
    }

    /// Open a v5 store: header, section directory, then the sections.
    ///
    /// Offsets stay in the map. The library tables and the IDF are copied out
    /// as `f32`, a few bytes per cell or feature.
    ///
    /// ### Params
    ///
    /// * `mmap` - The mapped file
    /// * `file` - The open file
    ///
    /// ### Returns
    ///
    /// The reader, or [`ScDataErrors::V5HeaderInvalid`] for a malformed file.
    fn open_v5(mmap: memmap2::Mmap, file: File) -> Result<Self, ScDataErrors> {
        let h = V5FileHeader::parse(&mmap)?;
        let sections = parse_sections(&mmap, &h)?;
        check_header_hash(&mmap, &h)?;
        let missing = |what: &str| ScDataErrors::V5HeaderInvalid(format!("no {what} section"));

        let offsets = find_section(&sections, SectionKind::ChunkOffsets)
            .ok_or_else(|| missing("chunk offset"))?;
        expect_len(&offsets, h.n_chunks + 1, 8)?;
        let chunks_len = mmap.len() - V5_HEADER_LEN;
        let mut prev = 0u64;
        for c in 0..=h.n_chunks {
            let off = le_u64(&mmap, offsets.offset + 8 * c);
            // every chunk starts at least its 8-byte length prefix before the
            // next offset, the last offset is the end of the chunks
            let min_gap = if c == 0 { 0 } else { 8 };
            if off < prev + min_gap || off as usize > chunks_len {
                return Err(ScDataErrors::V5HeaderInvalid(format!(
                    "chunk offset {c} out of order or outside the file"
                )));
            }
            prev = off;
        }

        let lookup = match find_section(&sections, SectionKind::OriginalIndex) {
            Some(s) => {
                expect_len(&s, h.n_chunks * 2, 8)?;
                ChunkLookup::Sorted(s.offset, h.n_chunks)
            }
            None => ChunkLookup::Identity,
        };

        let library_index = if h.cell_based {
            let s = find_section(&sections, SectionKind::LibraryIndex)
                .ok_or_else(|| missing("library index"))?;
            expect_len(&s, h.n_chunks, 8)?;
            Some(s.offset)
        } else {
            None
        };
        let cell_lib = match find_section(&sections, SectionKind::CellLibraryTable) {
            Some(s) if !h.cell_based => {
                if s.len % 8 != 0 {
                    return Err(ScDataErrors::V5HeaderInvalid(
                        "cell library table not a whole number of u64".into(),
                    ));
                }
                Some(Arc::new(
                    (0..s.len / 8)
                        .map(|i| le_u64(&mmap, s.offset + 8 * i) as f32)
                        .collect::<Vec<f32>>(),
                ))
            }
            _ => None,
        };
        let idf = match find_section(&sections, SectionKind::Idf) {
            Some(s) => {
                expect_len(&s, h.n_features, 4)?;
                Some(Arc::new(
                    mmap[s.offset..s.offset + s.len]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&b| f32::from_le_bytes(b))
                        .collect::<Vec<f32>>(),
                ))
            }
            None => None,
        };

        if let Some(s) = find_section(&sections, SectionKind::ChunkChecksums) {
            expect_len(&s, h.n_chunks, 8)?;
        }

        let cell_norm = cell_lib
            .as_ref()
            .filter(|_| h.target_size != 0.0 && h.norm_kind == NormKind::LogCpm)
            .map(|lib| Arc::new(gene_norm_table(lib, h.target_size)));

        Ok(Self {
            dims: StoreDims {
                total_cells: h.n_cells,
                total_genes: h.n_features,
                cell_based: h.cell_based,
            },
            n_chunks: h.n_chunks,
            offsets: ChunkOffsets::Mapped(offsets.offset),
            lookup,
            mmap: Arc::new(mmap),
            file,
            chunks_start: V5_HEADER_LEN as u64,
            target_size: h.target_size,
            tf_idf_scale: h.tf_idf_scale,
            library_index,
            version: SC_FILE_VERSION,
            cell_lib,
            cell_norm,
            feature_kind: h.feature_kind,
            norm_kind: h.norm_kind,
            n_layers: h.n_layers,
            layer: 0,
            cell_id_hash: h.cell_id_hash,
            idf,
            intervals: find_section(&sections, SectionKind::IntervalIndex)
                .map(|s| s.offset..s.offset + s.len),
            sections,
        })
    }

    /// Open a v3 or v4 store through its bincode headers.
    ///
    /// ### Params
    ///
    /// * `mmap` - The mapped file
    /// * `file` - The open file
    ///
    /// ### Returns
    ///
    /// The reader, or [`ScDataErrors::HeaderDecodeFailed`] for a malformed
    /// file.
    fn open_legacy(mmap: memmap2::Mmap, file: File) -> Result<Self, ScDataErrors> {
        let file_header_bytes = &mmap[0..64];
        let (file_header, _) =
            decode_from_slice::<FileHeader, _>(file_header_bytes, config::standard())
                .map_err(|_| ScDataErrors::HeaderDecodeFailed)?;

        if &file_header.magic != FILE_MAGIC {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }

        let main_header_offset = file_header.main_header_offset as usize;
        if main_header_offset.saturating_add(8) > mmap.len() {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }

        let header_size = le_u64(&mmap, main_header_offset) as usize;

        let header_end = main_header_offset
            .saturating_add(8)
            .saturating_add(header_size);
        if header_end > mmap.len() {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }

        let header_bytes = &mmap[main_header_offset + 8..header_end];
        let (header, _) =
            decode_from_slice::<SparseDataHeader, _>(header_bytes, config::standard())
                .map_err(|_| ScDataErrors::HeaderDecodeFailed)?;

        let n_chunks = header.no_chunks;
        if header.chunk_offsets.len() != n_chunks {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }
        // every chunk needs at least its 8-byte length prefix inside the file
        let chunks_len = mmap.len() as u64 - 64;
        if header
            .chunk_offsets
            .iter()
            .any(|&off| off.checked_add(8).is_none_or(|end| end > chunks_len))
        {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }
        let is_v4_gene = file_header.version == SC_FILE_VERSION_V4 && !header.cell_based;
        let library_index = if is_v4_gene {
            None
        } else {
            Self::check_library_index(
                &mmap,
                file_header.library_index_offset,
                header.cell_based,
                n_chunks,
            )?
        };
        let cell_lib = if is_v4_gene {
            Self::load_cell_library_table(&mmap, file_header.library_index_offset)?
        } else {
            None
        };
        let identity = header.index_map.len() == n_chunks
            && header
                .index_map
                .iter()
                .all(|(&original, &chunk)| original == chunk);

        Ok(Self {
            dims: StoreDims {
                total_cells: header.total_cells,
                total_genes: header.total_genes,
                cell_based: header.cell_based,
            },
            n_chunks,
            offsets: ChunkOffsets::Owned(header.chunk_offsets),
            lookup: if identity {
                ChunkLookup::Identity
            } else {
                ChunkLookup::Hash(header.index_map)
            },
            mmap: Arc::new(mmap),
            file,
            chunks_start: 64,
            target_size: file_header.target_size,
            tf_idf_scale: 0.0,
            library_index,
            version: file_header.version,
            cell_norm: cell_lib
                .as_ref()
                .filter(|_| file_header.target_size != 0.0)
                .map(|lib| Arc::new(gene_norm_table(lib, file_header.target_size))),
            cell_lib,
            feature_kind: FeatureKind::Gene,
            norm_kind: NormKind::LogCpm,
            n_layers: 1,
            layer: 0,
            cell_id_hash: [0; 16],
            idf: None,
            intervals: None,
            sections: Vec::new(),
        })
    }

    /// Read raw layer `layer` instead of layer 0.
    ///
    /// Every read then returns that layer as its raw counts, and the norm is
    /// derived from it. The library size a cell reports stays the sum of
    /// layer 0.
    ///
    /// ### Params
    ///
    /// * `layer` - Layer index
    ///
    /// ### Returns
    ///
    /// The reader, or [`ScDataErrors::LayerOutOfRange`].
    pub fn with_layer(mut self, layer: usize) -> Result<Self, ScDataErrors> {
        if layer >= self.n_layers {
            return Err(ScDataErrors::LayerOutOfRange {
                requested: layer,
                available: self.n_layers,
            });
        }
        self.layer = layer;
        Ok(self)
    }

    /// What the features of the store are.
    ///
    /// ### Returns
    ///
    /// The [`FeatureKind`]; genes for v4.
    pub fn feature_kind(&self) -> FeatureKind {
        self.feature_kind
    }

    /// How the norm of the store derives.
    ///
    /// ### Returns
    ///
    /// The [`NormKind`]; log-CPM for v4.
    pub fn norm_kind(&self) -> NormKind {
        self.norm_kind
    }

    /// Raw layers per chunk.
    ///
    /// ### Returns
    ///
    /// The layer count; one for v4.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Hash of the cell IDs the store was written with.
    ///
    /// ### Returns
    ///
    /// The hash, `None` when unset or pre-v5.
    pub fn cell_id_hash(&self) -> Option<[u8; 16]> {
        (self.cell_id_hash != [0; 16]).then_some(self.cell_id_hash)
    }

    /// Per-feature IDF of a TF-IDF store.
    ///
    /// ### Returns
    ///
    /// The table, `None` without one.
    pub fn idf(&self) -> Option<&[f32]> {
        self.idf.as_deref().map(Vec::as_slice)
    }

    /// The log-CPM target and the TF-IDF scale exactly as the header holds
    /// them, for writers that copy a store's identity.
    ///
    /// ### Returns
    ///
    /// `(target_size, tf_idf_scale)`, `0.0` where unset.
    pub(crate) fn header_scales(&self) -> (f32, f32) {
        (self.target_size, self.tf_idf_scale)
    }

    /// Whether reads can return the normalised layer: a log-CPM store, or a
    /// TF-IDF store with its IDF.
    ///
    /// ### Returns
    ///
    /// `true` when the norm can be derived.
    pub fn norm_available(&self) -> bool {
        self.v4_ctx().norm_available()
    }

    /// TF-IDF scale of a TF-IDF store.
    ///
    /// ### Returns
    ///
    /// The scale, `None` unless the store is TF-IDF with a scale set.
    pub fn tf_idf_scale(&self) -> Option<f32> {
        (self.norm_kind == NormKind::TfIdf && self.tf_idf_scale != 0.0).then_some(self.tf_idf_scale)
    }

    /// The scale the norm is derived with: the TF-IDF scale on a TF-IDF store,
    /// the log-CPM target otherwise.
    ///
    /// ### Returns
    ///
    /// The scale, `0.0` when unknown.
    pub fn norm_scale(&self) -> f32 {
        match self.norm_kind {
            NormKind::TfIdf => self.tf_idf_scale,
            _ => self.target_size,
        }
    }

    /// Payload of a v5 section.
    ///
    /// ### Params
    ///
    /// * `kind` - The section kind
    ///
    /// ### Returns
    ///
    /// The bytes, `None` when the file has no such section.
    fn section_bytes(&self, kind: SectionKind) -> Option<&[u8]> {
        find_section(&self.sections, kind).map(|s| &self.mmap[s.offset..s.offset + s.len])
    }

    /// Caller-owned metadata, see [`crate::store::v5::set_store_metadata`].
    ///
    /// ### Returns
    ///
    /// `(key, value)` pairs in file order; empty without any.
    pub fn store_metadata(&self) -> Vec<(String, String)> {
        self.section_bytes(SectionKind::Metadata)
            .map(parse_key_values)
            .unwrap_or_default()
    }

    /// Who wrote the file: writer version and codec.
    ///
    /// ### Returns
    ///
    /// `(key, value)` pairs; empty for v3, v4 and files written without it.
    pub fn provenance(&self) -> Vec<(String, String)> {
        self.section_bytes(SectionKind::Provenance)
            .map(parse_key_values)
            .unwrap_or_default()
    }

    /// Check every chunk against the checksum section.
    ///
    /// ### Returns
    ///
    /// `true` when all match, `false` without a checksum section, or
    /// [`ScDataErrors::V4ChunkCorrupt`] naming the first bad chunk.
    pub(crate) fn verify_chunks(&self) -> Result<bool, ScDataErrors> {
        let Some(sums) = self.section_bytes(SectionKind::ChunkChecksums) else {
            return Ok(false);
        };
        let bad = (0..self.n_chunks).into_par_iter().find_first(|&c| {
            let (stored, _) = match self.chunk_bytes(c) {
                Ok(b) => b,
                Err(_) => return true,
            };
            xxhash_rust::xxh3::xxh3_64(stored) != le_u64(sums, 8 * c)
        });
        match bad {
            Some(c) => Err(ScDataErrors::V4ChunkCorrupt(format!(
                "chunk {c} does not match its checksum"
            ))),
            None => Ok(true),
        }
    }

    /// The genome interval index of the store's features.
    ///
    /// Decoded from the map on every call; hold on to the result.
    ///
    /// ### Returns
    ///
    /// The index, `None` when the store has none, or
    /// [`ScDataErrors::V5HeaderInvalid`] for a malformed section.
    pub fn genome_intervals(&self) -> Result<Option<GenomeIntervals>, ScDataErrors> {
        self.intervals
            .clone()
            .map(|r| GenomeIntervals::decode(&self.mmap[r]))
            .transpose()
    }

    /// Chunks in the file.
    ///
    /// ### Returns
    ///
    /// The chunk count.
    pub(crate) fn n_chunks(&self) -> usize {
        self.n_chunks
    }

    /// Offset of chunk `c` relative to the end of the file header.
    ///
    /// `c == n_chunks` is the end of the chunks on v5; v3 and v4 do not store
    /// it.
    ///
    /// ### Params
    ///
    /// * `c` - Chunk ordinal, below `n_chunks` (validated on open)
    ///
    /// ### Returns
    ///
    /// The offset, `None` past the stored ones.
    #[inline]
    fn chunk_offset(&self, c: usize) -> Option<u64> {
        match &self.offsets {
            ChunkOffsets::Owned(v) => v.get(c).copied(),
            ChunkOffsets::Mapped(at) => {
                (c <= self.n_chunks).then(|| le_u64(&self.mmap, at + 8 * c))
            }
        }
    }

    /// Original index of every chunk, in file order.
    ///
    /// For the converters and the archive, which walk the file layout.
    ///
    /// ### Returns
    ///
    /// `order[chunk] = original index`, or [`ScDataErrors::V4ChunkCorrupt`]
    /// for a lookup that points outside the file.
    pub(crate) fn chunk_order(&self) -> Result<Vec<usize>, ScDataErrors> {
        let outside =
            |chunk: usize| ScDataErrors::V4ChunkCorrupt(format!("chunk {chunk} outside the file"));
        match &self.lookup {
            ChunkLookup::Identity => Ok((0..self.n_chunks).collect()),
            ChunkLookup::Hash(map) => {
                let mut order = vec![usize::MAX; self.n_chunks];
                for (&orig, &chunk) in map {
                    *order.get_mut(chunk).ok_or_else(|| outside(chunk))? = orig;
                }
                Ok(order)
            }
            &ChunkLookup::Sorted(at, n) => {
                let mut order = vec![usize::MAX; self.n_chunks];
                for i in 0..n {
                    let orig = le_u64(&self.mmap, at + 16 * i) as usize;
                    let chunk = le_u64(&self.mmap, at + 16 * i + 8) as usize;
                    *order.get_mut(chunk).ok_or_else(|| outside(chunk))? = orig;
                }
                Ok(order)
            }
        }
    }

    /// Load the per-cell library-size table of a v4 gene-based file.
    ///
    /// ### Params
    ///
    /// * `mmap` - The mapped file.
    /// * `offset` - `library_index_offset` from the file header; `0` means the
    ///   file has no table.
    ///
    /// ### Returns
    ///
    /// The library sizes as `f32`, `None` without a table, or
    /// [`ScDataErrors::HeaderDecodeFailed`] if the table is malformed.
    fn load_cell_library_table(
        mmap: &memmap2::Mmap,
        offset: u64,
    ) -> Result<Option<Arc<Vec<f32>>>, ScDataErrors> {
        if offset == 0 {
            return Ok(None);
        }
        let offset = offset as usize;
        if offset.saturating_add(8) > mmap.len() {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }
        let count = le_u64(mmap, offset) as usize;
        let end = count
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_add(offset + 8));
        if end.is_none_or(|end| end > mmap.len()) {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }
        Ok(Some(Arc::new(
            (0..count)
                .map(|i| le_u64(mmap, offset + 8 + 8 * i) as f32)
                .collect(),
        )))
    }

    /// Format version of the file this reader maps.
    ///
    /// ### Returns
    ///
    /// [`SC_FILE_VERSION`], [`SC_FILE_VERSION_V4`] or [`SC_FILE_VERSION_V3`].
    pub fn file_version(&self) -> u32 {
        self.version
    }

    /// Whether chunks use the v4 codec, i.e. the file is v4 or v5.
    ///
    /// ### Returns
    ///
    /// `false` only for v3.
    fn is_v4(&self) -> bool {
        self.version >= SC_FILE_VERSION_V4
    }

    /// Decode context for v4-codec chunks of this file.
    ///
    /// ### Returns
    ///
    /// The context.
    fn v4_ctx(&self) -> V4Ctx<'_> {
        V4Ctx {
            target_size: self.norm_scale(),
            cell_lib: self.cell_lib.as_deref().map(Vec::as_slice),
            cell_norm: self.cell_norm.as_deref().map(Vec::as_slice),
            is_cell: self.dims.cell_based,
            n_layers: self.n_layers,
            layer: self.layer,
            norm_f32: self.version >= SC_FILE_VERSION,
            norm_kind: self.norm_kind,
            weights: match self.norm_kind {
                NormKind::TfIdf => self.idf.as_deref().map(Vec::as_slice),
                _ => None,
            },
        }
    }

    /// Validate the library-size index a v3 / v4 file header points at.
    ///
    /// The index is a `u64` count followed by one `u64` per chunk. Like the
    /// main header, every bound is checked against the mmap, so a corrupt
    /// offset or count errors instead of panicking later in a read.
    ///
    /// ### Params
    ///
    /// * `mmap` - The mapped file.
    /// * `offset` - `library_index_offset` from the file header; `0` means the
    ///   file has no index.
    /// * `cell_based` - Whether the file is cell-based.
    /// * `n_chunks` - Chunks in the file.
    ///
    /// ### Returns
    ///
    /// The byte offset of the first entry, `None` without an index, or
    /// [`ScDataErrors::HeaderDecodeFailed`] if the index is malformed.
    fn check_library_index(
        mmap: &memmap2::Mmap,
        offset: u64,
        cell_based: bool,
        n_chunks: usize,
    ) -> Result<Option<usize>, ScDataErrors> {
        if offset == 0 {
            return Ok(None);
        }
        let offset = offset as usize;
        if !cell_based || offset.saturating_add(8) > mmap.len() {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }
        let count = le_u64(mmap, offset) as usize;
        let end = count
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_add(offset + 8));
        if count != n_chunks || end.is_none_or(|end| end > mmap.len()) {
            return Err(ScDataErrors::HeaderDecodeFailed);
        }
        Ok(Some(offset + 8))
    }

    /// Library sizes from the index: 8 bytes per cell, nothing decompressed.
    ///
    /// ### Params
    ///
    /// * `start` - Offset of the first index entry (validated on open).
    /// * `indices` - Original cell indices.
    ///
    /// ### Returns
    ///
    /// Library sizes in the order of `indices`.
    fn library_sizes_from_index(
        &self,
        start: usize,
        indices: &[usize],
    ) -> Result<Vec<usize>, ScDataErrors> {
        indices
            .iter()
            .map(|&original_index| {
                let chunk = self.chunk_index_of(original_index)?;
                Ok(le_u64(&self.mmap, start + 8 * chunk) as usize)
            })
            .collect()
    }

    /// Whether the file carries a library-size index. Lets tests in other
    /// modules check which path `read_cell_library_sizes` takes.
    ///
    /// ### Returns
    ///
    /// `true` if the index is present.
    #[cfg(test)]
    pub(crate) fn has_library_index(&self) -> bool {
        self.library_index.is_some()
    }

    /// Resolve an original cell or gene index to its chunk ordinal.
    ///
    /// A store written in one pass has chunk `i` holding original index `i`,
    /// the overwhelmingly common case, which needs no lookup at all. Otherwise
    /// v3 / v4 consult their hash map and v5 binary-searches its sorted
    /// original-index section in place.
    ///
    /// ### Params
    ///
    /// * `original_index` - Original index of the cell or gene in the data
    ///
    /// ### Returns
    ///
    /// The chunk ordinal, or [`ScDataErrors::ChunkIndexNotFound`].
    fn chunk_index_of(&self, original_index: usize) -> Result<usize, ScDataErrors> {
        let missing = ScDataErrors::ChunkIndexNotFound(original_index);
        let chunk = match &self.lookup {
            ChunkLookup::Identity => Some(original_index),
            ChunkLookup::Hash(map) => map.get(&original_index).copied(),
            &ChunkLookup::Sorted(at, n) => {
                let key = |i: usize| le_u64(&self.mmap, at + 16 * i) as usize;
                let (mut lo, mut hi) = (0usize, n);
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if key(mid) < original_index {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                (lo < n && key(lo) == original_index)
                    .then(|| le_u64(&self.mmap, at + 16 * lo + 8) as usize)
            }
        };
        chunk.filter(|&c| c < self.n_chunks).ok_or(missing)
    }

    /// Count the chunks that store their normalised layer verbatim.
    ///
    /// Only reads the fixed chunk headers. Such chunks skip the norm recompute
    /// on read and are larger on disk.
    ///
    /// ### Returns
    ///
    /// The number of norm-stored chunks, or
    /// [`ScDataErrors::FileVersionMismatch`] on a v3 file, which always stores
    /// the layer.
    pub fn count_norm_stored_chunks(&self) -> Result<usize, ScDataErrors> {
        if !self.is_v4() {
            return Err(ScDataErrors::FileVersionMismatch {
                expected: SC_FILE_VERSION,
                found: self.version,
            });
        }
        let offset = |c: usize| self.chunk_offset(c).expect("chunk below n_chunks");
        (0..self.n_chunks)
            .into_par_iter()
            .map(|c| {
                // chunk layout: u64 compressed length, then the v4 header
                let at = (self.chunks_start + offset(c)) as usize + 8;
                Ok(V4Header::parse(&self.mmap[at..])?.norm_stored() as usize)
            })
            .sum()
    }

    /// Decode the cells `start..end` of a v4 cell file in order, without the
    /// norm layer, handing each to `f`.
    ///
    /// Sequential and allocation free after the first cell: meant for callers
    /// that already fan out over cell ranges. Indices are always decoded, raw
    /// counts only with `want_raw`.
    ///
    /// ### Params
    ///
    /// * `start` - First cell (inclusive)
    /// * `end` - Last cell (exclusive)
    /// * `want_raw` - Decode raw counts; otherwise `f` gets an empty slice
    /// * `f` - Called as `f(cell, indices, raw)`
    ///
    /// ### Returns
    ///
    /// `Ok(())`, a mode or version mismatch, or the first decode error.
    pub(crate) fn visit_cells_v4(
        &self,
        start: usize,
        end: usize,
        want_raw: bool,
        mut f: impl FnMut(usize, &[u32], &[u32]),
    ) -> Result<(), ScDataErrors> {
        if !self.is_cell_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "gene-based",
                requested: "cell-based",
            });
        }
        if !self.is_v4() {
            return Err(ScDataErrors::FileVersionMismatch {
                expected: SC_FILE_VERSION,
                found: self.version,
            });
        }

        self.prefetch(&(start..end).collect::<Vec<_>>());
        let ctx = self.v4_ctx();
        let mut s = V4Decoded::default();
        for cell in start..end {
            let (chunk, _) = self.locate_chunk(cell)?;
            decode_v4(chunk, &ctx, want_raw, false, &mut s)?;
            let raw = if want_raw { &s.raw[..s.idx.len()] } else { &[] };
            f(cell, &s.idx, raw);
        }
        Ok(())
    }

    /// Locate the compressed bytes of a single chunk in the mapping.
    ///
    /// ### Params
    ///
    /// * `original_index` - Original index of the cell or gene in the data
    ///
    /// ### Returns
    ///
    /// The compressed slice, still carrying lz4's own size prefix, and the
    /// chunk's byte offset for error reporting.
    fn locate_chunk(&self, original_index: usize) -> Result<(&[u8], usize), ScDataErrors> {
        self.chunk_bytes(self.chunk_index_of(original_index)?)
    }

    /// The stored bytes of chunk `c`, after its length prefix.
    ///
    /// The length prefix is checked against the next chunk's offset (v5) or
    /// the file (v3, v4), so a corrupt length errors instead of panicking.
    ///
    /// ### Params
    ///
    /// * `c` - Chunk ordinal, below `n_chunks`
    ///
    /// ### Returns
    ///
    /// The slice and the chunk's byte offset, or
    /// [`ScDataErrors::V4ChunkCorrupt`].
    pub(crate) fn chunk_bytes(&self, c: usize) -> Result<(&[u8], usize), ScDataErrors> {
        let corrupt = || ScDataErrors::V4ChunkCorrupt(format!("chunk {c} length outside the file"));
        let start = self.chunks_start + self.chunk_offset(c).ok_or_else(corrupt)?;
        let limit = match self.chunk_offset(c + 1) {
            Some(next) if matches!(self.offsets, ChunkOffsets::Mapped(_)) => {
                self.chunks_start + next
            }
            _ => self.mmap.len() as u64,
        };
        let at = start as usize;
        let size = self
            .mmap
            .get(at..at + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
            .ok_or_else(corrupt)?;
        let end = (start + 8)
            .checked_add(size)
            .filter(|&e| e <= limit)
            .ok_or_else(corrupt)?;
        Ok((&self.mmap[at + 8..end as usize], at))
    }

    /// Locate, decompress and return the raw bytes of a single chunk
    ///
    /// Shared by every read path: resolve the original index to a chunk offset
    /// via the header index map, read the length prefix, then lz4 decompress.
    ///
    /// ### Params
    ///
    /// * `original_index` - Original index of the cell or gene in the data
    ///
    /// ### Returns
    ///
    /// The decompressed chunk bytes.
    fn decompress_chunk(&self, original_index: usize) -> Result<Vec<u8>, ScDataErrors> {
        let (compressed, chunk_offset) = self.locate_chunk(original_index)?;

        decompress_size_prepended(compressed)
            .map_err(|_| ScDataErrors::ChunkDecompressionFailed(chunk_offset as u64))
    }

    /// Decompress a chunk into a caller-owned buffer.
    ///
    /// The allocating [`Self::decompress_chunk`] costs one `Vec` plus one
    /// zero-fill per chunk, which for a shuffled minibatch is a fresh
    /// allocation per cell. Handing in a scratch buffer that is reused across
    /// the cells a worker owns removes both: `resize` only zeroes on growth,
    /// and after the first few cells it stops growing.
    ///
    /// ### Params
    ///
    /// * `original_index` - Original index of the cell or gene in the data
    /// * `scratch` - Buffer to decompress into. Resized to the payload.
    ///
    /// ### Returns
    ///
    /// `Ok(())` with `scratch` holding exactly the decompressed chunk.
    fn decompress_chunk_into(
        &self,
        original_index: usize,
        scratch: &mut Vec<u8>,
    ) -> Result<(), ScDataErrors> {
        let (compressed, chunk_offset) = self.locate_chunk(original_index)?;

        let (size, payload) = uncompressed_size(compressed)
            .map_err(|_| ScDataErrors::ChunkDecompressionFailed(chunk_offset as u64))?;

        scratch.resize(size, 0);
        let written = decompress_into(payload, scratch)
            .map_err(|_| ScDataErrors::ChunkDecompressionFailed(chunk_offset as u64))?;
        scratch.truncate(written);

        Ok(())
    }
}

//////////////
// Prefetch //
//////////////

/// Bytes per explicit read in [`ParallelSparseReader::prefetch`].
///
/// Large reads keep the SSD at a useful queue depth with few syscalls, which
/// page faults on a memory map do not: they arrive one page at a time per
/// thread. Not tuned beyond being large.
#[cfg(unix)]
const PREFETCH_BLOCK: usize = 8 * 1024 * 1024;

/// Smallest total span worth prefetching. Below it the faults of the decode
/// itself are cheaper than setting the reads up. One byte under test, so the
/// unit tests' small stores exercise the prefetch path.
#[cfg(unix)]
const PREFETCH_MIN_BYTES: usize = if cfg!(test) { 1 } else { 16 * 1024 * 1024 };

/// Two chunk spans closer than this are read as one range, so a sparse index
/// set does not turn into many tiny reads.
#[cfg(unix)]
const PREFETCH_MERGE_GAP: usize = 256 * 1024;

/// Ranges probed for residency before deciding whether to prefetch at all.
#[cfg(unix)]
const PREFETCH_SAMPLES: usize = 32;

/// A block with at least this fraction of its pages already cached is skipped,
/// so prefetching a warm store costs a residency probe instead of a copy.
#[cfg(unix)]
const PREFETCH_SKIP_RESIDENT: f64 = 0.9;

impl ParallelSparseReader {
    /// Pull the chunks about to be read into the page cache.
    ///
    /// Best effort: an I/O error or an out-of-range index is ignored here and
    /// surfaces in the read that follows. Chunk spans are merged, cut into
    /// [`PREFETCH_BLOCK`] reads and issued in parallel, skipping blocks that
    /// are already resident. The pages then serve the memory map the decode
    /// reads from.
    ///
    /// ### Params
    ///
    /// * `indices` - Original indices of the cells or genes about to be read
    #[cfg(unix)]
    fn prefetch(&self, indices: &[usize]) {
        use std::os::unix::fs::FileExt;

        let mut spans: Vec<(usize, usize)> = indices
            .iter()
            .filter_map(|&i| {
                let chunk = self.chunk_index_of(i).ok()?;
                let start = (self.chunks_start + self.chunk_offset(chunk)?) as usize;
                let end = self
                    .chunk_offset(chunk + 1)
                    .map_or(start + 8, |next| (self.chunks_start + next) as usize);
                Some((start, end))
            })
            .collect();
        if spans.iter().map(|&(s, e)| e - s).sum::<usize>() < PREFETCH_MIN_BYTES {
            return;
        }

        spans.sort_unstable();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (start, end) in spans {
            match ranges.last_mut() {
                Some(last) if start <= last.1 + PREFETCH_MERGE_GAP => last.1 = last.1.max(end),
                _ => ranges.push((start, end)),
            }
        }

        // SAFETY: `sysconf` with a valid name has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;

        // A handful of probes decide whether any of this is needed, so a warm
        // store pays almost nothing.
        let step = (ranges.len() / PREFETCH_SAMPLES).max(1);
        let probed: Vec<f64> = ranges
            .iter()
            .step_by(step)
            .filter(|&&(s, e)| s / page * page < self.mmap.len() && e > s)
            .map(|&(s, e)| {
                let s = s / page * page;
                let e = e.min(self.mmap.len());
                self.resident_fraction(s, e - s, page)
            })
            .collect();
        if probed.is_empty()
            || probed.iter().sum::<f64>() / probed.len() as f64 >= PREFETCH_SKIP_RESIDENT
        {
            return;
        }
        let blocks: Vec<(usize, usize)> = ranges
            .iter()
            .flat_map(|&(start, end)| {
                let start = start / page * page;
                (start..end)
                    .step_by(PREFETCH_BLOCK)
                    .map(move |s| (s, (s + PREFETCH_BLOCK).min(end)))
            })
            .filter(|&(s, e)| s < self.mmap.len() && e > s)
            .collect();

        blocks.par_iter().for_each_init(
            || vec![0u8; PREFETCH_BLOCK],
            |buf, &(start, end)| {
                let end = end.min(self.mmap.len());
                if self.resident_fraction(start, end - start, page) >= PREFETCH_SKIP_RESIDENT {
                    return;
                }
                let mut done = 0;
                while done < end - start {
                    match self
                        .file
                        .read_at(&mut buf[..end - start - done], (start + done) as u64)
                    {
                        Ok(0) | Err(_) => break,
                        Ok(n) => done += n,
                    }
                }
            },
        );
    }

    /// Fraction of the pages of a byte range that sit in the page cache.
    ///
    /// ### Params
    ///
    /// * `start` - Page-aligned byte offset into the file
    /// * `len` - Length in bytes
    /// * `page` - System page size
    ///
    /// ### Returns
    ///
    /// The resident fraction, `0.0` if the probe fails.
    #[cfg(unix)]
    fn resident_fraction(&self, start: usize, len: usize, page: usize) -> f64 {
        let mut vec = vec![0u8; len.div_ceil(page)];
        // SAFETY: `start..start + len` lies inside the mapping, `start` is
        // page aligned, and `vec` holds one byte per page as `mincore` needs.
        let rc = unsafe {
            libc::mincore(
                self.mmap.as_ptr().add(start) as *mut libc::c_void,
                len,
                vec.as_mut_ptr().cast(),
            )
        };
        if rc != 0 {
            return 0.0;
        }
        vec.iter().filter(|&&b| b & 1 != 0).count() as f64 / vec.len().max(1) as f64
    }

    /// No-op where explicit prefetching is not implemented.
    ///
    /// ### Params
    ///
    /// * `_indices` - Unused
    #[cfg(not(unix))]
    fn prefetch(&self, _indices: &[usize]) {}
}

///////////////////////
// Fused batch reads //
///////////////////////

/// Pieces per rayon thread in [`read_fused`].
///
/// One piece per thread maximises scratch reuse but leaves no work to steal,
/// so the slowest thread (an efficiency core, or the piece holding the
/// heaviest genes) sets the wall time. Several pieces per thread restore
/// stealing at the cost of a few more scratch buffers.
const FUSED_PIECES_PER_THREAD: usize = 8;

/// Genes per batch in [`SingleCellReading::read_genes_sparse_csr`].
///
/// Larger batches amortise the per-call read overhead; the transient batch
/// CSC grows with it and sits on top of the output at the peak.
const CSR_GENE_BATCH: usize = 200;

/// Row blocks per rayon thread in [`SingleCellReading::read_genes_sparse_csr`].
///
/// Each block transposes its rows of a batch locally. Too few blocks and the
/// scatter spills out of cache; too many and the per-batch row pointers and
/// binary searches start to cost, and the blocks get small enough that the
/// allocator keeps them in its heap instead of returning them.
const CSR_ROW_BLOCKS_PER_THREAD: usize = 32;

/// Smallest buffer a row block reserves once it is projected to matter.
///
/// glibc serves anything under its mmap threshold (at most 32 MiB) from the
/// heap, where a freed block stays resident, so the assembly would free
/// nothing. Reserving past the ceiling gives every block its own mapping,
/// which `free` returns and `realloc` grows by `mremap`. The reserve is
/// virtual: pages only commit as they are written.
const CSR_BLOCK_MIN_RESERVE_BYTES: usize = 40 << 20;

/// Projected row block size below which [`CSR_BLOCK_MIN_RESERVE_BYTES`] is
/// not applied: the whole matrix is then small enough that its peak does not
/// matter.
const CSR_BLOCK_RESERVE_FROM_BYTES: usize = 1 << 20;

/// One row block's share of every gene batch read so far.
///
/// All batches append to one buffer per layer, so the buffers grow large
/// enough to be mapped directly and are handed back whole on drop.
struct CsrRowBlock<T> {
    /// Per batch, row pointers into the buffers (absolute offsets), one more
    /// than the block's rows.
    ptrs: Vec<Vec<u32>>,
    /// Raw counts, when requested.
    data: Option<Vec<T>>,
    /// Normalised counts, when requested.
    data_2: Option<Vec<f32>>,
    /// Global column of each entry.
    indices: Vec<u32>,
}

impl<T: BixverseNumeric> CsrRowBlock<T> {
    /// Empty block.
    ///
    /// ### Params
    ///
    /// * `layers` - `(raw, norm)`: which layers to carry
    ///
    /// ### Returns
    ///
    /// The block.
    fn new(layers: (bool, bool)) -> Self {
        Self {
            ptrs: Vec::new(),
            data: layers.0.then(Vec::new),
            data_2: layers.1.then(Vec::new),
            indices: Vec::new(),
        }
    }

    /// Reserve the block's projected final size before its first allocation,
    /// so its buffers are mapped from the start rather than grown on the heap.
    ///
    /// ### Params
    ///
    /// * `first_batch` - Entries the first batch brings to this block
    /// * `n_batches` - Total gene batches
    fn reserve_projected(&mut self, first_batch: usize, n_batches: usize) {
        let projected = first_batch * n_batches;
        if projected * size_of::<u32>() < CSR_BLOCK_RESERVE_FROM_BYTES {
            return;
        }
        // a quarter of headroom for batches denser than the first
        let target = projected + projected / 4;
        let elems = |size: usize| target.max(CSR_BLOCK_MIN_RESERVE_BYTES / size);
        self.indices.reserve_exact(elems(size_of::<u32>()));
        if let Some(d) = self.data.as_mut() {
            d.reserve_exact(elems(size_of::<T>().max(1)));
        }
        if let Some(d) = self.data_2.as_mut() {
            d.reserve_exact(elems(size_of::<f32>()));
        }
    }

    /// Transpose the rows `r0..r1` of a gene batch onto the end of the block.
    ///
    /// Relies on the rows ascending within every gene, which
    /// [`SingleCellReading::read_genes_sparse`] guarantees, so each gene's
    /// share of the block is one binary-searched span. The first batch also
    /// sizes the block's buffers, see [`Self::reserve_projected`].
    ///
    /// ### Params
    ///
    /// * `csc` - The batch, cells x batch genes
    /// * `r0` / `r1` - Row range of the block
    /// * `c0` - Output column of the first gene in the batch
    /// * `n_batches` - Total gene batches
    fn append(
        &mut self,
        csc: &CompressedSparseData2<T, f32>,
        r0: u32,
        r1: u32,
        c0: u32,
        n_batches: usize,
    ) {
        let rows = (r1 - r0) as usize;
        let spans: Vec<(usize, usize)> = csc
            .indptr
            .windows(2)
            .map(|w| {
                let (s, e) = (w[0] as usize, w[1] as usize);
                let col = &csc.indices[s..e];
                (
                    s + col.partition_point(|&i| i < r0),
                    s + col.partition_point(|&i| i < r1),
                )
            })
            .collect();

        let mut ptr = vec![0u32; rows + 1];
        ptr[0] = self.indices.len() as u32;
        for &(lo, hi) in &spans {
            for &i in &csc.indices[lo..hi] {
                ptr[(i - r0) as usize + 1] += 1;
            }
        }
        for j in 0..rows {
            ptr[j + 1] += ptr[j];
        }

        let end = ptr[rows] as usize;
        if self.ptrs.is_empty() {
            self.reserve_projected(end, n_batches);
        }
        self.indices.resize(end, 0);
        if let Some(d) = self.data.as_mut() {
            d.resize(end, T::default());
        }
        if let Some(d) = self.data_2.as_mut() {
            d.resize(end, 0.0);
        }
        let src_2 = csc.data_2.as_deref();
        let mut cursor: Vec<u32> = ptr[..rows].to_vec();
        for (g, &(lo, hi)) in spans.iter().enumerate() {
            for k in lo..hi {
                let c = &mut cursor[(csc.indices[k] - r0) as usize];
                let pos = *c as usize;
                self.indices[pos] = c0 + g as u32;
                if let Some(d) = self.data.as_mut() {
                    d[pos] = csc.data[k];
                }
                if let (Some(d), Some(s)) = (self.data_2.as_mut(), src_2) {
                    d[pos] = s[k];
                }
                *c += 1;
            }
        }
        self.ptrs.push(ptr);
    }

    /// Copy the block's rows into its slot of the output, batch after batch
    /// within each row, then free the block.
    ///
    /// ### Params
    ///
    /// * `rows` - Rows in the block
    /// * `data` - Output raw counts slot, empty when not requested
    /// * `data_2` - Output normalised slot, empty when not requested
    /// * `indices` - Output column index slot
    fn copy_rows(self, rows: usize, data: &mut [T], data_2: &mut [f32], indices: &mut [u32]) {
        let mut pos = 0;
        for j in 0..rows {
            for ptr in &self.ptrs {
                let (s, e) = (ptr[j] as usize, ptr[j + 1] as usize);
                let len = e - s;
                indices[pos..pos + len].copy_from_slice(&self.indices[s..e]);
                if let Some(src) = self.data.as_ref() {
                    data[pos..pos + len].copy_from_slice(&src[s..e]);
                }
                if let Some(src) = self.data_2.as_ref() {
                    data_2[pos..pos + len].copy_from_slice(&src[s..e]);
                }
                pos += len;
            }
        }
    }
}

/// Body of [`SingleCellReading::read_genes_sparse_csr`], with the batch size
/// exposed so tests can force several batches on a toy store.
///
/// ### Params
///
/// * `reader` - The gene-major store
/// * `indices` - Gene indices, in output column order
/// * `cells_to_keep` - Cells to keep, in output row order, or `None`
/// * `data_layer` - Which layer(s) to materialise
/// * `batch` - Genes per batch
///
/// ### Returns
///
/// The CSR, or whatever [`SingleCellReading::read_genes_sparse`] raises.
fn genes_sparse_to_csr<S, T>(
    reader: &S,
    indices: &[usize],
    cells_to_keep: Option<&IndexSet<u32>>,
    data_layer: &DataLayerReturn,
    batch: usize,
) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
where
    S: SingleCellReading + ?Sized,
    T: BixverseNumeric + FromPrimitive,
{
    let n = cells_to_keep.map_or(reader.dims().total_cells, |s| s.len());
    let layers = wanted_layers(data_layer);
    let n_blocks = (rayon::current_num_threads() * CSR_ROW_BLOCKS_PER_THREAD)
        .min(n)
        .max(1);
    let block_rows = n.div_ceil(n_blocks).max(1);
    let bounds: Vec<(u32, u32)> = (0..n_blocks)
        .map(|r| {
            (
                (r * block_rows).min(n) as u32,
                ((r + 1) * block_rows).min(n) as u32,
            )
        })
        .collect();

    let mut blocks: Vec<CsrRowBlock<T>> = (0..n_blocks).map(|_| CsrRowBlock::new(layers)).collect();
    let n_batches = indices.len().div_ceil(batch);
    for (bi, genes) in indices.chunks(batch).enumerate() {
        let csc = reader.read_genes_sparse::<T>(genes, cells_to_keep, data_layer)?;
        let c0 = (bi * batch) as u32;
        blocks
            .par_iter_mut()
            .zip(bounds.par_iter())
            .for_each(|(block, &(r0, r1))| block.append(&csc, r0, r1, c0, n_batches));
    }

    // the batch reads leave their per-thread fragments on glibc's free lists,
    // resident; hand them back before the output starts to commit
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }

    let mut row_nnz = vec![0u32; n];
    row_nnz
        .par_chunks_mut(block_rows)
        .zip(blocks.par_iter())
        .for_each(|(out, block)| {
            for ptr in &block.ptrs {
                for (j, o) in out.iter_mut().enumerate() {
                    *o += ptr[j + 1] - ptr[j];
                }
            }
        });
    let mut indptr = vec![0u32; n + 1];
    for i in 0..n {
        indptr[i + 1] = indptr[i] + row_nnz[i];
    }
    drop(row_nnz);

    let nnz = indptr[n] as usize;
    let mut data: Vec<T> = if layers.0 {
        vec![T::default(); nnz]
    } else {
        Vec::new()
    };
    let mut data_2: Option<Vec<f32>> = layers.1.then(|| vec![0.0; nnz]);
    let mut out_indices: Vec<u32> = vec![0; nnz];

    let block_offsets: Vec<usize> = std::iter::once(0)
        .chain(bounds.iter().map(|&(_, r1)| indptr[r1 as usize] as usize))
        .collect();
    let data_parts = split_at_offsets(&mut data, &block_offsets, layers.0);
    let data_2_parts = split_at_offsets(
        data_2.as_deref_mut().unwrap_or_default(),
        &block_offsets,
        layers.1,
    );
    let index_parts = split_at_offsets(&mut out_indices, &block_offsets, true);

    blocks
        .into_par_iter()
        .zip(data_parts)
        .zip(data_2_parts)
        .zip(index_parts)
        .zip(bounds.par_iter())
        .for_each(|((((block, d), d2), idx), &(r0, r1))| {
            block.copy_rows((r1 - r0) as usize, d, d2, idx);
        });

    Ok(CompressedSparseData2 {
        data,
        indices: out_indices,
        indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2,
        shape: (n, indices.len()),
    })
}

/// One worker's share of a fused batch read, before the fragments are
/// concatenated.
///
/// Runs are CSR rows (cell reads) or CSC columns (gene reads). The layers that
/// were not requested stay `None` and cost nothing.
pub(crate) struct SparseFragment<T> {
    /// Raw counts, when requested.
    data: Option<Vec<T>>,
    /// Normalised counts, when requested.
    data_2: Option<Vec<f32>>,
    /// Minor-axis indices, renumbered when a row map is in play.
    indices: Vec<u32>,
    /// Entries per run, in order, for the prefix sum.
    lengths: Vec<usize>,
    /// Reused `(output row, entry position)` scratch for filtered runs.
    kept: Vec<(u32, u32)>,
}

impl<T: BixverseNumeric + FromPrimitive> SparseFragment<T> {
    /// Empty fragment.
    ///
    /// ### Params
    ///
    /// * `layers` - `(raw, norm)`: which layers to materialise
    /// * `n_runs` - Runs this worker will append
    ///
    /// ### Returns
    ///
    /// The fragment.
    fn new(layers: (bool, bool), n_runs: usize) -> Self {
        Self {
            data: layers.0.then(Vec::new),
            data_2: layers.1.then(Vec::new),
            indices: Vec::new(),
            lengths: Vec::with_capacity(n_runs),
            kept: Vec::new(),
        }
    }

    /// Append one v4 run.
    ///
    /// Without a `row_map` the chunk is decoded straight into this fragment's
    /// vectors, with no intermediate copy. With one, the run is decoded into
    /// `scratch` and filtered through [`Self::push_run`].
    ///
    /// ### Params
    ///
    /// * `chunk` - The v4 chunk, header first
    /// * `ctx` - File context
    /// * `scratch` - The worker's reusable buffers
    /// * `row_map` - Optional dense `index -> output row` map
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the first decode or [`ScDataErrors::RawCountOverflow`]
    /// error.
    pub(crate) fn push_v4(
        &mut self,
        chunk: &[u8],
        ctx: &V4Ctx,
        scratch: &mut V4Decoded,
        row_map: Option<&[u32]>,
    ) -> Result<(), ScDataErrors> {
        let (want_raw, want_norm) = self.layers();
        if row_map.is_some() {
            decode_v4(chunk, ctx, want_raw, want_norm, scratch)?;
            return self.push_run(&V4Run::new(scratch), row_map);
        }

        let h = decode_v4_into(
            chunk,
            ctx,
            want_raw,
            &mut scratch.bytes,
            &mut scratch.raw,
            &mut self.indices,
            self.data_2.as_mut(),
        )?;
        if let Some(data) = self.data.as_mut() {
            extend_raw_checked(data, scratch.raw.iter().copied())?;
        }
        self.lengths.push(h.nnz);
        Ok(())
    }

    /// Which layers this fragment materialises.
    ///
    /// ### Returns
    ///
    /// `(raw, norm)`.
    pub(crate) fn layers(&self) -> (bool, bool) {
        (self.data.is_some(), self.data_2.is_some())
    }

    /// Append one run.
    ///
    /// With a `row_map` the run is filtered and renumbered on the way in:
    /// entries whose index maps to [`INDEX_DROPPED`] (or past the map) are
    /// dropped, and the rest are reordered by output row. Stored indices
    /// ascend, so the sort only runs when the map itself is not monotone. This
    /// is the fused form of [`CscGeneChunk::filter_selected_cells_dense`] and
    /// gives identical output.
    ///
    /// ### Params
    ///
    /// * `run` - The stored run
    /// * `row_map` - Optional dense `index -> output row` map
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::RawCountOverflow`] for a count that does
    /// not fit `T`.
    #[inline]
    pub(crate) fn push_run(
        &mut self,
        run: &impl RunSource,
        row_map: Option<&[u32]>,
    ) -> Result<(), ScDataErrors> {
        let Some(map) = row_map else {
            if let Some(data) = self.data.as_mut() {
                run.extend_raw(data)?;
            }
            if let Some(data_2) = self.data_2.as_mut() {
                run.extend_norm(data_2);
            }
            run.extend_indices(&mut self.indices);
            self.lengths.push(run.nnz());
            return Ok(());
        };

        self.kept.clear();
        self.kept.extend(
            (0..run.nnz())
                .filter_map(|i| remap(map, run.index_at(i) as usize).map(|row| (row, i as u32))),
        );
        if !self.kept.is_sorted_by_key(|&(row, _)| row) {
            self.kept.sort_unstable_by_key(|&(row, _)| row);
        }

        if let Some(data) = self.data.as_mut() {
            for &(_, i) in &self.kept {
                data.push(raw_to(run.raw_at(i as usize))?);
            }
        }
        if let Some(data_2) = self.data_2.as_mut() {
            data_2.extend(self.kept.iter().map(|&(_, i)| run.norm_at(i as usize)));
        }
        self.indices.extend(self.kept.iter().map(|&(row, _)| row));
        self.lengths.push(self.kept.len());

        Ok(())
    }
}

/// Which layers a [`DataLayerReturn`] asks for.
///
/// ### Params
///
/// * `data_layer` - The requested layer(s)
///
/// ### Returns
///
/// `(raw, norm)`.
fn wanted_layers(data_layer: &DataLayerReturn) -> (bool, bool) {
    (
        matches!(
            data_layer,
            DataLayerReturn::Raw | DataLayerReturn::BothLayers
        ),
        matches!(
            data_layer,
            DataLayerReturn::Norm | DataLayerReturn::BothLayers
        ),
    )
}

/// Read a batch of runs straight into one compressed sparse matrix.
///
/// The engine behind [`SingleCellReading::read_cells_sparse`] and
/// [`SingleCellReading::read_genes_sparse`]. Work is split into one contiguous
/// piece per rayon worker rather than per run, so each piece reuses one
/// scratch buffer (and one [`SparseFragment`]) across all of its runs, and
/// allocation scales with the thread count rather than the batch size. The
/// pieces are concatenated in order, so the output run order follows
/// `indices`.
///
/// ### Params
///
/// * `indices` - Runs to read, in output order
/// * `data_layer` - Which layer(s) to materialise
/// * `cs_type` - Layout of the output: CSR for cell runs, CSC for gene runs
/// * `shape` - `(cells, genes)` of the output
/// * `fill` - Appends run `indices[i]` to the fragment; the [`V4Decoded`] is
///   the worker's reusable scratch
///
/// ### Returns
///
/// The batch, raw counts in `data` (empty when not requested) and normalised
/// counts in `data_2`.
pub(crate) fn read_fused<T, F>(
    indices: &[usize],
    data_layer: &DataLayerReturn,
    cs_type: CompressedSparseFormat,
    shape: (usize, usize),
    fill: F,
) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
where
    T: BixverseNumeric + FromPrimitive,
    F: Fn(usize, &mut V4Decoded, &mut SparseFragment<T>) -> Result<(), ScDataErrors> + Sync,
{
    let layers = wanted_layers(data_layer);
    let n_pieces = rayon::current_num_threads().max(1) * FUSED_PIECES_PER_THREAD;
    let piece = indices.len().div_ceil(n_pieces).max(1);

    let fragments: Vec<SparseFragment<T>> = indices
        .par_chunks(piece)
        .map(|runs| {
            let mut scratch = V4Decoded::default();
            let mut fragment = SparseFragment::new(layers, runs.len());
            for &run in runs {
                fill(run, &mut scratch, &mut fragment)?;
            }
            Ok(fragment)
        })
        .collect::<Result<Vec<_>, ScDataErrors>>()?;

    // Concatenate in parallel: each fragment copies into its own disjoint
    // slot of preallocated outputs. A serial append here costs more than the
    // whole parallel decode. The outputs are zero-allocated, so their pages
    // only commit as they are written, and each fragment is freed once
    // copied: the peak stays near one copy rather than two.
    let nnz_offsets: Vec<usize> = std::iter::once(0)
        .chain(fragments.iter().scan(0, |acc, f| {
            *acc += f.indices.len();
            Some(*acc)
        }))
        .collect();
    let run_offsets: Vec<usize> = std::iter::once(0)
        .chain(fragments.iter().scan(0, |acc, f| {
            *acc += f.lengths.len();
            Some(*acc)
        }))
        .collect();
    let nnz = *nnz_offsets.last().expect("seeded with 0");

    let mut data: Vec<T> = if layers.0 {
        vec![T::default(); nnz]
    } else {
        Vec::new()
    };
    let mut data_2: Option<Vec<f32>> = layers.1.then(|| vec![0.0; nnz]);
    let mut out_indices: Vec<u32> = vec![0; nnz];
    let mut indptr: Vec<u32> = vec![0; indices.len() + 1];

    let data_parts = split_at_offsets(&mut data, &nnz_offsets, layers.0);
    let data_2_parts = split_at_offsets(
        data_2.as_deref_mut().unwrap_or_default(),
        &nnz_offsets,
        layers.1,
    );
    let index_parts = split_at_offsets(&mut out_indices, &nnz_offsets, true);
    let indptr_parts = split_at_offsets(&mut indptr[1..], &run_offsets, true);

    fragments
        .into_par_iter()
        .zip(data_parts)
        .zip(data_2_parts)
        .zip(index_parts)
        .zip(indptr_parts)
        .zip(nnz_offsets.par_iter())
        .for_each(|(((((fragment, d), d2), idx), ptr), &base)| {
            if let Some(src) = fragment.data.as_ref() {
                d.copy_from_slice(src);
            }
            if let Some(src) = fragment.data_2.as_ref() {
                d2.copy_from_slice(src);
            }
            idx.copy_from_slice(&fragment.indices);
            let mut offset = base;
            for (out, &len) in ptr.iter_mut().zip(&fragment.lengths) {
                offset += len;
                *out = offset as u32;
            }
            drop(fragment);
        });

    Ok(CompressedSparseData2 {
        data,
        indices: out_indices,
        indptr,
        cs_type,
        data_2,
        shape,
    })
}

/// Split a slice into consecutive parts at the given offsets.
///
/// ### Params
///
/// * `buf` - The slice to split
/// * `offsets` - Prefix offsets with a leading 0, one more than the parts
/// * `active` - `false` for a layer that was not requested: `buf` is empty and
///   every part comes back empty, so the parts still zip one per fragment
///
/// ### Returns
///
/// The parts, in order.
fn split_at_offsets<'a, X>(buf: &'a mut [X], offsets: &[usize], active: bool) -> Vec<&'a mut [X]> {
    let n_parts = offsets.len().saturating_sub(1);
    if !active {
        return (0..n_parts).map(|_| &mut [][..]).collect();
    }
    let mut parts = Vec::with_capacity(n_parts);
    let mut rest = buf;
    for w in offsets.windows(2) {
        let (head, tail) = rest.split_at_mut(w[1] - w[0]);
        parts.push(head);
        rest = tail;
    }
    parts
}

/////////////////////
// Cell row subset //
/////////////////////

/// Dense map and output row count for a cell subset.
///
/// ### Params
///
/// * `cells_to_keep` - Cells to keep, in output order, or `None` for all
/// * `total_cells` - Cells in the store
///
/// ### Returns
///
/// `(map, n_rows)`: the map is `None` when no filtering is asked for.
pub(crate) fn cell_row_map(
    cells_to_keep: Option<&IndexSet<u32>>,
    total_cells: usize,
) -> (Option<Vec<u32>>, usize) {
    match cells_to_keep {
        Some(set) => (Some(keep_set_index_map(set)), set.len()),
        None => (None, total_cells),
    }
}

//////////////
// CsrBatch //
//////////////

/// A batch of cells as one CSR matrix.
///
/// The layout a training loop wants: contiguous `data`, `indices` and `indptr`
/// that hand straight to `scipy.sparse.csr_matrix` or `torch.sparse_csr_tensor`
/// without another pass. Whichever layer was not requested is `None`; both
/// share one sparsity pattern, so `indices` and `indptr` are never duplicated.
#[derive(Clone, Debug)]
pub struct CsrBatch {
    /// Raw counts as `f32`, in CSR order, when requested.
    pub data_raw: Option<Vec<f32>>,
    /// Normalised counts as `f32`, in CSR order, when requested.
    pub data_norm: Option<Vec<f32>>,
    /// Gene indices, one per non-zero.
    pub indices: Vec<u32>,
    /// Row pointers, of length `shape.0 + 1`.
    pub indptr: Vec<u32>,
    /// `(cells in the batch, total genes)`.
    pub shape: (usize, usize),
}

impl ParallelSparseReader {
    /// Read a batch of cells straight into CSR, as `f32`, optionally over a
    /// subset of genes.
    ///
    /// [`SingleCellReading::read_cells_sparse`] with `T = f32`, unpacked into
    /// the [`CsrBatch`] the Python loader hands to scipy or torch. A gene
    /// selection is applied while each cell is decoded, so no batch over every
    /// gene is built; column indices come out sorted within each row.
    ///
    /// ### Params
    ///
    /// * `indices` - Original cell indices, in the order the batch wants them
    /// * `genes` - Store genes to keep, in output column order, or `None` for
    ///   every gene in store order
    /// * `data_layer` - Which layer(s) to materialise
    ///
    /// ### Returns
    ///
    /// The batch as CSR, [`ScDataErrors::ReaderModeMismatch`] on a gene-major
    /// store, or [`ScDataErrors::InvalidArgument`] for an out-of-range or
    /// repeated gene.
    pub fn read_cells_csr(
        &self,
        indices: &[usize],
        genes: Option<&[usize]>,
        data_layer: &DataLayerReturn,
    ) -> Result<CsrBatch, ScDataErrors> {
        let gene_map = genes
            .map(|g| gene_subset_map(g, self.dims.total_genes))
            .transpose()?;
        let n_cols = genes.map_or(self.dims.total_genes, <[usize]>::len);
        let csr = self.read_cells_fused::<f32>(indices, gene_map.as_deref(), n_cols, data_layer)?;

        Ok(CsrBatch {
            data_raw: wanted_layers(data_layer).0.then_some(csr.data),
            data_norm: csr.data_2,
            indices: csr.indices,
            indptr: csr.indptr,
            shape: csr.shape,
        })
    }

    /// Read cells into CSR, filtering genes through a dense map, see
    /// [`read_fused`].
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices, in output row order
    /// * `gene_map` - Optional dense `gene -> output column` map
    /// * `n_cols` - Output columns: the kept genes, or every gene
    /// * `data_layer` - Which layer(s) to materialise
    ///
    /// ### Returns
    ///
    /// The batch as CSR.
    fn read_cells_fused<T>(
        &self,
        indices: &[usize],
        gene_map: Option<&[u32]>,
        n_cols: usize,
        data_layer: &DataLayerReturn,
    ) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
    where
        T: BixverseNumeric + FromPrimitive,
    {
        if !self.is_cell_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "gene-based",
                requested: "cell-based",
            });
        }

        self.prefetch(indices);

        read_fused(
            indices,
            data_layer,
            CompressedSparseFormat::Csr,
            (indices.len(), n_cols),
            |cell, scratch, fragment| {
                if self.is_v4() {
                    let (chunk, _) = self.locate_chunk(cell)?;
                    return fragment.push_v4(chunk, &self.v4_ctx(), scratch, gene_map);
                }
                self.decompress_chunk_into(cell, &mut scratch.bytes)?;
                let p = csr_payload(&scratch.bytes)?;
                fragment.push_run(&p, gene_map)
            },
        )
    }
}

//////////////////////////
// Trait implementation //
//////////////////////////

impl SingleCellReading for ParallelSparseReader {
    /// Read in cells by indices in a multi-threaded manner
    ///
    /// ### Params
    ///
    /// * `indices` - Slice of index positions of the cells to retrieve
    ///
    /// ### Returns
    ///
    /// Returns an array of `CsrCellChunk`.
    fn read_cells_parallel(&self, indices: &[usize]) -> Result<Vec<CsrCellChunk>, ScDataErrors> {
        if !self.is_cell_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "gene-based",
                requested: "cell-based",
            });
        }

        self.prefetch(indices);

        if self.is_v4() {
            let ctx = self.v4_ctx();
            return indices
                .par_iter()
                .map_init(V4Decoded::default, |scratch, &original_index| {
                    let (chunk, _) = self.locate_chunk(original_index)?;
                    v4_cell_chunk(chunk, &ctx, scratch)
                })
                .collect();
        }

        indices
            .par_iter()
            .map(|&original_index| {
                let decompressed = self.decompress_chunk(original_index)?;

                CsrCellChunk::read_from_buffer(&decompressed)
            })
            .collect()
    }

    /// Read a batch of cells straight into CSR
    ///
    /// Decompresses each cell into the worker's scratch buffer and appends
    /// its payload in place, see [`read_fused`].
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices, in output row order
    /// * `data_layer` - Which layer(s) to materialise
    ///
    /// ### Returns
    ///
    /// The batch as CSR.
    fn read_cells_sparse<T>(
        &self,
        indices: &[usize],
        data_layer: &DataLayerReturn,
    ) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
    where
        T: BixverseNumeric + FromPrimitive,
    {
        self.read_cells_fused(indices, None, self.dims.total_genes, data_layer)
    }

    /// Read a batch of genes straight into CSC, optionally filtering cells
    ///
    /// Decompresses each gene into the worker's scratch buffer and appends
    /// its payload in place, filtering through a dense cell map built once
    /// per call, see [`read_fused`].
    ///
    /// ### Params
    ///
    /// * `indices` - Gene indices, in output column order
    /// * `cells_to_keep` - Cells to keep, in output row order, or `None`
    /// * `data_layer` - Which layer(s) to materialise
    ///
    /// ### Returns
    ///
    /// The batch as CSC.
    fn read_genes_sparse<T>(
        &self,
        indices: &[usize],
        cells_to_keep: Option<&IndexSet<u32>>,
        data_layer: &DataLayerReturn,
    ) -> Result<CompressedSparseData2<T, f32>, ScDataErrors>
    where
        T: BixverseNumeric + FromPrimitive,
    {
        if !self.is_gene_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "cell-based",
                requested: "gene-based",
            });
        }

        let (row_map, n_rows) = cell_row_map(cells_to_keep, self.dims.total_cells);

        self.prefetch(indices);

        read_fused(
            indices,
            data_layer,
            CompressedSparseFormat::Csc,
            (n_rows, indices.len()),
            |gene, scratch, fragment| {
                if self.is_v4() {
                    let (chunk, _) = self.locate_chunk(gene)?;
                    return fragment.push_v4(chunk, &self.v4_ctx(), scratch, row_map.as_deref());
                }
                self.decompress_chunk_into(gene, &mut scratch.bytes)?;
                let p = csc_payload(&scratch.bytes)?;
                fragment.push_run(&p, row_map.as_deref())
            },
        )
    }

    /// Read in genes by indices in a multi-threaded manner
    ///
    /// ### Params
    ///
    /// * `indices` - Slice of index positions of the genes to retrieve
    ///
    /// ### Returns
    ///
    /// Returns an array of `CscGeneChunk`.
    fn read_gene_parallel(&self, indices: &[usize]) -> Result<Vec<CscGeneChunk>, ScDataErrors> {
        if !self.is_gene_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "cell-based",
                requested: "gene-based",
            });
        }

        self.prefetch(indices);

        if self.is_v4() {
            let ctx = self.v4_ctx();
            return indices
                .par_iter()
                .map_init(V4Decoded::default, |scratch, &original_index| {
                    let (chunk, _) = self.locate_chunk(original_index)?;
                    v4_gene_chunk(chunk, &ctx, scratch)
                })
                .collect();
        }

        indices
            .par_iter()
            .map(|&original_index| {
                let decompressed = self.decompress_chunk(original_index)?;

                CscGeneChunk::read_from_buffer(&decompressed)
            })
            .collect()
    }

    /// Read genes with raw counts and indices only.
    ///
    /// On v4 and v5 the norm layer is neither recomputed nor converted,
    /// `data_norm` comes back empty and the decoded indices move into the
    /// chunk. v3 reads both layers, as [`Self::read_gene_parallel`].
    ///
    /// ### Params
    ///
    /// * `indices` - Slice of index positions of the genes to retrieve
    ///
    /// ### Returns
    ///
    /// Returns an array of `CscGeneChunk` with an empty `data_norm` on v4 and
    /// v5.
    fn read_gene_parallel_raw(&self, indices: &[usize]) -> Result<Vec<CscGeneChunk>, ScDataErrors> {
        if !self.is_v4() {
            return self.read_gene_parallel(indices);
        }
        if !self.is_gene_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "cell-based",
                requested: "gene-based",
            });
        }

        self.prefetch(indices);

        let ctx = self.v4_ctx();
        indices
            .par_iter()
            .map_init(V4Decoded::default, |s, &original_index| {
                let (chunk, _) = self.locate_chunk(original_index)?;
                let h = decode_v4(chunk, &ctx, true, false, s)?;
                Ok(CscGeneChunk {
                    data_raw: RawCounts::from_u32_auto(&s.raw[..h.nnz]),
                    data_norm: Vec::new(),
                    nnz: h.nnz,
                    indices: std::mem::take(&mut s.idx),
                    original_index: h.original_index,
                    to_keep: h.to_keep,
                })
            })
            .collect()
    }

    /// Read in genes by indices in a multi-threaded manner
    ///
    /// And filter the genes by `cells_to_keep` in parallel. Overrides the
    /// default so that reading and filtering are fused into a single pass.
    ///
    /// ### Params
    ///
    /// * `indices` - Slice of index positions of the genes to retrieve
    /// * `cells_to_keep` - Cell index positions to keep, in output order
    ///
    /// ### Returns
    ///
    /// Returns an array of `CscGeneChunk`.
    fn read_gene_parallel_filtered(
        &self,
        indices: &[usize],
        cells_to_keep: &IndexSet<u32>,
    ) -> Result<Vec<CscGeneChunk>, ScDataErrors> {
        if !self.is_gene_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "cell-based",
                requested: "gene-based",
            });
        }

        let cell_map = keep_set_index_map(cells_to_keep);

        self.prefetch(indices);

        if self.is_v4() {
            let ctx = self.v4_ctx();
            return indices
                .par_iter()
                .map_init(V4Decoded::default, |scratch, &original_index| {
                    let (chunk, _) = self.locate_chunk(original_index)?;
                    let mut chunk = v4_gene_chunk(chunk, &ctx, scratch)?;
                    chunk.filter_selected_cells_dense(&cell_map);
                    Ok(chunk)
                })
                .collect();
        }

        indices
            .par_iter()
            .map(|&original_index| {
                let decompressed = self.decompress_chunk(original_index)?;

                let mut chunk = CscGeneChunk::read_from_buffer(&decompressed)?;

                chunk.filter_selected_cells_dense(&cell_map);

                Ok(chunk)
            })
            .collect()
    }

    /// Dimensions and orientation of the store.
    ///
    /// ### Returns
    ///
    /// The [`StoreDims`] read from the header.
    fn dims(&self) -> StoreDims {
        self.dims
    }

    /// Read only library sizes for specified cells
    ///
    /// More efficient than the default, which parses whole chunks. Files
    /// with a library-size index read 8 bytes per cell from it; older files
    /// decompress each chunk and read the header bytes (offset 12-19)
    /// without deserialising the rest.
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices
    ///
    /// ### Returns
    ///
    /// Vector of library sizes
    fn read_cell_library_sizes(&self, indices: &[usize]) -> Result<Vec<usize>, ScDataErrors> {
        if !self.is_cell_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "gene-based",
                requested: "cell-based",
            });
        }

        if let Some(start) = self.library_index {
            return self.library_sizes_from_index(start, indices);
        }

        if self.is_v4() {
            return indices
                .iter()
                .map(|&original_index| {
                    let (chunk, _) = self.locate_chunk(original_index)?;
                    Ok(V4Header::parse(chunk)?.aux as usize)
                })
                .collect();
        }

        indices
            .par_iter()
            .map(|&original_index| {
                let decompressed = self.decompress_chunk(original_index)?;

                Ok(le_u64(&decompressed, 12) as usize)
            })
            .collect()
    }

    /// Read the NNZ for specific genes
    ///
    /// Overrides the default, which parses whole chunks. Reads directly from
    /// the chunk header bytes (offset 16-23) instead.
    ///
    /// ### Params
    ///
    /// * `indices` - Gene indices
    ///
    /// ### Returns
    ///
    /// Vector of number of NNZ genes.
    fn read_gene_nnz(&self, indices: &[usize]) -> Result<Vec<usize>, ScDataErrors> {
        if !self.is_gene_based() {
            return Err(ScDataErrors::ReaderModeMismatch {
                actual: "cell-based",
                requested: "gene-based",
            });
        }

        if self.is_v4() {
            return indices
                .iter()
                .map(|&original_index| {
                    let (chunk, _) = self.locate_chunk(original_index)?;
                    Ok(V4Header::parse(chunk)?.nnz)
                })
                .collect();
        }

        indices
            .par_iter()
            .map(|&original_index| {
                let decompressed = self.decompress_chunk(original_index)?;

                Ok(le_u64(&decompressed, 16) as usize)
            })
            .collect()
    }

    /// Is the reader connected to a cell-based file?
    ///
    /// ### Returns
    ///
    /// Bool
    fn is_cell_based(&self) -> bool {
        self.dims.cell_based
    }

    /// Library size the log-CPM `data_norm` layer of this file is scaled to.
    ///
    /// ### Returns
    ///
    /// `Some(target_size)`, or `None` for raw-only files, files written
    /// before the field entered the header and stores whose norm is not
    /// log-CPM (see [`ParallelSparseReader::norm_scale`] for those).
    fn target_size(&self) -> Option<f32> {
        (self.target_size != 0.0 && self.norm_kind == NormKind::LogCpm).then_some(self.target_size)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::chunk::tests::{cell_chunk, gene_chunk};
    use crate::store::writer::{CellGeneSparseWriter, write_cell_chunks_parallel, write_cell_rows};
    use bincode::serde::encode_to_vec;
    use tempfile::{NamedTempFile, TempPath};

    ///////////////////////
    // Writer and reader //
    ///////////////////////

    /// Writer to reader round trip: orientation, target size and both chunks.
    #[test]
    fn test_cell_based_file_round_trip() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        let chunks = vec![
            cell_chunk(&[5, 70_000], &[0, 2], 0),
            cell_chunk(&[1, 2, 3], &[0, 1, 2], 1),
        ];

        let mut writer = CellGeneSparseWriter::new(temp.to_str().unwrap(), true, 2, 3, 1e4)
            .expect("writer opens");
        for chunk in chunks {
            writer.write_cell_chunk(chunk).expect("write");
        }
        writer.finalise().expect("finalise");

        let reader = ParallelSparseReader::new(temp.to_str().unwrap()).expect("reader opens");
        assert_eq!(reader.target_size(), Some(1e4));
        assert!(reader.is_cell_based());

        let back = reader.read_cells_parallel(&[0, 1]).expect("read");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].data_raw.iter().collect::<Vec<_>>(), vec![5, 70_000]);
        assert_eq!(back[1].indices, vec![0, 1, 2]);
    }

    /// Write a small cell-major store and hand back the guard and reader.
    ///
    /// The chunks deliberately differ in length and include a count above
    /// `u16::MAX`, so the `u32` raw element width is exercised alongside the
    /// row-length bookkeeping.
    ///
    /// ### Params
    ///
    /// * `name` - Test-unique file suffix.
    ///
    /// ### Returns
    ///
    /// The temp guard and an open reader over it.
    fn csr_test_store() -> (TempPath, ParallelSparseReader) {
        csr_test_store_with(false)
    }

    /// [`csr_test_store`] in the legacy v3 format when `legacy` is set.
    ///
    /// ### Params
    ///
    /// * `legacy` - Write v3
    ///
    /// ### Returns
    ///
    /// The path, removed on drop, and an open reader over it.
    fn csr_test_store_with(legacy: bool) -> (TempPath, ParallelSparseReader) {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        let chunks = vec![
            cell_chunk(&[5, 70_000], &[0, 2], 0),
            cell_chunk(&[1, 2, 3], &[0, 1, 2], 1),
            cell_chunk(&[9], &[3], 2),
        ];

        let ctor = if legacy {
            CellGeneSparseWriter::new_v3
        } else {
            CellGeneSparseWriter::new
        };
        let mut writer = ctor(temp.to_str().unwrap(), true, 3, 4, 1e4).expect("writer opens");
        for chunk in chunks {
            writer.write_cell_chunk(chunk).expect("write");
        }
        writer.finalise().expect("finalise");

        let reader = ParallelSparseReader::open_any(temp.to_str().unwrap()).expect("reader opens");
        (temp, reader)
    }

    #[test]
    fn test_read_cells_csr_matches_chunk_path() {
        let (_temp, reader) = csr_test_store();
        let order = [2usize, 0, 1];

        let batch = reader
            .read_cells_csr(&order, None, &DataLayerReturn::BothLayers)
            .expect("csr batch");
        let chunks = reader.read_cells_parallel(&order).expect("chunk batch");

        assert_eq!(batch.shape, (3, 4));
        assert_eq!(batch.indptr.len(), order.len() + 1);
        assert_eq!(batch.indptr[0], 0);
        assert_eq!(
            *batch.indptr.last().expect("non-empty"),
            batch.indices.len() as u32
        );

        let raw = batch.data_raw.as_ref().expect("raw requested");
        let norm = batch.data_norm.as_ref().expect("norm requested");

        for (row, chunk) in chunks.iter().enumerate() {
            let lo = batch.indptr[row] as usize;
            let hi = batch.indptr[row + 1] as usize;

            assert_eq!(&batch.indices[lo..hi], &chunk.indices[..]);
            assert_eq!(
                raw[lo..hi].to_vec(),
                chunk.data_raw.iter().map(|v| v as f32).collect::<Vec<_>>()
            );
            assert_eq!(norm[lo..hi].to_vec(), chunk.data_norm);
        }
    }

    #[test]
    fn test_read_cells_csr_layer_selection() {
        let (_temp, reader) = csr_test_store();

        let raw_only = reader
            .read_cells_csr(&[0, 1, 2], None, &DataLayerReturn::Raw)
            .expect("raw batch");
        assert!(raw_only.data_raw.is_some());
        assert!(raw_only.data_norm.is_none());

        let norm_only = reader
            .read_cells_csr(&[0, 1, 2], None, &DataLayerReturn::Norm)
            .expect("norm batch");
        assert!(norm_only.data_raw.is_none());
        assert!(norm_only.data_norm.is_some());
    }

    #[test]
    fn test_read_cells_csr_repeats_and_subsets() {
        let (_temp, reader) = csr_test_store();

        // A cell may legitimately appear twice in one batch: sampling with
        // replacement is a normal thing for a training loop to ask for.
        let batch = reader
            .read_cells_csr(&[1, 1], None, &DataLayerReturn::Raw)
            .expect("csr batch");

        assert_eq!(batch.shape.0, 2);
        assert_eq!(batch.indptr, vec![0, 3, 6]);
        assert_eq!(batch.indices, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_read_cells_csr_gene_subset() {
        // v5 decodes through push_v4, the legacy store through push_run
        for legacy in [false, true] {
            let (_temp, reader) = csr_test_store_with(legacy);

            // genes 2 then 0: a reversed selection must come back sorted
            let batch = reader
                .read_cells_csr(&[0, 1, 2], Some(&[2, 0]), &DataLayerReturn::BothLayers)
                .expect("subset batch");
            assert_eq!(batch.shape, (3, 2));
            assert_eq!(batch.indptr, vec![0, 2, 4, 4]);
            assert_eq!(batch.indices, vec![0, 1, 0, 1]);
            assert_eq!(batch.data_raw.unwrap(), vec![70_000.0, 5.0, 3.0, 1.0]);

            let full = reader
                .read_cells_csr(&[0, 1, 2], None, &DataLayerReturn::Norm)
                .unwrap()
                .data_norm
                .unwrap();
            // full rows: cell 0 is genes [0, 2], cell 1 genes [0, 1, 2]
            assert_eq!(
                batch.data_norm.unwrap(),
                vec![full[1], full[0], full[4], full[2]]
            );

            for bad in [&[0usize, 0][..], &[4][..]] {
                let err = reader
                    .read_cells_csr(&[0], Some(bad), &DataLayerReturn::Raw)
                    .expect_err("invalid gene selection");
                assert!(matches!(err, ScDataErrors::InvalidArgument(_)));
            }
        }
    }

    #[test]
    fn test_read_cells_csr_rejects_unknown_cell() {
        let (_temp, reader) = csr_test_store();

        let err = reader
            .read_cells_csr(&[99], None, &DataLayerReturn::Raw)
            .expect_err("index is out of range");
        assert!(matches!(err, ScDataErrors::ChunkIndexNotFound(99)));
    }

    #[test]
    fn test_read_cells_csr_rejects_gene_major_store() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        let mut writer = CellGeneSparseWriter::new(temp.to_str().unwrap(), false, 2, 1, 1e4)
            .expect("writer opens");
        writer
            .write_gene_chunk(gene_chunk(&[1, 2], &[0, 1], 0))
            .expect("write");
        writer.finalise().expect("finalise");

        let reader = ParallelSparseReader::new(temp.to_str().unwrap()).expect("reader opens");
        let err = reader
            .read_cells_csr(&[0], None, &DataLayerReturn::Raw)
            .expect_err("store is gene-major");
        assert!(matches!(err, ScDataErrors::ReaderModeMismatch { .. }));
    }

    /// Write a small gene-major store over six cells.
    ///
    /// Gene 0 carries a count above `u16::MAX` (u32 raw width), gene 3 is
    /// empty, and the rest differ in length.
    ///
    /// ### Params
    ///
    /// * `name` - Test-unique file suffix.
    ///
    /// ### Returns
    ///
    /// The temp guard and an open reader over it.
    fn csc_test_store() -> (TempPath, ParallelSparseReader) {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        let chunks = vec![
            gene_chunk(&[5, 70_000, 2], &[0, 3, 5], 0),
            gene_chunk(&[1, 2, 3, 4, 5, 6], &[0, 1, 2, 3, 4, 5], 1),
            gene_chunk(&[9, 11], &[1, 4], 2),
            gene_chunk(&[], &[], 3),
        ];

        let mut writer = CellGeneSparseWriter::new(temp.to_str().unwrap(), false, 6, 4, 1e4)
            .expect("writer opens");
        for chunk in chunks {
            writer.write_gene_chunk(chunk).expect("write");
        }
        writer.finalise().expect("finalise");

        let reader = ParallelSparseReader::new(temp.to_str().unwrap()).expect("reader opens");
        (temp, reader)
    }

    /// Assert two sparse matrices are identical, field by field.
    ///
    /// ### Params
    ///
    /// * `got` - Fused output
    /// * `want` - Chunk-path reference
    fn assert_same_sparse<T: BixverseNumeric + std::fmt::Debug + PartialEq>(
        got: &CompressedSparseData2<T, f32>,
        want: &CompressedSparseData2<T, f32>,
    ) {
        assert_eq!(got.shape, want.shape);
        assert_eq!(got.cs_type.is_csc(), want.cs_type.is_csc());
        assert_eq!(got.indptr, want.indptr);
        assert_eq!(got.indices, want.indices);
        assert_eq!(got.data, want.data);
        assert_eq!(got.data_2, want.data_2);
    }

    /// Fused gene reads equal the chunk path for every layer, with and
    /// without a cell subset, including a non-ascending, an empty and a
    /// repeated-gene case.
    #[test]
    fn test_read_genes_sparse_matches_chunk_path() {
        let (_temp, reader) = csc_test_store();
        let genes = [2usize, 0, 3, 1, 0];

        let subsets: Vec<Option<IndexSet<u32>>> = vec![
            None,
            Some([0_u32, 3, 4, 5].into_iter().collect()),
            Some([5_u32, 1, 3].into_iter().collect()),
            Some(IndexSet::new()),
        ];

        for layer in [
            DataLayerReturn::Raw,
            DataLayerReturn::Norm,
            DataLayerReturn::BothLayers,
        ] {
            for subset in &subsets {
                let (chunks, n_cells) = match subset {
                    Some(set) => (
                        reader
                            .read_gene_parallel_filtered(&genes, set)
                            .expect("chunks"),
                        set.len(),
                    ),
                    None => (reader.read_gene_parallel(&genes).expect("chunks"), 6),
                };
                let want = from_gene_chunks::<u32>(chunks, &layer, n_cells).expect("reference");
                let got = reader
                    .read_genes_sparse::<u32>(&genes, subset.as_ref(), &layer)
                    .expect("fused");
                assert_same_sparse(&got, &want);
            }
        }
    }

    /// The direct CSR read equals the CSC read transposed, for every layer and
    /// cell subset, with one batch and with several.
    #[test]
    fn test_read_genes_sparse_csr_matches_transposed_csc() {
        let (_temp, reader) = csc_test_store();
        let genes = [2usize, 0, 3, 1, 0];

        let subsets: Vec<Option<IndexSet<u32>>> = vec![
            None,
            Some([0_u32, 3, 4, 5].into_iter().collect()),
            Some([5_u32, 1, 3].into_iter().collect()),
            Some(IndexSet::new()),
        ];

        for layer in [
            DataLayerReturn::Raw,
            DataLayerReturn::Norm,
            DataLayerReturn::BothLayers,
        ] {
            for subset in &subsets {
                let want = reader
                    .read_genes_sparse::<u32>(&genes, subset.as_ref(), &layer)
                    .expect("csc")
                    .transform();
                for batch in [1, 2, CSR_GENE_BATCH] {
                    let got = genes_sparse_to_csr::<_, u32>(
                        &reader,
                        &genes,
                        subset.as_ref(),
                        &layer,
                        batch,
                    )
                    .expect("csr");
                    assert_same_sparse(&got, &want);
                }
            }
        }
    }

    /// Fused cell reads equal the chunk path for a non-f32 target.
    #[test]
    fn test_read_cells_sparse_matches_chunk_path() {
        let (_temp, reader) = csr_test_store();
        let order = [2usize, 0, 1, 0];

        for layer in [
            DataLayerReturn::Raw,
            DataLayerReturn::Norm,
            DataLayerReturn::BothLayers,
        ] {
            let chunks = reader.read_cells_parallel(&order).expect("chunks");
            let want = from_cell_chunks::<u32>(&chunks, &layer, 4).expect("reference");
            let got = reader
                .read_cells_sparse::<u32>(&order, &layer)
                .expect("fused");
            assert_same_sparse(&got, &want);
        }
    }

    /// A count above the target type's range is an error, as on the chunk
    /// path, on both the unfiltered (bulk) and filtered paths, also when it
    /// sits past a dropped cell.
    #[test]
    fn test_read_genes_sparse_reports_overflow() {
        let (_temp, reader) = csc_test_store();
        let keep: IndexSet<u32> = [3_u32].into_iter().collect();

        for cells in [None, Some(&keep)] {
            match reader.read_genes_sparse::<u16>(&[0], cells, &DataLayerReturn::Raw) {
                Err(ScDataErrors::RawCountOverflow { value, .. }) => assert_eq!(value, 70_000),
                other => panic!("expected RawCountOverflow, got {:?}", other.map(|_| ())),
            }
        }

        // Dropping the offending cell makes the narrow type fine.
        let keep: IndexSet<u32> = [0_u32, 5].into_iter().collect();
        let csc = reader
            .read_genes_sparse::<u16>(&[0], Some(&keep), &DataLayerReturn::Raw)
            .expect("fits u16");
        assert_eq!(csc.data, vec![5, 2]);
    }

    #[test]
    fn test_read_genes_sparse_rejects_cell_major_store() {
        let (_temp, reader) = csr_test_store();
        let err = reader
            .read_genes_sparse::<f32>(&[0], None, &DataLayerReturn::Raw)
            .expect_err("store is cell-major");
        assert!(matches!(err, ScDataErrors::ReaderModeMismatch { .. }));
    }

    #[test]
    fn test_decompress_chunk_into_matches_allocating_path() {
        let (_temp, reader) = csr_test_store_with(true);

        // A dirty buffer must be overwritten, not appended to: the scratch is
        // reused across every cell a worker owns.
        let mut scratch = vec![0xAAu8; 4096];
        for cell in 0..3 {
            reader
                .decompress_chunk_into(cell, &mut scratch)
                .expect("scratch decompress");
            let owned = reader.decompress_chunk(cell).expect("owned decompress");
            assert_eq!(scratch, owned);
        }
    }

    #[test]
    fn test_identity_index_map_is_detected() {
        let (_temp, reader) = csr_test_store();
        assert!(matches!(reader.lookup, ChunkLookup::Identity));
        assert_eq!(reader.chunk_index_of(2).expect("in range"), 2);
    }

    /// Rewrite the 64-byte header slot of an existing file, applying `edit` to
    /// the decoded [`FileHeader`]. Lets a test corrupt one field of an otherwise
    /// valid file without hand-assembling the whole layout.
    ///
    /// ### Params
    ///
    /// * `path` - The file
    /// * `edit` - Change to the decoded header
    fn patch_file_header(path: &str, edit: impl FnOnce(&mut FileHeader)) {
        let mut bytes = std::fs::read(path).expect("read back");
        let (mut header, _) = decode_from_slice::<FileHeader, _>(&bytes[0..64], config::standard())
            .expect("header decodes");

        edit(&mut header);

        let encoded = encode_to_vec(&header, config::standard()).expect("header encodes");
        let mut slot = [0_u8; 64];
        slot[..encoded.len()].copy_from_slice(&encoded);
        bytes[0..64].copy_from_slice(&slot);

        std::fs::write(path, bytes).expect("write back");
    }

    /// Build a minimal but valid cell-based file at `path`.
    ///
    /// ### Params
    ///
    /// * `path` - Output file
    fn write_valid_file(path: &str) {
        let mut writer = CellGeneSparseWriter::new_v4(path, true, 1, 3, 1e4).expect("writer opens");
        writer
            .write_cell_chunk(cell_chunk(&[1, 2], &[0, 1], 0))
            .expect("write");
        writer.finalise().expect("finalise");
    }

    /// Regression: `ParallelSparseReader::new` rejects any file whose version is
    /// not a supported version, and nothing exercised that guard.
    #[test]
    fn test_reader_rejects_a_version_mismatch() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        write_valid_file(temp.to_str().unwrap());
        patch_file_header(temp.to_str().unwrap(), |h| h.version = SC_FILE_VERSION + 1);

        match ParallelSparseReader::new(temp.to_str().unwrap()) {
            Err(ScDataErrors::FileVersionMismatch { expected, found }) => {
                assert_eq!(expected, SC_FILE_VERSION);
                assert_eq!(found, SC_FILE_VERSION + 1);
            }
            Err(other) => panic!("expected a version mismatch, got {other:?}"),
            Ok(_) => panic!("a mismatched version must not open"),
        }
    }

    /// A file that is not ours at all must be refused rather than parsed.
    #[test]
    fn test_reader_rejects_a_bad_magic() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        write_valid_file(temp.to_str().unwrap());
        patch_file_header(temp.to_str().unwrap(), |h| h.magic = *b"NOTOURS!");

        assert!(
            ParallelSparseReader::new(temp.to_str().unwrap()).is_err(),
            "a wrong magic string must not open"
        );
    }

    /// A header pointing past the end of the file must error rather than read
    /// out of bounds off the mmap.
    #[test]
    fn test_reader_rejects_a_main_header_offset_past_eof() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        write_valid_file(temp.to_str().unwrap());
        patch_file_header(temp.to_str().unwrap(), |h| {
            h.main_header_offset = u64::MAX / 2
        });

        assert!(
            ParallelSparseReader::new(temp.to_str().unwrap()).is_err(),
            "an out-of-range main header offset must not open"
        );
    }

    /// A file truncated mid-payload must error rather than panic.
    #[test]
    fn test_reader_rejects_a_truncated_file() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        write_valid_file(temp.to_str().unwrap());

        let bytes = std::fs::read(temp.to_str().unwrap()).expect("read back");
        std::fs::write(temp.to_str().unwrap(), &bytes[..bytes.len() / 2]).expect("truncate");

        assert!(
            ParallelSparseReader::new(temp.to_str().unwrap()).is_err(),
            "a truncated file must not open"
        );
    }

    /// The gene-based half of the same pin, with one chunk forced to u32 raw.
    #[test]
    fn test_gene_based_file_round_trip() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();

        let mut writer = CellGeneSparseWriter::new(temp.to_str().unwrap(), false, 4, 2, 1e5)
            .expect("writer opens");
        writer
            .write_gene_chunk(gene_chunk(&[9, 8], &[0, 3], 0))
            .expect("write");
        writer
            .write_gene_chunk(gene_chunk(&[100_000], &[2], 1))
            .expect("write");
        writer.finalise().expect("finalise");

        let reader = ParallelSparseReader::new(temp.to_str().unwrap()).expect("reader opens");
        assert_eq!(reader.target_size(), Some(1e5));
        assert!(reader.is_gene_based());

        let back = reader.read_gene_parallel(&[0, 1]).expect("read");
        assert_eq!(back[0].data_raw.iter().collect::<Vec<_>>(), vec![9, 8]);
        assert_eq!(back[1].data_raw.iter().collect::<Vec<_>>(), vec![100_000]);
    }

    /// Regression: the writer used to `assert!` on the orientation.
    #[test]
    fn test_writer_rejects_wrong_chunk_orientation() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        let mut writer = CellGeneSparseWriter::new(temp.to_str().unwrap(), true, 1, 1, 0.0)
            .expect("writer opens");

        match writer.write_gene_chunk(gene_chunk(&[1], &[0], 0)) {
            Err(ScDataErrors::ReaderModeMismatch { actual, requested }) => {
                assert_eq!(actual, "cell-based");
                assert_eq!(requested, "gene-based");
            }
            other => panic!("expected ReaderModeMismatch, got {other:?}"),
        }
    }

    /// A target size of 0 means raw only: reader and peek must both say `None`.
    #[test]
    fn test_raw_only_file_reports_no_target_size() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        let mut writer = CellGeneSparseWriter::new(temp.to_str().unwrap(), true, 1, 2, 0.0)
            .expect("writer opens");
        writer
            .write_cell_chunk(cell_chunk(&[1, 2], &[0, 1], 0))
            .expect("write");
        writer.finalise().expect("finalise");

        let reader = ParallelSparseReader::new(temp.to_str().unwrap()).expect("reader opens");
        assert_eq!(reader.target_size(), None);
        assert_eq!(
            peek_target_size(temp.to_str().unwrap()).expect("peek"),
            None
        );
    }

    ////////////////////////
    // Library-size index //
    ////////////////////////

    /// Read the decoded file header of a written file.
    ///
    /// ### Params
    ///
    /// * `path` - Path of the file.
    ///
    /// ### Returns
    ///
    /// The `FileHeader`.
    fn read_file_header(path: &str) -> FileHeader {
        let bytes = std::fs::read(path).expect("read back");
        decode_from_slice::<FileHeader, _>(&bytes[0..64], config::standard())
            .expect("header decodes")
            .0
    }

    /// Write three cells, out of index order and one with a u32 count,
    /// through the per-chunk or the pre-compressed write path, as v4 or v5.
    ///
    /// ### Params
    ///
    /// * `path` - Output file
    /// * `compressed` - Use the pre-compressed write path
    /// * `v5` - Write v5 rather than v4
    fn write_index_test_file(path: &str, compressed: bool, v5: bool) {
        let chunks = vec![
            cell_chunk(&[5, 70_000], &[0, 2], 2),
            cell_chunk(&[1, 2, 3], &[0, 1, 2], 0),
            cell_chunk(&[9], &[1], 1),
        ];
        let ctor = if v5 {
            CellGeneSparseWriter::new
        } else {
            CellGeneSparseWriter::new_v4
        };
        let mut writer = ctor(path, true, 3, 3, 1e4).expect("writer opens");
        if compressed {
            write_cell_chunks_parallel(&chunks, &mut writer).expect("write");
        } else {
            for chunk in chunks {
                writer.write_cell_chunk(chunk).expect("write");
            }
        }
        writer.finalise().expect("finalise");
    }

    /// Library sizes for `indices` as the reader answers them, and as the
    /// chunk headers store them.
    ///
    /// ### Params
    ///
    /// * `reader` - Reader over the cell file
    /// * `indices` - Cells to look up
    ///
    /// ### Returns
    ///
    /// `(from the reader, from the chunk headers)`.
    fn library_sizes_both_ways(
        reader: &ParallelSparseReader,
        indices: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let answered = reader.read_cell_library_sizes(indices).expect("read");
        let stored = reader
            .read_cells_parallel(indices)
            .expect("read chunks")
            .iter()
            .map(|chunk| chunk.library_size)
            .collect();
        (answered, stored)
    }

    /// Both write paths record an index that matches the chunk headers, in
    /// the order asked.
    #[test]
    fn test_library_index_matches_chunk_headers() {
        for (compressed, v5) in [(false, false), (true, false), (false, true), (true, true)] {
            let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
            write_index_test_file(temp.to_str().unwrap(), compressed, v5);

            let reader = ParallelSparseReader::new(temp.to_str().unwrap()).expect("reader opens");
            assert!(reader.library_index.is_some());
            let (answered, stored) = library_sizes_both_ways(&reader, &[1, 2, 0, 2]);
            assert_eq!(answered, vec![9, 70_005, 6, 70_005]);
            assert_eq!(answered, stored);
        }
    }

    /// v4 files without an index (offset 0) fall back to the chunk headers.
    #[test]
    fn test_library_sizes_fall_back_without_an_index() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        write_index_test_file(temp.to_str().unwrap(), false, false);
        patch_file_header(temp.to_str().unwrap(), |h| h.library_index_offset = 0);

        let reader = ParallelSparseReader::new(temp.to_str().unwrap()).expect("reader opens");
        assert!(reader.library_index.is_none());
        let (answered, _) = library_sizes_both_ways(&reader, &[0, 1, 2]);
        assert_eq!(answered, vec![6, 9, 70_005]);
    }

    /// An index offset past the end of the file must error, not panic.
    #[test]
    fn test_reader_rejects_a_library_index_past_eof() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        write_valid_file(temp.to_str().unwrap());
        let len = std::fs::metadata(temp.to_str().unwrap())
            .expect("stat")
            .len();
        patch_file_header(temp.to_str().unwrap(), |h| h.library_index_offset = len);

        assert!(matches!(
            ParallelSparseReader::new(temp.to_str().unwrap()),
            Err(ScDataErrors::HeaderDecodeFailed)
        ));
    }

    /// An index whose count disagrees with the chunk count must error.
    #[test]
    fn test_reader_rejects_a_library_index_with_the_wrong_count() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        write_valid_file(temp.to_str().unwrap());
        // The main header starts with its own byte length, not the chunk count.
        let main_header_offset = read_file_header(temp.to_str().unwrap()).main_header_offset;
        patch_file_header(temp.to_str().unwrap(), |h| {
            h.library_index_offset = main_header_offset
        });

        assert!(matches!(
            ParallelSparseReader::new(temp.to_str().unwrap()),
            Err(ScDataErrors::HeaderDecodeFailed)
        ));
    }

    /// The ingest path (mtx, h5ad, 10x) writes through `write_cell_rows`:
    /// the index must match the chunk headers and the sizes it reports.
    #[test]
    fn test_library_index_on_the_ingest_write_path() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        let mut rows: Vec<Vec<(u32, u32)>> = (0..300_u32)
            .map(|c| {
                (0..20_u32)
                    .filter(|g| (c * 7 + g * 3) % 5 == 0)
                    .map(|g| (g, 1 + (c + g) % 4 + if c == 7 { 70_000 } else { 0 }))
                    .collect()
            })
            .collect();
        let mut writer = CellGeneSparseWriter::new(temp.to_str().unwrap(), true, 300, 20, 1e4)
            .expect("writer opens");
        let (_, reported) = write_cell_rows(&mut rows, 0, 1e4, &mut writer).expect("write");
        writer.finalise().expect("finalise");

        let reader = ParallelSparseReader::new(temp.to_str().unwrap()).expect("reader opens");
        assert!(reader.library_index.is_some());
        let (answered, stored) = library_sizes_both_ways(&reader, &(0..300).collect::<Vec<_>>());
        assert_eq!(answered, stored);
        assert_eq!(answered, reported);
    }

    /// Gene-based files carry no library-size index.
    #[test]
    fn test_gene_based_file_has_no_library_index() {
        let temp = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        let mut writer = CellGeneSparseWriter::new(temp.to_str().unwrap(), false, 2, 1, 1e4)
            .expect("writer opens");
        writer
            .write_gene_chunk(gene_chunk(&[1, 2], &[0, 1], 0))
            .expect("write");
        writer.finalise().expect("finalise");

        assert_eq!(
            read_file_header(temp.to_str().unwrap()).library_index_offset,
            0
        );
        let reader = ParallelSparseReader::new(temp.to_str().unwrap()).expect("reader opens");
        assert!(reader.library_index.is_none());
    }
}
