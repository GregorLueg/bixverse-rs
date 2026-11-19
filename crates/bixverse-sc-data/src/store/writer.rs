//! Writers for the binary format: the streaming `CellGeneSparseWriter` and
//! the parallel cell write path.

use bincode::{config, serde::encode_to_vec};
use rayon::prelude::*;
use std::{
    fs::File,
    io::{BufWriter, Seek, SeekFrom, Write},
    marker::Sync,
    path::Path,
};

use crate::prelude::*;
use crate::store::chunk::{FileHeader, SparseDataHeader};
use crate::store::v4::{ChunkCodec, ChunkFormat};
use crate::store::v5::{
    FeatureKind, MAX_LAYERS, NormKind, SectionKind, SectionWriter, V5_HEADER_LEN, V5FileHeader,
    cell_id_hash,
};

/// Flush cadence for gene-based writers, in chunks.
///
/// Gene chunks are far fewer and far larger than cell chunks, so a tight
/// cadence keeps peak buffer occupancy bounded without costing many syscalls.
const GENE_FLUSH_FREQUENCY: usize = 1_000;

/// Flush cadence for cell-based writers, in chunks.
///
/// Cell chunks are small, so flushing this rarely effectively leaves the
/// 128 MiB [`BufWriter`] to manage itself. Kept as a backstop only.
const CELL_FLUSH_FREQUENCY: usize = 100_000;

//////////////////////
// Streaming writer //
//////////////////////

/// A compressed cell chunk for
/// [`CellGeneSparseWriter::write_compressed_cell_chunks`]. The library size
/// travels with it so the writer can build the library-size index without
/// decompressing anything.
pub struct CellPayload {
    /// Original index of the cell in the data.
    pub original_index: usize,
    /// Library size of the cell, as stored in the chunk header.
    pub library_size: usize,
    /// Size-prepended lz4 payload.
    pub bytes: Vec<u8>,
}

/// CellGeneSparseWriter
///
/// Implementation of a structure for writing in a streamed manner two different
/// types of sparse stored data.
pub struct CellGeneSparseWriter {
    /// Total cells in the store.
    total_cells: usize,
    /// Total genes (features) in the store.
    total_genes: usize,
    /// Offset of every chunk relative to `chunks_start_pos`, in chunk order.
    chunk_offsets: Vec<u64>,
    /// Original index of every chunk, in chunk order.
    original_indices: Vec<usize>,
    /// Buffered writer to the file.
    writer: BufWriter<File>,
    /// Where the chunks start: the end of the fixed file header.
    chunks_start_pos: u64,
    /// Bytes of chunks written so far. Tracked by hand: `stream_position` on a
    /// `BufWriter` flushes the buffer, once per chunk.
    chunks_len: u64,
    /// Boolean indicating if the writer is designed to write in an efficient
    /// manner for cells.
    cell_based: bool,
    /// The chunks written since the last flush
    chunks_since_flush: usize,
    /// The frequency of flushing, see [`CELL_FLUSH_FREQUENCY`] and
    /// [`GENE_FLUSH_FREQUENCY`].
    flush_frequency: usize,
    /// Library size the `data_norm` layer was normalised against (the TF-IDF
    /// scale for [`NormKind::TfIdf`]). Written into the file header on
    /// finalisation.
    target_size: f32,
    /// Library size of every cell chunk, in chunk order. Written as the
    /// library-size index on finalisation (cell-based files only).
    library_sizes: Vec<u64>,
    /// File format version written into the header.
    version: u32,
    /// Body codec of v4-codec chunks, see [`ChunkFormat`].
    codec: ChunkCodec,
    /// Library size per original cell index, written as the cell table of a
    /// gene-based file on finalisation.
    cell_library_table: Option<Vec<u64>>,
    /// What the features are (v5).
    feature_kind: FeatureKind,
    /// How the norm derives from the raw counts (v5).
    norm_kind: NormKind,
    /// Raw layers per chunk (v5).
    n_layers: usize,
    /// Hash of the cell IDs, all zero when unset (v5).
    cell_id_hash: [u8; 16],
    /// Per-feature IDF for [`NormKind::TfIdf`] (v5).
    idf: Option<Vec<f32>>,
}

impl CellGeneSparseWriter {
    /// Create a new sparse writer instance
    ///
    /// Writes the current file format (v5, lz4 bodies, gene features, log-CPM,
    /// one raw layer). Use [`Self::with_codec`] for another body codec,
    /// [`Self::with_kinds`] and [`Self::with_layers`] for other stores, or
    /// [`Self::new_v4`] and [`Self::new_v3`] for the legacy formats.
    ///
    /// ### Params
    ///
    /// * `path_f` - Path to the .bin file to which to write to.
    /// * `cell_based` - Shall the writer be set up for writing cell-based
    ///   (`true`) or gene-based chunks.
    /// * `total_cells` - Total cells in the data.
    /// * `total_genes` - Total genes in the data.
    /// * `target_size` - Library size the `data_norm` layer was normalised
    ///   against. Pass `0.0` on raw-only paths where no normalisation applies;
    ///   readers report that as `None`.
    ///
    /// ### Returns
    ///
    /// The `CellGeneSparseWriter`.
    pub fn new<P: AsRef<Path>>(
        path_f: P,
        cell_based: bool,
        total_cells: usize,
        total_genes: usize,
        target_size: f32,
    ) -> Result<Self, ScDataErrors> {
        Self::create(
            path_f,
            cell_based,
            total_cells,
            total_genes,
            target_size,
            SC_FILE_VERSION,
        )
    }

    /// Create a writer for the v4 format.
    ///
    /// Only needed to produce v4 files on purpose, such as the input of a
    /// conversion test. Same parameters as [`Self::new`].
    ///
    /// ### Params
    ///
    /// * `path_f` - Path to the .bin file to which to write to.
    /// * `cell_based` - Cell-based (`true`) or gene-based chunks.
    /// * `total_cells` - Total cells in the data.
    /// * `total_genes` - Total genes in the data.
    /// * `target_size` - Library size the normalised layer is scaled to.
    ///
    /// ### Returns
    ///
    /// The `CellGeneSparseWriter`.
    pub fn new_v4<P: AsRef<Path>>(
        path_f: P,
        cell_based: bool,
        total_cells: usize,
        total_genes: usize,
        target_size: f32,
    ) -> Result<Self, ScDataErrors> {
        Self::create(
            path_f,
            cell_based,
            total_cells,
            total_genes,
            target_size,
            SC_FILE_VERSION_V4,
        )
    }

    /// Create a writer for the legacy v3 format.
    ///
    /// Only needed to produce v3 files on purpose, such as the input of a
    /// conversion test. Same parameters as [`Self::new`].
    ///
    /// ### Params
    ///
    /// * `path_f` - Path to the .bin file to which to write to.
    /// * `cell_based` - Cell-based (`true`) or gene-based chunks.
    /// * `total_cells` - Total cells in the data.
    /// * `total_genes` - Total genes in the data.
    /// * `target_size` - Library size the normalised layer is scaled to.
    ///
    /// ### Returns
    ///
    /// The `CellGeneSparseWriter`.
    pub fn new_v3<P: AsRef<Path>>(
        path_f: P,
        cell_based: bool,
        total_cells: usize,
        total_genes: usize,
        target_size: f32,
    ) -> Result<Self, ScDataErrors> {
        Self::create(
            path_f,
            cell_based,
            total_cells,
            total_genes,
            target_size,
            SC_FILE_VERSION_V3,
        )
    }

    /// Open the file and reserve the fixed header of `version`.
    ///
    /// The header is a placeholder until [`Self::finalise`] rewrites it, so a
    /// file that is never finalised does not open.
    ///
    /// ### Params
    ///
    /// * `path_f` - Path to the .bin file to which to write to.
    /// * `cell_based` - Cell-based (`true`) or gene-based chunks.
    /// * `total_cells` - Total cells in the data.
    /// * `total_genes` - Total genes in the data.
    /// * `target_size` - Library size the normalised layer is scaled to.
    /// * `version` - File format version.
    ///
    /// ### Returns
    ///
    /// The `CellGeneSparseWriter`.
    fn create<P: AsRef<Path>>(
        path_f: P,
        cell_based: bool,
        total_cells: usize,
        total_genes: usize,
        target_size: f32,
        version: u32,
    ) -> Result<Self, ScDataErrors> {
        let file = File::create(path_f)?;
        let mut writer = BufWriter::with_capacity(128 * 1024 * 1024, file);

        let header_len = if version >= SC_FILE_VERSION {
            V5_HEADER_LEN
        } else {
            64
        };
        writer.write_all(&vec![0u8; header_len])?;
        writer.flush()?;

        let flush_frequency = if cell_based {
            CELL_FLUSH_FREQUENCY
        } else {
            GENE_FLUSH_FREQUENCY
        };

        Ok(Self {
            total_cells,
            total_genes,
            chunk_offsets: Vec::new(),
            original_indices: Vec::new(),
            writer,
            chunks_start_pos: header_len as u64,
            chunks_len: 0,
            cell_based,
            chunks_since_flush: 0_usize,
            flush_frequency,
            target_size,
            library_sizes: Vec::new(),
            version,
            codec: ChunkCodec::Lz4,
            cell_library_table: None,
            feature_kind: FeatureKind::Gene,
            norm_kind: NormKind::LogCpm,
            n_layers: 1,
            cell_id_hash: [0; 16],
            idf: None,
        })
    }

    /// Set what the features are and how the norm derives (v5).
    ///
    /// ### Params
    ///
    /// * `feature_kind` - What the features are
    /// * `norm_kind` - How the norm derives from the raw counts. TF-IDF needs
    ///   its IDF ([`crate::store::v5::set_tf_idf`]) before the norm can be
    ///   read.
    ///
    /// ### Returns
    ///
    /// The writer, or an error unless it writes v5.
    pub fn with_kinds(
        mut self,
        feature_kind: FeatureKind,
        norm_kind: NormKind,
    ) -> Result<Self, ScDataErrors> {
        self.require_v5()?;
        self.feature_kind = feature_kind;
        self.norm_kind = norm_kind;
        Ok(self)
    }

    /// Set the number of raw layers per chunk (v5).
    ///
    /// A store with more than one layer is written with
    /// [`Self::write_cell_layers`]; [`crate::store::gene_file::write_gene_file`]
    /// carries every layer into the gene file.
    ///
    /// ### Params
    ///
    /// * `n_layers` - Layers, 1 to [`MAX_LAYERS`]
    ///
    /// ### Returns
    ///
    /// The writer, or an error unless it writes v5 or for a count out of
    /// range.
    pub fn with_layers(mut self, n_layers: usize) -> Result<Self, ScDataErrors> {
        self.require_v5()?;
        if !(1..=MAX_LAYERS).contains(&n_layers) {
            return Err(ScDataErrors::LayerOutOfRange {
                requested: n_layers,
                available: MAX_LAYERS,
            });
        }
        self.n_layers = n_layers;
        Ok(self)
    }

    /// Record the cell IDs the store is aligned to (v5).
    ///
    /// Only their hash is stored, see [`cell_id_hash`].
    ///
    /// ### Params
    ///
    /// * `ids` - Cell IDs in store order
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or an error unless the writer writes v5.
    pub fn set_cell_ids<S: AsRef<str>>(&mut self, ids: &[S]) -> Result<(), ScDataErrors> {
        self.require_v5()?;
        self.cell_id_hash = cell_id_hash(ids);
        Ok(())
    }

    /// Copy the store-level fields of another v5 store: kinds, layer count,
    /// cell-ID hash and IDF.
    ///
    /// For the other orientation of the same store, and for conversions that
    /// keep the store's identity.
    ///
    /// ### Params
    ///
    /// * `reader` - The store to copy from
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or an error unless the writer writes v5.
    pub(crate) fn inherit_from(
        &mut self,
        reader: &ParallelSparseReader,
    ) -> Result<(), ScDataErrors> {
        self.require_v5()?;
        self.feature_kind = reader.feature_kind();
        self.norm_kind = reader.norm_kind();
        self.n_layers = reader.n_layers();
        self.cell_id_hash = reader.cell_id_hash().unwrap_or([0; 16]);
        self.idf = reader.idf().map(<[f32]>::to_vec);
        Ok(())
    }

    /// Error unless this writer writes v5.
    ///
    /// ### Returns
    ///
    /// `Ok(())` for v5, [`ScDataErrors::FileVersionMismatch`] otherwise.
    fn require_v5(&self) -> Result<(), ScDataErrors> {
        if self.version == SC_FILE_VERSION {
            Ok(())
        } else {
            Err(ScDataErrors::FileVersionMismatch {
                expected: SC_FILE_VERSION,
                found: self.version,
            })
        }
    }

    /// Set the body codec of the v4 chunks this writer encodes.
    ///
    /// Chunks that do not shrink are stored uncompressed whatever the codec.
    ///
    /// ### Params
    ///
    /// * `codec` - The codec
    ///
    /// ### Returns
    ///
    /// The writer, with the codec set.
    pub fn with_codec(mut self, codec: ChunkCodec) -> Self {
        self.codec = codec;
        self
    }

    /// How this writer encodes chunks.
    ///
    /// Parallel payload builders copy this out and encode with it, so what
    /// they hand to the `write_compressed_*` methods matches the file.
    ///
    /// ### Returns
    ///
    /// The format: file version, body codec and the header target size.
    pub fn format(&self) -> ChunkFormat {
        ChunkFormat {
            version: self.version,
            codec: self.codec,
            target_size: self.target_size,
        }
    }

    /// Set the per-cell library-size table of a v4 or v5 gene-based file.
    ///
    /// Gene chunks that drop their norm layer recompute it from the library
    /// size of each cell, so the table has to be set before
    /// [`Self::finalise`]. The chunks must have been encoded against the same
    /// table, see [`ChunkFormat::encode_gene`].
    ///
    /// ### Params
    ///
    /// * `table` - Library size per original cell index
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or an error unless this is a v4 or v5 gene-based writer.
    pub fn set_cell_library_table(&mut self, table: Vec<u64>) -> Result<(), ScDataErrors> {
        if !self.format().uses_v4_codec() {
            return Err(ScDataErrors::FileVersionMismatch {
                expected: SC_FILE_VERSION,
                found: self.version,
            });
        }
        self.check_mode(false)?;
        self.cell_library_table = Some(table);
        Ok(())
    }

    /// Verify the writer was opened in the orientation the caller expects.
    ///
    /// ### Params
    ///
    /// * `want_cell_based` - `true` when a cell (CSR) chunk is being written.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::ReaderModeMismatch`] on a mismatch.
    fn check_mode(&self, want_cell_based: bool) -> Result<(), ScDataErrors> {
        if self.cell_based == want_cell_based {
            return Ok(());
        }
        let label = |b: bool| if b { "cell-based" } else { "gene-based" };
        Err(ScDataErrors::ReaderModeMismatch {
            actual: label(self.cell_based),
            requested: label(want_cell_based),
        })
    }

    /// Append an already compressed chunk and record its offset.
    ///
    /// Flushes the underlying [`BufWriter`] every `flush_frequency` chunks.
    ///
    /// ### Params
    ///
    /// * `compressed` - The encoded chunk, in the format of this writer.
    /// * `original_index` - Original index of the cell or gene in the data.
    ///
    /// ### Returns
    ///
    /// `Ok(())` on success, an I/O error otherwise.
    fn append_compressed(
        &mut self,
        compressed: &[u8],
        original_index: usize,
    ) -> Result<(), ScDataErrors> {
        self.chunk_offsets.push(self.chunks_len);
        self.original_indices.push(original_index);

        self.writer
            .write_all(&(compressed.len() as u64).to_le_bytes())?;
        self.writer.write_all(compressed)?;
        self.chunks_len += 8 + compressed.len() as u64;

        self.chunks_since_flush += 1;

        if self.chunks_since_flush >= self.flush_frequency {
            self.writer.flush()?;
            self.chunks_since_flush = 0;
        }

        Ok(())
    }

    /// Append a compressed cell chunk and record its library size.
    ///
    /// ### Params
    ///
    /// * `compressed` - Size-prepended lz4 payload of a serialised cell chunk.
    /// * `original_index` - Original index of the cell in the data.
    /// * `library_size` - Library size stored in the chunk header.
    ///
    /// ### Returns
    ///
    /// `Ok(())` on success, an I/O error otherwise.
    fn append_cell(
        &mut self,
        compressed: &[u8],
        original_index: usize,
        library_size: usize,
    ) -> Result<(), ScDataErrors> {
        self.append_compressed(compressed, original_index)?;
        self.library_sizes.push(library_size as u64);
        Ok(())
    }

    /// Write a Cell (Chunk) to the file
    ///
    /// The data is represented in a CSR-type format.
    ///
    /// ### Params
    ///
    /// * `cell_chunk` - The data representing that specific cell.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::ReaderModeMismatch`] if the writer was
    /// opened for gene-based chunks.
    pub fn write_cell_chunk(&mut self, cell_chunk: CsrCellChunk) -> Result<(), ScDataErrors> {
        self.check_mode(true)?;
        self.check_single_layer()?;

        let bytes = self.format().encode_cell(&cell_chunk)?;
        self.append_cell(&bytes, cell_chunk.original_index, cell_chunk.library_size)
    }

    /// Write a cell with all its raw layers (v5).
    ///
    /// The norm is derived on read, never stored, so the store's norm kind
    /// decides what a norm read returns. The library size is the sum of
    /// layer 0.
    ///
    /// ### Params
    ///
    /// * `original_index` - Index of the cell
    /// * `indices` - Gene indices, ascending
    /// * `layers` - The store's raw layers, each as long as `indices`
    /// * `to_keep` - Keep flag
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or an error for a gene-based writer or a layer count other
    /// than the store's.
    pub fn write_cell_layers(
        &mut self,
        original_index: usize,
        indices: &[u32],
        layers: &[&[u32]],
        to_keep: bool,
    ) -> Result<(), ScDataErrors> {
        self.check_mode(true)?;
        self.check_layer_count(layers.len())?;
        let bytes = self
            .format()
            .encode_layers(indices, layers, original_index, to_keep)?;
        let library_size = layers[0].iter().map(|&x| x as usize).sum();
        self.append_cell(&bytes, original_index, library_size)
    }

    /// Error unless the store has one raw layer, which the chunk-struct
    /// writers assume.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::LayerOutOfRange`].
    fn check_single_layer(&self) -> Result<(), ScDataErrors> {
        self.check_layer_count(1)
    }

    /// Error unless `n` is the store's layer count.
    ///
    /// ### Params
    ///
    /// * `n` - Layers handed in
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::LayerOutOfRange`].
    fn check_layer_count(&self, n: usize) -> Result<(), ScDataErrors> {
        if n == self.n_layers {
            Ok(())
        } else {
            Err(ScDataErrors::LayerOutOfRange {
                requested: n,
                available: self.n_layers,
            })
        }
    }

    /// Write a Gene to the file
    ///
    /// A v4 gene chunk written this way keeps its normalised layer verbatim,
    /// since there is no per-cell library table to recompute it from. The bulk
    /// path, `write_gene_file`, drops it.
    ///
    /// ### Params
    ///
    /// * `gene_chunk` - The data representing that specific gene.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::ReaderModeMismatch`] if the writer was
    /// opened for cell-based chunks.
    pub fn write_gene_chunk(&mut self, gene_chunk: CscGeneChunk) -> Result<(), ScDataErrors> {
        self.check_mode(false)?;
        self.check_single_layer()?;

        let bytes = self.format().encode_gene(&gene_chunk, &[])?;
        self.append_compressed(&bytes, gene_chunk.original_index)
    }

    /// Write genes whose payloads were already built via
    /// [`ChunkFormat::encode_gene`] of this writer's [`Self::format`].
    ///
    /// ### Params
    ///
    /// * `chunks` - `(original_index, payload)` pairs, appended in the given
    ///   order.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::ReaderModeMismatch`] if the writer was
    /// opened for cell-based chunks.
    pub fn write_compressed_gene_chunks(
        &mut self,
        chunks: &[(usize, Vec<u8>)],
    ) -> Result<(), ScDataErrors> {
        self.check_mode(false)?;
        self.append_compressed_batch(chunks)
    }

    /// Write cells whose payloads were already built via
    /// [`ChunkFormat::encode_cell`] of this writer's [`Self::format`].
    ///
    /// ### Params
    ///
    /// * `chunks` - Cell payloads, appended in the given order.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::ReaderModeMismatch`] if the writer was
    /// opened for gene-based chunks.
    pub fn write_compressed_cell_chunks(
        &mut self,
        chunks: &[CellPayload],
    ) -> Result<(), ScDataErrors> {
        self.check_mode(true)?;
        for chunk in chunks {
            self.append_cell(&chunk.bytes, chunk.original_index, chunk.library_size)?;
        }
        Ok(())
    }

    /// Append a batch of compressed payloads in order.
    ///
    /// ### Params
    ///
    /// * `chunks` - `(original_index, payload)` pairs.
    ///
    /// ### Returns
    ///
    /// `Ok(())` on success, an I/O error otherwise.
    fn append_compressed_batch(&mut self, chunks: &[(usize, Vec<u8>)]) -> Result<(), ScDataErrors> {
        for (original_index, payload) in chunks {
            self.append_compressed(payload, *original_index)?;
        }
        Ok(())
    }

    /// Finalise the file
    ///
    /// v5: writes the tail sections and the section directory, then the
    /// 128-byte header. v3 / v4: writes the [`SparseDataHeader`] tail and the
    /// library-size index or table, then the 64-byte header.
    ///
    /// ### Returns
    ///
    /// `Ok(())` on success, [`ScDataErrors::HeaderEncodeFailed`] if either
    /// header could not be encoded.
    pub fn finalise(mut self) -> Result<(), ScDataErrors> {
        if self.version >= SC_FILE_VERSION {
            return self.finalise_v5();
        }

        let main_header_offset = self.writer.stream_position()?;

        let no_chunks = self.chunk_offsets.len();
        let header = SparseDataHeader {
            total_cells: self.total_cells,
            total_genes: self.total_genes,
            cell_based: self.cell_based,
            no_chunks,
            index_map: self
                .original_indices
                .iter()
                .enumerate()
                .map(|(chunk, &orig)| (orig, chunk))
                .collect(),
            chunk_offsets: std::mem::take(&mut self.chunk_offsets),
        };
        let header_data = encode_to_vec(&header, config::standard())
            .map_err(|_| ScDataErrors::HeaderEncodeFailed)?;
        let header_size = header_data.len() as u64;

        self.writer.write_all(&header_size.to_le_bytes())?;
        self.writer.write_all(&header_data)?;

        let mut file_header = FileHeader::new(self.cell_based, self.target_size);
        file_header.version = self.version;
        file_header.main_header_offset = main_header_offset;
        if self.cell_based {
            file_header.library_index_offset = self.write_library_index()?;
        } else if let Some(table) = self.cell_library_table.take() {
            file_header.library_index_offset = self.write_library_table(&table)?;
        }

        self.writer.seek(SeekFrom::Start(0))?;
        let file_header_enc = encode_to_vec(&file_header, config::standard())
            .map_err(|_| ScDataErrors::HeaderEncodeFailed)?;

        // ensure it's exactly 64 bytes
        if file_header_enc.len() < 64 {
            self.writer.write_all(&file_header_enc)?;
            self.writer
                .write_all(&vec![0u8; 64 - file_header_enc.len()])?;
        } else {
            self.writer.write_all(&file_header_enc[..64])?;
        }

        self.writer.flush()?;

        Ok(())
    }

    /// Finalise a v5 file: sections, directory, then the header.
    ///
    /// The original-index section is only written when chunk `i` does not
    /// hold original index `i`.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or an I/O error.
    fn finalise_v5(mut self) -> Result<(), ScDataErrors> {
        let n_chunks = self.chunk_offsets.len();
        let chunks_end = self.writer.stream_position()? - self.chunks_start_pos;

        let mut sections = SectionWriter::default();
        sections.push_u64(
            SectionKind::ChunkOffsets,
            self.chunk_offsets
                .iter()
                .copied()
                .chain(std::iter::once(chunks_end)),
        );
        let identity = self
            .original_indices
            .iter()
            .enumerate()
            .all(|(chunk, &orig)| chunk == orig);
        if !identity {
            let mut pairs: Vec<(u64, u64)> = self
                .original_indices
                .iter()
                .enumerate()
                .map(|(chunk, &orig)| (orig as u64, chunk as u64))
                .collect();
            pairs.sort_unstable();
            sections.push_u64(
                SectionKind::OriginalIndex,
                pairs.into_iter().flat_map(|(o, c)| [o, c]),
            );
        }
        if self.cell_based {
            assert_eq!(
                self.library_sizes.len(),
                n_chunks,
                "library-size index has {} entries for {} cell chunks",
                self.library_sizes.len(),
                n_chunks
            );
            sections.push_u64(
                SectionKind::LibraryIndex,
                self.library_sizes.iter().copied(),
            );
        } else if let Some(table) = self.cell_library_table.take() {
            sections.push_u64(SectionKind::CellLibraryTable, table);
        }
        if let Some(idf) = self.idf.take() {
            sections.push_f32(SectionKind::Idf, &idf);
        }

        let (dir_offset, n_sections) = sections.finish(&mut self.writer, &[])?;

        let header = V5FileHeader {
            cell_based: self.cell_based,
            feature_kind: self.feature_kind,
            norm_kind: self.norm_kind,
            n_layers: self.n_layers,
            target_size: self.target_size,
            n_cells: self.total_cells,
            n_features: self.total_genes,
            n_chunks,
            dir_offset,
            n_sections,
            cell_id_hash: self.cell_id_hash,
        };
        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(&header.encode())?;
        self.writer.flush()?;
        Ok(())
    }

    /// Write the library-size index at the current position: a `u64` count,
    /// then one `u64` per cell chunk in chunk order.
    ///
    /// ### Returns
    ///
    /// The offset the index starts at, for the [`FileHeader`].
    fn write_library_index(&mut self) -> Result<u64, ScDataErrors> {
        assert!(self.cell_based, "library-size index on a gene-based file");
        assert_eq!(
            self.library_sizes.len(),
            self.original_indices.len(),
            "library-size index has {} entries for {} cell chunks",
            self.library_sizes.len(),
            self.original_indices.len()
        );
        let offset = self.writer.stream_position()?;
        self.writer
            .write_all(&(self.library_sizes.len() as u64).to_le_bytes())?;
        for library_size in &self.library_sizes {
            self.writer.write_all(&library_size.to_le_bytes())?;
        }
        Ok(offset)
    }

    /// Write the per-cell library-size table of a v4 gene file at the current
    /// position: a `u64` count, then one `u64` per original cell index.
    ///
    /// ### Params
    ///
    /// * `table` - Library size per original cell index
    ///
    /// ### Returns
    ///
    /// The offset the table starts at, for the [`FileHeader`].
    fn write_library_table(&mut self, table: &[u64]) -> Result<u64, ScDataErrors> {
        let offset = self.writer.stream_position()?;
        self.writer.write_all(&(table.len() as u64).to_le_bytes())?;
        for library_size in table {
            self.writer.write_all(&library_size.to_le_bytes())?;
        }
        Ok(offset)
    }

    /// Update the number of cells in the header
    ///
    /// Helper function to update the number of cells to the number that was
    /// finally written on disk.
    ///
    /// ### Params
    ///
    /// * `no_cells` - New number of cells
    pub fn update_header_no_cells(&mut self, no_cells: usize) {
        self.total_cells = no_cells;
    }

    /// Update the number of cells in the header
    ///
    /// Helper function to update the number of genes to the number that was
    /// finally written on disk.
    ///
    /// ### Params
    ///
    /// * `no_genes` - New number of genes
    pub fn update_header_no_genes(&mut self, no_genes: usize) {
        self.total_genes = no_genes;
    }
}

/////////////////////////
// Parallel cell write //
/////////////////////////

/// Cells built and compressed per parallel batch in [`write_cell_rows`].
///
/// Large enough to amortise the rayon fan-out, small enough that the
/// compressed payloads held before the append stay a few MB.
const CELL_WRITE_BATCH: usize = 4_096;

/// Build one cell chunk from `(gene, count)` pairs and compress it.
///
/// ### Params
///
/// * `row` - The cell's `(gene, count)` pairs, sorted by gene here if needed
/// * `original_index` - Index the cell is stored under
/// * `target_size` - Target size for the library normalisation
/// * `format` - The format of the writer the payload goes to
///
/// ### Returns
///
/// `(nnz, payload)`; the payload carries the library size.
pub fn compress_cell_row(
    row: &mut [(u32, u32)],
    original_index: usize,
    target_size: f32,
    format: ChunkFormat,
) -> Result<(usize, CellPayload), ScDataErrors> {
    if row.windows(2).any(|w| w[0].0 > w[1].0) {
        row.sort_by_key(|&(g, _)| g);
    }
    let genes: Vec<u32> = row.iter().map(|&(g, _)| g).collect();
    let counts: Vec<u32> = row.iter().map(|&(_, c)| c).collect();
    // a norm scaled to another target than the file's does not recompute
    let fast = (target_size.to_bits() == format.target_size.to_bits())
        .then(|| format.encode_cell_counts(&genes, &counts, original_index))
        .flatten();
    if let Some((library_size, bytes)) = fast {
        let payload = CellPayload {
            original_index,
            library_size,
            bytes,
        };
        return Ok((row.len(), payload));
    }
    let chunk = CsrCellChunk::from_data(&counts, &genes, original_index, target_size, true);
    let (nnz, library_size) = chunk.get_qc_info();
    let payload = CellPayload {
        original_index,
        library_size,
        bytes: format.encode_cell(&chunk)?,
    };
    Ok((nnz, payload))
}

/// Write accumulated per-cell rows to the cell-based binary.
///
/// For the paths that have to gather a cell's values from a gene-major or
/// unsorted source (CSC h5ad, mtx) before writing. Rows are sorted, built and
/// compressed in parallel batches of [`CELL_WRITE_BATCH`], appended in order,
/// and freed as they go.
///
/// ### Params
///
/// * `rows` - One `(gene, count)` row per cell, in output order
/// * `cell_offset` - Index of the first row's cell in the output
/// * `target_size` - Target size for the library normalisation
/// * `writer` - Cell-based writer to append to
///
/// ### Returns
///
/// `(nnz, lib_size)` per written cell.
pub fn write_cell_rows(
    rows: &mut [Vec<(u32, u32)>],
    cell_offset: usize,
    target_size: f32,
    writer: &mut CellGeneSparseWriter,
) -> Result<(Vec<usize>, Vec<usize>), ScDataErrors> {
    let mut nnz = Vec::with_capacity(rows.len());
    let mut lib_size = Vec::with_capacity(rows.len());
    let format = writer.format();

    for (batch_idx, batch) in rows.chunks_mut(CELL_WRITE_BATCH).enumerate() {
        let first = cell_offset + batch_idx * CELL_WRITE_BATCH;
        let built = batch
            .par_iter_mut()
            .enumerate()
            .map(|(k, row)| {
                let mut row = std::mem::take(row);
                compress_cell_row(&mut row, first + k, target_size, format)
            })
            .collect::<Result<Vec<_>, ScDataErrors>>()?;

        let mut payloads = Vec::with_capacity(built.len());
        for (nnz_i, payload) in built {
            nnz.push(nnz_i);
            lib_size.push(payload.library_size);
            payloads.push(payload);
        }
        writer.write_compressed_cell_chunks(&payloads)?;
    }

    Ok((nnz, lib_size))
}

/// [`write_cell_rows`] over one flat buffer of rows.
///
/// ### Params
///
/// * `entries` - `(gene, count)` pairs of all rows back to back
/// * `offsets` - Row `k` is `entries[offsets[k]..offsets[k + 1]]`
/// * `cell_offset` - Index of the first row's cell in the output
/// * `target_size` - Target size for the library normalisation
/// * `writer` - Cell-based writer to append to
///
/// ### Returns
///
/// `(nnz, lib_size)` per written cell.
pub fn write_cell_rows_flat(
    entries: &mut [(u32, u32)],
    offsets: &[usize],
    cell_offset: usize,
    target_size: f32,
    writer: &mut CellGeneSparseWriter,
) -> Result<(Vec<usize>, Vec<usize>), ScDataErrors> {
    let n_rows = offsets.len() - 1;
    let mut nnz = Vec::with_capacity(n_rows);
    let mut lib_size = Vec::with_capacity(n_rows);
    let format = writer.format();

    let mut rest = &mut entries[..offsets[n_rows]];
    for start in (0..n_rows).step_by(CELL_WRITE_BATCH) {
        let end = (start + CELL_WRITE_BATCH).min(n_rows);
        let mut rows = Vec::with_capacity(end - start);
        for k in start..end {
            let (row, tail) = std::mem::take(&mut rest).split_at_mut(offsets[k + 1] - offsets[k]);
            rows.push(row);
            rest = tail;
        }
        let built = rows
            .into_par_iter()
            .enumerate()
            .map(|(k, row)| compress_cell_row(row, cell_offset + start + k, target_size, format))
            .collect::<Result<Vec<_>, ScDataErrors>>()?;

        let mut payloads = Vec::with_capacity(built.len());
        for (nnz_i, payload) in built {
            nnz.push(nnz_i);
            lib_size.push(payload.library_size);
            payloads.push(payload);
        }
        writer.write_compressed_cell_chunks(&payloads)?;
    }

    Ok((nnz, lib_size))
}

/// Build, compress and append cell chunks in parallel batches.
///
/// For writers whose chunks come from a closure, so that neither the build nor
/// the lz4 compression runs on the single writer thread. Batches of
/// [`CELL_WRITE_BATCH`] are compressed in parallel and appended in order.
///
/// ### Params
///
/// * `n_chunks` - Number of chunks to write
/// * `build` - Builds chunk `k` in `0..n_chunks`, its `original_index` set
/// * `writer` - Cell-based writer to append to
///
/// ### Returns
///
/// `(nnz, lib_size)` per written cell.
pub fn write_cell_chunks_built<F>(
    n_chunks: usize,
    build: F,
    writer: &mut CellGeneSparseWriter,
) -> Result<(Vec<usize>, Vec<usize>), ScDataErrors>
where
    F: Fn(usize) -> CsrCellChunk + Sync,
{
    let mut nnz = Vec::with_capacity(n_chunks);
    let mut lib_size = Vec::with_capacity(n_chunks);
    let format = writer.format();

    for start in (0..n_chunks).step_by(CELL_WRITE_BATCH) {
        let end = (start + CELL_WRITE_BATCH).min(n_chunks);
        let built = (start..end)
            .into_par_iter()
            .map(|k| {
                let chunk = build(k);
                let (nnz_i, lib_i) = chunk.get_qc_info();
                Ok::<_, ScDataErrors>((
                    nnz_i,
                    CellPayload {
                        original_index: chunk.original_index,
                        library_size: lib_i,
                        bytes: format.encode_cell(&chunk)?,
                    },
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut payloads = Vec::with_capacity(built.len());
        for (nnz_i, payload) in built {
            nnz.push(nnz_i);
            lib_size.push(payload.library_size);
            payloads.push(payload);
        }
        writer.write_compressed_cell_chunks(&payloads)?;
    }

    Ok((nnz, lib_size))
}

/// Compress and append already built cell chunks in parallel batches.
///
/// ### Params
///
/// * `chunks` - Chunks to write, in output order
/// * `writer` - Cell-based writer to append to
///
/// ### Returns
///
/// `Ok(())`, or the first compression or I/O error.
pub fn write_cell_chunks_parallel(
    chunks: &[CsrCellChunk],
    writer: &mut CellGeneSparseWriter,
) -> Result<(), ScDataErrors> {
    let format = writer.format();
    for batch in chunks.chunks(CELL_WRITE_BATCH) {
        let payloads = batch
            .par_iter()
            .map(|chunk| {
                Ok::<_, ScDataErrors>(CellPayload {
                    original_index: chunk.original_index,
                    library_size: chunk.library_size,
                    bytes: format.encode_cell(chunk)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        writer.write_compressed_cell_chunks(&payloads)?;
    }
    Ok(())
}
