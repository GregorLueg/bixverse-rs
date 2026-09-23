//! An in-memory [`SingleCellReading`] store over [`CompressedSparseData2`].
//!
//! Everything in `sc_analysis` is written against the streaming reader, which
//! assumes the counts live in the bixverse binary format on disk. Metacells do
//! not: they compress a whole experiment down to something that fits in memory
//! comfortably, and the R side hands them over as a sparse matrix. This adapter
//! presents such a matrix through the same trait so those methods can be reused
//! rather than reimplemented.
//!
//! Gene-major by default. Nothing that consumes it needs cell chunks, and
//! building a CSR twin purely to serve them would double the memory for no
//! gain, so [`SingleCellReading::read_cells_parallel`] refuses and the one
//! cell-side quantity that is actually needed, the library sizes, is summed
//! once up front.
//!
//! [`InMemorySparseReader::new_cell_major`] is the exception, for the callers
//! that genuinely need cell chunks: the analytic Pearson fit sums each cell
//! over its group's retained genes, which no gene-major sweep can answer. It
//! owns a real CSR copy of the matrix, so it costs the doubled memory the
//! default avoids. Worth it for metacells, which is what asks for it; do not
//! reach for it on a raw-cell matrix.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::math::sparse::transpose_sparse;
use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{
    CscGeneChunk, CsrCellChunk, RawCounts, SingleCellReading, SparseDataHeader,
};
use crate::single_cell::sc_traits::F16;

//////////////////////////
// InMemorySparseReader //
//////////////////////////

/// Gene-major in-memory store behind the [`SingleCellReading`] interface.
///
/// Reads are slices of the underlying buffers copied into the chunk types the
/// trait hands back, so a full gene sweep costs one copy of the matrix rather
/// than a decompression per chunk.
pub struct InMemorySparseReader<'a> {
    /// The counts, CSC with shape (cells, genes). `data` holds raw counts and
    /// `data_2` the normalised layer.
    ///
    /// In cell-major mode this still points at the caller's CSC matrix and the
    /// reads come off `transposed` instead. Keeping it lets both modes share
    /// the shape and the library-size sweep.
    matrix: &'a CompressedSparseData2<u32, f32>,
    /// The CSR twin, present only in cell-major mode.
    ///
    /// Owned, because it is a real scatter of `matrix` rather than a view: a
    /// CSC relabelled as CSR describes the transpose, which is the wrong axis
    /// here.
    transposed: Option<CompressedSparseData2<u32, f32>>,
    /// Total raw counts per cell, summed once in [`InMemorySparseReader::new`].
    library_sizes: Vec<usize>,
    /// Header describing the store. The file-layout fields carry no meaning
    /// here: `chunk_offsets` is empty and `index_map` is the identity.
    header: SparseDataHeader,
    /// Library size the normalised layer was scaled to, when known.
    target_size: Option<f32>,
}

impl<'a> InMemorySparseReader<'a> {
    /// Wrap a sparse matrix as a gene-major store.
    ///
    /// ### Params
    ///
    /// * `matrix` - The counts, CSC with shape (cells, genes). Raw counts in
    ///   `data`, normalised counts in `data_2`.
    /// * `target_size` - Library size `data_2` was normalised against, if the
    ///   caller knows it.
    ///
    /// ### Returns
    ///
    /// The reader, or [`BixverseErrors::SparseMatrixMustBeCsc`] for a CSR input
    /// and [`BixverseErrors::Data2NotAvailable`] when the normalised layer is
    /// missing.
    pub fn new(
        matrix: &'a CompressedSparseData2<u32, f32>,
        target_size: Option<f32>,
    ) -> Result<Self, BixverseErrors> {
        if !matrix.cs_type.is_csc() {
            return Err(BixverseErrors::SparseMatrixMustBeCsc);
        }
        if matrix.data_2.is_none() {
            return Err(BixverseErrors::Data2NotAvailable);
        }

        let (n_cells, n_genes) = matrix.shape;
        let library_sizes = cell_library_sizes(matrix, n_cells);

        let header = SparseDataHeader {
            total_cells: n_cells,
            total_genes: n_genes,
            cell_based: false,
            no_chunks: n_genes,
            chunk_offsets: Vec::new(),
            index_map: (0..n_genes)
                .map(|gene| (gene, gene))
                .collect::<FxHashMap<_, _>>(),
        };

        Ok(Self {
            matrix,
            transposed: None,
            library_sizes,
            header,
            target_size,
        })
    }

    /// Wrap the same matrix as a cell-major store.
    ///
    /// Builds and owns a CSR twin, so peak memory is roughly twice the matrix
    /// for as long as the reader lives. Use it only where cell chunks are
    /// genuinely needed, such as the per-cell totals the analytic Pearson fit
    /// sums over each group's retained genes.
    ///
    /// Note this is not [`CompressedSparseData2::transpose_and_convert`], which
    /// relabels a CSC as a CSR and swaps the shape, describing the transpose.
    /// What is wanted here is the same matrix in the other layout, which is a
    /// real scatter.
    ///
    /// ### Params
    ///
    /// * `matrix` - The counts, CSC with shape (cells, genes). Raw counts in
    ///   `data`, normalised counts in `data_2`.
    /// * `target_size` - Library size `data_2` was normalised against, if the
    ///   caller knows it.
    ///
    /// ### Returns
    ///
    /// The reader, or [`BixverseErrors::SparseMatrixMustBeCsc`] for a CSR input
    /// and [`BixverseErrors::Data2NotAvailable`] when the normalised layer is
    /// missing.
    pub fn new_cell_major(
        matrix: &'a CompressedSparseData2<u32, f32>,
        target_size: Option<f32>,
    ) -> Result<Self, BixverseErrors> {
        let mut reader = Self::new(matrix, target_size)?;
        let n_cells = matrix.shape.0;

        // `transpose_sparse` keeps the shape and changes the layout, so this is
        // CSR over (cells, genes): one run per cell, gene indices ascending.
        reader.transposed = Some(transpose_sparse(matrix));
        reader.header.cell_based = true;
        reader.header.no_chunks = n_cells;
        reader.header.index_map = (0..n_cells).map(|cell| (cell, cell)).collect();

        Ok(reader)
    }

    /// Build the chunk for one cell.
    ///
    /// ### Params
    ///
    /// * `cell` - Cell index
    ///
    /// ### Returns
    ///
    /// The [CsrCellChunk], or [`BixverseErrors::ReaderModeMismatch`] in
    /// gene-major mode and [`BixverseErrors::ChunkIndexNotFound`] when the
    /// index is past the cell axis.
    fn cell_chunk(&self, cell: usize) -> Result<CsrCellChunk, BixverseErrors> {
        let transposed = self
            .transposed
            .as_ref()
            .ok_or(BixverseErrors::ReaderModeMismatch {
                actual: "gene-based",
                requested: "cell-based",
            })?;

        if cell >= self.matrix.shape.0 {
            return Err(BixverseErrors::ChunkIndexNotFound(cell));
        }

        let lo = transposed.indptr[cell] as usize;
        let hi = transposed.indptr[cell + 1] as usize;

        let data_raw = RawCounts::from_u32_auto(&transposed.data[lo..hi]);
        let data_norm: Vec<F16> = transposed.data_2.as_ref().expect("checked in `new`")[lo..hi]
            .iter()
            .map(|&v| F16::from_f32(v))
            .collect();

        Ok(CsrCellChunk {
            data_raw,
            data_norm,
            library_size: self.library_sizes[cell],
            indices: transposed.indices[lo..hi].to_vec(),
            original_index: cell,
            to_keep: true,
        })
    }

    /// Build the chunk for one gene.
    ///
    /// ### Params
    ///
    /// * `gene` - Gene index
    ///
    /// ### Returns
    ///
    /// The [CscGeneChunk], or [`BixverseErrors::ChunkIndexNotFound`] when the
    /// index is past the gene axis.
    fn gene_chunk(&self, gene: usize) -> Result<CscGeneChunk, BixverseErrors> {
        if gene >= self.matrix.shape.1 {
            return Err(BixverseErrors::ChunkIndexNotFound(gene));
        }

        let lo = self.matrix.indptr[gene] as usize;
        let hi = self.matrix.indptr[gene + 1] as usize;

        let data_raw = RawCounts::from_u32_auto(&self.matrix.data[lo..hi]);
        let data_norm: Vec<F16> = self.matrix.data_2.as_ref().expect("checked in `new`")[lo..hi]
            .iter()
            .map(|&v| F16::from_f32(v))
            .collect();

        let avg_exp = F16::from_f32(data_norm.iter().map(|v| v.to_f32()).sum::<f32>());

        Ok(CscGeneChunk {
            data_raw,
            data_norm,
            avg_exp,
            nnz: hi - lo,
            indices: self.matrix.indices[lo..hi].to_vec(),
            original_index: gene,
            to_keep: true,
        })
    }
}

/// Total raw counts per cell of a gene-major matrix.
///
/// A CSC matrix has no per-cell run to sum, so this is a scatter-add over every
/// stored value. Each worker takes a contiguous block of genes and its own
/// `n_cells` accumulator, which for metacells is a few hundred kilobytes.
///
/// ### Params
///
/// * `matrix` - The counts, CSC with shape (cells, genes)
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// One library size per cell.
fn cell_library_sizes(matrix: &CompressedSparseData2<u32, f32>, n_cells: usize) -> Vec<usize> {
    let n_genes = matrix.shape.1;

    let totals = thread_chunks(n_genes)
        .par_iter()
        .map(|&(gene_start, gene_end)| {
            let mut acc = vec![0u64; n_cells];

            let lo = matrix.indptr[gene_start] as usize;
            let hi = matrix.indptr[gene_end] as usize;

            for (&cell, &count) in matrix.indices[lo..hi]
                .iter()
                .zip(matrix.data[lo..hi].iter())
            {
                acc[cell as usize] += count as u64;
            }

            acc
        })
        .reduce(
            || vec![0u64; n_cells],
            |mut a, b| {
                for (total, add) in a.iter_mut().zip(b.iter()) {
                    *total += add;
                }
                a
            },
        );

    totals.into_iter().map(|x| x as usize).collect()
}

impl SingleCellReading for InMemorySparseReader<'_> {
    /// Read cells by index in a multi-threaded manner
    ///
    /// Only available on a reader built with
    /// [`InMemorySparseReader::new_cell_major`]; the default gene-major reader
    /// has no CSR twin to read from.
    ///
    /// ### Params
    ///
    /// * `indices` - Index positions of the cells to retrieve
    ///
    /// ### Returns
    ///
    /// The [CsrCellChunk]s in the order given by `indices`, or
    /// [`BixverseErrors::ReaderModeMismatch`] in gene-major mode.
    fn read_cells_parallel(&self, indices: &[usize]) -> Result<Vec<CsrCellChunk>, BixverseErrors> {
        indices
            .par_iter()
            .map(|&cell| self.cell_chunk(cell))
            .collect()
    }

    /// Read genes by index in a multi-threaded manner
    ///
    /// ### Params
    ///
    /// * `indices` - Index positions of the genes to retrieve
    ///
    /// ### Returns
    ///
    /// The [CscGeneChunk]s in the order given by `indices`, or
    /// [`BixverseErrors::ReaderModeMismatch`] in cell-major mode.
    fn read_gene_parallel(&self, indices: &[usize]) -> Result<Vec<CscGeneChunk>, BixverseErrors> {
        if self.transposed.is_some() {
            return Err(BixverseErrors::ReaderModeMismatch {
                actual: "cell-based",
                requested: "gene-based",
            });
        }

        indices
            .par_iter()
            .map(|&gene| self.gene_chunk(gene))
            .collect()
    }

    /// Return the header of the underlying store
    ///
    /// ### Returns
    ///
    /// The [SparseDataHeader].
    fn get_header(&self) -> SparseDataHeader {
        self.header.clone()
    }

    /// Is the store laid out for fast cell retrieval?
    ///
    /// ### Returns
    ///
    /// `true` when built with [`InMemorySparseReader::new_cell_major`].
    fn is_cell_based(&self) -> bool {
        self.transposed.is_some()
    }

    /// Library size the normalised layer was scaled to.
    ///
    /// ### Returns
    ///
    /// Whatever the caller passed to [`InMemorySparseReader::new`].
    fn target_size(&self) -> Option<f32> {
        self.target_size
    }

    /// Total raw counts per cell
    ///
    /// Served from the sums taken in [`InMemorySparseReader::new`], which is
    /// why a gene-major store can answer a cell-side question at all.
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices
    ///
    /// ### Returns
    ///
    /// The library sizes in the order given by `indices`, or
    /// [`BixverseErrors::ChunkIndexNotFound`] for an index past the cell axis.
    fn read_cell_library_sizes(&self, indices: &[usize]) -> Result<Vec<usize>, BixverseErrors> {
        indices
            .iter()
            .map(|&cell| {
                self.library_sizes
                    .get(cell)
                    .copied()
                    .ok_or(BixverseErrors::ChunkIndexNotFound(cell))
            })
            .collect()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Four cells, three genes. Gene 1 is expressed everywhere, gene 2 in a
    /// single cell.
    fn toy_csc() -> CompressedSparseData2<u32, f32> {
        let data: Vec<u32> = vec![
            3, 5, // gene 0, cells 0 and 2
            1, 2, 4, 1, // gene 1, all four cells
            7, // gene 2, cell 3
        ];
        let data_2: Vec<f32> = data.iter().map(|&v| (v as f32).ln_1p()).collect();
        let indices: Vec<u32> = vec![0, 2, 0, 1, 2, 3, 3];
        let indptr: Vec<u32> = vec![0, 2, 6, 7];

        CompressedSparseData2::new_csc(&data, &indices, &indptr, Some(&data_2), (4, 3))
    }

    /// Library sizes sum down cells even though the storage is gene-major.
    #[test]
    fn test_library_sizes_are_the_cell_totals() {
        let matrix = toy_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();

        let sizes = reader.read_cell_library_sizes(&[0, 1, 2, 3]).unwrap();
        assert_eq!(sizes, vec![4, 2, 9, 8]);
    }

    /// Gene chunks come back in request order with the stored column untouched.
    #[test]
    fn test_gene_chunks_carry_the_column_verbatim() {
        let matrix = toy_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();

        let chunks = reader.read_gene_parallel(&[2, 0]).unwrap();

        // order follows the request, not the storage
        assert_eq!(chunks[0].original_index, 2);
        assert_eq!(chunks[0].indices, vec![3]);
        assert_eq!(chunks[0].data_raw.iter().collect::<Vec<u32>>(), vec![7]);
        assert_eq!(chunks[0].nnz, 1);

        assert_eq!(chunks[1].original_index, 0);
        assert_eq!(chunks[1].indices, vec![0, 2]);
        assert_eq!(chunks[1].data_raw.iter().collect::<Vec<u32>>(), vec![3, 5]);
    }

    /// The synthesised header reports the matrix shape and the reader stays gene-based only.
    #[test]
    fn test_header_and_mode() {
        let matrix = toy_csc();
        let reader = InMemorySparseReader::new(&matrix, Some(1e4)).unwrap();

        let header = reader.get_header();
        assert_eq!(header.total_cells, 4);
        assert_eq!(header.total_genes, 3);
        assert!(!reader.is_cell_based());
        assert!(reader.is_gene_based());
        assert_eq!(reader.target_size(), Some(1e4));
        assert!(reader.read_cells_parallel(&[0]).is_err());
    }

    /// The reader is CSC-only, so a CSR matrix must be refused at construction.
    #[test]
    fn test_csr_input_is_rejected() {
        let csc = toy_csc();
        let csr = csc.transform();

        assert!(InMemorySparseReader::new(&csr, None).is_err());
    }

    /// Out-of-range gene and cell indices error rather than reading past the buffers.
    #[test]
    fn test_out_of_range_indices_error() {
        let matrix = toy_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();

        assert!(reader.read_gene_parallel(&[3]).is_err());
        assert!(reader.read_cell_library_sizes(&[4]).is_err());
    }

    /// Cell chunks carry the row the CSC describes, in request order.
    #[test]
    fn test_cell_chunks_carry_the_row_verbatim() {
        let matrix = toy_csc();
        let reader = InMemorySparseReader::new_cell_major(&matrix, None).unwrap();

        let chunks = reader.read_cells_parallel(&[3, 0]).unwrap();

        assert_eq!(chunks[0].original_index, 3);
        assert_eq!(chunks[0].indices, vec![1, 2]);
        assert_eq!(chunks[0].data_raw.iter().collect::<Vec<u32>>(), vec![1, 7]);
        assert_eq!(chunks[0].library_size, 8);

        assert_eq!(chunks[1].original_index, 0);
        assert_eq!(chunks[1].indices, vec![0, 1]);
        assert_eq!(chunks[1].data_raw.iter().collect::<Vec<u32>>(), vec![3, 1]);
        assert_eq!(chunks[1].library_size, 4);
    }

    /// The two modes are exclusive: each serves its own axis and refuses the
    /// other, so a caller cannot silently get the wrong orientation.
    #[test]
    fn test_cell_major_mode_refuses_gene_reads() {
        let matrix = toy_csc();
        let reader = InMemorySparseReader::new_cell_major(&matrix, Some(1e4)).unwrap();

        assert!(reader.is_cell_based());
        assert!(!reader.is_gene_based());
        assert!(reader.get_header().cell_based);
        assert_eq!(reader.get_header().total_cells, 4);
        assert_eq!(reader.target_size(), Some(1e4));
        assert!(reader.read_gene_parallel(&[0]).is_err());
    }

    /// Every stored value survives the transpose, so the two orientations hold
    /// the same matrix.
    #[test]
    fn test_cell_major_preserves_every_entry() {
        let matrix = toy_csc();
        let gene_major = InMemorySparseReader::new(&matrix, None).unwrap();
        let cell_major = InMemorySparseReader::new_cell_major(&matrix, None).unwrap();

        let mut from_genes: Vec<(u32, usize, u32)> = Vec::new();
        for chunk in gene_major.read_gene_parallel(&[0, 1, 2]).unwrap() {
            for (&cell, count) in chunk.indices.iter().zip(chunk.data_raw.iter()) {
                from_genes.push((cell, chunk.original_index, count));
            }
        }

        let mut from_cells: Vec<(u32, usize, u32)> = Vec::new();
        for chunk in cell_major.read_cells_parallel(&[0, 1, 2, 3]).unwrap() {
            for (&gene, count) in chunk.indices.iter().zip(chunk.data_raw.iter()) {
                from_cells.push((chunk.original_index as u32, gene as usize, count));
            }
        }

        from_genes.sort_unstable();
        from_cells.sort_unstable();
        assert_eq!(from_genes, from_cells);
    }

    /// Out-of-range cell indices error rather than reading past the buffers.
    #[test]
    fn test_cell_major_out_of_range_errors() {
        let matrix = toy_csc();
        let reader = InMemorySparseReader::new_cell_major(&matrix, None).unwrap();

        assert!(reader.read_cells_parallel(&[4]).is_err());
    }
}
