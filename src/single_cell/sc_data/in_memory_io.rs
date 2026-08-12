//! An in-memory [`SingleCellReading`] store over [`CompressedSparseData2`].
//!
//! Everything in `sc_analysis` is written against the streaming reader, which
//! assumes the counts live in the bixverse binary format on disk. Metacells do
//! not: they compress a whole experiment down to something that fits in memory
//! comfortably, and the R side hands them over as a sparse matrix. This adapter
//! presents such a matrix through the same trait so those methods can be reused
//! rather than reimplemented.
//!
//! Gene-major only. Nothing that consumes it needs cell chunks, and building a
//! CSR twin purely to serve them would double the memory for no gain, so
//! [`SingleCellReading::read_cells_parallel`] refuses and the one cell-side
//! quantity that is actually needed, the library sizes, is summed once up
//! front.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

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
    matrix: &'a CompressedSparseData2<u32, f32>,
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
            library_sizes,
            header,
            target_size,
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

        // `avg_exp` is the sum of the normalised layer, matching what
        // `CscGeneChunk::from_conversion` writes on the ingest path.
        let avg_exp = data_norm.iter().sum::<F16>();

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
    /// Always fails: this store has no cell-major view.
    ///
    /// ### Params
    ///
    /// * `_indices` - Ignored
    ///
    /// ### Returns
    ///
    /// [`BixverseErrors::ReaderModeMismatch`].
    fn read_cells_parallel(&self, _indices: &[usize]) -> Result<Vec<CsrCellChunk>, BixverseErrors> {
        Err(BixverseErrors::ReaderModeMismatch {
            actual: "gene-based",
            requested: "cell-based",
        })
    }

    /// Read genes by index in a multi-threaded manner
    ///
    /// ### Params
    ///
    /// * `indices` - Index positions of the genes to retrieve
    ///
    /// ### Returns
    ///
    /// The [CscGeneChunk]s in the order given by `indices`.
    fn read_gene_parallel(&self, indices: &[usize]) -> Result<Vec<CscGeneChunk>, BixverseErrors> {
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
    /// Always `false`.
    fn is_cell_based(&self) -> bool {
        false
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

    #[test]
    fn test_library_sizes_are_the_cell_totals() {
        let matrix = toy_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();

        // cell 0: 3 + 1, cell 1: 2, cell 2: 5 + 4, cell 3: 1 + 7
        let sizes = reader.read_cell_library_sizes(&[0, 1, 2, 3]).unwrap();
        assert_eq!(sizes, vec![4, 2, 9, 8]);
    }

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

    #[test]
    fn test_csr_input_is_rejected() {
        let csc = toy_csc();
        let csr = csc.transform();

        assert!(InMemorySparseReader::new(&csr, None).is_err());
    }

    #[test]
    fn test_out_of_range_indices_error() {
        let matrix = toy_csc();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();

        assert!(reader.read_gene_parallel(&[3]).is_err());
        assert!(reader.read_cell_library_sizes(&[4]).is_err());
    }
}
