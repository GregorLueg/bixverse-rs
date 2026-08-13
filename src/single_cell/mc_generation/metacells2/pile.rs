//! Per-iteration working set of cells.
//!
//! A [`Pile`] is loaded from a [`ParallelSparseReader`] for a specified set of
//! cell indices. It always carries the raw CSR matrix and per-cell library
//! sizes; subsequent pipeline stages populate the optional fields
//! (`downsampled`, `selected_gene_indices`, `selected_dense`).
//!
//! The pile stays in the global gene index space so divide-and-conquer can
//! pool piles across recursion levels without remapping gene indices.

use faer::Mat;

use crate::prelude::*;

/// A working set of cells with derived data populated stage-by-stage.
pub struct Pile {
    /// Reader-level indices of cells in this pile. Used to project results back
    /// to the original dataset.
    pub cell_indices: Vec<usize>,
    /// Cells × genes raw UMI counts, CSR. `data_2` is `None` — the chunks' f16
    /// normalised layer is discarded because MC2 recomputes its own per-pile
    /// downsampling and normalisation.
    pub raw: CompressedSparseData2<u32, f32>,
    /// Per-cell library size, summed from `raw`. We do not reuse the chunk's
    /// `library_size` field because subsetting (and downstream downsampling)
    /// changes the row totals.
    pub umis_per_cell: Vec<f32>,
    /// Total gene count from the reader. All gene indices in this pile (in
    /// `raw.indices`, `selected_gene_indices`, etc.) live in `[0, n_genes)`.
    pub n_genes: usize,
    /// Same shape and sparsity as `raw`, with per-row counts capped via
    /// binomial subsampling. Populated by `downsample_pile`. Explicit zeros
    /// may appear where a non-zero was downsampled away.
    pub downsampled: Option<CompressedSparseData2<u32, f32>>,
    /// Reader-level indices of genes passing feature selection.
    pub selected_gene_indices: Option<Vec<usize>>,
    /// Dense `n_cells × selected_gene_indices.len()` view used for
    /// similarity computation. Populated alongside `selected_gene_indices`.
    pub selected_dense: Option<Mat<f32>>,
}

impl Pile {
    /// Load a pile from a reader.
    ///
    /// Streams the requested cells via
    /// [`ParallelSparseReader::read_cells_parallel`], assembles a CSR matrix
    /// in the global gene index space, and computes per-cell library sizes
    /// from the loaded raw counts.
    ///
    /// ### Params
    ///
    /// * `reader` - The reader to stream cells from. Must be cell-based.
    /// * `cell_indices` - Reader-level cell indices to include.
    ///
    /// ### Returns
    ///
    /// The [`Pile`] of the specified cells.
    pub fn from_reader<S: SingleCellReading>(
        reader: &S,
        cell_indices: &[usize],
    ) -> Result<Self, BixverseErrors> {
        let chunks = reader.read_cells_parallel(cell_indices)?;
        let n_genes = reader.get_header().total_genes;

        let raw = from_cell_chunks::<u32>(&chunks, &DataLayerReturn::Raw, n_genes)?;
        let umis_per_cell = sum_rows_csr(&raw);

        Ok(Self {
            cell_indices: cell_indices.to_vec(),
            raw,
            umis_per_cell,
            n_genes,
            downsampled: None,
            selected_gene_indices: None,
            selected_dense: None,
        })
    }

    /// Number of cells in this pile.
    ///
    /// ### Returns
    ///
    /// Number of cells
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.cell_indices.len()
    }
}

/// Sum each row of a CSR matrix into an `f32` vector.
///
/// ### Params
///
/// * `mat` - The [CompressedSparseData2] of the pile
///
/// ### Returns
///
/// A vector of the row sums as f32
fn sum_rows_csr(mat: &CompressedSparseData2<u32, f32>) -> Vec<f32> {
    let n_rows = mat.shape.0;
    let mut sums = vec![0.0f32; n_rows];
    for i in 0..n_rows {
        let s: u64 = (mat.indptr[i]..mat.indptr[i + 1])
            .map(|idx| mat.data[idx as usize] as u64)
            .sum();
        sums[i] = s as f32;
    }
    sums
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn make_csr(
        data: Vec<u32>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: (usize, usize),
    ) -> CompressedSparseData2<u32, f32> {
        CompressedSparseData2 {
            data,
            indices: indices.index_cast(),
            indptr: indptr.index_cast(),
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape,
        }
    }

    /// Row sums accumulate only the stored non-zeros and stay in cell order.
    #[test]
    fn sum_rows_csr_basic() {
        let mat = make_csr(vec![3, 5, 2], vec![0, 2, 1], vec![0, 2, 3], (2, 3));
        let sums = sum_rows_csr(&mat);
        assert_eq!(sums, vec![8.0, 2.0]);
    }

    /// A row with no stored entries must sum to zero rather than be skipped.
    #[test]
    fn sum_rows_csr_empty_row() {
        // Single cell, no non-zeros.
        let mat = make_csr(vec![], vec![], vec![0, 0], (1, 5));
        let sums = sum_rows_csr(&mat);
        assert_eq!(sums, vec![0.0]);
    }
}
