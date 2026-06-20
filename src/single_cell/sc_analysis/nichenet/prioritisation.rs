//! NicheNet prioritisation approach. Takes different approaches. Firstly,
//! uses the ligand activity scoring (see `activity_scoring.rs`), but
//! additionally also ligand and receptor specificity and expression levels
//! and DGEs (if applicable). This module provides:
//!
//! - Per-cluster expression aggregation for sets of genes:
//!   Streams gene chunks from the on-disk single-cell store and aggregates
//!   expression across user-supplied cell clusters. Output is two dense
//!   matrices of shape `(n_genes, n_clusters)`. Downstream specificity scores
//!   (min-max scaling per row, sender-vs-rest contrast, etc.) operate directly
//!   on these matrices.

use faer::Mat;
use rayon::prelude::*;

use crate::prelude::*;

/////////////
// Results //
/////////////

/// Per-cluster expression statistics for a gene set.
///
/// Rows match `gene_indices` order, columns match `clusters` order.
#[derive(Clone, Debug)]
pub struct ClusterExpressionStats<T> {
    /// Matrix containing the means
    pub mean: Mat<T>,
    /// Matrix containing the fraction expressing the genes of interest
    pub frac: Mat<T>,
}

///////////////////////////
// L/R expression levels //
///////////////////////////

/// Compute per-cluster mean expression and fraction-of-cells-expressing
/// for a set of genes.
///
/// Cells not in any cluster are ignored. Clusters are assumed disjoint;
/// if a cell appears in multiple, the last assignment in `clusters` wins.
///
/// ### Params
///
/// * `reader` - Reader pointing to the gene-based data
/// * `gene_indices` - The indices of the genes of interest
/// * `clusters` - Slice of vectors indicating the cell -> cluster membership
///
/// ### Returns
///
/// The [ClusterExpressionStats] results.
pub fn compute_cluster_expression_stats<T>(
    reader: &ParallelSparseReader,
    gene_indices: &[usize],
    clusters: &[Vec<usize>],
) -> Result<ClusterExpressionStats<T>, BixverseErrors>
where
    T: BixverseFloat + Send + Sync,
{
    if !reader.is_gene_based() {
        return Err(BixverseErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }

    let n_cells = reader.get_header().total_cells;
    let n_genes = gene_indices.len();
    let n_clusters = clusters.len();

    // 1. cell -> cluster lookup. None for cells outside all clusters.
    let mut cell_to_cluster: Vec<Option<u32>> = vec![None; n_cells];
    for (cluster_idx, cells) in clusters.iter().enumerate() {
        for &c in cells {
            cell_to_cluster[c] = Some(cluster_idx as u32);
        }
    }
    let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();

    // 2. stream gene chunks
    let chunks = reader.read_gene_parallel(gene_indices)?;

    // 3. aggregate per gene in parallel
    let aggregated: Vec<(Vec<T>, Vec<T>)> = chunks
        .par_iter()
        .map(|chunk| {
            let mut sum = vec![T::zero(); n_clusters];
            let mut count = vec![0usize; n_clusters];
            for (i, &cell_idx) in chunk.indices.iter().enumerate() {
                if let Some(cluster) = cell_to_cluster[cell_idx as usize] {
                    let v = T::from_f32(chunk.data_norm[i].to_f32()).unwrap();
                    sum[cluster as usize] += v;
                    count[cluster as usize] += 1;
                }
            }
            let mean: Vec<T> = sum
                .iter()
                .zip(cluster_sizes.iter())
                .map(|(&s, &n)| {
                    if n > 0 {
                        s / T::from_usize(n).unwrap()
                    } else {
                        T::zero()
                    }
                })
                .collect();
            let frac: Vec<T> = count
                .iter()
                .zip(cluster_sizes.iter())
                .map(|(&c, &n)| {
                    if n > 0 {
                        T::from_usize(c).unwrap() / T::from_usize(n).unwrap()
                    } else {
                        T::zero()
                    }
                })
                .collect();
            (mean, frac)
        })
        .collect();

    // 4. assemble into Mat<T>
    let mut mean_mat = Mat::zeros(n_genes, n_clusters);
    let mut frac_mat = Mat::zeros(n_genes, n_clusters);
    for (i, (mean_row, frac_row)) in aggregated.into_iter().enumerate() {
        for (j, v) in mean_row.into_iter().enumerate() {
            mean_mat[(i, j)] = v;
        }
        for (j, v) in frac_row.into_iter().enumerate() {
            frac_mat[(i, j)] = v;
        }
    }

    Ok(ClusterExpressionStats {
        mean: mean_mat,
        frac: frac_mat,
    })
}
