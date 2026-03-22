//! Helpers to extract genes for plotting. Can densify raw and normalised
//! counts, do Z-score normalisation for groups and caclulate proportion
//! of cells.

use indexmap::IndexSet;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::simd::*;

/////////////
// Helpers //
/////////////

/// Result for grouped gene statistics (dot plot / heatmap data).
pub struct GroupedGeneStats {
    /// Gene indices (original), length = n_genes
    pub gene_indices: Vec<usize>,
    /// Group labels in order, length = n_groups
    pub group_labels: Vec<String>,
    /// Mean normalised expression per gene per group,
    /// row-major (genes x groups)
    pub mean_expression: Vec<f32>,
    /// Fraction of cells with count > 0 per gene per group, same layout
    pub pct_expressed: Vec<f32>,
}

/// Z-score in place using SIMD helpers, optionally clamp.
///
/// ### Params
///
/// * `data` - Mutable reference to the data
/// * `clip` - Optional clipping values
fn scale_and_clip(data: &mut [f32], clip: Option<f32>) {
    let n = data.len() as f32;
    let mean = sum_simd_f32(data) / n;
    let var = variance_simd_f32(data, mean) / n;
    let sd = var.sqrt();
    if sd > 1e-8 {
        for x in data.iter_mut() {
            *x = (*x - mean) / sd;
        }
    }
    if let Some(c) = clip {
        for x in data.iter_mut() {
            *x = x.clamp(-c, c);
        }
    }
}

////////////////////
// Main functions //
////////////////////

/// Extract raw counts for a single gene as a dense vector over the given cells.
///
/// ### Params
///
/// * `f_path` -  Path to the gene-based binary file.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_index` - The gene index for which to get the values.
///
/// ### Returns
///
/// Returns a vector of length `cell_indices.len()` in the same order as
/// `cell_indices`.
pub fn extract_raw_counts(f_path: &str, cell_indices: &[usize], gene_index: usize) -> Vec<u32> {
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut chunk = reader
        .read_gene_parallel(&[gene_index])
        .into_iter()
        .next()
        .unwrap();
    chunk.filter_selected_cells(&cell_set);

    let n = cell_indices.len();
    let mut dense = vec![0u32; n];
    for (i, &cell_idx) in chunk.indices.iter().enumerate() {
        dense[cell_idx as usize] = chunk.data_raw.get(i);
    }
    dense
}

/// Extract normalised (log1p) counts for a single gene, dense, over given cells.
///
/// If `scale` is true, z-scores across all selected cells (including zeros).
/// If `clip` is provided, scaled values are clamped to `[-clip, clip]`.
///
/// ### Params
///
/// * `f_path` -  Path to the gene-based binary file.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_index` - The gene index for which to get the values.
/// * `scale` - Shall the data be scaled.
/// * `clip` - Shall scaled data be clipped between specific values.
///
/// ### Returns
///
/// Returns a vector of length `cell_indices.len()` in the same order as
/// `cell_indices`.
pub fn extract_norm_counts(
    f_path: &str,
    cell_indices: &[usize],
    gene_index: usize,
    scale: bool,
    clip: Option<f32>,
) -> Vec<f32> {
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut chunk = reader
        .read_gene_parallel(&[gene_index])
        .into_iter()
        .next()
        .unwrap();
    chunk.filter_selected_cells(&cell_set);

    let n = cell_indices.len();
    let mut dense = vec![0.0f32; n];
    for (i, &cell_idx) in chunk.indices.iter().enumerate() {
        dense[cell_idx as usize] = chunk.data_norm[i].to_f32();
    }

    if scale {
        scale_and_clip(&mut dense, clip);
    }

    dense
}

/// Extract normalised (log1p) counts for a multiple genes, dense, over given
/// cells.
///
/// If `scale` is true, z-scores across all selected cells (including zeros).
/// If `clip` is provided, scaled values are clamped to `[-clip, clip]`.
///
/// ### Params
///
/// * `f_path` -  Path to the gene-based binary file.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_index` - The gene index for which to get the values.
/// * `scale` - Shall the data be scaled.
/// * `clip` - Shall scaled data be clipped between specific values.
///
/// ### Returns
///
/// Returns a vector of vectors of length `cell_indices.len()` in the same order
/// as `cell_indices` with the densified gene expression values.
pub fn extract_norm_counts_multi(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    scale: bool,
    clip: Option<f32>,
) -> Vec<Vec<f32>> {
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n = cell_indices.len();

    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut chunks = reader.read_gene_parallel(gene_indices);

    chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    chunks
        .par_iter()
        .map(|chunk| {
            let mut dense = vec![0.0f32; n];
            for (i, &cell_idx) in chunk.indices.iter().enumerate() {
                dense[cell_idx as usize] = chunk.data_norm[i].to_f32();
            }
            if scale {
                scale_and_clip(&mut dense, clip);
            }
            dense
        })
        .collect()
}

/// Compute per-group mean expression and percent expressed for a set of genes.
///
/// `group_levels` defines the factor levels (labels). `group_ids` is parallel
/// to `cell_indices` and contains indices into `group_levels`, i.e.
/// `group_levels[group_ids[i]]` is the label for `cell_indices[i]`.
///
/// This mirrors R's factor representation: an integer vector plus a levels
/// attribute.
///
/// ### Params
///
/// * `f_path` - Path to the gene-based binary file.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `group_ids` - Integer group assignments parallel to `cell_indices`,
///   indexing into `group_levels`.
/// * `group_levels` - Factor level labels. `group_ids` values must be valid
///   indices into this slice.
///
/// ### Returns
///
/// A `GroupedGeneStats` containing per-gene, per-group mean normalised
/// expression and fraction of expressing cells. Output vectors are laid out
/// row-major (genes x n_levels).
pub fn extract_grouped_gene_stats(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    group_ids: &[usize],
    group_levels: &[String],
) -> GroupedGeneStats {
    assert_eq!(
        cell_indices.len(),
        group_ids.len(),
        "cell_indices and group_ids must have equal length"
    );

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_groups = group_levels.len();

    let mut group_sizes = vec![0usize; n_groups];
    for &g in group_ids {
        debug_assert!(
            g < n_groups,
            "group_id {} out of bounds for {} levels",
            g,
            n_groups
        );
        group_sizes[g] += 1;
    }

    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut chunks = reader.read_gene_parallel(gene_indices);

    chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    let n_genes = gene_indices.len();
    let mut mean_expression = vec![0.0f32; n_genes * n_groups];
    let mut pct_expressed = vec![0.0f32; n_genes * n_groups];

    for (g, chunk) in chunks.iter().enumerate() {
        let mut sums = vec![0.0f32; n_groups];
        let mut counts = vec![0usize; n_groups];

        for (i, &new_cell_idx) in chunk.indices.iter().enumerate() {
            let grp = group_ids[new_cell_idx as usize];
            sums[grp] += chunk.data_norm[i].to_f32();
            counts[grp] += 1;
        }

        let offset = g * n_groups;
        for grp in 0..n_groups {
            if group_sizes[grp] > 0 {
                let size = group_sizes[grp] as f32;
                mean_expression[offset + grp] = sums[grp] / size;
                pct_expressed[offset + grp] = counts[grp] as f32 / size;
            }
        }
    }

    GroupedGeneStats {
        gene_indices: chunks.iter().map(|c| c.original_index).collect(),
        group_labels: group_levels.to_vec(),
        mean_expression,
        pct_expressed,
    }
}
