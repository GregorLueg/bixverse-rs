//! Implements the SCENIC workflow from Aibar et al., Nat Methods, 2017 for meta
//! cells. It uses the same modifications as the from-disk streaming version.
//!
//! - **Quantisation and histogram-based splitting** to reduce predictor
//!   variable size.
//! - **Multi-output batching** for ExtraTrees and RandomForest learners,
//!   grouping genes to reduce the number of regression learners trained.
//! - **Gene batching strategies**: random assignment or SVD + k-means on
//!   gene loadings to group similar genes together.
//! - **GRNBoost2-style GBM** (Moerman et al., Bioinformatics, 2019) with
//!   histogram-based splits, parent-child subtraction, and OOB early
//!   stopping. Parallelism is exploited across targets rather than batching.

use ann_search_rs::prelude::*;
use ann_search_rs::utils::k_means_utils::*;
use faer::Mat;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::*;

use crate::prelude::*;
use crate::single_cell::mc_processing::hvg_pca::pca_on_metacells;
use crate::single_cell::sc_analysis::scenic::*;
use crate::single_cell::sc_utils::utils_tree::QuantisedStore;

/////////////
// Helpers //
/////////////

/// Extract a single target gene column from a cells x genes CSC matrix
/// as a `SparseAxis<u32, f32>`.
///
/// The normalised counts in `data_2` populate the `data_2` slot of the
/// returned axis; these drive the SCENIC tree fits. The raw counts are
/// cast to `u32` and placed in the `data` slot for parity with the
/// disk path's `CscGeneChunk::to_sparse_axis`, though the fitting code
/// only reads the normalised layer.
///
/// ### Params
///
/// * `csc` - Cells x genes CSC matrix with raw counts in `data` and
///   normalised counts in `data_2`.
/// * `gene` - Column index of the gene to extract.
/// * `n_cells` - Total number of rows (cells) in the matrix.
///
/// ### Returns
///
/// A `SparseAxis<u32, f32>` representing the gene column.
fn extract_target_column<T: Copy + Into<u32>>(
    csc: &CompressedSparseData2<T, f32>,
    gene: usize,
    n_cells: usize,
) -> SparseAxis<u32, f32> {
    let s = csc.indptr[gene];
    let e = csc.indptr[gene + 1];
    let indices: Vec<usize> = csc.indices[s..e].to_vec();
    let raw: Vec<u32> = csc.data[s..e].iter().map(|&v| v.into()).collect();
    let norm: Vec<f32> = csc
        .data_2
        .as_ref()
        .expect("normalised layer (data_2) required")[s..e]
        .to_vec();
    SparseAxis::new_csc(indices, raw, Some(norm), n_cells)
}

/// Build a `QuantisedStore` from selected gene columns of a CSC matrix.
///
/// Equivalent to subsetting the CSC to the selected columns and calling
/// `QuantisedStore::from_csc`, but avoids the intermediate allocation.
/// Each selected column is independently scaled to `[0, 255]` using its
/// observed min and max of the normalised values (`data_2`). The raw
/// layer is not consulted.
///
/// ### Params
///
/// * `csc` - Cells x genes CSC matrix with normalised counts in `data_2`.
/// * `tf_indices` - Column indices of the features (TFs) to include.
/// * `n_cells` - Total number of rows (cells) in the matrix.
///
/// ### Returns
///
/// A `QuantisedStore` with `tf_indices.len()` features in column-major
/// layout, ready for the SCENIC tree fitters.
fn build_tf_quantised_store<T>(
    csc: &CompressedSparseData2<T, f32>,
    tf_indices: &[usize],
    n_cells: usize,
) -> QuantisedStore
where
    T: Clone,
{
    let n_features = tf_indices.len();
    let mut data = vec![0u8; n_features * n_cells];
    let mut mins = Vec::with_capacity(n_features);
    let mut ranges = Vec::with_capacity(n_features);
    let vals = csc
        .data_2
        .as_ref()
        .expect("normalised layer (data_2) required");

    for (new_j, &g) in tf_indices.iter().enumerate() {
        let s = csc.indptr[g];
        let e = csc.indptr[g + 1];
        let col_indices = &csc.indices[s..e];
        let col_vals = &vals[s..e];

        let mut min_v = 0_f32;
        let mut max_v = 0_f32;
        for &v in col_vals {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        let range = max_v - min_v;
        mins.push(min_v);
        ranges.push(range);

        let offset = new_j * n_cells;
        if range > 1e-10 {
            let scale = 255.0 / range;
            for i in 0..col_indices.len() {
                let cell_idx = col_indices[i];
                let val = col_vals[i];
                data[offset + cell_idx] = ((val - min_v) * scale).round() as u8;
            }
        }
    }

    QuantisedStore {
        data,
        n_samples: n_cells,
        n_features,
        feature_min: mins,
        feature_range: ranges,
    }
}

/// Build a new CSC matrix over a subset of rows (cells) and columns
/// (genes), remapping cell indices to their position in `cell_subset`.
///
/// Used as a pre-processing step before `pca_on_metacells` in the
/// correlated gene-batching strategy. Columns in the result appear in
/// the order given by `gene_subset`; rows appear in the order given by
/// `cell_subset`.
///
/// ### Params
///
/// * `csc` - Cells x genes CSC matrix.
/// * `cell_subset` - Row indices to keep, in output row order.
/// * `gene_subset` - Column indices to keep, in output column order.
///
/// ### Returns
///
/// A new `CompressedSparseData2<T, f32>` of shape
/// `(cell_subset.len(), gene_subset.len())`.
fn subset_csc_for_pca<T: Clone>(
    csc: &CompressedSparseData2<T, f32>,
    cell_subset: &[usize],
    gene_subset: &[usize],
) -> CompressedSparseData2<T, f32> {
    let n_new_cells = cell_subset.len();
    let n_new_genes = gene_subset.len();

    let mut cell_map: FxHashMap<usize, usize> = FxHashMap::default();
    for (new_i, &old_i) in cell_subset.iter().enumerate() {
        cell_map.insert(old_i, new_i);
    }

    let vals = csc
        .data_2
        .as_ref()
        .expect("normalised layer (data_2) required");

    let mut new_data: Vec<T> = Vec::new();
    let mut new_data_2: Vec<f32> = Vec::new();
    let mut new_indices: Vec<usize> = Vec::new();
    let mut new_indptr: Vec<usize> = vec![0];

    for &g in gene_subset {
        let s = csc.indptr[g];
        let e = csc.indptr[g + 1];
        for idx in s..e {
            if let Some(&new_i) = cell_map.get(&csc.indices[idx]) {
                new_indices.push(new_i);
                new_data.push(csc.data[idx].clone());
                new_data_2.push(vals[idx]);
            }
        }
        new_indptr.push(new_data.len());
    }

    CompressedSparseData2 {
        data: new_data,
        indices: new_indices,
        indptr: new_indptr,
        cs_type: CompressedSparseFormat::Csc,
        data_2: Some(new_data_2),
        shape: (n_new_cells, n_new_genes),
    }
}

////////////////
// Clustering //
////////////////

/// In-memory variant of `batch_genes_correlated`.
///
/// Runs `pca_on_metacells` on a subsampled cells x (target genes)
/// slice of the input matrix, then k-means clusters the resulting gene
/// loadings. Genes are reordered so that members of the same cluster
/// sit contiguously, giving batches of co-expressed targets for the
/// multi-output tree fitter.
///
/// ### Params
///
/// * `csc` - Full cells x genes CSC matrix.
/// * `gene_indices` - Target gene indices to batch.
/// * `batch_size` - Target batch size (drives the number of k-means
///   centroids).
/// * `n_components` - PCs to compute on the subsampled matrix.
/// * `n_cells_subsample` - Maximum cells to use for the PCA. If the
///   full set is smaller, all cells are used.
/// * `seed` - RNG seed.
/// * `verbose` - Whether to print timing and progress.
///
/// ### Returns
///
/// Gene indices reordered so that co-expressed genes appear next to
/// each other in the returned vector.
fn batch_genes_correlated_in_memory<T: BixverseNumeric>(
    csc: &CompressedSparseData2<T, f32>,
    gene_indices: &[usize],
    batch_size: usize,
    n_components: usize,
    n_cells_subsample: usize,
    seed: usize,
    verbose: bool,
) -> Vec<usize> {
    let n_cells = csc.shape.0;
    let n_genes = gene_indices.len();
    let n_centroids = n_genes.div_ceil(batch_size);

    let all_cells: Vec<usize> = (0..n_cells).collect();
    let sub_cells = subsample_cells(&all_cells, n_cells_subsample.min(n_cells), seed);

    if verbose {
        println!(
            "Computing gene loadings: {} genes, {} subsampled cells, {} components",
            n_genes,
            sub_cells.len(),
            n_components
        );
    }

    let sub_csc = subset_csc_for_pca(csc, &sub_cells, gene_indices);
    let (_, loadings, _) = pca_on_metacells(&sub_csc, n_components, true, seed);

    let dim = loadings.ncols();
    let mut gene_loadings = vec![0.0f32; n_genes * dim];
    for g in 0..n_genes {
        for c in 0..dim {
            gene_loadings[g * dim + c] = loadings[(g, c)];
        }
    }

    if verbose {
        println!("Clustering {} genes into {} groups", n_genes, n_centroids);
    }

    let centroids = train_centroids(
        &gene_loadings,
        dim,
        n_genes,
        n_centroids,
        &Dist::Euclidean,
        50,
        seed,
        verbose,
    );

    let centroid_norms: Vec<f32> = (0..n_centroids)
        .map(|i| {
            let c = &centroids[i * dim..(i + 1) * dim];
            f32::dot_simd(c, c)
        })
        .collect();

    let data_norms: Vec<f32> = (0..n_genes)
        .map(|i| {
            let v = &gene_loadings[i * dim..(i + 1) * dim];
            f32::dot_simd(v, v)
        })
        .collect();

    let assignments = assign_all_parallel(
        &gene_loadings,
        &data_norms,
        dim,
        n_genes,
        &centroids,
        &centroid_norms,
        n_centroids,
        &Dist::Euclidean,
    );

    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); n_centroids];
    for (i, &cluster_id) in assignments.iter().enumerate() {
        clusters[cluster_id].push(gene_indices[i]);
    }

    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1) as u64);
    let mut result = Vec::with_capacity(n_genes);
    for cluster in &mut clusters {
        for i in (1..cluster.len()).rev() {
            let j = rng.random_range(0..=i);
            cluster.swap(i, j);
        }
        result.extend_from_slice(cluster);
    }

    result
}

/// Dispatch gene batching for the in-memory path.
///
/// Mirrors `batch_genes` but routes the correlated strategy through
/// `pca_on_metacells` rather than the disk-streaming PCA.
///
/// ### Params
///
/// * `csc` - Cells x genes CSC matrix.
/// * `gene_indices` - Target gene indices.
/// * `batch_size` - Target batch size.
/// * `strategy` - Batching strategy.
/// * `seed` - RNG seed.
/// * `verbose` - Whether to print progress.
///
/// ### Returns
///
/// Gene indices reordered so that consecutive chunks of `batch_size`
/// form sensible multi-output groups.
fn batch_genes_in_memory<T: BixverseNumeric>(
    csc: &CompressedSparseData2<T, f32>,
    gene_indices: &[usize],
    batch_size: usize,
    strategy: &GeneBatchStrategy,
    seed: usize,
    verbose: bool,
) -> Vec<usize> {
    match strategy {
        GeneBatchStrategy::Random => batch_genes_random(gene_indices, seed),
        GeneBatchStrategy::Correlated {
            n_comp,
            n_cells_subsample,
        } => {
            if gene_indices.len() <= batch_size {
                return batch_genes_random(gene_indices, seed);
            }
            batch_genes_correlated_in_memory(
                csc,
                gene_indices,
                batch_size,
                *n_comp,
                *n_cells_subsample,
                seed,
                verbose,
            )
        }
    }
}

//////////
// Main //
//////////

/// Multi-output RF/ET path for the in-memory SCENIC API.
///
/// Target gene columns are extracted in parallel from the CSC matrix,
/// batched according to the chosen strategy, and fitted via
/// `fit_multi_trees_sparse` with one Rayon task per batch. No I/O is
/// performed.
///
/// ### Params
///
/// * `csc` - Cells x genes CSC matrix (raw in `data`, normalised in
///   `data_2`).
/// * `gene_indices` - Target gene column indices.
/// * `tf_data` - Quantised TF feature store.
/// * `n_cells` - Number of rows in `csc`.
/// * `n_tfs` - Number of features in `tf_data`.
/// * `n_genes` - Number of target genes.
/// * `scenic_params` - SCENIC configuration.
/// * `seed` - Base random seed.
/// * `verbose` - Print progress to stdout.
/// * `start_total` - Timer from the top-level call for elapsed reporting.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` of normalised per-target
/// feature importances, in the order given by `gene_indices`.
#[allow(clippy::too_many_arguments)]
fn run_scenic_multi_output_in_memory<T>(
    csc: &CompressedSparseData2<T, f32>,
    gene_indices: &[usize],
    tf_data: &QuantisedStore,
    n_cells: usize,
    n_tfs: usize,
    n_genes: usize,
    scenic_params: &ScenicParams,
    seed: usize,
    verbose: bool,
    start_total: Instant,
) -> Mat<f32>
where
    T: BixverseNumeric + Copy + Into<u32>,
{
    let n_multi_output = scenic_params
        .gene_batch_size
        .unwrap_or(MULTI_OUTPUT_BATCH)
        .min(MULTI_OUTPUT_BATCH);

    let strategy = parse_gene_batch_strategy(
        &scenic_params.gene_batch_strategy,
        scenic_params.n_pcs,
        scenic_params.n_subsample,
    )
    .unwrap_or(GeneBatchStrategy::Random);

    let ordered_genes =
        batch_genes_in_memory(csc, gene_indices, n_multi_output, &strategy, seed, verbose);

    let gene_id_to_pos: FxHashMap<usize, usize> = gene_indices
        .iter()
        .enumerate()
        .map(|(pos, &gid)| (gid, pos))
        .collect();

    let start_extract = Instant::now();
    let all_sparse_cols: Vec<SparseAxis<u32, f32>> = ordered_genes
        .par_iter()
        .map(|&g| extract_target_column(csc, g, n_cells))
        .collect();

    if verbose {
        println!(
            "Extracted {} target columns in {:.2?}",
            n_genes,
            start_extract.elapsed()
        );
    }

    let id_batches: Vec<&[usize]> = ordered_genes.chunks(n_multi_output).collect();
    let col_batches: Vec<&[SparseAxis<u32, f32>]> =
        all_sparse_cols.chunks(n_multi_output).collect();
    let total_batches = col_batches.len();

    let config: &dyn TreeRegressorConfig = match &scenic_params.regression_learner {
        RegressionLearner::ExtraTrees(cfg) => cfg,
        RegressionLearner::RandomForest(cfg) => cfg,
        RegressionLearner::GradientBoosting(_) => unreachable!(),
    };

    let learner_name = match &scenic_params.regression_learner {
        RegressionLearner::ExtraTrees(_) => "ExtraTrees",
        RegressionLearner::RandomForest(_) => "RandomForest",
        RegressionLearner::GradientBoosting(_) => unreachable!(),
    };

    if verbose {
        println!(
            "Running SCENIC ({}, in-memory) on {} genes ({} TFs, {} cells, {} batches of up to {})",
            learner_name, n_genes, n_tfs, n_cells, total_batches, n_multi_output,
        );
    }

    let start_fit = Instant::now();
    let batches_done = AtomicUsize::new(0);

    let batch_results: Vec<(usize, Vec<Vec<f32>>)> = (0..total_batches)
        .into_par_iter()
        .map(|batch_idx| {
            let batch_seed = seed.wrapping_add(batch_idx.wrapping_mul(2654435761));
            let imp = fit_multi_trees_sparse(
                col_batches[batch_idx],
                tf_data,
                n_cells,
                config,
                batch_seed,
            );

            if verbose {
                let done = batches_done.fetch_add(1, Ordering::Relaxed) + 1;
                let pct = done * 100 / total_batches;
                let prev_pct = (done - 1) * 100 / total_batches;
                if pct / 10 > prev_pct / 10 || done == total_batches {
                    println!(
                        "  Progress: {}% ({}/{} batches, {:.2?} elapsed)",
                        pct,
                        done,
                        total_batches,
                        start_fit.elapsed()
                    );
                }
            }

            (batch_idx, imp)
        })
        .collect();

    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); n_genes];
    for (batch_idx, imp_vecs) in batch_results {
        let batch_gene_ids = id_batches[batch_idx];
        for (local_idx, imp) in imp_vecs.into_iter().enumerate() {
            let gene_id = batch_gene_ids[local_idx];
            let original_pos = gene_id_to_pos[&gene_id];
            importance_scores[original_pos] = imp;
        }
    }

    if verbose {
        println!(
            "SCENIC ({}, in-memory) GRN inference complete in {:.2?}",
            learner_name,
            start_total.elapsed()
        );
    }

    Mat::from_fn(n_genes, n_tfs, |i, j| {
        if j < importance_scores[i].len() {
            importance_scores[i][j]
        } else {
            0.0
        }
    })
}

/// GBM path for the in-memory SCENIC API.
///
/// Each target gene is fitted independently via `fit_grnboost2_sparse`.
/// Sparse columns are extracted up front in parallel; fitting is then
/// parallelised one-gene-per-task.
///
/// ### Params
///
/// * `csc` - Cells x genes CSC matrix.
/// * `gene_indices` - Target gene column indices.
/// * `tf_data` - Quantised TF feature store.
/// * `n_cells` - Number of rows in `csc`.
/// * `n_tfs` - Number of features in `tf_data`.
/// * `n_genes` - Number of target genes.
/// * `config` - GBM configuration.
/// * `seed` - Base random seed.
/// * `verbose` - Print progress to stdout.
/// * `start_total` - Timer from the top-level call for elapsed reporting.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` of normalised per-target
/// feature importances, in the order given by `gene_indices`.
#[allow(clippy::too_many_arguments)]
fn run_scenic_gbm_in_memory<T>(
    csc: &CompressedSparseData2<T, f32>,
    gene_indices: &[usize],
    tf_data: &QuantisedStore,
    n_cells: usize,
    n_tfs: usize,
    n_genes: usize,
    config: &GradientBoostingConfig,
    seed: usize,
    verbose: bool,
    start_total: Instant,
) -> Mat<f32>
where
    T: Copy + Into<u32> + Sync,
{
    let start_extract = Instant::now();
    let all_sparse_cols: Vec<SparseAxis<u32, f32>> = gene_indices
        .par_iter()
        .map(|&g| extract_target_column(csc, g, n_cells))
        .collect();

    if verbose {
        println!(
            "Extracted {} target columns in {:.2?}",
            n_genes,
            start_extract.elapsed()
        );
        println!(
            "Running GRNBoost2 (in-memory) on {} genes ({} TFs, {} cells)",
            n_genes, n_tfs, n_cells,
        );
    }

    let start_fit = Instant::now();
    let genes_done = AtomicUsize::new(0);

    let importance_scores: Vec<Vec<f32>> = all_sparse_cols
        .par_iter()
        .enumerate()
        .map(|(gene_idx, target)| {
            let gene_seed = seed.wrapping_add(gene_idx.wrapping_mul(2654435761));
            let imp = fit_grnboost2_sparse(target, tf_data, n_cells, config, gene_seed);

            if verbose {
                let done = genes_done.fetch_add(1, Ordering::Relaxed) + 1;
                let pct = done * 100 / n_genes;
                let prev_pct = (done - 1) * 100 / n_genes;
                if pct / 10 > prev_pct / 10 || done == n_genes {
                    println!(
                        "  Progress: {}% ({}/{} genes, {:.2?} elapsed)",
                        pct,
                        done,
                        n_genes,
                        start_fit.elapsed()
                    );
                }
            }

            imp
        })
        .collect();

    if verbose {
        println!(
            "GRNBoost2 (in-memory) GRN inference complete in {:.2?}",
            start_total.elapsed()
        );
    }

    Mat::from_fn(n_genes, n_tfs, |i, j| {
        if j < importance_scores[i].len() {
            importance_scores[i][j]
        } else {
            0.0
        }
    })
}

/// Run SCENIC GRN inference on an in-memory cells x genes CSC matrix.
///
/// Designed for meta-cell pipelines where the count matrix fits comfortably in
/// memory and disk streaming is unnecessary. All gene and cell pre-filtering is
/// assumed to have happened upstream, so the matrix rows are exactly the
/// samples used for regression and the columns cover all genes of interest
/// (both TFs and targets, selected via the respective index slices).
///
/// The matrix must hold raw counts in `data` and normalised counts in `data_2`;
/// only the normalised layer is consulted by the regressors. TFs are quantised
/// once into a `QuantisedStore` and shared across all target fits.
///
/// ### Params
///
/// * `expr_csc` - Cells x genes CSC matrix (raw in `data`, normalised
///   in `data_2`).
/// * `gene_indices` - Target gene column indices.
/// * `tf_indices` - Transcription factor column indices (predictors).
/// * `scenic_params` - SCENIC configuration (learner, batching, etc.).
/// * `seed` - Base random seed for reproducibility.
/// * `verbose` - Print progress to stdout.
///
/// ### Returns
///
/// A `Mat<f32>` of shape `(n_genes, n_tfs)` where entry `[i, j]` is the
/// normalised importance of TF `j` for target gene `i`, in the order
/// given by `gene_indices` and `tf_indices`.
pub fn run_scenic_grn_in_memory<T>(
    expr_csc: &CompressedSparseData2<T, f32>,
    gene_indices: &[usize],
    tf_indices: &[usize],
    scenic_params: &ScenicParams,
    seed: usize,
    verbose: bool,
) -> Mat<f32>
where
    T: BixverseNumeric + Copy + Into<u32> + Sync,
{
    // tranpose if need be
    let csc_owned;
    let csc: &CompressedSparseData2<T, f32> = match expr_csc.cs_type {
        CompressedSparseFormat::Csc => expr_csc,
        CompressedSparseFormat::Csr => {
            csc_owned = expr_csc.transform();
            &csc_owned
        }
    };

    let start_total = Instant::now();
    let n_cells = expr_csc.shape.0;
    let n_tfs = tf_indices.len();
    let n_genes = gene_indices.len();

    let start_quant = Instant::now();
    let tf_data = build_tf_quantised_store(csc, tf_indices, n_cells);
    if verbose {
        println!(
            "Quantised TF store (n: {}) in: {:.2?}",
            n_tfs.separate_with_underscores(),
            start_quant.elapsed()
        );
    }

    match &scenic_params.regression_learner {
        RegressionLearner::GradientBoosting(gbm_config) => run_scenic_gbm_in_memory(
            csc,
            gene_indices,
            &tf_data,
            n_cells,
            n_tfs,
            n_genes,
            gbm_config,
            seed,
            verbose,
            start_total,
        ),
        _ => run_scenic_multi_output_in_memory(
            csc,
            gene_indices,
            &tf_data,
            n_cells,
            n_tfs,
            n_genes,
            scenic_params,
            seed,
            verbose,
            start_total,
        ),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_target_column_basic() {
        // 4 cells, 3 genes
        // gene 0: cells 0, 2 -> raw 1, 3; norm 0.1, 0.3
        // gene 1: cell 1 -> raw 5; norm 0.5
        // gene 2: cells 0, 3 -> raw 2, 4; norm 0.2, 0.4
        let csc = CompressedSparseData2 {
            data: vec![1u16, 3, 5, 2, 4],
            indices: vec![0, 2, 1, 0, 3],
            indptr: vec![0, 2, 3, 5],
            cs_type: CompressedSparseFormat::Csc,
            data_2: Some(vec![0.1f32, 0.3, 0.5, 0.2, 0.4]),
            shape: (4, 3),
        };

        let t0 = extract_target_column(&csc, 0, 4);
        assert_eq!(t0.indices, vec![0, 2]);
        assert_eq!(t0.data, vec![1u32, 3]);
        assert_eq!(t0.data_2.as_ref().unwrap(), &vec![0.1f32, 0.3]);
        assert_eq!(t0.len, 4);

        let t1 = extract_target_column(&csc, 1, 4);
        assert_eq!(t1.indices, vec![1]);
        assert_eq!(t1.data, vec![5u32]);

        let t2 = extract_target_column(&csc, 2, 4);
        assert_eq!(t2.data, vec![2u32, 4]);
        assert_eq!(t2.data_2.as_ref().unwrap(), &vec![0.2f32, 0.4]);
    }

    #[test]
    fn build_tf_store_matches_from_csc_full() {
        let csc = CompressedSparseData2 {
            data: vec![1u16, 2, 3, 4, 5],
            indices: vec![0, 2, 1, 0, 2],
            indptr: vec![0, 2, 3, 5],
            cs_type: CompressedSparseFormat::Csc,
            data_2: Some(vec![0.1f32, 0.5, 0.2, 0.8, 1.0]),
            shape: (3, 3),
        };

        let full = QuantisedStore::from_csc(&csc, 3);
        let subset = build_tf_quantised_store(&csc, &[0, 1, 2], 3);

        assert_eq!(full.n_features, subset.n_features);
        assert_eq!(full.n_samples, subset.n_samples);
        assert_eq!(full.data, subset.data);
        assert_eq!(full.feature_min, subset.feature_min);
        assert_eq!(full.feature_range, subset.feature_range);
    }

    #[test]
    fn build_tf_store_respects_tf_order() {
        let csc = CompressedSparseData2 {
            data: vec![1u16, 2, 3, 4, 5],
            indices: vec![0, 2, 1, 0, 2],
            indptr: vec![0, 2, 3, 5],
            cs_type: CompressedSparseFormat::Csc,
            data_2: Some(vec![0.1f32, 0.5, 0.2, 0.8, 1.0]),
            shape: (3, 3),
        };

        let full = QuantisedStore::from_csc(&csc, 3);
        let reversed = build_tf_quantised_store(&csc, &[2, 1, 0], 3);

        for i in 0..3 {
            assert_eq!(full.get_col(2)[i], reversed.get_col(0)[i]);
            assert_eq!(full.get_col(1)[i], reversed.get_col(1)[i]);
            assert_eq!(full.get_col(0)[i], reversed.get_col(2)[i]);
        }
    }

    #[test]
    fn build_tf_store_gene_subset() {
        // 4 cells, 4 genes. Pick genes 1 and 3 as TFs.
        let csc = CompressedSparseData2 {
            data: vec![1u16, 1, 2, 2, 3, 3, 4, 4],
            indices: vec![0, 1, 2, 3, 0, 1, 2, 3],
            indptr: vec![0, 2, 4, 6, 8],
            cs_type: CompressedSparseFormat::Csc,
            data_2: Some(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            shape: (4, 4),
        };

        let tf_store = build_tf_quantised_store(&csc, &[1, 3], 4);
        assert_eq!(tf_store.n_features, 2);
        assert_eq!(tf_store.n_samples, 4);

        // Feature 0 (was gene 1): norm values [0.3, 0.4] at cells [2, 3]
        // min=0.0 (non-negative assumption), max=0.4 -> quantised linearly to 0..255
        let col0 = tf_store.get_col(0);
        assert_eq!(col0[0], 0);
        assert_eq!(col0[1], 0);
        assert_ne!(col0[2], 0);
        assert_ne!(col0[3], 0);
        // cell 3 (norm 0.4) should map to 255; cell 2 (norm 0.3) somewhere below
        assert_eq!(col0[3], 255);
        assert!(col0[2] < col0[3]);
    }

    #[test]
    fn subset_csc_rows_and_genes() {
        // 4 cells, 3 genes
        let csc = CompressedSparseData2 {
            data: vec![1u16, 3, 5, 2, 4],
            indices: vec![0, 2, 1, 0, 3],
            indptr: vec![0, 2, 3, 5],
            cs_type: CompressedSparseFormat::Csc,
            data_2: Some(vec![0.1f32, 0.3, 0.5, 0.2, 0.4]),
            shape: (4, 3),
        };

        // Keep cells [0, 2] (new rows 0, 1) and genes [0, 2]
        let sub = subset_csc_for_pca(&csc, &[0, 2], &[0, 2]);
        assert_eq!(sub.shape, (2, 2));
        // new gene 0 (was gene 0): cells 0, 2 -> new rows 0, 1
        assert_eq!(&sub.indices[sub.indptr[0]..sub.indptr[1]], &[0, 1]);
        assert_eq!(
            &sub.data_2.as_ref().unwrap()[sub.indptr[0]..sub.indptr[1]],
            &[0.1f32, 0.3]
        );
        // new gene 1 (was gene 2): only cell 0 survives (cell 3 dropped)
        assert_eq!(&sub.indices[sub.indptr[1]..sub.indptr[2]], &[0]);
        assert_eq!(
            &sub.data_2.as_ref().unwrap()[sub.indptr[1]..sub.indptr[2]],
            &[0.2f32]
        );
    }

    #[test]
    fn subset_csc_cell_reorder() {
        let csc = CompressedSparseData2 {
            data: vec![1u16, 2, 3],
            indices: vec![0, 1, 2],
            indptr: vec![0, 3],
            cs_type: CompressedSparseFormat::Csc,
            data_2: Some(vec![0.1f32, 0.2, 0.3]),
            shape: (3, 1),
        };

        // Reverse the cell order
        let sub = subset_csc_for_pca(&csc, &[2, 1, 0], &[0]);
        assert_eq!(sub.shape, (3, 1));
        // old cell 0 -> new row 2, old 1 -> new 1, old 2 -> new 0
        // Iteration walks old indices 0, 1, 2 in order -> new indices 2, 1, 0
        assert_eq!(&sub.indices[..], &[2, 1, 0]);
        assert_eq!(sub.data_2.as_ref().unwrap(), &vec![0.1f32, 0.2, 0.3]);
    }

    #[test]
    fn in_memory_scenic_gbm_smoke() {
        // 80 cells, 3 TFs + 2 targets. Target 0 is loosely driven by TF 0.
        let n_cells = 80;
        let n_cols = 5; // cols 0..3 = TFs, 3..5 = targets
        let mut rng = SmallRng::seed_from_u64(7);

        let mut data: Vec<u16> = Vec::new();
        let mut data_2: Vec<f32> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr: Vec<usize> = vec![0];

        // TF columns (0..3)
        let mut tf0_vals = vec![0.0f32; n_cells];
        for j in 0..3 {
            for c in 0..n_cells {
                let v: f32 = rng.random_range(0.0..5.0);
                if v > 1.0 {
                    indices.push(c);
                    data.push(v as u16);
                    data_2.push(v);
                    if j == 0 {
                        tf0_vals[c] = v;
                    }
                }
            }
            indptr.push(data.len());
        }

        // Target 0: driven by TF 0
        for c in 0..n_cells {
            let v = 2.0 * tf0_vals[c] + rng.random_range(-0.5..0.5f32);
            if v > 1.0 {
                indices.push(c);
                data.push(v as u16);
                data_2.push(v);
            }
        }
        indptr.push(data.len());

        // Target 1: pure noise
        for c in 0..n_cells {
            let v: f32 = rng.random_range(0.0..3.0);
            if v > 1.0 {
                indices.push(c);
                data.push(v as u16);
                data_2.push(v);
            }
        }
        indptr.push(data.len());

        assert_eq!(indptr.len(), n_cols + 1);

        let csc = CompressedSparseData2 {
            data,
            indices,
            indptr,
            cs_type: CompressedSparseFormat::Csc,
            data_2: Some(data_2),
            shape: (n_cells, n_cols),
        };

        let params = ScenicParams {
            min_counts: 0,
            min_cells: 0.0,
            regression_learner: RegressionLearner::GradientBoosting(GradientBoostingConfig {
                n_trees_max: 100,
                learning_rate: 0.05,
                max_depth: 3,
                min_samples_leaf: 5,
                early_stop_window: 15,
                subsample_rate: 0.9,
                n_features_split: 0,
            }),
            gene_batch_strategy: "random".to_string(),
            gene_batch_size: None,
            n_pcs: 10,
            n_subsample: 1000,
        };

        let result = run_scenic_grn_in_memory(&csc, &[3, 4], &[0, 1, 2], &params, 42, false);
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 3);

        // Target 0 should lean on TF 0 more than TF 1 or TF 2.
        assert!(
            result[(0, 0)] > result[(0, 1)] && result[(0, 0)] > result[(0, 2)],
            "TF 0 should dominate for target 0: {:?}",
            (result[(0, 0)], result[(0, 1)], result[(0, 2)])
        );
    }

    #[test]
    fn in_memory_scenic_extratrees_smoke() {
        // Same setup as the GBM smoke test, exercising the multi-output path.
        let n_cells = 80;
        let n_cols = 5;
        let mut rng = SmallRng::seed_from_u64(13);

        let mut data: Vec<u16> = Vec::new();
        let mut data_2: Vec<f32> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr: Vec<usize> = vec![0];

        let mut tf0_vals = vec![0.0f32; n_cells];
        for j in 0..3 {
            for c in 0..n_cells {
                let v: f32 = rng.random_range(0.0..5.0);
                if v > 1.0 {
                    indices.push(c);
                    data.push(v as u16);
                    data_2.push(v);
                    if j == 0 {
                        tf0_vals[c] = v;
                    }
                }
            }
            indptr.push(data.len());
        }
        for c in 0..n_cells {
            let v = 2.0 * tf0_vals[c] + rng.random_range(-0.5..0.5f32);
            if v > 1.0 {
                indices.push(c);
                data.push(v as u16);
                data_2.push(v);
            }
        }
        indptr.push(data.len());
        for c in 0..n_cells {
            let v: f32 = rng.random_range(0.0..3.0);
            if v > 1.0 {
                indices.push(c);
                data.push(v as u16);
                data_2.push(v);
            }
        }
        indptr.push(data.len());

        let csc = CompressedSparseData2 {
            data,
            indices,
            indptr,
            cs_type: CompressedSparseFormat::Csc,
            data_2: Some(data_2),
            shape: (n_cells, n_cols),
        };

        let params = ScenicParams {
            min_counts: 0,
            min_cells: 0.0,
            regression_learner: RegressionLearner::ExtraTrees(ExtraTreesConfig {
                n_trees: 100,
                min_samples_leaf: 5,
                n_features_split: 0,
                n_thresholds: 1,
                max_depth: Some(8),
                subsample_frac: None,
            }),
            gene_batch_strategy: "random".to_string(),
            gene_batch_size: None,
            n_pcs: 10,
            n_subsample: 1000,
        };

        let result = run_scenic_grn_in_memory(&csc, &[3, 4], &[0, 1, 2], &params, 42, false);
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 3);

        // Each row sums to ~1 (normalised) or to 0 if no variance was reducible.
        for i in 0..2 {
            let sum: f32 = (0..3).map(|j| result[(i, j)]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-4 || sum < 1e-6,
                "row {} sum = {}",
                i,
                sum
            );
        }
    }
}
