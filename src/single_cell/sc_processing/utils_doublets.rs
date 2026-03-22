//! Shared infrastructure for doublet detection methods.

use faer::{Mat, MatRef, concat};
use indexmap::IndexSet;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::core::math::pca_svd::*;
use crate::core::math::sparse::sparse_svd_lanczos;
use crate::prelude::*;
use crate::single_cell::sc_processing::hvg::*;
use crate::single_cell::sc_processing::pca::*;

////////////
// Params //
////////////

/// Common HVG parameter bundle
pub struct HvgOpts {
    /// HVG method to use
    pub method: String,
    /// Loess span parameter
    pub loess_span: f32,
    /// Optional clipping parameter
    pub clip_max: Option<f32>,
    /// Minimum percentile for the gene variance
    pub min_gene_var_pctl: f32,
}

/// Common PCA parameter bundle
pub struct PcaOpts {
    /// Shall the data be log transformed
    pub log_transform: bool,
    /// Shall the data be mean centred
    pub mean_center: bool,
    /// Shall the data have scaled variance
    pub normalise_variance: bool,
    /// Number of PCs to use
    pub no_pcs: usize,
    /// Shall randomised SVD be used
    pub random_svd: bool,
}

//////////////////////
// HVG + lib sizes  //
//////////////////////

/// Select highly variable genes
///
/// Identical logic used by Scrublet, Boost and ScDblFinder. Returns gene
/// indices sorted in ascending order.
///
/// ### Params
///
/// * `f_path_gene` - Path to the binary file storing the genes
/// * `cells_to_keep` - Indices of the cells to include in this analysis.
/// * `opts` - Hvg parameters
/// * `streaming` - Shall the data be streamed (reduces memory pressure)
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// Returns the indices of the HVG
pub fn select_hvg(
    f_path_gene: &str,
    cells_to_keep: &[usize],
    opts: &HvgOpts,
    streaming: bool,
    verbose: bool,
) -> Vec<usize> {
    let hvg_type = parse_hvg_method(&opts.method)
        .unwrap_or_else(|| panic!("Invalid HVG method: {}", &opts.method));

    let hvg_res: HvgRes = if streaming {
        match hvg_type {
            HvgMethod::Vst => get_hvg_vst_streaming(
                f_path_gene,
                cells_to_keep,
                opts.loess_span,
                opts.clip_max,
                verbose,
            ),
            HvgMethod::MeanVarBin => get_hvg_mvb_streaming(),
            HvgMethod::Dispersion => get_hvg_dispersion_streaming(),
        }
    } else {
        match hvg_type {
            HvgMethod::Vst => get_hvg_vst(
                f_path_gene,
                cells_to_keep,
                opts.loess_span,
                opts.clip_max,
                verbose,
            ),
            HvgMethod::MeanVarBin => get_hvg_mvb(),
            HvgMethod::Dispersion => get_hvg_dispersion(),
        }
    };

    let n_genes = hvg_res.mean.len() as f32;
    let n_to_take = (n_genes * (1.0 - opts.min_gene_var_pctl)).ceil() as usize;

    let mut indices: Vec<(usize, f64)> = hvg_res
        .var_std
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indices.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indices.truncate(n_to_take);

    let mut result: Vec<usize> = indices.into_iter().map(|(i, _)| i).collect();
    result.sort_unstable();
    result
}

/// Compute per-cell library sizes restricted to HVG genes
///
/// ### Params
///
/// * `f_path_gene` - Path to the binary file storing the cells
/// * `cells_to_keep` - Indices of the cells to include in this analysis.
/// * `hvg_genes` - Indices of the highly variable genes
///
/// ### Returns
///
/// Library sizes per cell given only the HVG genes
pub fn compute_hvg_library_sizes(
    f_path_cell: &str,
    cells_to_keep: &[usize],
    hvg_genes: &[usize],
) -> Vec<usize> {
    let hvg_set: FxHashSet<usize> = hvg_genes.iter().copied().collect();
    let reader = ParallelSparseReader::new(f_path_cell).unwrap();

    cells_to_keep
        .par_iter()
        .map(|&cell_idx| {
            let chunk = reader.read_cell(cell_idx);
            chunk
                .indices
                .iter()
                .enumerate()
                .filter(|&(_, &gene_idx)| hvg_set.contains(&(gene_idx as usize)))
                .map(|(i, _)| chunk.data_raw.get(i) as usize)
                .sum()
        })
        .collect()
}

/// Compute target library size (mean of HVG library sizes if not provided)
pub fn resolve_target_size(explicit: Option<f32>, hvg_library_sizes: &[usize]) -> f32 {
    explicit.unwrap_or_else(|| {
        let sum = hvg_library_sizes.iter().sum::<usize>() as f32;
        sum / hvg_library_sizes.len() as f32
    })
}

/////////
// PCA //
/////////

/// Type alias for Scrublet PCA results
///
/// ### Fields
///
/// * `0` - PCA scores
/// * `1` - PCA loadings
/// * `2` - Gene means
/// * `3` - Gene standard deviations
type DoubletPcaRes = (Mat<f32>, Mat<f32>, Vec<f32>, Vec<f32>);

/// Scale `Vec<CsrCellChunk>` using pre-calculated gene means and stds
///
/// This ensures simulated doublets are scaled using the SAME statistics
/// as the observed cells (critical for proper PCA projection)
///
/// ### Params
///
/// * `chunks` - Vector of cell chunks (simulated doublets)
/// * `gene_means` - Mean for each gene (from observed data)
/// * `gene_stds` - Std dev for each gene (from observed data)
/// * `mean_center` - Center the data around the mean
/// * `normalise_variance` - Normalise the variance
/// * `n_genes` - Total number of genes
///
/// ### Returns
///
/// Dense matrix (cells x genes) with z-scored values
pub fn scale_cell_chunks_with_stats(
    chunks: &[CsrCellChunk],
    gene_means: &[f32],
    gene_stds: &[f32],
    mean_center: bool,
    normalise_variance: bool,
    n_genes: usize,
) -> Mat<f32> {
    let n_cells = chunks.len();
    let mut result = Mat::<f32>::zeros(n_cells, n_genes);

    for (cell_idx, chunk) in chunks.iter().enumerate() {
        match (mean_center, normalise_variance) {
            (false, false) => {
                for i in 0..chunk.indices.len() {
                    let gene = chunk.indices[i] as usize;
                    let val = chunk.data_norm[i].to_f32();
                    *result.get_mut(cell_idx, gene) = val;
                }
            }
            (true, false) => {
                for gene in 0..n_genes {
                    *result.get_mut(cell_idx, gene) = -gene_means[gene];
                }
                for i in 0..chunk.indices.len() {
                    let gene = chunk.indices[i] as usize;
                    let val = chunk.data_norm[i].to_f32();
                    *result.get_mut(cell_idx, gene) = val - gene_means[gene];
                }
            }
            (false, true) => {
                for i in 0..chunk.indices.len() {
                    let gene = chunk.indices[i] as usize;
                    let val = chunk.data_norm[i].to_f32();
                    *result.get_mut(cell_idx, gene) = val / gene_stds[gene];
                }
            }
            (true, true) => {
                for gene in 0..n_genes {
                    *result.get_mut(cell_idx, gene) = -gene_means[gene] / gene_stds[gene];
                }
                for i in 0..chunk.indices.len() {
                    let gene = chunk.indices[i] as usize;
                    let val = chunk.data_norm[i].to_f32();
                    *result.get_mut(cell_idx, gene) = (val - gene_means[gene]) / gene_stds[gene];
                }
            }
        }
    }

    result
}

/// Calculate PCA for doublet detection methods using sparse SVD
///
/// Computes PCA on observed cells without densifying the full matrix. Gene
/// chunks are re-normalised using HVG-specific library sizes (matching the
/// Scrublet normalisation scheme), assembled into a CSC sparse matrix, and
/// decomposed via sparse randomised SVD or Lanczos. This avoids holding
/// an `n_cells x n_genes` dense matrix in memory, which is the primary
/// memory bottleneck in both Scrublet and Boost.
///
/// The returned gene means and standard deviations are computed from the
/// re-normalised sparse data and must be used to scale simulated doublets
/// before projecting them into the same PC space (via
/// `scale_cell_chunks_with_stats`).
///
/// ### Params
///
/// * `f_path_gene` - Path to the gene-based binary file (CSC on disk).
/// * `cell_indices` - Slice of cell indices to include in the analysis.
/// * `gene_indices` - Slice of gene indices (HVGs) to use.
/// * `hvg_library_sizes` - Per-cell library sizes computed over HVG genes
///   only. Must be in the same order as `cell_indices`.
/// * `target_size` - Normalisation target size (e.g. mean HVG library size).
/// * `log_transform` - Whether to apply `ln(1 + x)` after size-factor
///   normalisation.
/// * `mean_center` - Whether to implicitly centre columns during SVD.
/// * `normalise_variance` - Whether to implicitly scale columns to unit
///   variance during SVD.
/// * `no_pcs` - Number of principal components to compute.
/// * `random_svd` - If true, use randomised sparse SVD; otherwise use
///   Lanczos-based sparse SVD.
/// * `seed` - Seed for reproducibility (randomised SVD and Lanczos init).
/// * `verbose` - Controls verbosity of timing output.
///
/// ### Returns
///
/// A `ScrubletPcaRes` tuple of `(scores, loadings, gene_means, gene_stds)`
/// where scores is `n_cells x no_pcs`, loadings is `n_genes x no_pcs`,
/// and means/stds are per-gene vectors for downstream doublet projection.
#[allow(clippy::too_many_arguments)]
pub fn pca_observed(
    f_path_gene: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    hvg_library_sizes: &[usize],
    target_size: f32,
    log_transform: bool,
    mean_center: bool,
    normalise_variance: bool,
    no_pcs: usize,
    random_svd: bool,
    seed: usize,
    verbose: bool,
) -> DoubletPcaRes {
    let start_total = Instant::now();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = cell_indices.len();

    let start_reading = Instant::now();
    let reader = ParallelSparseReader::new(f_path_gene).unwrap();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices);
    if verbose {
        println!("Loaded in data: {:.2?}", start_reading.elapsed());
    }

    let start_prep = Instant::now();

    gene_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    // re-normalise using HVG library sizes
    gene_chunks.par_iter_mut().for_each(|chunk| {
        for (i, &pos) in chunk.indices.iter().enumerate() {
            let raw_count = chunk.data_raw.get(i) as f32;
            let lib_size = hvg_library_sizes[pos as usize] as f32;
            let val = if log_transform {
                ((raw_count / lib_size) * target_size).ln_1p()
            } else {
                (raw_count / lib_size) * target_size
            };
            chunk.data_norm[i] = F16::from(half::f16::from_f32(val.clamp(-65504.0, 65504.0)));
        }
    });

    // assemble into CSC -- remains sparse throughout
    let csc = from_gene_chunks::<f32>(&gene_chunks, n_cells);
    drop(gene_chunks);

    // Column statistics for (a) implicit centering/scaling in SVD and
    // (b) downstream projection of simulated doublets
    let col_means = sparse_csc_column_means(&csc, true);
    let col_stds = sparse_csc_column_stds(&csc, &col_means, true);

    let means_for_svd = if mean_center {
        Some(&col_means[..])
    } else {
        None
    };
    let stds_for_svd = if normalise_variance {
        Some(&col_stds[..])
    } else {
        None
    };

    if verbose {
        println!("Finished data preparation: {:.2?}", start_prep.elapsed());
    }

    let start_svd = Instant::now();

    let (scores, loadings) = if random_svd {
        let svd_res = randomised_sparse_svd::<f32, f32>(
            &csc,
            no_pcs,
            seed as u64,
            true,
            Some(100_usize),
            None,
            means_for_svd,
            stds_for_svd,
        );
        let scores = compute_pc_scores(&svd_res);
        let scores = scores.submatrix(0, 0, n_cells, no_pcs).to_owned();
        let loadings = svd_res
            .v()
            .submatrix(0, 0, gene_indices.len(), no_pcs)
            .to_owned();
        (scores, loadings)
    } else {
        let svd_res = sparse_svd_lanczos::<f32, f32, f32>(
            &csc,
            no_pcs,
            seed as u64,
            true,
            means_for_svd,
            stds_for_svd,
        );
        let scores = compute_pc_scores(&svd_res);
        let loadings = svd_res
            .v()
            .submatrix(0, 0, gene_indices.len(), no_pcs)
            .to_owned();
        (scores, loadings)
    };

    if verbose {
        println!("Finished PCA calculations: {:.2?}", start_svd.elapsed());
        println!(
            "Total run time PCA detection: {:.2?}",
            start_total.elapsed()
        );
    }

    (scores, loadings, col_means, col_stds)
}

/// Run PCA on observed cells, project simulated doublets, return combined
///
/// The observed cells are decomposed via sparse SVD (no densification).
/// Simulated doublet chunks are scaled using the observed statistics and
/// projected into the same PC space. Returns the vertically concatenated
/// scores: observed on top, simulated on bottom.
#[allow(clippy::too_many_arguments)]
pub fn pca_and_project(
    f_path_gene: &str,
    cells_to_keep: &[usize],
    hvg_genes: &[usize],
    hvg_library_sizes: &[usize],
    target_size: f32,
    sim_chunks: &[CsrCellChunk],
    opts: &PcaOpts,
    seed: usize,
    verbose: bool,
) -> (Mat<f32>, DoubletPcaRes) {
    let pca_res = pca_observed(
        f_path_gene,
        cells_to_keep,
        hvg_genes,
        hvg_library_sizes,
        target_size,
        opts.log_transform,
        opts.mean_center,
        opts.normalise_variance,
        opts.no_pcs,
        opts.random_svd,
        seed,
        verbose,
    );

    let scaled_sim = scale_cell_chunks_with_stats(
        sim_chunks,
        &pca_res.2,
        &pca_res.3,
        opts.mean_center,
        opts.normalise_variance,
        hvg_genes.len(),
    );

    let sim_pca = &scaled_sim * &pca_res.1;
    let combined = concat![[&pca_res.0], [sim_pca]];

    (combined, pca_res)
}

/////////
// kNN //
/////////

/// Dispatch to the correct kNN implementation
///
/// Pure routing function. The `k` value should already be adjusted by the
/// caller (e.g. scaled for combined obs+sim size).
///
/// ### Params
///
/// * `embd` - PCA embedding matrix of n_cells x features
/// * `k` - Number of neighbours to use
/// * `knn_params` - Parameters for the various kNN approximate nearest
///   neighbour searches
/// * `seed` - Seed for reproducibility
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The kNN indices for the cells
pub fn dispatch_knn(
    embd: MatRef<f32>,
    k: usize,
    knn_params: &KnnParams,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<usize>> {
    let method = parse_knn_method(&knn_params.knn_method).unwrap_or_default();

    match method {
        KnnSearch::Hnsw => generate_knn_hnsw(
            embd,
            &knn_params.ann_dist,
            k,
            knn_params.m,
            knn_params.ef_construction,
            knn_params.ef_search,
            seed,
            verbose,
        ),
        KnnSearch::Annoy => generate_knn_annoy(
            embd,
            &knn_params.ann_dist,
            k,
            knn_params.n_tree,
            knn_params.search_budget,
            seed,
            verbose,
        ),
        KnnSearch::NNDescent => generate_knn_nndescent(
            embd,
            &knn_params.ann_dist,
            k,
            knn_params.diversify_prob,
            knn_params.ef_budget,
            knn_params.delta,
            seed,
            verbose,
        ),
        KnnSearch::Exhaustive => generate_knn_exhaustive(embd, &knn_params.ann_dist, k, verbose),
        KnnSearch::Ivf => generate_knn_ivf(
            embd,
            &knn_params.ann_dist,
            k,
            knn_params.n_list,
            knn_params.n_list,
            seed,
            verbose,
        ),
    }
}

/// Compute adjusted k for combined observed + simulated kNN.
///
/// Scales k by `(1 + n_sim / n_obs)` to maintain effective neighbourhood
/// density when the graph contains both real and synthetic cells. If
/// `base_k` is 0, defaults to `round(0.5 * sqrt(n_obs))`.
///
/// ### Params
///
/// * `base_k` - Requested number of neighbours. If 0, auto-selected from
///   `n_obs`.
/// * `n_obs` - Number of observed cells.
/// * `n_sim` - Number of simulated doublets.
///
/// ### Returns
///
/// The adjusted k value to use for kNN construction on the combined
/// (observed + simulated) embedding.
pub fn adjusted_k(base_k: usize, n_obs: usize, n_sim: usize) -> usize {
    let k = if base_k == 0 {
        ((n_obs as f32).sqrt() * 0.5).round() as usize
    } else {
        base_k
    };
    let r = n_sim as f32 / n_obs as f32;
    (k as f32 * (1.0 + r)).round() as usize
}

////////////////////////
// Doublet simulation //
////////////////////////

/// Create simulated doublet chunks from explicit cell pairs.
///
/// Core simulation shared across all doublet detection methods. Each pair
/// of cell positions is read from disk, their HVG counts are summed,
/// normalised to `target_size` using the combined HVG library size, and
/// gene indices are remapped to contiguous HVG positions. Pair *selection*
/// strategy differs across methods (random for Scrublet/Boost,
/// cluster-aware for ScDblFinder) but chunk creation is identical.
///
/// ### Params
///
/// * `pairs` - Slice of `(pos_a, pos_b)` tuples indexing into
///   `cells_to_keep`.
/// * `cells_to_keep` - Original cell indices for disk retrieval.
/// * `hvg_library_sizes` - Per-cell library sizes over HVG genes, parallel
///   to `cells_to_keep`.
/// * `hvg_genes` - Sorted gene indices of the selected HVGs.
/// * `f_path_cell` - Path to the cell-based binary file (CSR format).
/// * `target_size` - Normalisation target library size.
/// * `log_transform` - Whether to apply `ln(1 + x)` after normalisation.
///
/// ### Returns
///
/// Vector of `CsrCellChunk`, one per pair, with gene indices remapped to
/// contiguous HVG positions (0-based).
pub fn simulate_from_pairs(
    pairs: &[(usize, usize)],
    cells_to_keep: &[usize],
    hvg_library_sizes: &[usize],
    hvg_genes: &[usize],
    f_path_cell: &str,
    target_size: f32,
    log_transform: bool,
) -> Vec<CsrCellChunk> {
    let hvg_set: FxHashSet<usize> = hvg_genes.iter().copied().collect();
    let gene_to_hvg_idx: FxHashMap<usize, u32> = hvg_genes
        .iter()
        .enumerate()
        .map(|(hvg_idx, &orig_idx)| (orig_idx, hvg_idx as u32))
        .collect();

    let reader = ParallelSparseReader::new(f_path_cell).unwrap();

    pairs
        .par_iter()
        .enumerate()
        .map(|(doublet_idx, &(pos_i, pos_j))| {
            let cell1 = reader.read_cell(cells_to_keep[pos_i]);
            let cell2 = reader.read_cell(cells_to_keep[pos_j]);

            let hvg_combined_lib_size = hvg_library_sizes[pos_i] + hvg_library_sizes[pos_j];

            let mut doublet = CsrCellChunk::add_cells_scrublet(
                &cell1,
                &cell2,
                &hvg_set,
                hvg_combined_lib_size,
                target_size,
                log_transform,
                doublet_idx,
            );

            for idx in doublet.indices.iter_mut() {
                *idx = gene_to_hvg_idx[&(*idx as usize)];
            }

            doublet
        })
        .collect()
}

/////////////////////
// Pair generation //
/////////////////////

/// Generate random cell pairs with replacement.
///
/// Each pair is drawn independently and uniformly from `[0, n_cells)`.
/// A cell can be paired with itself or appear in multiple pairs.
///
/// ### Params
///
/// * `n_cells` - Number of cells to sample from.
/// * `n_pairs` - Number of pairs to generate.
/// * `seed` - Seed for reproducibility.
///
/// ### Returns
///
/// Vector of `(pos_a, pos_b)` tuples, each indexing into the cell
/// population.
pub fn random_pairs(n_cells: usize, n_pairs: usize, seed: usize) -> Vec<(usize, usize)> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed as u64);
    (0..n_pairs)
        .map(|_| {
            let i = rng.random_range(0..n_cells);
            let j = rng.random_range(0..n_cells);
            (i, j)
        })
        .collect()
}

/////////////////////////
// Threshold detection //
/////////////////////////

/// Find the doublet score threshold using Otsu's method.
///
/// Maximises between-class variance of the observed score distribution to
/// find the optimal binary split. Robust to both bimodal and skewed
/// distributions.
///
/// ### Params
///
/// * `scores_obs` - Doublet scores for observed cells.
/// * `n_bins` - Number of histogram bins (50-100 works well).
///
/// ### Returns
///
/// The threshold score. Cells above this value are called doublets.
pub fn find_threshold_otsu(scores_obs: &[f32], n_bins: usize) -> f32 {
    let (max_score, min_score) = array_max_min(scores_obs);

    if (max_score - min_score).abs() < 1e-6 {
        return (min_score + max_score) / 2.0;
    }

    let bin_width = (max_score - min_score) / n_bins as f32;
    let mut hist = vec![0usize; n_bins];

    for &score in scores_obs {
        let bin = ((score - min_score) / bin_width).floor() as usize;
        hist[bin.min(n_bins - 1)] += 1;
    }

    let total = scores_obs.len() as f32;
    let prob: Vec<f32> = hist.iter().map(|&c| c as f32 / total).collect();

    let mut w0 = 0.0f32;
    let mut sum0 = 0.0f32;
    let total_mean: f32 = prob.iter().enumerate().map(|(i, &p)| i as f32 * p).sum();

    let mut best_variance = 0.0f32;
    let mut best_bin = 0usize;

    for i in 0..n_bins {
        w0 += prob[i];
        if w0 < 1e-10 {
            continue;
        }

        let w1 = 1.0 - w0;
        if w1 < 1e-10 {
            break;
        }

        sum0 += i as f32 * prob[i];

        let mu0 = sum0 / w0;
        let mu1 = (total_mean - sum0) / w1;
        let between_var = w0 * w1 * (mu0 - mu1).powi(2);

        if between_var > best_variance {
            best_variance = between_var;
            best_bin = i;
        }
    }

    min_score + (best_bin as f32 + 0.5) * bin_width
}
