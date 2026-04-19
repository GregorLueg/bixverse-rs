//! Shared infrastructure for doublet detection methods.

use faer::{Mat, MatRef, concat};
use half::f16;
use indexmap::IndexSet;
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
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
    /// Minimum percentile for the gene variance
    pub min_gene_var_pctl: f32,
    /// HVG method to use
    pub method: String,
    /// Loess span parameter
    pub loess_span: f32,
    /// Optional clipping parameter
    pub clip_max: Option<f32>,
    /// Number of bins
    pub n_bins: usize,
    /// Binning strategy, one of `"equal_width"` or `"equal_freq"`
    pub binning_strategy: String,
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

    let sort_key: Vec<f64> = match hvg_type {
        HvgMethod::Vst => {
            let res = if streaming {
                get_hvg_vst_streaming(
                    f_path_gene,
                    cells_to_keep,
                    opts.loess_span,
                    opts.clip_max,
                    verbose,
                )
            } else {
                get_hvg_vst(
                    f_path_gene,
                    cells_to_keep,
                    opts.loess_span,
                    opts.clip_max,
                    verbose,
                )
            };
            res.var_std
        }
        HvgMethod::Dispersion => {
            let res = if streaming {
                get_hvg_dispersion_streaming(
                    f_path_gene,
                    cells_to_keep,
                    &opts.binning_strategy,
                    opts.n_bins,
                    verbose,
                )
            } else {
                get_hvg_dispersion(
                    f_path_gene,
                    cells_to_keep,
                    &opts.binning_strategy,
                    opts.n_bins,
                    verbose,
                )
            };
            res.dispersion
        }
        HvgMethod::MeanVarBin => {
            let res = if streaming {
                get_hvg_mvb_streaming(
                    f_path_gene,
                    cells_to_keep,
                    &opts.binning_strategy,
                    opts.n_bins,
                    verbose,
                )
            } else {
                get_hvg_mvb(
                    f_path_gene,
                    cells_to_keep,
                    &opts.binning_strategy,
                    opts.n_bins,
                    verbose,
                )
            };
            res.dispersion_scaled
        }
    };

    let n_genes = sort_key.len() as f32;
    let n_to_take = (n_genes * (1.0 - opts.min_gene_var_pctl)).ceil() as usize;
    let mut indices: Vec<(usize, f64)> = sort_key.into_iter().enumerate().collect();
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

    let csc = from_gene_chunks::<f32>(&gene_chunks, n_cells);
    drop(gene_chunks);

    let col_means: Vec<f64> = sparse_csc_column_means(&csc, true);
    let col_stds: Vec<f64> = sparse_csc_column_stds(&csc, &col_means, true);

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
        let svd_res = randomised_sparse_svd::<f32, f64>(
            &csc,
            no_pcs,
            seed as u64,
            true,
            Some(100_usize),
            None,
            means_for_svd,
            stds_for_svd,
        );
        let scores_f64 = compute_pc_scores(&svd_res);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| scores_f64[(i, j)] as f32);
        let loadings = Mat::<f32>::from_fn(gene_indices.len(), no_pcs, |i, j| {
            svd_res.v()[(i, j)] as f32
        });
        (scores, loadings)
    } else {
        let svd_res = sparse_svd_lanczos::<f32, f32, f64>(
            &csc,
            no_pcs,
            seed as u64,
            true,
            means_for_svd,
            stds_for_svd,
        );
        let scores_f64 = compute_pc_scores(&svd_res);
        let scores = Mat::<f32>::from_fn(scores_f64.nrows(), scores_f64.ncols(), |i, j| {
            scores_f64[(i, j)] as f32
        });
        let loadings = Mat::<f32>::from_fn(svd_res.v().nrows(), svd_res.v().ncols(), |i, j| {
            svd_res.v()[(i, j)] as f32
        });
        (scores, loadings)
    };

    if verbose {
        println!("Finished PCA calculations: {:.2?}", start_svd.elapsed());
        println!(
            "Total run time PCA detection: {:.2?}",
            start_total.elapsed()
        );
    }

    // Cast means/stds back to f32 for the return type
    let col_means_f32: Vec<f32> = col_means.iter().map(|&x| x as f32).collect();
    let col_stds_f32: Vec<f32> = col_stds.iter().map(|&x| x as f32).collect();

    (scores, loadings, col_means_f32, col_stds_f32)
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
            false,
            verbose,
        ),
        KnnSearch::Annoy => generate_knn_annoy(
            embd,
            &knn_params.ann_dist,
            k,
            knn_params.n_tree,
            knn_params.search_budget,
            seed,
            false,
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
            false,
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
            false,
            verbose,
        ),
        KnnSearch::KmKnn => generate_knn_kmknn(
            embd,
            &knn_params.ann_dist,
            k,
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

/////////////////
// scDblFinder //
/////////////////

////////////////////
// Gene selection //
////////////////////

/// Simple overall-mean ranking path.
///
/// ### Params
///
/// * `gene_chunks` - Slice of the gene chunks for which to get the top N
///   expressed genes.
/// * `n_cells` - Number of total cells
/// * `n_top` - Total number of top genes to return
///
/// ### Returns
///
/// The indices of the genes to take forward
fn select_by_overall_mean(
    gene_chunks: &[CscGeneChunk],
    n_cells: usize,
    n_top: usize,
) -> Vec<usize> {
    let nf = n_cells as f64;
    let means: Vec<f64> = gene_chunks
        .par_iter()
        .map(|chunk| {
            let sum: f64 = (0..chunk.indices.len())
                .map(|i| chunk.data_raw.get(i) as f64)
                .sum();
            sum / nf
        })
        .collect();
    top_n_indices(&means, n_top)
}

/// Per-cluster round-robin selection matching R's selFeatures.
///
/// ### Params
///
/// * `gene_chunks` - Slice of the gene chunks for which to get the top N
///   expressed genes.
/// * `cluster_labels` - Cluster lables <- this will be used to identify the
///   top highly expressed genes across the clusters.
/// * `n_top` - Total number of top genes to return
///
/// ### Returns
///
/// The indices of the genes to take forward
fn select_by_cluster_roundrobin(
    gene_chunks: &[CscGeneChunk],
    cluster_labels: &[usize],
    n_top: usize,
) -> Vec<usize> {
    // build a dense cluster id mapping (original label -> 0..K index)
    let unique_clusters: Vec<usize> = {
        let mut s: FxHashSet<usize> = FxHashSet::default();
        for &c in cluster_labels {
            s.insert(c);
        }
        let mut v: Vec<usize> = s.into_iter().collect();
        v.sort_unstable();
        v
    };
    let n_clusters = unique_clusters.len();

    // degenerate case: one cluster is equivalent to no clusters
    if n_clusters <= 1 {
        return select_by_overall_mean(gene_chunks, cluster_labels.len(), n_top);
    }

    let cluster_idx: FxHashMap<usize, usize> = unique_clusters
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    // compute per-gene per-cluster sums of raw counts. after
    // filter_selected_cells, chunk.indices[i] is a position
    // into cells_to_keep (== cluster_labels).
    let cluster_sums: Vec<Vec<f64>> = gene_chunks
        .par_iter()
        .map(|chunk| {
            let mut sums = vec![0.0f64; n_clusters];
            for i in 0..chunk.indices.len() {
                let pos = chunk.indices[i] as usize;
                let c_idx = cluster_idx[&cluster_labels[pos]];
                sums[c_idx] += chunk.data_raw.get(i) as f64;
            }
            sums
        })
        .collect();

    let n_genes = gene_chunks.len();

    // for each cluster, rank genes by that cluster's sum descending,
    // take the top n_top indices. This gives us n_clusters lists.
    let per_cluster_rankings: Vec<Vec<usize>> = (0..n_clusters)
        .into_par_iter()
        .map(|c_idx| {
            let col: Vec<f64> = (0..n_genes).map(|g| cluster_sums[g][c_idx]).collect();
            top_n_indices_ordered(&col, n_top)
        })
        .collect();

    // round-robin flatten + dedupe + truncate to n_top
    let mut result = roundrobin_select(&per_cluster_rankings, n_top);

    // Fallback: if clusters have heavy rank overlap and we run out of
    // unique candidates, fill from overall sum ranking. R would return
    // NAs here; this is more robust.
    if result.len() < n_top {
        let totals: Vec<f64> = cluster_sums.iter().map(|sums| sums.iter().sum()).collect();
        let fallback = top_n_indices_ordered(&totals, gene_chunks.len());
        let mut seen: FxHashSet<usize> = result.iter().copied().collect();
        for g in fallback {
            if seen.insert(g) {
                result.push(g);
                if result.len() >= n_top {
                    break;
                }
            }
        }
    }

    result.sort_unstable();
    result
}

/// Round-robin across per-cluster rankings, collecting unique gene indices
/// until `n_top` are found.
///
/// Iterates rank 0 across all clusters, then rank 1, etc., matching R's
/// `as.numeric(t(apply(..., 2, ...)))` column-major flattening after transpose.
///
/// ### Params
///
/// * `per_cluster_rankings` - Rank on a per cluster basis
/// * `n_top` - Number of genes to return
///
/// ### Returns
///
fn roundrobin_select(per_cluster_rankings: &[Vec<usize>], n_top: usize) -> Vec<usize> {
    let n_clusters = per_cluster_rankings.len();
    let max_rank = per_cluster_rankings
        .iter()
        .map(|v| v.len())
        .max()
        .unwrap_or(0);

    let mut seen: FxHashSet<usize> = FxHashSet::default();
    let mut result: Vec<usize> = Vec::with_capacity(n_top);

    for rank in 0..max_rank {
        for c_idx in 0..n_clusters {
            if rank < per_cluster_rankings[c_idx].len() {
                let g = per_cluster_rankings[c_idx][rank];
                if seen.insert(g) {
                    result.push(g);
                    if result.len() >= n_top {
                        return result;
                    }
                }
            }
        }
    }
    result
}

/// Return the `n_top` indices of the largest values, sorted ascending by index.
///
/// ### Params
///
/// * `values` - The values for which to return the top N indices
/// * `n_top` - Number of top genes to return
///
/// ### Returns
///
/// The indices of the selected genes.
fn top_n_indices(values: &[f64], n_top: usize) -> Vec<usize> {
    let mut result = top_n_indices_ordered(values, n_top);
    result.sort_unstable();
    result
}

/// Like `top_n_indices` but returns them in ranking order
///
/// Ranked descending by value, not sorted by index. Needed for the round-robin
/// step since rank position matters.
///
/// ### Params
///
/// * `values` - The values for which to return the top N indices
/// * `n_top` - Number of top genes to return
///
/// ### Returns
///
/// The indices of the selected genes.
fn top_n_indices_ordered(values: &[f64], n_top: usize) -> Vec<usize> {
    let n = values.len();
    let k = n_top.min(n);
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/////////////////////////////
// Main selection function //
/////////////////////////////

/// Select the top `n_top` genes by mean raw expression, optionally
/// with per-cluster round-robin selection.
///
/// Matches R scDblFinder's `selFeatures()` with `propMarkers=0`. Operates on
/// RAW(!) counts -> library size implicitly drives selection, which is
/// intentional for doublet detection.
///
/// ### Details
///
/// #### With clusters
///
/// For each cluster, ranks genes by that cluster's summed raw counts
/// (descending) and takes the top `n_top`. Then round-robin through ranks,
/// gene at rank 1 from each cluster, then rank 2, etc., deduplicating and
/// stopping at `n_top` unique genes. This ensures each cluster contributes
/// representative features even if its cell count is small.
///
/// #### Without clusters
///
/// Simple top-N by overall mean raw expression.
///
/// ### Params
///
/// * `f_path_gene` - Path to the gene-based binary file (CSC).
/// * `cells_to_keep` - Cell indices to include.
/// * `clusters` - Optional cluster labels, parallel to `cells_to_keep`. If
///   `None`, falls back to overall mean ranking.
/// * `n_top` - Number of genes to select.
///
/// ### Returns
///
/// Sorted ascending vector of gene indices.
pub fn select_top_genes(
    f_path_gene: &str,
    cells_to_keep: &[usize],
    clusters: Option<&[usize]>,
    n_top: usize,
) -> Vec<usize> {
    let reader = ParallelSparseReader::new(f_path_gene).unwrap();
    let n_total_genes = reader.get_header().total_genes;

    if n_top >= n_total_genes {
        return (0..n_total_genes).collect();
    }
    if cells_to_keep.is_empty() {
        return (0..n_top).collect();
    }

    let cell_set: IndexSet<u32> = cells_to_keep.iter().map(|&x| x as u32).collect();
    let all_indices: Vec<usize> = (0..n_total_genes).collect();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(&all_indices);

    gene_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    match clusters {
        None => select_by_overall_mean(&gene_chunks, cells_to_keep.len(), n_top),
        Some(labels) => {
            assert_eq!(
                labels.len(),
                cells_to_keep.len(),
                "cluster labels must be parallel to cells_to_keep",
            );
            select_by_cluster_roundrobin(&gene_chunks, labels, n_top)
        }
    }
}

////////////////////////
// Doublet generation //
////////////////////////

/// Parameters controlling scDblFinder-style doublet simulation noise.
#[derive(Clone, Debug)]
pub struct ScDblSimParams {
    /// Fraction of doublets whose counts are Poisson-resampled.
    ///
    /// Each non-zero count `c` is replaced by `Poisson(c)`. This adds
    /// realistic sampling noise matching the stochastic capture process
    /// in droplet-based scRNA-seq. Default: 0.25.
    pub resamp_frac: f32,

    /// Fraction of doublets whose total counts are halved before any
    /// resampling.
    ///
    /// Real doublets share the reagent budget within a single droplet,
    /// so their effective library size is typically less than the sum
    /// of two singlets. Default: 0.25.
    pub half_size_frac: f32,

    /// Fraction of doublets with cluster-based size adjustment.
    ///
    /// The contribution of each parent cell is weighted by the average
    /// of (a) the actual library size ratio and (b) the ratio of
    /// cluster median library sizes. This prevents systematic bias
    /// when combining cells from clusters with very different depths.
    /// The total library size is preserved after reweighting.
    /// Default: 0.25.
    pub adjust_size_frac: f32,
}

impl Default for ScDblSimParams {
    fn default() -> Self {
        Self {
            resamp_frac: 0.25,
            half_size_frac: 0.25,
            adjust_size_frac: 0.25,
        }
    }
}

/// Pre-rolled per-doublet treatment flags.
///
/// Generated deterministically from the master seed before parallel
/// execution, so results are reproducible regardless of thread
/// scheduling.
#[derive(Clone, Debug)]
pub struct DoubletTreatment {
    /// Shall the size be adjusted
    adjust_size: bool,
    /// Shall the size be halved
    half_size: bool,
    /// Shall the counts be resampled via Poisson sampling
    resamp: bool,
    /// Per-doublet seed for Poisson resampling (ensures reproducibility
    /// under parallel execution).
    rng_seed: u64,
}

/////////////
// Helpers //
/////////////

/// Sample from Poisson(lambda) using the inverse transform method.
///
/// O(lambda) expected time. For lambda > 500, uses the normal
/// approximation with continuity correction to avoid excessive
/// iteration. Both regimes are adequate for scRNA-seq count data.
///
/// ### Params
///
/// * `rng` - Random number generator
/// * `lambda` - Lambda value
///
/// ### Returns
///
/// The poisson value for that lambda
fn poisson_sample(rng: &mut impl Rng, lambda: f64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    if lambda > 500.0 {
        // Box-Muller for normal variate
        let u1: f64 = rng.random::<f64>().max(1e-300);
        let u2: f64 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let val = lambda + lambda.sqrt() * z;
        return val.round().max(0.0) as u32;
    }
    let l = (-lambda).exp();
    let mut k = 0u32;
    let mut p = 1.0f64;
    loop {
        p *= rng.random::<f64>();
        if p < l {
            return k;
        }
        k += 1;
    }
}

/// Compute median library size per cluster.
///
/// Returns a map from cluster label to the median selected gene library size of
/// cells in that cluster. Used for size-adjusted doublet generation.
///
/// ### Params
///
/// * `cluster_labels` - The labels for the clusters
/// * `selected_gene_library_sizes` - Library sizes of the selected genes
///
/// ### Returns
///
/// HashMap with clusters and their median library sizes
fn compute_cluster_median_lib_sizes(
    cluster_labels: &[usize],
    selected_genes_library_sizes: &[usize],
) -> FxHashMap<usize, f64> {
    let mut per_cluster: FxHashMap<usize, Vec<f64>> = FxHashMap::default();
    for (i, &cl) in cluster_labels.iter().enumerate() {
        per_cluster
            .entry(cl)
            .or_default()
            .push(selected_genes_library_sizes[i] as f64);
    }
    per_cluster
        .into_iter()
        .map(|(cl, mut sizes)| {
            sizes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let med = if sizes.len() % 2 == 0 {
                (sizes[sizes.len() / 2 - 1] + sizes[sizes.len() / 2]) / 2.0
            } else {
                sizes[sizes.len() / 2]
            };
            (cl, med.max(1.0))
        })
        .collect()
}

////////////////
// Core logic //
////////////////

/// Generate a single doublet's raw counts from two dense parent count vectors.
///
/// This is the testable core of the simulation pipeline. It operates on dense
/// gene-count vectors (length = number of selected genes) and applies the full
/// R-matching noise model from scDblFinder.
///
/// ### Params
///
/// * `parent_a` - Dense selected genes count vector for parent cell A.
/// * `parent_b` - Dense selected genes count vector for parent cell B.
/// * `lib_a` - Selected genes library size of parent A.
/// * `lib_b` - Selected genes library size of parent B.
/// * `cluster_a` - Cluster label of parent A.
/// * `cluster_b` - Cluster label of parent B.
/// * `cluster_medians` - Median selected gene library size per cluster.
/// * `treatment` - Which noise treatments to apply.
///
/// ### Returns
///
/// `(counts, effective_library_size)` where `counts` is a dense
/// vector of noise-injected raw counts.
#[allow(clippy::too_many_arguments)]
pub fn generate_single_doublet(
    parent_a: &[u32],
    parent_b: &[u32],
    lib_a: usize,
    lib_b: usize,
    cluster_a: usize,
    cluster_b: usize,
    cluster_medians: &FxHashMap<usize, f64>,
    treatment: &DoubletTreatment,
) -> (Vec<u32>, usize) {
    let n_genes = parent_a.len();
    debug_assert_eq!(parent_b.len(), n_genes);

    let combined_lib = (lib_a + lib_b) as f64;
    let mut counts_f64 = vec![0.0f64; n_genes];

    if treatment.adjust_size {
        // Cluster-based size adjustment (R's adjustSize logic)
        //
        // factor = average of:
        //   (1) actual library size ratio: lib_a / (lib_a + lib_b)
        //   (2) cluster median ratio: med_a / (med_a + med_b)
        // Clamped to [0.2, 0.8].
        //
        // doublet = parent_a * factor + parent_b * (1 - factor)
        // Then rescaled so total counts = lib_a + lib_b.
        let actual_ratio = lib_a as f64 / combined_lib;
        let med_a = cluster_medians.get(&cluster_a).copied().unwrap_or(1.0);
        let med_b = cluster_medians.get(&cluster_b).copied().unwrap_or(1.0);
        let cluster_ratio = med_a / (med_a + med_b);
        let factor = ((actual_ratio + cluster_ratio) / 2.0).clamp(0.2, 0.8);

        for g in 0..n_genes {
            counts_f64[g] = parent_a[g] as f64 * factor + parent_b[g] as f64 * (1.0 - factor);
        }

        // Rescale to preserve combined library size
        let raw_sum: f64 = counts_f64.iter().sum();
        if raw_sum > 0.0 {
            let scale = combined_lib / raw_sum;
            for v in counts_f64.iter_mut() {
                *v *= scale;
            }
        }
    } else {
        // Simple sum (no adjustment)
        for g in 0..n_genes {
            counts_f64[g] = parent_a[g] as f64 + parent_b[g] as f64;
        }
    }

    // Library size halving
    if treatment.half_size {
        for v in counts_f64.iter_mut() {
            *v *= 0.5;
        }
    }

    // Poisson resampling or rounding
    let mut rng = StdRng::seed_from_u64(treatment.rng_seed);
    let counts: Vec<u32> = if treatment.resamp {
        counts_f64
            .iter()
            .map(|&v| poisson_sample(&mut rng, v))
            .collect()
    } else {
        counts_f64
            .iter()
            .map(|&v| v.round().max(0.0) as u32)
            .collect()
    };

    let effective_lib: usize = counts.iter().map(|&c| c as usize).sum();
    (counts, effective_lib)
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

        let data_raw = RawCounts::from_u32_auto(&raw_counts);
        let data_norm: Vec<F16> = norm_values
            .iter()
            .map(|&v| F16::from(f16::from_f32(v)))
            .collect();

        Self {
            data_raw,
            data_norm,
            library_size: effective_lib_size,
            indices,
            original_index: doublet_index,
            to_keep: true,
        }
    }
}

/// Simulate doublets with scDblFinder-style noise injection.
///
/// Replaces `simulate_from_pairs` in the scDblFinder pipeline. Each
/// pair of observed cells is read from disk, their HVG counts are
/// combined (with optional cluster-based size adjustment), then
/// optionally halved and/or Poisson-resampled. The result is stored
/// as sparse chunks with both raw and normalised values, ready for
/// consumption by `pca_combined` and `build_feature_matrix`.
///
/// ### Params
///
/// * `pairs` - `(pos_a, pos_b)` tuples indexing into `cells_to_keep`.
/// * `cells_to_keep` - Original cell indices for disk retrieval.
/// * `hvg_library_sizes` - Per-cell HVG library sizes, parallel to
///   `cells_to_keep`.
/// * `cluster_labels` - Per-cell cluster labels, parallel to
///   `cells_to_keep`.
/// * `hvg_genes` - Sorted gene indices of selected HVGs.
/// * `f_path_cell` - Path to the cell-based binary file (CSR).
/// * `target_size` - Normalisation target library size.
/// * `log_transform` - Whether to apply `ln(1 + x)` after
///   normalisation.
/// * `params` - Noise injection parameters.
/// * `seed` - Master seed for reproducibility.
///
/// ### Returns
///
/// `(chunks, library_sizes)` where `chunks` contains sparse
/// representations with noise-injected raw counts and normalised
/// values, and `library_sizes` holds the post-noise effective HVG
/// library size for each doublet.
#[allow(clippy::too_many_arguments)]
pub fn simulate_doublets_scdbl(
    pairs: &[(usize, usize)],
    cells_to_keep: &[usize],
    selected_genes_library_sizes: &[usize],
    cluster_labels: &[usize],
    selected_genes: &[usize],
    f_path_cell: &str,
    target_size: f32,
    log_transform: bool,
    params: &ScDblSimParams,
    seed: usize,
) -> (Vec<CsrCellChunk>, Vec<usize>) {
    let n_sim = pairs.len();
    let n_hvg = selected_genes.len();
    let gene_to_hvg_idx: FxHashMap<usize, usize> = selected_genes
        .iter()
        .enumerate()
        .map(|(pos, &orig)| (orig, pos))
        .collect();

    let cluster_medians =
        compute_cluster_median_lib_sizes(cluster_labels, selected_genes_library_sizes);

    // Pre-roll treatments deterministically
    let mut master_rng = StdRng::seed_from_u64(seed as u64 + 0xDEAD);
    let treatments: Vec<DoubletTreatment> = (0..n_sim)
        .map(|_| DoubletTreatment {
            adjust_size: master_rng.random::<f32>() < params.adjust_size_frac,
            half_size: master_rng.random::<f32>() < params.half_size_frac,
            resamp: master_rng.random::<f32>() < params.resamp_frac,
            rng_seed: master_rng.next_u64(),
        })
        .collect();

    let reader = ParallelSparseReader::new(f_path_cell).unwrap();

    let results: Vec<(CsrCellChunk, usize)> = (0..n_sim)
        .into_par_iter()
        .map(|di| {
            let (pos_a, pos_b) = pairs[di];
            let cell_a = reader.read_cell(cells_to_keep[pos_a]);
            let cell_b = reader.read_cell(cells_to_keep[pos_b]);

            // Extract dense HVG count vectors
            let mut dense_a = vec![0u32; n_hvg];
            let mut dense_b = vec![0u32; n_hvg];

            for (i, &gene_idx) in cell_a.indices.iter().enumerate() {
                let gi = gene_idx as usize;
                if let Some(&hvg_pos) = gene_to_hvg_idx.get(&gi) {
                    dense_a[hvg_pos] = cell_a.data_raw.get(i);
                }
            }
            for (i, &gene_idx) in cell_b.indices.iter().enumerate() {
                let gi = gene_idx as usize;
                if let Some(&hvg_pos) = gene_to_hvg_idx.get(&gi) {
                    dense_b[hvg_pos] = cell_b.data_raw.get(i);
                }
            }

            let (counts, effective_lib) = generate_single_doublet(
                &dense_a,
                &dense_b,
                selected_genes_library_sizes[pos_a],
                selected_genes_library_sizes[pos_b],
                cluster_labels[pos_a],
                cluster_labels[pos_b],
                &cluster_medians,
                &treatments[di],
            );

            // Pack sparse: collect non-zero entries
            let lib_f32 = (effective_lib as f32).max(1.0);
            let mut sp_indices: Vec<u32> = Vec::new();
            let mut sp_raw: Vec<u32> = Vec::new();
            let mut sp_norm: Vec<f32> = Vec::new();

            for (g, &c) in counts.iter().enumerate() {
                if c > 0 {
                    sp_indices.push(g as u32);
                    sp_raw.push(c);
                    let normed = if log_transform {
                        ((c as f32 / lib_f32) * target_size).ln_1p()
                    } else {
                        (c as f32 / lib_f32) * target_size
                    };
                    sp_norm.push(normed);
                }
            }

            let chunk = CsrCellChunk::from_doublet_simulation(
                sp_indices,
                sp_raw,
                sp_norm,
                effective_lib,
                di,
            );

            (chunk, effective_lib)
        })
        .collect();

    let mut chunks = Vec::with_capacity(n_sim);
    let mut lib_sizes = Vec::with_capacity(n_sim);
    for (chunk, lib) in results {
        chunks.push(chunk);
        lib_sizes.push(lib);
    }

    (chunks, lib_sizes)
}

/// Parameters for the unrecognisable-origin filter.
#[derive(Clone, Debug)]
pub struct UnrecognisableFilterParams {
    /// Minimum sims per origin to evaluate. R default: 5.
    pub min_size: usize,
    /// Minimum separation between sim median and the worst-case parent/global
    /// median. R default: 0.1.
    pub min_med_diff: f32,
}

impl Default for UnrecognisableFilterParams {
    fn default() -> Self {
        Self {
            min_size: 5,
            min_med_diff: 0.1,
        }
    }
}

/// Canonicalise a parent cluster pair. Returns `None` for homotypic
/// pairs (which R skips via its `grepl("+", ...)` filter).
#[inline]
pub fn canonical_pair(a: usize, b: usize) -> Option<(usize, usize)> {
    if a == b {
        None
    } else {
        Some((a.min(b), a.max(b)))
    }
}

/// Linear-interpolation quantile matching R's `type = 7` default.
///
/// Input must be sorted ascending.
fn quantile_sorted(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Identify canonical origins whose simulated doublets are indistinguishable
/// from their parent clusters' real cells.
///
/// Flags an origin `(A, B)` if EITHER:
///   (a) sim 10th percentile < max(parent A 90th, parent B 90th)
///   (b) sim median - max(global median, parent A median,
///       parent B median) < min_med_diff
pub fn identify_unrecognisable_origins(
    obs_scores: &[f32],
    obs_cluster_labels: &[usize],
    sim_scores: &[f32],
    sim_parent_clusters: &[(usize, usize)],
    params: &UnrecognisableFilterParams,
) -> FxHashSet<(usize, usize)> {
    assert_eq!(obs_scores.len(), obs_cluster_labels.len());
    assert_eq!(sim_scores.len(), sim_parent_clusters.len());

    // Global median of observed scores
    let global_median = {
        let mut sorted: Vec<f32> = obs_scores.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        quantile_sorted(&sorted, 0.5)
    };

    // Per-cluster 50th and 90th percentiles of observed scores
    let mut per_cluster: FxHashMap<usize, Vec<f32>> = FxHashMap::default();
    for (i, &cl) in obs_cluster_labels.iter().enumerate() {
        per_cluster.entry(cl).or_default().push(obs_scores[i]);
    }
    let mut cluster_pcts: FxHashMap<usize, (f32, f32)> = FxHashMap::default();
    for (cl, mut scores) in per_cluster {
        scores.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        cluster_pcts.insert(
            cl,
            (quantile_sorted(&scores, 0.5), quantile_sorted(&scores, 0.9)),
        );
    }

    // Group sim scores by canonical heterotypic origin
    let mut sims_by_origin: FxHashMap<(usize, usize), Vec<f32>> = FxHashMap::default();
    for (si, &(ca, cb)) in sim_parent_clusters.iter().enumerate() {
        if let Some(canon) = canonical_pair(ca, cb) {
            sims_by_origin
                .entry(canon)
                .or_default()
                .push(sim_scores[si]);
        }
    }

    // Evaluate predicate per origin
    let mut flagged: FxHashSet<(usize, usize)> = FxHashSet::default();
    for (origin, mut scores) in sims_by_origin {
        if scores.len() < params.min_size {
            continue;
        }
        scores.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let sim_p10 = quantile_sorted(&scores, 0.1);
        let sim_p50 = quantile_sorted(&scores, 0.5);

        let fallback = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        let (p50_a, p90_a) = cluster_pcts.get(&origin.0).copied().unwrap_or(fallback);
        let (p50_b, p90_b) = cluster_pcts.get(&origin.1).copied().unwrap_or(fallback);

        let cond_a = sim_p10 < p90_a.max(p90_b);
        let max_reference = global_median.max(p50_a).max(p50_b);
        let cond_b = (sim_p50 - max_reference) < params.min_med_diff;

        if cond_a || cond_b {
            flagged.insert(origin);
        }
    }

    flagged
}

/// Build a per-sim exclusion mask from a set of flagged canonical
/// origins. Homotypic sims are always false (not a valid origin).
pub fn mark_sims_from_flagged_origins(
    sim_parent_clusters: &[(usize, usize)],
    flagged: &FxHashSet<(usize, usize)>,
) -> Vec<bool> {
    sim_parent_clusters
        .iter()
        .map(|&(ca, cb)| {
            canonical_pair(ca, cb)
                .map(|c| flagged.contains(&c))
                .unwrap_or(false)
        })
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    ////////////////////
    // Feat selection //
    ////////////////////

    #[test]
    fn test_top_n_ordered_preserves_rank_order() {
        let values = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        // Top 3 in rank order: 5.0 (idx 1), 4.0 (idx 4), 3.0 (idx 2)
        assert_eq!(top_n_indices_ordered(&values, 3), vec![1, 4, 2]);
    }

    #[test]
    fn test_top_n_indices_sorts_by_index() {
        let values = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(top_n_indices(&values, 3), vec![1, 2, 4]);
    }

    #[test]
    fn test_top_n_ordered_ties_prefer_lower_index() {
        let values = vec![5.0, 5.0, 5.0, 1.0];
        assert_eq!(top_n_indices_ordered(&values, 2), vec![0, 1]);
    }

    #[test]
    fn test_roundrobin_basic() {
        // 3 clusters, distinct top picks
        let rankings = vec![vec![10, 20, 30], vec![11, 21, 31], vec![12, 22, 32]];
        // Rank 0 round: 10, 11, 12
        // Rank 1 round: 20, 21, 22
        // Rank 2 round: 30, 31, 32
        let result = roundrobin_select(&rankings, 5);
        assert_eq!(result, vec![10, 11, 12, 20, 21]);
    }

    #[test]
    fn test_roundrobin_dedupe() {
        // Overlapping rankings
        let rankings = vec![
            vec![1, 2, 3, 4],
            vec![1, 3, 5, 7], // 1 and 3 duplicate
            vec![2, 5, 8, 9], // 2 and 5 duplicate
        ];
        // Rank 0: 1 (new), 1 (dup), 2 (new)
        // Rank 1: 2 (dup), 3 (new), 5 (new)
        // Rank 2: 3 (dup), 5 (dup), 8 (new)
        let result = roundrobin_select(&rankings, 5);
        assert_eq!(result, vec![1, 2, 3, 5, 8]);
    }

    #[test]
    fn test_roundrobin_stops_at_n_top() {
        let rankings = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let result = roundrobin_select(&rankings, 3);
        assert_eq!(result, vec![1, 4, 2]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_roundrobin_small_per_cluster_lists() {
        // Some clusters have fewer ranks than others
        let rankings = vec![vec![1, 2], vec![3], vec![4, 5, 6]];
        // Rank 0: 1, 3, 4
        // Rank 1: 2, (skip), 5
        // Rank 2: (skip), (skip), 6
        let result = roundrobin_select(&rankings, 10);
        assert_eq!(result, vec![1, 3, 4, 2, 5, 6]);
    }

    #[test]
    fn test_roundrobin_insufficient_unique() {
        // All clusters have the same top picks; result is smaller than n_top
        let rankings = vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3]];
        let result = roundrobin_select(&rankings, 10);
        assert_eq!(result, vec![1, 2, 3]); // only 3 unique available
    }

    #[test]
    fn test_roundrobin_empty() {
        let rankings: Vec<Vec<usize>> = vec![];
        let result = roundrobin_select(&rankings, 5);
        assert_eq!(result, Vec::<usize>::new());
    }

    /////////////////
    // Doublet gen //
    /////////////////

    /// Verify Poisson sampler produces correct mean and variance
    /// for a range of lambda values.
    #[test]
    fn test_poisson_sampler_statistics() {
        let mut rng = StdRng::seed_from_u64(42);
        for &lambda in &[0.5, 2.0, 10.0, 50.0, 200.0, 1000.0] {
            let n = 50_000;
            let samples: Vec<u32> = (0..n).map(|_| poisson_sample(&mut rng, lambda)).collect();
            let mean = samples.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
            let var = samples
                .iter()
                .map(|&s| {
                    let d = s as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;

            // Poisson: E[X] = Var[X] = lambda
            let tol = 3.0 * (lambda / n as f64).sqrt(); // ~3 SE
            assert!(
                (mean - lambda).abs() < lambda * 0.05 + tol,
                "lambda={}: mean={:.2}, expected={:.2}",
                lambda,
                mean,
                lambda
            );
            assert!(
                (var - lambda).abs() < lambda * 0.1 + tol * lambda,
                "lambda={}: var={:.2}, expected={:.2}",
                lambda,
                var,
                lambda
            );
        }
    }

    #[test]
    fn test_poisson_sampler_zero() {
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..100 {
            assert_eq!(poisson_sample(&mut rng, 0.0), 0);
            assert_eq!(poisson_sample(&mut rng, -1.0), 0);
        }
    }

    #[test]
    fn test_cluster_median_lib_sizes() {
        let clusters = vec![0, 0, 0, 1, 1, 2];
        let lib_sizes = vec![100, 200, 300, 50, 150, 500];
        let medians = compute_cluster_median_lib_sizes(&clusters, &lib_sizes);

        assert!((medians[&0] - 200.0).abs() < 1e-6); // median of [100, 200, 300]
        assert!((medians[&1] - 100.0).abs() < 1e-6); // median of [50, 150]
        assert!((medians[&2] - 500.0).abs() < 1e-6); // single value
    }

    /// With all treatments disabled, output should be the exact sum
    /// of parent counts.
    #[test]
    fn test_no_treatment_is_simple_sum() {
        let parent_a = vec![0, 5, 10, 0, 3];
        let parent_b = vec![2, 0, 7, 1, 0];
        let cluster_medians = FxHashMap::default();
        let treatment = DoubletTreatment {
            adjust_size: false,
            half_size: false,
            resamp: false,
            rng_seed: 0,
        };

        let (counts, lib) = generate_single_doublet(
            &parent_a,
            &parent_b,
            18,
            10,
            0,
            1,
            &cluster_medians,
            &treatment,
        );

        assert_eq!(counts, vec![2, 5, 17, 1, 3]);
        assert_eq!(lib, 28);
    }

    /// Library size halving should produce ~0.5x the sum.
    #[test]
    fn test_half_size_halves_library() {
        let parent_a = vec![10, 20, 30, 40];
        let parent_b = vec![10, 20, 30, 40];
        let cluster_medians = FxHashMap::default();

        // Without halving
        let no_half = DoubletTreatment {
            adjust_size: false,
            half_size: false,
            resamp: false,
            rng_seed: 0,
        };
        let (_, lib_full) = generate_single_doublet(
            &parent_a,
            &parent_b,
            100,
            100,
            0,
            0,
            &cluster_medians,
            &no_half,
        );

        // With halving
        let with_half = DoubletTreatment {
            adjust_size: false,
            half_size: true,
            resamp: false,
            rng_seed: 0,
        };
        let (_, lib_half) = generate_single_doublet(
            &parent_a,
            &parent_b,
            100,
            100,
            0,
            0,
            &cluster_medians,
            &with_half,
        );

        assert_eq!(lib_full, 200);
        assert_eq!(lib_half, 100); // 200 / 2 = 100
    }

    /// Poisson resampling should add noise: repeated doublets from the same
    /// parents should NOT be identical.
    #[test]
    fn test_resamp_adds_noise() {
        let parent_a = vec![10, 20, 30, 40, 50];
        let parent_b = vec![5, 15, 25, 35, 45];
        let cluster_medians = FxHashMap::default();

        let mut all_equal = true;
        let mut first: Option<Vec<u32>> = None;

        for seed in 0..20u64 {
            let treatment = DoubletTreatment {
                adjust_size: false,
                half_size: false,
                resamp: true,
                rng_seed: seed * 7919 + 42,
            };
            let (counts, _) = generate_single_doublet(
                &parent_a,
                &parent_b,
                135,
                125,
                0,
                0,
                &cluster_medians,
                &treatment,
            );
            if let Some(ref f) = first {
                if &counts != f {
                    all_equal = false;
                    break;
                }
            } else {
                first = Some(counts);
            }
        }

        assert!(
            !all_equal,
            "Poisson resampling should produce different counts across seeds"
        );
    }

    /// Poisson resampling should preserve mean counts (law of large
    /// numbers).
    #[test]
    fn test_resamp_preserves_mean() {
        let parent_a = vec![0, 10, 50, 100];
        let parent_b = vec![5, 10, 50, 100];
        let expected_sum = [5, 20, 100, 200];
        let cluster_medians = FxHashMap::default();
        let n_reps = 5000;

        let mut accum = [0.0f64; 4];
        for rep in 0..n_reps {
            let treatment = DoubletTreatment {
                adjust_size: false,
                half_size: false,
                resamp: true,
                rng_seed: rep as u64 * 31 + 7,
            };
            let (counts, _) = generate_single_doublet(
                &parent_a,
                &parent_b,
                160,
                165,
                0,
                0,
                &cluster_medians,
                &treatment,
            );
            for (i, &c) in counts.iter().enumerate() {
                accum[i] += c as f64;
            }
        }

        for i in 0..4 {
            let mean = accum[i] / n_reps as f64;
            let expected = expected_sum[i] as f64;
            let tol = 3.0 * (expected / n_reps as f64).sqrt().max(0.5);
            assert!(
                (mean - expected).abs() < expected * 0.05 + tol,
                "gene {}: mean={:.2}, expected={:.2}",
                i,
                mean,
                expected
            );
        }
    }

    /// Size adjustment should preserve total library size while
    /// changing per-gene contributions.
    #[test]
    fn test_adjust_size_preserves_total_library() {
        let parent_a = vec![10, 20, 30, 40];
        let parent_b = vec![40, 30, 20, 10];
        let lib_a = 100;
        let lib_b = 100;

        let mut cluster_medians = FxHashMap::default();
        // Cluster 0 has 2x the median lib size of cluster 1
        cluster_medians.insert(0, 200.0);
        cluster_medians.insert(1, 100.0);

        let treatment = DoubletTreatment {
            adjust_size: true,
            half_size: false,
            resamp: false,
            rng_seed: 0,
        };

        let (counts, lib) = generate_single_doublet(
            &parent_a,
            &parent_b,
            lib_a,
            lib_b,
            0,
            1,
            &cluster_medians,
            &treatment,
        );

        // Total should be preserved (lib_a + lib_b = 200)
        assert_eq!(lib, lib_a + lib_b);

        // With asymmetric cluster medians, gene contributions should
        // differ from the simple sum
        let simple_sum: Vec<u32> = parent_a
            .iter()
            .zip(&parent_b)
            .map(|(&a, &b)| a + b)
            .collect();
        assert_ne!(
            counts, simple_sum,
            "adjusted counts should differ from simple sum"
        );
    }

    /// Size adjustment mixing factor should be clamped to [0.2, 0.8].
    #[test]
    fn test_adjust_size_clamping() {
        // Extreme library size ratio, but enough counts in parent_a
        // that clamping to 0.2 produces visible contribution.
        let parent_a = vec![100, 0, 0, 0];
        let parent_b = vec![0, 0, 0, 10000];
        let mut cluster_medians = FxHashMap::default();
        cluster_medians.insert(0, 1.0);
        cluster_medians.insert(1, 10000.0);

        let treatment = DoubletTreatment {
            adjust_size: true,
            half_size: false,
            resamp: false,
            rng_seed: 0,
        };

        let (counts, lib) = generate_single_doublet(
            &parent_a,
            &parent_b,
            100,
            10000,
            0,
            1,
            &cluster_medians,
            &treatment,
        );

        // Total library size preserved
        assert_eq!(lib, 10100);

        // Without clamping, factor ≈ 0.005 and gene 0 would get ~5 counts.
        // With clamping to 0.2, gene 0 gets: 100 * 0.2 = 20, rescaled by
        // 10100 / (20 + 8000) ≈ 1.26, so ~25 counts.
        assert!(
            counts[0] > 10,
            "clamping should give parent_a at least 20% weight; gene 0 got {}",
            counts[0]
        );

        // Gene 3 should get the bulk
        assert!(
            counts[3] > counts[0],
            "gene 3 ({}) should still dominate gene 0 ({})",
            counts[3],
            counts[0]
        );
    }

    /// Over many doublets with noise, library size distribution
    /// should NOT be a single spike at exactly 2x median.
    #[test]
    fn test_library_size_distribution_has_variance() {
        let parent = vec![10, 20, 30, 40, 50]; // lib = 150
        let cluster_medians = FxHashMap::default();
        let params = ScDblSimParams::default();
        let n = 1000;

        let mut master_rng = StdRng::seed_from_u64(123);
        let treatments: Vec<DoubletTreatment> = (0..n)
            .map(|_| DoubletTreatment {
                adjust_size: master_rng.random::<f32>() < params.adjust_size_frac,
                half_size: master_rng.random::<f32>() < params.half_size_frac,
                resamp: master_rng.random::<f32>() < params.resamp_frac,
                rng_seed: master_rng.next_u64(),
            })
            .collect();

        let lib_sizes: Vec<usize> = treatments
            .iter()
            .map(|t| {
                let (_, lib) =
                    generate_single_doublet(&parent, &parent, 150, 150, 0, 0, &cluster_medians, t);
                lib
            })
            .collect();

        let mean = lib_sizes.iter().sum::<usize>() as f64 / n as f64;
        let var = lib_sizes
            .iter()
            .map(|&l| {
                let d = l as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        // Without noise, all lib sizes would be exactly 300.
        // With noise (halving + resampling), we expect substantial
        // variance.
        let n_at_300 = lib_sizes.iter().filter(|&&l| l == 300).count();
        assert!(
            n_at_300 < n * 9 / 10,
            "expected fewer than 90% at exactly 2x; got {}/{} at 300",
            n_at_300,
            n
        );
        assert!(
            var > 100.0,
            "expected substantial variance in lib sizes; got var={:.1}",
            var
        );
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let parent_a = vec![5, 10, 15, 20];
        let parent_b = vec![8, 12, 3, 25];
        let cluster_medians = FxHashMap::default();

        let treatment = DoubletTreatment {
            adjust_size: false,
            half_size: true,
            resamp: true,
            rng_seed: 42,
        };

        let (c1, l1) = generate_single_doublet(
            &parent_a,
            &parent_b,
            50,
            48,
            0,
            1,
            &cluster_medians,
            &treatment,
        );
        let (c2, l2) = generate_single_doublet(
            &parent_a,
            &parent_b,
            50,
            48,
            0,
            1,
            &cluster_medians,
            &treatment,
        );

        assert_eq!(c1, c2);
        assert_eq!(l1, l2);
    }

    /////////////////////////////
    // Unrecognisable doublets //
    /////////////////////////////

    #[test]
    fn test_quantile_basic() {
        let s = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        assert!((quantile_sorted(&s, 0.0) - 0.1).abs() < 1e-6);
        assert!((quantile_sorted(&s, 0.5) - 0.3).abs() < 1e-6);
        assert!((quantile_sorted(&s, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_quantile_interpolation() {
        let s = vec![0.0, 10.0];
        assert!((quantile_sorted(&s, 0.5) - 5.0).abs() < 1e-6);
        assert!((quantile_sorted(&s, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantile_empty_and_single() {
        assert!(quantile_sorted(&[], 0.5).is_nan());
        assert_eq!(quantile_sorted(&[3.15], 0.5), 3.15);
    }

    #[test]
    fn test_canonical_pair_ordering() {
        assert_eq!(canonical_pair(2, 5), Some((2, 5)));
        assert_eq!(canonical_pair(5, 2), Some((2, 5)));
        assert_eq!(canonical_pair(3, 3), None);
    }

    #[test]
    fn test_too_few_sims_not_flagged() {
        // Only 4 sims for origin (0,1) -- below min_size=5
        let obs = vec![0.1; 10];
        let clust: Vec<usize> = (0..10).map(|i| i % 2).collect();
        let sim = vec![0.1, 0.15, 0.2, 0.25];
        let sp = vec![(0, 1); 4];
        let flagged = identify_unrecognisable_origins(&obs, &clust, &sim, &sp, &Default::default());
        assert!(flagged.is_empty());
    }

    #[test]
    fn test_clear_separation_not_flagged() {
        // Obs at 0.1, sim at 0.9+
        let obs = vec![0.1; 20];
        let clust: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let sim = vec![0.9, 0.92, 0.95, 0.97, 0.99, 0.88, 0.91, 0.94];
        let sp = vec![(0, 1); 8];
        let flagged = identify_unrecognisable_origins(&obs, &clust, &sim, &sp, &Default::default());
        assert!(flagged.is_empty());
    }

    #[test]
    fn test_cond_a_triggers_via_any_parent() {
        // Cluster 0 has noisy real cells (90th pct = 0.96)
        // Cluster 1 is clean
        // Sim (0,1) 10th pct ~0.57 < 0.96 -> cond_a triggers via cluster 0
        let obs = vec![
            0.6, 0.7, 0.8, 0.9, 1.0, // cluster 0 high
            0.1, 0.1, 0.1, 0.1, 0.1, // cluster 1 low
        ];
        let clust = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let sim = vec![0.5, 0.6, 0.65, 0.7, 0.75, 0.6, 0.65, 0.7];
        let sp = vec![(0, 1); 8];
        let flagged = identify_unrecognisable_origins(&obs, &clust, &sim, &sp, &Default::default());
        assert!(flagged.contains(&(0, 1)));
    }

    #[test]
    fn test_cond_b_triggers_on_median_proximity() {
        // Designed so cond_a does NOT trigger (sim 10th > max parent 90th)
        // but sim median is within 0.1 of worst parent median.
        // Cluster 0: all 0.1 -> p50=0.1, p90=0.1
        // Cluster 1: all 0.15 -> p50=0.15, p90=0.15
        // Sim 10th ~ 0.207 > 0.15 (cond_a false)
        // Sim 50th ~ 0.235, max ref = 0.15, diff 0.085 < 0.1 (cond_b true)
        let obs = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15];
        let clust = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let sim = vec![0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27];
        let sp = vec![(0, 1); 8];
        let flagged = identify_unrecognisable_origins(&obs, &clust, &sim, &sp, &Default::default());
        assert!(flagged.contains(&(0, 1)));
    }

    #[test]
    fn test_canonical_combines_reversed_pairs() {
        // Alternating (0,1) and (1,0) should combine into one origin
        // of 8 sims, triggering cond_b.
        let obs = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15];
        let clust = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let sim = vec![0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27];
        let sp = vec![
            (0, 1),
            (1, 0),
            (0, 1),
            (1, 0),
            (0, 1),
            (1, 0),
            (0, 1),
            (1, 0),
        ];
        let flagged = identify_unrecognisable_origins(&obs, &clust, &sim, &sp, &Default::default());
        assert!(flagged.contains(&(0, 1)));
    }

    #[test]
    fn test_homotypic_ignored() {
        // All sims have homotypic origin (0,0) -- skipped entirely
        let obs = vec![0.1; 10];
        let clust = vec![0; 10];
        let sim = vec![0.1; 8];
        let sp = vec![(0, 0); 8];
        let flagged = identify_unrecognisable_origins(&obs, &clust, &sim, &sp, &Default::default());
        assert!(flagged.is_empty());
    }

    #[test]
    fn test_mixed_flagged_and_kept() {
        // (0,1) clearly separated (kept)
        // (2,3) overlaps cluster 2's high-scoring cells (flagged)
        let obs = vec![
            0.1, 0.1, 0.1, 0.1, 0.1, // cluster 0
            0.1, 0.1, 0.1, 0.1, 0.1, // cluster 1
            0.6, 0.7, 0.8, 0.9, 1.0, // cluster 2 (high)
            0.1, 0.1, 0.1, 0.1, 0.1, // cluster 3
        ];
        let clust = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3];
        let sim = vec![
            0.90, 0.92, 0.95, 0.97, 0.99, 0.91, 0.93, 0.96, 0.50, 0.60, 0.65, 0.70, 0.75, 0.60,
            0.65, 0.70,
        ];
        let sp: Vec<(usize, usize)> = std::iter::repeat_n((0, 1), 8)
            .chain(std::iter::repeat_n((2, 3), 8))
            .collect();
        let flagged = identify_unrecognisable_origins(&obs, &clust, &sim, &sp, &Default::default());
        assert!(flagged.contains(&(2, 3)));
        assert!(!flagged.contains(&(0, 1)));
    }

    #[test]
    fn test_mark_sims_from_flagged_origins_canonical() {
        // Flagged set contains (0,1); sims with either (0,1) or (1,0)
        // should both be marked. (2,3) is not flagged. Homotypic (5,5) false.
        let sp = vec![(0, 1), (2, 3), (1, 0), (3, 2), (5, 5)];
        let mut flagged = FxHashSet::default();
        flagged.insert((0, 1));
        let mask = mark_sims_from_flagged_origins(&sp, &flagged);
        assert_eq!(mask, vec![true, false, true, false, false]);
    }

    #[test]
    fn test_empty_inputs() {
        let flagged = identify_unrecognisable_origins(&[], &[], &[], &[], &Default::default());
        assert!(flagged.is_empty());
    }
}
