//! Seurat CCA anchor-based batch correction.
//!
//! Ports the CCA integration path from Seurat v3 (Stuart et al., Cell, 2019).
//! Anchors are found in a per-pair L2-normalised canonical-correlation
//! embedding computed via a matrix-free randomised SVD of `X1^T @ X2`, so
//! the `N1 x N2` dense cross-product never materialises. Downstream anchor
//! filtering, scoring, weighting and correction all live in
//! [super::seurat_anchors].

use faer::{Mat, MatRef};
use indexmap::IndexSet;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;
use thousands::Separable;

use crate::core::math::pca_svd::randomised_svd_matfree;
use crate::prelude::*;
use crate::single_cell::sc_batch_correction::batch_utils::{
    batch_knn_search, cosine_normalise, standardise_per_column,
};
use crate::single_cell::sc_batch_correction::fast_mnn::{reorder_to_original, split_pca_by_batch};
use crate::single_cell::sc_batch_correction::seurat_anchors::{
    AnchorSet, build_sample_tree, filter_anchor_pairs, find_anchor_pairs, score_anchors,
    tree_merge_embeddings,
};
use crate::single_cell::sc_processing::pca::{
    SingleCellPcaParams, pca_on_sc, pca_on_sc_sparse, scale_csc_chunk,
};

////////////
// Params //
////////////

/// Parameters for Seurat CCA integration.
///
/// ### Fields
///
/// * `num_cc` - Number of canonical correlation dimensions to compute
/// * `dims` - CC dimensions used for anchor kNN queries (`<= num_cc`)
/// * `k_anchor` - Neighbours per cell for MNN anchor detection
/// * `k_filter` - Neighbours per cell in the gene-space filter
/// * `k_score` - Neighbours per cell in shared-neighbour scoring
/// * `k_weight` - Neighbours per query cell for kernel weighting
/// * `n_top_features` - Top-loading genes for [filter_anchor_pairs]
/// * `l2_norm` - Whether to L2-normalise the CC embedding
/// * `sd` - Kernel bandwidth divisor for weighting (Seurat default 1.0)
/// * `knn_params` - Backend for all kNN queries
/// * `pca_params` - Base PCA params (used when no pre-computed PCA is passed)
#[derive(Clone, Debug)]
pub struct SeuratCcaParams {
    /// Number of canonical correlation dimensions
    pub num_cc: usize,
    /// CC dimensions used for anchor kNN
    pub dims: usize,
    /// Anchor kNN size
    pub k_anchor: usize,
    /// Gene-space filter kNN size
    pub k_filter: usize,
    /// Shared-neighbour scoring kNN size
    pub k_score: usize,
    /// Weight kNN size
    pub k_weight: usize,
    /// Number of top-loading genes for the gene-space filter
    pub n_top_features: usize,
    /// L2-normalise the CC embedding
    pub l2_norm: bool,
    /// Gaussian kernel bandwidth divisor
    pub sd: f32,
    /// kNN backend parameters
    pub knn_params: KnnParams,
    /// Base PCA parameters
    pub pca_params: SingleCellPcaParams,
}

impl Default for SeuratCcaParams {
    fn default() -> Self {
        Self {
            num_cc: 30,
            dims: 30,
            k_anchor: 5,
            k_filter: 200,
            k_score: 30,
            k_weight: 100,
            n_top_features: 200,
            l2_norm: true,
            sd: 1.0,
            knn_params: KnnParams::default(),
            pca_params: SingleCellPcaParams::default(),
        }
    }
}

/////////////
// Loaders //
/////////////

/// Load log-normalised HVG expression for a set of cells and standardise
/// each cell (column) to mean 0, sd 1 across features. Rows are HVGs in
/// the order of `hvg_indices`; columns follow the order of
/// `batch_cell_indices`.
///
/// ### Params
///
/// * `reader` - Streaming reader for the gene-major binary
/// * `batch_cell_indices` - Absolute cell indices in the source file
/// * `hvg_indices` - Absolute HVG indices in the source file
/// * `clr_offsets` - Per-cell CLR offsets aligned to `batch_cell_indices`.
///   Required when `use_clr` is true.
/// * `use_clr` - Apply the shifted CLR transformation before standardising
/// * `size_factor` - Size factor used during original log-normalisation
///
/// ### Returns
///
/// Dense `Mat<f32>` of shape `(n_hvg, n_cells)`, per-column standardised.
pub(crate) fn load_hvg_standardised(
    reader: &ParallelSparseReader,
    batch_cell_indices: &[usize],
    hvg_indices: &[usize],
    clr_offsets: Option<&[f64]>,
    use_clr: bool,
    size_factor: f32,
) -> Result<Mat<f32>, BixverseErrors> {
    let cell_set: IndexSet<u32> = batch_cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = batch_cell_indices.len();
    let n_hvg = hvg_indices.len();

    let mut chunks = reader.read_gene_parallel_filtered(hvg_indices, &cell_set)?;
    if use_clr {
        chunks
            .par_iter_mut()
            .for_each(|chunk| chunk.transform_to_clr(size_factor));
    }

    let rows: Vec<Vec<f32>> = chunks
        .par_iter()
        .map(|chunk| {
            let (dense, _, _) = scale_csc_chunk(chunk, n_cells, false, false, clr_offsets);
            dense
        })
        .collect();

    let mat = Mat::from_fn(n_hvg, n_cells, |g, c| rows[g][c]);
    Ok(standardise_per_column(mat.as_ref()))
}

/// Load L2-normalised (per cell) expression on a fixed feature subset for
/// the gene-space anchor filter. Rows are cells, columns are the filter
/// features. Each row is normalised to unit L2.
///
/// ### Params
///
/// * `reader` - Streaming reader
/// * `batch_cell_indices` - Absolute cell indices for this batch
/// * `feature_indices` - Absolute gene indices to fetch (top-loading genes)
/// * `clr_offsets` - Optional per-cell CLR offsets aligned to
///   `batch_cell_indices`
/// * `use_clr` - Apply shifted CLR
/// * `size_factor` - Size factor for CLR
///
/// ### Returns
///
/// `(n_cells, n_features)` dense matrix, row-L2-normalised.
fn load_filter_expression(
    reader: &ParallelSparseReader,
    batch_cell_indices: &[usize],
    feature_indices: &[usize],
    clr_offsets: Option<&[f64]>,
    use_clr: bool,
    size_factor: f32,
) -> Result<Mat<f32>, BixverseErrors> {
    let cell_set: IndexSet<u32> = batch_cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = batch_cell_indices.len();
    let n_feat = feature_indices.len();

    let mut chunks = reader.read_gene_parallel_filtered(feature_indices, &cell_set)?;
    if use_clr {
        chunks
            .par_iter_mut()
            .for_each(|chunk| chunk.transform_to_clr(size_factor));
    }

    let rows: Vec<Vec<f32>> = chunks
        .par_iter()
        .map(|chunk| {
            let (dense, _, _) = scale_csc_chunk(chunk, n_cells, false, false, clr_offsets);
            dense
        })
        .collect();

    let mat = Mat::from_fn(n_cells, n_feat, |c, g| rows[g][c]);
    Ok(cosine_normalise(&mat))
}

//////////////////////
// CCA anchor space //
//////////////////////

/// Build a shared canonical-correlation embedding for two standardised
/// batches via matrix-free randomised SVD of `M = X1^T @ X2`.
///
/// `M` is never materialised: [randomised_svd_matfree] takes closures that
/// factor the matvecs through `X1` and `X2`. Memory stays
/// `O((n1 + n2) * (num_cc + oversampling))` for the sketch, plus the two
/// standardised inputs themselves.
///
/// ### Params
///
/// * `x1` - Standardised batch A, shape `(n_hvg, n_a)`
/// * `x2` - Standardised batch B, shape `(n_hvg, n_b)`
/// * `num_cc` - Number of canonical dimensions to keep
/// * `l2_norm` - Whether to L2-normalise each row of the concatenated
///   `[U; V]` embedding (Seurat default true)
/// * `seed` - Random seed for the sketch
///
/// ### Returns
///
/// `(cc_a, cc_b)` where `cc_a` is `(n_a, num_cc)` and `cc_b` is
/// `(n_b, num_cc)`, both in the same shared space.
fn build_cca_anchor_space(
    x1: MatRef<f32>,
    x2: MatRef<f32>,
    num_cc: usize,
    l2_norm: bool,
    seed: usize,
) -> Result<(Mat<f32>, Mat<f32>), BixverseErrors> {
    assert_eq!(x1.nrows(), x2.nrows(), "batch feature dimension mismatch");
    let n_a = x1.ncols();
    let n_b = x2.ncols();

    let apply = |v: MatRef<f32>| -> Mat<f32> {
        // M @ v = X1^T @ (X2 @ v); v is (n_b, k)
        let tmp = x2 * v; // (n_hvg, k)
        x1.transpose() * tmp.as_ref() // (n_a, k)
    };
    let apply_t = |u: MatRef<f32>| -> Mat<f32> {
        // M^T @ u = X2^T @ (X1 @ u); u is (n_a, k)
        let tmp = x1 * u; // (n_hvg, k)
        x2.transpose() * tmp.as_ref() // (n_b, k)
    };

    let svd = randomised_svd_matfree(n_a, n_b, num_cc, seed, None, None, apply, apply_t)?;

    // Truncate to num_cc columns (matfree may return more due to oversampling).
    let keep = num_cc.min(svd.s.len()).min(svd.u.ncols()).min(svd.v.ncols());
    let mut cc_a = Mat::from_fn(n_a, keep, |i, j| svd.u[(i, j)]);
    let mut cc_b = Mat::from_fn(n_b, keep, |i, j| svd.v[(i, j)]);

    // Sign convention: force the first non-zero entry of each column of the
    // concatenated embedding to be positive (matches Seurat's flip).
    for k in 0..keep {
        let mut pivot = 0.0_f32;
        for i in 0..n_a {
            if cc_a[(i, k)].abs() > 1e-12 {
                pivot = cc_a[(i, k)];
                break;
            }
        }
        if pivot == 0.0 {
            for i in 0..n_b {
                if cc_b[(i, k)].abs() > 1e-12 {
                    pivot = cc_b[(i, k)];
                    break;
                }
            }
        }
        if pivot < 0.0 {
            for i in 0..n_a {
                cc_a[(i, k)] = -cc_a[(i, k)];
            }
            for i in 0..n_b {
                cc_b[(i, k)] = -cc_b[(i, k)];
            }
        }
    }

    if l2_norm {
        cc_a = cosine_normalise(&cc_a);
        cc_b = cosine_normalise(&cc_b);
    }

    Ok((cc_a, cc_b))
}

/// Pick the top-`n_top` genes by maximum absolute PCA loading across the
/// first `dims` components. Used to prime [load_filter_expression] for the
/// gene-space anchor filter. Falls back to the number of available genes
/// when `n_top` exceeds them.
///
/// ### Params
///
/// * `loadings` - PCA loadings, `(n_genes, n_pcs)`
/// * `dims` - Number of PCs to consider
/// * `n_top` - Number of features to select
///
/// ### Returns
///
/// Sorted absolute indices into the loading rows, length
/// `min(n_top, n_genes)`.
fn top_loading_gene_positions(loadings: MatRef<f32>, dims: usize, n_top: usize) -> Vec<usize> {
    let n_genes = loadings.nrows();
    let k = dims.min(loadings.ncols());
    let mut scored: Vec<(usize, f32)> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let mut m = 0.0_f32;
            for j in 0..k {
                let v = loadings[(g, j)].abs();
                if v > m {
                    m = v;
                }
            }
            (g, m)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut kept: Vec<usize> = scored.into_iter().take(n_top).map(|(g, _)| g).collect();
    kept.sort_unstable();
    kept
}

///////////////////////////
// Per-pair anchor finder //
///////////////////////////

/// Anchor pairs plus their per-pair scores, one score per pair.
type AnchorPairsScored = (Vec<(u32, u32)>, Vec<f32>);

/// Find and score anchor pairs for a single batch pair in the CCA path.
///
/// Runs the four Seurat kNN queries in the shared L2-normalised CC space
/// (`nnaa`, `nnbb`, `nnab`, `nnba` with `k = max(k_anchor, k_score)`),
/// extracts MNN pairs via [find_anchor_pairs], optionally filters them
/// in gene-expression space via [filter_anchor_pairs], then scores the
/// survivors via [score_anchors] on the merged neighbourhood.
///
/// ### Params
///
/// * `cc_a` - Batch-A rows of the CC embedding, `(n_a, dims)`
/// * `cc_b` - Batch-B rows of the CC embedding, `(n_b, dims)`
/// * `filter_a` - Optional L2-normalised top-loading-gene expression for
///   batch A, `(n_a, n_top_features)`. When `Some`, `FilterAnchors` runs.
/// * `filter_b` - Same for batch B
/// * `params` - Method parameters
/// * `knn_params` - kNN backend parameters
/// * `seed` - Random seed for the kNN backend
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `(pairs, scores)` where `pairs[i] = (a, b)` are indices into batch A
/// / batch B and `scores[i]` is the rescaled shared-neighbour score.
/// Both are empty when no MNN survives.
#[allow(clippy::too_many_arguments)]
fn find_cca_anchors_for_pair(
    cc_a: MatRef<f32>,
    cc_b: MatRef<f32>,
    filter_a: Option<MatRef<f32>>,
    filter_b: Option<MatRef<f32>>,
    params: &SeuratCcaParams,
    knn_params: &KnnParams,
    seed: usize,
    verbose: usize,
) -> Result<AnchorPairsScored, BixverseErrors> {
    let k_nn = params.k_anchor.max(params.k_score);
    let k_aa = k_nn.min(cc_a.nrows());
    let k_bb = k_nn.min(cc_b.nrows());
    let k_ab = k_nn.min(cc_b.nrows());
    let k_ba = k_nn.min(cc_a.nrows());

    let (nnaa, _) = batch_knn_search(cc_a, cc_a, k_aa, knn_params, seed, verbose)?;
    let (nnbb, _) = batch_knn_search(cc_b, cc_b, k_bb, knn_params, seed, verbose)?;
    let (nnab, _) = batch_knn_search(cc_a, cc_b, k_ab, knn_params, seed, verbose)?;
    let (nnba, _) = batch_knn_search(cc_b, cc_a, k_ba, knn_params, seed, verbose)?;

    let mut pairs = find_anchor_pairs(&nnab, &nnba, params.k_anchor);

    if let (Some(fa), Some(fb)) = (filter_a, filter_b) {
        let k_filter = params.k_filter.min(fb.nrows());
        if k_filter > 0 {
            let (nn_filter, _) = batch_knn_search(fa, fb, k_filter, knn_params, seed, verbose)?;
            pairs = filter_anchor_pairs(&pairs, &nn_filter, params.k_filter);
        }
    }

    if pairs.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let scores = score_anchors(
        &pairs,
        &nnaa,
        &nnab,
        &nnba,
        &nnbb,
        params.k_score,
        cc_a.nrows(),
    );

    Ok((pairs, scores))
}

///////////////////
// Main function //
///////////////////

/// Perform Seurat CCA batch integration on single-cell data.
///
/// Computes a shared CCA embedding for every batch pair, extracts and
/// scores anchors, then applies Seurat's kernel-weighted correction on
/// the union PCA embedding in tree-merge order derived from the pairwise
/// anchor counts. The corrected embedding is returned in the original
/// cell order.
///
/// This port skips Seurat's per-gene `ScaleData` step and works directly
/// from the per-cell standardised log-normalised HVG expression. In
/// practice this produces close-to-identical anchor structure at
/// significantly reduced memory: `M = X1^T @ X2` is never materialised
/// (matrix-free randomised SVD) and the correction runs on the embedding
/// (dims x cells), not on full log-expression.
///
/// ### Params
///
/// * `f_path` - Path to the gene-major streaming binary
/// * `cell_indices` - Absolute cell indices to include (the union)
/// * `gene_indices` - Absolute HVG indices used for the CCA anchor space
/// * `batch_indices` - Zero-based batch label per cell in `cell_indices`
/// * `pre_computed_pca` - Optional pre-computed base PCA scores; if
///   provided, `pca_params` and PCA-related I/O are skipped
/// * `clr_offsets` - Optional per-cell CLR offsets aligned to
///   `cell_indices`
/// * `params` - Method parameters, see [SeuratCcaParams]
/// * `verbose` - `0` silent, `1` normal, `2` detailed
/// * `seed` - Random seed for SVD sketches and kNN backends
///
/// ### Returns
///
/// Corrected PCA embedding, `(n_cells, no_pcs)`, in the original cell
/// order supplied via `cell_indices`.
///
/// ### References
///
/// Stuart et al., Cell, 2019.
#[allow(clippy::too_many_arguments)]
pub fn seurat_cca_integration(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    batch_indices: &[usize],
    pre_computed_pca: Option<Mat<f32>>,
    clr_offsets: Option<&[f64]>,
    params: &SeuratCcaParams,
    verbose: usize,
    seed: usize,
) -> Result<Mat<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start_total = Instant::now();

    if batch_indices.len() != cell_indices.len() {
        return Err(BixverseErrors::NumberLabelsNotEqualSampleNumber {
            label_length: batch_indices.len(),
            n_samples: cell_indices.len(),
        });
    }

    // n_batches = max_label + 1 to stay consistent with split_pca_by_batch.
    // Empty slots (non-contiguous labels) are caught below by the
    // per-batch minimum-cell check.
    let n_batches = batch_indices.iter().copied().max().unwrap_or(0) + 1;
    if n_batches < 2 {
        return Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches });
    }

    let min_cells = params.k_anchor.max(params.k_score) + 1;
    let mut per_batch_positions: Vec<Vec<usize>> = vec![Vec::new(); n_batches];
    for (row, &b) in batch_indices.iter().enumerate() {
        per_batch_positions[b].push(row);
    }
    for (b, positions) in per_batch_positions.iter().enumerate() {
        if positions.len() < min_cells {
            return Err(BixverseErrors::TooFewCellsForAnchor {
                batch: b,
                n_cells: positions.len(),
                required: min_cells,
            });
        }
    }

    // Base PCA (union). Loadings are needed for the top-feature filter.
    let (pca_scores, pca_loadings, _s) = if let Some(scores) = pre_computed_pca {
        if verbosity.normal_verbosity() {
            println!("Using pre-computed PCA; computing loadings for filter step");
        }
        // No loadings available from a pre-computed embedding; compute a
        // cheap sparse PCA on the same data to recover them.
        let (_pc, load, s) = pca_on_sc_sparse(
            f_path,
            cell_indices,
            gene_indices,
            params.dims.max(scores.ncols()),
            &params.pca_params,
            clr_offsets,
            seed,
            verbose,
        )?;
        (scores, load, s)
    } else {
        if verbosity.normal_verbosity() {
            println!("Computing base PCA");
        }
        if params.pca_params.clr && clr_offsets.is_none() {
            return Err(BixverseErrors::OffsetsNotProvidedForClrPCA);
        }
        if params.pca_params.randomised {
            let (sc, load, s, _) = pca_on_sc(
                f_path,
                cell_indices,
                gene_indices,
                params.dims,
                &params.pca_params,
                clr_offsets,
                seed,
                false,
                verbose,
            )?;
            (sc, load, s)
        } else {
            let (sc, load, s) = pca_on_sc_sparse(
                f_path,
                cell_indices,
                gene_indices,
                params.dims,
                &params.pca_params,
                clr_offsets,
                seed,
                verbose,
            )?;
            (sc, load, s)
        }
    };

    // Top-loading genes (positions into gene_indices) for the gene-space
    // filter. Absolute file gene indices are `gene_indices[pos]`.
    let top_positions = top_loading_gene_positions(
        pca_loadings.as_ref(),
        params.dims,
        params.n_top_features,
    );
    let top_absolute_genes: Vec<usize> = top_positions
        .iter()
        .map(|&pos| gene_indices[pos])
        .collect();

    let reader = ParallelSparseReader::new(f_path)?;

    // Per-pair anchor computation: load only the two batches involved,
    // build CC space and filter expression, extract anchors, drop.
    let mut anchors_by_pair: FxHashMap<(usize, usize), AnchorSet> = FxHashMap::default();
    let mut anchor_counts: Vec<Vec<u32>> = vec![vec![0_u32; n_batches]; n_batches];

    for a in 0..n_batches {
        for b in (a + 1)..n_batches {
            let pair_start = Instant::now();
            if verbosity.normal_verbosity() {
                println!("CCA: finding anchors for batch pair ({a}, {b})");
            }

            let cells_a: Vec<usize> = per_batch_positions[a]
                .iter()
                .map(|&row| cell_indices[row])
                .collect();
            let cells_b: Vec<usize> = per_batch_positions[b]
                .iter()
                .map(|&row| cell_indices[row])
                .collect();

            let clr_a = clr_offsets.map(|offs| -> Vec<f64> {
                per_batch_positions[a].iter().map(|&row| offs[row]).collect()
            });
            let clr_b = clr_offsets.map(|offs| -> Vec<f64> {
                per_batch_positions[b].iter().map(|&row| offs[row]).collect()
            });

            let x1 = load_hvg_standardised(
                &reader,
                &cells_a,
                gene_indices,
                clr_a.as_deref(),
                params.pca_params.clr,
                params.pca_params.size_factor,
            )?;
            let x2 = load_hvg_standardised(
                &reader,
                &cells_b,
                gene_indices,
                clr_b.as_deref(),
                params.pca_params.clr,
                params.pca_params.size_factor,
            )?;

            let (cc_a, cc_b) = build_cca_anchor_space(
                x1.as_ref(),
                x2.as_ref(),
                params.num_cc.max(params.dims),
                params.l2_norm,
                seed,
            )?;
            drop(x1);
            drop(x2);

            // Restrict CC embedding to the anchor-kNN dimensions.
            let dims_use = params.dims.min(cc_a.ncols());
            let cc_a_dim = Mat::from_fn(cc_a.nrows(), dims_use, |i, j| cc_a[(i, j)]);
            let cc_b_dim = Mat::from_fn(cc_b.nrows(), dims_use, |i, j| cc_b[(i, j)]);

            let filter_a = if params.k_filter > 0 && !top_absolute_genes.is_empty() {
                Some(load_filter_expression(
                    &reader,
                    &cells_a,
                    &top_absolute_genes,
                    clr_a.as_deref(),
                    params.pca_params.clr,
                    params.pca_params.size_factor,
                )?)
            } else {
                None
            };
            let filter_b = if params.k_filter > 0 && !top_absolute_genes.is_empty() {
                Some(load_filter_expression(
                    &reader,
                    &cells_b,
                    &top_absolute_genes,
                    clr_b.as_deref(),
                    params.pca_params.clr,
                    params.pca_params.size_factor,
                )?)
            } else {
                None
            };

            let (pairs, scores) = find_cca_anchors_for_pair(
                cc_a_dim.as_ref(),
                cc_b_dim.as_ref(),
                filter_a.as_ref().map(|m| m.as_ref()),
                filter_b.as_ref().map(|m| m.as_ref()),
                params,
                &params.knn_params,
                seed,
                verbose,
            )?;

            let count = pairs.len();
            if verbosity.normal_verbosity() {
                println!(
                    "  found {} anchors in {:.2?}",
                    count.separate_with_underscores(),
                    pair_start.elapsed()
                );
            }
            anchor_counts[a][b] = count as u32;
            anchor_counts[b][a] = count as u32;
            anchors_by_pair.insert(
                (a, b),
                AnchorSet {
                    pairs,
                    scores,
                    batch_pair: (a, b),
                },
            );
        }
    }

    let merge_order = build_sample_tree(&anchor_counts);

    let (per_batch_scores, per_batch_union_indices) =
        split_pca_by_batch(&pca_scores, batch_indices);

    let (merged, index_map) = tree_merge_embeddings(
        per_batch_scores,
        per_batch_union_indices,
        &anchors_by_pair,
        &merge_order,
        params.k_weight,
        params.sd,
        &params.knn_params,
        seed,
        verbose,
    )?;

    let corrected = reorder_to_original(&merged, &index_map);

    if verbosity.normal_verbosity() {
        println!(
            "CCA integration complete in {:.2?}",
            start_total.elapsed()
        );
    }

    Ok(corrected)
}
