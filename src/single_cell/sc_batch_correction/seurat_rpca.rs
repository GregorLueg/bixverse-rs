//! Seurat reciprocal-PCA (rPCA) anchor-based batch correction.
//!
//! Same anchor pipeline as [super::seurat_cca] but with a cheaper per-pair
//! anchor space: each batch keeps its own PCA basis, and the other batch's HVG
//! expression is projected into it. Cross-batch kNN queries then live in these
//! projected bases. Anchor filtering is CCA-only in Seurat and is skipped here.

use faer::{Mat, MatRef};
use rustc_hash::FxHashMap;
use std::time::Instant;
use thousands::Separable;

use crate::core::math::pca_svd::randomised_svd;
use crate::prelude::*;
use crate::single_cell::sc_batch_correction::batch_utils::{batch_knn_search, cosine_normalise};
use crate::single_cell::sc_batch_correction::fast_mnn::{reorder_to_original, split_pca_by_batch};
use crate::single_cell::sc_batch_correction::seurat_anchors::{
    AnchorSet, build_sample_tree, find_anchor_pairs, score_anchors, tree_merge_embeddings,
};
use crate::single_cell::sc_batch_correction::seurat_cca::load_hvg_standardised;
use crate::single_cell::sc_processing::pca::{SingleCellPcaParams, pca_on_sc, pca_on_sc_sparse};

///////////
// Types //
///////////

/// Anchor pairs plus their per-pair scores, one score per pair.
type AnchorPairsScored = (Vec<(u32, u32)>, Vec<f32>);

////////////
// Params //
////////////

/// Parameters for Seurat rPCA integration.
#[derive(Clone, Debug)]
pub struct SeuratRpcaParams {
    /// PC dimensions for anchor kNN queries
    pub dims: usize,
    /// Anchor kNN size
    pub k_anchor: usize,
    /// Shared-neighbour scoring kNN size
    pub k_score: usize,
    /// Weight kNN size
    pub k_weight: usize,
    /// L2-normalise projected embeddings per cell
    pub l2_norm: bool,
    /// Gaussian kernel bandwidth divisor
    pub sd: f32,
    /// kNN backend parameters
    pub knn_params: KnnParams,
    /// Base PCA parameters
    pub pca_params: SingleCellPcaParams,
}

impl Default for SeuratRpcaParams {
    fn default() -> Self {
        Self {
            dims: 30,
            k_anchor: 5,
            k_score: 30,
            k_weight: 100,
            l2_norm: true,
            sd: 1.0,
            knn_params: KnnParams::default(),
            pca_params: SingleCellPcaParams::default(),
        }
    }
}

///////////////////////
// rPCA anchor space //
///////////////////////

/// Compute a per-batch PCA directly on per-cell-standardised HVG expression so
/// both loadings and scores live in the same regime as the projected data used
/// by [project_into_basis].
///
/// The SVD is done via [randomised_svd] on the standardised `(n_hvg,
/// n_cells)` matrix directly. Left singular vectors `U` are the gene-space
/// loadings; scores are `V * diag(S)` in cell space.
///
/// ### Params
///
/// * `x_standardised` - Per-cell standardised HVG matrix, `(n_hvg,
///   n_cells)`
/// * `dims` - Number of principal components to retain
/// * `seed` - Random seed for the SVD sketch
///
/// ### Returns
///
/// `(scores, loadings)` where `scores` is `(n_cells, dims)` and
/// `loadings` is `(n_hvg, dims)`.
fn per_batch_pca_standardised(
    x_standardised: MatRef<f32>,
    dims: usize,
    seed: usize,
) -> Result<(Mat<f32>, Mat<f32>), BixverseErrors> {
    let n_hvg = x_standardised.nrows();
    let n_cells = x_standardised.ncols();
    let effective = dims.min(n_hvg).min(n_cells);
    let svd = randomised_svd(x_standardised, effective, seed, None, None)?;

    // Truncate to `effective` columns (randomised SVD may return more due
    // to oversampling).
    let keep = effective
        .min(svd.s.len())
        .min(svd.u.ncols())
        .min(svd.v.ncols());
    let loadings = Mat::from_fn(n_hvg, keep, |i, j| svd.u[(i, j)]);
    let scores = Mat::from_fn(n_cells, keep, |i, j| svd.v[(i, j)] * svd.s[j]);
    Ok((scores, loadings))
}

/// Cross-project batch B's standardised HVG expression into batch A's PCA
/// basis. Returns `(n_cells_b, dims)` and, if requested, L2-normalised per
/// row so it lives in the same cosine space as batch A's own PCA scores.
///
/// ### Params
///
/// * `loadings_a` - PCA loadings of batch A, shape `(n_hvg, dims)`
/// * `x_b` - Standardised HVG expression of batch B, shape `(n_hvg, n_b)`
/// * `l2_norm` - Whether to L2-normalise per row after projection
///
/// ### Returns
///
/// Dense `(n_b, dims)` projected embedding.
fn project_into_basis(loadings_a: MatRef<f32>, x_b: MatRef<f32>, l2_norm: bool) -> Mat<f32> {
    // proj = loadings_a^T @ x_b, shape (dims, n_b). Transpose to (n_b, dims).
    let proj = loadings_a.transpose() * x_b;
    let n_b = x_b.ncols();
    let dims = loadings_a.ncols();
    let projected = Mat::from_fn(n_b, dims, |c, d| proj[(d, c)]);
    if l2_norm {
        cosine_normalise(&projected)
    } else {
        projected
    }
}

/// Take a per-batch PCA scores matrix and optionally L2-normalise its rows so
/// the batch's own scores sit in the same cosine space as cross-projected
/// embeddings for anchor kNN.
///
/// ### Params
///
/// * `scores` - Per-batch PCA scores, `(n_cells, dims)`
/// * `l2_norm` - Whether to L2-normalise each row
///
/// ### Returns
///
/// `(n_cells, dims)` matrix, cloned regardless of `l2_norm` so the
/// caller can freely consume it.
fn normalise_own_scores(scores: &Mat<f32>, l2_norm: bool) -> Mat<f32> {
    if l2_norm {
        cosine_normalise(scores)
    } else {
        scores.clone()
    }
}

////////////////////////////
// Per-pair anchor finder //
////////////////////////////

/// Find and score anchor pairs for a single batch pair in the rPCA path.
///
/// Runs Seurat's four cross-basis kNN queries:
/// - `nnaa`: batch-A self kNN in A's own basis
/// - `nnbb`: batch-B self kNN in B's own basis
/// - `nnab`: batch-A queries projected into B's basis, find nearest
///   batch-B cells in B's basis
/// - `nnba`: batch-B queries projected into A's basis, find nearest
///   batch-A cells in A's basis
///
/// Extracts MNN pairs via [find_anchor_pairs] and scores them via
/// [score_anchors]. Unlike CCA, no gene-space filter step runs (matches
/// Seurat's `FilterAnchors` being CCA-only).
///
/// ### Params
///
/// * `a_in_a` - Batch A in its own PCA basis, `(n_a, dims)`
/// * `b_in_b` - Batch B in its own PCA basis, `(n_b, dims)`
/// * `a_in_b` - Batch A projected into batch-B basis, `(n_a, dims)`
/// * `b_in_a` - Batch B projected into batch-A basis, `(n_b, dims)`
/// * `params` - Method parameters
/// * `knn_params` - kNN backend parameters
/// * `seed` - Random seed for the kNN backend
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `(pairs, scores)`. Empty when no MNN survives.
#[allow(clippy::too_many_arguments)]
fn find_rpca_anchors_for_pair(
    a_in_a: MatRef<f32>,
    b_in_b: MatRef<f32>,
    a_in_b: MatRef<f32>,
    b_in_a: MatRef<f32>,
    params: &SeuratRpcaParams,
    knn_params: &KnnParams,
    seed: usize,
    verbose: usize,
) -> Result<AnchorPairsScored, BixverseErrors> {
    let k_nn = params.k_anchor.max(params.k_score);
    let k_aa = k_nn.min(a_in_a.nrows());
    let k_bb = k_nn.min(b_in_b.nrows());
    let k_ab = k_nn.min(b_in_b.nrows());
    let k_ba = k_nn.min(a_in_a.nrows());

    let (nnaa, _) = batch_knn_search(a_in_a, a_in_a, k_aa, knn_params, seed, verbose)?;
    let (nnbb, _) = batch_knn_search(b_in_b, b_in_b, k_bb, knn_params, seed, verbose)?;
    let (nnab, _) = batch_knn_search(a_in_b, b_in_b, k_ab, knn_params, seed, verbose)?;
    let (nnba, _) = batch_knn_search(b_in_a, a_in_a, k_ba, knn_params, seed, verbose)?;

    let pairs = find_anchor_pairs(&nnab, &nnba, params.k_anchor);
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
        a_in_a.nrows(),
    );
    Ok((pairs, scores))
}

///////////////////
// Main function //
///////////////////

/// Perform Seurat rPCA batch integration on single-cell data.
///
/// Computes a per-batch PCA on HVGs, cross-projects standardised HVG expression
/// of each batch into every other batch's basis, extracts anchors via MNN in
/// these projected spaces, scores them, then applies Seurat's kernel-weighted
/// correction on the union PCA embedding in tree-merge order derived from
/// pairwise anchor counts. The corrected embedding is returned in the original
/// cell order.
///
/// Unlike CCA, no anchor gene-space filter is applied (matches Seurat).
///
/// ### Params
///
/// * `reader` - Reader over the gene-major count store
/// * `cell_indices` - Absolute cell indices to include
/// * `gene_indices` - Absolute HVG indices used for PCA and projections
/// * `batch_indices` - Zero-based batch label per cell
/// * `pre_computed_pca` - Optional pre-computed base union PCA scores.
///   Per-batch PCAs are always re-computed here (rPCA needs them).
/// * `clr_offsets` - Optional per-cell CLR offsets aligned to
///   `cell_indices`
/// * `params` - Method parameters, see [SeuratRpcaParams]
/// * `verbose` - `0` silent, `1` normal, `2` detailed
/// * `seed` - Random seed
///
/// ### Returns
///
/// Corrected PCA embedding, `(n_cells, dims)`, in the original cell order.
///
/// ### References
///
/// Stuart et al., Cell, 2019.
#[allow(clippy::too_many_arguments)]
pub fn seurat_rpca_integration<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    batch_indices: &[usize],
    pre_computed_pca: Option<Mat<f32>>,
    clr_offsets: Option<&[f64]>,
    params: &SeuratRpcaParams,
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

    if params.pca_params.clr && clr_offsets.is_none() {
        return Err(BixverseErrors::OffsetsNotProvidedForClrPCA);
    }

    // Base union PCA — used only for the merge-space embedding.
    let pca_scores = if let Some(scores) = pre_computed_pca {
        if verbosity.normal_verbosity() {
            println!("Using pre-computed union PCA");
        }
        scores
    } else {
        if verbosity.normal_verbosity() {
            println!("Computing base union PCA");
        }
        if params.pca_params.randomised {
            let (sc, _load, _s, _) = pca_on_sc(
                reader,
                cell_indices,
                gene_indices,
                params.dims,
                &params.pca_params,
                clr_offsets,
                seed,
                false,
                verbose,
            )?;
            sc
        } else {
            let (sc, _load, _s) = pca_on_sc_sparse(
                reader,
                cell_indices,
                gene_indices,
                params.dims,
                &params.pca_params,
                clr_offsets,
                seed,
                verbose,
            )?;
            sc
        }
    };

    // Cache per-batch standardised HVG expression: loaded once, reused
    // for per-batch PCA and every cross-projection.
    //
    // TODO: maybe worth thinking about off-loading data to disk
    // temporarily ... ?
    if verbosity.normal_verbosity() {
        println!("Loading per-batch standardised HVG expression");
    }
    let mut per_batch_standardised: Vec<Mat<f32>> = Vec::with_capacity(n_batches);
    for b in 0..n_batches {
        let cells_b: Vec<usize> = per_batch_positions[b]
            .iter()
            .map(|&row| cell_indices[row])
            .collect();
        let clr_b: Option<Vec<f64>> = clr_offsets.map(|offs| {
            per_batch_positions[b]
                .iter()
                .map(|&row| offs[row])
                .collect()
        });
        let x = load_hvg_standardised(
            reader,
            &cells_b,
            gene_indices,
            clr_b.as_deref(),
            params.pca_params.clr,
            params.pca_params.size_factor,
        )?;
        per_batch_standardised.push(x);
    }

    if verbosity.normal_verbosity() {
        println!("Computing per-batch PCAs ({n_batches} batches)");
    }
    let mut per_batch_pca_scores: Vec<Mat<f32>> = Vec::with_capacity(n_batches);
    let mut per_batch_loadings: Vec<Mat<f32>> = Vec::with_capacity(n_batches);
    for b in 0..n_batches {
        let (sc, load) =
            per_batch_pca_standardised(per_batch_standardised[b].as_ref(), params.dims, seed)?;
        per_batch_pca_scores.push(sc);
        per_batch_loadings.push(load);
    }

    // Per-pair anchor finding.
    let mut anchors_by_pair: FxHashMap<(usize, usize), AnchorSet> = FxHashMap::default();
    let mut anchor_counts: Vec<Vec<u32>> = vec![vec![0_u32; n_batches]; n_batches];

    for a in 0..n_batches {
        for b in (a + 1)..n_batches {
            let pair_start = Instant::now();
            if verbosity.normal_verbosity() {
                println!("rPCA: finding anchors for batch pair ({a}, {b})");
            }

            // Truncate loadings to `dims` used for anchoring.
            let dims_use = params
                .dims
                .min(per_batch_loadings[a].ncols())
                .min(per_batch_loadings[b].ncols());
            let load_a_trunc = Mat::from_fn(per_batch_loadings[a].nrows(), dims_use, |i, j| {
                per_batch_loadings[a][(i, j)]
            });
            let load_b_trunc = Mat::from_fn(per_batch_loadings[b].nrows(), dims_use, |i, j| {
                per_batch_loadings[b][(i, j)]
            });

            let a_in_a = normalise_own_scores(&per_batch_pca_scores[a], params.l2_norm);
            let b_in_b = normalise_own_scores(&per_batch_pca_scores[b], params.l2_norm);

            let a_in_b = project_into_basis(
                load_b_trunc.as_ref(),
                per_batch_standardised[a].as_ref(),
                params.l2_norm,
            );
            let b_in_a = project_into_basis(
                load_a_trunc.as_ref(),
                per_batch_standardised[b].as_ref(),
                params.l2_norm,
            );

            // restrict own-basis embeddings to `dims_use` columns too.
            let a_in_a = Mat::from_fn(a_in_a.nrows(), dims_use, |i, j| a_in_a[(i, j)]);
            let b_in_b = Mat::from_fn(b_in_b.nrows(), dims_use, |i, j| b_in_b[(i, j)]);

            let (pairs, scores) = find_rpca_anchors_for_pair(
                a_in_a.as_ref(),
                b_in_b.as_ref(),
                a_in_b.as_ref(),
                b_in_a.as_ref(),
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

    // drop everything unneeded
    drop(per_batch_standardised);
    drop(per_batch_loadings);
    drop(per_batch_pca_scores);

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
        println!("rPCA integration complete in {:.2?}", start_total.elapsed());
    }

    Ok(corrected)
}
