//! Seurat label transfer: `FindTransferAnchors(reduction = "cca")` and
//! `TransferData`.
//!
//! The reference and the query are two stores (or two cell sets of one) with
//! a shared feature list: RNA against RNA, or RNA against the gene activity of
//! an ATAC store. Each side is scaled per feature on its own (`ScaleData`,
//! clipped above at `scale_max`), then per cell (`Standardize`); the canonical
//! correlation vectors are the singular vectors of the cells x cells cross
//! product, computed exactly without forming it (`cross_product_svd`).
//! Anchors are mutual nearest neighbours in that space, L2 normalised per
//! cell, scored by shared neighbours. The labels then move through
//! Gaussian-kernel weights of each query cell to its nearest anchor cells in
//! an embedding of the query, for ATAC its LSI.
//!
//! Unlike Seurat's integration CCA in [`crate::sc_batch_correction::seurat_cca`]
//! this runs the per-feature scaling, so it follows Seurat to numerical noise.
//!
//! References: Stuart et al., Cell, 2019.

use faer::{Mat, MatRef};
use indexmap::IndexSet;
use rayon::prelude::*;

use crate::prelude::*;
use crate::sc_batch_correction::batch_utils::{cosine_normalise, par_for_each_col_mut};
use crate::sc_batch_correction::seurat_anchors::{
    AnchorSet, AnchorSpaces, find_weights, mnn_scored_anchors,
};
use crate::sc_processing::pca::scale_csc_chunk;
use bixverse_core::core::math::vector_helpers::mean_sd_f64;

////////////
// Params //
////////////

/// Parameters of [`find_transfer_anchors`]. Defaults follow Seurat.
#[derive(Clone, Debug)]
pub struct TransferAnchorParams {
    /// Canonical correlation vectors (Seurat `dims = 1:30`)
    pub n_cc: usize,
    /// Neighbours for the mutual nearest neighbour anchors
    pub k_anchor: usize,
    /// Neighbours for the shared neighbour score
    pub k_score: usize,
    /// Upper clip of the per-feature scaled values (`ScaleData` `scale.max`)
    pub scale_max: f32,
    /// L2 normalise the CCA space per cell
    pub l2_norm: bool,
    /// kNN backend
    pub knn_params: KnnParams,
}

impl Default for TransferAnchorParams {
    /// The default [`TransferAnchorParams`].
    ///
    /// ### Returns
    ///
    /// The defaults.
    fn default() -> Self {
        Self {
            n_cc: 30,
            k_anchor: 5,
            k_score: 30,
            scale_max: 10.0,
            l2_norm: true,
            knn_params: KnnParams::default(),
        }
    }
}

/// Parameters of [`transfer_labels`]. Defaults follow Seurat.
#[derive(Clone, Debug)]
pub struct TransferDataParams {
    /// Anchor pairs per query cell in the weights
    pub k_weight: usize,
    /// Gaussian kernel bandwidth divisor
    pub sd: f32,
    /// kNN backend
    pub knn_params: KnnParams,
}

impl Default for TransferDataParams {
    /// The default [`TransferDataParams`].
    ///
    /// ### Returns
    ///
    /// The defaults.
    fn default() -> Self {
        Self {
            k_weight: 50,
            sd: 1.0,
            knn_params: KnnParams::default(),
        }
    }
}

/////////////
// Helpers //
/////////////

/// Normalised expression of features, scaled per feature (`ScaleData`).
///
/// ### Params
///
/// * `reader` - Gene-based reader
/// * `cells` - Cells, output column order
/// * `features` - Features, output row order
/// * `scale_max` - Upper clip
///
/// ### Returns
///
/// `(features x cells, whether each feature varies)`. A feature without
/// variance is all zero.
fn scale_features<S: SingleCellReading>(
    reader: &S,
    cells: &[usize],
    features: &[usize],
    scale_max: f32,
) -> Result<(Mat<f32>, Vec<bool>), ScErrors> {
    let cell_set: IndexSet<u32> = cells.iter().map(|&c| c as u32).collect();
    let chunks = reader.read_gene_parallel_filtered(features, &cell_set)?;
    let rows: Vec<(Vec<f32>, bool)> = chunks
        .par_iter()
        .map(|chunk| {
            let (mut v, _, sd) = scale_csc_chunk(chunk, cells.len(), true, true, None);
            v.iter_mut().for_each(|x| *x = x.min(scale_max));
            let varies = sd > 0.0 && v.iter().any(|&x| x != 0.0);
            (v, varies)
        })
        .collect();
    let m = Mat::from_fn(features.len(), cells.len(), |g, c| rows[g].0[c]);
    Ok((m, rows.into_iter().map(|(_, ok)| ok).collect()))
}

/// Rows `keep` of a matrix, each column then standardised (`Standardize`:
/// mean 0, sd 1 over n - 1).
///
/// ### Params
///
/// * `m` - Features x cells
/// * `keep` - Rows to keep
///
/// ### Returns
///
/// `keep.len()` x cells.
fn standardise_cells(m: MatRef<f32>, keep: &[usize]) -> Mat<f32> {
    let mut out = Mat::from_fn(keep.len(), m.ncols(), |i, j| m[(keep[i], j)]);
    par_for_each_col_mut(&mut out, |_, col| {
        let (mean, sd) = mean_sd_f64(col.iter().copied());
        col.iter_mut()
            .for_each(|v| *v = ((*v as f64 - mean) / sd) as f32);
    });
    out
}

/// Leading singular vectors of `x1^T x2`, exact, without forming it.
///
/// With thin QRs `x1^T = Q1 R1` and `x2^T = Q2 R2`, `x1^T x2 = Q1 (R1 R2^T)
/// Q2^T`, so the SVD `U S V^T` of the small `R1 R2^T` (features x features at
/// most) gives the singular vectors `Q1 U` and `Q2 V`. Cost is linear in the
/// cells. Seurat's CCA is exact (irlba converges to 1e-11 here), and a
/// randomised sketch is not close enough when singular values crowd.
///
/// ### Params
///
/// * `x1` - Features x cells of the first set
/// * `x2` - Features x cells of the second set
/// * `k` - Number of singular vectors
///
/// ### Returns
///
/// `(u, v)`: cells of `x1` x `k` and cells of `x2` x `k`.
fn cross_product_svd(
    x1: MatRef<f32>,
    x2: MatRef<f32>,
    k: usize,
) -> Result<(Mat<f32>, Mat<f32>), ScErrors> {
    let qr1 = x1.transpose().qr();
    let qr2 = x2.transpose().qr();
    let small = qr1.thin_R() * qr2.thin_R().transpose();
    let svd = small
        .thin_svd()
        .map_err(|e| CoreErrors::FaerSvdError(format!("{e:?}")))?;
    let k = k.min(svd.S().dim());
    let u = qr1.compute_thin_Q() * svd.U().subcols(0, k);
    let v = qr2.compute_thin_Q() * svd.V().subcols(0, k);
    Ok((u, v))
}

///////////////
// Functions //
///////////////

/// Anchors between a reference and a query (Seurat `FindTransferAnchors` with
/// `reduction = "cca"`, `k.filter = NA`).
///
/// ### Params
///
/// * `ref_reader` - Gene-based reader of the reference
/// * `ref_cells` - Reference cells
/// * `ref_features` - Reference features
/// * `query_reader` - Gene-based reader of the query
/// * `query_cells` - Query cells
/// * `query_features` - Query features, the same features as `ref_features`
///   in the same order
/// * `params` - See [`TransferAnchorParams`]
/// * `seed` - Seed of the kNN backend
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [`AnchorSet`]: pairs `(reference cell, query cell)` as positions in
/// `ref_cells` and `query_cells`, with their scores.
#[allow(clippy::too_many_arguments)]
pub fn find_transfer_anchors<R: SingleCellReading, Q: SingleCellReading>(
    ref_reader: &R,
    ref_cells: &[usize],
    ref_features: &[usize],
    query_reader: &Q,
    query_cells: &[usize],
    query_features: &[usize],
    params: &TransferAnchorParams,
    seed: usize,
    verbose: usize,
) -> Result<AnchorSet, ScErrors> {
    if ref_features.len() != query_features.len() {
        return Err(CoreErrors::InvalidArgument(format!(
            "{} reference and {} query features",
            ref_features.len(),
            query_features.len()
        ))
        .into());
    }
    let k_nn = params.k_anchor.max(params.k_score);
    for (batch, n) in [(0, ref_cells.len()), (1, query_cells.len())] {
        if n <= k_nn {
            return Err(ScErrors::TooFewCellsForAnchor {
                batch,
                n_cells: n,
                required: k_nn + 1,
            });
        }
    }

    let (x_ref, var_ref) = scale_features(ref_reader, ref_cells, ref_features, params.scale_max)?;
    let (x_query, var_query) =
        scale_features(query_reader, query_cells, query_features, params.scale_max)?;
    // Seurat's CheckFeatures drops features without variance on either side
    let keep: Vec<usize> = (0..ref_features.len())
        .filter(|&g| var_ref[g] && var_query[g])
        .collect();
    let x1 = standardise_cells(x_ref.as_ref(), &keep);
    let x2 = standardise_cells(x_query.as_ref(), &keep);

    let (cc_ref, cc_query) = cross_product_svd(x1.as_ref(), x2.as_ref(), params.n_cc)?;
    let (cc_ref, cc_query) = if params.l2_norm {
        (cosine_normalise(&cc_ref), cosine_normalise(&cc_query))
    } else {
        (cc_ref, cc_query)
    };

    let (pairs, scores) = mnn_scored_anchors(
        &AnchorSpaces::shared(cc_ref.as_ref(), cc_query.as_ref()),
        params.k_anchor,
        params.k_score,
        None,
        &params.knn_params,
        seed,
        verbose,
    )?;
    Ok(AnchorSet {
        pairs,
        scores,
        batch_pair: (0, 1),
    })
}

/// Transfer categorical labels to the query (Seurat `TransferData`).
///
/// ### Params
///
/// * `anchors` - From [`find_transfer_anchors`]
/// * `ref_labels` - Label index of every reference cell, `0..n_labels`
/// * `n_labels` - Number of labels
/// * `weight_embedding` - Query cells x dimensions the weights are computed
///   in; Seurat's `weight.reduction` with its `dims` already selected (for
///   ATAC the LSI without component 1)
/// * `params` - See [`TransferDataParams`]
/// * `seed` - Seed of the kNN backend
///
/// ### Returns
///
/// Query cells x labels prediction scores; each row sums to 1 for a cell
/// with any anchor weight. The predicted label is the first maximum.
pub fn transfer_labels(
    anchors: &AnchorSet,
    ref_labels: &[usize],
    n_labels: usize,
    weight_embedding: MatRef<f32>,
    params: &TransferDataParams,
    seed: usize,
) -> Result<Mat<f32>, ScErrors> {
    if let Some(&l) = ref_labels.iter().find(|&&l| l >= n_labels) {
        return Err(CoreErrors::InvalidArgument(format!("label {l} for {n_labels} labels")).into());
    }
    if anchors
        .pairs
        .iter()
        .any(|&(r, _)| r as usize >= ref_labels.len())
    {
        return Err(CoreErrors::InvalidArgument(
            "an anchor points past the reference labels".into(),
        )
        .into());
    }
    let weights = find_weights(
        weight_embedding,
        anchors,
        params.k_weight,
        params.sd,
        &params.knn_params,
        seed,
    )?;
    let n_query = weight_embedding.nrows();
    let mut out = Mat::<f32>::zeros(n_query, n_labels);
    for c in 0..n_query {
        for p in weights.indptr[c] as usize..weights.indptr[c + 1] as usize {
            let pair = weights.indices[p] as usize;
            out[(c, ref_labels[anchors.pairs[pair].0 as usize])] += weights.data[p];
        }
    }
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_product_svd_matches_dense_svd() {
        let (g, n1, n2, k) = (12, 30, 25, 5);
        let x1 = Mat::from_fn(g, n1, |i, j| ((i * 7 + j * 3) % 11) as f32 - 5.0);
        let x2 = Mat::from_fn(g, n2, |i, j| ((i * 5 + j * 2) % 13) as f32 - 6.0);
        let (u, v) = cross_product_svd(x1.as_ref(), x2.as_ref(), k).unwrap();

        let m = x1.transpose() * &x2;
        let svd = m.thin_svd().unwrap();
        for j in 0..k {
            // singular vectors up to sign: |<u, u_ref>| = 1
            let cu: f32 = (0..n1).map(|i| u[(i, j)] * svd.U()[(i, j)]).sum();
            let cv: f32 = (0..n2).map(|i| v[(i, j)] * svd.V()[(i, j)]).sum();
            assert!((cu.abs() - 1.0).abs() < 1e-4, "u {j}: {cu}");
            assert!((cv.abs() - 1.0).abs() < 1e-4, "v {j}: {cv}");
        }
    }
}
