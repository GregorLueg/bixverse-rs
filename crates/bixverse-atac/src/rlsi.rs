//! Reciprocal LSI (rLSI) integration, Seurat's
//! `FindIntegrationAnchors(reduction = "rlsi")` followed by
//! `IntegrateEmbeddings`.
//!
//! Every batch gets its own TF-IDF (IDF over its own cells) and LSI. For a
//! batch pair, each side projects onto the other's LSI ([`project_lsi`]),
//! centred per component; the own LSI and the projection of the other batch
//! are stacked, divided by the column standard deviation of the stack and L2
//! normalised per cell. Anchors are mutual nearest neighbours across those
//! stacks, scored by shared neighbours on the unnormalised own LSI. The
//! correction itself runs on an embedding the caller supplies, usually the LSI
//! of all cells, through the anchor primitives of the rPCA and CCA paths.
//!
//! Seurat intersects the per-object feature sets; here all batches share one
//! feature set.

use faer::{Mat, MatRef};
use rustc_hash::FxHashMap;

use bixverse_sc::sc_batch_correction::batch_utils::{batch_knn_search, cosine_normalise};
use bixverse_sc::sc_batch_correction::fast_mnn::{reorder_to_original, split_pca_by_batch};
use bixverse_sc::sc_batch_correction::seurat_anchors::{
    AnchorSet, build_sample_tree, find_anchor_pairs, score_anchors, tree_merge_embeddings,
};
use bixverse_sc_data::v5_io::tf_idf_weights;

use crate::lsi::{LsiParams, library_sizes, lsi_on_matrix, project_lsi, read_tf_idf};
use crate::prelude::*;

///////////
// Types //
///////////

/// Parameters of the rLSI integration. Defaults follow Seurat.
#[derive(Clone, Debug)]
pub struct RlsiParams {
    /// 0-based LSI components the anchors are found on. Seurat's `dims = 2:30`
    /// is `1..30`, skipping the depth component.
    pub dims: Vec<usize>,
    /// Neighbours for the mutual nearest neighbour anchors
    pub k_anchor: usize,
    /// Neighbours for the shared neighbour score
    pub k_score: usize,
    /// Anchor cells per query cell in the correction weights
    pub k_weight: usize,
    /// Gaussian kernel bandwidth divisor of the weights
    pub sd: f32,
    /// Scale and L2 normalise the anchor spaces (Seurat `l2.norm`)
    pub l2_norm: bool,
    /// Per-batch LSI; `n_comp` must cover `dims`
    pub lsi: LsiParams,
    /// kNN backend for the anchors and the weights
    pub knn_params: KnnParams,
}

impl Default for RlsiParams {
    fn default() -> Self {
        Self {
            dims: (1..30).collect(),
            k_anchor: 5,
            k_score: 30,
            k_weight: 100,
            sd: 1.0,
            l2_norm: true,
            lsi: LsiParams {
                n_comp: 30,
                ..LsiParams::default()
            },
            knn_params: KnnParams::default(),
        }
    }
}

/// Anchor pairs `(cell in a, cell in b)` and their scores.
pub type ScoredAnchors = (Vec<(u32, u32)>, Vec<f32>);

/// One batch's LSI and TF-IDF, the input of the anchor search.
#[derive(Clone, Debug)]
pub struct BatchLsi {
    /// Cells x components, z-scored ([`crate::lsi::LsiResult::embedding`])
    pub embedding: Mat<f32>,
    /// Features x components
    pub loadings: Mat<f32>,
    /// Singular values
    pub singular_values: Vec<f32>,
    /// Cells x features CSR, TF-IDF with this batch's IDF in `data_2`
    pub tf_idf: CompressedSparseData2<f32>,
}

/////////////
// Helpers //
/////////////

/// Columns `dims` of a matrix.
///
/// ### Params
///
/// * `m` - The matrix
/// * `dims` - Column indices
///
/// ### Returns
///
/// `m[, dims]`.
fn select_columns(m: MatRef<f32>, dims: &[usize]) -> Mat<f32> {
    Mat::from_fn(m.nrows(), dims.len(), |i, j| m[(i, dims[j])])
}

/// Projection of `from` onto the LSI of `onto`, centred per component.
///
/// ### Params
///
/// * `from` - Projected batch
/// * `onto` - Batch whose LSI is projected onto
///
/// ### Returns
///
/// Cells of `from` x components of `onto`.
fn centred_projection(from: &BatchLsi, onto: &BatchLsi) -> Result<Mat<f32>, AtacErrors> {
    let mut p = project_lsi(&from.tf_idf, onto.loadings.as_ref(), &onto.singular_values)?;
    let n = p.nrows() as f64;
    for j in 0..p.ncols() {
        let mean = ((0..p.nrows()).map(|i| p[(i, j)] as f64).sum::<f64>() / n) as f32;
        for i in 0..p.nrows() {
            p[(i, j)] -= mean;
        }
    }
    Ok(p)
}

/// Stack an own embedding over a projection and, with `l2_norm`, divide each
/// column by its sample standard deviation over the stack, then L2 normalise
/// each row (Seurat `ReciprocalProject`).
///
/// ### Params
///
/// * `own` - Own LSI rows
/// * `projected` - Projected rows, same components
/// * `l2_norm` - Scale and normalise
///
/// ### Returns
///
/// `(own rows, projected rows)` after the transform.
fn anchor_space(own: &Mat<f32>, projected: &Mat<f32>, l2_norm: bool) -> (Mat<f32>, Mat<f32>) {
    if !l2_norm {
        return (own.clone(), projected.clone());
    }
    let (n_own, n_proj) = (own.nrows(), projected.nrows());
    let n = (n_own + n_proj) as f64;
    let sd: Vec<f32> = (0..own.ncols())
        .map(|j| {
            let col = (0..n_own)
                .map(|i| own[(i, j)] as f64)
                .chain((0..n_proj).map(|i| projected[(i, j)] as f64));
            let mean = col.clone().sum::<f64>() / n;
            let ss: f64 = col.map(|v| (v - mean).powi(2)).sum();
            (ss / (n - 1.0)).sqrt() as f32
        })
        .collect();
    let scale = |m: &Mat<f32>| {
        cosine_normalise(&Mat::from_fn(m.nrows(), m.ncols(), |i, j| {
            m[(i, j)] / sd[j]
        }))
    };
    (scale(own), scale(projected))
}

///////////////
// Functions //
///////////////

/// Scored rLSI anchors between two batches (Seurat `FindIntegrationAnchors`
/// with `reduction = "rlsi"` for one pair).
///
/// ### Params
///
/// * `a` - First batch
/// * `b` - Second batch
/// * `params` - See [`RlsiParams`]
/// * `seed` - Seed of the kNN backend
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `(pairs, scores)`: `(cell in a, cell in b)` and the rescaled shared
/// neighbour score of each pair. Empty when no mutual neighbour exists.
pub fn rlsi_anchors(
    a: &BatchLsi,
    b: &BatchLsi,
    params: &RlsiParams,
    seed: usize,
    verbose: usize,
) -> Result<ScoredAnchors, AtacErrors> {
    let dims = &params.dims;
    let n_comp = a.embedding.ncols().min(b.embedding.ncols());
    if dims.is_empty() || dims.iter().any(|&d| d >= n_comp) {
        return Err(AtacErrors::InvalidArgument(format!(
            "rLSI dims must be within the {n_comp} LSI components"
        )));
    }
    let (n_a, n_b) = (a.embedding.nrows(), b.embedding.nrows());
    let k_nn = params.k_anchor.max(params.k_score);
    for (batch, n) in [(0, n_a), (1, n_b)] {
        if n <= k_nn {
            return Err(ScErrors::TooFewCellsForAnchor {
                batch,
                n_cells: n,
                required: k_nn + 1,
            }
            .into());
        }
    }

    // reference space: a's LSI with b projected in; query space: the reverse.
    // Seurat scales and normalises over all components and only then keeps
    // `dims` (ReciprocalProject is called without them)
    let (ref_a, ref_b) = anchor_space(&a.embedding, &centred_projection(b, a)?, params.l2_norm);
    let (query_b, query_a) = anchor_space(&b.embedding, &centred_projection(a, b)?, params.l2_norm);
    let [ref_a, ref_b, query_a, query_b] =
        [ref_a, ref_b, query_a, query_b].map(|m| select_columns(m.as_ref(), dims));
    let own_a = select_columns(a.embedding.as_ref(), dims);
    let own_b = select_columns(b.embedding.as_ref(), dims);

    let knn = &params.knn_params;
    // within-batch neighbours include the cell itself, as Seurat's k + 1
    let (nnaa, _) = batch_knn_search(own_a.as_ref(), own_a.as_ref(), k_nn + 1, knn, seed, verbose)?;
    let (nnbb, _) = batch_knn_search(own_b.as_ref(), own_b.as_ref(), k_nn + 1, knn, seed, verbose)?;
    let (nnab, _) = batch_knn_search(query_a.as_ref(), query_b.as_ref(), k_nn, knn, seed, verbose)?;
    let (nnba, _) = batch_knn_search(ref_b.as_ref(), ref_a.as_ref(), k_nn, knn, seed, verbose)?;

    let pairs = find_anchor_pairs(&nnab, &nnba, params.k_anchor);
    if pairs.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let scores = score_anchors(&pairs, &nnaa, &nnab, &nnba, &nnbb, params.k_score, n_a);
    Ok((pairs, scores))
}

/// rLSI integration of an embedding from per-batch LSIs.
///
/// Anchors for every batch pair ([`rlsi_anchors`]), then Seurat's
/// `IntegrateEmbeddings`: batches merge along the anchor-count tree, each query
/// corrected by kernel-weighted anchor differences on `embedding` itself.
///
/// ### Params
///
/// * `batches` - One [`BatchLsi`] per batch, rows in the order the cells of
///   that batch appear in `batch_indices`
/// * `batch_indices` - 0-based batch per cell
/// * `embedding` - Cells x dimensions to integrate, rows aligned with
///   `batch_indices`; Seurat integrates the LSI of all cells, `1:30`
/// * `params` - See [`RlsiParams`]
/// * `seed` - Seed of the kNN backend
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The integrated embedding, rows in input order.
pub fn rlsi_integration(
    batches: &[BatchLsi],
    batch_indices: &[usize],
    embedding: &Mat<f32>,
    params: &RlsiParams,
    seed: usize,
    verbose: usize,
) -> Result<Mat<f32>, AtacErrors> {
    let n_batches = batches.len();
    if n_batches < 2 {
        return Err(ScErrors::NeedAtLeastTwoBatches { n_batches }.into());
    }
    let mut sizes = vec![0usize; n_batches];
    for &b in batch_indices {
        if b >= n_batches {
            return Err(AtacErrors::InvalidArgument(format!(
                "batch index {b} for {n_batches} batches"
            )));
        }
        sizes[b] += 1;
    }
    if embedding.nrows() != batch_indices.len()
        || sizes
            .iter()
            .zip(batches)
            .any(|(&n, lsi)| n != lsi.embedding.nrows())
    {
        return Err(AtacErrors::InvalidArgument(
            "rLSI: batch sizes, LSIs and embedding rows do not match".into(),
        ));
    }

    let mut anchors_by_pair: FxHashMap<(usize, usize), AnchorSet> = FxHashMap::default();
    let mut anchor_counts = vec![vec![0u32; n_batches]; n_batches];
    for a in 0..n_batches {
        for b in (a + 1)..n_batches {
            let (pairs, scores) = rlsi_anchors(&batches[a], &batches[b], params, seed, verbose)?;
            anchor_counts[a][b] = pairs.len() as u32;
            anchor_counts[b][a] = pairs.len() as u32;
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
    let (per_batch, per_batch_rows) = split_pca_by_batch(embedding, batch_indices);
    let (merged, index_map) = tree_merge_embeddings(
        per_batch,
        per_batch_rows,
        &anchors_by_pair,
        &merge_order,
        params.k_weight,
        params.sd,
        &params.knn_params,
        seed,
        verbose,
    )?;
    Ok(reorder_to_original(&merged, &index_map))
}

/// rLSI integration over a peak or tile store.
///
/// Per batch: IDF over its own cells, TF-IDF of `features`, LSI; then
/// [`rlsi_integration`]. The store's own norm layer is not used, so it need
/// not be TF-IDF.
///
/// ### Params
///
/// * `cell_path` - Cell-based file of the store
/// * `gene_path` - Gene-based (feature-major) file of the store
/// * `cells` - Cells, rows of `embedding`
/// * `features` - Features of the per-batch LSIs
/// * `batch_indices` - 0-based batch per cell
/// * `embedding` - Cells x dimensions to integrate, usually the LSI of all
///   cells
/// * `params` - See [`RlsiParams`]
/// * `seed` - Seed of the solvers and the kNN backend
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The integrated embedding, rows in `cells` order.
#[allow(clippy::too_many_arguments)]
pub fn rlsi_on_store(
    cell_path: &str,
    gene_path: &str,
    cells: &[usize],
    features: &[usize],
    batch_indices: &[usize],
    embedding: &Mat<f32>,
    params: &RlsiParams,
    seed: usize,
    verbose: usize,
) -> Result<Mat<f32>, AtacErrors> {
    if batch_indices.len() != cells.len() {
        return Err(AtacErrors::InvalidArgument(format!(
            "{} batch labels for {} cells",
            batch_indices.len(),
            cells.len()
        )));
    }
    let n_batches = batch_indices.iter().max().map_or(0, |&b| b + 1);
    let gene_reader = ParallelSparseReader::new(gene_path)?;
    let mut batches = Vec::with_capacity(n_batches);
    for b in 0..n_batches {
        let cells_b: Vec<usize> = cells
            .iter()
            .zip(batch_indices)
            .filter(|&(_, &bi)| bi == b)
            .map(|(&c, _)| c)
            .collect();
        let idf = tf_idf_weights(&gene_reader, Some(&cells_b))?;
        let tf_idf = read_tf_idf(cell_path, gene_path, &cells_b, features, Some(&idf))?;
        let depth = library_sizes(cell_path, &cells_b)?;
        let lsi = lsi_on_matrix(tf_idf.clone(), &depth, &params.lsi, seed, verbose)?;
        batches.push(BatchLsi {
            embedding: lsi.embedding,
            loadings: lsi.loadings,
            singular_values: lsi.singular_values,
            tf_idf,
        });
    }
    rlsi_integration(&batches, batch_indices, embedding, params, seed, verbose)
}
