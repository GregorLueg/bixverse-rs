//! Shared primitives for Seurat-style anchor-based batch correction.
//!
//! This module holds everything that is common to the CCA and rPCA
//! integration paths: the anchor set representation, mutual-nearest-neighbour
//! anchor extraction, shared-neighbour scoring, Gaussian-kernel weight
//! computation, correction application on an embedding, and the greedy
//! hierarchical merge tree.
//!
//! CCA and rPCA differ only in how the shared low-dimensional space between
//! a batch pair is constructed. Once that space (or spaces, for rPCA) is
//! available, callers issue four kNN queries themselves and hand the
//! neighbour lists here. Keeps this module free of the CCA/rPCA-specific
//! math.
//!
//! References: Stuart et al., Cell, 2019 (Seurat v3).

use faer::{Mat, MatRef};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::core::math::sparse::{CompressedSparseData2, CompressedSparseFormat};
use crate::prelude::*;

///////////////
// Structure //
///////////////

/// A set of anchor pairs between two batches with per-pair scores in [0, 1].
///
/// ### Fields
///
/// * `pairs` - `(cell_a, cell_b)` indices within their respective batches
/// * `scores` - Rescaled shared-neighbour scores, one per pair
/// * `batch_pair` - `(batch_a, batch_b)` indices in the caller's batch list
#[derive(Clone, Debug)]
pub struct AnchorSet {
    /// Anchor pairs: `(index in batch_a, index in batch_b)`
    pub pairs: Vec<(u32, u32)>,
    /// Per-anchor score in `[0, 1]`
    pub scores: Vec<f32>,
    /// Batch identifiers `(batch_a, batch_b)`
    pub batch_pair: (usize, usize),
}

impl AnchorSet {
    /// Number of anchors in this set
    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// True when no anchors survived filtering
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

//////////////////////
// Mutual neighbour //
//////////////////////

/// Extract MNN anchor pairs from cross-batch kNN graphs.
///
/// A candidate `(a, b)` is emitted iff `b` is in `a`'s `k_anchor`-nearest
/// batch-B neighbours AND `a` is in `b`'s `k_anchor`-nearest batch-A
/// neighbours. This is Seurat's `FindAnchorPairs`.
///
/// ### Params
///
/// * `nnab` - For each cell in batch A, its neighbours in batch B (row length
///   >= `k_anchor`; only the first `k_anchor` are consulted)
/// * `nnba` - Symmetric for batch B
/// * `k_anchor` - Neighbours to consider per side. Rows shorter than
///   `k_anchor` are consulted in full.
///
/// ### Returns
///
/// Deduplicated `(a, b)` pairs. Order is deterministic.
pub fn find_anchor_pairs(
    nnab: &[Vec<usize>],
    nnba: &[Vec<usize>],
    k_anchor: usize,
) -> Vec<(u32, u32)> {
    let ba_sets: Vec<FxHashSet<usize>> = nnba
        .iter()
        .map(|neighbours| {
            neighbours
                .iter()
                .take(k_anchor)
                .copied()
                .collect::<FxHashSet<usize>>()
        })
        .collect();

    let mut pairs: Vec<(u32, u32)> = nnab
        .par_iter()
        .enumerate()
        .fold(Vec::new, |mut acc, (a, neighbours)| {
            for &b in neighbours.iter().take(k_anchor) {
                if b < ba_sets.len() && ba_sets[b].contains(&a) {
                    acc.push((a as u32, b as u32));
                }
            }
            acc
        })
        .reduce(Vec::new, |mut left, right| {
            left.extend(right);
            left
        });

    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

//////////////
// Filtering //
//////////////

/// Filter anchor pairs by an additional cross-batch kNN graph.
///
/// Used for Seurat's `FilterAnchors` step (CCA-only): a candidate anchor is
/// kept only if `b` is among the `k_filter`-nearest batch-B cells of `a`
/// in the gene-space kNN passed as `nn_filter_ab`.
///
/// ### Params
///
/// * `pairs` - Candidate anchors to filter (from [find_anchor_pairs])
/// * `nn_filter_ab` - Gene-space kNN of every batch-A cell against batch-B
/// * `k_filter` - Neighbours to consider (defaults to the row length)
///
/// ### Returns
///
/// The subset of `pairs` that also live in the gene-space neighbourhood.
pub fn filter_anchor_pairs(
    pairs: &[(u32, u32)],
    nn_filter_ab: &[Vec<usize>],
    k_filter: usize,
) -> Vec<(u32, u32)> {
    let filter_sets: Vec<FxHashSet<usize>> = nn_filter_ab
        .iter()
        .map(|neighbours| neighbours.iter().take(k_filter).copied().collect())
        .collect();

    pairs
        .par_iter()
        .filter_map(|&(a, b)| {
            let ai = a as usize;
            if ai < filter_sets.len() && filter_sets[ai].contains(&(b as usize)) {
                Some((a, b))
            } else {
                None
            }
        })
        .collect()
}

////////////
// Scoring //
////////////

/// Score every anchor pair by the number of shared neighbours it induces on
/// the merged batch-A + batch-B neighbourhood graph, then rescale to `[0, 1]`
/// via the `(q_0.01, q_0.9)` interval. Matches Seurat's `ScoreAnchors`.
///
/// The merged neighbourhood of a batch-A cell `a` is the union of its
/// within-batch neighbours (`nnaa[a]`) and its cross-batch neighbours
/// (`nnab[a]`, offset by `offset_b` so indices don't collide).
///
/// ### Params
///
/// * `pairs` - Anchor pairs `(a, b)`
/// * `nnaa` - For every batch-A cell, its `k_score` within-batch neighbours
/// * `nnab` - For every batch-A cell, its `k_score` batch-B neighbours
/// * `nnba` - For every batch-B cell, its `k_score` batch-A neighbours
/// * `nnbb` - For every batch-B cell, its `k_score` within-batch neighbours
/// * `k_score` - Neighbours considered per side (rows are consulted up to
///   this many)
/// * `offset_b` - Amount added to batch-B indices in the merged graph. Should
///   equal `n_cells_a` so that batch-B cell `j` becomes `j + n_cells_a` in
///   the merged neighbourhood.
///
/// ### Returns
///
/// Rescaled scores, one per anchor pair, all in `[0, 1]`.
pub fn score_anchors(
    pairs: &[(u32, u32)],
    nnaa: &[Vec<usize>],
    nnab: &[Vec<usize>],
    nnba: &[Vec<usize>],
    nnbb: &[Vec<usize>],
    k_score: usize,
    offset_b: usize,
) -> Vec<f32> {
    let raw: Vec<u32> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let ai = a as usize;
            let bi = b as usize;

            let mut a_set: FxHashSet<usize> = nnaa
                .get(ai)
                .map(|n| n.iter().take(k_score).copied().collect())
                .unwrap_or_default();
            if let Some(cross) = nnab.get(ai) {
                for &x in cross.iter().take(k_score) {
                    a_set.insert(x + offset_b);
                }
            }

            let mut b_set: FxHashSet<usize> = nnbb
                .get(bi)
                .map(|n| n.iter().take(k_score).map(|x| x + offset_b).collect())
                .unwrap_or_default();
            if let Some(cross) = nnba.get(bi) {
                for &x in cross.iter().take(k_score) {
                    b_set.insert(x);
                }
            }

            a_set.intersection(&b_set).count() as u32
        })
        .collect();

    rescale_scores(&raw)
}

/// Rescale raw shared-neighbour counts to `[0, 1]` via
/// `(x - q_0.01) / (q_0.9 - q_0.01)`, clamped to `[0, 1]`.
///
/// Degenerate cases are treated as "trust all pairs equally": when there
/// is a single pair, or when the 1st and 90th percentiles coincide
/// (uniform counts), every anchor is returned with score `1.0`. Returning
/// zeros in this case would silently cancel the downstream correction.
///
/// ### Params
///
/// * `raw` - Raw shared-neighbour counts, one per anchor pair
///
/// ### Returns
///
/// Rescaled scores in `[0, 1]`, length matches `raw`.
fn rescale_scores(raw: &[u32]) -> Vec<f32> {
    if raw.is_empty() {
        return Vec::new();
    }
    if raw.len() == 1 {
        return vec![1.0];
    }

    let mut sorted: Vec<f32> = raw.iter().map(|&x| x as f32).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q01 = quantile_sorted(&sorted, 0.01);
    let q90 = quantile_sorted(&sorted, 0.90);
    let range = q90 - q01;
    if range < 1e-12 {
        return vec![1.0; raw.len()];
    }

    raw.iter()
        .map(|&x| ((x as f32 - q01) / range).clamp(0.0, 1.0))
        .collect()
}

/// Linear-interpolation quantile on a slice already sorted ascending.
/// Mirrors R's default `quantile(type = 7)` used inside Seurat's
/// `ScoreAnchors` rescale step.
///
/// ### Params
///
/// * `sorted` - Values in ascending order. Empty slice returns 0.
/// * `q` - Quantile in `[0, 1]`
///
/// ### Returns
///
/// Interpolated quantile value.
fn quantile_sorted(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() as f32 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

////////////
// Weights //
////////////

/// Build Seurat's per-anchor-pair integration matrix.
///
/// Row `i` of the returned matrix is `ref_embed[r_i] - query_embed[q_i]`
/// for anchor pair `i = (r_i, q_i)`. Shape `(n_pairs, dims)`. Feeds
/// [apply_correction] as `Delta` in `corrected = query - W^T @ Delta`.
///
/// ### Params
///
/// * `anchors` - The anchor set for this batch pair
/// * `ref_embed` - Reference batch embedding, `(n_ref, dims)`
/// * `query_embed` - Query batch embedding, `(n_query, dims)`
///
/// ### Returns
///
/// Dense integration matrix, one row per anchor pair.
pub fn build_integration_matrix(
    anchors: &AnchorSet,
    ref_embed: MatRef<f32>,
    query_embed: MatRef<f32>,
) -> Mat<f32> {
    let dims = ref_embed.ncols();
    let n_pairs = anchors.pairs.len();
    Mat::from_fn(n_pairs, dims, |i, d| {
        let (r, q) = anchors.pairs[i];
        ref_embed[(r as usize, d)] - query_embed[(q as usize, d)]
    })
}

/// Compute the sparse per-pair Gaussian-kernel weight matrix.
///
/// Matches Seurat's `FindWeightsC` (`src/integration.cpp:47-51`) exactly:
///   1. For each query cell `c`, find its `k_weight` nearest anchor
///      query cells in the current embedding.
///   2. Per-row-normalise the distances: `d_norm = 1 - d / d_max` so
///      values live in `[0, 1]` with `1 = closest` (Seurat's R-side
///      transform, `integration.R:4620`).
///   3. For every anchor pair `p` whose query end is that near unique
///      cell, push `w[p, c] = 1 - exp(- d_norm * score_p / (2/sd)^2)`.
///      Multiple pairs sharing the same query cell each get their own
///      row entry with their own score.
///   4. Column-normalise so each query cell's column sums to 1.
///
/// The output is a CSC matrix of shape `(n_pairs, n_query)` with at most
/// `k_weight * pairs_per_shared_cell_max` non-zeros per column.
///
/// ### Params
///
/// * `query_embed` - Query batch embedding in the current merge-space,
///   `(n_query, dims)`
/// * `anchors` - The anchor set for this batch pair
/// * `k_weight` - Neighbours per query cell (Seurat default 100)
/// * `sd` - Kernel bandwidth divisor (Seurat default 1.0); the kernel
///   scale is `(2 / sd)^2`
/// * `knn_params` - kNN backend parameters
/// * `seed` - Random seed for the kNN backend
///
/// ### Returns
///
/// CSC sparse matrix `(n_pairs, n_query)`, column-normalised.
pub fn find_weights(
    query_embed: MatRef<f32>,
    anchors: &AnchorSet,
    k_weight: usize,
    sd: f32,
    knn_params: &KnnParams,
    seed: usize,
) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
    use crate::single_cell::sc_batch_correction::batch_utils::batch_knn_search;

    let n_query = query_embed.nrows();
    let n_pairs = anchors.pairs.len();

    // Unique query anchor cells + per-cell list of pair indices sharing it.
    // Sorted ascending on the unique cell id for reproducibility.
    let (unique_query_cells, pairs_per_unique): (Vec<u32>, Vec<Vec<u32>>) = {
        let mut by_q: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        for (i, &(_r, q)) in anchors.pairs.iter().enumerate() {
            by_q.entry(q).or_default().push(i as u32);
        }
        let mut items: Vec<(u32, Vec<u32>)> = by_q.into_iter().collect();
        items.sort_by_key(|(q, _)| *q);
        items.into_iter().unzip()
    };
    let n_unique = unique_query_cells.len();

    if n_unique == 0 || n_pairs == 0 {
        // No anchors → empty (n_pairs, n_query) CSC.
        return Ok(CompressedSparseData2 {
            data: Vec::new(),
            indices: Vec::new(),
            indptr: vec![0_u32; n_query + 1],
            cs_type: CompressedSparseFormat::Csc,
            data_2: None,
            shape: (n_pairs, n_query),
        });
    }

    // Anchor-cell embedding by row-gathering query_embed at unique cells.
    let anchor_embed = Mat::from_fn(n_unique, query_embed.ncols(), |i, j| {
        query_embed[(unique_query_cells[i] as usize, j)]
    });

    let safe_k = k_weight.min(n_unique);
    let (nn_idx, nn_dist) =
        batch_knn_search(query_embed, anchor_embed.as_ref(), safe_k, knn_params, seed, 0)?;

    let kernel_denom = (2.0_f32 / sd.max(1e-6)).powi(2);

    // Per-column sparse entries: (pair_index, weight).
    let per_col: Vec<Vec<(u32, f32)>> = (0..n_query)
        .into_par_iter()
        .map(|c| {
            let d_max = nn_dist[c].iter().copied().fold(0.0_f32, f32::max);
            // If all neighbours are at zero distance, treat every unique
            // cell as maximally close: skip the divide-by-zero and use
            // d_norm = 1.
            let d_max_inv = if d_max > 0.0 { 1.0 / d_max } else { 0.0 };

            let mut entries: Vec<(u32, f32)> = Vec::with_capacity(safe_k);
            let mut total = 0.0_f32;
            for (&unique_idx, &d) in nn_idx[c].iter().zip(nn_dist[c].iter()) {
                let d_norm = if d_max > 0.0 {
                    1.0 - d * d_max_inv
                } else {
                    1.0
                };
                for &p in pairs_per_unique[unique_idx].iter() {
                    let score = anchors.scores[p as usize].max(0.0);
                    let arg = d_norm * score / kernel_denom;
                    let w = 1.0 - (-arg).exp();
                    if w > 0.0 {
                        entries.push((p, w));
                        total += w;
                    }
                }
            }
            if total > 0.0 {
                let inv = 1.0 / total;
                for (_, v) in entries.iter_mut() {
                    *v *= inv;
                }
            }
            entries.sort_by_key(|(i, _)| *i);
            entries
        })
        .collect();

    // Pack per-column entries into CSC.
    let mut data: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut indptr: Vec<u32> = Vec::with_capacity(n_query + 1);
    indptr.push(0);
    for col_entries in per_col.iter() {
        for &(row, v) in col_entries {
            indices.push(row);
            data.push(v);
        }
        indptr.push(data.len() as u32);
    }

    Ok(CompressedSparseData2 {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csc,
        data_2: None,
        shape: (n_pairs, n_query),
    })
}

////////////////
// Correction //
////////////////

/// Apply Seurat's anchor-based correction to a query embedding.
///
/// Computes `corrected = query_embed - W^T @ Delta` where `W` is the CSC
/// weight matrix from [find_weights], shape `(n_pairs, n_query)`, and
/// `Delta` is the per-pair integration matrix from
/// [build_integration_matrix], shape `(n_pairs, dims)`. Reference cells
/// are untouched here; the caller concatenates them post-hoc.
///
/// ### Params
///
/// * `query_embed` - Query embedding, `(n_query, dims)`
/// * `weights` - CSC weight matrix, `(n_pairs, n_query)`
/// * `delta` - Per-anchor-pair integration matrix, `(n_pairs, dims)`
///
/// ### Returns
///
/// Corrected query embedding, same shape as `query_embed`.
pub fn apply_correction(
    query_embed: MatRef<f32>,
    weights: &CompressedSparseData2<f32>,
    delta: MatRef<f32>,
) -> Mat<f32> {
    let n_query = query_embed.nrows();
    let dims = query_embed.ncols();

    let mut out = Mat::<f32>::zeros(n_query, dims);

    // One CSC column = one query cell; parallelise across cells.
    let cols: Vec<Vec<f32>> = (0..n_query)
        .into_par_iter()
        .map(|c| {
            let start = weights.indptr[c] as usize;
            let end = weights.indptr[c + 1] as usize;
            let mut contrib = vec![0.0_f32; dims];
            for idx in start..end {
                let row = weights.indices[idx] as usize;
                let w = weights.data[idx];
                for d in 0..dims {
                    contrib[d] += w * delta[(row, d)];
                }
            }
            contrib
        })
        .collect();

    for c in 0..n_query {
        for d in 0..dims {
            out[(c, d)] = query_embed[(c, d)] - cols[c][d];
        }
    }

    out
}

/////////////////
// Merge order //
/////////////////

/// Greedy hierarchical merge order from a pairwise anchor-count matrix.
///
/// At each step, pick the currently-active batch pair with the highest
/// anchor count and merge them (query into reference). The merged group
/// keeps the reference batch id; anchor counts of the merged group against
/// each remaining group are updated as `min(count[ref, k], count[query, k])`
/// (Seurat convention keeps small groups from dominating).
///
/// ### Params
///
/// * `anchor_counts` - Symmetric `n_batches x n_batches` count matrix
///   stored as row-major `Vec<Vec<u32>>`
///
/// ### Returns
///
/// A vector of `(ref_batch, query_batch)` merges in application order.
/// Length is `n_batches - 1`. On ties, the lowest `(i, j)` pair wins for
/// determinism.
pub fn build_sample_tree(anchor_counts: &[Vec<u32>]) -> Vec<(usize, usize)> {
    let n = anchor_counts.len();
    if n <= 1 {
        return Vec::new();
    }

    let mut counts: Vec<Vec<u32>> = anchor_counts.to_vec();

    let mut active: Vec<bool> = vec![true; n];
    let mut merges: Vec<(usize, usize)> = Vec::with_capacity(n - 1);

    for _ in 0..(n - 1) {
        let mut best: Option<(u32, usize, usize)> = None;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let c = counts[i][j];
                let take = match best {
                    None => true,
                    Some((bc, _, _)) => c > bc,
                };
                if take {
                    best = Some((c, i, j));
                }
            }
        }

        let (_, i, j) = best.expect("at least one active pair remains");
        merges.push((i, j));
        active[j] = false;

        for k in 0..n {
            if !active[k] || k == i {
                continue;
            }
            counts[i][k] = counts[i][k].min(counts[j][k]);
            counts[k][i] = counts[i][k];
        }
    }

    merges
}

////////////////
// Tree merge //
////////////////

/// One active group during the tree merge. Starts as a single batch and
/// absorbs query groups as merges proceed. Origin tracking lets us translate
/// original per-batch anchor pairs to the current row layout after any
/// number of prior merges.
///
/// ### Fields
///
/// * `embed` - Current embedding, rows are cells in the current merge order
/// * `origin` - For each row, the `(original_batch, original_index_in_batch)`
/// * `union_cell_indices` - For each row, the caller-supplied "union" cell
///   index used to reorder back into the original input order
#[derive(Clone, Debug)]
struct MergeGroup {
    embed: Mat<f32>,
    origin: Vec<(u16, u32)>,
    union_cell_indices: Vec<usize>,
}

impl MergeGroup {
    fn new_leaf(batch_id: u16, embed: Mat<f32>, union_indices: &[usize]) -> Self {
        let n = embed.nrows();
        assert_eq!(union_indices.len(), n, "union_indices length must match embed rows");
        let origin = (0..n).map(|i| (batch_id, i as u32)).collect();
        Self {
            embed,
            origin,
            union_cell_indices: union_indices.to_vec(),
        }
    }

    fn active_batches(&self) -> FxHashSet<u16> {
        self.origin.iter().map(|&(b, _)| b).collect()
    }

    fn origin_map(&self) -> FxHashMap<(u16, u32), u32> {
        self.origin
            .iter()
            .enumerate()
            .map(|(row, &orig)| (orig, row as u32))
            .collect()
    }
}

/// Tree-merge per-batch embeddings using pre-computed anchor sets and a
/// merge order. Applies Seurat's anchor correction on the embedding at
/// every merge step. The reference group's embedding is left unchanged;
/// only the query group is corrected via [apply_correction].
///
/// The [AnchorSet]s in `anchors_by_pair` are keyed by `(a, b)` with
/// `a < b` (canonical order). When a merge step wants anchors between two
/// current groups, this function translates each anchor's original
/// per-batch cell index into the current group's row via the tracked
/// `origin` map.
///
/// ### Params
///
/// * `per_batch_embed` - Base embedding per batch, `n_cells_in_batch x dims`
/// * `per_batch_union_indices` - Union-space cell index per row of each
///   `per_batch_embed[i]`, used for final reordering
/// * `anchors_by_pair` - Pre-computed anchors, keyed by canonical
///   `(a, b)` with `a < b`
/// * `merge_order` - Sequence of `(ref_group, query_group)` merges (from
///   [build_sample_tree])
/// * `k_weight` - Neighbours per query cell for [find_weights]
/// * `sd` - Kernel bandwidth divisor
/// * `knn_params` - kNN backend parameters
/// * `seed` - Random seed
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `(merged_embed, union_cell_indices)` where `merged_embed[i]` is the
/// corrected embedding of the cell at `union_cell_indices[i]` in the
/// caller's union.
#[allow(clippy::too_many_arguments)]
pub fn tree_merge_embeddings(
    per_batch_embed: Vec<Mat<f32>>,
    per_batch_union_indices: Vec<Vec<usize>>,
    anchors_by_pair: &FxHashMap<(usize, usize), AnchorSet>,
    merge_order: &[(usize, usize)],
    k_weight: usize,
    sd: f32,
    knn_params: &KnnParams,
    seed: usize,
    verbose: usize,
) -> Result<(Mat<f32>, Vec<usize>), BixverseErrors> {
    assert_eq!(per_batch_embed.len(), per_batch_union_indices.len());
    let verbosity = parse_verbosity_level(verbose);

    let mut groups: Vec<Option<MergeGroup>> = per_batch_embed
        .into_iter()
        .zip(per_batch_union_indices)
        .enumerate()
        .map(|(b, (embed, union))| Some(MergeGroup::new_leaf(b as u16, embed, &union)))
        .collect();

    for (step, &(ref_id, query_id)) in merge_order.iter().enumerate() {
        let ref_group = groups[ref_id]
            .clone()
            .expect("reference group should be active at merge step");
        let query_group = groups[query_id]
            .clone()
            .expect("query group should be active at merge step");

        let ref_map = ref_group.origin_map();
        let query_map = query_group.origin_map();
        let ref_batches = ref_group.active_batches();
        let query_batches = query_group.active_batches();

        let mut translated_pairs: Vec<(u32, u32)> = Vec::new();
        let mut translated_scores: Vec<f32> = Vec::new();

        for &b_r in &ref_batches {
            for &b_q in &query_batches {
                let (key, flip) = if (b_r as usize) < (b_q as usize) {
                    ((b_r as usize, b_q as usize), false)
                } else {
                    ((b_q as usize, b_r as usize), true)
                };
                let Some(anchors) = anchors_by_pair.get(&key) else {
                    continue;
                };
                for (i, &(a, b)) in anchors.pairs.iter().enumerate() {
                    // In canonical key (key.0, key.1): a lives in key.0, b in key.1.
                    // If flip, then key.0 == b_q and key.1 == b_r, so swap.
                    let (orig_ref, orig_query) = if flip { (b, a) } else { (a, b) };
                    let ref_row = ref_map.get(&(b_r, orig_ref)).copied();
                    let query_row = query_map.get(&(b_q, orig_query)).copied();
                    if let (Some(r), Some(q)) = (ref_row, query_row) {
                        translated_pairs.push((r, q));
                        translated_scores.push(anchors.scores[i]);
                    }
                }
            }
        }

        let merged_embed = if translated_pairs.is_empty() {
            if verbosity.normal_verbosity() {
                eprintln!(
                    "Tree merge step {}: no anchors between groups {} and {}; concatenating without correction",
                    step + 1,
                    ref_id,
                    query_id
                );
            }
            stack_vertical(&ref_group.embed, &query_group.embed)
        } else {
            let anchor_set = AnchorSet {
                pairs: translated_pairs,
                scores: translated_scores,
                batch_pair: (ref_id, query_id),
            };
            let delta = build_integration_matrix(
                &anchor_set,
                ref_group.embed.as_ref(),
                query_group.embed.as_ref(),
            );
            let weights = find_weights(
                query_group.embed.as_ref(),
                &anchor_set,
                k_weight,
                sd,
                knn_params,
                seed,
            )?;
            let corrected =
                apply_correction(query_group.embed.as_ref(), &weights, delta.as_ref());
            stack_vertical(&ref_group.embed, &corrected)
        };

        let mut merged_origin = ref_group.origin.clone();
        merged_origin.extend(query_group.origin.iter().copied());
        let mut merged_union = ref_group.union_cell_indices.clone();
        merged_union.extend(query_group.union_cell_indices.iter().copied());

        groups[ref_id] = Some(MergeGroup {
            embed: merged_embed,
            origin: merged_origin,
            union_cell_indices: merged_union,
        });
        groups[query_id] = None;
    }

    let survivor = groups
        .into_iter()
        .find_map(|g| g)
        .expect("at least one group should survive the merge");
    Ok((survivor.embed, survivor.union_cell_indices))
}

/// Concatenate two matrices vertically (`top` on top, `bottom` underneath).
/// Both must have the same number of columns; this is asserted.
///
/// ### Params
///
/// * `top` - Upper block
/// * `bottom` - Lower block
///
/// ### Returns
///
/// New `(top.nrows() + bottom.nrows(), top.ncols())` matrix.
fn stack_vertical(top: &Mat<f32>, bottom: &Mat<f32>) -> Mat<f32> {
    assert_eq!(top.ncols(), bottom.ncols());
    let n_top = top.nrows();
    let n_bot = bottom.nrows();
    let ncols = top.ncols();
    Mat::from_fn(n_top + n_bot, ncols, |i, j| {
        if i < n_top {
            top[(i, j)]
        } else {
            bottom[(i - n_top, j)]
        }
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_find_anchor_pairs_mutual() {
        // a=0 has b-neighbours [1, 2]; b=1 has a-neighbours [0]; b=2 has a-neighbours [3]
        // Only (0, 1) is mutual.
        let nnab = vec![vec![1_usize, 2], vec![], vec![], vec![]];
        let nnba = vec![vec![], vec![0_usize], vec![3_usize]];
        let pairs = find_anchor_pairs(&nnab, &nnba, 2);
        assert_eq!(pairs, vec![(0_u32, 1_u32)]);
    }

    #[test]
    fn test_rescale_scores() {
        let raw = vec![0_u32, 5, 10, 100];
        let out = rescale_scores(&raw);
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(*v >= 0.0 && *v <= 1.0, "{v} out of [0, 1]");
        }
        // Largest gets rescaled to 1.0 (100 > q_0.9 → clamps to 1.0)
        assert_relative_eq!(out[3], 1.0);
    }

    #[test]
    fn test_score_anchors_toy() {
        // Two anchors: (0, 0) and (1, 1). Design a merged neighbourhood so
        // (0, 0) shares more neighbours than (1, 1).
        let pairs = vec![(0_u32, 0_u32), (1_u32, 1_u32)];
        // With offset_b = 3:
        //   a=0: nnaa=[1] → {1}, nnab=[0,1] → {3, 4}
        //   b=0: nnbb=[1,2] → {4, 5}, nnba=[1] → {1}
        //   intersect({1,3,4}, {1,4,5}) = {1,4} → 2
        //   a=1: nnaa=[0] → {0}, nnab=[2] → {5}
        //   b=1: nnbb=[0] → {3}, nnba=[2] → {2}
        //   intersect({0,5}, {2,3}) = {} → 0
        let nnaa = vec![vec![1_usize], vec![0_usize]];
        let nnab = vec![vec![0_usize, 1_usize], vec![2_usize]];
        let nnba = vec![vec![1_usize], vec![2_usize], vec![]];
        let nnbb = vec![vec![1_usize, 2_usize], vec![0_usize], vec![]];
        let raw: Vec<u32> = {
            // Reproduce the raw-count step so the test doesn't lean on the
            // rescale specifics.
            let mut out = Vec::new();
            for &(a, b) in &pairs {
                let ai = a as usize;
                let bi = b as usize;
                let mut s: FxHashSet<usize> = nnaa[ai].iter().copied().collect();
                for &x in &nnab[ai] {
                    s.insert(x + 3);
                }
                let mut t: FxHashSet<usize> = nnbb[bi].iter().map(|&x| x + 3).collect();
                for &x in &nnba[bi] {
                    t.insert(x);
                }
                out.push(s.intersection(&t).count() as u32);
            }
            out
        };
        assert_eq!(raw, vec![2, 0]);
        let scored = score_anchors(&pairs, &nnaa, &nnab, &nnba, &nnbb, 3, 3);
        assert!(scored[0] > scored[1]);
    }

    #[test]
    fn test_build_integration_matrix_per_pair() {
        // Two anchors: (r=0, q=5) with score 0.4, (r=1, q=5) with score 0.9.
        // Under the per-pair formulation, integration matrix rows are
        // ref_embed[r_i] - query_embed[q_i]; both use the SAME query cell
        // 5 here so both rows subtract query_embed[5, :].
        let anchors = AnchorSet {
            pairs: vec![(0, 5), (1, 5)],
            scores: vec![0.4, 0.9],
            batch_pair: (0, 1),
        };
        let ref_e: Mat<f32> = Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f32);
        let query_e: Mat<f32> = Mat::from_fn(6, 3, |_, j| 0.1 * j as f32);

        let delta = build_integration_matrix(&anchors, ref_e.as_ref(), query_e.as_ref());
        assert_eq!(delta.nrows(), 2);
        assert_eq!(delta.ncols(), 3);
        for d in 0..3 {
            assert_relative_eq!(delta[(0, d)], ref_e[(0, d)] - query_e[(5, d)], epsilon = 1e-6);
            assert_relative_eq!(delta[(1, d)], ref_e[(1, d)] - query_e[(5, d)], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_apply_correction_identity_when_weights_zero() {
        // Empty CSC weight matrix (shape n_pairs x n_query = 2 x 4).
        // apply_correction should return the query embedding unchanged.
        let query: Mat<f32> = Mat::from_fn(4, 3, |i, j| (i as f32) + 0.1 * j as f32);
        let delta: Mat<f32> = Mat::from_fn(2, 3, |_, _| 5.0);
        let weights = CompressedSparseData2 {
            data: Vec::<f32>::new(),
            indices: Vec::<u32>::new(),
            indptr: vec![0_u32; 5],
            cs_type: CompressedSparseFormat::Csc,
            data_2: None,
            shape: (2, 4),
        };
        let out = apply_correction(query.as_ref(), &weights, delta.as_ref());
        for i in 0..4 {
            for j in 0..3 {
                assert_relative_eq!(out[(i, j)], query[(i, j)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_build_sample_tree_three_batches() {
        // Anchor count matrix:
        //   0-1: 100 (highest)
        //   0-2: 10
        //   1-2: 5
        // Expected merge order: (0, 1), then (0, 2)
        let counts = vec![
            vec![0_u32, 100, 10],
            vec![100, 0, 5],
            vec![10, 5, 0],
        ];
        let merges = build_sample_tree(&counts);
        assert_eq!(merges.len(), 2);
        assert_eq!(merges[0], (0, 1));
        assert_eq!(merges[1], (0, 2));
    }

    #[test]
    fn test_rescale_scores_single_and_uniform() {
        // Regression test for the silent-no-op case: a single anchor pair,
        // or all-equal counts, should be treated as fully-trusted (score 1.0)
        // rather than rescaled to 0.
        assert_eq!(rescale_scores(&[7]), vec![1.0]);
        assert_eq!(rescale_scores(&[5, 5, 5, 5]), vec![1.0; 4]);
    }

    #[test]
    fn test_find_weights_close_anchor_wins() {
        // Regression test for kernel direction: close anchors must get
        // higher weight than farther anchors. Before the per-row
        // `1 - d / d_max` transform, raw L2 distances flipped the
        // ordering — far anchors dominated.
        //
        // Seurat's transform makes `d_norm = 0` for the k-th neighbour so
        // its weight is exactly zero. We use three anchor cells and
        // check the two non-zero survivors: close > middle. The far one
        // is expected to drop out.
        let anchors = AnchorSet {
            pairs: vec![(0, 1), (1, 2), (2, 3)],
            scores: vec![1.0, 1.0, 1.0],
            batch_pair: (0, 1),
        };
        let query_embed = Mat::from_fn(4, 1, |i, _| match i {
            0 => 0.0_f32,
            1 => 1.0,
            2 => 5.0,
            3 => 100.0,
            _ => 0.0,
        });
        let knn_params = KnnParams {
            knn_method: "exhaustive".to_string(),
            ann_dist: "euclidean".to_string(),
            ..KnnParams::default()
        };
        let weights =
            find_weights(query_embed.as_ref(), &anchors, 3, 1.0, &knn_params, 42).unwrap();

        assert_eq!(weights.shape, (3, 4));
        let start = weights.indptr[0] as usize;
        let end = weights.indptr[1] as usize;
        let col0: Vec<(u32, f32)> = (start..end)
            .map(|i| (weights.indices[i], weights.data[i]))
            .collect();
        // Two anchors should survive (nearest two of three); the k-th
        // (furthest, x=100) normalises to weight 0 and is filtered out.
        assert_eq!(col0.len(), 2, "expected two non-zero anchor weights, got {col0:?}");
        let close = col0.iter().find(|(i, _)| *i == 0).unwrap().1;
        let middle = col0.iter().find(|(i, _)| *i == 1).unwrap().1;
        assert!(
            close > middle,
            "close anchor weight {close} must exceed middle anchor weight {middle}"
        );
        assert!(col0.iter().all(|(i, _)| *i != 2), "far anchor should have zero weight");
    }

    #[test]
    fn test_find_weights_columns_sum_to_one() {
        // Every query-cell column with any non-zero entries must sum to 1
        // (column-stochastic invariant). Mix of scores catches per-pair
        // normalisation bugs.
        let anchors = AnchorSet {
            pairs: vec![(0, 1), (1, 2), (2, 3)],
            scores: vec![0.5, 0.7, 0.9],
            batch_pair: (0, 1),
        };
        let query_embed = Mat::from_fn(4, 2, |i, j| i as f32 + j as f32 * 0.1);
        let knn_params = KnnParams {
            knn_method: "exhaustive".to_string(),
            ann_dist: "euclidean".to_string(),
            ..KnnParams::default()
        };
        let weights =
            find_weights(query_embed.as_ref(), &anchors, 3, 1.0, &knn_params, 42).unwrap();
        for c in 0..4 {
            let start = weights.indptr[c] as usize;
            let end = weights.indptr[c + 1] as usize;
            if end > start {
                let total: f32 = weights.data[start..end].iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-4,
                    "column {c} sums to {total}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn test_tree_merge_exercises_flip_branch() {
        // With 3 batches merged in order (0, 2) then (0, 1), the second
        // merge sees group 0 = {batch 0, batch 2} vs query group = {batch 1}.
        // When iterating batch pairs inside, `b_r = 2, b_q = 1` triggers
        // the canonical-key flip (b_r > b_q). Confirms the flip branch
        // resolves anchor indices correctly without panicking.
        let e0 = Mat::from_fn(3, 2, |i, j| if j == 0 { i as f32 * 0.1 } else { 0.0 });
        let e1 = Mat::from_fn(3, 2, |i, j| if j == 0 { 10.0 + i as f32 * 0.1 } else { 0.0 });
        let e2 = Mat::from_fn(3, 2, |i, j| if j == 0 { 0.05 + i as f32 * 0.1 } else { 0.0 });

        let mut anchors_by_pair: FxHashMap<(usize, usize), AnchorSet> = FxHashMap::default();
        anchors_by_pair.insert(
            (0, 1),
            AnchorSet {
                pairs: vec![(0, 0)],
                scores: vec![1.0],
                batch_pair: (0, 1),
            },
        );
        anchors_by_pair.insert(
            (0, 2),
            AnchorSet {
                pairs: vec![(0, 0), (1, 1), (2, 2)],
                scores: vec![1.0, 1.0, 1.0],
                batch_pair: (0, 2),
            },
        );
        anchors_by_pair.insert(
            (1, 2),
            AnchorSet {
                pairs: vec![(0, 0)],
                scores: vec![1.0],
                batch_pair: (1, 2),
            },
        );

        // Force merge order (0, 2) then (0, 1) via anchor counts.
        let counts = vec![vec![0, 1, 3], vec![1, 0, 1], vec![3, 1, 0]];
        let merge_order = build_sample_tree(&counts);
        assert_eq!(merge_order[0], (0, 2));

        let knn_params = KnnParams {
            knn_method: "exhaustive".to_string(),
            ann_dist: "euclidean".to_string(),
            ..KnnParams::default()
        };
        let (out, index_map) = tree_merge_embeddings(
            vec![e0, e1, e2],
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]],
            &anchors_by_pair,
            &merge_order,
            3,
            1.0,
            &knn_params,
            42,
            0,
        )
        .unwrap();
        assert_eq!(out.nrows(), 9);
        assert_eq!(out.ncols(), 2);
        assert_eq!(index_map.len(), 9);
        // Sanity: index_map covers each union cell exactly once.
        let mut sorted = index_map.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn test_build_sample_tree_updates_counts_by_min() {
        // Two batches merge first, then the merged-vs-remaining count is
        // min of the two contributing counts. With only three batches the
        // second merge is forced regardless of that update.
        let counts = vec![
            vec![0_u32, 50, 20],
            vec![50, 0, 5],
            vec![20, 5, 0],
        ];
        let merges = build_sample_tree(&counts);
        assert_eq!(merges, vec![(0, 1), (0, 2)]);
    }
}
