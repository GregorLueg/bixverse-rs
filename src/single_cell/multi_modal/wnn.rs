//! Weighted Nearest Neighbour analysis (WNN), per Hao et al. (Cell 2021).
//!
//! Input:  two per-modality embeddings (typically PCA, L2-normalised) plus
//!         per-modality kNN indices and distances at `knn_range`.
//! Output: a multimodal kNN graph (indices + pseudo-distances) and per-cell
//!         modality weights.
//!
//! Restricted to two modalities.

use ann_search_rs::utils::dist::SimdDistance;
use faer::MatRef;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_processing::snn::*;

////////////////////
// Structs, Enums //
////////////////////

/// How the per-cell kernel bandwidth (sigma) is computed.
#[derive(Clone, Copy, Debug, Default)]
pub enum SigmaMethod {
    /// Use mean Euclidean distance to the k cells with the smallest non-zero
    /// SNN edge weight, in embedding space (Seurat equivalent:
    /// `snn.far.nn = TRUE`).
    #[default]
    SnnFarthest,
    /// Use distance to a fixed kNN index (Seurat equivalent:
    /// `snn.far.nn = FALSE`).
    SigmaIdx,
}

/// Parse the sigma method
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// The option of the [SigmaMethod].
pub fn parse_sigma_method(s: &str) -> Option<SigmaMethod> {
    match s.to_lowercase().as_str() {
        "snn" => Some(SigmaMethod::SnnFarthest),
        "sigma" => Some(SigmaMethod::SigmaIdx),
        _ => None,
    }
}

/// Per-modality input. Embedding ideally L2-normalised; kNN must have self
/// removed and at least `knn_range` neighbours per cell.
pub struct ModalityInput<'a> {
    /// The embedding used to generate the kNN graph (PCA, batch-corrected
    /// embedding)
    pub embedding: MatRef<'a, f32>,
    /// The indices to the nearest neighbours with self removed
    pub knn_indices: &'a [Vec<usize>],
    /// The distances to the nearest neighbours with self removed
    pub knn_distances: &'a [Vec<f32>],
}

/// WNN output.
pub struct WnnResult {
    /// The indices of the weighted nearest neighbour graph
    pub wnn_indices: Vec<Vec<usize>>,
    /// The distances of the weighted nearest neighbour graph
    pub wnn_distances: Vec<Vec<f32>>,
    /// The modality weights
    pub modality_weights: Vec<Vec<f32>>,
}

////////////
// Params //
////////////

/// WNN parameters.
#[derive(Clone, Debug)]
pub struct WnnParams {
    /// Final number of multimodal neighbours per cell.
    pub k_nn: usize,
    /// Candidate pool size per modality. Each cell's kNN input must contain
    /// at least this many neighbours.
    pub knn_range: usize,
    /// Bandwidth method.
    pub sigma_method: SigmaMethod,
    /// `SigmaIdx` only: 0-based kNN index for bandwidth. Default k_nn - 1.
    pub sigma_idx: usize,
    /// `SnnFarthest` only: kNN size used to build the SNN graph. Default k_nn.
    pub s_nn: usize,
    /// Multiplier on sigma.
    pub sd_scale: f32,
    /// Kernel exponent power.
    pub kernel_power: f32,
    /// Cross-modality kernel stabiliser.
    pub cross_const: f32,
    /// Min sigma value (avoids div by zero).
    pub sigma_floor: f32,
}

/// Default implementation
impl Default for WnnParams {
    fn default() -> Self {
        Self {
            k_nn: 20,
            knn_range: 200,
            sigma_method: SigmaMethod::SnnFarthest,
            sigma_idx: 19,
            s_nn: 20,
            sd_scale: 1.0,
            kernel_power: 1.0,
            cross_const: 1e-4,
            sigma_floor: 1e-8,
        }
    }
}

/////////////
// Helpers //
/////////////

/// Row to vector
///
/// Copies a single row from a matrix reference into a heap-allocated vector.
///
/// ### Params
///
/// * `mat` - The source matrix.
/// * `i` - Row index to copy.
///
/// ### Returns
///
/// A `Vec<f32>` containing the values of row `i`.
#[inline]
fn row_to_vec(mat: MatRef<f32>, i: usize) -> Vec<f32> {
    let nc = mat.ncols();
    let mut v = Vec::with_capacity(nc);
    for j in 0..nc {
        v.push(mat[(i, j)]);
    }
    v
}

/// Mean of rows
///
/// Computes the element-wise mean of a subset of rows from a matrix.
///
/// ### Params
///
/// * `mat` - The source matrix.
/// * `indices` - Row indices to average over.
///
/// ### Returns
///
/// A `Vec<f32>` of length `mat.ncols()` containing the column-wise mean.
#[inline]
fn mean_of_rows(mat: MatRef<f32>, indices: &[usize]) -> Vec<f32> {
    let nc = mat.ncols();
    let mut acc = vec![0.0f32; nc];
    for &i in indices {
        for j in 0..nc {
            acc[j] += mat[(i, j)];
        }
    }
    let inv_n = 1.0 / indices.len() as f32;
    for v in &mut acc {
        *v *= inv_n;
    }
    acc
}

/// Euclidean distance
///
/// Computes the Euclidean distance between two equal-length slices using SIMD
/// acceleration.
///
/// ### Params
///
/// * `a` - First vector.
/// * `b` - Second vector.
///
/// ### Returns
///
/// The Euclidean distance as `f32`.
#[inline]
fn euclid(a: &[f32], b: &[f32]) -> f32 {
    f32::euclidean_simd(a, b).sqrt()
}

/// Build SNN graph for sigma calculation
///
/// Constructs a shared nearest-neighbour graph using Jaccard similarity with no
/// pruning, matching Seurat's `FindModalityWeights` defaults. Input KNN indices
/// are reordered into column-major layout before graph construction.
///
/// ### Params
///
/// * `knn_indices` - Per-cell KNN index lists.
/// * `s_nn` - Number of neighbours to use for SNN construction.
/// * `n_cells` - Total number of cells.
/// * `verbose` - Print timing information.
///
/// ### Returns
///
/// A `SparseGraph<f32>` of SNN edge weights.
fn build_snn_for_sigma(
    knn_indices: &[Vec<usize>],
    s_nn: usize,
    n_cells: usize,
    verbose: bool,
) -> SparseGraph<f32> {
    // generate_snn_full needs column-major flat input
    let mut flat = vec![0usize; s_nn * n_cells];
    for i in 0..n_cells {
        for j in 0..s_nn {
            flat[j * n_cells + i] = knn_indices[i][j];
        }
    }
    let (edges, weights) = generate_snn_full(
        &flat,
        s_nn,
        n_cells,
        0.0,
        SnnSimilarityMethod::Intersection,
        verbose,
    );
    snn_edges_to_sparse_graph(&edges, &weights, n_cells)
}

/// Per-cell sigma from SNN-farthest neighbours
///
/// Computes a per-cell bandwidth estimate as the mean Euclidean distance
/// (offset by the nearest-neighbour distance) to the `k_far` cells with the
/// smallest non-zero SNN edge weight. Ties at the `k_far`-th position are
/// retained, matching Seurat's `ComputeSNNwidth` behaviour.
///
/// ### Params
///
/// * `snn` - The pre-built SNN graph.
/// * `embedding` - Cell embedding matrix.
/// * `i` - Index of the query cell.
/// * `nearest` - Distance to the nearest neighbour of cell `i`.
/// * `k_far` - Number of far SNN neighbours to average over.
/// * `sd_scale` - Scaling factor applied to the mean distance.
/// * `sigma_floor` - Minimum sigma value returned.
///
/// ### Returns
///
/// The sigma estimate for cell `i` as `f32`.
fn sigma_from_snn(
    snn: &SparseGraph<f32>,
    embedding: MatRef<f32>,
    i: usize,
    nearest: f32,
    k_far: usize,
    sd_scale: f32,
    sigma_floor: f32,
) -> f32 {
    let (nbrs, weights) = snn.get_neighbours(i);
    if nbrs.is_empty() {
        return sigma_floor;
    }

    let mut pairs: Vec<(f32, usize)> = weights
        .iter()
        .zip(nbrs.iter())
        .map(|(&w, &n)| (w, n))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let take = k_far.min(pairs.len());
    let threshold = pairs[take - 1].0;
    let selected: Vec<usize> = pairs
        .iter()
        .take_while(|(w, _)| *w <= threshold)
        .map(|(_, c)| *c)
        .collect();

    let row_i = row_to_vec(embedding, i);
    let mut total = 0.0f32;
    for &c in &selected {
        let row_c = row_to_vec(embedding, c);
        total += (euclid(&row_i, &row_c) - nearest).max(0.0);
    }
    let mean = total / selected.len() as f32;
    (mean * sd_scale).max(sigma_floor)
}

//////////
// Main //
//////////

/// Compute weighted nearest-neighbour graph
///
/// Implements the WNN algorithm over two modalities. Computes per-cell modality
/// weights by comparing within-modality and cross-modality neighbourhood
/// distances scaled by a per-cell bandwidth (sigma). These weights are then
/// used to combine affinity scores from both modalities into a single fused KNN
/// graph.
///
/// ### Params
///
/// * `modalities` - Array of exactly two `ModalityInput` structs, each
///   carrying an embedding matrix, KNN indices, and KNN distances.
/// * `params` - Algorithm hyperparameters; see `WnnParams`.
/// * `verbose` - Print timing information.
///
/// ### Returns
///
/// A `WnnResult` containing fused KNN indices, fused KNN distances, and
/// per-cell modality weights for each of the two modalities.
pub fn compute_wnn(
    modalities: [ModalityInput<'_>; 2],
    params: &WnnParams,
    verbose: bool,
) -> WnnResult {
    let n_cells = modalities[0].embedding.nrows();
    assert_eq!(
        n_cells,
        modalities[1].embedding.nrows(),
        "both modalities must have the same number of cells"
    );
    assert!(params.k_nn <= params.knn_range);
    assert!(params.sigma_idx < params.knn_range);
    assert!(params.s_nn <= params.knn_range);

    for (r, m) in modalities.iter().enumerate() {
        assert_eq!(m.knn_indices.len(), n_cells, "modality {r} knn row count");
        assert_eq!(
            m.knn_distances.len(),
            n_cells,
            "modality {r} knn dist row count"
        );
        if !m.knn_indices.is_empty() {
            assert!(
                m.knn_indices[0].len() >= params.knn_range,
                "modality {r} needs >= knn_range neighbours per cell"
            );
            assert!(m.knn_distances[0].len() >= params.knn_range);
        }
    }

    // Build SNNs up-front if needed
    let snn_graphs: Option<Vec<SparseGraph<f32>>> = match params.sigma_method {
        SigmaMethod::SnnFarthest => {
            let t = Instant::now();
            let g: Vec<_> = (0..2)
                .map(|r| {
                    build_snn_for_sigma(modalities[r].knn_indices, params.s_nn, n_cells, false)
                })
                .collect();
            if verbose {
                println!("WNN: per-modality SNN built in {:.2?}", t.elapsed());
            }
            Some(g)
        }
        SigmaMethod::SigmaIdx => None,
    };

    // ---- Phase 1: per-cell quantities + modality weights ----
    let t_phase1 = Instant::now();
    let k_nn = params.k_nn;

    let per_modality: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0..2)
        .map(|r| {
            let m = &modalities[r];
            let m_other = &modalities[1 - r];
            let snn = snn_graphs.as_ref().map(|gs| &gs[r]);

            let rows: Vec<(f32, f32, f32, f32)> = (0..n_cells)
                .into_par_iter()
                .map(|i| {
                    let nn_r = &m.knn_indices[i][..k_nn];
                    let nn_other = &m_other.knn_indices[i][..k_nn];
                    let nearest = m.knn_distances[i][0];

                    let row_i = row_to_vec(m.embedding, i);
                    let within = mean_of_rows(m.embedding, nn_r);
                    let cross = mean_of_rows(m.embedding, nn_other);

                    let d_within = (euclid(&row_i, &within) - nearest).max(0.0);
                    let d_cross = (euclid(&row_i, &cross) - nearest).max(0.0);

                    let sigma = match params.sigma_method {
                        SigmaMethod::SigmaIdx => {
                            let raw =
                                (m.knn_distances[i][params.sigma_idx] - nearest) * params.sd_scale;
                            raw.max(params.sigma_floor)
                        }
                        SigmaMethod::SnnFarthest => sigma_from_snn(
                            snn.unwrap(),
                            m.embedding,
                            i,
                            nearest,
                            params.k_nn,
                            params.sd_scale,
                            params.sigma_floor,
                        ),
                    };

                    (nearest, d_within, d_cross, sigma)
                })
                .collect();

            let mut nearest = Vec::with_capacity(n_cells);
            let mut within = Vec::with_capacity(n_cells);
            let mut cross = Vec::with_capacity(n_cells);
            let mut sigma = Vec::with_capacity(n_cells);
            for (n, w, c, s) in rows {
                nearest.push(n);
                within.push(w);
                cross.push(c);
                sigma.push(s);
            }
            (nearest, within, cross, sigma)
        })
        .collect();

    let scores: Vec<Vec<f32>> = (0..2)
        .map(|r| {
            let (_n, within, cross, sigma) = &per_modality[r];
            (0..n_cells)
                .map(|i| {
                    let kw = (-within[i] / sigma[i]).exp();
                    let kc = (-cross[i] / sigma[i]).exp();
                    (kw / (kc + params.cross_const)).clamp(0.0, 200.0)
                })
                .collect()
        })
        .collect();

    let mut weights = vec![vec![0.0f32; n_cells]; 2];
    for i in 0..n_cells {
        let e0 = scores[0][i].exp();
        let e1 = scores[1][i].exp();
        let s = e0 + e1;
        weights[0][i] = e0 / s;
        weights[1][i] = e1 / s;
    }

    if verbose {
        println!("WNN: modality weights in {:.2?}", t_phase1.elapsed());
    }

    // ---- Phase 2: WNN graph construction ----
    let t_phase2 = Instant::now();
    let kernel_power = params.kernel_power;

    let wnn: Vec<(Vec<usize>, Vec<f32>)> = (0..n_cells)
        .into_par_iter()
        .map(|i| {
            let r0 = &modalities[0].knn_indices[i][..params.knn_range];
            let r1 = &modalities[1].knn_indices[i][..params.knn_range];

            let mut seen: FxHashSet<usize> =
                FxHashSet::with_capacity_and_hasher(r0.len() + r1.len(), Default::default());
            let mut cands: Vec<usize> = Vec::with_capacity(r0.len() + r1.len());
            for &c in r0.iter().chain(r1.iter()) {
                if c != i && seen.insert(c) {
                    cands.push(c);
                }
            }

            let row_i_0 = row_to_vec(modalities[0].embedding, i);
            let row_i_1 = row_to_vec(modalities[1].embedding, i);

            let nearest_0 = per_modality[0].0[i];
            let nearest_1 = per_modality[1].0[i];
            let sigma_0 = per_modality[0].3[i];
            let sigma_1 = per_modality[1].3[i];
            let w0 = weights[0][i];
            let w1 = weights[1][i];

            let mut combined: Vec<f32> = Vec::with_capacity(cands.len());
            for &c in &cands {
                let rc_0 = row_to_vec(modalities[0].embedding, c);
                let rc_1 = row_to_vec(modalities[1].embedding, c);

                let d0 = (euclid(&row_i_0, &rc_0) - nearest_0).max(0.0);
                let d1 = (euclid(&row_i_1, &rc_1) - nearest_1).max(0.0);

                let a0 = (-((d0 / sigma_0).powf(kernel_power))).exp();
                let a1 = (-((d1 / sigma_1).powf(kernel_power))).exp();

                combined.push(w0 * a0 + w1 * a1);
            }

            let mut order: Vec<usize> = (0..cands.len()).collect();
            order.sort_unstable_by(|&a, &b| {
                combined[b]
                    .partial_cmp(&combined[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            order.truncate(k_nn);

            let mut idx_out = Vec::with_capacity(k_nn);
            let mut dist_out = Vec::with_capacity(k_nn);
            for o in order {
                idx_out.push(cands[o]);
                let aff = combined[o].clamp(0.0, 1.0);
                dist_out.push(((1.0 - aff) / 2.0).max(0.0).sqrt());
            }
            (idx_out, dist_out)
        })
        .collect();

    if verbose {
        println!("WNN: graph constructed in {:.2?}", t_phase2.elapsed());
    }

    let (wnn_indices, wnn_distances): (Vec<_>, Vec<_>) = wnn.into_iter().unzip();

    WnnResult {
        wnn_indices,
        wnn_distances,
        modality_weights: weights,
    }
}
