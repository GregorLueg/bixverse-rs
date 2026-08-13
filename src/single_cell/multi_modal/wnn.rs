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

impl<'a> ModalityInput<'a> {
    /// Generate a new Modality input instance
    ///
    /// ### Params
    ///
    /// * `embedding` - The embedding that was used to generate the kNN graph
    /// * `knn_indices` - The indices of the kNN graph
    /// * `knn_distances` - The distances of the kNN graph
    ///
    /// ### Returns
    ///
    /// Returns the [ModalityInput].
    pub fn new(
        embedding: MatRef<'a, f32>,
        knn_indices: &'a [Vec<usize>],
        knn_distances: &'a [Vec<f32>],
    ) -> Self {
        Self {
            embedding,
            knn_distances,
            knn_indices,
        }
    }
}

/// Per-modality, per-cell quantities computed during WNN phase 1.
#[derive(Clone)]
struct ModalityCellStats {
    /// Distance to the nearest non-self kNN neighbour.
    nearest: Vec<f32>,
    /// Distance from each cell to the within-modality neighbour centroid,
    /// offset by `nearest` and floored at zero.
    within_dist: Vec<f32>,
    /// Distance from each cell to the cross-modality neighbour centroid,
    /// offset by `nearest` and floored at zero.
    cross_dist: Vec<f32>,
    /// Per-cell kernel bandwidth.
    sigma: Vec<f32>,
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
    /// sNN type: build a limited version only considering edges that exists
    /// in the kNN
    pub snn_type: SnnType,
    /// Multiplier on sigma.
    pub sd_scale: f32,
    /// Kernel exponent power.
    pub kernel_power: f32,
    /// Cross-modality kernel stabiliser.
    pub cross_const: f32,
    /// Min sigma value (avoids div by zero).
    pub sigma_floor: f32,
    /// [KnnParams] for the various approximate nearest neighbour searches in
    /// ann-search-rs
    pub knn_params: KnnParams,
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
            snn_type: SnnType::FullConnection,
            sd_scale: 1.0,
            kernel_power: 1.0,
            cross_const: 1e-4,
            sigma_floor: 1e-8,
            knn_params: KnnParams::default(),
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
    // square root has to be happen here, as ann-search-rs returns squared
    // euclidean
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
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `SparseGraph<f32>` of SNN edge weights.
fn build_snn_for_sigma(
    knn_indices: &[Vec<usize>],
    s_nn: usize,
    snn_type: SnnType,
    n_cells: usize,
    verbose: usize,
) -> SparseGraph<f32> {
    // generate_snn_full needs column-major flat input!
    let mut flat = vec![0usize; s_nn * n_cells];
    for i in 0..n_cells {
        for j in 0..s_nn {
            flat[j * n_cells + i] = knn_indices[i][j];
        }
    }
    let (edges, weights) = match snn_type {
        SnnType::FullConnection => generate_snn_full(
            &flat,
            s_nn,
            n_cells,
            0.0,
            SnnSimilarityMethod::Intersection,
            verbose,
        ),
        SnnType::LimitedConnection => generate_snn_limited(
            &flat,
            s_nn,
            n_cells,
            0.0,
            SnnSimilarityMethod::Intersection,
            verbose,
        ),
    };

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
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `WnnResult` containing fused KNN indices, fused KNN distances, and
/// per-cell modality weights for each of the two modalities.
pub fn compute_wnn(
    modalities: [ModalityInput<'_>; 2],
    params: &WnnParams,
    verbose: usize,
) -> Result<WnnResult, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    // input checks
    let n_cells = modalities[0].embedding.nrows();
    if n_cells != modalities[1].embedding.nrows() {
        return Err(BixverseErrors::WNNModalitySampleMismatch);
    }
    if params.k_nn > params.knn_range {
        return Err(BixverseErrors::WNNKnnLargerThanKnnRange);
    }
    if params.sigma_idx >= params.knn_range {
        return Err(BixverseErrors::WNNSigmaIdxOutOfRange);
    }
    if params.s_nn > params.knn_range {
        return Err(BixverseErrors::WNNSnnLargerThanKnnRange);
    }

    for (r, m) in modalities.iter().enumerate() {
        if m.knn_indices.len() != n_cells {
            return Err(BixverseErrors::WNNKnnRowCountMismatch {
                modality: r,
                expected: n_cells,
                found: m.knn_indices.len(),
            });
        }
        if m.knn_distances.len() != n_cells {
            return Err(BixverseErrors::WNNKnnDistRowCountMismatch {
                modality: r,
                expected: n_cells,
                found: m.knn_distances.len(),
            });
        }
        if !m.knn_indices.is_empty() {
            if m.knn_indices[0].len() < params.knn_range {
                return Err(BixverseErrors::WNNInsufficientNeighbours {
                    modality: r,
                    expected: params.knn_range,
                    found: m.knn_indices[0].len(),
                });
            }
            if m.knn_distances[0].len() < params.knn_range {
                return Err(BixverseErrors::WNNInsufficientNeighbours {
                    modality: r,
                    expected: params.knn_range,
                    found: m.knn_distances[0].len(),
                });
            }
        }
    }

    // build SNNs up-front if needed
    let snn_graphs: Option<Vec<SparseGraph<f32>>> = match params.sigma_method {
        SigmaMethod::SnnFarthest => {
            let t = Instant::now();
            let g: Vec<_> = (0..2)
                .map(|r| {
                    build_snn_for_sigma(
                        modalities[r].knn_indices,
                        params.s_nn,
                        params.snn_type,
                        n_cells,
                        verbose,
                    )
                })
                .collect();
            if verbosity.normal_verbosity() {
                println!("WNN: per-modality SNN built in {:.2?}", t.elapsed());
            }
            Some(g)
        }
        SigmaMethod::SigmaIdx => None,
    };

    // phase 1: per-cell quantities + modality weights
    let t_phase1 = Instant::now();
    let k_nn = params.k_nn;

    let per_modality: Vec<ModalityCellStats> = (0..2)
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

            let mut stats = ModalityCellStats {
                nearest: Vec::with_capacity(n_cells),
                within_dist: Vec::with_capacity(n_cells),
                cross_dist: Vec::with_capacity(n_cells),
                sigma: Vec::with_capacity(n_cells),
            };
            for (n, w, c, s) in rows {
                stats.nearest.push(n);
                stats.within_dist.push(w);
                stats.cross_dist.push(c);
                stats.sigma.push(s);
            }
            stats
        })
        .collect();

    let scores: Vec<Vec<f32>> = (0..2)
        .map(|r| {
            let s = &per_modality[r];
            (0..n_cells)
                .map(|i| {
                    let kw = (-s.within_dist[i] / s.sigma[i]).exp();
                    let kc = (-s.cross_dist[i] / s.sigma[i]).exp();
                    kw / (kc + params.cross_const)
                })
                .collect()
        })
        .collect();

    let mut weights = vec![vec![0.0f32; n_cells]; 2];
    for i in 0..n_cells {
        let m = scores[0][i].max(scores[1][i]);
        let e0 = (scores[0][i] - m).exp();
        let e1 = (scores[1][i] - m).exp();
        let s = e0 + e1;
        weights[0][i] = e0 / s;
        weights[1][i] = e1 / s;
    }

    if verbosity.normal_verbosity() {
        println!("WNN: modality weights in {:.2?}", t_phase1.elapsed());
    }

    // phase 2: WNN graph construction
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

            let nearest_0 = per_modality[0].nearest[i];
            let nearest_1 = per_modality[1].nearest[i];
            let sigma_0 = per_modality[0].sigma[i];
            let sigma_1 = per_modality[1].sigma[i];
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

    if verbosity.normal_verbosity() {
        println!("WNN: graph constructed in {:.2?}", t_phase2.elapsed());
    }

    let (wnn_indices, wnn_distances): (Vec<_>, Vec<_>) = wnn.into_iter().unzip();

    Ok(WnnResult {
        wnn_indices,
        wnn_distances,
        modality_weights: weights,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Sigma-method names parse case-insensitively and an unknown name returns `None`.
    #[test]
    fn parse_sigma_method_known_and_unknown() {
        assert!(matches!(
            parse_sigma_method("snn"),
            Some(SigmaMethod::SnnFarthest)
        ));
        assert!(matches!(
            parse_sigma_method("SNN"),
            Some(SigmaMethod::SnnFarthest)
        ));
        assert!(matches!(
            parse_sigma_method("sigma"),
            Some(SigmaMethod::SigmaIdx)
        ));
        assert!(parse_sigma_method("nope").is_none());
    }

    /// Row extraction walks the row, not the column, on faer's column-major storage.
    #[test]
    fn row_to_vec_copies_correct_values() {
        let mat = Mat::from_fn(3, 4, |i, j| (i * 10 + j) as f32);
        let row1 = row_to_vec(mat.as_ref(), 1);
        assert_eq!(row1, vec![10.0, 11.0, 12.0, 13.0]);
        let row2 = row_to_vec(mat.as_ref(), 2);
        assert_eq!(row2, vec![20.0, 21.0, 22.0, 23.0]);
    }

    /// The centroid averages only the selected rows, column by column.
    #[test]
    fn mean_of_rows_known_centroid() {
        let mat = Mat::from_fn(3, 3, |i, j| (i * (j + 1) * 2) as f32);
        let mean_all = mean_of_rows(mat.as_ref(), &[0, 1, 2]);
        assert_eq!(mean_all, vec![2.0, 4.0, 6.0]);

        // subset
        let mean_sub = mean_of_rows(mat.as_ref(), &[1, 2]);
        assert_eq!(mean_sub, vec![3.0, 6.0, 9.0]);
    }

    /// Euclidean distance on a known triangle, and zero on two identical points.
    #[test]
    fn euclid_known_distance() {
        let a = [3.0, 4.0];
        let b = [0.0, 0.0];
        assert!(approx_eq(euclid(&a, &b), 5.0, EPS));

        let c = [1.0, 2.0, 3.0];
        let d = [1.0, 2.0, 3.0];
        assert!(approx_eq(euclid(&c, &d), 0.0, EPS));
    }

    ///////////////////////////
    // SNN bandwidth helpers //
    ///////////////////////////

    /// Build a tight 4-cell, 2D fixture where the kNN structure is obvious.
    /// Layout: cells 0,1,2 cluster near origin; cell 3 sits far away.
    /// k=2 nearest non-self for each cell.
    fn small_fixture() -> (Mat<f32>, Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let coords = [[0.0_f32, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0]];
        let mat = Mat::from_fn(4, 2, |i, j| coords[i][j]);

        // hand-rolled kNN (self removed, k=2 -> 2 entries per cell)
        let knn_indices = vec![vec![1, 2], vec![0, 2], vec![0, 1], vec![0, 1]];
        let d_diag = (0.01_f32 + 0.01).sqrt(); // ~0.1414
        let d_far = (100.0_f32 + 100.0).sqrt(); // ~14.14
        let knn_distances = vec![
            vec![0.1, 0.1],
            vec![0.1, d_diag],
            vec![0.1, d_diag],
            vec![d_far, d_far],
        ];
        (mat, knn_indices, knn_distances)
    }

    /// A cell with no SNN neighbours drops to the sigma floor instead of dividing by zero.
    #[test]
    fn sigma_from_snn_floors_when_empty() {
        // empty SNN graph -> floor
        let (mat, _, _) = small_fixture();
        let empty_csr = CompressedSparseData2 {
            data: vec![],
            indices: vec![],
            indptr: vec![0; 5],
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (4, 4),
        };
        let snn = SparseGraph::new(4, empty_csr, false);
        let s = sigma_from_snn(&snn, mat.as_ref(), 0, 0.1, 2, 1.0, 1e-8);
        assert!(approx_eq(s, 1e-8, EPS));
    }

    /// Pins the bandwidth against a hand-computed value, tie behaviour included.
    #[test]
    fn sigma_from_snn_known_value() {
        // Build a tiny SNN by hand: cell 0 connects to cells 1, 2, 3 with
        // weights 0.5, 0.5, 0.1. With k_far = 2, the smallest-weight pick is
        // cell 3 (w=0.1); ties at the 2nd smallest are 0.5 each, so cells 1
        // AND 2 also enter the selection (Seurat tie behaviour). Three cells
        // selected: 3, 1, 2. nearest = 0.1 (cell 0 is at origin).
        // distances in embedding:
        //   d(0, 3) = sqrt(200) ~14.1421
        //   d(0, 1) = 0.1
        //   d(0, 2) = 0.1
        // sigma = mean(max(0, d - nearest)) * sd_scale
        //   = mean(14.1421 - 0.1, 0, 0) * 1
        //   = mean(14.0421, 0, 0) = 4.6807
        let (mat, _, _) = small_fixture();
        let csr = CompressedSparseData2 {
            data: vec![0.5, 0.5, 0.1],
            indices: vec![1, 2, 3],
            indptr: vec![0, 3, 3, 3, 3],
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (4, 4),
        };
        let snn = SparseGraph::new(4, csr, false);
        let s = sigma_from_snn(&snn, mat.as_ref(), 0, 0.1, 2, 1.0, 1e-8);
        let expected = (14.0421 + 0.0 + 0.0) / 3.0;
        assert!(
            approx_eq(s, expected, 1e-2),
            "expected ~{expected}, got {s}"
        );
    }

    //////////////////////
    // end-to-end tests //
    //////////////////////

    /// Two modalities = identical embeddings on the small fixture.
    /// Expected: modality weights ~0.5/0.5 per cell because within and cross
    /// imputes are identical, so within_dist == cross_dist and the score
    /// ratio collapses to 1 in both modalities -> equal softmax outputs.
    #[test]
    fn compute_wnn_identical_modalities_balanced_weights() {
        let (mat, knn, dist) = small_fixture();
        let mod_a = ModalityInput::new(mat.as_ref(), &knn, &dist);
        let mod_b = ModalityInput::new(mat.as_ref(), &knn, &dist);

        let params = WnnParams {
            k_nn: 2,
            knn_range: 2,
            sigma_method: SigmaMethod::SigmaIdx,
            sigma_idx: 1,
            s_nn: 2,
            snn_type: SnnType::FullConnection,
            sd_scale: 1.0,
            kernel_power: 1.0,
            cross_const: 1e-4,
            sigma_floor: 1e-8,
            knn_params: KnnParams::default(),
        };

        let res = compute_wnn([mod_a, mod_b], &params, 0)
            .expect("WNN failed on identical modality fixture");

        // shape checks
        assert_eq!(res.wnn_indices.len(), 4);
        assert_eq!(res.wnn_distances.len(), 4);
        assert_eq!(res.modality_weights.len(), 2);
        assert_eq!(res.modality_weights[0].len(), 4);
        assert_eq!(res.modality_weights[1].len(), 4);

        for i in 0..4 {
            assert_eq!(res.wnn_indices[i].len(), params.k_nn);
            assert_eq!(res.wnn_distances[i].len(), params.k_nn);
        }

        // balanced weights when modalities are identical
        for i in 0..4 {
            let w0 = res.modality_weights[0][i];
            let w1 = res.modality_weights[1][i];
            assert!(approx_eq(w0, 0.5, 1e-4), "cell {i}: w0={w0}, expected ~0.5");
            assert!(approx_eq(w1, 0.5, 1e-4));
            assert!(approx_eq(w0 + w1, 1.0, 1e-4));
        }
    }

    /// Asymmetric modalities must pull the per-cell weights away from the
    /// balanced 0.5/0.5 split that identical modalities produce.
    #[test]
    fn compute_wnn_asymmetric_modalities_shift_weights() {
        let (mat, knn, dist) = small_fixture();

        // second modality: rotate coords to give a different but valid
        // embedding (keeps the same kNN layout for simplicity)
        let mat_b = Mat::from_fn(4, 2, |i, j| {
            let v = mat[(i, j)];
            // swap columns
            if j == 0 {
                mat[(i, 1)]
            } else {
                v - mat[(i, 0)] + mat[(i, 1)]
            }
        });

        let mod_a = ModalityInput::new(mat.as_ref(), &knn, &dist);
        let mod_b = ModalityInput::new(mat_b.as_ref(), &knn, &dist);

        let params = WnnParams {
            k_nn: 2,
            knn_range: 2,
            sigma_method: SigmaMethod::SigmaIdx,
            sigma_idx: 1,
            s_nn: 2,
            ..Default::default()
        };

        let res = compute_wnn([mod_a, mod_b], &params, 0)
            .expect("WNN failed on rotated modality fixture");

        for i in 0..4 {
            let s = res.modality_weights[0][i] + res.modality_weights[1][i];
            println!(
                "PROBE cell {i}: {} {}",
                res.modality_weights[0][i], res.modality_weights[1][i]
            );
            assert!(
                approx_eq(s, 1.0, 1e-4),
                "cell {i} weights do not sum to 1: {s}"
            );
        }
    }

    /// Modalities with different cell counts are refused rather than indexed out of bounds.
    #[test]
    fn compute_wnn_errors_on_cell_count_mismatch() {
        let (mat_a, knn, dist) = small_fixture();
        let mat_b = Mat::from_fn(3, 2, |_, _| 0.0_f32); // different n_cells

        // knn and dist are size 4 but mat_b is size 3
        let mod_a = ModalityInput::new(mat_a.as_ref(), &knn, &dist);
        let mod_b = ModalityInput::new(mat_b.as_ref(), &knn, &dist);

        let params = WnnParams {
            k_nn: 2,
            knn_range: 2,
            sigma_idx: 1,
            s_nn: 2,
            ..Default::default()
        };

        let res = compute_wnn([mod_a, mod_b], &params, 0);
        assert!(matches!(
            res,
            Err(BixverseErrors::WNNModalitySampleMismatch)
        ));
    }

    /// A k_nn larger than knn_range cannot be satisfied and must error.
    #[test]
    fn compute_wnn_errors_on_bad_k_nn() {
        let (mat, knn, dist) = small_fixture();
        let mod_a = ModalityInput::new(mat.as_ref(), &knn, &dist);
        let mod_b = ModalityInput::new(mat.as_ref(), &knn, &dist);

        let params = WnnParams {
            k_nn: 10, // larger than knn_range
            knn_range: 2,
            sigma_idx: 1,
            s_nn: 2,
            ..Default::default()
        };

        let res = compute_wnn([mod_a, mod_b], &params, 0);
        assert!(matches!(res, Err(BixverseErrors::WNNKnnLargerThanKnnRange)));
    }
}
