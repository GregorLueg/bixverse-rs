//! Implementation of the miloR differential abundance approach on top of kNN
//! graphs, see Dann, et al., Nat Biotechnol, 2022
//!
//! The chain is: sample neighbourhood indices, refine them, build the
//! membership matrix with [`build_nhood_matrix`], count cells per sample with
//! [`count_nhood_cells`], test with
//! [`run_edger_ql`](crate::methods::dge_bulk::run_edger_ql) and correct with
//! [`spatial_fdr`].
//!
//! The test is edgeR's, unchanged, with neighbourhoods sitting where genes
//! normally do. There is no wrapper for it here because there would be nothing
//! in the wrapper: call `run_edger_ql` with `filter: false` and whatever
//! `min_mean` you want, since `filterByExpr` is a gene-expression heuristic and
//! means nothing for a neighbourhood.

use ann_search_rs::cpu::{annoy::AnnoyIndex, hnsw::HnswIndex, nndescent::NNDescent};
use ann_search_rs::utils::dist::{Dist, parse_ann_dist};
use faer::MatRef;
use rayon::prelude::*;

use crate::core::math::vector_helpers::median;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Structure for MiloR algorithm parameters
///
/// ### Fields
///
/// **MiloR params**
///
/// * `prop` - Proportion of cells to sample as neighbourhood indices
/// * `k_refine` - Number of neighbours to use for refinement
/// * `refinement_strategy` - Strategy for refining sampled indices
///   (`"approximate"`, `"bruteforce"`, or `"index"`)
/// * `index_type` - Type of kNN index to use (`"annoy"` or `"hnsw"`)
/// * `knn_params` - The knnParams via the `KnnParams` structure.
pub struct MiloRParams {
    /// Proportion of cells to sample as neighbourhood indices
    pub prop: f64,
    /// Number of neighbours to use for refinement
    pub k_refine: usize,
    /// Strategy for refining sampled indices (`"approximate"`, `"bruteforce"`,
    /// or `"index"`)
    pub refinement_strategy: String,
    /// Type of kNN index to use (`"annoy"`, `"hnsw"` or `"nndescent"`)
    pub index_type: String,
    /// [KnnParams] for the various approximate nearest neighbour searches
    /// in ann-search-rs. `self.knn_params.knn_method` is ignored in favour of
    /// `self.index_type`.
    pub knn_params: KnnParams,
}

/// Enum wrapper for different kNN index implementations
///
/// ### Variants
///
/// * `Annoy` - Approximate nearest neighbour index using trees
/// * `Hnsw` - Hierarchical navigable small world graph index
/// * `NNDescent` - Nearest neighbour descent index
pub enum KnnIndex {
    /// The Annoy index
    Annoy(AnnoyIndex<f32>),
    /// The HNSW index
    Hnsw(HnswIndex<f32>),
    /// NNDescent
    NNDescent(NNDescent<f32>),
}

impl KnnIndex {
    /// Generate a new instance of the kNN index
    ///
    /// ### Params
    ///
    /// * `embd` - The embedding matrix of cells x features to use to the
    ///   the generation
    /// * `knn_params` - The KnnParams with distance type, number of trees, etc.
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised `KnnIndex`.
    pub fn new(
        embd: MatRef<f32>,
        index_type: KnnIndexType,
        knn_params: &KnnParams,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, BixverseErrors> {
        match index_type {
            KnnIndexType::AnnoyIndex => {
                let dist = ann_search_rs::utils::dist::parse_ann_dist(&knn_params.ann_dist)
                    .unwrap_or_else(|| {
                        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
                        Dist::default()
                    });

                Ok(KnnIndex::Annoy(AnnoyIndex::new(
                    embd,
                    knn_params.n_tree,
                    dist,
                    seed,
                )?))
            }
            KnnIndexType::HnswIndex => {
                let dist = ann_search_rs::utils::dist::parse_ann_dist(&knn_params.ann_dist)
                    .unwrap_or_else(|| {
                        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
                        Dist::default()
                    });

                Ok(KnnIndex::Hnsw(HnswIndex::build(
                    embd,
                    knn_params.m,
                    knn_params.ef_construction,
                    &dist,
                    seed,
                    verbose,
                )))
            }
            KnnIndexType::NNDescentIndex => {
                let dist = ann_search_rs::utils::dist::parse_ann_dist(&knn_params.ann_dist)
                    .unwrap_or_else(|| {
                        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
                        Dist::default()
                    });

                Ok(KnnIndex::NNDescent(NNDescent::new(
                    embd,
                    dist,
                    None,
                    None,
                    None,
                    None,
                    knn_params.delta,
                    knn_params.diversify_prob,
                    seed,
                    verbose,
                )?))
            }
        }
    }

    /// Query for k nearest neighbours of a single point
    ///
    /// ### Params
    ///
    /// * `query_point` - The slice of values defining the query point
    /// * `knn_params` - The KnnParams with distance type, search budget, etc.
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(neighbour indices, distances to neighbours)`
    pub fn query_single(
        &self,
        query_point: &[f32],
        knn_params: &KnnParams,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<f32>), BixverseErrors> {
        match self {
            KnnIndex::Annoy(index) => Ok(index.query(query_point, k, knn_params.search_budget)?),
            KnnIndex::Hnsw(index) => Ok(index.query(query_point, k, knn_params.ef_search)?),
            KnnIndex::NNDescent(index) => Ok(index.query(query_point, k, None)?),
        }
    }
}

/// Enum specifying which kNN index type to use
///
/// ### Variants
///
/// * `AnnoyIndex` - Use Annoy index
/// * `HnswIndex` - Use HNSW index
#[allow(clippy::enum_variant_names)]
pub enum KnnIndexType {
    /// Annoy
    AnnoyIndex,
    /// HNSW
    HnswIndex,
    /// NNDescent
    NNDescentIndex,
}

//////////////
// Sampling //
//////////////

/// Enum specifying the refinement strategy for neighbourhood sampling
#[derive(Debug, Clone, Copy)]
pub enum RefinementStrategy {
    /// Search within k neighbours only
    Approximate,
    /// Linear search through all cells
    BruteForce,
    /// Use existing kNN index for search
    IndexBased,
}

/// Helper function to parse the refinement strategy
///
/// ### Params
///
/// * `s` - String specifying the strategy to use
///
/// ### Returns
///
/// The Option of the chosen `RefinementStrategy`
pub fn parse_refinement_strategy(s: &str) -> Option<RefinementStrategy> {
    match s.to_lowercase().as_str() {
        "approximate" => Some(RefinementStrategy::Approximate),
        "bruteforce" => Some(RefinementStrategy::BruteForce),
        "index" => Some(RefinementStrategy::IndexBased),
        _ => None,
    }
}

/// Helper function to parse the kNN index type
///
/// ### Params
///
/// * `s` - String specifying which kNN index to use
///
/// ### Returns
///
/// The Option of the chosen `KnnIndexType`
pub fn parse_index_type(s: &str) -> Option<KnnIndexType> {
    match s.to_lowercase().as_str() {
        "annoy" => Some(KnnIndexType::AnnoyIndex),
        "hnsw" => Some(KnnIndexType::HnswIndex),
        "nndescent" => Some(KnnIndexType::NNDescentIndex),
        _ => None,
    }
}

/// Helper function to compute the median positions
///
/// ### Params
///
/// * `embd` - The embedding matrix that was used for the generation of the kNN
///   graph.
/// * `neighbours` - Slice of indices for the neighbours
///
/// ### Returns
///
/// Vector of median features
fn compute_median_position(embd: MatRef<f32>, neighbours: &[usize]) -> Vec<f32> {
    let n_feature = embd.ncols();

    let mut median_point = vec![0.0f32; n_feature];
    for feat_idx in 0..n_feature {
        let values = neighbours
            .iter()
            .map(|&nb_idx| embd[(nb_idx, feat_idx)])
            .collect::<Vec<f32>>();

        median_point[feat_idx] = median(&values).unwrap_or(0_f32);
    }

    median_point
}

/// Find the cell nearest to a median position within a subset of candidates
///
/// ### Params
///
/// * `embd` - The embedding matrix
/// * `median_point` - The median position to query
/// * `candidates` - Indices of candidate cells to search within
/// * `metric` - Distance metric to use
///
/// ### Returns
///
/// Index of the nearest cell within the candidate subset
fn find_nearest_in_subset(
    embd: MatRef<f32>,
    median_point: &[f32],
    candidates: &[usize],
    metric: &Dist,
) -> usize {
    let median_row = MatRef::from_row_major_slice(median_point, 1, embd.ncols()).row(0);

    candidates
        .par_iter()
        .map(|&idx| {
            let dist = compute_distance_knn(median_row, embd.row(idx), metric);
            (idx, dist)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

/// Find the cell nearest to a median position using brute force search
///
/// ### Params
///
/// * `embd` - The embedding matrix
/// * `median_point` - The median position to query
/// * `metric` - Distance metric to use
///
/// ### Returns
///
/// Index of the nearest cell in the entire dataset
fn find_nearest_bruteforce(embd: MatRef<f32>, median_point: &[f32], metric: &Dist) -> usize {
    let median_row = MatRef::from_row_major_slice(median_point, 1, embd.ncols()).row(0);

    (0..embd.nrows())
        .into_par_iter()
        .map(|idx| {
            let dist = compute_distance_knn(median_row, embd.row(idx), metric);
            (idx, dist)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

/// Find the cell nearest to a median position using the kNN index
///
/// ### Params
///
/// * `index` - The kNN index to query
/// * `knn_params` - Parameters for the kNN search
/// * `median_point` - The median position to query
///
/// ### Returns
///
/// Index of the nearest cell
fn find_nearest_with_index(
    index: &KnnIndex,
    knn_params: &KnnParams,
    median_point: &[f32],
) -> Result<usize, BixverseErrors> {
    let (indices, _) = index.query_single(median_point, knn_params, 1)?;

    Ok(indices[0])
}

/// Refine neighbourhood sampling by shifting indices towards local median
/// positions
///
/// ### Params
///
/// * `embd` - The embedding matrix
/// * `knn_indices` - The kNN graph as adjacency list
/// * `sampled_indices` - Initial sampled cell indices
/// * `k_refine` - Number of neighbours to use for computing median
/// * `knn_params` - Parameters for distance calculation
/// * `strategy` - Refinement strategy to use
/// * `knn_index` - Optional kNN index for index-based strategy
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
///   detailed verbosity.
///
/// ### Returns
///
/// Refined cell indices after shifting to nearest median positions
#[allow(clippy::too_many_arguments)]
pub fn refine_sampling_with_strategy(
    embd: MatRef<f32>,
    knn_indices: &[Vec<usize>],
    sampled_indices: &[usize],
    k_refine: usize,
    knn_params: &KnnParams,
    strategy: &RefinementStrategy,
    knn_index: Option<&KnnIndex>,
    verbose: usize,
) -> Result<Vec<usize>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    if verbosity.normal_verbosity() {
        println!("Running refined sampling");
    }

    let mut refined = Vec::with_capacity(sampled_indices.len());

    let dist_metric = parse_ann_dist(&knn_params.ann_dist).unwrap_or_default();

    for &sample_idx in sampled_indices {
        let mut neighbours = Vec::with_capacity(k_refine);
        for j in 0..k_refine.min(knn_indices[0].len()) {
            let neighbour_idx = knn_indices[sample_idx][j];
            neighbours.push(neighbour_idx);
        }

        let median_point = compute_median_position(embd, &neighbours);

        let best_idx = match strategy {
            RefinementStrategy::Approximate => {
                find_nearest_in_subset(embd, &median_point, &neighbours, &dist_metric)
            }
            RefinementStrategy::BruteForce => {
                find_nearest_bruteforce(embd, &median_point, &dist_metric)
            }
            RefinementStrategy::IndexBased => {
                if let Some(index) = knn_index {
                    find_nearest_with_index(index, knn_params, &median_point)?
                } else {
                    // Fallback to brute force
                    find_nearest_bruteforce(embd, &median_point, &dist_metric)
                }
            }
        };

        refined.push(best_idx);
    }

    Ok(refined)
}

/// Compute distances to the k-th nearest neighbour for each index cell
///
/// ### Params
///
/// * `embd` - The embedding matrix
/// * `knn_indices` - The kNN graph as adjacency list
/// * `index_cells` - Indices of neighbourhood centre cells
/// * `kth_col` - Which neighbour to compute distance to (0-indexed)
///
/// ### Returns
///
/// Vector of distances to k-th neighbour for each index cell
pub fn compute_kth_distances_from_matrix(
    embd: MatRef<f32>,
    knn_indices: &[Vec<usize>],
    index_cells: &[usize],
    kth_col: usize,
) -> Vec<f64> {
    index_cells
        .par_iter()
        .map(|&cell_idx| {
            let kth_neighbour = knn_indices[cell_idx][kth_col];

            compute_distance_knn(
                embd.row(cell_idx),
                embd.row(kth_neighbour),
                &Dist::SquaredEuclidean,
            ) as f64
        })
        .collect()
}

/// Build sparse neighbourhood matrix in COO (triplet) format
///
/// Each neighbourhood includes the index cell plus its k nearest neighbours.
///
/// ### Params
///
/// * `knn_indices` - The kNN graph as adjacency list
/// * `index_cells` - Indices of neighbourhood centre cells
///
/// ### Returns
///
/// Tuple of `(row_indices, col_indices, values)` in COO format where
/// each neighbourhood is a column and non-zero entries indicate membership
pub fn build_nhood_matrix(
    knn_indices: &[Vec<usize>],
    index_cells: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let k = knn_indices[0].len();
    let n_nhoods = index_cells.len();

    // Pre-allocate (over-estimate)
    let mut row_indices = Vec::with_capacity(n_nhoods * (k + 1));
    let mut col_indices = Vec::with_capacity(n_nhoods * (k + 1));
    let mut values = Vec::with_capacity(n_nhoods * (k + 1));

    for (nh_idx, &cell_idx) in index_cells.iter().enumerate() {
        row_indices.push(cell_idx);
        col_indices.push(nh_idx);
        values.push(1.0);

        for j in 0..k {
            let neighbor_idx = knn_indices[cell_idx][j];
            row_indices.push(neighbor_idx);
            col_indices.push(nh_idx);
            values.push(1.0);
        }
    }

    (row_indices, col_indices, values)
}

/////////////////////////
// Neighbourhood tests //
/////////////////////////

/// Counts the cells of each sample in each neighbourhood.
///
/// The `t(nhoods) %*% onehot(sample)` contraction, and Milo's `countCells`.
///
/// A cell listed twice in the same neighbourhood is counted once. The dedup
/// relies on [`build_nhood_matrix`] emitting each neighbourhood's entries
/// contiguously, which it does; a COO interleaved across neighbourhoods would
/// double count.
///
/// ### Params
///
/// * `rows` - Cell index per non-zero, from [`build_nhood_matrix`]
/// * `cols` - Neighbourhood index per non-zero, matching `rows`
/// * `sample_ids` - Sample label per cell, values in `0..n_samples`
/// * `n_nhoods` - Number of neighbourhoods
/// * `n_samples` - Number of samples
///
/// ### Returns
///
/// The counts, row-major `n_nhoods * n_samples`, or
/// [`BixverseErrors::InvalidArgument`] if an index is out of range or the two
/// COO vectors disagree in length.
pub fn count_nhood_cells(
    rows: &[usize],
    cols: &[usize],
    sample_ids: &[usize],
    n_nhoods: usize,
    n_samples: usize,
) -> Result<Vec<f64>, BixverseErrors> {
    let n_cells = sample_ids.len();
    check_nhood_coo(rows, cols, n_nhoods, n_cells)?;
    if let Some(&bad) = sample_ids.iter().find(|&&s| s >= n_samples) {
        return Err(BixverseErrors::InvalidArgument(format!(
            "sample label {bad} is outside 0..{n_samples}."
        )));
    }

    let mut out = vec![0.0_f64; n_nhoods * n_samples];
    let mut seen = vec![usize::MAX; n_cells];

    for (&cell, &nhood) in rows.iter().zip(cols.iter()) {
        if seen[cell] == nhood {
            continue;
        }
        seen[cell] = nhood;
        out[nhood * n_samples + sample_ids[cell]] += 1.0;
    }

    Ok(out)
}

/// Checks a neighbourhood COO against the shape it claims.
///
/// ### Params
///
/// * `rows` - Cell index per non-zero
/// * `cols` - Neighbourhood index per non-zero
/// * `n_nhoods` - Number of neighbourhoods
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// `Ok(())`, or [`BixverseErrors::InvalidArgument`] naming the first problem.
fn check_nhood_coo(
    rows: &[usize],
    cols: &[usize],
    n_nhoods: usize,
    n_cells: usize,
) -> Result<(), BixverseErrors> {
    if rows.len() != cols.len() {
        return Err(BixverseErrors::InvalidArgument(format!(
            "the neighbourhood COO has {} rows but {} columns.",
            rows.len(),
            cols.len()
        )));
    }
    if let Some(&bad) = rows.iter().find(|&&c| c >= n_cells) {
        return Err(BixverseErrors::InvalidArgument(format!(
            "cell index {bad} is outside 0..{n_cells}."
        )));
    }
    if let Some(&bad) = cols.iter().find(|&&n| n >= n_nhoods) {
        return Err(BixverseErrors::InvalidArgument(format!(
            "neighbourhood index {bad} is outside 0..{n_nhoods}."
        )));
    }
    Ok(())
}

/// How many cells each neighbourhood shares with all the others.
///
/// Milo's `graph-overlap` weighting: the row sums of `t(nhoods) %*% nhoods`
/// with the diagonal zeroed, so a neighbourhood's own size does not count.
///
/// Never forms that product. Membership is binary, so
/// `sum_j!=i (N^T N)[i, j]` collapses to `sum_c in i (degree(c) - 1)`, where
/// `degree(c)` counts the neighbourhoods holding cell `c`. Two passes over the
/// non-zeros instead of the `O(n_nhoods^2)` intersection, which matters:
/// Milo samples thousands of neighbourhoods.
///
/// A cell listed twice in the same neighbourhood counts once, on the same
/// contiguity assumption [`count_nhood_cells`] makes.
///
/// ### Params
///
/// * `rows` - Cell index per non-zero, from [`build_nhood_matrix`]
/// * `cols` - Neighbourhood index per non-zero, matching `rows`
/// * `n_nhoods` - Number of neighbourhoods
/// * `n_cells` - Number of cells in the graph
///
/// ### Returns
///
/// One overlap total per neighbourhood, or
/// [`BixverseErrors::InvalidArgument`] if an index is out of range or the two
/// COO vectors disagree in length.
pub fn nhood_overlap(
    rows: &[usize],
    cols: &[usize],
    n_nhoods: usize,
    n_cells: usize,
) -> Result<Vec<f64>, BixverseErrors> {
    check_nhood_coo(rows, cols, n_nhoods, n_cells)?;

    let mut degree = vec![0.0_f64; n_cells];
    let mut seen = vec![usize::MAX; n_cells];
    for (&cell, &nhood) in rows.iter().zip(cols.iter()) {
        if seen[cell] == nhood {
            continue;
        }
        seen[cell] = nhood;
        degree[cell] += 1.0;
    }

    let mut out = vec![0.0_f64; n_nhoods];
    seen.iter_mut().for_each(|v| *v = usize::MAX);
    for (&cell, &nhood) in rows.iter().zip(cols.iter()) {
        if seen[cell] == nhood {
            continue;
        }
        seen[cell] = nhood;
        out[nhood] += degree[cell] - 1.0;
    }

    Ok(out)
}

/// Weighted Benjamini-Hochberg over overlapping neighbourhoods.
///
/// Milo's spatial FDR. Neighbourhoods overlap, so the tests are not
/// independent and a plain BH is anti-conservative. Each p-value is weighted by
/// the reciprocal of its connectivity, either the distance to the k-th
/// neighbour or the number of cells shared with other neighbourhoods, and the
/// step-up runs on those weights.
///
/// Non-finite p-values are carried through untouched and take no part in the
/// adjustment, as the upstream does with `NA`.
///
/// ### Params
///
/// * `p_values` - One raw p-value per tested neighbourhood
/// * `connectivity` - Matching connectivity per neighbourhood, either the k-th
///   neighbour distances from [`compute_kth_distances_from_matrix`] or the
///   overlaps from [`nhood_overlap`]. A zero connectivity gets a weight of one,
///   as in the upstream
///
/// ### Returns
///
/// The adjusted p-values, in the input order, or
/// [`BixverseErrors::InvalidArgument`] if the two disagree in length.
///
/// ### References
///
/// Dann et al., Nature Biotechnology 40, 245, 2022
pub fn spatial_fdr(p_values: &[f64], connectivity: &[f64]) -> Result<Vec<f64>, BixverseErrors> {
    if p_values.len() != connectivity.len() {
        return Err(BixverseErrors::InvalidArgument(format!(
            "{} p-values against {} connectivity values.",
            p_values.len(),
            connectivity.len()
        )));
    }

    let usable: Vec<usize> = (0..p_values.len())
        .filter(|&i| p_values[i].is_finite())
        .collect();
    let mut out = vec![f64::NAN; p_values.len()];
    if usable.is_empty() {
        return Ok(out);
    }

    let mut order = usable.clone();
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

    let weights: Vec<f64> = order
        .iter()
        .map(|&i| {
            let w = 1.0 / connectivity[i];
            if w.is_finite() { w } else { 1.0 }
        })
        .collect();
    let total: f64 = weights.iter().sum();

    // Step-up on the weighted cumulative sum, then the running minimum from the
    // largest p-value down, which is what keeps the adjustment monotone.
    let mut running = 0.0;
    let mut adjusted: Vec<f64> = Vec::with_capacity(order.len());
    for (rank, &i) in order.iter().enumerate() {
        running += weights[rank];
        adjusted.push(total * p_values[i] / running);
    }
    let mut floor = f64::INFINITY;
    for rank in (0..adjusted.len()).rev() {
        floor = floor.min(adjusted[rank]);
        adjusted[rank] = floor.min(1.0);
    }

    for (rank, &i) in order.iter().enumerate() {
        out[i] = adjusted[rank];
    }

    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Three neighbourhoods over five cells, the first two overlapping.
    ///
    /// ### Returns
    ///
    /// The COO rows and columns, in the neighbourhood-contiguous order
    /// [`build_nhood_matrix`] emits.
    fn nhood_coo() -> (Vec<usize>, Vec<usize>) {
        let rows = vec![0, 1, 2, 1, 2, 3, 4];
        let cols = vec![0, 0, 0, 1, 1, 1, 2];
        (rows, cols)
    }

    #[test]
    fn test_count_nhood_cells_contracts_membership_against_samples() {
        let (rows, cols) = nhood_coo();
        let sample_ids = vec![0, 0, 1, 1, 0];

        let got = count_nhood_cells(&rows, &cols, &sample_ids, 3, 2).expect("counting failed");

        assert_eq!(got, vec![2.0, 1.0, 1.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_count_nhood_cells_counts_a_repeated_cell_once() {
        let (mut rows, mut cols) = nhood_coo();
        // A kNN list that includes the index cell would look like this.
        rows.insert(1, 0);
        cols.insert(1, 0);
        let sample_ids = vec![0, 0, 1, 1, 0];

        let got = count_nhood_cells(&rows, &cols, &sample_ids, 3, 2).expect("counting failed");

        assert_eq!(got, vec![2.0, 1.0, 1.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_count_nhood_cells_rejects_bad_indices() {
        let (rows, cols) = nhood_coo();
        let sample_ids = vec![0, 0, 1, 1, 0];

        assert!(count_nhood_cells(&rows, &cols[..3], &sample_ids, 3, 2).is_err());
        assert!(count_nhood_cells(&rows, &cols, &sample_ids, 2, 2).is_err());
        assert!(count_nhood_cells(&rows, &cols, &[0, 0, 1, 1, 9], 3, 2).is_err());
    }

    #[test]
    fn test_nhood_overlap_excludes_the_diagonal() {
        let (rows, cols) = nhood_coo();

        // Neighbourhoods 0 and 1 share cells 1 and 2; 2 is disjoint from both.
        let got = nhood_overlap(&rows, &cols, 3, 5).expect("overlap failed");
        assert_eq!(got, vec![2.0, 2.0, 0.0]);
    }

    /// The closed form has to agree with the product it replaces.
    #[test]
    fn test_nhood_overlap_matches_the_crossproduct() {
        let (rows, cols) = nhood_coo();
        let n_nhoods = 3;
        let n_cells = 5;

        // `t(N) %*% N` with the diagonal zeroed, row summed. The definition,
        // written out.
        let mut membership = vec![false; n_nhoods * n_cells];
        for (&cell, &nhood) in rows.iter().zip(cols.iter()) {
            membership[nhood * n_cells + cell] = true;
        }
        let want: Vec<f64> = (0..n_nhoods)
            .map(|i| {
                (0..n_nhoods)
                    .filter(|&j| j != i)
                    .map(|j| {
                        (0..n_cells)
                            .filter(|&c| membership[i * n_cells + c] && membership[j * n_cells + c])
                            .count() as f64
                    })
                    .sum()
            })
            .collect();

        let got = nhood_overlap(&rows, &cols, n_nhoods, n_cells).expect("overlap failed");
        assert_eq!(got, want);
    }

    #[test]
    fn test_nhood_overlap_counts_a_repeated_cell_once() {
        let (mut rows, mut cols) = nhood_coo();
        rows.insert(1, 0);
        cols.insert(1, 0);

        let got = nhood_overlap(&rows, &cols, 3, 5).expect("overlap failed");
        assert_eq!(got, vec![2.0, 2.0, 0.0]);
    }

    /// Against `spatial_fdr_correction` in the bixverse R package.
    ///
    /// Twelve neighbourhoods, one with zero connectivity so its weight falls
    /// back to one, and one with no p-value at all.
    #[test]
    fn test_spatial_fdr_matches_the_r_implementation() {
        let p_values = [
            0.914806,
            0.937075,
            0.286140,
            0.830448,
            0.641746,
            0.519096,
            f64::NAN,
            0.134667,
            0.656992,
            0.705065,
            0.457742,
            0.719112,
        ];
        let connectivity = [
            3.771353, 1.394001, 0.000000, 3.790051, 3.923792, 0.911206, 2.162490, 2.461165,
            3.664110, 0.985486, 3.961121, 3.813339,
        ];
        let want = [
            0.937075000000,
            0.937075000000,
            0.915622825897,
            0.937075000000,
            0.915622825897,
            0.915622825897,
            f64::NAN,
            0.915622825897,
            0.915622825897,
            0.915622825897,
            0.915622825897,
            0.915622825897,
        ];

        let got = spatial_fdr(&p_values, &connectivity).expect("spatial_fdr failed");

        for (g, w) in got.iter().zip(want.iter()) {
            if w.is_nan() {
                assert!(g.is_nan(), "a missing p-value must stay missing");
            } else {
                assert_relative_eq!(*g, *w, max_relative = 1e-10);
            }
        }
    }

    /// Ties, which R breaks by position, and a run that the monotone pass has
    /// to flatten.
    #[test]
    fn test_spatial_fdr_handles_ties_the_way_r_does() {
        let p_values = [0.01, 0.01, 0.5, 0.5, 0.9];
        let connectivity = [1.0, 2.0, 0.5, 4.0, 1.5];
        let want = [
            0.029444444444,
            0.029444444444,
            0.588888888889,
            0.588888888889,
            0.900000000000,
        ];

        let got = spatial_fdr(&p_values, &connectivity).expect("spatial_fdr failed");

        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(*g, *w, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_spatial_fdr_rejects_mismatched_lengths() {
        assert!(spatial_fdr(&[0.1, 0.2], &[1.0]).is_err());
    }
}
