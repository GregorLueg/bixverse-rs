//! Consensus NMF: turns the random restarts from [`super::stabilised_nmf`] into
//! a stability score and a consensus factor, plus a driver loop over k so the
//! rank can be chosen from the stability and error curves.
//!
//! The pipeline follows cNMF: pool the components across restarts, L2-normalise
//! them, drop the ones sitting in a sparse neighbourhood, cluster the survivors
//! into k groups, score stability as the mean silhouette, and take the
//! per-cluster median as the consensus factor. The partner factor is then refit
//! against it with a frozen-factor NNLS pass.
//!
//! ### References
//!
//! Kotliar et al., eLife, 2019

use ann_search_rs::prelude::AnnSearchFloat;
use ann_search_rs::{build_exhaustive_index, query_exhaustive_self};
use faer::{Mat, MatRef};
use rayon::prelude::*;

use super::refit::{nmf_refit_h, nmf_refit_w};
use super::{HalsOpts, NmfInput, StabilisedNmfResult, stabilised_nmf};
use crate::ml::clustering::clustering_metrics::silhouette_cosine_unit;
use crate::ml::clustering::k_means::{KMeansParamsWrappers, k_means_clusters};
use crate::prelude::*;

////////////
// Consts //
////////////

/// Fraction of `n_runs` used as the local neighbourhood size for the density
/// filter. cNMF's `local_neighborhood_size` default.
const LOCAL_NEIGHBOURHOOD_FRACTION: f64 = 0.3;

/// Default cutoff for the density filter, as a mean COSINE distance. Cosine
/// distance is bounded by 2, so any value at or above 2 leaves every component
/// in place.
const DEFAULT_DENSITY_THRESHOLD: f64 = 0.5;

/// k-means restarts kept per consensus step, lowest inertia winning. cNMF leans
/// on scikit-learn's `n_init = 10` for the same purpose.
const DEFAULT_KMEANS_N_INIT: usize = 3;

/// Default Lloyd iterations per k-means restart.
const DEFAULT_KMEANS_ITERS: usize = 100;

/// Norm below which a pooled component counts as collapsed and is dropped
/// before anything else runs. A zero-norm row has no direction, so cosine
/// distance against it is meaningless.
const DEGENERATE_NORM_TOLERANCE: f64 = 1e-12;

////////////
// Params //
////////////

/// Which factor the consensus clustering operates on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConsensusTarget {
    /// Rows of the per-run H matrices stacked vertically, L2-normalised. These
    /// are the programs over features (the spectra), and what cNMF clusters. H
    /// rows carry the magnitudes out of [`super::nmf_hals`], so normalisation
    /// is required rather than cosmetic.
    #[default]
    HRows,
    /// Columns of `w_all`. Already unit L2 by construction, but these are
    /// sample-side activity patterns rather than the interpretable programs.
    /// The clustering then runs in sample space, which is cheap for bulk and
    /// expensive on the single-cell path, where that is the cell count.
    WColumns,
}

/// Parse the consensus target
///
/// ### Params
///
/// * `s` - String to parse. `"h"`, `"h_rows"`, `"spectra"` or `"programs"` for
///   [`ConsensusTarget::HRows`]; `"w"`, `"w_cols"`, `"w_columns"` or `"usage"`
///   for [`ConsensusTarget::WColumns`]. Case-insensitive.
///
/// ### Returns
///
/// The option of [ConsensusTarget].
pub fn parse_consensus_target(s: &str) -> Option<ConsensusTarget> {
    match s.to_lowercase().as_str() {
        "h" | "h_rows" | "spectra" | "programs" => Some(ConsensusTarget::HRows),
        "w" | "w_cols" | "w_columns" | "usage" => Some(ConsensusTarget::WColumns),
        _ => None,
    }
}

/// Options for the consensus step.
#[derive(Clone, Copy, Debug)]
pub struct ConsensusParams<F: BixverseFloat> {
    /// Which factor to cluster.
    pub target: ConsensusTarget,
    /// Neighbours used by the local-density filter. `None` or `Some(0)` resolve
    /// to `LOCAL_NEIGHBOURHOOD_FRACTION * n_runs`; both spellings mean "pick for
    /// me", so that a zero arriving from R behaves the same way.
    pub n_neighbours: Option<usize>,
    /// Mean cosine distance to those neighbours above which a component is
    /// dropped. `None` disables filtering entirely.
    pub density_threshold: Option<F>,
    /// Lloyd iterations per k-means restart.
    pub kmeans_iters: usize,
    /// Number of k-means restarts; the lowest-inertia labelling wins.
    pub kmeans_n_init: usize,
    /// Seed for the k-means restarts.
    pub seed: u64,
}

impl<F: BixverseFloat> ConsensusParams<F> {
    /// Generate a new instance of [ConsensusParams]
    ///
    /// ### Params
    ///
    /// * `target` - Which factor to cluster, see [ConsensusTarget].
    /// * `n_neighbours` - Neighbours for the density filter. `None` or `Some(0)`
    ///   for the `0.3 * n_runs` heuristic.
    /// * `density_threshold` - Mean cosine distance cutoff. `None` disables the
    ///   filter.
    /// * `kmeans_iters` - Lloyd iterations per k-means restart.
    /// * `kmeans_n_init` - Number of k-means restarts.
    /// * `seed` - Seed for the k-means restarts.
    ///
    /// ### Returns
    ///
    /// The initialised [ConsensusParams].
    pub fn new(
        target: ConsensusTarget,
        n_neighbours: Option<usize>,
        density_threshold: Option<F>,
        kmeans_iters: usize,
        kmeans_n_init: usize,
        seed: u64,
    ) -> Self {
        Self {
            target,
            n_neighbours,
            density_threshold,
            kmeans_iters,
            kmeans_n_init,
            seed,
        }
    }
}

impl<F: BixverseFloat> Default for ConsensusParams<F> {
    fn default() -> Self {
        Self {
            target: ConsensusTarget::default(),
            n_neighbours: None,
            density_threshold: Some(F::from_f64(DEFAULT_DENSITY_THRESHOLD).unwrap()),
            kmeans_iters: DEFAULT_KMEANS_ITERS,
            kmeans_n_init: DEFAULT_KMEANS_N_INIT,
            seed: 42,
        }
    }
}

/// Resolve the local neighbourhood size for the density filter.
///
/// Defaults to `LOCAL_NEIGHBOURHOOD_FRACTION * n_runs` rounded up, which keeps
/// the neighbourhood proportional to how many chances a real program had to
/// reappear across restarts, and `Some(0)` is taken to mean the same as `None`.
/// Clamped to at least one and to at most `n_components - 1`, since a point cannot
/// be its own neighbour; a pool of one therefore still reports one, which the
/// caller's guards make unreachable.
///
/// ### Params
///
/// * `n_neighbours` - Caller override, or `None` for the heuristic.
/// * `n_runs` - Number of restarts pooled.
/// * `n_components` - Components available to search over.
///
/// ### Returns
///
/// The neighbourhood size, at least one.
fn resolve_n_neighbours(n_neighbours: Option<usize>, n_runs: usize, n_components: usize) -> usize {
    // `Some(0)` is treated as `None` rather than clamped to one neighbour, so a
    // zero coming across the R boundary means the same thing as an absent value.
    let requested = match n_neighbours {
        Some(l) if l > 0 => l,
        _ => (LOCAL_NEIGHBOURHOOD_FRACTION * n_runs as f64).ceil() as usize,
    };
    requested.clamp(1, n_components.saturating_sub(1).max(1))
}

/////////////
// Results //
/////////////

/// Diagnostics from clustering the pooled components at one k.
#[derive(Clone, Debug)]
pub struct ConsensusClusters<F: BixverseFloat> {
    /// Cluster label per pooled component, `None` if the component was dropped.
    /// Length `k * n_runs`, indexed the same way as `w_all`'s columns.
    pub labels: Vec<Option<usize>>,
    /// Mean cosine distance to the nearest neighbours, per pooled component.
    /// `infinity` for components dropped as collapsed. Plot this to pick a
    /// `density_threshold`.
    pub local_density: Vec<F>,
    /// Pooled indices of the surviving components, ascending.
    pub kept: Vec<usize>,
    /// Silhouette per survivor, aligned with `kept`.
    pub silhouette: Vec<F>,
    /// Mean silhouette over the survivors. The stability score.
    pub stability: F,
    /// Number of survivors per cluster. Length k.
    pub sizes: Vec<usize>,
    /// Per-cluster median component, shape `k x dim`, rows renormalised to unit
    /// L2. `dim` is the feature count for [`ConsensusTarget::HRows`] and the
    /// sample count for [`ConsensusTarget::WColumns`].
    pub consensus: Mat<F>,
    /// Components removed, whether collapsed or filtered out.
    pub n_dropped: usize,
    /// Clusters that ended up with no members.
    pub n_empty_clusters: usize,
}

/// A consensus NMF fit at one k.
#[derive(Clone, Debug)]
pub struct ConsensusNmfResult<F: BixverseFloat> {
    /// Left factor, shape `m x k`.
    pub w: Mat<F>,
    /// Right factor, shape `k x n`.
    pub h: Mat<F>,
    /// `||V - WH||_F^2` of the consensus pair, divided by `||V||_F^2`.
    pub error: F,
    /// The clustering diagnostics behind the fit.
    pub clusters: ConsensusClusters<F>,
    /// Per-restart losses, each divided by `||V||_F^2`.
    pub run_errors: Vec<F>,
}

/// One row of the k sweep.
#[derive(Clone, Debug)]
pub struct KSweepEntry<F: BixverseFloat> {
    /// The rank this row describes.
    pub k: usize,
    /// Mean silhouette of the surviving components. Higher is more stable. `NaN`
    /// if the consensus step could not run at this k, which R reads as `NA`; see
    /// `consensus_failed`.
    pub stability: F,
    /// Loss of the best restart, divided by `||V||_F^2`. Always populated, since
    /// it comes from the restarts rather than the consensus.
    pub best_error: F,
    /// Median restart loss, divided by `||V||_F^2`.
    pub median_error: F,
    /// True if the density filter left fewer than `k` components, so the
    /// clustering never ran. The error columns are still meaningful; the
    /// stability one is not. Usually means the threshold is too tight for this k.
    pub consensus_failed: bool,
    /// Components dropped, whether collapsed or filtered out. A rising drop rate
    /// is itself a sign that k has gone past useful.
    pub n_dropped: usize,
    /// Clusters that ended up empty.
    pub n_empty_clusters: usize,
    /// Restarts that met the HALS tolerance.
    pub n_converged: usize,
}

/////////////
// Helpers //
/////////////

/// Pool the chosen factor across restarts into unit-L2 rows.
///
/// Pooled row `r * k + j` is component `j` of restart `r`, matching the column
/// blocks of `w_all`. Rows are normalised in place; `w_all` columns already have
/// unit norm out of HALS, but they are renormalised anyway so both targets carry
/// the same invariant.
///
/// ### Params
///
/// * `res` - Restarts from [`super::stabilised_nmf`].
/// * `k` - Components per restart.
/// * `target` - Which factor to pool.
///
/// ### Returns
///
/// A `(k * n_runs) x dim` matrix of unit-L2 rows. Collapsed components stay as
/// zero rows for the caller to drop.
fn pool_components<F: BixverseFloat>(
    res: &StabilisedNmfResult<F>,
    k: usize,
    target: ConsensusTarget,
) -> Mat<F> {
    let n_runs = res.h_per_run.len();
    let n_pooled = k * n_runs;

    let mut pooled = match target {
        ConsensusTarget::HRows => {
            let n = res.h_per_run[0].ncols();
            Mat::<F>::from_fn(n_pooled, n, |row, col| {
                res.h_per_run[row / k][(row % k, col)]
            })
        }
        ConsensusTarget::WColumns => {
            let m = res.w_all.nrows();
            Mat::<F>::from_fn(n_pooled, m, |row, col| res.w_all[(col, row)])
        }
    };

    let tol = F::from_f64(DEGENERATE_NORM_TOLERANCE).unwrap();
    let dim = pooled.ncols();
    for i in 0..n_pooled {
        let norm = row_norm(pooled.as_ref(), i);
        if norm <= tol {
            continue;
        }
        let inv = F::one() / norm;
        for j in 0..dim {
            pooled[(i, j)] *= inv;
        }
    }

    pooled
}

/// L2 norm of a single row.
///
/// ### Params
///
/// * `mat` - The matrix to read from.
/// * `row` - Row index.
///
/// ### Returns
///
/// The L2 norm of that row.
#[inline]
fn row_norm<F: BixverseFloat>(mat: MatRef<F>, row: usize) -> F {
    (0..mat.ncols())
        .fold(F::zero(), |acc, j| acc + mat[(row, j)] * mat[(row, j)])
        .sqrt()
}

/// Copy a subset of rows into a fresh matrix.
///
/// The clustering and kNN backends both want a contiguous matrix, so the subset
/// is materialised rather than viewed.
///
/// ### Params
///
/// * `mat` - Source matrix.
/// * `rows` - Row indices to keep, in the order wanted.
///
/// ### Returns
///
/// A `rows.len() x mat.ncols()` matrix.
fn subset_rows<F: BixverseFloat>(mat: MatRef<F>, rows: &[usize]) -> Mat<F> {
    Mat::<F>::from_fn(rows.len(), mat.ncols(), |i, j| mat[(rows[i], j)])
}

/// Mean cosine distance from each row to its `l` nearest neighbours.
///
/// The cNMF outlier criterion. A component that reappeared across restarts sits
/// in a tight neighbourhood of its near-duplicates and scores low; a one-off
/// artefact from a bad initialisation has nothing nearby and scores high.
/// Without this filter a handful of bad restarts smear the cluster medians and
/// depress the silhouette at every k, so the stability curve loses its shape.
///
/// Runs through the exhaustive index, which is exact and needs no recall tuning.
/// That makes it the one `O(n^2 * dim)` step in the pipeline.
///
/// ### Params
///
/// * `data` - Unit-L2 rows, shape `n x dim`.
/// * `l` - Neighbours per row, excluding self.
/// * `verbose` - Passed through to the kNN query.
///
/// ### Returns
///
/// One mean distance per row, or a [`BixverseErrors`] if the query fails.
fn local_densities<F>(data: MatRef<F>, l: usize, verbose: usize) -> Result<Vec<F>, BixverseErrors>
where
    F: BixverseFloat + AnnSearchFloat,
{
    let index = build_exhaustive_index(data, "cosine");
    // Self comes back in the results, so ask for one extra and drop it below.
    let (indices, distances) = query_exhaustive_self(&index, l + 1, true, verbose > 1)?;
    let distances = distances.expect("distances requested from the exhaustive index");

    Ok(indices
        .into_par_iter()
        .zip(distances)
        .enumerate()
        .map(|(i, (idx_row, dist_row))| {
            // Self is dropped by identity rather than by assuming it sits at
            // position 0. Equivalent in practice, since anything tied with self is
            // also at distance zero, but it does not depend on the backend's
            // ordering of ties.
            let mut sum = F::zero();
            let mut count = 0usize;
            for (&idx, &d) in idx_row.iter().zip(dist_row.iter()) {
                if idx == i {
                    continue;
                }
                sum += d;
                count += 1;
                if count == l {
                    break;
                }
            }

            if count == 0 {
                return F::zero();
            }
            sum / F::from_usize(count).unwrap()
        })
        .collect())
}

/// k-means over the survivors, keeping the lowest-inertia restart.
///
/// ### Params
///
/// * `data` - Unit-L2 rows to cluster, shape `n x dim`.
/// * `k` - Number of clusters.
/// * `params` - Consensus options supplying the iteration count, restart count
///   and seed.
///
/// ### Returns
///
/// One cluster label per row, or a [`BixverseErrors`] if k-means fails.
fn cluster_best_of_n<F>(
    data: MatRef<F>,
    k: usize,
    params: &ConsensusParams<F>,
) -> Result<Vec<usize>, BixverseErrors>
where
    F: BixverseFloat + AnnSearchFloat,
{
    // Both floored at one: a zero arriving from R would otherwise give an
    // init-only clustering, or no clustering at all, rather than an error.
    let kmeans_params = KMeansParamsWrappers::new(params.kmeans_iters.max(1), None, None);
    let n_init = params.kmeans_n_init.max(1);

    let mut best: Option<(F, Vec<usize>)> = None;
    for i in 0..n_init {
        let (centroids, assignments) = k_means_clusters(
            data,
            "euclidean",
            k,
            Some(kmeans_params),
            (params.seed + i as u64) as usize,
            false,
        )?;
        let inertia = inertia(data, centroids.as_ref(), &assignments);
        if best.as_ref().is_none_or(|(b, _)| inertia < *b) {
            best = Some((inertia, assignments));
        }
    }

    Ok(best.expect("at least one k-means restart runs").1)
}

/// Within-cluster sum of squared distances to the assigned centroid.
///
/// The k-means wrapper does not report inertia, so it is recomputed here to pick
/// between restarts. One pass, `O(n * dim)`.
///
/// ### Params
///
/// * `data` - The clustered rows, shape `n x dim`.
/// * `centroids` - Centroids, shape `k x dim`.
/// * `assignments` - Cluster label per row.
///
/// ### Returns
///
/// The total inertia.
fn inertia<F: BixverseFloat + Send + Sync>(
    data: MatRef<F>,
    centroids: MatRef<F>,
    assignments: &[usize],
) -> F {
    let dim = data.ncols();
    let total: f64 = assignments
        .par_iter()
        .enumerate()
        .map(|(i, &label)| {
            let mut acc = 0f64;
            for j in 0..dim {
                let d = (data[(i, j)] - centroids[(label, j)]).to_f64().unwrap();
                acc += d * d;
            }
            acc
        })
        .sum();

    F::from_f64(total).unwrap()
}

/// Per-cluster median component, rows renormalised to unit L2.
///
/// The median rather than the mean, following cNMF: a straggler that k-means put
/// in the wrong cluster shifts a mean but not a median. Parallelised over the
/// `dim` coordinates, since each is an independent selection problem.
///
/// ### Params
///
/// * `data` - Unit-L2 survivor rows, shape `n x dim`.
/// * `assignments` - Cluster label per row.
/// * `k` - Number of clusters.
///
/// ### Returns
///
/// A `k x dim` matrix. Empty clusters give zero rows; the caller decides whether
/// that is fatal.
fn median_consensus<F: BixverseFloat + Send + Sync>(
    data: MatRef<F>,
    assignments: &[usize],
    k: usize,
) -> Mat<F> {
    let dim = data.ncols();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &label) in assignments.iter().enumerate() {
        members[label].push(i);
    }

    let mut consensus = Mat::<F>::zeros(k, dim);
    for (c, rows) in members.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let column: Vec<F> = (0..dim)
            .into_par_iter()
            .map(|j| {
                let mut values: Vec<F> = rows.iter().map(|&i| data[(i, j)]).collect();
                median_in_place(&mut values)
            })
            .collect();
        for (j, value) in column.into_iter().enumerate() {
            consensus[(c, j)] = value;
        }
    }

    // Renormalise so the consensus carries the same unit-norm invariant as the
    // components it was built from.
    let tol = F::from_f64(DEGENERATE_NORM_TOLERANCE).unwrap();
    for c in 0..k {
        let norm = row_norm(consensus.as_ref(), c);
        if norm <= tol {
            continue;
        }
        let inv = F::one() / norm;
        for j in 0..dim {
            consensus[(c, j)] *= inv;
        }
    }

    consensus
}

/// Median of a slice, reordering it in the process.
///
/// ### Params
///
/// * `values` - The values, reordered in place. Must be non-empty.
///
/// ### Returns
///
/// The median; the mean of the two central values for an even count.
fn median_in_place<F: BixverseFloat>(values: &mut [F]) -> F {
    let n = values.len();
    let mid = n / 2;
    values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));

    if n.is_multiple_of(2) {
        let upper = values[mid];
        // The lower half is unordered but does sit below the pivot, so its max is
        // the other central value.
        let lower = values[..mid]
            .iter()
            .copied()
            .fold(F::neg_infinity(), |acc, x| if x > acc { x } else { acc });
        (lower + upper) / F::from_f64(2.0).unwrap()
    } else {
        values[mid]
    }
}

/// Median of a slice without touching it.
///
/// ### Params
///
/// * `values` - The values. Must be non-empty.
///
/// ### Returns
///
/// The median.
fn median<F: BixverseFloat>(values: &[F]) -> F {
    let mut copy = values.to_vec();
    median_in_place(&mut copy)
}

///////////////
// Workhorse //
///////////////

/// Cluster the pooled components of an existing set of restarts.
///
/// Split out from [`nmf_consensus`] so a density threshold can be re-picked
/// from the `local_density` histogram without paying for the NMF restarts
/// again, which is how cNMF is meant to be driven.
///
/// Steps: pool and L2-normalise the chosen factor, drop collapsed components,
/// compute each component's mean cosine distance to its nearest neighbours and
/// drop those above the threshold, k-means the survivors into k groups, score
/// the silhouette, and take the per-cluster median.
///
/// The k-means runs on unit-norm rows under squared Euclidean rather than
/// cosine distance. On unit-norm rows the two agree up to
/// `||a - b||^2 = 2(1 - cos)`.
///
/// ### Params
///
/// * `res` - Restarts from [`super::stabilised_nmf`].
/// * `k` - Rank the restarts were run at. Must be at least 2.
/// * `params` - Consensus options; see [`ConsensusParams`].
/// * `verbose` - If `0` -> silent, `1` or higher prints progress.
///
/// ### Returns
///
/// The [`ConsensusClusters`] diagnostics, or
/// [`BixverseErrors::NmfConsensusInvalidK`] /
/// [`BixverseErrors::NmfConsensusTooFewRuns`] for bad arguments,
/// [`BixverseErrors::NmfConsensusKMismatch`] if `k` does not match the restarts,
/// or [`BixverseErrors::NmfConsensusTooFewComponents`] if the density filter left
/// fewer than `k` survivors. An empty cluster is reported in the result rather
/// than raised.
pub fn consensus_from_restarts<F>(
    res: &StabilisedNmfResult<F>,
    k: usize,
    params: &ConsensusParams<F>,
    verbose: usize,
) -> Result<ConsensusClusters<F>, BixverseErrors>
where
    F: BixverseFloat + AnnSearchFloat,
{
    let n_runs = res.h_per_run.len();
    if k < 2 {
        return Err(BixverseErrors::NmfConsensusInvalidK { k });
    }
    if n_runs < 2 {
        return Err(BixverseErrors::NmfConsensusTooFewRuns { n_runs });
    }
    // k must be the rank the restarts were actually run at, or the pooling
    // would slice the per-run H matrices along the wrong stride. `w_all` is
    // checked separately: the fields of `StabilisedNmfResult` are public, so a
    // spliced value can disagree with `h_per_run` and would otherwise index out
    // of bounds inside faer rather than erroring.
    let restart_k = res.h_per_run[0].nrows();
    if restart_k != k || res.w_all.ncols() != k * n_runs {
        return Err(BixverseErrors::NmfConsensusKMismatch {
            requested: k,
            restarts: restart_k,
        });
    }

    let verbosity = parse_verbosity_level(verbose);
    let pooled = pool_components(res, k, params.target);
    let n_pooled = pooled.nrows();

    // Collapsed components are dropped before anything touches cosine distance.
    let tol = F::from_f64(DEGENERATE_NORM_TOLERANCE).unwrap();
    let mut local_density = vec![F::infinity(); n_pooled];
    let alive: Vec<usize> = (0..n_pooled)
        .filter(|&i| row_norm(pooled.as_ref(), i) > tol)
        .collect();

    if alive.len() < k {
        return Err(BixverseErrors::NmfConsensusTooFewComponents {
            k,
            n_surviving: alive.len(),
        });
    }

    // Local density over the non-collapsed components only.
    let alive_mat = subset_rows(pooled.as_ref(), &alive);
    let l = resolve_n_neighbours(params.n_neighbours, n_runs, alive.len());
    let densities = local_densities(alive_mat.as_ref(), l, verbose)?;
    for (&idx, &d) in alive.iter().zip(densities.iter()) {
        local_density[idx] = d;
    }

    let kept: Vec<usize> = match params.density_threshold {
        Some(threshold) => alive
            .iter()
            .copied()
            .filter(|&i| local_density[i] <= threshold)
            .collect(),
        None => alive.clone(),
    };

    if kept.len() < k {
        return Err(BixverseErrors::NmfConsensusTooFewComponents {
            k,
            n_surviving: kept.len(),
        });
    }

    if verbosity.normal_verbosity() {
        println!(
            "  Consensus NMF (k = {}): {} of {} components kept, {} neighbours per density estimate",
            k,
            kept.len(),
            n_pooled,
            l
        );
    }

    let survivors = subset_rows(pooled.as_ref(), &kept);
    let assignments = cluster_best_of_n(survivors.as_ref(), k, params)?;

    let mut sizes = vec![0usize; k];
    for &label in &assignments {
        sizes[label] += 1;
    }
    let n_empty_clusters = sizes.iter().filter(|&&s| s == 0).count();

    let (silhouette, stability) = silhouette_cosine_unit(survivors.as_ref(), &assignments, k);
    let consensus = median_consensus(survivors.as_ref(), &assignments, k);

    let mut labels = vec![None; n_pooled];
    for (&idx, &label) in kept.iter().zip(assignments.iter()) {
        labels[idx] = Some(label);
    }

    Ok(ConsensusClusters {
        labels,
        local_density,
        n_dropped: n_pooled - kept.len(),
        kept,
        silhouette,
        stability,
        sizes,
        consensus,
        n_empty_clusters,
    })
}

//////////////////
// Entry points //
//////////////////

/// Consensus NMF at a single k.
///
/// Runs `n_runs` random restarts, clusters the pooled components, and refits the
/// partner factor against the consensus one. For
/// [`ConsensusTarget::HRows`] the consensus is H and W is refit against it, so
/// the returned H rows have unit L2 norm and W carries the magnitudes. For
/// [`ConsensusTarget::WColumns`] it is the other way round.
///
/// ### Params
///
/// * `v` - Input matrix backend.
/// * `k` - Number of components. Must be at least 2.
/// * `n_runs` - Number of random restarts. Must be at least 2.
/// * `base_seed` - Seed offset; restart `i` uses `base_seed + i`.
/// * `opts` - HALS options. The `init` field is ignored by the restarts.
/// * `params` - Optional [`ConsensusParams`], defaulted if `None`.
/// * `verbose` - If `0` -> silent, `1` for normal, `2` for detailed verbosity.
///
/// ### Returns
///
/// A [`ConsensusNmfResult`], or a [`BixverseErrors`]. Errors on an empty cluster,
/// since the consensus factor would carry a zero row and the refit would be
/// degenerate.
pub fn nmf_consensus<F, In>(
    v: &In,
    k: usize,
    n_runs: usize,
    base_seed: u64,
    opts: &HalsOpts<F>,
    params: Option<ConsensusParams<F>>,
    verbose: usize,
) -> Result<ConsensusNmfResult<F>, BixverseErrors>
where
    F: BixverseFloat + AnnSearchFloat + Send + Sync,
    In: NmfInput<F> + Sync,
{
    let params = params.unwrap_or_default();
    if k < 2 {
        return Err(BixverseErrors::NmfConsensusInvalidK { k });
    }
    if n_runs < 2 {
        return Err(BixverseErrors::NmfConsensusTooFewRuns { n_runs });
    }

    let restarts = stabilised_nmf(v, k, n_runs, base_seed, opts, verbose)?;
    let clusters = consensus_from_restarts(&restarts, k, &params, verbose)?;

    if let Some(cluster) = clusters.sizes.iter().position(|&s| s == 0) {
        return Err(BixverseErrors::NmfConsensusEmptyCluster { cluster, k });
    }

    let sq_frob = v.sq_frob();
    let (w, h, loss) = match params.target {
        ConsensusTarget::HRows => {
            // The consensus is already k x n, so it is H directly.
            let h = clusters.consensus.clone();
            let (w, loss) = nmf_refit_w(v, h.as_ref(), opts)?;
            (w, h, loss)
        }
        ConsensusTarget::WColumns => {
            // The consensus is k x m, so transpose it into an m x k W.
            let w = clusters.consensus.transpose().to_owned();
            let (h, loss) = nmf_refit_h(v, w.as_ref(), opts)?;
            (w, h, loss)
        }
    };

    let denom = if sq_frob > F::zero() {
        sq_frob
    } else {
        F::one()
    };
    let run_errors: Vec<F> = restarts.losses.iter().map(|&l| l / denom).collect();

    Ok(ConsensusNmfResult {
        w,
        h,
        error: loss / denom,
        clusters,
        run_errors,
    })
}

/// Sweep k and report stability against error.
///
/// One row per k, with no refit and no factors retained, so sweeping a wide
/// range stays cheap in memory. Pick the k where stability is high and the
/// error curve has not yet flattened, then call [`nmf_consensus`] there.
///
/// k is swept sequentially: [`super::stabilised_nmf`] already parallelises
/// across its restarts and sizes its inner pool against the core count, so
/// nesting another layer here would only oversubscribe.
///
/// Two conditions that are properties of a particular k rather than of the
/// input are recorded rather than raised, so one awkward rank does not take the
/// whole sweep down: an empty cluster (see `n_empty_clusters`) and a density
/// filter that left fewer than k components (see `consensus_failed`, with
/// `stability` set to `NaN`). Anything else aborts.
///
/// ### Params
///
/// * `v` - Input matrix backend.
/// * `k_range` - Ranks to evaluate. Must be non-empty, every entry at least 2.
/// * `n_runs` - Restarts per k. Must be at least 2.
/// * `base_seed` - Seed offset. Each k gets a disjoint block of seeds, so two
///   ranks never share a restart.
/// * `opts` - HALS options. The `init` field is ignored by the restarts.
/// * `params` - Optional [`ConsensusParams`], defaulted if `None`.
/// * `verbose` - If `0` -> silent, `1` for normal, `2` for detailed verbosity.
///
/// ### Returns
///
/// One [`KSweepEntry`] per entry of `k_range`, in the order given.
pub fn nmf_k_sweep<F, In>(
    v: &In,
    k_range: &[usize],
    n_runs: usize,
    base_seed: u64,
    opts: &HalsOpts<F>,
    params: Option<ConsensusParams<F>>,
    verbose: usize,
) -> Result<Vec<KSweepEntry<F>>, BixverseErrors>
where
    F: BixverseFloat + AnnSearchFloat + Send + Sync,
    In: NmfInput<F> + Sync,
{
    if k_range.is_empty() {
        return Err(BixverseErrors::NmfKSweepEmptyRange);
    }
    if n_runs < 2 {
        return Err(BixverseErrors::NmfConsensusTooFewRuns { n_runs });
    }
    if let Some(&k) = k_range.iter().find(|&&k| k < 2) {
        return Err(BixverseErrors::NmfConsensusInvalidK { k });
    }

    let params = params.unwrap_or_default();
    let verbosity = parse_verbosity_level(verbose);
    let sq_frob = v.sq_frob();
    let denom = if sq_frob > F::zero() {
        sq_frob
    } else {
        F::one()
    };

    let mut out = Vec::with_capacity(k_range.len());
    for (i, &k) in k_range.iter().enumerate() {
        if verbosity.normal_verbosity() {
            println!(
                "Running NMF k sweep: k = {} ({} of {})",
                k,
                i + 1,
                k_range.len()
            );
        }

        let seed = base_seed + (i * n_runs) as u64;
        let restarts = stabilised_nmf(v, k, n_runs, seed, opts, verbose.saturating_sub(1))?;

        let clusters = match consensus_from_restarts(
            &restarts,
            k,
            &params,
            verbose.saturating_sub(1),
        ) {
            Ok(clusters) => Some(clusters),
            Err(BixverseErrors::NmfConsensusTooFewComponents { n_surviving, .. }) => {
                if verbosity.normal_verbosity() {
                    println!(
                        "  k = {}: only {} components survived the density filter (need {}); stability not scored",
                        k, n_surviving, k
                    );
                }
                None
            }
            Err(e) => return Err(e),
        };

        out.push(KSweepEntry {
            k,
            stability: clusters
                .as_ref()
                .map(|c| c.stability)
                .unwrap_or_else(F::nan),
            best_error: restarts.losses[restarts.best_idx] / denom,
            median_error: median(&restarts.losses) / denom,
            consensus_failed: clusters.is_none(),
            n_dropped: clusters.as_ref().map(|c| c.n_dropped).unwrap_or(k * n_runs),
            n_empty_clusters: clusters.as_ref().map(|c| c.n_empty_clusters).unwrap_or(0),
            n_converged: restarts.converged.iter().filter(|&&c| c).count(),
        });
    }

    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::nmf_hals::NmfInit;
    use crate::methods::nmf_hals::dense::DenseInput;
    use approx::assert_relative_eq;

    /// Exact rank-k non-negative product, built so the components are clearly
    /// distinguishable rather than near-collinear.
    fn rank_k_input(m: usize, n: usize, k: usize) -> Mat<f64> {
        let w: Mat<f64> = Mat::from_fn(m, k, |i, j| {
            // Each component owns a contiguous block of rows.
            if i % k == j { 5.0 } else { 0.3 }
        });
        let h: Mat<f64> = Mat::from_fn(k, n, |i, j| if j % k == i { 4.0 } else { 0.2 });
        Mat::from_fn(m, n, |i, j| (0..k).map(|r| w[(i, r)] * h[(r, j)]).sum())
    }

    fn test_opts() -> HalsOpts<f64> {
        HalsOpts::<f64>::new(200, 1e-8, 1e-12, 10, NmfInit::Nndsvd)
    }

    /// The pooled labels must cover every restart's components, with dropped
    /// entries reported as `None` rather than silently reindexed.
    #[test]
    fn consensus_labels_cover_every_pooled_component() {
        let (m, n, k, n_runs) = (25, 18, 2, 4);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();

        let restarts = stabilised_nmf(&input, k, n_runs, 0, &test_opts(), 0).unwrap();
        let params = ConsensusParams::<f64>::default();
        let clusters = consensus_from_restarts(&restarts, k, &params, 0).unwrap();

        assert_eq!(clusters.labels.len(), k * n_runs);
        assert_eq!(clusters.local_density.len(), k * n_runs);
        assert_eq!(clusters.kept.len(), clusters.silhouette.len());
        assert_eq!(clusters.sizes.iter().sum::<usize>(), clusters.kept.len());
        assert_eq!(clusters.n_dropped, k * n_runs - clusters.kept.len());

        for (i, label) in clusters.labels.iter().enumerate() {
            assert_eq!(label.is_some(), clusters.kept.contains(&i));
        }
    }

    /// The consensus rows must carry the unit-norm invariant the clustering
    /// assumes, or a downstream silhouette against them is meaningless.
    #[test]
    fn consensus_rows_are_unit_norm() {
        let (m, n, k, n_runs) = (25, 18, 2, 4);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();

        let restarts = stabilised_nmf(&input, k, n_runs, 0, &test_opts(), 0).unwrap();
        let clusters =
            consensus_from_restarts(&restarts, k, &ConsensusParams::<f64>::default(), 0).unwrap();

        assert_eq!(clusters.consensus.nrows(), k);
        assert_eq!(clusters.consensus.ncols(), n);
        for c in 0..k {
            assert_relative_eq!(
                row_norm(clusters.consensus.as_ref(), c),
                1.0,
                epsilon = 1e-10
            );
        }
    }

    /// Restarts that recover the same rank-k structure should cluster cleanly, so
    /// stability sits near the top of the scale.
    #[test]
    fn consensus_stability_high_on_exact_rank_k() {
        let (m, n, k, n_runs) = (30, 24, 3, 5);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();

        let restarts = stabilised_nmf(&input, k, n_runs, 7, &test_opts(), 0).unwrap();
        let clusters =
            consensus_from_restarts(&restarts, k, &ConsensusParams::<f64>::default(), 0).unwrap();

        assert!(
            clusters.stability > 0.95,
            "stability {}",
            clusters.stability
        );
    }

    /// Same seeds must give the same labels and the same stability, or a sweep
    /// cannot be compared against a rerun.
    #[test]
    fn consensus_is_deterministic() {
        let (m, n, k, n_runs) = (25, 18, 2, 4);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let params = ConsensusParams::<f64>::default();

        let first = stabilised_nmf(&input, k, n_runs, 3, &test_opts(), 0).unwrap();
        let a = consensus_from_restarts(&first, k, &params, 0).unwrap();
        let second = stabilised_nmf(&input, k, n_runs, 3, &test_opts(), 0).unwrap();
        let b = consensus_from_restarts(&second, k, &params, 0).unwrap();

        assert_eq!(a.labels, b.labels);
        assert_relative_eq!(a.stability, b.stability, epsilon = 1e-12);
    }

    /// A component with nothing near it must land above the threshold. This is the
    /// whole point of the filter, so it gets an explicit outlier rather than
    /// trusting a real restart to misbehave.
    #[test]
    fn density_filter_drops_an_isolated_component() {
        let (m, n, k, n_runs) = (25, 18, 2, 4);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let mut restarts = stabilised_nmf(&input, k, n_runs, 0, &test_opts(), 0).unwrap();

        // Overwrite the last restart's first component with a direction nothing
        // else points in: a single non-zero feature.
        let last = restarts.h_per_run.last_mut().unwrap();
        for j in 0..n {
            last[(0, j)] = if j == n - 1 { 1.0 } else { 0.0 };
        }
        let outlier = (n_runs - 1) * k;

        let unfiltered =
            ConsensusParams::<f64>::new(ConsensusTarget::HRows, None, None, 100, 3, 42);
        let all = consensus_from_restarts(&restarts, k, &unfiltered, 0).unwrap();
        assert_eq!(all.n_dropped, 0);

        // Its local density must be the worst of the pool.
        let worst = all
            .local_density
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(worst, outlier, "densities {:?}", all.local_density);

        let filtered =
            consensus_from_restarts(&restarts, k, &ConsensusParams::default(), 0).unwrap();
        assert!(filtered.labels[outlier].is_none());
        assert!(filtered.n_dropped >= 1);
    }

    /// A collapsed (zero-norm) component has no direction, so it must be dropped
    /// before it reaches the cosine kNN rather than poisoning it.
    #[test]
    fn collapsed_components_are_dropped() {
        let (m, n, k, n_runs) = (25, 18, 2, 5);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let mut restarts = stabilised_nmf(&input, k, n_runs, 0, &test_opts(), 0).unwrap();

        for j in 0..n {
            restarts.h_per_run[1][(0, j)] = 0.0;
        }
        let collapsed = k;

        let params = ConsensusParams::<f64>::new(ConsensusTarget::HRows, None, None, 100, 3, 42);
        let clusters = consensus_from_restarts(&restarts, k, &params, 0).unwrap();

        assert!(clusters.labels[collapsed].is_none());
        assert!(clusters.local_density[collapsed].is_infinite());
        assert_eq!(clusters.n_dropped, 1);
    }

    /// Both targets must run and return the documented orientation.
    #[test]
    fn both_targets_return_documented_shapes() {
        let (m, n, k, n_runs) = (25, 18, 2, 4);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let restarts = stabilised_nmf(&input, k, n_runs, 0, &test_opts(), 0).unwrap();

        let h_params = ConsensusParams::<f64>::new(ConsensusTarget::HRows, None, None, 100, 3, 42);
        let h_res = consensus_from_restarts(&restarts, k, &h_params, 0).unwrap();
        assert_eq!(h_res.consensus.ncols(), n);

        let w_params =
            ConsensusParams::<f64>::new(ConsensusTarget::WColumns, None, None, 100, 3, 42);
        let w_res = consensus_from_restarts(&restarts, k, &w_params, 0).unwrap();
        assert_eq!(w_res.consensus.ncols(), m);
    }

    /// The full fit must return factors of the right shape, in the scale
    /// convention its target documents, and reconstruct the input.
    #[test]
    fn nmf_consensus_returns_usable_factors() {
        let (m, n, k) = (30, 24, 3);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();

        let res = nmf_consensus(&input, k, 4, 0, &test_opts(), None, 0).unwrap();

        assert_eq!(res.w.nrows(), m);
        assert_eq!(res.w.ncols(), k);
        assert_eq!(res.h.nrows(), k);
        assert_eq!(res.h.ncols(), n);
        assert_eq!(res.run_errors.len(), 4);
        // The input is an exact rank-k product, so the consensus must reconstruct
        // it. A loose bound here is worthless: refitting against a random H of the
        // right shape lands around 0.5 and would sail through.
        assert!(res.error < 1e-6, "relative error {}", res.error);

        // HRows is the default, so H carries the unit-norm rows.
        for r in 0..k {
            let norm = (0..n).map(|j| res.h[(r, j)].powi(2)).sum::<f64>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    /// WColumns must put the unit norms on W instead.
    #[test]
    fn nmf_consensus_w_target_normalises_w() {
        let (m, n, k) = (30, 24, 3);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let params = ConsensusParams::<f64>::new(ConsensusTarget::WColumns, None, None, 100, 3, 42);

        let res = nmf_consensus(&input, k, 4, 0, &test_opts(), Some(params), 0).unwrap();

        assert_eq!(res.w.nrows(), m);
        assert_eq!(res.h.ncols(), n);
        assert!(res.error < 1e-6, "relative error {}", res.error);
        for c in 0..k {
            let norm = (0..m).map(|i| res.w[(i, c)].powi(2)).sum::<f64>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    /// The sweep must return one row per requested k, in order.
    #[test]
    fn k_sweep_returns_one_row_per_k() {
        let (m, n) = (30, 24);
        let v = rank_k_input(m, n, 3);
        let input = DenseInput::new(v.as_ref()).unwrap();

        let sweep = nmf_k_sweep(&input, &[2, 3], 3, 0, &test_opts(), None, 0).unwrap();

        assert_eq!(sweep.len(), 2);
        assert_eq!(sweep[0].k, 2);
        assert_eq!(sweep[1].k, 3);
        for entry in &sweep {
            assert!(entry.n_converged <= 3);
            assert!(entry.best_error <= entry.median_error + 1e-12);
        }
    }

    /// Guard rails: k below 2 has no silhouette, one restart has nothing to pool,
    /// and an empty range has nothing to sweep.
    #[test]
    fn invalid_arguments_error() {
        let v = rank_k_input(20, 15, 2);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let opts = test_opts();

        assert!(matches!(
            nmf_consensus(&input, 1, 4, 0, &opts, None, 0),
            Err(BixverseErrors::NmfConsensusInvalidK { k: 1 })
        ));
        assert!(matches!(
            nmf_consensus(&input, 2, 1, 0, &opts, None, 0),
            Err(BixverseErrors::NmfConsensusTooFewRuns { n_runs: 1 })
        ));
        assert!(matches!(
            nmf_k_sweep(&input, &[], 4, 0, &opts, None, 0),
            Err(BixverseErrors::NmfKSweepEmptyRange)
        ));
        assert!(matches!(
            nmf_k_sweep(&input, &[2, 1], 4, 0, &opts, None, 0),
            Err(BixverseErrors::NmfConsensusInvalidK { k: 1 })
        ));
    }

    /// Applying a k the restarts were not run at would slice the per-run H
    /// matrices along the wrong stride, so it must be rejected.
    #[test]
    fn consensus_rejects_mismatched_k() {
        let v = rank_k_input(25, 18, 2);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let restarts = stabilised_nmf(&input, 2, 4, 0, &test_opts(), 0).unwrap();

        assert!(matches!(
            consensus_from_restarts(&restarts, 3, &ConsensusParams::<f64>::default(), 0),
            Err(BixverseErrors::NmfConsensusKMismatch {
                requested: 3,
                restarts: 2
            })
        ));
    }

    /// Pins the density values themselves on a hand-computable case. It does not
    /// discriminate the identity-based self-exclusion from a positional one: since
    /// anything tied with self is also at distance zero, the two agree for every
    /// reachable input.
    #[test]
    fn local_density_matches_hand_computed_values() {
        // Three identical rows plus one orthogonal to them.
        let data: Mat<f64> = Mat::from_fn(4, 2, |i, j| match (i, j) {
            (3, 0) => 0.0,
            (3, 1) => 1.0,
            (_, 0) => 1.0,
            _ => 0.0,
        });

        // Each duplicate has two other duplicates at distance 0.
        let densities = local_densities(data.as_ref(), 2, 0).unwrap();
        for d in densities.iter().take(3) {
            assert_relative_eq!(d, &0.0, epsilon = 1e-12);
        }
        // The odd one out only has the duplicates, all at distance 1.
        assert_relative_eq!(densities[3], 1.0, epsilon = 1e-12);
    }

    /// The neighbourhood heuristic must stay inside the pool and never reach zero.
    #[test]
    fn neighbourhood_resolution_is_clamped() {
        // 0.3 * 20 = 6.
        assert_eq!(resolve_n_neighbours(None, 20, 100), 6);
        // Rounds up rather than down to zero.
        assert_eq!(resolve_n_neighbours(None, 2, 100), 1);
        // Cannot exceed the pool minus self.
        assert_eq!(resolve_n_neighbours(Some(50), 20, 10), 9);
        // Never zero, even for a pool of one.
        assert_eq!(resolve_n_neighbours(Some(0), 20, 1), 1);
    }

    /// Even and odd counts, since the even case has to average two central values
    /// out of a partially ordered slice.
    #[test]
    fn median_handles_both_parities() {
        assert_relative_eq!(median(&[3.0, 1.0, 2.0]), 2.0, epsilon = 1e-12);
        assert_relative_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5, epsilon = 1e-12);
        assert_relative_eq!(median(&[5.0]), 5.0, epsilon = 1e-12);
        assert_relative_eq!(median(&[2.0, 2.0, 2.0, 8.0]), 2.0, epsilon = 1e-12);
    }

    /// The median must be the per-coordinate median of the cluster members, and a
    /// single outlier in a cluster must not move it the way a mean would.
    #[test]
    fn median_consensus_resists_an_outlier() {
        // Three near-identical rows plus one wild one, all in cluster 0.
        let data: Mat<f64> = Mat::from_fn(4, 2, |i, j| match (i, j) {
            (3, 0) => 0.0,
            (3, 1) => 1.0,
            (_, 0) => 1.0,
            _ => 0.0,
        });
        let consensus = median_consensus(data.as_ref(), &[0, 0, 0, 0], 1);

        assert_relative_eq!(consensus[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(consensus[(0, 1)], 0.0, epsilon = 1e-10);
    }

    /// An empty cluster gives a zero consensus row, which `nmf_consensus` treats
    /// as fatal but the sweep only records.
    #[test]
    fn median_consensus_leaves_empty_clusters_zero() {
        let data: Mat<f64> = Mat::from_fn(2, 3, |_, j| if j == 0 { 1.0 } else { 0.0 });
        let consensus = median_consensus(data.as_ref(), &[0, 0], 2);

        assert_relative_eq!(row_norm(consensus.as_ref(), 0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(row_norm(consensus.as_ref(), 1), 0.0, epsilon = 1e-10);
    }

    /// The whole point of the sweep: stability must peak at the rank the data was
    /// actually generated at, and the error must keep falling as k grows.
    #[test]
    #[cfg(feature = "large-test")]
    // 200 x 120 rank-5 product with 5% noise, k in 2..=8, 20 restarts per k.
    fn k_sweep_peaks_at_the_true_rank() {
        let (m, n, k_true) = (200, 120, 5);

        // Block-structured non-negative factors, so the true rank is unambiguous.
        let w: Mat<f64> = Mat::from_fn(
            m,
            k_true,
            |i, j| {
                if i * k_true / m == j { 6.0 } else { 0.2 }
            },
        );
        let h: Mat<f64> = Mat::from_fn(
            k_true,
            n,
            |i, j| {
                if j * k_true / n == i { 5.0 } else { 0.15 }
            },
        );

        // Deterministic multiplicative noise at roughly 5%.
        let mut state = 987654321u64;
        let mut v = Mat::<f64>::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let jitter = ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let clean: f64 = (0..k_true).map(|r| w[(i, r)] * h[(r, j)]).sum();
                v[(i, j)] = (clean * (1.0 + 0.05 * jitter)).max(0.0);
            }
        }

        let input = DenseInput::new(v.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(400, 1e-7, 1e-12, 10, NmfInit::Nndsvd);
        let k_range: Vec<usize> = (2..=8).collect();

        let sweep = nmf_k_sweep(&input, &k_range, 20, 11, &opts, None, 0).unwrap();

        let best = sweep
            .iter()
            .max_by(|a, b| a.stability.partial_cmp(&b.stability).unwrap())
            .unwrap();
        assert_eq!(
            best.k,
            k_true,
            "stability curve: {:?}",
            sweep.iter().map(|e| (e.k, e.stability)).collect::<Vec<_>>()
        );

        // More components can only fit the data better, up to solver slack.
        for pair in sweep.windows(2) {
            assert!(
                pair[1].best_error <= pair[0].best_error + 1e-3,
                "error rose from k = {} ({}) to k = {} ({})",
                pair[0].k,
                pair[0].best_error,
                pair[1].k,
                pair[1].best_error
            );
        }
    }

    /// The pooled index ordering for `WColumns` must line up with `w_all`'s column
    /// blocks, or `labels` is silently misaligned with `local_density`. Planting an
    /// isolated direction at a known column and asserting the density peak lands at
    /// the matching pooled index is the only thing that pins it.
    #[test]
    fn w_columns_pooled_ordering_matches_w_all() {
        let (m, n, k, n_runs) = (25, 18, 2, 4);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let mut restarts = stabilised_nmf(&input, k, n_runs, 0, &test_opts(), 0).unwrap();

        // Restart 2, component 1 -> pooled index 2 * k + 1.
        let (run, comp) = (2usize, 1usize);
        let target_col = run * k + comp;
        for row in 0..m {
            restarts.w_all[(row, target_col)] = if row == m - 1 { 1.0 } else { 0.0 };
        }

        let params = ConsensusParams::<f64>::new(ConsensusTarget::WColumns, None, None, 100, 3, 42);
        let clusters = consensus_from_restarts(&restarts, k, &params, 0).unwrap();

        let worst = clusters
            .local_density
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(worst, target_col, "densities {:?}", clusters.local_density);
    }

    /// The same ordering claim for `HRows`, which indexes `h_per_run[r]` row `j`.
    #[test]
    fn h_rows_pooled_ordering_matches_run_blocks() {
        let (m, n, k, n_runs) = (25, 18, 2, 4);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let mut restarts = stabilised_nmf(&input, k, n_runs, 0, &test_opts(), 0).unwrap();

        let (run, comp) = (2usize, 1usize);
        for j in 0..n {
            restarts.h_per_run[run][(comp, j)] = if j == n - 1 { 1.0 } else { 0.0 };
        }

        let params = ConsensusParams::<f64>::new(ConsensusTarget::HRows, None, None, 100, 3, 42);
        let clusters = consensus_from_restarts(&restarts, k, &params, 0).unwrap();

        let worst = clusters
            .local_density
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(worst, run * k + comp);
    }

    /// Every kept component must be at or below the threshold and every dropped
    /// live one above it. Without this the filter could be applied with the
    /// comparison inverted and the other tests would not notice.
    #[test]
    fn density_filter_partitions_on_the_threshold() {
        let (m, n, k, n_runs) = (25, 18, 2, 5);
        let v = rank_k_input(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let restarts = stabilised_nmf(&input, k, n_runs, 4, &test_opts(), 0).unwrap();

        // Tight enough to actually bite, loose enough to leave k survivors.
        let unfiltered = consensus_from_restarts(
            &restarts,
            k,
            &ConsensusParams::<f64>::new(ConsensusTarget::HRows, None, None, 100, 3, 42),
            0,
        )
        .unwrap();
        let mut densities: Vec<f64> = unfiltered
            .local_density
            .iter()
            .copied()
            .filter(|d| d.is_finite())
            .collect();
        densities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = densities[densities.len() / 2];

        let params =
            ConsensusParams::<f64>::new(ConsensusTarget::HRows, None, Some(threshold), 100, 3, 42);
        let clusters = consensus_from_restarts(&restarts, k, &params, 0).unwrap();

        for &i in &clusters.kept {
            assert!(
                clusters.local_density[i] <= threshold,
                "kept component {i} has density {} above {threshold}",
                clusters.local_density[i]
            );
        }
        for i in 0..clusters.labels.len() {
            if clusters.labels[i].is_none() && clusters.local_density[i].is_finite() {
                assert!(
                    clusters.local_density[i] > threshold,
                    "dropped component {i} has density {} at or below {threshold}",
                    clusters.local_density[i]
                );
            }
        }
    }

    /// A pool collapsed onto fewer directions than k cannot fill every cluster, so
    /// k-means leaves one empty and the count must report it. `nmf_consensus`
    /// refuses on that condition; the sweep only records it.
    #[test]
    fn degenerate_pool_leaves_an_empty_cluster() {
        let (m, n, k, n_runs) = (25, 18, 3, 4);
        let v = rank_k_input(m, n, 2);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let mut restarts = stabilised_nmf(&input, k, n_runs, 0, &test_opts(), 0).unwrap();

        // Collapse every pooled component onto two directions, so a third cluster
        // has nothing to claim.
        for r in 0..n_runs {
            for c in 0..k {
                for j in 0..n {
                    restarts.h_per_run[r][(c, j)] = if j % 2 == c % 2 { 1.0 } else { 0.0 };
                }
            }
        }

        let params = ConsensusParams::<f64>::new(ConsensusTarget::HRows, None, None, 100, 1, 42);
        let clusters = consensus_from_restarts(&restarts, k, &params, 0).unwrap();
        assert!(clusters.n_empty_clusters > 0, "sizes {:?}", clusters.sizes);
    }

    /// A threshold that prunes everything is a property of that k, not a broken
    /// input, so the sweep must record it and carry on rather than discarding the
    /// ranks it already computed.
    #[test]
    fn k_sweep_records_an_over_pruned_k() {
        let (m, n) = (25, 18);
        let v = rank_k_input(m, n, 2);
        let input = DenseInput::new(v.as_ref()).unwrap();
        // Negative threshold: nothing can survive.
        let params =
            ConsensusParams::<f64>::new(ConsensusTarget::HRows, None, Some(-1.0), 100, 3, 42);

        let sweep = nmf_k_sweep(&input, &[2, 3], 3, 0, &test_opts(), Some(params), 0).unwrap();

        assert_eq!(sweep.len(), 2);
        for entry in &sweep {
            assert!(entry.consensus_failed);
            assert!(entry.stability.is_nan());
            // The error columns come from the restarts, so they stay meaningful.
            assert!(entry.best_error.is_finite());
            assert!(entry.median_error.is_finite());
        }
    }

    /// Over-pruning is recorded by the sweep but is still an error for a single
    /// consensus call, which has no way to return a factor.
    #[test]
    fn over_pruning_errors_for_a_single_k() {
        let v = rank_k_input(25, 18, 2);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let params =
            ConsensusParams::<f64>::new(ConsensusTarget::HRows, None, Some(-1.0), 100, 3, 42);

        assert!(matches!(
            nmf_consensus(&input, 2, 4, 0, &test_opts(), Some(params), 0),
            Err(BixverseErrors::NmfConsensusTooFewComponents { k: 2, .. })
        ));
    }

    /// A spliced `StabilisedNmfResult` whose `w_all` disagrees with `h_per_run`
    /// must error rather than panicking inside faer, since the fields are public.
    #[test]
    fn inconsistent_w_all_errors_rather_than_panicking() {
        let v = rank_k_input(25, 18, 2);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let mut restarts = stabilised_nmf(&input, 2, 4, 0, &test_opts(), 0).unwrap();

        // Truncate w_all so it no longer holds k * n_runs columns.
        restarts.w_all = Mat::<f64>::from_fn(25, 2, |i, j| restarts.w_all[(i, j)]);

        assert!(matches!(
            consensus_from_restarts(&restarts, 2, &ConsensusParams::<f64>::default(), 0),
            Err(BixverseErrors::NmfConsensusKMismatch { .. })
        ));
    }

    /// The single-cell path is `f32` only, so the whole pipeline gets exercised at
    /// that precision. The silhouette's cancellation is severe enough that an
    /// `f32` accumulation put scores outside `[-1, 1]` here.
    #[test]
    fn consensus_runs_in_f32() {
        let (m, n, k, n_runs) = (30, 24, 3, 4);
        let v: Mat<f32> = {
            let w: Mat<f32> = Mat::from_fn(m, k, |i, j| if i % k == j { 5.0 } else { 0.3 });
            let h: Mat<f32> = Mat::from_fn(k, n, |i, j| if j % k == i { 4.0 } else { 0.2 });
            Mat::from_fn(m, n, |i, j| (0..k).map(|r| w[(i, r)] * h[(r, j)]).sum())
        };
        let input = DenseInput::new(v.as_ref()).unwrap();
        let opts = HalsOpts::<f32>::new(200, 1e-6, 1e-10, 10, NmfInit::Nndsvd);

        let restarts = stabilised_nmf(&input, k, n_runs, 0, &opts, 0).unwrap();
        let clusters =
            consensus_from_restarts(&restarts, k, &ConsensusParams::<f32>::default(), 0).unwrap();

        for s in &clusters.silhouette {
            assert!(
                *s <= 1.0 + 1e-5 && *s >= -1.0 - 1e-5,
                "f32 silhouette {s} outside [-1, 1]"
            );
        }
        assert!(
            clusters.stability > 0.9,
            "f32 stability {}",
            clusters.stability
        );

        let fit = nmf_consensus(&input, k, n_runs, 0, &opts, None, 0).unwrap();
        assert!(
            fit.error.is_finite() && fit.error < 1e-2,
            "error {}",
            fit.error
        );
    }

    /// `Some(0)` neighbours must mean the same as `None`, so a zero crossing the R
    /// boundary does not silently become a one-neighbour density estimate.
    #[test]
    fn zero_neighbours_means_the_heuristic() {
        assert_eq!(resolve_n_neighbours(Some(0), 20, 100), 6);
        assert_eq!(resolve_n_neighbours(None, 20, 100), 6);
    }

    /// Target strings are the R-facing surface, so unknown input must be `None`
    /// rather than silently defaulting.
    #[test]
    fn target_strings_parse() {
        assert_eq!(parse_consensus_target("h"), Some(ConsensusTarget::HRows));
        assert_eq!(
            parse_consensus_target("SPECTRA"),
            Some(ConsensusTarget::HRows)
        );
        assert_eq!(
            parse_consensus_target("w_cols"),
            Some(ConsensusTarget::WColumns)
        );
        assert_eq!(parse_consensus_target("nonsense"), None);
    }
}
