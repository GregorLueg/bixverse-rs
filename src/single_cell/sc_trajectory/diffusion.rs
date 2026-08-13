//! Multiscale diffusion space for Palantir, plus the geometric bookkeeping the
//! trajectory pipeline does on top of it.
//!
//! The kernel and eigendecomposition are reused from the SEACells
//! implementation. Two things are done differently here: a.) the eigenvectors
//! are back-transformed from the symmetrically normalised problem to the
//! row-stochastic one Palantir actually defines its diffusion coordinates on,
//! and b.) the multiscale scaling carries an eigenvalue clamp plus the
//! reference's second-largest-gap fallback.
//!
//! Data is kept as row-major `Vec<Vec<f32>>` (cells x components) throughout,
//! which is what [determine_multiscale_space] emits and what
//! [max_min_sampling] consumes. With ten or so components the column-wise
//! passes here are free, and no layout conversion is needed anywhere.
//!
//! ### References
//!
//! Setty, et al., Nat. Biotechnol., 2019.

use rustc_hash::FxHashSet;

use crate::prelude::*;
use crate::single_cell::mc_generation::seacells::{
    compute_diffusion_kernel, determine_multiscale_space, diffusion_map_from_kernel_diag,
    max_min_sampling,
};

/////////////
// Structs //
/////////////

/// The multiscale diffusion space plus the eigensolver's own verdict on it.
///
/// The diagnostics travel with the components on purpose. A budget-exhausted
/// Lanczos solve returns an embedding that is noise, and nothing downstream can
/// tell: the pseudotime refinement happily converges on it, so a run reports
/// success end to end.
pub struct MultiscaleSpace {
    /// Multiscale components, `n_cells` rows by the retained component count.
    pub components: Vec<Vec<f32>>,
    /// Whether the eigensolve met its tolerance rather than running out of
    /// restarts.
    pub converged: bool,
    /// Largest achieved `||A x - lambda x||` over the returned pairs.
    pub residual: f64,
}

////////////
// Consts //
////////////

/// Largest eigenvalue admitted into the `lambda / (1 - lambda)` multiscale
/// scaling.
const MAX_MULTISCALE_EIGENVALUE: f64 = 1.0 - 1e-10;

/// Floor on the number of eigenvectors kept by the eigengap heuristic.
///
/// Matches the reference's `n_eigs = 3` fallback. Dropping the trivial first
/// eigenvector leaves two multiscale components, which is the minimum that
/// still describes a branching manifold. The reference applies this floor only
/// when it picks the count itself, and so does [resolve_n_eigs].
const MIN_MULTISCALE_EIGS: usize = 3;

/// Eigenvalues needed before a multiscale space can be built at all.
///
/// One eigenvalue is the trivial one, which is always dropped, so two is the
/// minimum that leaves a single usable coordinate.
const MIN_MULTISCALE_EIGENVALUES: usize = 2;

//////////////////////
// Multiscale space //
//////////////////////

/// Build the multiscale diffusion space Palantir operates on.
///
/// Adaptive-bandwidth kernel, symmetric eigendecomposition, back-transform to
/// the row-stochastic eigenvectors, then the multiscale scaling.
///
/// The eigensolver runs on `D^(-1/2) K D^(-1/2)` while Palantir defines its
/// coordinates on the row-stochastic `T = D^(-1) K`. The two are similar,
/// `T = D^(-1/2) M D^(1/2)`, so they share eigenvalues and their eigenvectors
/// differ by `psi = D^(-1/2) v`. Taking the symmetric route and
/// back-transforming is both exact and far better conditioned than a
/// non-symmetric solve.
///
/// ### Params
///
/// * `knn_indices` - kNN indices per cell, self excluded.
/// * `knn_distances` - kNN distances per cell, aligned with `knn_indices`.
/// * `squared_dist` - Whether `knn_distances` holds squared distances, as the
///   ANN backends return for Euclidean.
/// * `n_dcs` - Usable diffusion components to extract before scaling. The
///   eigensolver is asked for `n_dcs + 1` pairs, since the trivial leading
///   eigenvector is always dropped.
/// * `n_eigs` - **Eigenvectors** to retain, not components: the trivial leading
///   one is included in the count and then dropped, so `Some(3)` yields two
///   multiscale components. `None` applies the eigengap heuristic.
/// * `seed` - Seed for the Lanczos start vector.
/// * `lanczos_params` - Optional [LanczosParams] for the eigensolver, defaulted
///   when `None`.
///
/// ### Returns
///
/// The [MultiscaleSpace]: components `n_cells` rows by `n_eigs - 1` columns
/// with the trivial first eigenvector dropped, plus the eigensolver's
/// convergence flag and residual.
pub fn multiscale_components(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    squared_dist: bool,
    n_dcs: usize,
    n_eigs: Option<usize>,
    seed: u64,
    lanczos_params: Option<LanczosParams>,
) -> Result<MultiscaleSpace, BixverseErrors> {
    if n_dcs == 0 {
        return Err(BixverseErrors::InvalidArgument(
            "Palantir: n_dcs must be positive".to_string(),
        ));
    }

    let mut kernel = compute_diffusion_kernel(knn_indices, knn_distances, squared_dist)?;
    let degrees = kernel_row_sums(&kernel);

    // mutates the kernel into its symmetrically normalised form, so the degrees
    // have to be taken before this call. The `_diag` variant is the point: an
    // under-resolved embedding is otherwise indistinguishable from a good one.
    let solve = diffusion_map_from_kernel_diag(&mut kernel, n_dcs + 1, seed, lanczos_params)?;

    // the solver caps the pair count at the matrix dimension and can stop early
    // on an invariant subspace, so the count it returns is a ceiling, not a
    // promise.
    if solve.eigenvalues.len() < MIN_MULTISCALE_EIGENVALUES {
        return Err(BixverseErrors::InvalidArgument(
            "Palantir: fewer than two diffusion components could be resolved".to_string(),
        ));
    }

    let eigenvectors = back_transform_eigenvectors(&solve.eigenvectors, &degrees);
    let n_keep = resolve_n_eigs(&solve.eigenvalues, n_eigs)?;
    let clamped = clamp_eigenvalues(&solve.eigenvalues)?;

    Ok(MultiscaleSpace {
        components: determine_multiscale_space(&clamped, &eigenvectors, Some(n_keep)),
        converged: solve.converged,
        residual: solve.residual,
    })
}

/// Row sums of a CSR kernel, i.e. the diffusion degrees.
///
/// ### Params
///
/// * `kernel` - Symmetric CSR kernel.
///
/// ### Returns
///
/// One degree per row.
fn kernel_row_sums(kernel: &CompressedSparseData2<f32>) -> Vec<f32> {
    (0..kernel.shape.0)
        .map(|i| {
            (kernel.indptr[i]..kernel.indptr[i + 1])
                .map(|idx| kernel.data[idx as usize])
                .sum()
        })
        .collect()
}

/// Map symmetric eigenvectors onto the row-stochastic ones.
///
/// Applies `psi = D^(-1/2) v` and renormalises each component to unit L2, which
/// is what the reference does after its non-symmetric solve.
///
/// The zero-degree branch is defensive only, not a guard: a zero-degree row has
/// already been divided by its own square root inside
/// [diffusion_map_from_kernel], so the kernel and hence `v` are non-finite well
/// before this runs.
///
/// ### Params
///
/// * `eigenvectors` - Symmetric eigenvectors, `n_cells` rows by components.
/// * `degrees` - Diffusion degrees per cell.
///
/// ### Returns
///
/// The back-transformed eigenvectors in the same layout.
fn back_transform_eigenvectors(eigenvectors: &[Vec<f32>], degrees: &[f32]) -> Vec<Vec<f32>> {
    let n = eigenvectors.len();
    if n == 0 {
        return Vec::new();
    }
    let n_comps = eigenvectors[0].len();

    let inv_sqrt_degree: Vec<f32> = degrees
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let mut out = vec![vec![0.0f32; n_comps]; n];
    for c in 0..n_comps {
        let mut norm_sq = 0.0f64;
        for i in 0..n {
            let v = eigenvectors[i][c] * inv_sqrt_degree[i];
            out[i][c] = v;
            norm_sq += (v as f64) * (v as f64);
        }
        let norm = norm_sq.sqrt() as f32;
        if norm > 0.0 {
            for row in out.iter_mut() {
                row[c] /= norm;
            }
        }
    }

    out
}

/// Clamp eigenvalues away from one before the multiscale scaling.
///
/// Non-finite eigenvalues are rejected rather than clamped. `f64::min` returns
/// the other operand when one side is `NaN`, so a `NaN` eigenvalue would come
/// out as [MAX_MULTISCALE_EIGENVALUE] and go on to weight a `NaN` eigenvector at
/// full scale, which looks exactly like a converged component.
///
/// ### Params
///
/// * `eigenvalues` - Raw eigenvalues, descending.
///
/// ### Returns
///
/// The same values with anything above [MAX_MULTISCALE_EIGENVALUE] pulled down,
/// or an error when any eigenvalue is non-finite.
fn clamp_eigenvalues(eigenvalues: &[f64]) -> Result<Vec<f64>, BixverseErrors> {
    eigenvalues
        .iter()
        .map(|&l| {
            if l.is_finite() {
                Ok(l.min(MAX_MULTISCALE_EIGENVALUE))
            } else {
                Err(BixverseErrors::PalantirNonFiniteEigenvalue { eigenvalue: l })
            }
        })
        .collect()
}

/// Pick how many eigenvectors to retain.
///
/// The reference's rule: the largest eigengap wins; if that leaves fewer than
/// three, fall back to the second largest; if that still leaves fewer than
/// three, take three. Ties resolve to the *last* index.
///
/// The floor of three belongs to the heuristic alone. An explicit `requested`
/// bypasses it, as it does in the reference, so `Some(2)` genuinely means two
/// eigenvectors and one multiscale component. Anything below
/// [MIN_MULTISCALE_EIGENVALUES] is refused rather than raised: silently turning
/// a pinned count into a different one changes every waypoint distance,
/// boundary and geodesic downstream.
///
/// ### Params
///
/// * `eigenvalues` - Eigenvalues, descending.
/// * `requested` - Explicit count, bypassing the heuristic.
///
/// ### Returns
///
/// The number of eigenvectors to feed to the multiscale scaling, never above
/// the number available, or an error when `requested` is below the minimum.
fn resolve_n_eigs(eigenvalues: &[f64], requested: Option<usize>) -> Result<usize, BixverseErrors> {
    let available = eigenvalues.len();
    if let Some(n) = requested {
        if n < MIN_MULTISCALE_EIGENVALUES {
            return Err(BixverseErrors::PalantirNEigsTooSmall {
                n_eigs: n,
                minimum: MIN_MULTISCALE_EIGENVALUES,
            });
        }
        return Ok(n.min(available));
    }

    let gaps: Vec<f64> = eigenvalues.windows(2).map(|w| w[0] - w[1]).collect();
    if gaps.is_empty() {
        return Ok(MIN_MULTISCALE_EIGS.min(available));
    }

    let mut order: Vec<usize> = (0..gaps.len()).collect();
    order.sort_by(|&a, &b| gaps[a].total_cmp(&gaps[b]));

    let mut n_eigs = order[order.len() - 1] + 1;
    if n_eigs < MIN_MULTISCALE_EIGS && order.len() >= 2 {
        n_eigs = order[order.len() - 2] + 1;
    }
    if n_eigs < MIN_MULTISCALE_EIGS {
        n_eigs = MIN_MULTISCALE_EIGS;
    }

    Ok(n_eigs.min(available))
}

///////////////////////////
// Geometric bookkeeping //
///////////////////////////

/// Min-max scale every component to `[0, 1]` in place.
///
/// ### Notes
///
/// A constant component maps to all zeros rather than `NaN`, matching
/// `sklearn.preprocessing.minmax_scale`.
///
/// ### Params
///
/// * `data` - Multiscale components, cells by components. Mutated in place.
///
/// ### Returns
///
/// Nothing; `data` is rewritten.
pub fn minmax_scale_columns(data: &mut [Vec<f32>]) {
    if data.is_empty() {
        return;
    }
    let n_cols = data[0].len();

    for c in 0..n_cols {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for row in data.iter() {
            lo = lo.min(row[c]);
            hi = hi.max(row[c]);
        }

        let range = hi - lo;
        if range > 0.0 {
            for row in data.iter_mut() {
                row[c] = (row[c] - lo) / range;
            }
        } else {
            for row in data.iter_mut() {
                row[c] = 0.0;
            }
        }
    }
}

/// Cells sitting at the extremes of the multiscale space.
///
/// The per-component argmax and argmin cells, deduplicated. These are the
/// candidate endpoints of the manifold: the start cell is snapped to one of
/// them, and so is every terminal state candidate.
///
/// ### Notes
///
/// Ties resolve to the first index, matching `pandas.Series.idxmax`.
///
/// ### Params
///
/// * `data` - Multiscale components, cells by components.
///
/// ### Returns
///
/// Boundary cell indices, ascending and unique.
pub fn boundary_cells(data: &[Vec<f32>]) -> Vec<usize> {
    if data.is_empty() {
        return Vec::new();
    }
    let n_cols = data[0].len();
    let mut found: FxHashSet<usize> = FxHashSet::default();

    for c in 0..n_cols {
        let (mut arg_lo, mut arg_hi) = (0usize, 0usize);
        let (mut lo, mut hi) = (data[0][c], data[0][c]);
        for (i, row) in data.iter().enumerate().skip(1) {
            // Strict comparisons keep the first index on ties.
            if row[c] < lo {
                lo = row[c];
                arg_lo = i;
            }
            if row[c] > hi {
                hi = row[c];
                arg_hi = i;
            }
        }
        found.insert(arg_lo);
        found.insert(arg_hi);
    }

    let mut out: Vec<usize> = found.into_iter().collect();
    out.sort_unstable();
    out
}

/// Nearest candidate to a query cell in the multiscale space.
///
/// ### Params
///
/// * `data` - Multiscale components, cells by components.
/// * `candidates` - Cell indices to search over. Must not be empty.
/// * `query` - The cell to match.
///
/// ### Returns
///
/// The candidate at the smallest Euclidean distance, first index on ties, or an
/// error when `candidates` is empty. Both call sites pass the boundary set, so
/// the error names that rather than terminal states, which are not involved in
/// the start-cell snap at all.
pub fn nearest_candidate(
    data: &[Vec<f32>],
    candidates: &[usize],
    query: usize,
) -> Result<usize, BixverseErrors> {
    let first = *candidates
        .first()
        .ok_or(BixverseErrors::PalantirNoBoundaryCells)?;

    let mut best = first;
    let mut best_dist = f32::INFINITY;

    for &c in candidates {
        let dist: f32 = data[c]
            .iter()
            .zip(data[query].iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        if dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }

    Ok(best)
}

/// Assemble the waypoint set.
///
/// Max-min samples, boundary cells and any user-supplied terminal states, with
/// the start cell pulled to the front.
///
/// The reference orders the tail lexicographically by cell name; here it is
/// ascending by cell index, which is the deterministic equivalent.
///
/// ### Params
///
/// * `data` - Multiscale components, cells by components.
/// * `num_waypoints` - Target sample count, raised to at least
///   `max(3, n_components)` as the reference does.
/// * `boundaries` - Boundary cells from [boundary_cells].
/// * `terminal_states` - Optional user-supplied terminal states.
/// * `start_cell` - The start cell, always placed first.
/// * `seed` - Seed for the max-min sampler.
///
/// ### Returns
///
/// Waypoint cell indices, `start_cell` first and the remainder ascending.
pub fn assemble_waypoints(
    data: &[Vec<f32>],
    num_waypoints: usize,
    boundaries: &[usize],
    terminal_states: Option<&[usize]>,
    start_cell: usize,
    seed: u64,
) -> Vec<usize> {
    if data.is_empty() {
        return Vec::new();
    }

    let n_dims = data[0].len();
    let target = num_waypoints.max(MIN_MULTISCALE_EIGS.max(n_dims));

    let mut set: FxHashSet<usize> = max_min_sampling(data, target, seed).into_iter().collect();
    set.extend(boundaries.iter().copied());
    if let Some(states) = terminal_states {
        set.extend(states.iter().copied());
    }
    set.remove(&start_cell);

    let mut rest: Vec<usize> = set.into_iter().collect();
    rest.sort_unstable();

    let mut out = Vec::with_capacity(rest.len() + 1);
    out.push(start_cell);
    out.extend(rest);
    out
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Column scaling maps each component onto 0 to 1 independently.
    #[test]
    fn test_minmax_scale_maps_to_unit_range() {
        let mut data = vec![vec![1.0, 10.0], vec![3.0, 20.0], vec![5.0, 30.0]];
        minmax_scale_columns(&mut data);

        assert_relative_eq!(data[0][0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(data[1][0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(data[2][0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data[1][1], 0.5, epsilon = 1e-6);
    }

    /// A constant column has zero range, so it scales to zero instead of dividing by it.
    #[test]
    fn test_minmax_scale_constant_column_is_zero_not_nan() {
        let mut data = vec![vec![7.0, 1.0], vec![7.0, 2.0]];
        minmax_scale_columns(&mut data);

        assert!(data[0][0].is_finite() && data[1][0].is_finite());
        assert_relative_eq!(data[0][0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(data[1][0], 0.0, epsilon = 1e-6);
    }

    /// Boundary cells are the argmin and argmax of every component.
    #[test]
    fn test_boundary_cells_are_column_extremes() {
        // Column 0 extremes are cells 3 and 0; column 1 extremes are 1 and 2.
        let data = vec![
            vec![0.0, 0.5],
            vec![1.0, 0.9],
            vec![2.0, 0.1],
            vec![3.0, 0.4],
        ];

        assert_eq!(boundary_cells(&data), vec![0, 1, 2, 3]);
    }

    /// Ties resolve to the first index, and the boundary set is deduplicated.
    #[test]
    fn test_boundary_cells_take_first_index_on_ties() {
        // Every cell shares the same value, so the first index wins both ways.
        let data = vec![vec![1.0], vec![1.0], vec![1.0]];

        assert_eq!(boundary_cells(&data), vec![0]);
    }

    /// The closest candidate wins, and an empty candidate set errors rather than returning zero.
    #[test]
    fn test_nearest_candidate_picks_closest() {
        let data = vec![vec![0.0, 0.0], vec![10.0, 0.0], vec![1.0, 1.0]];

        assert_eq!(nearest_candidate(&data, &[0, 1], 2).unwrap(), 0);
        assert!(nearest_candidate(&data, &[], 0).is_err());
    }

    /// Waypoints lead with the start cell, then run ascending with no repeat of it, keeping boundaries and pinned cells.
    #[test]
    fn test_waypoints_start_first_then_ascending() {
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32, (i % 7) as f32, (i % 3) as f32])
            .collect();
        let boundaries = boundary_cells(&data);

        let waypoints = assemble_waypoints(&data, 20, &boundaries, Some(&[42]), 5, 42);

        assert_eq!(waypoints[0], 5);
        assert!(waypoints.windows(2).skip(1).all(|w| w[0] < w[1]));
        assert!(!waypoints[1..].contains(&5));
        assert!(waypoints.contains(&42));
        for b in &boundaries {
            if *b != 5 {
                assert!(waypoints.contains(b), "boundary {b} missing");
            }
        }
    }

    /// The eigengap heuristic falls back through the gaps until one clears the floor of three.
    #[test]
    fn test_resolve_n_eigs_takes_largest_gap() {
        // Gaps are [0.1, 0.5, 0.05]; the largest sits at index 1 -> n_eigs 2,
        // which is below the floor, so the second largest at index 0 is tried
        // (n_eigs 1), and that also fails, leaving the floor of three.
        let eigenvalues = [1.0, 0.9, 0.4, 0.35];

        assert_eq!(resolve_n_eigs(&eigenvalues, None).unwrap(), 3);
    }

    /// A gap late in the spectrum is taken when it is the largest, with the floor never binding.
    #[test]
    fn test_resolve_n_eigs_uses_a_late_gap_when_it_is_largest() {
        // Gaps are [0.02, 0.03, 0.02, 0.63]; the largest sits at index 3, so
        // n_eigs is 4 and the floor never comes into play.
        let eigenvalues = [1.0, 0.98, 0.95, 0.93, 0.30];

        assert_eq!(resolve_n_eigs(&eigenvalues, None).unwrap(), 4);
    }

    /// An explicit request is passed through, floor included, and capped at what exists.
    #[test]
    fn test_resolve_n_eigs_honours_explicit_requests_below_the_floor() {
        // The floor of three is the *heuristic's*, not the function's. The
        // reference applies it only inside the `n_eigs is None` branch, so a
        // pinned `Some(2)` genuinely means two eigenvectors and one component.
        // Raising it silently would change every waypoint distance downstream.
        let eigenvalues = [1.0, 0.9, 0.5, 0.2];

        assert_eq!(resolve_n_eigs(&eigenvalues, Some(2)).unwrap(), 2);
        assert_eq!(resolve_n_eigs(&eigenvalues, Some(4)).unwrap(), 4);
        assert_eq!(resolve_n_eigs(&eigenvalues, Some(99)).unwrap(), 4);
    }

    /// Fewer than two eigenvectors leaves nothing to embed, so it errors.
    #[test]
    fn test_resolve_n_eigs_rejects_a_request_below_two() {
        let eigenvalues = [1.0, 0.9, 0.5, 0.2];

        assert!(matches!(
            resolve_n_eigs(&eigenvalues, Some(1)),
            Err(BixverseErrors::PalantirNEigsTooSmall { n_eigs: 1, .. })
        ));
        assert!(matches!(
            resolve_n_eigs(&eigenvalues, Some(0)),
            Err(BixverseErrors::PalantirNEigsTooSmall { n_eigs: 0, .. })
        ));
    }

    /// Regression: every return path handed back the floor of three, past the end of a shorter spectrum.
    #[test]
    fn test_resolve_n_eigs_never_exceeds_what_is_available() {
        // Every return path used to hand back the floor of three regardless,
        // and the caller then indexed eigenvalues[2] on a two-element slice.
        let two = [1.0, 0.5];
        let one = [1.0];

        assert_eq!(resolve_n_eigs(&two, Some(2)).unwrap(), 2);
        assert_eq!(resolve_n_eigs(&two, Some(99)).unwrap(), 2);
        assert_eq!(resolve_n_eigs(&two, None).unwrap(), 2);
        assert_eq!(resolve_n_eigs(&one, Some(3)).unwrap(), 1);
        assert_eq!(resolve_n_eigs(&one, None).unwrap(), 1);
        assert_eq!(resolve_n_eigs(&[], None).unwrap(), 0);
    }

    /// The clamp binds at its ceiling, leaves everything below untouched, and sits above real signal.
    #[test]
    fn test_clamp_eigenvalues_pins_the_ceiling() {
        // A scale of 1e300 is finite too, so "the scale is finite" pins
        // nothing. What matters is where the ceiling actually sits and that
        // everything below it is untouched.
        let clamped = clamp_eigenvalues(&[1.0, 1.5, 0.999_999, 0.5]).unwrap();

        assert_eq!(clamped[0], MAX_MULTISCALE_EIGENVALUE);
        assert_eq!(clamped[1], MAX_MULTISCALE_EIGENVALUE);
        assert_eq!(clamped[2], 0.999_999);
        assert_eq!(clamped[3], 0.5);

        // The clamp must leave a genuine leading eigenvalue alone. The leading
        // spectral gap of a knn = 30 diffusion operator is around 4e-5 at 10k
        // cells, so a ceiling anywhere near 1 - 1e-4 would bind on real signal.
        const {
            assert!(1.0 - MAX_MULTISCALE_EIGENVALUE < 4e-5);
        }
    }

    /// A non-finite eigenvalue errors rather than passing the clamp unchanged.
    #[test]
    fn test_clamp_eigenvalues_rejects_non_finite_input() {
        // `f64::min` returns the other operand for a NaN, so the clamp would
        // otherwise hand back MAX_MULTISCALE_EIGENVALUE and weight a NaN
        // eigenvector at full scale.
        assert!(matches!(
            clamp_eigenvalues(&[1.0, f64::NAN]),
            Err(BixverseErrors::PalantirNonFiniteEigenvalue { .. })
        ));
        assert!(clamp_eigenvalues(&[f64::INFINITY]).is_err());
    }

    /// Back-transformed eigenvectors come out unit L2 with their direction intact.
    #[test]
    fn test_back_transform_renormalises_to_unit_l2() {
        let eigenvectors = vec![vec![0.6, 1.0], vec![0.8, 2.0]];
        let degrees = vec![4.0, 9.0];

        let out = back_transform_eigenvectors(&eigenvectors, &degrees);

        for c in 0..2 {
            let norm: f32 = out.iter().map(|r| r[c] * r[c]).sum::<f32>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }

        // Direction is preserved: v / sqrt(D) before renormalisation gives
        // 0.6/2 = 0.3 and 0.8/3 = 0.2667, so cell 0 stays the larger entry.
        assert!(out[0][0] > out[1][0]);
    }

    /// A zero-degree row back-transforms to zero instead of dividing by zero.
    #[test]
    fn test_back_transform_survives_zero_degree_rows() {
        let eigenvectors = vec![vec![1.0], vec![2.0]];
        let degrees = vec![0.0, 4.0];

        let out = back_transform_eigenvectors(&eigenvectors, &degrees);

        assert!(out.iter().all(|r| r[0].is_finite()));
        assert_relative_eq!(out[0][0], 0.0, epsilon = 1e-6);
    }
}
