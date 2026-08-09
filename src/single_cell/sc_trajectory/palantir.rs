//! Palantir trajectory inference.
//!
//! Diffusion kernel, multiscale space, waypoint sampling, geodesic pseudotime,
//! a pseudotime-directed Markov chain over the waypoints, terminal states, and
//! absorption probabilities projected back onto every cell.
//!
//! Memory is dominated by the geodesic and weight matrices at
//! `n_waypoints * n_cells * 12` bytes, so 14.4 KB per cell at the reference's
//! 1200 waypoints. `num_waypoints` is the knob for that and for the dense
//! absorbing solve.
//!
//! ### Deliberate divergences from the reference implementation
//!
//! 1. Waypoints after the start cell are ordered ascending by cell index; the
//!    reference sorts lexicographically by cell name. Same set, different order,
//!    which propagates into the row order of the internal matrices.
//! 2. Terminal states are likewise ordered by index, which sets the column order
//!    of `branch_probs`.
//! 3. The stationary distribution comes from power iteration on the lazy chain
//!    `(I + T) / 2` rather than ARPACK with an unseeded start vector. Same
//!    eigenvector, deterministic, and immune to periodicity.
//! 4. Diffusion components come from a symmetric Lanczos solve plus a
//!    back-transform rather than a non-symmetric solve. Mathematically identical
//!    and far better conditioned.
//! 5. Eigenvalues are clamped away from one before the multiscale scaling.
//! 6. A zero adaptive bandwidth is replaced by the smallest positive one instead
//!    of silently killing the row.
//! 7. A waypoint whose neighbours are all pruned keeps its nearest neighbour at
//!    full weight. The reference leaves the row empty, which makes the waypoint
//!    a dead end that the absorbing solve then drops; this keeps the chain
//!    row-stochastic and the waypoint attached to the manifold instead.
//! 8. `(I - Q)` is densified and solved with a partial-pivot LU rather than a
//!    sparse SuperLU factorisation, with a documented waypoint ceiling.
//!    Waypoints that cannot reach any terminal state are dropped from the system
//!    and reported through `stranded_waypoints`; the reference lets the
//!    factorisation fail and falls back to a pseudo-inverse.
//! 9. Graph repair adds each bridging edge from the farthest reachable cell in
//!    one pass per component rather than iterating the reference's full search.
//! 10. Degenerate cases the reference turns into `NaN` or empty frames return
//!     typed errors here.
//! 11. `alpha`, the anisotropic kernel exponent, is not exposed. Its reference
//!     default of zero makes the step a no-op.
//!
//! ### References
//!
//! Setty, et al., Nat. Biotechnol., 2019. "Characterization of cell fate
//! probabilities in single-cell data with Palantir."

use faer::Mat;
use rustc_hash::FxHashMap;

use crate::prelude::*;
use crate::single_cell::sc_trajectory::diffusion::{
    assemble_waypoints, boundary_cells, minmax_scale_columns, multiscale_components,
    nearest_candidate,
};
use crate::single_cell::sc_trajectory::markov::{
    absorption_probabilities, branch_entropy, build_waypoint_transitions, detect_terminal_states,
    project_to_cells,
};
use crate::single_cell::sc_trajectory::pseudotime::{
    build_symmetric_knn_graph, compute_pseudotime, connect_graph,
};

//////////////////////
// Enums and strucs //
//////////////////////

/// Parameters for Palantir trajectory inference.
#[derive(Clone, Debug)]
pub struct PalantirParams {
    /// Diffusion components to extract before the multiscale scaling.
    /// Reference default: 10.
    pub n_dcs: usize,
    /// Multiscale components to retain. `None` selects the count from the
    /// largest eigengap, as the reference does.
    pub n_eigs: Option<usize>,
    /// Neighbours in the reference's self-inclusive convention. Reference
    /// default: 30. The crate's kNN search drops the self hit, so it is asked
    /// for `knn - 1`.
    pub knn: usize,
    /// Target waypoint count for the max-min sampler. Reference default: 1200.
    /// Drives both the geodesic matrix size and the dense absorbing solve.
    pub num_waypoints: usize,
    /// Min-max scale each multiscale component to `[0, 1]` before any distance
    /// is taken. Reference default: true.
    pub scale_components: bool,
    /// Use `early_cell` directly rather than snapping it to the nearest
    /// diffusion-map boundary cell. Reference default: false.
    pub use_early_cell_as_start: bool,
    /// Iteration cap for the pseudotime refinement. The counter starts at one,
    /// so this permits at most `max_iterations - 1` passes, matching the
    /// reference, and anything below two is rejected rather than silently
    /// skipping the refinement. Reference default: 25.
    pub max_iterations: usize,
    /// Fate probabilities below this are zeroed in the returned matrix without
    /// renormalisation. Entropy is computed before it is applied. Reference
    /// default: 0.01.
    pub branch_prob_threshold: f32,
    /// Backend settings for the kNN search over the multiscale space. `k` and
    /// `ann_dist` are overridden by this module; only the method choice and its
    /// backend-specific tuning are read.
    pub knn_params: KnnParams,
    /// [LanczosParams] for the diffusion-map eigendecomposition. A trajectory is
    /// a long, thin manifold, which is the worst case for the eigensolver: the
    /// diffusion eigenvalues cluster as `O(1 / n_cells^2)` and an
    /// under-resolved decomposition scrambles the whole embedding. This is the
    /// knob to reach for when pseudotime does not track the manifold.
    pub lanczos_params: LanczosParams,
}

impl PalantirParams {
    /// Parameters matching the reference's `run_palantir` defaults.
    ///
    /// ### Returns
    ///
    /// Self with the reference defaults.
    pub fn new() -> Self {
        Self {
            n_dcs: 10,
            n_eigs: None,
            knn: 30,
            num_waypoints: 1200,
            scale_components: true,
            use_early_cell_as_start: false,
            max_iterations: 25,
            branch_prob_threshold: 0.01,
            knn_params: KnnParams::new(),
            lanczos_params: LanczosParams::new(),
        }
    }
}

impl Default for PalantirParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Results of a Palantir run.
pub struct PalantirResult {
    /// Pseudotime per cell, min-max scaled to `[0, 1]` with the start cell at
    /// 0.
    pub pseudotime: Vec<f32>,
    /// Differentiation entropy per cell, natural log, computed on the
    /// row-renormalised fate probabilities *before* thresholding.
    pub entropy: Vec<f32>,
    /// Fate probabilities, cells by terminal states. Column `t` corresponds to
    /// `terminal_states[t]`. Sub-threshold values are zeroed without
    /// renormalisation, so rows need not sum to one.
    pub branch_probs: Mat<f32>,
    /// Terminal state cell indices, ascending. Sets the column order above.
    pub terminal_states: Vec<usize>,
    /// Waypoint cell indices. `waypoints[0]` is the start cell; the remainder
    /// is ascending.
    pub waypoints: Vec<usize>,
    /// The start cell actually used.
    pub start_cell: usize,
    /// Multiscale components, cells by components, after scaling if requested.
    pub multiscale: Mat<f32>,
    /// Refinement passes actually run.
    pub iterations: usize,
    /// Whether the refinement met the correlation criterion before the cap.
    pub converged: bool,
    /// Bridging edges the connectivity repair had to add. Anything non-zero
    /// means the kNN graph was disconnected and `knn` is probably too small.
    pub repair_edges: usize,
    /// Waypoints the pseudotime pruning stranded, meaning no terminal state is
    /// reachable from them. They carry no fate probabilities, so cells whose
    /// weight sits on them have rows summing below one. A large count means the
    /// chain is over-pruned.
    pub stranded_waypoints: usize,
}

/////////////
// Helpers //
/////////////

/// Build the symmetric kNN graph geodesics are measured over.
///
/// The search runs on the multiscale space rather than the caller's original
/// embedding, because that is where the reference defines its geodesics. `k`
/// and the metric are forced; the caller's backend choice and its tuning are
/// kept.
///
/// ### Params
///
/// * `multiscale` - Multiscale components, cells by components.
/// * `params` - The run parameters, read for `knn` and the kNN backend.
/// * `seed` - Seed for the kNN backend.
/// * `verbose` - Progress reporting for the search.
///
/// ### Returns
///
/// A symmetric CSR adjacency over the cells.
fn geodesic_graph(
    multiscale: faer::MatRef<f32>,
    params: &PalantirParams,
    seed: u64,
    verbose: bool,
) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
    let n_cells = multiscale.nrows();

    let mut knn_params = params.knn_params.clone();

    // the reference's knn counts the self hit; the crate's search drops it.
    knn_params.k = params
        .knn
        .saturating_sub(1)
        .clamp(1, n_cells.saturating_sub(1).max(1));
    knn_params.ann_dist = "euclidean".to_string();

    let (indices, distances) =
        generate_knn_with_dist(multiscale, &knn_params, true, false, seed as usize, verbose)?;
    let distances = distances.ok_or(BixverseErrors::InvalidArgument(
        "Palantir: the multiscale kNN search returned no distances".to_string(),
    ))?;

    // euclidean distances come back squared from the ANN backends.
    build_symmetric_knn_graph(&indices, &distances, true)
}

/// Map terminal state cell indices onto their waypoint positions.
///
/// ### Params
///
/// * `waypoints` - Waypoint cell indices.
/// * `states` - Terminal state cell indices.
///
/// ### Returns
///
/// Positions within `waypoints`, ascending and unique, or an error when a state
/// is not a waypoint. [assemble_waypoints] guarantees it is, so this only fires
/// when the two are called out of step.
fn waypoint_positions(waypoints: &[usize], states: &[usize]) -> Result<Vec<usize>, BixverseErrors> {
    let lookup: FxHashMap<usize, usize> = waypoints
        .iter()
        .enumerate()
        .map(|(pos, &cell)| (cell, pos))
        .collect();

    let mut out: Vec<usize> = states
        .iter()
        .map(|&s| {
            lookup
                .get(&s)
                .copied()
                .ok_or(BixverseErrors::PalantirTerminalStateNotAWaypoint {
                    state: s,
                    n_waypoints: waypoints.len(),
                })
        })
        .collect::<Result<Vec<usize>, BixverseErrors>>()?;

    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Zero fate probabilities below a threshold, without renormalising.
///
/// The reference does the same, so rows generally stop summing to one.
///
/// ### Params
///
/// * `branch_probs` - Fate probabilities, cells by terminal states. Mutated.
/// * `threshold` - Values strictly below this become zero.
///
/// ### Returns
///
/// Nothing; `branch_probs` is rewritten.
fn apply_probability_threshold(branch_probs: &mut Mat<f32>, threshold: f32) {
    if threshold <= 0.0 {
        return;
    }
    for i in 0..branch_probs.nrows() {
        for j in 0..branch_probs.ncols() {
            if branch_probs[(i, j)] < threshold {
                branch_probs[(i, j)] = 0.0;
            }
        }
    }
}

//////////
// Main //
//////////

/// Run Palantir trajectory inference on a precomputed kNN graph.
///
/// The supplied kNN graph feeds the diffusion kernel only. Geodesics are taken
/// over a second kNN graph built on the multiscale space itself, which is where
/// the reference measures them.
///
/// ### Params
///
/// * `knn_indices` - kNN indices per cell, self excluded.
/// * `knn_distances` - kNN distances per cell, aligned with `knn_indices`.
/// * `squared_dist` - Whether `knn_distances` holds squared distances, as
///   [crate::prelude::generate_knn_with_dist] returns for `"euclidean"`.
/// * `early_cell` - Index of the user's early cell.
/// * `terminal_states` - Optional terminal state cell indices. When `None` they
///   are detected from the waypoint Markov chain.
/// * `params` - Optional [PalantirParams], defaulted when `None`.
/// * `seed` - Seed for the Lanczos start vector, the waypoint sampler and the
///   multiscale kNN search.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The [PalantirResult].
///
/// ### References
///
/// Setty, et al., Nat. Biotechnol., 2019.
#[allow(clippy::too_many_arguments)]
pub fn run_palantir(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    squared_dist: bool,
    early_cell: usize,
    terminal_states: Option<&[usize]>,
    params: Option<PalantirParams>,
    seed: u64,
    verbose: usize,
) -> Result<PalantirResult, BixverseErrors> {
    let params = params.unwrap_or_default();
    let verbosity = parse_verbosity_level(verbose);
    let n_cells = knn_indices.len();

    if n_cells == 0 {
        return Err(BixverseErrors::InvalidArgument(
            "Palantir: the kNN graph is empty".to_string(),
        ));
    }
    if early_cell >= n_cells {
        return Err(BixverseErrors::PalantirEarlyCellOutOfRange {
            early_cell,
            n_cells,
        });
    }
    if let Some(states) = terminal_states
        && let Some(&bad) = states.iter().find(|&&s| s >= n_cells)
    {
        return Err(BixverseErrors::PalantirTerminalStateOutOfRange {
            state: bad,
            n_cells,
        });
    }

    if verbosity.normal_verbosity() {
        println!("Building the multiscale diffusion space...");
    }
    let mut multiscale = multiscale_components(
        knn_indices,
        knn_distances,
        squared_dist,
        params.n_dcs,
        params.n_eigs,
        seed,
        Some(params.lanczos_params),
    )?;
    if params.scale_components {
        minmax_scale_columns(&mut multiscale);
    }
    let n_dims = multiscale[0].len();

    let boundaries = boundary_cells(&multiscale);
    let start_cell = if params.use_early_cell_as_start {
        early_cell
    } else {
        nearest_candidate(&multiscale, &boundaries, early_cell)?
    };

    let waypoints = assemble_waypoints(
        &multiscale,
        params.num_waypoints,
        &boundaries,
        terminal_states,
        start_cell,
        seed,
    );

    if verbosity.normal_verbosity() {
        println!(
            "Sampled {} waypoints; start cell is {start_cell}.",
            waypoints.len()
        );
    }

    let multiscale_mat = Mat::<f32>::from_fn(n_cells, n_dims, |i, j| multiscale[i][j]);
    let graph = geodesic_graph(
        multiscale_mat.as_ref(),
        &params,
        seed,
        verbosity.detailed_verbosity(),
    )?;
    let (graph, repair_edges) = connect_graph(&graph, &multiscale, start_cell)?;
    if repair_edges > 0 && verbosity.normal_verbosity() {
        println!(
            "[WARNING] The kNN graph was disconnected; added {repair_edges} bridging edges. Consider a larger knn."
        );
    }

    let pseudotime = compute_pseudotime(
        &graph,
        &waypoints,
        start_cell,
        params.max_iterations,
        verbosity,
    )?;

    let wp_data: Vec<Vec<f32>> = waypoints.iter().map(|&w| multiscale[w].clone()).collect();
    let wp_pseudotime: Vec<f32> = waypoints
        .iter()
        .map(|&w| pseudotime.pseudotime[w])
        .collect();

    if verbosity.normal_verbosity() {
        println!("Constructing the waypoint Markov chain...");
    }
    let transitions = build_waypoint_transitions(
        &wp_data,
        &wp_pseudotime,
        params.knn,
        verbosity.detailed_verbosity(),
    )?;

    let mut absorbing = match terminal_states {
        Some(states) => waypoint_positions(&waypoints, states)?,
        None => detect_terminal_states(&transitions, &wp_data, &wp_pseudotime)?,
    };

    // order by cell index, not waypoint position: `terminal_states` is exported
    // ascending and it has to stay aligned with the `branch_probs` columns.
    absorbing.sort_unstable_by_key(|&p| waypoints[p]);

    if verbosity.normal_verbosity() {
        println!("Solving for {} terminal states...", absorbing.len());
    }
    let (branch_probs_wp, stranded_waypoints) = absorption_probabilities(&transitions, &absorbing)?;
    if stranded_waypoints > 0 && verbosity.normal_verbosity() {
        println!(
            "[WARNING] {stranded_waypoints} waypoints cannot reach any terminal state and were left without fate probabilities."
        );
    }

    let mut branch_probs = project_to_cells(
        &pseudotime.weights,
        pseudotime.n_waypoints,
        pseudotime.n_cells,
        branch_probs_wp.as_ref(),
    )?;

    // entropy is taken before the threshold, matching the reference.
    let entropy = branch_entropy(branch_probs.as_ref());
    apply_probability_threshold(&mut branch_probs, params.branch_prob_threshold);

    let terminal_states: Vec<usize> = absorbing.iter().map(|&p| waypoints[p]).collect();

    Ok(PalantirResult {
        pseudotime: pseudotime.pseudotime,
        entropy,
        branch_probs,
        terminal_states,
        waypoints,
        start_cell,
        multiscale: multiscale_mat,
        iterations: pseudotime.iterations,
        converged: pseudotime.converged,
        repair_edges,
        stranded_waypoints,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic jitter in `[-0.5, 0.5)`, so fixtures need no RNG.
    ///
    /// ### Params
    ///
    /// * `i` - Sample index.
    /// * `salt` - Distinguishes independent noise streams.
    ///
    /// ### Returns
    ///
    /// A reproducible pseudo-random offset.
    fn jitter(i: usize, salt: usize) -> f32 {
        let h = (i.wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(40_503)) % 10_007;
        h as f32 / 10_007.0 - 0.5
    }

    /// kNN graph over a synthetic embedding, as `run_palantir` expects it.
    ///
    /// ### Params
    ///
    /// * `coords` - Cell coordinates, cells by dimensions.
    /// * `k` - Neighbours per cell, self excluded.
    ///
    /// ### Returns
    ///
    /// `(indices, squared distances)` from an exhaustive Euclidean search.
    fn knn_of(coords: &[Vec<f32>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let n = coords.len();
        let d = coords[0].len();
        let mat = Mat::<f32>::from_fn(n, d, |i, j| coords[i][j]);

        let mut params = KnnParams::new();
        params.knn_method = "exhaustive".to_string();
        params.ann_dist = "euclidean".to_string();
        params.k = k;

        let (indices, distances) =
            generate_knn_with_dist(mat.as_ref(), &params, true, false, 42, false).unwrap();
        (indices, distances.unwrap())
    }

    /// A noisy one-dimensional trajectory.
    ///
    /// Genuinely noisy on purpose. A clean lattice makes every neighbourhood
    /// identical, which is not what the adaptive bandwidths are built for.
    ///
    /// ### Params
    ///
    /// * `n` - Cell count.
    ///
    /// ### Returns
    ///
    /// Coordinates ordered along the trajectory.
    fn linear_manifold(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                vec![t * 10.0, jitter(i, 1) * 0.6, jitter(i, 2) * 0.6]
            })
            .collect()
    }

    /// Rate at which the two arms separate in the Y fixture, per unit of arc.
    ///
    /// Load bearing in both directions. Too slow and the arms sit inside each
    /// other's noise for most of their length, so the kNN bridges them and the
    /// branch never resolves. Too fast and they detach from the trunk entirely,
    /// giving a kNN graph with three connected components; the diffusion kernel
    /// then has eigenvalue one with multiplicity three, the eigenvectors inside
    /// that degenerate eigenspace are arbitrary rotations, and every result
    /// downstream turns on the eigensolver's convergence budget rather than on
    /// the data.
    const ARM_DIVERGENCE: f32 = 1.4;

    /// A Y-shaped manifold: a shared trunk splitting into two arms.
    ///
    /// Cells `0..trunk` run the trunk; the next `arm` cells are the first
    /// branch, the last `arm` cells the second.
    ///
    /// ### Params
    ///
    /// * `trunk` - Cells on the shared trunk.
    /// * `arm` - Cells per branch.
    ///
    /// ### Returns
    ///
    /// Coordinates for `trunk + 2 * arm` cells.
    fn y_manifold(trunk: usize, arm: usize) -> Vec<Vec<f32>> {
        let mut coords = Vec::with_capacity(trunk + 2 * arm);

        for i in 0..trunk {
            let t = i as f32 / trunk as f32 * 5.0;
            coords.push(vec![t, jitter(i, 3) * 0.3]);
        }
        // The arms step away from the axis immediately. Letting them start at
        // y = 0 puts them within the noise of each other at the branch point,
        // and the kNN then bridges the two, smearing the split.
        for i in 0..arm {
            let t = (i + 1) as f32 / arm as f32 * 5.0;
            coords.push(vec![5.0 + t, t * ARM_DIVERGENCE + jitter(i, 4) * 0.3]);
        }
        for i in 0..arm {
            let t = (i + 1) as f32 / arm as f32 * 5.0;
            coords.push(vec![5.0 + t, -t * ARM_DIVERGENCE + jitter(i, 5) * 0.3]);
        }

        coords
    }

    /// Parameters sized for the small synthetic fixtures.
    ///
    /// ### Returns
    ///
    /// [PalantirParams] with a modest waypoint count and an exact kNN backend.
    fn test_params() -> PalantirParams {
        let mut params = PalantirParams::new();
        params.knn = 24;
        params.num_waypoints = 250;
        params.knn_params.knn_method = "exhaustive".to_string();
        // Pin the component count. The eigengap heuristic keeps six or more on
        // these fixtures, and the higher harmonics of a simple manifold
        // oscillate across the full scaled range, which inflates the arc length
        // until the spatial bandwidth swamps every pseudotime gap and the
        // directed pruning stops cutting anything. Unrelated to the eigensolver.
        params.n_eigs = Some(3);
        params
    }

    #[test]
    fn test_linear_trajectory_has_monotone_pseudotime() {
        let coords = linear_manifold(120);
        let (indices, distances) = knn_of(&coords, 15);

        let res = run_palantir(
            &indices,
            &distances,
            true,
            0,
            None,
            Some(test_params()),
            42,
            0,
        )
        .unwrap();

        let truth: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let corr = crate::core::math::vector_helpers::pearson_correlation(&res.pseudotime, &truth)
            .unwrap();

        assert!(
            corr.abs() > 0.95,
            "pseudotime does not track the manifold: r = {corr}"
        );
        assert_eq!(res.terminal_states.len(), 1);
    }

    #[test]
    fn test_y_branch_terminal_states_lie_in_the_arms() {
        // What automatic detection can be held to on a fixture this size, and
        // no more. The candidates are drawn from the per-axis extremes of the
        // multiscale space, and a 600-cell thin manifold has a diffusion
        // spectrum clustered to within 1e-5, so the individual components are
        // not determined well enough to place all three tips at extremes. That
        // is the reference's heuristic behaving as designed, not a defect, so
        // this pins the properties that are genuinely determined: something is
        // found, it sits in a branch rather than the trunk, and the export
        // ordering holds. `test_y_branch_fate_probabilities_split_at_the_root`
        // covers the downstream machinery with the tips supplied explicitly.
        let (trunk, arm) = (200usize, 200usize);
        let coords = y_manifold(trunk, arm);
        let (indices, distances) = knn_of(&coords, 15);

        let res = run_palantir(
            &indices,
            &distances,
            true,
            0,
            None,
            Some(test_params()),
            42,
            0,
        )
        .unwrap();

        // The arms must stay attached to the trunk. Detaching them gives a kNN
        // graph with three connected components, which puts eigenvalue one at
        // multiplicity three and makes every result downstream an artefact of
        // the eigensolver's convergence budget rather than of the data.
        assert_eq!(
            res.repair_edges, 0,
            "the Y fixture is disconnected; the arms have detached from the trunk"
        );

        assert!(
            !res.terminal_states.is_empty(),
            "no terminal states detected"
        );
        assert!(
            res.terminal_states.iter().all(|&c| c >= trunk),
            "a trunk cell was called terminal: {:?}",
            res.terminal_states
        );
        assert!(
            res.terminal_states.windows(2).all(|w| w[0] < w[1]),
            "terminal states are not ascending: {:?}",
            res.terminal_states
        );
    }

    #[test]
    fn test_y_branch_fate_probabilities_split_at_the_root() {
        // Terminal states supplied, so this exercises the Markov chain, the
        // absorbing solve, the projection and the entropy without depending on
        // the detection heuristic. Both arm tips are given, so the root has two
        // reachable fates and its entropy should approach ln 2.
        let (trunk, arm) = (200usize, 200usize);
        let coords = y_manifold(trunk, arm);
        let (indices, distances) = knn_of(&coords, 15);
        let tips = [trunk + arm - 1, trunk + 2 * arm - 1];

        let res = run_palantir(
            &indices,
            &distances,
            true,
            0,
            Some(&tips),
            Some(test_params()),
            42,
            0,
        )
        .unwrap();

        assert_eq!(res.terminal_states, tips.to_vec());

        // Measured on this fixture: tips sit at 0.952 and 0.886 for their own
        // fate, the root at 0.593/0.407, mid-trunk at 0.518/0.482. Bounds are
        // set from those with headroom rather than at the idealised values. A
        // per-cell probability is a convex combination over nearby waypoints,
        // so even a tip picks up some of its neighbours' fates; only the
        // waypoint row itself is one-hot.
        for (col, &tip) in res.terminal_states.iter().enumerate() {
            assert!(
                res.branch_probs[(tip, col)] > 0.8,
                "tip {tip} gives only {} to its own fate",
                res.branch_probs[(tip, col)]
            );
            assert!(
                res.entropy[tip] < 0.45,
                "terminal state {tip} has entropy {}",
                res.entropy[tip]
            );
        }

        // The shared trunk is genuinely undecided: both fates are close to even
        // and the entropy sits near ln 2 = 0.693. This is the assertion that
        // actually says the branch probabilities are splitting rather than
        // collapsing onto one arm.
        let mid_trunk = trunk / 2;
        let split = (res.branch_probs[(mid_trunk, 0)] - res.branch_probs[(mid_trunk, 1)]).abs();
        assert!(
            split < 0.2,
            "mid-trunk fates are not close to even, difference {split}"
        );
        assert!(
            res.entropy[mid_trunk] > 0.6,
            "mid-trunk entropy is only {}, expected near ln 2",
            res.entropy[mid_trunk]
        );
        assert!(
            res.entropy[0] > 0.6,
            "root entropy is only {}, expected near ln 2",
            res.entropy[0]
        );

        // Nothing stranded and nothing repaired on a well-formed fixture.
        assert_eq!(res.stranded_waypoints, 0);
        assert_eq!(res.repair_edges, 0);
    }

    #[test]
    fn test_branch_probs_are_bounded_and_split_across_two_absorbers() {
        // Two absorbers, so the row sums say something about the solve rather
        // than restating that a convex combination of ones is one. Terminal
        // states are supplied so the column count is pinned at two.
        //
        // The bound is one-sided on purpose. Waypoints the pseudotime pruning
        // strands absorb nowhere, so a cell whose weight sits on them sums
        // below one, and a cell whose weight sits entirely on them sums to
        // zero. Both are the correct answer; summing *above* one never is, and
        // that is exactly what rescaling the surviving edges used to produce.
        let (trunk, arm) = (200usize, 200usize);
        let coords = y_manifold(trunk, arm);
        let (indices, distances) = knn_of(&coords, 15);
        let supplied = [trunk + arm - 1, trunk + 2 * arm - 1];

        let mut params = test_params();
        params.branch_prob_threshold = 0.0;

        let res = run_palantir(
            &indices,
            &distances,
            true,
            0,
            Some(&supplied),
            Some(params),
            42,
            0,
        )
        .unwrap();

        assert_eq!(res.branch_probs.ncols(), 2);
        for i in 0..res.branch_probs.nrows() {
            let sum: f32 = (0..2).map(|j| res.branch_probs[(i, j)]).sum();
            assert!(
                (0.0..=1.0 + 1e-4).contains(&sum),
                "cell {i} has fate probabilities summing to {sum}"
            );
        }

        // Neither column may be dead, or the row sums are trivial again for
        // want of anything to split between.
        for j in 0..2 {
            let column_mass: f32 = (0..res.branch_probs.nrows())
                .map(|i| res.branch_probs[(i, j)])
                .sum();
            assert!(column_mass > 0.0, "fate {j} carries no mass at all");
        }

        // The exact numbers for a multi-absorber chain, including the one where
        // a strand makes the ratio 1:10, are pinned in `markov.rs` against a
        // hand-solved example. Anything embedding-dependent belongs there rather
        // than here, where the diffusion map decides the answer.
    }

    #[test]
    fn test_user_supplied_terminal_states_are_honoured() {
        let coords = y_manifold(200, 200);
        let (indices, distances) = knn_of(&coords, 15);
        let supplied = [399usize, 599usize];

        let res = run_palantir(
            &indices,
            &distances,
            true,
            0,
            Some(&supplied),
            Some(test_params()),
            42,
            0,
        )
        .unwrap();

        assert_eq!(res.terminal_states, vec![399, 599]);
    }

    // `test_run_is_reproducible` used to live here. It pinned the kNN backend to
    // `"exhaustive"`, seeded the eigensolver and the sampler, and every rayon
    // site in the pipeline is an order-preserving `map().collect()`, so
    // determinism held by construction and the test could not fail. The
    // nondeterminism that does exist lives in the approximate ANN backends,
    // which this fixture never touched.

    #[test]
    fn test_rejects_out_of_range_early_cell() {
        let coords = linear_manifold(60);
        let (indices, distances) = knn_of(&coords, 10);

        assert!(matches!(
            run_palantir(
                &indices,
                &distances,
                true,
                999,
                None,
                Some(test_params()),
                42,
                0
            ),
            Err(BixverseErrors::PalantirEarlyCellOutOfRange { .. })
        ));
    }

    #[test]
    fn test_rejects_out_of_range_terminal_state() {
        let coords = linear_manifold(60);
        let (indices, distances) = knn_of(&coords, 10);

        assert!(matches!(
            run_palantir(
                &indices,
                &distances,
                true,
                0,
                Some(&[999]),
                Some(test_params()),
                42,
                0
            ),
            Err(BixverseErrors::PalantirTerminalStateOutOfRange { .. })
        ));
    }

    #[test]
    fn test_rejects_knn_below_the_bandwidth_floor() {
        let coords = linear_manifold(60);
        let (indices, distances) = knn_of(&coords, 10);

        let mut params = test_params();
        params.knn = 4;

        assert!(matches!(
            run_palantir(&indices, &distances, true, 0, None, Some(params), 42, 0),
            Err(BixverseErrors::PalantirKnnTooSmall { .. })
        ));
    }
}
