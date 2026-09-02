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
//! 2. Terminal states are likewise ordered by index, which sets the column
//!    order of `branch_probs`.
//! 3. The stationary distribution comes from power iteration on the lazy chain
//!    `(I + T) / 2` rather than ARPACK with an unseeded start vector. Same
//!    eigenvector, deterministic, and immune to periodicity.
//! 4. Diffusion components come from a symmetric Lanczos solve plus a
//!    back-transform rather than a non-symmetric solve. Mathematically
//!    identical and far better conditioned.
//! 5. Eigenvalues are clamped away from one before the multiscale scaling.
//! 6. A zero adaptive bandwidth is replaced by the smallest positive one
//!    instead of silently killing the row. This applies to both the waypoint
//!    chain and the diffusion kernel; a duplicate cell in a narrow kNN graph
//!    otherwise makes the kernel `NaN`.
//! 7. A waypoint whose neighbours are all pruned keeps the *latest* of them at
//!    full weight, and falls back to a self loop when even that one sits
//!    earlier in pseudotime. The reference leaves the row empty; this keeps the
//!    chain row-stochastic while still leaving a branch tip a sink, which is
//!    what terminal-state detection needs from it.
//! 8. `(I - Q)` is densified and solved with a partial-pivot LU rather than a
//!    sparse SuperLU factorisation, with a documented waypoint ceiling.
//!    Waypoints that cannot reach any terminal state are dropped from the
//!    system and reported through `stranded_waypoints`; the reference lets the
//!    factorisation fail and falls back to a pseudo-inverse.
//! 9. Graph repair is capped at `n` passes and then raises
//!    `PalantirDisconnectedGraph`, where the reference loops until the graph is
//!    connected with no bound at all. On a duplicate `(lo, hi)` pair the repair
//!    keeps the smaller weight; the reference overwrites with the later one.
//!    The per-pass structure itself is the same on both sides: one edge added,
//!    Dijkstra recomputed.
//! 10. Degenerate cases the reference turns into `NaN` or empty frames return
//!     typed errors here.
//! 11. `alpha`, the anisotropic kernel exponent, is not exposed. Its reference
//!     default of zero makes the step a no-op.
//! 12. `use_early_cell_as_start` defaults to `true`. The reference defaults to
//!     `false`, meaning it silently relocates `early_cell` to whichever
//!     diffusion-map boundary cell is nearest in the multiscale space. The
//!     boundary set is only the per-column argmin and argmax, so it holds at
//!     most `2 * n_components` cells, and "nearest" among so few extremes is
//!     close to arbitrary once the components themselves are not pinned down.
//!     Measured over 40 waypoint seeds on the Y fixture: with the eigensolve
//!     converged the snap never moved a trunk-root `early_cell` at all, but at
//!     a restart budget leaving the residual above the leading eigengap it
//!     moved it into an arm 5 times at the heuristic component count and 16
//!     times with `n_eigs` pinned to three, taking the trunk pseudotime
//!     backwards on 2 of those. So the failure is a symptom of an
//!     under-resolved multiscale space rather than of the snap on its own, and
//!     the reference has no `eigen_converged` equivalent to tell a caller they
//!     are in that regime. Naming an early cell is an explicit statement about
//!     the data, so it is honoured by default here. Set the flag to `false` for
//!     the reference's behaviour.
//! 13. `n_dcs` counts *usable* components, so the eigensolver is asked for
//!     `n_dcs + 1` pairs and the eigengap heuristic runs over all of them. The
//!     reference's `n_components` counts eigenvectors, runs the heuristic over
//!     those, and ends up with one fewer usable component at the same setting.
//!     On identical data the two can therefore retain different component
//!     counts, which shifts every downstream distance.
//! 14. An explicit `n_eigs` is honoured exactly, where the reference's floor of
//!     three is heuristic-only. Below two eigenvectors nothing survives dropping
//!     the trivial one, so that is refused rather than silently raised.
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
    /// Usable diffusion components to extract before the multiscale scaling.
    /// The eigensolver is asked for `n_dcs + 1` pairs because the trivial
    /// leading eigenvector is always dropped. Reference default: 10, which the
    /// reference reads as an eigenvector count rather than a component count;
    /// see divergence 13.
    pub n_dcs: usize,
    /// **Eigenvectors** to retain, the trivial leading one included. It is then
    /// dropped, so `Some(3)` leaves two multiscale components. `None` selects
    /// the count from the largest eigengap, as the reference does; an explicit
    /// count bypasses the heuristic's floor of three and anything below two is
    /// rejected.
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
    /// diffusion-map boundary cell.
    ///
    /// Defaults to `true`, against the reference's `false`. The snap picks from
    /// at most `2 * n_components` extremes, which is reliable only while the
    /// components are resolved; on an under-converged embedding it relocates
    /// the root often enough to invert the trajectory. See divergence 12 in the
    /// module docs for the measurements. Set to `false` to reproduce the
    /// reference. Reference default: false.
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
    /// [LanczosParams] for the diffusion-map eigendecomposition.
    pub lanczos_params: LanczosParams,
}

impl PalantirParams {
    /// Parameters matching the reference's `run_palantir` defaults, except for
    /// `use_early_cell_as_start`.
    ///
    /// That one flag is flipped on purpose; divergence 12 in the module docs has
    /// the reasoning and the numbers behind it.
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
            use_early_cell_as_start: true,
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
    /// Pseudotime per cell, min-max scaled to `[0, 1]`.
    ///
    /// The start cell is not pinned to zero. `refine_once` gives it
    /// `sum_w w (t_w - d(w, start))`, which nothing bounds below every other
    /// cell, so on a noisy manifold the minimum can land on a neighbour
    /// instead. A start cell far from zero means the refinement disagreed with
    /// the anchor, which is worth looking at.
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
    /// Whether the *pseudotime refinement* met the correlation criterion before
    /// the cap.
    pub converged: bool,
    /// Whether the diffusion eigensolve met its tolerance rather than running
    /// out of restarts. `false` means the embedding is under-resolved and every
    /// distance taken on it is suspect; raise `lanczos_params.max_restarts`.
    pub eigen_converged: bool,
    /// Largest achieved `||A x - lambda x||` from the diffusion eigensolve.
    pub eigen_residual: f64,
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

    build_symmetric_knn_graph(&indices, &distances)
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
    early_cell: usize,
    terminal_states: Option<&[usize]>,
    params: PalantirParams,
    seed: u64,
    verbose: usize,
) -> Result<PalantirResult, BixverseErrors> {
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
    let space = multiscale_components(
        knn_indices,
        knn_distances,
        params.n_dcs,
        params.n_eigs,
        seed,
        Some(params.lanczos_params),
    )?;
    let (eigen_converged, eigen_residual) = (space.converged, space.residual);
    if !eigen_converged && verbosity.normal_verbosity() {
        println!(
            "[WARNING] The diffusion eigensolve did not converge (residual {eigen_residual:e}); the embedding is under-resolved. Raise lanczos_max_restarts."
        );
    }

    let mut multiscale = space.components;
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
        eigen_converged,
        eigen_residual,
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

    /// Deterministic jitter in `[-0.5, 0.5)`, so fixtures need no RNG. `salt`
    /// distinguishes independent noise streams.
    fn jitter(i: usize, salt: usize) -> f32 {
        let h = (i.wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(40_503)) % 10_007;
        h as f32 / 10_007.0 - 0.5
    }

    /// kNN graph over a synthetic embedding, as `run_palantir` expects it:
    /// `(indices, distances)` from an exhaustive Euclidean search.
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

    /// A one-dimensional trajectory, coordinates ordered along it. Genuinely
    /// noisy on purpose: a clean lattice makes every neighbourhood identical,
    /// which is not what the adaptive bandwidths are built for.
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
    /// The second arm is the exact reflection of the first, noise included.
    /// That is what gives "the trunk is undecided between the two fates" a
    /// ground truth to be asserted against: with an independent noise stream
    /// per arm the two fates are not equivalent, an uneven trunk split is then
    /// a correct answer, and the assertion has nothing to hold on to. Measured
    /// over 40 waypoint seeds, independent noise put the mid-trunk split
    /// anywhere up to 0.54 and dropped the root into the detected terminal
    /// states on 4 of them; the reflected arms hold at 0.31 and 0 of 40. The
    /// trunk carries its own noise stream and does not sit on `y = 0`, so the
    /// manifold as a whole is still asymmetric and the diffusion operator has
    /// no exact reflection symmetry to make its eigenvalues degenerate.
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
            coords.push(vec![5.0 + t, -(t * ARM_DIVERGENCE + jitter(i, 4) * 0.3)]);
        }

        coords
    }

    /// Trunk and arm sizes for the Y fixture.
    ///
    /// A full Palantir run over the 600-cell version costs around five seconds
    /// in a debug build, and four of them dominated the whole crate's test time.
    /// Everything that supplies its terminal states explicitly holds at 300, and
    /// measurably better: over 40 waypoint seeds the mid-trunk split worsens
    /// from 0.31 to only 0.12. Automatic detection is the one thing that does
    /// not survive the shrink, so it keeps the large fixture and goes behind
    /// `large-test`.
    const Y_TRUNK: usize = 100;
    const Y_ARM: usize = 100;

    /// Trunk and arm sizes automatic terminal-state detection needs.
    ///
    /// At `Y_TRUNK` the detection misses one of the two arms on 2 of 40 waypoint
    /// seeds, and below 240 cells the waypoint chain stops resolving the branch
    /// at all. The detected states stay inside the arms throughout, which is
    /// what the cheap test asserts.
    #[cfg(feature = "large-test")]
    const Y_TRUNK_LARGE: usize = 200;
    #[cfg(feature = "large-test")]
    const Y_ARM_LARGE: usize = 200;

    /// [PalantirParams] for the small synthetic fixtures, with an exact kNN
    /// backend and a pinned spectrum budget. `num_waypoints` has to track the
    /// fixture size.
    fn test_params(num_waypoints: usize) -> PalantirParams {
        let mut params = PalantirParams::new();
        params.knn = 24;
        params.num_waypoints = num_waypoints;
        params.knn_params.knn_method = "exhaustive".to_string();
        // Pin the component count. The eigengap heuristic keeps six or more on
        // these fixtures, and the higher harmonics of a simple manifold
        // oscillate across the full scaled range, which inflates the arc length
        // until the spatial bandwidth swamps every pseudotime gap and the
        // directed pruning stops cutting anything. Unrelated to the eigensolver.
        params.n_eigs = Some(3);
        // These fixtures' spectra sit within 1e-5 of one across the whole
        // retained range, so the leading eigenvectors do not separate until the
        // residual drops below that gap. The crate default stops an order of
        // magnitude short, leaving the embedding a function of the restart count
        // rather than of the data. Real data needs nothing like this budget.
        params.lanczos_params.max_restarts = 256;
        params
    }

    /// Pseudotime tracks the manifold in the right direction on a single-fate trajectory.
    #[test]
    fn test_linear_trajectory_has_monotone_pseudotime() {
        let coords = linear_manifold(120);
        let (indices, distances) = knn_of(&coords, 15);

        // 250 rather than the cell count: `max_min_sampling` runs
        // `num_waypoints / n_dims` iterations per component, so a target equal
        // to `n` samples barely half the cells per axis and detection loses the
        // single terminal state on this fixture.
        let res = run_palantir(&indices, &distances, 0, None, test_params(250), 42, 0).unwrap();

        let truth: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let corr = crate::core::math::vector_helpers::pearson_correlation(&res.pseudotime, &truth)
            .unwrap();

        // No `.abs()`. A fully inverted trajectory gives r = -1 and would sail
        // through the absolute version, which is precisely the failure
        // divergence 12 exists to prevent, so the one test that could catch a
        // regression in `use_early_cell_as_start` has to be signed.
        assert!(
            corr > 0.95,
            "pseudotime does not track the manifold: r = {corr}"
        );
        // Cell 0 is both the start cell and the earliest cell on the manifold,
        // so it has to land at the bottom of the scaled range.
        assert!(
            res.pseudotime[0] < 0.05,
            "the start cell sits at pseudotime {}, so the trajectory runs backwards",
            res.pseudotime[0]
        );
        assert_eq!(res.terminal_states.len(), 1);
    }

    /// The cheap half of detection coverage: whatever is detected sits in a branch, never the trunk.
    #[test]
    fn test_y_branch_terminal_states_lie_in_the_arms() {
        // The cheap half of detection coverage: whatever comes back is in a
        // branch and not in the trunk, and the export ordering holds. True on
        // all 40 waypoint seeds swept at this size, where pinning the exact
        // answer is not: two of them find only one of the two arms.
        // `test_y_branch_detection_finds_one_state_per_arm` pins the full answer
        // on the fixture that supports it.
        let (trunk, arm) = (Y_TRUNK, Y_ARM);
        let coords = y_manifold(trunk, arm);
        let (indices, distances) = knn_of(&coords, 15);

        let res = run_palantir(&indices, &distances, 0, None, test_params(120), 42, 0).unwrap();

        // The arms must stay attached to the trunk, for the reason spelled out
        // at [ARM_DIVERGENCE].
        assert_eq!(
            res.repair_edges, 0,
            "the Y fixture is disconnected; the arms have detached from the trunk"
        );

        // Terminal states are read off the extremes of the multiscale space, so
        // an under-resolved embedding decides this test. Two components leave
        // only four per-axis extremes, one of which is the root, and an
        // unconverged solve reliably hands that root back as a terminal state.
        assert!(
            res.eigen_converged,
            "the eigensolve stopped at residual {:e}; the embedding is not determined",
            res.eigen_residual
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

    /// Detection held to the full answer: exactly one terminal state per arm.
    #[test]
    // 800 cells, a full Palantir run, around five seconds in a debug build.
    #[cfg(feature = "large-test")]
    fn test_y_branch_detection_finds_one_state_per_arm() {
        // Detection held to the full answer: exactly one terminal state per arm.
        // It needs the large fixture; the cheap sibling above covers the same
        // machinery at a size where only the weaker property is determined.
        let (trunk, arm) = (Y_TRUNK_LARGE, Y_ARM_LARGE);
        let coords = y_manifold(trunk, arm);
        let (indices, distances) = knn_of(&coords, 15);

        let res = run_palantir(&indices, &distances, 0, None, test_params(250), 42, 0).unwrap();

        assert_eq!(res.repair_edges, 0);
        assert!(
            res.eigen_converged,
            "the eigensolve stopped at residual {:e}",
            res.eigen_residual
        );
        assert_eq!(
            res.terminal_states.len(),
            2,
            "expected one terminal state per arm, got {:?}",
            res.terminal_states
        );
        // Checked by arm rather than only by `>= trunk`, or both states landing
        // in the same arm would pass.
        assert!(
            (trunk..trunk + arm).contains(&res.terminal_states[0])
                && (trunk + arm..trunk + 2 * arm).contains(&res.terminal_states[1]),
            "terminal states do not sit one per arm: {:?}",
            res.terminal_states
        );
    }

    /// The Markov chain, absorbing solve, projection and entropy, with detection taken out of the loop.
    #[test]
    fn test_y_branch_fate_probabilities_split_at_the_root() {
        // Terminal states supplied, so this exercises the Markov chain, the
        // absorbing solve, the projection and the entropy without depending on
        // the detection heuristic. Both arm tips are given, so the root has two
        // reachable fates and its entropy should approach ln 2.
        let (trunk, arm) = (Y_TRUNK, Y_ARM);
        let coords = y_manifold(trunk, arm);
        let (indices, distances) = knn_of(&coords, 15);
        let tips = [trunk + arm - 1, trunk + 2 * arm - 1];

        let res = run_palantir(
            &indices,
            &distances,
            0,
            Some(&tips),
            test_params(120),
            42,
            0,
        )
        .unwrap();

        assert_eq!(res.terminal_states, tips.to_vec());

        // Bounds come from a 40-seed sweep of the waypoint sampler rather than
        // from the single seed run here, so a sampler change shows up as a
        // failure instead of drifting quietly. Worst case over that sweep: tips
        // hold 0.979 of their own fate at entropy 0.103, the mid-trunk split
        // reaches 0.123 at entropy 0.686. A per-cell probability is a convex
        // combination over nearby waypoints, so even a tip picks up some of its
        // neighbours' fates; only the waypoint row itself is one-hot.
        for (col, &tip) in res.terminal_states.iter().enumerate() {
            assert!(
                res.branch_probs[(tip, col)] > 0.95,
                "tip {tip} gives only {} to its own fate",
                res.branch_probs[(tip, col)]
            );
            assert!(
                res.entropy[tip] < 0.2,
                "terminal state {tip} has entropy {}",
                res.entropy[tip]
            );
        }

        // The shared trunk is genuinely undecided: the arms are exact
        // reflections of each other, so an even split is the right answer and
        // the entropy belongs near ln 2 = 0.693. This is the assertion that
        // actually says the branch probabilities are splitting rather than
        // collapsing onto one arm.
        let mid_trunk = trunk / 2;
        let split = (res.branch_probs[(mid_trunk, 0)] - res.branch_probs[(mid_trunk, 1)]).abs();
        assert!(
            split < 0.2,
            "mid-trunk fates are not close to even, difference {split}"
        );
        assert!(
            res.entropy[mid_trunk] > 0.65,
            "mid-trunk entropy is only {}, expected near ln 2",
            res.entropy[mid_trunk]
        );
        assert!(
            res.entropy[0] > 0.65,
            "root entropy is only {}, expected near ln 2",
            res.entropy[0]
        );

        // Nothing stranded and nothing repaired on a well-formed fixture.
        assert_eq!(res.stranded_waypoints, 0);
        assert_eq!(res.repair_edges, 0);
    }

    /// Regression: rescaling the surviving edges used to push fate probabilities above one.
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
        let (trunk, arm) = (Y_TRUNK, Y_ARM);
        let coords = y_manifold(trunk, arm);
        let (indices, distances) = knn_of(&coords, 15);
        let supplied = [trunk + arm - 1, trunk + 2 * arm - 1];

        let mut params = test_params(120);
        params.branch_prob_threshold = 0.0;

        let res = run_palantir(&indices, &distances, 0, Some(&supplied), params, 42, 0).unwrap();

        assert_eq!(res.branch_probs.ncols(), 2);
        for i in 0..res.branch_probs.nrows() {
            let sum: f32 = (0..2).map(|j| res.branch_probs[(i, j)]).sum();
            assert!(
                (0.0..=1.0 + 1e-4).contains(&sum),
                "cell {i} has fate probabilities summing to {sum}"
            );
        }

        // Neither column may be dead, or the row sums are trivial again for
        // want of anything to split between. The bound is a real fraction of
        // the cell count rather than `> 0.0`, which passes on 1e-38.
        let n_cells = res.branch_probs.nrows();
        for j in 0..2 {
            let column_mass: f32 = (0..n_cells).map(|i| res.branch_probs[(i, j)]).sum();
            assert!(
                column_mass > 0.1 * n_cells as f32,
                "fate {j} carries only {column_mass} across {n_cells} cells"
            );
        }

        // The exact numbers for a multi-absorber chain, including the one where
        // a strand makes the ratio 1:10, are pinned in `markov.rs` against a
        // hand-solved example. Anything embedding-dependent belongs there rather
        // than here, where the diffusion map decides the answer.
    }

    /// Supplied terminal states are used verbatim rather than re-detected.
    #[test]
    fn test_user_supplied_terminal_states_are_honoured() {
        let coords = y_manifold(Y_TRUNK, Y_ARM);
        let (indices, distances) = knn_of(&coords, 15);
        // Deliberately not the arm tips: an interior cell of each arm, so this
        // cannot pass by agreeing with what detection would have found anyway.
        let supplied = [Y_TRUNK + Y_ARM - 40, Y_TRUNK + 2 * Y_ARM - 40];

        let res = run_palantir(
            &indices,
            &distances,
            0,
            Some(&supplied),
            test_params(120),
            42,
            0,
        )
        .unwrap();

        assert_eq!(res.terminal_states, supplied.to_vec());
    }

    /// An early cell past the cell count errors rather than indexing out of bounds.
    #[test]
    fn test_rejects_out_of_range_early_cell() {
        let coords = linear_manifold(60);
        let (indices, distances) = knn_of(&coords, 10);

        assert!(matches!(
            run_palantir(&indices, &distances, 999, None, test_params(250), 42, 0),
            Err(BixverseErrors::PalantirEarlyCellOutOfRange { .. })
        ));
    }

    /// Same guard on supplied terminal states.
    #[test]
    fn test_rejects_out_of_range_terminal_state() {
        let coords = linear_manifold(60);
        let (indices, distances) = knn_of(&coords, 10);

        assert!(matches!(
            run_palantir(
                &indices,
                &distances,
                0,
                Some(&[999]),
                test_params(250),
                42,
                0
            ),
            Err(BixverseErrors::PalantirTerminalStateOutOfRange { .. })
        ));
    }

    /// A `knn` too small for the adaptive bandwidth rank errors up front.
    #[test]
    fn test_rejects_knn_below_the_bandwidth_floor() {
        let coords = linear_manifold(60);
        let (indices, distances) = knn_of(&coords, 10);

        let mut params = test_params(250);
        params.knn = 4;

        assert!(matches!(
            run_palantir(&indices, &distances, 0, None, params, 42, 0),
            Err(BixverseErrors::PalantirKnnTooSmall { .. })
        ));
    }
}
