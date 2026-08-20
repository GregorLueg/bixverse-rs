//! The directed waypoint Markov chain and everything derived from it: terminal
//! states, absorption probabilities, and the projection back onto all cells.
//!
//! Everything here is sized by the waypoint count rather than the cell count,
//! so it is small and dense where the pseudotime half is enormous and sparse.
//! That is also why the absorbing system is solved by densifying `(I - Q)` and
//! handing it to a partial-pivot LU rather than reaching for a sparse
//! factorisation.
//!
//! ### References
//!
//! Setty, et al., Nat. Biotechnol., 2019.

use ann_search_rs::{build_exhaustive_index, query_exhaustive_self};
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::Solve;
use faer::{Accum, Mat, MatRef};
use rayon::prelude::*;

use crate::core::math::vector_helpers::{mad, median};
use crate::graph::graph_components::induced_components;
use crate::prelude::*;
use crate::single_cell::sc_trajectory::diffusion::{boundary_cells, nearest_candidate};
use crate::utils::faer_parallelism;

////////////
// Consts //
////////////

/// Cap on the neighbour rank used for the back-edge bandwidth.
///
/// The reference takes `min(floor(knn / 3) - 1, 30)` into a self-inclusive
/// neighbour list. Note this is one rank tighter than the diffusion kernel's
/// `floor(knn / 3)`; the discrepancy is the reference's, not a transcription
/// slip, and it makes the chain marginally less willing to keep back edges.
const ADAPTIVE_K_CAP: usize = 30;

/// Smallest neighbour count that yields a usable back-edge bandwidth.
///
/// Below six, `floor(knn / 3) - 1` collapses to zero and there is no non-self
/// neighbour left to take a bandwidth from.
const MIN_KNN: usize = 6;

/// Outlier threshold on the stationary ranks, `scipy.stats.norm.ppf(0.9999)`.
///
/// Applied against the *raw* median absolute deviation, without the normal
/// consistency factor, matching the reference. Do not substitute
/// [crate::prelude::MAD_SCALE] here.
const TERMINAL_CUTOFF_Z: f64 = 3.719_016_485_455_680_4;

/// Iteration cap for the stationary-distribution power iteration.
///
/// Reaching this is expected rather than exceptional; see
/// [stationary_distribution]. It is not a convergence budget: the lazy chain's
/// second eigenvalue is `(1 + lambda_2) / 2`, so a near-reducible chain with
/// `lambda_2 = 1 - 1e-4` only decays its error to `exp(-0.25)` of the starting
/// value over the full 5000 passes. What settles far sooner is the *ranking*
/// [detect_terminal_states] reads, which is all the cap has to buy.
const POWER_ITER_MAX: usize = 5_000;

/// L1 change per pass below which the iteration stops early.
///
/// The vector is L1-normalised each pass, so this is an absolute tolerance on a
/// probability distribution and needs no scaling with the state count.
const POWER_ITER_TOL: f64 = 1e-12;

/// Largest transient count accepted for the densified `(I - Q)` solve.
///
/// The matrix is `n_transient^2` in `f64`, so 20,000 waypoints already needs
/// 3.2 GB. `partial_piv_lu` takes `&self` and hands back an owned factorisation,
/// so both live at once and the real peak is 6.4 GB, plus a cubic factorisation
/// to match. The reference uses a sparse SuperLU factorisation instead; if this
/// ceiling ever binds in practice that is the upgrade path, not a larger
/// allocation.
const MAX_DENSE_TRANSIENT: usize = 20_000;

/// Slack allowed above one on an absorption row sum.
///
/// Rows may legitimately sum to *less* than one, because mass that leaks into a
/// stranded state never absorbs anywhere. Summing to more than one is not
/// legitimate under any configuration, so only the upper side is bounded.
const ABSORPTION_ROW_SUM_TOL: f64 = 1e-3;

/// Tolerance on `‖(I - Q) B - R‖_inf`, the achieved solve residual.
///
/// This catches a factorisation that failed outright. It is **not** a check on
/// whether the system was the right one: `(I - Q)` and `R` are `f64` matrices by
/// the time they get here, so the attainable residual is `f64` rounding at
/// around `1e-16` no matter how inaccurate the `f32` chain behind them was. A
/// consistent solve of a subtly wrong system passes this comfortably, which is
/// exactly why the row-sum check exists alongside it.
const ABSORPTION_RESIDUAL_TOL: f64 = 1e-5;

///////////////////////
// Transition matrix //
///////////////////////

/// Build the pseudotime-directed Markov chain over the waypoints.
///
/// A kNN graph on the waypoints is pruned by pseudotime: the edge `i -> j`
/// survives when `t_j >= t_i - sigma_i`, so forward edges always live while
/// backward ones only survive within one local bandwidth. Affinities then come
/// from an adaptive Gaussian using both endpoints' bandwidths, and rows are
/// normalised to give a row-stochastic matrix.
///
/// Unlike the diffusion kernel this is deliberately **not** symmetrised: the
/// asymmetry is the directionality.
///
/// ### Params
///
/// * `wp_data` - Multiscale components restricted to the waypoints, waypoints
///   by components.
/// * `wp_pseudotime` - Pseudotime per waypoint, aligned with `wp_data`.
/// * `knn` - Neighbour count in the reference's self-inclusive convention.
///   Clamped to the waypoint count. Too small a `knn` and too few waypoints are
///   rejected separately, so the error names whichever the caller can actually
///   act on.
/// * `verbose` - Passed through to the exhaustive search.
///
/// ### Returns
///
/// A row-stochastic CSR transition matrix, square in the waypoint count.
pub fn build_waypoint_transitions(
    wp_data: &[Vec<f32>],
    wp_pseudotime: &[f32],
    knn: usize,
    verbose: bool,
) -> Result<CompressedSparseData2<f32>, BixverseErrors> {
    let n_wp = wp_data.len();
    if n_wp == 0 {
        return Err(BixverseErrors::InvalidArgument(
            "Palantir: the waypoint set is empty".to_string(),
        ));
    }
    if wp_pseudotime.len() != n_wp {
        return Err(BixverseErrors::PalantirWaypointDataMismatch {
            n_waypoints: n_wp,
            n_pseudotime: wp_pseudotime.len(),
        });
    }

    if knn < MIN_KNN {
        return Err(BixverseErrors::PalantirKnnTooSmall {
            knn,
            minimum: MIN_KNN,
        });
    }
    if n_wp < MIN_KNN {
        return Err(BixverseErrors::PalantirTooFewWaypoints {
            n_waypoints: n_wp,
            minimum: MIN_KNN,
        });
    }
    let knn = knn.min(n_wp);

    let (indices, distances) = waypoint_knn(wp_data, knn, verbose)?;
    let adaptive_std = adaptive_bandwidths(&distances, knn);

    let mut indptr: Vec<u32> = vec![0];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();

    for i in 0..n_wp {
        let sigma_i = adaptive_std[i];
        let cut = wp_pseudotime[i] - sigma_i;

        let start = data.len();
        for (slot, &j) in indices[i].iter().enumerate() {
            let d = distances[i][slot];
            if d <= 0.0 {
                continue;
            }
            if wp_pseudotime[j] < cut {
                continue;
            }
            let sigma_j = adaptive_std[j];
            let exponent = -0.5 * d * d * (1.0 / (sigma_i * sigma_i) + 1.0 / (sigma_j * sigma_j));
            col_idx.push(j as u32);
            data.push(exponent.exp());
        }

        if data.len() == start {
            let mut best: Option<usize> = None;
            let mut best_pt = f32::NEG_INFINITY;
            for (slot, &j) in indices[i].iter().enumerate() {
                // Strict `>` keeps the first index on ties.
                if j != i && distances[i][slot] > 0.0 && wp_pseudotime[j] > best_pt {
                    best_pt = wp_pseudotime[j];
                    best = Some(j);
                }
            }
            let target = match best {
                Some(j) if wp_pseudotime[j] >= wp_pseudotime[i] => j,
                _ => i,
            };
            col_idx.push(target as u32);
            data.push(1.0);
        }

        let row_sum: f32 = data[start..].iter().sum();
        if row_sum > 0.0 {
            for v in data[start..].iter_mut() {
                *v /= row_sum;
            }
        }
        indptr.push(col_idx.len() as u32);
    }

    Ok(CompressedSparseData2::new_csr(
        &data,
        &col_idx,
        &indptr,
        None,
        (n_wp, n_wp),
    ))
}

/// Exhaustive self-kNN over the waypoints.
///
/// Uses the SIMD flat-layout exhaustive index from `ann-search-rs` rather than
/// the crate's [crate::prelude::generate_knn_with_dist]: at a few thousand
/// waypoints an exact scan beats any index build, and this entry point keeps
/// the self hit that the reference's convention indexes against.
///
/// ### Params
///
/// * `wp_data` - Waypoint rows, waypoints by components.
/// * `k` - Neighbours to retrieve, self inclusive.
/// * `verbose` - Progress reporting for the search.
///
/// ### Returns
///
/// `(indices, distances)` with un-squared Euclidean distances, ascending.
fn waypoint_knn(wp_data: &[Vec<f32>], k: usize, verbose: bool) -> ScKnnResults {
    let n_wp = wp_data.len();
    let n_dims = wp_data[0].len();

    let mat = Mat::<f32>::from_fn(n_wp, n_dims, |i, j| wp_data[i][j]);
    let index = build_exhaustive_index(mat.as_ref(), "euclidean");
    let (indices, distances) = query_exhaustive_self(&index, k, true, verbose)?;

    // The backend reports squared Euclidean distances.
    let distances = distances
        .map(|d| {
            d.into_iter()
                .map(|row| row.into_iter().map(|v| v.sqrt()).collect())
                .collect::<Vec<Vec<f32>>>()
        })
        .ok_or(BixverseErrors::InvalidArgument(
            "Palantir: the exhaustive waypoint search returned no distances".to_string(),
        ))?;

    Ok((indices, distances))
}

/// Per-waypoint bandwidth for the back-edge tolerance and the affinity kernel.
///
/// Matches the reference's min(floor(knn / 3) - 1, 30) rank into a
/// self-inclusive neighbour list. Since the self hit is retrieved too, no
/// adjustment is needed to land on the same non-self neighbour.
///
/// Duplicate waypoints produce zero bandwidths; these are replaced with the
/// smallest positive bandwidth in the set. The reference instead divides by
/// zero and silently kills the row.
///
/// ### Params
///
/// * `distances` - Sorted neighbour distances per waypoint, self first.
/// * `knn` - Neighbour count in the self-inclusive convention, already clamped
///   to the retrieved neighbour count by the caller.
///
/// ### Returns
///
/// One strictly positive bandwidth per waypoint.
fn adaptive_bandwidths(distances: &[Vec<f32>], knn: usize) -> Vec<f32> {
    let adaptive_k = (knn / 3).saturating_sub(1).min(ADAPTIVE_K_CAP);

    let mut sigma: Vec<f32> = distances
        .iter()
        .map(|row| {
            let idx = adaptive_k.min(row.len().saturating_sub(1));
            row.get(idx).copied().unwrap_or(0.0)
        })
        .collect();

    let smallest_positive = sigma
        .iter()
        .copied()
        .filter(|s| *s > 0.0)
        .fold(f32::INFINITY, f32::min);
    let fallback = if smallest_positive.is_finite() {
        smallest_positive
    } else {
        1.0
    };

    for s in sigma.iter_mut() {
        if !s.is_finite() || *s <= 0.0 {
            *s = fallback;
        }
    }

    sigma
}

/////////////////////
// Terminal states //
/////////////////////

/// Per-row rescaling factors that make an `f32` chain exactly row-stochastic.
///
/// [build_waypoint_transitions] normalises in `f32`, so its rows sum to one only
/// to about 3e-7 over a 24-term sum. That error is per *step* and per *row*, so
/// no downstream global renormalisation can undo it: it perturbs the operator
/// itself rather than the vector it acts on. Reading the row mass back in `f64`
/// and dividing by it costs one extra pass over the data and makes every
/// consumer see the same exactly sub-stochastic matrix.
///
/// Both consumers need this and for different reasons.
/// [absorption_probabilities] multiplies the per-step error by the expected
/// absorption time, which a pruned corridor pushes to ~3e5.
/// [stationary_distribution] is sensitive to the perturbation as
/// `1 / (1 - |lambda_2|)`, and a trajectory with several fates is near-reducible
/// by construction, so `lambda_2` sits within rounding of one.
///
/// ### Params
///
/// * `transitions` - Row-stochastic CSR transition matrix.
///
/// ### Returns
///
/// One factor per row: `1 / row_mass`, or `1` for a row with no mass to scale.
fn row_rescale_factors(transitions: &CompressedSparseData2<f32>) -> Vec<f64> {
    (0..transitions.shape.0)
        .map(|i| {
            let mass: f64 = (transitions.indptr[i] as usize..transitions.indptr[i + 1] as usize)
                .map(|idx| transitions.data[idx] as f64)
                .sum();
            if mass > 0.0 { 1.0 / mass } else { 1.0 }
        })
        .collect()
}

/// Stationary distribution of a row-stochastic chain.
///
/// Power iteration on the lazy chain (I + T) / 2. Same stationary vector as
/// T, but strictly positive eigenvalues, so periodicity from the pseudotime
/// pruning can't stall the iteration. The reference uses ARPACK with an
/// unseeded start vector, so its terminal-state detection isn't reproducible
/// run to run; this is deterministic.
///
/// The iteration is capped rather than converged, on purpose. A trajectory
/// with several fates gives a chain that's nearly reducible: the eigenvalue at
/// one is nearly repeated, and the stationary vector is genuinely
/// ill-determined. No method recovers a unique answer here, ARPACK just hides
/// it behind its random start. What survives the degeneracy is exactly what
/// [detect_terminal_states] relies on: mass piles up on the near-absorbing
/// ends, and their ranking against the rest settles long before the vector
/// itself does. Starting from uniform makes the choice within that degenerate
/// subspace reproducible.
///
/// Rows are rescaled in `f64` by `row_rescale_factors` before the iteration
/// starts. Without that the operator itself carries the `f32` chain's per-row
/// drift, and the stationary vector amplifies it by `1 / (1 - |lambda_2|)` on
/// exactly the near-reducible chains this is asked about.
///
/// Sequential by design: at a few thousand waypoints the sparse product takes
/// a few hundred microseconds, and rayon's overhead would dominate.
///
/// ### Params
///
/// * transitions - Row-stochastic CSR transition matrix.
///
/// ### Returns
///
/// The L1-normalised stationary distribution, or an error if the chain loses
/// all its mass, meaning the input wasn't row-stochastic. That guard can't
/// fire on a genuinely row-stochastic chain: the lazy step keeps at least
/// half of an L1-normalised non-negative vector, so the total stays in
/// [0.5, 1]. It's there for callers passing in non-finite chain data.
pub fn stationary_distribution(
    transitions: &CompressedSparseData2<f32>,
) -> Result<Vec<f64>, BixverseErrors> {
    if !transitions.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    let n = transitions.shape.0;
    let scale = row_rescale_factors(transitions);

    let mut v = vec![1.0 / n as f64; n];
    let mut next = vec![0.0f64; n];

    for iteration in 1..=POWER_ITER_MAX {
        // next = 0.5 * v + 0.5 * T^T v, the transpose being a scatter over rows.
        next.iter_mut()
            .zip(v.iter())
            .for_each(|(o, &x)| *o = 0.5 * x);
        for i in 0..n {
            let vi = v[i];
            if vi == 0.0 {
                continue;
            }
            let scaled = 0.5 * scale[i] * vi;
            for idx in transitions.indptr[i] as usize..transitions.indptr[i + 1] as usize {
                let j = transitions.indices[idx] as usize;
                next[j] += transitions.data[idx] as f64 * scaled;
            }
        }

        let total: f64 = next.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(BixverseErrors::PalantirStationaryMassLost { iteration, total });
        }
        for x in next.iter_mut() {
            *x /= total;
        }

        let residual: f64 = next
            .iter()
            .zip(v.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        std::mem::swap(&mut v, &mut next);

        if residual < POWER_ITER_TOL {
            break;
        }
    }

    Ok(v)
}

/// Detect terminal states from the waypoint chain.
///
/// Waypoints whose stationary mass is a robust outlier are grouped into weakly
/// connected components; the latest one per component becomes a candidate, and
/// each candidate is snapped to its nearest boundary waypoint. Note the
/// boundaries here are recomputed over the **waypoint rows only**, which is a
/// different set from the boundaries used to pick the start cell.
///
/// ### Params
///
/// * `transitions` - Row-stochastic CSR transition matrix over the waypoints.
/// * `wp_data` - Multiscale components restricted to the waypoints.
/// * `wp_pseudotime` - Pseudotime per waypoint.
///
/// ### Returns
///
/// Terminal state positions **within the waypoint array**, ascending and
/// unique, or an error when nothing clears the cutoff.
pub fn detect_terminal_states(
    transitions: &CompressedSparseData2<f32>,
    wp_data: &[Vec<f32>],
    wp_pseudotime: &[f32],
) -> Result<Vec<usize>, BixverseErrors> {
    let ranks: Vec<f64> = stationary_distribution(transitions)?
        .into_iter()
        .map(f64::abs)
        .collect();

    let centre = median(&ranks).ok_or(BixverseErrors::PalantirNoTerminalStates)?;
    let spread = mad(&ranks, None).ok_or(BixverseErrors::PalantirNoTerminalStates)?;

    // A zero MAD means more than half the waypoints share a stationary rank, so
    // the cutoff collapses onto the median and roughly half of everything clears
    // it. `induced_components` then merges the lot and the branching structure
    // silently becomes one terminal state. The "nothing is above" case is
    // guarded below; this is the opposite failure and needs its own.
    if spread <= 0.0 || !spread.is_finite() {
        return Err(BixverseErrors::PalantirStationaryRanksDegenerate {
            n_waypoints: ranks.len(),
        });
    }

    let cutoff = centre + TERMINAL_CUTOFF_Z * spread;

    let above: Vec<usize> = (0..ranks.len()).filter(|&i| ranks[i] > cutoff).collect();
    if above.is_empty() {
        return Err(BixverseErrors::PalantirNoTerminalStates);
    }

    let (n_comps, labels) = induced_components(transitions, &above)?;

    // Latest waypoint per component; strict `>` keeps the first index on ties,
    // matching `pandas.Series.idxmax`.
    let mut candidates: Vec<usize> = vec![usize::MAX; n_comps];
    let mut best: Vec<f32> = vec![f32::NEG_INFINITY; n_comps];
    for (slot, &wp) in above.iter().enumerate() {
        let comp = labels[slot];
        if wp_pseudotime[wp] > best[comp] {
            best[comp] = wp_pseudotime[wp];
            candidates[comp] = wp;
        }
    }

    let boundaries = boundary_cells(wp_data);
    if boundaries.is_empty() {
        return Err(BixverseErrors::PalantirNoBoundaryCells);
    }

    let mut terminal: Vec<usize> = candidates
        .into_iter()
        .filter(|&c| c != usize::MAX)
        .map(|c| nearest_candidate(wp_data, &boundaries, c))
        .collect::<Result<Vec<usize>, BixverseErrors>>()?;

    terminal.sort_unstable();
    terminal.dedup();

    if terminal.is_empty() {
        return Err(BixverseErrors::PalantirNoTerminalStates);
    }

    Ok(terminal)
}

//////////////////////
// Absorption probs //
//////////////////////

/// Absorption probabilities of the waypoint chain.
///
/// Makes the terminal states absorbing and solves (I - Q) B = R for the
/// transient rows, where Q is the transient-to-transient block and R the
/// transient-to-absorbing one. The absorbing rows are never read, so zeroing
/// them and setting their diagonal to one is skipped rather than performed.
///
/// The system is densified and factorised with a partial-pivot LU. faer's LU
/// reads the global parallelism setting, which this crate never touches, so it
/// runs sequentially and costs O(n_transient^3). That's milliseconds at the
/// reference's waypoint count, and is refused outright above
/// `MAX_DENSE_TRANSIENT`.
///
/// Transient states with no path to any absorbing state are dropped from the
/// system and left at zero rather than aborting the solve. Their absorption
/// probabilities are genuinely undefined, since the walk never absorbs from
/// there, and a handful of such waypoints is normal on a noisy manifold where
/// the pseudotime pruning strands a local maximum. Including them would make
/// (I - Q) singular; the reference lets the sparse factorisation fail and
/// falls back to a pseudo-inverse, which returns numbers with no meaning.
///
/// Edges from a solved state into a stranded one are kept at their raw
/// probability and simply lead nowhere, exactly as in the reference. That mass
/// is lost, so the affected rows sum to less than one. Rescaling the surviving
/// edges to recover a sum of one would be neither the absorption probability
/// nor the probability conditional on absorbing, which needs a Doob
/// h-transform, and it would silently rewrite the fate ratios of every state
/// upstream of a strand.
///
/// ### Params
///
/// * `transitions` - Row-stochastic CSR transition matrix over the waypoints.
/// * `absorbing` - Terminal state positions within the waypoint array. Unique;
///   the order sets the column order of the result and is otherwise free.
///
/// ### Returns
///
/// The n_waypoints x n_absorbing matrix and the number of stranded transient
/// states. Absorbing rows are one-hot; solvable transient rows hold the
/// absorption probabilities with negatives clamped to zero and no
/// renormalisation, matching the reference.
pub fn absorption_probabilities(
    transitions: &CompressedSparseData2<f32>,
    absorbing: &[usize],
) -> Result<(Mat<f32>, usize), BixverseErrors> {
    if !transitions.cs_type.is_csr() {
        return Err(BixverseErrors::SparseMatrixMustBeCsr);
    }
    let n = transitions.shape.0;
    let n_abs = absorbing.len();
    if n_abs == 0 {
        return Err(BixverseErrors::PalantirNoTerminalStates);
    }

    let mut pos_abs = vec![-1i64; n];
    for (slot, &a) in absorbing.iter().enumerate() {
        if a >= n {
            return Err(BixverseErrors::SliceIndexOutOfBounds { index: a, len: n });
        }
        pos_abs[a] = slot as i64;
    }

    let reachable = reaches_absorbing(transitions, &pos_abs);

    let transient: Vec<usize> = (0..n).filter(|&i| pos_abs[i] < 0 && reachable[i]).collect();
    let stranded = (0..n).filter(|&i| pos_abs[i] < 0 && !reachable[i]).count();
    let n_trans = transient.len();
    if n_trans > MAX_DENSE_TRANSIENT {
        return Err(BixverseErrors::PalantirTooManyWaypointsForDenseSolve {
            n_transient: n_trans,
            limit: MAX_DENSE_TRANSIENT,
        });
    }

    let mut pos_trans = vec![-1i64; n];
    for (slot, &t) in transient.iter().enumerate() {
        pos_trans[t] = slot as i64;
    }

    let mut out = Mat::<f32>::zeros(n, n_abs);
    for (slot, &a) in absorbing.iter().enumerate() {
        out[(a, slot)] = 1.0;
    }
    if n_trans == 0 {
        return Ok((out, stranded));
    }

    // Renormalise the rows in `f64` rather than trusting the `f32` chain; see
    // [row_rescale_factors]. Without it the solve returns row sums past
    // ABSORPTION_ROW_SUM_TOL while remaining a perfectly consistent answer to
    // the system it was handed.
    let scale = row_rescale_factors(transitions);

    let mut lhs = Mat::<f64>::zeros(n_trans, n_trans);
    let mut rhs = Mat::<f64>::zeros(n_trans, n_abs);
    for i in 0..n {
        let ti = pos_trans[i];
        if ti < 0 {
            continue;
        }
        let ti = ti as usize;

        for idx in transitions.indptr[i] as usize..transitions.indptr[i + 1] as usize {
            let j = transitions.indices[idx] as usize;
            let v = transitions.data[idx] as f64 * scale[i];
            if pos_trans[j] >= 0 {
                lhs[(ti, pos_trans[j] as usize)] -= v;
            } else if pos_abs[j] >= 0 {
                rhs[(ti, pos_abs[j] as usize)] += v;
            }
        }
        lhs[(ti, ti)] += 1.0;
    }

    let solved = lhs.partial_piv_lu().solve(&rhs);

    let mut residual = Mat::<f64>::zeros(n_trans, n_abs);
    matmul(
        residual.as_mut(),
        Accum::Replace,
        lhs.as_ref(),
        solved.as_ref(),
        1.0f64,
        faer_parallelism(),
    );
    let mut max_residual = 0.0f64;
    for ti in 0..n_trans {
        for k in 0..n_abs {
            let r = (residual[(ti, k)] - rhs[(ti, k)]).abs();
            // `f64::max` swallows NaN, so it is carried through explicitly.
            if r.is_nan() || r > max_residual {
                max_residual = r;
            }
        }
    }
    if !max_residual.is_finite() || max_residual > ABSORPTION_RESIDUAL_TOL {
        return Err(BixverseErrors::PalantirAbsorbingSolveFailed {
            residual: max_residual,
            n_transient: n_trans,
        });
    }

    for ti in 0..n_trans {
        let mut row_sum = 0.0f64;
        for k in 0..n_abs {
            let p = solved[(ti, k)].max(0.0);
            row_sum += p;
            out[(transient[ti], k)] = p as f32;
        }
        // Split, because the two say completely different things. A non-finite
        // sum points at a NaN in the solve; a sum above one points at a
        // normalisation problem. Reporting "NaN, which is above one" sends the
        // reader after the wrong bug.
        if !row_sum.is_finite() {
            return Err(BixverseErrors::PalantirAbsorbingRowNotFinite {
                row_sum,
                waypoint: transient[ti],
                n_transient: n_trans,
            });
        }
        if row_sum > 1.0 + ABSORPTION_ROW_SUM_TOL {
            return Err(BixverseErrors::PalantirAbsorbingRowSum {
                row_sum,
                waypoint: transient[ti],
                n_transient: n_trans,
            });
        }
    }

    Ok((out, stranded))
}

/// Mark the states that can reach an absorbing one.
///
/// Backwards breadth-first search from the absorbing set over the transposed
/// chain. A transient state with no path to an absorbent contributes a zero row
/// and a zero column to `(I - Q)`, making it singular, so these are excluded
/// from the system rather than solved for.
///
/// Edges are followed only where they carry positive probability. The affinity
/// kernel's `exponent.exp()` can store an exact `0.0f32`, and walking the
/// structural nonzeros alone would count a waypoint whose only route out is a
/// zero-weight edge as reachable. It then solves to zero and never shows up in
/// the `stranded_waypoints` diagnostic, which is the user's headline signal for
/// over-pruning.
///
/// ### Params
///
/// * `transitions` - Row-stochastic CSR transition matrix.
/// * `pos_abs` - Absorbing position per state, `-1` when transient.
///
/// ### Returns
///
/// One flag per state, true when an absorbing state is reachable from it along
/// edges of positive probability.
fn reaches_absorbing(transitions: &CompressedSparseData2<f32>, pos_abs: &[i64]) -> Vec<bool> {
    let n = transitions.shape.0;
    let reversed = transitions.transpose_and_convert();

    let mut reached = vec![false; n];
    let mut stack: Vec<usize> = (0..n).filter(|&i| pos_abs[i] >= 0).collect();
    for &s in &stack {
        reached[s] = true;
    }

    while let Some(u) = stack.pop() {
        for idx in reversed.indptr[u] as usize..reversed.indptr[u + 1] as usize {
            let v = reversed.indices[idx] as usize;
            if reversed.data[idx] > 0.0 && !reached[v] {
                reached[v] = true;
                stack.push(v);
            }
        }
    }

    reached
}

////////////////////////////
// Projection and entropy //
////////////////////////////

/// Project waypoint absorption probabilities onto every cell.
///
/// The weight columns sum to one, so `W^T B` is a convex combination of the
/// waypoint fates per cell. The weights are viewed in place as a `f32` matrix
/// and transposed for free, so nothing is copied on the way into the product.
///
/// ### Params
///
/// * `weights` - Row-major `n_waypoints x n_cells` waypoint weights.
/// * `n_waypoints` - Row count of `weights`.
/// * `n_cells` - Column count of `weights`.
/// * `branch_probs_wp` - Waypoint absorption probabilities, waypoints by
///   terminal states.
///
/// ### Returns
///
/// Per-cell fate probabilities, cells by terminal states.
pub fn project_to_cells(
    weights: &[f32],
    n_waypoints: usize,
    n_cells: usize,
    branch_probs_wp: MatRef<f32>,
) -> Result<Mat<f32>, BixverseErrors> {
    if weights.len() != n_waypoints * n_cells {
        return Err(BixverseErrors::DimensionMisMatchSparse {
            indices_len: weights.len(),
            data_len: n_waypoints * n_cells,
        });
    }
    if branch_probs_wp.nrows() != n_waypoints {
        return Err(BixverseErrors::ShapeMismatch {
            expected: (n_waypoints, branch_probs_wp.ncols()),
            got: (branch_probs_wp.nrows(), branch_probs_wp.ncols()),
        });
    }

    let w = MatRef::<f32>::from_row_major_slice(weights, n_waypoints, n_cells);
    let mut out = Mat::<f32>::zeros(n_cells, branch_probs_wp.ncols());

    matmul(
        out.as_mut(),
        Accum::Replace,
        w.transpose(),
        branch_probs_wp,
        1.0f32,
        faer_parallelism(),
    );

    Ok(out)
}

/// Differentiation entropy of per-cell fate probabilities.
///
/// Each row is renormalised to sum to one and then scored with the natural-log
/// Shannon entropy, `0 ln 0` taken as zero, matching `scipy.stats.entropy`.
///
/// ### Params
///
/// * `branch_probs` - Fate probabilities, cells by terminal states.
///
/// ### Returns
///
/// One entropy value per cell. An all-zero row scores zero rather than `NaN`.
pub fn branch_entropy(branch_probs: MatRef<f32>) -> Vec<f32> {
    let n_states = branch_probs.ncols();

    (0..branch_probs.nrows())
        .into_par_iter()
        .map(|i| {
            let total: f64 = (0..n_states).map(|k| branch_probs[(i, k)] as f64).sum();
            if total <= 0.0 || !total.is_finite() {
                return 0.0;
            }
            let h: f64 = (0..n_states)
                .map(|k| {
                    let p = branch_probs[(i, k)] as f64 / total;
                    if p > 0.0 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            h as f32
        })
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A CSR transition matrix from dense rows that already sum to one, with
    /// structural zeros dropped.
    fn chain_from_rows(rows: &[Vec<f32>]) -> CompressedSparseData2<f32> {
        let n = rows.len();
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in rows {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    indices.push(j as u32);
                    data.push(v);
                }
            }
            indptr.push(indices.len() as u32);
        }
        CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n))
    }

    /// The waypoint chain is row-stochastic and no waypoint transitions to itself.
    #[test]
    fn test_transitions_rows_sum_to_one_and_have_no_self_loops() {
        // A short pseudotime-ordered line of waypoints.
        let wp_data: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32, 0.0]).collect();
        let wp_pt: Vec<f32> = (0..12).map(|i| i as f32 / 11.0).collect();

        let t = build_waypoint_transitions(&wp_data, &wp_pt, 6, false).unwrap();

        for i in 0..12 {
            let lo = t.indptr[i] as usize;
            let hi = t.indptr[i + 1] as usize;
            let sum: f32 = t.data[lo..hi].iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
            assert!(
                !t.indices[lo..hi].contains(&(i as u32)),
                "self loop on waypoint {i}"
            );
        }
    }

    /// Edges running far backwards in pseudotime are pruned out of the chain.
    #[test]
    fn test_transitions_prune_far_back_edges() {
        // Widely spaced pseudotime means backward edges cannot survive the
        // bandwidth, so every kept edge must run forwards.
        let wp_data: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32 * 0.01]).collect();
        let wp_pt: Vec<f32> = (0..12).map(|i| i as f32).collect();

        let t = build_waypoint_transitions(&wp_data, &wp_pt, 6, false).unwrap();

        for i in 0..12 {
            for idx in t.indptr[i] as usize..t.indptr[i + 1] as usize {
                let j = t.indices[idx] as usize;
                assert!(
                    wp_pt[j] >= wp_pt[i] - 1.0,
                    "edge {i} -> {j} runs too far backwards"
                );
            }
        }
    }

    /// A `knn` below the bandwidth rank errors rather than silently degrading.
    #[test]
    fn test_transitions_reject_small_knn() {
        let wp_data: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32]).collect();
        let wp_pt: Vec<f32> = (0..12).map(|i| i as f32).collect();

        assert!(matches!(
            build_waypoint_transitions(&wp_data, &wp_pt, 4, false),
            Err(BixverseErrors::PalantirKnnTooSmall { .. })
        ));
    }

    /// Regression: an oversized `knn` fed the raw value to the bandwidth and stopped the pruning cutting.
    #[test]
    fn test_transitions_clamp_knn_to_the_waypoint_count() {
        // Asking for more neighbours than there are waypoints must behave
        // exactly like asking for all of them. Feeding the raw knn to the
        // bandwidth instead saturates the rank on the farthest neighbour, which
        // widens the back-edge tolerance and stops the pruning cutting.
        let wp_data: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32 * 0.01]).collect();
        let wp_pt: Vec<f32> = (0..12).map(|i| i as f32 * 0.02).collect();

        let oversized = build_waypoint_transitions(&wp_data, &wp_pt, 40, false).unwrap();
        let exact = build_waypoint_transitions(&wp_data, &wp_pt, 12, false).unwrap();

        assert_eq!(oversized.indptr, exact.indptr);
        assert_eq!(oversized.indices, exact.indices);
    }

    /// The diagnostic names the waypoint count, not the `knn` the caller passed.
    #[test]
    fn test_transitions_reject_too_few_waypoints_for_the_bandwidth() {
        // Four waypoints cannot support a rank-`knn / 3 - 1` bandwidth however
        // large the requested knn is. The diagnostic has to say so: reporting
        // "knn 4 is below the minimum of 6" to someone who passed knn = 30
        // sends them after a parameter that is not the problem.
        let wp_data: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32]).collect();
        let wp_pt: Vec<f32> = (0..4).map(|i| i as f32).collect();

        assert!(matches!(
            build_waypoint_transitions(&wp_data, &wp_pt, 30, false),
            Err(BixverseErrors::PalantirTooFewWaypoints {
                n_waypoints: 4,
                minimum: 6
            })
        ));
    }

    /// Mismatched waypoint and pseudotime lengths are refused up front.
    #[test]
    fn test_transitions_reject_a_waypoint_pseudotime_mismatch() {
        let wp_data: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32]).collect();
        let wp_pt: Vec<f32> = (0..3).map(|i| i as f32).collect();

        assert!(matches!(
            build_waypoint_transitions(&wp_data, &wp_pt, 6, false),
            Err(BixverseErrors::PalantirWaypointDataMismatch {
                n_waypoints: 8,
                n_pseudotime: 3
            })
        ));
    }

    /// Regression: a fully pruned row handed its mass to the nearest neighbour, suppressing the terminal state.
    #[test]
    fn test_fully_pruned_row_never_runs_backwards() {
        // Widely spaced pseudotime against tightly spaced coordinates, so every
        // back edge is pruned for every waypoint. The last waypoint has nothing
        // ahead of it, and handing its whole row to the nearest neighbour would
        // push all of its mass back down the trunk, suppressing exactly the
        // terminal state it is.
        let wp_data: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32 * 0.01]).collect();
        let wp_pt: Vec<f32> = (0..12).map(|i| i as f32).collect();

        let t = build_waypoint_transitions(&wp_data, &wp_pt, 6, false).unwrap();

        for i in 0..12 {
            for idx in t.indptr[i] as usize..t.indptr[i + 1] as usize {
                let j = t.indices[idx] as usize;
                assert!(
                    wp_pt[j] >= wp_pt[i],
                    "edge {i} -> {j} runs backwards in pseudotime"
                );
            }
        }

        // The last waypoint falls back to a self loop, which leaves it a sink.
        let last = 11usize;
        let lo = t.indptr[last] as usize;
        let hi = t.indptr[last + 1] as usize;
        assert_eq!(&t.indices[lo..hi], &[last as u32]);
    }

    /// The stationary distribution of a hand-solved two-state chain.
    #[test]
    fn test_stationary_of_two_state_chain() {
        // P(0->1) = 0.25, P(1->0) = 0.5; stationary is (2/3, 1/3).
        let t = chain_from_rows(&[vec![0.75, 0.25], vec![0.5, 0.5]]);

        let pi = stationary_distribution(&t).unwrap();

        assert_relative_eq!(pi[0], 2.0 / 3.0, epsilon = 1e-9);
        assert_relative_eq!(pi[1], 1.0 / 3.0, epsilon = 1e-9);
    }

    /// Regression: per-row `f32` drift shifted the stationary ranks past the terminal-state cutoff.
    #[test]
    fn test_stationary_ignores_per_row_scale_drift() {
        // Row 0 scaled up, magnifying the per-row `f32` residue that the L1
        // renormalisation cannot undo. Without the `f64` rescale the answer is
        // 0.6702 / 0.3298 against a true 2/3, and `detect_terminal_states`
        // thresholds those ranks at `median + 3.719 * MAD`, so a shift that size
        // moves an arm tip across the cutoff and costs a whole branch.
        let drift = 1.05f32;
        let t = chain_from_rows(&[vec![0.75 * drift, 0.25 * drift], vec![0.5, 0.5]]);

        let pi = stationary_distribution(&t).unwrap();

        assert_relative_eq!(pi[0], 2.0 / 3.0, epsilon = 1e-9);
        assert_relative_eq!(pi[1], 1.0 / 3.0, epsilon = 1e-9);
    }

    /// The lazy chain converges where the raw one oscillates forever.
    #[test]
    fn test_stationary_survives_a_periodic_chain() {
        // A pure two-cycle: the raw chain never converges, the lazy one does.
        let t = chain_from_rows(&[vec![0.0, 1.0], vec![1.0, 0.0]]);

        let pi = stationary_distribution(&t).unwrap();

        assert_relative_eq!(pi[0], 0.5, epsilon = 1e-9);
        assert_relative_eq!(pi[1], 0.5, epsilon = 1e-9);
    }

    /// Absorption probabilities on a hand-solved chain, with one-hot absorbing rows.
    #[test]
    fn test_absorption_on_a_three_state_chain() {
        // State 1 is transient with a 30/70 split onto absorbers 0 and 2.
        let t = chain_from_rows(&[
            vec![1.0, 0.0, 0.0],
            vec![0.3, 0.0, 0.7],
            vec![0.0, 0.0, 1.0],
        ]);

        let (b, stranded) = absorption_probabilities(&t, &[0, 2]).unwrap();
        assert_eq!(stranded, 0);

        assert_relative_eq!(b[(1, 0)], 0.3, epsilon = 1e-6);
        assert_relative_eq!(b[(1, 1)], 0.7, epsilon = 1e-6);
        // Absorbing rows are one-hot.
        assert_relative_eq!(b[(0, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(b[(2, 1)], 1.0, epsilon = 1e-6);
    }

    /// A transient passing through another transient inherits its fate split.
    #[test]
    fn test_absorption_through_a_transient_cascade() {
        // 1 -> 2 -> {0, 3}: the analytic answer for state 1 matches state 2.
        let t = chain_from_rows(&[
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.25, 0.0, 0.0, 0.75],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);

        let (b, _) = absorption_probabilities(&t, &[0, 3]).unwrap();

        assert_relative_eq!(b[(1, 0)], 0.25, epsilon = 1e-6);
        assert_relative_eq!(b[(1, 1)], 0.75, epsilon = 1e-6);
        assert_relative_eq!(b[(2, 0)], 0.25, epsilon = 1e-6);
    }

    /// With every transient reachable, each row of the solve is a full distribution.
    #[test]
    fn test_absorption_rows_sum_to_one() {
        let t = chain_from_rows(&[
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.4, 0.5, 0.0],
            vec![0.0, 0.2, 0.3, 0.5],
            vec![0.0, 0.0, 0.0, 1.0],
        ]);

        let (b, _) = absorption_probabilities(&t, &[0, 3]).unwrap();

        for i in 0..4 {
            let sum: f32 = (0..2).map(|k| b[(i, k)]).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    /// A transient that can never be absorbed is dropped and counted, not left to make `(I - Q)` singular.
    #[test]
    fn test_absorption_reports_a_stranded_transient() {
        // State 1 only ever loops back to itself, so it can never be absorbed.
        // Including it would make (I - Q) singular; it is dropped and counted.
        let t = chain_from_rows(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.0, 0.5],
        ]);

        let (b, stranded) = absorption_probabilities(&t, &[0]).unwrap();

        assert_eq!(stranded, 1);
        assert_relative_eq!(b[(1, 0)], 0.0, epsilon = 1e-6);
        // State 2 still absorbs into state 0 with certainty.
        assert_relative_eq!(b[(2, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(b[(0, 0)], 1.0, epsilon = 1e-6);
    }

    /// Regression: rescaling surviving edges turned a 1:10 fate ratio into an even split.
    #[test]
    fn test_absorption_does_not_redistribute_mass_lost_to_a_strand() {
        // 0 and 1 absorb, 2 is stranded on a self loop, and state 5 leaks half
        // its mass towards it. The true fates of state 5 are 0.05 and 0.5, a
        // 1:10 ratio. Rescaling the surviving edges to sum to one gives
        // 0.5 / 0.5, a plausible-looking answer to a different question.
        let t = chain_from_rows(&[
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.0, 0.9, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        ]);

        let (b, stranded) = absorption_probabilities(&t, &[0, 1]).unwrap();

        assert_eq!(stranded, 1);
        assert_relative_eq!(b[(5, 0)], 0.05, epsilon = 1e-6);
        assert_relative_eq!(b[(5, 1)], 0.5, epsilon = 1e-6);
        assert_relative_eq!(b[(3, 0)], 0.1, epsilon = 1e-6);
        assert_relative_eq!(b[(3, 1)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(b[(4, 1)], 1.0, epsilon = 1e-6);
        // The stranded state itself carries nothing.
        assert_relative_eq!(b[(2, 0)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(b[(2, 1)], 0.0, epsilon = 1e-6);
    }

    /// Leaking into a strand may put a row below one; nothing may put it above.
    #[test]
    fn test_absorption_rows_never_sum_above_one() {
        // Same chain as above: leaking into the strand means rows are allowed
        // below one, but nothing may ever come out above it.
        let t = chain_from_rows(&[
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.0, 0.9, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        ]);

        let (b, _) = absorption_probabilities(&t, &[0, 1]).unwrap();

        for i in 0..6 {
            let sum: f32 = (0..2).map(|k| b[(i, k)]).sum();
            assert!((0.0..=1.0 + 1e-5).contains(&sum), "row {i} sums to {sum}");
        }
    }

    /// Regression: `f32` weight residue broke the row-sum check on a well-conditioned solve.
    #[test]
    fn test_absorption_survives_a_chain_with_a_huge_absorption_time() {
        // A birth-death chain biased away from its absorber: 30 states at
        // 0.6 / 0.4 reaches roughly 1.9e5 expected steps. The weights are the
        // point, since `0.4f32 + 0.6f32` is 1.0000000298. Without the `f64`
        // rescale in [absorption_probabilities] the row sums to 1.0606, sixty
        // times ABSORPTION_ROW_SUM_TOL, on a machine-precision solve.
        const N: usize = 31;
        let mut rows = vec![vec![0.0f32; N]; N];
        rows[0][1] = 1.0;
        for i in 1..N - 1 {
            rows[i][i + 1] = 0.4;
            rows[i][i - 1] = 0.6;
        }
        rows[N - 1][N - 1] = 1.0;
        let t = chain_from_rows(&rows);

        let (b, stranded) = absorption_probabilities(&t, &[N - 1]).unwrap();

        assert_eq!(stranded, 0);
        // One absorber, every state reaches it, so every row is exactly one.
        for i in 0..N {
            assert_relative_eq!(b[(i, 0)], 1.0, epsilon = 1e-4);
        }
    }

    /// No terminal states is an error, not an empty result.
    #[test]
    fn test_absorption_rejects_an_empty_absorbing_set() {
        let t = chain_from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]]);

        assert!(matches!(
            absorption_probabilities(&t, &[]),
            Err(BixverseErrors::PalantirNoTerminalStates)
        ));
    }

    /// Cell-level fates are the waypoint weights applied to the waypoint fates.
    #[test]
    fn test_projection_is_a_convex_combination() {
        // Two waypoints, three cells, weights split 0.25 / 0.75 per cell.
        let weights = vec![0.25, 0.25, 0.25, 0.75, 0.75, 0.75];
        let b_wp = Mat::<f32>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });

        let out = project_to_cells(&weights, 2, 3, b_wp.as_ref()).unwrap();

        for c in 0..3 {
            assert_relative_eq!(out[(c, 0)], 0.25, epsilon = 1e-6);
            assert_relative_eq!(out[(c, 1)], 0.75, epsilon = 1e-6);
        }
    }

    /// Branch entropy against hand-computed values, unnormalised rows included.
    #[test]
    fn test_entropy_matches_known_values() {
        let bp = Mat::<f32>::from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (0, 1) => 0.0,
            (1, _) => 0.5,
            // Unnormalised row: the renormalisation must still give ln 2.
            (2, _) => 4.0,
            _ => 0.0,
        });

        let h = branch_entropy(bp.as_ref());

        assert_relative_eq!(h[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(h[1], std::f32::consts::LN_2, epsilon = 1e-6);
        assert_relative_eq!(h[2], std::f32::consts::LN_2, epsilon = 1e-6);
    }

    /// An all-zero row has no fates to be uncertain between, so its entropy is zero.
    #[test]
    fn test_entropy_of_an_all_zero_row_is_zero_not_nan() {
        let bp = Mat::<f32>::zeros(2, 3);

        let h = branch_entropy(bp.as_ref());

        assert!(h.iter().all(|v| v.is_finite()));
        assert_relative_eq!(h[0], 0.0, epsilon = 1e-6);
    }

    /// Detection picks the late boundary of a line, not merely some boundary.
    #[test]
    fn test_detect_terminal_states_finds_the_late_end_of_a_line() {
        // A pseudotime-ordered line: the late end accumulates the mass and is
        // the only thing that should come back. Asserting that the result is a
        // boundary waypoint pins nothing, since `detect_terminal_states` returns
        // `nearest_candidate(_, &boundaries, _)` by construction; a line has
        // exactly two boundaries and picking the *early* one would be the bug.
        let wp_data: Vec<Vec<f32>> = (0..30).map(|i| vec![i as f32 / 29.0]).collect();
        let wp_pt: Vec<f32> = (0..30).map(|i| i as f32 / 29.0).collect();

        let t = build_waypoint_transitions(&wp_data, &wp_pt, 6, false).unwrap();
        let terminal = detect_terminal_states(&t, &wp_data, &wp_pt).unwrap();

        assert_eq!(terminal, vec![29]);
    }

    /// A zero MAD would let half the waypoints clear the cutoff, so it errors instead.
    #[test]
    fn test_detect_terminal_states_rejects_a_flat_stationary_spread() {
        // A uniform cycle: every waypoint carries the same stationary mass, so
        // the MAD is exactly zero and the cutoff lands on the median. Roughly
        // half of everything then clears it, `induced_components` merges the
        // lot, and the branching structure silently collapses to one terminal
        // state. The empty case is guarded; this is the opposite failure.
        let n = 8usize;
        let mut rows = vec![vec![0.0f32; n]; n];
        for (i, row) in rows.iter_mut().enumerate() {
            row[(i + 1) % n] = 1.0;
        }
        let t = chain_from_rows(&rows);
        let wp_data: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32]).collect();
        let wp_pt: Vec<f32> = (0..n).map(|i| i as f32).collect();

        assert!(matches!(
            detect_terminal_states(&t, &wp_data, &wp_pt),
            Err(BixverseErrors::PalantirStationaryRanksDegenerate { n_waypoints: 8 })
        ));
    }

    /// Regression: an exact-zero edge counted as reachable and hid the waypoint from the strand diagnostic.
    #[test]
    fn test_stranded_waypoints_count_a_zero_weight_route_out() {
        // State 1's only edge to the absorber carries an exact zero. Walking
        // structural nonzeros alone calls it reachable, so it solves to zero and
        // never shows up in `stranded_waypoints`, which is the user's headline
        // diagnostic for over-pruning. The affinity kernel's `exp()` really does
        // underflow to `0.0f32` on a pruned corridor, so this is not a
        // hypothetical.
        let data = vec![1.0f32, 0.0, 1.0, 0.5, 0.5];
        let indices = vec![0u32, 0, 1, 0, 2];
        let indptr = vec![0u32, 1, 3, 5];
        let t = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (3, 3));

        let (b, stranded) = absorption_probabilities(&t, &[0]).unwrap();

        assert_eq!(stranded, 1);
        assert_relative_eq!(b[(1, 0)], 0.0, epsilon = 1e-6);
    }
}
