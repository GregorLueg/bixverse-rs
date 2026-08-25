//! Sparse multiple canonical correlation analysis by penalised matrix
//! decomposition.
//!
//! Given `K` matrices measured on the same rows, find sparse weight vectors
//! maximising `sum_{i<j} w_i' X_i' X_j w_j` subject to `||w_i||_2 = 1` and
//! `||w_i||_1 <= c_i`. Block coordinate ascent: each block's update is a
//! soft-threshold of its cross products against the other blocks' current
//! variates, with the threshold bisected to sit exactly on the L1 budget.
//! Components come out one at a time, each deflated against its predecessors.
//!
//! A port of `PMA::MultiCCA`. Three things in the original are easy to
//! reimplement subtly differently and all three change the answer, so they are
//! called out where they happen: the deflation uses a *diagonalised* cross
//! product rather than a Hotelling deflation, the convergence test compares
//! against a criterion computed from the previous sweep's weights, and the L2
//! normaliser floors at 0.05 instead of erroring on a zero vector.
//!
//! ### References
//!
//! Witten & Tibshirani, Statistical Applications in Genetics and Molecular
//! Biology, 2009. Witten, Tibshirani & Hastie, Biostatistics 10(3), 2009.

use faer::{ColRef, Mat, MatRef};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::core::math::matrix_helpers::scale_matrix_col;
use crate::core::math::vector_helpers::pearson_correlation;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Bisection budget for the L1 threshold search.
///
/// The bracket starts at `[0, max|g|]` and halves every step, so 150 steps take
/// any realistic gradient far below the interval tolerance. It is a runaway
/// guard rather than a working limit, and matches `PMA::BinarySearch`.
const L1_SEARCH_MAX_ITER: usize = 150;

/// Absolute width at which the L1 bisection stops.
const L1_SEARCH_TOL: f64 = 1e-6;

/// Backoff subtracted from the upper end of the bisection bracket.
///
/// At exactly `max|g|` the soft-threshold annihilates the whole vector, and the
/// L1-to-L2 ratio it is bisecting on is then undefined. `PMA` steps just inside.
const L1_SEARCH_UPPER_BACKOFF: f64 = 1e-5;

/// Floor for the L2 normaliser, from `PMA::l2n`.
///
/// A block whose gradient is entirely annihilated by the threshold would
/// otherwise divide by zero. Dividing by 0.05 leaves the zero vector a zero
/// vector, so the block simply contributes nothing that sweep.
const L2_NORM_FLOOR: f64 = 0.05;

/// Relative change in the criterion below which a component has converged.
const DEFAULT_TOL: f64 = 1e-3;

/// Sweeps per component. `PMA`'s own default is 25; DIALOGUE asks for 100.
const DEFAULT_N_ITER: usize = 100;

/// Number of penalty values in an auto-generated grid, from
/// `PMA::MultiCCA.permute`.
const PENALTY_GRID_SIZE: usize = 10;

/// Lowest multiplier of `sqrt(p_k)` in an auto-generated penalty grid.
const PENALTY_GRID_LOWER: f64 = 0.1;

/// Highest multiplier of `sqrt(p_k)` in an auto-generated penalty grid.
const PENALTY_GRID_UPPER: f64 = 0.8;

/// Floor on any auto-generated penalty. The L1 bound must exceed one or the
/// constraint set is empty.
const PENALTY_GRID_FLOOR: f64 = 1.1;

/// Sweeps per component while tuning. Tuning only needs the first component to
/// roughly the right place, and it refits once per grid point per permutation.
const PERMUTE_N_ITER: usize = 3;

/// Permutations per grid point, from `PMA::MultiCCA.permute`.
const PERMUTE_N_PERMS: usize = 10;

/// Stabiliser on the permutation standard deviation in the z-statistic.
///
/// Without it a grid point whose permuted correlations happen to agree closely
/// wins on a near-zero denominator rather than on a large gap.
const PERMUTE_SD_FLOOR: f64 = 0.05;

////////////////////
// MultiCcaParams //
////////////////////

/// Tuning knobs for [multi_cca].
#[derive(Clone, Debug)]
pub struct MultiCcaParams<T> {
    /// Number of components to extract.
    pub n_components: usize,
    /// Maximum block coordinate ascent sweeps per component.
    pub n_iter: usize,
    /// Relative change in the criterion below which a component has converged.
    pub tol: T,
    /// L1 bound per block. `None` picks `sqrt(p_1) / 2` for every block, which
    /// is what DIALOGUE uses; a bound must lie in `[1, sqrt(p_k)]`.
    pub penalties: Option<Vec<T>>,
    /// Centre and scale every column to unit variance first, with an `n - 1`
    /// denominator. `PMA` does this unconditionally by default.
    pub standardise: bool,
}

impl<T: BixverseFloat> MultiCcaParams<T> {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `n_components` - Number of components to extract
    /// * `n_iter` - Maximum sweeps per component
    /// * `tol` - Relative convergence tolerance on the criterion
    /// * `penalties` - Per-block L1 bounds, or `None` to derive them
    /// * `standardise` - Centre and scale the columns first
    ///
    /// ### Returns
    ///
    /// The parameter set. Nothing is validated here; [multi_cca] checks the
    /// values against the data it is given.
    pub fn new(
        n_components: usize,
        n_iter: usize,
        tol: T,
        penalties: Option<Vec<T>>,
        standardise: bool,
    ) -> Self {
        Self {
            n_components,
            n_iter,
            tol,
            penalties,
            standardise,
        }
    }
}

impl<T: BixverseFloat> Default for MultiCcaParams<T> {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_iter: DEFAULT_N_ITER,
            tol: T::from_f64(DEFAULT_TOL).unwrap(),
            penalties: None,
            standardise: true,
        }
    }
}

/// What [multi_cca] found.
#[derive(Clone, Debug)]
pub struct MultiCcaResult<T> {
    /// One `p_k x n_components` weight matrix per block. Sparse: entries the
    /// L1 bound annihilated are exactly zero.
    pub ws: Vec<Mat<T>>,
    /// `sum_{i<j} cor(X_i w_i, X_j w_j)` per component.
    pub cors: Vec<T>,
    /// The L1 bound actually applied to each block.
    pub penalties: Vec<T>,
}

////////////////////
// The main solve //
////////////////////

/// Sparse multiple canonical correlation analysis.
///
/// Every block must have the same number of rows and at least two columns.
/// Components are extracted sequentially; each sweep cycles the blocks in
/// order, so the block order is part of the specification rather than an
/// implementation detail.
///
/// Initialised from the leading right singular vectors of each block. Sign
/// conventions for singular vectors are arbitrary and differ between LAPACK
/// and `faer`, and the objective is *not* invariant to flipping a single
/// block's weights, only to flipping all of them at once. A deterministic
/// convention is applied here (largest-magnitude entry made positive) so
/// repeated runs agree; a run of `PMA` on the same data may still land on a
/// different local optimum.
///
/// ### Params
///
/// * `x` - The `K` blocks, each `n x p_k`, rows aligned across blocks
/// * `params` - Optional [MultiCcaParams]; [MultiCcaParams::default] otherwise
///
/// ### Returns
///
/// The [MultiCcaResult], or [BixverseErrors::InvalidArgument] for fewer than
/// two blocks, a block with fewer than two columns, or an L1 bound outside
/// `[1, sqrt(p_k)]`; [BixverseErrors::ShapeMismatch] when the row counts
/// disagree.
///
/// ### References
///
/// Witten & Tibshirani, Statistical Applications in Genetics and Molecular
/// Biology, 2009
pub fn multi_cca<T: BixverseFloat>(
    x: &[MatRef<T>],
    params: Option<MultiCcaParams<T>>,
) -> Result<MultiCcaResult<T>, BixverseErrors> {
    let params = params.unwrap_or_default();
    let n_comp = params.n_components;
    if n_comp == 0 {
        return Err(BixverseErrors::InvalidArgument(
            "multi-CCA needs at least one component.".to_string(),
        ));
    }
    let blocks = prepare_blocks(x, params.standardise)?;
    let k = blocks.len();

    let penalties = resolve_penalties(&blocks, params.penalties.as_deref())?;

    // Leading right singular vectors, one column per component.
    let mut init: Vec<Mat<T>> = Vec::with_capacity(k);
    for block in blocks.iter() {
        init.push(svd_init(block.as_ref(), n_comp)?);
    }

    let mut ws_final: Vec<Mat<T>> = blocks
        .iter()
        .map(|b| Mat::<T>::zeros(b.ncols(), n_comp))
        .collect();
    let mut cors: Vec<T> = Vec::with_capacity(n_comp);

    for comp in 0..n_comp {
        let diag = deflation_table(&blocks, &ws_final, comp);
        let mut w: Vec<Vec<T>> = init
            .iter()
            .map(|m| (0..m.nrows()).map(|i| m[(i, comp)]).collect())
            .collect();

        // The criterion is evaluated at the top of the sweep, so it lags the
        // weights by one. Seeding the pair at -10 and -20 is what makes the
        // first test pass unconditionally and the loop run at least twice.
        // Both come straight from PMA; changing either changes where it stops.
        let mut crit_old = T::from_f64(-10.0).unwrap();
        let mut crit = T::from_f64(-20.0).unwrap();
        let mut sweep = 0;
        while sweep < params.n_iter
            && ((crit_old - crit).abs() / crit_old.abs()) > params.tol
            && crit_old != T::zero()
        {
            crit_old = crit;
            crit = criterion(&blocks, &w);
            sweep += 1;
            for i in 0..k {
                w[i] = update_block(&blocks, i, &w, &ws_final, comp, &diag, penalties[i]);
            }
        }

        for (i, wi) in w.iter().enumerate() {
            for (j, &v) in wi.iter().enumerate() {
                ws_final[i][(j, comp)] = v;
            }
        }
        cors.push(sum_pairwise_correlations(&blocks, &w));
    }

    Ok(MultiCcaResult {
        ws: ws_final,
        cors,
        penalties,
    })
}

/// Validates the blocks and optionally standardises their columns.
///
/// ### Params
///
/// * `x` - The blocks as supplied
/// * `standardise` - Centre and scale each column
///
/// ### Returns
///
/// Owned copies ready for the solve, or the relevant [BixverseErrors].
fn prepare_blocks<T: BixverseFloat>(
    x: &[MatRef<T>],
    standardise: bool,
) -> Result<Vec<Mat<T>>, BixverseErrors> {
    if x.len() < 2 {
        return Err(BixverseErrors::InvalidArgument(format!(
            "multi-CCA needs at least two blocks; got {}.",
            x.len()
        )));
    }
    let n = x[0].nrows();
    for (i, block) in x.iter().enumerate() {
        if block.nrows() != n {
            return Err(BixverseErrors::ShapeMismatch {
                expected: (n, block.ncols()),
                got: (block.nrows(), block.ncols()),
            });
        }
        if block.ncols() < 2 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "block {i} has {} columns; multi-CCA needs at least 2.",
                block.ncols()
            )));
        }
    }
    Ok(x.iter()
        .map(|b| {
            if standardise {
                scale_matrix_col(b, true)
            } else {
                b.to_owned()
            }
        })
        .collect())
}

/// Resolves the per-block L1 bounds.
///
/// `None` gives `sqrt(p_1) / 2` for *every* block, using the first block's
/// column count throughout. That is what DIALOGUE passes and it is deliberate
/// there, where every block has the same number of features; it is worth
/// knowing when the blocks differ in width.
///
/// ### Params
///
/// * `blocks` - The prepared blocks
/// * `penalties` - Supplied bounds, one per block, or `None`
///
/// ### Returns
///
/// One bound per block, or [BixverseErrors::InvalidArgument] when a bound is
/// outside `[1, sqrt(p_k)]` or the count does not match.
fn resolve_penalties<T: BixverseFloat>(
    blocks: &[Mat<T>],
    penalties: Option<&[T]>,
) -> Result<Vec<T>, BixverseErrors> {
    let k = blocks.len();
    let resolved: Vec<T> = match penalties {
        Some(p) if p.len() == 1 => vec![p[0]; k],
        Some(p) if p.len() == k => p.to_vec(),
        Some(p) => {
            return Err(BixverseErrors::InvalidArgument(format!(
                "expected 1 or {k} penalties; got {}.",
                p.len()
            )));
        }
        None => {
            let default =
                T::from_usize(blocks[0].ncols()).unwrap().sqrt() / T::from_f64(2.0).unwrap();
            vec![default; k]
        }
    };

    for (i, (&p, block)) in resolved.iter().zip(blocks.iter()).enumerate() {
        let upper = T::from_usize(block.ncols()).unwrap().sqrt();
        if p < T::one() || p > upper {
            return Err(BixverseErrors::InvalidArgument(format!(
                "the L1 bound for block {i} must lie in [1, sqrt(p_k)]; got {p} against an \
                 upper limit of {upper}."
            )));
        }
    }
    Ok(resolved)
}

/// Leading right singular vectors of a block, sign-normalised.
///
/// ### Params
///
/// * `block` - One prepared block
/// * `n_comp` - How many singular vectors to take
///
/// ### Returns
///
/// A `p_k x n_comp` matrix, each column with its largest-magnitude entry
/// positive, or [BixverseErrors::FaerSvdError].
fn svd_init<T: BixverseFloat>(block: MatRef<T>, n_comp: usize) -> Result<Mat<T>, BixverseErrors> {
    let svd = block
        .thin_svd()
        .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;
    let v = svd.V();
    let p = block.ncols();
    let available = v.ncols();
    let mut out = Mat::<T>::zeros(p, n_comp);
    for c in 0..n_comp {
        // A block of rank below n_comp has no further singular vectors. Fall
        // back to a unit basis vector so the ascent still has somewhere to
        // start rather than an all-zero column it can never leave.
        if c < available {
            for i in 0..p {
                out[(i, c)] = v[(i, c)];
            }
        } else {
            out[(c % p, c)] = T::one();
        }
        normalise_sign(&mut out, c);
    }
    Ok(out)
}

/// Flips a column so its largest-magnitude entry is positive.
///
/// ### Params
///
/// * `m` - Matrix to adjust in place
/// * `col` - Column index
fn normalise_sign<T: BixverseFloat>(m: &mut Mat<T>, col: usize) {
    let mut best = T::zero();
    let mut sign = T::one();
    for i in 0..m.nrows() {
        let v = m[(i, col)];
        if v.abs() > best {
            best = v.abs();
            sign = if v < T::zero() { -T::one() } else { T::one() };
        }
    }
    if sign < T::zero() {
        for i in 0..m.nrows() {
            m[(i, col)] = -m[(i, col)];
        }
    }
}

/// One block's coordinate ascent update.
///
/// Accumulates `sum_{j != i} X_i' X_j w_j`, less a deflation term against the
/// components already extracted, then soft-thresholds onto the L1 budget and
/// renormalises to unit L2.
///
/// The deflation is `PMA`'s, and it is not a Hotelling deflation: the
/// cross-product `W_i' X_i' X_j W_j` between the two blocks' *finished*
/// components is stripped of its off-diagonal before being applied. That makes
/// it an approximate orthogonality constraint, and reproducing it exactly is
/// the difference between matching `PMA` and merely resembling it.
///
/// ### Params
///
/// * `blocks` - The prepared blocks
/// * `i` - Block being updated
/// * `w` - Current weights for every block, this component
/// * `ws_final` - Components already extracted, zero-filled beyond `comp`
/// * `comp` - Component being extracted
/// * `diag` - Deflation table from [deflation_table], indexed
///   `(i * K + j) * comp + c`
/// * `penalty` - L1 bound for this block
///
/// ### Returns
///
/// The updated weight vector for block `i`.
fn update_block<T: BixverseFloat>(
    blocks: &[Mat<T>],
    i: usize,
    w: &[Vec<T>],
    ws_final: &[Mat<T>],
    comp: usize,
    diag: &[T],
    penalty: T,
) -> Vec<T> {
    let p_i = blocks[i].ncols();
    let mut totals = vec![T::zero(); p_i];

    for j in 0..blocks.len() {
        if j == i {
            continue;
        }
        // X_i' (X_j w_j)
        let xj_wj = mat_vec(&blocks[j], &w[j]);
        let cross = blocks[i].transpose() * ColRef::from_slice(&xj_wj);
        for (t, c) in totals.iter_mut().zip(cross.iter()) {
            *t += *c;
        }

        if comp == 0 {
            continue;
        }
        // Deflation: W_i (diag(W_i' X_i' X_j W_j) (W_j' w_j)). Only the
        // projection onto the current weights varies across sweeps; the
        // diagonal was computed once for the whole component.
        let k = blocks.len();
        let mut correction = vec![T::zero(); comp];
        for (c, corr) in correction.iter_mut().enumerate() {
            let proj: T = (0..blocks[j].ncols())
                .fold(T::zero(), |acc, r| acc + ws_final[j][(r, c)] * w[j][r]);
            *corr = diag[(i * k + j) * comp + c] * proj;
        }
        for (r, t) in totals.iter_mut().enumerate() {
            for (c, &corr) in correction.iter().enumerate() {
                *t -= ws_final[i][(r, c)] * corr;
            }
        }
    }

    let lambda = l1_threshold(&totals, penalty);
    let thresholded = soft_threshold(&totals, lambda);
    let norm = l2_norm_floored(&thresholded);
    thresholded.into_iter().map(|v| v / norm).collect()
}

/// Diagonal cross products between every pair of blocks' finished components.
///
/// `diag[(i * K + j) * comp + c]` is `(X_i W_i[:,c]) . (X_j W_j[:,c])`. None of
/// it depends on the weights currently being solved for, so it is constant for
/// the whole of a component's sweeps -- which is why it is built once here
/// rather than rebuilt inside every block update, where it cost a factor of
/// `n_iter` in matrix-vector products.
///
/// ### Params
///
/// * `blocks` - The prepared blocks
/// * `ws_final` - Components already extracted
/// * `comp` - Component being extracted; the table covers `0..comp`
///
/// ### Returns
///
/// The table, empty when `comp` is zero.
fn deflation_table<T: BixverseFloat>(
    blocks: &[Mat<T>],
    ws_final: &[Mat<T>],
    comp: usize,
) -> Vec<T> {
    let k = blocks.len();
    if comp == 0 {
        return Vec::new();
    }
    let variates: Vec<Vec<Vec<T>>> = (0..k)
        .map(|b| {
            (0..comp)
                .map(|c| {
                    let column: Vec<T> = (0..blocks[b].ncols())
                        .map(|r| ws_final[b][(r, c)])
                        .collect();
                    mat_vec(&blocks[b], &column)
                })
                .collect()
        })
        .collect();

    let mut out = vec![T::zero(); k * k * comp];
    for i in 0..k {
        for j in 0..k {
            for c in 0..comp {
                out[(i * k + j) * comp + c] = variates[i][c]
                    .iter()
                    .zip(variates[j][c].iter())
                    .fold(T::zero(), |acc, (a, b)| acc + *a * *b);
            }
        }
    }
    out
}

/// `sum_{i<j} w_i' X_i' X_j w_j`, the objective being maximised.
///
/// ### Params
///
/// * `blocks` - The prepared blocks
/// * `w` - Current weights for every block
///
/// ### Returns
///
/// The criterion value.
fn criterion<T: BixverseFloat>(blocks: &[Mat<T>], w: &[Vec<T>]) -> T {
    let variates: Vec<Vec<T>> = blocks
        .iter()
        .zip(w.iter())
        .map(|(b, wi)| mat_vec(b, wi))
        .collect();
    let mut total = T::zero();
    for i in 1..blocks.len() {
        for j in 0..i {
            total += variates[i]
                .iter()
                .zip(variates[j].iter())
                .fold(T::zero(), |acc, (a, b)| acc + *a * *b);
        }
    }
    total
}

/// `sum_{i<j} cor(X_i w_i, X_j w_j)`, with a degenerate pair scoring zero.
///
/// ### Params
///
/// * `blocks` - The prepared blocks
/// * `w` - Current weights for every block
///
/// ### Returns
///
/// The summed pairwise correlation.
fn sum_pairwise_correlations<T: BixverseFloat>(blocks: &[Mat<T>], w: &[Vec<T>]) -> T {
    let variates: Vec<Vec<T>> = blocks
        .iter()
        .zip(w.iter())
        .map(|(b, wi)| mat_vec(b, wi))
        .collect();
    let mut total = T::zero();
    for i in 1..blocks.len() {
        for j in 0..i {
            let r = pearson_correlation(&variates[i], &variates[j]).unwrap_or(0.0);
            total += T::from_f64(r).unwrap_or_else(T::zero);
        }
    }
    total
}

/// Soft-threshold, `sign(x) max(0, |x| - lambda)`.
///
/// ### Params
///
/// * `x` - Vector to threshold
/// * `lambda` - Threshold, non-negative
///
/// ### Returns
///
/// The thresholded vector, same length.
fn soft_threshold<T: BixverseFloat>(x: &[T], lambda: T) -> Vec<T> {
    x.iter()
        .map(|&v| {
            let shrunk = v.abs() - lambda;
            if shrunk > T::zero() {
                if v < T::zero() { -shrunk } else { shrunk }
            } else {
                T::zero()
            }
        })
        .collect()
}

/// L2 norm with `PMA`'s floor: a zero vector reports 0.05 rather than zero.
///
/// ### Params
///
/// * `x` - Vector
///
/// ### Returns
///
/// The L2 norm, or 0.05 when it would be exactly zero.
fn l2_norm_floored<T: BixverseFloat>(x: &[T]) -> T {
    let norm = x.iter().fold(T::zero(), |acc, &v| acc + v * v).sqrt();
    if norm == T::zero() {
        T::from_f64(L2_NORM_FLOOR).unwrap()
    } else {
        norm
    }
}

/// Bisects the soft-threshold that puts `||w||_1 / ||w||_2` on the budget.
///
/// Returns zero when the unthresholded direction already satisfies the bound,
/// which is the case whenever the L1 constraint is slack.
///
/// ### Params
///
/// * `x` - Gradient to threshold
/// * `budget` - L1 bound on the unit-norm result
///
/// ### Returns
///
/// The threshold.
fn l1_threshold<T: BixverseFloat>(x: &[T], budget: T) -> T {
    let norm = l2_norm_floored(x);
    let raw_l1 = x.iter().fold(T::zero(), |acc, &v| acc + (v / norm).abs());
    if raw_l1 <= budget {
        return T::zero();
    }

    let mut lower = T::zero();
    let mut upper = x.iter().fold(T::zero(), |acc, &v| acc.max(v.abs()))
        - T::from_f64(L1_SEARCH_UPPER_BACKOFF).unwrap();
    let tol = T::from_f64(L1_SEARCH_TOL).unwrap();

    for _ in 0..L1_SEARCH_MAX_ITER {
        let mid = (lower + upper) / T::from_f64(2.0).unwrap();
        let candidate = soft_threshold(x, mid);
        let cand_norm = l2_norm_floored(&candidate);
        let l1 = candidate
            .iter()
            .fold(T::zero(), |acc, &v| acc + (v / cand_norm).abs());
        if l1 < budget {
            upper = mid;
        } else {
            lower = mid;
        }
        if (upper - lower) < tol {
            return (lower + upper) / T::from_f64(2.0).unwrap();
        }
    }
    (lower + upper) / T::from_f64(2.0).unwrap()
}

/// `X v`.
///
/// ### Params
///
/// * `m` - Matrix, `n x p`
/// * `v` - Vector, length `p`
///
/// ### Returns
///
/// The product, length `n`.
fn mat_vec<T: BixverseFloat>(m: &Mat<T>, v: &[T]) -> Vec<T> {
    let n = m.nrows();
    let p = m.ncols();
    let mut out = vec![T::zero(); n];
    for j in 0..p {
        let vj = v[j];
        if vj == T::zero() {
            continue;
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o += m[(i, j)] * vj;
        }
    }
    out
}

//////////////////
// Penalty tuning //
//////////////////////

/// Tuning knobs for [multi_cca_permute].
#[derive(Clone, Debug)]
pub struct MultiCcaPermuteParams<T> {
    /// Candidate L1 bounds, one row per grid point and one entry per block.
    /// `None` builds `PMA`'s grid: ten points spanning
    /// `[0.1, 0.8] * sqrt(p_k)`, floored at 1.1.
    pub grid: Option<Vec<Vec<T>>>,
    /// Permutations per grid point.
    pub n_perms: usize,
    /// Sweeps per fit while tuning.
    pub n_iter: usize,
    /// Seed for the row permutations.
    pub seed: u64,
}

impl<T: BixverseFloat> Default for MultiCcaPermuteParams<T> {
    fn default() -> Self {
        Self {
            grid: None,
            n_perms: PERMUTE_N_PERMS,
            n_iter: PERMUTE_N_ITER,
            seed: 42,
        }
    }
}

/// What [multi_cca_permute] found.
#[derive(Clone, Debug)]
pub struct MultiCcaPermuteResult<T> {
    /// The grid point with the highest z-statistic, one bound per block.
    pub best_penalties: Vec<T>,
    /// z-statistic per grid point.
    pub z_stats: Vec<T>,
    /// Fraction of permutations matching or beating the real fit, per grid
    /// point.
    pub p_values: Vec<T>,
    /// The grid that was searched.
    pub grid: Vec<Vec<T>>,
}

/// Picks an L1 bound by permutation.
///
/// For each candidate, fits the first component on the real data and on
/// `n_perms` row-permuted copies, then scores it by
/// `(real - mean(permuted)) / (sd(permuted) + 0.05)`. Rows are permuted
/// independently per block, which is what breaks the cross-block association
/// while leaving each block's own covariance intact.
///
/// Only worth calling when the grid has more than one point. DIALOGUE's default
/// path passes a single fixed bound, in which case `PMA` still runs every
/// permutation and then picks the only candidate; skip it and call [multi_cca]
/// directly instead.
///
/// ### Params
///
/// * `x` - The `K` blocks, each `n x p_k`
/// * `params` - Optional [MultiCcaPermuteParams]; the default otherwise
///
/// ### Returns
///
/// The [MultiCcaPermuteResult], or whatever [multi_cca] rejects.
pub fn multi_cca_permute<T: BixverseFloat>(
    x: &[MatRef<T>],
    params: Option<MultiCcaPermuteParams<T>>,
) -> Result<MultiCcaPermuteResult<T>, BixverseErrors> {
    let params = params.unwrap_or_default();
    let blocks = prepare_blocks(x, true)?;
    let grid = match params.grid {
        Some(g) => g,
        None => default_penalty_grid(&blocks),
    };
    if grid.is_empty() {
        return Err(BixverseErrors::InvalidArgument(
            "the penalty grid is empty.".to_string(),
        ));
    }

    // Blocks are standardised above, so the fits below must not repeat it.
    let fit = |b: &[Mat<T>], penalties: &[T], n_iter: usize| -> Result<T, BixverseErrors> {
        let refs: Vec<MatRef<T>> = b.iter().map(|m| m.as_ref()).collect();
        let res = multi_cca(
            &refs,
            Some(MultiCcaParams {
                n_components: 1,
                n_iter,
                tol: T::from_f64(DEFAULT_TOL).unwrap(),
                penalties: Some(penalties.to_vec()),
                standardise: false,
            }),
        )?;
        Ok(res.cors[0])
    };

    let real: Vec<T> = grid
        .iter()
        .map(|p| fit(&blocks, p, params.n_iter))
        .collect::<Result<_, _>>()?;

    let mut rng = StdRng::seed_from_u64(params.seed);
    let n = blocks[0].nrows();
    let mut permuted: Vec<Vec<T>> = vec![Vec::with_capacity(params.n_perms); grid.len()];
    for _ in 0..params.n_perms {
        let shuffled: Vec<Mat<T>> = blocks
            .iter()
            .map(|b| {
                let mut order: Vec<usize> = (0..n).collect();
                order.shuffle(&mut rng);
                Mat::<T>::from_fn(n, b.ncols(), |i, j| b[(order[i], j)])
            })
            .collect();
        for (g, point) in grid.iter().enumerate() {
            permuted[g].push(fit(&shuffled, point, params.n_iter)?);
        }
    }

    let sd_floor = T::from_f64(PERMUTE_SD_FLOOR).unwrap();
    let mut z_stats = Vec::with_capacity(grid.len());
    let mut p_values = Vec::with_capacity(grid.len());
    for (g, &observed) in real.iter().enumerate() {
        let null = &permuted[g];
        let m = T::from_usize(null.len()).unwrap();
        let mean = null.iter().fold(T::zero(), |a, &v| a + v) / m;
        let var = null
            .iter()
            .fold(T::zero(), |a, &v| a + (v - mean) * (v - mean))
            / (m - T::one()).max(T::one());
        z_stats.push((observed - mean) / (var.sqrt() + sd_floor));
        let hits = null.iter().filter(|&&v| v >= observed).count();
        p_values.push(T::from_usize(hits).unwrap() / m);
    }

    let best = z_stats
        .iter()
        .enumerate()
        .fold((0usize, T::neg_infinity()), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        })
        .0;

    Ok(MultiCcaPermuteResult {
        best_penalties: grid[best].clone(),
        z_stats,
        p_values,
        grid,
    })
}

/// `PMA`'s default penalty grid: ten points over `[0.1, 0.8] * sqrt(p_k)`,
/// floored at 1.1 so the L1 constraint stays feasible.
///
/// ### Params
///
/// * `blocks` - The prepared blocks
///
/// ### Returns
///
/// One row per grid point, one entry per block.
fn default_penalty_grid<T: BixverseFloat>(blocks: &[Mat<T>]) -> Vec<Vec<T>> {
    let floor = T::from_f64(PENALTY_GRID_FLOOR).unwrap();
    let steps = PENALTY_GRID_SIZE;
    (0..steps)
        .map(|s| {
            let frac = PENALTY_GRID_LOWER
                + (PENALTY_GRID_UPPER - PENALTY_GRID_LOWER) * s as f64 / (steps - 1) as f64;
            let frac = T::from_f64(frac).unwrap();
            blocks
                .iter()
                .map(|b| {
                    let upper = T::from_usize(b.ncols()).unwrap().sqrt();
                    (frac * upper).max(floor).min(upper)
                })
                .collect()
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

    /// Three blocks on 12 shared rows, the first two features of each loaded on
    /// a common latent factor. Generated in R and run through `PMA::MultiCCA`
    /// itself for the reference weights below.
    fn pma_fixture() -> (Mat<f64>, Mat<f64>, Mat<f64>) {
        let x1 = [
            -0.1, -1.13, 0.55, 0.78, 0.38, 1.01, -0.1, -1.89, -0.06, -0.08, -0.5, -0.71, 0.85,
            2.24, -0.31, 0.25, -0.33, -0.69, 1.85, 0.22, -0.38, -0.03, -1.02, 1.03, 0.61, -1.02,
            -0.69, -0.04, -1.07, -0.28, -0.42, 2.8, -0.21, 1.37, 0.3, -1.22, 1.12, 0.84, -1.27,
            -0.23, 0.45, 0.18, -1.61, -1.43, 2.17, 1.52, 0.05, -0.14, -1.69, 0.28, 1.21, -1.55,
            0.92, 0.01, -0.62, 0.48, -1.12, 0.58, 2.05, 0.39, 0.08, 1.92, -0.4, 0.12, -0.49, -0.37,
            -0.41, 1.01, -0.47, 0.22, -2.31, 0.64,
        ];
        let x2 = [
            -0.89, 1.51, -0.38, 0.12, 0.04, 0.06, 1.26, 0.92, -0.95, -0.42, 2.97, 1.63, -0.58,
            -0.49, -2.05, 0.52, -0.94, 0.61, -0.26, 1.13, -0.17, -0.56, -1.62, 1.84, -1.46, 3.21,
            2.31, -0.06, -0.65, 0.74, 1.55, 0.31, 0.52, 0.24, 1.91, -0.97, -1.87, 0.3, 0.08, -1.44,
            -0.59, -1.78, 0.11, -0.96, 0.7, -1.16, -0.58, -0.64, -0.07, -0.26, 2.83, 0.68, -0.85,
            1.44, -1.57, -0.17, -1.24, -1.02, 0.45, -1.51,
        ];
        let x3 = [
            -2.05, 0.11, 0.52, -1.06, -0.72, -0.56, 0.37, 1.26, -0.21, 2.22, -0.22, -0.35, 0.74,
            -0.32, 0.07, -0.87, 2.2, 1.16, -0.03, -0.24, 0.09, 0.32, 2.13, -0.2, 1.16, -0.89,
            -0.74, 1.11, -0.24, 2.23, -1.1, 0.08, -0.22, -0.97, 0.04, 0.75, -1.36, -0.06, 0.31,
            -0.5, 0.86, 1.62, 0.44, 0.21, 0.01, -0.2, -0.46, -0.32,
        ];
        (
            Mat::from_fn(12, 6, |i, j| x1[i * 6 + j]),
            Mat::from_fn(12, 5, |i, j| x2[i * 5 + j]),
            Mat::from_fn(12, 4, |i, j| x3[i * 4 + j]),
        )
    }

    /// `sqrt(ncol(x1)) / 2` applied to every block, as DIALOGUE does.
    const PMA_PENALTY: f64 = 1.224_744_871_391_589;

    /// Parameters matching the reference `PMA::MultiCCA` call.
    fn pma_params() -> MultiCcaParams<f64> {
        MultiCcaParams {
            n_components: 2,
            n_iter: 100,
            tol: 1e-3,
            penalties: Some(vec![PMA_PENALTY; 3]),
            standardise: true,
        }
    }

    /// The reference weights from `PMA::MultiCCA`, row-major per block.
    fn pma_expected() -> [Vec<f64>; 3] {
        [
            vec![
                0.0,
                0.0,
                -0.965_925_812_666_518_6,
                0.0,
                0.0,
                0.0,
                0.0,
                0.965_925_840_039_351_2,
                0.0,
                -0.258_818_993_785_760_6,
                0.25881909594256314,
                0.0,
            ],
            vec![
                -0.965_925_811_281_016_2,
                0.0,
                -0.258_819_101_113_327,
                0.0,
                0.0,
                -0.1398497689827376,
                0.0,
                0.099_760_763_370_438_63,
                0.0,
                -0.985_134_423_420_084_6,
            ],
            vec![
                -0.12522129863695705,
                0.0,
                -0.1139615511233123,
                0.965_925_827_364_77,
                -0.985_561_967_221_362_9,
                0.0,
                0.0,
                -0.258_819_041_087_947,
            ],
        ]
    }

    /// Weights against `PMA::MultiCCA` on the same data.
    ///
    /// Compared on magnitude, not signed value. The objective is invariant only
    /// under flipping every block at once, and the SVD that seeds the ascent
    /// has no shared sign convention between LAPACK and `faer`, so a signed
    /// comparison would be testing the singular value decomposition rather than
    /// the ascent.
    #[test]
    fn test_multi_cca_matches_pma() {
        let (x1, x2, x3) = pma_fixture();
        let blocks = [x1.as_ref(), x2.as_ref(), x3.as_ref()];
        let res = multi_cca(&blocks, Some(pma_params())).unwrap();

        for (b, expected) in pma_expected().iter().enumerate() {
            let w = &res.ws[b];
            for c in 0..2 {
                for r in 0..w.nrows() {
                    let got = w[(r, c)];
                    let want = expected[r * 2 + c];
                    assert_relative_eq!(got.abs(), want.abs(), epsilon = 1e-7);
                    // The active set must match exactly, and the zeros must be
                    // exact rather than merely small: DIALOGUE counts non-zero
                    // weights to decide which features enter a programme.
                    if want == 0.0 {
                        assert_eq!(got, 0.0, "block {b} entry ({r}, {c}) should be zero");
                    } else {
                        assert_ne!(got, 0.0, "block {b} entry ({r}, {c}) should be non-zero");
                    }
                }
            }
        }
    }

    /// The summed pairwise correlations `PMA` reports per component.
    #[test]
    fn test_multi_cca_reports_pma_correlations() {
        let (x1, x2, x3) = pma_fixture();
        let blocks = [x1.as_ref(), x2.as_ref(), x3.as_ref()];
        let res = multi_cca(&blocks, Some(pma_params())).unwrap();
        assert_relative_eq!(res.cors[0].abs(), 1.8951346746976176, epsilon = 1e-7);
        assert_relative_eq!(res.cors[1].abs(), 1.6787391871183392, epsilon = 1e-7);
    }

    /// The L1 bound binds: with `sqrt(p) / 2` the solution is 2-sparse, and the
    /// surviving pair sits at `(cos 15, sin 15)` because that is the only unit
    /// vector with two non-zeros whose L1 norm is `sqrt(6) / 2`.
    #[test]
    fn test_multi_cca_respects_the_l1_budget() {
        let (x1, x2, x3) = pma_fixture();
        let blocks = [x1.as_ref(), x2.as_ref(), x3.as_ref()];
        let res = multi_cca(&blocks, Some(pma_params())).unwrap();

        for w in res.ws.iter() {
            for c in 0..2 {
                let col: Vec<f64> = (0..w.nrows()).map(|r| w[(r, c)]).collect();
                let l2: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
                let l1: f64 = col.iter().map(|v| v.abs()).sum();
                assert_relative_eq!(l2, 1.0, epsilon = 1e-9);
                assert!(l1 <= PMA_PENALTY + 1e-6, "L1 norm {l1} exceeded the budget");
            }
        }
    }

    /// A loose bound leaves the constraint slack, so nothing is thresholded
    /// away and every weight stays non-zero.
    #[test]
    fn test_multi_cca_slack_penalty_keeps_every_feature() {
        let (x1, x2, x3) = pma_fixture();
        let blocks = [x1.as_ref(), x2.as_ref(), x3.as_ref()];
        let params = MultiCcaParams {
            n_components: 1,
            n_iter: 100,
            tol: 1e-3,
            // sqrt(p_k) is the largest admissible bound and never binds.
            penalties: Some(vec![6f64.sqrt(), 5f64.sqrt(), 4f64.sqrt()]),
            standardise: true,
        };
        let res = multi_cca(&blocks, Some(params)).unwrap();
        for w in res.ws.iter() {
            let zeros = (0..w.nrows()).filter(|&r| w[(r, 0)] == 0.0).count();
            assert_eq!(zeros, 0);
        }
    }

    #[test]
    fn test_multi_cca_rejects_bad_input() {
        let (x1, x2, x3) = pma_fixture();
        // Fewer than two blocks.
        assert!(multi_cca(&[x1.as_ref()], None).is_err());
        // Mismatched row counts.
        let short = Mat::<f64>::zeros(11, 4);
        assert!(multi_cca(&[x1.as_ref(), short.as_ref()], None).is_err());
        // A single-column block.
        let thin = Mat::<f64>::zeros(12, 1);
        assert!(multi_cca(&[x1.as_ref(), thin.as_ref()], None).is_err());
        // An L1 bound below one leaves the constraint set empty, and one above
        // sqrt(p_k) can never bind.
        let blocks = [x1.as_ref(), x2.as_ref(), x3.as_ref()];
        let too_small = MultiCcaParams {
            penalties: Some(vec![0.5; 3]),
            ..MultiCcaParams::default()
        };
        assert!(multi_cca(&blocks, Some(too_small)).is_err());
        let too_large = MultiCcaParams {
            penalties: Some(vec![10.0; 3]),
            ..MultiCcaParams::default()
        };
        assert!(multi_cca(&blocks, Some(too_large)).is_err());
    }

    /// Tuning prefers a bound that separates the real fit from its permutation
    /// null, and always returns a point from the grid it searched.
    #[test]
    fn test_multi_cca_permute_picks_from_the_grid() {
        let (x1, x2, x3) = pma_fixture();
        let blocks = [x1.as_ref(), x2.as_ref(), x3.as_ref()];
        let res = multi_cca_permute(&blocks, None).unwrap();

        // Every grid point must stay inside [1, sqrt(p_k)] or multi_cca would
        // have rejected it on the way through.
        for point in res.grid.iter() {
            for (p, upper) in point.iter().zip([6f64.sqrt(), 5f64.sqrt(), 4f64.sqrt()]) {
                assert!(*p >= 1.0 && *p <= upper);
            }
        }
        for p in res.p_values.iter() {
            assert!((0.0..=1.0).contains(p));
        }
    }

    /// A planted shared factor is recovered: the leading component's weight
    /// lands on the loaded features rather than the noise ones.
    #[test]
    fn test_multi_cca_recovers_a_planted_factor() {
        use rand::Rng;
        use rand_distr::StandardNormal;

        let n = 40;
        let mut rng = StdRng::seed_from_u64(7);
        let latent: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
        // Two loaded features, six of pure noise, in each of three blocks. The
        // noise has to be genuinely independent across blocks: anything with
        // shared structure is a real cross-block signal and multi-CCA is
        // entitled to find it instead.
        let mut build = |load: f64| {
            let draws: Vec<f64> = (0..n * 8).map(|_| rng.sample(StandardNormal)).collect();
            Mat::<f64>::from_fn(n, 8, |i, j| {
                let base = draws[i * 8 + j];
                if j < 2 { base + load * latent[i] } else { base }
            })
        };
        let b1 = build(2.0);
        let b2 = build(2.0);
        let b3 = build(2.0);
        let blocks = [b1.as_ref(), b2.as_ref(), b3.as_ref()];

        let params = MultiCcaParams {
            n_components: 1,
            ..MultiCcaParams::default()
        };
        let res = multi_cca(&blocks, Some(params)).unwrap();

        for (b, w) in res.ws.iter().enumerate() {
            let loaded: f64 = (0..2).map(|r| w[(r, 0)].abs()).sum();
            let rest: f64 = (2..8).map(|r| w[(r, 0)].abs()).sum();
            assert!(
                loaded > rest,
                "block {b}: loaded weight {loaded} did not beat the noise weight {rest}"
            );
        }
    }
}
