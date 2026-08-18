//! Fixed-factor NNLS refits. Solves for one factor while the other is frozen,
//! which is what a consensus NMF needs once the consensus factor has been
//! assembled from the clustered restarts.
//!
//! Freezing a factor makes both of its data products constant, so `W^T V` (or
//! `V H^T`) is computed once up front rather than once per iteration. That is
//! the expensive part of a HALS sweep, so a refit is far cheaper than a full
//! solve of the same iteration count.

use faer::{Mat, MatRef};
use rayon::prelude::*;

use super::{
    HalsOpts, NmfInput, compute_objective, gram_h_ht, gram_wt_w, hals_sweep_cols, hals_sweep_rows,
};
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Denominator for the relative-decrease convergence test.
///
/// Matches [`super::nmf_hals`]: `||V||_F^2` floored at one, so `tol` reads as a
/// fraction of the total energy and tiny-norm inputs do not divide by a value
/// below one.
///
/// ### Params
///
/// * `sq_frob` - The squared Frobenius norm of V.
///
/// ### Returns
///
/// The floored denominator.
#[inline]
fn relative_denominator<F: BixverseFloat>(sq_frob: F) -> F {
    if sq_frob > F::one() {
        sq_frob
    } else {
        F::one()
    }
}

/// Reconstruction error with H frozen.
///
/// Evaluates `||V - WH||_F^2` as
/// `||V||_F^2 - 2<W, V H^T> + <W^T W, H H^T>`. The middle term uses the
/// precomputed `V H^T` rather than `W^T V`, so a changing W costs no extra pass
/// over V. Accumulated in `f64` regardless of `F`, matching
/// [`super::compute_objective`], because the two large terms nearly cancel.
///
/// ### Params
///
/// * `sq_frob_v` - Precomputed `||V||_F^2`.
/// * `w` - Left factor `m x k`.
/// * `vht` - Product `V H^T`, shape `m x k`.
/// * `wtw` - Gram matrix `W^T W`, shape `k x k`.
/// * `hht` - Gram matrix `H H^T`, shape `k x k`.
///
/// ### Returns
///
/// The scalar reconstruction error, clamped at zero.
fn objective_fixed_h<F: BixverseFloat + Send + Sync>(
    sq_frob_v: F,
    w: MatRef<F>,
    vht: MatRef<F>,
    wtw: MatRef<F>,
    hht: MatRef<F>,
) -> F {
    let m = w.nrows();
    let k = w.ncols();

    let inner_w_vht: f64 = (0..m)
        .into_par_iter()
        .with_min_len(64)
        .map(|i| {
            let mut acc = 0f64;
            for r in 0..k {
                acc += w[(i, r)].to_f64().unwrap() * vht[(i, r)].to_f64().unwrap();
            }
            acc
        })
        .reduce(|| 0f64, |x, y| x + y);

    let mut inner_grams = 0f64;
    for j in 0..k {
        for i in 0..k {
            inner_grams += wtw[(i, j)].to_f64().unwrap() * hht[(i, j)].to_f64().unwrap();
        }
    }

    let result = sq_frob_v.to_f64().unwrap() - 2.0 * inner_w_vht + inner_grams;
    F::from_f64(result.max(0.0)).unwrap()
}

////////////
// Refits //
////////////

/// Refit H against a frozen W.
///
/// Initialises H at `opts.eps` and runs HALS row sweeps until the relative
/// objective decrease drops below `opts.tol` or `opts.max_iter` is reached.
/// `W^T W` and `W^T V` are loop invariants here, so only `H H^T` is recomputed
/// when the objective is evaluated. No column normalisation is applied: W is
/// frozen, so its scale convention is the caller's.
///
/// ### Params
///
/// * `v` - Input matrix backend.
/// * `w` - Frozen left factor, shape `m x k`. Must match `v`'s row count.
/// * `opts` - Solver options. The `init` field is ignored.
///
/// ### Returns
///
/// Tuple of `(H of shape k x n, final ||V - WH||_F^2)`, or
/// [`BixverseErrors::NmfDimensionMismatch`] if `w` does not match `v`.
pub fn nmf_refit_h<F, In>(
    v: &In,
    w: MatRef<F>,
    opts: &HalsOpts<F>,
) -> Result<(Mat<F>, F), BixverseErrors>
where
    F: BixverseFloat + Send + Sync,
    In: NmfInput<F> + Sync,
{
    let (m, n) = v.shape();
    if w.nrows() != m {
        return Err(BixverseErrors::NmfDimensionMismatch {
            expected: m,
            found: w.nrows(),
        });
    }
    let k = w.ncols();
    let sq_frob = v.sq_frob();
    let denom = relative_denominator(sq_frob);

    // Both loop invariants, because W never changes.
    let mut wtw = Mat::<F>::zeros(k, k);
    gram_wt_w(w, &mut wtw);
    let mut wtv = Mat::<F>::zeros(k, n);
    v.wt_v(w, &mut wtv);

    let mut h = Mat::<F>::full(k, n, opts.eps);
    let mut hht = Mat::<F>::zeros(k, k);

    let mut final_loss = F::infinity();
    let mut last_loss = F::infinity();
    let mut n_iter = 0usize;

    for iter in 0..opts.max_iter {
        n_iter = iter + 1;
        hals_sweep_rows(&mut h, wtw.as_ref(), wtv.as_ref(), opts.eps);

        if n_iter.is_multiple_of(opts.check_every) {
            gram_h_ht(h.as_ref(), &mut hht);
            let loss = compute_objective(
                sq_frob,
                h.as_ref(),
                wtv.as_ref(),
                wtw.as_ref(),
                hht.as_ref(),
            );
            final_loss = loss;

            if n_iter > opts.check_every && (last_loss - loss).abs() / denom < opts.tol {
                break;
            }
            last_loss = loss;
        }
    }

    // `n_iter == 0` when `max_iter` is 0, and zero is a multiple of everything,
    // so without the first clause the loss would be returned as infinity.
    if n_iter == 0 || !n_iter.is_multiple_of(opts.check_every) {
        gram_h_ht(h.as_ref(), &mut hht);
        final_loss = compute_objective(
            sq_frob,
            h.as_ref(),
            wtv.as_ref(),
            wtw.as_ref(),
            hht.as_ref(),
        );
    }

    Ok((h, final_loss))
}

/// Refit W against a frozen H.
///
/// Mirror of [`nmf_refit_h`]. `H H^T` and `V H^T` are the loop invariants, and
/// the objective goes through [`objective_fixed_h`] so that a changing W never
/// forces another pass over V. No row normalisation is applied: H is frozen, so
/// its scale convention is the caller's.
///
/// ### Params
///
/// * `v` - Input matrix backend.
/// * `h` - Frozen right factor, shape `k x n`. Must match `v`'s column count.
/// * `opts` - Solver options. The `init` field is ignored.
///
/// ### Returns
///
/// Tuple of `(W of shape m x k, final ||V - WH||_F^2)`, or
/// [`BixverseErrors::NmfDimensionMismatch`] if `h` does not match `v`.
pub fn nmf_refit_w<F, In>(
    v: &In,
    h: MatRef<F>,
    opts: &HalsOpts<F>,
) -> Result<(Mat<F>, F), BixverseErrors>
where
    F: BixverseFloat + Send + Sync,
    In: NmfInput<F> + Sync,
{
    let (m, n) = v.shape();
    if h.ncols() != n {
        return Err(BixverseErrors::NmfDimensionMismatch {
            expected: n,
            found: h.ncols(),
        });
    }
    let k = h.nrows();
    let sq_frob = v.sq_frob();
    let denom = relative_denominator(sq_frob);

    // Both loop invariants, because H never changes.
    let mut hht = Mat::<F>::zeros(k, k);
    gram_h_ht(h, &mut hht);
    let mut vht = Mat::<F>::zeros(m, k);
    v.v_ht(h, &mut vht);

    let mut w = Mat::<F>::full(m, k, opts.eps);
    let mut wtw = Mat::<F>::zeros(k, k);

    let mut final_loss = F::infinity();
    let mut last_loss = F::infinity();
    let mut n_iter = 0usize;

    for iter in 0..opts.max_iter {
        n_iter = iter + 1;
        hals_sweep_cols(&mut w, hht.as_ref(), vht.as_ref(), opts.eps);

        if n_iter.is_multiple_of(opts.check_every) {
            gram_wt_w(w.as_ref(), &mut wtw);
            let loss = objective_fixed_h(
                sq_frob,
                w.as_ref(),
                vht.as_ref(),
                wtw.as_ref(),
                hht.as_ref(),
            );
            final_loss = loss;

            if n_iter > opts.check_every && (last_loss - loss).abs() / denom < opts.tol {
                break;
            }
            last_loss = loss;
        }
    }

    // `n_iter == 0` when `max_iter` is 0, and zero is a multiple of everything,
    // so without the first clause the loss would be returned as infinity.
    if n_iter == 0 || !n_iter.is_multiple_of(opts.check_every) {
        gram_wt_w(w.as_ref(), &mut wtw);
        final_loss = objective_fixed_h(
            sq_frob,
            w.as_ref(),
            vht.as_ref(),
            wtw.as_ref(),
            hht.as_ref(),
        );
    }

    Ok((w, final_loss))
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

    /// Exact rank-k non-negative product plus its squared Frobenius norm.
    fn rank_k_product(m: usize, n: usize, k: usize) -> (Mat<f64>, Mat<f64>, Mat<f64>, f64) {
        let w_true: Mat<f64> = Mat::from_fn(m, k, |i, j| ((i * 7 + j * 3) % 5 + 1) as f64);
        let h_true: Mat<f64> = Mat::from_fn(k, n, |i, j| ((i * 5 + j * 2) % 4 + 1) as f64);
        let v: Mat<f64> = Mat::from_fn(m, n, |i, j| {
            (0..k).map(|r| w_true[(i, r)] * h_true[(r, j)]).sum()
        });
        let sq_frob = (0..m)
            .map(|i| (0..n).map(|j| v[(i, j)].powi(2)).sum::<f64>())
            .sum();
        (w_true, h_true, v, sq_frob)
    }

    /// Squared Frobenius reconstruction error of an explicit `W H` product.
    fn reconstruction_error(w: MatRef<f64>, h: MatRef<f64>, v: MatRef<f64>) -> f64 {
        let k = w.ncols();
        (0..v.nrows())
            .map(|i| {
                (0..v.ncols())
                    .map(|j| {
                        let wh: f64 = (0..k).map(|r| w[(i, r)] * h[(r, j)]).sum();
                        (v[(i, j)] - wh).powi(2)
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    /// With the true W frozen the system is exactly solvable, so the refit must
    /// drive the reconstruction to zero. Note the contract is the reconstruction,
    /// not recovery of `h_true`: the frozen factor need not have full rank, in
    /// which case many exact non-negative H exist.
    #[test]
    fn refit_h_reconstructs_exactly() {
        let (m, n, k) = (30, 20, 3);
        let (w_true, _, v, sq_frob) = rank_k_product(m, n, k);

        let input = DenseInput::new(v.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(600, 1e-12, 1e-12, 5, NmfInit::Nndsvd);
        let (h, loss) = nmf_refit_h(&input, w_true.as_ref(), &opts).unwrap();

        assert!(
            loss / sq_frob < 1e-8,
            "reported rel loss {}",
            loss / sq_frob
        );
        // The reported loss must agree with an explicit W H, not just be small.
        let direct = reconstruction_error(w_true.as_ref(), h.as_ref(), v.as_ref());
        assert!(
            direct / sq_frob < 1e-8,
            "direct rel loss {}",
            direct / sq_frob
        );
        assert!((0..k).all(|r| (0..n).all(|j| h[(r, j)] >= 0.0)));
    }

    /// Mirror of the above with the roles swapped.
    #[test]
    fn refit_w_reconstructs_exactly() {
        let (m, n, k) = (30, 20, 3);
        let (_, h_true, v, sq_frob) = rank_k_product(m, n, k);

        let input = DenseInput::new(v.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(600, 1e-12, 1e-12, 5, NmfInit::Nndsvd);
        let (w, loss) = nmf_refit_w(&input, h_true.as_ref(), &opts).unwrap();

        assert!(
            loss / sq_frob < 1e-8,
            "reported rel loss {}",
            loss / sq_frob
        );
        let direct = reconstruction_error(w.as_ref(), h_true.as_ref(), v.as_ref());
        assert!(
            direct / sq_frob < 1e-8,
            "direct rel loss {}",
            direct / sq_frob
        );
        assert!((0..m).all(|i| (0..k).all(|r| w[(i, r)] >= 0.0)));
    }

    /// When the frozen factor has full rank the NNLS solution is unique, so the
    /// true partner is recovered outright.
    #[test]
    fn refit_recovers_unique_partner() {
        // Full-rank 2 x 3 frozen H, so W is pinned by the normal equations.
        let h_true: Mat<f64> = Mat::from_fn(2, 3, |i, j| if i == j { 2.0 } else { 0.5 });
        let w_true: Mat<f64> = Mat::from_fn(5, 2, |i, j| (i + 2 * j + 1) as f64);
        let v: Mat<f64> = Mat::from_fn(5, 3, |i, j| {
            (0..2).map(|r| w_true[(i, r)] * h_true[(r, j)]).sum()
        });

        let input = DenseInput::new(v.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(2000, 1e-14, 1e-14, 5, NmfInit::Nndsvd);
        let (w, _) = nmf_refit_w(&input, h_true.as_ref(), &opts).unwrap();

        for i in 0..5 {
            for r in 0..2 {
                assert_relative_eq!(w[(i, r)], w_true[(i, r)], epsilon = 1e-6);
            }
        }
    }

    /// The objective must never increase across the refit, which is the property
    /// that makes the convergence test meaningful.
    #[test]
    fn refit_loss_is_monotone() {
        let (m, n, k) = (20, 15, 3);
        let (w_true, _, v, _) = rank_k_product(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();

        let mut previous = f64::INFINITY;
        for max_iter in [5, 10, 20, 40, 80] {
            let opts = HalsOpts::<f64>::new(max_iter, 0.0, 1e-12, 5, NmfInit::Nndsvd);
            let (_, loss) = nmf_refit_h(&input, w_true.as_ref(), &opts).unwrap();
            assert!(
                loss <= previous + 1e-9,
                "loss rose from {previous} to {loss} at max_iter {max_iter}"
            );
            previous = loss;
        }
    }

    /// The fixed-H objective shortcut must agree with the shared one, which goes
    /// through `W^T V` instead of `V H^T`.
    #[test]
    fn fixed_h_objective_matches_shared() {
        let (m, n, k) = (12, 9, 3);
        let (w_true, h_true, v, sq_frob) = rank_k_product(m, n, k);
        // Perturb W so the two terms do not cancel to exactly zero.
        let w: Mat<f64> = Mat::from_fn(m, k, |i, j| w_true[(i, j)] + 0.3 * (i + j) as f64);

        let input = DenseInput::new(v.as_ref()).unwrap();

        let mut wtw = Mat::<f64>::zeros(k, k);
        gram_wt_w(w.as_ref(), &mut wtw);
        let mut hht = Mat::<f64>::zeros(k, k);
        gram_h_ht(h_true.as_ref(), &mut hht);
        let mut wtv = Mat::<f64>::zeros(k, n);
        input.wt_v(w.as_ref(), &mut wtv);
        let mut vht = Mat::<f64>::zeros(m, k);
        input.v_ht(h_true.as_ref(), &mut vht);

        let shared = compute_objective(
            sq_frob,
            h_true.as_ref(),
            wtv.as_ref(),
            wtw.as_ref(),
            hht.as_ref(),
        );
        let fixed = objective_fixed_h(
            sq_frob,
            w.as_ref(),
            vht.as_ref(),
            wtw.as_ref(),
            hht.as_ref(),
        );

        assert_relative_eq!(shared, fixed, max_relative = 1e-10);
    }

    /// `max_iter = 0` runs no sweeps, but the loss must still be evaluated on the
    /// starting factor. Returning infinity would propagate into
    /// `ConsensusNmfResult::error`, and `HalsOpts` is reachable from R.
    #[test]
    fn refit_with_zero_iterations_reports_a_finite_loss() {
        let (m, n, k) = (12, 9, 2);
        let (w_true, h_true, v, _) = rank_k_product(m, n, k);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::new(0, 1e-9, 1e-12, 10, NmfInit::Nndsvd);

        let (h, loss_h) = nmf_refit_h(&input, w_true.as_ref(), &opts).unwrap();
        assert!(loss_h.is_finite(), "H refit loss {loss_h}");
        assert_eq!((h.nrows(), h.ncols()), (k, n));

        let (w, loss_w) = nmf_refit_w(&input, h_true.as_ref(), &opts).unwrap();
        assert!(loss_w.is_finite(), "W refit loss {loss_w}");
        assert_eq!((w.nrows(), w.ncols()), (m, k));
    }

    /// A factor whose shared dimension does not match V is a caller error and
    /// must not be silently truncated by the GEMMs.
    #[test]
    fn refit_rejects_dimension_mismatch() {
        let (_, _, v, _) = rank_k_product(10, 8, 2);
        let input = DenseInput::new(v.as_ref()).unwrap();
        let opts = HalsOpts::<f64>::default();

        let bad_w: Mat<f64> = Mat::full(7, 2, 1.0);
        assert!(nmf_refit_h(&input, bad_w.as_ref(), &opts).is_err());

        let bad_h: Mat<f64> = Mat::full(2, 5, 1.0);
        assert!(nmf_refit_w(&input, bad_h.as_ref(), &opts).is_err());
    }
}
