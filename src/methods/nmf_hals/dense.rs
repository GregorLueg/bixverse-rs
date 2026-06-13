//! Dense backend for NMF. Borrows V as a faer `MatRef` and routes both data
//! products through `faer::linalg::matmul`.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef};

use super::NmfInput;
use crate::core::math::pca_svd::{SvdResults, randomised_svd};
use crate::prelude::*;
use crate::utils::faer_parallelism;

/// Dense, non-negative V wrapped behind the NMF trait. Validates non-negativity
/// and finiteness once at construction.
#[derive(Debug)]
pub struct DenseInput<'a, F: BixverseFloat> {
    /// The input matrix v
    v: MatRef<'a, F>,
    /// The squared Frobenius norm
    sq_frob: F,
}

impl<'a, F: BixverseFloat> DenseInput<'a, F> {
    /// Construct a [`DenseInput`] from a borrowed matrix.
    ///
    /// Validates that all entries are finite and non-negative, then computes
    /// `||V||_F^2`.
    ///
    /// ### Params
    ///
    /// * `v` - The input matrix to factorise.
    ///
    /// ### Returns
    ///
    /// A [`DenseInput`] wrapping `v`.
    pub fn new(v: MatRef<'a, F>) -> Result<Self, BixverseErrors> {
        let (n, m) = (v.nrows(), v.ncols());
        let mut sq_frob = F::zero();
        for i in 0..n {
            for j in 0..m {
                let x = v[(i, j)];
                if !x.is_finite() {
                    return Err(BixverseErrors::NmfNonFinite);
                }
                if x < F::zero() {
                    return Err(BixverseErrors::NmfNonNegativeViolated);
                }
                sq_frob += x * x;
            }
        }
        Ok(Self { v, sq_frob })
    }
}

/// NmfInput trait implementation for [DenseInput]
impl<F: BixverseFloat> NmfInput<F> for DenseInput<'_, F> {
    fn shape(&self) -> (usize, usize) {
        (self.v.nrows(), self.v.ncols())
    }

    fn sq_frob(&self) -> F {
        self.sq_frob
    }

    fn wt_v(&self, w: MatRef<F>, out: &mut Mat<F>) {
        matmul(
            out.as_mut(),
            Accum::Replace,
            w.transpose(),
            self.v,
            F::one(),
            faer_parallelism(),
        );
    }

    fn v_ht(&self, h: MatRef<F>, out: &mut Mat<F>) {
        matmul(
            out.as_mut(),
            Accum::Replace,
            self.v,
            h.transpose(),
            F::one(),
            faer_parallelism(),
        );
    }

    fn top_k_svd(&self, k: usize) -> Result<SvdResults<F>, BixverseErrors> {
        let (n, m) = (self.v.nrows(), self.v.ncols());
        let min_dim = m.min(n);

        if min_dim < k {
            return Err(BixverseErrors::NmfRankTooLarge {
                requested: k,
                available: min_dim,
            });
        }

        // randomised when k is small relative to min(m, n) and the matrix is
        // large enough that the projection pays off; otherwise thin_svd.
        let use_randomised = k * 4 < min_dim && min_dim > 64;

        if use_randomised {
            let rsvd = randomised_svd(self.v, k, 42, None, None)?;
            // randomised_svd returns rank + oversampling triplets; truncate to k.
            let u = Mat::<F>::from_fn(rsvd.u.nrows(), k, |i, j| rsvd.u[(i, j)]);
            let v = Mat::<F>::from_fn(rsvd.v.nrows(), k, |i, j| rsvd.v[(i, j)]);
            let s: Vec<F> = rsvd.s.into_iter().take(k).collect();
            Ok(SvdResults { u, v, s })
        } else {
            let svd = self
                .v
                .thin_svd()
                .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;

            let u_full = svd.U();
            let v_full = svd.V();
            let s_full = svd.S().column_vector();

            let u = Mat::<F>::from_fn(u_full.nrows(), k, |i, j| u_full[(i, j)]);
            let v = Mat::<F>::from_fn(v_full.nrows(), k, |i, j| v_full[(i, j)]);
            let s: Vec<F> = (0..k).map(|i| s_full[i]).collect();

            Ok(SvdResults { u, v, s })
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn shape_and_sq_frob() {
        let v: Mat<f64> = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
        let input = DenseInput::new(v.as_ref()).unwrap();
        assert_eq!(input.shape(), (3, 2));
        // entries: 0,1,2,1,2,3 -> squares sum = 0+1+4+1+4+9 = 19
        assert!((input.sq_frob() - 19.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_negative() {
        let v: Mat<f64> = Mat::from_fn(2, 2, |i, _| if i == 0 { 1.0 } else { -1.0 });
        let err = DenseInput::new(v.as_ref()).unwrap_err();
        assert!(matches!(err, BixverseErrors::NmfNonNegativeViolated));
    }

    #[test]
    fn rejects_non_finite() {
        let v: Mat<f64> = Mat::from_fn(2, 2, |i, _| if i == 0 { 1.0 } else { f64::NAN });
        let err = DenseInput::new(v.as_ref()).unwrap_err();
        assert!(matches!(err, BixverseErrors::NmfNonFinite));

        let v: Mat<f64> = Mat::from_fn(2, 2, |i, _| if i == 0 { 1.0 } else { f64::INFINITY });
        let err = DenseInput::new(v.as_ref()).unwrap_err();
        assert!(matches!(err, BixverseErrors::NmfNonFinite));
    }

    #[test]
    fn wt_v_matches_reference() {
        let (m, n, k) = (4, 3, 2);
        let v_mat: Mat<f64> = Mat::from_fn(m, n, |i, j| (i * n + j + 1) as f64);
        let w: Mat<f64> = Mat::from_fn(m, k, |i, j| (i + j + 1) as f64);
        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let mut out = Mat::<f64>::zeros(k, n);
        input.wt_v(w.as_ref(), &mut out);
        for r in 0..k {
            for j in 0..n {
                let expected: f64 = (0..m).map(|i| w[(i, r)] * v_mat[(i, j)]).sum();
                assert!((out[(r, j)] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn v_ht_matches_reference() {
        let (m, n, k) = (4, 3, 2);
        let v_mat: Mat<f64> = Mat::from_fn(m, n, |i, j| (i * n + j + 1) as f64);
        let h: Mat<f64> = Mat::from_fn(k, n, |i, j| (i + j + 1) as f64);
        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let mut out = Mat::<f64>::zeros(m, k);
        input.v_ht(h.as_ref(), &mut out);
        for i in 0..m {
            for r in 0..k {
                let expected: f64 = (0..n).map(|j| v_mat[(i, j)] * h[(r, j)]).sum();
                assert!((out[(i, r)] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn top_k_svd_shapes_and_ordering() {
        let v_mat: Mat<f64> = Mat::from_fn(5, 4, |i, j| (i * 4 + j + 1) as f64);
        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let svd = input.top_k_svd(2).unwrap();
        assert_eq!(svd.u.shape(), (5, 2));
        assert_eq!(svd.v.shape(), (4, 2));
        assert_eq!(svd.s.len(), 2);
        assert!(svd.s[0] >= svd.s[1]);
        assert!(svd.s[1] >= 0.0);
    }

    #[test]
    fn top_k_svd_rejects_too_large_k() {
        let v_mat: Mat<f64> = Mat::from_fn(3, 2, |i, j| (i + j + 1) as f64);
        let input = DenseInput::new(v_mat.as_ref()).unwrap();
        let err = input.top_k_svd(3).unwrap_err();
        assert!(matches!(err, BixverseErrors::NmfRankTooLarge { .. }));
    }
}
