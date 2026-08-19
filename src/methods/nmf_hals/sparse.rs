//! Sparse backend for NMF. Holds the chosen layer of V as CSC and CSR with
//! storage type `S`; widens to compute type `F` per element inside two
//! hand-rolled spMM kernels.

use faer::{Mat, MatRef};
use num_traits::{Float, ToPrimitive};
use rayon::prelude::*;

use super::NmfInput;
use crate::core::math::pca_svd::SvdResults;
use crate::core::math::sparse::sparse_svd_lanczos;
use crate::core::math::vector_helpers::sum_sq_f64;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Cast and validate a slice of values for NMF input.
///
/// Casts each element from `T` to `S`, then checks finiteness and
/// non-negativity.
///
/// ### Params
///
/// * `src` - Source slice to cast and validate.
///
/// ### Returns
///
/// A validated `Vec<S>`, or [`BixverseErrors::NmfNonFinite`] if any cast
/// value is non-finite, or [`BixverseErrors::NmfNonNegativeViolated`] if
/// any cast value is negative.
fn cast_and_validate<T, S>(src: &[T]) -> Result<Vec<S>, BixverseErrors>
where
    T: Copy + Into<S>,
    S: Float,
{
    let mut out = Vec::with_capacity(src.len());
    for &v in src {
        let s: S = v.into();
        if !s.is_finite() {
            return Err(BixverseErrors::NmfNonFinite);
        }
        if s < S::zero() {
            return Err(BixverseErrors::NmfNonNegativeViolated);
        }
        out.push(s);
    }
    Ok(out)
}

/// Squared Frobenius norm of a flat data buffer.
///
/// Accumulates in `f64` via [`sum_sq_f64`] and casts once, rather than summing in
/// `F`. Over 3e7 non-zeros the `F = f32` version measured 4.4% low, and that error
/// lands directly in every relative loss the solver reports.
///
/// ### Params
///
/// * `data` - Flat non-zero value buffer.
///
/// ### Returns
///
/// The scalar `||V||_F^2`.
fn compute_sq_frob<S, F>(data: &[S]) -> F
where
    S: BixverseNumeric + ToPrimitive,
    F: BixverseFloat,
{
    F::from_f64(sum_sq_f64(data)).unwrap()
}

/// Build a matched (CSR, CSC) pair from a source matrix and its transpose.
///
/// Branches on the source orientation so the returned tuple is always
/// `(csr, csc)` regardless of which was passed as `source`.
///
/// ### Params
///
/// * `source` - The primary sparse matrix.
/// * `other` - Its transpose in the opposite format.
/// * `source_cast` - Cast non-zero values for `source`.
/// * `other_cast` - Cast non-zero values for `other`.
/// * `shape` - Logical shape `(m, n)` of the matrix.
///
/// ### Returns
///
/// A `(csr, csc)` pair with indices and indptr borrowed from the inputs.
fn build_pair<S, T, U>(
    source: &CompressedSparseData2<T, U>,
    other: CompressedSparseData2<T, U>,
    source_cast: Vec<S>,
    other_cast: Vec<S>,
    shape: (usize, usize),
) -> (CompressedSparseData2<S>, CompressedSparseData2<S>)
where
    S: Clone,
    T: Clone,
    U: Clone,
{
    match source.cs_type {
        CompressedSparseFormat::Csr => {
            let csr = CompressedSparseData2 {
                data: source_cast,
                indices: source.indices.clone(),
                indptr: source.indptr.clone(),
                cs_type: CompressedSparseFormat::Csr,
                data_2: None,
                shape,
            };
            let csc = CompressedSparseData2 {
                data: other_cast,
                indices: other.indices,
                indptr: other.indptr,
                cs_type: CompressedSparseFormat::Csc,
                data_2: None,
                shape,
            };
            (csr, csc)
        }
        CompressedSparseFormat::Csc => {
            let csc = CompressedSparseData2 {
                data: source_cast,
                indices: source.indices.clone(),
                indptr: source.indptr.clone(),
                cs_type: CompressedSparseFormat::Csc,
                data_2: None,
                shape,
            };
            let csr = CompressedSparseData2 {
                data: other_cast,
                indices: other.indices,
                indptr: other.indptr,
                cs_type: CompressedSparseFormat::Csr,
                data_2: None,
                shape,
            };
            (csr, csc)
        }
    }
}

/// Sparse W^T V via CSC.
///
/// Computes `out = W^T V` using the CSC layout of V, parallelising over output
/// columns.
///
/// ### Params
///
/// * `csc` - CSC representation of V, shape `m x n`.
/// * `w` - Left factor, shape `m x k`.
/// * `out` - Output buffer `k x n`. Overwritten on return.
fn spmm_csc_wt_v<S, F>(csc: &CompressedSparseData2<S>, w: MatRef<F>, out: &mut Mat<F>)
where
    S: BixverseNumeric + Into<F> + Sync,
    F: BixverseFloat + Send + Sync,
{
    let n = csc.shape.1;
    let k = w.ncols();

    let out_ptr = out.as_ptr_mut() as usize;
    let out_row_stride = out.row_stride();
    let out_col_stride = out.col_stride();

    (0..n).into_par_iter().for_each_init(
        || vec![F::zero(); k],
        |acc, j| {
            acc.fill(F::zero());

            let start = csc.indptr[j] as usize;
            let end = csc.indptr[j + 1] as usize;
            for idx in start..end {
                let i = csc.indices[idx] as usize;
                let val_f: F = csc.data[idx].into();
                for r in 0..k {
                    acc[r] += val_f * w[(i, r)];
                }
            }

            let base = out_ptr as *mut F;
            for r in 0..k {
                unsafe {
                    let ptr =
                        base.offset(r as isize * out_row_stride + j as isize * out_col_stride);
                    *ptr = acc[r];
                }
            }
        },
    );
}

/// Sparse V H^T via CSR.
///
/// Computes `out = V H^T` using the CSR layout of V, parallelising over output
/// rows.
///
/// ### Params
///
/// * `csr` - CSR representation of V, shape `m x n`.
/// * `h` - Right factor, shape `k x n`.
/// * `out` - Output buffer `m x k`. Overwritten on return.
fn spmm_csr_v_ht<S, F>(csr: &CompressedSparseData2<S>, h: MatRef<F>, out: &mut Mat<F>)
where
    S: BixverseNumeric + Into<F> + Sync,
    F: BixverseFloat + Send + Sync,
{
    let m = csr.shape.0;
    let k = h.nrows();

    let out_ptr = out.as_ptr_mut() as usize;
    let out_row_stride = out.row_stride();
    let out_col_stride = out.col_stride();

    (0..m).into_par_iter().for_each_init(
        || vec![F::zero(); k],
        |acc, i| {
            acc.fill(F::zero());

            let start = csr.indptr[i] as usize;
            let end = csr.indptr[i + 1] as usize;
            for idx in start..end {
                let j = csr.indices[idx] as usize;
                let val_f: F = csr.data[idx].into();
                for r in 0..k {
                    acc[r] += val_f * h[(r, j)];
                }
            }

            let base = out_ptr as *mut F;
            for r in 0..k {
                unsafe {
                    let ptr =
                        base.offset(i as isize * out_row_stride + r as isize * out_col_stride);
                    *ptr = acc[r];
                }
            }
        },
    );
}

//////////
// Main //
//////////

/// Sparse NMF input backed by a matched (CSR, CSC) pair.
///
/// Validates non-negativity and finiteness once at construction, then computes
/// and caches `||V||_F^2`.
#[derive(Debug)]
pub struct SparseInput<S, F>
where
    S: BixverseNumeric + Into<F> + Float,
    F: BixverseFloat,
{
    /// CSR layout of V, used for `V H^T`.
    csr: CompressedSparseData2<S>,
    /// CSC layout of V, used for `W^T V`.
    csc: CompressedSparseData2<S>,
    /// Cached `||V||_F^2`.
    sq_frob: F,
    /// Logical shape `(m, n)` of V.
    shape: (usize, usize),
}

impl<S, F> SparseInput<S, F>
where
    S: BixverseNumeric + BixverseSimd + Into<F> + Float + Clone,
    F: BixverseFloat + BixverseSimd + std::iter::Sum,
{
    /// Construct a [`SparseInput`] from the primary data layer of `matrix`.
    ///
    /// Casts and validates `data`, derives the transpose, and builds the
    /// matched (CSR, CSC) pair.
    ///
    /// ### Params
    ///
    /// * `matrix` - Source sparse matrix whose `data` field is used.
    ///
    /// ### Returns
    ///
    /// A [`SparseInput`], or [`BixverseErrors::NmfNonFinite`] /
    /// [`BixverseErrors::NmfNonNegativeViolated`] if validation fails.
    pub fn from_primary<T, U>(matrix: &CompressedSparseData2<T, U>) -> Result<Self, BixverseErrors>
    where
        T: BixverseNumeric + Into<S>,
        U: BixverseNumeric,
    {
        let shape = matrix.shape;
        let other = matrix.transform_single_layer(false)?;

        let primary: Vec<S> = cast_and_validate::<T, S>(&matrix.data)?;
        let secondary: Vec<S> = other.data.iter().map(|&v| v.into()).collect();

        let (csr, csc) = build_pair(matrix, other, primary, secondary, shape);
        let sq_frob = compute_sq_frob::<S, F>(&csr.data);

        Ok(Self {
            csr,
            csc,
            sq_frob,
            shape,
        })
    }

    /// Construct a [`SparseInput`] from the secondary data layer of `matrix`.
    ///
    /// Casts and validates `data_2`, derives the transpose, and builds the
    /// matched (CSR, CSC) pair. Intended for the log1p-normalised layer in the
    /// single-cell case.
    ///
    /// ### Params
    ///
    /// * `matrix` - Source sparse matrix whose `data_2` field is used.
    ///
    /// ### Returns
    ///
    /// A [`SparseInput`], or [`BixverseErrors::Data2NotAvailable`] if
    /// `data_2` is absent, or [`BixverseErrors::NmfNonFinite`] /
    /// [`BixverseErrors::NmfNonNegativeViolated`] if validation fails.
    pub fn from_secondary<T, U>(
        matrix: &CompressedSparseData2<T, U>,
    ) -> Result<Self, BixverseErrors>
    where
        T: BixverseNumeric,
        U: BixverseNumeric + Into<S>,
    {
        let shape = matrix.shape;
        let other = matrix.transform_single_layer(true)?;

        let m_data2 = matrix
            .data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?;
        let o_data2 = other
            .data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?;

        let primary: Vec<S> = cast_and_validate::<U, S>(m_data2)?;
        let secondary: Vec<S> = o_data2.iter().map(|&v| v.into()).collect();

        let (csr, csc) = build_pair(matrix, other, primary, secondary, shape);
        let sq_frob = compute_sq_frob::<S, F>(&csr.data);

        Ok(Self {
            csr,
            csc,
            sq_frob,
            shape,
        })
    }

    /// The CSR layout of V, used for `V H^T`.
    ///
    /// Exposed so a GPU backend can upload the already-validated matched pair
    /// rather than rebuilding and revalidating it.
    ///
    /// ### Returns
    ///
    /// A reference to the CSR layer.
    pub fn csr(&self) -> &CompressedSparseData2<S> {
        &self.csr
    }

    /// The CSC layout of V, used for `W^T V`.
    ///
    /// Exposed for the same reason as [`Self::csr`].
    ///
    /// ### Returns
    ///
    /// A reference to the CSC layer.
    pub fn csc(&self) -> &CompressedSparseData2<S> {
        &self.csc
    }
}

/// NmfInput trait implementation for [SparseInput]
impl<S, F> NmfInput<F> for SparseInput<S, F>
where
    S: BixverseNumeric + BixverseSimd + Into<F> + Float + Clone + Sync,
    F: BixverseFloat + BixverseSimd + Send + Sync + std::iter::Sum,
{
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn sq_frob(&self) -> F {
        self.sq_frob
    }

    fn wt_v(&self, w: MatRef<F>, out: &mut Mat<F>) {
        spmm_csc_wt_v::<S, F>(&self.csc, w, out);
    }

    fn v_ht(&self, h: MatRef<F>, out: &mut Mat<F>) {
        spmm_csr_v_ht::<S, F>(&self.csr, h, out);
    }

    fn top_k_svd(&self, k: usize) -> Result<SvdResults<F>, BixverseErrors> {
        sparse_svd_lanczos::<S, S, F>(&self.csr, k, 42, false, None, None, None)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::nmf_hals::dense::DenseInput;
    use faer::Mat;

    /// One 3x4 matrix in both CSR and dense form, for sparse against dense parity checks.
    fn fixture_csr() -> (CompressedSparseData2<f64, f64>, Mat<f64>) {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices: Vec<u32> = vec![0, 2, 1, 3, 0, 3];
        let indptr: Vec<u32> = vec![0, 2, 4, 6];
        let csr =
            CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, (3, 4));
        let dense = Mat::from_fn(3, 4, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (0, 2) => 2.0,
            (1, 1) => 3.0,
            (1, 3) => 4.0,
            (2, 0) => 5.0,
            (2, 3) => 6.0,
            _ => 0.0,
        });
        (csr, dense)
    }

    /// `||V||_F^2` over the non-zeros has to be accurate on data where summing in
    /// `F` visibly is not. Sparse is the path where this bit hardest, because the
    /// term count is the non-zero count: measured 4.4% low over 3e7 non-zeros
    /// before the accumulator moved to f64.
    #[test]
    fn sq_frob_survives_a_swamped_accumulator() {
        // One leading non-zero squaring to 1e8, then a million ones. At 1e8 the
        // f32 spacing is 8, so a running f32 total swallows every one of them.
        let n_ones = 1_000_000usize;
        let mut data = vec![1e4f32];
        data.extend(std::iter::repeat_n(1.0f32, n_ones));
        let nnz = data.len();
        let indices: Vec<u32> = (0..nnz as u32).collect();
        let indptr: Vec<u32> = vec![0, nnz as u32];
        let csr =
            CompressedSparseData2::<f32, f32>::new_csr(&data, &indices, &indptr, None, (1, nnz));

        let sparse_in: SparseInput<f32, f32> = SparseInput::from_primary(&csr).unwrap();
        let exact = 1e8 + n_ones as f64;
        let got = sparse_in.sq_frob() as f64;
        assert!(
            (got - exact).abs() <= 1e-5 * exact,
            "sq_frob is off: got {got}, exact {exact}"
        );

        let naive = data.iter().fold(0f32, |acc, &v| acc + v * v) as f64;
        assert!(
            (naive - exact).abs() > 0.005 * exact,
            "the f32 reference was supposed to be visibly wrong, so this test proves nothing"
        );
    }

    /// The sparse backend reports the same shape and squared Frobenius norm as the dense one.
    #[test]
    fn shape_and_sq_frob_match_dense() {
        let (csr, dense) = fixture_csr();
        let sparse_in: SparseInput<f64, f64> = SparseInput::from_primary(&csr).unwrap();
        let dense_in = DenseInput::new(dense.as_ref()).unwrap();
        assert_eq!(sparse_in.shape(), dense_in.shape());
        assert!((sparse_in.sq_frob() - dense_in.sq_frob()).abs() < 1e-12);
    }

    /// A negative non-zero is refused at construction rather than silently factorised.
    #[test]
    fn rejects_negative() {
        let data = vec![1.0, -2.0];
        let indices: Vec<u32> = vec![0, 1];
        let indptr: Vec<u32> = vec![0, 2];
        let csr =
            CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, (1, 2));
        let err = SparseInput::<f64, f64>::from_primary(&csr).unwrap_err();
        assert!(matches!(err, BixverseErrors::NmfNonNegativeViolated));
    }

    /// A NaN non-zero is refused at construction.
    #[test]
    fn rejects_non_finite() {
        let data = vec![1.0, f64::NAN];
        let indices: Vec<u32> = vec![0, 1];
        let indptr: Vec<u32> = vec![0, 2];
        let csr =
            CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, (1, 2));
        let err = SparseInput::<f64, f64>::from_primary(&csr).unwrap_err();
        assert!(matches!(err, BixverseErrors::NmfNonFinite));
    }

    /// The hand-rolled CSC `W^T V` kernel agrees with the dense faer matmul.
    #[test]
    fn wt_v_matches_dense() {
        let (csr, dense) = fixture_csr();
        let k = 2;
        let w: Mat<f64> = Mat::from_fn(3, k, |i, j| (i + j + 1) as f64);
        let sparse_in: SparseInput<f64, f64> = SparseInput::from_primary(&csr).unwrap();
        let dense_in = DenseInput::new(dense.as_ref()).unwrap();
        let mut out_s = Mat::<f64>::zeros(k, 4);
        let mut out_d = Mat::<f64>::zeros(k, 4);
        sparse_in.wt_v(w.as_ref(), &mut out_s);
        dense_in.wt_v(w.as_ref(), &mut out_d);
        for r in 0..k {
            for j in 0..4 {
                assert!((out_s[(r, j)] - out_d[(r, j)]).abs() < 1e-9);
            }
        }
    }

    /// The hand-rolled CSR `V H^T` kernel agrees with the dense faer matmul.
    #[test]
    fn v_ht_matches_dense() {
        let (csr, dense) = fixture_csr();
        let k = 2;
        let h: Mat<f64> = Mat::from_fn(k, 4, |i, j| (i + j + 1) as f64);
        let sparse_in: SparseInput<f64, f64> = SparseInput::from_primary(&csr).unwrap();
        let dense_in = DenseInput::new(dense.as_ref()).unwrap();
        let mut out_s = Mat::<f64>::zeros(3, k);
        let mut out_d = Mat::<f64>::zeros(3, k);
        sparse_in.v_ht(h.as_ref(), &mut out_s);
        dense_in.v_ht(h.as_ref(), &mut out_d);
        for i in 0..3 {
            for r in 0..k {
                assert!((out_s[(i, r)] - out_d[(i, r)]).abs() < 1e-9);
            }
        }
    }

    /// A CSC source takes the other arm of `build_pair`, and must still yield the same products.
    #[test]
    fn csc_input_produces_same_products() {
        // Same logical matrix but constructed as CSC.
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0];
        let indices: Vec<u32> = vec![0, 2, 1, 0, 1, 2];
        let indptr: Vec<u32> = vec![0, 2, 3, 4, 6];
        let csc =
            CompressedSparseData2::<f64, f64>::new_csc(&data, &indices, &indptr, None, (3, 4));
        let dense = Mat::from_fn(3, 4, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (0, 2) => 2.0,
            (1, 1) => 3.0,
            (1, 3) => 4.0,
            (2, 0) => 5.0,
            (2, 3) => 6.0,
            _ => 0.0,
        });

        let sparse_in: SparseInput<f64, f64> = SparseInput::from_primary(&csc).unwrap();
        let dense_in = DenseInput::new(dense.as_ref()).unwrap();

        let k = 2;
        let w: Mat<f64> = Mat::from_fn(3, k, |i, j| (i + j + 1) as f64);
        let mut out_s = Mat::<f64>::zeros(k, 4);
        let mut out_d = Mat::<f64>::zeros(k, 4);
        sparse_in.wt_v(w.as_ref(), &mut out_s);
        dense_in.wt_v(w.as_ref(), &mut out_d);
        for r in 0..k {
            for j in 0..4 {
                assert!((out_s[(r, j)] - out_d[(r, j)]).abs() < 1e-9);
            }
        }
    }

    /// `from_secondary` reads the `data_2` layer, so the cached norm comes from it, not `data`.
    #[test]
    fn from_secondary_uses_data_2() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let data_2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices: Vec<u32> = vec![0, 2, 1, 3, 0, 3];
        let indptr: Vec<u32> = vec![0, 2, 4, 6];
        let csr = CompressedSparseData2::<f64, f64>::new_csr(
            &data,
            &indices,
            &indptr,
            Some(&data_2),
            (3, 4),
        );

        let sparse_in: SparseInput<f64, f64> = SparseInput::from_secondary(&csr).unwrap();
        // sq_frob is from data_2, not data
        let expected: f64 = data_2.iter().map(|v| v * v).sum();
        assert!((sparse_in.sq_frob() - expected).abs() < 1e-12);
    }

    /// The Lanczos SVD returns k triplets of the right shape, in descending singular value order.
    #[test]
    fn top_k_svd_shapes() {
        let (csr, _) = fixture_csr();
        let sparse_in: SparseInput<f64, f64> = SparseInput::from_primary(&csr).unwrap();
        let svd = sparse_in.top_k_svd(2).unwrap();
        assert_eq!(svd.u.shape(), (3, 2));
        assert_eq!(svd.v.shape(), (4, 2));
        assert_eq!(svd.s.len(), 2);
        assert!(svd.s[0] >= svd.s[1]);
    }
}
