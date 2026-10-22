//! Sparse matrix formats, sparse operations and helpers to transform different
//! formats into each other.

use crate::utils::gemm::gemm;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

pub use bixverse_sparse::sparse::*;

use crate::core::math::pca_svd::SvdResults;
use crate::core::math::vector_helpers::sum_sq_f64;
use crate::prelude::*;
use crate::utils::faer_parallelism;
use crate::utils::simd::{sum_squared_dev_widen_simd_f32, sum_widen_simd_f32};

////////////////////////
// Dense construction //
////////////////////////

/// Builds a [CompressedSparseData2] from a dense `faer` matrix.
///
/// An extension trait rather than an inherent method, as the sparse structure
/// lives in `bixverse-sparse`, which does not depend on `faer`.
pub trait SparseFromDense<T> {
    /// Generates a sparse matrix from a dense matrix
    ///
    /// ### Params
    ///
    /// * `mat`: The dense matrix to convert to a sparse matrix
    /// * `format`: The format of the sparse matrix to generate
    ///
    /// ### Returns
    ///
    /// * `Self`: The sparse matrix generated from the dense matrix
    fn from_dense_matrix(mat: MatRef<T>, format: CompressedSparseFormat) -> Self;
}

impl<T, U> SparseFromDense<T> for CompressedSparseData2<T, U>
where
    T: BixverseFloat + BixverseNumeric,
    U: BixverseNumeric,
{
    fn from_dense_matrix(mat: MatRef<T>, format: CompressedSparseFormat) -> Self {
        Self::from_entry_fn(mat.nrows(), mat.ncols(), format, |i, j| mat[(i, j)])
    }
}

/////////////////////////////
// Sparse dense operations //
/////////////////////////////

//////////////////////////
// Sparse dense: consts //
//////////////////////////

/// Output rows accumulated per tile in [`csr_sparse_matmul_dense`]
const MATMUL_DENSE_ROW_BLOCK: usize = 256;

////////////////////////////////////
// Sparse dense: public functions //
////////////////////////////////////

/// Sparse CSR @ sparse CSR -> dense
///
/// Parallel over output rows. For cases where you assume that the resulting
/// matrix will become dense.
///
/// ### Params
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// ### Returns
///
/// Dense matrix product of both CSR matrices
pub fn csr_sparse_matmul_dense<T>(
    a: &CompressedSparseData2<T>,
    b: &CompressedSparseData2<T>,
) -> Result<Mat<T>, CoreErrors>
where
    T: BixverseFloat + Send + Sync + Default,
{
    require_matmul_operands(a, b)?;

    let n_rows = a.nrows();
    let n_cols = b.ncols();
    let m_indptr = &a.indptr;
    let m_indices = &a.indices;
    let m_data = &a.data;
    let g_indptr = &b.indptr;
    let g_indices = &b.indices;
    let g_data = &b.data;

    // Row blocks are accumulated into a row-major tile, then transposed into
    // the column-major output in parallel over output columns.
    let mut out = Mat::<T>::zeros(n_rows, n_cols);
    for block_start in (0..n_rows).step_by(MATMUL_DENSE_ROW_BLOCK) {
        let rows = MATMUL_DENSE_ROW_BLOCK.min(n_rows - block_start);
        let mut tile = vec![T::zero(); rows * n_cols];
        tile.par_chunks_mut(n_cols.max(1))
            .take(rows)
            .enumerate()
            .for_each(|(r, row)| {
                let i = block_start + r;
                for p in m_indptr[i] as usize..m_indptr[i + 1] as usize {
                    let j = m_indices[p] as usize;
                    let w = m_data[p];
                    let gs = g_indptr[j] as usize;
                    let ge = g_indptr[j + 1] as usize;
                    for q in gs..ge {
                        let t = g_indices[q] as usize;
                        row[t] += w * g_data[q];
                    }
                }
            });

        out.par_col_iter_mut().enumerate().for_each(|(j, mut col)| {
            for r in 0..rows {
                col[block_start + r] = tile[r * n_cols + j];
            }
        });
    }

    Ok(out)
}

/// Apply a CSR operator to a dense row-major block: `out = a @ block`
///
/// The kernel behind repeated diffusion, `T @ (T @ (T @ X))`. Applying the
/// operator `t` times is always preferable to forming `T^t`: at `knn = 30` the
/// operator carries ~30 non-zeros per row, `T^2` ~900 and `T^3` ~27,000, so at
/// 100k rows the explicit cube is on the order of 2.7e9 non-zeros while `t`
/// applications stay at ~30 per row for the same answer.
///
/// Both buffers are flat row-major rather than [faer::Mat] on purpose. The
/// operation is `out_row_i = sum_j w_ij * block_row_j`, so a row-major layout
/// makes every inner step a contiguous SIMD axpy of `width` elements, where
/// faer's column-major storage would turn it into a strided scatter.
///
/// Parallel over output rows, which are disjoint, so no synchronisation.
///
/// Note that `single_cell::sc_analysis::meld` has an older private
/// column-at-a-time version of this (`chebyshev_apply_columns`) that restreams
/// the whole operator once per column. It should move over here at some point.
///
/// ### Params
///
/// * `a` - The operator, CSR, `n x k`
/// * `block` - Dense input, row-major, `k * width` elements
/// * `width` - Columns in the block
/// * `out` - Dense output, row-major, `n * width` elements, overwritten
///
/// ### Returns
///
/// `Ok(())`, or [CoreErrors::SparseMatrixMustBeCsr] if `a` is in CSC, or
/// [CoreErrors::SparseMatrixMultiplication] if the buffers do not match the
/// operator's shape.
///
/// ### Panics
///
/// If `a` violates its own CSR invariants: a column index at or above
/// `a.shape().1`, or an `indptr` shorter than `a.shape().0 + 1`. Both index
/// safe slices, so this is a panic rather than unsoundness, but it surfaces
/// inside a rayon closure. `CompressedSparseData2` constructors do not validate,
/// so run [assert_invariants](CompressedSparseData2::assert_invariants) on a
/// matrix of uncertain provenance.
pub fn csr_matmul_dense_block<T>(
    a: &CompressedSparseData2<T>,
    block: &[T],
    width: usize,
    out: &mut [T],
) -> Result<(), CoreErrors>
where
    T: BixverseFloat + BixverseSimd,
{
    require_csr(a)?;

    let (n_rows, n_cols) = a.shape();

    // `width == 0` would make both buffers empty regardless of the operator, so
    // the length checks below could not tell a shape mismatch from a no-op.
    if width == 0 || block.len() != n_cols * width || out.len() != n_rows * width {
        return Err(CoreErrors::SparseMatrixMultiplication {
            n_col_a: n_cols,
            n_row_b: block.len().checked_div(width).unwrap_or(0),
        });
    }

    let indptr = &a.indptr;
    let indices = &a.indices;
    let data = &a.data;

    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, out_row)| {
            out_row.fill(T::zero());
            for p in indptr[i] as usize..indptr[i + 1] as usize {
                let j = indices[p] as usize;
                T::bxv_axpy_simd(out_row, data[p], &block[j * width..(j + 1) * width]);
            }
        });

    Ok(())
}

///////////////////////
// Sparse statistics //
///////////////////////

////////////////////////
// Statistics: consts //
////////////////////////

/// Stored non-zeros from which the column statistics fan out over rayon
const PAR_COLUMN_STATS_MIN_NNZ: usize = 1 << 20;

/////////////////////////
// Statistics: helpers //
/////////////////////////

/// The data layer a column statistic runs over.
///
/// ### Params
///
/// * `csc` - The matrix, which must be CSC
/// * `use_second_layer` - Use `data_2` instead of `data`
///
/// ### Returns
///
/// The layer's stored values, or [CoreErrors::SparseMatrixMustBeCsc] /
/// [CoreErrors::Data2NotAvailable].
fn csc_layer<T>(csc: &CompressedSparseData2<T>, use_second_layer: bool) -> Result<&[T], CoreErrors>
where
    T: BixverseNumeric,
{
    require_csc(csc)?;

    if use_second_layer {
        Ok(csc
            .data_2
            .as_ref()
            .ok_or(CoreErrors::Data2NotAvailable)?
            .as_slice())
    } else {
        Ok(csc.data.as_slice())
    }
}

/// Evaluate a per-column statistic for every column, in parallel once the
/// matrix is large enough.
///
/// ### Params
///
/// * `ncols` - Number of columns
/// * `nnz` - Stored values of the layer, gating the parallel path via
///   [PAR_COLUMN_STATS_MIN_NNZ]
/// * `stat` - Statistic of column `j`
///
/// ### Returns
///
/// One value per column.
fn collect_per_column<T, F>(ncols: usize, nnz: usize, stat: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if nnz >= PAR_COLUMN_STATS_MIN_NNZ {
        (0..ncols).into_par_iter().map(stat).collect()
    } else {
        (0..ncols).map(stat).collect()
    }
}

//////////////////////////////////
// Statistics: public functions //
//////////////////////////////////

/// Calculate the column means for CSC [CompressedSparseData2]
///
/// ### Params
///
/// * `csc` - The [CompressedSparseData2] (needs to have floats)
/// * `use_second_layer` - Use the second data layer
///
/// ### Returns
///
/// The column means
pub fn sparse_col_means_csc<T>(
    csc: &CompressedSparseData2<T>,
    use_second_layer: bool,
) -> Result<Vec<T>, CoreErrors>
where
    T: BixverseFloat + BixverseSimd,
{
    let active_data = csc_layer(csc, use_second_layer)?;

    let (nrows, ncols) = csc.shape();
    let nrows_t = T::from_usize(nrows).unwrap();

    Ok(collect_per_column(ncols, active_data.len(), |j| {
        let start = csc.indptr[j] as usize;
        let end = csc.indptr[j + 1] as usize;
        T::bxv_sum(&active_data[start..end]) / nrows_t
    }))
}

/// Calculate the column standard deviations for CSC [CompressedSparseData2]
///
/// ### Params
///
/// * `csc` - The [CompressedSparseData2] (needs to have floats)
/// * `use_second_layer` - Use the second data layer
///
/// ### Returns
///
/// The column standard deviations
pub fn sparse_col_sds_csc<T>(
    csc: &CompressedSparseData2<T>,
    use_second_layer: bool,
) -> Result<Vec<T>, CoreErrors>
where
    T: BixverseFloat + BixverseSimd,
{
    let active_data = csc_layer(csc, use_second_layer)?;

    let (nrows, ncols) = csc.shape();
    let nrows_t = T::from_usize(nrows).unwrap();
    let denom = nrows_t - T::one();

    Ok(collect_per_column(ncols, active_data.len(), |j| {
        let start = csc.indptr[j] as usize;
        let end = csc.indptr[j + 1] as usize;
        let col_slice = &active_data[start..end];
        let implicit_zeros = T::from_usize(nrows - (end - start)).unwrap();

        let mean = T::bxv_sum(col_slice) / nrows_t;
        let ssd_nonzero = T::bxv_sum_squared_deviation(col_slice, mean);
        let ssd_total = ssd_nonzero + implicit_zeros * mean * mean;
        (ssd_total / denom).sqrt()
    }))
}

/// Compute Frobenius norm of sparse matrix
///
/// Accumulated in `f64` via [`sum_sq_f64`] and narrowed once at the end. Summing
/// in `f32` drifts by percentage points over tens of millions of non-zeros, and
/// the square root then hides how far off it was.
///
/// ### Params
///
/// * `mat` - Sparse matrix in CSR or CSC format
///
/// ### Returns
///
/// Frobenius norm ||A||_F = sqrt(sum(A_ij^2))
pub fn frobenius_norm<T>(mat: &CompressedSparseData2<T>) -> f32
where
    T: BixverseNumeric + ToPrimitive,
{
    sum_sq_f64(&mat.data).sqrt() as f32
}

/// Squared Frobenius norm of a sparse matrix, in `f64`.
///
/// Prefer this over squaring [`frobenius_norm`] wherever the result is one of
/// several large terms that cancel: narrowing to `f32` and squaring again throws
/// away the precision the cancellation needs.
///
/// ### Params
///
/// * `mat` - Sparse matrix in CSR or CSC format
///
/// ### Returns
///
/// `||A||_F^2`.
pub fn frobenius_norm_sq_f64<T>(mat: &CompressedSparseData2<T>) -> f64
where
    T: BixverseNumeric + ToPrimitive,
{
    sum_sq_f64(&mat.data)
}

/// Counts the zeroes in a given faer matrix
///
/// ### Params
///
/// * `mat` - The respective faer matrix
///
/// ### Returns
///
/// A tuple with the first being the total zeroes, the second the zeroes per
/// row and the last element being the column zeroes.
pub fn count_zeroes<T>(mat: &MatRef<T>) -> (usize, Vec<usize>, Vec<usize>)
where
    T: BixverseFloat,
{
    let (nrow, ncol) = mat.shape();
    let mut total_zeroes = 0_usize;
    let mut row_zeroes = vec![0_usize; nrow];
    let mut col_zeroes = vec![0_usize; ncol];

    let zero = T::zero();

    for j in 0..ncol {
        for i in 0..nrow {
            let val = unsafe { mat.get_unchecked(i, j) };
            if *val == zero {
                total_zeroes += 1;
                row_zeroes[i] += 1;
                col_zeroes[j] += 1;
            }
        }
    }

    (total_zeroes, row_zeroes, col_zeroes)
}

/////////////////////////
// Sparse correlations //
/////////////////////////

//////////////////////////
// Correlations: consts //
//////////////////////////

/// Below this a column counts as constant and its correlations are reported as
/// `0.0` rather than as a division by a vanishing standard deviation.
const SPARSE_COR_SD_EPS: f64 = 1e-8;

///////////////////////////
// Correlations: structs //
///////////////////////////

/// One sparse column reduced to what [`sparse_pairwise_correlations`] needs.
///
/// The stored entries are kept as they arrive; nothing here densifies, so the
/// memory is `O(nnz)` per column rather than `O(n_rows)`.
#[derive(Clone, Debug)]
pub struct SparseColMoments {
    /// Row indices of the stored entries. Ascending order is not required.
    pub indices: Vec<u32>,
    /// Values at those indices. The raw values for Pearson, the zero-shifted
    /// ranks for Spearman. Every non-stored position is exactly zero in both
    /// cases, which is what makes the closed forms below valid.
    pub values: Vec<f32>,
    /// Mean over all `n_rows` entries, structural zeros included.
    pub mean: f64,
    /// Standard deviation over all `n_rows` entries, sample (n - 1)
    /// denominator.
    pub sd: f64,
}

///////////////////////////
// Correlations: helpers //
///////////////////////////

/// Average ranks of one sparse column, shifted so the zero block sits at zero.
///
/// ### Params
///
/// * `values` - The stored values of the column.
/// * `n_rows` - Full length of the column, structural zeros included.
///
/// ### Returns
///
/// The shifted average ranks, aligned to `values`.
fn shifted_ranks_sparse(values: &[f32], n_rows: usize) -> Vec<f32> {
    let n_stored = values.len();
    let n_implicit = n_rows.saturating_sub(n_stored);

    let n_neg = values.iter().filter(|&&v| v < 0.0).count();
    let n_stored_zero = values.iter().filter(|&&v| v == 0.0).count();
    let zero_block = n_stored_zero + n_implicit;

    // The zero group occupies 1-based ranks (n_neg + 1) ..= (n_neg + zero_block).
    let r_zero = n_neg as f64 + (zero_block as f64 + 1.0) / 2.0;

    let mut order: Vec<u32> = (0..n_stored as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        values[a as usize]
            .partial_cmp(&values[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0_f32; n_stored];
    let mut i = 0;
    while i < n_stored {
        let value = values[order[i] as usize];
        let start = i;
        while i < n_stored && values[order[i] as usize] == value {
            i += 1;
        }

        let avg_rank = if value == 0.0 {
            r_zero
        } else {
            // A stored value at sorted position `p` sits at 1-based full rank
            // `p + 1` when negative, and `p + 1 + n_implicit` when positive:
            // the implicit zeros are spliced in ahead of it.
            let offset = if value < 0.0 { 0 } else { n_implicit };
            (start + i + 1 + 2 * offset) as f64 / 2.0
        };

        let shifted = (avg_rank - r_zero) as f32;
        for &slot in &order[start..i] {
            ranks[slot as usize] = shifted;
        }
    }

    ranks
}

////////////////////////////////////
// Correlations: public functions //
////////////////////////////////////

/// Reduce one sparse column to its moments without densifying it.
///
/// ### Params
///
/// * `indices` - Row indices of the stored entries, each below `n_rows` and
///   without duplicates. Order does not matter.
/// * `values` - Values at those indices, aligned to `indices`.
/// * `n_rows` - Full length of the column, structural zeros included. Must be
///   at least `values.len()`.
/// * `spearman` - Rank the column first, for Spearman rather than Pearson.
///
/// ### Returns
///
/// The [`SparseColMoments`] for this column.
pub fn sparse_col_moments(
    indices: &[u32],
    values: &[f32],
    n_rows: usize,
    spearman: bool,
) -> SparseColMoments {
    let stored: Vec<f32> = if spearman {
        shifted_ranks_sparse(values, n_rows)
    } else {
        values.to_vec()
    };

    let n = n_rows as f64;
    let mean = sum_widen_simd_f32(&stored) / n;
    let ss_stored = sum_squared_dev_widen_simd_f32(&stored, mean);
    let ss = ss_stored + n_rows.saturating_sub(stored.len()) as f64 * mean * mean;
    let sd = if n_rows < 2 {
        0.0
    } else {
        (ss / (n - 1.0)).sqrt()
    };

    SparseColMoments {
        indices: indices.to_vec(),
        values: stored,
        mean,
        sd,
    }
}

/// Pearson correlation of specified column pairs, over the sparsity patterns.
///
/// Uses the raw-moment form of the covariance,
///
/// ```text
/// sum (x_a - m_a)(x_b - m_b) = sum x_a x_b - n * m_a * m_b
/// ```
///
/// because `sum x_a x_b` only picks up the intersection of the two sparsity
/// patterns. On single-cell like counts that is a hundredth of the column
/// rather than all of it.
///
/// ### Params
///
/// * `moments` - Per-column moments, see [`sparse_col_moments`].
/// * `pairs` - Index pairs into `moments`, one per requested correlation.
/// * `n_rows` - Full column length, structural zeros included.
///
/// ### Returns
///
/// One correlation per entry of `pairs`, in the same order, clamped to
/// `[-1, 1]`. A column whose standard deviation is below
/// [`SPARSE_COR_SD_EPS`] yields `0.0`.
pub fn sparse_pairwise_correlations(
    moments: &[SparseColMoments],
    pairs: &[(usize, usize)],
    n_rows: usize,
) -> Vec<f32> {
    let mut grouped: FxHashMap<usize, Vec<(usize, usize)>> = FxHashMap::default();
    for (slot, &(a, b)) in pairs.iter().enumerate() {
        grouped.entry(a).or_default().push((slot, b));
    }
    let groups: Vec<(usize, Vec<(usize, usize)>)> = grouped.into_iter().collect();

    let n = n_rows as f64;
    let denom = n - 1.0;

    let solved: Vec<Vec<(usize, f32)>> = groups
        .par_iter()
        .map_init(
            || vec![0_f32; n_rows],
            |scratch, (first, partners)| {
                let a = &moments[*first];
                for (&idx, &value) in a.indices.iter().zip(a.values.iter()) {
                    scratch[idx as usize] = value;
                }

                let out = partners
                    .iter()
                    .map(|&(slot, second)| {
                        let b = &moments[second];
                        if a.sd < SPARSE_COR_SD_EPS || b.sd < SPARSE_COR_SD_EPS {
                            return (slot, 0_f32);
                        }
                        // scratch is zero wherever `a` has no entry, so the
                        // product drops out and no branch is needed.
                        let mut cross = 0_f64;
                        for (&idx, &value) in b.indices.iter().zip(b.values.iter()) {
                            cross += scratch[idx as usize] as f64 * value as f64;
                        }
                        let cov = cross - n * a.mean * b.mean;
                        let cor = cov / (denom * a.sd * b.sd);
                        (slot, (cor as f32).clamp(-1_f32, 1_f32))
                    })
                    .collect();

                // Clear only what was written; refilling the whole buffer would
                // cost `n_rows` per group instead of `nnz`.
                for &idx in &a.indices {
                    scratch[idx as usize] = 0_f32;
                }

                out
            },
        )
        .collect();

    let mut res = vec![0_f32; pairs.len()];
    for group in solved {
        for (slot, value) in group {
            res[slot] = value;
        }
    }
    res
}

/////////////////////////
// Lanczos eigensolver //
/////////////////////////

/////////////////////
// Lanczos: consts //
/////////////////////

/// Ritz vectors retained across a restart, above the requested pair count.
const LANCZOS_RESTART_GUARD: usize = 5;

/// Norm ratio below which a second Gram-Schmidt pass is run.
///
/// The DGKS criterion. A drop past `1 / sqrt(2)` means the projection cancelled
/// at least half the vector, which is where the round-off it leaves behind
/// stops being negligible. Above that the second pass is measurably wasted work.
const LANCZOS_ORTHO_REFINE_RATIO: f64 = 0.707;

/// Lanczos breakdown threshold, relative to the running `||A||` estimate.
///
/// An absolute threshold is meaningless for a matrix that is not `O(1)`. The
/// attainable floor on the residual norm after two Gram-Schmidt passes is
/// `~eps * sqrt(n) * ||A||`, so this sits comfortably above it while still only
/// firing on a genuinely invariant subspace.
const LANCZOS_BREAKDOWN_RATIO: f64 = 1e-12;

//////////////////////
// Lanczos: structs //
//////////////////////

/// Parameters for the restarted Lanczos eigensolver.
#[derive(Clone, Copy, Debug)]
pub struct LanczosParams {
    /// Krylov basis vectors held at once. `None` derives it from the requested
    /// component count as `max(2k + 10, k + 20)`, which leaves enough room
    /// above the wanted pairs for the restart to make progress.
    pub basis_size: Option<usize>,
    /// Maximum restart cycles. Each cycle costs one basis worth of
    /// matrix-vector products plus the orthogonalisation against the basis, so
    /// this is the only knob that bounds the run time.
    pub max_restarts: usize,
    /// Relative residual tolerance. A Ritz pair counts as converged once
    /// `||A x - lambda x||` drops below `tol * ||A||`. The scaling matters: an
    /// absolute threshold is unreachable for a matrix whose norm is large and
    /// trivially met for one whose norm is tiny.
    pub tol: f64,
}

impl LanczosParams {
    /// Defaults tuned on a single cell diffusion kernel.
    ///
    /// ### Returns
    ///
    /// Self with a derived basis size, 64 restarts and a relative tolerance of
    /// `1e-8`.
    pub fn new() -> Self {
        Self {
            basis_size: None,
            max_restarts: 64,
            tol: 1e-8,
        }
    }

    /// Resolve the basis size for a given component count.
    ///
    /// ### Params
    ///
    /// * `n_components` - Eigenpairs requested, already capped at `n`.
    /// * `n` - Matrix dimension, which caps the basis.
    ///
    /// ### Returns
    ///
    /// The number of basis vectors to hold, at least one and never above `n`.
    fn resolve_basis(&self, n_components: usize, n: usize) -> usize {
        self.basis_size
            .unwrap_or_else(|| (n_components * 2 + 10).max(n_components + 20))
            .max(n_components + 1)
            .min(n)
            .max(1)
    }
}

impl Default for LanczosParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Outcome of a restarted Lanczos solve, with the convergence diagnostics.
///
/// The diagnostics exist because a budget-exhausted solve and a converged one
/// are otherwise indistinguishable from the outside, and the difference is
/// several digits of accuracy in the eigenvalues.
#[derive(Clone, Debug)]
pub struct LanczosResult {
    /// Eigenvalues sorted descending. Kept in `f64` because the callers form
    /// `lambda / (1 - lambda)`, which is catastrophic in `f32` once `lambda`
    /// approaches one, as it does for any diffusion operator.
    pub eigenvalues: Vec<f64>,
    /// `eigenvectors[i][j]` is element `i` of eigenvector `j`. Always exactly
    /// `eigenvalues.len()` columns wide.
    pub eigenvectors: Vec<Vec<f32>>,
    /// Largest residual `||A x - lambda x||` over the returned pairs.
    pub residual: f64,
    /// Scale the residual was tested against, i.e. the `||A||` estimate.
    pub norm_estimate: f64,
    /// Whether `residual` met `tol * norm_estimate`.
    pub converged: bool,
    /// Restart cycles actually run.
    pub restarts: usize,
}

/// Ritz pairs and convergence diagnostics from [lanczos_thick_restart] and
/// [block_lanczos_thick_restart].
#[derive(Clone, Debug)]
pub struct RitzPairs {
    /// Ritz values sorted descending.
    pub values: Vec<f64>,
    /// Unit-norm Ritz vectors, column-major, `n` rows by `values.len()`
    /// columns.
    pub vectors: Vec<f64>,
    /// Largest residual `||A x - lambda x||` over the returned pairs.
    pub residual: f64,
    /// Scale the residual was tested against, i.e. the `||A||` estimate.
    pub norm_estimate: f64,
    /// Whether `residual` met `tol * norm_estimate`.
    pub converged: bool,
    /// Restart cycles actually run.
    pub restarts: usize,
}

//////////////////////
// Lanczos: helpers //
//////////////////////

/// Helper function for dot product of two vectors
///
/// ### Params
///
/// * `a` - Vector a
/// * `b` - Vector b
///
/// ### Returns
///
/// Dot product of the two vectors
fn dot<T>(a: &[T], b: &[T]) -> T
where
    T: BixverseSimd,
{
    assert_same_len!(a, b);
    T::bxv_dot_simd(a, b)
}

/// Helper function to normalise a vector
///
/// ### Params
///
/// * `v` - Initial vector
///
/// ### Returns
///
/// Normalised dot product of the vector `v`
fn norm<T>(v: &[T]) -> T
where
    T: BixverseSimd + BixverseFloat,
{
    let dot = dot(v, v);
    dot.sqrt()
}

/// Helper function to normalise a vector
///
/// ### Params
///
/// * `v` - Mutable reference of the vector to normalise
fn normalise<T>(v: &mut [T])
where
    T: BixverseSimd + BixverseFloat,
{
    let n = norm(v);
    v.par_iter_mut().for_each(|x| *x /= n);
}

/// Largest eigenpairs of a symmetric operator via thick-restarted Lanczos.
///
/// Each cycle builds a Krylov basis of
/// [LanczosParams::basis_size] vectors with full reorthogonalisation, does a
/// Rayleigh-Ritz extraction, and restarts from the best Ritz vectors plus the
/// residual direction. Restarting is the key point: the basis stays bounded
/// while the iteration keeps going, so a clustered spectrum converges without
/// the quadratic orthogonalisation cost a single long run would need.
///
/// The projected matrix is accumulated explicitly as `H[i][j] = <v_i, A v_j>`
/// rather than as a tridiagonal recurrence. After a thick restart the leading
/// basis vectors are Ritz vectors, not Lanczos vectors, so the projection is
/// arrow-shaped rather than tridiagonal; forming it directly handles that
/// without special-casing.
///
/// The basis is held column-major as one flat buffer, so the projection, the
/// reorthogonalisation, the restart and the final Ritz vectors are all faer
/// products rather than scalar loops. Orthogonalisation is classical
/// Gram-Schmidt with a DGKS-conditional second pass, which matches the accuracy
/// of two unconditional modified passes while being expressible as two matrix
/// products instead of 2 * basis_size scalar loops.
///
/// Convergence uses the exact Arnoldi residual |beta_m| * |y[m-1]|, which
/// still holds under thick restart because the relation A V = V H + beta v
/// e^T is preserved across cycles, tested against tol * ||A||.
///
/// Internals are f64 regardless of the caller's float type, the operator
/// included.
///
/// ### Params
///
/// * `n` - Dimension of the operator.
/// * `n_components` - Number of eigenpairs to compute. Capped at `n`.
/// * `seed` - Seed for the starting vector.
/// * `params` - [LanczosParams] for basis size, restarts and tolerance.
/// * `matvec` - Symmetric operator, writes `A x` into `y`.
///
/// ### Returns
///
/// [RitzPairs] with at most `min(n_components, n)` pairs, fewer if the
/// iteration hit an invariant subspace first.
fn lanczos_thick_restart<M>(
    n: usize,
    n_components: usize,
    seed: u64,
    params: LanczosParams,
    matvec: M,
) -> Result<RitzPairs, CoreErrors>
where
    M: Fn(&[f64], &mut [f64]),
{
    if n_components == 0 || n == 0 {
        return Err(CoreErrors::MustBePositive(
            "n_components and the matrix dimension".to_string(),
        ));
    }

    let n_keep = n_components.min(n);
    let m = params.resolve_basis(n_keep, n);

    // Column-major basis, one flat buffer, so every pass over it is a `faer`
    // product and column `j` is still a contiguous slice for the matrix product.
    let mut basis = vec![0.0f64; m * n];
    let mut h = Mat::<f64>::zeros(m, m);
    let mut w = vec![0.0f64; n];
    let mut coeffs = vec![0.0f64; m];

    let mut rng = StdRng::seed_from_u64(seed);
    for x in basis[..n].iter_mut() {
        *x = rng.random::<f64>() - 0.5;
    }
    normalise(&mut basis[..n]);

    // Basis vectors carried over from the previous cycle. Zero on the first.
    let mut locked = 0usize;
    let mut ritz_values: Vec<f64> = Vec::new();
    let mut ritz_coeffs = Mat::<f64>::zeros(0, 0);
    let mut m_final = 0usize;

    // Lower bound on ||A||_2, refined as the iteration sees more of the matrix.
    let mut norm_estimate = 0.0f64;
    let mut residual = f64::INFINITY;
    let mut converged = false;
    let mut restarts = 0usize;

    let budget = params.max_restarts.max(1);

    for cycle in 0..budget {
        restarts = cycle + 1;
        let mut m_eff = m;
        let mut last_beta = 0.0f64;

        for j in locked..m {
            matvec(&basis[j * n..(j + 1) * n], &mut w);

            let width = j + 1;

            // Rayleigh-Ritz coefficients against the whole basis so far. Taken
            // before any subtraction, so `h` is the exact projection, and they
            // double as the first Gram-Schmidt coefficient set.
            basis_project(&basis, n, width, &w, &mut coeffs[..width]);
            for i in 0..width {
                h[(i, j)] = coeffs[i];
                h[(j, i)] = coeffs[i];
            }

            let before = norm(&w);
            norm_estimate = norm_estimate.max(before);

            basis_subtract(&basis, n, width, &coeffs[..width], &mut w);
            last_beta = norm(&w);

            // DGKS: refine only when the projection cancelled enough of the
            // vector for the round-off it left behind to matter.
            if last_beta < LANCZOS_ORTHO_REFINE_RATIO * before {
                basis_project(&basis, n, width, &w, &mut coeffs[..width]);
                basis_subtract(&basis, n, width, &coeffs[..width], &mut w);
                last_beta = norm(&w);
            }

            // Breakdown: the basis already spans an invariant subspace, so the
            // Ritz pairs from it are exact and there is nothing left to add.
            if last_beta <= LANCZOS_BREAKDOWN_RATIO * norm_estimate {
                m_eff = width;
                last_beta = 0.0;
                break;
            }

            if width < m {
                let inv = 1.0 / last_beta;
                let next = &mut basis[width * n..(width + 1) * n];
                for (dst, &src) in next.iter_mut().zip(w.iter()) {
                    *dst = src * inv;
                }
            }
        }

        let (values, vectors) = symmetric_eigen_descending(h.as_ref(), m_eff)?;

        // Residual of Ritz pair i is |beta| * |y[m_eff - 1, i]|, exact under the
        // Arnoldi relation that thick restart preserves.
        let n_out = n_keep.min(m_eff);
        residual = (0..n_out).fold(0.0f64, |acc, i| {
            acc.max(last_beta * vectors[(m_eff - 1, i)].abs())
        });

        // The extreme Ritz values are the sharpest available lower bound on
        // ||A||, so scale the tolerance by them rather than testing an absolute
        // threshold that a large matrix can never reach.
        norm_estimate = norm_estimate
            .max(values[0].abs())
            .max(values[m_eff - 1].abs());
        converged = residual <= params.tol * norm_estimate;

        ritz_values = values;
        ritz_coeffs = vectors;
        m_final = m_eff;

        if converged || m_eff < m || last_beta == 0.0 || cycle + 1 == budget {
            break;
        }

        // Thick restart: keep the best Ritz vectors, then hand the residual
        // direction over as the next basis vector so the Krylov space continues
        // to grow from where it stopped.
        // Never retain more than half the basis: past that the cycle buys too
        // few new Krylov vectors to pay for the restart, and at `keep = m - 1`
        // it buys exactly one and the iteration effectively stops.
        let keep = (n_keep + LANCZOS_RESTART_GUARD).min(m / 2).max(1);
        let mut restarted = vec![0.0f64; keep * n];
        basis_expand(&basis, n, m_eff, ritz_coeffs.as_ref(), keep, &mut restarted);

        let inv = 1.0 / last_beta;
        for x in w.iter_mut() {
            *x *= inv;
        }

        h.fill(0.0);
        basis[..keep * n].copy_from_slice(&restarted);
        for (i, &value) in ritz_values.iter().take(keep).enumerate() {
            h[(i, i)] = value;
        }
        basis[keep * n..(keep + 1) * n].copy_from_slice(&w);
        locked = keep;
    }

    // Ritz vectors are built once, on the way out. Building them every cycle is
    // pure waste: only the last cycle's are ever returned, and the retained
    // block of the restart is bit-identical to the leading columns anyway.
    let n_out = n_keep.min(m_final);
    let mut ritz = vec![0.0f64; n_out * n];
    basis_expand(&basis, n, m_final, ritz_coeffs.as_ref(), n_out, &mut ritz);

    for column in ritz.chunks_exact_mut(n) {
        let scale: f64 = column.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        column.iter_mut().for_each(|v| *v *= inv);
    }

    Ok(RitzPairs {
        values: ritz_values[..n_out].to_vec(),
        vectors: ritz,
        residual,
        norm_estimate,
        converged,
        restarts,
    })
}

/// Project a vector onto the leading columns of the basis.
///
/// ### Params
///
/// * `basis` - Column-major basis buffer with `n` rows.
/// * `n` - Length of a basis vector.
/// * `width` - Leading columns to project against.
/// * `v` - Vector of length `n`.
/// * `out` - Receives `V^T v`; must be exactly `width` long.
///
/// ### Returns
///
/// Nothing; `out` is overwritten.
fn basis_project(basis: &[f64], n: usize, width: usize, v: &[f64], out: &mut [f64]) {
    let b = MatRef::from_column_major_slice(&basis[..width * n], n, width);
    let v = MatRef::from_column_major_slice(v, n, 1);
    let mut out = MatMut::from_column_major_slice_mut(out, width, 1);

    gemm(
        out.as_mut(),
        Accum::Replace,
        b.transpose(),
        v,
        1.0,
        faer_parallelism(),
    );
}

/// Subtract the basis expansion of a coefficient vector in place.
///
/// ### Params
///
/// * `basis` - Column-major basis buffer with `n` rows.
/// * `n` - Length of a basis vector.
/// * `width` - Leading columns to expand.
/// * `coeffs` - Coefficients, exactly `width` long.
/// * `v` - Vector of length `n`, updated to `v - V c`.
///
/// ### Returns
///
/// Nothing; `v` is updated in place.
fn basis_subtract(basis: &[f64], n: usize, width: usize, coeffs: &[f64], v: &mut [f64]) {
    let b = MatRef::from_column_major_slice(&basis[..width * n], n, width);
    let c = MatRef::from_column_major_slice(coeffs, width, 1);
    let mut v = MatMut::from_column_major_slice_mut(v, n, 1);

    gemm(v.as_mut(), Accum::Add, b, c, -1.0, faer_parallelism());
}

/// Expand Ritz coefficient columns back into the full space.
///
/// ### Params
///
/// * `basis` - Column-major basis buffer with `n` rows.
/// * `n` - Length of a basis vector.
/// * `width` - Basis columns spanned by the coefficients.
/// * `coeffs` - Rayleigh-Ritz eigenvectors, column-wise.
/// * `n_out` - Leading coefficient columns to expand.
/// * `out` - Receives the `n_out` expanded vectors, column-major.
///
/// ### Returns
///
/// Nothing; `out` is overwritten.
fn basis_expand(
    basis: &[f64],
    n: usize,
    width: usize,
    coeffs: MatRef<f64>,
    n_out: usize,
    out: &mut [f64],
) {
    let b = MatRef::from_column_major_slice(&basis[..width * n], n, width);
    let mut out = MatMut::from_column_major_slice_mut(out, n, n_out);

    gemm(
        out.as_mut(),
        Accum::Replace,
        b,
        coeffs.subcols(0, n_out),
        1.0,
        faer_parallelism(),
    );
}

/// Eigen-decompose the leading `size` block of a symmetric matrix, descending.
///
/// ### Params
///
/// * `h` - Symmetric matrix; only the leading `size` by `size` block is read.
/// * `size` - Size of the block to decompose.
///
/// ### Returns
///
/// `(eigenvalues, eigenvectors)` sorted by descending eigenvalue, the
/// eigenvectors held column-wise.
pub(crate) fn symmetric_eigen_descending(
    h: MatRef<f64>,
    size: usize,
) -> Result<(Vec<f64>, Mat<f64>), CoreErrors> {
    let block = Mat::<f64>::from_fn(size, size, |i, j| h[(i, j)]);

    let eig = block
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| CoreErrors::FaerEigenError)?;

    let raw_values = eig.S();
    let raw_vectors = eig.U();

    let mut order: Vec<usize> = (0..size).collect();
    order.sort_by(|&a, &b| raw_values[b].total_cmp(&raw_values[a]));

    let values: Vec<f64> = order.iter().map(|&i| raw_values[i]).collect();
    let vectors = Mat::<f64>::from_fn(size, size, |i, j| raw_vectors[(i, order[j])]);

    Ok((values, vectors))
}

///////////////////////////////
// Lanczos: public functions //
///////////////////////////////

/// Compute the largest eigenpairs of a symmetric sparse matrix.
///
/// Thin wrapper over [compute_largest_eigenpairs_lanczos_diag] for callers that
/// do not inspect the convergence diagnostics.
///
/// ### Params
///
/// * `matrix` - Symmetric sparse matrix. CSC input is converted to CSR.
/// * `n_components` - Number of eigenpairs to compute.
/// * `seed` - Seed for the starting vector.
/// * `params` - Optional [LanczosParams], defaulted when `None`.
///
/// ### Returns
///
/// `(eigenvalues, eigenvectors)` sorted by descending eigenvalue, where
/// `eigenvectors[i][j]` is element `i` of eigenvector `j`. Both outputs carry
/// the same number of components, which is `min(n_components, n)` and can be
/// smaller still if the iteration hit an invariant subspace first.
pub fn compute_largest_eigenpairs_lanczos<T>(
    matrix: &CompressedSparseData2<T>,
    n_components: usize,
    seed: u64,
    params: Option<LanczosParams>,
) -> Result<(Vec<f64>, Vec<Vec<f32>>), CoreErrors>
where
    T: BixverseNumeric + BixverseSimd + Into<f64>,
{
    let res = compute_largest_eigenpairs_lanczos_diag(matrix, n_components, seed, params)?;
    Ok((res.eigenvalues, res.eigenvectors))
}

/// Compute the largest eigenpairs of a symmetric sparse matrix, with
/// diagnostics.
///
/// Thick-restarted Lanczos, see [lanczos_thick_restart] for the algorithm.
///
/// ### Params
///
/// * `matrix` - Symmetric sparse matrix. CSC input is converted to CSR.
/// * `n_components` - Number of eigenpairs to compute. Capped at the matrix
///   dimension rather than erroring, so a request for more pairs than the
///   matrix has simply returns all of them.
/// * `seed` - Seed for the starting vector.
/// * `params` - Optional [LanczosParams], defaulted when None.
///
/// ### Returns
///
/// A [LanczosResult]. The eigenvalue count and the eigenvector column count
/// are always equal.
pub fn compute_largest_eigenpairs_lanczos_diag<T>(
    matrix: &CompressedSparseData2<T>,
    n_components: usize,
    seed: u64,
    params: Option<LanczosParams>,
) -> Result<LanczosResult, CoreErrors>
where
    T: BixverseNumeric + BixverseSimd + Into<f64>,
{
    let n = matrix.shape.0;

    // CSR gives contiguous rows for the matrix-vector product.
    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => matrix.clone(),
        CompressedSparseFormat::Csc => matrix.transform(),
    };
    let data_f64: Vec<f64> = csr.data.iter().map(|&v| v.into()).collect();

    let matvec = |x: &[f64], y: &mut [f64]| {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = 0.0;
            for idx in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                let j = csr.indices[idx] as usize;
                sum += data_f64[idx] * x[j];
            }
            *yi = sum;
        });
    };

    let ritz = lanczos_thick_restart(n, n_components, seed, params.unwrap_or_default(), matvec)?;

    let n_out = ritz.values.len();
    let mut eigenvectors = vec![vec![0.0f32; n_out]; n];
    for comp in 0..n_out {
        let column = &ritz.vectors[comp * n..(comp + 1) * n];
        for (point, &v) in column.iter().enumerate() {
            eigenvectors[point][comp] = v as f32;
        }
    }

    Ok(LanczosResult {
        eigenvalues: ritz.values,
        eigenvectors,
        residual: ritz.residual,
        norm_estimate: ritz.norm_estimate,
        converged: ritz.converged,
        restarts: ritz.restarts,
    })
}

///////////////////
// Block Lanczos //
///////////////////

///////////////////////////
// Block Lanczos: consts //
///////////////////////////

/// New blocks per restart cycle the derived basis leaves room for.
///
/// With a single new block per cycle the iteration degenerates into subspace
/// iteration with Rayleigh-Ritz, which crawls on a clustered spectrum. Not
/// tuned beyond that.
const BLOCK_LANCZOS_MIN_NEW_BLOCKS: usize = 3;

////////////////////////////
// Block Lanczos: structs //
////////////////////////////

/// Parameters for the block thick-restart Lanczos eigensolver.
#[derive(Clone, Copy, Debug)]
pub struct BlockLanczosParams {
    /// Vectors per block, i.e. the width of every operator call. Wider blocks
    /// mean fewer, larger calls, which is what pays where each call carries a
    /// fixed launch and readback cost. Capped at half the operator dimension.
    pub block_size: usize,
    /// Krylov basis vectors held at once, rounded down to whole blocks. `None`
    /// derives it as for [LanczosParams], rounded up to whole blocks, at least
    /// twice the retained Ritz block and with room for
    /// `BLOCK_LANCZOS_MIN_NEW_BLOCKS` new blocks per cycle.
    pub basis_size: Option<usize>,
    /// Maximum restart cycles.
    pub max_restarts: usize,
    /// Relative residual tolerance, tested against `tol * ||A||` as in
    /// [LanczosParams::tol].
    pub tol: f64,
}

impl BlockLanczosParams {
    /// Defaults, chosen for the cubecl GPU path. Backends whose per-call cost
    /// grows faster with the block width override the block size.
    ///
    /// ### Returns
    ///
    /// Self with blocks of 32, a derived basis size, 64 restarts and a
    /// relative tolerance of `1e-6`. That is tight enough that an f32
    /// operator's singular values agree with an f64 solve to f32 precision,
    /// while the scalar solver's `1e-8` sits near the f32 round-off and would
    /// mostly burn the restart budget.
    pub fn new() -> Self {
        Self {
            block_size: 32,
            basis_size: None,
            max_restarts: 64,
            tol: 1e-6,
        }
    }

    /// Resolve block width, basis size and retained Ritz vectors.
    ///
    /// Everything is a whole number of blocks, so the operator is only ever
    /// called at full width.
    ///
    /// ### Params
    ///
    /// * `n_keep` - Eigenpairs requested, already capped at `n`.
    /// * `n` - Operator dimension.
    ///
    /// ### Returns
    ///
    /// `(block, basis, keep)`. `basis + block > n` means the operator is too
    /// small for Lanczos to pay, see [block_lanczos_thick_restart].
    fn resolve(&self, n_keep: usize, n: usize) -> (usize, usize, usize) {
        let b = self.block_size.min(n / 2).max(1);
        let round_up = |x: usize| x.div_ceil(b) * b;
        let floor = |x: usize| (x / b) * b;

        let keep_target = round_up(n_keep + LANCZOS_RESTART_GUARD);
        let basis = match self.basis_size {
            Some(size) => floor(size),
            None => round_up((n_keep * 2 + 10).max(n_keep + 20))
                .max(2 * keep_target)
                .max(keep_target + BLOCK_LANCZOS_MIN_NEW_BLOCKS * b),
        }
        .max(2 * b)
        .min(floor(n))
        .max(b);
        let keep = keep_target.min(floor(basis / 2)).max(b);

        (b, basis, keep)
    }
}

impl Default for BlockLanczosParams {
    fn default() -> Self {
        Self::new()
    }
}

////////////////////////////
// Block Lanczos: helpers //
////////////////////////////

/// Check that a block operator returned the `n x width` block it was asked for.
///
/// ### Params
///
/// * `out` - What the operator returned
/// * `n` - Dimension of the operator
/// * `width` - Columns of the block that was applied
///
/// ### Returns
///
/// `Ok(())`, or [CoreErrors::InvalidArgument] naming both shapes.
fn check_operator_output(out: &Mat<f64>, n: usize, width: usize) -> Result<(), CoreErrors> {
    if out.nrows() != n || out.ncols() != width {
        return Err(CoreErrors::InvalidArgument(format!(
            "block Lanczos: operator returned {} x {}, expected {n} x {width}",
            out.nrows(),
            out.ncols()
        )));
    }
    Ok(())
}

/// Leading eigenpairs of a small operator, formed densely.
///
/// ### Params
///
/// * `n` - Dimension of the operator.
/// * `n_keep` - Eigenpairs to return, at most `n`.
/// * `b` - Columns per operator call.
/// * `op` - Symmetric operator, returns `A X` for an `n x w` block `X`.
///
/// ### Returns
///
/// [RitzPairs] with `n_keep` exact pairs, zero residual and zero restarts.
fn dense_eigenpairs<E, M>(n: usize, n_keep: usize, b: usize, op: &mut M) -> Result<RitzPairs, E>
where
    E: From<CoreErrors>,
    M: FnMut(MatRef<f64>) -> Result<Mat<f64>, E>,
{
    let mut a = Mat::<f64>::zeros(n, n);
    for col in (0..n).step_by(b) {
        let w = b.min(n - col);
        let unit = Mat::<f64>::from_fn(n, w, |i, c| if i == col + c { 1.0 } else { 0.0 });
        let out = op(unit.as_ref())?;
        check_operator_output(&out, n, w)?;
        a.subcols_mut(col, w).copy_from(&out);
    }
    // An f32 operator is only symmetric to round-off.
    let sym = Mat::<f64>::from_fn(n, n, |i, j| 0.5 * (a[(i, j)] + a[(j, i)]));

    let (values, vectors) = symmetric_eigen_descending(sym.as_ref(), n)?;
    let flat: Vec<f64> = (0..n_keep)
        .flat_map(|c| vectors.col(c).iter().copied().collect::<Vec<_>>())
        .collect();

    Ok(RitzPairs {
        values: values[..n_keep].to_vec(),
        vectors: flat,
        residual: 0.0,
        norm_estimate: values[0].abs().max(values[n - 1].abs()),
        converged: true,
        restarts: 0,
    })
}

/// Euclidean norm of every column.
///
/// ### Params
///
/// * `x` - Matrix whose columns to measure.
///
/// ### Returns
///
/// One norm per column.
fn column_norms(x: MatRef<f64>) -> Vec<f64> {
    (0..x.ncols()).map(|c| x.col(c).norm_l2()).collect()
}

/// Orthonormalise a block against itself, column by column.
///
/// Two classical Gram-Schmidt passes per column against the columns before it.
/// A column whose remainder falls below `threshold` is dependent: it gets a
/// zero diagonal in `R` and a random replacement, orthogonalised twice against
/// `basis` and the columns so far. Later columns are still projected onto the
/// replacement and the coefficients land in `R`, so `W = Q R` holds exactly
/// up to the dropped remainders, which is what keeps the Arnoldi relation
/// intact. A Householder QR cannot do this without pivoting: zeroing a row of
/// its `R` also drops the off-diagonal terms of every later column.
///
/// ### Params
///
/// * `w` - Block to orthonormalise, `n x b`, already orthogonal to `basis`.
/// * `basis` - Basis the replacements must stay orthogonal to.
/// * `threshold` - Remainder norm at or below which a column is dependent.
/// * `rng` - Source of the replacement vectors.
///
/// ### Returns
///
/// `(q, r, n_deficient)`: orthonormal `q` (`n x b`), upper triangular `r`
/// (`b x b`) and the number of dependent columns.
fn orthonormalise_block(
    w: MatRef<f64>,
    basis: MatRef<f64>,
    threshold: f64,
    rng: &mut StdRng,
) -> (Mat<f64>, Mat<f64>, usize) {
    let (n, b) = w.shape();
    let mut q = w.to_owned();
    let mut r = Mat::<f64>::zeros(b, b);
    let mut n_deficient = 0usize;

    for c in 0..b {
        for _ in 0..2 {
            for i in 0..c {
                let d = q.col(i).transpose() * q.col(c);
                r[(i, c)] += d;
                for k in 0..n {
                    q[(k, c)] -= d * q[(k, i)];
                }
            }
        }

        let remainder = q.col(c).norm_l2();
        if remainder > threshold {
            r[(c, c)] = remainder;
            let inv = 1.0 / remainder;
            q.col_mut(c).iter_mut().for_each(|x| *x *= inv);
            continue;
        }

        n_deficient += 1;
        let mut x = Mat::<f64>::from_fn(n, 1, |_, _| rng.random::<f64>() - 0.5);
        for _ in 0..2 {
            let mut coeffs = Mat::<f64>::zeros(basis.ncols(), 1);
            gemm(
                coeffs.as_mut(),
                Accum::Replace,
                basis.transpose(),
                x.as_ref(),
                1.0,
                Par::Seq,
            );
            gemm(
                x.as_mut(),
                Accum::Add,
                basis,
                coeffs.as_ref(),
                -1.0,
                Par::Seq,
            );
            for i in 0..c {
                let d = q.col(i).transpose() * x.col(0);
                for k in 0..n {
                    x[(k, 0)] -= d * q[(k, i)];
                }
            }
        }
        let scale = x.col(0).norm_l2();
        for k in 0..n {
            q[(k, c)] = x[(k, 0)] / scale;
        }
    }

    (q, r, n_deficient)
}

/////////////////////////////////////
// Block Lanczos: public functions //
/////////////////////////////////////

/// Largest eigenpairs of a symmetric operator via block thick-restarted
/// Lanczos.
///
/// The block twin of [lanczos_thick_restart], with the same structure: an
/// explicit projection `H = V^T A V` (arrow-shaped after a restart),
/// classical Gram-Schmidt with a DGKS-conditional second pass, Rayleigh-Ritz,
/// and a thick restart that keeps the best Ritz vectors plus the residual
/// block. The operator is applied to a whole block per call, which is the
/// point: on a device every call carries a fixed cost, and a block of `b`
/// vectors pays it once instead of `b` times.
///
/// New blocks are orthonormalised among themselves by a Householder QR. If
/// that block is rank deficient, the dependent directions are replaced by
/// random vectors orthogonal to everything so far; the dropped residual is
/// below the breakdown threshold, so the Arnoldi relation
/// `A V = V H + Q R E^T` survives. A block that vanishes entirely means the
/// basis spans an invariant subspace and the Ritz pairs are exact.
///
/// Convergence is the exact block residual `||R y_last||`, with `y_last` the
/// last block of the Ritz coefficient vector, against `tol * ||A||`.
///
/// When the basis plus one block would reach the operator dimension, the
/// residual block has no room left to stay orthogonal to the basis. The
/// operator is then formed densely, a block of unit vectors per call, and
/// decomposed exactly; that costs no more calls than a single cycle.
///
/// Internals are f64 regardless of the operator's precision.
///
/// ### Params
///
/// * `n` - Dimension of the operator.
/// * `n_components` - Number of eigenpairs to compute. Capped at `n`.
/// * `seed` - Seed for the starting block.
/// * `params` - [BlockLanczosParams].
/// * `op` - Symmetric operator, returns `A X` for an `n x b` block `X`.
///
/// ### Returns
///
/// [RitzPairs] with at most `min(n_components, n)` pairs, fewer if the
/// iteration hit an invariant subspace first.
pub fn block_lanczos_thick_restart<E, M>(
    n: usize,
    n_components: usize,
    seed: u64,
    params: BlockLanczosParams,
    mut op: M,
) -> Result<RitzPairs, E>
where
    E: From<CoreErrors>,
    M: FnMut(MatRef<f64>) -> Result<Mat<f64>, E>,
{
    if n_components == 0 || n == 0 {
        return Err(CoreErrors::MustBePositive(
            "n_components and the operator dimension".to_string(),
        )
        .into());
    }

    let n_keep = n_components.min(n);
    let (b, m, keep) = params.resolve(n_keep, n);
    let par = faer_parallelism();

    if m + b > n {
        return dense_eigenpairs(n, n_keep, b, &mut op);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut basis = Mat::<f64>::zeros(n, m);
    let mut h = Mat::<f64>::zeros(m, m);
    let start = Mat::<f64>::from_fn(n, b, |_, _| rng.random::<f64>() - 0.5);
    basis
        .subcols_mut(0, b)
        .copy_from(start.qr().compute_thin_Q());

    // Residual block of the last cycle, `A V = V H + resid_q resid_r E^T`.
    let mut resid_q = Mat::<f64>::zeros(n, b);
    let mut resid_r = Mat::<f64>::zeros(b, b);

    let mut locked = 0usize;
    let mut ritz_values: Vec<f64> = Vec::new();
    let mut ritz_coeffs = Mat::<f64>::zeros(0, 0);
    let mut m_final = 0usize;

    let mut norm_estimate = 0.0f64;
    let mut residual = f64::INFINITY;
    let mut converged = false;
    let mut restarts = 0usize;

    let budget = params.max_restarts.max(1);

    for cycle in 0..budget {
        restarts = cycle + 1;
        let mut m_eff = m;
        let mut invariant = false;

        let mut j = locked;
        while j < m {
            let width = j + b;
            let mut w = op(basis.subcols(j, b))?;
            check_operator_output(&w, n, b)?;

            // Rayleigh-Ritz coefficients against the basis so far, taken before
            // any subtraction so `h` is the exact projection.
            let v = basis.subcols(0, width);
            let mut coeffs = Mat::<f64>::zeros(width, b);
            gemm(
                coeffs.as_mut(),
                Accum::Replace,
                v.transpose(),
                w.as_ref(),
                1.0,
                par,
            );
            for c in 0..b {
                for i in 0..width {
                    h[(i, j + c)] = coeffs[(i, c)];
                    h[(j + c, i)] = coeffs[(i, c)];
                }
            }

            let before = column_norms(w.as_ref());
            norm_estimate = before.iter().fold(norm_estimate, |acc, &x| acc.max(x));

            gemm(w.as_mut(), Accum::Add, v, coeffs.as_ref(), -1.0, par);
            let after = column_norms(w.as_ref());

            if after
                .iter()
                .zip(&before)
                .any(|(&a, &bf)| a < LANCZOS_ORTHO_REFINE_RATIO * bf)
            {
                gemm(
                    coeffs.as_mut(),
                    Accum::Replace,
                    v.transpose(),
                    w.as_ref(),
                    1.0,
                    par,
                );
                gemm(w.as_mut(), Accum::Add, v, coeffs.as_ref(), -1.0, par);
            }

            let threshold = LANCZOS_BREAKDOWN_RATIO * norm_estimate;
            let (q, r, n_deficient) = orthonormalise_block(w.as_ref(), v, threshold, &mut rng);

            if n_deficient == b {
                m_eff = width;
                invariant = true;
                break;
            }

            if width < m {
                basis.subcols_mut(width, b).copy_from(&q);
            } else {
                resid_q = q;
                resid_r = r;
            }
            j = width;
        }

        let (values, vectors) = symmetric_eigen_descending(h.as_ref(), m_eff)?;

        let n_out = n_keep.min(m_eff);
        residual = if invariant {
            0.0
        } else {
            let mut y_last = Mat::<f64>::zeros(b, n_out);
            y_last.copy_from(vectors.submatrix(m_eff - b, 0, b, n_out));
            let mut ry = Mat::<f64>::zeros(b, n_out);
            gemm(
                ry.as_mut(),
                Accum::Replace,
                resid_r.as_ref(),
                y_last.as_ref(),
                1.0,
                Par::Seq,
            );
            column_norms(ry.as_ref()).into_iter().fold(0.0f64, f64::max)
        };

        norm_estimate = norm_estimate
            .max(values[0].abs())
            .max(values[m_eff - 1].abs());
        converged = residual <= params.tol * norm_estimate;

        ritz_values = values;
        ritz_coeffs = vectors;
        m_final = m_eff;

        if converged || invariant || cycle + 1 == budget {
            break;
        }

        // Thick restart: best `keep` Ritz vectors, then the residual block.
        let mut restarted = Mat::<f64>::zeros(n, keep);
        gemm(
            restarted.as_mut(),
            Accum::Replace,
            basis.subcols(0, m_eff),
            ritz_coeffs.subcols(0, keep),
            1.0,
            par,
        );

        h.fill(0.0);
        basis.subcols_mut(0, keep).copy_from(&restarted);
        for (i, &value) in ritz_values.iter().take(keep).enumerate() {
            h[(i, i)] = value;
        }
        basis.subcols_mut(keep, b).copy_from(&resid_q);
        locked = keep;
    }

    let n_out = n_keep.min(m_final);
    let mut ritz = Mat::<f64>::zeros(n, n_out);
    gemm(
        ritz.as_mut(),
        Accum::Replace,
        basis.subcols(0, m_final),
        ritz_coeffs.subcols(0, n_out),
        1.0,
        par,
    );

    let mut vectors = vec![0.0f64; n * n_out];
    for (c, column) in vectors.chunks_exact_mut(n).enumerate() {
        let src = ritz.col(c);
        let scale = src.norm_l2();
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (dst, &v) in column.iter_mut().zip(src.iter()) {
            *dst = v * inv;
        }
    }

    Ok(RitzPairs {
        values: ritz_values[..n_out].to_vec(),
        vectors,
        residual,
        norm_estimate,
        converged,
        restarts,
    })
}

/////////////////
// Lanczos SVD //
/////////////////

///////////////////////////////////
// Lanczos SVD: public functions //
///////////////////////////////////

/// Compute sparse SVD using Lanczos on A^T A or AA^T
///
/// Thick-restarted with the default [LanczosParams], so the trailing
/// components converge on a flat spectrum too.
///
/// ### Params
///
/// * `matrix` - Sparse matrix (CSR or CSC)
/// * `n_components` - Number of singular values/vectors to compute
/// * `seed` - For reproducibility
/// * `use_second_layer` - If true, use data_2 instead of data
/// * `col_means` - Optional column means for implicit mean centering
/// * `col_stds` - Optional column sds for implicit variance normalising
/// * `row_offsets` - Additional offsets (for example for CLR-type PCA in single
///   cell).
///
/// ### Returns
///
/// `SvdResults` containing U (n×k), S (length k), and V (m×k)
pub fn sparse_svd_lanczos<T, U, F>(
    matrix: &CompressedSparseData2<T, U>,
    n_components: usize,
    seed: u64,
    use_second_layer: bool,
    col_means: Option<&[F]>,
    col_stds: Option<&[F]>,
    row_offsets: Option<&[F]>,
) -> Result<SvdResults<F>, CoreErrors>
where
    T: BixverseNumeric + BixverseSimd + Into<F> + Clone,
    U: BixverseNumeric + Into<F> + Clone,
    F: BixverseFloat + BixverseSimd + std::iter::Sum,
{
    let (n, m) = matrix.shape;
    let use_ata = n > m;
    let krylov_dim = if use_ata { m } else { n };

    // keep both representations to make matvec operations fast
    let (csr, csc);
    let csr_owned;
    let csc_owned;

    match matrix.cs_type {
        CompressedSparseFormat::Csr => {
            csr = matrix;
            csc_owned = matrix.transform_single_layer(use_second_layer)?;
            csc = &csc_owned;
        }
        CompressedSparseFormat::Csc => {
            csc = matrix;
            csr_owned = matrix.transform_single_layer(use_second_layer)?;
            csr = &csr_owned;
        }
    };

    // helper to extract the right data layer and cast to F
    let extract_data = |mat: &CompressedSparseData2<T, U>| -> Vec<F> {
        if use_second_layer {
            mat.data_2
                .as_ref()
                .expect("data_2 is None but use_second_layer is true")
                .iter()
                .copied()
                .map(|v| v.into())
                .collect()
        } else {
            mat.data.iter().copied().map(|v| v.into()).collect()
        }
    };

    let data_csr_f = extract_data(csr);
    let data_csc_f = extract_data(csc);

    // matrix-vector product for A (using CSR)
    let matvec_a = |x: &[F], y: &mut [F]| {
        let x_scaled: Vec<F> = if let Some(sd) = col_stds {
            x.iter().enumerate().map(|(j, &v)| v / sd[j]).collect()
        } else {
            x.to_vec()
        };
        let mean_dot: F = if let Some(mu) = col_means {
            x_scaled.iter().enumerate().map(|(j, &v)| mu[j] * v).sum()
        } else {
            F::zero()
        };
        let x_sum: F = if row_offsets.is_some() {
            x_scaled.iter().copied().sum()
        } else {
            F::zero()
        };

        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut sum = F::zero();
            for idx in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                let j = csr.indices[idx] as usize;
                sum += data_csr_f[idx] * x_scaled[j];
            }
            if col_means.is_some() {
                sum -= mean_dot;
            }
            if let Some(off) = row_offsets {
                sum -= off[i] * x_sum;
            }
            *yi = sum;
        });
    };

    let matvec_at = |x: &[F], y: &mut [F]| {
        let x_sum: F = x.iter().copied().sum();
        let m_dot_x: F = if let Some(off) = row_offsets {
            off.iter().zip(x.iter()).map(|(&m_i, &xi)| m_i * xi).sum()
        } else {
            F::zero()
        };

        y.par_iter_mut().enumerate().for_each(|(j, yj)| {
            let mut sum = F::zero();
            for idx in csc.indptr[j] as usize..csc.indptr[j + 1] as usize {
                let i = csc.indices[idx] as usize;
                sum += data_csc_f[idx] * x[i];
            }
            if let Some(mu) = col_means {
                sum -= mu[j] * x_sum;
            }
            if row_offsets.is_some() {
                sum -= m_dot_x;
            }
            if let Some(sd) = col_stds {
                sum /= sd[j];
            }
            *yj = sum;
        });
    };

    // select Gram matrix operator
    #[allow(clippy::type_complexity)]
    let matvec_gram: Box<dyn Fn(&[F], &mut [F]) + Sync> = if use_ata {
        Box::new(|x: &[F], y: &mut [F]| {
            let mut temp = vec![F::zero(); n];
            matvec_a(x, &mut temp);
            matvec_at(&temp, y);
        })
    } else {
        Box::new(|x: &[F], y: &mut [F]| {
            let mut temp = vec![F::zero(); m];
            matvec_at(x, &mut temp);
            matvec_a(&temp, y);
        })
    };

    // the restarted solver runs in f64; the Gram operator stays in F
    let matvec_gram_f64 = |x: &[f64], y: &mut [f64]| {
        let x_f: Vec<F> = x
            .iter()
            .map(|&v| F::from(v).expect("f64 casts into any float type"))
            .collect();
        let mut y_f = vec![F::zero(); krylov_dim];
        matvec_gram(&x_f, &mut y_f);
        for (dst, src) in y.iter_mut().zip(y_f) {
            *dst = src.to_f64().expect("float casts into f64");
        }
    };

    let ritz = lanczos_thick_restart(
        krylov_dim,
        n_components,
        seed,
        LanczosParams::default(),
        matvec_gram_f64,
    )?;

    let mut singular_values: Vec<F> = Vec::with_capacity(n_components);
    let mut u_vecs: Vec<Vec<F>> = Vec::with_capacity(n_components);
    let mut v_vecs: Vec<Vec<F>> = Vec::with_capacity(n_components);

    for (idx, &eval) in ritz.values.iter().enumerate() {
        if eval <= 0.0 {
            continue;
        }

        let sigma = F::from(eval.sqrt()).expect("f64 casts into any float type");
        singular_values.push(sigma);

        let gram_evec: Vec<F> = ritz.vectors[idx * krylov_dim..(idx + 1) * krylov_dim]
            .iter()
            .map(|&v| F::from(v).expect("f64 casts into any float type"))
            .collect();

        if use_ata {
            let mut u_vec = vec![F::zero(); n];
            matvec_a(&gram_evec, &mut u_vec);
            for x in &mut u_vec {
                *x /= sigma;
            }
            u_vecs.push(u_vec);
            v_vecs.push(gram_evec);
        } else {
            let mut v_vec = vec![F::zero(); m];
            matvec_at(&gram_evec, &mut v_vec);
            for x in &mut v_vec {
                *x /= sigma;
            }
            u_vecs.push(gram_evec);
            v_vecs.push(v_vec);
        }
    }

    let u = Mat::from_fn(n, singular_values.len(), |i, j| u_vecs[j][i]);
    let v = Mat::from_fn(m, singular_values.len(), |i, j| v_vecs[j][i]);

    Ok(SvdResults {
        u,
        s: singular_values,
        v,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::vector_helpers::rank_vector;
    use faer::Mat;

    /// The zero counts and the dense-to-CSR conversion agree on what is structurally zero.
    #[test]
    fn test_from_dense_and_count_zeroes() {
        let mat: Mat<f64> = Mat::from_fn(3, 2, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
        let (total_zeroes, row_zeroes, col_zeroes) = count_zeroes(&mat.as_ref());

        assert_eq!(total_zeroes, 4);
        assert_eq!(row_zeroes, vec![1, 1, 2]);
        assert_eq!(col_zeroes, vec![2, 2]);

        let csr = CompressedSparseData2::<f64, f64>::from_dense_matrix(
            mat.as_ref(),
            CompressedSparseFormat::Csr,
        );
        assert_eq!(csr.shape, (3, 2));
        assert_eq!(csr.data, vec![1.0, 2.0]);
    }

    /// Lanczos recovers the one nonzero eigenpair of a symmetric rank-one matrix.
    #[test]
    fn test_lanczos_eigenpairs_logic() {
        // Symmetric rank-1 matrix M = x * x^T
        // Let x = [1.0, 0.0, 2.0, 0.0]^T
        let data = vec![1.0, 2.0, 2.0, 4.0];
        let indices = vec![0, 2, 0, 2];
        let indptr = vec![0, 2, 2, 4, 4]; // Rows 1 and 3 are empty
        let shape = (4, 4);

        let csr = CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, shape);

        // Lanczos expects symmetric matrix, this one is symmetric
        let (evals, evecs) = compute_largest_eigenpairs_lanczos(&csr, 1, 42, None).unwrap();

        // True top eigenvalue should be exactly sum(x_i^2) = 1.0 + 4.0 = 5.0
        assert!((evals[0] - 5.0).abs() < 1e-3);

        // Eigenvectors are returned transposed: evecs[point_idx][comp_idx]
        // So the first principal component is [evecs[0][0], evecs[1][0], evecs[2][0], evecs[3][0]]
        let x_norm = 5.0_f32.sqrt();
        let dot_x = (evecs[0][0] * 1.0 + evecs[2][0] * 2.0) / x_norm;

        assert!(dot_x.abs() > 0.999);
    }

    /// Lanczos SVD recovers both singular vectors of a sparse rank-one matrix.
    #[test]
    fn test_sparse_svd_lanczos_logic() {
        // Sparse rank-1 matrix A = x * y^T
        // x = [0.0, 2.0, 0.0, 4.0]^T
        // y = [1.0, 0.0, 0.5]^T
        let data = vec![2.0, 1.0, 4.0, 2.0];
        let indices = vec![0, 2, 0, 2];
        let indptr = vec![0, 0, 2, 2, 4];
        let shape = (4, 3);

        let csr = CompressedSparseData2::<f64, f64>::new_csr(&data, &indices, &indptr, None, shape);
        let no_params: Option<&[f64]> = None;

        let svd = sparse_svd_lanczos(&csr, 1, 42, false, no_params, no_params, None).unwrap();

        // Test correlation with theoretical U
        let u_col = svd.u.col(0);
        let x_norm = 20.0_f64.sqrt();
        let dot_u = (u_col[1] * 2.0 + u_col[3] * 4.0) / x_norm;
        assert!(dot_u.abs() > 0.999);

        // Test correlation with theoretical V
        let v_col = svd.v.col(0);
        let y_norm = 1.25_f64.sqrt();
        let dot_v = (v_col[0] * 1.0 + v_col[2] * 0.5) / y_norm;
        assert!(dot_v.abs() > 0.999);
    }

    /// Thirty components of a centred uniform noise matrix match a dense SVD.
    ///
    /// The spectrum is flat past the first component, which is where a single
    /// unrestarted Lanczos pass leaves the trailing pairs unconverged.
    #[test]
    fn test_sparse_svd_lanczos_flat_spectrum() {
        let (n, m, k) = (400usize, 200usize, 30usize);
        let mut rng = StdRng::seed_from_u64(1);
        let vals: Vec<f64> = (0..n * m).map(|_| rng.random::<f64>()).collect();
        let indices: Vec<u32> = (0..n).flat_map(|_| 0..m as u32).collect();
        let indptr: Vec<u32> = (0..=n).map(|i| (i * m) as u32).collect();
        let csr =
            CompressedSparseData2::<f64, f64>::new_csr(&vals, &indices, &indptr, None, (n, m));

        let means: Vec<f64> = (0..m)
            .map(|j| (0..n).map(|i| vals[i * m + j]).sum::<f64>() / n as f64)
            .collect();
        let svd = sparse_svd_lanczos(&csr, k, 42, false, Some(&means[..]), None, None).unwrap();

        let centred = Mat::<f64>::from_fn(n, m, |i, j| vals[i * m + j] - means[j]);
        let reference = centred.thin_svd().unwrap();
        let s_ref = reference.S().column_vector();

        assert_eq!(svd.s.len(), k);
        for i in 0..k {
            assert_relative_eq!(svd.s[i], s_ref[i], max_relative = 1e-6);
        }
    }

    use approx::assert_relative_eq;

    /// Adjacency matrix of a path graph on `n` nodes.
    ///
    /// Eigenvalues are `2 cos(k pi / (n + 1))`, separated by `O(1 / n^2)`, which
    /// makes it the cheapest genuinely clustered spectrum to build.
    fn path_graph(n: usize) -> CompressedSparseData2<f64, f64> {
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for i in 0..n - 1 {
            rows[i].push(((i + 1) as u32, 1.0));
            rows[i + 1].push((i as u32, 1.0));
        }
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in rows.iter_mut() {
            row.sort_by_key(|&(j, _)| j);
            for &(j, v) in row.iter() {
                indices.push(j);
                data.push(v);
            }
            indptr.push(indices.len() as u32);
        }

        CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n))
    }

    /// Diagonal matrix in CSR, storing every entry given, zeros included.
    fn diagonal(diag: &[f64]) -> CompressedSparseData2<f64, f64> {
        let indices: Vec<u32> = (0..diag.len() as u32).collect();
        let indptr: Vec<u32> = (0..=diag.len() as u32).collect();

        CompressedSparseData2::new_csr(diag, &indices, &indptr, None, (diag.len(), diag.len()))
    }

    /// `||A x - lambda x||` for the `k`-th returned pair, with `vectors` indexed `[i][k]`.
    fn pair_residual(
        mat: &CompressedSparseData2<f64, f64>,
        values: &[f64],
        vectors: &[Vec<f32>],
        k: usize,
    ) -> f64 {
        let n = mat.shape.0;
        let v: Vec<f64> = (0..n).map(|i| vectors[i][k] as f64).collect();
        let lam = values[k];

        (0..n)
            .map(|i| {
                let mut sum = 0.0;
                for x in mat.indptr[i] as usize..mat.indptr[i + 1] as usize {
                    sum += mat.data[x] * v[mat.indices[x] as usize];
                }
                (sum - lam * v[i]).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Regression: a tightly clustered spectrum used to come back as noise.
    #[test]
    fn test_lanczos_resolves_a_clustered_spectrum_at_defaults() {
        let n = 300usize;
        let mat = path_graph(n);

        let (values, vectors) = compute_largest_eigenpairs_lanczos(&mat, 4, 42, None).unwrap();

        for k in 0..4 {
            let expected = 2.0 * ((k + 1) as f64 * std::f64::consts::PI / (n + 1) as f64).cos();
            assert_relative_eq!(values[k], expected, epsilon = 1e-3);
        }

        // The leading eigenvector is sin(pi x / (n + 1)), so it has no sign
        // change anywhere. Noise does.
        let sign = vectors[0][0].signum();
        assert!(
            (0..n).all(|i| vectors[i][0].signum() == sign || vectors[i][0] == 0.0),
            "leading eigenvector of a path graph changed sign"
        );
    }

    /// Given a big enough restart budget the same clustered spectrum does converge.
    #[test]
    fn test_lanczos_restarts_converge_a_clustered_spectrum() {
        // Same fixture with a budget large enough to actually converge, which
        // is what the restart machinery exists for. 300 nodes, ~40 cycles.
        let n = 300usize;
        let mat = path_graph(n);
        let params = LanczosParams {
            basis_size: None,
            max_restarts: 64,
            tol: 1e-8,
        };

        let res = compute_largest_eigenpairs_lanczos_diag(&mat, 4, 42, Some(params)).unwrap();

        assert!(res.converged, "residual stalled at {:e}", res.residual);
        for k in 0..4 {
            let expected = 2.0 * ((k + 1) as f64 * std::f64::consts::PI / (n + 1) as f64).cos();
            assert_relative_eq!(res.eigenvalues[k], expected, epsilon = 1e-6);

            let residual = pair_residual(&mat, &res.eigenvalues, &res.eigenvectors, k);
            assert!(residual < 1e-5, "pair {k} has residual {residual:e}");
        }
    }

    /// Regression: asking for more components than the matrix has rows used to panic.
    #[test]
    fn test_lanczos_caps_components_at_the_matrix_dimension() {
        // Used to panic inside `clamp` before it ever got to the iteration.
        let mat = diagonal(&[4.0, 3.0, 2.0, 1.0]);

        for requested in [4usize, 5, 40] {
            let (values, vectors) =
                compute_largest_eigenpairs_lanczos(&mat, requested, 11, None).unwrap();

            assert_eq!(values.len(), 4, "requested {requested}");
            assert_eq!(vectors[0].len(), values.len(), "requested {requested}");
            assert_relative_eq!(values[0], 4.0, epsilon = 1e-9);
            assert_relative_eq!(values[3], 1.0, epsilon = 1e-9);
        }
    }

    /// Regression: an early Krylov breakdown used to leave values and vectors different lengths.
    #[test]
    fn test_lanczos_breakdown_keeps_values_and_vectors_the_same_length() {
        // Rank two, so the Krylov space closes after three vectors and there is
        // no way to produce the ten pairs asked for. Both outputs must agree on
        // how many there are; they used to disagree and index out of bounds
        // downstream.
        let mut diag = vec![0.0f64; 12];
        diag[0] = 5.0;
        diag[1] = 3.0;
        let mat = diagonal(&diag);

        let res = compute_largest_eigenpairs_lanczos_diag(&mat, 10, 3, None).unwrap();

        assert!(res.eigenvalues.len() < 10, "expected an early breakdown");
        assert_eq!(res.eigenvalues.len(), res.eigenvectors[0].len());
        assert_relative_eq!(res.eigenvalues[0], 5.0, epsilon = 1e-9);
        assert_relative_eq!(res.eigenvalues[1], 3.0, epsilon = 1e-9);
    }

    /// A solve that ran out of restarts has to report itself as unconverged.
    #[test]
    fn test_lanczos_reports_budget_exhaustion() {
        // One cycle on a path graph cannot converge, and the caller has to be
        // able to tell that apart from a solve that did.
        let mat = path_graph(300);
        let params = LanczosParams {
            basis_size: None,
            max_restarts: 1,
            tol: 1e-8,
        };

        let res = compute_largest_eigenpairs_lanczos_diag(&mat, 4, 42, Some(params)).unwrap();

        assert!(!res.converged);
        assert_eq!(res.restarts, 1);
        assert!(res.residual > params.tol * res.norm_estimate);
    }

    /// Scaling the whole matrix must not change whether the solve converges.
    #[test]
    fn test_lanczos_tolerance_is_relative_to_the_matrix_norm() {
        // Scaling the matrix by 1e8 scales every residual by 1e8 too. An
        // absolute tolerance would make the scaled problem unconvergeable; a
        // relative one converges both in the same number of cycles.
        let small = diagonal(&[4.0, 3.0, 2.0, 1.0]);
        let large = diagonal(&[4e8, 3e8, 2e8, 1e8]);

        let a = compute_largest_eigenpairs_lanczos_diag(&small, 2, 5, None).unwrap();
        let b = compute_largest_eigenpairs_lanczos_diag(&large, 2, 5, None).unwrap();

        assert!(a.converged && b.converged);
        assert_relative_eq!(b.eigenvalues[0] / a.eigenvalues[0], 1e8, epsilon = 1e-1);
    }

    /// A basis smaller than the default still resolves the top eigenvalues via restarts.
    #[test]
    fn test_lanczos_params_basis_size_is_honoured() {
        let n = 40usize;
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            indices.push(i as u32);
            data.push((i + 1) as f64);
            indptr.push(indices.len() as u32);
        }
        let mat = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n));

        // Diagonal matrix: the largest eigenvalue is n regardless of the basis.
        let tight = LanczosParams {
            basis_size: Some(6),
            max_restarts: 200,
            tol: 1e-10,
        };
        let (values, _) = compute_largest_eigenpairs_lanczos(&mat, 3, 7, Some(tight)).unwrap();

        assert_relative_eq!(values[0], 40.0, epsilon = 1e-8);
        assert_relative_eq!(values[1], 39.0, epsilon = 1e-8);
        assert_relative_eq!(values[2], 38.0, epsilon = 1e-8);
    }

    ///////////////////
    // Block Lanczos //
    ///////////////////

    /// Block operator over a symmetric CSR, as the GPU drivers present it.
    fn block_op(
        mat: &CompressedSparseData2<f64, f64>,
        x: MatRef<f64>,
    ) -> Result<Mat<f64>, CoreErrors> {
        let n = mat.shape.0;
        Ok(Mat::from_fn(n, x.ncols(), |i, c| {
            (mat.indptr[i] as usize..mat.indptr[i + 1] as usize)
                .map(|k| mat.data[k] * x[(mat.indices[k] as usize, c)])
                .sum()
        }))
    }

    /// Block Lanczos converges a clustered spectrum to the same values as the scalar solver.
    #[test]
    fn test_block_lanczos_matches_scalar_on_a_clustered_spectrum() {
        let n = 300usize;
        let mat = path_graph(n);
        let scalar = compute_largest_eigenpairs_lanczos_diag(&mat, 4, 42, None).unwrap();

        // Wider blocks need more restarts here: at a fixed basis they buy a
        // shallower Krylov space per cycle, and this spectrum is clustered.
        for block_size in [1usize, 4, 16] {
            let params = BlockLanczosParams {
                block_size,
                basis_size: None,
                max_restarts: 1000,
                tol: 1e-8,
            };
            let res = block_lanczos_thick_restart(n, 4, 42, params, |x| block_op(&mat, x)).unwrap();

            assert!(
                res.converged,
                "b = {block_size}: residual {:e}",
                res.residual
            );
            for k in 0..4 {
                assert_relative_eq!(res.values[k], scalar.eigenvalues[k], epsilon = 1e-7);
                let v: Vec<Vec<f32>> = (0..n)
                    .map(|i| vec![res.vectors[k * n + i] as f32])
                    .collect();
                let residual = pair_residual(&mat, &res.values[k..k + 1], &v, 0);
                assert!(residual < 1e-5, "b = {block_size}, pair {k}: {residual:e}");
            }
        }
    }

    /// A low-rank operator closes the Krylov space mid-block; the dependent directions are replaced.
    #[test]
    fn test_block_lanczos_low_rank_recovers_the_nonzero_pairs() {
        // Rank two with a block of four: the first residual block is rank
        // deficient, so the replacement path runs before the space closes.
        let mut diag = vec![0.0f64; 200];
        diag[0] = 5.0;
        diag[1] = 3.0;
        let mat = diagonal(&diag);
        let params = BlockLanczosParams {
            block_size: 4,
            ..BlockLanczosParams::new()
        };

        let res = block_lanczos_thick_restart(200, 2, 3, params, |x| block_op(&mat, x)).unwrap();

        assert!(res.converged);
        assert_relative_eq!(res.values[0], 5.0, epsilon = 1e-9);
        assert_relative_eq!(res.values[1], 3.0, epsilon = 1e-9);
        assert_relative_eq!(res.vectors[0].abs(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(res.vectors[200 + 1].abs(), 1.0, epsilon = 1e-6);
    }

    /// An operator too small for Lanczos is solved densely, every pair coming back.
    #[test]
    fn test_block_lanczos_caps_the_block_at_the_dimension() {
        let mat = diagonal(&[4.0, 3.0, 2.0, 1.0]);
        let params = BlockLanczosParams {
            block_size: 64,
            ..BlockLanczosParams::new()
        };

        let res = block_lanczos_thick_restart(4, 10, 1, params, |x| block_op(&mat, x)).unwrap();

        assert_eq!(res.values.len(), 4);
        assert_eq!(res.vectors.len(), 16);
        assert_relative_eq!(res.values[0], 4.0, epsilon = 1e-9);
        assert_relative_eq!(res.values[3], 1.0, epsilon = 1e-9);
    }

    /// Regression: a basis one block short of the dimension used to fill the residual block with noise.
    #[test]
    fn test_block_lanczos_ragged_dimension_is_exact() {
        // 50 is not a multiple of 16, and 48 basis vectors leave the residual
        // block 2 dimensions to live in. Replacements there were noise and the
        // leading value came out several times too large.
        let diag: Vec<f64> = (0..50).map(|i| 1.0 + i as f64).collect();
        let mat = diagonal(&diag);
        let params = BlockLanczosParams {
            block_size: 16,
            ..BlockLanczosParams::new()
        };

        let res = block_lanczos_thick_restart(50, 3, 9, params, |x| block_op(&mat, x)).unwrap();

        assert!(res.converged);
        assert_relative_eq!(res.values[0], 50.0, epsilon = 1e-9);
        assert_relative_eq!(res.values[1], 49.0, epsilon = 1e-9);
        assert_relative_eq!(res.values[2], 48.0, epsilon = 1e-9);
    }

    /// An operator returning the wrong shape is an error, not a panic.
    #[test]
    fn test_block_lanczos_rejects_a_misshapen_operator() {
        let res = block_lanczos_thick_restart::<CoreErrors, _>(
            10,
            2,
            1,
            BlockLanczosParams::new(),
            |_| Ok(Mat::<f64>::zeros(10, 1)),
        );

        assert!(matches!(res, Err(CoreErrors::InvalidArgument(_))));
    }

    /// Build a small random CSR operator plus a dense block, both deterministic.
    fn dense_block_fixture(
        n: usize,
        k: usize,
        width: usize,
        seed: u64,
    ) -> (CompressedSparseData2<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indptr = vec![0u32];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f64> = Vec::new();

        for _ in 0..n {
            // Ascending, unique column indices; some rows deliberately empty.
            for j in 0..k {
                if rng.random::<f64>() < 0.35 {
                    indices.push(j as u32);
                    data.push(rng.random::<f64>() * 2.0 - 1.0);
                }
            }
            indptr.push(indices.len() as u32);
        }

        let csr = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, k));
        let block: Vec<f64> = (0..k * width)
            .map(|_| rng.random::<f64>() * 2.0 - 1.0)
            .collect();

        (csr, block)
    }

    /// The SIMD dense-block kernel must agree with the obvious scalar triple
    /// loop, including on rows that carry no non-zeros at all.
    #[test]
    fn test_csr_matmul_dense_block_matches_scalar() {
        // Widths straddling the 2-, 4-, 8- and 16-lane boundaries so every
        // axpy tail path is hit.
        for width in [1usize, 3, 4, 7, 16, 17, 33] {
            let (csr, block) = dense_block_fixture(11, 9, width, 4242 + width as u64);

            let mut got = vec![0.0f64; 11 * width];
            csr_matmul_dense_block(&csr, &block, width, &mut got).unwrap();

            let mut want = vec![0.0f64; 11 * width];
            for i in 0..11 {
                for p in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                    let j = csr.indices[p] as usize;
                    for c in 0..width {
                        want[i * width + c] += csr.data[p] * block[j * width + c];
                    }
                }
            }

            for (g, w) in got.iter().zip(want.iter()) {
                assert_relative_eq!(g, w, epsilon = 1e-12, max_relative = 1e-12);
            }
        }
    }

    /// Applying the operator `t` times must equal multiplying by the explicit
    /// `T^t`. This is the whole justification for never materialising the power
    /// in MAGIC, so it is worth pinning at a size where the explicit cube is
    /// still legal to build.
    #[test]
    fn test_csr_matmul_dense_block_repeated_matches_explicit_power() {
        let width = 5usize;
        let n = 8usize;
        let (csr, block) = dense_block_fixture(n, n, width, 77);

        // Three applications, ping-ponging between two buffers.
        let mut buf_a = block.clone();
        let mut buf_b = vec![0.0f64; n * width];
        for _ in 0..3 {
            csr_matmul_dense_block(&csr, &buf_a, width, &mut buf_b).unwrap();
            std::mem::swap(&mut buf_a, &mut buf_b);
        }

        // The explicit cube, applied once.
        let sq = csr_matmul_csr(&csr, &csr).unwrap();
        let cube = csr_matmul_csr(&sq, &csr).unwrap();
        let mut direct = vec![0.0f64; n * width];
        csr_matmul_dense_block(&cube, &block, width, &mut direct).unwrap();

        for (rep, dir) in buf_a.iter().zip(direct.iter()) {
            assert_relative_eq!(rep, dir, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    /// A row-stochastic operator preserves a constant column, which is the
    /// invariant that makes imputed values sit on the input's scale.
    #[test]
    fn test_csr_matmul_dense_block_preserves_constant_column() {
        let n = 6usize;
        let width = 3usize;

        // Dense positive rows, so nothing is isolated, then row-normalise.
        let mut indptr = vec![0u32];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                indices.push(j as u32);
                data.push(1.0 + ((i * n + j) % 3) as f64);
            }
            indptr.push(indices.len() as u32);
        }
        let mut csr = CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (n, n));
        normalise_csr_rows_l1(&mut csr).unwrap();

        let block = vec![2.5f64; n * width];
        let mut out = vec![0.0f64; n * width];
        csr_matmul_dense_block(&csr, &block, width, &mut out).unwrap();

        for v in out {
            assert_relative_eq!(v, 2.5, epsilon = 1e-12);
        }
    }

    /// Shape mismatches and a CSC operator must be rejected, not read out of
    /// bounds.
    #[test]
    fn test_csr_matmul_dense_block_rejects_bad_shapes() {
        let (csr, block) = dense_block_fixture(4, 4, 2, 1);
        let mut out = vec![0.0f64; 4 * 2];

        // Wrong width for the block that was built.
        assert!(csr_matmul_dense_block(&csr, &block, 3, &mut out).is_err());
        // Zero width.
        assert!(csr_matmul_dense_block(&csr, &block, 0, &mut out).is_err());
        // Output too short.
        let mut short = vec![0.0f64; 4];
        assert!(csr_matmul_dense_block(&csr, &block, 2, &mut short).is_err());

        let csc = csr.transform();
        assert!(matches!(
            csr_matmul_dense_block(&csc, &block, 2, &mut out),
            Err(CoreErrors::SparseMatrixMustBeCsr)
        ));
    }

    /// The sparse ranking must reproduce the dense one, up to the shift.
    #[test]
    fn test_shifted_ranks_sparse_matches_dense_rank_vector() {
        let cases: Vec<Vec<f32>> = vec![
            vec![3.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![-2.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            vec![-1.0, -3.0, 0.0, 2.0, 4.0, 0.0, 0.0],
            vec![2.0, 2.0, 0.0, 2.0, 0.0, 5.0],
            vec![-4.0, -4.0, 0.0, 0.0, -4.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-1.0, -2.0, -3.0],
            vec![7.0],
        ];

        for dense in cases {
            let n = dense.len();
            let indices: Vec<u32> = (0..n as u32)
                .filter(|&i| dense[i as usize] != 0.0)
                .collect();
            let values: Vec<f32> = indices.iter().map(|&i| dense[i as usize]).collect();

            let got = shifted_ranks_sparse(&values, n);
            let dense_ranks = rank_vector(&dense);
            assert_eq!(got.len(), indices.len());

            if indices.is_empty() {
                continue;
            }

            // The ranks themselves must match the dense ones. Only up to a
            // constant, because the correlation is shift invariant and a fully
            // dense column has no zero block to pin the shift to.
            for (slot, &i) in indices.iter().enumerate() {
                assert_relative_eq!(
                    got[slot] - got[0],
                    dense_ranks[i as usize] - dense_ranks[indices[0] as usize],
                    epsilon = 1e-6
                );
            }

            // Once a structural zero exists the shift is pinned: it has to put
            // that whole tie group at exactly zero.
            if indices.len() < n {
                let r_zero = dense_ranks[(0..n).find(|&i| dense[i] == 0.0).unwrap()];
                for (slot, &i) in indices.iter().enumerate() {
                    assert_relative_eq!(
                        got[slot],
                        dense_ranks[i as usize] - r_zero,
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    /// An explicitly stored zero must land on exactly the same shifted rank as
    /// a structural one, namely zero.
    ///
    /// This is the invariant the closed-form variance and the scatter/gather
    /// both stand on: every position the caller did not store contributes a
    /// known constant. A stored zero that ranked as its own tie group would
    /// break that silently, and CSC data from R does carry explicit zeros.
    #[test]
    fn test_shifted_ranks_sparse_merges_explicit_zeros() {
        // Six rows, four stored, one of them an explicit zero.
        let values: Vec<f32> = vec![-2.0, 0.0, 3.0, 1.0];
        let got = shifted_ranks_sparse(&values, 6);

        assert_relative_eq!(got[1], 0.0, epsilon = 1e-6);
        assert!(got[0] < 0.0, "the negative must rank below the zero block");
        assert!(got[2] > got[3], "3.0 must rank above 1.0");
        assert!(got[3] > 0.0, "positives must rank above the zero block");
    }
}
