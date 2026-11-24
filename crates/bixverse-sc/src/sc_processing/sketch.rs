//! Leverage score sketching of cells, after Seurat's `LeverageScore` and
//! `SketchData`.
//!
//! The statistical leverage of a cell is the squared norm of its row of the
//! left singular vectors: how much it sticks out of the span of the others.
//! Large data takes the randomised route of Clarkson and Woodruff (CountSketch,
//! QR of the sketch, a sparse Johnson-Lindenstrauss projection); small data
//! takes the exact Gram eigenproblem. Scores run over the full feature space,
//! not a rank-k subspace, so outliers and doublets score high: clean first.

use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Mat, MatRef};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::prelude::*;
use bixverse_core::utils::faer_parallelism;

////////////
// Consts //
////////////

/// Singular vectors of the exact fallback, as Seurat's `LeverageScore`
/// hardcodes it.
const LEVERAGE_EXACT_DIMS: usize = 50;

/// Sketch size as a multiple of the feature count. The QR needs a sketch
/// taller than wide, and Seurat raises it to this when the requested size is
/// too small.
const LEVERAGE_SKETCH_FEATURE_RATIO: f64 = 1.1;

/// Cell count, as a multiple of the sketch size, below which the exact route
/// runs. Same threshold as Seurat.
const LEVERAGE_EXACT_CELL_RATIO: f64 = 1.5;

/// Relative eigenvalue below which a direction of the Gram counts as null
/// space in the exact route.
const LEVERAGE_EIGEN_TOL: f64 = 1e-10;

////////////
// Params //
////////////

/// Parameters of [leverage_scores].
#[derive(Clone, Copy, Debug)]
pub struct LeverageParams {
    /// Rows of the CountSketch. Raised to `1.1 * n_features` when smaller.
    pub n_sketch: usize,
    /// Distortion of the Johnson-Lindenstrauss projection, in (0, 1]. Sets
    /// its width to `4 ln(n) / (eps^2 / 2 - eps^3 / 3)`.
    pub eps: f64,
}

impl LeverageParams {
    /// Construct the parameters.
    ///
    /// ### Params
    ///
    /// * `n_sketch` - Rows of the CountSketch
    /// * `eps` - Distortion of the JL projection
    ///
    /// ### Returns
    ///
    /// The [LeverageParams].
    pub fn new(n_sketch: usize, eps: f64) -> Self {
        Self { n_sketch, eps }
    }
}

impl Default for LeverageParams {
    /// Seurat's defaults: a 5000 row sketch and `eps = 0.5`.
    ///
    /// ### Returns
    ///
    /// The defaults.
    fn default() -> Self {
        Self::new(5000, 0.5)
    }
}

/////////////
// Helpers //
/////////////

/// `S X` for a CountSketch `S`: every cell lands in one random row with a
/// random sign.
///
/// Cells are bucketed by row first, so each sketch row is summed by one task
/// and no thread needs a full `n_sketch x n_features` accumulator.
///
/// ### Params
///
/// * `x` - CSR, cells x features
/// * `n_sketch` - Rows of the sketch
/// * `rng` - Random number generator
///
/// ### Returns
///
/// The dense `n_sketch x n_features` sketch.
fn count_sketch<T>(x: &CompressedSparseData2<T>, n_sketch: usize, rng: &mut StdRng) -> Mat<f64>
where
    T: BixverseFloat + BixverseNumeric,
{
    let (n, p) = x.shape;
    let mut buckets: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_sketch];
    for i in 0..n {
        let row = rng.random_range(0..n_sketch);
        let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        buckets[row].push((i, sign));
    }

    let rows: Vec<Vec<f64>> = buckets
        .par_iter()
        .map(|bucket| {
            let mut acc = vec![0.0; p];
            for &(i, sign) in bucket {
                let (s, e) = (x.indptr[i] as usize, x.indptr[i + 1] as usize);
                for q in s..e {
                    acc[x.indices[q] as usize] += sign * x.data[q].to_f64().unwrap();
                }
            }
            acc
        })
        .collect();

    Mat::from_fn(n_sketch, p, |r, j| rows[r][j])
}

/// Sparse Johnson-Lindenstrauss matrix with entries in `{-1, 0, 1}` (Li et
/// al.): zero with probability `1 - 1/s`, `s = ceil(sqrt(width))`.
///
/// ### Params
///
/// * `n_rows` - Rows, the feature count
/// * `width` - Columns of the projection
/// * `rng` - Random number generator
///
/// ### Returns
///
/// The `n_rows x width` matrix.
fn jl_matrix(n_rows: usize, width: usize, rng: &mut StdRng) -> Mat<f64> {
    let s = (width as f64).sqrt().ceil();
    let half = 1.0 / (2.0 * s);
    Mat::from_fn(n_rows, width, |_, _| {
        let u: f64 = rng.random();
        if u < half {
            -1.0
        } else if u < 2.0 * half {
            1.0
        } else {
            0.0
        }
    })
}

/// Squared row norms of `X M` without materialising it.
///
/// ### Params
///
/// * `x` - CSR, cells x features
/// * `m` - Dense, features x k
///
/// ### Returns
///
/// One value per cell.
fn row_norms_sq_of_product<T>(x: &CompressedSparseData2<T>, m: MatRef<f64>) -> Vec<f64>
where
    T: BixverseFloat + BixverseNumeric,
{
    let k = m.ncols();
    // row-major copy so each stored entry adds a contiguous row
    let m_rm: Vec<f64> = (0..m.nrows())
        .flat_map(|j| (0..k).map(move |c| m[(j, c)]))
        .collect();
    (0..x.shape.0)
        .into_par_iter()
        .map_init(
            || vec![0.0f64; k],
            |z, i| {
                z.fill(0.0);
                let (s, e) = (x.indptr[i] as usize, x.indptr[i + 1] as usize);
                for q in s..e {
                    let v = x.data[q].to_f64().unwrap();
                    let row = &m_rm[x.indices[q] as usize * k..][..k];
                    z.iter_mut().zip(row).for_each(|(a, &b)| *a += v * b);
                }
                z.iter().map(|v| v * v).sum()
            },
        )
        .collect()
}

/// Exact leverage over the leading right singular directions.
///
/// Eigendecomposition of the `p x p` Gram `X' X = V S^2 V'`, then
/// `U = X V S^-1` row by row. The SVD is uncentred, as Seurat's `irlba` call.
///
/// ### Params
///
/// * `x` - CSR, cells x features
/// * `n_dims` - Singular directions to keep
///
/// ### Returns
///
/// One score per cell.
fn exact_leverage<T>(x: &CompressedSparseData2<T>, n_dims: usize) -> Result<Vec<f64>, ScErrors>
where
    T: BixverseFloat + BixverseNumeric,
{
    let p = x.shape.1;
    let gram = (0..x.shape.0)
        .into_par_iter()
        .fold(
            || vec![0.0f64; p * p],
            |mut acc, i| {
                let (s, e) = (x.indptr[i] as usize, x.indptr[i + 1] as usize);
                for a in s..e {
                    let (ja, va) = (x.indices[a] as usize, x.data[a].to_f64().unwrap());
                    for b in s..e {
                        acc[ja * p + x.indices[b] as usize] += va * x.data[b].to_f64().unwrap();
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; p * p],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                a
            },
        );
    let eig = MatRef::from_row_major_slice(&gram, p, p)
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| CoreErrors::FaerEigenError)?;
    let (vals, vecs) = (eig.S(), eig.U());

    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
    let top = vals[order[0]].max(0.0);
    let keep: Vec<usize> = order
        .into_iter()
        .take(n_dims)
        .filter(|&c| vals[c] > top * LEVERAGE_EIGEN_TOL)
        .collect();

    let scaled = Mat::<f64>::from_fn(p, keep.len(), |j, c| {
        vecs[(j, keep[c])] / vals[keep[c]].sqrt()
    });
    Ok(row_norms_sq_of_product(x, scaled.as_ref()))
}

//////////
// Main //
//////////

/// Statistical leverage score per cell.
///
/// With fewer than `1.5 * n_sketch` cells, or too few cells for a sketch taller
/// than the feature count, the score is the exact leverage over the top 50
/// singular directions. Otherwise `S X` (CountSketch) gets a column-pivoted QR
/// and the score is `||x R^-1 G||^2` over the numerically independent features,
/// with `G` a sparse JL matrix: leverage up to the JL distortion, at the cost of
/// one pass over `X`. Scores are only defined up
/// to scale, which is all sampling needs.
///
/// ### Params
///
/// * `x` - CSR, cells x features. Seurat passes the log-normalised HVGs,
///   uncentred
/// * `params` - The [LeverageParams]
/// * `seed` - Seed of the sketch and the projection
///
/// ### Returns
///
/// One non-negative score per cell.
///
/// ### References
///
/// Clarkson & Woodruff, STOC, 2013; Li et al., KDD, 2006; Hao et al., Nat
/// Biotechnol, 2024
pub fn leverage_scores<T>(
    x: &CompressedSparseData2<T>,
    params: &LeverageParams,
    seed: usize,
) -> Result<Vec<f64>, ScErrors>
where
    T: BixverseFloat + BixverseNumeric,
{
    if !x.cs_type.is_csr() {
        return Err(CoreErrors::InvalidArgument("leverage_scores needs CSR input.".into()).into());
    }
    if !(params.eps > 0.0 && params.eps <= 1.0) {
        return Err(CoreErrors::InvalidArgument(format!(
            "eps must lie in (0, 1], got {}.",
            params.eps
        ))
        .into());
    }
    let (n, p) = x.shape;
    if n < 2 || p < 2 {
        return Err(CoreErrors::InvalidArgument(format!(
            "Leverage scores need at least two cells and two features, got {n} x {p}."
        ))
        .into());
    }

    let n_sketch = params
        .n_sketch
        .max((LEVERAGE_SKETCH_FEATURE_RATIO * p as f64).ceil() as usize);
    if (n as f64) < LEVERAGE_EXACT_CELL_RATIO * params.n_sketch as f64 || n_sketch >= n {
        return exact_leverage(x, LEVERAGE_EXACT_DIMS.min(p - 1).min(n - 1));
    }

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let sa = count_sketch(x, n_sketch, &mut rng);

    // Column-pivoted so collinear features (rare genes sharing their few cells)
    // surface as negligible trailing pivots. Dropping them keeps the column
    // space, so the leverage of the kept span is the leverage of `X`.
    let qr = sa.col_piv_qr();
    let r = qr.thin_R();
    let (pivots, _) = qr.P().arrays();
    let tol = r[(0, 0)].abs() * (n_sketch.max(p) as f64) * f64::EPSILON;
    let rank = (0..p).take_while(|&k| r[(k, k)].abs() > tol).count();
    if rank == 0 {
        return Err(CoreErrors::InvalidArgument(
            "The sketched matrix is numerically zero; no leverage to compute.".into(),
        )
        .into());
    }

    let eps = params.eps;
    let width = (4.0 * (n as f64).ln() / (eps * eps / 2.0 - eps.powi(3) / 3.0)).floor() as usize;
    let mut proj = jl_matrix(rank, width.max(1), &mut rng);
    solve_upper_triangular_in_place(
        r.submatrix(0, 0, rank, rank),
        proj.as_mut(),
        faer_parallelism(),
    );
    // back to feature order; dropped features get zero rows
    let mut full = Mat::<f64>::zeros(p, proj.ncols());
    for k in 0..rank {
        full.row_mut(pivots[k]).copy_from(proj.row(k));
    }

    Ok(row_norms_sq_of_product(x, full.as_ref()))
}

/// Weighted sampling without replacement (Efraimidis and Spirakis).
///
/// Each item draws `ln(u) / w` and the `n` largest keys win, which gives the
/// same distribution as drawing one at a time proportional to the remaining
/// weights, as R's `sample(prob = )` does. Items with zero weight are never
/// drawn.
///
/// ### Params
///
/// * `weights` - Non-negative weight per item
/// * `n` - Items to draw; capped at the number with positive weight
/// * `seed` - Seed
///
/// ### Returns
///
/// The drawn indices, ascending.
pub fn weighted_sample_without_replacement(weights: &[f64], n: usize, seed: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut keys: Vec<(f64, usize)> = weights
        .iter()
        .enumerate()
        .filter(|&(_, &w)| w > 0.0 && w.is_finite())
        .map(|(i, &w)| {
            let u: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
            (u.ln() / w, i)
        })
        .collect();
    let n = n.min(keys.len());
    if n < keys.len() {
        keys.select_nth_unstable_by(n, |a, b| b.0.total_cmp(&a.0));
    }
    let mut out: Vec<usize> = keys[..n].iter().map(|&(_, i)| i).collect();
    out.sort_unstable();
    out
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use bixverse_core::core::math::sparse::SparseFromDense;

    /// Deterministic sparse-ish test matrix with three planted outlier rows.
    ///
    /// ### Params
    ///
    /// * `n` - Rows
    /// * `p` - Columns
    ///
    /// ### Returns
    ///
    /// The CSR matrix and the outlier row indices.
    fn planted(n: usize, p: usize) -> (CompressedSparseData2<f64>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(42);
        let outliers = vec![7, n / 2, n - 3];
        let mut m = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                if rng.random_bool(0.4) {
                    m[(i, j)] = rng.random_range(0.0..2.0);
                }
            }
        }
        for (k, &i) in outliers.iter().enumerate() {
            m[(i, k)] += 60.0;
        }
        (
            CompressedSparseData2::from_dense_matrix(m.as_ref(), CompressedSparseFormat::Csr),
            outliers,
        )
    }

    /// Leverage from a dense thin SVD, the reference.
    ///
    /// ### Params
    ///
    /// * `x` - The CSR matrix
    /// * `k` - Leading singular vectors to keep
    ///
    /// ### Returns
    ///
    /// Squared row norms of the first `k` columns of `U`.
    fn reference_leverage(x: &CompressedSparseData2<f64>, k: usize) -> Vec<f64> {
        let (n, p) = x.shape;
        let mut dense = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for q in x.indptr[i] as usize..x.indptr[i + 1] as usize {
                dense[(i, x.indices[q] as usize)] = x.data[q];
            }
        }
        let svd = dense.thin_svd().unwrap();
        let u = svd.U();
        (0..n)
            .map(|i| (0..k).map(|c| u[(i, c)] * u[(i, c)]).sum())
            .collect()
    }

    #[test]
    fn test_leverage_exact_route_matches_svd() {
        let (x, _) = planted(200, 12);
        let got = leverage_scores(&x, &LeverageParams::default(), 1).unwrap();
        // Seurat keeps min(50, p - 1) directions
        let want = reference_leverage(&x, 11);
        for (g, w) in got.iter().zip(&want) {
            approx::assert_relative_eq!(g, w, epsilon = 1e-9, max_relative = 1e-7);
        }
        approx::assert_relative_eq!(got.iter().sum::<f64>(), 11.0, max_relative = 1e-8);
    }

    #[test]
    fn test_leverage_sketch_route_tracks_exact() {
        let (x, outliers) = planted(3000, 20);
        let params = LeverageParams::new(200, 0.5);
        let got = leverage_scores(&x, &params, 7).unwrap();
        let want = reference_leverage(&x, 20);

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (mg, mw) = (mean(&got), mean(&want));
        let cov: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - mg) * (w - mw))
            .sum();
        let sg: f64 = got.iter().map(|g| (g - mg).powi(2)).sum::<f64>().sqrt();
        let sw: f64 = want.iter().map(|w| (w - mw).powi(2)).sum::<f64>().sqrt();
        assert!(cov / (sg * sw) > 0.9, "correlation {}", cov / (sg * sw));

        let mut order: Vec<usize> = (0..got.len()).collect();
        order.sort_by(|&a, &b| got[b].total_cmp(&got[a]));
        for o in outliers {
            assert!(order[..10].contains(&o), "outlier {o} not in the top ten");
        }
    }

    #[test]
    fn test_leverage_sketch_survives_collinear_features() {
        let (x, outliers) = planted(3000, 20);
        // column 19 becomes twice column 0: rank 19, as rare genes sharing
        // their cells make it
        let mut data = x.data.clone();
        let mut indices = x.indices.clone();
        let mut indptr = vec![0u32];
        let (mut d2, mut i2) = (Vec::new(), Vec::new());
        for i in 0..x.shape.0 {
            let (s, e) = (x.indptr[i] as usize, x.indptr[i + 1] as usize);
            let mut row: Vec<(u32, f64)> = (s..e)
                .filter(|&q| x.indices[q] != 19)
                .map(|q| (x.indices[q], x.data[q]))
                .collect();
            if let Some(&(_, v)) = row.iter().find(|e| e.0 == 0) {
                row.push((19, 2.0 * v));
            }
            for (c, v) in row {
                i2.push(c);
                d2.push(v);
            }
            indptr.push(i2.len() as u32);
        }
        std::mem::swap(&mut data, &mut d2);
        std::mem::swap(&mut indices, &mut i2);
        let x = CompressedSparseData2::from_parts(
            data,
            indices,
            indptr,
            None,
            CompressedSparseFormat::Csr,
            x.shape,
        );

        let got = leverage_scores(&x, &LeverageParams::new(200, 0.5), 7).unwrap();
        assert!(got.iter().all(|v| v.is_finite()));
        let want = reference_leverage(&x, 19);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (mg, mw) = (mean(&got), mean(&want));
        let cov: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - mg) * (w - mw))
            .sum();
        let sg: f64 = got.iter().map(|g| (g - mg).powi(2)).sum::<f64>().sqrt();
        let sw: f64 = want.iter().map(|w| (w - mw).powi(2)).sum::<f64>().sqrt();
        assert!(cov / (sg * sw) > 0.9, "correlation {}", cov / (sg * sw));
        let mut order: Vec<usize> = (0..got.len()).collect();
        order.sort_by(|&a, &b| got[b].total_cmp(&got[a]));
        for o in outliers {
            assert!(order[..10].contains(&o), "outlier {o} not in the top ten");
        }
    }

    #[test]
    fn test_leverage_sketch_reproducible() {
        let (x, _) = planted(3000, 20);
        let params = LeverageParams::new(200, 0.5);
        assert_eq!(
            leverage_scores(&x, &params, 3).unwrap(),
            leverage_scores(&x, &params, 3).unwrap()
        );
    }

    #[test]
    fn test_weighted_sample_skips_zero_weights() {
        let mut w = vec![1.0; 100];
        w[..50].fill(0.0);
        let s = weighted_sample_without_replacement(&w, 30, 5);
        assert_eq!(s.len(), 30);
        assert!(s.iter().all(|&i| i >= 50));
        assert!(s.windows(2).all(|p| p[0] < p[1]));
        assert_eq!(weighted_sample_without_replacement(&w, 80, 5).len(), 50);
    }

    #[test]
    fn test_weighted_sample_favours_heavy_items() {
        let mut w = vec![1.0; 1000];
        w[..10].fill(1000.0);
        let s = weighted_sample_without_replacement(&w, 20, 11);
        assert!((0..10).all(|i| s.contains(&i)));
    }
}
