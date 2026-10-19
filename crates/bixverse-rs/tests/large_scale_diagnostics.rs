//! Large-n correctness gates for the sparse PCA path.
//!
//! Checks, up to 500k rows, that the pieces the sparse SVDs rely on hold at
//! scale: `data_2` survives a CSC/CSR transpose, the implicit centre-and-scale
//! matvecs match an explicitly centred dense matrix, the implied Gram operator
//! is symmetric, the Lanczos SVD reconstructs `A v = s u`, and the sparse
//! randomised SVD gives the same PC scores as the dense one. The references are
//! all dense f64 computations on the same synthetic matrix.
//!
//! Every test asserts, so the whole file sits behind `large-test`. The file
//! name predates the split from `large_scale_diagnostics`.

#![allow(clippy::needless_range_loop)]
#![cfg(all(feature = "single-cell", feature = "large-test"))]

use faer::Mat;
use rand::prelude::*;
use rand_distr::Normal;

use bixverse_rs::core::math::pca_svd::{compute_pc_scores, randomised_sparse_svd, randomised_svd};
use bixverse_rs::core::math::sparse::{sparse_svd_lanczos, transpose_sparse};
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_processing::pca::{
    sparse_csc_column_means, sparse_csc_column_stds,
};

///////////////
// Constants //
///////////////

/// Row counts for the full sweep.
const N_ROWS_FULL: [usize; 4] = [1000, 10_000, 100_000, 500_000];
/// Row counts for the tests that build dense operators per call.
const N_ROWS_SHORT: [usize; 3] = [1000, 10_000, 100_000];
/// Columns in the transpose round-trip fixture.
const TRANSPOSE_N_COLS: usize = 2000;
/// Non-zero density in the transpose round-trip fixture.
const TRANSPOSE_DENSITY: f64 = 0.05;
/// Columns everywhere else. Small enough to keep a dense copy at 500k rows.
const N_COLS: usize = 500;
/// Non-zero density everywhere else.
const DENSITY: f64 = 0.1;
/// Number of singular triplets requested.
const N_COMP: usize = 10;
/// Oversampling for both randomised SVDs.
const OVERSAMPLING: usize = 100;
/// Power iterations for both randomised SVDs.
const N_POWER_ITER: usize = 2;

/////////////
// Helpers //
/////////////

/// Synthetic non-negative sparse matrix plus its dense twin.
///
/// Values are `|N(0, 2)|`, so non-negative like counts. `data` and `data_2`
/// hold the same values, so either layer can be used.
///
/// ### Params
///
/// * `n` - Number of rows
/// * `m` - Number of columns
/// * `density` - Probability an entry is non-zero
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// `(csc, dense)`, the CSC matrix and the same values as a dense f64 matrix.
fn make_test_matrix(
    n: usize,
    m: usize,
    density: f64,
    seed: u64,
) -> (CompressedSparseData2<f32, f32>, Mat<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::<f64>::new(0.0, 2.0).expect("valid normal");

    let mut dense = Mat::<f64>::zeros(n, m);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    let mut indptr = vec![0usize];

    for j in 0..m {
        for i in 0..n {
            if rng.random::<f64>() < density {
                let val: f32 = normal.sample(&mut rng).abs() as f32;
                dense[(i, j)] = val as f64;
                row_indices.push(i);
                values.push(val);
            }
        }
        indptr.push(values.len());
    }

    let csc = CompressedSparseData2 {
        data: values.clone(),
        indices: row_indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csc,
        data_2: Some(values),
        shape: (n, m),
    };

    (csc, dense)
}

///////////
// Tests //
///////////

/// `data_2` survives a CSC -> CSR -> CSC round trip exactly, and every CSC
/// column holds the same `(row, value)` pairs as the CSR rows. The Lanczos
/// dual-representation approach is broken if this fails.
#[test]
fn test_transpose_data2_consistency() {
    for &n in &N_ROWS_FULL {
        let m = TRANSPOSE_N_COLS;
        let (csc, _dense) = make_test_matrix(n, m, TRANSPOSE_DENSITY, 42);

        let csr = transpose_sparse(&csc);
        let csc_roundtrip = transpose_sparse(&csr);

        let orig_d2 = csc.data_2.as_ref().expect("data_2 set");
        let rt_d2 = csc_roundtrip.data_2.as_ref().expect("data_2 set");
        let csr_d2 = csr.data_2.as_ref().expect("data_2 set");

        assert_eq!(orig_d2.len(), rt_d2.len(), "n={n}: data_2 length mismatch");

        let max_diff: f32 = orig_d2
            .iter()
            .zip(rt_d2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff == 0.0,
            "n={n}: data_2 round-trip max diff = {max_diff}"
        );

        for j in 0..m {
            let csc_start = csc.indptr[j] as usize;
            let csc_end = csc.indptr[j + 1] as usize;
            let mut csc_pairs: Vec<(usize, f32)> = csc.indices[csc_start..csc_end]
                .iter()
                .zip(orig_d2[csc_start..csc_end].iter())
                .map(|(&i, &v)| (i as usize, v))
                .collect();
            csc_pairs.sort_by_key(|&(i, _)| i);

            let mut csr_pairs: Vec<(usize, f32)> = Vec::new();
            for i in 0..n {
                let csr_start = csr.indptr[i] as usize;
                let csr_end = csr.indptr[i + 1] as usize;
                for idx in csr_start..csr_end {
                    if csr.indices[idx] == j as u32 {
                        csr_pairs.push((i, csr_d2[idx]));
                    }
                }
            }
            csr_pairs.sort_by_key(|&(i, _)| i);

            assert_eq!(
                csc_pairs.len(),
                csr_pairs.len(),
                "n={n}, col={j}: nnz mismatch between CSC and CSR"
            );
            for (a, b) in csc_pairs.iter().zip(csr_pairs.iter()) {
                assert_eq!(a.0, b.0, "n={n}, col={j}: index mismatch");
                assert!(
                    (a.1 - b.1).abs() == 0.0,
                    "n={n}, col={j}, row={}: value mismatch {} vs {}",
                    a.0,
                    a.1,
                    b.1
                );
            }
        }
    }
}

/// Implicitly centred and scaled `A x` over CSR matches the explicit dense
/// product. This is the operator the sparse SVDs apply.
#[test]
fn test_implicit_vs_explicit_centring() {
    for &n in &N_ROWS_FULL {
        let m = N_COLS;
        let (csc, dense) = make_test_matrix(n, m, DENSITY, 123);

        let col_means = sparse_csc_column_means(&csc, false, None).expect("column means");
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false, None).expect("column stds");

        let dense_cs =
            Mat::<f64>::from_fn(n, m, |i, j| (dense[(i, j)] - col_means[j]) / col_stds[j]);

        let mut rng = StdRng::seed_from_u64(999);
        let x: Vec<f64> = (0..m).map(|_| rng.random::<f64>() - 0.5).collect();

        let mut y_explicit = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..m {
                y_explicit[i] += dense_cs[(i, j)] * x[j];
            }
        }

        let csr = transpose_sparse(&csc);
        let data_csr_f: Vec<f64> = csr.data.iter().map(|&v| v as f64).collect();

        let x_scaled: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(j, &v)| v / col_stds[j])
            .collect();
        let mean_dot: f64 = x_scaled
            .iter()
            .enumerate()
            .map(|(j, &v)| col_means[j] * v)
            .sum();

        let mut y_implicit = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0f64;
            for idx in csr.indptr[i]..csr.indptr[i + 1] {
                let idx = idx as usize;
                let j = csr.indices[idx];
                sum += data_csr_f[idx] * x_scaled[j as usize];
            }
            sum -= mean_dot;
            y_implicit[i] = sum;
        }

        let max_diff: f64 = y_explicit
            .iter()
            .zip(y_implicit.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        let y_norm: f64 = y_explicit.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel_err = max_diff / y_norm.max(1e-15);

        assert!(
            rel_err < 1e-10,
            "n={n}: implicit centring diverged, rel_err={rel_err:.2e}"
        );
    }
}

/// Implicitly centred and scaled `A^T x` over CSC matches the explicit dense
/// product.
#[test]
fn test_implicit_vs_explicit_transpose_matvec() {
    for &n in &N_ROWS_FULL {
        let m = N_COLS;
        let (csc, dense) = make_test_matrix(n, m, DENSITY, 456);

        let col_means = sparse_csc_column_means(&csc, false, None).expect("column means");
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false, None).expect("column stds");

        let dense_cs =
            Mat::<f64>::from_fn(n, m, |i, j| (dense[(i, j)] - col_means[j]) / col_stds[j]);

        let mut rng = StdRng::seed_from_u64(777);
        let x: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();

        let mut y_explicit = vec![0.0f64; m];
        for j in 0..m {
            for i in 0..n {
                y_explicit[j] += dense_cs[(i, j)] * x[i];
            }
        }

        let data_csc_f: Vec<f64> = csc.data.iter().map(|&v| v as f64).collect();
        let x_sum: f64 = x.iter().sum();

        let mut y_implicit = vec![0.0f64; m];
        for j in 0..m {
            let mut sum = 0.0f64;
            for idx in csc.indptr[j]..csc.indptr[j + 1] {
                let idx = idx as usize;
                let i = csc.indices[idx];
                sum += data_csc_f[idx] * x[i as usize];
            }
            sum -= col_means[j] * x_sum;
            sum /= col_stds[j];
            y_implicit[j] = sum;
        }

        let max_diff: f64 = y_explicit
            .iter()
            .zip(y_implicit.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        let y_norm: f64 = y_explicit.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel_err = max_diff / y_norm.max(1e-15);

        assert!(
            rel_err < 1e-10,
            "n={n}: transpose matvec diverged, rel_err={rel_err:.2e}"
        );
    }
}

/// `<G x, y> == <x, G y>` for the Gram operator built from the implicit `A`
/// and `A^T` matvecs. An inconsistency between the two (e.g. a `data_2`
/// transpose bug) breaks this, and with it Lanczos.
#[test]
fn test_gram_symmetry() {
    for &n in &N_ROWS_SHORT {
        let m = N_COLS;
        let (csc, _dense) = make_test_matrix(n, m, DENSITY, 789);

        let col_means = sparse_csc_column_means(&csc, false, None).expect("column means");
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false, None).expect("column stds");

        let csr = transpose_sparse(&csc);
        let data_csr_f: Vec<f64> = csr.data.iter().map(|&v| v as f64).collect();
        let data_csc_f: Vec<f64> = csc.data.iter().map(|&v| v as f64).collect();

        let use_ata = n > m;
        let krylov_dim = if use_ata { m } else { n };

        let matvec_a = |x: &[f64]| -> Vec<f64> {
            let x_scaled: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(j, &v)| v / col_stds[j])
                .collect();
            let mean_dot: f64 = x_scaled
                .iter()
                .enumerate()
                .map(|(j, &v)| col_means[j] * v)
                .sum();
            let mut y = vec![0.0f64; n];
            for i in 0..n {
                let mut sum = 0.0;
                for idx in csr.indptr[i]..csr.indptr[i + 1] {
                    let idx = idx as usize;
                    let j = csr.indices[idx];
                    sum += data_csr_f[idx] * x_scaled[j as usize];
                }
                y[i] = sum - mean_dot;
            }
            y
        };

        let matvec_at = |x: &[f64]| -> Vec<f64> {
            let x_sum: f64 = x.iter().sum();
            let mut y = vec![0.0f64; m];
            for j in 0..m {
                let mut sum = 0.0;
                for idx in csc.indptr[j]..csc.indptr[j + 1] {
                    let idx = idx as usize;
                    let i = csc.indices[idx];
                    sum += data_csc_f[idx] * x[i as usize];
                }
                y[j] = (sum - col_means[j] * x_sum) / col_stds[j];
            }
            y
        };

        let gram = |x: &[f64]| -> Vec<f64> {
            if use_ata {
                matvec_at(&matvec_a(x))
            } else {
                matvec_a(&matvec_at(x))
            }
        };

        let mut rng = StdRng::seed_from_u64(42);
        let x: Vec<f64> = (0..krylov_dim).map(|_| rng.random::<f64>() - 0.5).collect();
        let y: Vec<f64> = (0..krylov_dim).map(|_| rng.random::<f64>() - 0.5).collect();

        let gx = gram(&x);
        let gy = gram(&y);

        let dot_gx_y: f64 = gx.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let dot_x_gy: f64 = x.iter().zip(gy.iter()).map(|(a, b)| a * b).sum();

        let rel_diff = (dot_gx_y - dot_x_gy).abs() / dot_gx_y.abs().max(1e-15);

        assert!(
            rel_diff < 1e-10,
            "n={n}: Gram operator not symmetric, rel_diff={rel_diff:.2e}"
        );
    }
}

/// Lanczos SVD at scale: singular values are non-negative and the leading
/// three triplets satisfy `||A v - s u|| / ||A v|| < 1e-4`.
#[test]
fn test_lanczos_svd_reconstruction() {
    for &n in &N_ROWS_SHORT {
        let m = N_COLS;
        let (csc, _) = make_test_matrix(n, m, DENSITY, 321);
        let col_means = sparse_csc_column_means(&csc, false, None).expect("column means");
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false, None).expect("column stds");

        let svd = sparse_svd_lanczos::<f32, f32, f64>(
            &csc,
            N_COMP,
            42,
            false,
            Some(&col_means),
            Some(&col_stds),
            None,
        )
        .expect("Lanczos SVD failed");

        let k = svd.s.len();

        for i in 1..k {
            assert!(
                svd.s[i] >= 0.0,
                "n={n}: negative singular value s[{i}] = {}",
                svd.s[i]
            );
        }

        let csr = transpose_sparse(&csc);
        let data_csr_f: Vec<f64> = csr.data.iter().map(|&v| v as f64).collect();

        for comp in 0..k.min(3) {
            let v_scaled: Vec<f64> = (0..m).map(|j| svd.v[(j, comp)] / col_stds[j]).collect();
            let mean_dot: f64 = v_scaled
                .iter()
                .enumerate()
                .map(|(j, &v)| col_means[j] * v)
                .sum();

            let mut av = vec![0.0f64; n];
            for i in 0..n {
                let mut sum = 0.0;
                for idx in csr.indptr[i]..csr.indptr[i + 1] {
                    let idx = idx as usize;
                    let j = csr.indices[idx];
                    sum += data_csr_f[idx] * v_scaled[j as usize];
                }
                av[i] = sum - mean_dot;
            }

            let sigma = svd.s[comp];
            let residual: f64 = (0..n)
                .map(|i| {
                    let diff = av[i] - sigma * svd.u[(i, comp)];
                    diff * diff
                })
                .sum::<f64>()
                .sqrt();

            let av_norm: f64 = av.iter().map(|v| v * v).sum::<f64>().sqrt();
            let rel_residual = residual / av_norm.max(1e-15);

            assert!(
                rel_residual < 1e-4,
                "n={n}, comp={comp}: SVD reconstruction error too large: {rel_residual:.2e}"
            );
        }
    }
}

/// Sparse randomised SVD with implicit centring gives the same PC scores as a
/// dense randomised SVD on the explicitly centred matrix, up to sign
/// (`|cor| > 0.99` per component).
#[test]
fn test_sparse_vs_dense_svd_scores() {
    for &n in &N_ROWS_FULL {
        let m = N_COLS;
        let (csc, dense) = make_test_matrix(n, m, DENSITY, 555);

        let col_means = sparse_csc_column_means(&csc, false, None).expect("column means");
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false, None).expect("column stds");

        let dense_cs =
            Mat::<f64>::from_fn(n, m, |i, j| (dense[(i, j)] - col_means[j]) / col_stds[j]);
        let dense_svd = randomised_svd(
            dense_cs.as_ref(),
            N_COMP,
            42,
            Some(OVERSAMPLING),
            Some(N_POWER_ITER),
        )
        .expect("dense randomised SVD failed");
        let dense_scores = compute_pc_scores(&dense_svd);

        let sparse_svd = randomised_sparse_svd::<f32, f64>(
            csc,
            N_COMP,
            42,
            false,
            Some(OVERSAMPLING),
            Some(N_POWER_ITER),
            Some(&col_means),
            Some(&col_stds),
            None,
        )
        .expect("sparse randomised SVD failed");
        let sparse_scores = compute_pc_scores(&sparse_svd);

        for comp in 0..N_COMP {
            let mut corr = 0.0f64;
            let mut norm_d = 0.0f64;
            let mut norm_s = 0.0f64;
            for i in 0..n {
                let d = dense_scores[(i, comp)];
                let s = sparse_scores[(i, comp)];
                corr += d * s;
                norm_d += d * d;
                norm_s += s * s;
            }
            let abs_corr = corr.abs() / (norm_d.sqrt() * norm_s.sqrt()).max(1e-15);

            assert!(
                abs_corr > 0.99,
                "n={n}, PC{comp}: sparse vs dense correlation = {abs_corr:.6}"
            );
        }
    }
}
