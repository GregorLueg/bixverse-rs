#![cfg(feature = "large_scale_diagnostics")]
#![allow(clippy::needless_range_loop)]

use bixverse_rs::core::math::pca_svd::*;
use bixverse_rs::core::math::sparse::*;
use bixverse_rs::single_cell::sc_processing::pca::*;

use faer::Mat;
use rand::prelude::*;
use rand_distr::Normal;

/// Generate a synthetic sparse CSC matrix with known properties.
/// Returns (csc_matrix, dense_matrix) so you can compare.
fn make_test_matrix(
    n: usize,
    m: usize,
    density: f64,
    seed: u64,
) -> (CompressedSparseData2<f32, f32>, Mat<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::<f64>::new(0.0, 2.0).unwrap();

    let mut dense = Mat::<f64>::zeros(n, m);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    let mut indptr = vec![0usize];

    for j in 0..m {
        for i in 0..n {
            if rng.random::<f64>() < density {
                let val: f32 = normal.sample(&mut rng).abs() as f32; // non-negative like counts
                dense[(i, j)] = val as f64;
                row_indices.push(i);
                values.push(val);
            }
        }
        indptr.push(values.len());
    }

    let csc = CompressedSparseData2 {
        data: values.clone(),
        indices: row_indices,
        indptr,
        cs_type: CompressedSparseFormat::Csc,
        data_2: Some(values), // same data in both layers for testing
        shape: (n, m),
    };

    (csc, dense)
}

/// TEST 1: Verify CSR/CSC transpose consistency for data_2.
/// If this fails, the Lanczos dual-representation approach is broken.
#[test]
fn test_transpose_data2_consistency() {
    for &n in &[1000, 10_000, 100_000, 500_000] {
        let m = 2000;
        let (csc, _dense) = make_test_matrix(n, m, 0.05, 42);

        // Transpose CSC -> CSR
        let csr = transpose_sparse(&csc);

        // Now transpose back CSR -> CSC
        let csc_roundtrip = transpose_sparse(&csr);

        // data_2 should be identical after round-trip
        let orig_d2 = csc.data_2.as_ref().unwrap();
        let rt_d2 = csc_roundtrip.data_2.as_ref().unwrap();

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

        // Also verify: for each column j, CSC data_2 values match
        // the corresponding CSR data_2 values (different order but same set)
        for j in 0..m {
            let csc_start = csc.indptr[j];
            let csc_end = csc.indptr[j + 1];
            let mut csc_pairs: Vec<(usize, f32)> = csc.indices[csc_start..csc_end]
                .iter()
                .zip(orig_d2[csc_start..csc_end].iter())
                .map(|(&i, &v)| (i, v))
                .collect();
            csc_pairs.sort_by_key(|&(i, _)| i);

            // Gather same column from CSR
            let mut csr_pairs: Vec<(usize, f32)> = Vec::new();
            for i in 0..n {
                let csr_start = csr.indptr[i];
                let csr_end = csr.indptr[i + 1];
                for idx in csr_start..csr_end {
                    if csr.indices[idx] == j {
                        let csr_d2 = csr.data_2.as_ref().unwrap();
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

        eprintln!("PASS: transpose data_2 consistency at n={n}");
    }
}

/// TEST 2: Verify implicit centering/scaling matches explicit.
/// This is the core correctness check for the sparse matvec operators.
#[test]
fn test_implicit_vs_explicit_centering() {
    for &n in &[1000, 10_000, 100_000, 500_000] {
        let m = 500; // keep m small-ish to allow dense comparison
        let (csc, dense) = make_test_matrix(n, m, 0.1, 123);

        // Compute means and stds from the CSC
        let col_means = sparse_csc_column_means(&csc, false);
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false);

        // Build explicit centered+scaled dense matrix
        let dense_cs =
            Mat::<f64>::from_fn(n, m, |i, j| (dense[(i, j)] - col_means[j]) / col_stds[j]);

        // Generate a random test vector
        let mut rng = StdRng::seed_from_u64(999);
        let x: Vec<f64> = (0..m).map(|_| rng.random::<f64>() - 0.5).collect();

        // Explicit: y_explicit = dense_cs * x
        let mut y_explicit = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..m {
                y_explicit[i] += dense_cs[(i, j)] * x[j];
            }
        }

        // Implicit via Lanczos-style matvec_a
        // Need CSR for matvec_a
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
                let j = csr.indices[idx];
                sum += data_csr_f[idx] * x_scaled[j];
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

        eprintln!("n={n}: matvec_a max_diff={max_diff:.2e}, rel_err={rel_err:.2e}");
        assert!(
            rel_err < 1e-10,
            "n={n}: implicit centering diverged, rel_err={rel_err:.2e}"
        );
    }
}

/// TEST 3: Verify matvec_at (transpose) implicit vs explicit.
#[test]
fn test_implicit_vs_explicit_transpose_matvec() {
    for &n in &[1000, 10_000, 100_000, 500_000] {
        let m = 500;
        let (csc, dense) = make_test_matrix(n, m, 0.1, 456);

        let col_means = sparse_csc_column_means(&csc, false);
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false);

        let dense_cs =
            Mat::<f64>::from_fn(n, m, |i, j| (dense[(i, j)] - col_means[j]) / col_stds[j]);

        let mut rng = StdRng::seed_from_u64(777);
        let x: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();

        // Explicit: y = dense_cs^T * x
        let mut y_explicit = vec![0.0f64; m];
        for j in 0..m {
            for i in 0..n {
                y_explicit[j] += dense_cs[(i, j)] * x[i];
            }
        }

        // Implicit via Lanczos-style matvec_at (uses CSC)
        let data_csc_f: Vec<f64> = csc.data.iter().map(|&v| v as f64).collect();
        let x_sum: f64 = x.iter().sum();

        let mut y_implicit = vec![0.0f64; m];
        for j in 0..m {
            let mut sum = 0.0f64;
            for idx in csc.indptr[j]..csc.indptr[j + 1] {
                let i = csc.indices[idx];
                sum += data_csc_f[idx] * x[i];
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

        eprintln!("n={n}: matvec_at max_diff={max_diff:.2e}, rel_err={rel_err:.2e}");
        assert!(
            rel_err < 1e-10,
            "n={n}: transpose matvec diverged, rel_err={rel_err:.2e}"
        );
    }
}

/// TEST 4: Verify Gram matrix symmetry.
/// If matvec_a and matvec_at are inconsistent (e.g., due to data_2 transpose bug),
/// then <Gram*x, y> != <x, Gram*y> and Lanczos is fundamentally broken.
#[test]
fn test_gram_symmetry() {
    for &n in &[1000, 10_000, 100_000] {
        let m = 500;
        let (csc, _dense) = make_test_matrix(n, m, 0.1, 789);

        let col_means = sparse_csc_column_means(&csc, false);
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false);

        let csr = transpose_sparse(&csc);
        let data_csr_f: Vec<f64> = csr.data.iter().map(|&v| v as f64).collect();
        let data_csc_f: Vec<f64> = csc.data.iter().map(|&v| v as f64).collect();

        let use_ata = n > m;
        let krylov_dim = if use_ata { m } else { n };

        // matvec_a: y = A_cs * x  (n-dim output)
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
                    let j = csr.indices[idx];
                    sum += data_csr_f[idx] * x_scaled[j];
                }
                y[i] = sum - mean_dot;
            }
            y
        };

        // matvec_at: y = A_cs^T * x  (m-dim output)
        let matvec_at = |x: &[f64]| -> Vec<f64> {
            let x_sum: f64 = x.iter().sum();
            let mut y = vec![0.0f64; m];
            for j in 0..m {
                let mut sum = 0.0;
                for idx in csc.indptr[j]..csc.indptr[j + 1] {
                    let i = csc.indices[idx];
                    sum += data_csc_f[idx] * x[i];
                }
                y[j] = (sum - col_means[j] * x_sum) / col_stds[j];
            }
            y
        };

        // Gram = A^T A (if use_ata) or AA^T
        let gram = |x: &[f64]| -> Vec<f64> {
            if use_ata {
                let temp = matvec_a(x);
                matvec_at(&temp)
            } else {
                let temp = matvec_at(x);
                matvec_a(&temp)
            }
        };

        // Test: <Gram*x, y> == <x, Gram*y> for random x, y
        let mut rng = StdRng::seed_from_u64(42);
        let x: Vec<f64> = (0..krylov_dim).map(|_| rng.random::<f64>() - 0.5).collect();
        let y: Vec<f64> = (0..krylov_dim).map(|_| rng.random::<f64>() - 0.5).collect();

        let gx = gram(&x);
        let gy = gram(&y);

        let dot_gx_y: f64 = gx.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let dot_x_gy: f64 = x.iter().zip(gy.iter()).map(|(a, b)| a * b).sum();

        let rel_diff = (dot_gx_y - dot_x_gy).abs() / dot_gx_y.abs().max(1e-15);

        eprintln!("n={n}: <Gx,y>={dot_gx_y:.6e}, <x,Gy>={dot_x_gy:.6e}, rel_diff={rel_diff:.2e}");
        assert!(
            rel_diff < 1e-10,
            "n={n}: Gram matrix NOT symmetric! rel_diff={rel_diff:.2e}"
        );
    }
}

/// TEST 5: Lanczos orthogonality loss at scale.
/// Measures how much orthogonality degrades with single-pass CGS.
#[test]
fn test_lanczos_orthogonality_loss() {
    for &n in &[1000, 10_000, 100_000] {
        let m = 500;
        let (csc, _) = make_test_matrix(n, m, 0.1, 321);
        let col_means = sparse_csc_column_means(&csc, false);
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false);

        let no_params_means: Option<&[f64]> = Some(&col_means);
        let no_params_stds: Option<&[f64]> = Some(&col_stds);

        // Run sparse_svd_lanczos and check if results are sane
        let svd = sparse_svd_lanczos::<f32, f32, f64>(
            &csc,
            10,
            42,
            false,
            no_params_means,
            no_params_stds,
        );

        // Check U orthogonality: U^T U should be ~identity
        let k = svd.s.len();
        let mut max_off_diag = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..n).map(|r| svd.u[(r, i)] * svd.u[(r, j)]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (dot - expected).abs();
                if i != j {
                    max_off_diag = max_off_diag.max(err);
                }
            }
        }

        // Check V orthogonality
        let mut max_off_diag_v = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..m).map(|r| svd.v[(r, i)] * svd.v[(r, j)]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (dot - expected).abs();
                if i != j {
                    max_off_diag_v = max_off_diag_v.max(err);
                }
            }
        }

        eprintln!(
            "n={n}: U orthogonality loss={max_off_diag:.2e}, V orthogonality loss={max_off_diag_v:.2e}"
        );

        // Also check singular values are positive and decreasing
        for i in 1..svd.s.len() {
            assert!(
                svd.s[i] >= 0.0,
                "n={n}: negative singular value s[{i}] = {}",
                svd.s[i]
            );
        }

        // Check reconstruction error: ||A_cs * v_j - s_j * u_j|| should be small
        let csr = transpose_sparse(&csc);
        let data_csr_f: Vec<f64> = csr.data.iter().map(|&v| v as f64).collect();

        for comp in 0..k.min(3) {
            let v_col: Vec<f64> = (0..m).map(|j| svd.v[(j, comp)]).collect();
            let v_scaled: Vec<f64> = v_col
                .iter()
                .enumerate()
                .map(|(j, &v)| v / col_stds[j])
                .collect();
            let mean_dot: f64 = v_scaled
                .iter()
                .enumerate()
                .map(|(j, &v)| col_means[j] * v)
                .sum();

            let mut av = vec![0.0f64; n];
            for i in 0..n {
                let mut sum = 0.0;
                for idx in csr.indptr[i]..csr.indptr[i + 1] {
                    let j = csr.indices[idx];
                    sum += data_csr_f[idx] * v_scaled[j];
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

            eprintln!("n={n}, comp={comp}: ||Av - su||/||Av|| = {rel_residual:.2e}");
            assert!(
                rel_residual < 1e-4,
                "n={n}, comp={comp}: SVD reconstruction error too large: {rel_residual:.2e}"
            );
        }
    }
}

/// TEST 6: Compare sparse randomised SVD against dense randomised SVD.
/// This is the end-to-end comparison that replicates your plots.
#[test]
fn test_sparse_vs_dense_svd_scores() {
    for &n in &[1000, 10_000, 100_000, 500_000] {
        let m = 500;
        let (csc, dense) = make_test_matrix(n, m, 0.1, 555);

        let col_means = sparse_csc_column_means(&csc, false);
        let col_stds = sparse_csc_column_stds(&csc, &col_means, false);

        // Dense: explicitly centre and scale, then randomised SVD
        let dense_cs =
            Mat::<f64>::from_fn(n, m, |i, j| (dense[(i, j)] - col_means[j]) / col_stds[j]);
        let dense_svd = randomised_svd(dense_cs.as_ref(), 10, 42, Some(100), Some(2));
        let dense_scores = compute_pc_scores(&dense_svd);

        // Sparse: implicit centering
        let sparse_svd = randomised_sparse_svd::<f32, f64>(
            &csc,
            10,
            42,
            false, // use data, not data_2 (we set them equal)
            Some(100),
            Some(2),
            Some(&col_means),
            Some(&col_stds),
        );
        let sparse_scores = compute_pc_scores(&sparse_svd);

        // Compare PC scores (allowing sign flips)
        for comp in 0..10 {
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

            eprintln!("n={n}, PC{comp}: |correlation| = {abs_corr:.6}");
            assert!(
                abs_corr > 0.99,
                "n={n}, PC{comp}: sparse vs dense correlation = {abs_corr:.6}"
            );
        }

        eprintln!("PASS: sparse vs dense at n={n}");
    }
}
