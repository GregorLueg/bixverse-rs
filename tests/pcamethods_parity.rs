//! Parity of `ppca` / `bpca` against pcaMethods on LCG-built data with
//! missing values. Fixtures from `dev/gen_pcamethods_fixtures.R`.

use bixverse_rs::core::math::pca_missing::*;
use bixverse_rs::prelude::*;
use faer::{Mat, MatRef};

mod pcamethods_fixtures;
use pcamethods_fixtures::*;

#[cfg(feature = "large-test")]
#[path = "pcamethods_fixtures/large.rs"]
mod pcamethods_large;

/// Numerical Recipes LCG, identical to the R side.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = (1664525 * self.0 + 1013904223) % 4294967296;
        self.0
    }
}

/// Rebuild the fixture matrix (N x D, `NaN` for missing) the R script used.
fn build_case(n: usize, d: usize, k: usize, miss_pct: u64, seed: u64) -> Mat<f64> {
    let mut lcg = Lcg(seed);
    let mut z = vec![0.0; n * k];
    let mut w = vec![0.0; d * k];
    for v in z.iter_mut() {
        *v = (lcg.next() % 11) as f64 - 5.0;
    }
    for v in w.iter_mut() {
        *v = (lcg.next() % 7) as f64 - 3.0;
    }
    let mut y = Mat::<f64>::zeros(n, d);
    for i in 0..n {
        for j in 0..d {
            let signal: f64 = (0..k).map(|l| z[i * k + l] * w[j * k + l]).sum();
            y[(i, j)] = signal + ((lcg.next() % 2001) as f64 - 1000.0) / 250.0;
        }
    }
    let mask = loop {
        let mask: Vec<bool> = (0..n * d).map(|_| lcg.next() % 100 < miss_pct).collect();
        let rows_ok = (0..n).all(|i| (0..d).any(|j| !mask[i * d + j]));
        let cols_ok = (0..d).all(|j| (0..n).any(|i| !mask[i * d + j]));
        if rows_ok && cols_ok {
            break mask;
        }
    };
    for i in 0..n {
        for j in 0..d {
            if mask[i * d + j] {
                y[(i, j)] = f64::NAN;
            }
        }
    }
    y
}

/// Column-major fixture slice as a matrix.
fn fixture(x: &[f64], nrow: usize, ncol: usize) -> MatRef<'_, f64> {
    MatRef::from_column_major_slice(x, nrow, ncol)
}

/// Max absolute difference after flipping each column of `ours` onto the sign
/// of `theirs`.
fn max_diff_signed(ours: MatRef<f64>, theirs: MatRef<f64>) -> f64 {
    let mut worst: f64 = 0.0;
    for l in 0..ours.ncols() {
        let dot: f64 = (0..ours.nrows())
            .map(|i| ours[(i, l)] * theirs[(i, l)])
            .sum();
        let s = dot.signum();
        for i in 0..ours.nrows() {
            worst = worst.max((s * ours[(i, l)] - theirs[(i, l)]).abs());
        }
    }
    worst
}

/// Max absolute difference.
fn max_diff(ours: MatRef<f64>, theirs: MatRef<f64>) -> f64 {
    let mut worst: f64 = 0.0;
    for j in 0..ours.ncols() {
        for i in 0..ours.nrows() {
            worst = worst.max((ours[(i, j)] - theirs[(i, j)]).abs());
        }
    }
    worst
}

/// Max absolute difference of two slices.
fn max_diff_vec(ours: &[f64], theirs: &[f64]) -> f64 {
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Per-quantity max absolute differences of one fit against R.
struct Diffs {
    scores: f64,
    loadings: f64,
    r2: f64,
    completed: f64,
}

/// Compare one fit against its R fixture, aligning column signs of scores
/// and loadings.
#[allow(clippy::too_many_arguments)]
fn compare(
    res: &MissingPcaResults<f64>,
    n: usize,
    d: usize,
    k: usize,
    scores: &[f64],
    loadings: &[f64],
    r2: &[f64],
    completed: &[f64],
) -> Diffs {
    Diffs {
        scores: max_diff_signed(res.scores.as_ref(), fixture(scores, n, k)),
        loadings: max_diff_signed(res.loadings.as_ref(), fixture(loadings, d, k)),
        r2: max_diff_vec(&res.r2_cum, r2),
        completed: max_diff(res.completed.as_ref(), fixture(completed, n, d)),
    }
}

/// Tolerance on every compared quantity. The first runs measured at most
/// 2.3e-13 on the small cases and 5.9e-12 on the large one (PPCA scores both
/// times); the gap is summation order, not method.
const TOL: f64 = 1e-10;

fn assert_diffs(label: &str, x: &Diffs) {
    for (what, v) in [
        ("scores", x.scores),
        ("loadings", x.loadings),
        ("r2", x.r2),
        ("completed", x.completed),
    ] {
        assert!(v < TOL, "{label} {what}: max abs diff {v:.3e}");
    }
}

/// Both fits on a tall (60 x 12, 20 % missing) and a wide (16 x 80, 25 %
/// missing) case. In the wide case R's BPCA prunes the third component, so
/// the ARD path is covered too.
#[test]
fn test_ppca_bpca_match_pcamethods() {
    for (tag, n, d, k, miss, seed) in [
        ("tall", TALL_N, TALL_D, TALL_K, TALL_MISS_PCT, TALL_SEED),
        ("wide", WIDE_N, WIDE_D, WIDE_K, WIDE_MISS_PCT, WIDE_SEED),
    ] {
        let y = build_case(n, d, k, miss, seed);
        let tall = tag == "tall";

        let c0 = if tall { TALL_PPCA_C0 } else { WIDE_PPCA_C0 };
        let pp_params = PpcaParams {
            n_pcs: k,
            ..PpcaParams::default()
        };
        let pp = ppca_from_init(
            y.as_ref(),
            fixture(c0, d, k),
            Some(pp_params),
            Verbosity::Quiet,
        )
        .unwrap();
        let pp_d = if tall {
            compare(
                &pp,
                n,
                d,
                k,
                TALL_PPCA_SCORES,
                TALL_PPCA_LOADINGS,
                TALL_PPCA_R2CUM,
                TALL_PPCA_COMPLETED,
            )
        } else {
            compare(
                &pp,
                n,
                d,
                k,
                WIDE_PPCA_SCORES,
                WIDE_PPCA_LOADINGS,
                WIDE_PPCA_R2CUM,
                WIDE_PPCA_COMPLETED,
            )
        };
        assert!(pp.converged, "{tag} ppca did not converge");
        assert_diffs(&format!("{tag} ppca"), &pp_d);

        let bp_params = BpcaParams {
            n_pcs: k,
            ..BpcaParams::default()
        };
        let bp = bpca(y.as_ref(), Some(bp_params), Verbosity::Quiet).unwrap();
        let bp_d = if tall {
            compare(
                &bp,
                n,
                d,
                k,
                TALL_BPCA_SCORES,
                TALL_BPCA_LOADINGS,
                TALL_BPCA_R2CUM,
                TALL_BPCA_COMPLETED,
            )
        } else {
            compare(
                &bp,
                n,
                d,
                k,
                WIDE_BPCA_SCORES,
                WIDE_BPCA_LOADINGS,
                WIDE_BPCA_R2CUM,
                WIDE_BPCA_COMPLETED,
            )
        };
        assert_diffs(&format!("{tag} bpca"), &bp_d);
    }
}

/// Proteomics-sized case: every row has missing values, so the BPCA per-row
/// solve path carries the whole E-step. No completed matrix in the fixture;
/// scores and loadings determine it.
#[test]
#[cfg(feature = "large-test")]
// 100 x 1000, k = 5, 40 % missing
fn test_ppca_bpca_match_pcamethods_large() {
    use pcamethods_large::*;
    let (n, d, k) = (LARGE_N, LARGE_D, LARGE_K);
    let y = build_case(n, d, k, LARGE_MISS_PCT, LARGE_SEED);

    let pp = ppca_from_init(
        y.as_ref(),
        fixture(LARGE_PPCA_C0, d, k),
        Some(PpcaParams {
            n_pcs: k,
            ..PpcaParams::default()
        }),
        Verbosity::Quiet,
    )
    .unwrap();
    assert!(pp.converged, "large ppca did not converge");
    let bp = bpca(
        y.as_ref(),
        Some(BpcaParams {
            n_pcs: k,
            ..BpcaParams::default()
        }),
        Verbosity::Quiet,
    )
    .unwrap();

    for (label, res, scores, loadings, r2) in [
        (
            "large ppca",
            &pp,
            LARGE_PPCA_SCORES,
            LARGE_PPCA_LOADINGS,
            LARGE_PPCA_R2CUM,
        ),
        (
            "large bpca",
            &bp,
            LARGE_BPCA_SCORES,
            LARGE_BPCA_LOADINGS,
            LARGE_BPCA_R2CUM,
        ),
    ] {
        for (what, v) in [
            (
                "scores",
                max_diff_signed(res.scores.as_ref(), fixture(scores, n, k)),
            ),
            (
                "loadings",
                max_diff_signed(res.loadings.as_ref(), fixture(loadings, d, k)),
            ),
            ("r2", max_diff_vec(&res.r2_cum, r2)),
        ] {
            assert!(v < TOL, "{label} {what}: max abs diff {v:.3e}");
        }
    }
}
