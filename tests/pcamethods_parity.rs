//! Parity of `ppca` / `bpca` against pcaMethods on LCG-built data with
//! missing values. Fixtures in `tests/pcamethods_fixtures/*.txt`, from
//! `dev/gen_pcamethods_fixtures.R`.

use bixverse_rs::core::math::pca_missing::*;
use bixverse_rs::prelude::*;
use faer::{Mat, MatRef};
use rustc_hash::FxHashMap;

/// Tolerance on every compared quantity. The first runs measured at most
/// 2.3e-13 on the small cases and 5.9e-12 on the large one (PPCA scores both
/// times); the gap is summation order, not method.
const TOL: f64 = 1e-10;

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

/// Parse a fixture file: `#` lines are comments, every other line is a key
/// followed by whitespace-separated values.
fn parse_fixture(text: &str) -> FxHashMap<&str, Vec<f64>> {
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            let key = it.next().expect("fixture line has a key");
            let vals = it
                .map(|v| v.parse().expect("fixture value parses as f64"))
                .collect();
            (key, vals)
        })
        .collect()
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

/// Fit both methods on one fixture case and assert every stored quantity
/// against R, aligning column signs of scores and loadings. `*_COMPLETED` is
/// compared only when the fixture carries it.
fn check_case(label: &str, text: &str) {
    let fx = parse_fixture(text);
    let int = |key: &str| fx[key][0] as usize;
    let (n, d, k) = (int("N"), int("D"), int("K"));
    let y = build_case(n, d, k, int("MISS_PCT") as u64, int("SEED") as u64);

    let pp = ppca_from_init(
        y.as_ref(),
        fixture(&fx["PPCA_C0"], d, k),
        Some(PpcaParams {
            n_pcs: k,
            ..PpcaParams::default()
        }),
        Verbosity::Quiet,
    )
    .unwrap();
    assert!(pp.converged, "{label} ppca did not converge");
    let bp = bpca(
        y.as_ref(),
        Some(BpcaParams {
            n_pcs: k,
            ..BpcaParams::default()
        }),
        Verbosity::Quiet,
    )
    .unwrap();

    for (method, res) in [("PPCA", &pp), ("BPCA", &bp)] {
        let get = |what: &str| fx.get(format!("{method}_{what}").as_str());
        let mut diffs = vec![
            (
                "scores",
                max_diff_signed(res.scores.as_ref(), fixture(get("SCORES").unwrap(), n, k)),
            ),
            (
                "loadings",
                max_diff_signed(
                    res.loadings.as_ref(),
                    fixture(get("LOADINGS").unwrap(), d, k),
                ),
            ),
            ("r2", max_diff_vec(&res.r2_cum, get("R2CUM").unwrap())),
        ];
        if let Some(completed) = get("COMPLETED") {
            diffs.push((
                "completed",
                max_diff(res.completed.as_ref(), fixture(completed, n, d)),
            ));
        }
        for (what, v) in diffs {
            assert!(v < TOL, "{label} {method} {what}: max abs diff {v:.3e}");
        }
    }
}

/// Both fits on a tall (60 x 12, 20 % missing) and a wide (16 x 80, 25 %
/// missing) case. In the wide case R's BPCA prunes the third component, so
/// the ARD path is covered too.
#[test]
fn test_ppca_bpca_match_pcamethods() {
    check_case("tall", include_str!("pcamethods_fixtures/tall.txt"));
    check_case("wide", include_str!("pcamethods_fixtures/wide.txt"));
}

/// Proteomics-sized case: every row has missing values, so the BPCA per-row
/// solve path carries the whole E-step. The fixture leaves out the completed
/// matrix; scores and loadings determine it.
#[test]
#[cfg(feature = "large-test")]
// 100 x 1000, k = 5, 40 % missing
fn test_ppca_bpca_match_pcamethods_large() {
    check_case("large", include_str!("pcamethods_fixtures/large.txt"));
}
