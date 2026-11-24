//! Parity of `kernel_pls` against `pls::kernelpls.fit`, fixtures from
//! `dev/gen_pls_fixtures.R`.

mod pls_fixtures;

use approx::assert_relative_eq;
use bixverse_core::core::math::pls::*;
use bixverse_core::core::math::sparse::SparseFromDense;
use bixverse_core::prelude::*;
use faer::{Mat, MatRef};
use pls_fixtures::*;

/// The fixture's predictors, rebuilt from the same LCG as the R script.
///
/// ### Returns
///
/// The `N_OBS x N_PRED` matrix.
fn lcg_predictors() -> Mat<f64> {
    let mut state = LCG_SEED;
    let mut next = || {
        state = (1664525 * state + 1013904223) % 4294967296;
        state
    };
    let mut x = Mat::<f64>::zeros(N_OBS, N_PRED);
    for i in 0..N_OBS {
        for j in 0..N_PRED {
            let s = next();
            x[(i, j)] = if s % 5 < 2 {
                0.0
            } else {
                (s % 13) as f64 / 3.0
            };
        }
    }
    x
}

/// Compare a fitted matrix with a row-major fixture, column signs aligned.
///
/// ### Params
///
/// * `got` - The Rust result
/// * `want` - Row-major fixture values of the same shape
/// * `signs` - Per-component sign that aligns `got` with R
fn assert_matches(got: MatRef<f64>, want: &[f64], signs: &[f64]) {
    let (nr, nc) = (got.nrows(), got.ncols());
    assert_eq!(want.len(), nr * nc);
    for i in 0..nr {
        for j in 0..nc {
            assert_relative_eq!(
                got[(i, j)] * signs[j],
                want[i * nc + j],
                epsilon = 1e-9,
                max_relative = 1e-8
            );
        }
    }
}

/// Fit one fixture case and check every output against R.
///
/// ### Params
///
/// * `n_pred` - Leading predictors to use
/// * `n_resp` - Response columns
/// * `n_comp` - Components
/// * `y` - Row-major responses
/// * `expected` - Projection, weights, x loadings, y loadings, scores
fn check_case(n_pred: usize, n_resp: usize, n_comp: usize, y: &[f64], expected: [&[f64]; 5]) {
    let full = lcg_predictors();
    let x = full.as_ref().subcols(0, n_pred).to_owned();
    let csr: CompressedSparseData2<f64> =
        CompressedSparseData2::from_dense_matrix(x.as_ref(), CompressedSparseFormat::Csr);
    let y = MatRef::from_row_major_slice(y, N_OBS, n_resp);

    let fit = kernel_pls(&csr, y, n_comp).unwrap();

    let [projection, weights, x_loadings, y_loadings, scores] = expected;
    let signs: Vec<f64> = (0..n_comp)
        .map(|a| {
            let (got, want) = (fit.projection[(0, a)], projection[a]);
            if (got >= 0.0) == (want >= 0.0) {
                1.0
            } else {
                -1.0
            }
        })
        .collect();

    assert_matches(fit.projection.as_ref(), projection, &signs);
    assert_matches(fit.weights.as_ref(), weights, &signs);
    assert_matches(fit.x_loadings.as_ref(), x_loadings, &signs);
    assert_matches(fit.y_loadings.as_ref(), y_loadings, &signs);
    assert_matches(fit.x_scores.as_ref(), scores, &signs);
}

#[test]
fn test_kernel_pls_single_response_matches_r() {
    check_case(
        SINGLE_NPRED,
        SINGLE_NRESP,
        SINGLE_NCOMP,
        SINGLE_Y,
        [
            SINGLE_PROJECTION,
            SINGLE_WEIGHTS,
            SINGLE_X_LOADINGS,
            SINGLE_Y_LOADINGS,
            SINGLE_SCORES,
        ],
    );
}

#[test]
fn test_kernel_pls_multi_response_matches_r() {
    check_case(
        MULTI_NPRED,
        MULTI_NRESP,
        MULTI_NCOMP,
        MULTI_Y,
        [
            MULTI_PROJECTION,
            MULTI_WEIGHTS,
            MULTI_X_LOADINGS,
            MULTI_Y_LOADINGS,
            MULTI_SCORES,
        ],
    );
}

#[test]
fn test_kernel_pls_more_responses_than_predictors_matches_r() {
    check_case(
        WIDE_NPRED,
        WIDE_NRESP,
        WIDE_NCOMP,
        WIDE_Y,
        [
            WIDE_PROJECTION,
            WIDE_WEIGHTS,
            WIDE_X_LOADINGS,
            WIDE_Y_LOADINGS,
            WIDE_SCORES,
        ],
    );
}

#[test]
fn test_kernel_pls_x_means_match_r() {
    let csr: CompressedSparseData2<f64> = CompressedSparseData2::from_dense_matrix(
        lcg_predictors().as_ref(),
        CompressedSparseFormat::Csr,
    );
    let y = MatRef::from_row_major_slice(MULTI_Y, N_OBS, MULTI_NRESP);
    let fit = kernel_pls(&csr, y, 1).unwrap();
    for (got, want) in fit.x_means.iter().zip(MULTI_X_MEANS) {
        assert_relative_eq!(*got, *want, max_relative = 1e-12);
    }
}

#[test]
fn test_kernel_pls_rejects_too_many_components() {
    let csr: CompressedSparseData2<f64> = CompressedSparseData2::from_dense_matrix(
        lcg_predictors().as_ref(),
        CompressedSparseFormat::Csr,
    );
    let y = MatRef::from_row_major_slice(MULTI_Y, N_OBS, MULTI_NRESP);
    assert!(matches!(
        kernel_pls(&csr, y, N_PRED + 1),
        Err(CoreErrors::InvalidArgument(_))
    ));
}
