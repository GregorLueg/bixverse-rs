//! End-to-end parity for the limma linear model chain, against limma 3.66.0
//! and edgeR 4.8.2.
//!
//! Same point as `tests/edger_bulk.rs`: `edge-rs` gates its own pieces against
//! R already, so what is pinned here is that the chain assembled in
//! `methods::dge_bulk` is the chain limma runs. The library sizes carried
//! through the subset, the TMM factors folded in before voom rather than
//! recomputed inside it, and the empirical Bayes trend following the route are
//! all things that produce plausible numbers when got wrong.
//!
//! The fixture is the same 200 genes over eight samples the edgeR file uses.

#![cfg(feature = "dge")]

mod edger_fixtures;

use approx::assert_relative_eq;
use bixverse_rs::methods::dge_bulk::{LimmaParams, LimmaRoute, run_limma_dge};
use edge_rs::glm::test::Tested;

use edger_fixtures as fx;

////////////////
// Tolerances //
////////////////

/// Everything that comes out of the fit: log-fold changes, average
/// expressions, the moderated t and the B statistic. Worst observed is 1.9e-12,
/// on the voom coefficients, where the weighted solve runs against weights the
/// two sides derive through different orderings of the same lowess.
const TOL_FIT: f64 = 1e-11;

/// P-values and the adjusted ones. Worst observed is 1.5e-13. Tighter than the
/// statistics themselves because the t tail is flat where the error lives.
const TOL_P: f64 = 1e-12;

///////////
// Tests //
///////////

/// `voomLmFit` -> `eBayes` -> `topTable`.
#[test]
fn test_run_limma_voom_matches_limma() {
    let counts = fx::counts();
    let design = fx::design();

    let got = run_limma_dge(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![1]),
        &LimmaParams::default(),
    )
    .expect("the voom chain runs");

    let kept: Vec<usize> = got
        .genes_to_keep
        .iter()
        .enumerate()
        .filter(|(_, k)| **k)
        .map(|(g, _)| g)
        .collect();
    assert_eq!(kept, fx::KEPT, "filterByExpr keeps the same genes");

    for (i, want) in fx::VOOM_LOG_FC.iter().enumerate() {
        assert_relative_eq!(got.log_fc[i], want, max_relative = TOL_FIT);
        assert_relative_eq!(
            got.ave_expr[i],
            fx::VOOM_AVE_EXPR[i],
            max_relative = TOL_FIT
        );
        assert_relative_eq!(got.t_stat[i], fx::VOOM_T[i], max_relative = TOL_FIT);
        assert_relative_eq!(got.b_stat[i], fx::VOOM_B[i], max_relative = TOL_FIT);
        assert_relative_eq!(got.p_val[i], fx::VOOM_P[i], max_relative = TOL_P);
        assert_relative_eq!(got.fdr[i], fx::VOOM_ADJ_P[i], max_relative = TOL_P);
    }
}

/// log-CPM -> `lmFit` -> `eBayes(trend = TRUE)` -> `topTable`.
#[test]
fn test_run_limma_trend_matches_limma() {
    let counts = fx::counts();
    let design = fx::design();

    let got = run_limma_dge(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![1]),
        &LimmaParams {
            route: LimmaRoute::Trend,
            ..Default::default()
        },
    )
    .expect("the limma-trend chain runs");

    for (i, want) in fx::TREND_LOG_FC.iter().enumerate() {
        assert_relative_eq!(got.log_fc[i], want, max_relative = TOL_FIT);
        assert_relative_eq!(
            got.ave_expr[i],
            fx::TREND_AVE_EXPR[i],
            max_relative = TOL_FIT
        );
        assert_relative_eq!(got.t_stat[i], fx::TREND_T[i], max_relative = TOL_FIT);
        assert_relative_eq!(got.b_stat[i], fx::TREND_B[i], max_relative = TOL_FIT);
        assert_relative_eq!(got.p_val[i], fx::TREND_P[i], max_relative = TOL_P);
        assert_relative_eq!(got.fdr[i], fx::TREND_ADJ_P[i], max_relative = TOL_P);
    }
}

/// A one-column contrast picking out the group coefficient has to reproduce
/// the coefficient test, which pins that `contrasts_fit` runs before `ebayes`
/// and that the contrast layout is read column-major.
#[test]
fn test_a_contrast_reproduces_the_coefficient_test() {
    let counts = fx::counts();
    let design = fx::design();
    let params = LimmaParams::default();

    let by_coef = run_limma_dge(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![1]),
        &params,
    )
    .expect("the coefficient test runs");

    let by_contrast = run_limma_dge(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Contrast {
            values: vec![0.0, 1.0],
            n_contrasts: 1,
        },
        &params,
    )
    .expect("the contrast test runs");

    for i in 0..by_coef.log_fc.len() {
        assert_relative_eq!(
            by_contrast.log_fc[i],
            by_coef.log_fc[i],
            max_relative = TOL_FIT
        );
        assert_relative_eq!(
            by_contrast.t_stat[i],
            by_coef.t_stat[i],
            max_relative = TOL_FIT
        );
        assert_relative_eq!(by_contrast.p_val[i], by_coef.p_val[i], max_relative = TOL_P);
    }
}

/// `topTable` tabulates one column, so several coefficients at once has to be
/// refused rather than silently tested as the first of them.
#[test]
fn test_several_coefficients_are_rejected() {
    let counts = fx::counts();
    let design = fx::design();

    let got = run_limma_dge(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![0, 1]),
        &LimmaParams::default(),
    );

    assert!(got.is_err(), "a multi-coefficient test is not tabulable");
}

/// The mean-count floor composes with `filterByExpr`, and an impossible one is
/// an error rather than an empty fit.
#[test]
fn test_the_filters_compose() {
    let counts = fx::counts();
    let design = fx::design();

    let got = run_limma_dge(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![1]),
        &LimmaParams {
            min_mean: 40.0,
            ..Default::default()
        },
    )
    .expect("the floor leaves genes behind");

    let n_kept = got.genes_to_keep.iter().filter(|k| **k).count();
    assert!(n_kept < fx::KEPT.len(), "the floor removed something");
    assert_eq!(got.log_fc.len(), n_kept, "one row per kept gene");

    for (gene, keep) in got.genes_to_keep.iter().enumerate() {
        if *keep {
            let mean = counts[gene * fx::N_SAMPLES..(gene + 1) * fx::N_SAMPLES]
                .iter()
                .sum::<f64>()
                / fx::N_SAMPLES as f64;
            assert!(mean >= 40.0, "gene {gene} survived below the floor");
        }
    }

    let empty = run_limma_dge(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![1]),
        &LimmaParams {
            min_mean: 1e9,
            ..Default::default()
        },
    );
    assert!(empty.is_err(), "an empty filter is rejected");
}
