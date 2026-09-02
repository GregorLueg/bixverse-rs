//! End-to-end parity for the edgeR quasi-likelihood chain, against edgeR
//! 4.8.2.
//!
//! `edge-rs` gates its own pieces against R already. What this file checks is
//! that the chain assembled in `methods::dge_bulk` is the chain edgeR runs:
//! the same filter, the same TMM factors, the same library sizes carried
//! through the subset, the same dispersion route, and the same F-test on top.
//! Getting any one of those in the wrong order still produces plausible
//! numbers, which is exactly why it is worth pinning.
//!
//! The fixture is 200 genes over eight samples in two groups of four. See
//! `tests/edger_fixtures/mod.rs` for how the counts are reproduced on both
//! sides without any float data crossing as text.

#![cfg(feature = "dge")]

mod edger_fixtures;

use approx::assert_relative_eq;
use bixverse_rs::methods::dge_bulk::{EdgeRQlParams, run_edger_ql};
use edge_rs::core::normalisation::NormMethod;
use edge_rs::glm::test::Tested;

use edger_fixtures as fx;

////////////////
// Tolerances //
////////////////

/// Log-fold changes and abundances. Both are closed-form given the fit, so
/// they carry only the fit's own error.
const TOL_ESTIMATE: f64 = 1e-9;

/// The F statistic, which divides two quantities that each carry the fit's
/// error.
const TOL_F: f64 = 1e-8;

/// P-values. The F tail amplifies whatever error is in the statistic, and the
/// adjusted values inherit that.
const TOL_P: f64 = 1e-7;

///////////
// Tests //
///////////

/// The whole chain against `filterByExpr` -> `calcNormFactors` -> `glmQLFit`
/// -> `glmQLFTest` -> `topTags`.
#[test]
fn test_run_edger_ql_matches_edger() {
    let counts = fx::counts();
    let design = fx::design();

    let got = run_edger_ql(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![1]),
        &EdgeRQlParams::default(),
    )
    .expect("run_edger_ql failed");

    let kept: Vec<usize> = got
        .genes_to_keep
        .iter()
        .enumerate()
        .filter(|(_, k)| **k)
        .map(|(gene, _)| gene)
        .collect();
    assert_eq!(kept, fx::KEPT, "filterByExpr disagrees");

    assert_relative_eq!(
        got.log_fc.as_slice(),
        fx::LOG_FC,
        max_relative = TOL_ESTIMATE
    );
    assert_relative_eq!(
        got.log_cpm.as_slice(),
        fx::LOG_CPM,
        max_relative = TOL_ESTIMATE
    );
    assert_relative_eq!(got.f_stat.as_slice(), fx::F_STAT, max_relative = TOL_F);
    assert_relative_eq!(got.p_val.as_slice(), fx::P_VALUE, max_relative = TOL_P);
    assert_relative_eq!(got.fdr.as_slice(), fx::FDR, max_relative = TOL_P);
}

/// A one-column contrast on the group is the same test as naming its
/// coefficient.
#[test]
fn test_a_contrast_reproduces_the_coefficient_test() {
    let counts = fx::counts();
    let design = fx::design();
    let params = EdgeRQlParams::default();

    let by_coef = run_edger_ql(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![1]),
        &params,
    )
    .expect("run_edger_ql failed");

    let by_contrast = run_edger_ql(
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
    .expect("run_edger_ql failed");

    assert_relative_eq!(
        by_contrast.p_val.as_slice(),
        by_coef.p_val.as_slice(),
        max_relative = TOL_P
    );
    assert_relative_eq!(
        by_contrast.log_fc.as_slice(),
        by_coef.log_fc.as_slice(),
        max_relative = TOL_ESTIMATE
    );
}

/// Turning the filter off keeps every gene, and `min_mean` cuts on top of it.
#[test]
fn test_the_filters_compose() {
    let counts = fx::counts();
    let design = fx::design();

    let unfiltered = run_edger_ql(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![1]),
        &EdgeRQlParams {
            filter: false,
            norm_method: NormMethod::None,
            ..Default::default()
        },
    )
    .expect("run_edger_ql failed");
    assert_eq!(
        unfiltered.genes_to_keep.iter().filter(|k| **k).count(),
        fx::N_GENES
    );
    assert_eq!(unfiltered.p_val.len(), fx::N_GENES);

    let by_mean = run_edger_ql(
        &counts,
        fx::N_GENES,
        fx::N_SAMPLES,
        &design,
        fx::N_COEF,
        &Tested::Coef(vec![1]),
        &EdgeRQlParams {
            filter: false,
            min_mean: 20.0,
            norm_method: NormMethod::None,
            ..Default::default()
        },
    )
    .expect("run_edger_ql failed");

    let kept = by_mean.genes_to_keep.iter().filter(|k| **k).count();
    assert!(kept > 0 && kept < fx::N_GENES, "min_mean kept {kept} genes");
    assert_eq!(by_mean.p_val.len(), kept);

    // The mask is over the whole universe, so it has to line up with the mean.
    for (gene, flag) in by_mean.genes_to_keep.iter().enumerate() {
        let mean = counts[gene * fx::N_SAMPLES..(gene + 1) * fx::N_SAMPLES]
            .iter()
            .sum::<f64>()
            / fx::N_SAMPLES as f64;
        assert_eq!(*flag, mean >= 20.0, "gene {gene} at mean {mean}");
    }
}

/// Filtering everything out is an error, not an empty table.
#[test]
fn test_an_empty_filter_is_rejected() {
    let counts = fx::counts();
    let design = fx::design();

    assert!(
        run_edger_ql(
            &counts,
            fx::N_GENES,
            fx::N_SAMPLES,
            &design,
            fx::N_COEF,
            &Tested::Coef(vec![1]),
            &EdgeRQlParams {
                filter: false,
                min_mean: 1e9,
                ..Default::default()
            },
        )
        .is_err()
    );
}
