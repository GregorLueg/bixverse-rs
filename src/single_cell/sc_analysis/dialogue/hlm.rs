//! DIALOGUE stage two: hierarchical modelling.
//!
//! For every ordered pair of cell types and every programme they share, ask of
//! each candidate gene: does a cell's own programme score track how much of
//! that gene the *other* cell type expresses in the same sample? The model is
//!
//! ```text
//! score ~ (1 | sample) + partner_pseudobulk_of_gene + cellQ + tme.qc
//! ```
//!
//! fitted by REML with Satterthwaite degrees of freedom. The answer kept is a
//! signed `-log10 p` on the gene's coefficient.

use rayon::prelude::*;
use rustc_hash::FxHashSet;

use crate::core::math::mixed_model::{
    RandomInterceptParams, RandomInterceptStats, fit_random_intercept,
};
use crate::prelude::*;
use crate::single_cell::mc_generation::cell_aggregation_utils::{
    PseudoBulk, pseudo_bulk_genes_dense,
};
use crate::single_cell::sc_analysis::dialogue::params::DialogueParams;
use crate::single_cell::sc_analysis::dialogue::pmd::{CellTypeView, DialogueStep1Result};

use faer::Mat;

/////////////////////
// GeneAssociation //
/////////////////////

/// One gene's association with one partner cell type, for one programme.
#[derive(Clone, Copy, Debug)]
pub struct GeneAssociation {
    /// Cell type the gene belongs to, whose signature it came from.
    pub cell_type: usize,
    /// Cell type whose programme score was the response.
    pub partner: usize,
    /// Programme index.
    pub programme: usize,
    /// Gene index.
    pub gene: usize,
    /// Whether the gene entered from the up or the down side of the signature.
    pub up: bool,
    /// Coefficient on the pseudo-bulk predictor.
    pub estimate: f64,
    /// Two-sided p-value on that coefficient.
    pub p_value: f64,
    /// Signed `-log10 p`, positive for a positive coefficient.
    ///
    /// Upstream calls this a z-score and it is not one: `|Z| = 1` corresponds
    /// to a one-sided p of 0.1, not to a standard deviation. The name is kept
    /// because every threshold downstream is expressed in it.
    pub z: f64,
}

/// What stage two produced.
#[derive(Clone, Debug, Default)]
pub struct DialogueStep2Result {
    /// Every fit that converged, one row per gene, partner and programme.
    pub associations: Vec<GeneAssociation>,
}

///////////////////////////
// Sufficient statistics //
///////////////////////////

/// Per-sample reductions of everything that does not change between genes.
///
/// See the module documentation: these six numbers plus a sample-level
/// predictor and covariate reconstruct the whole design.
#[derive(Clone, Debug, Default)]
struct SampleSummary {
    /// Cells contributed by this sample.
    n: f64,
    /// Sum of the quality covariate.
    sum_q: f64,
    /// Sum of its square.
    sum_qq: f64,
    /// Sum of the response.
    sum_y: f64,
    /// Sum of its square.
    sum_yy: f64,
    /// Sum of their product.
    sum_qy: f64,
}

/// Reduces one cell type's cells to per-sample summaries for one programme.
///
/// ### Params
///
/// * `view` - The responding cell type
/// * `scores` - Its programme scores
/// * `programme` - Which programme is the response
/// * `sample_slots` - Sample slot per cell, `usize::MAX` for cells to skip
/// * `n_slots` - Number of retained samples
///
/// ### Returns
///
/// One summary per retained sample.
fn summarise(
    view: &CellTypeView,
    scores: &Mat<f64>,
    programme: usize,
    sample_slots: &[usize],
    n_slots: usize,
) -> Vec<SampleSummary> {
    let mut out = vec![SampleSummary::default(); n_slots];
    for row in 0..view.cells.len() {
        let slot = sample_slots[row];
        if slot == usize::MAX {
            continue;
        }
        let q = view.quality[row];
        let y = scores[(row, programme)];
        let s = &mut out[slot];
        s.n += 1.0;
        s.sum_q += q;
        s.sum_qq += q * q;
        s.sum_y += y;
        s.sum_yy += y * y;
        s.sum_qy += q * y;
    }
    out
}

/// Assembles the mixed-model sufficient statistics for one gene.
///
/// The design is `[1, x]`, then the quality covariate and the partner's mean
/// quality if they are switched on. `x` and `tme_qc` are sample-level, so every
/// cross product involving them factors out of the per-sample sums.
///
/// ### Params
///
/// * `summaries` - Per-sample reductions
/// * `predictor` - The gene's pseudo-bulk value per retained sample
/// * `tme_qc` - The partner's mean quality per retained sample
/// * `use_q` - Include the cell-level quality covariate
/// * `use_t` - Include `tme_qc`
///
/// ### Returns
///
/// The statistics, ready for [fit_random_intercept]. Column 1 is always the
/// gene's coefficient.
fn assemble(
    summaries: &[SampleSummary],
    predictor: &[f64],
    tme_qc: &[f64],
    use_q: bool,
    use_t: bool,
) -> RandomInterceptStats {
    let p = 2 + usize::from(use_q) + usize::from(use_t);
    let g = summaries.len();

    let mut xtx = Mat::<f64>::zeros(p, p);
    let mut xty = vec![0.0_f64; p];
    let mut yty = 0.0_f64;
    let mut group_s = Mat::<f64>::zeros(g, p);
    let mut group_t = vec![0.0_f64; g];
    let mut group_n = vec![0.0_f64; g];
    let mut n_total = 0.0_f64;

    // Column 0 is the intercept, 1 the gene, then the optional covariates.
    let q_idx = if use_q { Some(2) } else { None };
    let t_idx = if use_t {
        Some(if use_q { 3 } else { 2 })
    } else {
        None
    };
    // Per-column scalars for the sample-level entries. Only the gene column
    // changes between samples, so the rest is set once.
    let mut scalar = vec![0.0_f64; p];
    scalar[0] = 1.0;

    for (k, s) in summaries.iter().enumerate() {
        scalar[1] = predictor[k];
        if let Some(ti) = t_idx {
            scalar[ti] = tme_qc[k];
        }

        // Accumulated straight in: everything sample-level factors out of the
        // per-sample sums, and only the quality covariate carries its own
        // second moments.
        for i in 0..p {
            for j in 0..p {
                xtx[(i, j)] += if Some(i) != q_idx && Some(j) != q_idx {
                    s.n * scalar[i] * scalar[j]
                } else if Some(i) == q_idx && Some(j) == q_idx {
                    s.sum_qq
                } else if Some(i) == q_idx {
                    s.sum_q * scalar[j]
                } else {
                    s.sum_q * scalar[i]
                };
            }
            xty[i] += if Some(i) == q_idx {
                s.sum_qy
            } else {
                scalar[i] * s.sum_y
            };
            // Per-sample column sums of the design.
            group_s[(k, i)] = if Some(i) == q_idx {
                s.sum_q
            } else {
                s.n * scalar[i]
            };
        }
        yty += s.sum_yy;
        group_t[k] = s.sum_y;
        group_n[k] = s.n;
        n_total += s.n;
    }

    RandomInterceptStats {
        xtx,
        xty,
        yty,
        n: n_total as usize,
        p,
        group_n,
        group_s,
        group_t,
    }
}

/// Signed `-log10 p` from a coefficient and its two-sided p-value.
///
/// Upstream builds this by halving the two-sided p on the side the coefficient
/// points, taking the complement on the other, and reporting `-log10` of the
/// smaller tail with the sign of the larger. The net effect is what it says on
/// the tin, with a floor so a p of exactly zero does not become infinite.
///
/// ### Params
///
/// * `estimate` - The coefficient
/// * `p_value` - Its two-sided p-value
/// * `floor` - Smallest non-zero p seen, substituted for an exact zero
///
/// ### Returns
///
/// The signed score, or `f64::NAN` if either input is not finite.
fn signed_log_p(estimate: f64, p_value: f64, floor: f64) -> f64 {
    if !estimate.is_finite() || !p_value.is_finite() {
        return f64::NAN;
    }
    let p = if p_value <= 0.0 { floor } else { p_value };
    let one_sided = if estimate > 0.0 {
        p / 2.0
    } else {
        1.0 - p / 2.0
    };
    if one_sided > 0.5 {
        // The other tail is the informative one, and it carries the sign.
        (1.0 - one_sided).log10()
    } else {
        -one_sided.log10()
    }
}

///////////////
// The stage //
///////////////

/// Runs DIALOGUE stage two.
///
/// ### Params
///
/// * `reader` - Gene-major reader
/// * `views` - One resolved view per cell type
/// * `step1` - Stage one output
/// * `params` - The full parameter set
/// * `verbose` - `0` silent, non-zero prints progress
///
/// ### Returns
///
/// The [Step2Result], or the first error encountered.
pub(crate) fn run_step2<S: SingleCellReading>(
    reader: &S,
    views: &[CellTypeView],
    step1: &DialogueStep1Result,
    params: &DialogueParams,
    verbose: usize,
) -> Result<DialogueStep2Result, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let n_types = views.len();
    let hlm = &params.hlm;
    let k = params.pmd.k;

    let fit_params = RandomInterceptParams {
        satterthwaite: hlm.satterthwaite,
        ..Default::default()
    };

    let mut associations = Vec::new();
    let mut n_failed = 0usize;

    for own in 0..n_types {
        // Depends only on `own`, so it is built once rather than per partner.
        let own_ok: FxHashSet<usize> = views[own]
            .samples
            .iter()
            .zip(views[own].rows_by_sample.iter())
            .filter(|(_, rows)| rows.len() >= hlm.min_cells_per_sample)
            .map(|(s, _)| *s)
            .collect();

        for partner in 0..n_types {
            if own == partner {
                continue;
            }

            // Samples carrying enough cells in both cell types.
            let shared: Vec<usize> = views[partner]
                .samples
                .iter()
                .zip(views[partner].rows_by_sample.iter())
                .filter(|(s, rows)| rows.len() >= hlm.min_cells_per_sample && own_ok.contains(s))
                .map(|(s, _)| *s)
                .collect();
            if shared.len() < 3 {
                if verbosity.normal_verbosity() {
                    println!(
                        "Cell types {own} -> {partner}: only {} usable samples, skipping.",
                        shared.len()
                    );
                }
                continue;
            }

            // Slot per cell of the responding cell type.
            let mut partner_slots = vec![usize::MAX; views[partner].cells.len()];
            for (slot, s) in shared.iter().enumerate() {
                let idx = views[partner]
                    .samples
                    .binary_search(s)
                    .expect("shared sample is present");
                for &row in views[partner].rows_by_sample[idx].iter() {
                    partner_slots[row] = slot;
                }
            }

            // Resolve each shared sample against the gene-owning cell type
            // once; both quantities below are derived from the same rows.
            let own_rows: Vec<&Vec<usize>> = shared
                .iter()
                .map(|s| {
                    let idx = views[own].samples.binary_search(s).expect("shared sample");
                    &views[own].rows_by_sample[idx]
                })
                .collect();

            // The partner's mean quality per sample, upstream's tme.qc.
            let tme_qc: Vec<f64> = own_rows
                .iter()
                .map(|rows| {
                    rows.iter().map(|&r| views[own].quality[r]).sum::<f64>() / rows.len() as f64
                })
                .collect();

            // Cells of the gene-owning cell type, grouped by shared sample.
            let own_groups: Vec<Vec<usize>> = own_rows
                .iter()
                .map(|rows| rows.iter().map(|&r| views[own].cells[r]).collect())
                .collect();

            for programme in 0..k {
                if !step1.mcp_cell_types[programme].contains(&own)
                    || !step1.mcp_cell_types[programme].contains(&partner)
                {
                    continue;
                }
                let signature = &step1.signatures[own][programme];
                let genes = signature.all();
                if genes.is_empty() {
                    continue;
                }
                let up: FxHashSet<usize> = signature.up.iter().copied().collect();

                let pseudo =
                    pseudo_bulk_genes_dense(reader, &genes, &own_groups, PseudoBulk::Norm, 0)?;
                let summaries = summarise(
                    &views[partner],
                    &step1.scores[partner],
                    programme,
                    &partner_slots,
                    shared.len(),
                );

                let fitted: Vec<Option<GeneAssociation>> = genes
                    .par_iter()
                    .enumerate()
                    .map(|(row, &gene)| {
                        let predictor: Vec<f64> =
                            (0..shared.len()).map(|s| pseudo[(row, s)]).collect();
                        // A gene the partner never expresses carries no
                        // information and makes the design rank deficient.
                        let first = predictor[0];
                        if predictor.iter().all(|v| (v - first).abs() < f64::EPSILON) {
                            return None;
                        }
                        let stats = assemble(
                            &summaries,
                            &predictor,
                            &tme_qc,
                            hlm.use_cell_quality,
                            hlm.use_tme_qc,
                        );
                        let fit = fit_random_intercept(&stats, Some(fit_params)).ok()?;
                        let estimate = fit.beta[1];
                        let p_value = fit.p_value[1];
                        if !estimate.is_finite() || !p_value.is_finite() {
                            return None;
                        }
                        Some(GeneAssociation {
                            cell_type: own,
                            partner,
                            programme,
                            gene,
                            up: up.contains(&gene),
                            estimate,
                            p_value,
                            z: f64::NAN,
                        })
                    })
                    .collect();

                n_failed += fitted.iter().filter(|f| f.is_none()).count();
                associations.extend(fitted.into_iter().flatten());
            }
        }
    }

    // The signed score needs a floor for the exact zeros, and upstream takes it
    // from the smallest non-zero p in the batch rather than from machine
    // epsilon.
    let floor = associations
        .iter()
        .map(|a| a.p_value)
        .filter(|p| *p > 0.0)
        .fold(f64::INFINITY, f64::min);
    let floor = if floor.is_finite() {
        floor
    } else {
        f64::MIN_POSITIVE
    };
    for a in associations.iter_mut() {
        a.z = signed_log_p(a.estimate, a.p_value, floor);
    }

    if verbosity.normal_verbosity() {
        println!(
            "Stage two: {} gene associations fitted, {} skipped.",
            associations.len(),
            n_failed
        );
    }

    Ok(DialogueStep2Result { associations })
}

/// Converts a signed `-log10 p` back to a one-sided p-value.
///
/// The inverse of [signed_log_p], and what the meta-analysis in stage three
/// needs to combine partners. Upstream's `get.pval.from.zscores`.
///
/// ### Params
///
/// * `z` - Signed `-log10 p`
/// * `positive_side` - Test the positive direction, or the negative one
///
/// ### Returns
///
/// The one-sided p-value, or `f64::NAN` for a non-finite input.
pub(crate) fn p_from_signed_log(z: f64, positive_side: bool) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    let signed = if positive_side { z } else { -z };
    let p = 10f64.powf(-signed.abs());
    if signed < 0.0 { 1.0 - p } else { p }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A synthetic run of the shape stage two actually sees: unbalanced
    /// samples, a cell-level response and quality covariate, and a
    /// sample-level predictor and second covariate.
    struct Fixture {
        view: CellTypeView,
        scores: Mat<f64>,
        slots: Vec<usize>,
        predictor: Vec<f64>,
        tme_qc: Vec<f64>,
    }

    fn fixture() -> Fixture {
        let sizes = [5usize, 3, 7, 4, 6];
        let cells: Vec<usize> = (0..sizes.iter().sum::<usize>()).collect();
        let mut sample_ids = Vec::new();
        for (s, &n) in sizes.iter().enumerate() {
            sample_ids.extend(std::iter::repeat_n(s, n));
        }
        // Deterministic but irregular; the exact values do not matter, only
        // that nothing is degenerate.
        let quality: Vec<f64> = (0..cells.len())
            .map(|i| ((i * 13 % 17) as f64 - 8.0) / 5.0)
            .collect();
        let view = CellTypeView::new(&cells, &sample_ids, &quality).unwrap();

        let scores = Mat::<f64>::from_fn(cells.len(), 2, |i, j| {
            ((i * 7 + j * 29) % 23) as f64 / 11.0 - 1.0
        });
        let slots = sample_ids.clone();
        let predictor = vec![0.4, -1.2, 2.3, 0.05, -0.7];
        let tme_qc = vec![-0.3, 0.9, 0.15, -1.1, 0.6];

        Fixture {
            view,
            scores,
            slots,
            predictor,
            tme_qc,
        }
    }

    /// Builds the design matrix explicitly, the way a naive implementation
    /// would, so the reduced form can be checked against it.
    fn explicit_design(f: &Fixture, use_q: bool, use_t: bool) -> (Mat<f64>, Vec<f64>) {
        let n = f.view.cells.len();
        let p = 2 + usize::from(use_q) + usize::from(use_t);
        let design = Mat::<f64>::from_fn(n, p, |i, j| {
            let s = f.slots[i];
            match j {
                0 => 1.0,
                1 => f.predictor[s],
                2 if use_q => f.view.quality[i],
                2 => f.tme_qc[s],
                _ => f.tme_qc[s],
            }
        });
        (design, (0..n).map(|i| f.scores[(i, 0)]).collect())
    }

    /// The whole point of stage two: the sufficient statistics rebuilt from six
    /// numbers per sample must equal the ones a full pass over the cells would
    /// produce.
    ///
    /// If this drifts, every mixed model in the stage is quietly wrong and
    /// nothing else would catch it.
    #[test]
    fn test_assembled_statistics_match_the_explicit_design() {
        let f = fixture();
        for (use_q, use_t) in [(true, true), (true, false), (false, true), (false, false)] {
            let summaries = summarise(&f.view, &f.scores, 0, &f.slots, f.predictor.len());
            let reduced = assemble(&summaries, &f.predictor, &f.tme_qc, use_q, use_t);

            let (design, y) = explicit_design(&f, use_q, use_t);
            let direct =
                RandomInterceptStats::from_design(design.as_ref(), &y, &f.slots, f.predictor.len())
                    .unwrap();

            assert_eq!(reduced.n, direct.n, "n mismatch at ({use_q}, {use_t})");
            assert_eq!(reduced.p, direct.p);
            assert_relative_eq!(reduced.yty, direct.yty, max_relative = 1e-12);
            for i in 0..reduced.p {
                assert_relative_eq!(reduced.xty[i], direct.xty[i], epsilon = 1e-10);
                for j in 0..reduced.p {
                    assert_relative_eq!(reduced.xtx[(i, j)], direct.xtx[(i, j)], epsilon = 1e-10);
                }
            }
            for g in 0..reduced.group_n.len() {
                assert_relative_eq!(reduced.group_n[g], direct.group_n[g], epsilon = 1e-12);
                assert_relative_eq!(reduced.group_t[g], direct.group_t[g], epsilon = 1e-10);
                for i in 0..reduced.p {
                    assert_relative_eq!(
                        reduced.group_s[(g, i)],
                        direct.group_s[(g, i)],
                        epsilon = 1e-10
                    );
                }
            }
        }
    }

    /// And the fits agree, which is the property that actually matters.
    #[test]
    fn test_assembled_fit_matches_the_explicit_fit() {
        let f = fixture();
        let summaries = summarise(&f.view, &f.scores, 0, &f.slots, f.predictor.len());
        let reduced = assemble(&summaries, &f.predictor, &f.tme_qc, true, true);
        let (design, y) = explicit_design(&f, true, true);
        let direct =
            RandomInterceptStats::from_design(design.as_ref(), &y, &f.slots, f.predictor.len())
                .unwrap();

        let a = fit_random_intercept(&reduced, None).unwrap();
        let b = fit_random_intercept(&direct, None).unwrap();

        assert_relative_eq!(a.lambda, b.lambda, max_relative = 1e-9);
        assert_relative_eq!(a.sigma_sq, b.sigma_sq, max_relative = 1e-10);
        for j in 0..a.beta.len() {
            assert_relative_eq!(a.beta[j], b.beta[j], epsilon = 1e-9);
            assert_relative_eq!(a.se[j], b.se[j], epsilon = 1e-9);
            assert_relative_eq!(a.df[j], b.df[j], max_relative = 1e-6);
            assert_relative_eq!(a.p_value[j], b.p_value[j], epsilon = 1e-9);
        }
    }

    /// Changing the gene changes only one column, so the reduction has to keep
    /// tracking the explicit design as the predictor moves.
    #[test]
    fn test_assembled_statistics_track_a_changing_predictor() {
        let mut f = fixture();
        let summaries = summarise(&f.view, &f.scores, 1, &f.slots, f.predictor.len());
        for shift in [0.0, 1.5, -3.0] {
            f.predictor = f.predictor.iter().map(|v| v + shift).collect();
            let reduced = assemble(&summaries, &f.predictor, &f.tme_qc, true, true);
            let (design, _) = explicit_design(&f, true, true);
            // The response is programme 1, matching the summaries above.
            let y: Vec<f64> = (0..f.view.cells.len()).map(|i| f.scores[(i, 1)]).collect();
            let direct =
                RandomInterceptStats::from_design(design.as_ref(), &y, &f.slots, f.predictor.len())
                    .unwrap();
            for i in 0..reduced.p {
                assert_relative_eq!(reduced.xty[i], direct.xty[i], epsilon = 1e-10);
                for j in 0..reduced.p {
                    assert_relative_eq!(reduced.xtx[(i, j)], direct.xtx[(i, j)], epsilon = 1e-10);
                }
            }
        }
    }

    /// The signed score is a `-log10 p` carrying the coefficient's sign, and
    /// `|Z| = 1` is a one-sided p of 0.1, not a standard deviation.
    #[test]
    fn test_signed_log_p_semantics() {
        // A positive coefficient with a two-sided p of 0.2 is a one-sided p of
        // 0.1, so Z is exactly 1.
        assert_relative_eq!(signed_log_p(0.5, 0.2, 1e-300), 1.0, max_relative = 1e-12);
        // The same p with a negative coefficient mirrors it.
        assert_relative_eq!(signed_log_p(-0.5, 0.2, 1e-300), -1.0, max_relative = 1e-12);
        // A strong positive result.
        assert_relative_eq!(signed_log_p(1.0, 2e-6, 1e-300), 6.0, max_relative = 1e-9);
        // A p of exactly zero falls back to the supplied floor.
        assert!(signed_log_p(1.0, 0.0, 1e-20).is_finite());
        assert!(signed_log_p(f64::NAN, 0.1, 1e-300).is_nan());
    }

    /// And the inverse round-trips on the side the gene came in on.
    #[test]
    fn test_p_from_signed_log_round_trips() {
        for (estimate, p) in [(0.5, 0.2), (-0.5, 0.2), (1.0, 2e-6), (-2.0, 0.5)] {
            let z = signed_log_p(estimate, p, 1e-300);
            let recovered = p_from_signed_log(z, estimate > 0.0);
            assert_relative_eq!(recovered, p / 2.0, max_relative = 1e-9);
            // The opposite side is the complement.
            let other = p_from_signed_log(z, estimate <= 0.0);
            assert_relative_eq!(recovered + other, 1.0, max_relative = 1e-9);
        }
    }
}
