//! Bulk differential expression: edgeR's quasi-likelihood chain, and
//! limma-voom.
//!
//! The numerics live in [`edge_rs`], gated against edgeR 4.8.2 and limma
//! 3.66.0. Both routes assemble the same front half, filter and normalise, and
//! diverge from there. Anything with a genes-by-samples count matrix can call
//! either, so pseudobulk and Milo neighbourhood counts both come through here.
//!
//! ### Two edgeR pipelines
//!
//! `legacy` picks edgeR's pre-4.0 route, which shrinks the raw residual
//! deviance and needs a negative binomial dispersion handed to it. That is the
//! one `estimateDisp` exists for, and the only one where the Poisson bound
//! bites. The current route estimates its own dispersion from the most abundant
//! genes and skips `estimateDisp`, which is where most of the runtime went.
//!
//! That mirrors `glmQLFit(y, design, legacy = FALSE)` on a `DGEList` that never
//! saw `estimateDisp`. Run `estimateDisp` first in R and edgeR feeds the mean
//! of the top decile's trended dispersions in instead, which moves the answer
//! slightly. Skipping it is edgeR 4's own recommendation for this pipeline.
//!
//! ### The linear model route
//!
//! [run_limma_dge] is the other bulk route. `LimmaRoute::Voom` is edgeR's
//! `voomLmFit`: a mean-variance trend over the log-CPM, precision weights from
//! it, and a weighted least squares fit. `LimmaRoute::Trend` skips the weights
//! and lets the empirical Bayes prior absorb the mean-variance relationship
//! instead, which is what limma recommends when the library sizes are not too
//! variable. Both then moderate and tabulate.

use edge_rs::core::dgelist::DgeList;
use edge_rs::core::expression::{ave_log_cpm, cpm};
use edge_rs::core::filtering::filter_by_expr;
use edge_rs::core::normalisation::{NormMethod, calc_norm_factors};
use edge_rs::dispersion::estimate::estimate_disp;
use edge_rs::glm::ql_fit::{QlFitParams, glm_ql_fit};
use edge_rs::glm::test::{GlmTestInput, Tested, glm_ql_ftest};
use edge_rs::limma::contrasts::contrasts_fit;
use edge_rs::limma::ebayes::{EBayesParams, EBayesTrend, ebayes};
use edge_rs::limma::lm_fit::{LmFitResult, lm_fit};
use edge_rs::limma::marray::MArrayLm;
use edge_rs::limma::toptable::{TopTableParams, TopTableSort, top_table};
use edge_rs::limma::voom::{VoomParams, voom_lmfit};
use edge_rs::numeric::stats::p_adjust_bh;
use edge_rs::prelude::Recycled;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Prior count `aveLogCPM` adds before the log. edgeR's default.
const AVE_LOG_CPM_PRIOR: f64 = 2.0;

/// Prior count voom adds before the log. Hardcoded at this upstream.
const VOOM_PRIOR: f64 = 0.5;

/// Prior count the limma-trend route adds before the log. edgeR's `cpm`
/// default, and what limma's own guide uses for the trend.
const TREND_PRIOR: f64 = 2.0;

/////////////
// Helpers //
/////////////

/// Builds the gene mask both routes filter on.
///
/// `filterByExpr` first, then the mean-count floor on top of it. Split out so
/// the two pipelines cannot drift apart on which genes they see.
///
/// ### Params
///
/// * `counts` - Raw counts, gene-major and row-major, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Predictors, row-major `n_samples * n_coef`
/// * `n_coef` - Number of design columns
/// * `filter` - Run `filterByExpr`
/// * `min_mean` - Mean-count floor, `0.0` to turn it off
///
/// ### Returns
///
/// One flag per gene, or [`edge_rs::errors::EdgeErrors::NoGenesAfterFiltering`]
/// if nothing survived.
fn keep_mask(
    counts: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    filter: bool,
    min_mean: f64,
) -> Result<Vec<bool>, BixverseErrors> {
    let mut keep = if filter {
        filter_by_expr(
            counts,
            n_genes,
            n_samples,
            None,
            None,
            Some((design, n_coef)),
            None,
        )?
    } else {
        vec![true; n_genes]
    };
    if min_mean > 0.0 {
        for (gene, flag) in keep.iter_mut().enumerate() {
            let mean = counts[gene * n_samples..(gene + 1) * n_samples]
                .iter()
                .sum::<f64>()
                / n_samples as f64;
            *flag &= mean >= min_mean;
        }
    }
    if !keep.iter().any(|k| *k) {
        return Err(edge_rs::errors::EdgeErrors::NoGenesAfterFiltering { n_genes }.into());
    }
    Ok(keep)
}

////////////
// Params //
////////////

/// Parameters for [run_edger_ql].
#[derive(Clone, Copy, Debug)]
pub struct EdgeRQlParams {
    /// Library size normalisation. [`NormMethod::None`] leaves every factor at
    /// one, which is what Milo's `logMS` amounts to.
    pub norm_method: NormMethod,
    /// Run `filterByExpr` before fitting.
    pub filter: bool,
    /// Drop genes whose mean count across samples is below this. Applied on top
    /// of `filter`, and `0.0` turns it off.
    pub min_mean: f64,
    /// Robust empirical Bayes squeezing, giving outlier genes their own smaller
    /// prior degrees of freedom.
    pub robust: bool,
    /// Take edgeR's pre-4.0 quasi-likelihood pipeline.
    pub legacy: bool,
}

impl Default for EdgeRQlParams {
    /// edgeR's defaults, filtering on.
    fn default() -> Self {
        Self {
            norm_method: NormMethod::Tmm,
            filter: true,
            min_mean: 0.0,
            robust: false,
            legacy: false,
        }
    }
}

/////////////////
// EdgeRDgeRes //
/////////////////

/// The edgeR quasi-likelihood F-test, one row per gene that survived the
/// filters.
///
/// `genes_to_keep` spans the whole gene universe and is the mask back onto it,
/// the same convention `DgeMannWhitneyRes` uses on the single-cell side.
#[derive(Clone, Debug)]
pub struct EdgeRDgeRes {
    /// Which genes made it past the filters.
    pub genes_to_keep: Vec<bool>,
    /// Log2 fold change of the tested coefficient or contrast.
    pub log_fc: Vec<f64>,
    /// Average log2 counts per million.
    pub log_cpm: Vec<f64>,
    /// Quasi-likelihood F statistic.
    pub f_stat: Vec<f64>,
    /// Raw p-value.
    pub p_val: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value.
    pub fdr: Vec<f64>,
}

//////////
// Main //
//////////

/// Runs the edgeR quasi-likelihood chain and tests one coefficient or contrast.
///
/// ### Params
///
/// * `counts` - Raw counts, gene-major and row-major, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Predictors, row-major `n_samples * n_coef`, including an
///   intercept. At least two columns, since the null model has to keep one
/// * `n_coef` - Number of design columns
/// * `tested` - Coefficients to drop from the null, or a contrast over them
/// * `params` - See [EdgeRQlParams]
///
/// ### Returns
///
/// The [EdgeRDgeRes], or an [`edge_rs`] error if the shapes disagree, the
/// design is rank deficient, or nothing survived the filters.
///
/// ### References
///
/// Chen, Lun and Smyth, F1000Research 5:1438, 2016
pub fn run_edger_ql(
    counts: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    tested: &Tested,
    params: &EdgeRQlParams,
) -> Result<EdgeRDgeRes, BixverseErrors> {
    let dge = DgeList::new(counts.to_vec(), n_genes, n_samples, None)?;

    let keep = keep_mask(
        &dge.counts,
        n_genes,
        n_samples,
        design,
        n_coef,
        params.filter,
        params.min_mean,
    )?;

    let mut dge = dge.subset_genes(&keep)?;
    let n_kept = dge.n_genes;

    // The library sizes are the full matrix's, not the filtered one's. edgeR
    // keeps them through `[.DGEList` and normalises against them, so passing
    // `None` here and letting the column sums be recomputed silently shifts
    // every TMM factor and everything downstream of it.
    let lib_size = dge.lib_size.clone();
    dge.norm_factors = calc_norm_factors(
        &dge.counts,
        n_kept,
        n_samples,
        Some(&lib_size),
        params.norm_method,
        None,
        None,
    )?;
    let offset = dge.offset()?;
    let abundance = ave_log_cpm(
        &dge.counts,
        n_kept,
        n_samples,
        None,
        Some(&offset),
        AVE_LOG_CPM_PRIOR,
        None,
    )?;

    // `glmQLFit.DGEList` needs a trended dispersion on the legacy path and
    // errors without one, which is what `estimateDisp` is here for. On the
    // current path it takes whatever the DGEList carries, and a DGEList that
    // never saw `estimateDisp` carries nothing, so the fit self-estimates from
    // the most abundant genes. Skipping the call there is edgeR 4's own advice
    // and drops the most expensive step in the chain.
    let dispersion = if params.legacy {
        let disp = estimate_disp(
            &dge.counts,
            n_kept,
            n_samples,
            design,
            n_coef,
            &offset,
            None,
            Some(&abundance),
            None,
        )?;
        let per_gene = disp.trended.unwrap_or_else(|| vec![disp.common; n_kept]);
        Some(Recycled::by_gene(per_gene))
    } else {
        None
    };

    let fit = glm_ql_fit(
        &dge.counts,
        n_kept,
        n_samples,
        design,
        n_coef,
        dispersion.as_ref(),
        &offset,
        None,
        &abundance,
        Some(QlFitParams {
            robust: params.robust,
            legacy: params.legacy,
            ..Default::default()
        }),
    )?;

    let base = fit.as_glm_fit();
    let ql = fit.ql_summary();
    let input = GlmTestInput {
        counts: &dge.counts,
        n_genes: n_kept,
        n_samples,
        design,
        n_coef,
        dispersion: &fit.dispersion,
        offset: &offset,
        weights: None,
        log_cpm: Some(&abundance),
    };

    // edgeR's `poisson.bound` default. It only bites on the legacy pipeline;
    // `ql_summary` leaves `df_residual_zeros` empty on the current one, which
    // is what switches it off there.
    let test = glm_ql_ftest(&input, &base, &ql, tested, true)?;
    let fdr = p_adjust_bh(&test.p_value);

    Ok(EdgeRDgeRes {
        genes_to_keep: keep,
        log_fc: test.log_fc,
        log_cpm: test.log_cpm.unwrap_or(abundance),
        f_stat: test.statistic,
        p_val: test.p_value,
        fdr,
    })
}

/////////////////
// LimmaParams //
/////////////////

/// Which linear model route the counts go through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LimmaRoute {
    /// edgeR's `voomLmFit`: mean-variance trend, precision weights, and a
    /// weighted least squares fit on top of them.
    #[default]
    Voom,
    /// limma-trend: log-CPM straight into `lmFit`, with the mean-variance
    /// relationship absorbed by the empirical Bayes prior instead.
    Trend,
}

/// Parses the limma route from its R spelling.
///
/// ### Params
///
/// * `s` - `"voom"` or `"trend"`
///
/// ### Returns
///
/// The [LimmaRoute], or `None` for anything else.
pub fn parse_limma_route(s: &str) -> Option<LimmaRoute> {
    match s {
        "voom" => Some(LimmaRoute::Voom),
        "trend" => Some(LimmaRoute::Trend),
        _ => None,
    }
}

/// Parameters for [run_limma_dge].
#[derive(Clone, Copy, Debug)]
pub struct LimmaParams {
    /// Which route the linear model is fitted on.
    pub route: LimmaRoute,
    /// Library size normalisation, applied before either route.
    pub norm_method: NormMethod,
    /// Run `filterByExpr` before fitting.
    pub filter: bool,
    /// Drop genes whose mean count across samples is below this. Applied on top
    /// of `filter`, and `0.0` turns it off.
    pub min_mean: f64,
    /// Winsorise the empirical Bayes moments, so a handful of outlier genes
    /// cannot drag the prior degrees of freedom down. limma's
    /// `eBayes(robust = TRUE)`.
    pub robust: bool,
    /// Count added before the log. `None` takes the route's own default,
    /// [VOOM_PRIOR] or [TREND_PRIOR].
    pub prior_count: Option<f64>,
    /// Derive the lowess span from the gene count rather than reading `span`.
    /// Only the voom route fits a trend, so the other one ignores it.
    pub adaptive_span: bool,
    /// Lowess span for the voom trend, read only when `adaptive_span` is off.
    pub span: f64,
    /// Assumed proportion of differentially expressed genes. Only the
    /// B-statistic reads it.
    pub proportion: f64,
}

impl Default for LimmaParams {
    /// limma's own defaults, filtering and TMM on.
    fn default() -> Self {
        Self {
            route: LimmaRoute::Voom,
            norm_method: NormMethod::Tmm,
            filter: true,
            min_mean: 0.0,
            robust: false,
            prior_count: None,
            adaptive_span: true,
            span: 0.5,
            proportion: 0.01,
        }
    }
}

impl LimmaParams {
    /// Prior count for the route, falling back to its upstream default.
    ///
    /// voom and `cpm` disagree here, `0.5` against `2.0`, so the fallback has
    /// to know which route it is resolving for.
    ///
    /// ### Returns
    ///
    /// The prior count to add before the log.
    fn resolve_prior_count(&self) -> f64 {
        self.prior_count.unwrap_or(match self.route {
            LimmaRoute::Voom => VOOM_PRIOR,
            LimmaRoute::Trend => TREND_PRIOR,
        })
    }
}

/////////////////
// LimmaDgeRes //
/////////////////

/// The limma moderated t-test, one row per gene that survived the filters.
///
/// `genes_to_keep` spans the whole gene universe and is the mask back onto it,
/// the same convention [EdgeRDgeRes] uses.
#[derive(Clone, Debug)]
pub struct LimmaDgeRes {
    /// Which genes made it past the filters.
    pub genes_to_keep: Vec<bool>,
    /// Log2 fold change of the tested coefficient or contrast.
    pub log_fc: Vec<f64>,
    /// Average log2 expression. limma's `AveExpr`, an average log2 count per
    /// million on both routes.
    pub ave_expr: Vec<f64>,
    /// Moderated t statistic.
    pub t_stat: Vec<f64>,
    /// Raw p-value.
    pub p_val: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value.
    pub fdr: Vec<f64>,
    /// Log-odds of differential expression. limma's `B`.
    pub b_stat: Vec<f64>,
}

////////////////
// limma-voom //
////////////////

/// Runs the limma linear model chain and tests one coefficient or contrast.
///
/// Filter and normalise as [run_edger_ql] does, then either `voomLmFit` or
/// limma-trend, then `contrasts.fit`, `eBayes` and `topTable`. The table comes
/// back unsorted and untruncated, one row per kept gene in input order.
///
/// The empirical Bayes trend follows the route rather than being a knob: the
/// voom weights already carry the mean-variance relationship, so a trended
/// prior on top of them double counts it, while limma-trend has nothing else
/// to absorb it.
///
/// ### Params
///
/// * `counts` - Raw counts, gene-major and row-major, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Predictors, row-major `n_samples * n_coef`, including an
///   intercept and full rank
/// * `n_coef` - Number of design columns
/// * `tested` - One coefficient to tabulate, or a contrast over them. Several
///   coefficients at once is the moderated F question and is rejected
/// * `params` - See [LimmaParams]
///
/// ### Returns
///
/// The [LimmaDgeRes], or an [`edge_rs`] error if the shapes disagree, the
/// design is rank deficient, or nothing survived the filters.
///
/// ### References
///
/// Law, Chen, Shi and Smyth, Genome Biology 15:R29, 2014
///
/// Smyth, Statistical Applications in Genetics and Molecular Biology 3(1), 2004
pub fn run_limma_dge(
    counts: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    tested: &Tested,
    params: &LimmaParams,
) -> Result<LimmaDgeRes, BixverseErrors> {
    if let Tested::Coef(coef) = tested
        && coef.len() != 1
    {
        return Err(BixverseErrors::LimmaMultiCoef { n_coef: coef.len() });
    }

    let dge = DgeList::new(counts.to_vec(), n_genes, n_samples, None)?;

    let keep = keep_mask(
        &dge.counts,
        n_genes,
        n_samples,
        design,
        n_coef,
        params.filter,
        params.min_mean,
    )?;

    let mut dge = dge.subset_genes(&keep)?;
    let n_kept = dge.n_genes;

    // Same trap as the edgeR route: the library sizes are the full matrix's,
    // and normalising against the filtered column sums instead shifts every
    // factor. See the comment in `run_edger_ql`.
    let lib_size = dge.lib_size.clone();
    dge.norm_factors = calc_norm_factors(
        &dge.counts,
        n_kept,
        n_samples,
        Some(&lib_size),
        params.norm_method,
        None,
        None,
    )?;

    let prior_count = params.resolve_prior_count();
    let (fit, amean): (LmFitResult, Vec<f64>) = match params.route {
        LimmaRoute::Voom => {
            // `VoomParams::normalize_method` would recompute the factors
            // against the filtered counts, so the effective sizes go in
            // pre-multiplied instead. This is `voomLmFit` on a DGEList that has
            // already been through `calcNormFactors`.
            let eff_lib: Vec<f64> = lib_size
                .iter()
                .zip(dge.norm_factors.iter())
                .map(|(l, f)| l * f)
                .collect();
            let (voom, lm) = voom_lmfit(
                &dge.counts,
                n_kept,
                n_samples,
                design,
                n_coef,
                Some(&eff_lib),
                None,
                Some(VoomParams {
                    normalize_method: NormMethod::None,
                    span: params.span,
                    adaptive_span: params.adaptive_span,
                    prior_count,
                    save_trend: false,
                }),
            )?;
            (lm, voom.amean)
        }
        LimmaRoute::Trend => {
            let offset = dge.offset()?;
            let e = cpm(
                &dge.counts,
                n_kept,
                n_samples,
                None,
                Some(&offset),
                true,
                prior_count,
            )?;
            let amean: Vec<f64> = e
                .chunks_exact(n_samples)
                .map(|row| row.iter().sum::<f64>() / n_samples as f64)
                .collect();
            let lm = lm_fit(&e, n_kept, n_samples, design, n_coef, None, None, None)?;
            (lm, amean)
        }
    };

    let fit = MArrayLm::from_lm_fit(fit, design, n_coef, n_samples, Some(amean))?;

    // `Tested::Contrast` already carries the column-major layout `contrasts_fit`
    // wants, so it passes straight through. Rotating has to happen before
    // `ebayes`, which drops any moderated statistic it finds on the fit.
    let (fit, coef) = match tested {
        Tested::Coef(coef) => (fit, coef[0]),
        Tested::Contrast {
            values,
            n_contrasts,
        } => (contrasts_fit(fit, values, *n_contrasts)?, 0),
    };

    let trend = match params.route {
        LimmaRoute::Voom => EBayesTrend::None,
        LimmaRoute::Trend => EBayesTrend::Amean,
    };
    let fit = ebayes(
        fit,
        Some(EBayesParams {
            proportion: params.proportion,
            trend,
            robust: params.robust,
            ..Default::default()
        }),
    )?;

    // Unsorted and untruncated, so the rows come back in input order and
    // `adj_p_value` is already Benjamini-Hochberg over every kept gene.
    let top = top_table(
        &fit,
        coef,
        Some(TopTableParams {
            number: usize::MAX,
            sort_by: TopTableSort::None,
            ..Default::default()
        }),
    )?;

    // `from_lm_fit` was handed an `amean`, so this cannot be `None`.
    let ave_expr = top.ave_expr.ok_or(BixverseErrors::DgeShapeMismatch {
        name: "ave_expr",
        expected: n_kept,
        got: 0,
    })?;

    Ok(LimmaDgeRes {
        genes_to_keep: keep,
        log_fc: top.log_fc,
        ave_expr,
        t_stat: top.t,
        p_val: top.p_value,
        fdr: top.adj_p_value,
        b_stat: top.b,
    })
}
