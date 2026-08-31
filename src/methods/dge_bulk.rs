//! edgeR's quasi-likelihood differential expression, see Chen, Lun and Smyth,
//! F1000Research, 2016.
//!
//! The numerics live in [`edge_rs`], gated against edgeR 4.8.2. This is the
//! chain assembled: filter, normalise, abundance, fit, test. Anything with a
//! genes-by-samples count matrix can call it, so pseudobulk and Milo
//! neighbourhood counts both come through here.
//!
//! ### Two pipelines
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

use edge_rs::core::dgelist::DgeList;
use edge_rs::core::expression::ave_log_cpm;
use edge_rs::core::filtering::filter_by_expr;
use edge_rs::core::normalisation::{NormMethod, calc_norm_factors};
use edge_rs::dispersion::estimate::estimate_disp;
use edge_rs::glm::ql_fit::{QlFitParams, glm_ql_fit};
use edge_rs::glm::test::{GlmTestInput, Tested, glm_ql_ftest};
use edge_rs::numeric::stats::p_adjust_bh;
use edge_rs::prelude::Recycled;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Prior count `aveLogCPM` adds before the log. edgeR's default.
const AVE_LOG_CPM_PRIOR: f64 = 2.0;

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

/////////////
// Results //
/////////////

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

///////////////
// Front end //
///////////////

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

    let mut keep = if params.filter {
        filter_by_expr(
            &dge.counts,
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
    if params.min_mean > 0.0 {
        for (gene, flag) in keep.iter_mut().enumerate() {
            let mean = counts[gene * n_samples..(gene + 1) * n_samples]
                .iter()
                .sum::<f64>()
                / n_samples as f64;
            *flag &= mean >= params.min_mean;
        }
    }
    if !keep.iter().any(|k| *k) {
        return Err(edge_rs::errors::EdgeErrors::NoGenesAfterFiltering { n_genes }.into());
    }

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
