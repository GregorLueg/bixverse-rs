//! scTransform v2: regularised negative binomial normalisation.
//!
//! The model is fitted on a small subsample, regularised across genes against
//! their abundance, and only then applied to everything. That split is what
//! makes it cheap: step 1 is a fixed 2000 genes by 2000 cells whatever the
//! dataset size, and every later stage is a per-gene reduction over the
//! gene-major store.
//!
//! This module holds the parameters, the per-gene statistics the regularisation
//! needs, and the regularisation itself. The per-gene fit lives in
//! [`super::nb_fit`].
//!
//! ### References
//!
//! Hafemeister & Satija, Genome Biology, 2019; Choudhary & Satija, Genome
//! Biology, 2022

use crate::core::base::kernel_smooth::{bw_sj, ksmooth_normal};
use crate::core::math::stats::is_outlier;
use crate::core::math::vector_helpers::median;
use crate::errors::BixverseErrors;

use super::nb_fit::{LOG_UMI_COEF, NbOffsetFit};

////////////
// Consts //
////////////

/// Mean count below which a gene is treated as Poisson regardless of what its
/// variance says.
///
/// Under this the variance estimate is too noisy to distinguish the two, so
/// sctransform short circuits to the closed-form offset model.
const POISSON_MEAN_FLOOR: f64 = 1e-3;

/// Divisor on the median non-zero UMI count that sets the variance floor.
///
/// `min_variance = (median_nonzero / 5)^2`, sctransform's `min_variance =
/// "umi_median"`, which v2 turns on. It stops a gene with a near-zero fitted
/// mean from producing enormous residuals off a single count.
const UMI_MEDIAN_DIVISOR: f64 = 5.0;

////////////
// Params //
////////////

/// Tuning knobs for a scTransform v2 run.
///
/// The defaults are sctransform's own with `vst.flavor = "v2"` applied, which
/// fixes `n_cells` at 2000 and turns on Poisson exclusion and the UMI-median
/// variance floor.
#[derive(Clone, Copy, Debug)]
pub struct SctParams {
    /// Genes sampled for the step-1 fit, by inverse density over
    /// `log10(geometric mean)`.
    pub n_genes: usize,
    /// Cells sampled for the step-1 fit.
    pub n_cells: usize,
    /// Minimum number of cells a gene must be detected in to be modelled.
    pub min_cells: usize,
    /// Multiplier on the Sheather-Jones bandwidth used for the smoothing.
    pub bw_adjust: f64,
    /// Pseudocount in the geometric mean, `row_gmean(eps = )`.
    pub gmean_eps: f64,
    /// Absolute robust z-score above which a step-1 parameter is an outlier.
    pub outlier_th: f64,
    /// Ratio of method-of-moments theta to fitted theta below which a gene is
    /// declared Poisson and its theta set to infinity.
    pub poisson_diff_theta: f64,
    /// Residual clipping range. `None` resolves to `+/- sqrt(n_cells)`,
    /// sctransform's default. Seurat's `SCTransform()` wrapper uses
    /// `+/- sqrt(n_cells / 30)` instead, so this is worth setting explicitly
    /// when matching Seurat rather than sctransform.
    pub clip_range: Option<(f64, f64)>,
}

impl Default for SctParams {
    fn default() -> Self {
        Self {
            n_genes: 2000,
            n_cells: 2000,
            min_cells: 5,
            bw_adjust: 3.0,
            gmean_eps: 1.0,
            outlier_th: 10.0,
            poisson_diff_theta: 1e-3,
            clip_range: None,
        }
    }
}

impl SctParams {
    /// Resolves the residual clipping range against the number of cells.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells the residuals are computed over.
    ///
    /// ### Returns
    ///
    /// The clipping range, either as supplied or `+/- sqrt(n_cells)`.
    pub fn resolve_clip_range(&self, n_cells: usize) -> (f64, f64) {
        self.clip_range.unwrap_or_else(|| {
            let c = (n_cells as f64).sqrt();
            (-c, c)
        })
    }
}

////////////////
// Covariates //
////////////////

/// Cell-level covariates the model regresses out alongside the library size.
///
/// sctransform's `latent_var` beyond `log_umi`. The library size is never one
/// of these: it enters as a fixed offset with its slope pinned at `ln(10)`,
/// which is what distinguishes v2 from v1, so putting it here would fit it
/// twice.
///
/// Empty is the common case and costs nothing: the design collapses to a single
/// intercept column and the residual arithmetic skips the dot product.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SctCovariates {
    /// Row-major `n_cells * n_covariates`, excluding the intercept column.
    pub values: Vec<f64>,
    /// One name per covariate, for reporting the fitted coefficients back.
    pub names: Vec<String>,
}

impl SctCovariates {
    /// Builds a covariate set from one column per covariate.
    ///
    /// ### Params
    ///
    /// * `columns` - One `(name, values)` pair per covariate, each of length
    ///   `n_cells`.
    ///
    /// ### Returns
    ///
    /// The covariate set, or [`BixverseErrors::LengthMismatch`] when the
    /// columns disagree in length.
    pub fn from_columns(columns: &[(String, Vec<f64>)]) -> Result<Self, BixverseErrors> {
        if columns.is_empty() {
            return Ok(Self::default());
        }
        let n_cells = columns[0].1.len();
        if let Some((name, col)) = columns.iter().find(|(_, c)| c.len() != n_cells) {
            return Err(BixverseErrors::SctCovariateLengthMismatch {
                name: name.clone(),
                expected: n_cells,
                found: col.len(),
            });
        }

        // Row-major, so a cell's covariates are contiguous and the per-cell dot
        // product in the residual is a single cache line on any realistic
        // covariate count.
        let mut values = Vec::with_capacity(n_cells * columns.len());
        for cell in 0..n_cells {
            for (_, col) in columns {
                values.push(col[cell]);
            }
        }

        Ok(Self {
            values,
            names: columns.iter().map(|(n, _)| n.clone()).collect(),
        })
    }

    /// Number of covariates, excluding the intercept.
    ///
    /// ### Returns
    ///
    /// The covariate count.
    pub fn n_covariates(&self) -> usize {
        self.names.len()
    }

    /// Whether there are no covariates at all.
    ///
    /// ### Returns
    ///
    /// `true` when only the intercept is fitted.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// One cell's covariate values.
    ///
    /// ### Params
    ///
    /// * `cell` - Position within the cell set.
    ///
    /// ### Returns
    ///
    /// The slice, empty when there are no covariates.
    pub fn row(&self, cell: usize) -> &[f64] {
        let k = self.n_covariates();
        if k == 0 {
            &[]
        } else {
            &self.values[cell * k..(cell + 1) * k]
        }
    }

    /// Checks the covariates cover the expected number of cells.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Cells the model runs over.
    ///
    /// ### Returns
    ///
    /// `()`, or [`BixverseErrors::LengthMismatch`].
    pub fn validate(&self, n_cells: usize) -> Result<(), BixverseErrors> {
        let expected = n_cells * self.n_covariates();
        if self.values.len() != expected {
            return Err(BixverseErrors::LengthMismatch {
                name: "covariate values",
                expected,
                found: self.values.len(),
            });
        }
        Ok(())
    }

    /// Builds the design matrix the fit uses: an intercept column followed by
    /// the covariates, row-major `n_cells * (1 + n_covariates)`.
    ///
    /// The library size is deliberately absent. It is the offset.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of rows.
    ///
    /// ### Returns
    ///
    /// The design matrix.
    pub fn design(&self, n_cells: usize) -> Vec<f64> {
        let k = self.n_covariates();
        let mut design = Vec::with_capacity(n_cells * (k + 1));
        for cell in 0..n_cells {
            design.push(1.0);
            design.extend_from_slice(self.row(cell));
        }
        design
    }

    /// The median of each covariate.
    ///
    /// The correction places every cell at the median of every latent
    /// variable, so this is what the corrected counts are evaluated at.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells.
    ///
    /// ### Returns
    ///
    /// One median per covariate.
    pub fn medians(&self, n_cells: usize) -> Vec<f64> {
        (0..self.n_covariates())
            .map(|k| {
                let column: Vec<f64> = (0..n_cells).map(|c| self.row(c)[k]).collect();
                median(&column).unwrap_or(0.0)
            })
            .collect()
    }

    /// Gathers the rows of a subset of cells.
    ///
    /// Used to hand one group's cells to a per-group fit. Empty covariates stay
    /// empty rather than becoming a zero-column matrix of the wrong height.
    ///
    /// ### Params
    ///
    /// * `cells` - Positions of the cells to keep, within the current set.
    ///
    /// ### Returns
    ///
    /// The subset, or [`BixverseErrors::SctCovariateLengthMismatch`] when a
    /// position is out of range.
    pub fn subset(&self, cells: &[usize]) -> Result<Self, BixverseErrors> {
        if self.is_empty() {
            return Ok(Self::default());
        }

        let k = self.n_covariates();
        let n_cells = self.values.len() / k;
        let mut values = Vec::with_capacity(cells.len() * k);
        for &c in cells {
            if c >= n_cells {
                return Err(BixverseErrors::SctCovariateLengthMismatch {
                    name: "subset".to_string(),
                    expected: n_cells,
                    found: c,
                });
            }
            values.extend_from_slice(self.row(c));
        }

        Ok(Self {
            values,
            names: self.names.clone(),
        })
    }
}

/// The per-cell terms every residual needs.
///
/// Bundled because the library size and the covariates always travel together
/// and every consumer of a residual needs both.
#[derive(Clone, Copy, Debug)]
pub struct SctCellContext<'a> {
    /// `log10(total UMI)` per cell.
    pub log10_umi: &'a [f64],
    /// Cell-level covariates, in the same cell order.
    pub covariates: &'a SctCovariates,
}

impl<'a> SctCellContext<'a> {
    /// Builds a context and checks the two agree on the cell count.
    ///
    /// ### Params
    ///
    /// * `log10_umi` - `log10(total UMI)` per cell.
    /// * `covariates` - Cell-level covariates.
    ///
    /// ### Returns
    ///
    /// The context, or [`BixverseErrors::LengthMismatch`].
    pub fn new(
        log10_umi: &'a [f64],
        covariates: &'a SctCovariates,
    ) -> Result<Self, BixverseErrors> {
        covariates.validate(log10_umi.len())?;
        Ok(Self {
            log10_umi,
            covariates,
        })
    }

    /// Number of cells described.
    ///
    /// ### Returns
    ///
    /// The cell count.
    pub fn n_cells(&self) -> usize {
        self.log10_umi.len()
    }
}

/////////////////
// Gene stats //
/////////////////

/// Per-gene summaries over the full cell set, one pass over the gene-major
/// store.
///
/// All three vectors are parallel and indexed by position in the gene set being
/// modelled, not by the store's own gene index.
#[derive(Clone, Debug)]
pub struct SctGeneStats {
    /// `log10` of the geometric mean of the counts, with the `gmean_eps`
    /// pseudocount.
    pub log_gmean: Vec<f64>,
    /// Arithmetic mean of the counts.
    pub amean: Vec<f64>,
    /// Variance of the counts.
    pub var: Vec<f64>,
}

impl SctGeneStats {
    /// Number of genes described.
    ///
    /// ### Returns
    ///
    /// The gene count.
    pub fn len(&self) -> usize {
        self.log_gmean.len()
    }

    /// Whether any genes are described.
    ///
    /// ### Returns
    ///
    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.log_gmean.is_empty()
    }

    /// Checks that all three vectors agree in length.
    ///
    /// ### Returns
    ///
    /// `()`, or [`BixverseErrors::LengthMismatch`] naming the offending field.
    fn validate(&self) -> Result<(), BixverseErrors> {
        let n = self.log_gmean.len();
        if self.amean.len() != n {
            return Err(BixverseErrors::LengthMismatch {
                name: "amean",
                expected: n,
                found: self.amean.len(),
            });
        }
        if self.var.len() != n {
            return Err(BixverseErrors::LengthMismatch {
                name: "var",
                expected: n,
                found: self.var.len(),
            });
        }
        Ok(())
    }
}

///////////
// Model //
///////////

/// The regularised per-gene model, one entry per modelled gene.
///
/// `mu_gc = exp(intercept_g) * total_umi_c` and
/// `var_gc = max(mu_gc + mu_gc^2 / theta_g, min_variance)`, so these three
/// numbers plus the per-cell library sizes regenerate any gene's residual row
/// without touching the counts of any other gene. That is what lets the
/// residual and corrected-count stages stream.
#[derive(Clone, Debug)]
pub struct SctModel {
    /// Store gene indices this model covers, ascending. Lets a caller holding
    /// store indices, such as an HVG set, look a gene up without carrying a
    /// separate mapping around.
    pub genes: Vec<usize>,
    /// Regularised inverse overdispersion. `f64::INFINITY` for a Poisson gene.
    pub theta: Vec<f64>,
    /// Regularised coefficients, row-major `n_genes * n_coef`. Column zero is
    /// the intercept, the rest are the covariates in `covariate_names` order.
    pub coefficients: Vec<f64>,
    /// Design columns per gene, at least one.
    pub n_coef: usize,
    /// Names of the fitted covariates, length `n_coef - 1`.
    pub covariate_names: Vec<String>,
    /// Coefficient on `log10(total UMI)`, always `ln(10)`. See
    /// [`LOG_UMI_COEF`].
    pub log_umi_coef: f64,
    /// Variance floor, `(median non-zero UMI / 5)^2`.
    pub min_variance: f64,
    /// Residual clipping range.
    pub clip_range: (f64, f64),
}

impl SctModel {
    /// Number of genes modelled.
    ///
    /// ### Returns
    ///
    /// The gene count.
    pub fn len(&self) -> usize {
        self.theta.len()
    }

    /// One gene's coefficients, intercept first.
    ///
    /// ### Params
    ///
    /// * `pos` - Position within the model.
    ///
    /// ### Returns
    ///
    /// The `n_coef` coefficients.
    pub fn coefficients_for(&self, pos: usize) -> &[f64] {
        &self.coefficients[pos * self.n_coef..(pos + 1) * self.n_coef]
    }

    /// One gene's intercept.
    ///
    /// ### Params
    ///
    /// * `pos` - Position within the model.
    ///
    /// ### Returns
    ///
    /// The intercept on the natural-log scale.
    pub fn intercept(&self, pos: usize) -> f64 {
        self.coefficients[pos * self.n_coef]
    }

    /// Whether the model regresses out anything beyond the library size.
    ///
    /// ### Returns
    ///
    /// `true` when covariates were fitted.
    pub fn has_covariates(&self) -> bool {
        self.n_coef > 1
    }

    /// Whether any genes are modelled.
    ///
    /// ### Returns
    ///
    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.theta.is_empty()
    }

    /// Position of a store gene index within this model.
    ///
    /// `genes` is ascending, so this is a binary search rather than a hash
    /// lookup: the caller is usually walking an HVG set in order and the
    /// probes land close together.
    ///
    /// ### Params
    ///
    /// * `gene` - Store gene index.
    ///
    /// ### Returns
    ///
    /// The position, or `None` when the gene is not modelled.
    pub fn position(&self, gene: usize) -> Option<usize> {
        self.genes.binary_search(&gene).ok()
    }

    /// Whether a gene carries the Poisson variance rather than a negative
    /// binomial one.
    ///
    /// Derived from `theta` rather than stored: an infinite theta *is* the
    /// Poisson case, since `mu^2 / theta` vanishes. Keeping a parallel flag
    /// would be state that can disagree with the parameter it describes.
    ///
    /// ### Params
    ///
    /// * `pos` - Position within the model.
    ///
    /// ### Returns
    ///
    /// `true` when the gene's variance is Poisson.
    pub fn is_poisson(&self, pos: usize) -> bool {
        self.theta[pos].is_infinite()
    }
}

////////////////////
// Regularisation //
////////////////////

/// Regularises the step-1 fits across genes and extends them to every gene.
///
/// The step-1 parameters are noisy per gene but vary smoothly with abundance,
/// so each is kernel-smoothed against `log10(geometric mean)` and read off at
/// every gene's own abundance. Three things are excluded from the smoothing
/// first: parameter outliers, genes whose fit hit the Poisson boundary, and
/// genes whose variance does not exceed their mean. Those last get the exact
/// offset-model parameters instead of an extrapolated curve value.
///
/// Theta is not smoothed directly. It is transformed to the overdispersion
/// factor `log10(1 + gmean / theta)`, which is the multiple by which the NB
/// variance exceeds the Poisson one, smoothed there, and transformed back. That
/// keeps the smoothing on a scale where the parameter is well behaved across
/// four orders of magnitude of abundance.
///
/// The fits are expected to have been through v2's post-fit Poisson check
/// already, which overrides theta where the second moment disagrees with the
/// likelihood. [`super::stream::fit_sctransform`] does that; a caller
/// driving this directly has to, or genes that carry no real overdispersion
/// will pull the smoothing curve.
///
/// ### Params
///
/// * `step1` - Per-gene fits from the step-1 subsample.
/// * `step1_idx` - Position of each step-1 gene in the full gene set, parallel
///   to `step1`.
/// * `stats` - Per-gene statistics over the full cell set, for every gene.
/// * `covariate_names` - Names of the fitted covariates, so the model can
///   report which coefficient is which. Its length plus one must match the
///   coefficient count in `step1`.
/// * `mean_cell_sum` - Mean library size, for the Poisson genes' closed-form
///   intercept.
/// * `min_variance` - Variance floor. See [`min_variance_from_umi_median`].
/// * `n_cells` - Cell count, for the default clipping range.
/// * `params` - Tuning knobs.
///
/// ### Returns
///
/// The [`SctModel`] over every gene in `stats`, or a [`BixverseErrors`] when
/// the inputs disagree in length, too few genes survive the exclusions to
/// smooth, or the smoothing leaves a gene without a value.
#[allow(clippy::too_many_arguments)]
pub fn regularise_sct_model(
    step1: &[NbOffsetFit],
    step1_idx: &[usize],
    stats: &SctGeneStats,
    covariate_names: &[String],
    mean_cell_sum: f64,
    min_variance: f64,
    n_cells: usize,
    params: &SctParams,
) -> Result<SctModel, BixverseErrors> {
    stats.validate()?;
    if step1.len() != step1_idx.len() {
        return Err(BixverseErrors::LengthMismatch {
            name: "step1_idx",
            expected: step1.len(),
            found: step1_idx.len(),
        });
    }
    let n_genes = stats.len();
    if let Some(&bad) = step1_idx.iter().find(|&&i| i >= n_genes) {
        return Err(BixverseErrors::SctGeneIndexOutOfRange {
            index: bad,
            n_genes,
        });
    }

    let n_coef = covariate_names.len() + 1;
    if let Some(f) = step1.iter().find(|f| f.coefficients.len() != n_coef) {
        return Err(BixverseErrors::LengthMismatch {
            name: "step-1 coefficients",
            expected: n_coef,
            found: f.coefficients.len(),
        });
    }

    // Genes that get the closed-form offset model rather than a smoothed fit:
    // no excess variance over Poisson, or too low a mean to tell.
    let poisson: Vec<bool> = (0..n_genes)
        .map(|g| stats.var[g] - stats.amean[g] <= 0.0 || stats.amean[g] < POISSON_MEAN_FLOOR)
        .collect();

    // Overdispersion factor: the multiple by which the NB variance exceeds the
    // Poisson one. This, not theta, is what gets smoothed.
    let disp_par: Vec<f64> = step1
        .iter()
        .zip(step1_idx.iter())
        .map(|(f, &g)| (1.0 + 10.0_f64.powf(stats.log_gmean[g]) / f.theta).log10())
        .collect();
    let gmean_s1: Vec<f64> = step1_idx.iter().map(|&g| stats.log_gmean[g]).collect();

    // One column per parameter that gets smoothed: the overdispersion factor
    // followed by every design coefficient.
    let columns: Vec<Vec<f64>> = std::iter::once(disp_par.clone())
        .chain((0..n_coef).map(|k| step1.iter().map(|f| f.coefficients[k]).collect()))
        .collect();

    // The `log_umi` column sctransform also carries is the constant ln(10).
    // Within any bin its median is itself and its MAD is zero, so its robust
    // z-score is exactly zero and it can never flag an outlier, nor can
    // smoothing a constant change it. Leaving it out is a saving, not a
    // behaviour change.
    let mut outlier = vec![false; step1.len()];
    for column in &columns {
        for (slot, flagged) in
            outlier
                .iter_mut()
                .zip(is_outlier(column, &gmean_s1, params.outlier_th)?)
        {
            *slot |= flagged;
        }
    }

    let keep: Vec<usize> = (0..step1.len())
        .filter(|&i| !outlier[i] && step1[i].theta.is_finite() && !poisson[step1_idx[i]])
        .collect();

    if keep.len() < 2 {
        return Err(BixverseErrors::SctTooFewGenesToRegularise {
            kept: keep.len(),
            total: step1.len(),
        });
    }

    let fit_x: Vec<f64> = keep.iter().map(|&i| gmean_s1[i]).collect();
    let bw = bw_sj(&fit_x)? * params.bw_adjust;

    // Predict at every gene's abundance, clamped into the range the smoothing
    // actually saw so the tails are held flat rather than extrapolated.
    let (lo, hi) = fit_x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
    let x_points: Vec<f64> = stats.log_gmean.iter().map(|&v| v.clamp(lo, hi)).collect();

    // `ksmooth_normal` returns its values in sorted `x_points` order, so the
    // permutation has to be undone, exactly as sctransform's
    // `model_pars_fit[o, i] <- ...` does.
    let mut order: Vec<usize> = (0..n_genes).collect();
    order.sort_by(|&a, &b| {
        x_points[a]
            .partial_cmp(&x_points[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let smooth = |column: &[f64]| -> Result<Vec<f64>, BixverseErrors> {
        let y: Vec<f64> = keep.iter().map(|&i| column[i]).collect();
        let (_, smoothed) = ksmooth_normal(&fit_x, &y, &x_points, bw)?;
        let mut out = vec![f64::NAN; n_genes];
        for (k, &g) in order.iter().enumerate() {
            out[g] = smoothed[k];
        }
        Ok(out)
    };

    let mut disp_fit = smooth(&columns[0])?;
    let mut coefficients = vec![0.0_f64; n_genes * n_coef];
    for k in 0..n_coef {
        let column = smooth(&columns[k + 1])?;
        for (g, &v) in column.iter().enumerate() {
            coefficients[g * n_coef + k] = v;
        }
    }

    // Poisson genes take the exact offset model: theta infinite, the intercept
    // straight off the gene mean, and every covariate coefficient zeroed. A
    // gene with no overdispersion has nothing for a covariate to explain, and
    // sctransform pads its offset parameters with zeros for the same reason.
    for g in 0..n_genes {
        if poisson[g] {
            disp_fit[g] = 0.0;
            coefficients[g * n_coef] = stats.amean[g].ln() - mean_cell_sum.ln();
            for k in 1..n_coef {
                coefficients[g * n_coef + k] = 0.0;
            }
        }
    }

    if let Some(g) = (0..n_genes).find(|&g| {
        disp_fit[g].is_nan()
            || coefficients[g * n_coef..(g + 1) * n_coef]
                .iter()
                .any(|v| v.is_nan())
    }) {
        return Err(BixverseErrors::SctSmoothingLeftAGeneUnfitted { gene: g });
    }

    let theta: Vec<f64> = (0..n_genes)
        .map(|g| {
            if poisson[g] {
                f64::INFINITY
            } else {
                10.0_f64.powf(stats.log_gmean[g]) / (10.0_f64.powf(disp_fit[g]) - 1.0)
            }
        })
        .collect();

    Ok(SctModel {
        // Positional by default. `fit_sctransform` overwrites this with the
        // store indices once it knows them.
        genes: (0..n_genes).collect(),
        theta,
        coefficients,
        n_coef,
        covariate_names: covariate_names.to_vec(),
        log_umi_coef: LOG_UMI_COEF,
        min_variance,
        clip_range: params.resolve_clip_range(n_cells),
    })
}

//////////////////
// Residual row //
//////////////////

/// Pearson residuals for one gene across the selected cells.
///
/// ```text
/// mu_c  = exp(intercept + log_umi_coef * log10_umi_c + x_c . beta)
/// var_c = max(mu_c + mu_c^2 / theta, min_variance)
/// r_c   = clamp((y_c - mu_c) / sqrt(var_c), clip_range)
/// ```
///
/// The row is dense even though the counts are sparse: a zero count still has a
/// non-zero residual `-mu / sqrt(var)`. That is why residuals are generated on
/// demand rather than stored, and why the store's sparse layout cannot hold
/// them.
///
/// The exponential is written as `exp(b0 + ln(10) * log10(umi))` rather than the
/// algebraically identical `exp(b0) * umi` so that it rounds the way
/// sctransform's `exp(tcrossprod(coefs, regressor_data))` rounds.
///
/// ### Params
///
/// * `counts` - The gene's non-zero counts.
/// * `indices` - Cell positions of those counts, within `0..n_cells`.
/// * `gene_pos` - Position of this gene within the model.
/// * `model` - The fitted model.
/// * `cells` - Per-cell library sizes and covariates.
/// * `out` - Destination row, overwritten in full.
///
/// ### Returns
///
/// `()`, or a [`BixverseErrors`] when `gene_pos` is outside the model or the
/// lengths disagree.
pub fn sct_residual_row(
    counts: &[f64],
    indices: &[u32],
    gene_pos: usize,
    model: &SctModel,
    cells: &SctCellContext<'_>,
    out: &mut [f32],
) -> Result<(), BixverseErrors> {
    if gene_pos >= model.len() {
        return Err(BixverseErrors::SctGeneIndexOutOfRange {
            index: gene_pos,
            n_genes: model.len(),
        });
    }
    let n_cells = cells.n_cells();
    if out.len() != n_cells {
        return Err(BixverseErrors::LengthMismatch {
            name: "out",
            expected: n_cells,
            found: out.len(),
        });
    }
    if cells.covariates.n_covariates() + 1 != model.n_coef {
        return Err(BixverseErrors::SctCovariateCountMismatch {
            model: model.n_coef - 1,
            supplied: cells.covariates.n_covariates(),
        });
    }

    let gene = SctGeneParams::new(model, gene_pos);
    fill_residual_row(counts, indices, cells, out, |_| &gene);

    Ok(())
}

/// One gene's residual parameters, pulled out of the model once.
///
/// Hoisting the slice and scalar lookups out of the per-cell loop is what keeps
/// the grouped path as cheap as the single-model one: a group's parameters are
/// resolved once per gene, not once per cell.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SctGeneParams<'a> {
    /// Coefficients, intercept first.
    beta: &'a [f64],
    /// Inverse overdispersion, possibly infinite.
    theta: f64,
    /// Variance floor.
    min_variance: f64,
    /// Residual clipping range.
    clip: (f64, f64),
    /// Coefficient on `log10(total UMI)`.
    log_umi_coef: f64,
    /// Whether anything beyond the intercept was fitted.
    has_covariates: bool,
}

impl<'a> SctGeneParams<'a> {
    /// Pulls one gene's parameters out of a model.
    ///
    /// ### Params
    ///
    /// * `model` - The fitted model.
    /// * `pos` - Position of the gene within it.
    ///
    /// ### Returns
    ///
    /// The parameters.
    pub(crate) fn new(model: &'a SctModel, pos: usize) -> Self {
        Self {
            beta: model.coefficients_for(pos),
            theta: model.theta[pos],
            min_variance: model.min_variance,
            clip: model.clip_range,
            log_umi_coef: model.log_umi_coef,
            has_covariates: model.has_covariates(),
        }
    }

    /// One cell's residual.
    ///
    /// ### Params
    ///
    /// * `log10_umi` - The cell's `log10(total UMI)`.
    /// * `covariates` - The cell's covariate row, empty when none were fitted.
    /// * `y` - The observed count.
    ///
    /// ### Returns
    ///
    /// The clipped Pearson residual.
    #[inline(always)]
    pub(crate) fn residual(&self, log10_umi: f64, covariates: &[f64], y: f64) -> f32 {
        let mut eta = self.beta[0] + self.log_umi_coef * log10_umi;
        if self.has_covariates {
            for (b, x) in self.beta[1..].iter().zip(covariates) {
                eta += b * x;
            }
        }
        let mu = eta.exp();
        let var = (mu + mu * mu / self.theta).max(self.min_variance);
        let (lo, hi) = self.clip;
        ((y - mu) / var.sqrt()).clamp(lo, hi) as f32
    }
}

/// Writes a dense residual row, resolving each cell's parameters through
/// `gene_for`.
///
/// Every cell starts at its zero-count residual and the stored non-zeros then
/// overwrite their own positions: one pass over the cells plus one over the
/// non-zeros, rather than densifying the counts first.
///
/// ### Params
///
/// * `counts` - The gene's non-zero counts.
/// * `indices` - Cell positions of those counts, within `0..n_cells`.
/// * `cells` - Per-cell library sizes and covariates.
/// * `out` - Destination row, overwritten in full.
/// * `gene_for` - The gene's parameters for a given cell position. Constant for
///   a single-model source, the cell's own group's parameters for a grouped one.
#[inline]
pub(crate) fn fill_residual_row<'a, F>(
    counts: &[f64],
    indices: &[u32],
    cells: &SctCellContext<'_>,
    out: &mut [f32],
    gene_for: F,
) where
    F: Fn(usize) -> &'a SctGeneParams<'a>,
{
    let covariates = cells.covariates;
    let simple = covariates.is_empty();

    for (c, slot) in out.iter_mut().enumerate() {
        let cov = if simple { &[][..] } else { covariates.row(c) };
        *slot = gene_for(c).residual(cells.log10_umi[c], cov, 0.0);
    }

    for (&i, &y) in indices.iter().zip(counts.iter()) {
        let c = i as usize;
        let cov = if simple { &[][..] } else { covariates.row(c) };
        out[c] = gene_for(c).residual(cells.log10_umi[c], cov, y);
    }
}

/// The variance floor from the median non-zero UMI count.
///
/// sctransform's `min_variance = "umi_median"`, which `vst.flavor = "v2"` turns
/// on. It is a single scalar over the whole matrix, not a per-gene quantity.
///
/// ### Params
///
/// * `median_nonzero_umi` - Median of every non-zero count in the matrix.
///
/// ### Returns
///
/// The variance floor.
pub fn min_variance_from_umi_median(median_nonzero_umi: f64) -> f64 {
    (median_nonzero_umi / UMI_MEDIAN_DIVISOR).powi(2)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Mean library size the reference used.
    const MEAN_CELL_SUM: f64 = 3007.0;

    /// Step-1 gene positions, from `scratchpad/reg_ref.R`. These come from R's
    /// own `sample()`, which no LCG here can reproduce, so the indices cross as
    /// text. They are integers, not floats, so nothing is lost doing so.
    const STEP1_IDX: [usize; 300] = [
        1, 3, 7, 8, 10, 11, 12, 15, 16, 18, 19, 21, 22, 23, 24, 25, 26, 29, 30, 31, 33, 34, 35, 40,
        43, 44, 45, 46, 48, 50, 52, 56, 60, 61, 62, 64, 65, 67, 69, 71, 72, 73, 74, 75, 76, 78, 82,
        84, 85, 86, 87, 90, 91, 92, 93, 95, 96, 98, 99, 100, 103, 104, 108, 110, 111, 113, 114,
        115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 128, 129, 130, 131, 134, 135, 136,
        141, 143, 144, 145, 146, 148, 150, 153, 155, 156, 158, 159, 162, 164, 167, 168, 169, 171,
        172, 173, 177, 178, 179, 180, 182, 184, 185, 186, 187, 190, 192, 193, 194, 196, 197, 198,
        201, 202, 203, 205, 206, 209, 211, 212, 214, 215, 218, 219, 220, 221, 224, 225, 226, 227,
        228, 229, 234, 235, 240, 242, 243, 244, 248, 249, 251, 252, 253, 255, 256, 257, 258, 259,
        261, 263, 265, 266, 268, 269, 270, 272, 274, 275, 276, 277, 278, 283, 284, 285, 286, 288,
        289, 290, 291, 292, 293, 294, 295, 297, 299, 301, 304, 305, 306, 307, 308, 309, 310, 311,
        312, 313, 314, 316, 319, 320, 323, 324, 325, 326, 329, 330, 331, 332, 334, 335, 336, 338,
        339, 341, 344, 346, 347, 352, 354, 356, 357, 358, 359, 361, 364, 366, 368, 369, 370, 372,
        373, 375, 377, 378, 384, 385, 388, 390, 391, 396, 397, 400, 402, 404, 405, 406, 407, 410,
        411, 412, 414, 415, 418, 420, 422, 424, 425, 427, 429, 431, 432, 433, 434, 436, 437, 438,
        439, 442, 443, 444, 447, 448, 449, 450, 451, 452, 454, 455, 456, 457, 458, 459, 461, 467,
        468, 469, 472, 473, 476, 477, 478, 479, 480, 481, 482, 485, 488, 490, 493, 497, 499,
    ];

    /// Numerical Recipes LCG, as `dev/gen_edger_fixtures.R` uses it.
    struct Lcg(u64);

    impl Lcg {
        fn unifs(&mut self, n: usize) -> Vec<f64> {
            (0..n)
                .map(|_| {
                    self.0 = (1_664_525_u64
                        .wrapping_mul(self.0)
                        .wrapping_add(1_013_904_223))
                        % 4_294_967_296;
                    self.0 as f64 / 4_294_967_296.0
                })
                .collect()
        }
    }

    /// Rebuilds the inputs `scratchpad/reg_ref.R` fed to
    /// `sctransform:::reg_model_pars`. The draw order matters and matches the
    /// R script call for call.
    fn reg_fixture() -> (SctGeneStats, Vec<NbOffsetFit>) {
        let (n_genes, n_step1) = (500_usize, 300_usize);
        let mut lcg = Lcg(20_260_101);

        let j = lcg.unifs(n_genes);
        let log_gmean: Vec<f64> = (0..n_genes)
            .map(|i| -4.0 + 4.0 * (i as f64 / (n_genes - 1) as f64) + (j[i] - 0.5) * 0.3)
            .collect();

        let a = lcg.unifs(n_genes);
        let amean: Vec<f64> = (0..n_genes)
            .map(|i| 10.0_f64.powf(log_gmean[i]) * (1.0 + a[i]))
            .collect();

        let u = lcg.unifs(n_genes);
        let var: Vec<f64> = (0..n_genes)
            .map(|i| {
                if u[i] < 0.08 {
                    amean[i] * 0.5
                } else {
                    let disp = 0.5 + 6.0 / (1.0 + (4.0 * (log_gmean[i] + 2.0)).exp());
                    amean[i] * (1.0 + amean[i] / disp)
                }
            })
            .collect();

        let m = lcg.unifs(n_step1);
        let inf_u = lcg.unifs(n_step1);
        let int_j = lcg.unifs(n_step1);

        let step1: Vec<NbOffsetFit> = (0..n_step1)
            .map(|k| {
                let g = STEP1_IDX[k];
                let theta = if inf_u[k] < 0.05 {
                    f64::INFINITY
                } else {
                    (0.5 + 6.0 / (1.0 + (4.0 * (log_gmean[g] + 2.0)).exp())) * (0.7 + 0.6 * m[k])
                };
                NbOffsetFit {
                    theta,
                    coefficients: vec![amean[g].ln() - 3000.0_f64.ln() + (int_j[k] - 0.5) * 0.2],
                }
            })
            .collect();

        (
            SctGeneStats {
                log_gmean,
                amean,
                var,
            },
            step1,
        )
    }

    /// R: `sctransform:::reg_model_pars(..., theta_regularization = "od_factor",
    /// exclude_poisson = TRUE, bw_adjust = 3)` on the same inputs.
    ///
    /// The reference run reported 141 Poisson genes and 12 parameter outliers,
    /// so this exercises the outlier drop, the Poisson exclusion and the
    /// offset-model override, not just the smoothing.
    #[test]
    fn test_regularise_matches_sctransform() {
        let (stats, step1) = reg_fixture();
        let params = SctParams::default();

        let model = regularise_sct_model(
            &step1,
            &STEP1_IDX,
            &stats,
            &[],
            MEAN_CELL_SUM,
            1.0,
            800,
            &params,
        )
        .unwrap();

        assert_eq!(model.len(), 500);
        assert_eq!(
            (0..model.len()).filter(|&g| model.is_poisson(g)).count(),
            141
        );

        // The leading genes are all below the 1e-3 mean floor, so they take the
        // closed-form offset model rather than a smoothed value.
        let int_head = [
            -17.353_708_150_040_36,
            -16.725_809_467_272_3,
            -16.551_839_121_323_83,
            -16.206_994_267_881_16,
            -16.962_119_107_905_046,
            -16.735_240_757_788_93,
            -16.771_623_732_989_518,
            -16.919_318_538_563_182,
            -17.012_739_495_553_93,
            -16.381_698_448_994_32,
            -16.746_081_708_430_64,
            -16.527_553_026_856_687,
        ];
        for (g, &e) in int_head.iter().enumerate() {
            assert!(model.theta[g].is_infinite(), "gene {g} should be Poisson");
            assert_relative_eq!(model.intercept(g), e, max_relative = 1e-12);
        }

        // The tail is the smoothed branch, which is what the bandwidth and the
        // kernel actually decide.
        let theta_tail = [
            0.638_060_996_011_783_7,
            0.869_965_647_883_973_5,
            0.750_698_028_787_386,
            0.635_926_212_826_603_2,
            1.101_069_258_148_505_3,
            0.684_416_184_721_164_2,
        ];
        for (k, &e) in theta_tail.iter().enumerate() {
            let g = 500 - 6 + k;
            assert_relative_eq!(model.theta[g], e, max_relative = 1e-10);
        }

        let theta_sum: f64 = model.theta.iter().filter(|t| t.is_finite()).sum();
        assert_relative_eq!(theta_sum, 592.180_275_674_843, max_relative = 1e-10);
        let int_sum: f64 = (0..model.len()).map(|g| model.intercept(g)).sum();
        assert_relative_eq!(int_sum, -6_122.860_122_515_623, max_relative = 1e-12);
    }

    /// The Poisson branch is the offset model exactly: theta infinite and the
    /// intercept read straight off the gene mean, never a curve value.
    #[test]
    fn test_regularise_uses_the_offset_model_for_poisson_genes() {
        let (stats, step1) = reg_fixture();

        let model = regularise_sct_model(
            &step1,
            &STEP1_IDX,
            &stats,
            &[],
            MEAN_CELL_SUM,
            1.0,
            800,
            &SctParams::default(),
        )
        .unwrap();

        for g in 0..model.len() {
            if !model.is_poisson(g) {
                continue;
            }
            assert_relative_eq!(
                model.intercept(g),
                stats.amean[g].ln() - MEAN_CELL_SUM.ln(),
                max_relative = 1e-14
            );
        }
    }

    #[test]
    fn test_regularise_rejects_an_out_of_range_step1_index() {
        let (stats, step1) = reg_fixture();
        let mut idx = STEP1_IDX.to_vec();
        idx[0] = 10_000;

        assert!(matches!(
            regularise_sct_model(
                &step1,
                &idx,
                &stats,
                &[],
                MEAN_CELL_SUM,
                1.0,
                800,
                &SctParams::default()
            ),
            Err(BixverseErrors::SctGeneIndexOutOfRange { index: 10_000, .. })
        ));
    }

    /// Nothing left to smooth is an error rather than a silent all-Poisson
    /// model, because it means the subsample was useless.
    #[test]
    fn test_regularise_rejects_too_few_surviving_genes() {
        let stats = SctGeneStats {
            log_gmean: vec![-1.0, -0.5, 0.0],
            // Every gene under the Poisson line, so none survive.
            amean: vec![1.0, 2.0, 3.0],
            var: vec![0.5, 1.0, 1.5],
        };
        let step1 = vec![
            NbOffsetFit {
                theta: 2.0,
                coefficients: vec![-7.0],
            },
            NbOffsetFit {
                theta: 3.0,
                coefficients: vec![-6.0],
            },
            NbOffsetFit {
                theta: 4.0,
                coefficients: vec![-5.0],
            },
        ];

        assert!(matches!(
            regularise_sct_model(
                &step1,
                &[0, 1, 2],
                &stats,
                &[],
                3000.0,
                1.0,
                100,
                &SctParams::default()
            ),
            Err(BixverseErrors::SctTooFewGenesToRegularise { kept: 0, total: 3 })
        ));
    }

    #[test]
    fn test_clip_range_defaults_to_sqrt_n_cells() {
        let p = SctParams::default();
        let (lo, hi) = p.resolve_clip_range(10_000);
        assert_relative_eq!(hi, 100.0, max_relative = 1e-14);
        assert_relative_eq!(lo, -100.0, max_relative = 1e-14);

        let p2 = SctParams {
            clip_range: Some((-5.0, 5.0)),
            ..SctParams::default()
        };
        assert_eq!(p2.resolve_clip_range(10_000), (-5.0, 5.0));
    }

    #[test]
    fn test_min_variance_from_umi_median() {
        assert_relative_eq!(
            min_variance_from_umi_median(10.0),
            4.0,
            max_relative = 1e-14
        );
    }
}
