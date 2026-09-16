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
//! [`super::sct_nb_fit`].
//!
//! ### References
//!
//! Hafemeister & Satija, Genome Biology, 2019; Choudhary & Satija, Genome
//! Biology, 2022

use crate::core::base::kernel_smooth::{bw_sj, ksmooth_normal};
use crate::core::math::stats::is_outlier;
use crate::errors::BixverseErrors;

use super::sct_nb_fit::{LOG_UMI_COEF, NbOffsetFit};

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
    /// Regularised inverse overdispersion. `f64::INFINITY` for a Poisson gene.
    pub theta: Vec<f64>,
    /// Regularised intercept on the natural-log scale.
    pub intercept: Vec<f64>,
    /// Coefficient on `log10(total UMI)`, always `ln(10)`. See
    /// [`LOG_UMI_COEF`].
    pub log_umi_coef: f64,
    /// Variance floor, `(median non-zero UMI / 5)^2`.
    pub min_variance: f64,
    /// Residual clipping range.
    pub clip_range: (f64, f64),
    /// Which genes were treated as Poisson and given the closed-form offset
    /// parameters rather than a smoothed fit.
    pub poisson: Vec<bool>,
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

    /// Whether any genes are modelled.
    ///
    /// ### Returns
    ///
    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.theta.is_empty()
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
/// ### Params
///
/// * `step1` - Per-gene fits from the step-1 subsample.
/// * `step1_idx` - Position of each step-1 gene in the full gene set, parallel
///   to `step1`.
/// * `stats` - Per-gene statistics over the full cell set, for every gene.
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
pub fn regularise_sct_model(
    step1: &[NbOffsetFit],
    step1_idx: &[usize],
    stats: &SctGeneStats,
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
    let intercept_s1: Vec<f64> = step1.iter().map(|f| f.intercept).collect();
    let gmean_s1: Vec<f64> = step1_idx.iter().map(|&g| stats.log_gmean[g]).collect();

    // The third parameter column, `log_umi`, is the constant ln(10). Within any
    // bin its median is itself and its MAD is zero, so its robust z-score is
    // exactly zero and it can never flag an outlier. Skipping it is a saving,
    // not a behaviour change.
    let out_disp = is_outlier(&disp_par, &gmean_s1, params.outlier_th)?;
    let out_int = is_outlier(&intercept_s1, &gmean_s1, params.outlier_th)?;

    let keep: Vec<usize> = (0..step1.len())
        .filter(|&i| {
            !out_disp[i] && !out_int[i] && step1[i].theta.is_finite() && !poisson[step1_idx[i]]
        })
        .collect();

    if keep.len() < 2 {
        return Err(BixverseErrors::SctTooFewGenesToRegularise {
            kept: keep.len(),
            total: step1.len(),
        });
    }

    let fit_x: Vec<f64> = keep.iter().map(|&i| gmean_s1[i]).collect();
    let fit_disp: Vec<f64> = keep.iter().map(|&i| disp_par[i]).collect();
    let fit_int: Vec<f64> = keep.iter().map(|&i| intercept_s1[i]).collect();

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

    let scatter = |smoothed: Vec<f64>| -> Vec<f64> {
        let mut out = vec![f64::NAN; n_genes];
        for (k, &g) in order.iter().enumerate() {
            out[g] = smoothed[k];
        }
        out
    };

    let (_, sm_disp) = ksmooth_normal(&fit_x, &fit_disp, &x_points, bw)?;
    let (_, sm_int) = ksmooth_normal(&fit_x, &fit_int, &x_points, bw)?;
    let mut disp_fit = scatter(sm_disp);
    let mut intercept = scatter(sm_int);

    // Poisson genes take the exact offset model: theta infinite and the
    // intercept straight off the gene mean, not a curve value.
    for g in 0..n_genes {
        if poisson[g] {
            disp_fit[g] = 0.0;
            intercept[g] = stats.amean[g].ln() - mean_cell_sum.ln();
        }
    }

    if let Some(g) = (0..n_genes).find(|&g| disp_fit[g].is_nan() || intercept[g].is_nan()) {
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
        theta,
        intercept,
        log_umi_coef: LOG_UMI_COEF,
        min_variance,
        clip_range: params.resolve_clip_range(n_cells),
        poisson,
    })
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
                    intercept: amean[g].ln() - 3000.0_f64.ln() + (int_j[k] - 0.5) * 0.2,
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

        let model =
            regularise_sct_model(&step1, &STEP1_IDX, &stats, MEAN_CELL_SUM, 1.0, 800, &params)
                .unwrap();

        assert_eq!(model.len(), 500);
        assert_eq!(model.poisson.iter().filter(|&&p| p).count(), 141);

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
            assert_relative_eq!(model.intercept[g], e, max_relative = 1e-12);
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
        let int_sum: f64 = model.intercept.iter().sum();
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
            MEAN_CELL_SUM,
            1.0,
            800,
            &SctParams::default(),
        )
        .unwrap();

        for g in 0..model.len() {
            if !model.poisson[g] {
                continue;
            }
            assert!(model.theta[g].is_infinite());
            assert_relative_eq!(
                model.intercept[g],
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
                intercept: -7.0,
            },
            NbOffsetFit {
                theta: 3.0,
                intercept: -6.0,
            },
            NbOffsetFit {
                theta: 4.0,
                intercept: -5.0,
            },
        ];

        assert!(matches!(
            regularise_sct_model(
                &step1,
                &[0, 1, 2],
                &stats,
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
