//! Per-gene negative binomial fit for scTransform v2.
//!
//! The v2 model is an NB GLM with the cell's total UMI count as a **fixed
//! offset** rather than a fitted coefficient:
//!
//! ```text
//! mu_c  = exp(x_c . beta) * total_umi_c
//! var_c = mu_c + alpha * mu_c^2        (alpha = 1 / theta)
//! ```
//!
//! With no covariates the design is a single intercept column and only two
//! numbers are learned per gene, which is the common case. Additional
//! cell-level covariates, sctransform's `latent_var` beyond `log_umi`, add
//! columns to `x` and coefficients to `beta`; the offset is unaffected, which
//! is what keeps the library-size slope pinned.
//!
//! sctransform gets these from `glmGamPoi::glm_gp(design, offset =
//! log(total_umi))`, which maximises the Cox-Reid adjusted profile likelihood
//! over the overdispersion.
//!
//! Every gene is independent, which is what makes the whole of scTransform
//! streamable: one gene's fit needs that gene's counts and the shared per-cell
//! offset, nothing else.

use edge_rs::dispersion::apl::apl_at;
use edge_rs::glm::levenberg::{LevenbergParams, mglm_levenberg};
use edge_rs::prelude::*;

use crate::core::math::optimise::brent_fmin;
use crate::errors::BixverseErrors;

////////////
// Consts //
////////////

/// Lower end of the overdispersion search, on the log scale.
///
/// `exp(-25)` is far enough below any dispersion that survives the Poisson test
/// below that the bracket never binds in practice, and the objective is flat
/// there, so a tighter bound would only cost iterations.
const LOG_ALPHA_LOWER: f64 = -25.0;

/// Upper end of the overdispersion search, on the log scale.
///
/// `exp(5)` is an overdispersion of about 150, well past anything a real gene
/// reaches. glmGamPoi's own bound is the same order.
const LOG_ALPHA_UPPER: f64 = 5.0;

/// Convergence tolerance for the overdispersion search, on the log scale.
///
/// Tighter than R's `optimize` default because the answer is exponentiated and
/// then inverted into theta, so a slack tolerance here is magnified twice.
const LOG_ALPHA_TOL: f64 = 1e-8;

/// Overdispersion below which the gene is called Poisson and theta set to
/// infinity.
///
/// `glm_gp` reports an overdispersion of exactly `0.0` when the CR-APL is
/// maximised at the boundary, but a bounded search on the log scale can only
/// approach it. This is the boundary in that parameterisation: below it the
/// negative binomial and the Poisson are indistinguishable at any sample size
/// this code will see.
const ALPHA_POISSON_FLOOR: f64 = 1e-10;

/// Relative deviance tolerance for the coefficient fit.
///
/// Far tighter than edgeR's `1e-6` default because these coefficients are a
/// parity target, not an input to a test statistic. At the default the
/// intercept lands within 1.8e-7 of the reference where this brings it inside
/// 1e-9, and the extra iterations do not show in the fit timings.
const COEF_TOL: f64 = 1e-14;

/// `ln(10)`, the coefficient on `log10(total UMI)` that a natural-log offset is
/// algebraically equivalent to.
///
/// sctransform carries it as a real column of `model_pars` so that the
/// regularisation step and the residual step can use one code path for the
/// offset model and the general one. It is a constant, so smoothing it is a
/// no-op, which is how "fixed slope" survives regularisation.
pub const LOG_UMI_COEF: f64 = std::f64::consts::LN_10;

/////////////
// Results //
/////////////

/// One gene's fitted negative binomial parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct NbOffsetFit {
    /// Inverse overdispersion, `1 / alpha`. `f64::INFINITY` for a gene that
    /// hit the Poisson boundary.
    pub theta: f64,
    /// Coefficients on the natural-log scale, one per design column. Entry `0`
    /// is the intercept, so with no covariates `mu_c = exp(coefficients[0]) *
    /// total_umi_c`.
    pub coefficients: Vec<f64>,
}

impl NbOffsetFit {
    /// The intercept, design column zero.
    ///
    /// ### Returns
    ///
    /// The intercept on the natural-log scale.
    pub fn intercept(&self) -> f64 {
        self.coefficients[0]
    }
}

/////////
// Fit //
/////////

/// Fits the negative binomial GLM with a log offset for one gene.
///
/// Maximises the Cox-Reid adjusted profile likelihood over `log(alpha)` with
/// [`brent_fmin`], then takes the coefficients at the maximiser. [`apl_at`]
/// already refits the coefficients at every dispersion it evaluates, so the two
/// are consistent by construction rather than by alternation.
///
/// ### Params
///
/// * `counts` - This gene's counts, one per cell.
/// * `log_offset` - `ln(total UMI)` per cell, parallel to `counts`.
/// * `design` - Design matrix, row-major `n_cells * n_coef`, column zero the
///   intercept. The library size is the offset and must not appear here.
/// * `n_coef` - Number of design columns.
///
/// ### Returns
///
/// The [`NbOffsetFit`], or [`BixverseErrors::LengthMismatch`] when the inputs
/// disagree, or an [`BixverseErrors::EdgeRsError`] from the underlying
/// `edge-rs` call.
///
/// ### References
///
/// Choudhary & Satija, Genome Biology, 2022; Ahlmann-Eltze & Huber,
/// Bioinformatics, 2021 (glmGamPoi)
pub fn fit_nb_offset_gene(
    counts: &[f64],
    log_offset: &[f64],
    design: &[f64],
    n_coef: usize,
) -> Result<NbOffsetFit, BixverseErrors> {
    let n = counts.len();
    if log_offset.len() != n {
        return Err(BixverseErrors::LengthMismatch {
            name: "log_offset",
            expected: n,
            found: log_offset.len(),
        });
    }
    if n == 0 || n_coef == 0 {
        return Err(BixverseErrors::LengthMismatch {
            name: "counts",
            expected: 1,
            found: n,
        });
    }
    if design.len() != n * n_coef {
        return Err(BixverseErrors::LengthMismatch {
            name: "design",
            expected: n * n_coef,
            found: design.len(),
        });
    }

    if counts.iter().all(|&y| y == 0.0) {
        let mut coefficients = vec![0.0; n_coef];
        coefficients[0] = f64::NEG_INFINITY;
        return Ok(NbOffsetFit {
            theta: f64::INFINITY,
            coefficients,
        });
    }

    let offset = Recycled::BySample(log_offset.to_vec());

    let neg_apl = |log_alpha: f64| -> f64 {
        match apl_at(counts, 1, n, design, n_coef, log_alpha.exp(), &offset, None) {
            Ok(v) => -v[0],
            Err(_) => f64::INFINITY,
        }
    };

    let log_alpha = brent_fmin(LOG_ALPHA_LOWER, LOG_ALPHA_UPPER, neg_apl, LOG_ALPHA_TOL);
    let alpha = log_alpha.exp();

    // A gene whose likelihood is still climbing at the lower bound is Poisson.
    // glm_gp reports that as an overdispersion of exactly zero.
    let (theta, dispersion) = if alpha <= ALPHA_POISSON_FLOOR {
        (f64::INFINITY, 0.0)
    } else {
        (1.0 / alpha, alpha)
    };

    let fit = mglm_levenberg(
        counts,
        1,
        n,
        design,
        n_coef,
        &Recycled::Scalar(dispersion),
        &offset,
        None,
        None,
        Some(LevenbergParams {
            tol: COEF_TOL,
            ..LevenbergParams::default()
        }),
    )
    .map_err(BixverseErrors::from)?;

    Ok(NbOffsetFit {
        theta,
        coefficients: fit.coefficients,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Genes whose overdispersion sits at the Poisson boundary. The profile
    /// likelihood is flat there, so theta carries no information and is only
    /// ever checked for being large.
    const POISSON_THETA_FLOOR: f64 = 1e5;

    /// The fixture `scratchpad/nbfit_ref.R` builds: 12 genes over 400 cells,
    /// counts straight out of the Numerical Recipes LCG with a per-gene
    /// modulus, so the abundance range spans a near-Poisson gene through to a
    /// strongly overdispersed one. Nothing crosses as text.
    fn fixture() -> (Vec<Vec<f64>>, Vec<f64>) {
        const MODULUS: [u64; 12] = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377];
        let (n_genes, n_cells) = (12_usize, 400_usize);

        let mut state = 20_260_101_u64;
        let mut counts = vec![vec![0.0_f64; n_cells]; n_genes];
        for (g, row) in counts.iter_mut().enumerate() {
            for slot in row.iter_mut() {
                state = (1_664_525_u64
                    .wrapping_mul(state)
                    .wrapping_add(1_013_904_223))
                    % 4_294_967_296;
                *slot = (state % MODULUS[g]) as f64;
            }
        }

        let total_umi: Vec<f64> = (0..n_cells)
            .map(|c| counts.iter().map(|row| row[c]).sum())
            .collect();

        (counts, total_umi)
    }

    /// The fixture has to agree with R's before any parameter comparison means
    /// anything.
    #[test]
    fn test_nb_fixture_matches_r() {
        let (counts, total_umi) = fixture();

        assert_eq!(&counts[0][..10], &[0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]);
        assert_eq!(&total_umi[..5], &[537.0, 364.0, 659.0, 379.0, 256.0]);
        let row_sums: Vec<f64> = counts.iter().map(|r| r.iter().sum()).collect();
        assert_eq!(
            row_sums,
            vec![
                200.0, 386.0, 749.0, 1400.0, 2435.0, 3847.0, 6590.0, 10701.0, 17882.0, 28504.0,
                47179.0, 74795.0
            ]
        );
    }

    /// The gate: does this compute the Cox-Reid adjusted joint MLE it claims
    /// to?
    ///
    /// Reference from `scratchpad/cr_ref.R`, which builds the same estimator
    /// independently in R out of `uniroot` on the score equation and
    /// `optimize` on the CR-adjusted profile likelihood, touching none of the
    /// same code. Agreement is to about 1e-9, so this is a real gate rather
    /// than a tolerance fudge.
    #[test]
    fn test_nb_offset_fit_is_the_cox_reid_joint_mle() {
        let (counts, total_umi) = fixture();
        let log_offset: Vec<f64> = total_umi.iter().map(|u| u.ln()).collect();
        // Intercept-only design: a single column of ones.
        let ones = vec![1.0_f64; log_offset.len()];

        let r_theta = [
            60_488_309.625_924_04,
            2_191_807_574.316_569_3,
            5.513_574_095_376_203,
            3.133_668_357_936_107_5,
            2.368_659_443_652_557,
            1.957_655_842_984_292_3,
            1.854_434_151_652_939_7,
            1.817_632_203_480_953_3,
            2.126_488_146_916_778,
            1.799_514_028_743_560_3,
            1.960_104_840_351_798_4,
            2.438_988_023_082_542_3,
        ];
        let r_intercept = [
            -6.880_733_455_250_665,
            -6.223_213_452_934_95,
            -5.538_823_940_048_09,
            -4.892_388_460_356_183,
            -4.325_969_879_985_012,
            -3.854_724_993_628_699,
            -3.314_049_780_990_182,
            -2.843_407_974_978_344,
            -2.327_788_350_269_296,
            -1.884_251_074_082_975_2,
            -1.420_506_386_373_765,
            -1.012_693_909_316_931_6,
        ];

        for (g, (&rt, &ri)) in r_theta.iter().zip(r_intercept.iter()).enumerate() {
            let fit = fit_nb_offset_gene(&counts[g], &log_offset, &ones, 1).unwrap();

            let rel_i = ((fit.intercept() - ri) / ri).abs();
            assert!(
                rel_i < 1e-8,
                "gene {g}: intercept {} vs R {ri} (rel {rel_i:.3e})",
                fit.intercept()
            );

            if rt > POISSON_THETA_FLOOR {
                assert!(
                    fit.theta > POISSON_THETA_FLOOR,
                    "gene {g}: theta {} should be at the Poisson boundary",
                    fit.theta
                );
                continue;
            }
            let rel_t = ((fit.theta - rt) / rt).abs();
            assert!(
                rel_t < 1e-6,
                "gene {g}: theta {} vs R {rt} (rel {rel_t:.3e})",
                fit.theta
            );
        }
    }

    /// How far this sits from what sctransform v2 actually ships, which is
    /// `glmGamPoi::glm_gp(design = ~1, offset = log(total_umi))`.
    ///
    /// The gap is real and deliberate: `glm_gp` fits its coefficient at an
    /// intermediate overdispersion and never refits at the final one, so its
    /// `Beta` is not at the optimum of its own reported dispersion. On the
    /// near-Poisson gene 0, where the optimum is the closed-form Poisson MLE
    /// `-6.880733`, `glm_gp` returns `-6.864962`, a log-likelihood gap of 0.025
    /// nats. This test pins the size of the disagreement so that a future
    /// change to either side shows up as a number rather than a surprise.
    #[test]
    fn test_nb_offset_fit_band_against_glmgampoi() {
        let (counts, total_umi) = fixture();
        let log_offset: Vec<f64> = total_umi.iter().map(|u| u.ln()).collect();
        // Intercept-only design: a single column of ones.
        let ones = vec![1.0_f64; log_offset.len()];

        let gp_theta = [
            f64::INFINITY,
            2_960_918.101_872_272,
            5.358_462_058_249_337,
            3.126_993_566_920_339,
            2.369_260_436_554_973,
            1.958_116_250_503_067_4,
            1.854_705_676_226_394,
            1.817_738_562_108_434_5,
            2.126_576_339_459_727,
            1.799_560_754_399_995_4,
            1.960_152_479_199_029,
            2.439_049_644_472_123,
        ];

        let mut worst = 0.0_f64;
        for (g, &gt) in gp_theta.iter().enumerate() {
            if gt > POISSON_THETA_FLOOR {
                continue;
            }
            let fit = fit_nb_offset_gene(&counts[g], &log_offset, &ones, 1).unwrap();
            worst = worst.max(((fit.theta - gt) / gt).abs());
        }

        assert!(
            worst < 0.03,
            "theta drifted {worst:.4} from glmGamPoi, further than the measured 2.9%"
        );
    }

    #[test]
    fn test_nb_offset_fit_rejects_length_mismatch() {
        assert!(matches!(
            fit_nb_offset_gene(&[1.0, 2.0], &[0.0], &[1.0, 1.0], 1),
            Err(BixverseErrors::LengthMismatch { .. })
        ));
    }

    /// An all-zero gene carries no information about either parameter, so the
    /// likelihood is flat in both and a search would return wherever it
    /// happened to stop. The short circuit makes that explicit instead.
    #[test]
    fn test_nb_offset_fit_short_circuits_an_all_zero_gene() {
        let counts = vec![0.0_f64; 50];
        let log_offset: Vec<f64> = (0..50).map(|i| (100.0 + i as f64).ln()).collect();

        let fit = fit_nb_offset_gene(&counts, &log_offset, &vec![1.0; counts.len()], 1).unwrap();

        assert!(fit.theta.is_infinite());
        assert_eq!(fit.intercept(), f64::NEG_INFINITY);
    }

    /// A gene that is underdispersed relative to Poisson has its CR-adjusted
    /// profile likelihood maximised at the boundary, which must come back as
    /// theta infinite and the closed-form Poisson intercept rather than a
    /// huge finite theta from wherever the search stalled.
    #[test]
    fn test_nb_offset_fit_reports_the_poisson_boundary() {
        // Perfectly regular counts: zero variance, so far below the Poisson
        // mean-variance line that the overdispersion is pinned at the boundary.
        let counts = vec![5.0_f64; 200];
        let total_umi = vec![1000.0_f64; 200];
        let log_offset: Vec<f64> = total_umi.iter().map(|u| u.ln()).collect();
        // Intercept-only design: a single column of ones.
        let ones = vec![1.0_f64; counts.len()];

        let fit = fit_nb_offset_gene(&counts, &log_offset, &ones, 1).unwrap();

        assert!(
            fit.theta > POISSON_THETA_FLOOR,
            "theta {} should be at the Poisson boundary",
            fit.theta
        );
        let sum_y: f64 = counts.iter().sum();
        let sum_off: f64 = total_umi.iter().sum();
        let poisson_mle = (sum_y / sum_off).ln();
        assert!((fit.intercept() - poisson_mle).abs() < 1e-10);
    }
}
