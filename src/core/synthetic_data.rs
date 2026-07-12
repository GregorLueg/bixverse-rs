//! Synthetic bulk RNAseq data with configurable co-expression topology plus
//! Splatter-style sequencing-depth sparsification.
//!
//! Two things this module is meant to stress-test: co-expression module
//! detection (WGCNA-like TOM plus downstream clustering) and how well those
//! methods survive scRNA-seq-style dropout. On top of that, four generator
//! variants make the same synthetic layout usable as a fair benchmark for
//! NMF, ICA and dictionary-learning methods (DGRDL) — see
//! [`BulkDataGenerator`] for the trade-offs.
//!
//! The counts come from a Gamma-Poisson (NB) mixture around a per-gene mean.
//! When modules are on, a per-sample latent factor is drawn per module and
//! each module gene loads on it (linearly on the log scale) plus per-gene
//! noise. The variant chooses which factor and loading distributions are
//! drawn from, which controls whether the resulting structure has hubs, a
//! non-negative activity matrix (NMF-friendly), or non-Gaussian sources
//! (ICA-friendly).
//!
//! The sparsification path is a Splatter-style multinomial downsample: a
//! per-sample capture efficiency `s_j ~ LogNormal(0, sigma_s)` sets a target
//! library size, and each gene's count is binomially thinned to hit it. This
//! replaces the previous logistic and power-decay dropout functions, which
//! had magic clamps and no biological anchor.
//!
//! ### References
//!
//! * Zhang B, Horvath S. A general framework for weighted gene co-expression
//!   network analysis. Stat Appl Genet Mol Biol, 2005.
//! * Khanin R, Wit E. How scale-free are biological networks? J Comput Biol,
//!   2006. (Caveat on the strict scale-free claim.)
//! * Zappia L, Phipson B, Oshlack A. Splatter: simulation of single-cell RNA
//!   sequencing data. Genome Biology, 2017.

use faer::{Mat, MatRef};
use rand::prelude::*;
use rand_distr::{Beta, Binomial, Distribution, Gamma, LogNormal, Normal, Poisson};
use rayon::prelude::*;

use crate::errors::BixverseErrors;
use crate::prelude::BixverseFloat;

////////////////////
// Default consts //
////////////////////

/// Shape of the Gamma prior on per-gene mean expression. Combined with the
/// scale below, gives a mean of ~50 counts and a heavy right tail.
const DEFAULT_MEAN_EXP_SHAPE: f64 = 5.0;
/// Scale of the Gamma prior on per-gene mean expression.
const DEFAULT_MEAN_EXP_SCALE: f64 = 10.0;
/// Intercept of the NB dispersion trend `disp = 1 / (a + b * mean)`.
const DEFAULT_DISP_INTERCEPT: f64 = 0.2;
/// Slope of the NB dispersion trend.
const DEFAULT_DISP_SLOPE: f64 = 0.3;
/// Per-gene per-sample noise std on the latent log-signal.
const DEFAULT_NOISE_STD: f64 = 0.3;
/// Std of the Normal latent factor. Kept below 1 so even hub genes stay in a
/// workable log-signal range — otherwise `exp(z)` spans too many orders of
/// magnitude, the marginal-mean rescale concentrates mass on a single sample,
/// and the linear correlation between the hub and the rest of the module
/// collapses.
const DEFAULT_NORMAL_FACTOR_STD: f64 = 0.3;
/// Shape of the Gamma latent factor used for the non-negative variant. With
/// scale = 0.3, mean = 0.6 and mode = 0.3 — non-negative, unimodal, mild tail.
const DEFAULT_GAMMA_FACTOR_SHAPE: f64 = 2.0;
/// Scale of the Gamma latent factor for the non-negative variant.
const DEFAULT_GAMMA_FACTOR_SCALE: f64 = 0.3;
/// Scale of the Laplace latent factor for the non-Gaussian variant.
/// `b = 0.3` gives std ≈ 0.42, comparable to the Normal factor default.
const DEFAULT_LAPLACE_FACTOR_SCALE: f64 = 0.3;
/// LogNormal location for hub-style loadings. `mu = 0` gives median loading 1.
const DEFAULT_LOADING_LN_MU: f64 = 0.0;
/// LogNormal scale for hub-style loadings. `sigma = 0.7` gives q99 loading of
/// about 5, so a hub's log-signal std is `~5 * factor_std ≈ 1.5` — clean exp
/// range, tail heavy enough to drive a hub-degree gradient.
const DEFAULT_LOADING_LN_SIGMA: f64 = 0.7;
/// Fraction of module genes flagged as hubs (top X% by loading).
const DEFAULT_HUB_PERCENTILE: f64 = 0.1;
/// Reference library size for the sequencing-depth dropout.
const DEFAULT_TARGET_LIB_SIZE: f64 = 20_000.0;
/// LogNormal scale for per-sample size factors (Splatter default region).
const DEFAULT_CAPTURE_SIGMA: f64 = 0.5;

/////////////////////
// Generator enums //
/////////////////////

/// Shared config for LogNormal-loading variants (all except [`BulkDataGenerator::Modular`]).
///
/// ### Fields
///
/// The tail of the LogNormal drives hub behaviour: bigger `sigma` = heavier
/// tail = a few module genes with much larger loadings than the rest.
#[derive(Clone, Debug)]
pub struct LoadingLogNormal<T: BixverseFloat> {
    /// LogNormal location parameter for the loading distribution.
    pub mu: T,
    /// LogNormal scale parameter for the loading distribution.
    pub sigma: T,
    /// Top-fraction of module genes flagged as hubs by loading rank.
    pub hub_percentile: T,
}

impl<T: BixverseFloat> Default for LoadingLogNormal<T> {
    fn default() -> Self {
        Self {
            mu: T::from_f64(DEFAULT_LOADING_LN_MU).unwrap(),
            sigma: T::from_f64(DEFAULT_LOADING_LN_SIGMA).unwrap(),
            hub_percentile: T::from_f64(DEFAULT_HUB_PERCENTILE).unwrap(),
        }
    }
}

/// Config for [`BulkDataGenerator::Modular`].
///
/// ### Fields
///
/// Uses a Beta(5, 2) loading distribution (mean ~0.71, narrow) and a Normal
/// factor. No hubs.
#[derive(Clone, Debug)]
pub struct ModularConfig<T: BixverseFloat> {
    /// Std of the Normal factor drawn per sample.
    pub factor_std: T,
}

impl<T: BixverseFloat> Default for ModularConfig<T> {
    fn default() -> Self {
        Self {
            factor_std: T::from_f64(DEFAULT_NORMAL_FACTOR_STD).unwrap(),
        }
    }
}

/// Config for [`BulkDataGenerator::HubModular`].
///
/// ### Fields
///
/// LogNormal loadings + Normal factor. Heavy-tailed loading gives a hub /
/// non-hub degree gradient after correlation thresholding — useful for
/// WGCNA-style benchmarks.
#[derive(Clone, Debug)]
pub struct HubModularConfig<T: BixverseFloat> {
    /// Std of the Normal factor drawn per sample.
    pub factor_std: T,
    /// LogNormal loading config.
    pub loading: LoadingLogNormal<T>,
}

impl<T: BixverseFloat> Default for HubModularConfig<T> {
    fn default() -> Self {
        Self {
            factor_std: T::from_f64(DEFAULT_NORMAL_FACTOR_STD).unwrap(),
            loading: LoadingLogNormal::default(),
        }
    }
}

/// Config for [`BulkDataGenerator::NonNegativeFactor`].
///
/// ### Fields
///
/// LogNormal loadings + Gamma factor. The Gamma factor is non-negative, so
/// the ground-truth activity matrix `H` is directly comparable to what NMF
/// recovers (NMF constrains `H >= 0`).
#[derive(Clone, Debug)]
pub struct NonNegativeFactorConfig<T: BixverseFloat> {
    /// Shape of the Gamma factor drawn per sample.
    pub factor_shape: T,
    /// Scale of the Gamma factor drawn per sample.
    pub factor_scale: T,
    /// LogNormal loading config.
    pub loading: LoadingLogNormal<T>,
}

impl<T: BixverseFloat> Default for NonNegativeFactorConfig<T> {
    fn default() -> Self {
        Self {
            factor_shape: T::from_f64(DEFAULT_GAMMA_FACTOR_SHAPE).unwrap(),
            factor_scale: T::from_f64(DEFAULT_GAMMA_FACTOR_SCALE).unwrap(),
            loading: LoadingLogNormal::default(),
        }
    }
}

/// Config for [`BulkDataGenerator::NonGaussianFactor`].
///
/// ### Fields
///
/// LogNormal loadings + Laplace factor. Laplace is symmetric heavy-tailed —
/// non-Gaussian by construction — which is the identifiability condition ICA
/// needs on the source signals.
#[derive(Clone, Debug)]
pub struct NonGaussianFactorConfig<T: BixverseFloat> {
    /// Laplace scale (`b`) of the factor. Std of Laplace(0, b) is `b * sqrt(2)`.
    pub factor_scale: T,
    /// LogNormal loading config.
    pub loading: LoadingLogNormal<T>,
}

impl<T: BixverseFloat> Default for NonGaussianFactorConfig<T> {
    fn default() -> Self {
        Self {
            factor_scale: T::from_f64(DEFAULT_LAPLACE_FACTOR_SCALE).unwrap(),
            loading: LoadingLogNormal::default(),
        }
    }
}

/// Topology + factor / loading distribution to inject into the module
/// structure.
///
/// Each variant carries its own config so downstream code and R-side
/// dispatch can forward a single parameter struct.
#[derive(Clone, Debug)]
pub enum BulkDataGenerator<T: BixverseFloat> {
    /// Beta(5, 2) loadings, Normal factor. Homogeneous within-module
    /// correlation, no hubs.
    Modular(ModularConfig<T>),
    /// LogNormal loadings, Normal factor. Heavy-tailed loading distribution
    /// makes a few module genes act as hubs. WGCNA-style benchmark default.
    HubModular(HubModularConfig<T>),
    /// LogNormal loadings, Gamma (non-negative) factor. NMF-friendly ground
    /// truth: the activity matrix `H` is non-negative by construction.
    NonNegativeFactor(NonNegativeFactorConfig<T>),
    /// LogNormal loadings, Laplace (non-Gaussian) factor. ICA-friendly ground
    /// truth: source signals are non-Gaussian, satisfying ICA identifiability.
    NonGaussianFactor(NonGaussianFactorConfig<T>),
}

impl<T: BixverseFloat> Default for BulkDataGenerator<T> {
    fn default() -> Self {
        BulkDataGenerator::HubModular(HubModularConfig::default())
    }
}

impl<T: BixverseFloat> BulkDataGenerator<T> {
    /// Access the shared LogNormal loading config, if the variant has one.
    /// [`BulkDataGenerator::Modular`] returns `None` (Beta loadings).
    fn loading(&self) -> Option<&LoadingLogNormal<T>> {
        match self {
            BulkDataGenerator::Modular(_) => None,
            BulkDataGenerator::HubModular(c) => Some(&c.loading),
            BulkDataGenerator::NonNegativeFactor(c) => Some(&c.loading),
            BulkDataGenerator::NonGaussianFactor(c) => Some(&c.loading),
        }
    }

    /// Human-readable label for logs and error messages.
    fn label(&self) -> &'static str {
        match self {
            BulkDataGenerator::Modular(_) => "modular",
            BulkDataGenerator::HubModular(_) => "hub_modular",
            BulkDataGenerator::NonNegativeFactor(_) => "non_negative_factor",
            BulkDataGenerator::NonGaussianFactor(_) => "non_gaussian_factor",
        }
    }
}

/// Parse a generator label into a [`BulkDataGenerator`] with default config.
///
/// ### Params
///
/// * `s` - String to parse; matched case-insensitively.
///
/// ### Returns
///
/// `Some(BulkDataGenerator)` on match with default config for that variant,
/// `None` otherwise. Use the enum constructors directly if you need to
/// override the defaults.
pub fn parse_bulk_data_generator<T: BixverseFloat>(s: &str) -> Option<BulkDataGenerator<T>> {
    match s.to_lowercase().as_str() {
        "modular" => Some(BulkDataGenerator::Modular(ModularConfig::default())),
        "hub_modular" | "hubmodular" | "hub-modular" => Some(BulkDataGenerator::HubModular(
            HubModularConfig::default(),
        )),
        "non_negative_factor" | "nonnegative" | "nmf" => Some(
            BulkDataGenerator::NonNegativeFactor(NonNegativeFactorConfig::default()),
        ),
        "non_gaussian_factor" | "nongaussian" | "ica" => Some(
            BulkDataGenerator::NonGaussianFactor(NonGaussianFactorConfig::default()),
        ),
        _ => None,
    }
}

/// Dropout / sparsification strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropoutStrategy {
    /// Splatter-style: per-sample size factor `s_j ~ LogNormal(0, sigma_s)`,
    /// target library `L_j = target * s_j`, per-gene binomial thinning to hit
    /// the target.
    SequencingDepth,
}

/// Parse the dropout strategy from a string (for R-side dispatch).
///
/// ### Params
///
/// * `s` - String to parse; matched case-insensitively.
///
/// ### Returns
///
/// `Some(DropoutStrategy)` on match, `None` otherwise.
pub fn parse_dropout_strategy(s: &str) -> Option<DropoutStrategy> {
    match s.to_lowercase().as_str() {
        "seq_depth" | "sequencing_depth" | "sequencingdepth" => {
            Some(DropoutStrategy::SequencingDepth)
        }
        _ => None,
    }
}

////////////
// Params //
////////////

/// Parameters for [`generate_bulk_rnaseq`].
///
/// The generator variant lives inside [`SyntheticRnaSeqParams::generator`];
/// everything else — dimensions, module layout, the NB count model, noise —
/// is shared across variants.
///
/// ### Fields
///
/// See per-field docs. The [`Default`] impl gives 100 samples, 2000 genes,
/// five modules of decreasing size and [`BulkDataGenerator::HubModular`]
/// with its defaults.
#[derive(Clone, Debug)]
pub struct SyntheticRnaSeqParams<T: BixverseFloat> {
    /// Number of samples (columns) in the count matrix.
    pub num_samples: usize,
    /// Number of genes (rows) in the count matrix.
    pub num_genes: usize,
    /// Base seed. Per-gene RNGs are derived by wrapping-add on this.
    pub seed: u64,
    /// Sizes of the co-expression modules. Empty means no modules.
    /// `sum(module_sizes) <= num_genes` is enforced.
    pub module_sizes: Vec<usize>,
    /// Which topology / distribution family to inject. See
    /// [`BulkDataGenerator`].
    pub generator: BulkDataGenerator<T>,
    /// Gamma shape for per-gene mean expression.
    pub mean_exp_gamma_shape: T,
    /// Gamma scale for per-gene mean expression.
    pub mean_exp_gamma_scale: T,
    /// Intercept of the NB dispersion trend `disp = 1 / (a + b * mean)`.
    pub disp_intercept: T,
    /// Slope of the NB dispersion trend.
    pub disp_slope: T,
    /// Per-gene per-sample noise std on the latent log-signal. Smaller =
    /// tighter tracking of the module factor.
    pub noise_std: T,
}

impl<T: BixverseFloat> Default for SyntheticRnaSeqParams<T> {
    fn default() -> Self {
        Self {
            num_samples: 100,
            num_genes: 2_000,
            seed: 42,
            module_sizes: vec![300, 250, 200, 300, 500],
            generator: BulkDataGenerator::default(),
            mean_exp_gamma_shape: T::from_f64(DEFAULT_MEAN_EXP_SHAPE).unwrap(),
            mean_exp_gamma_scale: T::from_f64(DEFAULT_MEAN_EXP_SCALE).unwrap(),
            disp_intercept: T::from_f64(DEFAULT_DISP_INTERCEPT).unwrap(),
            disp_slope: T::from_f64(DEFAULT_DISP_SLOPE).unwrap(),
            noise_std: T::from_f64(DEFAULT_NOISE_STD).unwrap(),
        }
    }
}

impl<T: BixverseFloat> SyntheticRnaSeqParams<T> {
    /// Sanity-check the parameters.
    ///
    /// ### Returns
    ///
    /// `Ok(())` on success, [`BixverseErrors::InvalidArgument`] on failure.
    pub fn validate(&self) -> Result<(), BixverseErrors> {
        if self.num_samples == 0 {
            return Err(BixverseErrors::InvalidArgument(
                "num_samples must be > 0".into(),
            ));
        }
        if self.num_genes == 0 {
            return Err(BixverseErrors::InvalidArgument(
                "num_genes must be > 0".into(),
            ));
        }
        let total: usize = self.module_sizes.iter().sum();
        if total > self.num_genes {
            return Err(BixverseErrors::InvalidArgument(format!(
                "sum(module_sizes)={total} exceeds num_genes={}",
                self.num_genes
            )));
        }
        if self.mean_exp_gamma_shape.to_f64().unwrap() <= 0.0
            || self.mean_exp_gamma_scale.to_f64().unwrap() <= 0.0
        {
            return Err(BixverseErrors::InvalidArgument(
                "mean_exp gamma shape/scale must be > 0".into(),
            ));
        }
        if self.noise_std.to_f64().unwrap() < 0.0 {
            return Err(BixverseErrors::InvalidArgument(
                "noise_std must be >= 0".into(),
            ));
        }
        // per-variant validation
        match &self.generator {
            BulkDataGenerator::Modular(c) => {
                if c.factor_std.to_f64().unwrap() <= 0.0 {
                    return Err(BixverseErrors::InvalidArgument(
                        "modular: factor_std must be > 0".into(),
                    ));
                }
            }
            BulkDataGenerator::HubModular(c) => {
                if c.factor_std.to_f64().unwrap() <= 0.0 {
                    return Err(BixverseErrors::InvalidArgument(
                        "hub_modular: factor_std must be > 0".into(),
                    ));
                }
                validate_loading(&c.loading, "hub_modular")?;
            }
            BulkDataGenerator::NonNegativeFactor(c) => {
                if c.factor_shape.to_f64().unwrap() <= 0.0
                    || c.factor_scale.to_f64().unwrap() <= 0.0
                {
                    return Err(BixverseErrors::InvalidArgument(
                        "non_negative_factor: factor_shape/factor_scale must be > 0".into(),
                    ));
                }
                validate_loading(&c.loading, "non_negative_factor")?;
            }
            BulkDataGenerator::NonGaussianFactor(c) => {
                if c.factor_scale.to_f64().unwrap() <= 0.0 {
                    return Err(BixverseErrors::InvalidArgument(
                        "non_gaussian_factor: factor_scale must be > 0".into(),
                    ));
                }
                validate_loading(&c.loading, "non_gaussian_factor")?;
            }
        }
        Ok(())
    }
}

/// Validate the LogNormal loading sub-config.
fn validate_loading<T: BixverseFloat>(
    l: &LoadingLogNormal<T>,
    label: &str,
) -> Result<(), BixverseErrors> {
    if l.sigma.to_f64().unwrap() <= 0.0 {
        return Err(BixverseErrors::InvalidArgument(format!(
            "{label}: loading.sigma must be > 0"
        )));
    }
    let hub_p = l.hub_percentile.to_f64().unwrap();
    if !(hub_p > 0.0 && hub_p <= 1.0) {
        return Err(BixverseErrors::InvalidArgument(format!(
            "{label}: loading.hub_percentile must be in (0, 1]"
        )));
    }
    Ok(())
}

/// Parameters for [`apply_dropout`].
///
/// ### Fields
///
/// The [`Default`] impl gives Splatter-style sequencing-depth dropout with a
/// reference library size of 20,000 and per-sample size factor std of 0.5.
#[derive(Clone, Debug)]
pub struct SparsityParams<T: BixverseFloat> {
    /// Which dropout strategy to run.
    pub strategy: DropoutStrategy,
    /// Reference library size. Actual per-sample target is
    /// `target_library_size * s_j` where `s_j ~ LogNormal(0, sigma_s)`.
    pub target_library_size: T,
    /// Std of the LogNormal size-factor distribution.
    pub capture_efficiency_sigma: T,
    /// Seed for the dropout RNG.
    pub seed: u64,
}

impl<T: BixverseFloat> Default for SparsityParams<T> {
    fn default() -> Self {
        Self {
            strategy: DropoutStrategy::SequencingDepth,
            target_library_size: T::from_f64(DEFAULT_TARGET_LIB_SIZE).unwrap(),
            capture_efficiency_sigma: T::from_f64(DEFAULT_CAPTURE_SIGMA).unwrap(),
            seed: 42,
        }
    }
}

impl<T: BixverseFloat> SparsityParams<T> {
    /// Sanity-check the parameters.
    ///
    /// ### Returns
    ///
    /// `Ok(())` on success, [`BixverseErrors::InvalidArgument`] on failure.
    pub fn validate(&self) -> Result<(), BixverseErrors> {
        if self.target_library_size.to_f64().unwrap() <= 0.0 {
            return Err(BixverseErrors::InvalidArgument(
                "target_library_size must be > 0".into(),
            ));
        }
        if self.capture_efficiency_sigma.to_f64().unwrap() <= 0.0 {
            return Err(BixverseErrors::InvalidArgument(
                "capture_efficiency_sigma must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/////////////
// Results //
/////////////

/// Output of [`generate_bulk_rnaseq`].
///
/// ### Fields
///
/// Beyond the count matrix, the ground-truth latent factors and per-gene
/// loadings are returned so downstream tests can compute analytical
/// expectations (e.g. correlate a recovered module eigengene against the
/// true factor) without materialising a full gene x gene correlation matrix.
#[derive(Clone, Debug)]
pub struct SyntheticRnaSeqData<T: BixverseFloat> {
    /// Count matrix, shape `(num_genes, num_samples)`.
    pub count_matrix: Mat<T>,
    /// Per-gene module assignment. `0` = background, `1..=K` = module id.
    pub gene_modules: Vec<usize>,
    /// Indices of the genes flagged as hubs. Empty under
    /// [`BulkDataGenerator::Modular`].
    pub module_hubs: Vec<usize>,
    /// Per-gene loading on its module's latent factor. `0` for background.
    pub loadings: Vec<T>,
    /// Latent factor matrix, shape `(num_modules, num_samples)`. Empty when
    /// no modules were requested.
    pub module_factors: Mat<T>,
    /// Noise std used for the latent log-signal, echoed back for analytical
    /// ground truth.
    pub sigma_noise: T,
}

////////////////////
// Bulk generator //
////////////////////

/// Generate synthetic bulk RNAseq counts with the requested co-expression
/// topology.
///
/// Per-gene mean expression comes from a Gamma prior. When `module_sizes` is
/// non-empty, each module gets one latent factor per sample and each module
/// gene loads on it (linearly on the log scale) plus per-gene noise. The
/// [`BulkDataGenerator`] variant controls which factor and loading
/// distributions are drawn from. Counts are Gamma-Poisson (NB) sampled with
/// dispersion trending `1 / (a + b * mean)`.
///
/// ### Params
///
/// * `params` - See [`SyntheticRnaSeqParams`].
///
/// ### Returns
///
/// [`SyntheticRnaSeqData`] on success, [`BixverseErrors::InvalidArgument`]
/// on invalid params. The count matrix has shape `(num_genes, num_samples)`.
///
/// ### References
///
/// Zhang & Horvath 2005 (WGCNA), Khanin & Wit 2006 (scale-free caveat).
pub fn generate_bulk_rnaseq<T>(
    params: &SyntheticRnaSeqParams<T>,
) -> Result<SyntheticRnaSeqData<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    params.validate()?;

    let num_samples = params.num_samples;
    let num_genes = params.num_genes;
    let seed = params.seed;
    let sigma_noise_f64 = params.noise_std.to_f64().unwrap();

    let mut rng = StdRng::seed_from_u64(seed);

    // ---- gene-level statistics --------------------------------------------
    let gamma_mean = Gamma::new(
        params.mean_exp_gamma_shape.to_f64().unwrap(),
        params.mean_exp_gamma_scale.to_f64().unwrap(),
    )
    .map_err(|e| BixverseErrors::InvalidArgument(format!("gamma(mean) init: {e}")))?;
    let mean_exp: Vec<f64> = (0..num_genes).map(|_| gamma_mean.sample(&mut rng)).collect();
    let disp_intercept = params.disp_intercept.to_f64().unwrap();
    let disp_slope = params.disp_slope.to_f64().unwrap();
    let dispersion: Vec<f64> = mean_exp
        .iter()
        .map(|&m| 1.0 / (disp_intercept + disp_slope * m))
        .collect();

    // ---- module assignment ------------------------------------------------
    let mut gene_modules = vec![0usize; num_genes];
    let mut gene_idx = 0usize;
    for (m_id, &size) in params.module_sizes.iter().enumerate() {
        for _ in 0..size {
            if gene_idx >= num_genes {
                break;
            }
            gene_modules[gene_idx] = m_id + 1;
            gene_idx += 1;
        }
    }
    let num_modules = params.module_sizes.len();

    // ---- module latent factors --------------------------------------------
    // faer Mat is column-major; storing as (n_modules, n_samples) means each
    // sample is a column. We keep a Vec<Vec<f64>> for cheap per-thread reads
    // in the parallel gene loop, then copy into a Mat<T> at the end.
    let module_factor_vecs =
        draw_factors(num_modules, num_samples, &params.generator, &mut rng)
            .map_err(|e| BixverseErrors::InvalidArgument(format!("factor draw: {e}")))?;

    // ---- per-gene loadings ------------------------------------------------
    let loadings_f64 = draw_loadings(&gene_modules, num_genes, &params.generator, &mut rng)?;

    // ---- hub identification -----------------------------------------------
    let module_hubs = match &params.generator {
        BulkDataGenerator::Modular(_) => Vec::new(),
        _ => {
            let hub_p = params
                .generator
                .loading()
                .map(|l| l.hub_percentile.to_f64().unwrap())
                .unwrap_or(DEFAULT_HUB_PERCENTILE);
            identify_hubs(&gene_modules, &loadings_f64, hub_p)
        }
    };

    // ---- per-gene NB sampling ---------------------------------------------
    let inv_num_samples = 1.0 / num_samples as f64;
    let has_modules = num_modules > 0;

    let gene_data: Vec<Vec<f64>> = (0..num_genes)
        .into_par_iter()
        .map(|i| {
            let mut local_rng = StdRng::seed_from_u64(seed.wrapping_add((i as u64) * 1000 + 1));
            let mut counts = Vec::with_capacity(num_samples);

            let module_id = gene_modules[i];
            let in_module = has_modules && module_id > 0;

            if in_module {
                let m = module_id - 1;
                let loading = loadings_f64[i];
                let factor = &module_factor_vecs[m];

                // eps ~ Normal(0, sigma_N); latent log-signal z = loading * f + eps
                let noise = if sigma_noise_f64 > 0.0 {
                    Some(Normal::new(0.0, sigma_noise_f64).unwrap())
                } else {
                    None
                };
                let mut latent_exp = Vec::with_capacity(num_samples);
                let mut sum_signal = 0.0;
                for j in 0..num_samples {
                    let eps = match noise {
                        Some(n) => n.sample(&mut local_rng),
                        None => 0.0,
                    };
                    let z = loading * factor[j] + eps;
                    let s = z.exp();
                    latent_exp.push(s);
                    sum_signal += s;
                }
                let mean_signal = sum_signal * inv_num_samples;
                let scale = if mean_signal > 0.0 {
                    mean_exp[i] / mean_signal
                } else {
                    1.0
                };

                let r = 1.0 / dispersion[i];
                for j in 0..num_samples {
                    let mu_ij = latent_exp[j] * scale;
                    let count = nb_sample(r, mu_ij, &mut local_rng);
                    counts.push(count);
                }
            } else {
                // background: pure NB around mean_exp[i]
                let r = 1.0 / dispersion[i];
                for _ in 0..num_samples {
                    let count = nb_sample(r, mean_exp[i], &mut local_rng);
                    counts.push(count);
                }
            }

            counts
        })
        .collect();

    // ---- write into faer Mats --------------------------------------------
    let mut count_matrix: Mat<T> = Mat::zeros(num_genes, num_samples);
    for (i, gene_counts) in gene_data.into_iter().enumerate() {
        for (j, c) in gene_counts.into_iter().enumerate() {
            count_matrix[(i, j)] = T::from_f64(c).unwrap();
        }
    }

    let mut module_factors: Mat<T> = Mat::zeros(num_modules, num_samples);
    for (m, row) in module_factor_vecs.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            module_factors[(m, j)] = T::from_f64(v).unwrap();
        }
    }

    let loadings: Vec<T> = loadings_f64
        .into_iter()
        .map(|v| T::from_f64(v).unwrap())
        .collect();

    let _label = params.generator.label(); // reserved for future logging

    Ok(SyntheticRnaSeqData {
        count_matrix,
        gene_modules,
        module_hubs,
        loadings,
        module_factors,
        sigma_noise: params.noise_std,
    })
}

// ---- helpers ----------------------------------------------------------

/// Draw the per-module latent factors for every sample.
///
/// One row per module, one column per sample. The factor distribution is
/// picked from `generator`. For [`BulkDataGenerator::NonGaussianFactor`], the
/// factor is drawn from Laplace(0, b) via inverse-CDF sampling — rand_distr
/// 0.5 has no Laplace distribution so we synthesise it inline.
///
/// ### Params
///
/// * `num_modules` - Number of modules (rows).
/// * `num_samples` - Samples per module (columns).
/// * `generator` - Which factor distribution to draw from.
/// * `rng` - Seeded RNG.
///
/// ### Returns
///
/// One inner `Vec<f64>` per module of length `num_samples`, or a distribution
/// construction error propagated as a string.
fn draw_factors<T: BixverseFloat>(
    num_modules: usize,
    num_samples: usize,
    generator: &BulkDataGenerator<T>,
    rng: &mut StdRng,
) -> Result<Vec<Vec<f64>>, String> {
    let mut out = Vec::with_capacity(num_modules);
    match generator {
        BulkDataGenerator::Modular(c) => {
            let normal = Normal::new(0.0, c.factor_std.to_f64().unwrap())
                .map_err(|e| format!("normal init: {e}"))?;
            for _ in 0..num_modules {
                out.push((0..num_samples).map(|_| normal.sample(rng)).collect());
            }
        }
        BulkDataGenerator::HubModular(c) => {
            let normal = Normal::new(0.0, c.factor_std.to_f64().unwrap())
                .map_err(|e| format!("normal init: {e}"))?;
            for _ in 0..num_modules {
                out.push((0..num_samples).map(|_| normal.sample(rng)).collect());
            }
        }
        BulkDataGenerator::NonNegativeFactor(c) => {
            let gamma = Gamma::new(
                c.factor_shape.to_f64().unwrap(),
                c.factor_scale.to_f64().unwrap(),
            )
            .map_err(|e| format!("gamma(factor) init: {e}"))?;
            for _ in 0..num_modules {
                out.push((0..num_samples).map(|_| gamma.sample(rng)).collect());
            }
        }
        BulkDataGenerator::NonGaussianFactor(c) => {
            let b = c.factor_scale.to_f64().unwrap();
            for _ in 0..num_modules {
                out.push((0..num_samples).map(|_| laplace_sample(0.0, b, rng)).collect());
            }
        }
    }
    Ok(out)
}

/// Sample one draw from Laplace(mu, b) via inverse-CDF. Symmetric heavy-tailed,
/// std = `b * sqrt(2)`.
///
/// ### Params
///
/// * `mu` - Location.
/// * `b` - Scale.
/// * `rng` - RNG.
///
/// ### Returns
///
/// A single f64 sample.
#[inline]
fn laplace_sample(mu: f64, b: f64, rng: &mut StdRng) -> f64 {
    // U in (-0.5, 0.5). Guard the endpoints so ln(0) can't occur.
    let mut u: f64 = rng.random::<f64>() - 0.5;
    if u == -0.5 {
        u = -0.5 + f64::EPSILON;
    }
    let sign = if u >= 0.0 { -1.0 } else { 1.0 };
    mu + sign * b * (1.0 - 2.0 * u.abs()).ln()
}

/// Draw per-gene loadings under the requested generator variant. Zero for
/// background genes.
///
/// ### Params
///
/// * `gene_modules` - Length `num_genes`; `0` means background.
/// * `num_genes` - Total number of genes.
/// * `generator` - Picks Beta (Modular) vs LogNormal (all other variants).
/// * `rng` - Seeded RNG.
///
/// ### Returns
///
/// Loadings vector of length `num_genes`, or
/// [`BixverseErrors::BetaDistribution`] / [`BixverseErrors::InvalidArgument`]
/// on distribution construction failure.
fn draw_loadings<T: BixverseFloat>(
    gene_modules: &[usize],
    num_genes: usize,
    generator: &BulkDataGenerator<T>,
    rng: &mut StdRng,
) -> Result<Vec<f64>, BixverseErrors> {
    let mut loadings = vec![0.0f64; num_genes];
    match generator {
        BulkDataGenerator::Modular(_) => {
            let beta =
                Beta::new(5.0, 2.0).map_err(|e| BixverseErrors::BetaDistribution(e.to_string()))?;
            for (i, &m) in gene_modules.iter().enumerate() {
                if m > 0 {
                    loadings[i] = beta.sample(rng);
                }
            }
        }
        _ => {
            let l = generator.loading().expect("non-Modular has loading");
            let ln = LogNormal::new(l.mu.to_f64().unwrap(), l.sigma.to_f64().unwrap())
                .map_err(|e| BixverseErrors::InvalidArgument(format!("lognormal init: {e}")))?;
            for (i, &m) in gene_modules.iter().enumerate() {
                if m > 0 {
                    loadings[i] = ln.sample(rng);
                }
            }
        }
    }
    Ok(loadings)
}

/// Flag the top-fraction of module-gene loadings as hubs.
///
/// Percentile is computed globally across all module genes (not per module),
/// so a module with unusually large loadings will get proportionally more
/// hubs, which matches the intuition that some modules are hub-dense and
/// others aren't.
///
/// ### Params
///
/// * `gene_modules` - Length `num_genes`; `0` means background.
/// * `loadings` - Per-gene loading values.
/// * `hub_percentile` - Fraction in `(0, 1]`.
///
/// ### Returns
///
/// Sorted indices of the hub genes. Empty if there are no module genes.
fn identify_hubs(gene_modules: &[usize], loadings: &[f64], hub_percentile: f64) -> Vec<usize> {
    let mut module_idx: Vec<usize> = gene_modules
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m > 0 { Some(i) } else { None })
        .collect();
    if module_idx.is_empty() {
        return Vec::new();
    }
    module_idx.sort_by(|&a, &b| {
        loadings[b]
            .partial_cmp(&loadings[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = ((module_idx.len() as f64 * hub_percentile).ceil() as usize).max(1);
    let mut hubs: Vec<usize> = module_idx.into_iter().take(k).collect();
    hubs.sort_unstable();
    hubs
}

/// Sample from `NegBin(r, mu)` via a Gamma-Poisson mixture.
///
/// `r` is the NB size, `mu` the mean. Returns 0 when `mu` is small enough
/// that the Gamma scale would underflow (protects against zero-mean latent
/// signal after log-scale noise pushes into the extreme left tail) and when
/// either distribution constructor otherwise rejects — the caller doesn't
/// need to distinguish "mean was zero" from "sampler said no".
///
/// ### Params
///
/// * `r` - NB size (inverse dispersion).
/// * `mu` - NB mean.
/// * `rng` - RNG.
///
/// ### Returns
///
/// A single non-negative count as `f64` (integer-valued in practice).
#[inline]
fn nb_sample(r: f64, mu: f64, rng: &mut StdRng) -> f64 {
    // rand_distr Gamma rejects scale <= f64::MIN_POSITIVE. scale = (1-p)/p = mu/r,
    // so mu < r * MIN_POSITIVE is unusable. Also protect against mu near 0
    // producing a Poisson lambda that Poisson::new would reject.
    let min_mu = r * f64::MIN_POSITIVE.max(1e-300) * 1e6;
    if mu <= min_mu || !mu.is_finite() {
        return 0.0;
    }
    let p = r / (r + mu);
    let scale = (1.0 - p) / p;
    let gamma = match Gamma::new(r, scale) {
        Ok(g) => g,
        Err(_) => return 0.0,
    };
    let lambda = gamma.sample(rng);
    if !(lambda > 0.0 && lambda.is_finite()) {
        return 0.0;
    }
    match Poisson::new(lambda) {
        Ok(p) => p.sample(rng),
        Err(_) => 0.0,
    }
}

////////////////////
// Sparsification //
////////////////////

/// Apply a dropout / sparsification strategy to a dense count matrix.
///
/// ### Params
///
/// * `counts` - Input count matrix, shape `(n_genes, n_samples)`. Values are
///   read as f64 counts; non-integer inputs are floored inside the binomial
///   thinning step.
/// * `params` - See [`SparsityParams`].
///
/// ### Returns
///
/// A new count matrix with the same shape as `counts`, sparsified per the
/// strategy. Errors propagate through [`BixverseErrors::InvalidArgument`].
///
/// ### References
///
/// Zappia et al. 2017 (Splatter). Per-sample size factors from a LogNormal
/// with a target library size, then binomial thinning per gene.
pub fn apply_dropout<T>(
    counts: MatRef<T>,
    params: &SparsityParams<T>,
) -> Result<Mat<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    params.validate()?;
    match params.strategy {
        DropoutStrategy::SequencingDepth => apply_dropout_seq_depth(counts, params),
    }
}

/// Splatter-style sequencing-depth dropout.
///
/// Per sample j, draw a size factor `s_j ~ LogNormal(0, sigma_s)`, compute a
/// target library `L_j = target * s_j`, and binomially thin each gene's
/// count so the column library approaches `L_j`. Retention probability is
/// capped at 1 — when the target exceeds the current library, no counts are
/// added (we don't upsample).
///
/// ### Params
///
/// * `counts` - Input count matrix, shape `(n_genes, n_samples)`.
/// * `params` - See [`SparsityParams`].
///
/// ### Returns
///
/// A new count matrix, same shape as `counts`.
fn apply_dropout_seq_depth<T>(
    counts: MatRef<T>,
    params: &SparsityParams<T>,
) -> Result<Mat<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let (n_genes, n_samples) = counts.shape();
    let seed = params.seed;
    let target_lib = params.target_library_size.to_f64().unwrap();
    let sigma_s = params.capture_efficiency_sigma.to_f64().unwrap();

    // per-sample size factor and current library
    let mut rng = StdRng::seed_from_u64(seed);
    let ln = LogNormal::new(0.0, sigma_s)
        .map_err(|e| BixverseErrors::InvalidArgument(format!("lognormal(seq_depth) init: {e}")))?;
    let size_factors: Vec<f64> = (0..n_samples).map(|_| ln.sample(&mut rng)).collect();
    let current_lib: Vec<f64> = (0..n_samples)
        .map(|j| {
            (0..n_genes)
                .map(|i| counts.get(i, j).to_f64().unwrap())
                .sum()
        })
        .collect();

    // per-sample retention probability, capped at 1.0
    let retention: Vec<f64> = (0..n_samples)
        .map(|j| {
            let target_j = target_lib * size_factors[j];
            if current_lib[j] > 0.0 {
                (target_j / current_lib[j]).min(1.0)
            } else {
                0.0
            }
        })
        .collect();

    // parallel over samples: per-column binomial thinning. Columns are the
    // faer-native storage axis so this is cache-friendly.
    let per_sample: Vec<Vec<T>> = (0..n_samples)
        .into_par_iter()
        .map(|j| {
            let mut local_rng = StdRng::seed_from_u64(seed.wrapping_add((j as u64) * 1000 + 7));
            let p = retention[j];
            let mut col = Vec::with_capacity(n_genes);
            for i in 0..n_genes {
                let c = counts.get(i, j).to_f64().unwrap();
                let n = c.max(0.0).floor() as u64;
                let out = if n == 0 || p <= 0.0 {
                    0.0
                } else if p >= 1.0 {
                    n as f64
                } else {
                    let b = Binomial::new(n, p).unwrap();
                    b.sample(&mut local_rng) as f64
                };
                col.push(T::from_f64(out).unwrap());
            }
            col
        })
        .collect();

    let mut out: Mat<T> = Mat::zeros(n_genes, n_samples);
    for (j, col) in per_sample.into_iter().enumerate() {
        for (i, v) in col.into_iter().enumerate() {
            out[(i, j)] = v;
        }
    }
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -----------------------------------------------------

    /// log1p on every entry.
    fn log1p_mat(counts: &Mat<f64>) -> Mat<f64> {
        let (r, c) = counts.shape();
        let mut out = Mat::<f64>::zeros(r, c);
        for i in 0..r {
            for j in 0..c {
                out[(i, j)] = (counts[(i, j)] + 1.0).ln();
            }
        }
        out
    }

    /// Pearson correlation on two equal-length slices.
    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for (a, b) in x.iter().zip(y.iter()) {
            let da = *a - mx;
            let db = *b - my;
            num += da * db;
            sx += da * da;
            sy += db * db;
        }
        let denom = (sx * sy).sqrt();
        if denom > 0.0 { num / denom } else { 0.0 }
    }

    /// Full gene x gene pearson correlation on a `(n_genes, n_samples)` matrix.
    fn gene_gene_corr(log_mat: &Mat<f64>) -> Mat<f64> {
        let (n, s) = log_mat.shape();
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            rows.push((0..s).map(|j| log_mat[(i, j)]).collect());
        }
        let mut out = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let r = pearson(&rows[i], &rows[j]);
                out[(i, j)] = r;
                out[(j, i)] = r;
            }
        }
        out
    }

    /// Extract row `i` of a `Mat<f64>` as a `Vec<f64>`.
    fn take_row(m: &Mat<f64>, i: usize) -> Vec<f64> {
        (0..m.ncols()).map(|j| m[(i, j)]).collect()
    }

    /// Excess kurtosis of a slice. Normal distribution → 0.
    fn excess_kurtosis(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let m = x.iter().sum::<f64>() / n;
        let mut m2 = 0.0;
        let mut m4 = 0.0;
        for &v in x {
            let d = v - m;
            m2 += d * d;
            m4 += d * d * d * d;
        }
        m2 /= n;
        m4 /= n;
        if m2 > 0.0 { m4 / (m2 * m2) - 3.0 } else { 0.0 }
    }

    fn hub_modular_default() -> BulkDataGenerator<f64> {
        BulkDataGenerator::HubModular(HubModularConfig::default())
    }

    // ---- basic dimensions --------------------------------------------

    #[test]
    fn dims_no_modules() {
        let params = SyntheticRnaSeqParams::<f64> {
            num_samples: 10,
            num_genes: 50,
            module_sizes: vec![],
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&params).unwrap();
        assert_eq!(data.count_matrix.nrows(), 50);
        assert_eq!(data.count_matrix.ncols(), 10);
        assert_eq!(data.gene_modules.len(), 50);
        assert!(data.gene_modules.iter().all(|&m| m == 0));
        assert!(data.module_hubs.is_empty());
        assert_eq!(data.module_factors.nrows(), 0);
        assert_eq!(data.module_factors.ncols(), 10);
    }

    #[test]
    fn dims_with_modules() {
        let params = SyntheticRnaSeqParams::<f64> {
            num_samples: 20,
            num_genes: 100,
            module_sizes: vec![20, 30],
            generator: hub_modular_default(),
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&params).unwrap();
        assert_eq!(data.count_matrix.nrows(), 100);
        let counts_by_id = |k: usize| data.gene_modules.iter().filter(|&&m| m == k).count();
        assert_eq!(counts_by_id(1), 20);
        assert_eq!(counts_by_id(2), 30);
        assert_eq!(counts_by_id(0), 50);
        assert_eq!(data.module_factors.nrows(), 2);
        assert_eq!(data.module_factors.ncols(), 20);
        assert!(!data.module_hubs.is_empty());
        assert!(data.module_hubs.iter().all(|&i| data.gene_modules[i] > 0));
    }

    // ---- validation --------------------------------------------------

    #[test]
    fn invalid_module_sizes_rejected() {
        let params = SyntheticRnaSeqParams::<f64> {
            num_samples: 5,
            num_genes: 10,
            module_sizes: vec![20],
            ..Default::default()
        };
        assert!(matches!(
            generate_bulk_rnaseq(&params),
            Err(BixverseErrors::InvalidArgument(_))
        ));
    }

    // ---- correlation structure --------------------------------------

    #[test]
    fn within_module_corr_stronger_than_cross() {
        let params = SyntheticRnaSeqParams::<f64> {
            num_samples: 200,
            num_genes: 110,
            seed: 7,
            module_sizes: vec![30, 30],
            generator: hub_modular_default(),
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&params).unwrap();
        let log = log1p_mat(&data.count_matrix);
        let corr = gene_gene_corr(&log);
        let mods = &data.gene_modules;

        let (mut in_sum, mut in_n) = (0.0, 0usize);
        let (mut xr_sum, mut xr_n) = (0.0, 0usize);
        for i in 0..mods.len() {
            for j in (i + 1)..mods.len() {
                if mods[i] == 0 || mods[j] == 0 {
                    continue;
                }
                if mods[i] == mods[j] {
                    in_sum += corr[(i, j)];
                    in_n += 1;
                } else {
                    xr_sum += corr[(i, j)];
                    xr_n += 1;
                }
            }
        }
        let in_mean = in_sum / in_n as f64;
        let cross_mean = xr_sum / xr_n as f64;
        assert!(
            in_mean - cross_mean > 0.15,
            "within-module mean {in_mean} - cross-module mean {cross_mean} < 0.15"
        );
    }

    #[test]
    fn hubs_have_higher_degree_than_non_hubs() {
        let params = SyntheticRnaSeqParams::<f64> {
            num_samples: 200,
            num_genes: 100,
            seed: 11,
            module_sizes: vec![60],
            generator: hub_modular_default(),
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&params).unwrap();
        let log = log1p_mat(&data.count_matrix);
        let corr = gene_gene_corr(&log);
        let n = corr.nrows();

        let threshold = 0.5;
        let mut degrees = vec![0usize; n];
        for i in 0..n {
            for j in 0..n {
                if i != j && corr[(i, j)].abs() >= threshold {
                    degrees[i] += 1;
                }
            }
        }

        let hubs: std::collections::HashSet<usize> = data.module_hubs.iter().copied().collect();
        let mut hub_sum = 0.0;
        let mut hub_n = 0.0;
        let mut non_hub_sum = 0.0;
        let mut non_hub_n = 0.0;
        for (i, &m) in data.gene_modules.iter().enumerate() {
            if m == 0 {
                continue;
            }
            if hubs.contains(&i) {
                hub_sum += degrees[i] as f64;
                hub_n += 1.0;
            } else {
                non_hub_sum += degrees[i] as f64;
                non_hub_n += 1.0;
            }
        }
        let hub_mean = hub_sum / hub_n;
        let non_hub_mean = non_hub_sum / non_hub_n;
        assert!(
            hub_mean > non_hub_mean,
            "hub mean degree {hub_mean} not > non-hub {non_hub_mean}"
        );
    }

    // ---- reproducibility --------------------------------------------

    #[test]
    fn same_seed_same_output() {
        let params = SyntheticRnaSeqParams::<f64> {
            num_samples: 12,
            num_genes: 40,
            seed: 999,
            module_sizes: vec![10, 10],
            ..Default::default()
        };
        let a = generate_bulk_rnaseq(&params).unwrap();
        let b = generate_bulk_rnaseq(&params).unwrap();
        for i in 0..a.count_matrix.nrows() {
            for j in 0..a.count_matrix.ncols() {
                assert_eq!(a.count_matrix[(i, j)], b.count_matrix[(i, j)]);
            }
        }
        for m in 0..a.module_factors.nrows() {
            for j in 0..a.module_factors.ncols() {
                assert_eq!(a.module_factors[(m, j)], b.module_factors[(m, j)]);
            }
        }
    }

    // ---- module factor round-trip ------------------------------------

    #[test]
    fn module_pc1_tracks_ground_truth_factor() {
        let params = SyntheticRnaSeqParams::<f64> {
            num_samples: 150,
            num_genes: 60,
            seed: 21,
            module_sizes: vec![60],
            generator: hub_modular_default(),
            noise_std: 0.3,
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&params).unwrap();
        let log = log1p_mat(&data.count_matrix);
        let (n, s) = log.shape();
        let mut centered = Mat::<f64>::zeros(n, s);
        for i in 0..n {
            let mut m = 0.0;
            for j in 0..s {
                m += log[(i, j)];
            }
            m /= s as f64;
            for j in 0..s {
                centered[(i, j)] = log[(i, j)] - m;
            }
        }
        let svd = centered.thin_svd().expect("thin_svd");
        let v = svd.V();
        let pc1: Vec<f64> = (0..s).map(|j| v[(j, 0)]).collect();
        let factor = take_row(
            &Mat::<f64>::from_fn(1, s, |_, j| data.module_factors[(0, j)]),
            0,
        );
        let r = pearson(&pc1, &factor).abs();
        assert!(r > 0.7, "PC1 vs ground-truth factor |r|={r} <= 0.7");
    }

    // ---- non-negative factor variant ---------------------------------

    #[test]
    fn non_negative_factor_is_non_negative() {
        let params = SyntheticRnaSeqParams::<f64> {
            num_samples: 100,
            num_genes: 60,
            seed: 15,
            module_sizes: vec![30],
            generator: BulkDataGenerator::NonNegativeFactor(NonNegativeFactorConfig::default()),
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&params).unwrap();
        for m in 0..data.module_factors.nrows() {
            for j in 0..data.module_factors.ncols() {
                assert!(
                    data.module_factors[(m, j)] >= 0.0,
                    "factor {} at (m={m}, j={j}) is negative",
                    data.module_factors[(m, j)]
                );
            }
        }
        // sanity: recovery still works
        let log = log1p_mat(&data.count_matrix);
        let corr = gene_gene_corr(&log);
        let mods = &data.gene_modules;
        let (mut in_sum, mut in_n) = (0.0, 0usize);
        for i in 0..mods.len() {
            for j in (i + 1)..mods.len() {
                if mods[i] > 0 && mods[i] == mods[j] {
                    in_sum += corr[(i, j)];
                    in_n += 1;
                }
            }
        }
        assert!(in_sum / in_n as f64 > 0.1, "non-negative variant not producing correlation");
    }

    // ---- non-Gaussian factor variant ---------------------------------

    #[test]
    fn non_gaussian_factor_is_non_gaussian() {
        let params = SyntheticRnaSeqParams::<f64> {
            num_samples: 500,
            num_genes: 60,
            seed: 23,
            module_sizes: vec![30],
            generator: BulkDataGenerator::NonGaussianFactor(NonGaussianFactorConfig::default()),
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&params).unwrap();
        let f: Vec<f64> = (0..data.module_factors.ncols())
            .map(|j| data.module_factors[(0, j)])
            .collect();
        // Laplace excess kurtosis is 3.
        let ex_k = excess_kurtosis(&f);
        assert!(
            ex_k > 1.0,
            "excess kurtosis {ex_k} not clearly non-Gaussian (Laplace expected ~3)"
        );
    }

    // ---- sparsification ---------------------------------------------

    #[test]
    fn dropout_reduces_total_counts() {
        let base = SyntheticRnaSeqParams::<f64> {
            num_samples: 30,
            num_genes: 100,
            seed: 3,
            module_sizes: vec![],
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&base).unwrap();
        let params = SparsityParams::<f64> {
            strategy: DropoutStrategy::SequencingDepth,
            target_library_size: 100.0,
            capture_efficiency_sigma: 0.5,
            seed: 4,
        };
        let dropped = apply_dropout(data.count_matrix.as_ref(), &params).unwrap();
        let mut sum_before = 0.0;
        for i in 0..data.count_matrix.nrows() {
            for j in 0..data.count_matrix.ncols() {
                sum_before += data.count_matrix[(i, j)];
            }
        }
        let mut sum_after = 0.0;
        for i in 0..dropped.nrows() {
            for j in 0..dropped.ncols() {
                sum_after += dropped[(i, j)];
            }
        }
        assert!(
            sum_after < sum_before,
            "sum_after {sum_after} >= sum_before {sum_before}"
        );
    }

    #[test]
    fn dropout_preserves_within_module_correlation_sign() {
        let base = SyntheticRnaSeqParams::<f64> {
            num_samples: 200,
            num_genes: 80,
            seed: 5,
            module_sizes: vec![40],
            generator: hub_modular_default(),
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&base).unwrap();
        let params = SparsityParams::<f64> {
            strategy: DropoutStrategy::SequencingDepth,
            target_library_size: 1_000.0,
            capture_efficiency_sigma: 0.2,
            seed: 6,
        };
        let dropped = apply_dropout(data.count_matrix.as_ref(), &params).unwrap();
        let log = log1p_mat(&dropped);
        let corr = gene_gene_corr(&log);
        let mods = &data.gene_modules;
        let (mut in_sum, mut in_n) = (0.0, 0usize);
        for i in 0..mods.len() {
            for j in (i + 1)..mods.len() {
                if mods[i] > 0 && mods[i] == mods[j] {
                    in_sum += corr[(i, j)];
                    in_n += 1;
                }
            }
        }
        let mean = in_sum / in_n as f64;
        assert!(mean > 0.0, "within-module mean corr {mean} not positive after dropout");
    }

    #[test]
    fn dropout_reproducible() {
        let base = SyntheticRnaSeqParams::<f64> {
            num_samples: 15,
            num_genes: 40,
            seed: 8,
            module_sizes: vec![],
            ..Default::default()
        };
        let data = generate_bulk_rnaseq(&base).unwrap();
        let params = SparsityParams::<f64> {
            strategy: DropoutStrategy::SequencingDepth,
            target_library_size: 500.0,
            capture_efficiency_sigma: 0.5,
            seed: 99,
        };
        let a = apply_dropout(data.count_matrix.as_ref(), &params).unwrap();
        let b = apply_dropout(data.count_matrix.as_ref(), &params).unwrap();
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_eq!(a[(i, j)], b[(i, j)]);
            }
        }
    }

    // ---- parse helpers ----------------------------------------------

    #[test]
    fn parse_helpers_work() {
        assert!(matches!(
            parse_bulk_data_generator::<f64>("modular"),
            Some(BulkDataGenerator::Modular(_))
        ));
        assert!(matches!(
            parse_bulk_data_generator::<f64>("Hub_Modular"),
            Some(BulkDataGenerator::HubModular(_))
        ));
        assert!(matches!(
            parse_bulk_data_generator::<f64>("nmf"),
            Some(BulkDataGenerator::NonNegativeFactor(_))
        ));
        assert!(matches!(
            parse_bulk_data_generator::<f64>("ica"),
            Some(BulkDataGenerator::NonGaussianFactor(_))
        ));
        assert!(parse_bulk_data_generator::<f64>("nope").is_none());
        assert_eq!(
            parse_dropout_strategy("seq_depth"),
            Some(DropoutStrategy::SequencingDepth)
        );
        assert_eq!(parse_dropout_strategy("nope"), None);
    }
}
