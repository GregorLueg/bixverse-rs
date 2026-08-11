//! Specific R wrappers over structures or functions within the `core` module

use extendr_api::*;
use std::collections::{BTreeMap, HashMap};

use crate::core::mat_struct::NamedMatrix;
use crate::core::synthetic_data::{
    BulkDataGenerator, DropoutStrategy, HubModularConfig, LoadingLogNormal, ModularConfig,
    NonGaussianFactorConfig, NonNegativeFactorConfig, SparsityParams, SyntheticRnaSeqParams,
    parse_dropout_strategy,
};
use crate::prelude::*;

///////////////////////////////
// NamedMatrix from R matrix //
///////////////////////////////

impl<'a, T> NamedMatrix<'a, T>
where
    T: BixverseFloat,
{
    /// Generate a NamedMatrix from an RMatrix.
    pub fn from_r_matrix(x: &'a RMatrix<f64>) -> NamedMatrix<'a, f64> {
        let col_names: BTreeMap<String, usize> = x
            .get_colnames()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i))
            .collect();
        let row_names: BTreeMap<String, usize> = x
            .get_rownames()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i))
            .collect();
        let mat = r_matrix_to_faer(x);
        NamedMatrix {
            col_names,
            row_names,
            values: mat,
        }
    }
}

///////////////////////////////////
// Synthetic data - loading base //
///////////////////////////////////

impl<T: BixverseFloat> LoadingLogNormal<T> {
    /// Read the LogNormal loading sub-config from a flat R list.
    ///
    /// Keys: `loading_mu`, `loading_sigma`, `hub_percentile`. Missing keys
    /// fall back to [`LoadingLogNormal::default()`].
    ///
    /// ### Params
    ///
    /// * `params` - Parsed R list contents (already flattened to a HashMap).
    ///
    /// ### Returns
    ///
    /// A [`LoadingLogNormal`] populated with the provided or default values.
    pub fn from_r_map(params: &HashMap<&str, Robj>) -> LoadingLogNormal<T> {
        let defaults: LoadingLogNormal<T> = LoadingLogNormal::default();

        let mu = params
            .get("loading_mu")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.mu);

        let sigma = params
            .get("loading_sigma")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.sigma);

        let hub_percentile = params
            .get("hub_percentile")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.hub_percentile);

        LoadingLogNormal {
            mu,
            sigma,
            hub_percentile,
        }
    }
}

/////////////////////////////////
// Synthetic data - generators //
/////////////////////////////////

impl<T: BixverseFloat> ModularConfig<T> {
    /// Read [`ModularConfig`] from an R list.
    ///
    /// Keys: `factor_std`. Missing keys fall back to [`ModularConfig::default`].
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list containing the parameters.
    ///
    /// ### Returns
    ///
    /// The [`ModularConfig`] populated with provided or default values.
    pub fn from_r_list(r_list: List) -> Result<ModularConfig<T>> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults: ModularConfig<T> = ModularConfig::default();

        let factor_std = params
            .get("factor_std")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.factor_std);

        Ok(ModularConfig { factor_std })
    }
}

impl<T: BixverseFloat> HubModularConfig<T> {
    /// Read [`HubModularConfig`] from an R list.
    ///
    /// Keys: `factor_std`, `loading_mu`, `loading_sigma`, `hub_percentile`.
    /// Missing keys fall back to [`HubModularConfig::default`].
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list containing the parameters.
    ///
    /// ### Returns
    ///
    /// The [`HubModularConfig`] populated with provided or default values.
    pub fn from_r_list(r_list: List) -> Result<HubModularConfig<T>> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults: HubModularConfig<T> = HubModularConfig::default();

        let factor_std = params
            .get("factor_std")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.factor_std);

        let loading = LoadingLogNormal::<T>::from_r_map(&params);

        Ok(HubModularConfig {
            factor_std,
            loading,
        })
    }
}

impl<T: BixverseFloat> NonNegativeFactorConfig<T> {
    /// Read [`NonNegativeFactorConfig`] from an R list.
    ///
    /// Keys: `factor_shape`, `factor_scale`, `loading_mu`, `loading_sigma`,
    /// `hub_percentile`. Missing keys fall back to
    /// [`NonNegativeFactorConfig::default`].
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list containing the parameters.
    ///
    /// ### Returns
    ///
    /// The [`NonNegativeFactorConfig`] populated with provided or default values.
    pub fn from_r_list(r_list: List) -> Result<NonNegativeFactorConfig<T>> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults: NonNegativeFactorConfig<T> = NonNegativeFactorConfig::default();

        let factor_shape = params
            .get("factor_shape")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.factor_shape);

        let factor_scale = params
            .get("factor_scale")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.factor_scale);

        let loading = LoadingLogNormal::<T>::from_r_map(&params);

        Ok(NonNegativeFactorConfig {
            factor_shape,
            factor_scale,
            loading,
        })
    }
}

impl<T: BixverseFloat> NonGaussianFactorConfig<T> {
    /// Read [`NonGaussianFactorConfig`] from an R list.
    ///
    /// Keys: `factor_scale`, `loading_mu`, `loading_sigma`, `hub_percentile`.
    /// Missing keys fall back to [`NonGaussianFactorConfig::default`].
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list containing the parameters.
    ///
    /// ### Returns
    ///
    /// The [`NonGaussianFactorConfig`] populated with provided or default values.
    pub fn from_r_list(r_list: List) -> Result<NonGaussianFactorConfig<T>> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults: NonGaussianFactorConfig<T> = NonGaussianFactorConfig::default();

        let factor_scale = params
            .get("factor_scale")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.factor_scale);

        let loading = LoadingLogNormal::<T>::from_r_map(&params);

        Ok(NonGaussianFactorConfig {
            factor_scale,
            loading,
        })
    }
}

//////////////////////////////////////
// Synthetic data - top-level params //
//////////////////////////////////////

impl<T: BixverseFloat> SyntheticRnaSeqParams<T> {
    /// Generate [`SyntheticRnaSeqParams`] from an R list.
    ///
    /// The list is expected to contain all top-level synthetic-data parameters
    /// plus the fields required by the chosen generator variant. Generator
    /// configuration parameters are read from the same flat list, following
    /// the SCENIC convention.
    ///
    /// The `generator` string picks the variant: `"modular"`, `"hub_modular"`,
    /// `"non_negative_factor"` (aliases: `"nonnegative"`, `"nmf"`), or
    /// `"non_gaussian_factor"` (aliases: `"nongaussian"`, `"ica"`). Missing
    /// values fall back to [`SyntheticRnaSeqParams::default`].
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list with the synthetic-data parameters.
    ///
    /// ### Returns
    ///
    /// The [`SyntheticRnaSeqParams`] with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<SyntheticRnaSeqParams<T>> {
        let defaults: SyntheticRnaSeqParams<T> = SyntheticRnaSeqParams::default();
        let params: HashMap<&str, Robj> = r_list_to_map(r_list.clone())?;

        let num_samples = params
            .get("num_samples")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.num_samples);

        let num_genes = params
            .get("num_genes")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.num_genes);

        let seed = params
            .get("seed")
            .and_then(|v| v.as_integer())
            .map(|v| v as u64)
            .unwrap_or(defaults.seed);

        let module_sizes = params
            .get("module_sizes")
            .and_then(|v| v.as_integer_slice())
            .map(|s| s.iter().map(|&x| x as usize).collect::<Vec<_>>())
            .unwrap_or(defaults.module_sizes);

        let mean_exp_gamma_shape = params
            .get("mean_exp_gamma_shape")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.mean_exp_gamma_shape);

        let mean_exp_gamma_scale = params
            .get("mean_exp_gamma_scale")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.mean_exp_gamma_scale);

        let disp_intercept = params
            .get("disp_intercept")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.disp_intercept);

        let disp_slope = params
            .get("disp_slope")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.disp_slope);

        let noise_std = params
            .get("noise_std")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.noise_std);

        let generator_kind = params
            .get("generator")
            .and_then(|v| v.as_str())
            .unwrap_or("hub_modular");

        let generator = match generator_kind.to_lowercase().as_str() {
            "modular" => BulkDataGenerator::Modular(ModularConfig::from_r_list(r_list.clone())?),
            "non_negative_factor" | "nonnegative" | "nmf" => BulkDataGenerator::NonNegativeFactor(
                NonNegativeFactorConfig::from_r_list(r_list.clone())?,
            ),
            "non_gaussian_factor" | "nongaussian" | "ica" => BulkDataGenerator::NonGaussianFactor(
                NonGaussianFactorConfig::from_r_list(r_list.clone())?,
            ),
            _ => BulkDataGenerator::HubModular(HubModularConfig::from_r_list(r_list.clone())?),
        };

        Ok(SyntheticRnaSeqParams {
            num_samples,
            num_genes,
            seed,
            module_sizes,
            generator,
            mean_exp_gamma_shape,
            mean_exp_gamma_scale,
            disp_intercept,
            disp_slope,
            noise_std,
        })
    }
}

////////////////////////////////////////
// Synthetic data - sparsity / dropout //
////////////////////////////////////////

impl<T: BixverseFloat> SparsityParams<T> {
    /// Generate [`SparsityParams`] from an R list.
    ///
    /// Keys: `strategy` (currently only `"seq_depth"` /
    /// `"sequencing_depth"` — falls back to
    /// [`DropoutStrategy::SequencingDepth`]), `target_library_size`,
    /// `capture_efficiency_sigma`, `seed`. Missing values fall back to
    /// [`SparsityParams::default`].
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list with the sparsity parameters.
    ///
    /// ### Returns
    ///
    /// The [`SparsityParams`] with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<SparsityParams<T>> {
        let defaults: SparsityParams<T> = SparsityParams::default();
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let strategy = params
            .get("strategy")
            .and_then(|v| v.as_str())
            .and_then(parse_dropout_strategy)
            .unwrap_or(DropoutStrategy::SequencingDepth);

        let target_library_size = params
            .get("target_library_size")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.target_library_size);

        let capture_efficiency_sigma = params
            .get("capture_efficiency_sigma")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.capture_efficiency_sigma);

        let seed = params
            .get("seed")
            .and_then(|v| v.as_integer())
            .map(|v| v as u64)
            .unwrap_or(defaults.seed);

        Ok(SparsityParams {
            strategy,
            target_library_size,
            capture_efficiency_sigma,
            seed,
        })
    }
}

/////////////////////////
// Lanczos eigensolver //
/////////////////////////

impl LanczosParams {
    /// Generate LanczosParams from a flattened R list
    ///
    /// Takes the already-flattened map rather than a `List`, so it can be nested
    /// inside the parameter block of whichever method owns the eigensolver.
    /// Anything missing falls back to [LanczosParams::default], so the R and
    /// Rust defaults cannot drift apart.
    ///
    /// ### Params
    ///
    /// * `params` - Parsed R list contents, already flattened to a HashMap.
    ///
    /// ### Returns
    ///
    /// A [LanczosParams] populated with the provided or default values, or an
    /// error when a count is present but unusable.
    pub fn from_r_list(params: &HashMap<&str, Robj>) -> Result<LanczosParams> {
        let defaults = LanczosParams::default();

        // `None` derives the basis from the requested component count.
        // `r_list_count` is load bearing here: `NA_integer_` is `i32::MIN` and a
        // plain `as usize` turns it, or any negative, into ~1.8e19, which for
        // the restart budget is an unbounded loop rather than an error.
        let basis_size = r_list_count(params, "lanczos_basis_size")?;

        let max_restarts =
            r_list_count(params, "lanczos_max_restarts")?.unwrap_or(defaults.max_restarts);

        let tol = params
            .get("lanczos_tol")
            .and_then(|v| v.as_real())
            .unwrap_or(defaults.tol);

        Ok(LanczosParams {
            basis_size,
            max_restarts,
            tol,
        })
    }
}
