//! Wrapper functions to run NMF on (likely) bulkRNAseq experiments of shape
//! samples x features

use faer::MatRef;

use crate::errors::BixverseErrors;
use crate::methods::nmf_hals::dense::DenseInput;
use crate::methods::nmf_hals::nmf_preprocessing::*;
use crate::methods::nmf_hals::*;
use crate::prelude::BixverseFloat;

////////////////
// Single run //
////////////////

/// Run NMF (single run)
///
/// This version runs NMF on dense inputs (think bulk RNAseq experiments).
/// Assumes samples x features.
///
/// ### Params
///
/// * `data` - The data on which to run a single NMF run. Assumes samples x
///   features.
/// * `k` - Number of k to return
/// * `preprocessing` - String. Shall additional pre-processing be applied,
///   see [parse_nmf_processing].
/// * `nmf_hals_params` - Optional [HalsOpts] parameters.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// [NmfResult] or errors.
pub fn nmf_single_run<T>(
    data: MatRef<T>,
    k: usize,
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<T>>,
    verbose: usize,
) -> Result<NmfResult<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let nmf_preprocessing = parse_nmf_processing(preprocessing).unwrap_or_else(|| {
        println!(
            "Unknown string provided: {:?}. Using the default (No additional pre-processing)",
            preprocessing
        );
        NmfPreprocessing::default()
    });

    let hals_opt = nmf_hals_params.unwrap_or_default();

    let data_processed = nmf_process_dense(data, &nmf_preprocessing);

    let nmf_input = DenseInput::new(data_processed.as_ref())?;

    let nmf_res = nmf_hals(&nmf_input, k, &hals_opt, verbose)?;

    Ok(nmf_res)
}

///////////////////
// Multiple runs //
///////////////////

/// Run NMF (multiple runs)
///
/// This version runs NMF on dense inputs (think bulk RNAseq experiments)
/// across a set of random initialisations and returns the results across these
/// runs. Assumes samples x features.
///
/// ### Params
///
/// * `data` - The data on which to run a single NMF run. Assumes samples x
///   features.
/// * `k` - Number of k to return
/// * `preprocessing` - String. Shall additional pre-processing be applied,
///   see [parse_nmf_processing].
/// * `nmf_hals_params` - Optional [HalsOpts] parameters.
/// * `n_runs` - Number of runs at this k
/// * `base_seed` - The base seed.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// [StabilisedNmfResult] or errors.
pub fn nmf_multiple_run<T>(
    data: MatRef<T>,
    k: usize,
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<T>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<StabilisedNmfResult<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let nmf_preprocessing = parse_nmf_processing(preprocessing).unwrap_or_else(|| {
        println!(
            "Unknown string provided: {:?}. Using the default (No additional pre-processing)",
            preprocessing
        );
        NmfPreprocessing::default()
    });

    let hals_opt = nmf_hals_params.unwrap_or_default();

    let data_processed = nmf_process_dense(data, &nmf_preprocessing);

    let nmf_input = DenseInput::new(data_processed.as_ref())?;

    let res = stabilised_nmf(&nmf_input, k, n_runs, base_seed as u64, &hals_opt, verbose)?;

    Ok(res)
}
