//! Wrapper functions to run NMF on (likely) bulkRNAseq experiments of shape
//! samples x features

use faer::MatRef;

use ann_search_rs::prelude::AnnSearchFloat;

use crate::errors::BixverseErrors;
use crate::methods::nmf_hals::consensus::*;
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
    let hals_opt = nmf_hals_params.unwrap_or_default();

    let data_processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));

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
    let hals_opt = nmf_hals_params.unwrap_or_default();

    let data_processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));

    let nmf_input = DenseInput::new(data_processed.as_ref())?;

    let res = stabilised_nmf(&nmf_input, k, n_runs, base_seed as u64, &hals_opt, verbose)?;

    Ok(res)
}

///////////////
// Consensus //
///////////////

/// Run consensus NMF at a single k
///
/// Dense inputs (think bulk RNAseq experiments), assumes samples x features.
/// Runs `n_runs` random restarts, clusters the pooled components and refits the
/// partner factor against the consensus one. See [nmf_consensus] for the scale
/// convention of the returned factors, which depends on the
/// [ConsensusTarget].
///
/// ### Params
///
/// * `data` - The data to factorise. Assumes samples x features.
/// * `k` - Number of components. Must be at least 2.
/// * `preprocessing` - String. Shall additional pre-processing be applied,
///   see [parse_nmf_processing].
/// * `nmf_hals_params` - Optional [HalsOpts] parameters.
/// * `consensus_params` - Optional [ConsensusParams] parameters.
/// * `n_runs` - Number of restarts at this k. Must be at least 2.
/// * `base_seed` - The base seed.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// [ConsensusNmfResult] or errors.
#[allow(clippy::too_many_arguments)]
pub fn nmf_consensus_run<T>(
    data: MatRef<T>,
    k: usize,
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<T>>,
    consensus_params: Option<ConsensusParams<T>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<ConsensusNmfResult<T>, BixverseErrors>
where
    T: BixverseFloat + AnnSearchFloat,
{
    let hals_opt = nmf_hals_params.unwrap_or_default();
    let data_processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));
    let nmf_input = DenseInput::new(data_processed.as_ref())?;

    nmf_consensus(
        &nmf_input,
        k,
        n_runs,
        base_seed as u64,
        &hals_opt,
        consensus_params,
        verbose,
    )
}

/// Sweep k and report consensus stability against reconstruction error
///
/// Dense inputs (think bulk RNAseq experiments), assumes samples x features.
/// Returns diagnostics only, so a wide `k_range` stays cheap. Pick the k where
/// stability is high and the error curve has not yet flattened, then call
/// [nmf_consensus_run] there.
///
/// ### Params
///
/// * `data` - The data to factorise. Assumes samples x features.
/// * `k_range` - Ranks to evaluate. Must be non-empty, every entry at least 2.
/// * `preprocessing` - String. Shall additional pre-processing be applied,
///   see [parse_nmf_processing].
/// * `nmf_hals_params` - Optional [HalsOpts] parameters.
/// * `consensus_params` - Optional [ConsensusParams] parameters.
/// * `n_runs` - Number of restarts per k. Must be at least 2.
/// * `base_seed` - The base seed.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// One [KSweepEntry] per entry of `k_range`, or errors.
#[allow(clippy::too_many_arguments)]
pub fn nmf_k_sweep_run<T>(
    data: MatRef<T>,
    k_range: &[usize],
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<T>>,
    consensus_params: Option<ConsensusParams<T>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<Vec<KSweepEntry<T>>, BixverseErrors>
where
    T: BixverseFloat + AnnSearchFloat,
{
    let hals_opt = nmf_hals_params.unwrap_or_default();
    let data_processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));
    let nmf_input = DenseInput::new(data_processed.as_ref())?;

    nmf_k_sweep(
        &nmf_input,
        k_range,
        n_runs,
        base_seed as u64,
        &hals_opt,
        consensus_params,
        verbose,
    )
}

/////////////
// Helpers //
/////////////

/// Parse the pre-processing string, warning and defaulting on unknown input.
///
/// ### Params
///
/// * `preprocessing` - String to parse, see [parse_nmf_processing].
///
/// ### Returns
///
/// The [NmfPreprocessing] to apply.
fn resolve_preprocessing(preprocessing: &str) -> NmfPreprocessing {
    parse_nmf_processing(preprocessing).unwrap_or_else(|| {
        println!(
            "Unknown string provided: {:?}. Using the default (No additional pre-processing)",
            preprocessing
        );
        NmfPreprocessing::default()
    })
}
