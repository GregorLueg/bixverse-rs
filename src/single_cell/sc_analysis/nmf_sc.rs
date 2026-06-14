//! Single-cell version of NMF. Leverages the sparse implementation of the
//! HALS version.

use crate::methods::nmf_hals::nmf_preprocessing::*;
use crate::methods::nmf_hals::sparse::*;
use crate::methods::nmf_hals::*;
use crate::prelude::*;

/////////////
// Workers //
/////////////

/////////////////////////
// Single run (sparse) //
/////////////////////////

use crate::prelude::CompressedSparseData2;

/// Run NMF (single run)
///
/// This version runs NMF on sparse inputs (think single cell or spatial
/// transcriptomics experiments). Assumes samples x features.
///
/// ### Params
///
/// * `data` - The data on which to run a single NMF run. Assumes samples x
///   features and [CompressedSparseFormat::Csc] in terms of storage format.
/// * `k` - Number of k to return
/// * `use_second_layer` - Shall the layer in data_2 in the
///   [CompressedSparseData2] be used. Needs to be filled out or will error.
/// * `preprocessing` - String. Shall additional pre-processing be applied,
///   see [parse_nmf_processing].
/// * `nmf_hals_params` - Optional [HalsOpts] parameters.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// [NmfResult] or errors.
pub fn nmf_single_run_sc<T>(
    data: CompressedSparseData2<T>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<T>>,
    verbose: usize,
) -> Result<NmfResult<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseNumeric + std::iter::Sum<T> + BixverseSimd,
{
    let nmf_preprocessing = parse_nmf_processing(preprocessing).unwrap_or_else(|| {
        println!(
            "Unknown string provided: {:?}. Using the default (No additional pre-processing)",
            preprocessing
        );
        NmfPreprocessing::default()
    });

    let hals_opt = nmf_hals_params.unwrap_or_default();

    let data_processed = nmf_process_sparse(&data, &nmf_preprocessing, use_second_layer)?;

    let nmf_input: SparseInput<T, T> = if use_second_layer {
        SparseInput::from_secondary(&data_processed)
    } else {
        SparseInput::from_primary(&data_processed)
    }?;

    let nmf_res = nmf_hals(&nmf_input, k, &hals_opt, verbose)?;

    Ok(nmf_res)
}

//////////////////////////////////
// Multiple runs (single cells) //
//////////////////////////////////

/// Run NMF (multiple run)
///
/// This version runs NMF on sparse inputs (think single cell or spatial
/// transcriptomics experiments) across a set of random initialisations and
/// returns the results across these runs. Assumes samples x features.
///
/// ### Params
///
/// * `data` - The data on which to run a single NMF run. Assumes samples x
///   features and [CompressedSparseFormat::Csc] in terms of storage format.
/// * `k` - Number of k to return
/// * `use_second_layer` - Shall the layer in data_2 in the
///   [CompressedSparseData2] be used. Needs to be filled out or will error.
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
#[allow(clippy::too_many_arguments)]
pub fn nmf_multiple_run_sc<T>(
    data: CompressedSparseData2<T>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<T>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<StabilisedNmfResult<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseNumeric + std::iter::Sum<T> + BixverseSimd,
{
    let nmf_preprocessing = parse_nmf_processing(preprocessing).unwrap_or_else(|| {
        println!(
            "Unknown string provided: {:?}. Using the default (No additional pre-processing)",
            preprocessing
        );
        NmfPreprocessing::default()
    });

    let hals_opt = nmf_hals_params.unwrap_or_default();

    let data_processed = nmf_process_sparse(&data, &nmf_preprocessing, use_second_layer)?;

    let nmf_input: SparseInput<T, T> = if use_second_layer {
        SparseInput::from_secondary(&data_processed)
    } else {
        SparseInput::from_primary(&data_processed)
    }?;

    let res = stabilised_nmf(&nmf_input, k, n_runs, base_seed as u64, &hals_opt, verbose)?;

    Ok(res)
}

//////////////////////////////
// Single cells (from disk) //
//////////////////////////////
