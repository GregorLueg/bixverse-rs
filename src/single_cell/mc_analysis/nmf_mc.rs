//! Meta cells version of NMF. Leverages the sparse implementation of the
//! HALS version.

use std::time::Instant;

use crate::methods::nmf_hals::*;
use crate::prelude::*;
use crate::single_cell::sc_analysis::nmf_sc::{nmf_multiple_run_sparse, nmf_single_run_sparse};

/// Run NMF on meta cells
///
/// This function runs NMF on the chosen the provided meta cells x genes with
/// the given `k`. To note, during the fitting of the HALS-based NMF two copies
/// of the data layer are held in memory for speedy matrix operations. This
/// doubles the memory of that layer.
///
/// ### Params
///
/// * `data` - The [CompressedSparseData2] on which to run the the NMF. If
///   supplied as
/// * `k` - Number of latent variables to return
/// * `preprocessing` - String to forward to
///   [crate::methods::nmf_hals::nmf_preprocessing::parse_nmf_processing].
/// * `use_second_layer` - Shall the second data layer (normalised counts) be
///   used.
/// * `nmf_hals_params` - Optional parameters for the [HalsOpts].
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// [NmfResult] or respective errors if something went wrong.
#[allow(clippy::too_many_arguments)]
pub fn nmf_single_run_mc(
    data: CompressedSparseData2<f32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    verbose: usize,
) -> Result<NmfResult<f32>, BixverseErrors> {
    let start_total = Instant::now();
    let verbosity = parse_verbosity_level(verbose);

    let data = if data.cs_type.is_csr() {
        if verbosity.detailed_verbosity() {
            println!("NMF: meta cell data was provided as CSR. Transposing to CSC.")
        }
        data.transform()
    } else {
        data
    };

    if verbosity.normal_verbosity() {
        println!("NMF: Running NMF for metacells cells ...")
    }
    let res = nmf_single_run_sparse(
        data,
        k,
        preprocessing,
        use_second_layer,
        nmf_hals_params,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start_total.elapsed())
    }

    Ok(res)
}

/// Run multiple rounds of NMF on a subset of cells and genes loaded from disk
///
/// This function runs multiple rounds of NMF with different random
/// initilisations on the provided meta cells x genes with the given `k`.
/// To note, during the fitting of the HALS-based NMF two copies
/// of the data layer are held in memory for speedy matrix operations. This
/// doubles the memory of that layer.
///
/// ### Params
///
/// * `data` - The [CompressedSparseData2] on which to run the the NMF. If
///   supplied as
/// * `k` - Number of latent variables to return
/// * `preprocessing` - String to forward to
///   [crate::methods::nmf_hals::nmf_preprocessing::parse_nmf_processing].
/// * `use_second_layer` - Shall the second data layer (normalised counts) be
///   used.
/// * `nmf_hals_params` - Optional parameters for the [HalsOpts].
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// [NmfResult] or respective errors if something went wrong.
#[allow(clippy::too_many_arguments)]
pub fn nmf_multiple_run_mc(
    data: CompressedSparseData2<f32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<StabilisedNmfResult<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start_total = Instant::now();

    let data = if data.cs_type.is_csr() {
        if verbosity.detailed_verbosity() {
            println!("NMF: meta cell data was provided as CSR. Transposing to CSC.")
        }
        data.transform()
    } else {
        data
    };

    if verbosity.normal_verbosity() {
        println!("NMF: Running multiple NMF runs for meta cells ...")
    }
    let res = nmf_multiple_run_sparse(
        data,
        k,
        preprocessing,
        use_second_layer,
        nmf_hals_params,
        n_runs,
        base_seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start_total.elapsed())
    }

    Ok(res)
}
