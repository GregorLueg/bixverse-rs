//! Single cells version of NMF. Leverages the sparse implementation of the
//! HALS version.

use indexmap::IndexSet;
use std::time::Instant;

use crate::methods::nmf_hals::nmf_preprocessing::*;
use crate::methods::nmf_hals::sparse::*;
use crate::methods::nmf_hals::*;
use crate::prelude::*;
use crate::single_cell::sc_data::data_io::from_gene_chunks;

/////////////
// Workers //
/////////////

/////////////////////////
// Single run (sparse) //
/////////////////////////

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
pub fn nmf_single_run_sparse<T>(
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
pub fn nmf_multiple_run_sparse<T>(
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

/// Run NMF on a subset of cells and genes loaded from disk
///
/// This function runs NMF on the chosen subset of cells x genes with the
/// given `k`. To note, during the fitting of the HALS-based NMF two copies
/// of the data layer are held in memory for speedy matrix operations. This
/// doubles the memory of that layer.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `gene_indices` - The indices of the genes to include. Typically, people
///   take 2000 to 5000 HVG within the subset of cells.
/// * `cell_indices` - The cells on which to run NMF.
/// * `k` - Number of latent variables to return
/// * `preprocessing` - String to forward to [parse_nmf_processing].
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
pub fn nmf_single_run_sc<S: SingleCellReading>(
    reader: &S,
    gene_indices: &[usize],
    cell_indices: &[usize],
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    verbose: usize,
) -> Result<NmfResult<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let n_cells = cell_indices.len();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let start_total = Instant::now();
    if verbosity.normal_verbosity() {
        println!("NMF: Loading data from disk ...")
    }

    let layer = if use_second_layer {
        DataLayerReturn::Norm
    } else {
        DataLayerReturn::Raw
    };

    let gene_chunks: Vec<CscGeneChunk> =
        reader.read_gene_parallel_filtered(gene_indices, &cell_set)?;
    let csc: CompressedSparseData2<f32> = from_gene_chunks::<f32>(gene_chunks, &layer, n_cells);

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start_total.elapsed())
    }

    if verbosity.normal_verbosity() {
        println!("NMF: Running NMF for single cells ...")
    }
    let res = nmf_single_run_sparse(
        csc,
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
/// initilisations on the chosen subset of cells x genes with the given `k`.
/// To note, during the fitting of the HALS-based NMF two copies
/// of the data layer are held in memory for speedy matrix operations. This
/// doubles the memory of that layer.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `gene_indices` - The indices of the genes to include. Typically, people
///   take 2000 to 5000 HVG within the subset of cells.
/// * `cell_indices` - The cells on which to run NMF.
/// * `k` - Number of latent variables to return
/// * `preprocessing` - String to forward to [parse_nmf_processing].
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
pub fn nmf_multiple_run_sc<S: SingleCellReading>(
    reader: &S,
    gene_indices: &[usize],
    cell_indices: &[usize],
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<StabilisedNmfResult<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let n_cells = cell_indices.len();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let start_total = Instant::now();
    if verbosity.normal_verbosity() {
        println!("NMF: Loading data from disk ...")
    }

    let layer = if use_second_layer {
        DataLayerReturn::Norm
    } else {
        DataLayerReturn::Raw
    };

    let gene_chunks: Vec<CscGeneChunk> =
        reader.read_gene_parallel_filtered(gene_indices, &cell_set)?;
    let csc: CompressedSparseData2<f32> = from_gene_chunks::<f32>(gene_chunks, &layer, n_cells);

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start_total.elapsed())
    }

    if verbosity.normal_verbosity() {
        println!("NMF: Running multiple NMF runs for single cells ...")
    }
    let res = nmf_multiple_run_sparse(
        csc,
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
