//! Single cell-related GPU-accelerated SVD functions. Interface to a
//! GPU-accelerated sparse, randomised version to calculate rapidly PCAs and
//! leverage the GPU for the multiplication operations. Assumes that the host
//! has enough GPU memory.

use cubecl::Device;
use std::time::Instant;

use crate::prelude::*;
use bixverse_core::core::math::pca_svd::*;
use bixverse_core::core::math::{MAX_OVERSAMPLING_SINGLE_CELL, N_POWER_ITERS_SINGLE_CELL};
use bixverse_gpu::linalg::sparse_rand_svd_gpu::{RandSvdGpuParams, randomised_sparse_svd_gpu};
use bixverse_sc::sc_processing::pca::SingleCellPcaParams;
use bixverse_sc::sc_processing::pca::*;

/////////////////////////////////
// Sparse randomised PCA (GPU) //
/////////////////////////////////

/// Calculate randomised SVD in sparse on GPU
///
/// This version will stream in the data from disk, calculate the gene
/// expression means and standard deviations and leverage GPU-accelerated
/// matrix operations to calculate a randomised sparse SVD from the data while
/// scaling the data during the process.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `clr_offsets` - Per-cell CLR offsets, required if `params_pca.clr`.
/// * `params_pca` - The [SingleCellPcaParams] for this run.
/// * `seed` - Seed for randomised SVD.
/// * `device` - CubeCL runtime device
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Return
///
/// A tuple of the samples projected on the PC space, gene loadings and singular
/// values.
#[allow(clippy::too_many_arguments)]
pub fn pca_on_sc_sparse_gpu<S>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    clr_offsets: Option<&[f32]>,
    params_pca: &SingleCellPcaParams,
    seed: usize,
    device: Device,
    verbose: usize,
) -> SingleCellPcaRes<ScGpuErrors>
where
    S: SingleCellReading,
{
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let (csc, col_means, col_std) = sparse_pca_accelerator_input(
        reader,
        cell_indices,
        gene_indices,
        clr_offsets,
        params_pca,
        "GPU-accelerated",
        verbose,
    )?;

    let start_svd = Instant::now();

    // safe guard against degenerate cases
    let max_s = std::cmp::min(cell_indices.len(), gene_indices.len()).saturating_sub(1);
    let oversampling = std::cmp::min(MAX_OVERSAMPLING_SINGLE_CELL, max_s.saturating_sub(no_pcs));
    let svd_params = RandSvdGpuParams::new(N_POWER_ITERS_SINGLE_CELL, oversampling);

    let svd_res = randomised_sparse_svd_gpu::<f32, f32>(
        csc,
        &col_means,
        &col_std,
        clr_offsets,
        no_pcs,
        Some(svd_params),
        seed as u64,
        device,
        verbose,
    )?;

    let scores = compute_pc_scores(&svd_res);

    let end_svd = start_svd.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Sparse PCA (GPU-accelerated): finished calculations in {:.2?}",
            end_svd
        );
    }

    let end_total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Sparse PCA (GPU-accelerated): total run time -> {:.2?}",
            end_total
        );
    }

    Ok((scores, svd_res.v().to_owned(), svd_res.s().to_owned()))
}
