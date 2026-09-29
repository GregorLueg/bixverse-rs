//! Bonsai over single cell counts with Sanity on the GPU.
//!
//! Only Sanity moves to the device. It is gene-parallel and was 60 to 80 times
//! faster there in sanity-sc-rs's own benchmarks; the tree search stays on the
//! CPU in [`run_bonsai_sc`].

use bonsai_rs::sanity_sc_rs::gpu::sanity_gpu;
use cubecl::Runtime;

use crate::prelude::*;
use crate::single_cell::sc_analysis::bonsai::{
    BonsaiScParams, BonsaiScResult, run_bonsai_sc, sanity_counts, sanity_params,
};
use crate::single_cell::sc_data::data_io::SingleCellReading;

/// Counts on disk to a laid-out Bonsai tree, Sanity on the GPU.
///
/// The device computes in `f32` with the likelihood over the variance grid
/// assembled in `f64` on the host, so the posteriors match the CPU run to
/// within the variance grid's resolution rather than bit for bit.
///
/// ### Params
///
/// * `gene_reader` - Reader over the gene-major file
/// * `cell_reader` - Reader over the cell-major file
/// * `cell_indices` - Cells to keep, 0-indexed. Sets the leaf order
/// * `gene_indices` - Genes to use, 0-indexed
/// * `params` - Sanity, Bonsai and layout parameters
/// * `device` - The device to run Sanity on
/// * `verbosity` - How much to print
///
/// ### Returns
///
/// The tree, its layout, and the genes it was built on.
pub fn sanity_bonsai_sc_gpu<R, G, C>(
    gene_reader: &G,
    cell_reader: &C,
    cell_indices: &[usize],
    gene_indices: &[usize],
    params: &BonsaiScParams,
    device: R::Device,
    verbosity: Verbosity,
) -> Result<BonsaiScResult, BixverseErrors>
where
    R: Runtime,
    G: SingleCellReading,
    C: SingleCellReading,
{
    let input = sanity_counts(
        gene_reader,
        cell_reader,
        cell_indices,
        gene_indices,
        verbosity,
    )?;
    let client = R::client(&device);
    let post = sanity_gpu::<f32, R>(
        &input.counts,
        &input.cell_totals,
        Some(sanity_params(params, verbosity)),
        &client,
    )?;
    drop(input.counts);
    run_bonsai_sc(post, &input.genes, params, verbosity)
}
