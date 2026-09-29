//! Bonsai over single cell counts with Sanity on the GPU.
//!
//! Only Sanity moves to the device. It is gene-parallel and was 60 to 80 times
//! faster there in sanity-sc-rs's own benchmarks; the tree search stays on the
//! CPU in [`run_bonsai_sc`].

use bonsai_rs::sanity_sc_rs::gpu::sanity_gpu_select;
use cubecl::Runtime;
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_analysis::bonsai::{
    BonsaiScParams, BonsaiScResult, SANITY_GENE_CHUNK, chunk_sanity_params, keep_for_bonsai,
    run_bonsai_sc, stream_sanity,
};
use crate::single_cell::sc_data::data_io::SingleCellReading;

/// Counts on disk to a laid-out Bonsai tree, Sanity on the GPU.
///
/// Streams genes in chunks as the CPU path does, with `sanity_gpu_select` per
/// chunk. The device computes in `f32` with the likelihood over the variance
/// grid assembled in `f64` on the host, so the posteriors match the CPU run to
/// within the variance grid's resolution rather than bit for bit, and a gene
/// right at a filter threshold can land on the other side.
///
/// ### Params
///
/// * `gene_reader` - Reader over the gene-major file
/// * `cell_reader` - Reader over the cell-major file
/// * `cell_indices` - Cells to keep, 0-indexed. Sets the leaf order
/// * `gene_indices` - Genes to consider, 0-indexed; only the ones passing
///   Bonsai's ingest filters are kept
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
    let started = Instant::now();
    let client = R::client(&device);
    let sanity_params = chunk_sanity_params(params);
    let keep = keep_for_bonsai(params);
    let post = stream_sanity(
        gene_reader,
        cell_reader,
        cell_indices,
        gene_indices,
        SANITY_GENE_CHUNK,
        verbosity,
        |counts, totals| {
            Ok(sanity_gpu_select(
                counts,
                totals,
                Some(sanity_params),
                &client,
                &keep,
            )?)
        },
    )?;
    let t_sanity = started.elapsed().as_secs_f64();
    let mut res = run_bonsai_sc(post, gene_indices.len(), params, verbosity)?;
    res.timings.insert(0, ("sanity".to_string(), t_sanity));
    Ok(res)
}
