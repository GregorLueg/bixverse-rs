//! NEBULA over the streamed single-cell store with stage two on the device.
//!
//! The adapter is [`run_nebula_with`]: cell ordering, batching, the filter-skip,
//! the dispersion shrinkage and the Wald test are all shared with the CPU path.
//! The only thing this module supplies is the per-batch fit,
//! [`edge_rs::gpu::stage_two::nebula_sparse_gpu`], and the client it runs on.
//!
//! The answers are not bit-identical to the CPU path: stage two's penalised fits
//! run in `f32` on the device and the host finishes each one in `f64`, so the
//! variance components land within the tolerance `edge-rs` gates rather than on
//! the CPU's values.

use cubecl::prelude::*;
use edge_rs::gpu::stage_two::nebula_sparse_gpu;

use crate::prelude::*;
use crate::single_cell::sc_analysis::nebula::{NebulaScParams, NebulaScRes, run_nebula_with};

/// Fits NEBULA to every requested gene with stage two on the device.
///
/// Takes and returns exactly what
/// [`run_nebula`](crate::single_cell::sc_analysis::nebula::run_nebula) does,
/// plus the device.
///
/// ### Params
///
/// * `gene_reader` - Gene-major store the counts come from
/// * `cell_reader` - Cell-major store the library sizes come from. Pass
///   `gene_reader` again for an in-memory store, which serves both
/// * `cells_to_keep` - Global indices of the cells to analyse, in any order
/// * `gene_indices` - Indices of the genes to fit
/// * `subject_ids` - Subject label per global cell
/// * `design` - Predictors, row-major `cells_to_keep.len() * n_coef`, rows
///   aligned to `cells_to_keep` and including an intercept
/// * `n_coef` - Number of design columns
/// * `offset` - Strictly positive scaling factor per selected cell, aligned to
///   `cells_to_keep`, or `None` to use the library sizes
/// * `params` - See [NebulaScParams]. `params.nebula.reml` is rejected, the
///   device fit does not implement it
/// * `device` - CubeCL device
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [NebulaScRes], or as
/// [`run_nebula`](crate::single_cell::sc_analysis::nebula::run_nebula), plus
/// [`BixverseErrors::InvalidArgument`] for `reml` and an [`edge_rs`] GPU error
/// if the device rejects the work.
///
/// ### References
///
/// He et al., Communications Biology 4, 629, 2021
#[allow(clippy::too_many_arguments)]
pub fn run_nebula_gpu<R, S>(
    gene_reader: &S,
    cell_reader: &S,
    cells_to_keep: &[usize],
    gene_indices: &[usize],
    subject_ids: &[usize],
    design: &[f64],
    n_coef: usize,
    offset: Option<&[f64]>,
    params: &NebulaScParams,
    device: R::Device,
    verbose: usize,
) -> Result<NebulaScRes, BixverseErrors>
where
    R: Runtime,
    S: SingleCellReading,
{
    // Upstream rejects this too, but only once the first batch has been read off
    // disk.
    if params.nebula.reml {
        return Err(BixverseErrors::InvalidArgument(
            "The GPU NEBULA path does not implement `reml`; use the CPU path.".to_string(),
        ));
    }

    let client = R::client(&device);

    run_nebula_with(
        gene_reader,
        cell_reader,
        cells_to_keep,
        gene_indices,
        subject_ids,
        design,
        n_coef,
        offset,
        params,
        verbose,
        |counts, subject_run, design_ordered, n_coef, offsets, nebula_params| {
            nebula_sparse_gpu::<f64, R>(
                counts,
                subject_run,
                design_ordered,
                n_coef,
                Some(offsets),
                Some(nebula_params),
                &client,
            )
        },
    )
}
