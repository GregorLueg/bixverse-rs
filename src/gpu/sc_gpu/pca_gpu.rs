//! Single cell-related GPU-accelerated SVD functions. Interface to a
//! GPU-accelerated sparse, randomised version to calculate rapidly PCAs and
//! leverage the GPU for the multiplication operations. Assumes that the host
//! has enough GPU memory.

use cubecl::Runtime;
use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::pca_svd::*;
use crate::gpu::linalg::sparse_rand_svd_gpu::{RandSvdGpuParams, randomised_sparse_svd_gpu};
use crate::prelude::*;
use crate::single_cell::sc_processing::pca::SingleCellPcaParams;
use crate::single_cell::sc_processing::pca::*;

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
/// * `random_svd` - Shall randomised sparse singular value decompostion be
///   used. This has the advantage of speed-ups, but loses precision.
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
pub fn pca_on_sc_sparse_gpu<R, S>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    clr_offsets: Option<&[f32]>,
    params_pca: &SingleCellPcaParams,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> SingleCellPcaRes
where
    R: Runtime,
    S: SingleCellReading,
{
    if params_pca.clr && clr_offsets.is_none() {
        return Err(BixverseErrors::OffsetsNotProvidedForClrPCA);
    }
    if params_pca.clr
        && let Some(offs) = clr_offsets
        && offs.len() != cell_indices.len()
    {
        return Err(BixverseErrors::OffsetsLengthDoesNotMatchNCells {
            len_offset: offs.len(),
            n_cells: cell_indices.len(),
        });
    }

    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let start_reading = Instant::now();

    let mut gene_chunks: Vec<CscGeneChunk> =
        reader.read_gene_parallel_filtered(gene_indices, &cell_set)?;

    if params_pca.clr {
        gene_chunks
            .par_iter_mut()
            .for_each(|chunk| chunk.transform_to_clr(params_pca.size_factor));
    }

    let row_offsets_f64: Option<Vec<f64>> = if params_pca.clr {
        Some(clr_offsets.unwrap().iter().map(|&x| x as f64).collect())
    } else {
        None
    };

    let end_reading = start_reading.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Sparse PCA (GPU-accelerated): Loaded in data in {:.2?}",
            end_reading
        );
    }

    let start_data_prep = Instant::now();

    let n_cells = cell_set.len();

    let csc: CompressedSparseData2<f32> =
        from_gene_chunks::<f32>(gene_chunks, &DataLayerReturn::Norm, n_cells);

    let end_data_prep = start_data_prep.elapsed();

    let real_col_means: Vec<f64> = sparse_csc_column_means(&csc, true, row_offsets_f64.as_deref())?;

    let col_stds: Vec<f64> = if params_pca.normalise_variance {
        sparse_csc_column_stds(&csc, &real_col_means, true, row_offsets_f64.as_deref())?
    } else {
        vec![1.0; real_col_means.len()]
    };

    let col_means: Vec<f64> = if params_pca.mean_center {
        real_col_means
    } else {
        vec![0.0; col_stds.len()]
    };

    // down cast here to avoid trait bound chaos
    let col_means: Vec<f32> = col_means.iter().map(|x| *x as f32).collect();
    let col_std: Vec<f32> = col_stds.iter().map(|x| *x as f32).collect();

    if verbosity.normal_verbosity() {
        println!(
            "Sparse PCA (GPU-accelerated): finished the data preparations in {:.2?}",
            end_data_prep
        );
    }

    let start_svd = Instant::now();

    // safe guard against degenerate cases
    let max_s = std::cmp::min(cell_indices.len(), gene_indices.len()).saturating_sub(1);
    let oversampling = std::cmp::min(100, max_s.saturating_sub(no_pcs));
    let svd_params = RandSvdGpuParams::new(2, oversampling);

    let svd_res = randomised_sparse_svd_gpu::<R, f32, f32>(
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
