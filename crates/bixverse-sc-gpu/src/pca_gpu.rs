//! Single cell-related GPU-accelerated SVD functions. Sparse PCA with the
//! SpMMs on the GPU, either randomised or exact via block Lanczos. Assumes
//! that the host has enough GPU memory.

use cubecl::Device;
use std::time::Instant;

use crate::prelude::*;
use bixverse_core::core::math::pca_svd::*;
use bixverse_core::core::math::{MAX_OVERSAMPLING_SINGLE_CELL, N_POWER_ITERS_SINGLE_CELL};
use bixverse_gpu::linalg::sparse_lanczos_gpu::sparse_svd_lanczos_gpu;
use bixverse_gpu::linalg::sparse_rand_svd_gpu::{RandSvdGpuParams, randomised_sparse_svd_gpu};
use bixverse_sc::sc_processing::pca::SingleCellPcaParams;
use bixverse_sc::sc_processing::pca::*;

//////////////////////
// Sparse PCA (GPU) //
//////////////////////

/// Calculate a sparse PCA on the GPU
///
/// This version will stream in the data from disk, calculate the gene
/// expression means and standard deviations and leverage GPU-accelerated
/// SpMMs to calculate a sparse SVD from the data while scaling the data during
/// the process. `PcaSolver::Randomised` runs the randomised SVD; every other
/// solver runs the exact block Lanczos, as there is no GPU covariance path.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `clr_offsets` - Per-cell CLR offsets, required if `params_pca.clr`.
/// * `params_pca` - The [SingleCellPcaParams] for this run.
/// * `seed` - Seed for the random sketch or the Lanczos starting block.
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

    let svd_res = match params_pca.svd_solver {
        PcaSolver::Randomised => {
            // safe guard against degenerate cases
            let max_s = std::cmp::min(cell_indices.len(), gene_indices.len()).saturating_sub(1);
            let oversampling =
                std::cmp::min(MAX_OVERSAMPLING_SINGLE_CELL, max_s.saturating_sub(no_pcs));
            let svd_params = RandSvdGpuParams::new(N_POWER_ITERS_SINGLE_CELL, oversampling);

            randomised_sparse_svd_gpu::<f32, f32>(
                csc,
                &col_means,
                &col_std,
                clr_offsets,
                no_pcs,
                Some(svd_params),
                seed as u64,
                device,
                verbose,
            )?
        }
        PcaSolver::Exact | PcaSolver::Auto | PcaSolver::Covariance => {
            sparse_svd_lanczos_gpu::<f32>(
                csc,
                &col_means,
                &col_std,
                clr_offsets,
                no_pcs,
                None,
                seed as u64,
                device,
                verbose,
            )?
        }
    };

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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use bixverse_sc_data::in_memory_io::InMemorySparseReader;
    use cubecl::wgpu::WgpuDevice;

    /// Synthetic CSC (cells x genes), rank-4 signal plus LCG noise, `data_2`
    /// holding `log1p(count)`.
    fn synthetic_store(n_cells: usize, n_genes: usize) -> CompressedSparseData2<u32, f32> {
        let mut state: u64 = 42;
        let mut lcg = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64)
        };
        let a: Vec<f64> = (0..n_cells * 4).map(|_| lcg() * 2.0).collect();
        let b: Vec<f64> = (0..4 * n_genes).map(|_| lcg() * 2.0).collect();

        let mut raw = Vec::new();
        let mut norm = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for g in 0..n_genes {
            for c in 0..n_cells {
                let rate: f64 = (0..4).map(|r| a[c * 4 + r] * b[r * n_genes + g]).sum();
                let count = (rate + lcg() - 2.0).floor();
                if count >= 1.0 {
                    raw.push(count as u32);
                    norm.push((count as f32).ln_1p());
                    indices.push(c as u32);
                }
            }
            indptr.push(indices.len() as u32);
        }
        CompressedSparseData2::new_csc(&raw, &indices, &indptr, Some(&norm), (n_cells, n_genes))
    }

    /// Every exact solver runs the GPU Lanczos and matches the CPU exact solve;
    /// randomised still runs the randomised path.
    #[test]
    fn test_pca_on_sc_sparse_gpu_dispatches_on_the_solver() {
        let device = WgpuDevice::default();
        if std::panic::catch_unwind(|| cubecl::Device::from(WgpuDevice::default()).client())
            .is_err()
        {
            return;
        }
        // 300 genes keeps the operator above the dense fallback.
        let matrix = synthetic_store(500, 300);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let cells: Vec<usize> = (0..500).collect();
        let genes: Vec<usize> = (0..300).collect();
        let n_pcs = 4;

        let params_of = |svd_solver| SingleCellPcaParams {
            svd_solver,
            ..SingleCellPcaParams::default()
        };
        let (_, _, cpu_s) = pca_on_sc_sparse(
            &reader,
            &cells,
            &genes,
            n_pcs,
            &params_of(PcaSolver::Exact),
            None,
            1,
            0,
        )
        .unwrap();

        for solver in [
            PcaSolver::Exact,
            PcaSolver::Auto,
            PcaSolver::Covariance,
            PcaSolver::Randomised,
        ] {
            let (scores, loadings, s) = pca_on_sc_sparse_gpu(
                &reader,
                &cells,
                &genes,
                n_pcs,
                None,
                &params_of(solver),
                1,
                device.clone().into(),
                0,
            )
            .unwrap();

            assert_eq!(scores.shape(), (500, n_pcs), "{solver:?}");
            assert_eq!(loadings.shape(), (300, n_pcs), "{solver:?}");
            // Randomised gets the leading values to its own tolerance only.
            let tol = if matches!(solver, PcaSolver::Randomised) {
                1e-2
            } else {
                1e-4
            };
            for j in 0..n_pcs {
                let rel = (s[j] - cpu_s[j]).abs() / cpu_s[j];
                assert!(rel < tol, "{solver:?} PC {j}: s {} vs {}", s[j], cpu_s[j]);
            }
        }
    }
}
