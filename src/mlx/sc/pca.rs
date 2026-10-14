//! Sparse randomised single-cell PCA on MLX. Same streaming, column statistics
//! and CLR handling as `gpu::sc_gpu::pca_gpu`, with the SVD on
//! `mlx::linalg::rand_svd`.

use std::time::Instant;

use crate::core::math::pca_svd::*;
use crate::core::math::{MAX_OVERSAMPLING_SINGLE_CELL, N_POWER_ITERS_SINGLE_CELL};
use crate::mlx::linalg::rand_svd::{RandSvdMlxParams, randomised_sparse_svd_mlx};
use crate::prelude::*;
use crate::single_cell::sc_processing::pca::*;

/////////////////////////////////
// Sparse randomised PCA (MLX) //
/////////////////////////////////

/// Calculate a sparse randomised PCA on MLX.
///
/// Streams the data in, computes the gene means and standard deviations on the
/// host and runs the randomised sparse SVD on MLX with the centring and
/// scaling applied implicitly, so the matrix is never densified.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `clr_offsets` - Per-cell CLR offsets, required if `params_pca.clr`.
/// * `params_pca` - The [SingleCellPcaParams] for this run. `svd_solver` is
///   ignored, this is always randomised.
/// * `seed` - Seed for randomised SVD.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The [SingleCellPcaRes]: scores (cells x PCs), gene loadings (genes x PCs)
/// and singular values.
#[allow(clippy::too_many_arguments)]
pub fn pca_on_sc_sparse_mlx<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    clr_offsets: Option<&[f32]>,
    params_pca: &SingleCellPcaParams,
    seed: usize,
    verbose: usize,
) -> SingleCellPcaRes {
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let (csc, col_means, col_std) = sparse_pca_accelerator_input(
        reader,
        cell_indices,
        gene_indices,
        clr_offsets,
        params_pca,
        "MLX",
        verbose,
    )?;

    let start_svd = Instant::now();

    // safe guard against degenerate cases
    let max_s = std::cmp::min(cell_indices.len(), gene_indices.len()).saturating_sub(1);
    let oversampling = std::cmp::min(MAX_OVERSAMPLING_SINGLE_CELL, max_s.saturating_sub(no_pcs));
    let svd_params = RandSvdMlxParams::new(N_POWER_ITERS_SINGLE_CELL, oversampling);

    let svd_res = randomised_sparse_svd_mlx::<f32>(
        csc,
        &col_means,
        &col_std,
        clr_offsets,
        no_pcs,
        Some(svd_params),
        seed as u64,
        verbose,
    )?;

    let scores = compute_pc_scores(&svd_res);

    if verbosity.normal_verbosity() {
        println!(
            "Sparse PCA (MLX): finished calculations in {:.2?}",
            start_svd.elapsed()
        );
        println!(
            "Sparse PCA (MLX): total run time -> {:.2?}",
            start_total.elapsed()
        );
    }

    Ok((scores, svd_res.v, svd_res.s))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_data::in_memory_io::InMemorySparseReader;
    use faer::Mat;

    /// Number of cells in the synthetic store.
    const N_CELLS: usize = 400;
    /// Number of genes in the synthetic store.
    const N_GENES: usize = 80;
    /// Rank of the latent signal.
    const RANK: usize = 4;

    /// Synthetic CSC (cells x genes) with a rank-`RANK` signal and LCG noise,
    /// counts thresholded to give roughly half zeros. `data_2` holds
    /// `log1p(count)`.
    fn synthetic_store() -> CompressedSparseData2<u32, f32> {
        let mut state: u64 = 42;
        let mut lcg = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64)
        };
        let a: Vec<f64> = (0..N_CELLS * RANK).map(|_| lcg() * 2.0).collect();
        let b: Vec<f64> = (0..RANK * N_GENES).map(|_| lcg() * 2.0).collect();

        let mut raw = Vec::new();
        let mut norm = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for g in 0..N_GENES {
            for c in 0..N_CELLS {
                let rate: f64 = (0..RANK)
                    .map(|r| a[c * RANK + r] * b[r * N_GENES + g])
                    .sum();
                let count = (rate + lcg() - 2.0).floor();
                if count >= 1.0 {
                    raw.push(count as u32);
                    norm.push((count as f32).ln_1p());
                    indices.push(c as u32);
                }
            }
            indptr.push(indices.len() as u32);
        }
        CompressedSparseData2::new_csc(&raw, &indices, &indptr, Some(&norm), (N_CELLS, N_GENES))
    }

    /// Absolute Pearson correlation between two columns, sign-invariant.
    fn abs_corr(x: &Mat<f32>, y: &Mat<f32>, j: usize) -> f64 {
        let n = x.nrows();
        let (mx, my) = (0..n).fold((0f64, 0f64), |(a, b), i| {
            (a + x[(i, j)] as f64, b + y[(i, j)] as f64)
        });
        let (mx, my) = (mx / n as f64, my / n as f64);
        let (mut sxy, mut sxx, mut syy) = (0f64, 0f64, 0f64);
        for i in 0..n {
            let (dx, dy) = (x[(i, j)] as f64 - mx, y[(i, j)] as f64 - my);
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        (sxy / (sxx * syy).sqrt()).abs()
    }

    /// Leading components and singular values agree with the exact CPU solve.
    #[test]
    fn test_pca_on_sc_sparse_mlx_matches_cpu_exact() {
        let matrix = synthetic_store();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let cells: Vec<usize> = (0..N_CELLS).collect();
        let genes: Vec<usize> = (0..N_GENES).collect();
        let n_pcs = 3;

        let (mlx_scores, _, mlx_s) = pca_on_sc_sparse_mlx(
            &reader,
            &cells,
            &genes,
            n_pcs,
            None,
            &SingleCellPcaParams::default(),
            1,
            0,
        )
        .unwrap();

        let params_cpu = SingleCellPcaParams {
            svd_solver: PcaSolver::Exact,
            ..SingleCellPcaParams::default()
        };
        let (cpu_scores, _, cpu_s) =
            pca_on_sc_sparse(&reader, &cells, &genes, n_pcs, &params_cpu, None, 1, 0).unwrap();

        assert_eq!(mlx_scores.nrows(), N_CELLS);
        assert_eq!(mlx_scores.ncols(), n_pcs);
        for j in 0..n_pcs {
            let rel = (mlx_s[j] - cpu_s[j]).abs() / cpu_s[j];
            assert!(rel < 1e-3, "PC {j}: s {} vs {}", mlx_s[j], cpu_s[j]);
            let r = abs_corr(&mlx_scores, &cpu_scores, j);
            assert!(r > 0.999, "PC {j}: |corr| = {r}");
        }
    }

    /// CLR without offsets is rejected before anything is read.
    #[test]
    fn test_pca_on_sc_sparse_mlx_clr_needs_offsets() {
        let matrix = synthetic_store();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let params = SingleCellPcaParams {
            clr: true,
            ..SingleCellPcaParams::default()
        };
        let res = pca_on_sc_sparse_mlx(&reader, &[0, 1, 2], &[0, 1, 2], 2, None, &params, 1, 0);
        assert!(matches!(
            res,
            Err(BixverseErrors::OffsetsNotProvidedForClrPCA)
        ));
    }

    /// MLX and wgpu give the same components on the same store.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_pca_on_sc_sparse_mlx_matches_wgpu() {
        use crate::gpu::sc_gpu::pca_gpu::pca_on_sc_sparse_gpu;
        use cubecl::wgpu::WgpuDevice;

        let matrix = synthetic_store();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let cells: Vec<usize> = (0..N_CELLS).collect();
        let genes: Vec<usize> = (0..N_GENES).collect();
        let params = SingleCellPcaParams::default();
        let n_pcs = 3;

        let (mlx_scores, _, mlx_s) =
            pca_on_sc_sparse_mlx(&reader, &cells, &genes, n_pcs, None, &params, 1, 0).unwrap();
        let (gpu_scores, _, gpu_s) = pca_on_sc_sparse_gpu(
            &reader,
            &cells,
            &genes,
            n_pcs,
            None,
            &params,
            1,
            WgpuDevice::default().into(),
            0,
        )
        .unwrap();

        for j in 0..n_pcs {
            let rel = (mlx_s[j] - gpu_s[j]).abs() / gpu_s[j];
            assert!(rel < 1e-3, "PC {j}: s {} vs {}", mlx_s[j], gpu_s[j]);
            let r = abs_corr(&mlx_scores, &gpu_scores, j);
            assert!(r > 0.999, "PC {j}: |corr| = {r}");
        }
    }
}
