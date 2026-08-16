//! GPU-accelerated Scrublet doublet detection.
//!
//! Same algorithm as `single_cell::sc_processing::scrublet`, with the stages
//! that dominate the wall-clock moved to the device:
//!
//! 1. The PCA of the observed cells, via `randomised_sparse_svd_gpu`.
//! 2. The projection of the simulated doublets into that PC space, via the
//!    SpMM kernel rather than a dense `n_sim x n_hvg` intermediate.
//! 3. The kNN over the combined embedding, by default via the `ann-search-rs`
//!    GPU indices. The CPU indices stay selectable, and which index to use is
//!    a real choice, see [`ScrubletKnnBackend`].
//!
//! HVG selection, doublet simulation, scoring and Otsu thresholding stay on
//! the CPU and are reused unchanged from the CPU module.

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use faer::{Mat, concat};
use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::pca_svd::compute_pc_scores;
use crate::core::math::{DEFAULT_N_POWER_ITERS_RAND_SVD, MAX_OVERSAMPLING_SINGLE_CELL};
use crate::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use crate::gpu::linalg::sparse_rand_svd_gpu::{RandSvdGpuParams, randomised_sparse_svd_gpu};
use crate::gpu::linalg::spmm::{launch_dense_column_weighted_sum, launch_spmm_csr_forward};
use crate::gpu::sc_gpu::knn_gpu::{KnnParamsGpu, dispatch_knn_gpu};
use crate::prelude::*;
use crate::single_cell::sc_processing::knn::KnnParams;
use crate::single_cell::sc_processing::pca::{sparse_csc_column_means, sparse_csc_column_stds};
use crate::single_cell::sc_processing::scrublet::{
    FinalScrubletRes, calculate_doublet_scores, call_doublets,
};
use crate::single_cell::sc_processing::utils_doublets::*;

///////////////
// Constants //
///////////////

/// Floor applied to the per-gene standard deviations before they reach the
/// GPU SVD, which documents strictly positive standard deviations as a caller
/// obligation and does not enforce them itself.
const MIN_GENE_STD: f32 = 1e-8;

////////////
// Params //
////////////

/// Parameters for GPU-accelerated Scrublet.
///
/// Mirrors `ScrubletParams`, swapping the CPU `KnnParams` for the GPU
/// [`KnnParamsGpu`]. The `random_svd` flag has no counterpart here: the GPU
/// path is always randomised, since there is no GPU Lanczos.
#[derive(Clone, Debug)]
pub struct ScrubletParamsGpu {
    // -- processing --
    /// Whether to log-transform counts after normalisation.
    pub log_transform: bool,
    /// Whether to mean-centre genes before PCA.
    pub mean_center: bool,
    /// Whether to scale genes to unit variance before PCA.
    pub normalise_variance: bool,
    /// Optional target library size. Defaults to the mean HVG library size.
    pub target_size: Option<f32>,

    // -- hvg --
    /// Percentile threshold for HVG selection.
    pub min_gene_var_pctl: f32,
    /// HVG method: `"vst"`, `"mvb"`, or `"dispersion"`.
    pub hvg_method: String,
    /// Loess span for VST fitting.
    pub loess_span: f64,
    /// Optional clip max for variance stabilisation.
    pub clip_max: Option<f32>,
    /// Binning strategy.
    pub binning_strategy: String,
    /// Number of bins for HVG selection.
    pub n_bins: usize,

    // -- pca --
    /// Number of principal components.
    pub no_pcs: usize,

    // -- scrublet --
    /// Ratio of simulated doublets to observed cells.
    pub sim_doublet_ratio: f32,
    /// Expected doublet rate (typically 0.05-0.10).
    pub expected_doublet_rate: f32,
    /// Uncertainty in the expected doublet rate.
    pub stdev_doublet_rate: f32,
    /// Number of histogram bins for Otsu threshold detection.
    pub n_bins_hist: usize,
    /// Optional manual threshold. If `None`, Otsu's method is used.
    pub manual_threshold: Option<f32>,

    // -- knn --
    /// Which nearest neighbour backend to use, and its parameters. Defaults to
    /// GPU exhaustive, see [`ScrubletKnnBackend`].
    pub knn_params: ScrubletKnnBackend,
}

/// Nearest neighbour backend for GPU Scrublet.
///
/// [`ScrubletKnnBackend::Cpu`] stays available for the exact CPU indices.
#[derive(Clone, Debug)]
pub enum ScrubletKnnBackend {
    /// CPU nearest neighbour search, via `dispatch_knn`.
    Cpu(KnnParams),
    /// GPU nearest neighbour search, via [`dispatch_knn_gpu`].
    Gpu(KnnParamsGpu),
}

impl ScrubletKnnBackend {
    /// The requested number of neighbours, before the doublet adjustment.
    ///
    /// ### Returns
    ///
    /// The `k` held by whichever backend is selected.
    pub fn k(&self) -> usize {
        match self {
            Self::Cpu(p) => p.k,
            Self::Gpu(p) => p.k,
        }
    }
}

/// Default implementation for ScrubletKnnBackend
impl Default for ScrubletKnnBackend {
    fn default() -> Self {
        Self::Gpu(KnnParamsGpu::new())
    }
}

impl ScrubletParamsGpu {
    /// Generate a version of this with sensible base parameters.
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            log_transform: true,
            mean_center: true,
            normalise_variance: true,
            target_size: None,
            min_gene_var_pctl: 0.85,
            hvg_method: "vst".to_string(),
            loess_span: 0.3,
            clip_max: None,
            binning_strategy: "equal_width".to_string(),
            n_bins: 20,
            no_pcs: 30,
            sim_doublet_ratio: 2.0,
            expected_doublet_rate: 0.1,
            stdev_doublet_rate: 0.02,
            n_bins_hist: 100,
            manual_threshold: None,
            knn_params: ScrubletKnnBackend::default(),
        }
    }
}

/// Default implementation for ScrubletParamsGpu
impl Default for ScrubletParamsGpu {
    fn default() -> Self {
        Self::new()
    }
}

/////////////
// Helpers //
/////////////

/// Resolve the oversampling for the randomised SVD.
///
/// Clamped against the rank of the matrix so a degenerate input (few cells or
/// few HVGs) cannot ask for a sketch wider than the data supports.
/// [MAX_OVERSAMPLING_SINGLE_CELL] is what the CPU doublet path asks for, and it
/// can exceed the rank on small inputs.
///
/// ### Params
///
/// * `n_cells` - Number of observed cells.
/// * `n_genes` - Number of HVGs.
/// * `no_pcs` - Number of principal components requested.
///
/// ### Returns
///
/// The number of extra sketch columns to sample.
fn resolve_oversampling(n_cells: usize, n_genes: usize, no_pcs: usize) -> usize {
    let max_s = std::cmp::min(n_cells, n_genes).saturating_sub(1);
    std::cmp::min(MAX_OVERSAMPLING_SINGLE_CELL, max_s.saturating_sub(no_pcs))
}

/// PCA of the observed cells on the GPU.
///
/// Mirrors `pca_observed` up to the assembly of the CSC matrix, then hands off
/// to `randomised_sparse_svd_gpu` instead of the CPU SVD. Gene chunks are
/// re-normalised against the HVG-only library sizes, which is what makes the
/// Scrublet normalisation differ from the standard PCA path.
///
/// The `mean_center` and `normalise_variance` flags are applied by filling the
/// statistic vectors with zeros and ones respectively, since the GPU SVD takes
/// them unconditionally rather than as options.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based (CSC) count store.
/// * `cell_indices` - Cells to include.
/// * `gene_indices` - HVG indices.
/// * `hvg_library_sizes` - Per-cell library sizes over HVGs only, in the same
///   order as `cell_indices`.
/// * `target_size` - Normalisation target size.
/// * `log_transform` - Whether to apply `ln(1 + x)` after normalisation.
/// * `mean_center` - Whether to implicitly centre columns during the SVD.
/// * `normalise_variance` - Whether to implicitly scale columns to unit
///   variance during the SVD.
/// * `no_pcs` - Number of principal components.
/// * `seed` - Seed for the random projection.
/// * `device` - CubeCL runtime device.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
///   detailed verbosity.
///
/// ### Returns
///
/// A `DoubletPcaRes` of `(scores, loadings, gene_means, gene_stds)`. The
/// returned standard deviations carry the `MIN_GENE_STD` floor, so the
/// downstream projection divides by the same values the SVD used.
#[allow(clippy::too_many_arguments)]
pub fn pca_observed_gpu<R, S>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    hvg_library_sizes: &[usize],
    target_size: f32,
    log_transform: bool,
    mean_center: bool,
    normalise_variance: bool,
    no_pcs: usize,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<DoubletPcaRes, BixverseErrors>
where
    R: Runtime,
    S: SingleCellReading,
{
    let verbosity = parse_verbosity_level(verbose);
    let start_total = Instant::now();

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = cell_indices.len();
    let n_genes = gene_indices.len();

    let start_reading = Instant::now();
    let mut gene_chunks: Vec<CscGeneChunk> =
        reader.read_gene_parallel_filtered(gene_indices, &cell_set)?;
    if verbosity.normal_verbosity() {
        println!("Loaded in data: {:.2?}", start_reading.elapsed());
    }

    let start_prep = Instant::now();

    gene_chunks.par_iter_mut().for_each(|chunk| {
        for (i, &pos) in chunk.indices.iter().enumerate() {
            let raw_count = chunk.data_raw.get(i) as f32;
            let lib_size = hvg_library_sizes[pos as usize] as f32;
            let val = if log_transform {
                ((raw_count / lib_size) * target_size).ln_1p()
            } else {
                (raw_count / lib_size) * target_size
            };
            chunk.data_norm[i] = F16::from(half::f16::from_f32(val.clamp(-65504.0, 65504.0)));
        }
    });

    let csc = from_gene_chunks::<f32>(gene_chunks, &DataLayerReturn::Norm, n_cells)?;

    let real_col_means: Vec<f64> = sparse_csc_column_means(&csc, true, None)?;

    // The floor is what keeps a constant HVG column from pushing an infinity
    // through the sketch. It applies whether or not scaling is on, so the
    // projection later divides by exactly what the SVD used.
    let col_stds: Vec<f32> = if normalise_variance {
        let stds = sparse_csc_column_stds(&csc, &real_col_means, true, None)?;
        stds.iter().map(|&x| (x as f32).max(MIN_GENE_STD)).collect()
    } else {
        vec![1.0; n_genes]
    };
    let col_means: Vec<f32> = if mean_center {
        real_col_means.iter().map(|&x| x as f32).collect()
    } else {
        vec![0.0; n_genes]
    };

    if verbosity.normal_verbosity() {
        println!("Finished data preparation: {:.2?}", start_prep.elapsed());
    }

    let start_svd = Instant::now();

    let oversampling = resolve_oversampling(n_cells, n_genes, no_pcs);
    let svd_params = RandSvdGpuParams::new(DEFAULT_N_POWER_ITERS_RAND_SVD, oversampling);

    let svd_res = randomised_sparse_svd_gpu::<R, f32, f32>(
        csc,
        &col_means,
        &col_stds,
        None,
        no_pcs,
        Some(svd_params),
        seed as u64,
        device,
        verbose,
    )?;

    let scores = compute_pc_scores(&svd_res);
    let loadings = svd_res.v.clone();

    if verbosity.normal_verbosity() {
        println!("Finished PCA calculations: {:.2?}", start_svd.elapsed());
        println!(
            "Total run time PCA detection: {:.2?}",
            start_total.elapsed()
        );
    }

    Ok((scores, loadings, col_means, col_stds))
}

/// Project the simulated doublets into the observed PC space on the GPU.
///
/// Replaces the dense `scale_cell_chunks_with_stats` plus faer GEMM of the CPU
/// path. `launch_spmm_csr_forward` evaluates `Y = A @ Omega - 1 * c^T`, so
/// with `Omega[g, j] = V[g, j] / sigma[g]` and `c = mu^T @ Omega` the result is
/// `((X_sim - mu) / sigma) @ V`, i.e. the projection, without materialising the
/// scaled dense matrix.
///
/// ### Params
///
/// * `sim_chunks` - Simulated doublets, indexed in HVG space.
/// * `loadings` - PCA loadings `[n_genes, no_pcs]` from the observed cells.
/// * `gene_means` - Per-gene means from the observed cells.
/// * `gene_stds` - Per-gene standard deviations from the observed cells,
///   already floored.
/// * `mean_center` - Whether centring was applied to the observed PCA.
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// The simulated doublet scores, `[n_sim, no_pcs]`.
fn project_sim_gpu<R: Runtime>(
    sim_chunks: &[CsrCellChunk],
    loadings: &Mat<f32>,
    gene_means: &[f32],
    gene_stds: &[f32],
    mean_center: bool,
    client: &ComputeClient<R>,
) -> Result<Mat<f32>, BixverseErrors> {
    let n_sim = sim_chunks.len();
    let n_genes = gene_means.len();
    let no_pcs = loadings.ncols();

    let csr = from_cell_chunks::<f32>(sim_chunks, &DataLayerReturn::Norm, n_genes)?;
    let csr_gpu =
        GpuCompressedSparseData::<R, f32>::from_compressed_sparse_data_2(&csr, true, client)?;
    drop(csr);

    // Omega = V / sigma, rowwise. Row-major [n_genes, no_pcs] to match what
    // the SpMM kernel expects of the dense operand.
    let omega: Vec<f32> = (0..n_genes)
        .flat_map(|g| {
            let inv_sd = 1.0 / gene_stds[g];
            (0..no_pcs).map(move |j| loadings[(g, j)] * inv_sd)
        })
        .collect();
    let omega_gpu = GpuTensor::<R, f32>::from_slice(&omega, vec![n_genes, no_pcs], client)?;

    // Centring off means no correction term. `empty` does not zero the
    // buffer, so the uncentred arm has to upload zeros explicitly.
    let c_buf = if mean_center {
        let c = GpuTensor::<R, f32>::empty(vec![no_pcs], client)?;
        let mu_gpu = GpuTensor::<R, f32>::from_slice(gene_means, vec![n_genes], client)?;
        launch_dense_column_weighted_sum::<R, f32>(
            &mu_gpu, &omega_gpu, &c, n_genes, no_pcs, client,
        )?;
        c
    } else {
        GpuTensor::<R, f32>::from_slice(&vec![0.0f32; no_pcs], vec![no_pcs], client)?
    };

    project_with_correction::<R>(&csr_gpu, &omega_gpu, &c_buf, n_sim, no_pcs, client)
}

/// Run the SpMM and read the projection back to the host.
///
/// Split out of [`project_sim_gpu`] so the centred and uncentred arms share
/// the launch and the readback. There is no CLR in the doublet path, so the
/// row-offset and column-sum corrections are zero throughout.
///
/// ### Params
///
/// * `csr_gpu` - Device-resident CSR of the simulated doublets.
/// * `omega_gpu` - The scaled loadings `[n_genes, no_pcs]`.
/// * `c_buf` - The centring correction `[no_pcs]`.
/// * `n_sim` - Number of simulated doublets.
/// * `no_pcs` - Number of principal components.
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// The projected scores, `[n_sim, no_pcs]`.
fn project_with_correction<R: Runtime>(
    csr_gpu: &GpuCompressedSparseData<R, f32>,
    omega_gpu: &GpuTensor<R, f32>,
    c_buf: &GpuTensor<R, f32>,
    n_sim: usize,
    no_pcs: usize,
    client: &ComputeClient<R>,
) -> Result<Mat<f32>, BixverseErrors> {
    let zero_n = vec![0.0f32; n_sim];
    let zero_s = vec![0.0f32; no_pcs];
    let row_offsets = GpuTensor::<R, f32>::from_slice(&zero_n, vec![n_sim], client)?;
    let x_sum = GpuTensor::<R, f32>::from_slice(&zero_s, vec![no_pcs], client)?;

    let y_buf = GpuTensor::<R, f32>::empty(vec![n_sim, no_pcs], client)?;
    launch_spmm_csr_forward::<R, f32, f32>(
        csr_gpu,
        omega_gpu,
        c_buf,
        &row_offsets,
        &x_sum,
        &y_buf,
        no_pcs,
        client,
    )?;

    let host = y_buf.read(client)?;
    Ok(Mat::<f32>::from_fn(n_sim, no_pcs, |i, j| {
        host[i * no_pcs + j]
    }))
}

//////////
// Main //
//////////

/// Run the full Scrublet pipeline on the GPU.
///
/// Same algorithm and same return type as `Scrublet::run_scrublet`, with the
/// PCA, the doublet projection and the kNN on the device. HVG selection,
/// doublet simulation, scoring and thresholding are reused unchanged from the
/// CPU module.
///
/// ### Params
///
/// * `gene_reader` - Reader over the gene-based (CSC) count store.
/// * `cell_reader` - Reader over the cell-based (CSR) count store.
/// * `params` - The [`ScrubletParamsGpu`] for this run.
/// * `cell_indices` - Cell indices to include in the analysis.
/// * `streaming` - Stream HVG computation to reduce memory pressure.
/// * `seed` - Seed for reproducibility.
/// * `device` - CubeCL runtime device.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
///   detailed verbosity.
/// * `return_combined_pca` - Whether to return the combined PCA embedding.
/// * `return_pairs` - Whether to return the parent indices of each doublet.
///
/// ### Returns
///
/// `FinalScrubletRes` containing predictions, optional PCA and pair data.
///
/// ### Errors
///
/// * `CubeclUtils` if a dispatch grid or a workspace buffer busts a device
///   limit. The `[n, s]` sketch inside the SVD reaches it first.
/// * `AnnSearchRsError` if the GPU kNN index build or query fails.
/// * `FaerCholeskyError` if a CholeskyQR2 pass inside the SVD hits a non-SPD
///   Gram matrix.
///
/// ### References
///
/// Wolock, et al., Cell Systems, 2019
#[allow(clippy::too_many_arguments)]
pub fn run_scrublet_gpu<R, S>(
    gene_reader: &S,
    cell_reader: &S,
    params: &ScrubletParamsGpu,
    cell_indices: &[usize],
    streaming: bool,
    seed: usize,
    device: R::Device,
    verbose: usize,
    return_combined_pca: bool,
    return_pairs: bool,
) -> Result<FinalScrubletRes, BixverseErrors>
where
    R: Runtime,
    S: SingleCellReading,
{
    let verbosity = parse_verbosity_level(verbose);
    let n_cells = cell_indices.len();

    let hvg_opts = HvgOpts {
        method: params.hvg_method.clone(),
        loess_span: params.loess_span as f32,
        clip_max: params.clip_max,
        min_gene_var_pctl: params.min_gene_var_pctl,
        binning_strategy: params.binning_strategy.clone(),
        n_bins: params.n_bins,
    };

    if verbosity.normal_verbosity() {
        println!("Identifying highly variable genes...");
    }
    let start_all = Instant::now();
    let start_hvg = Instant::now();

    let hvg_genes = select_hvg(gene_reader, cell_indices, &hvg_opts, streaming, verbose)?;

    if verbosity.normal_verbosity() {
        println!(
            "Using {} highly variable genes. Done in {:.2?}",
            hvg_genes.len(),
            start_hvg.elapsed()
        );
    }

    let hvg_library_sizes = compute_hvg_library_sizes(cell_reader, cell_indices, &hvg_genes)?;
    let target_size = resolve_target_size(params.target_size, &hvg_library_sizes);

    // -- Doublet simulation --
    if verbosity.normal_verbosity() {
        println!("Simulating doublets...");
    }
    let start_sim = Instant::now();

    let n_cells_sim = (n_cells as f32 * params.sim_doublet_ratio) as usize;
    let pairs = random_pairs(cell_indices.len(), n_cells_sim, seed);
    let (pair_1, pair_2): (Vec<usize>, Vec<usize>) = pairs.iter().copied().unzip();

    let sim_chunks = simulate_from_pairs(
        &pairs,
        cell_indices,
        &hvg_library_sizes,
        &hvg_genes,
        cell_reader,
        target_size,
        params.log_transform,
    )?;

    if verbosity.normal_verbosity() {
        println!(
            "Simulated {} doublets. Done in {:.2?}",
            sim_chunks.len(),
            start_sim.elapsed()
        );
    }

    // -- PCA on the observed cells --
    if verbosity.normal_verbosity() {
        println!("Running PCA (GPU-accelerated)...");
    }

    let pca_res = pca_observed_gpu::<R, S>(
        gene_reader,
        cell_indices,
        &hvg_genes,
        &hvg_library_sizes,
        target_size,
        params.log_transform,
        params.mean_center,
        params.normalise_variance,
        params.no_pcs,
        seed,
        device.clone(),
        verbose,
    )?;

    // -- Project the simulated doublets --
    if verbosity.normal_verbosity() {
        println!("Projecting simulated doublets (GPU-accelerated)...");
    }
    let start_proj = Instant::now();

    let client = R::client(&device);
    let sim_pca = project_sim_gpu::<R>(
        &sim_chunks,
        &pca_res.1,
        &pca_res.2,
        &pca_res.3,
        params.mean_center,
        &client,
    )?;
    drop(client);

    let combined_pca = concat![[&pca_res.0], [sim_pca]];

    if verbosity.normal_verbosity() {
        println!("Done with projection in {:.2?}", start_proj.elapsed());
    }

    // -- kNN --
    if verbosity.normal_verbosity() {
        println!("Building kNN graph (GPU-accelerated)...");
    }
    let start_knn = Instant::now();

    let k_adj = adjusted_k(params.knn_params.k(), n_cells, n_cells_sim);
    if verbosity.normal_verbosity() {
        println!("Using {} neighbours in the kNN generation.", k_adj);
    }

    let knn_indices = match &params.knn_params {
        ScrubletKnnBackend::Cpu(knn_params) => dispatch_knn(
            combined_pca.as_ref(),
            k_adj,
            knn_params,
            seed,
            verbosity.detailed_verbosity(),
        )?,
        ScrubletKnnBackend::Gpu(knn_params) => dispatch_knn_gpu::<R>(
            combined_pca.as_ref(),
            k_adj,
            knn_params,
            device,
            seed,
            verbose,
        )?,
    };

    if verbosity.normal_verbosity() {
        println!("Done with kNN generation in {:.2?}", start_knn.elapsed());
    }

    // -- Scoring and calling --
    if verbosity.normal_verbosity() {
        println!("Calculating doublet scores...");
    }
    let start_score = Instant::now();

    let doublet_scores = calculate_doublet_scores(
        &knn_indices,
        n_cells,
        n_cells_sim,
        k_adj,
        params.expected_doublet_rate,
        params.stdev_doublet_rate,
    );
    let res = call_doublets(
        doublet_scores,
        params.manual_threshold,
        params.n_bins_hist,
        params.expected_doublet_rate,
        verbose,
    );

    if verbosity.normal_verbosity() {
        println!(
            "Done with doublet scoring and calling {:.2?}",
            start_score.elapsed()
        );
        println!("Finished Scrublet (GPU) {:.2?}", start_all.elapsed());
    }

    let pca_out = if return_combined_pca {
        Some(combined_pca)
    } else {
        None
    };
    let pair_1_out = if return_pairs { Some(pair_1) } else { None };
    let pair_2_out = if return_pairs { Some(pair_2) } else { None };

    Ok((res, pca_out, pair_1_out, pair_2_out))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::sc_gpu::knn_gpu::{KnnSearchGpu, parse_knn_method_gpu};
    use crate::single_cell::sc_data::data_io::{CellGeneSparseWriter, ParallelSparseReader};
    use crate::single_cell::sc_processing::knn::KnnParams;
    use crate::single_cell::sc_processing::scrublet::{Scrublet, ScrubletParams};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    const TEST_CELLS: usize = 600;
    const TEST_GENES: usize = 150;
    /// The fixture's three cell types span two real components; everything
    /// below sits in a flat noise floor where any two SVDs disagree on
    /// arbitrary directions. Keep this small enough that every retained PC is
    /// signal, or the CPU/GPU parity gate measures the noise subspace.
    const TEST_PCS: usize = 5;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /////////////
    // Helpers //
    /////////////

    /// Deterministic pseudo-random value in `[0, 1)`.
    fn hash01(a: usize, b: usize) -> f32 {
        let mut h = (a as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
        h ^= h >> 29;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 32;
        ((h >> 40) as f32) / 16_777_216.0
    }

    /// Synthetic counts with three cell types and type-specific marker genes,
    /// so HVG selection has real structure to find.
    fn synthetic_counts() -> Vec<Vec<u32>> {
        (0..TEST_CELLS)
            .map(|cell| {
                let cell_type = cell % 3;
                (0..TEST_GENES)
                    .map(|gene| {
                        let base = if gene % 3 == cell_type { 30.0 } else { 3.0 };
                        let v = base * (0.5 + hash01(cell, gene));
                        // Roughly 30% dropout, so the matrix stays sparse.
                        if hash01(gene, cell + 7) < 0.3 {
                            0
                        } else {
                            v as u32
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Counts with gene 0 unexpressed in every cell.
    ///
    /// This is the realistic route to a zero-variance column: a gene carrying
    /// no counts across the selected cells has an all-zero normalised column,
    /// so both its mean and its standard deviation are exactly zero whatever
    /// the per-cell library sizes are. A gene holding the same *non-zero* count
    /// everywhere would not do, because `pca_observed_gpu` renormalises against
    /// each cell's HVG library size and those differ.
    ///
    /// ### Returns
    ///
    /// Per-cell raw count vectors of length `TEST_GENES`, with index 0 zeroed.
    fn zero_gene_counts() -> Vec<Vec<u32>> {
        let mut counts = synthetic_counts();
        for row in counts.iter_mut() {
            row[0] = 0;
        }
        counts
    }

    /// Write the same counts to a gene-major and a cell-major store, since
    /// Scrublet needs both views.
    fn write_stores(counts: &[Vec<u32>], gene_path: &str, cell_path: &str) {
        let mut gene_writer =
            CellGeneSparseWriter::new(gene_path, false, TEST_CELLS, TEST_GENES, 1e4)
                .expect("gene writer");
        for gene in 0..TEST_GENES {
            let mut data_raw: Vec<u32> = Vec::new();
            let mut indices: Vec<usize> = Vec::new();
            for (cell, row) in counts.iter().enumerate() {
                if row[gene] > 0 {
                    data_raw.push(row[gene]);
                    indices.push(cell);
                }
            }
            let sum: f32 = data_raw.iter().map(|&x| x as f32).sum();
            let data_norm: Vec<F16> = data_raw
                .iter()
                .map(|&x| F16::from(half::f16::from_f32((x as f32 / sum * 1e4).ln_1p())))
                .collect();
            let chunk = CscGeneChunk::from_conversion(
                RawCounts::from_u32_auto(&data_raw),
                &data_norm,
                &indices,
                gene,
                true,
            );
            gene_writer.write_gene_chunk(chunk).expect("write gene");
        }
        gene_writer.finalise().expect("finalise gene store");

        let mut cell_writer =
            CellGeneSparseWriter::new(cell_path, true, TEST_CELLS, TEST_GENES, 1e4)
                .expect("cell writer");
        for (cell, row) in counts.iter().enumerate() {
            let mut data_raw: Vec<u32> = Vec::new();
            let mut indices: Vec<usize> = Vec::new();
            for (gene, &v) in row.iter().enumerate() {
                if v > 0 {
                    data_raw.push(v);
                    indices.push(gene);
                }
            }
            let chunk = CsrCellChunk::from_data(&data_raw, &indices, cell, 1e4, true);
            cell_writer.write_cell_chunk(chunk).expect("write cell");
        }
        cell_writer.finalise().expect("finalise cell store");
    }

    fn pearson(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        let ma = a.iter().sum::<f32>() / n;
        let mb = b.iter().sum::<f32>() / n;
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            let dx = x - ma;
            let dy = y - mb;
            num += dx * dy;
            da += dx * dx;
            db += dy * dy;
        }
        if da <= 0.0 || db <= 0.0 {
            return 0.0;
        }
        num / (da.sqrt() * db.sqrt())
    }

    fn cpu_params() -> ScrubletParams {
        ScrubletParams {
            log_transform: true,
            mean_center: true,
            normalise_variance: true,
            target_size: None,
            min_gene_var_pctl: 0.7,
            hvg_method: "vst".to_string(),
            loess_span: 0.5,
            clip_max: None,
            binning_strategy: "equal_width".to_string(),
            n_bins: 10,
            no_pcs: TEST_PCS,
            random_svd: true,
            sim_doublet_ratio: 2.0,
            expected_doublet_rate: 0.1,
            stdev_doublet_rate: 0.02,
            n_bins_hist: 50,
            manual_threshold: None,
            knn_params: KnnParams {
                knn_method: "exhaustive".to_string(),
                ..KnnParams::new()
            },
        }
    }

    fn gpu_params(knn: ScrubletKnnBackend) -> ScrubletParamsGpu {
        ScrubletParamsGpu {
            min_gene_var_pctl: 0.7,
            loess_span: 0.5,
            n_bins: 10,
            no_pcs: TEST_PCS,
            n_bins_hist: 50,
            knn_params: knn,
            ..ScrubletParamsGpu::new()
        }
    }

    fn gpu_knn_exhaustive() -> ScrubletKnnBackend {
        ScrubletKnnBackend::Gpu(KnnParamsGpu {
            knn_method: "exhaustive".to_string(),
            ..KnnParamsGpu::new()
        })
    }

    fn cpu_knn_exhaustive() -> ScrubletKnnBackend {
        ScrubletKnnBackend::Cpu(KnnParams {
            knn_method: "exhaustive".to_string(),
            ..KnnParams::new()
        })
    }

    ///////////
    // Tests //
    ///////////

    /// Oversampling must never exceed the rank the data supports, or the
    /// sketch is wider than the matrix.
    #[test]
    fn test_resolve_oversampling_clamps_on_small_inputs() {
        // Plenty of room: the constant ceiling applies. Spelled as a literal
        // rather than the const, so swapping the const out changes the expected
        // value here instead of silently agreeing with itself. That is exactly
        // how the ceiling once dropped to 10 without this test noticing.
        assert_eq!(resolve_oversampling(10_000, 3_000, 30), 100);

        // Rank-limited: 20 genes, 30 PCs requested, nothing left to sample.
        assert_eq!(resolve_oversampling(10_000, 20, 30), 0);

        // Just above the requested rank.
        assert_eq!(resolve_oversampling(10_000, 41, 30), 10);
    }

    /// The default must be the exact GPU index. IVF trades recall, which biases
    /// a doublet score built from a neighbour count, so it is opt-in. See
    /// [`ScrubletKnnBackend`].
    #[test]
    fn test_default_knn_backend_is_exact_gpu() {
        match ScrubletParamsGpu::new().knn_params {
            ScrubletKnnBackend::Gpu(p) => {
                assert_eq!(
                    parse_knn_method_gpu(&p.knn_method),
                    Some(KnnSearchGpu::ExhaustiveGpu)
                );
            }
            other => panic!("default kNN backend should be exact GPU, got {:?}", other),
        }
    }

    /// GPU Scrublet against the CPU reference on the same seed and store, both
    /// on an exact kNN. Run once per kNN backend, so the GPU search inside the
    /// driver is exercised as well as the default CPU one. The two randomised
    /// SVDs draw different sketches, so this asserts on correlation rather
    /// than equality.
    #[test]
    // Moderate: 600 cells x 150 genes, the CPU pipeline plus two GPU runs.
    fn test_run_scrublet_gpu_matches_cpu() {
        let Some(device) = try_device() else { return };

        let gene_path = std::env::temp_dir().join("bixverse_scrublet_gpu_genes.bin");
        let cell_path = std::env::temp_dir().join("bixverse_scrublet_gpu_cells.bin");
        let gene_str = gene_path.to_str().expect("utf8 path");
        let cell_str = cell_path.to_str().expect("utf8 path");

        write_stores(&synthetic_counts(), gene_str, cell_str);

        let gene_reader = ParallelSparseReader::new(gene_str).expect("gene reader");
        let cell_reader = ParallelSparseReader::new(cell_str).expect("cell reader");
        let cell_indices: Vec<usize> = (0..TEST_CELLS).collect();

        let mut cpu = Scrublet::new(&gene_reader, &cell_reader, cpu_params(), &cell_indices);
        let (cpu_res, _, _, _) = cpu
            .run_scrublet(false, 42, 0, false, false)
            .expect("cpu run");

        for (label, backend) in [
            ("cpu-knn", cpu_knn_exhaustive()),
            ("gpu-knn", gpu_knn_exhaustive()),
        ] {
            let (gpu_res, gpu_pca, _, _) = run_scrublet_gpu::<WgpuRuntime, _>(
                &gene_reader,
                &cell_reader,
                &gpu_params(backend),
                &cell_indices,
                false,
                42,
                device.clone(),
                0,
                true,
                false,
            )
            .expect("gpu run");

            // A GPU dispatch that busts a device limit returns zeros without
            // erroring, so check the embedding is real work before trusting
            // the scores at all. The observed and simulated blocks come from
            // different kernels, the SVD and the SpMM projection, so they have
            // to be checked separately: summing the whole matrix would let a
            // no-op projection hide behind the SVD's output.
            let pca = gpu_pca.expect("combined pca requested");
            assert_eq!(pca.nrows(), TEST_CELLS * 3);
            assert_eq!(pca.ncols(), TEST_PCS);

            let block_energy = |rows: std::ops::Range<usize>| -> f64 {
                rows.map(|i| {
                    (0..pca.ncols())
                        .map(|j| (pca[(i, j)] as f64).abs())
                        .sum::<f64>()
                })
                .sum()
            };
            let obs_energy = block_energy(0..TEST_CELLS);
            let sim_energy = block_energy(TEST_CELLS..pca.nrows());

            assert!(
                obs_energy.is_finite() && obs_energy > 1.0,
                "{}: observed block is degenerate (energy {}), the GPU SVD did no work",
                label,
                obs_energy
            );
            assert!(
                sim_energy.is_finite() && sim_energy > 1.0,
                "{}: simulated block is degenerate (energy {}), the SpMM projection did no work",
                label,
                sim_energy
            );

            assert_eq!(gpu_res.doublet_scores_obs.len(), TEST_CELLS);
            assert!(
                gpu_res.doublet_scores_obs.iter().all(|x| x.is_finite()),
                "{}: GPU doublet scores contain non-finite values",
                label
            );

            let r = pearson(&cpu_res.doublet_scores_obs, &gpu_res.doublet_scores_obs);
            println!(
                "Scrublet CPU vs GPU ({}): obs score Pearson {:.4}, thresholds {:.4} / {:.4}",
                label, r, cpu_res.threshold, gpu_res.threshold
            );
            // Measured 0.9952 on both backends. The gate sits just below that
            // rather than at a token value, because the two paths differ only
            // by the random sketch: anything that actually drifts shows up
            // well before 0.97.
            assert!(
                r > 0.97,
                "{}: GPU doublet scores only correlate {:.4} with the CPU reference",
                label,
                r
            );
        }

        let _ = std::fs::remove_file(&gene_path);
        let _ = std::fs::remove_file(&cell_path);
    }

    /// A zero-variance HVG column must reach the GPU SVD carrying the
    /// `MIN_GENE_STD` floor rather than the `f64::EPSILON` that
    /// `sparse_csc_column_stds` hands back, or the sketch amplifies that column
    /// by 4.5e15 in an f32 pipeline. See the constant's own docs for why that
    /// is headroom rather than a divide-by-zero guard.
    ///
    /// The fixture matters here. `pca_observed_gpu` rewrites `data_norm` per
    /// cell against that cell's HVG library size, so a gene holding the same
    /// non-zero count in every cell comes out varying, not constant, and the
    /// floor is never reached. [`zero_gene_counts`] uses an unexpressed gene
    /// instead, which is zero-variance regardless of library size.
    #[test]
    fn test_pca_observed_gpu_survives_zero_variance_gene() {
        let Some(device) = try_device() else { return };

        let gene_path = std::env::temp_dir().join("bixverse_scrublet_gpu_constant.bin");
        let cell_path = std::env::temp_dir().join("bixverse_scrublet_gpu_constant_cells.bin");
        let gene_str = gene_path.to_str().expect("utf8 path");
        let cell_str = cell_path.to_str().expect("utf8 path");

        write_stores(&zero_gene_counts(), gene_str, cell_str);

        let gene_reader = ParallelSparseReader::new(gene_str).expect("gene reader");
        let cell_reader = ParallelSparseReader::new(cell_str).expect("cell reader");
        let cell_indices: Vec<usize> = (0..TEST_CELLS).collect();
        let gene_indices: Vec<usize> = (0..TEST_GENES).collect();
        let lib_sizes =
            compute_hvg_library_sizes(&cell_reader, &cell_indices, &gene_indices).expect("libs");
        let target_size = resolve_target_size(None, &lib_sizes);

        let (scores, _, _, stds) = pca_observed_gpu::<WgpuRuntime, _>(
            &gene_reader,
            &cell_indices,
            &gene_indices,
            &lib_sizes,
            target_size,
            true,
            true,
            true,
            5,
            42,
            device,
            0,
        )
        .expect("gpu pca on a zero-variance column");

        // Gene 0 is the unexpressed one, so the floor must be exactly what came
        // back for it. Anything larger means the column still had variance and
        // the test is not exercising what it claims.
        assert_eq!(
            stds[0], MIN_GENE_STD,
            "gene 0 should have been floored, got a standard deviation of {}",
            stds[0]
        );
        assert!(
            stds[1..].iter().all(|&s| s > MIN_GENE_STD),
            "the expressed genes should not have been floored"
        );
        assert!(
            (0..scores.nrows()).all(|i| (0..scores.ncols()).all(|j| scores[(i, j)].is_finite())),
            "zero-variance gene produced non-finite PC scores"
        );

        let _ = std::fs::remove_file(&gene_path);
        let _ = std::fs::remove_file(&cell_path);
    }
}
