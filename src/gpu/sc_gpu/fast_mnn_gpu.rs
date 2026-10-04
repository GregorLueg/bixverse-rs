//! GPU version of the fastMNN batch correction from
//! `sc_batch_correction/fast_mnn.rs`. Only the neighbour searches move to the
//! device; centring, MNN pairing and the tricube correction are the CPU code,
//! shared via `fast_mnn_embedding`.

use cubecl::Runtime;
use faer::{Mat, MatRef};

use crate::gpu::sc_gpu::knn_gpu::{KnnParamsGpu, dispatch_knn_query_gpu};
use crate::prelude::*;
use crate::single_cell::sc_batch_correction::fast_mnn::fast_mnn_embedding;
use crate::single_cell::sc_processing::knn::to_true_distances;

////////////
// Params //
////////////

/// Parameters for GPU fastMNN.
///
/// Mirrors `FastMnnParams` with the GPU nearest neighbour parameters in place
/// of the CPU ones. There are no PCA fields, since the GPU entry point takes
/// the embedding directly.
#[derive(Clone, Debug)]
pub struct FastMnnParamsGpu {
    /// Number of median distances for tricube kernel bandwidth.
    pub ndist: f32,
    /// Apply cosine normalisation before computing distances.
    pub cos_norm: bool,
    /// [`KnnParamsGpu`] for the MNN and tricube searches. `k` sets the
    /// neighbour count for both; `extract_knn` is ignored, since every search
    /// here is a cross-query.
    pub knn_params: KnnParamsGpu,
}

impl FastMnnParamsGpu {
    /// Generate a version of this with sensible base parameters.
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            ndist: 3.0,
            cos_norm: true,
            knn_params: KnnParamsGpu::new(),
        }
    }
}

/// Default implementation for FastMnnParamsGpu
impl Default for FastMnnParamsGpu {
    fn default() -> Self {
        Self::new()
    }
}

///////////////
// Main code //
///////////////

/// Run fastMNN batch correction with the neighbour searches on the GPU
///
/// Same algorithm as the CPU `fast_mnn_main`, minus the PCA: three GPU index
/// builds and queries per merge (both MNN directions plus the tricube search
/// against the MNN-involved cells).
///
/// ### Params
///
/// * `embd` - The embedding matrix to correct. Usually PCA. cells = rows,
///   features = columns.
/// * `batch_labels` - Batch assignment for each cell.
/// * `params` - [`FastMnnParamsGpu`] for this run.
/// * `seed` - Random seed for the index builds.
/// * `device` - CubeCL runtime device.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Batch-corrected embedding (cells x features) in the original cell order.
///
/// ### References
///
/// Haghverdi, et al., Nat Biotechnol, 2018
pub fn fast_mnn_gpu<R: Runtime>(
    embd: MatRef<f32>,
    batch_labels: &[usize],
    params: &FastMnnParamsGpu,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors>
where
    R::Device: Clone,
{
    let verbosity = parse_verbosity_level(verbose);

    if verbosity.normal_verbosity() {
        println!("fastMNN (GPU): correcting the embedding.")
    }

    let search = |query: MatRef<f32>, reference: MatRef<f32>, k: usize| -> ScKnnResults {
        let (indices, dist) = dispatch_knn_query_gpu::<R>(
            reference,
            query,
            k,
            &params.knn_params,
            device.clone(),
            seed,
            true,
            verbosity.detailed_verbosity(),
        )?;
        let mut dist = dist.expect("distances were requested from the index");
        to_true_distances(&mut dist, &params.knn_params.ann_dist);
        Ok((indices, dist))
    };

    fast_mnn_embedding(
        &embd.to_owned(),
        batch_labels,
        params.cos_norm,
        params.knn_params.k,
        params.ndist,
        &search,
        verbose,
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_batch_correction::batch_utils::batch_knn_search;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Three batches of well-separated blobs, each batch shifted by its own
    /// offset, cells interleaved across batches.
    fn batched_data(n: usize, dim: usize, n_batches: usize) -> (Mat<f32>, Vec<usize>) {
        let hash = |a: usize, b: usize| -> f32 {
            let mut h = (a as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            ((h >> 40) as f32) / 16_777_216.0 - 0.5
        };
        let labels: Vec<usize> = (0..n).map(|i| i % n_batches).collect();
        let mat = Mat::from_fn(n, dim, |i, j| {
            let centre = hash((i / n_batches) % 5 + 1000, j) * 20.0;
            let offset = hash(labels[i] + 2000, j) * 4.0;
            centre + offset + hash(i, j)
        });
        (mat, labels)
    }

    /// Exhaustive GPU search against the exact CPU search on the same data:
    /// the corrected embeddings must agree to float noise.
    #[test]
    fn test_fast_mnn_gpu_matches_cpu_exhaustive() {
        let Some(device) = try_device() else {
            eprintln!("No GPU device, skipping");
            return;
        };

        let (embd, labels) = batched_data(900, 16, 3);
        let params = FastMnnParamsGpu::new();

        let gpu =
            fast_mnn_gpu::<WgpuRuntime>(embd.as_ref(), &labels, &params, 42, device, 0).unwrap();

        let mut cpu_knn = KnnParams::new();
        cpu_knn.knn_method = "exhaustive".to_string();
        let search =
            |q: MatRef<f32>, r: MatRef<f32>, k: usize| batch_knn_search(q, r, k, &cpu_knn, 42, 0);
        let cpu = fast_mnn_embedding(
            &embd,
            &labels,
            params.cos_norm,
            params.knn_params.k,
            params.ndist,
            &search,
            0,
        )
        .unwrap();

        assert_eq!(gpu.nrows(), cpu.nrows());
        assert_eq!(gpu.ncols(), cpu.ncols());
        let max_diff = (0..cpu.nrows())
            .flat_map(|i| (0..cpu.ncols()).map(move |j| (i, j)))
            .map(|(i, j)| (gpu[(i, j)] - cpu[(i, j)]).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-3, "max abs diff GPU vs CPU: {max_diff}");
    }

    /// A single batch has nothing to merge and must error, not pass through.
    #[test]
    fn test_fast_mnn_gpu_single_batch_errors() {
        let Some(device) = try_device() else {
            eprintln!("No GPU device, skipping");
            return;
        };

        let (embd, _) = batched_data(60, 8, 1);
        let labels = vec![0; 60];
        let res = fast_mnn_gpu::<WgpuRuntime>(
            embd.as_ref(),
            &labels,
            &FastMnnParamsGpu::new(),
            42,
            device,
            0,
        );
        assert!(matches!(
            res,
            Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches: 1 })
        ));
    }
}
