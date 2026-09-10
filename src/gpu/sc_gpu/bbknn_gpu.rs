//! GPU version of the BBKNN batch correction from
//! `sc_batch_correction/bbknn.rs`. Only the neighbour search moves to the
//! device: one GPU index per batch, queried by every cell, with the UMAP
//! connectivity pipeline shared with the CPU path via `bbknn_graph_from_knn`.
//!
//! The GPU backends hand back squared euclidean distances, which is not what
//! the CPU BBKNN works in. It costs nothing to sidestep: the batch indices are
//! batch-local, so both paths have to recompute the distance against the global
//! matrix anyway. The queries therefore run with `return_dist: false`.

#![allow(missing_docs)]

use ann_search_rs::utils::dist::{Dist, parse_ann_dist};
use ann_search_rs::{
    build_exhaustive_index_gpu, build_ivf_index_gpu, build_nndescent_index_gpu,
    query_exhaustive_index_gpu, query_ivf_index_gpu, query_nndescent_index_gpu,
};
use cubecl::Runtime;
use faer::MatRef;
use rayon::prelude::*;

use crate::core::mat_struct::MatSliceView;
use crate::gpu::sc_gpu::knn_gpu::{
    KnnParamsGpu, KnnSearchGpu, cagra_query_params, parse_knn_method_gpu,
};
use crate::prelude::*;
use crate::single_cell::sc_batch_correction::batch_utils::process_batch_labels;
use crate::single_cell::sc_batch_correction::bbknn::{bbknn_graph_from_knn, check_batch_size};

////////////
// Params //
////////////

/// Parameters for GPU BBKNN.
///
/// Mirrors `BbknnParams` with the GPU nearest neighbour parameters in place of
/// the CPU ones.
#[derive(Clone, Debug)]
pub struct BbknnParamsGpu {
    /// How many neighbours per batch to identify
    pub neighbours_within_batch: usize,
    /// Mixing ratio between union (1.0) and intersection (0.0).
    pub set_op_mix_ratio: f32,
    /// UMAP connectivity computation parameter, how many nearest neighbours of
    /// each cell are assumed to be fully connected.
    pub local_connectivity: f32,
    /// Trim the neighbours of each cell to these many to connectivities. May
    /// help with population independence and improve the tidiness of clustering.
    pub trim: Option<usize>,
    /// [`KnnParamsGpu`] for the per-batch searches. Two of its fields do not
    /// apply here: `k` is ignored, since `neighbours_within_batch` sets the
    /// neighbour count, and `extract_knn` is ignored, since extraction hands
    /// back the index's own graph rather than the results of a query.
    pub knn_params: KnnParamsGpu,
}

impl BbknnParamsGpu {
    /// Generate a version of this with sensible base parameters.
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            neighbours_within_batch: 3,
            set_op_mix_ratio: 1.0,
            local_connectivity: 1.0,
            trim: Some(30),
            knn_params: KnnParamsGpu::new(),
        }
    }
}

/// Default implementation for BbknnParamsGpu
impl Default for BbknnParamsGpu {
    fn default() -> Self {
        Self::new()
    }
}

/////////////
// Helpers //
/////////////

/// Query one batch index with every cell on the GPU.
///
/// Pure dispatch over the three GPU backends, mirroring `dispatch_knn_gpu` but
/// on the cross-query entry points: the index covers one batch, the query
/// matrix covers all cells. An unrecognised method string falls back to the
/// default rather than erroring, matching the rest of the crate.
///
/// ### Params
///
/// * `sub_matrix` - The cells of this batch, the data the index is built on.
/// * `mat` - The full embedding matrix, the query side.
/// * `k` - Neighbours to return per cell, self included.
/// * `params` - The [`KnnParamsGpu`] for this run.
/// * `device` - CubeCL runtime device.
/// * `seed` - Random seed for the index build.
/// * `verbose` - Detailed verbosity flag handed to the index build and query.
///
/// ### Returns
///
/// Batch-local neighbour indices, one row per cell of `mat`.
fn query_batch_index_gpu<R: Runtime>(
    sub_matrix: MatRef<f32>,
    mat: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    device: R::Device,
    seed: usize,
    verbose: bool,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    let method = parse_knn_method_gpu(&params.knn_method).unwrap_or_else(|| {
        println!(
            "Unrecognised GPU kNN method provided: {:?}. Defaulting to exhaustive GPU.",
            params.knn_method
        );
        KnnSearchGpu::default()
    });

    let (indices, _) = match method {
        KnnSearchGpu::ExhaustiveGpu => {
            let index = build_exhaustive_index_gpu::<f32, R>(sub_matrix, &params.ann_dist, device)?;
            query_exhaustive_index_gpu(mat, &index, k, false, verbose)?
        }
        KnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_gpu::<f32, R>(
                sub_matrix,
                params.n_list,
                None,
                &params.ann_dist,
                seed,
                verbose,
                device,
            )?;
            query_ivf_index_gpu(mat, &index, k, params.n_probe, None, false, verbose)?
        }
        KnnSearchGpu::CagraGpu => {
            // `retain_gpu` has to be set: the cross-query path needs the
            // vectors to stay device-resident after the build.
            let mut index = build_nndescent_index_gpu::<f32, R>(
                sub_matrix,
                &params.ann_dist,
                params.graph_k,
                params.k_build,
                None,
                params.n_tree,
                Some(params.delta),
                params.rho,
                params.refine_knn,
                seed,
                verbose,
                true,
                device,
            )?;

            let query_params = cagra_query_params(params, k, params.graph_k);

            query_nndescent_index_gpu(mat, &mut index, k, Some(query_params), false, verbose)?
        }
    };

    Ok(indices)
}

/// Generate a batch balanced kNN graph on the GPU
///
/// The GPU mirror of `get_batch_balanced_knn`: a GPU index per batch, every
/// cell queried against it, and the batch-local indices mapped back to global
/// ones. Batches run one after the other, since the device serialises the work
/// anyway. The distance recompute inside each batch is parallel over cells.
///
/// ### Params
///
/// * `mat` - The embedding matrix to use. Usually PCA. cells = rows, features
///   = columns.
/// * `batch_labels` - Slice indicating which cell belongs to which batch.
/// * `bbknn_params` - [`BbknnParamsGpu`] for this run.
/// * `device` - CubeCL runtime device.
/// * `seed` - Random seed.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A tuple with (nearest_neighbour_indices, nearest_neighbour_distances)
fn get_batch_balanced_knn_gpu<R: Runtime>(
    mat: MatRef<f32>,
    batch_labels: &[usize],
    bbknn_params: &BbknnParamsGpu,
    device: R::Device,
    seed: usize,
    verbose: usize,
) -> ScKnnResults
where
    R::Device: Clone,
{
    let verbosity = parse_verbosity_level(verbose);

    let n_cells = mat.nrows();
    let (unique_batches, n_batches) = process_batch_labels(batch_labels);

    if n_batches == 1 {
        return Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches });
    }

    let dist_metric: Dist = parse_ann_dist(&bbknn_params.knn_params.ann_dist).unwrap_or_default();

    let n_per_batch = bbknn_params.neighbours_within_batch;
    let mut all_indices = vec![vec![0; n_per_batch * n_batches]; n_cells];
    let mut all_distances = vec![vec![0.0; n_per_batch * n_batches]; n_cells];
    let col_indices: Vec<usize> = (0..mat.ncols()).collect();

    for (batch_idx, &batch) in unique_batches.iter().enumerate() {
        if verbosity.normal_verbosity() {
            println!(
                "Processing batch {} / {}: {}",
                batch_idx + 1,
                n_batches,
                batch
            );
        }

        let batch_cell_indices: Vec<usize> = batch_labels
            .iter()
            .enumerate()
            .filter(|(_, b)| **b == batch)
            .map(|(i, _)| i)
            .collect();

        check_batch_size(batch, batch_cell_indices.len(), n_per_batch)?;

        let sub_matrix = MatSliceView::new(mat, &batch_cell_indices, &col_indices).to_owned();

        let neighbour_indices = query_batch_index_gpu::<R>(
            sub_matrix.as_ref(),
            mat,
            n_per_batch + 1,
            &bbknn_params.knn_params,
            device.clone(),
            seed,
            verbosity.detailed_verbosity(),
        )?;

        let col_start = batch_idx * n_per_batch;

        all_indices
            .par_iter_mut()
            .zip(all_distances.par_iter_mut())
            .enumerate()
            .for_each(|(cell_idx, (indices, distances))| {
                let mut added = 0;
                let mut k_idx = 0;

                while added < n_per_batch {
                    let global_idx = batch_cell_indices[neighbour_indices[cell_idx][k_idx]];

                    if global_idx != cell_idx {
                        let dist = compute_distance_knn(
                            mat.row(cell_idx),
                            mat.row(global_idx),
                            &dist_metric,
                        );

                        indices[col_start + added] = global_idx;
                        distances[col_start + added] = dist;
                        added += 1;
                    }

                    k_idx += 1;
                }
            });
    }

    Ok((all_indices, all_distances))
}

////////////////////
// Main BBKNN GPU //
////////////////////

/// Run batch-balanced KNN on the GPU
///
/// Same output as the CPU [`bbknn`](crate::single_cell::sc_batch_correction::bbknn::bbknn),
/// with the per-batch neighbour searches on the device.
///
/// ### Params
///
/// * `mat` - The embedding matrix to use. Usually PCA. cells = rows, features
///   = columns.
/// * `batch_labels` - Slice indicating which cell belongs to which batch.
/// * `params` - [`BbknnParamsGpu`] with the parameters for the BBKNN batch
///   correction.
/// * `seed` - Random seed.
/// * `device` - CubeCL runtime device.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A tuple of two CompressedSparseData2 with `(distances, connectivities)`.
///
/// ### References
///
/// Polański, et al., Bioinformatics, 2019
pub fn bbknn_gpu<R: Runtime>(
    mat: MatRef<f32>,
    batch_labels: &[usize],
    params: &BbknnParamsGpu,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<(CompressedSparseData2<f32>, CompressedSparseData2<f32>), BixverseErrors>
where
    R::Device: Clone,
{
    let verbosity = parse_verbosity_level(verbose);

    if verbosity.normal_verbosity() {
        println!("BBKNN (GPU): generating the batch balanced kNN values.")
    }

    let (mut knn_indices, mut knn_dists) =
        get_batch_balanced_knn_gpu::<R>(mat, batch_labels, params, device, seed, verbose)?;

    bbknn_graph_from_knn(
        &mut knn_indices,
        &mut knn_dists,
        mat.nrows(),
        params.set_op_mix_ratio,
        params.local_connectivity,
        params.trim,
        verbose,
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_batch_correction::bbknn::{BbknnParams, bbknn};
    use approx::assert_relative_eq;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use faer::Mat;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Deterministic clustered data with a per-batch offset, so the batches
    /// really do need mixing.
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

        let batch_labels: Vec<usize> = (0..n).map(|i| i % n_batches).collect();
        let mat = Mat::<f32>::from_fn(n, dim, |i, j| {
            let cluster = (i / n_batches) % 4;
            let centre = hash(cluster + 1000, j) * 10.0;
            centre + hash(i, j) * 2.0 + batch_labels[i] as f32 * 0.5
        });

        (mat, batch_labels)
    }

    /// Exhaustive is exact on both sides and both recompute the distances
    /// against the global matrix, so the two graphs have to agree exactly.
    #[test]
    fn test_bbknn_gpu_matches_cpu_exhaustive() {
        let Some(device) = try_device() else { return };

        let (mat, batch_labels) = batched_data(300, 12, 3);

        let params_gpu = BbknnParamsGpu {
            neighbours_within_batch: 3,
            knn_params: KnnParamsGpu {
                knn_method: "exhaustive".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let params_cpu = BbknnParams {
            neighbours_within_batch: 3,
            set_op_mix_ratio: params_gpu.set_op_mix_ratio,
            local_connectivity: params_gpu.local_connectivity,
            trim: params_gpu.trim,
            knn_params: KnnParams {
                knn_method: "exhaustive".to_string(),
                ann_dist: "euclidean".to_string(),
                ..Default::default()
            },
        };

        let (dist_gpu, conn_gpu) =
            bbknn_gpu::<WgpuRuntime>(mat.as_ref(), &batch_labels, &params_gpu, 42, device, 0)
                .unwrap();
        let (dist_cpu, conn_cpu) = bbknn(mat.as_ref(), &batch_labels, &params_cpu, 42, 0).unwrap();

        assert_eq!(conn_gpu.indptr, conn_cpu.indptr);
        assert_eq!(conn_gpu.indices, conn_cpu.indices);
        assert_eq!(dist_gpu.indptr, dist_cpu.indptr);
        assert_eq!(dist_gpu.indices, dist_cpu.indices);

        for (g, c) in conn_gpu.data.iter().zip(conn_cpu.data.iter()) {
            assert_relative_eq!(g, c, epsilon = 1e-5);
        }
        for (g, c) in dist_gpu.data.iter().zip(dist_cpu.data.iter()) {
            assert_relative_eq!(g, c, epsilon = 1e-5);
        }
    }

    /// A batch smaller than `neighbours_within_batch + 1` used to walk off the
    /// end of the neighbour row.
    #[test]
    fn test_bbknn_gpu_rejects_tiny_batch() {
        let Some(device) = try_device() else { return };

        let (mat, mut batch_labels) = batched_data(60, 8, 2);
        // Leave batch 1 with a single cell.
        for label in batch_labels.iter_mut() {
            *label = 0;
        }
        batch_labels[1] = 1;

        let params = BbknnParamsGpu::default();
        let res = bbknn_gpu::<WgpuRuntime>(mat.as_ref(), &batch_labels, &params, 42, device, 0);

        assert!(matches!(
            res,
            Err(BixverseErrors::BbknnBatchTooSmall { .. })
        ));
    }

    /// A single batch has nothing to balance.
    #[test]
    fn test_bbknn_gpu_rejects_single_batch() {
        let Some(device) = try_device() else { return };

        let (mat, _) = batched_data(60, 8, 2);
        let batch_labels = vec![0usize; 60];

        let params = BbknnParamsGpu::default();
        let res = bbknn_gpu::<WgpuRuntime>(mat.as_ref(), &batch_labels, &params, 42, device, 0);

        assert!(matches!(
            res,
            Err(BixverseErrors::NeedAtLeastTwoBatches { .. })
        ));
    }

    /// The approximate backends cannot be checked value by value, so assert the
    /// structural properties instead: right shape and no self-edges.
    #[test]
    fn test_bbknn_gpu_approximate_backends_graph_shape() {
        let Some(device) = try_device() else { return };

        let (mat, batch_labels) = batched_data(400, 12, 2);

        for method in ["ivf", "cagra"] {
            let params = BbknnParamsGpu {
                neighbours_within_batch: 3,
                knn_params: KnnParamsGpu {
                    knn_method: method.to_string(),
                    ..Default::default()
                },
                ..Default::default()
            };

            let (_, conn) = bbknn_gpu::<WgpuRuntime>(
                mat.as_ref(),
                &batch_labels,
                &params,
                42,
                device.clone(),
                0,
            )
            .unwrap();

            assert_eq!(conn.shape, (400, 400), "{}: wrong shape", method);
            assert_eq!(conn.indptr.len(), 401, "{}: wrong indptr length", method);

            for i in 0..400 {
                let start = conn.indptr[i] as usize;
                let end = conn.indptr[i + 1] as usize;
                assert!(
                    !conn.indices[start..end].contains(&(i as u32)),
                    "{}: row {} contains a self-edge",
                    method,
                    i
                );
            }
        }
    }
}
