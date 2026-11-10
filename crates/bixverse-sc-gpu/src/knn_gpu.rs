//! GPU-accelerated approximate nearest neighbour search for single cell data,
//! specifically with Scrublet in mind. The callers here run at high `k`, which
//! is why exhaustive stays the default: CAGRA and the NN-Descent graph are at
//! their best on low `k`.

use ann_search_rs::prelude::{CagraGpuSearchParams, NNDescentGpuKnnParams};
use ann_search_rs::{
    build_exhaustive_index_gpu, build_ivf_index_gpu, build_knn_graph_nndescent_gpu,
    build_nndescent_index_gpu, query_exhaustive_index_gpu, query_exhaustive_index_gpu_self,
    query_ivf_index_gpu, query_ivf_index_gpu_self, query_nndescent_index_gpu,
    query_nndescent_index_gpu_self,
};
use bixverse_sc::sc_processing::knn::{knn_graph_to_rows, to_true_distances, top_up_short_rows};
use bixverse_sc::sc_processing::knn_device::cagra_beam;
pub use bixverse_sc::sc_processing::knn_device::{
    KnnParamsGpu, KnnSearchGpu, de_ann_dist_gpu, de_knn_method_gpu, parse_knn_method_gpu,
};
use cubecl::Device;
use faer::MatRef;
use rayon::prelude::*;

use crate::prelude::*;

///////////
// Types //
///////////

/// Neighbour indices from a GPU query and, if requested, the distances.
pub(crate) type GpuKnnQueryResult = Result<(Vec<Vec<usize>>, Option<Vec<Vec<f32>>>), ScGpuErrors>;

/// Neighbour indices and distances from a GPU self-query, one row per cell.
pub type GpuKnnWithDist = Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), ScGpuErrors>;

/////////////
// Helpers //
/////////////

/// The wgpu CAGRA query parameters, beam scaled to `k` by [`cagra_beam`].
///
/// ### Params
///
/// * `params` - The [`KnnParamsGpu`] for this run.
/// * `k_query` - Number of neighbours asked of the index, self included.
///
/// ### Returns
///
/// The [`CagraGpuSearchParams`] to hand to the query.
fn cagra_query_params(params: &KnnParamsGpu, k_query: usize) -> CagraGpuSearchParams {
    let (beam_width, max_beam_iters, n_entry_points) = cagra_beam(params, k_query);
    CagraGpuSearchParams::new(beam_width, max_beam_iters, n_entry_points, None)
}

/// Drop each query point from its own neighbour list and truncate to `k`.
///
/// The GPU self-queries return the query point itself, so they are issued with
/// `k + 1` and trimmed here. Matches the CPU `build_and_query_knn` behaviour.
///
/// ### Params
///
/// * `raw` - Neighbour indices as returned by the GPU self-query, one row per
///   query point, each of length up to `k + 1`.
/// * `k` - Number of neighbours to keep per row.
///
/// ### Returns
///
/// Neighbour indices with self removed, each row at most `k` long.
fn drop_self_neighbours(raw: Vec<Vec<usize>>, k: usize) -> Vec<Vec<usize>> {
    raw.into_par_iter()
        .enumerate()
        .map(|(i, idx)| idx.into_iter().filter(|&j| j != i).take(k).collect())
        .collect()
}

//////////////
// Dispatch //
//////////////

/// Dispatch to the correct GPU kNN implementation.
///
/// Pure routing, mirroring `dispatch_knn` on the CPU side. The `k` value must
/// already be adjusted by the caller. An unrecognised method string warns and
/// falls back to the default rather than erroring, matching the CPU behaviour.
///
/// ### Params
///
/// * `embd` - Embedding matrix of `n_cells x features`.
/// * `k` - Number of neighbours to return per cell.
/// * `params` - The [`KnnParamsGpu`] for this run.
/// * `device` - CubeCL runtime device.
/// * `seed` - Seed for reproducibility of the index builds.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
///   detailed verbosity.
///
/// ### Returns
///
/// The kNN indices for the cells, self excluded, each row at most `k` long.
///
/// ### Errors
///
/// * `AnnSearchRsError` if an index build or query fails, including the typed
///   shared-memory refusals when `k` or the embedding dimensionality is too
///   large for the device.
pub fn dispatch_knn_gpu(
    embd: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<usize>>, ScGpuErrors> {
    let (raw_indices, _) = self_query_gpu(embd, k, params, device, seed, verbose, false)?;
    Ok(drop_self_neighbours(raw_indices, k))
}

/// GPU kNN with the distances, self excluded.
///
/// The sibling of [`dispatch_knn_gpu`] for callers that keep the distances,
/// as the CPU `generate_knn_with_dist` does. Euclidean distances come back as
/// true L2, not squared, via `to_true_distances`.
///
/// ### Params
///
/// * `embd` - Embedding matrix of `n_cells x features`.
/// * `k` - Number of neighbours to return per cell.
/// * `params` - The [`KnnParamsGpu`] for this run.
/// * `device` - CubeCL runtime device.
/// * `seed` - Seed for reproducibility of the index builds.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
///   detailed verbosity.
///
/// ### Returns
///
/// The kNN indices and distances, one row per cell, each at most `k` long and
/// sorted by distance.
///
/// ### Errors
///
/// As [`dispatch_knn_gpu`].
pub fn knn_with_dist_gpu(
    embd: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    device: Device,
    seed: usize,
    verbose: usize,
) -> GpuKnnWithDist {
    let (raw_indices, raw_dist) = self_query_gpu(embd, k, params, device, seed, verbose, true)?;
    let raw_dist = raw_dist.expect("every GPU self-query returns distances when asked");
    let (indices, mut dist): (Vec<Vec<usize>>, Vec<Vec<f32>>) = raw_indices
        .into_par_iter()
        .zip(raw_dist)
        .enumerate()
        .map(|(i, (idx, d))| {
            idx.into_iter()
                .zip(d)
                .filter(|&(j, _)| j != i)
                .take(k)
                .unzip()
        })
        .unzip();
    to_true_distances(&mut dist, &params.ann_dist);
    Ok((indices, dist))
}

/// Run the GPU self-query the parameters ask for.
///
/// ### Params
///
/// * `embd` - Embedding matrix of `n_cells x features`.
/// * `k` - Number of neighbours wanted per cell, self excluded.
/// * `params` - The [`KnnParamsGpu`] for this run.
/// * `device` - CubeCL runtime device.
/// * `seed` - Seed for reproducibility of the index builds.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
///   detailed verbosity.
/// * `return_dist` - Also return the distances, as the backend reports them.
///
/// ### Returns
///
/// The raw rows, up to `k + 1` long and possibly holding the query point, and
/// the distances if asked for.
fn self_query_gpu(
    embd: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    device: Device,
    seed: usize,
    verbose: usize,
    return_dist: bool,
) -> GpuKnnQueryResult {
    // The index builds and queries only print at detailed verbosity here.
    let ann_verbosity = verbosity_from_flag(parse_verbosity_level(verbose).detailed_verbosity());

    let method = parse_knn_method_gpu(&params.knn_method).unwrap_or_else(|| {
        println!(
            "Unrecognised GPU kNN method provided: {:?}. Defaulting to exhaustive GPU.",
            params.knn_method
        );
        KnnSearchGpu::default()
    });

    // Every backend self-queries, so ask for one extra and drop the query
    // point afterwards.
    let k_query = k + 1;

    let res = match method {
        KnnSearchGpu::ExhaustiveGpu => {
            let index =
                build_exhaustive_index_gpu::<f32>(embd, &params.ann_dist, device, ann_verbosity)?;
            query_exhaustive_index_gpu_self(&index, k_query, return_dist, ann_verbosity)?
        }
        KnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_gpu::<f32>(
                embd,
                params.n_list,
                None,
                &params.ann_dist,
                seed,
                ann_verbosity,
                device,
            )?;
            query_ivf_index_gpu_self(
                &index,
                k_query,
                params.n_probe,
                None,
                return_dist,
                ann_verbosity,
            )?
        }
        KnnSearchGpu::CagraGpu => {
            let mut index = build_nndescent_index_gpu::<f32>(
                embd,
                &params.ann_dist,
                params.graph_k,
                params.k_build,
                None,
                params.n_tree,
                Some(params.delta),
                params.rho,
                params.refine_knn,
                seed,
                ann_verbosity,
                true,
                device,
            )?;
            let query_params = cagra_query_params(params, k_query);

            query_nndescent_index_gpu_self(
                &mut index,
                k_query,
                Some(query_params),
                return_dist,
                ann_verbosity,
            )?
        }
        KnnSearchGpu::NNDescentGraphGpu => {
            // No self-edge requested, so dropping self afterwards only
            // truncates.
            let graph_params = NNDescentGpuKnnParams::new(
                k,
                params.k_build,
                None,
                params.n_tree,
                Some(params.delta),
                params.rho,
                params.refine_knn,
                false,
                return_dist,
            );
            let graph = build_knn_graph_nndescent_gpu::<f32>(
                embd,
                &params.ann_dist,
                Some(graph_params),
                None,
                seed,
                ann_verbosity,
                device,
            )?;
            let (mut indices, mut distances) = knn_graph_to_rows(graph);
            top_up_short_rows(
                embd,
                &mut indices,
                distances.as_deref_mut(),
                k,
                &params.ann_dist,
                parse_verbosity_level(verbose).normal_verbosity(),
            )?;
            (indices, distances)
        }
    };

    Ok(res)
}

/// Build a GPU index on `reference` and query it with every row of `query`.
///
/// The cross-query sibling of [`dispatch_knn_gpu`], shared by the batch
/// correction methods. Nothing is dropped from the rows, since query and
/// reference are different matrices. An unrecognised method string warns and
/// falls back to the default rather than erroring.
///
/// ### Params
///
/// * `reference` - The data the index is built on (cells x features).
/// * `query` - The query points (cells x features).
/// * `k` - Neighbours to return per query point.
/// * `params` - The [`KnnParamsGpu`] for this run. `k` is ignored, and
///   `"nndescent"` runs as CAGRA, since a graph-only build cannot answer a
///   cross-query.
/// * `device` - CubeCL runtime device.
/// * `seed` - Random seed for the index build.
/// * `return_dist` - Return the distances as well. These come back in the
///   index's metric, so squared for Euclidean.
/// * `verbose` - Detailed verbosity flag handed to the index build and query.
///
/// ### Returns
///
/// Reference-local neighbour indices, one row per query point, and the
/// distances if requested.
///
/// ### Errors
///
/// * `AnnSearchRsError` if an index build or query fails.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_knn_query_gpu(
    reference: MatRef<f32>,
    query: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    device: Device,
    seed: usize,
    return_dist: bool,
    verbose: bool,
) -> GpuKnnQueryResult {
    let method = parse_knn_method_gpu(&params.knn_method).unwrap_or_else(|| {
        println!(
            "Unrecognised GPU kNN method provided: {:?}. Defaulting to exhaustive GPU.",
            params.knn_method
        );
        KnnSearchGpu::default()
    });
    let verbosity = verbosity_from_flag(verbose);

    let res = match method {
        KnnSearchGpu::ExhaustiveGpu => {
            let index =
                build_exhaustive_index_gpu::<f32>(reference, &params.ann_dist, device, verbosity)?;
            query_exhaustive_index_gpu(query, &index, k, return_dist, verbosity)?
        }
        KnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_gpu::<f32>(
                reference,
                params.n_list,
                None,
                &params.ann_dist,
                seed,
                verbosity,
                device,
            )?;
            query_ivf_index_gpu(
                query,
                &index,
                k,
                params.n_probe,
                None,
                return_dist,
                verbosity,
            )?
        }
        KnnSearchGpu::CagraGpu | KnnSearchGpu::NNDescentGraphGpu => {
            // `retain_gpu` has to be set: the cross-query path needs the
            // vectors to stay device-resident after the build.
            let mut index = build_nndescent_index_gpu::<f32>(
                reference,
                &params.ann_dist,
                params.graph_k,
                params.k_build,
                None,
                params.n_tree,
                Some(params.delta),
                params.rho,
                params.refine_knn,
                seed,
                verbosity,
                true,
                device,
            )?;

            let query_params = cagra_query_params(params, k);

            query_nndescent_index_gpu(
                query,
                &mut index,
                k,
                Some(query_params),
                return_dist,
                verbosity,
            )?
        }
    };

    Ok(res)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use ann_search_rs::{build_exhaustive_index, query_exhaustive_self};
    use cubecl::wgpu::WgpuDevice;
    use faer::Mat;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::default();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::Device::from(device.clone()).client();
        }))
        .ok()
        .map(|_| device)
    }

    /////////////
    // Helpers //
    /////////////

    /// Deterministic clustered data, `n` points in `dim` dimensions spread over
    /// `n_clusters` well-separated Gaussian-ish blobs.
    fn clustered_data(n: usize, dim: usize, n_clusters: usize) -> Mat<f32> {
        let hash = |a: usize, b: usize| -> f32 {
            let mut h = (a as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            ((h >> 40) as f32) / 16_777_216.0 - 0.5
        };

        Mat::<f32>::from_fn(n, dim, |i, j| {
            let cluster = i % n_clusters;
            let centre = hash(cluster + 1000, j) * 20.0;
            centre + hash(i, j) * 2.0
        })
    }

    /// Recall of `got` against `want`, averaged over rows.
    fn recall(got: &[Vec<usize>], want: &[Vec<usize>]) -> f32 {
        let total: f32 = got
            .iter()
            .zip(want.iter())
            .map(|(g, w)| {
                let truth: rustc_hash::FxHashSet<usize> = w.iter().copied().collect();
                let hits = g.iter().filter(|x| truth.contains(x)).count();
                hits as f32 / w.len().max(1) as f32
            })
            .sum();
        total / got.len().max(1) as f32
    }

    fn cpu_reference(data: &Mat<f32>, k: usize) -> Vec<Vec<usize>> {
        let index = build_exhaustive_index(data.as_ref(), "euclidean", Verbosity::Quiet);
        let (idx, _) = query_exhaustive_self(&index, k + 1, false, Verbosity::Quiet).unwrap();
        idx.into_iter()
            .enumerate()
            .map(|(i, row)| row.into_iter().filter(|&j| j != i).take(k).collect())
            .collect()
    }

    ///////////
    // Tests //
    ///////////

    /// The string parser must accept the bare names and the `_gpu` suffix, and
    /// reject anything the GPU side does not implement.
    #[test]
    fn test_parse_knn_method_gpu() {
        assert_eq!(
            parse_knn_method_gpu("exhaustive"),
            Some(KnnSearchGpu::ExhaustiveGpu)
        );
        assert_eq!(parse_knn_method_gpu("IVF_GPU"), Some(KnnSearchGpu::IvfGpu));
        assert_eq!(parse_knn_method_gpu("cagra"), Some(KnnSearchGpu::CagraGpu));
        assert_eq!(
            parse_knn_method_gpu("CAGRA_MLX"),
            Some(KnnSearchGpu::CagraGpu)
        );
        for name in ["nndescent", "nndescent_gpu", "NNDescent_MLX"] {
            assert_eq!(
                parse_knn_method_gpu(name),
                Some(KnnSearchGpu::NNDescentGraphGpu),
                "{}",
                name
            );
        }
        assert_eq!(parse_knn_method_gpu("nndescent_graph"), None);
        assert_eq!(parse_knn_method_gpu("nndescent_graph_gpu"), None);
        assert_eq!(parse_knn_method_gpu("kmknn"), None);
        assert_eq!(parse_knn_method_gpu("annoy"), None);
    }

    /// CAGRA and the graph-only NN-Descent build at the low `k` they are
    /// actually good at. The beam search fills every row, and the graph path
    /// has its short rows topped up, so both must hand back exactly `k`.
    #[test]
    fn test_cagra_and_nndescent_gpu() {
        let Some(device) = try_device() else { return };

        let (n, dim, k) = (2000usize, 24usize, 15usize);
        let data = clustered_data(n, dim, 10);
        let want = cpu_reference(&data, k);

        for method in ["cagra", "nndescent"] {
            let params = KnnParamsGpu {
                knn_method: method.to_string(),
                k,
                refine_knn: Some(1),
                ..Default::default()
            };
            let got =
                dispatch_knn_gpu(data.as_ref(), k, &params, device.clone().into(), 42, 0).unwrap();

            assert_eq!(got.len(), n, "{}: wrong row count", method);
            for (i, row) in got.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    k,
                    "{}: row {} holds {} != {}",
                    method,
                    i,
                    row.len(),
                    k
                );
                assert!(!row.contains(&i), "row {} contains itself", i);
            }

            let r = recall(&got, &want);
            assert!(r > 0.85, "{}: recall {:.4} below the 0.85 floor", method, r);
        }
    }

    /// Self must never appear in its own neighbour list, and rows must be
    /// truncated to `k`.
    #[test]
    fn test_drop_self_neighbours() {
        let raw = vec![vec![0, 3, 1, 2], vec![1, 0, 2, 3], vec![5, 2, 0, 1]];
        let out = drop_self_neighbours(raw, 2);

        assert_eq!(out, vec![vec![3, 1], vec![0, 2], vec![5, 0]]);
    }

    /// Exhaustive GPU is exact, so it must reproduce the CPU reference. Small
    /// enough to stay ungated.
    #[test]
    fn test_exhaustive_gpu_matches_cpu() {
        let Some(device) = try_device() else { return };

        let (n, dim, k) = (512usize, 16usize, 10usize);
        let data = clustered_data(n, dim, 8);

        let params = KnnParamsGpu {
            knn_method: "exhaustive".to_string(),
            ..Default::default()
        };
        let got =
            dispatch_knn_gpu(data.as_ref(), k, &params, device.clone().into(), 42, 0).unwrap();

        assert_eq!(got.len(), n);
        for (i, row) in got.iter().enumerate() {
            assert_eq!(row.len(), k, "row {} has {} neighbours", i, row.len());
            assert!(!row.contains(&i), "row {} contains itself", i);
        }

        let want = cpu_reference(&data, k);
        let r = recall(&got, &want);
        assert!(r > 0.99, "exhaustive GPU is exact but recall was {:.4}", r);
    }

    /// The high-`k` regime that doublet detection actually runs in. Every
    /// backend is measured against exact CPU neighbours at `k = 150`, which is
    /// an order of magnitude above ordinary single-cell work. CAGRA belongs
    /// here specifically because its beam has to scale with `k`: pinned at the
    /// constant default it returns a recall of 0.10 rather than 1.00, and no
    /// low-`k` test sees that.
    #[test]
    // Moderate: two index builds at n = 4000, dim = 30, k = 150.
    fn test_gpu_knn_recall_at_high_k() {
        let Some(device) = try_device() else { return };

        let (n, dim, k) = (4000usize, 30usize, 150usize);
        let data = clustered_data(n, dim, 12);
        let want = cpu_reference(&data, k);

        for (method, floor) in [("exhaustive", 0.99f32), ("ivf", 0.90), ("cagra", 0.90)] {
            let params = KnnParamsGpu {
                knn_method: method.to_string(),
                ..Default::default()
            };
            let got =
                dispatch_knn_gpu(data.as_ref(), k, &params, device.clone().into(), 42, 0).unwrap();

            // A backend that silently no-ops returns empty or short rows, so
            // check the shape before trusting the recall number.
            assert_eq!(got.len(), n, "{}: wrong row count", method);
            let short = got.iter().filter(|r| r.len() < k).count();
            assert_eq!(
                short, 0,
                "{}: {} of {} rows returned fewer than {} neighbours",
                method, short, n, k
            );

            let r = recall(&got, &want);
            println!("{:>10} GPU: recall {:.4} at k = {}", method, r, k);
            assert!(
                r > floor,
                "{}: recall {:.4} below the {:.2} floor at k = {}",
                method,
                r,
                floor,
                k
            );
        }
    }
}
