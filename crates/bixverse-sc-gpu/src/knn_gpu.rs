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
use bixverse_sc::sc_processing::knn::knn_graph_to_rows;
use cubecl::Device;
use faer::MatRef;
use rayon::prelude::*;

use crate::prelude::*;

///////////////
// Constants //
///////////////

/// Default CAGRA graph degree, matching `ann-search-rs`.
///
/// Mirrored rather than imported because the crate does not export it. Sizes
/// the CAGRA beam when the caller leaves `graph_k` unset.
const NNDESCENT_GPU_DEFAULT_DEGREE: usize = 30;

///////////
// Types //
///////////

/// Neighbour indices from a GPU cross-query and, if requested, the distances.
pub(crate) type GpuKnnQueryResult = Result<(Vec<Vec<usize>>, Option<Vec<Vec<f32>>>), ScGpuErrors>;

//////////
// Enum //
//////////

/// The GPU-accelerated approximate nearest neighbour methods.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KnnSearchGpu {
    /// Brute-force GPU search. Exact and parameter-free, but quadratic in `n`.
    /// The default, because exactness is usually worth more to the callers here
    /// than wall-clock.
    #[default]
    ExhaustiveGpu,
    /// Inverted file index. Approximate, cost flat in `k`, and the faster of
    /// the two from roughly n = 50k upwards.
    IvfGpu,
    /// GPU NN-Descent with a CAGRA navigational graph, beam searched. Cheap to
    /// query, but recall falls away as `k` climbs.
    CagraGpu,
    /// The GPU NN-Descent kNN graph itself, built through the clustered build
    /// without an index and handed back without a search, so datasets past the
    /// per-binding limit work too. Self-kNN only: the cross-query paths fall
    /// back to [`KnnSearchGpu::CagraGpu`]. Rows can come back shorter than `k`
    /// where the descent never filled them.
    NNDescentGraphGpu,
}

/// Parse a GPU kNN method from a string.
///
/// ### Params
///
/// * `s` - Method name, case-insensitive. `"exhaustive"`, `"ivf"`, `"cagra"`
///   or `"nndescent_graph"`, each also accepted with a `_gpu` suffix.
///   `"nndescent"` and `"nndescent_gpu"` map to CAGRA, whose graph is built by
///   NN-descent.
///
/// ### Returns
///
/// The matching [`KnnSearchGpu`], or `None` if the string is unrecognised.
pub fn parse_knn_method_gpu(s: &str) -> Option<KnnSearchGpu> {
    match s.to_lowercase().as_str() {
        "exhaustive" | "exhaustive_gpu" => Some(KnnSearchGpu::ExhaustiveGpu),
        "ivf" | "ivf_gpu" => Some(KnnSearchGpu::IvfGpu),
        "cagra" | "cagra_gpu" | "nndescent" | "nndescent_gpu" => Some(KnnSearchGpu::CagraGpu),
        "nndescent_graph" | "nndescent_graph_gpu" => Some(KnnSearchGpu::NNDescentGraphGpu),
        _ => None,
    }
}

////////////
// Params //
////////////

/// Parameters for the GPU nearest neighbour searches.
///
/// The GPU backends take a different parameter set from the CPU ones, so this
/// is a sibling of `KnnParams` rather than a reuse of it. Fields left as
/// `None` fall back to the heuristics documented on each field.
#[derive(Clone, Debug)]
pub struct KnnParamsGpu {
    /// Which GPU method to use. One of `"exhaustive"`, `"ivf"`, `"cagra"` or
    /// `"nndescent_graph"`.
    pub knn_method: String,
    /// Distance metric. One of `"euclidean"` or `"cosine"`. Manhattan is not
    /// supported by the GPU kernels.
    pub ann_dist: String,
    /// Number of neighbours to return.
    pub k: usize,
    /// IVF: number of lists. Defaults to `sqrt(n)` upstream.
    pub n_list: Option<usize>,
    /// IVF: number of lists to probe. Defaults to `sqrt(n_list)` upstream.
    pub n_probe: Option<usize>,
    /// CAGRA: final node degree of the graph after pruning. Defaults to 30
    /// upstream.
    pub graph_k: Option<usize>,
    /// CAGRA and NNDescent graph: build node degree, the initial degree prior
    /// to pruning. Defaults to `max(k, floor(1.5 * k))` upstream.
    pub k_build: Option<usize>,
    /// CAGRA and NNDescent graph: number of trees for the forest that seeds
    /// the descent.
    pub n_tree: Option<usize>,
    /// CAGRA and NNDescent graph: termination criterium for the NNDescent
    /// iterations.
    pub delta: f32,
    /// CAGRA and NNDescent graph: sampling rate for the NNDescent iterations.
    pub rho: Option<f32>,
    /// CAGRA and NNDescent graph: 2-hop refinement sweeps after the descent
    /// loop. Buys graph quality at a linear cost. Defaults to `0` upstream.
    pub refine_knn: Option<usize>,
    /// CAGRA: beam width during querying. Auto-determined if `None`.
    pub beam_width: Option<usize>,
    /// CAGRA: beam search iterations during querying. Auto-determined if
    /// `None`.
    pub max_beam_iters: Option<usize>,
    /// CAGRA: number of entry points during querying. Auto-determined if
    /// `None`.
    pub n_entry_points: Option<usize>,
}

impl KnnParamsGpu {
    /// Generate a version of this with sensible base parameters.
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new() -> Self {
        Self {
            knn_method: "exhaustive".to_string(),
            ann_dist: "euclidean".to_string(),
            k: 15,
            n_list: None,
            n_probe: None,
            graph_k: None,
            k_build: None,
            n_tree: None,
            delta: 0.001,
            rho: None,
            refine_knn: None,
            beam_width: None,
            max_beam_iters: None,
            n_entry_points: None,
        }
    }
}

/// Default implementation for KnnParamsGpu
impl Default for KnnParamsGpu {
    fn default() -> Self {
        Self::new()
    }
}

/////////////
// Helpers //
/////////////

/// Beam parameters for a CAGRA query, with the beam scaled to `k`.
///
/// Wrapping the params in `Some(..)` suppresses the crate's own k-scaled
/// fallback, so the beam has to be backfilled here or the raw BEAM_WIDTH of 16
/// caps the row at 15 after the self-filter, costing 90% of the recall at the
/// `k` these callers run at.
///
/// ### Params
///
/// * `params` - The [`KnnParamsGpu`] for this run. Any beam field the caller
///   pinned wins over the scaled value.
/// * `k_query` - Number of neighbours asked of the index, self included.
/// * `graph_k` - The resolved graph degree, or `None` for the crate default.
///
/// ### Returns
///
/// The [`CagraGpuSearchParams`] to hand to the query.
pub(crate) fn cagra_query_params(
    params: &KnnParamsGpu,
    k_query: usize,
    graph_k: Option<usize>,
) -> CagraGpuSearchParams {
    let k_graph = graph_k.unwrap_or(NNDESCENT_GPU_DEFAULT_DEGREE);
    let scaled_bw = k_query.max(k_graph).max(16) * 2;

    CagraGpuSearchParams::new(
        params.beam_width.or(Some(scaled_bw)),
        params.max_beam_iters.or(Some(scaled_bw * 3)),
        params.n_entry_points,
        None,
    )
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

    let (raw_indices, _) = match method {
        KnnSearchGpu::ExhaustiveGpu => {
            let index =
                build_exhaustive_index_gpu::<f32>(embd, &params.ann_dist, device, ann_verbosity)?;
            query_exhaustive_index_gpu_self(&index, k_query, false, ann_verbosity)?
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
            query_ivf_index_gpu_self(&index, k_query, params.n_probe, None, false, ann_verbosity)?
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
            let query_params = cagra_query_params(params, k_query, params.graph_k);

            query_nndescent_index_gpu_self(
                &mut index,
                k_query,
                Some(query_params),
                false,
                ann_verbosity,
            )?
        }
        KnnSearchGpu::NNDescentGraphGpu => {
            // No self-edge requested, so `drop_self_neighbours` below only
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
                false,
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
            knn_graph_to_rows(graph)
        }
    };

    Ok(drop_self_neighbours(raw_indices, k))
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
/// * `params` - The [`KnnParamsGpu`] for this run. `k` is ignored, and the
///   NNDescent graph method runs as CAGRA, since a graph-only build cannot
///   answer a cross-query.
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

            let query_params = cagra_query_params(params, k, params.graph_k);

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
            parse_knn_method_gpu("nndescent_gpu"),
            Some(KnnSearchGpu::CagraGpu)
        );
        assert_eq!(
            parse_knn_method_gpu("nndescent"),
            Some(KnnSearchGpu::CagraGpu)
        );
        assert_eq!(
            parse_knn_method_gpu("NNDescent_Graph_GPU"),
            Some(KnnSearchGpu::NNDescentGraphGpu)
        );
        assert_eq!(parse_knn_method_gpu("kmknn"), None);
        assert_eq!(parse_knn_method_gpu("annoy"), None);
    }

    /// CAGRA and the graph-only NN-Descent build at the low `k` they are
    /// actually good at. The beam search fills every row; the graph path may
    /// not, which is why the shape assertion only bounds the row length.
    #[test]
    fn test_cagra_and_nndescent_graph_gpu() {
        let Some(device) = try_device() else { return };

        let (n, dim, k) = (2000usize, 24usize, 15usize);
        let data = clustered_data(n, dim, 10);
        let want = cpu_reference(&data, k);

        for method in ["cagra", "nndescent_graph"] {
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
                assert!(row.len() <= k, "row {} returned {} > {}", i, row.len(), k);
                assert!(!row.contains(&i), "row {} contains itself", i);
            }

            let r = recall(&got, &want);
            println!("{} GPU: recall {:.4}", method, r);
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
