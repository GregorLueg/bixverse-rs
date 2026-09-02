//! GPU-accelerated approximate nearest neighbour search for single cell data,
//! specifically with Scrublet in mind. The callers here run at high `k`, which
//! is why exhaustive stays the default: CAGRA is at its best on low `k`, and
//! its extraction path is capped by the build-time graph degree.

use ann_search_rs::prelude::CagraGpuSearchParams;
use ann_search_rs::{
    build_exhaustive_index_gpu, build_ivf_index_gpu, build_nndescent_index_gpu,
    extract_nndescent_knn_gpu, query_exhaustive_index_gpu_self, query_ivf_index_gpu_self,
    query_nndescent_index_gpu_self,
};
use cubecl::Runtime;
use faer::MatRef;
use rayon::prelude::*;

use crate::prelude::*;

///////////////
// Constants //
///////////////

/// Default CAGRA graph degree, matching `ann-search-rs`.
///
/// Mirrored rather than imported because the crate does not export it. Only a
/// floor: widening the degree for extraction must never narrow it below what
/// the query path would have built.
const NNDESCENT_GPU_DEFAULT_DEGREE: usize = 30;

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
    /// GPU NN-Descent with a CAGRA navigational graph. Either beam searched or,
    /// with `extract_knn`, handed back as the descent left it. Cheapest of the
    /// three to query, but recall falls away as `k` climbs.
    CagraGpu,
}

/// Parse a GPU kNN method from a string.
///
/// ### Params
///
/// * `s` - Method name, case-insensitive. `"exhaustive"`, `"ivf"` or
///   `"cagra"`, each also accepted with a `_gpu` suffix.
///
/// ### Returns
///
/// The matching [`KnnSearchGpu`], or `None` if the string is unrecognised.
pub fn parse_knn_method_gpu(s: &str) -> Option<KnnSearchGpu> {
    match s.to_lowercase().as_str() {
        "exhaustive" | "exhaustive_gpu" => Some(KnnSearchGpu::ExhaustiveGpu),
        "ivf" | "ivf_gpu" => Some(KnnSearchGpu::IvfGpu),
        "cagra" | "cagra_gpu" | "nndescent_gpu" => Some(KnnSearchGpu::CagraGpu),
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
    /// Which GPU method to use. One of `"exhaustive"`, `"ivf"` or `"cagra"`.
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
    /// upstream. Widened to cover the request when `extract_knn` is set, since
    /// extraction cannot return more neighbours than the graph holds.
    pub graph_k: Option<usize>,
    /// CAGRA: build node degree, the initial degree prior to pruning. Defaults
    /// to `max(k, floor(1.5 * k))` upstream.
    pub k_build: Option<usize>,
    /// CAGRA: number of trees for the forest that seeds the descent.
    pub n_tree: Option<usize>,
    /// CAGRA: termination criterium for the NNDescent iterations.
    pub delta: f32,
    /// CAGRA: sampling rate for the NNDescent iterations.
    pub rho: Option<f32>,
    /// CAGRA: 2-hop refinement sweeps after the descent loop. Buys graph
    /// quality at a linear cost. Defaults to `0` upstream.
    pub refine_knn: Option<usize>,
    /// CAGRA: beam width during querying. Auto-determined if `None`.
    pub beam_width: Option<usize>,
    /// CAGRA: beam search iterations during querying. Auto-determined if
    /// `None`.
    pub max_beam_iters: Option<usize>,
    /// CAGRA: number of entry points during querying. Auto-determined if
    /// `None`.
    pub n_entry_points: Option<usize>,
    /// CAGRA: hand back the graph the descent already built instead of running
    /// a beam search over it. No search runs at all.
    ///
    /// `graph_k` is widened to cover the request, and `beam_width`,
    /// `max_beam_iters` and `n_entry_points` are ignored. No effect on the
    /// other GPU backends.
    pub extract_knn: bool,
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
            extract_knn: false,
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
pub fn dispatch_knn_gpu<R: Runtime>(
    embd: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    device: R::Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let detailed = verbosity.detailed_verbosity();

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
            let index = build_exhaustive_index_gpu::<f32, R>(embd, &params.ann_dist, device)?;
            query_exhaustive_index_gpu_self(&index, k_query, false, detailed)?
        }
        KnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_gpu::<f32, R>(
                embd,
                params.n_list,
                None,
                &params.ann_dist,
                seed,
                detailed,
                device,
            )?;
            query_ivf_index_gpu_self(&index, k_query, params.n_probe, None, false, detailed)?
        }
        KnnSearchGpu::CagraGpu => {
            // Extraction can only hand back what the graph holds, so the degree
            // has to cover the request. Anything the caller pinned wins; a
            // `None` is widened rather than left at the crate's 30.
            let graph_k = if params.extract_knn {
                Some(
                    params
                        .graph_k
                        .unwrap_or(NNDESCENT_GPU_DEFAULT_DEGREE)
                        .max(k_query),
                )
            } else {
                params.graph_k
            };

            let mut index = build_nndescent_index_gpu::<f32, R>(
                embd,
                &params.ann_dist,
                graph_k,
                params.k_build,
                None,
                params.n_tree,
                Some(params.delta),
                params.rho,
                params.refine_knn,
                seed,
                detailed,
                !params.extract_knn,
                device,
            )?;

            if params.extract_knn {
                extract_nndescent_knn_gpu(&index, Some(k_query), true, false)?
            } else {
                // Wrapping the params in `Some(..)` suppresses the crate's own
                // k-scaled fallback, so the beam has to be backfilled here or
                // the raw BEAM_WIDTH of 16 caps the row at 15 after the
                // self-filter, costing 90% of the recall at the `k` these
                // callers run at.
                let k_graph = graph_k.unwrap_or(NNDESCENT_GPU_DEFAULT_DEGREE);
                let scaled_bw = k_query.max(k_graph).max(16) * 2;
                let query_params = CagraGpuSearchParams::new(
                    params.beam_width.or(Some(scaled_bw)),
                    params.max_beam_iters.or(Some(scaled_bw * 3)),
                    params.n_entry_points,
                    None,
                );

                query_nndescent_index_gpu_self(&mut index, k_query, Some(query_params), false)?
            }
        }
    };

    Ok(drop_self_neighbours(raw_indices, k))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use ann_search_rs::{build_exhaustive_index, query_exhaustive_self};
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
        let index = build_exhaustive_index(data.as_ref(), "euclidean");
        let (idx, _) = query_exhaustive_self(&index, k + 1, false, false).unwrap();
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
        assert_eq!(parse_knn_method_gpu("kmknn"), None);
        assert_eq!(parse_knn_method_gpu("annoy"), None);
    }

    /// CAGRA at the low `k` it is actually good at, both exits. The beam search
    /// fills every row; the extraction path is capped by the graph degree and
    /// may not, which is exactly why it gets a separate shape assertion.
    #[test]
    fn test_cagra_gpu_both_exits() {
        let Some(device) = try_device() else { return };

        let (n, dim, k) = (2000usize, 24usize, 15usize);
        let data = clustered_data(n, dim, 10);
        let want = cpu_reference(&data, k);

        for extract in [false, true] {
            let params = KnnParamsGpu {
                knn_method: "cagra".to_string(),
                k,
                extract_knn: extract,
                refine_knn: Some(1),
                ..Default::default()
            };
            let got =
                dispatch_knn_gpu::<WgpuRuntime>(data.as_ref(), k, &params, device.clone(), 42, 0)
                    .unwrap();

            assert_eq!(got.len(), n, "extract = {}: wrong row count", extract);
            for (i, row) in got.iter().enumerate() {
                assert!(row.len() <= k, "row {} returned {} > {}", i, row.len(), k);
                assert!(!row.contains(&i), "row {} contains itself", i);
            }

            let r = recall(&got, &want);
            println!("CAGRA GPU (extract = {}): recall {:.4}", extract, r);
            assert!(
                r > 0.85,
                "extract = {}: recall {:.4} below the 0.85 floor",
                extract,
                r
            );
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
            dispatch_knn_gpu::<WgpuRuntime>(data.as_ref(), k, &params, device, 42, 0).unwrap();

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
                dispatch_knn_gpu::<WgpuRuntime>(data.as_ref(), k, &params, device.clone(), 42, 0)
                    .unwrap();

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
