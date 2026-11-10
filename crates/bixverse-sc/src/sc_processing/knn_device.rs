//! Backend-agnostic parameters for the device kNN searches. The wgpu path
//! (`bixverse-sc-gpu`) and the MLX path (`bixverse-mlx`) do not depend on
//! each other, so the method names and parameters they share live here.

use serde::{Deserialize, Deserializer, de::Error as _};

///////////////
// Constants //
///////////////

/// Default CAGRA graph degree, matching `ann-search-rs`.
///
/// Mirrored rather than imported because the crate does not export it. Sizes
/// the CAGRA beam when the caller leaves `graph_k` unset.
const CAGRA_DEFAULT_DEGREE: usize = 30;

/// Smallest beam width the k-scaled CAGRA beam goes down to, the raw
/// `ann-search-rs` default.
const CAGRA_MIN_BEAM_WIDTH: usize = 16;

//////////
// Enum //
//////////

/// The device (wgpu or MLX) nearest neighbour methods.
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
    /// NN-Descent with a pruned CAGRA navigational graph, always beam
    /// searched, self-query included. The explicit opt-in when the graph alone
    /// is too lossy.
    CagraGpu,
    /// NN-Descent. A self-kNN is the descent graph itself, built without an
    /// index and handed back without a search (on wgpu through the clustered
    /// build, so datasets past the per-binding limit work too), short rows
    /// topped up exactly. A cross-query falls back to
    /// [`KnnSearchGpu::CagraGpu`], since a graph-only build cannot answer it.
    NNDescentGraphGpu,
}

/// Parse a device kNN method from a string.
///
/// ### Params
///
/// * `s` - Method name, case-insensitive. `"exhaustive"`, `"ivf"`,
///   `"nndescent"` or `"cagra"`, each also accepted with a `_gpu` or `_mlx`
///   suffix.
///
/// ### Returns
///
/// The matching [`KnnSearchGpu`], or `None` if the string is unrecognised.
pub fn parse_knn_method_gpu(s: &str) -> Option<KnnSearchGpu> {
    match s.to_lowercase().as_str() {
        "exhaustive" | "exhaustive_gpu" | "exhaustive_mlx" => Some(KnnSearchGpu::ExhaustiveGpu),
        "ivf" | "ivf_gpu" | "ivf_mlx" => Some(KnnSearchGpu::IvfGpu),
        "nndescent" | "nndescent_gpu" | "nndescent_mlx" => Some(KnnSearchGpu::NNDescentGraphGpu),
        "cagra" | "cagra_gpu" | "cagra_mlx" => Some(KnnSearchGpu::CagraGpu),
        _ => None,
    }
}

/// Deserialise a device kNN method name, checked through [`parse_knn_method_gpu`]
///
/// ### Params
///
/// * `d` - The deserialiser, holding a string
///
/// ### Returns
///
/// The name as given, or an error naming the unrecognised value.
pub fn de_knn_method_gpu<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let s = String::deserialize(d)?;
    match parse_knn_method_gpu(&s) {
        Some(_) => Ok(s),
        None => Err(D::Error::custom(format!(
            "knn_method must be one of 'exhaustive', 'ivf', 'nndescent' or 'cagra', \
             got '{s}'"
        ))),
    }
}

/// Deserialise a device distance metric name
///
/// The wgpu and MLX kernels have no Manhattan, so only `"euclidean"` and `"cosine"`
/// pass.
///
/// ### Params
///
/// * `d` - The deserialiser, holding a string
///
/// ### Returns
///
/// The name as given, or an error naming the unsupported value.
pub fn de_ann_dist_gpu<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let s = String::deserialize(d)?;
    match s.as_str() {
        "euclidean" | "cosine" => Ok(s),
        _ => Err(D::Error::custom(format!(
            "ann_dist must be 'euclidean' or 'cosine' on the GPU, got '{s}'"
        ))),
    }
}

////////////
// Params //
////////////

/// Parameters for the device (wgpu and MLX) nearest neighbour searches.
///
/// The device backends take a different parameter set from the CPU ones, so
/// this is a sibling of `KnnParams` rather than a reuse of it. Fields left as
/// `None` fall back to the heuristics documented on each field.
///
/// Deserialises from the keys of the R `params_knn_gpu_defaults()` list;
/// missing keys take the defaults.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct KnnParamsGpu {
    /// Which device method to use. One of `"exhaustive"`, `"ivf"`,
    /// `"nndescent"` or `"cagra"`.
    #[serde(deserialize_with = "de_knn_method_gpu")]
    pub knn_method: String,
    /// Distance metric. One of `"euclidean"` or `"cosine"`. Manhattan is not
    /// supported by the device kernels.
    #[serde(deserialize_with = "de_ann_dist_gpu")]
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
    /// NNDescent and CAGRA: build node degree, the initial degree prior
    /// to pruning. Defaults to `max(k, floor(1.5 * k))` upstream.
    pub k_build: Option<usize>,
    /// NNDescent and CAGRA: number of trees for the forest that seeds
    /// the descent.
    pub n_tree: Option<usize>,
    /// NNDescent and CAGRA: termination criterium for the NNDescent
    /// iterations.
    pub delta: f32,
    /// NNDescent and CAGRA: sampling rate for the NNDescent iterations.
    pub rho: Option<f32>,
    /// NNDescent and CAGRA: 2-hop refinement sweeps after the descent
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
/// Handing explicit params to the query suppresses the crate's own k-scaled
/// fallback, so the beam has to be backfilled here or the raw beam width of
/// 16 caps the row at 15 after the self-filter, costing most of the recall at
/// the `k` these callers run at. Shared by the wgpu and MLX dispatches.
///
/// ### Params
///
/// * `params` - The [`KnnParamsGpu`] for this run. Any beam field the caller
///   pinned wins over the scaled value.
/// * `k_query` - Number of neighbours asked of the index, self included.
///
/// ### Returns
///
/// `(beam_width, max_beam_iters, n_entry_points)`.
pub fn cagra_beam(
    params: &KnnParamsGpu,
    k_query: usize,
) -> (Option<usize>, Option<usize>, Option<usize>) {
    let k_graph = params.graph_k.unwrap_or(CAGRA_DEFAULT_DEGREE);
    let scaled_bw = k_query.max(k_graph).max(CAGRA_MIN_BEAM_WIDTH) * 2;

    (
        params.beam_width.or(Some(scaled_bw)),
        params.max_beam_iters.or(Some(scaled_bw * 3)),
        params.n_entry_points,
    )
}
