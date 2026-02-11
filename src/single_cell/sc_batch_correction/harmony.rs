////////////
// Params //
////////////

/// Parameters for Harmony batch correction.
///
/// ### Fields
///
/// * `k`: number of clusters
/// * `sigma`: cluster diversity weights
/// * `theta`: batch diversity penalties
/// * `lambda`: ridge parameters (None = auto-estimate)
/// * `alpha`: lambda estimation coefficient
/// * `max_iter_harmony`: outer iterations
/// * `max_iter_cluster`: inner k-means iterations
/// * `block_size`: fraction of cells per update block
/// * `epsilon_cluster`: k-means convergence
/// * `epsilon_harmony`: harmony convergence
pub struct HarmonyParams {
    pub k: usize,
    pub sigma: Vec<f32>,
    pub theta: Vec<f32>,
    pub lambda: Option<Vec<f32>>,
    pub alpha: f32,
    pub max_iter_harmony: usize,
    pub max_iter_cluster: usize,
    pub block_size: f32,
    pub epsilon_cluster: f32,
    pub epsilon_harmony: f32,
}
