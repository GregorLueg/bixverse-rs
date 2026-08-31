//! Implementation of the HotSpot method to identify genes that vary across
//! various potential graphs in single cell 'omics, see DeTomaso and Yosef,
//! Cell Syst., 2021

use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::{Accum, Mat, MatRef};
use indexmap::IndexSet;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use statrs::distribution::{ContinuousCDF, Normal};
use std::cmp::Ordering;
use std::time::Instant;

use crate::core::math::linear_algebra::{linear_regression, linear_regression_widen};
use crate::core::math::stats::{inv_logit, logit, p_adjust_fdr, z_scores_to_pval};
use crate::prelude::*;

use crate::single_cell::sc_processing::knn::knn_distance_weights;
use crate::single_cell::sc_utils::simd::*;
use crate::utils::faer_parallelism;
use crate::utils::simd::{
    sum_squared_dev_widen_simd_f32, sum_squares_simd_f32, sum_widen_simd_f32,
};

/////////////
// Hotspot //
/////////////

////////////
// Consts //
////////////

/// Gene batch size for Hotspot streaming
const GENE_BATCH_SIZE: usize = 1000;

///////////
// Types //
///////////

/// Loaded panel: centred counts, wy (both n_cells x panel), eg2, per-gene max.
type PanelData = (Mat<f32>, Mat<f32>, Vec<f32>, Vec<f32>);

////////////
// Params //
////////////

/// HotSpot parameters
pub struct HotSpotParams {
    /// The model to use for modelling the GEX. Choice of `"danb"`,
    /// `"bernoulli"` or `"normal"`.
    pub model: String,
    /// Shall the data be normalised
    pub normalise: bool,
    /// [KnnParams] for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
    /// How the kNN distances become edge weights, see [HotSpotGraphParams]
    pub graph_params: HotSpotGraphParams,
}

/// How the kNN distances are turned into edge weights.
///
/// Mirrors `Hotspot.create_knn_graph` in the reference implementation, right
/// down to the default: upstream ships `weighted_graph=False`, so every
/// retained edge weighs one and the distances only decide who is a neighbour.
#[derive(Clone, Copy, Debug)]
pub struct HotSpotGraphParams {
    /// Apply the Gaussian kernel of [`knn_distance_weights`] to the neighbour
    /// distances. `false` gives every edge a weight of one.
    pub weighted_graph: bool,
    /// Kernel width is the `ceil(k / neighborhood_factor)`-th neighbour
    /// distance. Only read when `weighted_graph` is `true`.
    pub neighborhood_factor: f32,
    /// Whether the supplied distances already hold `d^2`. Depends entirely on
    /// the metric the neighbours came from, see
    /// [`distances_are_squared`]. Only read when `weighted_graph` is `true`.
    pub squared_distances: bool,
}

impl Default for HotSpotGraphParams {
    fn default() -> Self {
        Self {
            weighted_graph: false,
            neighborhood_factor: 3.0,
            // upstream runs on scikit-learn / pynndescent distances, which are
            // never pre-squared
            squared_distances: false,
        }
    }
}

impl HotSpotGraphParams {
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// * `weighted_graph` - Weight the edges by the Gaussian kernel
    /// * `neighborhood_factor` - Divisor picking the kernel width neighbour
    /// * `squared_distances` - `true` when the distances already hold `d^2`
    ///
    /// ### Returns
    ///
    /// The initialised parameters.
    pub fn new(weighted_graph: bool, neighborhood_factor: f32, squared_distances: bool) -> Self {
        Self {
            weighted_graph,
            neighborhood_factor,
            squared_distances,
        }
    }
}

/////////////
// Helpers //
/////////////

/// Gene expression module to use for HotSpot
#[derive(Debug, Clone)]
pub enum GexModel {
    /// Use depth-adjusted negative binomial model
    DephAdjustNegBinom,
    /// Uses Bernoulli distribution to model prediction probability
    Bernoulli,
    /// Use depth-adjusted normal model
    Normal,
    /// Assume the caller already standardised the data: `mu = 0`, `var = 1`.
    ///
    /// The escape hatch for anything the other three cannot express, since none
    /// of them accept covariates. Standardise within batch, cell line or
    /// whatever else needs regressing out, then run with this.
    PreStandardised,
}

/// Parse the model to use gene expression
///
/// ### Params
///
/// * `s` - Type of model to use the model
///
/// ### Returns
///
/// The [GexModel] to use, or `None` when the string names no model.
pub fn parse_gex_model(s: &str) -> Option<GexModel> {
    match s.to_lowercase().as_str() {
        "danb" => Some(GexModel::DephAdjustNegBinom),
        "bernoulli" => Some(GexModel::Bernoulli),
        "normal" => Some(GexModel::Normal),
        "none" => Some(GexModel::PreStandardised),
        _ => None,
    }
}

/////////////
// Results //
/////////////

/// Structure for the gene results
#[derive(Debug, Clone)]
pub struct HotSpotGeneRes {
    /// Gene index of the analysed gene
    pub gene_idx: Vec<usize>,
    /// Geary's C statistic for this gene
    pub c: Vec<f64>,
    /// Z-score for this gene
    pub z: Vec<f64>,
    /// P-value for this gene
    pub pval: Vec<f64>,
    /// FDR for this gene
    pub fdr: Vec<f64>,
}

/// Structure for pair-wise correlations
#[derive(Debug, Clone)]
pub struct HotSpotPairRes {
    /// Symmetric matrix with cor coefficients (N_genes x N_genes)
    pub cor: Mat<f32>,
    /// Symmetric matrix with Z scores (N_genex x N_genes)
    pub z_scores: Mat<f32>,
}

/////////////////
// GeneScratch //
/////////////////

/// Per-thread reusable buffers for the autocorrelation path.
#[derive(Debug, Clone)]
struct GeneScratch {
    /// Dense gene values: raw counts, optionally centred in place.
    vals: Vec<f32>,
    /// Per-cell mean from the model fit.
    mu: Vec<f32>,
    /// Per-cell variance from the model fit.
    var: Vec<f32>,
    /// Per-cell second moment from the model fit.
    x2: Vec<f32>,
    /// Scratch for mu² (uncentered only).
    mu_sq: Vec<f32>,
    /// Scratch for W @ mu (uncentered only).
    t1: Vec<f32>,
    /// Scratch for W² @ mu² (uncentered only).
    u_musq: Vec<f32>,
    /// Scratch for W² @ x2 (uncentered only).
    u_x2: Vec<f32>,
}

impl GeneScratch {
    /// Allocate scratch for one worker.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells
    /// * `need_moments` - Allocate the four moment buffers (uncentered path)
    ///
    /// ### Returns
    ///
    /// Zeroed scratch buffers.
    fn new(n_cells: usize, need_moments: bool) -> Self {
        let opt = |n: usize| {
            if need_moments {
                vec![0.0_f32; n]
            } else {
                Vec::new()
            }
        };
        Self {
            vals: vec![0.0_f32; n_cells],
            mu: vec![0.0_f32; n_cells],
            var: vec![0.0_f32; n_cells],
            x2: vec![0.0_f32; n_cells],
            mu_sq: opt(n_cells),
            t1: opt(n_cells),
            u_musq: opt(n_cells),
            u_x2: opt(n_cells),
        }
    }
}

///////////////
// Graph CSR //
///////////////

/// Symmetric graph in CSR layout for the pair path.
///
/// Built from the non-redundant (upper-triangular combined) weights produced by
/// [make_weights_non_redundant]. Each undirected edge is stored in *both* rows
/// with its combined weight, which is what makes both statistics a single
/// sparse mat-vec `wy = W_sym @ c`:
///
/// - `lc(x, y) = dot(x, W_sym @ y)` (the pair test statistic, no extra factor)
/// - `eg2(x)   = sum_squares(W_sym @ x)`
#[derive(Clone, Debug)]
struct GraphCsr {
    /// Row offsets, length n_nodes + 1.
    offsets: Vec<usize>,
    /// Column indices (the neighbour node), compacted (no zero-weight entries).
    indices: Vec<u32>,
    /// Edge weights, aligned with `indices`.
    weights: Vec<f32>,
}

impl GraphCsr {
    /// Build the symmetric CSR from the non-redundant neighbour/weight arrays.
    ///
    /// Iterates the non-zero (canonical) entries of `weights` and scatters each
    /// into both endpoints' rows. Zero-weight reciprocal entries are dropped.
    ///
    /// ### Params
    ///
    /// * `neighbours` - Neighbour indices for each node
    /// * `weights` - Non-redundant edge weights for each neighbour connection
    ///
    /// ### Returns
    ///
    /// A new `GraphCsr` with symmetric edges in CSR layout.
    fn from_non_redundant(neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> Self {
        let n = neighbours.len();
        let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];

        for i in 0..n {
            for (k, &j) in neighbours[i].iter().enumerate() {
                let w = weights[i][k];
                if w == 0.0 {
                    continue;
                }
                rows[i].push((j as u32, w));
                rows[j].push((i as u32, w));
            }
        }

        let mut offsets = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut ws = Vec::new();
        offsets.push(0);
        for row in &rows {
            for &(j, w) in row {
                indices.push(j);
                ws.push(w);
            }
            offsets.push(indices.len());
        }

        Self {
            offsets,
            indices,
            weights: ws,
        }
    }

    /// Number of nodes in the graph
    ///
    /// ### Returns
    ///
    /// The number of nodes.
    #[inline]
    fn n_nodes(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Sparse matrix-vector product: `out = W_sym @ c`
    ///
    /// Multiplies the symmetric CSR graph by a dense vector.
    ///
    /// ### Params
    ///
    /// * `c` - Input vector of length `n_nodes`
    /// * `out` - Output vector of length `n_nodes`, written in place
    ///
    /// ### Returns
    ///
    /// Nothing; writes result into `out`.
    fn spmv(&self, c: &[f32], out: &mut [f32]) {
        for i in 0..self.n_nodes() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let mut acc = 0.0_f32;
            for k in start..end {
                acc += self.weights[k] * c[self.indices[k] as usize];
            }
            out[i] = acc;
        }
    }

    /// Quadratic form `cᵀ W_sym c`, fused (no intermediate `W_sym @ c`).
    ///
    /// The autocorrelation statistic is `g = 0.5 * quadratic_form(vals)`,
    /// because `W_sym` carries each undirected edge in both rows. Fusing the
    /// mat-vec and the dot means the neighbour gather happens once and nothing
    /// is allocated.
    ///
    /// ### Params
    ///
    /// * `c` - Input vector of length `n_nodes`
    ///
    /// ### Returns
    ///
    /// The scalar `cᵀ W_sym c`.
    fn quadratic_form(&self, c: &[f32]) -> f32 {
        let mut total = 0.0_f32;
        for i in 0..self.n_nodes() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let mut acc = 0.0_f32;
            for k in start..end {
                acc += self.weights[k] * c[self.indices[k] as usize];
            }
            total += c[i] * acc;
        }
        total
    }

    /// Sparse mat-vec against the squared weights: `out = W_sym² @ c`.
    ///
    /// Same traversal as [GraphCsr::spmv] with each weight squared. Used by the
    /// moment computation for the `wᵢⱼ²` accumulators.
    ///
    /// ### Params
    ///
    /// * `c` - Input vector of length `n_nodes`
    /// * `out` - Output vector of length `n_nodes`, written in place
    ///
    /// ### Returns
    ///
    /// Nothing; writes result into `out`.
    fn spmv_sq(&self, c: &[f32], out: &mut [f32]) {
        for i in 0..self.n_nodes() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let mut acc = 0.0_f32;
            for k in start..end {
                let w = self.weights[k];
                acc += w * w * c[self.indices[k] as usize];
            }
            out[i] = acc;
        }
    }
}

//////////////////////
// Momentum weights //
//////////////////////

/// Expected value and expected squared value of the local covariance under the
/// null, computed via CSR graph products.
///
/// ### Params
///
/// * `graph` - Symmetric CSR graph
/// * `mu` - Mean expression per cell
/// * `x2` - Second moment per cell
/// * `mu_sq` - Scratch, overwritten with mu²
/// * `t1` - Scratch, overwritten with W @ mu
/// * `u_musq` - Scratch, overwritten with W² @ mu²
/// * `u_x2` - Scratch, overwritten with W² @ x2
///
/// ### Returns
///
/// Tuple `(eg, eg2)`.
fn compute_moments_weights(
    graph: &GraphCsr,
    mu: &[f32],
    x2: &[f32],
    mu_sq: &mut [f32],
    t1: &mut [f32],
    u_musq: &mut [f32],
    u_x2: &mut [f32],
) -> (f32, f32) {
    let n = mu.len();

    for i in 0..n {
        mu_sq[i] = mu[i] * mu[i];
    }

    graph.spmv(mu, t1);
    graph.spmv_sq(mu_sq, u_musq);
    graph.spmv_sq(x2, u_x2);

    let mut eg = 0.0_f32;
    for i in 0..n {
        eg += mu[i] * t1[i];
    }
    eg *= 0.5;

    let mut eg2 = 0.0_f32;
    for i in 0..n {
        eg2 += (x2[i] - mu_sq[i]) * (t1[i] * t1[i] - u_musq[i]);
    }

    let mut b1 = 0.0_f32;
    let mut b2 = 0.0_f32;
    for i in 0..n {
        b1 += x2[i] * u_x2[i];
        b2 += mu_sq[i] * u_musq[i];
    }
    eg2 += 0.5 * (b1 - b2);

    eg2 += eg * eg;
    (eg, eg2)
}

/// Turn neighbour distances into the edge weights the statistics run on.
///
/// ### Params
///
/// * `distances` - Neighbour distances per node, ascending, self excluded
/// * `params` - See [HotSpotGraphParams]
///
/// ### Returns
///
/// One weight per neighbour, in the same layout as `distances`.
fn graph_weights(distances: &[Vec<f32>], params: &HotSpotGraphParams) -> Vec<Vec<f32>> {
    if params.weighted_graph {
        knn_distance_weights(
            distances,
            params.neighborhood_factor,
            params.squared_distances,
        )
    } else {
        distances
            .iter()
            .map(|row| vec![1.0_f32; row.len()])
            .collect()
    }
}

/// Combine the two directions of every reciprocal edge onto one slot
///
/// Self-loops are dropped. Upstream has no guard because it builds its own kNN
/// and drops the query point, but this crate accepts a caller-supplied graph,
/// and a kNN that returns each node as its own first neighbour would add
/// `w * x_i^2` to `G` for every gene: a uniform positive bias that looks
/// exactly like universal spatial autocorrelation.
///
/// Note that the result is *not* upper-triangular. A non-mutual edge where `i`
/// lists `j` but `j` does not list `i` is left in place even when `j < i`,
/// which matches upstream and is what every downstream traversal assumes.
///
/// ### Params
///
/// * `neighbours` - Neighbour indices for each node
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Modified weights with redundant edges and self-loops zeroed
fn make_weights_non_redundant(neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut w_no_redundant = weights.to_vec();

    for i in 0..neighbours.len() {
        for k in 0..neighbours[i].len() {
            let j = neighbours[i][k];

            if j == i {
                w_no_redundant[i][k] = 0.0;
                continue;
            }

            if j < i {
                continue;
            }

            // check if j has i as a neighbour
            for k2 in 0..neighbours[j].len() {
                if neighbours[j][k2] == i {
                    let w_ji = w_no_redundant[j][k2];
                    w_no_redundant[j][k2] = 0.0;
                    w_no_redundant[i][k] += w_ji;
                    break;
                }
            }
        }
    }

    w_no_redundant
}

/// Compute node degree from edge weights
///
/// Calculates the degree (sum of incident edge weights) for each node.
/// Each edge contributes to the degree of both its endpoints.
///
/// ### Params
///
/// * `neighbours` - Neighbour indices for each node
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Vector of degree values for each node
fn compute_node_degree(neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> Vec<f32> {
    let mut d = vec![0.0_f32; neighbours.len()];

    for i in 0..neighbours.len() {
        for k in 0..neighbours[i].len() {
            let j = neighbours[i][k];
            let w_ij = weights[i][k];

            d[i] += w_ij;
            d[j] += w_ij;
        }
    }

    d
}

/// Compute maximum possible local covariance
///
/// Calculates the theoretical maximum value of the local covariance statistic
/// given the node degrees and expression values. Used to normalise Geary's C.
///
/// ### Params
///
/// * `node_degrees` - Sum of edge weights for each node
/// * `vals` - Gene expression values for each cell
///
/// ### Returns
///
/// Maximum possible local covariance
fn compute_local_cov_max(node_degrees: &[f32], vals: &[f32]) -> f32 {
    fused_mul_square_sum_simd(node_degrees, vals) / 2.0
}

/// Center (Z-score) the values
///
/// Transforms values to have zero mean and unit variance using the
/// provided stats.
///
/// ### Params
///
/// * `vals` - Mutable reference to the values to scale
/// * `mu` - The mean values
/// * `var` - The variance of the values
///
/// ### Returns
///
/// Nothing; modifies `vals` in place.
fn center_values(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    assert_same_len!(vals, mu, var);

    center_values_simd(vals, mu, var);
}

//////////////////
// Corr helpers //
//////////////////

/// Replace counts by their detection indicator, in place.
///
/// ### Params
///
/// * `vals` - Dense counts, overwritten with `1.0` where positive
///
/// ### Returns
///
/// Nothing; modifies `vals` in place.
#[inline]
fn binarise(vals: &mut [f32]) {
    for v in vals.iter_mut() {
        *v = if *v > 0.0 { 1.0 } else { 0.0 };
    }
}

/// Centre gene counts for correlation computation
///
/// Standardises gene expression using the specified model, transforming to
/// zero mean and unit variance.
///
/// ### Params
///
/// * `gene` - Reference to gene expression data
/// * `umi_counts` - Total UMI counts per cell
/// * `n_cells` - Number of cells
/// * `model` - Statistical model to use
///
/// ### Returns
///
/// Vector of centred expression values
fn create_centered_counts_gene(
    gene: &CscGeneChunk,
    umi_counts: &[f32],
    n_cells: usize,
    model: &GexModel,
) -> Vec<f32> {
    let mut vals = vec![0_f32; n_cells];
    for (&idx, val) in gene.indices.iter().zip(gene.data_raw.iter()) {
        vals[idx as usize] = val as f32;
    }

    // Bernoulli models detection, so it fits and centres the indicator rather
    // than the counts, see `local_stats_pairs.py:417`.
    if matches!(model, GexModel::Bernoulli) {
        binarise(&mut vals);
    }

    let (mu, var, _) = match model {
        GexModel::DephAdjustNegBinom => danb_model(gene, umi_counts, n_cells),
        GexModel::Bernoulli => bernoulli_model(gene, umi_counts, n_cells),
        GexModel::Normal => normal_model(gene, umi_counts, n_cells),
        GexModel::PreStandardised => return vals,
    };

    center_values(&mut vals, &mu, &var);

    vals
}

////////////////
// DANB model //
////////////////

/// Depth-adjusted negative binomial (DANB) model
///
/// Fits a negative binomial distribution to gene expression data, adjusting
/// for sequencing depth differences between cells.
///
/// ### Params
///
/// * `gene` - Reference to the CscGeneChunk on which to apply the model.
/// * `umi_counts` - Slice of the UMI counts across these cells (i.e.,
///   sequencing depth).
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// Tuple of (mu, var, x2) where:
/// - mu: Mean expression for each cell
/// - var: Variance for each cell
/// - x2: Second moment (var + mu²) for each cell
fn danb_model(
    gene: &CscGeneChunk,
    umi_counts: &[f32],
    n_cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = n_cells as f64;
    // f64 throughout the moment matching, see `Hotspot::fit_danb`
    let total = sum_widen_simd_f32(umi_counts);
    let tj: f64 = gene.data_raw.iter().map(|x| x as f64).sum();

    let scale = (tj / total) as f32;
    let mu: Vec<f32> = umi_counts.iter().map(|&ti| scale * ti).collect();

    // Build dense array for O(1) lookups
    let mut data_dense = vec![0.0f32; n_cells];
    for (&idx, val) in gene.indices.iter().zip(gene.data_raw.iter()) {
        data_dense[idx as usize] = val as f32;
    }

    let mut sum_sq = 0_f64;
    for i in 0..n_cells {
        let diff = (data_dense[i] - mu[i]) as f64;
        sum_sq += diff * diff;
    }

    let vv = sum_sq / (n - 1.0);
    let tis_sq_sum = sum_squared_dev_widen_simd_f32(umi_counts, 0.0);
    let mut size = ((tj * tj) / total) * (tis_sq_sum / total) / ((n - 1.0) * vv - tj);

    if size < 0.0 {
        size = 1e9;
    } else if size < 1e-10 {
        size = 1e-10;
    }
    let size = size as f32;

    let var: Vec<f32> = mu.iter().map(|&m| m * (1.0 + m / size)).collect();
    let x2: Vec<f32> = var.iter().zip(&mu).map(|(&v, &m)| v + m * m).collect();

    (mu, var, x2)
}

/////////////////////
// Bernoulli model //
/////////////////////

/// Bin gene detections by UMI count bins
///
/// Calculates the detection rate within each bin, applying Laplace smoothing
/// to handle edge cases (0% or 100% detection).
///
/// ### Params
///
/// * `detected_gene` - Binary detection indicators (0 or 1) for each cell
/// * `umi_count_bins` - Bin assignment for each cell
/// * `n_bins` - Total number of bins
///
/// ### Returns
///
/// Vector of detection rates per bin (with Laplace smoothing)
fn bin_gene_detection(detected_gene: &[f32], umi_count_bins: &[usize], n_bins: usize) -> Vec<f32> {
    let mut bin_detects = vec![0_f32; n_bins];
    let mut bin_totals = vec![0_f32; n_bins];

    for i in 0..detected_gene.len() {
        let bin_i = umi_count_bins[i];
        bin_detects[bin_i] += detected_gene[i];
        bin_totals[bin_i] += 1.0;
    }

    // laplace smoothing
    bin_detects
        .iter()
        .zip(&bin_totals)
        .map(|(&d, &t)| (d + 1.0) / (t + 2.0))
        .collect()
}

/// Quantile-based binning with duplicate edge handling
///
/// Generates quantile-based bins from data, dropping duplicate bin edges when
/// they would result in empty bins.
///
/// ### Params
///
/// * `data` - Input data to bin
/// * `n_bins` - Target number of bins
///
/// ### Returns
///
/// Tuple of (bin_assignments, bin_edges) where:
/// - bin_assignments: Vector of bin indices for each data point
/// - bin_edges: Vector of bin edge values (length = n_bins + 1)
fn quantile_cut(data: &[f32], n_bins: usize) -> (Vec<usize>, Vec<f32>) {
    let mut data_sorted = data.to_vec();
    data_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data_sorted.len();
    let mut edges = vec![data_sorted[0]];

    for i in 1..n_bins {
        let idx = (i * n) / n_bins;
        let value = data_sorted[idx.min(n - 1)];

        if value > *edges.last().unwrap() {
            edges.push(value);
        }
    }

    let max_val = data_sorted[n - 1];
    if max_val > *edges.last().unwrap() {
        edges.push(max_val + 1e-6);
    } else {
        *edges.last_mut().unwrap() += 1e-6;
    }

    let n_actual_bins = edges.len() - 1;

    // binary search is faster here...
    let bin_assignments: Vec<usize> = data
        .iter()
        .map(|&x| {
            edges
                .partition_point(|&edge| edge <= x)
                .saturating_sub(1)
                .min(n_actual_bins - 1)
        })
        .collect();

    (bin_assignments, edges)
}

/// Bernoulli model for gene expression
///
/// Models the probability of detecting gene expression using a Bernoulli
/// distribution. Fits a logistic regression model on binned UMI counts to
/// predict detection probability.
///
/// ### Params
///
/// * `gene` - Reference to the CscGeneChunk containing gene expression data
/// * `umi_counts` - Total UMI counts per cell
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// Tuple of (mu, var, x2) where:
/// - mu: Detection probability for each cell
/// - var: Variance (p * (1-p)) for each cell
/// - x2: Second moment (equal to mu for Bernoulli)
fn bernoulli_model(
    gene: &CscGeneChunk,
    umi_counts: &[f32],
    n_cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    const N_BIN_TARGET: usize = 30;

    let mut detected_gene = vec![0_f32; n_cells];
    for idx in &gene.indices {
        detected_gene[*idx as usize] = 1.0;
    }

    let log_umi: Vec<f32> = umi_counts
        .iter()
        .map(|&x| if x > 0.0 { x.log10() } else { 0.0 })
        .collect();

    let (umi_count_bins, bin_edges) = quantile_cut(&log_umi, N_BIN_TARGET);
    let n_bins = bin_edges.len() - 1;

    let bin_centers: Vec<f32> = (0..n_bins)
        .map(|i| (bin_edges[i] + bin_edges[i + 1]) / 2.0)
        .collect();

    let bin_detects = bin_gene_detection(&detected_gene, &umi_count_bins, n_bins);

    let lbin_detects: Vec<f32> = bin_detects.iter().map(|&p| logit(p)).collect();
    let coef = linear_regression(&bin_centers, &lbin_detects);

    let mu: Vec<f32> = log_umi
        .iter()
        .map(|&log_u| inv_logit(coef.0 + coef.1 * log_u))
        .collect();

    let var: Vec<f32> = mu.iter().map(|&p| p * (1.0 - p)).collect();
    let x2: Vec<f32> = mu.clone();

    (mu, var, x2)
}

//////////////////
// Normal model //
//////////////////

/// Normal model for gene expression
///
/// Simplest model just using the normalised counts in the data.
///
/// ### Params
///
/// * `gene` - Reference to the CscGeneChunk containing gene expression data
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// Tuple of (mu, var, x2) where:
/// - mu: Mean expression for each cell (from linear regression)
/// - var: Residual variance (constant across cells)
/// - x2: Second moment (var + mu²) for each cell
fn normal_model(
    gene: &CscGeneChunk,
    umi_counts: &[f32],
    n_cells: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut gene_raw = vec![0_f32; n_cells];
    for (&idx, val) in gene.indices.iter().zip(gene.data_raw.iter()) {
        gene_raw[idx as usize] = val as f32;
    }

    let n = n_cells as f64;
    let umi_mean = sum_widen_simd_f32(umi_counts) / n;
    let umi_var = sum_squared_dev_widen_simd_f32(umi_counts, umi_mean);

    // Degenerate design: the regression has nothing to fit against, so fall
    // back on the gene's own moments.
    if umi_var == 0.0 {
        let mean = sum_widen_simd_f32(&gene_raw) / n;
        let sq = sum_squared_dev_widen_simd_f32(&gene_raw, mean);
        let (mu_val, var_val) = (mean as f32, (sq / n) as f32);
        return (
            vec![mu_val; n_cells],
            vec![var_val; n_cells],
            vec![var_val + mu_val * mu_val; n_cells],
        );
    }

    // Expression ~ raw library size, matching `normal_model.py`
    let (intercept, slope) = linear_regression_widen(umi_counts, &gene_raw);
    let mu: Vec<f32> = umi_counts.iter().map(|&x| intercept + slope * x).collect();

    // `np.var` of the residuals, so `n` and not the regression's `n - 2`
    let residuals_sq: f64 = gene_raw
        .iter()
        .zip(&mu)
        .map(|(&obs, &pred)| {
            let d = (obs - pred) as f64;
            d * d
        })
        .sum();
    let var_val = (residuals_sq / n) as f32;

    let var = vec![var_val; n_cells];
    let x2: Vec<f32> = mu.iter().map(|&m| var_val + m * m).collect();

    (mu, var, x2)
}

//////////
// Main //
//////////

/// HotSpot structure
///
/// Main structure for computing spatial autocorrelation and gene <> gene
/// correlations in spatially-resolved transcriptomics data.
#[derive(Clone, Debug)]
pub struct Hotspot<'a, S: SingleCellReading> {
    /// Reader over the gene-based count store.
    gene_reader: &'a S,
    /// Symmetric CSR graph both the autocorrelation and the pair path run on.
    graph: GraphCsr,
    /// Slice of cells to analyse/keep in this analysis.
    cells_to_keep: &'a [usize],
    /// Pre-computed node-degree for each cell based on the weights.
    node_degrees: Vec<f32>,
    /// Total UMI counts per cell, read eagerly in `new`.
    umi_counts: Vec<f32>,
    /// Sum of squared weights
    wtot2: f32,
    /// Total number of cells analysed in the experiment.
    n_cells: usize,
    /// Sum of `umi_counts` (hoisted out of the DANB fit). `f64` because
    /// metacell depths sum well past what `f32` represents exactly.
    umi_total: f64,
    /// Sum of squared `umi_counts` (hoisted out of the DANB fit), in `f64` for
    /// the same reason as `umi_total`, only more so.
    umi_sq_sum: f64,
    /// log10(umi) per cell (hoisted out of the Bernoulli fit).
    log10_umi: Vec<f32>,
    /// Whether the library sizes carry no variance, in which case the Normal
    /// model's regression is degenerate.
    umi_variance_zero: bool,
    /// Per-cell quantile bin on log10(umi); depends only on depth.
    umi_bins: Vec<usize>,
    /// Bin centres for the Bernoulli logistic fit.
    bin_centers: Vec<f32>,
    /// Cells per bin (Laplace-smoothing denominator).
    bin_totals: Vec<f32>,
    /// Number of Bernoulli bins.
    n_bins: usize,
}

impl<'a, S: SingleCellReading> Hotspot<'a, S> {
    /// Initialise a new instance
    ///
    /// Reads the per-cell UMI counts from `cell_reader` eagerly and builds both
    /// the non-redundant weights (autocorrelation path) and the symmetric CSR
    /// graph (pair path).
    ///
    /// ### Params
    ///
    /// * `gene_reader` - Reader over the gene-based count store.
    /// * `cell_reader` - Reader over the cell-based count store.
    /// * `cells_to_keep` - Slice if the indices of the cells to include in this
    ///   analysis.
    /// * `neighbours` - Slice of the indices of the neighbours of the given
    ///   cell.
    /// * `distances` - Slice of the distances to the neighbours of a given
    ///   cell, ascending. These are distances, not weights: the edge weights
    ///   are derived here, see `graph_weights`.
    /// * `graph_params` - See [HotSpotGraphParams]. `None` takes the upstream
    ///   defaults, i.e. an unweighted graph.
    ///
    /// ### Returns
    ///
    /// `Result` with the initialised `Hotspot`
    pub fn new(
        gene_reader: &'a S,
        cell_reader: &S,
        cells_to_keep: &'a [usize],
        neighbours: &[Vec<usize>],
        distances: &[Vec<f32>],
        graph_params: Option<HotSpotGraphParams>,
    ) -> Result<Self, BixverseErrors> {
        let n_cells = neighbours.len();

        let weights = graph_weights(distances, &graph_params.unwrap_or_default());
        let weights = make_weights_non_redundant(neighbours, &weights);
        let node_degrees = compute_node_degree(neighbours, &weights);
        let graph = GraphCsr::from_non_redundant(neighbours, &weights);
        let wtot2: f32 = weights.iter().flatten().map(|&w| w * w).sum();

        let reader = cell_reader;
        let lib_sizes = reader.read_cell_library_sizes(cells_to_keep)?;
        let umi_counts: Vec<f32> = lib_sizes.iter().map(|x| *x as f32).collect();

        // Depth-derived quantities are constant across genes; compute once.
        let umi_total = sum_widen_simd_f32(&umi_counts);
        let umi_sq_sum = sum_squared_dev_widen_simd_f32(&umi_counts, 0.0);
        let log10_umi: Vec<f32> = umi_counts
            .iter()
            .map(|&x| if x > 0.0 { x.log10() } else { 0.0 })
            .collect();
        let umi_variance_zero =
            sum_squared_dev_widen_simd_f32(&umi_counts, umi_total / n_cells as f64) == 0.0;

        // Bernoulli binning depends only on depth, so the (sorting) quantile
        // cut moves out of the per-gene fit.
        const N_BIN_TARGET: usize = 30;
        let (umi_bins, bin_edges) = quantile_cut(&log10_umi, N_BIN_TARGET);
        let n_bins = bin_edges.len() - 1;
        let bin_centers: Vec<f32> = (0..n_bins)
            .map(|i| (bin_edges[i] + bin_edges[i + 1]) / 2.0)
            .collect();
        let mut bin_totals = vec![0.0_f32; n_bins];
        for &b in &umi_bins {
            bin_totals[b] += 1.0;
        }

        Ok(Self {
            gene_reader,
            graph,
            cells_to_keep,
            node_degrees,
            umi_counts,
            umi_total,
            umi_sq_sum,
            log10_umi,
            umi_variance_zero,
            umi_bins,
            bin_centers,
            bin_totals,
            n_bins,
            wtot2,
            n_cells,
        })
    }

    ///////////////////////
    // Auto correlations //
    ///////////////////////

    /// Compute spatial autocorrelation for all specified genes
    ///
    /// Calculates Geary's C statistic and Z-scores for spatial autocorrelation
    /// across the specified genes.
    ///
    /// ### Params
    ///
    /// * `gene_indices` - Indices of genes to analyse
    /// * `model` - Statistical model to use ("danb", "bernoulli", or "normal")
    /// * `centered` - Whether to centre the data before computing statistics
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// `Result<HotSpotGeneRes>` with gene indices, Geary's C, Z-scores, derived
    /// p-values and FDR.
    pub fn compute_all_genes(
        &mut self,
        gene_indices: &[usize],
        model: &str,
        centered: bool,
        verbose: usize,
    ) -> Result<HotSpotGeneRes, BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        let gex_model = parse_gex_model(model)
            .ok_or_else(|| BixverseErrors::HotSpotWrongModel(model.to_string()))?;

        let cell_set: IndexSet<u32> = self.cells_to_keep.iter().map(|&x| x as u32).collect();

        let start_reading = Instant::now();
        let reader = self.gene_reader;
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices)?;
        gene_chunks.par_iter_mut().for_each(|chunk| {
            chunk.filter_selected_cells(&cell_set);
        });

        if verbosity.normal_verbosity() {
            println!("Loaded in data: {:.2?}", start_reading.elapsed());
        }

        let start_calculation = Instant::now();
        let res: Vec<(usize, f32, f32)> = gene_chunks
            .par_iter()
            .map_init(
                || GeneScratch::new(self.n_cells, !centered),
                |sc, chunk| self.compute_single_gene(chunk, &gex_model, centered, sc),
            )
            .collect();

        let mut gene_idx: Vec<usize> = Vec::with_capacity(res.len());
        let mut gaery_c: Vec<f64> = Vec::with_capacity(res.len());
        let mut z_scores: Vec<f64> = Vec::with_capacity(res.len());
        for (idx, c, z) in res {
            if z.is_finite() {
                gene_idx.push(idx);
                gaery_c.push(c as f64);
                z_scores.push(z as f64);
            }
        }

        if verbosity.normal_verbosity() {
            println!(
                "Finished the calculations: {:.2?}",
                start_calculation.elapsed()
            );
        }

        // upstream tests positive autocorrelation only, see `local_stats.py:239`
        let p_vals = z_scores_to_pval(&z_scores, "greater");
        let fdrs = p_adjust_fdr(&p_vals);

        Ok(HotSpotGeneRes {
            gene_idx,
            c: gaery_c,
            z: z_scores,
            pval: p_vals,
            fdr: fdrs,
        })
    }

    /// Compute spatial autocorrelation with streaming (memory-efficient)
    ///
    /// Processes genes in batches to reduce memory usage for large datasets.
    ///
    /// ### Params
    ///
    /// * `gene_indices` - Indices of genes to analyse
    /// * `model` - Statistical model to use ("danb", "bernoulli", or "normal")
    /// * `centered` - Whether to centre the data before computing statistics
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// `Result<HotSpotGeneRes>` with gene indices, Geary's C, Z-scores, derived
    /// p-values and FDR.
    pub fn compute_all_genes_streaming(
        &mut self,
        gene_indices: &[usize],
        model: &str,
        centered: bool,
        verbose: usize,
    ) -> Result<HotSpotGeneRes, BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        let start_all = Instant::now();

        let no_genes = gene_indices.len();
        let no_batches = no_genes.div_ceil(GENE_BATCH_SIZE);
        let cell_set: IndexSet<u32> = self.cells_to_keep.iter().map(|&x| x as u32).collect();
        let reader = self.gene_reader;

        let gex_model = parse_gex_model(model)
            .ok_or_else(|| BixverseErrors::HotSpotWrongModel(model.to_string()))?;

        let mut gene_indices_out: Vec<usize> = Vec::new();
        let mut gaery_c: Vec<f64> = Vec::new();
        let mut z_scores: Vec<f64> = Vec::new();

        for batch_idx in 0..no_batches {
            if verbosity.normal_verbosity() && batch_idx % 5 == 0 {
                let progress = (batch_idx + 1) as f32 / no_batches as f32 * 100.0;
                println!("  Progress: {:.1}%", progress);
            }

            let start_gene = batch_idx * GENE_BATCH_SIZE;
            let end_gene = ((batch_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
            let batch_gene_indices = &gene_indices[start_gene..end_gene];

            let start_loading = Instant::now();
            let mut gene_chunks = reader.read_gene_parallel(batch_gene_indices)?;
            gene_chunks.par_iter_mut().for_each(|chunk| {
                chunk.filter_selected_cells(&cell_set);
            });
            if verbosity.detailed_verbosity() {
                println!("   Loaded batch in: {:.2?}.", start_loading.elapsed());
            }

            let start_calc = Instant::now();
            let batch_res: Vec<(usize, f32, f32)> = gene_chunks
                .par_iter()
                .map_init(
                    || GeneScratch::new(self.n_cells, !centered),
                    |sc, chunk| self.compute_single_gene(chunk, &gex_model, centered, sc),
                )
                .collect();
            if verbosity.detailed_verbosity() {
                println!("   Finished calculations in: {:.2?}.", start_calc.elapsed());
            }

            for (idx, c, z) in batch_res {
                if z.is_finite() {
                    gene_indices_out.push(idx);
                    gaery_c.push(c as f64);
                    z_scores.push(z as f64);
                }
            }
        }

        // upstream tests positive autocorrelation only, see `local_stats.py:239`
        let p_vals = z_scores_to_pval(&z_scores, "greater");
        let fdrs = p_adjust_fdr(&p_vals);

        if verbosity.normal_verbosity() {
            println!("Finished the full run in : {:.2?}.", start_all.elapsed());
        }

        Ok(HotSpotGeneRes {
            gene_idx: gene_indices_out,
            c: gaery_c,
            z: z_scores,
            pval: p_vals,
            fdr: fdrs,
        })
    }

    /// Compute a single gene's spatial autocorrelation.
    ///
    /// ### Params
    ///
    /// * `gene_chunk` - Gene expression data
    /// * `gex_model` - Statistical model to apply
    /// * `centered` - Whether to centre the data
    /// * `sc` - Reusable per-thread scratch
    ///
    /// ### Returns
    ///
    /// Tuple of (gene_index, Geary's C, Z-score)
    fn compute_single_gene(
        &self,
        gene_chunk: &CscGeneChunk,
        gex_model: &GexModel,
        centered: bool,
        sc: &mut GeneScratch,
    ) -> (usize, f32, f32) {
        sc.vals.fill(0.0);
        for (&idx, val) in gene_chunk.indices.iter().zip(gene_chunk.data_raw.iter()) {
            sc.vals[idx as usize] = val as f32;
        }

        // Bernoulli models detection, so both the fit and the statistic run on
        // the indicator, see `local_stats.py:254`.
        if matches!(gex_model, GexModel::Bernoulli) {
            binarise(&mut sc.vals);
        }

        match gex_model {
            GexModel::DephAdjustNegBinom => {
                self.fit_danb(gene_chunk, sc);
            }
            GexModel::Bernoulli => self.fit_bernoulli(gene_chunk, sc),
            GexModel::Normal => self.fit_normal(sc),
            GexModel::PreStandardised => self.fit_pre_standardised(sc),
        }

        if centered && !matches!(gex_model, GexModel::PreStandardised) {
            center_values(&mut sc.vals, &sc.mu, &sc.var);
        }

        let g = 0.5 * self.graph.quadratic_form(&sc.vals);

        let (eg, eg2) = if centered {
            (0.0, self.wtot2)
        } else {
            compute_moments_weights(
                &self.graph,
                &sc.mu,
                &sc.x2,
                &mut sc.mu_sq,
                &mut sc.t1,
                &mut sc.u_musq,
                &mut sc.u_x2,
            )
        };

        let std_g = (eg2 - eg * eg).sqrt();
        let z = (g - eg) / std_g;

        let g_max = compute_local_cov_max(&self.node_degrees, &sc.vals);
        let c = (g - eg) / g_max;

        (gene_chunk.original_index, c, z)
    }

    /////////////
    // Helpers //
    /////////////

    /// Fit the DANB model into scratch (reads dense raw from `sc.vals`).
    ///
    /// ### Params
    ///
    /// * `gene` - Gene chunk
    /// * `sc` - Scratch; `vals` must already hold the dense raw counts
    ///
    /// ### Returns
    ///
    /// Nothing; writes `mu`, `var`, `x2`.
    fn fit_danb(&self, gene: &CscGeneChunk, sc: &mut GeneScratch) {
        let n = self.n_cells as f64;
        let total = self.umi_total;
        let tj: f64 = gene.data_raw.iter().map(|x| x as f64).sum();

        let scale = (tj / total) as f32;
        for i in 0..self.n_cells {
            sc.mu[i] = scale * self.umi_counts[i];
        }

        let mut sum_sq = 0.0_f64;
        for i in 0..self.n_cells {
            let diff = (sc.vals[i] - sc.mu[i]) as f64;
            sum_sq += diff * diff;
        }
        let vv = sum_sq / (n - 1.0);

        let mut size = ((tj * tj) / total) * (self.umi_sq_sum / total) / ((n - 1.0) * vv - tj);
        if size < 0.0 {
            size = 1e9;
        } else if size < 1e-10 {
            size = 1e-10;
        }
        let size = size as f32;

        for i in 0..self.n_cells {
            let m = sc.mu[i];
            let v = m * (1.0 + m / size);
            sc.var[i] = v;
            sc.x2[i] = v + m * m;
        }
    }

    /// Fit the Bernoulli model into scratch.
    ///
    /// Binning, bin centres and bin totals are precomputed in `new`; only the
    /// per-gene detection counts and the logistic fit happen here.
    ///
    /// ### Params
    ///
    /// * `gene` - Gene chunk
    /// * `sc` - Scratch
    ///
    /// ### Returns
    ///
    /// Nothing; writes `mu`, `var`, `x2` (`x2 == mu`).
    fn fit_bernoulli(&self, gene: &CscGeneChunk, sc: &mut GeneScratch) {
        let mut bin_detects = vec![0.0_f32; self.n_bins];
        for &idx in &gene.indices {
            bin_detects[self.umi_bins[idx as usize]] += 1.0;
        }

        let lbin: Vec<f32> = bin_detects
            .iter()
            .zip(&self.bin_totals)
            .map(|(&d, &t)| logit((d + 1.0) / (t + 2.0)))
            .collect();
        let coef = linear_regression(&self.bin_centers, &lbin);

        for i in 0..self.n_cells {
            let p = inv_logit(coef.0 + coef.1 * self.log10_umi[i]);
            sc.mu[i] = p;
            sc.var[i] = p * (1.0 - p);
            sc.x2[i] = p;
        }
    }

    /// Fit the Normal model into scratch (reads dense raw from `sc.vals`).
    ///
    /// OLS of the counts on the raw library size, with the residual variance
    /// taken as the uncorrected (`ddof = 0`) second moment and shared across
    /// cells. When the depths carry no variance the regression is degenerate
    /// and the gene's own mean and variance are used instead.
    ///
    /// ### Params
    ///
    /// * `sc` - Scratch; `vals` must already hold the dense raw counts
    ///
    /// ### Returns
    ///
    /// Nothing; writes `mu`, `var` (constant), `x2`.
    ///
    /// ### References
    ///
    /// DeTomaso and Yosef, Cell Systems, 2021
    fn fit_normal(&self, sc: &mut GeneScratch) {
        let n = self.n_cells as f64;

        if self.umi_variance_zero {
            let mean = sum_widen_simd_f32(&sc.vals) / n;
            let mut sq = 0.0_f64;
            for i in 0..self.n_cells {
                let d = sc.vals[i] as f64 - mean;
                sq += d * d;
            }
            let (mu_val, var_val) = ((mean as f32), ((sq / n) as f32));
            for i in 0..self.n_cells {
                sc.mu[i] = mu_val;
                sc.var[i] = var_val;
                sc.x2[i] = var_val + mu_val * mu_val;
            }
            return;
        }

        // Raw depth, not its log: upstream regresses on `umi_counts` directly.
        let (intercept, slope) = linear_regression_widen(&self.umi_counts, &sc.vals);
        for i in 0..self.n_cells {
            sc.mu[i] = intercept + slope * self.umi_counts[i];
        }

        let mut resid_sq = 0.0_f64;
        for i in 0..self.n_cells {
            let d = (sc.vals[i] - sc.mu[i]) as f64;
            resid_sq += d * d;
        }
        // `np.var` of the residuals, so `n` rather than the regression's `n - 2`
        let var_val = (resid_sq / n) as f32;

        for i in 0..self.n_cells {
            sc.var[i] = var_val;
            sc.x2[i] = var_val + sc.mu[i] * sc.mu[i];
        }
    }

    /// Fit the pre-standardised model into scratch: `mu = 0`, `var = x2 = 1`.
    ///
    /// Centring is then the identity, which is why `compute_single_gene` skips
    /// it for this model rather than dividing by one.
    ///
    /// ### Params
    ///
    /// * `sc` - Scratch
    ///
    /// ### Returns
    ///
    /// Nothing; writes `mu`, `var`, `x2`.
    ///
    /// ### References
    ///
    /// DeTomaso and Yosef, Cell Systems, 2021 (`hotspot/none_model.py`)
    fn fit_pre_standardised(&self, sc: &mut GeneScratch) {
        sc.mu.fill(0.0);
        sc.var.fill(1.0);
        sc.x2.fill(1.0);
    }

    //////////////////
    // Correlations //
    //////////////////

    /// Compute pairwise gene correlations (in-memory version)
    ///
    /// Calculates local spatial correlations between all pairs of specified
    /// genes. Loads all gene data into memory for faster computation.
    ///
    /// WARNING: This holds three dense `n_cells x n_genes` blocks transiently
    /// (centred counts, wy, and their `Mat` copies during construction). Use
    /// the streaming variant for large gene sets.
    ///
    /// The pair statistic is computed as a single GEMM: with the symmetric
    /// graph `W`, centred counts stacked column-wise as `C`, and
    /// `WY = W @ C`, the local covariance matrix is `LC = C^T @ WY`.
    ///
    /// ### Params
    ///
    /// * `gene_indices` - Indices of genes to analyse
    /// * `model` - Statistical model to use ("danb", "bernoulli", or "normal")
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// Result containing HotSpotPairRes with correlation and Z-score matrices
    pub fn compute_gene_cor(
        &mut self,
        gene_indices: &[usize],
        model: &str,
        verbose: usize,
    ) -> Result<HotSpotPairRes, BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);

        let gex_model = parse_gex_model(model)
            .ok_or_else(|| BixverseErrors::HotSpotWrongModel(model.to_string()))?;

        let cell_set: IndexSet<u32> = self.cells_to_keep.iter().map(|&x| x as u32).collect();

        if verbosity.normal_verbosity() {
            println!("Loading {} genes...", gene_indices.len());
        }

        let start_loading = Instant::now();
        let reader = self.gene_reader;
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices)?;

        gene_chunks.par_iter_mut().for_each(|chunk| {
            chunk.filter_selected_cells(&cell_set);
        });

        let n_genes = gene_chunks.len();

        if verbosity.normal_verbosity() {
            println!("Loaded data in {:.2?}", start_loading.elapsed());
            println!("Centering gene expression...");
        }

        // Centred counts, column-major flat: column j == gene j over n_cells.
        let start_center = Instant::now();
        let mut cc = vec![0_f32; self.n_cells * n_genes];
        cc.par_chunks_mut(self.n_cells)
            .zip(gene_chunks.par_iter())
            .for_each(|(col, gene)| {
                let centered =
                    create_centered_counts_gene(gene, &self.umi_counts, self.n_cells, &gex_model);
                col.copy_from_slice(&centered);
            });
        drop(gene_chunks);

        if verbosity.normal_verbosity() {
            println!("Centered in {:.2?}", start_center.elapsed());
            println!("Computing wy, eg2 and per-gene max values...");
        }

        // wy = W @ c per gene; eg2 = sum_squares(wy); max from centred counts.
        let start_eg2 = Instant::now();
        let mut wy = vec![0_f32; self.n_cells * n_genes];
        let mut eg2s = vec![0_f32; n_genes];
        let mut gene_maxs = vec![0_f32; n_genes];
        wy.par_chunks_mut(self.n_cells)
            .zip(cc.par_chunks(self.n_cells))
            .zip(eg2s.par_iter_mut().zip(gene_maxs.par_iter_mut()))
            .for_each(|((wy_col, c_col), (e, m))| {
                self.graph.spmv(c_col, wy_col);
                *e = sum_squares_simd_f32(wy_col);
                *m = compute_local_cov_max(&self.node_degrees, c_col);
            });

        let c_mat = Mat::<f32>::from_fn(self.n_cells, n_genes, |i, j| cc[j * self.n_cells + i]);
        drop(cc);
        let wy_mat = Mat::<f32>::from_fn(self.n_cells, n_genes, |i, j| wy[j * self.n_cells + i]);
        drop(wy);

        if verbosity.normal_verbosity() {
            println!("Computed wy/eg2/maxs in {:.2?}", start_eg2.elapsed());
            println!("Computing pairwise correlations (GEMM)...");
        }

        // LC = C^T @ WY is symmetric: compute the lower triangle, then reflect.
        let start_pairs = Instant::now();
        let mut lc = Mat::<f32>::zeros(n_genes, n_genes);
        triangular_matmul(
            &mut lc,
            BlockStructure::TriangularLower,
            Accum::Replace,
            c_mat.transpose(),
            BlockStructure::Rectangular,
            &wy_mat,
            BlockStructure::Rectangular,
            1.0_f32,
            faer_parallelism(),
        );
        for j in 0..n_genes {
            for i in 0..j {
                lc[(i, j)] = lc[(j, i)];
            }
        }

        if verbosity.normal_verbosity() {
            println!("Computed GEMM in {:.2?}", start_pairs.elapsed());
        }

        let mut lc_mat = Mat::<f32>::zeros(n_genes, n_genes);
        let mut z_mat = Mat::<f32>::zeros(n_genes, n_genes);

        for i in 0..n_genes {
            for j in (i + 1)..n_genes {
                Self::write_pair(
                    &mut lc_mat,
                    &mut z_mat,
                    i,
                    j,
                    lc[(i, j)],
                    eg2s[i],
                    eg2s[j],
                    gene_maxs[i],
                    gene_maxs[j],
                );
            }
        }

        if verbosity.normal_verbosity() {
            println!("Done!");
        }

        Ok(HotSpotPairRes {
            cor: lc_mat,
            z_scores: z_mat,
        })
    }

    /// Compute pairwise gene correlations (streaming version)
    ///
    /// Calculates local spatial correlations between all pairs of specified
    /// genes using panel-tiled block GEMM. Genes are split into panels of
    /// `panel_size`; the upper triangle of panel pairs is computed as block
    /// products `LC_block = C_i^T @ WY_j`. Within a panel load, gene chunks are
    /// read from disk in sub-chunks of `batch_size`.
    ///
    /// Memory: at most two panels are resident at once. Per panel peak is
    /// roughly `2 * n_cells * panel_size * 4` bytes (centred counts + wy), with
    /// a transient extra copy during `Mat` construction. `panel_size` trades
    /// memory for fewer panel reloads (reloads scale ~`n_panels^2 / 2`, and
    /// `n_panels = ceil(n_genes / panel_size)`); `panel_size >= n_genes`
    /// degenerates to the non-streaming case (single panel, no reload).
    ///
    /// ### Params
    ///
    /// * `gene_indices` - Indices of genes to analyse
    /// * `model` - Statistical model to use ("danb", "bernoulli", or "normal")
    /// * `batch_size` - Genes read from disk per `read_gene_parallel` call
    /// * `panel_size` - Genes held resident as one GEMM operand block
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// Result containing HotSpotPairRes with correlation and Z-score matrices
    pub fn compute_gene_cor_streaming(
        &mut self,
        gene_indices: &[usize],
        model: &str,
        batch_size: usize,
        panel_size: usize,
        verbose: usize,
    ) -> Result<HotSpotPairRes, BixverseErrors> {
        assert!(batch_size >= 1, "batch_size must be >= 1");
        assert!(panel_size >= 1, "panel_size must be >= 1");

        let verbosity = parse_verbosity_level(verbose);

        let gex_model = parse_gex_model(model)
            .ok_or_else(|| BixverseErrors::HotSpotWrongModel(model.to_string()))?;

        let cell_set: IndexSet<u32> = self.cells_to_keep.iter().map(|&x| x as u32).collect();
        let reader = self.gene_reader;

        let n_genes = gene_indices.len();
        let n_cells = self.n_cells;

        // load a panel: centred counts (Mat), wy (Mat), eg2 and per-gene max.
        let load_panel = |panel_indices: &[usize]| -> Result<PanelData, BixverseErrors> {
            let p = panel_indices.len();
            let mut cc = vec![0_f32; n_cells * p];

            let mut written = 0usize;
            for chunk_indices in panel_indices.chunks(batch_size) {
                let mut chunks = reader.read_gene_parallel(chunk_indices)?;
                chunks.par_iter_mut().for_each(|c| {
                    c.filter_selected_cells(&cell_set);
                });

                let base = written;
                cc[base * n_cells..(base + chunk_indices.len()) * n_cells]
                    .par_chunks_mut(n_cells)
                    .zip(chunks.par_iter())
                    .for_each(|(col, gene)| {
                        let centered = create_centered_counts_gene(
                            gene,
                            &self.umi_counts,
                            n_cells,
                            &gex_model,
                        );
                        col.copy_from_slice(&centered);
                    });
                written += chunk_indices.len();
            }

            let mut wy = vec![0_f32; n_cells * p];
            let mut eg2 = vec![0_f32; p];
            let mut maxs = vec![0_f32; p];
            wy.par_chunks_mut(n_cells)
                .zip(cc.par_chunks(n_cells))
                .zip(eg2.par_iter_mut().zip(maxs.par_iter_mut()))
                .for_each(|((wy_col, c_col), (e, m))| {
                    self.graph.spmv(c_col, wy_col);
                    *e = sum_squares_simd_f32(wy_col);
                    *m = compute_local_cov_max(&self.node_degrees, c_col);
                });

            let c_mat = Mat::<f32>::from_fn(n_cells, p, |i, j| cc[j * n_cells + i]);
            drop(cc);
            let wy_mat = Mat::<f32>::from_fn(n_cells, p, |i, j| wy[j * n_cells + i]);
            Ok((c_mat, wy_mat, eg2, maxs))
        };

        let panel_starts: Vec<usize> = (0..n_genes).step_by(panel_size).collect();
        let n_panels = panel_starts.len();

        if verbosity.normal_verbosity() {
            println!(
                "Processing {} genes in {} panels of up to {}",
                n_genes, n_panels, panel_size
            );
        }

        let mut lc_mat = Mat::<f32>::zeros(n_genes, n_genes);
        let mut z_mat = Mat::<f32>::zeros(n_genes, n_genes);

        for (pi, &pstart_i) in panel_starts.iter().enumerate() {
            let pend_i = (pstart_i + panel_size).min(n_genes);
            let li = pend_i - pstart_i;

            let start_panel = Instant::now();
            let (ci, wyi, eg2i, maxi) = load_panel(&gene_indices[pstart_i..pend_i])?;

            if verbosity.normal_verbosity() {
                println!(
                    "Loaded panel {} / {} (genes {}-{}) in {:.2?}",
                    pi + 1,
                    n_panels,
                    pstart_i,
                    pend_i - 1,
                    start_panel.elapsed()
                );
            }

            for &pstart_j in panel_starts.iter().skip(pi) {
                let pend_j = (pstart_j + panel_size).min(n_genes);
                let lj = pend_j - pstart_j;
                let diagonal = pstart_i == pstart_j;

                if verbosity.detailed_verbosity() {
                    println!("    Panel pair ({}, {})", pstart_i, pstart_j);
                }

                if diagonal {
                    // Symmetric block: lower triangle then reflect.
                    let mut block = Mat::<f32>::zeros(li, li);
                    triangular_matmul(
                        &mut block,
                        BlockStructure::TriangularLower,
                        Accum::Replace,
                        ci.transpose(),
                        BlockStructure::Rectangular,
                        &wyi,
                        BlockStructure::Rectangular,
                        1.0_f32,
                        faer_parallelism(),
                    );
                    for b in 0..li {
                        for a in 0..b {
                            block[(a, b)] = block[(b, a)];
                        }
                    }

                    for a in 0..li {
                        for b in (a + 1)..li {
                            let gi = pstart_i + a;
                            let gj = pstart_i + b;
                            Self::write_pair(
                                &mut lc_mat,
                                &mut z_mat,
                                gi,
                                gj,
                                block[(a, b)],
                                eg2i[a],
                                eg2i[b],
                                maxi[a],
                                maxi[b],
                            );
                        }
                    }
                } else {
                    let (_, wyj, eg2j, maxj) = load_panel(&gene_indices[pstart_j..pend_j])?;

                    // Rectangular block C_i^T @ WY_j.
                    let mut block = Mat::<f32>::zeros(li, lj);
                    matmul(
                        &mut block,
                        Accum::Replace,
                        ci.transpose(),
                        &wyj,
                        1.0_f32,
                        faer_parallelism(),
                    );

                    for a in 0..li {
                        for b in 0..lj {
                            let gi = pstart_i + a;
                            let gj = pstart_j + b; // pstart_i < pstart_j => gi < gj
                            Self::write_pair(
                                &mut lc_mat,
                                &mut z_mat,
                                gi,
                                gj,
                                block[(a, b)],
                                eg2i[a],
                                eg2j[b],
                                maxi[a],
                                maxj[b],
                            );
                        }
                    }
                }
            }
        }

        if verbosity.normal_verbosity() {
            println!("Done!");
        }

        Ok(HotSpotPairRes {
            cor: lc_mat,
            z_scores: z_mat,
        })
    }

    /// Write one symmetric pair entry into the output matrices
    ///
    /// Normalises the local covariance and computes the Z-score for a gene
    /// pair, writing both values symmetrically into `lc_mat` and `z_mat`.
    ///
    /// ### Params
    ///
    /// * `lc_mat` - Mutable reference to the local covariance matrix
    /// * `z_mat` - Mutable reference to the Z-score matrix
    /// * `gi` - Row/column index of the first gene
    /// * `gj` - Row/column index of the second gene
    /// * `lc` - Raw local covariance value for this pair
    /// * `eg2_i` - Expected squared covariance for gene `i`
    /// * `eg2_j` - Expected squared covariance for gene `j`
    /// * `max_i` - Maximum possible local covariance for gene `i`
    /// * `max_j` - Maximum possible local covariance for gene `j`
    ///
    /// ### Returns
    ///
    /// Nothing; modifies `lc_mat` and `z_mat` in place.
    #[allow(clippy::too_many_arguments)]
    fn write_pair(
        lc_mat: &mut Mat<f32>,
        z_mat: &mut Mat<f32>,
        gi: usize,
        gj: usize,
        lc: f32,
        eg2_i: f32,
        eg2_j: f32,
        max_i: f32,
        max_j: f32,
    ) {
        let lc_max = (max_i + max_j) * 0.5;
        let normalised_lc = if lc_max > 0.0 { lc / lc_max } else { 0.0 };
        // Smaller-magnitude z == divide by the larger denominator.
        let z = lc / eg2_i.max(eg2_j).sqrt();

        if z.is_finite() {
            lc_mat[(gi, gj)] = normalised_lc;
            lc_mat[(gj, gi)] = normalised_lc;
            z_mat[(gi, gj)] = z;
            z_mat[(gj, gi)] = z;
        }
    }
}

////////////////
// Clustering //
////////////////

/////////////////////////
// BH threshold search //
/////////////////////////

/// Bins spanning the candidate Z range in [bh_z_threshold].
///
/// 256 KB of `u32` per worker, so the histogram update stays L2-resident. At
/// m = 1.3e8 pairs this leaves on the order of 1e4 entries in the one bin that
/// has to be resolved exactly.
const Z_HIST_BINS: usize = 1 << 16;

/// Upper edge of the histogram.
///
/// Above roughly `|z| = 38.6` the normal pdf underflows in `f64`, so the
/// upper-tail p-value is exactly zero and every entry there clears
/// Benjamini-Hochberg unconditionally. One open bucket is therefore exact
/// rather than approximate.
const Z_HIST_MAX: f64 = 40.0;

/// Slack subtracted from the histogram floor.
///
/// `inverse_cdf` rounds, and excluding a true candidate would be a silent wrong
/// answer, whereas including a false one costs one extra exact re-test.
const Z_ALPHA_RELAX: f64 = 1e-9;

/// Largest dimension for which the symmetry precondition is fully asserted in
/// debug builds. A full check is O(n^2) and would make a debug run at
/// n = 16000 unusable.
const SYMMETRY_CHECK_MAX_DIM: usize = 512;

/// Upper-tail p-value of a single Z score.
///
/// Routed through [z_scores_to_pval] rather than calling `statrs` directly, so
/// the threshold search and the reported p-values cannot drift apart.
///
/// ### Params
///
/// * `z` - The Z score
///
/// ### Returns
///
/// `P(Z > z)` under the standard normal.
#[inline]
fn upper_tail_pval(z: f64) -> f64 {
    z_scores_to_pval(&[z], "greater")[0]
}

////////////////
// ZHistogram //
////////////////

/// Counts of the strict upper triangle of a Z matrix over a uniform grid.
#[derive(Debug)]
struct ZHistogram {
    /// Bin counts over `[lo, hi)`, `Z_HIST_BINS` of them.
    counts: Vec<u32>,
    /// Lower edge of bin zero. Entries below are dropped uncounted.
    lo: f64,
    /// Width of one bin.
    width: f64,
    /// Entries at or above the top edge.
    overflow: u32,
    /// Smallest entry at or above the top edge, `INFINITY` when there are none.
    overflow_min: f64,
}

impl ZHistogram {
    /// Bin the strict upper triangle of `z_mat` over `[lo, Z_HIST_MAX)`.
    ///
    /// Parallel over columns. For column `j` the strict upper entries are rows
    /// `0..j`, a contiguous prefix of a column-major matrix, so the pass is
    /// sequential in memory.
    ///
    /// ### Params
    ///
    /// * `z_mat` - Symmetric Z matrix
    /// * `lo` - Lower edge; entries below cannot clear BH and are dropped
    ///
    /// ### Returns
    ///
    /// The populated histogram.
    fn build(z_mat: MatRef<f64>, lo: f64) -> Self {
        let n = z_mat.nrows();
        let width = (Z_HIST_MAX - lo) / Z_HIST_BINS as f64;

        let (counts, overflow, overflow_min) = (0..n)
            .into_par_iter()
            .fold(
                || (vec![0_u32; Z_HIST_BINS], 0_u32, f64::INFINITY),
                |(mut counts, mut overflow, mut overflow_min), j| {
                    for i in 0..j {
                        let z = z_mat[(i, j)];
                        if z < lo {
                            continue;
                        }
                        if z >= Z_HIST_MAX {
                            overflow += 1;
                            overflow_min = overflow_min.min(z);
                            continue;
                        }
                        let bin = (((z - lo) / width) as usize).min(Z_HIST_BINS - 1);
                        counts[bin] += 1;
                    }
                    (counts, overflow, overflow_min)
                },
            )
            .reduce(
                || (vec![0_u32; Z_HIST_BINS], 0_u32, f64::INFINITY),
                |(mut a_counts, a_over, a_min), (b_counts, b_over, b_min)| {
                    for (a, b) in a_counts.iter_mut().zip(b_counts.iter()) {
                        *a += b;
                    }
                    (a_counts, a_over + b_over, a_min.min(b_min))
                },
            );

        Self {
            counts,
            lo,
            width,
            overflow,
            overflow_min,
        }
    }

    /// Lower edge of a bin.
    ///
    /// ### Params
    ///
    /// * `bin` - Bin index; `Z_HIST_BINS` gives the top edge
    ///
    /// ### Returns
    ///
    /// The edge value.
    #[inline]
    fn edge(&self, bin: usize) -> f64 {
        self.lo + self.width * bin as f64
    }
}

/// Smallest Z whose Benjamini-Hochberg q-value falls below `fdr_threshold`.
///
/// With `R(v) = #{z_i >= v}` the answer is
/// `min { v observed : p(v) < fdr_threshold * R(v) / m }`, which a histogram
/// plus one exact resolution pass finds in two streaming passes and 256 KB,
/// rather than materialising the O(m) upper triangle and its BH adjustment.
///
/// The tail is the upper one, matching upstream Hotspot (`modules.py` derives
/// the threshold from `norm.sf`), not the two-sided tail.
///
/// ### Params
///
/// * `z_mat` - Symmetric Z matrix with a zero diagonal. Must be finite: a `NaN`
///   makes the BH adjustment itself ill-defined, not merely this shortcut.
/// * `fdr_threshold` - The BH level
///
/// ### Returns
///
/// The threshold, or `f64::INFINITY` when nothing is significant, which stops
/// every merge downstream.
fn bh_z_threshold(z_mat: MatRef<f64>, fdr_threshold: f64) -> f64 {
    let n = z_mat.nrows();
    if n < 2 {
        return f64::INFINITY;
    }

    debug_assert!(
        n > SYMMETRY_CHECK_MAX_DIM
            || (0..n).all(|j| (0..j).all(|i| z_mat[(i, j)] == z_mat[(j, i)])),
        "bh_z_threshold requires a symmetric Z matrix"
    );

    let m = n * (n - 1) / 2;

    if fdr_threshold > 1.0 {
        return (0..n)
            .into_par_iter()
            .map(|j| {
                (0..j)
                    .map(|i| z_mat[(i, j)])
                    .fold(f64::INFINITY, |a, b| a.min(b))
            })
            .reduce(|| f64::INFINITY, |a, b| a.min(b));
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    let lo = (normal.inverse_cdf(1.0 - fdr_threshold) - Z_ALPHA_RELAX)
        .clamp(-Z_HIST_MAX, Z_HIST_MAX - 1.0);

    let hist = ZHistogram::build(z_mat, lo);

    let mut at_or_above = vec![0_u32; Z_HIST_BINS + 1];
    at_or_above[Z_HIST_BINS] = hist.overflow;
    for bin in (0..Z_HIST_BINS).rev() {
        at_or_above[bin] = at_or_above[bin + 1] + hist.counts[bin];
    }

    let m_f = m as f64;
    for bin in 0..Z_HIST_BINS {
        if hist.counts[bin] == 0 {
            continue;
        }

        let p_floor = upper_tail_pval(hist.edge(bin + 1));
        if p_floor >= fdr_threshold * at_or_above[bin] as f64 / m_f {
            continue;
        }

        if let Some(v) = resolve_threshold_bin(
            z_mat,
            hist.edge(bin),
            hist.edge(bin + 1),
            at_or_above[bin + 1],
            hist.counts[bin] as usize,
            m_f,
            fdr_threshold,
        ) {
            return v;
        }
    }

    if hist.overflow > 0 {
        return hist.overflow_min;
    }

    f64::INFINITY
}

/// Exact BH test over the members of one histogram bin.
///
/// The optimistic bin test in [bh_z_threshold] is necessary but not sufficient,
/// so the members are collected and tested individually. p-values come from
/// [z_scores_to_pval] in a single call, and the comparison is spelled as in
/// `calc_fdr`, which is what makes the result bit-identical to the long way
/// round.
///
/// ### Params
///
/// * `z_mat` - Symmetric Z matrix
/// * `lo` - Bin lower edge, inclusive
/// * `hi` - Bin upper edge, exclusive
/// * `above` - Entries at or above `hi`
/// * `capacity` - Member count, known from the histogram
/// * `m_f` - Total pair count as a float
/// * `fdr_threshold` - The BH level
///
/// ### Returns
///
/// The smallest passing member, or `None` when none passes.
#[allow(clippy::too_many_arguments)]
fn resolve_threshold_bin(
    z_mat: MatRef<f64>,
    lo: f64,
    hi: f64,
    above: u32,
    capacity: usize,
    m_f: f64,
    fdr_threshold: f64,
) -> Option<f64> {
    let n = z_mat.nrows();
    let mut members: Vec<f64> = Vec::with_capacity(capacity);
    for j in 0..n {
        for i in 0..j {
            let z = z_mat[(i, j)];
            if z >= lo && z < hi {
                members.push(z);
            }
        }
    }
    members.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let pvals = z_scores_to_pval(&members, "greater");
    let len = members.len();

    let mut t = 0usize;
    while t < len {
        let mut group_end = t + 1;
        while group_end < len && members[group_end] == members[t] {
            group_end += 1;
        }

        let rank = above as f64 + (len - t) as f64;
        if (m_f / rank) * pvals[t] < fdr_threshold {
            return Some(members[t]);
        }

        t = group_end;
    }

    None
}

//////////////////////////
// Average-linkage tree //
//////////////////////////

/// One merge in the average-linkage dendrogram.
///
/// Laid out like a row of a scipy linkage matrix, except that `height` carries
/// the average-linkage Z rather than the `maxZ - Z` distance upstream feeds to
/// `scipy.linkage`. The two are affine with a negative slope, so upstream's
/// `Z[i, 2] > offset - z_threshold` test becomes `height < z_threshold` here.
#[derive(Clone, Copy, Debug)]
struct Merge {
    /// Left child node id: a leaf below `n`, an internal node at or above it.
    left: usize,
    /// Right child node id, same encoding.
    right: usize,
    /// Average-linkage Z between the two children at the point they merged.
    height: f64,
    /// Number of leaves below this node.
    size: usize,
}

/// Average-linkage dendrogram over a dense similarity matrix, via
/// nearest-neighbour chain.
///
/// The algorithm `scipy.cluster.hierarchy.linkage(method='average')` uses, so
/// this is a port of upstream's clustering rather than a lookalike. Average
/// linkage on similarities is reducible (`sim(k, a u b)` is a convex
/// combination of `sim(k,a)` and `sim(k,b)`, so it can never exceed their max),
/// which is what makes the chain valid. O(n^2) time and O(n) working state on
/// top of the similarity matrix.
///
/// The returned merges are ordered by descending height, i.e. ascending
/// distance, so a bottom-up walk sees children before parents.
///
/// ### Params
///
/// * `z_mat` - Symmetric similarity matrix with a zero diagonal
///
/// ### Returns
///
/// `n - 1` merges, or an empty vector when `n < 2`.
fn average_linkage_nn_chain(z_mat: MatRef<f64>) -> Vec<Merge> {
    let n = z_mat.nrows();
    if n < 2 {
        return Vec::new();
    }

    let mut sim = vec![0.0_f64; n * n];
    sim.par_chunks_mut(n).enumerate().for_each(|(j, col)| {
        for (i, slot) in col.iter_mut().enumerate() {
            *slot = z_mat[(i, j)];
        }
    });

    let mut size = vec![1_usize; n];
    let mut active = vec![true; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut raw: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);

    for _ in 0..(n - 1) {
        if chain.is_empty() {
            let seed = (0..n)
                .find(|&i| active[i])
                .expect("an active cluster remains");
            chain.push(seed);
        }

        let (x, y, best) = loop {
            let x = chain[chain.len() - 1];

            let (mut y, mut best) = if chain.len() > 1 {
                let prev = chain[chain.len() - 2];
                (prev, sim[prev + x * n])
            } else {
                (usize::MAX, f64::NEG_INFINITY)
            };

            let col = &sim[x * n..(x + 1) * n];
            for k in 0..n {
                if !active[k] || k == x {
                    continue;
                }
                if col[k] > best {
                    best = col[k];
                    y = k;
                }
            }

            if chain.len() > 1 && y == chain[chain.len() - 2] {
                chain.pop();
                chain.pop();
                break (x, y, best);
            }
            chain.push(y);
        };

        let (sx, sy) = (size[x], size[y]);
        let total = (sx + sy) as f64;
        let (wx, wy) = (sx as f64, sy as f64);
        for k in 0..n {
            if !active[k] || k == x || k == y {
                continue;
            }
            let merged = (sim[k + x * n] * wx + sim[k + y * n] * wy) / total;
            sim[k + y * n] = merged;
            sim[y + k * n] = merged;
        }

        active[x] = false;
        size[y] = sx + sy;
        raw.push((x, y, best));
    }

    canonicalise_linkage(raw, n)
}

/// Turn raw nearest-neighbour-chain merges into a canonical dendrogram.
///
/// The chain emits merges in whatever order it walks the tree and identifies
/// clusters by an original leaf slot. Sorting by descending height and mapping
/// slots to node ids through a union-find reproduces scipy's post-`nn_chain`
/// labelling step, which is what makes the node ids meaningful.
///
/// ### Params
///
/// * `raw` - `(dropped_slot, surviving_slot, height)` in chain order
/// * `n` - Number of leaves
///
/// ### Returns
///
/// The merges ordered by descending height, with node ids assigned so that
/// merge `i` is node `n + i`.
fn canonicalise_linkage(mut raw: Vec<(usize, usize, f64)>, n: usize) -> Vec<Merge> {
    raw.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

    let mut parent: Vec<usize> = (0..n).collect();
    let mut node_id: Vec<usize> = (0..n).collect();
    let mut leaves: Vec<usize> = vec![1; n];

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    let mut out = Vec::with_capacity(raw.len());
    for (idx, (x, y, height)) in raw.into_iter().enumerate() {
        let rx = find(&mut parent, x);
        let ry = find(&mut parent, y);
        let (a, b) = if node_id[rx] < node_id[ry] {
            (node_id[rx], node_id[ry])
        } else {
            (node_id[ry], node_id[rx])
        };
        let size = leaves[rx] + leaves[ry];

        out.push(Merge {
            left: a,
            right: b,
            height,
            size,
        });

        parent[rx] = ry;
        node_id[ry] = n + idx;
        leaves[ry] = size;
    }

    out
}

/// Assign gene modules over a dendrogram, as upstream `assign_modules_core`.
///
/// A bottom-up pass decides, per merge, whether the merged node names a module,
/// inherits one from a child, or names none. `prop_label` then pushes labels
/// down to the leaves, and the first non-empty label encountered wins the whole
/// subtree below it: a merge that declines to join two already-large children
/// leaves both children's modules standing rather than discarding them.
///
/// ### Params
///
/// * `merges` - Dendrogram from [average_linkage_nn_chain], descending height
/// * `n` - Number of genes
/// * `z_threshold` - Merges below this height cannot form or extend a module
/// * `min_cluster_genes` - Minimum genes for a subtree to count as a module
///
/// ### Returns
///
/// Per-gene module label, densely renumbered from zero, `None` for unassigned.
///
/// ### References
///
/// DeTomaso and Yosef, Cell Systems, 2021 (`hotspot/modules.py::assign_modules_core`)
fn assign_modules_core(
    merges: &[Merge],
    n: usize,
    z_threshold: f64,
    min_cluster_genes: usize,
) -> Vec<Option<usize>> {
    /// Leaves below a child node.
    fn child_size(merges: &[Merge], n: usize, node: usize) -> usize {
        if node < n { 1 } else { merges[node - n].size }
    }

    /// Module label of a child node; leaves carry none.
    fn child_label(labels: &[Option<usize>], n: usize, node: usize) -> Option<usize> {
        if node < n { None } else { labels[node - n] }
    }

    let mut labels: Vec<Option<usize>> = vec![None; merges.len()];
    let mut next_label = 0usize;

    for (i, merge) in merges.iter().enumerate() {
        let n_a = child_size(merges, n, merge.left);
        let n_b = child_size(merges, n, merge.right);
        let big_a = n_a >= min_cluster_genes;
        let big_b = n_b >= min_cluster_genes;

        labels[i] = if (big_a && big_b) || (merge.height < z_threshold) {
            None
        } else if big_a {
            child_label(&labels, n, merge.left)
        } else if big_b {
            child_label(&labels, n, merge.right)
        } else if n_a + n_b >= min_cluster_genes {
            let label = next_label;
            next_label += 1;
            Some(label)
        } else {
            None
        };
    }

    let mut out: Vec<Option<usize>> = vec![None; n];
    if merges.is_empty() {
        return out;
    }

    let root = merges.len() - 1;
    let mut stack: Vec<(usize, Option<usize>)> = vec![(root, labels[root])];
    while let Some((idx, inherited)) = stack.pop() {
        let label = inherited.or(labels[idx]);
        for child in [merges[idx].left, merges[idx].right] {
            if child < n {
                out[child] = label;
            } else {
                stack.push((child - n, label));
            }
        }
    }

    renumber_modules(out)
}

/// Renumber module labels densely from zero, in ascending order of the raw
/// label.
///
/// Upstream renumbers from the sorted unique set so that the ids are stable
/// against the order the tree happened to be walked in; without this they
/// depend on dendrogram traversal, which is an implementation detail.
///
/// ### Params
///
/// * `labels` - Per-gene raw labels
///
/// ### Returns
///
/// The same labels renumbered from zero.
fn renumber_modules(labels: Vec<Option<usize>>) -> Vec<Option<usize>> {
    let mut seen: Vec<usize> = labels.iter().flatten().copied().collect();
    seen.sort_unstable();
    seen.dedup();

    let remap: FxHashMap<usize, usize> = seen
        .into_iter()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect();

    labels
        .into_iter()
        .map(|l| l.and_then(|old| remap.get(&old).copied()))
        .collect()
}

/// Cluster the HotSpot Z matrix into gene modules.
///
/// Builds the average-linkage dendrogram over the Z matrix, then assigns
/// modules over it exactly as upstream `compute_modules` does: the FDR-derived
/// Z threshold decides which merges may form a module, and
/// `assign_modules_core` walks the tree.
///
/// ### Params
///
/// * `z_mat` - Symmetric Z matrix with a zero diagonal, as
///   [HotSpotPairRes::z_scores] produces. Must be finite.
/// * `fdr_threshold` - BH level at which a pair Z counts as significant.
/// * `min_cluster_genes` - Minimum genes for a subtree to count as a module.
///   Upstream's `Hotspot.create_modules` ships 20 here, while `compute_modules`
///   itself defaults to 10; this takes whatever the caller passes.
///
/// ### Returns
///
/// Per-gene module label as a float, renumbered densely from zero. `NaN` means
/// unassigned.
///
/// ### References
///
/// DeTomaso and Yosef, Cell Systems, 2021
pub fn hotspot_gene_clusters(
    z_mat: MatRef<f64>,
    fdr_threshold: f64,
    min_cluster_genes: usize,
) -> Vec<f64> {
    let n = z_mat.nrows();
    let z_threshold = bh_z_threshold(z_mat, fdr_threshold);
    let merges = average_linkage_nn_chain(z_mat);

    assign_modules_core(&merges, n, z_threshold, min_cluster_genes)
        .into_iter()
        .map(|l| l.map_or(f64::NAN, |v| v as f64))
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    // A small undirected graph over 4 nodes. Reciprocal edges present so that
    // make_weights_non_redundant exercises the combine-and-zero path.
    fn small_graph() -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let neighbours = vec![
            vec![1, 2],    // 0
            vec![0, 2],    // 1
            vec![0, 1, 3], // 2
            vec![2],       // 3
        ];
        let weights = vec![
            vec![0.5, 0.25],
            vec![0.5, 0.75],
            vec![0.25, 0.75, 1.0],
            vec![1.0],
        ];
        (neighbours, weights)
    }

    // Reference implementations matching the pre-optimisation behaviour.

    // Old `local_cov_pair(x, y) * 2.0`, i.e. the value the pair caller used.
    fn ref_lc_x2(x: &[f32], y: &[f32], neigh: &[Vec<usize>], w_nr: &[Vec<f32>]) -> f32 {
        let mut out = 0.0_f32;
        for (i, (ns, ws)) in neigh.iter().zip(w_nr.iter()).enumerate() {
            let xi = x[i];
            let yi = y[i];
            for (&j, &wij) in ns.iter().zip(ws.iter()) {
                out += wij * (xi * y[j] + yi * x[j]);
            }
        }
        out
    }

    // Old `conditional_eg2`.
    fn ref_eg2(x: &[f32], neigh: &[Vec<usize>], w_nr: &[Vec<f32>]) -> f32 {
        let n = neigh.len();
        let mut t = vec![0.0_f32; n];
        for i in 0..n {
            for k in 0..neigh[i].len() {
                let j = neigh[i][k];
                let wij = w_nr[i][k];
                if wij == 0.0 {
                    continue;
                }
                t[i] += wij * x[j];
                t[j] += wij * x[i];
            }
        }
        t.iter().map(|&v| v * v).sum()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// The default is upstream's `weighted_graph=False`: distances decide who
    /// is a neighbour and nothing else.
    #[test]
    fn test_unweighted_graph_is_the_default() {
        let distances = vec![vec![0.5, 1.0, 1.5], vec![2.0, 4.0, 8.0]];

        let weights = graph_weights(&distances, &HotSpotGraphParams::default());

        assert_eq!(weights, vec![vec![1.0_f32; 3], vec![1.0_f32; 3]]);
    }

    /// Each undirected edge keeps its combined weight on one endpoint and zero on the other.
    #[test]
    fn non_redundant_combines_and_zeroes() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);

        // edge 0~1: combined onto node 0's entry (0.5 + 0.5), node 1's zeroed.
        assert_eq!(nr[0][0], 1.0); // 0 -> 1
        assert_eq!(nr[1][0], 0.0); // 1 -> 0 zeroed
        // edge 0~2: 0.25 + 0.25 onto node 0.
        assert_eq!(nr[0][1], 0.5);
        assert_eq!(nr[2][0], 0.0);
        // edge 1~2: 0.75 + 0.75 onto node 1.
        assert_eq!(nr[1][1], 1.5);
        assert_eq!(nr[2][1], 0.0);
        // edge 2~3: 1.0 + 1.0 onto node 2.
        assert_eq!(nr[2][2], 2.0);
        assert_eq!(nr[3][0], 0.0);
    }

    /// Degree sums every incident edge regardless of which endpoint the weight was folded onto.
    #[test]
    fn node_degree_counts_both_endpoints() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let d = compute_node_degree(&neigh, &nr);

        // node 0: edges 0~1 (1.0) + 0~2 (0.5) = 1.5
        assert!((d[0] - 1.5).abs() < 1e-6);
        // node 3: edge 2~3 (2.0)
        assert!((d[3] - 2.0).abs() < 1e-6);
    }

    /// The CSR spmv reproduces the symmetric scatter loop it replaced.
    #[test]
    fn spmv_matches_reference_t1x() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let graph = GraphCsr::from_non_redundant(&neigh, &nr);

        let x = [1.0_f32, -2.0, 3.0, 0.5];
        let mut wy = vec![0.0_f32; x.len()];
        graph.spmv(&x, &mut wy);

        // reference t1x (symmetric scatter)
        let mut t = vec![0.0_f32; x.len()];
        for i in 0..neigh.len() {
            for k in 0..neigh[i].len() {
                let j = neigh[i][k];
                let wij = nr[i][k];
                if wij == 0.0 {
                    continue;
                }
                t[i] += wij * x[j];
                t[j] += wij * x[i];
            }
        }

        for (a, b) in wy.iter().zip(t.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    /// E[G^2] collapses to the sum of squares of `W @ x`, matching the old `conditional_eg2`.
    #[test]
    fn eg2_equals_sum_squares_wy() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let graph = GraphCsr::from_non_redundant(&neigh, &nr);

        let x = [0.3_f32, 1.7, -0.4, 2.2];
        let mut wy = vec![0.0_f32; x.len()];
        graph.spmv(&x, &mut wy);
        let new_eg2: f32 = wy.iter().map(|&v| v * v).sum();

        let old = ref_eg2(&x, &neigh, &nr);
        assert!((new_eg2 - old).abs() < 1e-5, "{new_eg2} vs {old}");
    }

    /// Regression: `dot(x, W @ y)` must equal the old `local_cov_pair * 2.0`, or a
    /// botched symmetrisation halves every pair value.
    #[test]
    fn pair_formula_equivalence() {
        let (neigh, w) = small_graph();
        let nr = make_weights_non_redundant(&neigh, &w);
        let graph = GraphCsr::from_non_redundant(&neigh, &nr);

        let x = [1.0_f32, 0.5, -1.5, 2.0];
        let y = [-0.5_f32, 2.0, 0.25, -1.0];

        let old = ref_lc_x2(&x, &y, &neigh, &nr);

        let mut wy = vec![0.0_f32; y.len()];
        graph.spmv(&y, &mut wy);
        let new = dot(&x, &wy);

        assert!((new - old).abs() < 1e-5, "new {new} vs old {old}");

        // symmetry: dot(y, W @ x) gives the same value
        let mut wx = vec![0.0_f32; x.len()];
        graph.spmv(&x, &mut wx);
        let new_sym = dot(&y, &wx);
        assert!((new_sym - old).abs() < 1e-5, "sym {new_sym} vs old {old}");
    }

    /// Bin edges come out strictly increasing and every point lands in a valid bin index.
    #[test]
    fn quantile_cut_assigns_and_orders() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let (bins, edges) = quantile_cut(&data, 10);

        assert_eq!(bins.len(), data.len());
        // edges strictly increasing
        for win in edges.windows(2) {
            assert!(win[1] > win[0]);
        }
        // every assignment is a valid bin index
        let n_bins = edges.len() - 1;
        assert!(bins.iter().all(|&b| b < n_bins));
    }

    ////////////////////////////
    // Clustering: references //
    ////////////////////////////

    // The pre-rewrite threshold search and merge loop, kept as the parity
    // oracle for the histogram threshold and the NN-chain linkage. They must
    // not be "tidied" to match whatever the production code grows into.

    use std::collections::BinaryHeap;

    /// Entry in the pre-rewrite merge priority queue.
    #[derive(Clone, Debug)]
    struct MergeCandidate {
        /// Z-score between the two clusters (higher = merge first).
        z: f64,
        /// First cluster index (always the smaller index).
        i: usize,
        /// Second cluster index.
        j: usize,
        /// Generation stamp of `i` at insertion time.
        gen_i: u32,
        /// Generation stamp of `j` at insertion time.
        gen_j: u32,
    }

    impl PartialEq for MergeCandidate {
        fn eq(&self, other: &Self) -> bool {
            self.z == other.z
        }
    }

    impl Eq for MergeCandidate {}

    impl PartialOrd for MergeCandidate {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for MergeCandidate {
        fn cmp(&self, other: &Self) -> Ordering {
            self.z.partial_cmp(&other.z).unwrap_or(Ordering::Equal)
        }
    }

    /// Pre-rewrite Z threshold: materialise the upper triangle, its upper-tail
    /// p-values and their FDRs, then take the smallest surviving `z`.
    ///
    /// ### Params
    ///
    /// * `z_mat` - Symmetric Z matrix
    /// * `fdr_threshold` - Below which FDR an entry counts as significant
    ///
    /// ### Returns
    ///
    /// The threshold, or `f64::INFINITY` when nothing is significant.
    fn z_threshold_reference(z_mat: MatRef<f64>, fdr_threshold: f64) -> f64 {
        let z_upper_triangle = faer_mat_to_upper_triangle(z_mat, 1);
        let pvals = z_scores_to_pval(&z_upper_triangle, "greater");
        let fdrs = p_adjust_fdr(&pvals);

        z_upper_triangle
            .iter()
            .zip(fdrs.iter())
            .filter(|&(_, &fdr)| fdr < fdr_threshold)
            .map(|(&z, _)| z)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f64::INFINITY)
    }

    /// Pre-rewrite greedy merge loop, with the threshold injected.
    ///
    /// Average-linkage agglomerative clustering driven by a lazily invalidated
    /// binary heap, plus the original label bookkeeping.
    ///
    /// ### Params
    ///
    /// * `z_mat` - Symmetric Z matrix
    /// * `z_threshold` - Stop merging once the best remaining Z drops below this
    ///
    /// ### Returns
    ///
    /// The induced partition as sorted gene groups, and the merge heights in
    /// merge order.
    fn greedy_merge_reference(z_mat: MatRef<f64>, z_threshold: f64) -> (Vec<Vec<usize>>, Vec<f64>) {
        let n = z_mat.nrows();
        let mut z = z_mat.to_owned();

        let mut sizes: Vec<usize> = vec![1; n];
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut gene_to_cluster: Vec<usize> = (0..n).collect();
        let mut generation: Vec<u32> = vec![0; n];
        let mut active = vec![true; n];
        let mut heights: Vec<f64> = Vec::new();

        let mut heap = BinaryHeap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let zij = z[(i, j)];
                if zij >= z_threshold {
                    heap.push(MergeCandidate {
                        z: zij,
                        i,
                        j,
                        gen_i: 0,
                        gen_j: 0,
                    });
                }
            }
        }

        while let Some(candidate) = heap.pop() {
            if candidate.z < z_threshold {
                break;
            }

            let ci = candidate.i;
            let cj = candidate.j;

            if !active[ci]
                || !active[cj]
                || generation[ci] != candidate.gen_i
                || generation[cj] != candidate.gen_j
            {
                continue;
            }

            let new_size = sizes[ci] + sizes[cj];
            heights.push(candidate.z);

            generation[ci] += 1;
            active[cj] = false;

            for k in 0..n {
                if k == ci || k == cj || !active[k] {
                    continue;
                }

                let new_z = (z[(ci, k)] * sizes[ci] as f64 + z[(cj, k)] * sizes[cj] as f64)
                    / new_size as f64;
                z[(ci, k)] = new_z;
                z[(k, ci)] = new_z;

                if new_z >= z_threshold {
                    heap.push(MergeCandidate {
                        z: new_z,
                        i: ci.min(k),
                        j: ci.max(k),
                        gen_i: generation[ci.min(k)],
                        gen_j: generation[ci.max(k)],
                    });
                }
            }

            let cj_genes: Vec<usize> = clusters[cj].drain(..).collect();
            clusters[ci].extend(cj_genes);
            sizes[ci] = new_size;

            for &gene in &clusters[ci] {
                gene_to_cluster[gene] = ci;
            }
        }

        (canonical_partition(&gene_to_cluster), heights)
    }

    /// Canonical form of a partition: sorted groups, sorted among themselves.
    ///
    /// ### Params
    ///
    /// * `assignment` - Per-gene cluster id, ids need not be dense
    ///
    /// ### Returns
    ///
    /// The groups, each sorted, ordered by their smallest member.
    fn canonical_partition(assignment: &[usize]) -> Vec<Vec<usize>> {
        let mut groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for (gene, &cluster) in assignment.iter().enumerate() {
            groups.entry(cluster).or_default().push(gene);
        }
        let mut out: Vec<Vec<usize>> = groups.into_values().collect();
        for g in out.iter_mut() {
            g.sort_unstable();
        }
        out.sort_unstable();
        out
    }

    /// The partition a dendrogram induces when every merge below `z_threshold`
    /// is cut.
    ///
    /// ### Params
    ///
    /// * `merges` - Dendrogram, descending height
    /// * `n` - Number of genes
    /// * `z_threshold` - Merges below this are not taken
    ///
    /// ### Returns
    ///
    /// The partition in the canonical form of [canonical_partition].
    fn linkage_partition(merges: &[Merge], n: usize, z_threshold: f64) -> Vec<Vec<usize>> {
        let mut assignment: Vec<usize> = (0..n).collect();
        // node id -> the representative gene of its subtree
        let mut rep: Vec<usize> = (0..n + merges.len()).collect();

        for (i, merge) in merges.iter().enumerate() {
            if merge.height < z_threshold {
                continue;
            }
            let target = rep[merge.left].min(rep[merge.right]);
            let (a, b) = (rep[merge.left], rep[merge.right]);
            for slot in assignment.iter_mut() {
                if *slot == a || *slot == b {
                    *slot = target;
                }
            }
            rep[n + i] = target;
        }

        canonical_partition(&assignment)
    }

    /// A reproducible symmetric matrix with a zero diagonal, as
    /// `HotSpotPairRes::z_scores` produces.
    ///
    /// ### Params
    ///
    /// * `n` - Number of genes
    /// * `scale` - Multiplier on the standard normal draws; controls how much
    ///   of the matrix clears any given FDR threshold
    /// * `seed` - RNG seed
    ///
    /// ### Returns
    ///
    /// The symmetric `n x n` matrix.
    fn random_symmetric_z(n: usize, scale: f64, seed: u64) -> Mat<f64> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in (i + 1)..n {
                // Box-Muller, so the tail is a real normal tail rather than a
                // scaled uniform: the threshold search keys off the tail shape.
                let u1: f64 = rng.random_range(1e-12..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let v = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                let z = v * scale;
                out[(i, j)] = z;
                out[(j, i)] = z;
            }
        }
        out
    }

    /// Two well-separated blocks joined weakly, the shape the module
    /// assignment has to get right.
    ///
    /// ### Params
    ///
    /// * `block` - Genes per block
    /// * `within` - Z within a block
    /// * `between` - Z across the two blocks
    ///
    /// ### Returns
    ///
    /// The symmetric `2 * block` square matrix.
    fn two_block_z(block: usize, within: f64, between: f64) -> Mat<f64> {
        let n = 2 * block;
        Mat::<f64>::from_fn(n, n, |i, j| {
            if i == j {
                0.0
            } else if (i < block) == (j < block) {
                within
            } else {
                between
            }
        })
    }

    /// NN-chain and the pre-rewrite greedy loop must agree on the clustering
    /// itself: the same merges get taken at any given threshold, so they induce
    /// the same partition, and the merge heights are the same multiset.
    ///
    /// Not bit equality. NN-chain does the Lance-Williams combinations in a
    /// different temporal order, so the heights differ in the last ulps. That
    /// order is scipy's, which is what upstream Hotspot runs.
    #[test]
    fn nn_chain_matches_greedy_clustering() {
        for (n, scale, seed) in [(20, 1.0, 1), (60, 2.0, 2), (40, 0.5, 3), (80, 1.5, 4)] {
            let z = random_symmetric_z(n, scale, seed);
            let merges = average_linkage_nn_chain(z.as_ref());
            assert_eq!(merges.len(), n - 1);

            // Heights are monotone: average linkage admits no inversions, and a
            // violation here would mean the chain took merges out of order.
            for w in merges.windows(2) {
                assert!(
                    w[0].height >= w[1].height,
                    "non-monotone dendrogram at n={n} scale={scale}"
                );
            }

            for threshold in [f64::NEG_INFINITY, -1.0, 0.0, 0.5, 1.0, 2.0] {
                let (expected, ref_heights) = greedy_merge_reference(z.as_ref(), threshold);
                let actual = linkage_partition(&merges, n, threshold);

                assert_eq!(
                    expected, actual,
                    "partition mismatch at n={n} scale={scale} threshold={threshold}"
                );

                // Same merges taken, so the same heights, up to rounding.
                let mut a: Vec<f64> = ref_heights;
                let mut b: Vec<f64> = merges
                    .iter()
                    .map(|m| m.height)
                    .filter(|h| *h >= threshold)
                    .collect();
                a.sort_by(|x, y| x.partial_cmp(y).unwrap());
                b.sort_by(|x, y| x.partial_cmp(y).unwrap());
                assert_eq!(a.len(), b.len(), "merge count at threshold={threshold}");
                for (x, y) in a.iter().zip(b.iter()) {
                    assert_relative_eq!(x, y, epsilon = 1e-12);
                }
            }
        }
    }

    /// The fix for the label bug: two labelled modules joined above the
    /// threshold must both survive, where the pre-rewrite code wiped both.
    #[test]
    fn assign_modules_core_keeps_both_declined_children() {
        let z = two_block_z(6, 10.0, 8.0);
        let merges = average_linkage_nn_chain(z.as_ref());

        let labels = assign_modules_core(&merges, 12, 1.0, 3);

        let assigned: Vec<usize> = labels.iter().flatten().copied().collect();
        assert_eq!(assigned.len(), 12, "every gene should land in a module");

        let mut distinct = assigned.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct, vec![0, 1], "expected exactly two modules");

        // and the split must follow the blocks
        let first = labels[0].unwrap();
        assert!(labels[..6].iter().all(|l| *l == Some(first)));
        assert!(labels[6..].iter().all(|l| *l == Some(1 - first)));
    }

    /// A merge below the threshold cannot form a module, so a matrix whose only
    /// links are weak leaves everything unassigned.
    #[test]
    fn assign_modules_core_respects_the_threshold() {
        let z = two_block_z(6, 0.5, 0.1);
        let merges = average_linkage_nn_chain(z.as_ref());

        let labels = assign_modules_core(&merges, 12, 5.0, 3);
        assert!(labels.iter().all(|l| l.is_none()));
    }

    ///////////////////////////
    // Clustering: threshold //
    ///////////////////////////

    /// The histogram search must reproduce the materialised BH pipeline exactly,
    /// on the same (upper-tail, signed) semantics. Bit equality, not closeness:
    /// the claim is that this is a pure optimisation.
    #[test]
    fn bh_z_threshold_matches_reference() {
        // scales chosen so K = 0, 0 < K < m and K = m all occur
        for (n, scale, seed) in [
            (20, 0.3, 11),
            (20, 3.0, 12),
            (60, 1.0, 13),
            (60, 2.5, 14),
            (120, 0.8, 15),
            (120, 4.0, 16),
        ] {
            let z = random_symmetric_z(n, scale, seed);
            for alpha in [1e-6, 0.001, 0.05, 0.5, 1.0, 1.5] {
                let expected = z_threshold_reference(z.as_ref(), alpha);
                let actual = bh_z_threshold(z.as_ref(), alpha);

                assert_eq!(
                    expected, actual,
                    "n={n} scale={scale} alpha={alpha}: {expected} vs {actual}"
                );
            }
        }
    }

    /// Degenerate shapes and the paths the histogram cannot represent directly:
    /// the zero-width case, the underflow plateau above the top edge, and heavy
    /// ties landing in one bin.
    #[test]
    fn bh_z_threshold_edge_cases() {
        // no pairs at all
        assert_eq!(
            bh_z_threshold(Mat::<f64>::zeros(0, 0).as_ref(), 0.05),
            f64::INFINITY
        );
        assert_eq!(
            bh_z_threshold(Mat::<f64>::zeros(1, 1).as_ref(), 0.05),
            f64::INFINITY
        );

        // nothing significant
        let flat = Mat::<f64>::zeros(10, 10);
        assert_eq!(bh_z_threshold(flat.as_ref(), 0.05), f64::INFINITY);

        // every entry above the underflow plateau, so every p-value is exactly
        // zero and the smallest entry wins
        let huge = two_block_z(5, 60.0, 50.0);
        assert_eq!(bh_z_threshold(huge.as_ref(), 0.05), 50.0);

        // heavy ties: one distinct off-diagonal value, all in a single bin
        let tied = two_block_z(8, 7.0, 7.0);
        assert_eq!(
            bh_z_threshold(tied.as_ref(), 0.05),
            z_threshold_reference(tied.as_ref(), 0.05)
        );

        // alpha above one passes everything, so the global minimum wins
        let mixed = two_block_z(6, 4.0, -2.0);
        assert_eq!(bh_z_threshold(mixed.as_ref(), 1.5), -2.0);
    }
}
