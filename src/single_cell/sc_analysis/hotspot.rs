//! Implementation of the HotSpot method to identify genes that vary across
//! various potential graphs in single cell 'omics, see DeTomaso and Yosef,
//! Cell Syst., 2021

use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as triangular_matmul};
use faer::{Accum, Mat, MatRef};
use indexmap::IndexSet;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::core::math::linear_algebra::linear_regression;
use crate::core::math::stats::{calc_fdr, inv_logit, logit, z_scores_to_pval};
use crate::prelude::*;
use crate::single_cell::sc_utils::simd::*;
use crate::utils::faer_parallelism;
use crate::utils::simd::{sum_simd_f32, sum_squares_simd_f32};

/////////////
// Hotspot //
/////////////

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
}

/// Parse the model to use gene expression
///
/// ### Params
///
/// * `s` - Type of model to use the model
///
/// ### Returns
///
/// Option of the GexModel to use (some not yet implemented)
pub fn parse_gex_model(s: &str) -> Option<GexModel> {
    match s.to_lowercase().as_str() {
        "danb" => Some(GexModel::DephAdjustNegBinom),
        "bernoulli" => Some(GexModel::Bernoulli),
        "normal" => Some(GexModel::Normal),
        _ => None,
    }
}

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

///////////////
// Graph CSR //
///////////////

/// Symmetric graph in CSR layout for the pair path.
///
/// Built from the non-redundant (upper-triangular combined) weights produced by
/// [make_weights_non_redundant]. Each undirected edge is stored in *both* rows
/// with its combined weight, so that a single sparse mat-vec
/// `wy = W_sym @ c` reproduces exactly the symmetric neighbour-weighted vector
/// `t1x` that the old `conditional_eg2` built by hand. With that:
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

    #[inline]
    fn n_nodes(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Sparse mat-vec: `out = W_sym @ c`. `out` and `c` are length n_nodes.
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
}

/// Compute momentum weights
///
/// Calculates the expected value (EG) and expected squared value (EG2) of the
/// local covariance statistic under the null hypothesis of no spatial
/// autocorrelation.
///
/// ### Params
///
/// * `mu` - Mean expression values for each cell
/// * `x2` - Second moment (variance + mean²) for each cell
/// * `neighbours` - Neighbour indices for each cell
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Tuple of (EG, EG2) where:
///
/// - EG: Expected value of the spatial covariance statistic
/// - EG2: Expected value of the squared spatial covariance statistic
fn compute_moments_weights(
    mu: &[f32],
    x2: &[f32],
    neighbours: &[Vec<usize>],
    weights: &[Vec<f32>],
) -> (f32, f32) {
    let n = neighbours.len();
    let mu_sq: Vec<f32> = mu.iter().map(|&m| m * m).collect();

    let mut eg = 0_f32;
    let mut t1 = vec![0_f32; n];
    let mut t2 = vec![0_f32; n];

    for i in 0..n {
        let mu_i = mu[i];

        for (k, &j) in neighbours[i].iter().enumerate() {
            let wij = weights[i][k];
            let mu_j = mu[j];

            eg += wij * mu_i * mu_j;

            t1[i] += wij * mu_j;
            let wij_sq = wij * wij;
            t2[i] += wij_sq * mu_j * mu_j;

            // Add these back:
            t1[j] += wij * mu_i;
            t2[j] += wij_sq * mu_i * mu_i;
        }
    }

    let mut eg2 = 0_f32;

    for i in 0..n {
        eg2 += (x2[i] - mu_sq[i]) * (t1[i] * t1[i] - t2[i]);
    }

    for i in 0..n {
        let x2_i = x2[i];
        let mu_sq_i = mu_sq[i];

        for (k, &j) in neighbours[i].iter().enumerate() {
            let wij = weights[i][k];
            eg2 += wij * wij * (x2_i * x2[j] - mu_sq_i * mu_sq[j]);
        }
    }

    eg2 += eg * eg;

    (eg, eg2)
}

/// Remove redundancy in bidirectional edge weights
///
/// Consolidates weights from bidirectional edges by accumulating both
/// directions into the lower-indexed node's edge and zeroing the
/// higher-indexed node's reciprocal edge.
///
/// ### Params
///
/// * `neighbours` - Neighbour indices for each node
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// Modified weights with redundant edges zeroed
fn make_weights_non_redundant(neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut w_no_redundant = weights.to_vec();

    for i in 0..neighbours.len() {
        for k in 0..neighbours[i].len() {
            let j = neighbours[i][k];

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

/// Compute local covariance using edge weights
///
/// Calculates the weighted local covariance statistic for spatial
/// autocorrelation. This is the numerator of Geary's C statistic.
///
/// ### Params
///
/// * `vals` - Gene expression values for each cell
/// * `neighbours` - Neighbour indices for each cell
/// * `weights` - Edge weights for each neighbour connection
///
/// ### Returns
///
/// The local covariance statistic
fn local_cov_weights(vals: &[f32], neighbours: &[Vec<usize>], weights: &[Vec<f32>]) -> f32 {
    let mut out = 0.0;

    for i in 0..vals.len() {
        let xi = vals[i];
        if xi == 0.0 {
            continue;
        }

        for (k, &j) in neighbours[i].iter().enumerate() {
            let xj = vals[j];
            let wij = weights[i][k];

            if xj != 0.0 && wij != 0.0 {
                out += xi * xj * wij;
            }
        }
    }

    out
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
/// Transforms values to have zero means and unit variance of one using the
/// provided stats.
///
/// ### Params
///
/// * `vals` - Mutable reference to the values to scale
/// * `mu` - The mean values
/// * `var` - The variance of the values
fn center_values(vals: &mut [f32], mu: &[f32], var: &[f32]) {
    assert_same_len!(vals, mu, var);

    center_values_simd(vals, mu, var);
}

//////////////////
// Corr helpers //
//////////////////

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
    let (mu, var, _) = match model {
        GexModel::DephAdjustNegBinom => danb_model(gene, umi_counts, n_cells),
        GexModel::Bernoulli => bernoulli_model(gene, umi_counts, n_cells),
        GexModel::Normal => normal_model(gene, umi_counts, n_cells),
    };

    let mut vals = vec![0_f32; n_cells];
    for (&idx, val) in gene.indices.iter().zip(gene.data_raw.iter()) {
        vals[idx as usize] = val as f32;
    }

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
    let n = n_cells as f32;
    let total: f32 = sum_simd_f32(umi_counts);
    let tj: f32 = gene.data_raw.iter().map(|x| x as f32).sum();

    let mu: Vec<f32> = umi_counts.iter().map(|&ti| tj * ti / total).collect();

    // Build dense array for O(1) lookups
    let mut data_dense = vec![0.0f32; n_cells];
    for (&idx, val) in gene.indices.iter().zip(gene.data_raw.iter()) {
        data_dense[idx as usize] = val as f32;
    }

    let mut sum_sq = 0_f32;
    for i in 0..n_cells {
        let diff = data_dense[i] - mu[i];
        sum_sq += diff * diff;
    }

    let vv = sum_sq / (n - 1.0);
    let tis_sq_sum: f32 = sum_squares_simd_f32(umi_counts);
    let mut size = ((tj * tj) / total) * (tis_sq_sum / total) / ((n - 1.0) * vv - tj);

    if size < 0.0 {
        size = 1e9;
    } else if size < 1e-10 {
        size = 1e-10;
    }

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

    // Fit linear regression: expression ~ log(umi_counts)
    let log_umi: Vec<f32> = umi_counts
        .iter()
        .map(|&x| if x > 0.0 { x.ln() } else { 0.0 })
        .collect();

    let (intercept, slope) = linear_regression(&log_umi, &gene_raw);

    // Cell-specific mu from regression
    let mu: Vec<f32> = log_umi.iter().map(|&x| intercept + slope * x).collect();

    // Residual variance (constant across cells)
    let residuals_sq: f32 = gene_raw
        .iter()
        .zip(&mu)
        .map(|(&obs, &pred)| (obs - pred).powi(2))
        .sum();
    let var_val = residuals_sq / (n_cells as f32 - 2.0);

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
///
/// Two graph representations are held: the original non-redundant
/// `neighbours`/`weights` arrays drive the autocorrelation path
/// (`compute_all_genes*`), while the symmetric [GraphCsr] drives the pair path
/// (`compute_gene_cor*`).
#[derive(Clone, Debug)]
pub struct Hotspot<'a> {
    /// File path to the gene-based binary file.
    f_path_gene: String,
    /// Slice if the indices of the cells to include in this analysis.
    neighbours: &'a [Vec<usize>],
    /// Non-redundant edge weights (used by the autocorrelation path).
    weights: Vec<Vec<f32>>,
    /// Symmetric CSR graph (used by the pair path).
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
}

impl<'a> Hotspot<'a> {
    /// Initialise a new instance
    ///
    /// Reads the per-cell UMI counts from `f_path_cell` eagerly and builds both
    /// the non-redundant weights (autocorrelation path) and the symmetric CSR
    /// graph (pair path).
    ///
    /// ### Params
    ///
    /// * `f_path_gene` - File path to the gene-based binary file.
    /// * `f_path_cell` - File path to the cell-based binary file.
    /// * `cells_to_keep` - Slice if the indices of the cells to include in this
    ///   analysis.
    /// * `neighbours` - Slice of the indices of the neighbours of the given
    ///   cell.
    /// * `weights` - Slice of the distances to the neighbours of a given cell.
    ///
    /// ### Return
    ///
    /// `Result` with the initialised `Hotspot` (reading the cell file can fail).
    pub fn new(
        f_path_gene: String,
        f_path_cell: String,
        cells_to_keep: &'a [usize],
        neighbours: &'a [Vec<usize>],
        weights: &mut [Vec<f32>],
    ) -> Result<Self, BixverseErrors> {
        let n_cells = neighbours.len();

        let weights = make_weights_non_redundant(neighbours, weights);

        let node_degrees = compute_node_degree(neighbours, &weights);

        let graph = GraphCsr::from_non_redundant(neighbours, &weights);

        let wtot2: f32 = weights.iter().flatten().map(|&w| w * w).sum();

        let reader = ParallelSparseReader::new(&f_path_cell)?;
        let lib_sizes = reader.read_cell_library_sizes(cells_to_keep)?;
        let umi_counts: Vec<f32> = lib_sizes.iter().map(|x| *x as f32).collect();

        Ok(Self {
            f_path_gene,
            neighbours,
            weights,
            graph,
            cells_to_keep,
            node_degrees,
            umi_counts,
            wtot2,
            n_cells,
        })
    }

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

        let reader = ParallelSparseReader::new(&self.f_path_gene)?;
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices)?;

        gene_chunks.par_iter_mut().for_each(|chunk| {
            chunk.filter_selected_cells(&cell_set);
        });

        let end_reading = start_reading.elapsed();

        if verbosity.normal_verbosity() {
            println!("Loaded in data: {:.2?}", end_reading);
        }

        let start_calculation = Instant::now();

        let res: Vec<(usize, f32, f32)> = gene_chunks
            .par_iter()
            .map(|chunk| self.compute_single_gene(chunk, &gex_model, centered))
            .collect();

        let mut gene_indices: Vec<usize> = Vec::with_capacity(res.len());
        let mut gaery_c: Vec<f64> = Vec::with_capacity(res.len());
        let mut z_scores: Vec<f64> = Vec::with_capacity(res.len());

        for (idx, c, z) in res {
            if !z.is_nan() {
                gene_indices.push(idx);
                gaery_c.push(c as f64);
                z_scores.push(z as f64);
            }
        }

        let end_calculations = start_calculation.elapsed();

        if verbosity.normal_verbosity() {
            println!("Finsished the calculations: {:.2?}", end_calculations);
        }

        let p_vals = z_scores_to_pval(&z_scores, "twosided");
        let fdrs = calc_fdr(&p_vals);

        Ok(HotSpotGeneRes {
            gene_idx: gene_indices,
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

        const GENE_BATCH_SIZE: usize = 1000;

        let start_all = Instant::now();

        let no_genes = gene_indices.len();
        let no_batches = no_genes.div_ceil(GENE_BATCH_SIZE);
        let cell_set: IndexSet<u32> = self.cells_to_keep.iter().map(|&x| x as u32).collect();
        let reader = ParallelSparseReader::new(&self.f_path_gene)?;

        let gex_model = parse_gex_model(model)
            .ok_or_else(|| BixverseErrors::HotSpotWrongModel(model.to_string()))?;

        let mut results: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)> = Vec::with_capacity(no_batches);

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

            let end_loading = start_loading.elapsed();

            if verbosity.detailed_verbosity() {
                println!("   Loaded batch in: {:.2?}.", end_loading);
            }

            let start_calc = Instant::now();

            let batch_res: Vec<(usize, f32, f32)> = gene_chunks
                .par_iter()
                .map(|chunk| self.compute_single_gene(chunk, &gex_model, centered))
                .collect();

            let mut batch_gene_indices: Vec<usize> = Vec::with_capacity(batch_res.len());
            let mut batch_gaery_c: Vec<f64> = Vec::with_capacity(batch_res.len());
            let mut batch_z_scores: Vec<f64> = Vec::with_capacity(batch_res.len());

            for (idx, c, z) in batch_res {
                if z.is_finite() {
                    batch_gene_indices.push(idx);
                    batch_gaery_c.push(c as f64);
                    batch_z_scores.push(z as f64);
                }
            }

            let end_calc = start_calc.elapsed();

            if verbosity.detailed_verbosity() {
                println!("   Finished calculations in: {:.2?}.", end_calc);
            }

            results.push((batch_gene_indices, batch_gaery_c, batch_z_scores));
        }

        let mut gene_indices_out: Vec<usize> = Vec::new();
        let mut gaery_c: Vec<f64> = Vec::new();
        let mut z_scores: Vec<f64> = Vec::new();

        for (idx, c, z) in results {
            gene_indices_out.extend(idx);
            gaery_c.extend(c);
            z_scores.extend(z);
        }

        let p_vals = z_scores_to_pval(&z_scores, "twosided");
        let fdrs = calc_fdr(&p_vals);

        let end_total = start_all.elapsed();

        if verbosity.normal_verbosity() {
            println!("Finished the full run in : {:.2?}.", end_total);
        }

        Ok(HotSpotGeneRes {
            gene_idx: gene_indices_out,
            c: gaery_c,
            z: z_scores,
            pval: p_vals,
            fdr: fdrs,
        })
    }

    /// Compute a single gene's spatial autocorrelation
    ///
    /// Internal method for calculating Geary's C and Z-score for one gene.
    ///
    /// ### Params
    ///
    /// * `gene_chunk` - Gene expression data
    /// * `gex_model` - Statistical model to apply
    /// * `centered` - Whether to centre the data
    ///
    /// ### Returns
    ///
    /// Tuple of (gene_index, Geary's C, Z-score)
    fn compute_single_gene(
        &self,
        gene_chunk: &CscGeneChunk,
        gex_model: &GexModel,
        centered: bool,
    ) -> (usize, f32, f32) {
        let (mu, var, x2) = match gex_model {
            GexModel::DephAdjustNegBinom => danb_model(gene_chunk, &self.umi_counts, self.n_cells),
            GexModel::Bernoulli => bernoulli_model(gene_chunk, &self.umi_counts, self.n_cells),
            GexModel::Normal => normal_model(gene_chunk, &self.umi_counts, self.n_cells),
        };

        let mut vals = vec![0_f32; self.n_cells];
        for (&idx, val) in gene_chunk.indices.iter().zip(gene_chunk.data_raw.iter()) {
            vals[idx as usize] = val as f32;
        }

        if centered {
            center_values(&mut vals, &mu, &var);
        }

        let g = local_cov_weights(&vals, self.neighbours, &self.weights);

        let (eg, eg2) = if centered {
            (0.0, self.wtot2)
        } else {
            compute_moments_weights(&mu, &x2, self.neighbours, &self.weights)
        };

        let std_g = (eg2 - eg * eg).sqrt();
        let z = (g - eg) / std_g;

        let g_max = compute_local_cov_max(&self.node_degrees, &vals);
        let c = (g - eg) / g_max;

        (gene_chunk.original_index, c, z)
    }

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
        let reader = ParallelSparseReader::new(&self.f_path_gene)?;
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
                let lc_ij = lc[(i, j)];

                let lc_max = (gene_maxs[i] + gene_maxs[j]) * 0.5;
                let normalised_lc = if lc_max > 0.0 { lc_ij / lc_max } else { 0.0 };

                // Smaller-magnitude z == divide by the larger denominator.
                let z = lc_ij / eg2s[i].max(eg2s[j]).sqrt();

                if z.is_finite() {
                    lc_mat[(i, j)] = normalised_lc;
                    lc_mat[(j, i)] = normalised_lc;
                    z_mat[(i, j)] = z;
                    z_mat[(j, i)] = z;
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
        let reader = ParallelSparseReader::new(&self.f_path_gene)?;

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

    /// Write one symmetric pair entry (normalised lc and z) into the outputs.
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

/// Entry in the merge priority queue.
#[derive(Clone, Debug)]
struct MergeCandidate {
    /// Z-score between the two clusters (higher = merge first).
    z: f64,
    /// First cluster index (always the smaller index).
    i: usize,
    /// Second cluster index.
    j: usize,
    /// Generation stamps at insertion time. If either cluster has been
    /// merged since this entry was created, it is stale and must be
    /// discarded.
    gen_i: u32,
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

/// Cluster the HotSpot Z matrix using heap-accelerated agglomerative
/// clustering.
///
/// Merges cluster pairs in descending Z-score order using average linkage.
/// Stops when the best remaining Z falls below the FDR-derived threshold.
///
/// ### Params
///
/// * `z_mat` - Reference to the Z-score matrix.
/// * `fdr_threshold` - Below which FDR to stop merging.
/// * `min_cluster_genes` - Minimum genes per cluster to assign a label.
///
/// ### Returns
///
/// Per-gene cluster assignment. `NaN` indicates no assignment.
pub fn hotspot_gene_clusters(
    z_mat: MatRef<f64>,
    fdr_threshold: f64,
    min_cluster_genes: usize,
) -> Vec<f64> {
    let z_upper_triangle = faer_mat_to_upper_triangle(z_mat, 1);
    let pvals = z_scores_to_pval(&z_upper_triangle, "twosided");
    let fdrs = calc_fdr(&pvals);
    let n = z_mat.nrows();

    let z_threshold = z_upper_triangle
        .iter()
        .zip(fdrs.iter())
        .filter(|&(_, &fdr)| fdr < fdr_threshold)
        .map(|(&z, _)| z.abs())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(f64::INFINITY);

    // mutable Z matrix for average-linkage updates
    let mut z = z_mat.to_owned();

    // per-cluster state
    let mut sizes: Vec<usize> = vec![1; n];
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut labels: Vec<Option<usize>> = vec![None; n];
    let mut gene_to_cluster: Vec<usize> = (0..n).collect();

    // generation counter per cluster; incremented on each merge so stale
    // heap entries can be detected cheaply.
    let mut generation: Vec<u32> = vec![0; n];

    // track which clusters are still active
    let mut active = vec![true; n];

    let mut next_label = 0usize;

    // seed the heap with all pairs above threshold
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

        // stale entry: one or both clusters have been merged since this
        // candidate was inserted.
        if !active[ci]
            || !active[cj]
            || generation[ci] != candidate.gen_i
            || generation[cj] != candidate.gen_j
        {
            continue;
        }

        // merge cj into ci
        let new_size = sizes[ci] + sizes[cj];
        let both_labelled = labels[ci].is_some() && labels[cj].is_some();

        // average-linkage Z update and push new candidates
        generation[ci] += 1;
        active[cj] = false;

        for k in 0..n {
            if k == ci || k == cj || !active[k] {
                continue;
            }

            let new_z =
                (z[(ci, k)] * sizes[ci] as f64 + z[(cj, k)] * sizes[cj] as f64) / new_size as f64;
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

        // Transfer genes from cj to ci
        let cj_genes: Vec<usize> = clusters[cj].drain(..).collect();
        clusters[ci].extend(cj_genes);
        sizes[ci] = new_size;

        for &gene in &clusters[ci] {
            gene_to_cluster[gene] = ci;
        }

        if new_size >= min_cluster_genes && !both_labelled {
            labels[ci] = Some(next_label);
            next_label += 1;
        } else if both_labelled {
            labels[ci] = None;
        }
    }

    let mut gene_labels = vec![f64::NAN; n];
    for gene in 0..n {
        let cluster = gene_to_cluster[gene];
        if let Some(label) = labels[cluster] {
            gene_labels[gene] = label as f64;
        }
    }

    gene_labels
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
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

    // The centrepiece: dot(x, W @ y) must equal the old `local_cov_pair * 2.0`.
    // This is what catches a half-value bug from getting symmetrisation wrong.
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

    #[test]
    fn z_simplification_matches_min_abs_branch() {
        let lc = -1.3_f32;
        let eg2_i = 4.0_f32;
        let eg2_j = 9.0_f32;

        let z_xy = lc / eg2_i.sqrt();
        let z_yx = lc / eg2_j.sqrt();
        let old = if z_xy.abs() < z_yx.abs() { z_xy } else { z_yx };

        let new = lc / eg2_i.max(eg2_j).sqrt();
        assert!((new - old).abs() < 1e-6, "{new} vs {old}");
    }

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
}
