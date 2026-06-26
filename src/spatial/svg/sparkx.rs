//! SPARK-X spatially variable gene detection (Zhu et al., Genome Biology 2021).
//!
//! Per-sample, per-gene score test against a small bank of low-rank
//! covariance kernels built from spot coordinates. For each kernel `l`, the
//! test statistic is
//!
//! ```text
//! T_l = (n - 1) * || phi_l^c' y_c ||^2 / SST_y
//! ```
//!
//! where `phi_l^c` is the (column-centred) low-rank factor of the centred
//! kernel matrix `M K_l M`, `y_c` is the centred gene vector, and
//! `SST_y = y_c' y_c`. Under H0 (no spatial pattern),
//! `T_l ~ sum_k lambda_k * chi^2_1` with `lambda_k = eigvals(phi_l^c' phi_l^c)`
//! (these are exactly the non-zero eigvals of the centred kernel matrix). The
//! upper-tail p-value is evaluated via a Kuonen (1999) saddle-point
//! approximation to the weighted-chi-squared CDF.
//!
//! P-values across kernels are combined via the Cauchy combination test
//! (Liu & Xie 2020), which is robust to dependence:
//!
//! ```text
//! p_combined = 0.5 - atan(mean_l tan((0.5 - p_l) * pi)) / pi
//! ```
//!
//! Kernel bank: by default five Gaussian + five cosine bandwidths derived
//! from the pairwise-distance distribution on a coordinate sub-sample (5
//! quantiles each, paper default). Gaussian kernels are factorised by a
//! Nystroem approximation against k-means++ landmarks; cosine kernels admit
//! an exact closed-form factorisation per dimension via cos/sin features.
//!
//! Raw counts are used unconditionally. The test statistic is scale-invariant
//! per gene (numerator and denominator scale together), so the normalisation
//! choice does not matter and raw counts match the SPARK-X paper exactly.

use faer::Mat;
use indexmap::IndexSet;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::stats::calc_fdr;
use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{CscGeneChunk, ParallelSparseReader};
use crate::spatial::svg::shared::{Assay, center_inplace, materialise_gene_dense};

////////////
// Params //
////////////

/// A single SPARK-X kernel specification.
///
/// The `bandwidth` parameter has different meaning per variant:
///
/// - `Gaussian { bandwidth: sigma }`: `K(x, y) = exp(-||x - y||^2 / (2 sigma^2))`.
///   Approximated by Nystroem against [`SparkXParams::n_landmarks`] landmarks.
/// - `Cosine { bandwidth: period }`: per-dimension cosine kernel
///   `K(x, y) = sum_d cos(2 pi (x_d - y_d) / period)`. Has an exact rank-4
///   factorisation for 2D coords via the `cos(a-b) = cos a cos b + sin a sin b`
///   identity (two cos/sin features per dimension).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparkXKernel {
    /// Gaussian / RBF kernel with bandwidth `sigma`.
    Gaussian {
        /// Kernel bandwidth `sigma`.
        bandwidth: f32,
    },
    /// Per-dimension cosine kernel with period `l`.
    Cosine {
        /// Kernel period `l`.
        bandwidth: f32,
    },
}

impl SparkXKernel {
    /// Short human-readable label used in [`SparkXRes::kernels`].
    pub fn label(&self) -> String {
        match self {
            SparkXKernel::Gaussian { bandwidth } => format!("gauss_{bandwidth:.4}"),
            SparkXKernel::Cosine { bandwidth } => format!("cos_{bandwidth:.4}"),
        }
    }
}

/// Parameters for [`SparkX`].
///
/// Empty `kernels` triggers the paper-default 5 Gaussian + 5 cosine bandwidth
/// bank, derived from quantiles of the pairwise coordinate distance
/// distribution on a sub-sample.
#[derive(Debug, Clone)]
pub struct SparkXParams {
    /// Kernel bank. Empty -> use defaults from coords (5 + 5).
    pub kernels: Vec<SparkXKernel>,
    /// Number of Nystroem landmarks per Gaussian kernel. Default 20.
    pub n_landmarks: usize,
    /// Sub-sample size for pairwise-distance bandwidth estimation. Default
    /// 1000. Skipped when `kernels` is non-empty.
    pub bandwidth_subsample: usize,
    /// Seed for landmark selection and bandwidth sub-sampling.
    pub seed: u64,
}

impl Default for SparkXParams {
    fn default() -> Self {
        Self {
            kernels: Vec::new(),
            n_landmarks: 20,
            bandwidth_subsample: 1000,
            seed: 42,
        }
    }
}

/////////////
// Results //
/////////////

/// Output of a SPARK-X run across a set of genes.
///
/// Per-kernel matrices are stored row-major as flat `Vec<f64>` of length
/// `n_genes * n_kernels` (gene-major).
#[derive(Debug, Clone)]
pub struct SparkXRes {
    /// Original (on-disk) gene indices that were tested.
    pub gene_idx: Vec<usize>,
    /// Number of kernels in the bank.
    pub n_kernels: usize,
    /// Per-kernel human-readable labels. Length `n_kernels`.
    pub kernels: Vec<String>,
    /// Per-(gene, kernel) test statistic, row-major. Length `n_genes * n_kernels`.
    pub stat_per_kernel: Vec<f64>,
    /// Per-(gene, kernel) p-value, row-major. Length `n_genes * n_kernels`.
    pub pval_per_kernel: Vec<f64>,
    /// Cauchy-combined p-value per gene. Length `n_genes`.
    pub pval_combined: Vec<f64>,
    /// BH-FDR over `pval_combined`. Length `n_genes`.
    pub fdr: Vec<f64>,
}

////////////
// Struct //
////////////

/// Per-sample SPARK-X driver.
///
/// Mirrors the shape of [`crate::spatial::svg::morans_i::MoransI`]. All
/// kernel-dependent quantities (`phi_l` and eigenvalues `lambdas_l`) are
/// precomputed in [`SparkX::new`]; the per-gene work in
/// [`SparkX::compute_all_genes`] reduces to a single dense matvec
/// `phi_l^T y_c` of size `n × r_l` per kernel plus the saddle-point tail and
/// the Cauchy combination.
#[derive(Clone, Debug)]
pub struct SparkX<'a> {
    /// File path to the gene-based binary file.
    f_path_gene: String,
    /// Global spot indices to keep (R-side per-sample selection).
    spots_to_keep: &'a [usize],
    /// Number of spots in this sample (= `spots_to_keep.len()`).
    n_spots: usize,
    /// `IndexSet` form of `spots_to_keep`, reused per gene chunk to drive
    /// `filter_selected_cells`.
    spot_set: IndexSet<u32>,
    /// Resolved kernel bank (after default expansion).
    kernels: Vec<SparkXKernel>,
    /// One `phi_l^c` per kernel. Column-centred. Stored row-major as
    /// `Vec<f32>` of length `n_spots * r_l`.
    phis: Vec<KernelFactor>,
}

/// Per-kernel low-rank factor.
#[derive(Clone, Debug)]
struct KernelFactor {
    /// Column-centred factor matrix, row-major: `data[i*r + k]` is row `i`
    /// column `k`. `r` = `lambdas.len()`.
    data: Vec<f32>,
    /// Number of columns (rank).
    r: usize,
    /// Non-zero eigenvalues of the centred kernel `phi_c phi_c'`. Equals the
    /// eigenvalues of the small `r × r` Gram matrix `phi_c' phi_c`.
    lambdas: Vec<f64>,
}

impl<'a> SparkX<'a> {
    /// Build a SPARK-X driver for a single sample.
    ///
    /// ### Params
    ///
    /// * `f_path_gene` - Gene-based binary file.
    /// * `spots_to_keep` - Global spot indices that make up this sample.
    /// * `coordinates` - Per-spot 2D coordinates in physical units, aligned
    ///   with `spots_to_keep`.
    /// * `params` - Run parameters; empty `kernels` triggers the default
    ///   5+5 bank derived from coord-pairwise-distance quantiles.
    pub fn new(
        f_path_gene: String,
        spots_to_keep: &'a [usize],
        coordinates: &[(f32, f32)],
        params: SparkXParams,
    ) -> Result<Self, BixverseErrors> {
        let n_spots = spots_to_keep.len();
        if coordinates.len() != n_spots {
            return Err(BixverseErrors::SpatialCoordSpotMismatch {
                n_coords: coordinates.len(),
                n_spots,
            });
        }
        if n_spots < 3 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "SPARK-X needs at least 3 spots, got {n_spots}"
            )));
        }

        let kernels = if params.kernels.is_empty() {
            default_kernel_bank(coordinates, params.bandwidth_subsample, params.seed)
        } else {
            params.kernels.clone()
        };
        if kernels.is_empty() {
            return Err(BixverseErrors::InvalidArgument(
                "SPARK-X: empty kernel bank after defaults".to_string(),
            ));
        }

        let phis: Result<Vec<KernelFactor>, BixverseErrors> = kernels
            .iter()
            .map(|k| build_kernel_factor(*k, coordinates, params.n_landmarks, params.seed))
            .collect();
        let phis = phis?;

        let spot_set: IndexSet<u32> = spots_to_keep.iter().map(|&x| x as u32).collect();

        Ok(Self {
            f_path_gene,
            spots_to_keep,
            n_spots,
            spot_set,
            kernels,
            phis,
        })
    }

    /// Spot count for this sample.
    #[inline]
    pub fn n_spots(&self) -> usize {
        self.n_spots
    }

    /// Resolved kernels (after default-bank expansion).
    #[inline]
    pub fn kernels(&self) -> &[SparkXKernel] {
        &self.kernels
    }

    /// Spots in this sample (passthrough of the slice given to `new`).
    #[inline]
    pub fn spots_to_keep(&self) -> &[usize] {
        self.spots_to_keep
    }

    /// Compute SPARK-X for all specified genes in a single in-memory pass.
    ///
    /// Loads all gene chunks for `gene_indices` up-front, then runs per-gene
    /// in parallel via rayon. Use [`SparkX::compute_all_genes_streaming`]
    /// when the full chunk set does not comfortably fit in memory.
    pub fn compute_all_genes(
        &self,
        gene_indices: &[usize],
        verbose: usize,
    ) -> Result<SparkXRes, BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);
        let start_all = Instant::now();

        let reader = ParallelSparseReader::new(&self.f_path_gene)?;
        let start_load = Instant::now();
        let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices)?;
        gene_chunks
            .par_iter_mut()
            .for_each(|chunk| chunk.filter_selected_cells(&self.spot_set));
        if verbosity.normal_verbosity() {
            println!(
                "SPARK-X: loaded {} gene chunks in {:.2?}",
                gene_chunks.len(),
                start_load.elapsed()
            );
        }

        let start_calc = Instant::now();
        let per_gene: Vec<GeneRes> = gene_chunks
            .par_iter()
            .map_init(
                || PerGeneScratch::new(self.n_spots, &self.phis),
                |scratch, chunk| self.compute_single_gene(chunk, scratch),
            )
            .collect();
        if verbosity.normal_verbosity() {
            println!(
                "SPARK-X: per-gene calculations in {:.2?}",
                start_calc.elapsed()
            );
        }

        let res = self.finalise(per_gene);
        if verbosity.normal_verbosity() {
            println!("SPARK-X: total {:.2?}", start_all.elapsed());
        }
        Ok(res)
    }

    /// Streaming variant: loads gene chunks in fixed-size batches.
    pub fn compute_all_genes_streaming(
        &self,
        gene_indices: &[usize],
        verbose: usize,
    ) -> Result<SparkXRes, BixverseErrors> {
        let verbosity = parse_verbosity_level(verbose);
        let start_all = Instant::now();

        const GENE_BATCH_SIZE: usize = 1000;
        let no_genes = gene_indices.len();
        let no_batches = no_genes.div_ceil(GENE_BATCH_SIZE);
        let reader = ParallelSparseReader::new(&self.f_path_gene)?;

        let mut per_gene: Vec<GeneRes> = Vec::with_capacity(no_genes);

        for batch_idx in 0..no_batches {
            if verbosity.normal_verbosity() && batch_idx % 5 == 0 {
                let progress = (batch_idx + 1) as f32 / no_batches as f32 * 100.0;
                println!("  SPARK-X progress: {:.1}%", progress);
            }

            let start_gene = batch_idx * GENE_BATCH_SIZE;
            let end_gene = ((batch_idx + 1) * GENE_BATCH_SIZE).min(no_genes);
            let batch = &gene_indices[start_gene..end_gene];

            let start_load = Instant::now();
            let mut chunks = reader.read_gene_parallel(batch)?;
            chunks
                .par_iter_mut()
                .for_each(|chunk| chunk.filter_selected_cells(&self.spot_set));
            if verbosity.detailed_verbosity() {
                println!("    Loaded batch in: {:.2?}.", start_load.elapsed());
            }

            let start_calc = Instant::now();
            let batch_res: Vec<GeneRes> = chunks
                .par_iter()
                .map_init(
                    || PerGeneScratch::new(self.n_spots, &self.phis),
                    |scratch, chunk| self.compute_single_gene(chunk, scratch),
                )
                .collect();
            if verbosity.detailed_verbosity() {
                println!("    Calculations in: {:.2?}.", start_calc.elapsed());
            }
            per_gene.extend(batch_res);
        }

        let res = self.finalise(per_gene);
        if verbosity.normal_verbosity() {
            println!("SPARK-X (streaming): total {:.2?}", start_all.elapsed());
        }
        Ok(res)
    }

    /////////////
    // Internals
    /////////////

    fn compute_single_gene(&self, chunk: &CscGeneChunk, scratch: &mut PerGeneScratch) -> GeneRes {
        materialise_gene_dense(chunk, Assay::Raw, &mut scratch.y);
        let (_mean, sst) = center_inplace(&mut scratch.y);
        let n = self.n_spots;
        let n_kernels = self.phis.len();

        if !sst.is_finite() || sst <= 0.0 || n < 2 {
            return GeneRes {
                original_index: chunk.original_index,
                stats: vec![f64::NAN; n_kernels],
                pvals: vec![f64::NAN; n_kernels],
                p_combined: f64::NAN,
            };
        }

        let sst_f64 = sst as f64;
        let scale = (n as f64 - 1.0) / sst_f64;

        let mut stats = Vec::with_capacity(n_kernels);
        let mut pvals = Vec::with_capacity(n_kernels);

        for (kernel_idx, factor) in self.phis.iter().enumerate() {
            // proj = phi^T y_c, length r
            let proj = &mut scratch.proj_bufs[kernel_idx];
            phi_t_y(&factor.data, factor.r, n, &scratch.y, proj);

            let mut num: f64 = 0.0;
            for &p in proj.iter() {
                let p64 = p as f64;
                num += p64 * p64;
            }
            // T_observed (scaled) ~ weighted-chisq with weights `lambdas`
            let q_obs = num * scale;
            stats.push(q_obs);
            let p = saddle_point_upper_tail(q_obs, &factor.lambdas);
            pvals.push(p);
        }

        let p_combined = cauchy_combination(&pvals);

        GeneRes {
            original_index: chunk.original_index,
            stats,
            pvals,
            p_combined,
        }
    }

    fn finalise(&self, per_gene: Vec<GeneRes>) -> SparkXRes {
        let n_genes = per_gene.len();
        let n_kernels = self.phis.len();
        let mut gene_idx = Vec::with_capacity(n_genes);
        let mut stat_per_kernel = Vec::with_capacity(n_genes * n_kernels);
        let mut pval_per_kernel = Vec::with_capacity(n_genes * n_kernels);
        let mut pval_combined = Vec::with_capacity(n_genes);

        for gr in per_gene {
            gene_idx.push(gr.original_index);
            stat_per_kernel.extend_from_slice(&gr.stats);
            pval_per_kernel.extend_from_slice(&gr.pvals);
            pval_combined.push(gr.p_combined);
        }

        let fdr = calc_fdr(&pval_combined);
        let kernels = self.kernels.iter().map(|k| k.label()).collect();

        SparkXRes {
            gene_idx,
            n_kernels,
            kernels,
            stat_per_kernel,
            pval_per_kernel,
            pval_combined,
            fdr,
        }
    }
}

/// Scratch buffers reused across genes within a rayon worker thread.
struct PerGeneScratch {
    y: Vec<f32>,
    /// One `r_l`-length buffer per kernel for `phi^T y`.
    proj_bufs: Vec<Vec<f32>>,
}

impl PerGeneScratch {
    fn new(n_spots: usize, phis: &[KernelFactor]) -> Self {
        Self {
            y: vec![0.0; n_spots],
            proj_bufs: phis.iter().map(|f| vec![0.0; f.r]).collect(),
        }
    }
}

/// Per-gene result before flat-Vec packing.
struct GeneRes {
    original_index: usize,
    stats: Vec<f64>,
    pvals: Vec<f64>,
    p_combined: f64,
}

////////////
// Kernel //
////////////

/// Build the rank-`r` centred factor and the small eigenvalue spectrum for
/// a single kernel.
fn build_kernel_factor(
    kernel: SparkXKernel,
    coords: &[(f32, f32)],
    n_landmarks: usize,
    seed: u64,
) -> Result<KernelFactor, BixverseErrors> {
    let n = coords.len();
    let (mut phi, r) = match kernel {
        SparkXKernel::Gaussian { bandwidth } => {
            if !bandwidth.is_finite() || bandwidth <= 0.0 {
                return Err(BixverseErrors::InvalidArgument(format!(
                    "SPARK-X Gaussian bandwidth must be positive and finite, got {bandwidth}"
                )));
            }
            build_gaussian_nystroem(coords, bandwidth, n_landmarks, seed)?
        }
        SparkXKernel::Cosine { bandwidth } => {
            if !bandwidth.is_finite() || bandwidth <= 0.0 {
                return Err(BixverseErrors::InvalidArgument(format!(
                    "SPARK-X cosine bandwidth must be positive and finite, got {bandwidth}"
                )));
            }
            build_cosine_features(coords, bandwidth)
        }
    };

    centre_columns(&mut phi, n, r);
    let lambdas = phi_c_eigvals(&phi, n, r)?;

    Ok(KernelFactor {
        data: phi,
        r,
        lambdas,
    })
}

/// Nystroem decomposition of the Gaussian kernel.
///
/// Picks `m = min(n_landmarks, n)` landmarks via k-means++ on the (x, y)
/// coordinates, builds `K_mm` and `K_nm`, eigendecomposes `K_mm = U S U'`
/// with floored eigenvalues for numerical safety, and returns
/// `phi = K_nm U S^{-1/2}` (shape `n × m_eff`, row-major), where `m_eff` is
/// the count of eigenvalues above the floor.
fn build_gaussian_nystroem(
    coords: &[(f32, f32)],
    sigma: f32,
    n_landmarks: usize,
    seed: u64,
) -> Result<(Vec<f32>, usize), BixverseErrors> {
    let n = coords.len();
    let m = n_landmarks.min(n).max(1);

    let landmarks = kmeans_pp_indices(coords, m, seed);
    let m = landmarks.len();
    let landmark_pts: Vec<(f32, f32)> = landmarks.iter().map(|&i| coords[i]).collect();

    let two_sigma_sq = 2.0_f32 * sigma * sigma;

    // K_mm
    let mut k_mm = Mat::<f64>::zeros(m, m);
    for a in 0..m {
        for b in 0..m {
            let dx = landmark_pts[a].0 - landmark_pts[b].0;
            let dy = landmark_pts[a].1 - landmark_pts[b].1;
            let d2 = dx * dx + dy * dy;
            k_mm[(a, b)] = (-d2 / two_sigma_sq).exp() as f64;
        }
    }

    let eig = k_mm
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| BixverseErrors::FaerEigenError)?;
    let evals_full: Vec<f64> = eig.S().column_vector().iter().copied().collect();
    let evecs_full = eig.U().to_owned();

    // Drop eigenvalues below a small fraction of the max (rank deficiency)
    let max_ev = evals_full.iter().cloned().fold(0.0_f64, f64::max);
    let floor = (max_ev * 1e-8_f64).max(1e-12_f64);
    let mut keep_cols: Vec<usize> = (0..m).filter(|&j| evals_full[j] > floor).collect();
    // Stable order (ascending eigenvalues already).
    if keep_cols.is_empty() {
        // Degenerate: just keep the largest.
        let (j_max, _) = evals_full
            .iter()
            .enumerate()
            .fold((0_usize, f64::NEG_INFINITY), |(jb, vb), (j, &v)| {
                if v > vb { (j, v) } else { (jb, vb) }
            });
        keep_cols.push(j_max);
    }
    let m_eff = keep_cols.len();

    // K_nm (n × m), then phi = K_nm U S^{-1/2}, kept-cols only.
    // We compute row-by-row to avoid allocating K_nm explicitly:
    //   phi[i, k] = sum_a K_nm[i, a] * U[a, keep_cols[k]] / sqrt(evals[keep_cols[k]])
    let mut phi = vec![0.0_f32; n * m_eff];
    let inv_sqrt: Vec<f64> = keep_cols
        .iter()
        .map(|&j| 1.0 / evals_full[j].sqrt())
        .collect();

    // Cache U columns for kept eigenvalues, flat row-major (m × m_eff).
    let mut u_keep = vec![0.0_f64; m * m_eff];
    for (kk, &j) in keep_cols.iter().enumerate() {
        for a in 0..m {
            u_keep[a * m_eff + kk] = evecs_full[(a, j)];
        }
    }

    let mut k_row = vec![0.0_f64; m];
    for i in 0..n {
        let (xi, yi) = coords[i];
        for a in 0..m {
            let dx = xi - landmark_pts[a].0;
            let dy = yi - landmark_pts[a].1;
            let d2 = dx * dx + dy * dy;
            k_row[a] = (-d2 / two_sigma_sq).exp() as f64;
        }
        for kk in 0..m_eff {
            let mut acc = 0.0_f64;
            for a in 0..m {
                acc += k_row[a] * u_keep[a * m_eff + kk];
            }
            phi[i * m_eff + kk] = (acc * inv_sqrt[kk]) as f32;
        }
    }

    Ok((phi, m_eff))
}

/// Closed-form rank-4 factorisation of the per-dimension cosine kernel.
///
/// `K(x, y; l) = sum_d cos(2 pi (x_d - y_d) / l)` factorises via
/// `cos(a - b) = cos a cos b + sin a sin b` into `r = 2 * D` features per
/// spot. For 2D coords `r = 4`, columns
/// `[cos(omega x_1), sin(omega x_1), cos(omega x_2), sin(omega x_2)]` with
/// `omega = 2 pi / l`.
fn build_cosine_features(coords: &[(f32, f32)], period: f32) -> (Vec<f32>, usize) {
    let n = coords.len();
    let r = 4_usize;
    let omega = (2.0 * std::f32::consts::PI) / period;
    let mut phi = vec![0.0_f32; n * r];
    for i in 0..n {
        let (x, y) = coords[i];
        let ax = omega * x;
        let ay = omega * y;
        phi[i * r] = ax.cos();
        phi[i * r + 1] = ax.sin();
        phi[i * r + 2] = ay.cos();
        phi[i * r + 3] = ay.sin();
    }
    (phi, r)
}

/// Subtract each column mean from a row-major `n × r` matrix in place.
fn centre_columns(phi: &mut [f32], n: usize, r: usize) {
    if n == 0 || r == 0 {
        return;
    }
    let inv_n = 1.0_f32 / n as f32;
    let mut means = vec![0.0_f32; r];
    for i in 0..n {
        let row = &phi[i * r..i * r + r];
        for k in 0..r {
            means[k] += row[k];
        }
    }
    for k in 0..r {
        means[k] *= inv_n;
    }
    for i in 0..n {
        let row = &mut phi[i * r..i * r + r];
        for k in 0..r {
            row[k] -= means[k];
        }
    }
}

/// Eigenvalues of `phi^T phi` (`r × r`). These equal the non-zero
/// eigenvalues of the full kernel `phi phi^T` (`n × n`, rank ≤ r).
fn phi_c_eigvals(phi: &[f32], n: usize, r: usize) -> Result<Vec<f64>, BixverseErrors> {
    if r == 0 {
        return Ok(Vec::new());
    }
    let mut g = Mat::<f64>::zeros(r, r);
    for k1 in 0..r {
        for k2 in k1..r {
            let mut acc = 0.0_f64;
            for i in 0..n {
                acc += phi[i * r + k1] as f64 * phi[i * r + k2] as f64;
            }
            g[(k1, k2)] = acc;
            g[(k2, k1)] = acc;
        }
    }
    let eig = g
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| BixverseErrors::FaerEigenError)?;
    let lambdas: Vec<f64> = eig
        .S()
        .column_vector()
        .iter()
        .copied()
        .filter(|&v| v > 0.0 && v.is_finite())
        .collect();
    Ok(lambdas)
}

/// Compute `out = phi^T y` for row-major `phi` of shape `(n, r)`.
fn phi_t_y(phi: &[f32], r: usize, n: usize, y: &[f32], out: &mut [f32]) {
    for k in 0..r {
        out[k] = 0.0;
    }
    for i in 0..n {
        let yi = y[i];
        let row = &phi[i * r..i * r + r];
        for k in 0..r {
            out[k] += row[k] * yi;
        }
    }
}

/////////////////////
// Bandwidth bank  //
/////////////////////

/// Build the paper-default 5+5 kernel bank from coord pairwise-distance
/// quantiles on a sub-sample of size `n_sub`.
fn default_kernel_bank(
    coords: &[(f32, f32)],
    n_sub: usize,
    seed: u64,
) -> Vec<SparkXKernel> {
    let dists = pairwise_distance_quantiles(coords, n_sub, seed, &[0.2, 0.4, 0.6, 0.8, 1.0]);
    let mut bank = Vec::with_capacity(10);
    for &d in &dists {
        if d > 0.0 && d.is_finite() {
            bank.push(SparkXKernel::Gaussian { bandwidth: d });
        }
    }
    for &d in &dists {
        if d > 0.0 && d.is_finite() {
            bank.push(SparkXKernel::Cosine { bandwidth: d });
        }
    }
    bank
}

/// Approximate quantiles of the upper-triangular pairwise Euclidean
/// distances on a sub-sample of `coords` of size `n_sub`. Returns one value
/// per requested quantile in `[0, 1]`. Quantiles are clamped to in-range
/// before lookup.
fn pairwise_distance_quantiles(
    coords: &[(f32, f32)],
    n_sub: usize,
    seed: u64,
    qs: &[f64],
) -> Vec<f32> {
    let n = coords.len();
    let m = n_sub.min(n);
    if m < 2 {
        return vec![0.0; qs.len()];
    }
    let idx: Vec<usize> = if m == n {
        (0..n).collect()
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut buf: Vec<usize> = (0..n).collect();
        buf.shuffle(&mut rng);
        buf.into_iter().take(m).collect()
    };

    let mut dists: Vec<f32> = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        let (xi, yi) = coords[idx[i]];
        for j in (i + 1)..m {
            let (xj, yj) = coords[idx[j]];
            let dx = xi - xj;
            let dy = yi - yj;
            dists.push((dx * dx + dy * dy).sqrt());
        }
    }
    dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = dists.len();
    qs.iter()
        .map(|&q| {
            let qc = q.clamp(0.0, 1.0);
            let pos = ((len as f64 - 1.0) * qc).round() as usize;
            dists[pos.min(len - 1)]
        })
        .collect()
}

///////////////////
// k-means++ pts //
///////////////////

/// k-means++ seeding on (x, y) coordinates. Picks `k` distinct point indices.
fn kmeans_pp_indices(coords: &[(f32, f32)], k: usize, seed: u64) -> Vec<usize> {
    let n = coords.len();
    let k = k.min(n).max(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    let first = rng.random_range(0..n);
    chosen.push(first);

    // Squared min-distance to nearest chosen.
    let mut min_d2: Vec<f32> = (0..n)
        .map(|i| sq_dist(coords[i], coords[first]))
        .collect();

    while chosen.len() < k {
        let sum: f64 = min_d2.iter().map(|&v| v as f64).sum();
        if !(sum.is_finite() && sum > 0.0) {
            // All remaining points are identical to a chosen centre. Fill
            // arbitrarily.
            for i in 0..n {
                if !chosen.contains(&i) {
                    chosen.push(i);
                    if chosen.len() == k {
                        break;
                    }
                }
            }
            break;
        }
        let pick: f64 = rng.random_range(0.0..sum);
        let mut acc = 0.0_f64;
        let mut idx = n - 1;
        for i in 0..n {
            acc += min_d2[i] as f64;
            if acc >= pick {
                idx = i;
                break;
            }
        }
        // De-dup against existing chosen.
        if chosen.contains(&idx) {
            // Pick any other unchosen point with non-zero distance.
            if let Some(i) = (0..n).find(|i| !chosen.contains(i) && min_d2[*i] > 0.0) {
                chosen.push(i);
            } else {
                break;
            }
        } else {
            chosen.push(idx);
        }
        let new = *chosen.last().unwrap();
        for i in 0..n {
            let d = sq_dist(coords[i], coords[new]);
            if d < min_d2[i] {
                min_d2[i] = d;
            }
        }
    }
    chosen
}

#[inline]
fn sq_dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

//////////////////////
// Saddle-point CDF //
//////////////////////

/// Upper-tail probability `P(sum_k lambda_k * Q_k > q)` where `Q_k ~ chi^2_1`
/// independently, via the Kuonen (1999) saddle-point approximation.
///
/// `lambdas` must be all positive. Returns `1.0` for `q <= 0`. When the
/// saddle-point is close to zero (`q` near the mean), the standard
/// Barndorff-Nielsen formula degenerates and a normal approximation is
/// inaccurate for skewed mixtures with few eigenvalues; in that regime we
/// fall back to Liu's moment-matched chi-squared approximation
/// ([`liu_upper_tail`]).
pub(crate) fn saddle_point_upper_tail(q: f64, lambdas: &[f64]) -> f64 {
    if lambdas.is_empty() {
        return f64::NAN;
    }
    if !q.is_finite() {
        return if q > 0.0 { 0.0 } else { 1.0 };
    }
    if q <= 0.0 {
        return 1.0;
    }

    let mean: f64 = lambdas.iter().sum();
    if !mean.is_finite() || mean <= 0.0 {
        return f64::NAN;
    }

    let lambda_max = lambdas.iter().cloned().fold(0.0_f64, f64::max);
    if lambda_max <= 0.0 {
        return f64::NAN;
    }
    let t_max = 1.0 / (2.0 * lambda_max);

    let t_hat = solve_saddle(q, lambdas, t_max);
    if !t_hat.is_finite() {
        return liu_upper_tail(q, lambdas);
    }
    // Near-zero saddle: Barndorff-Nielsen formula degenerates here; Liu's
    // approximation handles the q ≈ mean region correctly even for skewed
    // (small-r) mixtures.
    if t_hat.abs() < 1e-5 {
        return liu_upper_tail(q, lambdas);
    }

    let mut k_t = 0.0_f64;
    let mut k_pp = 0.0_f64;
    for &l in lambdas {
        let d = 1.0 - 2.0 * l * t_hat;
        if d <= 0.0 || !d.is_finite() {
            return liu_upper_tail(q, lambdas);
        }
        k_t += -0.5 * d.ln();
        let r = l / d;
        k_pp += 2.0 * r * r;
    }
    let arg = 2.0 * (t_hat * q - k_t);
    if arg <= 0.0 || !arg.is_finite() || k_pp <= 0.0 || !k_pp.is_finite() {
        return liu_upper_tail(q, lambdas);
    }
    let w = t_hat.signum() * arg.sqrt();
    let v = t_hat * k_pp.sqrt();
    if w.abs() < 1e-10 || v.abs() < 1e-300 {
        return liu_upper_tail(q, lambdas);
    }

    let z_star = w + (v / w).ln() / w;
    let p = normal_upper(z_star);
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return liu_upper_tail(q, lambdas);
    }
    p
}

/// Upper-tail probability `P(sum_k lambda_k * Q_k > q)` via Liu's
/// moment-matched chi-squared approximation (Liu, Lin & Ghosh 2009, "Liu
/// modified" / 3-parameter form). Matches mean, variance, and skewness to a
/// scaled-and-shifted central chi-squared.
///
/// Handles `q = mean` correctly (the saddle-point formula does not). Used
/// both standalone and as the fallback in [`saddle_point_upper_tail`].
pub(crate) fn liu_upper_tail(q: f64, lambdas: &[f64]) -> f64 {
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    if lambdas.is_empty() {
        return f64::NAN;
    }
    if !q.is_finite() {
        return if q > 0.0 { 0.0 } else { 1.0 };
    }
    if q <= 0.0 {
        return 1.0;
    }

    let c1: f64 = lambdas.iter().sum();
    let c2: f64 = lambdas.iter().map(|l| l * l).sum();
    let c3: f64 = lambdas.iter().map(|l| l * l * l).sum();
    if !(c1.is_finite() && c2.is_finite() && c3.is_finite()) {
        return f64::NAN;
    }
    if c2 <= 0.0 {
        return f64::NAN;
    }
    let mu_q = c1;
    let sigma_q = (2.0 * c2).sqrt();
    if sigma_q <= 0.0 || !sigma_q.is_finite() {
        return f64::NAN;
    }

    // Match skewness of a central chi^2_df: gamma1(chi^2_df) = sqrt(8/df).
    // gamma1(Q) = sqrt(8) * c3 / c2^(3/2)  =>  df = c2^3 / c3^2.
    // When c3 == 0 (all eigenvalues equal, distribution exactly chi^2_k * lambda),
    // df defaults to a large value -> distribution treated as normal, which
    // matches the symmetric limit.
    let df = if c3 > 0.0 {
        c2 * c2 * c2 / (c3 * c3)
    } else {
        // Symmetric: use a large df so chi^2 looks ≈ normal.
        1e6_f64
    };
    if !df.is_finite() || df <= 0.0 {
        return f64::NAN;
    }

    let mu_x = df;
    let sigma_x = (2.0 * df).sqrt();

    // X corresponds to q at the same standardised distance from each mean.
    let q_std = (q - mu_q) / sigma_q;
    let x = q_std * sigma_x + mu_x;
    if x <= 0.0 {
        return 1.0;
    }

    let chi = match ChiSquared::new(df) {
        Ok(c) => c,
        Err(_) => return f64::NAN,
    };
    (1.0 - chi.cdf(x)).clamp(0.0, 1.0)
}

/// Solve `K'(t) = q` for the saddle-point `t`, with `t` constrained to
/// `(-inf, t_max)` where `t_max = 1/(2 * max(lambda))`. Returns `NaN` on
/// numerical failure.
fn solve_saddle(q: f64, lambdas: &[f64], t_max: f64) -> f64 {
    // Newton on K'(t) = q, with a small bracketed safeguard against
    // overshooting past t_max (which would push some 1 - 2*lambda*t to 0/neg).
    let lo = -1e9_f64;
    let hi = t_max * (1.0 - 1e-6); // strict interior
    let mut t = 0.0_f64;

    for _ in 0..200 {
        let mut kp = 0.0_f64;
        let mut kpp = 0.0_f64;
        for &l in lambdas {
            let d = 1.0 - 2.0 * l * t;
            if d <= 0.0 || !d.is_finite() {
                return f64::NAN;
            }
            kp += l / d;
            // K''(t) = 2 * sum_k lambda_k^2 / (1 - 2 lambda_k t)^2
            let r = l / d;
            kpp += 2.0 * r * r;
        }
        let f = kp - q;
        if !f.is_finite() || !kpp.is_finite() || kpp <= 0.0 {
            return f64::NAN;
        }
        let mut step = f / kpp;
        // Clip step so we stay strictly inside (lo, hi).
        let mut t_new = t - step;
        if t_new >= hi {
            // Half-step toward hi.
            t_new = 0.5 * (t + hi);
            step = t - t_new;
        }
        if t_new <= lo {
            t_new = 0.5 * (t + lo);
            step = t - t_new;
        }
        if step.abs() < 1e-12 || (t_new - t).abs() < 1e-12 {
            return t_new;
        }
        t = t_new;
    }
    t
}

/// Upper tail of the standard normal: `P(Z > z) = 0.5 * erfc(z / sqrt(2))`.
#[inline]
fn normal_upper(z: f64) -> f64 {
    use statrs::function::erf::erfc;
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

//////////////////////////
// Cauchy combination   //
//////////////////////////

/// Cauchy combination test (Liu & Xie 2020) for `L` p-values.
///
/// `T = (1/L) * sum_l tan((0.5 - p_l) * pi)`,
/// `p_combined = 0.5 - atan(T) / pi`. Clamps inputs to `[eps, 1-eps]` so the
/// `tan` arguments stay finite. Drops NaN inputs from the combination; if
/// all inputs are NaN, returns NaN.
pub(crate) fn cauchy_combination(pvals: &[f64]) -> f64 {
    let mut t_sum = 0.0_f64;
    let mut n_used = 0_usize;
    let eps = 1e-15_f64;
    for &p in pvals {
        if !p.is_finite() {
            continue;
        }
        let pc = p.clamp(eps, 1.0 - eps);
        // tan((0.5 - p) * pi). For p in (0, 1), this is finite after the clamp.
        let arg = (0.5 - pc) * std::f64::consts::PI;
        t_sum += arg.tan();
        n_used += 1;
    }
    if n_used == 0 {
        return f64::NAN;
    }
    let t_avg = t_sum / n_used as f64;
    0.5 - t_avg.atan() / std::f64::consts::PI
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Cosine features ----

    #[test]
    fn cosine_features_reconstruct_per_dim_kernel() {
        let coords = vec![(0.0_f32, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 1.5)];
        let period = 4.0_f32;
        let (phi, r) = build_cosine_features(&coords, period);
        assert_eq!(r, 4);
        // Direct K(i, j) = cos(omega (x_i - x_j)) + cos(omega (y_i - y_j))
        let omega = (2.0 * std::f32::consts::PI) / period;
        let n = coords.len();
        for i in 0..n {
            for j in 0..n {
                let direct = (omega * (coords[i].0 - coords[j].0)).cos()
                    + (omega * (coords[i].1 - coords[j].1)).cos();
                let mut via_phi = 0.0_f32;
                for k in 0..r {
                    via_phi += phi[i * r + k] * phi[j * r + k];
                }
                assert!(
                    (direct - via_phi).abs() < 1e-5,
                    "K[{i},{j}]: {direct} vs {via_phi}"
                );
            }
        }
    }

    // ---- phi^T phi spectrum vs full K spectrum ----

    #[test]
    fn phi_t_phi_eigvals_match_phi_phi_t_nonzero() {
        // Tiny Phi (n=5, r=2), centred. The non-zero eigvals of K = Phi Phi'
        // (5 × 5, rank ≤ 2) must equal the eigvals of Phi' Phi (2 × 2).
        let n = 5_usize;
        let r = 2_usize;
        let mut phi = vec![
            1.0_f32, 0.5,
            0.7, 1.1,
            -0.4, 0.9,
            -0.8, -0.2,
            0.3, -1.0,
        ];
        // Centre columns to match our pipeline.
        centre_columns(&mut phi, n, r);
        let lambdas_small = phi_c_eigvals(&phi, n, r).unwrap();

        // Build K = Phi Phi' explicitly and decompose.
        let mut k = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f64;
                for s in 0..r {
                    acc += phi[i * r + s] as f64 * phi[j * r + s] as f64;
                }
                k[(i, j)] = acc;
            }
        }
        let eig = k.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let mut evals_full: Vec<f64> = eig
            .S()
            .column_vector()
            .iter()
            .copied()
            .filter(|v| v.abs() > 1e-8)
            .collect();
        evals_full.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut sm = lambdas_small.clone();
        sm.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(sm.len(), evals_full.len(), "rank mismatch");
        for (a, b) in sm.iter().zip(evals_full.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    // ---- Saddle-point against closed-form cases ----

    fn normal_cdf(z: f64) -> f64 {
        use statrs::function::erf::erfc;
        0.5 * erfc(-z / std::f64::consts::SQRT_2)
    }

    #[test]
    fn saddle_single_chi_sq_matches_closed_form() {
        // P(chi_sq_1 > q) = 2 * (1 - Phi(sqrt(q))). With Liu fallback at the
        // mean, the function matches the closed form across all q tested.
        let lambdas = vec![1.0_f64];
        for &q in &[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0] {
            let p_sp = saddle_point_upper_tail(q, &lambdas);
            let p_ref = 2.0 * (1.0 - normal_cdf(q.sqrt()));
            // Saddle-point is asymptotic; tail accuracy ~ a few percent of the
            // reference (the Liu-fallback path is exact for this single-lambda
            // case, but the regular saddle branch still has approximation error).
            let tol = 0.05_f64 * p_ref + 1e-3;
            assert!(
                (p_sp - p_ref).abs() < tol,
                "q={q}: saddle={p_sp} vs closed={p_ref}"
            );
        }
    }

    #[test]
    fn saddle_chi_sq_two_dof_matches_exp() {
        // sum_k lambda_k chi_sq_1 with two unit weights = chi_sq_2.
        // P(chi_sq_2 > q) = exp(-q/2)
        let lambdas = vec![1.0_f64, 1.0];
        for &q in &[0.5, 1.0, 2.0, 4.0, 6.0, 8.0] {
            let p_sp = saddle_point_upper_tail(q, &lambdas);
            let p_ref = (-q / 2.0).exp();
            let tol = 0.02_f64 * p_ref + 1e-3;
            assert!(
                (p_sp - p_ref).abs() < tol,
                "q={q}: saddle={p_sp} vs closed={p_ref}"
            );
        }
    }

    #[test]
    fn saddle_scaled_chi_sq_two() {
        // 0.5 * chi_sq_2: P(0.5 Y > q) = P(Y > 2q) = exp(-q)
        let lambdas = vec![0.5_f64, 0.5];
        for &q in &[0.25, 0.5, 1.0, 2.0, 3.0] {
            let p_sp = saddle_point_upper_tail(q, &lambdas);
            let p_ref = (-q).exp();
            let tol = 0.02_f64 * p_ref + 1e-3;
            assert!(
                (p_sp - p_ref).abs() < tol,
                "q={q}: saddle={p_sp} vs closed={p_ref}"
            );
        }
    }

    #[test]
    fn saddle_q_le_zero_returns_one() {
        let lambdas = vec![1.0_f64, 2.0, 0.5];
        assert_eq!(saddle_point_upper_tail(0.0, &lambdas), 1.0);
        assert_eq!(saddle_point_upper_tail(-0.1, &lambdas), 1.0);
    }

    #[test]
    fn saddle_at_mean_recovers_chi_sq_one() {
        // q = mean = 1 for chi_sq_1; saddle degenerates at t=0, Liu fallback
        // recovers the correct ~0.317 (= 2 * (1 - Phi(1))).
        let lambdas = vec![1.0_f64];
        let p = saddle_point_upper_tail(1.0, &lambdas);
        let p_ref = 2.0 * (1.0 - normal_cdf(1.0));
        assert!((p - p_ref).abs() < 0.02, "{p} vs {p_ref}");
    }

    // ---- Liu's approximation: independent checks ----

    #[test]
    fn liu_single_chi_sq_matches_closed_form() {
        // For single lambda=1, Liu reduces to chi^2_1.
        let lambdas = vec![1.0_f64];
        for &q in &[0.1, 0.5, 1.0, 2.0, 3.0, 5.0] {
            let p_liu = liu_upper_tail(q, &lambdas);
            let p_ref = 2.0 * (1.0 - normal_cdf(q.sqrt()));
            assert!(
                (p_liu - p_ref).abs() < 1e-6,
                "q={q}: liu={p_liu} vs closed={p_ref}"
            );
        }
    }

    #[test]
    fn liu_chi_sq_two_dof_matches_exp() {
        let lambdas = vec![1.0_f64, 1.0];
        for &q in &[0.5, 1.0, 2.0, 4.0, 6.0, 8.0] {
            let p_liu = liu_upper_tail(q, &lambdas);
            let p_ref = (-q / 2.0).exp();
            assert!(
                (p_liu - p_ref).abs() < 1e-6,
                "q={q}: liu={p_liu} vs closed={p_ref}"
            );
        }
    }

    #[test]
    fn liu_q_le_zero_returns_one() {
        assert_eq!(liu_upper_tail(0.0, &[1.0_f64, 2.0]), 1.0);
        assert_eq!(liu_upper_tail(-1.0, &[1.0_f64, 2.0]), 1.0);
    }

    // ---- Cauchy combination ----

    #[test]
    fn cauchy_combination_uniform_inputs_match_single() {
        // All p_l equal -> p_combined = p_l (combine identical p-values).
        let p = 0.05;
        let pvals = vec![p; 5];
        let c = cauchy_combination(&pvals);
        assert!((c - p).abs() < 1e-10, "{c} vs {p}");
    }

    #[test]
    fn cauchy_combination_one_zero_dominates() {
        // One p = 0 forces combined ≈ 0.
        let pvals = vec![0.0_f64, 0.5, 0.5, 0.5];
        let c = cauchy_combination(&pvals);
        assert!(c < 1e-10, "{c}");
    }

    #[test]
    fn cauchy_combination_drops_nans() {
        let pvals = vec![f64::NAN, 0.1, 0.1];
        let c = cauchy_combination(&pvals);
        assert!((c - 0.1).abs() < 1e-10, "{c}");
    }

    #[test]
    fn cauchy_combination_all_nan_is_nan() {
        let pvals = vec![f64::NAN, f64::NAN];
        assert!(cauchy_combination(&pvals).is_nan());
    }

    // ---- k-means++ ----

    #[test]
    fn kmeans_pp_returns_distinct_indices() {
        let coords: Vec<(f32, f32)> = (0..50).map(|i| (i as f32, (i * i) as f32 * 0.1)).collect();
        let k = 5;
        let chosen = kmeans_pp_indices(&coords, k, 42);
        assert_eq!(chosen.len(), k);
        let mut sorted = chosen.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), k, "indices must be distinct");
        for &i in &chosen {
            assert!(i < coords.len());
        }
    }

    #[test]
    fn kmeans_pp_more_landmarks_than_points_capped() {
        let coords = vec![(0.0_f32, 0.0), (1.0, 1.0)];
        let chosen = kmeans_pp_indices(&coords, 5, 0);
        assert_eq!(chosen.len(), 2);
    }

    // ---- Bandwidth bank ----

    #[test]
    fn default_kernel_bank_is_five_plus_five() {
        let coords: Vec<(f32, f32)> = (0..30)
            .map(|i| ((i % 6) as f32, (i / 6) as f32))
            .collect();
        let bank = default_kernel_bank(&coords, 1000, 0);
        assert_eq!(bank.len(), 10);
        let n_gauss = bank
            .iter()
            .filter(|k| matches!(k, SparkXKernel::Gaussian { .. }))
            .count();
        let n_cos = bank
            .iter()
            .filter(|k| matches!(k, SparkXKernel::Cosine { .. }))
            .count();
        assert_eq!(n_gauss, 5);
        assert_eq!(n_cos, 5);
    }

    // ---- End-to-end small kernel factor ----

    #[test]
    fn gaussian_nystroem_kernel_factor_positive_semidef() {
        let coords: Vec<(f32, f32)> = (0..20)
            .map(|i| ((i % 5) as f32, (i / 5) as f32))
            .collect();
        let factor = build_kernel_factor(
            SparkXKernel::Gaussian { bandwidth: 1.5 },
            &coords,
            8,
            0,
        )
        .unwrap();
        assert!(!factor.lambdas.is_empty());
        for &l in &factor.lambdas {
            assert!(l > 0.0, "negative eigenvalue from Nystroem: {l}");
        }
    }

    #[test]
    fn cosine_kernel_factor_has_rank_four() {
        let coords: Vec<(f32, f32)> = (0..30)
            .map(|i| (i as f32 * 0.3, (i as f32).sin()))
            .collect();
        let factor = build_kernel_factor(
            SparkXKernel::Cosine { bandwidth: 3.0 },
            &coords,
            0,
            0,
        )
        .unwrap();
        assert_eq!(factor.r, 4);
        assert!(factor.lambdas.len() <= 4);
    }

    // ---- Constructor validation ----

    #[test]
    fn new_rejects_coord_count_mismatch() {
        let spots = vec![0_usize, 1, 2, 3, 4];
        let coords = vec![(0.0_f32, 0.0); 4]; // wrong length
        let err = SparkX::new(
            "/dev/null".to_string(),
            &spots,
            &coords,
            SparkXParams::default(),
        )
        .unwrap_err();
        match err {
            BixverseErrors::SpatialCoordSpotMismatch { n_coords, n_spots } => {
                assert_eq!(n_coords, 4);
                assert_eq!(n_spots, 5);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn new_rejects_too_few_spots() {
        let spots = vec![0_usize, 1];
        let coords = vec![(0.0_f32, 0.0), (1.0, 0.0)];
        let err = SparkX::new(
            "/dev/null".to_string(),
            &spots,
            &coords,
            SparkXParams::default(),
        )
        .unwrap_err();
        match err {
            BixverseErrors::InvalidArgument(_) => {}
            other => panic!("wrong error: {other:?}"),
        }
    }

    // ---- Per-gene reference path ----
    //
    // Compute T_l directly and verify it matches the projection-based path:
    // T = (n-1) * ||phi_c' y_c||^2 / SST
    //   = (n-1) * y_c' phi_c phi_c' y_c / SST.
    #[test]
    fn projection_vs_full_kernel_quadratic_agrees() {
        let coords: Vec<(f32, f32)> = (0..15)
            .map(|i| ((i % 5) as f32, (i / 5) as f32))
            .collect();
        let factor =
            build_kernel_factor(SparkXKernel::Gaussian { bandwidth: 1.0 }, &coords, 6, 7).unwrap();
        let n = coords.len();
        let r = factor.r;

        // synthetic centred y
        let mut y: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.7).sin()) as f32).collect();
        let (_m, sst) = center_inplace(&mut y);
        assert!(sst > 0.0);

        // Path A: projection
        let mut proj = vec![0.0_f32; r];
        phi_t_y(&factor.data, r, n, &y, &mut proj);
        let num_proj: f64 = proj.iter().map(|&p| (p as f64).powi(2)).sum();

        // Path B: build K = phi_c phi_c' and compute y' K y
        let mut k_mat = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f64;
                for s in 0..r {
                    acc += factor.data[i * r + s] as f64 * factor.data[j * r + s] as f64;
                }
                k_mat[i * n + j] = acc;
            }
        }
        let mut num_full = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                num_full += y[i] as f64 * y[j] as f64 * k_mat[i * n + j];
            }
        }

        assert!(
            (num_proj - num_full).abs() < 1e-6_f64.max(1e-6 * num_full.abs()),
            "proj={num_proj} vs full={num_full}"
        );
    }
}
