//! Implementation of the MELD method from Burkhardt, et al., Nat. Biotechnol.,
//! 2021. For large scale data sets, the landmark method is available. Estimates
//! per-condition cell densities by spectrally low-pass filtering one-hot sample
//! indicator vectors over a cell-cell kNN graph using a Chebyshev polynomial
//! approximation of the filter.

use faer::{Mat, MatRef};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;
use thousands::Separable;

use crate::core::math::sparse::*;
use crate::prelude::*;
use crate::single_cell::mc_generation::seacells::compute_diffusion_kernel;

///////////
// Types //
///////////

/// MELD results
///
/// ### Fields
///
/// * `0` - Raw scores
/// * `1` - Normalised scores (negative clamped away)
pub type MeldResult = Result<(Mat<f32>, Mat<f32>), BixverseErrors>;

///////////
// Enums //
///////////

/// MELD filter family.
#[derive(Clone, Copy, Debug, Default)]
pub enum MeldFilter {
    #[default]
    /// Heat version: `h(λ) = exp(-β |λ/λ_max - offset|^order)`
    Heat,
    /// Laplacian: `h(λ) = 1 / (1 + (β |λ/λ_max - offset|)^order)`
    Laplacian,
}

/// Type of Laplacian to use for spectral filtering.
#[derive(Clone, Copy, Debug, Default)]
pub enum LaplacianType {
    /// Combinatorial: `L = D - A`
    #[default]
    Combinatorial,
    /// Normalised: `L = I - D^(-1/2) A D^(-1/2)`
    Normalised,
}

/// Helper to parse the filter type from a string.
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// The option of [MeldFilter]
pub fn parse_meld_filter(s: &str) -> Option<MeldFilter> {
    match s.to_lowercase().as_str() {
        "heat" => Some(MeldFilter::Heat),
        "laplacian" => Some(MeldFilter::Laplacian),
        _ => None,
    }
}

/// Helper to parse the Laplacian type from a string.
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// The option of [LaplacianType]
pub fn parse_lap_type(s: &str) -> Option<LaplacianType> {
    match s.to_lowercase().as_str() {
        "combinatorial" => Some(LaplacianType::Combinatorial),
        "normalised" | "normalized" => Some(LaplacianType::Normalised),
        _ => None,
    }
}

/// Parameters for MELD.
/// * `order` - Filter falloff sharpness. Larger values approach a square
///   low-pass.
/// * `filter` - Filter family ([`MeldFilter::Heat`] or
///   [`MeldFilter::Laplacian`]).
/// * `chebyshev_order` - Number of Chebyshev coefficients (polynomial terms).
///   Reference default: 50. Must be >= 2.
/// * `lap_type` - [`LaplacianType::Combinatorial`] or
///   [`LaplacianType::Normalised`].
/// * `normalise_indicators` - If true, each column of the indicator matrix is
///   divided by its column sum before filtering, so cross-condition densities
///   are comparable regardless of cells-per-condition.
/// * `knn_params` - kNN parameters used by the landmark variant for the
///   landmark-landmark kNN.
#[derive(Clone, Debug)]
pub struct MeldParams {
    /// Smoothing strength. Larger values produce smoother densities. Reference
    /// default: 60.
    pub beta: f32,
    /// Shift of the filter centre in the rescaled spectrum `[0, 1]`.
    pub offset: f32,
    /// Filter falloff sharpness. Larger values approach a square low-pass.
    pub order: f32,
    /// Filter type, i.e. [MeldFilter]
    pub filter: MeldFilter,
    /// Number of Chebyshev coefficients (polynomial terms). Must be >= 2.
    pub chebyshev_order: usize,
    /// The [LaplacianType] type enum indicating which Laplacian to use.
    pub lap_type: LaplacianType,
    /// If true, each column of the indicator matrix is divided by its column
    /// sum before filtering, so cross-condition densities are comparable
    /// regardless of cells-per-condition.
    pub normalise_indicators: bool,
    /// [KnnParams] for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
}

impl MeldParams {
    /// Sensible defaults matching the reference Python MELD constructor.
    pub fn new() -> Self {
        Self {
            beta: 60.0,
            offset: 0.0,
            order: 1.0,
            filter: MeldFilter::Heat,
            chebyshev_order: 50,
            lap_type: LaplacianType::Combinatorial,
            normalise_indicators: true,
            knn_params: KnnParams::new(),
        }
    }
}

impl Default for MeldParams {
    fn default() -> Self {
        Self::new()
    }
}

/////////////
// Helpers //
/////////////

/// Build an N x p one-hot indicator matrix from category labels.
///
/// Optionally L1-column-normalises so each column sums to 1, making
/// per-condition densities comparable across conditions of different size.
///
/// ### Params
///
/// * `labels` - Per-cell category labels, integers in `0..n_groups`.
/// * `n_groups` - Number of unique groups (columns of the output).
/// * `normalise` - If true, each column is divided by its sum.
///
/// ### Returns
///
/// `Mat<f32>` of shape samples x labels
fn build_indicator_matrix(
    labels: &[usize],
    n_groups: usize,
    normalise: bool,
) -> Result<Mat<f32>, BixverseErrors> {
    let n = labels.len();
    let mut counts = vec![0_usize; n_groups];
    for &l in labels {
        if l > n_groups {
            return Err(BixverseErrors::MELDLabelOutOfRange { label: l, n_groups });
        }
        counts[l] += 1;
    }
    let scales: Vec<f32> = if normalise {
        counts
            .iter()
            .map(|&c| if c > 0 { 1.0 / c as f32 } else { 0.0 })
            .collect()
    } else {
        vec![1.0_f32; n_groups]
    };

    Ok(Mat::from_fn(n, n_groups, |i, j| {
        if labels[i] == j { scales[j] } else { 0.0 }
    }))
}

/// Row sums of a CSR adjacency.
///
/// ### Params
///
/// * `adj` - The adjacency matrix
///
/// ### Returns
///
/// The node degree (row sums of the CSR matrix)
fn compute_degrees(adj: &CompressedSparseData2<f32>) -> Vec<f32> {
    (0..adj.shape.0)
        .map(|i| {
            (adj.indptr[i]..adj.indptr[i + 1])
                .map(|idx| adj.data[idx as usize])
                .sum::<f32>()
        })
        .collect()
}

/// Build the combinatorial Laplacian `L = D - A` as CSR.
///
/// ### Params
///
/// * `adj` - The adjacency matrix
///
/// ### Returns
///
/// The combinatorial Laplacian as [CompressedSparseData2]
fn build_combinatorial_laplacian(adj: &CompressedSparseData2<f32>) -> CompressedSparseData2<f32> {
    let n = adj.shape.0;
    let degrees = compute_degrees(adj);

    let mut rows = Vec::with_capacity(adj.get_nnz() + n);
    let mut cols = Vec::with_capacity(adj.get_nnz() + n);
    let mut vals = Vec::with_capacity(adj.get_nnz() + n);

    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(degrees[i]);
        let diag_pos = vals.len() - 1;

        for idx in adj.indptr[i]..adj.indptr[i + 1] {
            let idx_usize = idx as usize;

            let j = adj.indices[idx_usize] as usize;
            if j == i {
                vals[diag_pos] -= adj.data[idx_usize];
            } else {
                rows.push(i);
                cols.push(j);
                vals.push(-adj.data[idx_usize]);
            }
        }
    }

    coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, n))
}

/// Build the symmetric normalised Laplacian `L = I - D^(-1/2) A D^(-1/2)` as
/// CSR.
///
/// ### Params
///
/// * `adj` - The adjacency matrix
///
/// ### Returns
///
/// The combinatorial Laplacian as [CompressedSparseData2]
fn build_normalised_laplacian(adj: &CompressedSparseData2<f32>) -> CompressedSparseData2<f32> {
    let n = adj.shape.0;
    let degrees = compute_degrees(adj);
    let inv_sqrt_d: Vec<f32> = degrees
        .iter()
        .map(|&d| if d > 1e-12 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let mut rows = Vec::with_capacity(adj.get_nnz() + n);
    let mut cols = Vec::with_capacity(adj.get_nnz() + n);
    let mut vals = Vec::with_capacity(adj.get_nnz() + n);

    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(1.0);
        let diag_pos = vals.len() - 1;

        for idx in adj.indptr[i]..adj.indptr[i + 1] {
            let idx_usize = idx as usize;

            let j = adj.indices[idx_usize] as usize;
            let scaled = adj.data[idx_usize] * inv_sqrt_d[i] * inv_sqrt_d[j];
            if j == i {
                vals[diag_pos] -= scaled;
            } else {
                rows.push(i);
                cols.push(j);
                vals.push(-scaled);
            }
        }
    }

    coo_to_csr(&rows.index_cast(), &cols.index_cast(), &vals, (n, n))
}

/// Estimate the largest eigenvalue of a sparse symmetric matrix via Lanczos.
///
/// ### Params
///
/// * `lap` - The Laplacian for which to estimate the largest eigenvalue
///
/// ### Returns
///
/// The largest eigenvalue of the provided Laplacian matrix
fn estimate_lmax(lap: &CompressedSparseData2<f32>, seed: u64) -> Result<f32, BixverseErrors> {
    let (evals, _) = compute_largest_eigenpairs_lanczos(lap, 1, seed)?;
    Ok(evals[0])
}

/// Compute Chebyshev coefficients of the MELD filter on `[0, lmax]`.
///
/// Uses standard discrete cosine sampling at Chebyshev nodes mapped to the
/// interval `[0, lmax]`. Reconstruction convention is:
///
/// `h(λ) ≈ c[0]/2 + Σ c[k] T_k((2λ - λ_max) / λ_max)`.
///
/// ### Params
///
/// * `filter` - Filter family ([MeldFilter::Heat] or [MeldFilter::Laplacian]).
/// * `beta` - Smoothing strength. Larger values produce sharper roll-off.
/// * `offset` - Shift of the filter centre in the rescaled spectrum `[0, 1]`.
/// * `order` - Filter falloff sharpness. Higher values approach a square
///   low-pass.
/// * `lmax` - Largest Laplacian eigenvalue, used to map `[0, lmax]` onto
///   `[-1, 1]`.
/// * `n_coeffs` - Number of Chebyshev coefficients to compute. Must be `>= 2`.
///
/// ### Returns
///
/// Vector of `n_coeffs` Chebyshev coefficients
fn chebyshev_coefficients(
    filter: MeldFilter,
    beta: f32,
    offset: f32,
    order: f32,
    lmax: f32,
    n_coeffs: usize,
) -> Result<Vec<f32>, BixverseErrors> {
    if n_coeffs < 2 {
        return Err(BixverseErrors::MELDChebyshevCoefTooLow);
    }

    let a = lmax / 2.0;
    let b = lmax / 2.0;
    let n_f = n_coeffs as f32;

    let thetas: Vec<f32> = (0..n_coeffs).map(|j| PI * (j as f32 + 0.5) / n_f).collect();

    let samples: Vec<f32> = thetas
        .iter()
        .map(|&theta| {
            let lambda = a + b * theta.cos();
            let arg = (lambda / lmax - offset).abs();
            match filter {
                MeldFilter::Heat => (-beta * arg.powf(order)).exp(),
                MeldFilter::Laplacian => 1.0 / (1.0 + (beta * arg).powf(order)),
            }
        })
        .collect();

    let res = (0..n_coeffs)
        .map(|k| {
            let s: f32 = thetas
                .iter()
                .zip(samples.iter())
                .map(|(&theta, &sample)| sample * (k as f32 * theta).cos())
                .sum();
            (2.0 / n_f) * s
        })
        .collect();

    Ok(res)
}

/// Apply a Chebyshev polynomial approximation of `h(L)` to a single signal.
///
/// Evaluates `h(L) v ≈ (c[0]/2) v + Σ c[k] T_k(M) v` via the three-term
/// recurrence on the rescaled operator `M = (2 L - λ_max I) / λ_max`:
/// `T_0 = v`, `T_1 = M v`, `T_k = 2 M T_{k-1} - T_{k-2}`. Each iteration
/// costs one sparse matvec through `lap`, so total cost is `O(K · nnz(L))`
/// for `K = coeffs.len()`.
///
/// ### Params
///
/// * `lap` - Sparse Laplacian in CSR format.
/// * `coeffs` - Chebyshev coefficients from [`chebyshev_coefficients`]. Length
///   is the polynomial degree plus one.
/// * `lmax` - Largest Laplacian eigenvalue used to rescale the spectrum onto
///   `[-1, 1]`. Must match the value used to compute `coeffs`.
/// * `signal` - The signal `v` to filter, length `lap.shape.0`.
///
/// ### Returns
///
/// Filtered signal `h(L) v` of the same length as `signal`.
fn chebyshev_apply(
    lap: &CompressedSparseData2<f32>,
    coeffs: &[f32],
    lmax: f32,
    signal: &[f32],
) -> Result<Vec<f32>, BixverseErrors> {
    let n = signal.len();
    let n_coeffs = coeffs.len();
    let a = lmax / 2.0;
    let b = lmax / 2.0;
    let inv_b = 1.0 / b;

    let mut t_old = signal.to_vec();

    let lv = csr_matvec(lap, signal)?;
    let mut t_cur: Vec<f32> = lv
        .iter()
        .zip(signal.iter())
        .map(|(&l, &s)| (l - a * s) * inv_b)
        .collect();

    let mut result: Vec<f32> = (0..n)
        .map(|i| 0.5 * coeffs[0] * t_old[i] + coeffs[1] * t_cur[i])
        .collect();

    for k in 2..n_coeffs {
        let lt = csr_matvec(lap, &t_cur)?;
        let mut t_new = vec![0.0_f32; n];
        for i in 0..n {
            t_new[i] = 2.0 * (lt[i] - a * t_cur[i]) * inv_b - t_old[i];
            result[i] += coeffs[k] * t_new[i];
        }
        t_old = std::mem::replace(&mut t_cur, t_new);
    }

    Ok(result)
}

/// Apply a Chebyshev filter to all columns of an `N x p` signal in parallel.
///
/// Each column is filtered independently via [`chebyshev_apply`] and the
/// per-column work is dispatched across threads with rayon. Suited to MELD's
/// indicator matrix, where the `p` columns (one per condition) are
/// independent low-pass-filtering problems sharing the same Laplacian.
///
/// ### Params
///
/// * `lap` - Sparse Laplacian in CSR format.
/// * `coeffs` - Chebyshev coefficients from [`chebyshev_coefficients`].
/// * `lmax` - Largest Laplacian eigenvalue (must match `coeffs`).
/// * `signal` - The `N x p` signal matrix to filter, with `N == lap.shape.0`.
///
/// ### Returns
///
/// Filtered `N x p` matrix where column `j` is `h(L)` applied to column `j`
/// of `signal`.
fn chebyshev_apply_columns(
    lap: &CompressedSparseData2<f32>,
    coeffs: &[f32],
    lmax: f32,
    signal: MatRef<f32>,
) -> Result<Mat<f32>, BixverseErrors> {
    let n = signal.nrows();
    let p = signal.ncols();
    let columns: Vec<Vec<f32>> = (0..p)
        .into_par_iter()
        .map(|j| {
            let col: Vec<f32> = (0..n).map(|i| *signal.get(i, j)).collect();
            chebyshev_apply(lap, coeffs, lmax, &col)
        })
        .collect::<Result<Vec<_>, BixverseErrors>>()?;
    Ok(Mat::from_fn(n, p, |i, j| columns[j][i]))
}

/// Apply L1 normalisation to each row.
///
/// Normalises each row (cell) to unit L1 norm, turning per-condition MELD
/// densities into per-cell condition probabilities that sum to 1. Rows with
/// near-zero norm are left as zeros to avoid division by zero.
///
/// ### Params
///
/// * `mat` - The matrix to row-normalise (cells x conditions)
///
/// ### Returns
///
/// Per-row L1-normalised data
fn l1_normalise(mat: &Mat<f32>) -> Mat<f32> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    let norms: Vec<f32> = (0..nrows)
        .into_par_iter()
        .map(|row| (0..ncols).map(|col| mat[(row, col)].max(0.0)).sum::<f32>())
        .collect();

    Mat::from_fn(nrows, ncols, |row, col| {
        let norm = norms[row];
        if norm > 1e-8 {
            mat[(row, col)].max(0.0) / norm
        } else {
            0.0
        }
    })
}

//////////
// Main //
//////////

/// Run MELD on the full N x N graph.
///
/// ### Params
///
/// * `knn_indices` - kNN indices (without self).
/// * `knn_distances` - kNN distances (matching `knn_indices`).
/// * `labels` - Per-cell category labels in `0..n_groups`.
/// * `n_groups` - Number of unique groups (>= 2).
/// * `squared_dist` - Whether `knn_distances` are squared (e.g. squared
///   Euclidean from the kNN search).
/// * `params` - [`MeldParams`].
/// * `seed` - Seed for Lanczos lmax estimation.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Mat<f32>` of shape `(n_cells, n_groups)` containing per-condition density
/// estimates for each cell.
#[allow(clippy::too_many_arguments)]
pub fn meld(
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    labels: &[usize],
    n_groups: usize,
    squared_dist: bool,
    params: &MeldParams,
    seed: u64,
    verbose: usize,
) -> MeldResult {
    let verbosity = parse_verbosity_level(verbose);

    let n = knn_indices.len();
    if labels.len() != n {
        return Err(BixverseErrors::MELDLabelUnequalsSamples);
    }
    if n_groups < 2 {
        return Err(BixverseErrors::MELDOnlyOneGroup);
    }

    let start = Instant::now();
    let knn_k = knn_indices[0].len();

    if verbosity.normal_verbosity() {
        println!(
            "MELD: building kernel adjacency from kNN ({} cells)...",
            n.separate_with_underscores()
        );
    }
    let adj = compute_diffusion_kernel(knn_indices, knn_distances, knn_k, squared_dist)?;
    if verbosity.normal_verbosity() {
        println!(" Done in {:.2?}", start.elapsed())
    }

    if verbosity.normal_verbosity() {
        println!("MELD: building Laplacian...");
    }
    let lap = match params.lap_type {
        LaplacianType::Combinatorial => build_combinatorial_laplacian(&adj),
        LaplacianType::Normalised => build_normalised_laplacian(&adj),
    };
    drop(adj);
    if verbosity.normal_verbosity() {
        println!(" Done in {:.2?}", start.elapsed())
    }

    if verbosity.normal_verbosity() {
        println!("MELD: estimating largest Laplacian eigenvalue...");
    }
    let lmax = estimate_lmax(&lap, seed)?;
    if verbosity.normal_verbosity() {
        println!("MELD: lmax = {:.4}", lmax);
    }
    if verbosity.normal_verbosity() {
        println!(" Done in {:.2?}", start.elapsed())
    }

    let coeffs = chebyshev_coefficients(
        params.filter,
        params.beta,
        params.offset,
        params.order,
        lmax,
        params.chebyshev_order,
    )?;

    let indicators = build_indicator_matrix(labels, n_groups, params.normalise_indicators)?;

    if verbosity.normal_verbosity() {
        println!(
            "MELD: applying Chebyshev filter (order={}) to {} signals...",
            params.chebyshev_order, n_groups
        );
    }
    let densities = chebyshev_apply_columns(&lap, &coeffs, lmax, indicators.as_ref())?;
    if verbosity.normal_verbosity() {
        println!(" Done in {:.2?}", start.elapsed())
    }

    if verbosity.normal_verbosity() {
        println!("MELD: done in {:.2?}", start.elapsed());
    }

    Ok((densities.clone(), l1_normalise(&densities)))
}
