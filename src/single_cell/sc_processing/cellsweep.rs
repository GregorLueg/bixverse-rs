//! CellSweep ambient and bulk contamination removal for single cell data.
//!
//! Implements the multinomial mixture model of the CellSweep reference
//! implementation (Pachter lab). Every observed count in a barcode is split
//! three ways by EM: ambient contamination drawn from a per-emulsion profile
//! `a`, bulk contamination drawn from a global profile `m`, and true expression
//! drawn from the barcode's cell-type profile `p_k`. Subtracting the first two
//! from the observed counts gives the denoised matrix.
//!
//! The model needs cell-type labels on input, so this is not a pre-processing
//! step: it runs after clustering and annotation. It also needs the empty
//! droplets, since `a` is estimated from their pooled counts and they stay in
//! the EM with `alpha` frozen at 1. The ambient profile is a property of one
//! emulsion, so one fit covers one sample.
//!
//! ### Deviations from the reference
//!
//! * Per-row log-likelihood, `A_n` and `alpha` accumulate in `f64`. The
//!   reference keeps them in `f32`, which loses resolution at large N, so the
//!   two implementations agree to a tolerance rather than bit-exactly.
//! * Cell-type profiles are seeded from real cells only. The reference derives
//!   its category list from the non-empty barcodes but then takes the mean over
//!   every barcode carrying the label.
//! * A non-empty barcode with no cell-type label indexes `p[-1]` in the
//!   reference, silently wrapping onto the last cell type. Here barcodes
//!   partition three ways (empty, real-and-annotated, neither) and the third
//!   group is excluded from the fit and from the output.
//!
//! ### References
//!
//! Sullivan et al., CellSweep, 2025

use ann_search_rs::utils::dist::SimdDistance;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::vector_helpers::quantile_sorted;
use crate::prelude::*;
use crate::single_cell::sc_data::data_io::CellGeneSparseWriter;
use crate::single_cell::sc_utils::simd::ln_dot_simd;
use crate::utils::simd::sum_simd_f64;

////////////
// Consts //
////////////

/// Inner iterations of the ambient profile update per M-step.
///
/// The `a = u @ p` fixed point converges fast, and the reference hardcodes
/// three passes. Kept identical so the two implementations track each other.
const AMBIENT_UPDATE_ITERS: usize = 3;

/// Slack added to `alpha_cap` before a barcode counts as exceeding it.
///
/// Guards against a barcode oscillating in and out of the exclusion set purely
/// on floating point noise once `alpha` has settled on the cap.
const ALPHA_CAP_SLACK: f64 = 1e-6;

/// Quantile of the per-cell `|delta f|` used for the convergence check.
const F_CONVERGENCE_QUANTILE: f64 = 0.9;

/// Empty droplets below which the ambient profile cannot be estimated at all.
///
/// The reference drops to `freeze_ambient_profile = false` here rather than
/// erroring. We error instead: silently switching the model out from under the
/// caller buys nothing when the R layer can ask for that branch explicitly.
pub const MIN_EMPTY_DROPLETS: usize = 30;

/// Empty droplets below which a frozen ambient profile is worth a warning.
pub const RECOMMENDED_MIN_EMPTY_DROPLETS: usize = 10_000;

/// Sigma of the Gaussian smoother in the knee detector, in ranks.
const KNEE_SMOOTHING_SIGMA: f64 = 2.0;

/// Half-width of the Gaussian smoothing kernel, in units of sigma.
const KNEE_SMOOTHING_TRUNCATE: f64 = 4.0;

/// Barcodes below this many UMIs are dropped before the knee search.
const KNEE_MIN_COUNTS: u32 = 10;

//////////////////////
// EmptyDropletCall //
//////////////////////

/// How the empty droplets are identified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptyDropletCall {
    /// The caller supplies the mask, e.g. from a CellRanger filtered barcode
    /// list. Nothing is inferred.
    Supplied,
    /// Barcodes with a library size strictly below this are empty.
    UmiCutoff(u32),
    /// Take the library size of the nth largest barcode as the cutoff.
    ExpectedCells(usize),
    /// Infer the cutoff from the curvature of the rank / log-count curve.
    Knee,
}

/// Parse the empty droplet calling strategy from a string.
///
/// ### Params
///
/// * `s` - String, one of `"supplied"`, `"umi_cutoff"`, `"expected_cells"`,
///   `"knee"`.
///
/// ### Returns
///
/// The matching [EmptyDropletCall] with a zero threshold where one is carried,
/// or `None` for an unrecognised value.
pub fn parse_empty_droplet_call(s: &str) -> Option<EmptyDropletCall> {
    match s {
        "supplied" => Some(EmptyDropletCall::Supplied),
        "umi_cutoff" => Some(EmptyDropletCall::UmiCutoff(0)),
        "expected_cells" => Some(EmptyDropletCall::ExpectedCells(0)),
        "knee" => Some(EmptyDropletCall::Knee),
        _ => None,
    }
}

/// Parameters for the CellSweep EM fit.
#[derive(Clone, Copy, Debug)]
pub struct CellSweepParams {
    // -- model structure --
    /// Keep the contamination fraction of empty droplets pinned at 1 rather
    /// than re-estimating it.
    pub freeze_empties: bool,
    /// Keep the ambient profile `a` at its empty-droplet estimate rather than
    /// re-estimating it as a mixture over cell-type profiles. When `false`,
    /// `alpha_cap`, the repulsion terms and cell-type reassignment are all
    /// disabled.
    pub freeze_ambient_profile: bool,

    // -- initialisation --
    /// Starting ambient fraction for every real barcode.
    pub init_alpha: f64,
    /// Starting bulk contamination fraction. Initialised below `init_alpha` to
    /// bias unassignable contamination towards ambient, which is the more
    /// identifiable of the two.
    pub init_beta: f64,

    // -- stage one schedule --
    /// Ceiling on `alpha` before the log-likelihood has converged. Barcodes
    /// wanting to exceed it are excluded from the `p_k` update and allowed to
    /// switch cell type.
    pub alpha_cap: f64,
    /// Strength of the repulsion pushing cell-type profiles away from the
    /// ambient profile.
    pub repulsion_strength: f32,
    /// Ceiling on the fraction of any single `p_k` entry that repulsion may
    /// remove.
    pub max_frac_gene_repulsion: f32,

    // -- pseudocounts --
    /// Pseudocount smoothing the cell-type profile update.
    pub celltype_lambda: f32,
    /// Pseudocount smoothing the ambient profile estimate.
    pub ambient_lambda: f32,
    /// Pseudocount smoothing the bulk profile estimate.
    pub bulk_lambda: f32,

    // -- numerics --
    /// Floor on denominators.
    pub eps: f64,
    /// Floor on the argument of `ln`.
    pub log_eps: f64,

    // -- stopping --
    /// Hard cap on EM iterations.
    pub max_iter: usize,
    /// Log-likelihood change, as a fraction of the first EM step's change,
    /// below which stage one ends and parameter convergence is checked.
    pub del0_ll_tol: f64,
    /// Floor on the adaptive tolerance, relative to the current
    /// log-likelihood. Stops `del0_ll_tol` chasing floating point noise.
    pub min_ll_tol: f64,
    /// Convergence threshold on the maximum row-wise L1 change in `p`.
    pub tol_p: f32,
    /// Convergence threshold on the change in the total contamination fraction
    /// `f = (1 - beta) * alpha + beta`.
    pub tol_f: f64,

    // -- output --
    /// Derive the normalised layer from the integerised counts rather than the
    /// denoised floats. Consistent across the two layers at the cost of the
    /// sub-integer signal, which is where CellSweep is most informative.
    pub norm_from_rounded: bool,
    /// Seed for the stochastic rounding of the denoised counts.
    pub seed: u64,
}

/// Default implementation for [`CellSweepParams`].
impl Default for CellSweepParams {
    fn default() -> Self {
        Self {
            freeze_empties: true,
            freeze_ambient_profile: true,
            init_alpha: 0.9,
            init_beta: 0.1,
            alpha_cap: 0.9,
            repulsion_strength: 1e-4,
            max_frac_gene_repulsion: 0.2,
            celltype_lambda: 50.0,
            ambient_lambda: 50.0,
            bulk_lambda: 10.0,
            eps: 1e-12,
            log_eps: 1e-300,
            max_iter: 2000,
            del0_ll_tol: 1e-3,
            min_ll_tol: 1e-6,
            tol_p: 1e-4,
            tol_f: 1e-4,
            norm_from_rounded: false,
            seed: 42,
        }
    }
}

impl CellSweepParams {
    /// Create a new parameter set.
    ///
    /// Every field is taken as given; use [CellSweepParams::default] for the
    /// reference implementation's defaults and override from there.
    ///
    /// ### Returns
    ///
    /// The populated [CellSweepParams].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        freeze_empties: bool,
        freeze_ambient_profile: bool,
        init_alpha: f64,
        init_beta: f64,
        alpha_cap: f64,
        repulsion_strength: f32,
        max_frac_gene_repulsion: f32,
        celltype_lambda: f32,
        ambient_lambda: f32,
        bulk_lambda: f32,
        eps: f64,
        log_eps: f64,
        max_iter: usize,
        del0_ll_tol: f64,
        min_ll_tol: f64,
        tol_p: f32,
        tol_f: f64,
        norm_from_rounded: bool,
        seed: u64,
    ) -> Self {
        Self {
            freeze_empties,
            freeze_ambient_profile,
            init_alpha,
            init_beta,
            alpha_cap,
            repulsion_strength,
            max_frac_gene_repulsion,
            celltype_lambda,
            ambient_lambda,
            bulk_lambda,
            eps,
            log_eps,
            max_iter,
            del0_ll_tol,
            min_ll_tol,
            tol_p,
            tol_f,
            norm_from_rounded,
            seed,
        }
    }
}

/// One sample's barcodes and labels.
///
/// Indices are positions in the underlying store, so they survive unchanged
/// into the parent object's index space.
#[derive(Clone, Debug)]
pub struct CellSweepSample {
    /// Identifier of the sample, used in progress output and errors.
    pub sample_id: String,
    /// Store indices of the annotated barcodes that passed QC.
    pub real_cells: Vec<usize>,
    /// Store indices of the empty droplets.
    pub empty_cells: Vec<usize>,
    /// Cell-type code in `0..n_celltypes` for each entry of `real_cells`.
    pub celltype_idx: Vec<usize>,
    /// Number of distinct cell types. Codes index into this range.
    pub n_celltypes: usize,
}

/// Fitted parameters and diagnostics for one sample.
#[derive(Clone, Debug)]
pub struct CellSweepFit {
    /// Ambient fraction per real barcode, in `real_cells` order.
    pub alpha: Vec<f64>,
    /// Cell-type assignment per real barcode after any reassignment, in
    /// `real_cells` order.
    pub z_hat: Vec<usize>,
    /// Fitted bulk contamination fraction.
    pub beta: f64,
    /// Fitted ambient profile over genes, summing to 1.
    pub ambient: Vec<f32>,
    /// Fitted cell-type profiles, row-major `n_celltypes x n_genes`, each row
    /// summing to 1.
    pub celltype_profiles: Vec<f32>,
    /// Mean per-barcode log-likelihood at the last iteration. Relative, not a
    /// complete multinomial log-likelihood.
    pub log_likelihood: f64,
    /// EM iterations run.
    pub n_iter: usize,
    /// Whether both parameter convergence criteria were met.
    pub converged: bool,
}

///////////////
// SampleCsr //
///////////////

/// A sample's counts as a flat CSR block, plus the model's view of each row.
///
/// Held in memory for the whole EM: the E-step touches every non-zero on every
/// iteration, so streaming it from disk would mean thousands of full passes
/// over the store. At `f32` data and `u32` indices this is 8 bytes per
/// non-zero, which is what bounds a single fit and why fits are per-sample.
struct SampleCsr {
    /// Non-zero counts, row-major.
    data: Vec<f32>,
    /// Gene index of each non-zero.
    indices: Vec<u32>,
    /// Row offsets into `data` and `indices`, length `n_rows + 1`.
    indptr: Vec<usize>,
    /// Number of genes.
    n_genes: usize,
    /// Rows `0..n_real` are the real barcodes; the rest are empty droplets.
    n_real: usize,
    /// Cell-type code per row, `usize::MAX` for the empty droplets.
    gamma_idx: Vec<usize>,
    /// Number of cell types.
    n_celltypes: usize,
}

impl SampleCsr {
    /// Total number of rows, real barcodes followed by empty droplets.
    ///
    /// ### Returns
    ///
    /// Number of rows
    fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Non-zero slice of one row.
    ///
    /// ### Returns
    ///
    /// `(index, value)` as slices
    #[inline]
    fn row(&self, i: usize) -> (&[u32], &[f32]) {
        let (rs, re) = (self.indptr[i], self.indptr[i + 1]);
        (&self.indices[rs..re], &self.data[rs..re])
    }

    /// Whether the row is an empty droplet.
    ///
    /// ### Return
    ///
    /// `true` if empty droplet.
    #[inline]
    fn is_empty_droplet(&self, i: usize) -> bool {
        i >= self.n_real
    }
}

///////////////////////////
// Empty droplet calling //
///////////////////////////

/////////////
// Helpers //
/////////////

/// Reflect an out-of-range index back into `0..n`, half-sample symmetric.
///
/// ### Params
///
/// * `idx` - Possibly out-of-range index.
/// * `n` - Length of the signal. Must be non-zero.
///
/// ### Returns
///
/// The reflected in-range index.
#[inline]
fn reflect_index(idx: isize, n: usize) -> usize {
    let n_i = n as isize;
    let mut i = idx;
    // Loops rather than a closed form because a kernel wider than the signal
    // can reflect more than once.
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n_i {
            i = 2 * n_i - i - 1;
        } else {
            return i as usize;
        }
    }
}

/// Reflect-padded Gaussian smoothing of a 1D signal.
///
/// Matches `scipy.ndimage.gaussian_filter1d` at its default `mode="reflect"`
/// and `truncate=4.0`, which the knee detector depends on to land on the same
/// rank as the reference.
///
/// ### Params
///
/// * `y` - Signal to smooth.
/// * `sigma` - Standard deviation of the kernel, in samples.
///
/// ### Returns
///
/// The smoothed signal, same length as `y`.
fn gaussian_smooth_1d(y: &[f64], sigma: f64) -> Vec<f64> {
    let radius = (KNEE_SMOOTHING_TRUNCATE * sigma + 0.5) as usize;

    let mut kernel: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-0.5 * x * x / (sigma * sigma)).exp()
        })
        .collect();
    let kernel_sum: f64 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= kernel_sum);

    let n = y.len();

    (0..n)
        .map(|i| {
            kernel
                .iter()
                .enumerate()
                .map(|(k, &w)| {
                    let offset = k as isize - radius as isize;
                    w * y[reflect_index(i as isize + offset, n)]
                })
                .sum()
        })
        .collect()
}

/// Central-difference gradient with one-sided ends.
///
/// Matches `np.gradient` on a unit-spaced signal.
///
/// ### Params
///
/// * `y` - Signal to differentiate. Must have at least two elements.
///
/// ### Returns
///
/// The gradient, same length as `y`.
fn central_gradient(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n];

    out[0] = y[1] - y[0];
    out[n - 1] = y[n - 1] - y[n - 2];
    for i in 1..n - 1 {
        out[i] = 0.5 * (y[i + 1] - y[i - 1]);
    }

    out
}

/// Library size of the nth largest barcode.
///
/// ### Params
///
/// * `library_sizes` - Library size per barcode.
/// * `expected` - Number of real cells expected in the run.
///
/// ### Returns
///
/// The cutoff, i.e. the library size at rank `expected`.
fn cutoff_for_expected_cells(
    library_sizes: &[u32],
    expected: usize,
) -> Result<u32, BixverseErrors> {
    if expected == 0 || expected > library_sizes.len() {
        return Err(BixverseErrors::CellSweepBadExpectedCells {
            expected,
            barcodes: library_sizes.len(),
        });
    }

    let mut sorted = library_sizes.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));

    Ok(sorted[expected - 1])
}

/// Locate the knee of the rank / log-count curve.
///
/// Sorts library sizes descending, drops the tail below [KNEE_MIN_COUNTS],
/// smooths `log10(counts)` against rank with a Gaussian kernel, and takes the
/// rank of most negative second derivative. Experimental, and flagged as such
/// in the reference too: prefer a supplied mask or an explicit cutoff.
///
/// ### Params
///
/// * `library_sizes` - Library size per barcode.
///
/// ### Returns
///
/// The library size at the knee, used as the empty droplet cutoff.
fn knee_umi_cutoff(library_sizes: &[u32]) -> Result<u32, BixverseErrors> {
    let mut counts: Vec<u32> = library_sizes
        .iter()
        .copied()
        .filter(|&c| c > KNEE_MIN_COUNTS)
        .collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));

    // Three points is the minimum a central-difference second derivative can
    // be taken over.
    if counts.len() < 3 {
        return Err(BixverseErrors::CellSweepKneeNotFound {
            barcodes: counts.len(),
        });
    }

    let log_counts: Vec<f64> = counts.iter().map(|&c| (c as f64).log10()).collect();
    let smoothed = gaussian_smooth_1d(&log_counts, KNEE_SMOOTHING_SIGMA);

    // Ranks are 1..=n and unit-spaced, so `np.gradient(y, x)` reduces to
    // `np.gradient(y)`.
    let d1 = central_gradient(&smoothed);
    let d2 = central_gradient(&d1);

    let knee_idx = d2
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .ok_or(BixverseErrors::CellSweepKneeNotFound {
            barcodes: counts.len(),
        })?;

    Ok(counts[knee_idx])
}

//////////
// Main //
//////////

/// Turn a library size vector into an empty droplet mask.
///
/// ### Params
///
/// * `library_sizes` - Library size per barcode, in store order.
/// * `call` - Which strategy to apply. [EmptyDropletCall::Supplied] is a
///   caller error here, since there is nothing to infer.
///
/// ### Returns
///
/// A mask that is `true` where the barcode is an empty droplet.
pub fn infer_empty_droplets(
    library_sizes: &[u32],
    call: EmptyDropletCall,
) -> Result<Vec<bool>, BixverseErrors> {
    let cutoff = match call {
        EmptyDropletCall::Supplied => {
            return Err(BixverseErrors::CellSweepEmptyMaskMissing);
        }
        EmptyDropletCall::UmiCutoff(cutoff) => cutoff,
        EmptyDropletCall::ExpectedCells(expected) => {
            cutoff_for_expected_cells(library_sizes, expected)?
        }
        EmptyDropletCall::Knee => knee_umi_cutoff(library_sizes)?,
    };

    Ok(library_sizes.iter().map(|&s| s < cutoff).collect())
}

////////////
// E step //
////////////

/// Per-barcode accumulators produced by one E-step.
///
/// Packed into one structure so a parallel sweep can hand out disjoint `&mut`
/// slices of a single vector rather than of four.
#[derive(Clone, Copy, Debug, Default)]
struct RowStats {
    /// Expected true-expression counts in this barcode.
    gamma: f64,
    /// Expected ambient counts in this barcode.
    ambient: f64,
    /// Expected bulk counts in this barcode.
    bulk: f64,
    /// Contribution of this barcode to the log-likelihood.
    log_likelihood: f64,
}

/// Non-zeros per pass of the tiled row kernels.
///
/// The row loop stays fused; the tile exists only to bound the two buffers the
/// vectorised `ln` reads, and to keep them in L1 between the pass that fills
/// them and the pass that consumes them. Rows shorter than this run as one
/// short tile.
const E_STEP_TILE: usize = 256;

/// Per-thread buffers for the vectorised `ln`.
///
/// One allocation per rayon chunk rather than per row. Rows are short and
/// numerous, so a per-row allocation would cost more than the vectorisation
/// saves.
struct RowScratch {
    /// Observed counts of the current tile, widened to `f64`.
    counts: Vec<f64>,
    /// Total mixture probability per non-zero of the current tile.
    p_tot: Vec<f64>,
    /// Ambient plus bulk per non-zero. The part of the mixture that does not
    /// depend on the cell type, so the reassignment builds it once per tile
    /// rather than once per tile and cell type. Only filled for an excluded
    /// barcode.
    base: Vec<f64>,
    /// `base` plus the candidate cell type's weighted profile.
    mix: Vec<f64>,
    /// Log-likelihood of this barcode under each cell type, accumulated across
    /// tiles. Only used by an excluded barcode.
    ll_k: Vec<f64>,
}

impl RowScratch {
    /// Allocate the buffers for one thread.
    ///
    /// ### Params
    ///
    /// * `n_celltypes` - Number of cell types, the length of `ll_k`.
    ///
    /// ### Returns
    ///
    /// Zeroed [RowScratch].
    fn new(n_celltypes: usize) -> Self {
        Self {
            counts: vec![0.0; E_STEP_TILE],
            p_tot: vec![0.0; E_STEP_TILE],
            base: vec![0.0; E_STEP_TILE],
            mix: vec![0.0; E_STEP_TILE],
            ll_k: vec![0.0; n_celltypes],
        }
    }
}

/// Immutable model state the E-step reads.
struct EStepState<'a> {
    /// Ambient fraction per barcode.
    alpha: &'a [f64],
    /// Bulk contamination fraction.
    beta: f64,
    /// Ambient profile over genes.
    ambient: &'a [f32],
    /// Bulk profile over genes.
    bulk: &'a [f32],
    /// Cell-type profiles, row-major `n_celltypes x n_genes`.
    profiles: &'a [f32],
    /// Cell-type code per barcode.
    gamma_idx: &'a [usize],
    /// Barcodes excluded from the `p_k` update and free to switch cell type.
    excluded: &'a [bool],
    /// Whether the ambient profile is being re-estimated.
    update_ambient: bool,
}

/// Everything the row kernels read that is the same for every row.
///
/// Bundled rather than passed one by one: the kernels take the model state, the
/// counts, the tolerances and a precomputed profile, and threading four
/// references through two call sites buys nothing over one.
struct RowCtx<'a> {
    /// The sample's counts.
    csr: &'a SampleCsr,
    /// Current model state.
    state: &'a EStepState<'a>,
    /// Numerical tolerances.
    params: &'a CellSweepParams,
    /// `beta * bulk[g]` per gene. `beta` is fixed across an E-step, so this
    /// lifts a multiply out of the per-non-zero path.
    beta_bulk: &'a [f64],
}

/// Sufficient statistics reduced out of one E-step.
///
/// Accumulated in `f64` rather than the reference's `f32`: a single `p_numer`
/// entry sums an expected count over every barcode of its cell type, so at 1e5
/// barcodes the small contributions vanish in `f32`.
struct EStepTotals {
    /// Expected true-expression counts per cell type and gene, row-major.
    p_numer: Vec<f64>,
    /// Expected ambient counts per gene. Empty when the ambient profile is
    /// frozen, since nothing reads it.
    a_numer: Vec<f64>,
}

impl EStepTotals {
    /// Allocate zeroed accumulators.
    ///
    /// ### Params
    ///
    /// * `n_celltypes` - Number of cell types.
    /// * `n_genes` - Number of genes.
    /// * `update_ambient` - Whether the ambient numerator is needed at all.
    ///
    /// ### Returns
    ///
    /// Zeroed [EStepTotals].
    fn zeros(n_celltypes: usize, n_genes: usize, update_ambient: bool) -> Self {
        Self {
            p_numer: vec![0.0; n_celltypes * n_genes],
            a_numer: vec![0.0; if update_ambient { n_genes } else { 0 }],
        }
    }

    /// Fold another chunk's accumulators into this one.
    ///
    /// ### Params
    ///
    /// * `other` - Accumulators to absorb.
    fn merge(&mut self, other: Self) {
        self.p_numer
            .iter_mut()
            .zip(other.p_numer)
            .for_each(|(a, b)| *a += b);
        self.a_numer
            .iter_mut()
            .zip(other.a_numer)
            .for_each(|(a, b)| *a += b);
    }
}

/// Run one E-step over every barcode of a sample.
///
/// Rows are independent, so the sweep fans out over row chunks sized to one
/// chunk per thread. That bounds the `n_celltypes x n_genes` accumulator to one
/// allocation per thread, the same footprint as the reference's preallocated
/// thread-local block but without pinning it to a thread count.
///
/// ### Params
///
/// * `csr` - The sample's counts.
/// * `state` - Current model state.
/// * `row_stats` - Per-barcode output, overwritten. Length `csr.n_rows()`.
/// * `gamma_idx_out` - Cell-type codes after reassignment, overwritten. Length
///   `csr.n_rows()`.
/// * `params` - Numerical tolerances.
///
/// ### Returns
///
/// The reduced sufficient statistics.
fn e_step(
    csr: &SampleCsr,
    state: &EStepState<'_>,
    row_stats: &mut [RowStats],
    gamma_idx_out: &mut [usize],
    params: &CellSweepParams,
) -> EStepTotals {
    let n_genes = csr.n_genes;
    let n_celltypes = csr.n_celltypes;
    let n_rows = csr.n_rows();

    let n_threads = rayon::current_num_threads();
    let chunk_size = n_rows.div_ceil(n_threads.max(1)).max(1);

    gamma_idx_out.copy_from_slice(state.gamma_idx);

    let beta_bulk: Vec<f64> = state.bulk.iter().map(|&m| state.beta * m as f64).collect();
    let ctx = RowCtx {
        csr,
        state,
        params,
        beta_bulk: &beta_bulk,
    };

    row_stats
        .par_chunks_mut(chunk_size)
        .zip(gamma_idx_out.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(chunk_idx, (stats, gammas))| {
            let mut totals = EStepTotals::zeros(n_celltypes, n_genes, state.update_ambient);
            let mut scratch = RowScratch::new(n_celltypes);
            let row_offset = chunk_idx * chunk_size;

            for (local, stat) in stats.iter_mut().enumerate() {
                let n = row_offset + local;
                let (indices, values) = csr.row(n);
                *stat = RowStats::default();

                if indices.is_empty() {
                    continue;
                }

                if csr.is_empty_droplet(n) {
                    e_step_empty_row(indices, values, n, &ctx, &mut scratch, &mut totals, stat);
                } else {
                    e_step_cell_row(
                        indices,
                        values,
                        n,
                        &ctx,
                        &mut scratch,
                        &mut totals,
                        stat,
                        &mut gammas[local],
                    );
                }
            }

            totals
        })
        .reduce(
            || EStepTotals::zeros(n_celltypes, n_genes, state.update_ambient),
            |mut acc, other| {
                acc.merge(other);
                acc
            },
        )
}

/// E-step for one empty droplet.
///
/// An empty droplet has no cell, so only the ambient and bulk components apply.
/// Same shape as [e_step_cell_row], minus the cell component and the `p_numer`
/// scatter.
///
/// ### Params
///
/// * `indices` - Gene indices of the barcode's non-zeros.
/// * `values` - Counts of the barcode's non-zeros.
/// * `n` - Row index, used to look up `alpha`.
/// * `ctx` - Model state and tolerances.
/// * `scratch` - This thread's buffers.
/// * `totals` - Accumulators to add into.
/// * `stat` - Per-barcode output for this row.
fn e_step_empty_row(
    indices: &[u32],
    values: &[f32],
    n: usize,
    ctx: &RowCtx<'_>,
    scratch: &mut RowScratch,
    totals: &mut EStepTotals,
    stat: &mut RowStats,
) {
    let (state, params) = (ctx.state, ctx.params);
    let w_ambient = (1.0 - state.beta) * state.alpha[n];

    for (idx, val) in indices.chunks(E_STEP_TILE).zip(values.chunks(E_STEP_TILE)) {
        for (j, (&gene, &value)) in idx.iter().zip(val).enumerate() {
            let g = gene as usize;
            let value = value as f64;

            let wa = w_ambient * state.ambient[g] as f64;
            let wm = ctx.beta_bulk[g];
            let p_tot = wa + wm;
            let scale = value / p_tot.max(params.eps);

            let c_ambient = scale * wa;

            stat.ambient += c_ambient;
            stat.bulk += scale * wm;

            scratch.counts[j] = value;
            scratch.p_tot[j] = p_tot;

            if state.update_ambient {
                totals.a_numer[g] += c_ambient;
            }
        }

        let len = idx.len();
        stat.log_likelihood += ln_dot_simd(
            &scratch.counts[..len],
            &scratch.p_tot[..len],
            params.log_eps,
        );
    }
}

/// E-step for one cell-containing barcode.
///
/// Adds the cell-type component on top of ambient and bulk. Barcodes excluded
/// from the `p_k` update get a hard cell-type reassignment on the full
/// likelihood, which is what lets a barcode whose `alpha` wants to run away
/// find a better-fitting profile instead of dragging its own profile towards
/// the ambient one.
///
/// The loop stays fused and scalar, because that is where the gathers and the
/// scattered `p_numer` write live and neither vectorises. The one thing lifted
/// out of it is the `ln`: `p_tot` is stashed per tile and the log-likelihood
/// taken in a second, vectorised pass. Worth 1.27x on this path. Note that the
/// `ln` is a smaller share of a row than a micro-benchmark of it suggests,
/// since the divide, the accumulations and the scatters give the out-of-order
/// engine plenty to hide a libm call behind.
///
/// ### Params
///
/// * `indices` - Gene indices of the barcode's non-zeros.
/// * `values` - Counts of the barcode's non-zeros.
/// * `n` - Row index, used to look up `alpha` and the cell-type code.
/// * `ctx` - Model state and tolerances.
/// * `scratch` - This thread's buffers.
/// * `totals` - Accumulators to add into.
/// * `stat` - Per-barcode output for this row.
/// * `gamma_out` - Cell-type code for this row, overwritten on reassignment.
///   Only touched for an excluded barcode, which is also the only case where
///   the reference reassigns.
#[allow(clippy::too_many_arguments)]
fn e_step_cell_row(
    indices: &[u32],
    values: &[f32],
    n: usize,
    ctx: &RowCtx<'_>,
    scratch: &mut RowScratch,
    totals: &mut EStepTotals,
    stat: &mut RowStats,
    gamma_out: &mut usize,
) {
    let (csr, state, params) = (ctx.csr, ctx.state, ctx.params);
    let n_genes = csr.n_genes;
    let alpha_n = state.alpha[n];
    let w_ambient = (1.0 - state.beta) * alpha_n;
    let w_cell = (1.0 - state.beta) * (1.0 - alpha_n);
    let k = state.gamma_idx[n];
    let allow_p_update = !state.excluded[n];
    let profile = &state.profiles[k * n_genes..(k + 1) * n_genes];

    if !allow_p_update {
        scratch.ll_k.iter_mut().for_each(|ll| *ll = 0.0);
    }

    for (idx, val) in indices.chunks(E_STEP_TILE).zip(values.chunks(E_STEP_TILE)) {
        for (j, (&gene, &value)) in idx.iter().zip(val).enumerate() {
            let g = gene as usize;
            let value = value as f64;

            let wa = w_ambient * state.ambient[g] as f64;
            let wm = ctx.beta_bulk[g];
            let wc = w_cell * profile[g] as f64;
            let p_tot = wa + wm + wc;
            let scale = value / p_tot.max(params.eps);

            let c_ambient = scale * wa;
            let c_cell = scale * wc;

            stat.ambient += c_ambient;
            stat.bulk += scale * wm;
            stat.gamma += c_cell;

            scratch.counts[j] = value;
            scratch.p_tot[j] = p_tot;

            if allow_p_update {
                totals.p_numer[k * n_genes + g] += c_cell;
            } else {
                scratch.base[j] = wa + wm;
            }
            if state.update_ambient {
                totals.a_numer[g] += c_ambient;
            }
        }

        let len = idx.len();
        stat.log_likelihood += ln_dot_simd(
            &scratch.counts[..len],
            &scratch.p_tot[..len],
            params.log_eps,
        );

        if !allow_p_update {
            accumulate_celltype_ll(idx, w_cell, ctx, scratch);
        }
    }

    if !allow_p_update {
        *gamma_out = best_celltype(&scratch.ll_k, k);
    }
}

/// Add one tile's contribution to the per-cell-type log-likelihood.
///
/// The ambient and bulk halves of the mixture do not depend on the cell type,
/// and the pass above has already weighted them into `base`, so this rebuilds
/// only what changes. What is left per cell type is one gather, one
/// multiply-add and the `ln`, which is the one loop in CellSweep that really is
/// `ln`-bound: there is nothing else in it for the out-of-order engine to hide
/// the libm call behind.
///
/// ### Params
///
/// * `idx` - Gene indices of this tile's non-zeros.
/// * `w_cell` - Weight on the cell-type profile, `(1 - beta) * (1 - alpha_n)`.
/// * `ctx` - Model state and tolerances.
/// * `scratch` - This thread's buffers. `ll_k` is added into.
fn accumulate_celltype_ll(idx: &[u32], w_cell: f64, ctx: &RowCtx<'_>, scratch: &mut RowScratch) {
    let n_genes = ctx.csr.n_genes;
    let len = idx.len();

    for k in 0..ctx.csr.n_celltypes {
        let profile = &ctx.state.profiles[k * n_genes..(k + 1) * n_genes];
        for (j, &gene) in idx.iter().enumerate() {
            scratch.mix[j] = scratch.base[j] + w_cell * profile[gene as usize] as f64;
        }
        scratch.ll_k[k] += ln_dot_simd(
            &scratch.counts[..len],
            &scratch.mix[..len],
            ctx.params.log_eps,
        );
    }
}

/// Pick the cell type maximising this barcode's full mixture likelihood.
///
/// ### Params
///
/// * `ll_k` - Log-likelihood of the barcode under each cell type.
/// * `current_k` - Cell-type code held going in, kept when `ll_k` is empty.
///
/// ### Returns
///
/// The best-fitting cell-type code, ties to the lowest code.
fn best_celltype(ll_k: &[f64], current_k: usize) -> usize {
    let mut best_k = current_k;
    let mut best_ll = f64::NEG_INFINITY;

    for (k, &ll) in ll_k.iter().enumerate() {
        if ll > best_ll {
            best_ll = ll;
            best_k = k;
        }
    }

    best_k
}

/////////////
// EmState //
/////////////

/// Mutable model state carried through the EM.
struct EmState {
    /// Ambient fraction per barcode.
    alpha: Vec<f64>,
    /// Bulk contamination fraction.
    beta: f64,
    /// Ambient profile over genes, sums to 1.
    ambient: Vec<f32>,
    /// Bulk profile over genes, sums to 1. Fixed after initialisation.
    bulk: Vec<f32>,
    /// Cell-type mixing weights of the ambient profile. Only used when the
    /// ambient profile is re-estimated.
    u: Vec<f64>,
    /// Cell-type profiles, row-major `n_celltypes x n_genes`, rows sum to 1.
    profiles: Vec<f32>,
    /// Cell-type code per barcode, `usize::MAX` for empty droplets.
    gamma_idx: Vec<usize>,
    /// Barcodes excluded from the `p_k` update this iteration.
    excluded: Vec<bool>,
}

/// Seed the model state from the raw counts.
///
/// Cell-type profiles start at the smoothed mean expression of each type over
/// the real barcodes, the bulk profile at the smoothed column sums over every
/// barcode, and the ambient profile either at the smoothed column sums of the
/// empty droplets (frozen case) or as the cell-type mixture implied by the
/// label frequencies.
///
/// ### Params
///
/// * `csr` - The sample's counts.
/// * `params` - Model parameters. The pseudocounts are applied here, already
///   divided by the gene count.
///
/// ### Returns
///
/// The initialised [EmState].
fn init_em(csr: &SampleCsr, params: &CellSweepParams) -> EmState {
    let (n_genes, n_celltypes) = (csr.n_genes, csr.n_celltypes);
    let inv_genes = 1.0 / n_genes as f64;

    // bulk profile: column sums over every barcode, including the empties
    let mut bulk_raw = column_sums(csr, 0..csr.n_rows());
    bulk_raw
        .iter_mut()
        .for_each(|v| *v += params.bulk_lambda as f64 * inv_genes);
    let bulk = normalise_to_f32(&bulk_raw);

    // cell-type profiles: mean expression per type over the real barcodes
    let celltype_lambda = params.celltype_lambda as f64 * inv_genes;
    let mut profiles = vec![0.0_f32; n_celltypes * n_genes];
    let means = celltype_mean_expression(csr);
    for k in 0..n_celltypes {
        let row = &means[k * n_genes..(k + 1) * n_genes];
        let denom = sum_simd_f64(row) + n_genes as f64 * celltype_lambda;
        for (g, &value) in row.iter().enumerate() {
            profiles[k * n_genes + g] = ((value + celltype_lambda) / denom) as f32;
        }
    }

    // ambient profile
    let mut u = vec![0.0_f64; n_celltypes];
    let ambient = if params.freeze_ambient_profile {
        let mut ambient_raw = column_sums(csr, csr.n_real..csr.n_rows());
        ambient_raw
            .iter_mut()
            .for_each(|v| *v += params.ambient_lambda as f64 * inv_genes);
        normalise_to_f32(&ambient_raw)
    } else {
        let mut counts = vec![0.0_f64; n_celltypes];
        csr.gamma_idx[..csr.n_real]
            .iter()
            .for_each(|&k| counts[k] += 1.0);
        let total = counts.iter().sum::<f64>().max(params.eps);
        counts
            .iter()
            .enumerate()
            .for_each(|(k, &c)| u[k] = c / total);
        normalise_to_f32(&mixture_profile(&u, &profiles, n_genes, n_celltypes))
    };

    // alpha: the reference clips to (eps, 1 - eps) and then pins the empties
    let mut alpha = vec![params.init_alpha.clamp(params.eps, 1.0 - params.eps); csr.n_rows()];
    alpha[csr.n_real..].iter_mut().for_each(|a| *a = 1.0);

    EmState {
        alpha,
        beta: params.init_beta,
        ambient,
        bulk,
        u,
        profiles,
        gamma_idx: csr.gamma_idx.clone(),
        excluded: vec![false; csr.n_rows()],
    }
}

/// Column sums over a contiguous range of rows.
///
/// ### Params
///
/// * `csr` - The sample's counts.
/// * `rows` - Row range to sum over.
///
/// ### Returns
///
/// Per-gene sums, length `csr.n_genes`.
fn column_sums(csr: &SampleCsr, rows: std::ops::Range<usize>) -> Vec<f64> {
    let n_genes = csr.n_genes;
    let n_threads = rayon::current_num_threads();
    let chunk_size = rows.len().div_ceil(n_threads.max(1)).max(1);

    rows.collect::<Vec<usize>>()
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut acc = vec![0.0_f64; n_genes];
            for &n in chunk {
                let (indices, values) = csr.row(n);
                for (&gene, &value) in indices.iter().zip(values) {
                    acc[gene as usize] += value as f64;
                }
            }
            acc
        })
        .reduce(
            || vec![0.0_f64; n_genes],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        )
}

/// Mean expression of every gene within each cell type.
///
/// Restricted to the real barcodes. The reference derives its category list
/// from the non-empty barcodes but then takes the mean over every barcode
/// carrying the label, which lets an empty droplet contribute.
///
/// ### Params
///
/// * `csr` - The sample's counts.
///
/// ### Returns
///
/// Row-major `n_celltypes x n_genes` means.
fn celltype_mean_expression(csr: &SampleCsr) -> Vec<f64> {
    let (n_genes, n_celltypes) = (csr.n_genes, csr.n_celltypes);
    let n_threads = rayon::current_num_threads();
    let chunk_size = csr.n_real.div_ceil(n_threads.max(1)).max(1);

    let (mut sums, counts) = (0..csr.n_real)
        .collect::<Vec<usize>>()
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut sums = vec![0.0_f64; n_celltypes * n_genes];
            let mut counts = vec![0.0_f64; n_celltypes];
            for &n in chunk {
                let k = csr.gamma_idx[n];
                counts[k] += 1.0;
                let (indices, values) = csr.row(n);
                for (&gene, &value) in indices.iter().zip(values) {
                    sums[k * n_genes + gene as usize] += value as f64;
                }
            }
            (sums, counts)
        })
        .reduce(
            || {
                (
                    vec![0.0_f64; n_celltypes * n_genes],
                    vec![0.0_f64; n_celltypes],
                )
            },
            |mut a, b| {
                a.0.iter_mut().zip(b.0).for_each(|(x, y)| *x += y);
                a.1.iter_mut().zip(b.1).for_each(|(x, y)| *x += y);
                a
            },
        );

    for k in 0..n_celltypes {
        let n_cells = counts[k].max(1.0);
        sums[k * n_genes..(k + 1) * n_genes]
            .iter_mut()
            .for_each(|v| *v /= n_cells);
    }

    sums
}

/// Mix the cell-type profiles into a single gene profile.
///
/// ### Params
///
/// * `weights` - Mixing weight per cell type.
/// * `profiles` - Cell-type profiles, row-major.
/// * `n_genes` - Number of genes, the stride of `profiles`.
/// * `n_celltypes` - Number of cell types.
///
/// ### Returns
///
/// The mixed profile, length `n_genes`. Not normalised.
fn mixture_profile(
    weights: &[f64],
    profiles: &[f32],
    n_genes: usize,
    n_celltypes: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_genes];
    for k in 0..n_celltypes {
        let w = weights[k];
        let profile = &profiles[k * n_genes..(k + 1) * n_genes];
        out.iter_mut()
            .zip(profile)
            .for_each(|(o, &p)| *o += w * p as f64);
    }
    out
}

/// Scale a non-negative vector to sum to one and narrow it to `f32`.
///
/// ### Params
///
/// * `v` - Values to normalise. A zero total returns a uniform profile, which
///   is the only sensible answer for a gene set that saw no counts at all.
///
/// ### Returns
///
/// The normalised profile.
fn normalise_to_f32(v: &[f64]) -> Vec<f32> {
    let total = sum_simd_f64(v);
    if total <= 0.0 {
        return vec![1.0 / v.len() as f32; v.len()];
    }
    v.iter().map(|&x| (x / total) as f32).collect()
}

////////////
// M step //
////////////

/// Update every parameter from one E-step's sufficient statistics.
///
/// Stage one, which runs until the log-likelihood has converged and only when
/// the ambient profile is frozen, additionally caps `alpha`, excludes the
/// barcodes that wanted to exceed the cap from the `p_k` update, and repels the
/// cell-type profiles away from the ambient profile. Stage two drops all three.
///
/// ### Params
///
/// * `csr` - The sample's counts.
/// * `state` - Model state, updated in place.
/// * `row_stats` - Per-barcode statistics from the E-step.
/// * `totals` - Reduced sufficient statistics from the E-step.
/// * `stage_one` - Whether the stage one schedule is still active.
/// * `params` - Model parameters.
///
/// ### Returns
///
/// The total contamination fraction `f = (1 - beta) * alpha + beta` per real
/// barcode, which the convergence check consumes.
fn m_step(
    csr: &SampleCsr,
    state: &mut EmState,
    row_stats: &[RowStats],
    totals: &EStepTotals,
    stage_one: bool,
    params: &CellSweepParams,
) -> Vec<f64> {
    let (n_genes, n_celltypes) = (csr.n_genes, csr.n_celltypes);

    // -- alpha --

    state
        .alpha
        .iter_mut()
        .zip(row_stats)
        .for_each(|(alpha, stat)| {
            *alpha = stat.ambient / (stat.ambient + stat.gamma).max(params.eps);
        });
    state.alpha[csr.n_real..].iter_mut().for_each(|a| *a = 1.0);

    if stage_one {
        for n in 0..csr.n_rows() {
            state.excluded[n] =
                state.alpha[n] > params.alpha_cap + ALPHA_CAP_SLACK && !csr.is_empty_droplet(n);
        }
        state
            .alpha
            .iter_mut()
            .for_each(|a| *a = a.min(params.alpha_cap));
    } else {
        state.excluded.iter_mut().for_each(|e| *e = false);
    }

    // -- beta --

    let (bulk_total, ambient_total, gamma_total) = row_stats
        .iter()
        .fold((0.0_f64, 0.0_f64, 0.0_f64), |(bulk, ambient, gamma), s| {
            (bulk + s.bulk, ambient + s.ambient, gamma + s.gamma)
        });
    state.beta = bulk_total / (bulk_total + ambient_total + gamma_total).max(params.eps);

    // -- ambient profile --

    if !params.freeze_ambient_profile {
        for _ in 0..AMBIENT_UPDATE_ITERS {
            for k in 0..n_celltypes {
                let u_k = state.u[k];
                let profile = &state.profiles[k * n_genes..(k + 1) * n_genes];
                state.u[k] = profile
                    .iter()
                    .zip(&state.ambient)
                    .zip(&totals.a_numer)
                    .map(|((&p, &a), &numer)| u_k * p as f64 / (a as f64).max(params.eps) * numer)
                    .sum();
            }
            let total = sum_simd_f64(&state.u).max(params.eps);
            state.u.iter_mut().for_each(|u| *u /= total);
            state.ambient = normalise_to_f32(&mixture_profile(
                &state.u,
                &state.profiles,
                n_genes,
                n_celltypes,
            ));
        }
    }

    // -- cell-type profiles --

    let celltype_lambda = params.celltype_lambda as f64 / n_genes as f64;
    let repulsion = params.repulsion_strength as f64;
    let max_frac = params.max_frac_gene_repulsion as f64;
    let (ambient, eps) = (&state.ambient, params.eps);

    state
        .profiles
        .par_chunks_mut(n_genes)
        .zip(totals.p_numer.par_chunks(n_genes))
        .for_each(|(profile, numer)| {
            let cluster_mass = sum_simd_f64(numer);
            let repel = repulsion * cluster_mass;

            let mut updated: Vec<f64> = numer.iter().map(|&v| v + celltype_lambda).collect();

            if stage_one {
                updated
                    .iter_mut()
                    .zip(ambient)
                    .for_each(|(value, &ambient_g)| {
                        let sub = (repel * ambient_g as f64).min(max_frac * *value);
                        *value = (*value - sub).max(eps);
                    });
            }

            let total = sum_simd_f64(&updated).max(eps);
            profile
                .iter_mut()
                .zip(updated)
                .for_each(|(slot, value)| *slot = (value / total) as f32);
        });

    state.alpha[..csr.n_real]
        .iter()
        .map(|&a| (1.0 - state.beta) * a + state.beta)
        .collect()
}

/////////////////
// Convergence //
/////////////////

/// Maximum row-wise L1 distance between two profile matrices.
///
/// Profile rows are contiguous, so each row is a straight
/// [SimdDistance::manhattan_simd] call rather than a hand-rolled scalar loop.
///
/// ### Params
///
/// * `current` - Profiles after the latest M-step, row-major.
/// * `previous` - Profiles from the M-step before, row-major.
/// * `n_genes` - Row stride.
///
/// ### Returns
///
/// The largest per-cell-type L1 change.
fn max_row_l1_change(current: &[f32], previous: &[f32], n_genes: usize) -> f64 {
    current
        .par_chunks(n_genes)
        .zip(previous.par_chunks(n_genes))
        .map(|(now, before)| f32::manhattan_simd(now, before) as f64)
        .reduce(|| 0.0_f64, f64::max)
}

///////////////
// Denoising //
///////////////

/// Subtract the expected ambient and bulk counts from the real barcodes.
///
/// Runs as its own pass rather than inside the E-step so the E-step never has
/// to carry a per-non-zero output buffer. Called with the state the last
/// E-step ran on, which is one M-step behind the returned parameters, exactly
/// as in the reference.
///
/// ### Params
///
/// * `csr` - The sample's counts.
/// * `state` - Model state the last E-step used.
/// * `params` - Numerical tolerances.
///
/// ### Returns
///
/// Denoised values for the non-zeros of the real barcodes, clamped at zero.
/// Length `csr.indptr[csr.n_real]`.
fn denoise_real_cells(csr: &SampleCsr, state: &EmState, params: &CellSweepParams) -> Vec<f32> {
    let n_genes = csr.n_genes;
    let nnz_real = csr.indptr[csr.n_real];
    let mut out = vec![0.0_f32; nnz_real];

    let n_threads = rayon::current_num_threads();
    let chunk_size = csr.n_real.div_ceil(n_threads.max(1)).max(1);

    // Row chunks do not fall on equal non-zero boundaries, so the output is
    // carved up by `indptr` rather than by `par_chunks_mut`.
    let mut blocks: Vec<(std::ops::Range<usize>, &mut [f32])> = Vec::new();
    let mut rest = &mut out[..];
    let mut row = 0;
    while row < csr.n_real {
        let end = (row + chunk_size).min(csr.n_real);
        let len = csr.indptr[end] - csr.indptr[row];
        let (head, tail) = rest.split_at_mut(len);
        blocks.push((row..end, head));
        rest = tail;
        row = end;
    }

    blocks.into_par_iter().for_each(|(rows, block)| {
        let base = csr.indptr[rows.start];
        for n in rows {
            let (rs, re) = (csr.indptr[n], csr.indptr[n + 1]);

            let alpha_n = state.alpha[n];
            let w_ambient = (1.0 - state.beta) * alpha_n;
            let w_cell = (1.0 - state.beta) * (1.0 - alpha_n);
            let k = state.gamma_idx[n];
            let profile = &state.profiles[k * n_genes..(k + 1) * n_genes];

            for jj in rs..re {
                let g = csr.indices[jj] as usize;
                let value = csr.data[jj] as f64;

                let wa = w_ambient * state.ambient[g] as f64;
                let wm = state.beta * state.bulk[g] as f64;
                let wc = w_cell * profile[g] as f64;
                let scale = value / (wa + wm + wc).max(params.eps);

                block[jj - base] = (value - scale * wa - scale * wm).max(0.0) as f32;
            }
        }
    });

    out
}

/// Stochastically round non-negative floats to integers.
///
/// Floor plus a Bernoulli draw on the residual, so the rounded counts are
/// unbiased for the denoised expectation. Needed because the store's raw layer
/// is integral and the negative binomial methods downstream depend on that.
///
/// ### Params
///
/// * `values` - Non-negative values to round.
/// * `rng` - Seeded generator, advanced once per value with a fractional part.
///
/// ### Returns
///
/// The rounded counts.
fn stochastic_round(values: &[f32], rng: &mut StdRng) -> Vec<u32> {
    values
        .iter()
        .map(|&value| {
            let floor = value.floor();
            let residual = value - floor;
            let bump = if residual > 0.0 && rng.random::<f32>() < residual {
                1
            } else {
                0
            };
            floor as u32 + bump
        })
        .collect()
}

/////////////
// EM loop //
/////////////

/// Fit the mixture model to one sample.
///
/// ### Params
///
/// * `csr` - The sample's counts, real barcodes first.
/// * `sample_id` - Identifier used in progress output and errors.
/// * `params` - Model parameters.
/// * `verbosity` - [Verbosity::Detailed] prints one line per EM iteration.
///
/// ### Returns
///
/// The fitted parameters and the denoised values for the real barcodes'
/// non-zeros, clamped at zero and in row-major order.
fn fit_em(
    csr: &SampleCsr,
    sample_id: &str,
    params: &CellSweepParams,
    verbosity: Verbosity,
) -> Result<(CellSweepFit, Vec<f32>), BixverseErrors> {
    let (n_rows, n_genes) = (csr.n_rows(), csr.n_genes);
    let mut state = init_em(csr, params);
    let mut row_stats = vec![RowStats::default(); n_rows];
    let mut gamma_out = vec![0_usize; n_rows];

    if params.freeze_ambient_profile {
        let view = state.as_e_step_state(false);
        e_step(csr, &view, &mut row_stats, &mut gamma_out, params);
        for n in 0..csr.n_real {
            let stat = row_stats[n];
            let alpha_test = stat.ambient / (stat.ambient + stat.gamma).max(params.eps);
            state.excluded[n] = alpha_test > params.alpha_cap + ALPHA_CAP_SLACK;
        }
    }

    let mut denoised: Option<Vec<f32>> = None;
    let mut prev_ll = 0.0_f64;
    let mut prev_f: Vec<f64> = Vec::new();
    let mut tol_adaptive = 0.0_f64;
    let mut ll_converged = false;
    let mut converged = false;
    let mut log_likelihood = 0.0_f64;
    let mut n_iter = 0;

    for iteration in 1..=params.max_iter {
        let done = iteration == params.max_iter || converged;
        let stage_one = !ll_converged && params.freeze_ambient_profile;

        let view = state.as_e_step_state(!params.freeze_ambient_profile);
        let totals = e_step(csr, &view, &mut row_stats, &mut gamma_out, params);

        log_likelihood = row_stats.iter().map(|s| s.log_likelihood).sum::<f64>() / n_rows as f64;
        if !log_likelihood.is_finite() {
            return Err(BixverseErrors::CellSweepDiverged {
                sample_id: sample_id.to_string(),
                iteration,
            });
        }

        if done {
            denoised = Some(denoise_real_cells(csr, &state, params));
        }

        state.gamma_idx.copy_from_slice(&gamma_out);

        let prev_profiles = state.profiles.clone();
        let f = m_step(csr, &mut state, &row_stats, &totals, stage_one, params);

        if verbosity.detailed_verbosity() {
            println!(
                "  EM iter {iteration:4}: ll = {log_likelihood:.6}, beta = {:.6}, mean alpha = {:.6}",
                state.beta,
                state.alpha[..csr.n_real].iter().sum::<f64>() / csr.n_real.max(1) as f64
            );
        }

        if iteration == 2 {
            tol_adaptive = (log_likelihood - prev_ll).abs() * params.del0_ll_tol;
        }

        if iteration > 1 && !converged {
            let delta_p = max_row_l1_change(&state.profiles, &prev_profiles, n_genes);
            let mut deltas: Vec<f64> = f
                .iter()
                .zip(&prev_f)
                .map(|(now, before)| (now - before).abs())
                .collect();

            deltas.sort_unstable_by(f64::total_cmp);
            let delta_f = quantile_sorted(&deltas, F_CONVERGENCE_QUANTILE);

            tol_adaptive = tol_adaptive.max(params.min_ll_tol * prev_ll.abs().max(1.0));

            if !ll_converged && (log_likelihood - prev_ll).abs() < tol_adaptive {
                ll_converged = true;
            }
            if ll_converged && delta_p < params.tol_p as f64 && delta_f < params.tol_f {
                converged = true;
            }
        }

        prev_ll = log_likelihood;
        prev_f = f;
        n_iter = iteration;

        if done {
            break;
        }
    }

    let fit = CellSweepFit {
        alpha: state.alpha[..csr.n_real].to_vec(),
        z_hat: state.gamma_idx[..csr.n_real].to_vec(),
        beta: state.beta,
        ambient: state.ambient,
        celltype_profiles: state.profiles,
        log_likelihood,
        n_iter,
        converged,
    };

    // `denoised` is set on the iteration where `done` first holds, and the loop
    // always reaches one: `done` is unconditionally true at `max_iter`.
    let denoised = denoised.expect("the EM loop always runs its final iteration");

    Ok((fit, denoised))
}

impl EmState {
    /// Borrow the state as the E-step's read-only view.
    ///
    /// ### Params
    ///
    /// * `update_ambient` - Whether the E-step should accumulate the ambient
    ///   numerator.
    ///
    /// ### Returns
    ///
    /// The borrowed [EStepState].
    fn as_e_step_state(&self, update_ambient: bool) -> EStepState<'_> {
        EStepState {
            alpha: &self.alpha,
            beta: self.beta,
            ambient: &self.ambient,
            bulk: &self.bulk,
            profiles: &self.profiles,
            gamma_idx: &self.gamma_idx,
            excluded: &self.excluded,
            update_ambient,
        }
    }
}

/////////////////
// CSR loading //
/////////////////

/// Load one sample's counts into a flat in-memory CSR block.
///
/// Real barcodes come first so the denoised output and the fitted `alpha` and
/// `z_hat` vectors are all a prefix of the row space. Kept in memory for the
/// whole fit: the E-step touches every non-zero on every iteration, so
/// streaming would mean thousands of full passes over the store.
///
/// ### Params
///
/// * `reader` - Cell-based (CSR) store to read from.
/// * `sample` - The sample's barcodes and labels.
/// * `n_genes` - Number of genes in the store.
/// * `params` - Model parameters, for the empty droplet requirement.
///
/// ### Returns
///
/// The sample's counts as a [SampleCsr].
fn build_sample_csr<S: SingleCellReading>(
    reader: &S,
    sample: &CellSweepSample,
    n_genes: usize,
    params: &CellSweepParams,
) -> Result<SampleCsr, BixverseErrors> {
    // -- assertions --
    if sample.real_cells.is_empty() || sample.n_celltypes == 0 {
        return Err(BixverseErrors::CellSweepNoRealCells {
            sample_id: sample.sample_id.clone(),
        });
    }
    if sample.celltype_idx.len() != sample.real_cells.len() {
        return Err(BixverseErrors::CellSweepLabelLengthMismatch {
            sample_id: sample.sample_id.clone(),
            n_cells: sample.real_cells.len(),
            n_labels: sample.celltype_idx.len(),
        });
    }
    if let Some(&code) = sample
        .celltype_idx
        .iter()
        .find(|&&code| code >= sample.n_celltypes)
    {
        return Err(BixverseErrors::CellSweepCelltypeOutOfRange {
            sample_id: sample.sample_id.clone(),
            code,
            n_celltypes: sample.n_celltypes,
        });
    }
    if params.freeze_ambient_profile && sample.empty_cells.len() < MIN_EMPTY_DROPLETS {
        return Err(BixverseErrors::CellSweepTooFewEmptyDroplets {
            sample_id: sample.sample_id.clone(),
            found: sample.empty_cells.len(),
            required: MIN_EMPTY_DROPLETS,
        });
    }

    let n_real = sample.real_cells.len();
    let n_rows = n_real + sample.empty_cells.len();

    let mut data: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut indptr: Vec<usize> = Vec::with_capacity(n_rows + 1);
    indptr.push(0);

    for batch in sample
        .real_cells
        .iter()
        .chain(&sample.empty_cells)
        .copied()
        .collect::<Vec<usize>>()
        .chunks(CELL_BATCH_SIZE)
    {
        for chunk in reader.read_cells_parallel(batch)? {
            data.extend(chunk.data_raw.iter().map(|c| c as f32));
            indices.extend_from_slice(&chunk.indices);
            indptr.push(indices.len());
        }
    }

    let mut gamma_idx = vec![usize::MAX; n_rows];
    gamma_idx[..n_real].copy_from_slice(&sample.celltype_idx);

    Ok(SampleCsr {
        data,
        indices,
        indptr,
        n_genes,
        n_real,
        gamma_idx,
        n_celltypes: sample.n_celltypes,
    })
}

////////////////////
// Denoised store //
////////////////////

/// Everything the R layer needs after a CellSweep run.
#[derive(Clone, Debug)]
pub struct CellSweepRun {
    /// One fit per sample, in the order the samples were given.
    pub fits: Vec<CellSweepFit>,
    /// Store indices of the written barcodes, in output order. Lets the caller
    /// subset the parent's obs table without guessing at the ordering.
    pub cell_order: Vec<usize>,
    /// Denoised library size per written barcode, in output order.
    pub library_size: Vec<usize>,
    /// Non-zeros per written barcode after the clamp, in output order.
    pub nnz: Vec<usize>,
}

/// Build one output chunk from a barcode's denoised values.
///
/// Values that the clamp drove to zero are dropped, so the output is sparser
/// than the input. The raw layer takes the stochastically rounded counts, since
/// the store is integral and the negative binomial methods downstream depend on
/// that; the normalised layer keeps the floats by default, which is where the
/// sub-integer part of the denoised signal survives.
///
/// ### Params
///
/// * `indices` - Gene indices of the barcode's non-zeros.
/// * `denoised` - Denoised values, same length as `indices`.
/// * `original_index` - Row index in the new store.
/// * `target_size` - Library size the normalised layer is scaled to.
/// * `norm_from_rounded` - Derive the normalised layer from the rounded counts
///   rather than the floats.
/// * `rng` - Seeded generator for the stochastic rounding.
///
/// ### Returns
///
/// The [CsrCellChunk] ready to write.
fn build_denoised_chunk(
    indices: &[u32],
    denoised: &[f32],
    original_index: usize,
    target_size: f32,
    norm_from_rounded: bool,
    rng: &mut StdRng,
) -> CsrCellChunk {
    let all_rounded = stochastic_round(denoised, rng);
    let kept: Vec<usize> = (0..denoised.len())
        .filter(|&i| all_rounded[i] > 0)
        .collect();

    let values: Vec<f32> = kept.iter().map(|&i| denoised[i]).collect();
    let kept_indices: Vec<u32> = kept.iter().map(|&i| indices[i]).collect();
    let rounded: Vec<u32> = kept.iter().map(|&i| all_rounded[i]).collect();

    let norm_source: Vec<f32> = if norm_from_rounded {
        rounded.iter().map(|&c| c as f32).collect()
    } else {
        values.clone()
    };
    let norm_total: f32 = norm_source.iter().sum();

    let data_norm: Vec<F16> = norm_source
        .iter()
        .map(|&value| {
            let scaled = if norm_total > 0.0 {
                (value / norm_total * target_size).ln_1p()
            } else {
                0.0
            };
            F16::from_f32(scaled)
        })
        .collect();

    CsrCellChunk {
        data_raw: RawCounts::from_u32_auto(&rounded),
        data_norm,
        library_size: rounded.iter().map(|&c| c as usize).sum(),
        indices: kept_indices,
        original_index,
        to_keep: true,
    }
}

/// Fit CellSweep per sample and write the denoised barcodes to a new store.
///
/// One EM per sample, since the ambient profile is a property of a single
/// emulsion. Only the real barcodes are written: empty droplets exist to train
/// the ambient profile, and barcodes that are neither empty nor annotated are
/// not part of the model at all.
///
/// Writes the cell-based (CSR) file only. The gene-based companion has to be
/// regenerated afterwards, as it is for a merge.
///
/// ### Params
///
/// * `reader` - Cell-based (CSR) store holding the raw counts, including the
///   empty droplets.
/// * `samples` - One entry per sample. Barcodes must not repeat across
///   samples.
/// * `params` - Model parameters.
/// * `output_path` - Path of the new cells .bin to write.
/// * `target_size` - Library size the normalised layer is scaled to.
/// * `verbose` - `0` silent, `1` per-sample progress, `2` per-EM-iteration.
///
/// ### Returns
///
/// [CellSweepRun] with one fit per sample plus the per-barcode bookkeeping the
/// caller needs to rebuild its obs table.
pub fn run_cellsweep<S, P>(
    reader: &S,
    samples: &[CellSweepSample],
    params: CellSweepParams,
    output_path: P,
    target_size: f32,
    verbose: usize,
) -> Result<CellSweepRun, BixverseErrors>
where
    S: SingleCellReading,
    P: AsRef<std::path::Path>,
{
    // -- assertions --
    if !params.freeze_empties {
        return Err(BixverseErrors::CellSweepFreezeEmptiesUnsupported);
    }
    if !reader.is_cell_based() {
        return Err(BixverseErrors::ReaderModeMismatch {
            actual: "gene-based",
            requested: "cell-based",
        });
    }

    let verbosity = parse_verbosity_level(verbose);
    let header = reader.get_header();
    let n_genes = header.total_genes;
    let total_cells: usize = samples.iter().map(|s| s.real_cells.len()).sum();

    let start = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "CellSweep: {} samples, {} barcodes to denoise, {} genes",
            samples.len(),
            total_cells,
            n_genes
        );
    }

    let mut writer =
        CellGeneSparseWriter::new(output_path, true, total_cells, n_genes, target_size)?;
    let mut rng = StdRng::seed_from_u64(params.seed);

    let mut run = CellSweepRun {
        fits: Vec::with_capacity(samples.len()),
        cell_order: Vec::with_capacity(total_cells),
        library_size: Vec::with_capacity(total_cells),
        nnz: Vec::with_capacity(total_cells),
    };

    for (sample_idx, sample) in samples.iter().enumerate() {
        if verbosity.normal_verbosity() {
            println!(
                "  Sample {}/{} ('{}'): {} barcodes, {} empty droplets",
                sample_idx + 1,
                samples.len(),
                sample.sample_id,
                sample.real_cells.len(),
                sample.empty_cells.len()
            );
            if params.freeze_ambient_profile
                && sample.empty_cells.len() < RECOMMENDED_MIN_EMPTY_DROPLETS
            {
                println!(
                    "   Warning: fewer than {RECOMMENDED_MIN_EMPTY_DROPLETS} empty droplets, the ambient profile may be unreliable."
                );
            }
        }

        let csr = build_sample_csr(reader, sample, n_genes, &params)?;
        let (fit, denoised) = fit_em(&csr, &sample.sample_id, &params, verbosity)?;

        if verbosity.normal_verbosity() {
            println!(
                "   {} EM iterations, converged = {}, ll = {:.6}, beta = {:.6}",
                fit.n_iter, fit.converged, fit.log_likelihood, fit.beta
            );
        }

        for local in 0..csr.n_real {
            let (rs, re) = (csr.indptr[local], csr.indptr[local + 1]);
            let chunk = build_denoised_chunk(
                &csr.indices[rs..re],
                &denoised[rs..re],
                run.cell_order.len(),
                target_size,
                params.norm_from_rounded,
                &mut rng,
            );

            run.cell_order.push(sample.real_cells[local]);
            run.library_size.push(chunk.library_size);
            run.nnz.push(chunk.indices.len());
            writer.write_cell_chunk(chunk)?;
        }

        run.fits.push(fit);
    }

    writer.finalise()?;

    if verbosity.normal_verbosity() {
        println!("CellSweep finished in {:.2?}", start.elapsed());
    }

    Ok(run)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// RAII guard that removes a test's temp file even if an assert fails.
    struct TempBin(std::path::PathBuf);

    /// Drop implementation for [`TempBin`]. Errors are ignored: the file may
    /// already be gone, and this runs during unwind.
    impl Drop for TempBin {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempBin {
        /// Reserve a uniquely named scratch file in the system temp directory.
        ///
        /// ### Params
        ///
        /// * `name` - Test-unique suffix.
        ///
        /// ### Returns
        ///
        /// The guard; the path is available via [`Self::path`].
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_cellsweep_{name}.bin")))
        }

        /// Path of the guarded file as a `&str`.
        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// Genes in the two-type fixture.
    const FIXTURE_GENES: usize = 8;

    /// Marker genes of cell type 0 in the two-type fixture.
    const MARKERS_0: std::ops::Range<usize> = 0..4;

    /// Marker genes of cell type 1 in the two-type fixture.
    const MARKERS_1: std::ops::Range<usize> = 4..8;

    /// Weight of [MARKERS_0] in the fixture's ambient profile.
    ///
    /// A flat ambient profile is unidentifiable: adding a constant to every
    /// `p_k` explains it exactly as well, and both this implementation and the
    /// reference duly collapse `alpha` to near zero. Skewing it is what makes
    /// the fixture test anything.
    const FIXTURE_AMBIENT_SKEW: f32 = 4.0;

    /// Build a [SampleCsr] straight from dense rows, bypassing the store.
    ///
    /// ### Params
    ///
    /// * `real` - Dense rows of the annotated barcodes.
    /// * `empty` - Dense rows of the empty droplets.
    /// * `labels` - Cell-type code per real barcode.
    /// * `n_celltypes` - Number of cell types.
    ///
    /// ### Returns
    ///
    /// The populated [SampleCsr].
    fn csr_from_dense(
        real: &[Vec<f32>],
        empty: &[Vec<f32>],
        labels: &[usize],
        n_celltypes: usize,
    ) -> SampleCsr {
        let n_genes = real[0].len();
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0_usize];

        for row in real.iter().chain(empty) {
            for (g, &value) in row.iter().enumerate() {
                if value > 0.0 {
                    data.push(value);
                    indices.push(g as u32);
                }
            }
            indptr.push(indices.len());
        }

        let mut gamma_idx = vec![usize::MAX; real.len() + empty.len()];
        gamma_idx[..real.len()].copy_from_slice(labels);

        SampleCsr {
            data,
            indices,
            indptr,
            n_genes,
            n_real: real.len(),
            gamma_idx,
            n_celltypes,
        }
    }

    /// Ambient profile of the two-type fixture, skewed towards [MARKERS_0].
    ///
    /// ### Returns
    ///
    /// Per-gene ambient weights, summing to one.
    fn fixture_ambient() -> Vec<f32> {
        let weights: Vec<f32> = (0..FIXTURE_GENES)
            .map(|g| {
                if MARKERS_0.contains(&g) {
                    FIXTURE_AMBIENT_SKEW
                } else {
                    1.0
                }
            })
            .collect();
        let total: f32 = weights.iter().sum();
        weights.iter().map(|w| w / total).collect()
    }

    /// Two cell types on disjoint marker blocks, contaminated from a skewed
    /// ambient profile with a per-barcode budget.
    ///
    /// Two things make the contamination identifiable, and the fixture needs
    /// both. The skew (see [FIXTURE_AMBIENT_SKEW]) stops the ambient profile
    /// being absorbable into every `p_k` at once. The per-barcode budget stops
    /// it being absorbable into any single `p_k`: a shared profile can explain
    /// a constant amount of contamination but not one that varies tenfold from
    /// barcode to barcode, which only the per-cell `alpha` can.
    ///
    /// ### Params
    ///
    /// * `n_per_type` - Real barcodes per cell type.
    /// * `n_empty` - Empty droplets to generate.
    ///
    /// ### Returns
    ///
    /// The real rows, the empty rows and the cell-type labels. Genes
    /// [MARKERS_0] are cell type 0's markers, [MARKERS_1] cell type 1's.
    fn two_type_fixture(
        n_per_type: usize,
        n_empty: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
        let ambient = fixture_ambient();
        let mut real = Vec::new();
        let mut labels = Vec::new();

        for k in 0..2 {
            let own = if k == 0 { MARKERS_0 } else { MARKERS_1 };

            for i in 0..n_per_type {
                // Deterministic but wide spread, so alpha has something to fit.
                let budget = 20.0 + (17 * i % 180) as f32;
                let row = (0..FIXTURE_GENES)
                    .map(|g| {
                        let signal = if own.contains(&g) {
                            25.0 + ((i + g) % 5) as f32
                        } else {
                            0.0
                        };
                        (signal + budget * ambient[g]).round()
                    })
                    .collect();
                real.push(row);
                labels.push(k);
            }
        }

        let empty = (0..n_empty)
            .map(|i| {
                let total = 200.0 + (23 * i % 200) as f32;
                (0..FIXTURE_GENES)
                    .map(|g| (total * ambient[g]).round())
                    .collect()
            })
            .collect();

        (real, empty, labels)
    }

    // -- empty droplet calling --

    #[test]
    fn test_expected_cells_takes_the_nth_largest_library() {
        let sizes = vec![100, 5, 50, 1, 20];
        assert_eq!(cutoff_for_expected_cells(&sizes, 1).unwrap(), 100);
        assert_eq!(cutoff_for_expected_cells(&sizes, 3).unwrap(), 20);
        assert_eq!(cutoff_for_expected_cells(&sizes, 5).unwrap(), 1);
    }

    #[test]
    fn test_expected_cells_rejects_out_of_range() {
        let sizes = vec![10, 20, 30];
        assert!(cutoff_for_expected_cells(&sizes, 0).is_err());
        assert!(cutoff_for_expected_cells(&sizes, 4).is_err());
    }

    #[test]
    fn test_infer_empty_droplets_masks_below_the_cutoff() {
        let sizes = vec![1000, 500, 40, 3];
        let mask = infer_empty_droplets(&sizes, EmptyDropletCall::UmiCutoff(100)).unwrap();
        assert_eq!(mask, vec![false, false, true, true]);
    }

    #[test]
    fn test_infer_empty_droplets_errors_when_mask_was_promised() {
        let sizes = vec![10, 20];
        assert!(infer_empty_droplets(&sizes, EmptyDropletCall::Supplied).is_err());
    }

    #[test]
    fn test_knee_lands_on_the_cliff_of_a_two_plateau_curve() {
        // 200 real barcodes near 10k counts, then a cliff to 2000 empties near
        // 50. The knee has to fall in the transition, not out in either
        // plateau.
        let mut sizes: Vec<u32> = (0..200).map(|i| 12_000 - 10 * i as u32).collect();
        sizes.extend((0..2000).map(|i| 60 - (i as u32 % 20)));

        let cutoff = knee_umi_cutoff(&sizes).unwrap();
        assert!(
            cutoff > 60 && cutoff <= 12_000,
            "knee cutoff {cutoff} fell inside a plateau"
        );

        // Every true empty has to be caught. Smoothing puts the curvature
        // minimum a few ranks ahead of the cliff, so the real barcodes nearest
        // the transition get swept up too; the reference behaves the same way,
        // which is why this detector is the fallback rather than the default.
        let mask = infer_empty_droplets(&sizes, EmptyDropletCall::Knee).unwrap();
        assert!(mask[200..].iter().all(|&m| m));
        let called_empty = mask.iter().filter(|&&m| m).count();
        assert!(
            (2000..=2010).contains(&called_empty),
            "knee called {called_empty} barcodes empty, expected close to 2000"
        );
    }

    #[test]
    fn test_knee_errors_on_too_few_barcodes() {
        assert!(knee_umi_cutoff(&[500, 400]).is_err());
        assert!(knee_umi_cutoff(&[1, 2, 3]).is_err());
    }

    // -- numerical helpers --

    #[test]
    fn test_gaussian_smooth_leaves_a_constant_alone() {
        let y = vec![3.0; 50];
        let smoothed = gaussian_smooth_1d(&y, KNEE_SMOOTHING_SIGMA);
        smoothed
            .iter()
            .for_each(|&v| assert_relative_eq!(v, 3.0, epsilon = 1e-12));
    }

    #[test]
    fn test_reflect_index_folds_both_ends() {
        assert_eq!(reflect_index(-1, 5), 0);
        assert_eq!(reflect_index(-3, 5), 2);
        assert_eq!(reflect_index(5, 5), 4);
        assert_eq!(reflect_index(7, 5), 2);
        assert_eq!(reflect_index(2, 5), 2);
        // A kernel wider than the signal reflects more than once.
        assert!(reflect_index(-20, 3) < 3);
        assert!(reflect_index(20, 3) < 3);
    }

    #[test]
    fn test_central_gradient_matches_a_linear_ramp() {
        let y: Vec<f64> = (0..6).map(|i| 2.0 * i as f64).collect();
        let grad = central_gradient(&y);
        grad.iter()
            .for_each(|&g| assert_relative_eq!(g, 2.0, epsilon = 1e-12));
    }

    #[test]
    fn test_max_row_l1_change_takes_the_worst_row() {
        // Row 0 moves by 0.3, row 1 by 0.02.
        let current = vec![0.5, 0.3, 0.2, 0.30, 0.40, 0.30];
        let previous = vec![0.4, 0.4, 0.2, 0.31, 0.39, 0.30];
        assert_relative_eq!(
            max_row_l1_change(&current, &previous, 3),
            0.2,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_normalise_to_f32_handles_an_all_zero_profile() {
        let profile = normalise_to_f32(&[0.0, 0.0, 0.0, 0.0]);
        profile
            .iter()
            .for_each(|&v| assert_relative_eq!(v, 0.25, epsilon = 1e-6));
    }

    // -- stochastic rounding --

    #[test]
    fn test_stochastic_round_is_reproducible_for_a_seed() {
        let values = vec![0.4, 1.6, 2.5, 0.01, 9.99];
        let first = stochastic_round(&values, &mut StdRng::seed_from_u64(7));
        let second = stochastic_round(&values, &mut StdRng::seed_from_u64(7));
        assert_eq!(first, second);
    }

    #[test]
    fn test_stochastic_round_preserves_the_expectation() {
        // The only meaningful test of the rounding: the reference draws from
        // numpy's generator, which cannot be reproduced here, so parity is on
        // the mean rather than the values.
        let values = vec![0.3_f32; 20_000];
        let rounded = stochastic_round(&values, &mut StdRng::seed_from_u64(11));
        let mean = rounded.iter().map(|&c| c as f64).sum::<f64>() / values.len() as f64;
        assert_relative_eq!(mean, 0.3, epsilon = 0.02);
    }

    #[test]
    fn test_stochastic_round_keeps_integers_exact() {
        let values = vec![0.0, 1.0, 4.0, 17.0];
        let rounded = stochastic_round(&values, &mut StdRng::seed_from_u64(3));
        assert_eq!(rounded, vec![0, 1, 4, 17]);
    }

    // -- initialisation --

    #[test]
    fn test_celltype_means_ignore_empty_droplets() {
        // One real barcode per type plus an empty droplet loaded on the genes
        // of type 0. The means must not see it.
        let real = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
        let empty = vec![vec![100.0, 0.0]];
        let csr = csr_from_dense(&real, &empty, &[0, 1], 2);

        let means = celltype_mean_expression(&csr);
        assert_relative_eq!(means[0] as f32, 10.0, epsilon = 1e-6);
        assert_relative_eq!(means[1] as f32, 0.0, epsilon = 1e-6);
        assert_relative_eq!(means[2] as f32, 0.0, epsilon = 1e-6);
        assert_relative_eq!(means[3] as f32, 10.0, epsilon = 1e-6);
    }

    #[test]
    fn test_init_seeds_the_ambient_profile_from_the_empties() {
        let (real, empty, labels) = two_type_fixture(20, 40);
        let csr = csr_from_dense(&real, &empty, &labels, 2);
        let state = init_em(&csr, &CellSweepParams::default());

        // The empties carry the skewed ambient shape, so the recovered profile
        // has to reproduce it.
        let expected: f32 = fixture_ambient()[MARKERS_0].iter().sum();
        let ambient_on_type_0: f32 = state.ambient[MARKERS_0].iter().sum();
        assert_relative_eq!(ambient_on_type_0, expected, epsilon = 0.02);

        // Profiles and empties both normalise to one, and the empties are
        // pinned.
        assert_relative_eq!(state.ambient.iter().sum::<f32>(), 1.0, epsilon = 1e-5);
        assert_relative_eq!(state.bulk.iter().sum::<f32>(), 1.0, epsilon = 1e-5);
        for k in 0..2 {
            let row: f32 = state.profiles[k * FIXTURE_GENES..(k + 1) * FIXTURE_GENES]
                .iter()
                .sum();
            assert_relative_eq!(row, 1.0, epsilon = 1e-5);
        }
        assert!(state.alpha[csr.n_real..].iter().all(|&a| a == 1.0));
    }

    // -- the fit --

    #[test]
    fn test_em_strips_the_ambient_block_and_keeps_the_signal() {
        let (real, empty, labels) = two_type_fixture(40, 60);
        let csr = csr_from_dense(&real, &empty, &labels, 2);
        let params = CellSweepParams {
            max_iter: 200,
            ..Default::default()
        };

        let (fit, denoised) = fit_em(&csr, "toy", &params, Verbosity::Quiet).unwrap();

        // Denoising can only ever remove mass.
        for (jj, &value) in denoised.iter().enumerate() {
            assert!(
                value <= csr.data[jj] + 1e-4,
                "denoised value {value} exceeds the raw count {}",
                csr.data[jj]
            );
        }

        // For a type 1 barcode the counts on [MARKERS_0] are pure
        // contamination drawn from the ambient-heavy block, so they have to
        // lose far more of their mass than the counts on its own markers.
        let (mut own_raw, mut own_denoised) = (0.0_f64, 0.0_f64);
        let (mut foreign_raw, mut foreign_denoised) = (0.0_f64, 0.0_f64);
        for n in 0..csr.n_real {
            if labels[n] != 1 {
                continue;
            }
            for jj in csr.indptr[n]..csr.indptr[n + 1] {
                let g = csr.indices[jj] as usize;
                if MARKERS_1.contains(&g) {
                    own_raw += csr.data[jj] as f64;
                    own_denoised += denoised[jj] as f64;
                } else {
                    foreign_raw += csr.data[jj] as f64;
                    foreign_denoised += denoised[jj] as f64;
                }
            }
        }

        let own_kept = own_denoised / own_raw;
        let foreign_kept = foreign_denoised / foreign_raw;
        assert!(
            own_kept > 0.7,
            "kept only {own_kept} of the cell-type signal"
        );
        assert!(
            foreign_kept < 0.4,
            "kept {foreign_kept} of the foreign-marker contamination"
        );
        assert!(
            own_kept > foreign_kept + 0.3,
            "signal and contamination were stripped at similar rates"
        );

        // Fitted profiles stay simplex-valued.
        assert_relative_eq!(fit.ambient.iter().sum::<f32>(), 1.0, epsilon = 1e-4);
        for k in 0..2 {
            let row: f32 = fit.celltype_profiles[k * FIXTURE_GENES..(k + 1) * FIXTURE_GENES]
                .iter()
                .sum();
            assert_relative_eq!(row, 1.0, epsilon = 1e-4);
        }
        assert!((0.0..=1.0).contains(&fit.beta));
        assert!(fit.alpha.iter().all(|&a| (0.0..=1.0).contains(&a)));
        assert_eq!(fit.alpha.len(), csr.n_real);
        assert_eq!(fit.z_hat.len(), csr.n_real);
    }

    #[test]
    fn test_em_converges_before_the_iteration_cap() {
        let (real, empty, labels) = two_type_fixture(30, 50);
        let csr = csr_from_dense(&real, &empty, &labels, 2);
        // Converges around iteration 560 on this fixture; the cap is only
        // here so a regression that stops it converging still fails.
        let params = CellSweepParams {
            max_iter: 1500,
            ..Default::default()
        };

        let (fit, _) = fit_em(&csr, "toy", &params, Verbosity::Quiet).unwrap();
        assert!(
            fit.converged,
            "did not converge in {} iterations",
            fit.n_iter
        );
        assert!(fit.n_iter < params.max_iter);
    }

    #[test]
    fn test_em_is_deterministic() {
        let (real, empty, labels) = two_type_fixture(20, 40);
        let csr = csr_from_dense(&real, &empty, &labels, 2);
        let params = CellSweepParams {
            max_iter: 100,
            ..Default::default()
        };

        let (first, first_denoised) = fit_em(&csr, "toy", &params, Verbosity::Quiet).unwrap();
        let (second, second_denoised) = fit_em(&csr, "toy", &params, Verbosity::Quiet).unwrap();

        assert_eq!(first.n_iter, second.n_iter);
        assert_relative_eq!(first.beta, second.beta, epsilon = 1e-12);
        assert_eq!(first_denoised, second_denoised);
    }

    #[test]
    fn test_em_denoises_against_the_pre_reassignment_profile() {
        // The reference writes its per-entry ambient and bulk shares inside
        // the E-step's entry loop, on the cell type the barcode held going in,
        // and only reassigns after that loop. So a barcode that switches type
        // on a `done` pass must be denoised against its OLD profile.
        //
        // Isolating that needs the reassignment to happen on the final pass
        // specifically, not an earlier one. `max_iter = 1` guarantees it: the
        // single iteration is the `done` pass, the warm-up has already filled
        // the exclusion set, and denoising runs before the first M-step. So
        // the state used for denoising is exactly what `init_em` produced,
        // which the test can rebuild and compare against bit for bit.
        let (real, empty, mut labels) = two_type_fixture(20, 40);
        // Mislabel a handful so the likelihood actually prefers a different
        // type. Reassignment fires for every excluded barcode, but with
        // disjoint marker blocks a correctly labelled one always keeps its own
        // profile, so nothing would change and nothing would be tested.
        for n in [0, 2, 21, 23] {
            labels[n] = 1 - labels[n];
        }
        let csr = csr_from_dense(&real, &empty, &labels, 2);
        let params = CellSweepParams {
            alpha_cap: 0.1,
            max_iter: 1,
            ..Default::default()
        };

        let (fit, denoised) = fit_em(&csr, "reassign", &params, Verbosity::Quiet).unwrap();

        // Confirm the path was actually taken, or the test proves nothing.
        let reassigned = fit
            .z_hat
            .iter()
            .zip(&labels)
            .filter(|&(&got, &want)| got != want)
            .count();
        assert!(
            reassigned > 0,
            "no barcode was reassigned on the final pass, so the ordering is untested"
        );

        let initial = init_em(&csr, &params);
        assert_eq!(
            initial.gamma_idx[..csr.n_real],
            labels[..],
            "init_em should start from the given labels"
        );

        let expected = denoise_real_cells(&csr, &initial, &params);
        assert_eq!(
            denoised, expected,
            "denoised values do not match the pre-reassignment state"
        );

        // And the reassigned assignment must give something different, or the
        // assertion above would hold either way.
        let mut reassigned_state = initial;
        reassigned_state.gamma_idx[..csr.n_real].copy_from_slice(&fit.z_hat);
        let wrong = denoise_real_cells(&csr, &reassigned_state, &params);
        assert_ne!(
            denoised, wrong,
            "the two assignments denoise identically, so this cannot discriminate"
        );
    }

    #[test]
    fn test_em_runs_the_ambient_update_branch() {
        // freeze_ambient_profile = false takes the `u @ p` re-estimation path
        // and disables the cap, repulsion and reassignment.
        let (real, empty, labels) = two_type_fixture(20, 40);
        let csr = csr_from_dense(&real, &empty, &labels, 2);
        let params = CellSweepParams {
            freeze_ambient_profile: false,
            max_iter: 100,
            ..Default::default()
        };

        let (fit, denoised) = fit_em(&csr, "toy", &params, Verbosity::Quiet).unwrap();

        assert_relative_eq!(fit.ambient.iter().sum::<f32>(), 1.0, epsilon = 1e-4);
        assert!(denoised.iter().all(|&v| v >= 0.0));
        // The ambient profile is now a mixture of the cell-type profiles
        // rather than the empty droplets' pooled counts.
        assert_relative_eq!(fit.ambient.iter().sum::<f32>(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_em_runs_with_a_single_celltype() {
        let real = vec![vec![5.0, 7.0, 1.0], vec![6.0, 8.0, 2.0]];
        let empty: Vec<Vec<f32>> = (0..40).map(|_| vec![0.0, 0.0, 3.0]).collect();
        let csr = csr_from_dense(&real, &empty, &[0, 0], 1);
        let params = CellSweepParams {
            max_iter: 50,
            ..Default::default()
        };

        let (fit, denoised) = fit_em(&csr, "single", &params, Verbosity::Quiet).unwrap();
        assert!(fit.z_hat.iter().all(|&k| k == 0));
        assert!(denoised.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn test_em_tolerates_more_celltypes_than_barcodes() {
        let real = vec![vec![5.0, 0.0, 0.0], vec![0.0, 5.0, 0.0]];
        let empty: Vec<Vec<f32>> = (0..40).map(|_| vec![0.0, 0.0, 2.0]).collect();
        let csr = csr_from_dense(&real, &empty, &[0, 1], 4);
        let params = CellSweepParams {
            max_iter: 50,
            ..Default::default()
        };

        let (fit, _) = fit_em(&csr, "sparse-labels", &params, Verbosity::Quiet).unwrap();
        // The two unused cell types get pseudocount-only profiles, which are
        // uniform rather than degenerate.
        for k in 2..4 {
            let row = &fit.celltype_profiles[k * 3..(k + 1) * 3];
            row.iter()
                .for_each(|&v| assert_relative_eq!(v, 1.0 / 3.0, epsilon = 1e-5));
        }
    }

    #[test]
    fn test_em_tolerates_an_all_zero_barcode() {
        let real = vec![vec![5.0, 1.0, 0.0], vec![0.0, 0.0, 0.0]];
        let empty: Vec<Vec<f32>> = (0..40).map(|_| vec![0.0, 0.0, 2.0]).collect();
        let csr = csr_from_dense(&real, &empty, &[0, 0], 1);
        let params = CellSweepParams {
            max_iter: 50,
            ..Default::default()
        };

        let (fit, denoised) = fit_em(&csr, "empty-row", &params, Verbosity::Quiet).unwrap();
        assert!(fit.log_likelihood.is_finite());
        assert!(denoised.iter().all(|&v| v.is_finite()));
    }

    // -- output chunks --

    #[test]
    fn test_denoised_chunk_drops_the_clamped_zeros() {
        let mut rng = StdRng::seed_from_u64(1);
        let chunk = build_denoised_chunk(
            &[0, 1, 2, 3],
            &[4.0, 0.0, 6.0, 0.0],
            7,
            1e4,
            false,
            &mut rng,
        );

        assert_eq!(chunk.indices, vec![0, 2]);
        assert_eq!(chunk.data_raw.len(), 2);
        assert_eq!(chunk.data_norm.len(), 2);
        assert_eq!(chunk.library_size, 10);
        assert_eq!(chunk.original_index, 7);
    }

    #[test]
    fn test_denoised_chunk_never_stores_an_explicit_zero() {
        // Sub-integer values can round to zero. They have to leave the chunk
        // entirely, or `library_size` stops being the sum of `data_raw` and
        // every consumer computing a fraction of the library divides by a
        // total that does not match the entries it summed.
        for seed in 0..64_u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let chunk = build_denoised_chunk(
                &[0, 1, 2, 3, 4, 5],
                &[0.2, 0.4, 0.5, 0.6, 0.8, 3.0],
                0,
                1e4,
                false,
                &mut rng,
            );

            let raw: Vec<u32> = chunk.data_raw.iter().collect();
            assert!(
                raw.iter().all(|&c| c > 0),
                "seed {seed} stored an explicit zero: {raw:?}"
            );
            assert_eq!(
                chunk.library_size,
                raw.iter().map(|&c| c as usize).sum::<usize>(),
                "seed {seed}: library_size disagrees with data_raw"
            );
            assert_eq!(chunk.indices.len(), raw.len());
            assert_eq!(chunk.data_norm.len(), raw.len());
        }
    }

    #[test]
    fn test_denoised_chunk_survives_a_fully_rounded_away_barcode() {
        // Every value below one, all Bernoulli draws lost: a legitimate
        // outcome that has to produce a coherent empty chunk rather than a
        // library size that disagrees with the data.
        let mut rng = StdRng::seed_from_u64(0);
        let chunk = build_denoised_chunk(&[0, 1], &[0.0, 0.0], 3, 1e4, false, &mut rng);

        assert!(chunk.indices.is_empty());
        assert_eq!(chunk.data_raw.len(), 0);
        assert_eq!(chunk.data_norm.len(), 0);
        assert_eq!(chunk.library_size, 0);
        assert_eq!(chunk.original_index, 3);
    }

    #[test]
    fn test_denoised_chunk_normalises_to_the_target_size() {
        let mut rng = StdRng::seed_from_u64(1);
        let chunk = build_denoised_chunk(&[0, 1], &[3.0, 1.0], 0, 1e4, false, &mut rng);

        // ln1p(0.75 * 1e4) and ln1p(0.25 * 1e4), within f16 resolution.
        let values: Vec<f32> = chunk.data_norm.iter().map(|v| v.to_f32()).collect();
        assert_relative_eq!(values[0], (0.75_f32 * 1e4).ln_1p(), epsilon = 1e-2);
        assert_relative_eq!(values[1], (0.25_f32 * 1e4).ln_1p(), epsilon = 1e-2);
    }

    #[test]
    fn test_denoised_chunk_can_take_the_norm_layer_from_the_rounded_counts() {
        // 0.4 rounds to 0 or 1; with `norm_from_rounded` the normalised layer
        // has to agree with whichever it was.
        let mut rng = StdRng::seed_from_u64(5);
        let chunk = build_denoised_chunk(&[0, 1], &[0.4, 8.0], 0, 1e4, true, &mut rng);

        let raw: Vec<u32> = chunk.data_raw.iter().collect();
        let norm: Vec<f32> = chunk.data_norm.iter().map(|v| v.to_f32()).collect();
        for (&count, &value) in raw.iter().zip(&norm) {
            assert_eq!(count == 0, value == 0.0);
        }
    }

    // -- the full run --

    /// Write a raw store with the two-type fixture and return the reader.
    ///
    /// ### Params
    ///
    /// * `path` - Where to write the .bin.
    /// * `n_per_type` - Real barcodes per cell type.
    /// * `n_empty` - Empty droplets.
    ///
    /// ### Returns
    ///
    /// The reader plus the cell-type labels of the real barcodes, which occupy
    /// store indices `0..2 * n_per_type`.
    fn write_fixture_store(
        path: &str,
        n_per_type: usize,
        n_empty: usize,
    ) -> (ParallelSparseReader, Vec<usize>) {
        let (real, empty, labels) = two_type_fixture(n_per_type, n_empty);
        let n_genes = real[0].len();
        let mut writer =
            CellGeneSparseWriter::new(path, true, real.len() + empty.len(), n_genes, 1e4)
                .expect("writer opens");

        for (i, row) in real.iter().chain(&empty).enumerate() {
            let indices: Vec<u32> = (0..n_genes as u32)
                .filter(|&g| row[g as usize] > 0.0)
                .collect();
            let counts: Vec<u32> = indices.iter().map(|&g| row[g as usize] as u32).collect();
            writer
                .write_cell_chunk(CsrCellChunk::from_data(&counts, &indices, i, 1e4, true))
                .expect("chunk writes");
        }
        writer.finalise().expect("store finalises");

        (
            ParallelSparseReader::new(path).expect("store opens"),
            labels,
        )
    }

    #[test]
    fn test_run_cellsweep_round_trips_through_the_store() {
        let raw = TempBin::new("run_raw");
        let out = TempBin::new("run_out");
        let (reader, labels) = write_fixture_store(raw.path(), 25, 50);

        let sample = CellSweepSample {
            sample_id: "s1".to_string(),
            real_cells: (0..50).collect(),
            empty_cells: (50..100).collect(),
            celltype_idx: labels,
            n_celltypes: 2,
        };

        let params = CellSweepParams {
            max_iter: 100,
            ..Default::default()
        };
        let run = run_cellsweep(&reader, &[sample], params, out.path(), 1e4, 0).unwrap();

        assert_eq!(run.fits.len(), 1);
        assert_eq!(run.cell_order, (0..50).collect::<Vec<usize>>());
        assert_eq!(run.library_size.len(), 50);

        let denoised_reader = ParallelSparseReader::new(out.path()).expect("output opens");
        let header = denoised_reader.get_header();
        assert_eq!(header.total_cells, 50);
        assert_eq!(header.total_genes, FIXTURE_GENES);
        assert!(denoised_reader.is_cell_based());
        assert_eq!(denoised_reader.target_size(), Some(1e4));

        // Every written barcode must have lost mass and the store has to agree
        // with the bookkeeping handed back to the caller.
        let raw_libs = reader
            .read_cell_library_sizes(&(0..50).collect::<Vec<usize>>())
            .unwrap();
        for (i, chunk) in denoised_reader.get_all_cells().unwrap().iter().enumerate() {
            assert_eq!(chunk.library_size, run.library_size[i]);
            assert_eq!(chunk.indices.len(), run.nnz[i]);
            assert_eq!(chunk.data_raw.len(), chunk.data_norm.len());
            assert!(
                chunk.library_size <= raw_libs[i],
                "barcode {i} gained counts: {} > {}",
                chunk.library_size,
                raw_libs[i]
            );
        }

        // Denoising can only drop non-zeros, never add them. Nothing actually
        // clamps to zero on this fixture, since every gene carries a
        // substantial count; `test_denoised_chunk_drops_the_clamped_zeros`
        // covers the clamping path directly.
        let total_nnz: usize = run.nnz.iter().sum();
        let raw_nnz: usize = reader
            .read_cells_parallel(&(0..50).collect::<Vec<usize>>())
            .unwrap()
            .iter()
            .map(|c| c.indices.len())
            .sum();
        assert!(total_nnz <= raw_nnz, "denoising invented non-zeros");
    }

    #[test]
    fn test_run_cellsweep_writes_samples_back_to_back() {
        let raw = TempBin::new("multi_raw");
        let out = TempBin::new("multi_out");
        let (reader, labels) = write_fixture_store(raw.path(), 25, 80);

        // Split the 50 real barcodes across two samples, each with its own
        // 40 empty droplets.
        let samples = vec![
            CellSweepSample {
                sample_id: "s1".to_string(),
                real_cells: (0..25).collect(),
                empty_cells: (50..90).collect(),
                celltype_idx: labels[..25].to_vec(),
                n_celltypes: 2,
            },
            CellSweepSample {
                sample_id: "s2".to_string(),
                real_cells: (25..50).collect(),
                empty_cells: (90..130).collect(),
                celltype_idx: labels[25..].to_vec(),
                n_celltypes: 2,
            },
        ];

        let params = CellSweepParams {
            max_iter: 100,
            ..Default::default()
        };
        let run = run_cellsweep(&reader, &samples, params, out.path(), 1e4, 0).unwrap();

        assert_eq!(run.fits.len(), 2);
        assert_eq!(run.cell_order, (0..50).collect::<Vec<usize>>());

        // Each sample got its own ambient profile and its own beta.
        assert_eq!(run.fits[0].ambient.len(), FIXTURE_GENES);
        assert_eq!(run.fits[1].ambient.len(), FIXTURE_GENES);
        assert_eq!(run.fits[0].alpha.len(), 25);
        assert_eq!(run.fits[1].alpha.len(), 25);

        let denoised_reader = ParallelSparseReader::new(out.path()).expect("output opens");
        assert_eq!(denoised_reader.get_header().total_cells, 50);
        // Output indices are the new store's row numbers, not the parent's.
        for (i, chunk) in denoised_reader.get_all_cells().unwrap().iter().enumerate() {
            assert_eq!(chunk.original_index, i);
        }
    }

    // -- guards --

    #[test]
    fn test_run_cellsweep_rejects_unfreezing_the_empties() {
        let raw = TempBin::new("guard_freeze");
        let out = TempBin::new("guard_freeze_out");
        let (reader, labels) = write_fixture_store(raw.path(), 5, 40);

        let sample = CellSweepSample {
            sample_id: "s1".to_string(),
            real_cells: (0..10).collect(),
            empty_cells: (10..50).collect(),
            celltype_idx: labels,
            n_celltypes: 2,
        };
        let params = CellSweepParams {
            freeze_empties: false,
            ..Default::default()
        };

        assert!(matches!(
            run_cellsweep(&reader, &[sample], params, out.path(), 1e4, 0),
            Err(BixverseErrors::CellSweepFreezeEmptiesUnsupported)
        ));
    }

    #[test]
    fn test_build_sample_csr_rejects_too_few_empty_droplets() {
        let raw = TempBin::new("guard_empties");
        let (reader, labels) = write_fixture_store(raw.path(), 5, 40);

        let sample = CellSweepSample {
            sample_id: "s1".to_string(),
            real_cells: (0..10).collect(),
            empty_cells: (10..15).collect(),
            celltype_idx: labels,
            n_celltypes: 2,
        };

        assert!(matches!(
            build_sample_csr(&reader, &sample, FIXTURE_GENES, &CellSweepParams::default()),
            Err(BixverseErrors::CellSweepTooFewEmptyDroplets { found: 5, .. })
        ));
    }

    #[test]
    fn test_build_sample_csr_rejects_bad_labels() {
        let raw = TempBin::new("guard_labels");
        let (reader, _) = write_fixture_store(raw.path(), 5, 40);
        let params = CellSweepParams::default();

        let out_of_range = CellSweepSample {
            sample_id: "s1".to_string(),
            real_cells: (0..2).collect(),
            empty_cells: (10..50).collect(),
            celltype_idx: vec![0, 9],
            n_celltypes: 2,
        };
        assert!(matches!(
            build_sample_csr(&reader, &out_of_range, FIXTURE_GENES, &params),
            Err(BixverseErrors::CellSweepCelltypeOutOfRange { code: 9, .. })
        ));

        let wrong_length = CellSweepSample {
            sample_id: "s1".to_string(),
            real_cells: (0..3).collect(),
            empty_cells: (10..50).collect(),
            celltype_idx: vec![0, 1],
            n_celltypes: 2,
        };
        assert!(matches!(
            build_sample_csr(&reader, &wrong_length, FIXTURE_GENES, &params),
            Err(BixverseErrors::CellSweepLabelLengthMismatch { .. })
        ));

        let no_cells = CellSweepSample {
            sample_id: "s1".to_string(),
            real_cells: Vec::new(),
            empty_cells: (10..50).collect(),
            celltype_idx: Vec::new(),
            n_celltypes: 2,
        };
        assert!(matches!(
            build_sample_csr(&reader, &no_cells, FIXTURE_GENES, &params),
            Err(BixverseErrors::CellSweepNoRealCells { .. })
        ));
    }

    #[test]
    fn test_parse_empty_droplet_call_rejects_nonsense() {
        assert_eq!(
            parse_empty_droplet_call("knee"),
            Some(EmptyDropletCall::Knee)
        );
        assert_eq!(
            parse_empty_droplet_call("supplied"),
            Some(EmptyDropletCall::Supplied)
        );
        assert!(parse_empty_droplet_call("emptyDrops").is_none());
    }
}
