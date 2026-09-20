//! Streaming stages of scTransform v2 over the gene-major store.
//!
//! Every stage here is a per-gene reduction. A gene's contribution needs that
//! gene's counts and the shared per-cell library sizes, nothing from any other
//! gene, so the whole of scTransform runs in
//! `n_threads * n_cells * size_of::<f64>()` plus a handful of gene-length
//! vectors, whatever the dataset size. That is the difference from the R
//! implementation, which preallocates a dense genes-by-cells residual matrix
//! before it starts.
//!
//! The one stage that is not linear in the data is the step-1 fit, and it is
//! bounded by construction: 2000 genes by 2000 cells regardless of the input.

use half::f16;
use indexmap::IndexSet;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::time::Instant;

use crate::core::base::kernel_smooth::bw_nrd;
use crate::core::math::vector_helpers::{interp_linear_at, median};
use crate::errors::BixverseErrors;
use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CscGeneChunk, RawCounts, SingleCellReading,
};

use crate::single_cell::sc_processing::residuals::{
    ResidualSource, distinct_cell_set, intersect_gene_sets, residual_variance, validate_groups,
};

use super::model::{
    SctCellContext, SctCovariates, SctGeneStats, SctModel, SctParams, min_variance_from_umi_median,
    regularise_sct_model,
};
use super::nb_fit::{NbOffsetFit, fit_nb_offset_gene};
use super::residuals::SctResiduals;

////////////
// Consts //
////////////

/// Genes read from disk per batch during the full-gene passes.
///
/// Matches the HVG sweep's block size. Large enough that the per-batch
/// dispatch is amortised, small enough that a batch of decompressed chunks
/// stays well inside cache-friendly territory.
const SCT_GENE_BATCH_SIZE: usize = 1000;

/// Grid resolution for the density estimate driving step-1 gene sampling.
///
/// R's `density()` default. The sampling weight is one over the density at
/// each gene's abundance, so a coarse grid would flatten the weighting exactly
/// where the abundance spectrum is sparsest, which is the part it exists to
/// upweight.
const DENSITY_GRID_POINTS: usize = 512;

/// Grid padding either side of the data, in bandwidths.
///
/// R's `density()` spans `min(x) - cut * bw` to `max(x) + cut * bw` and its
/// `cut` default is 3, so this has to be 3 to lay the 512 points over the same
/// interval. There is no kernel truncation here: the sum below runs over every
/// point, where R uses a binned FFT.
const DENSITY_CUTOFF: f64 = 3.0;

/////////////
// Options //
/////////////

/// Disk and reporting knobs for the streaming stages.
#[derive(Clone, Copy, Debug)]
pub struct SctStreamOpts {
    /// Genes read per disk batch. `None` reads every gene in one go.
    pub gene_batch_size: Option<usize>,
    /// Seed for the step-1 gene and cell subsampling.
    pub seed: u64,
    /// `0` silent, `1` normal, `2` detailed.
    pub verbose: usize,
}

impl Default for SctStreamOpts {
    fn default() -> Self {
        Self {
            gene_batch_size: Some(SCT_GENE_BATCH_SIZE),
            seed: 42,
            verbose: 0,
        }
    }
}

/////////////
// Helpers //
/////////////

/// Corrects one gene's counts to the median library size.
///
/// ### Params
///
/// * `chunk` - The gene's raw counts, reindexed to the selected cells.
/// * `gene_pos` - Position of this gene within the shared gene axis.
/// * `n_cells` - Number of cells represented.
/// * `source` - The fitted models and the per-cell group map.
/// * `median_log10_umi` - Median of `log10_umi` over every selected cell, the
///   library size everything is corrected to.
/// * `median_covariates` - Median of each covariate, what they are held at.
///
/// ### Returns
///
/// The corrected gene chunk, with zeros dropped and `original_index` set to the
/// gene's position on the shared axis rather than in the source store, since
/// the output's gene axis is the modelled set.
fn correct_one_gene(
    chunk: &CscGeneChunk,
    gene_pos: usize,
    n_cells: usize,
    source: &SctResiduals<'_>,
    median_log10_umi: f64,
    median_covariates: &[f64],
) -> CscGeneChunk {
    let cells = source.cells();
    let simple = cells.covariates.is_empty();

    // One set of parameters per group, resolved once per gene. The target
    // predictor holds every latent variable at its median and so is constant
    // across the cells of a group, but not across groups: each sample keeps its
    // own intercept, which is the sample-level expression difference the
    // correction is not trying to remove.
    let per_group: Vec<CorrectionParams<'_>> = source
        .models()
        .iter()
        .enumerate()
        .map(|(group, model)| {
            let pos = source.model_gene_position(group, gene_pos);
            let beta = model.coefficients_for(pos);
            let theta = model.theta[pos];
            let mut eta_target = beta[0] + model.log_umi_coef * median_log10_umi;
            for (b, x) in beta[1..].iter().zip(median_covariates) {
                eta_target += b * x;
            }
            let mu_target = eta_target.exp();
            CorrectionParams {
                beta,
                theta,
                log_umi_coef: model.log_umi_coef,
                mu_target,
                sd_target: (mu_target + mu_target * mu_target / theta).sqrt(),
            }
        })
        .collect();

    let mut counts = vec![0.0_f64; n_cells];
    match &chunk.data_raw {
        RawCounts::U16(v) => {
            for (&i, &x) in chunk.indices.iter().zip(v.iter()) {
                counts[i as usize] = x as f64;
            }
        }
        RawCounts::U32(v) => {
            for (&i, &x) in chunk.indices.iter().zip(v.iter()) {
                counts[i as usize] = x as f64;
            }
        }
    }

    let groups = source.group_of_cell_slice();
    let mut raw = Vec::new();
    let mut indices = Vec::new();
    let mut norms = Vec::new();

    for (c, &y) in counts.iter().enumerate() {
        let p = &per_group[groups[c] as usize];
        let mut eta = p.beta[0] + p.log_umi_coef * cells.log10_umi[c];
        if !simple {
            for (b, x) in p.beta[1..].iter().zip(cells.covariates.row(c)) {
                eta += b * x;
            }
        }
        let mu = eta.exp();
        let residual = (y - mu) / (mu + mu * mu / p.theta).sqrt();
        // R's `round()` is round-half-to-even, not half-away-from-zero.
        let corrected = (p.mu_target + residual * p.sd_target).round_ties_even();

        if corrected >= 1.0 {
            let value = corrected as u32;
            raw.push(value);
            indices.push(c);
            norms.push(F16::from(f16::from_f32((value as f32).ln_1p())));
        }
    }

    CscGeneChunk::from_conversion(
        RawCounts::from_u32_auto(&raw),
        &norms,
        &indices,
        gene_pos,
        true,
    )
}

/// One group's parameters for the count correction of a single gene.
///
/// Distinct from
/// [`SctGeneParams`](super::model::SctGeneParams) because the correction
/// deliberately drops the clipping and the variance floor, and carries the
/// target-depth terms instead.
struct CorrectionParams<'a> {
    /// Coefficients, intercept first.
    beta: &'a [f64],
    /// Inverse overdispersion.
    theta: f64,
    /// Coefficient on `log10(total UMI)`.
    log_umi_coef: f64,
    /// Fitted mean at the median library size.
    mu_target: f64,
    /// Standard deviation at that mean.
    sd_target: f64,
}

/// Overrides the fitted theta with infinity for genes that are really Poisson.
///
/// v2's own check, after the fit rather than before it. The likelihood fit can
/// run theta off to a very large but finite value, which is the boundary in all
/// but name: at that point `mu^2 / theta` has vanished and the gene is Poisson.
/// The method-of-moments estimate `amean^2 / (var - amean)` supplies the scale
/// to judge "very large" against, since what counts as large depends on the
/// gene's abundance. A ratio below the threshold means the fit has gone orders
/// of magnitude past what the second moment supports, so theta is pinned at
/// infinity outright.
///
/// The comparison is a plain ratio so the degenerate cases fall out the way R's
/// do. A gene with `var <= amean` gives a negative moment estimate and a
/// negative ratio, which is below any positive threshold and flags, and that is
/// the right answer for a gene with no excess variance at all. A gene already
/// at infinity gives a ratio of zero and stays there.
///
/// ### Params
///
/// * `fits` - The step-1 fits, consumed.
/// * `fit_idx` - Store gene index of each fit, parallel to `fits`.
/// * `stats` - Per-gene statistics over the full cell set, by store index.
/// * `threshold` - Ratio below which the gene is declared Poisson.
///
/// ### Returns
///
/// The fits, with flagged genes carrying an infinite theta.
fn flag_poisson_genes(
    fits: Vec<NbOffsetFit>,
    fit_idx: &[usize],
    stats: &SctGeneStats,
    threshold: f64,
) -> Vec<NbOffsetFit> {
    fits.into_iter()
        .zip(fit_idx.iter())
        .map(|(fit, &g)| {
            let amean = stats.amean[g];
            let moment_theta = amean * amean / (stats.var[g] - amean);
            if moment_theta / fit.theta < threshold {
                NbOffsetFit {
                    theta: f64::INFINITY,
                    ..fit
                }
            } else {
                fit
            }
        })
        .collect()
}

/// Counts how many of a cell subset each gene is detected in.
///
/// ### Params
///
/// * `reader` - Gene-major reader.
/// * `genes` - Store indices to count.
/// * `cell_set` - Cells to count within.
/// * `opts` - Disk knobs.
///
/// ### Returns
///
/// Detection counts indexed by **store** gene index, zero for genes not asked
/// about.
fn count_detected<S: SingleCellReading>(
    reader: &S,
    genes: &[usize],
    cell_set: &IndexSet<u32>,
    opts: SctStreamOpts,
) -> Result<Vec<usize>, BixverseErrors> {
    let n_genes = reader.get_header().total_genes;
    let mut out = vec![0_usize; n_genes];
    let step = opts.gene_batch_size.unwrap_or(genes.len()).max(1);

    for block in genes.chunks(step) {
        for chunk in reader.read_gene_parallel_filtered(block, cell_set)? {
            out[chunk.original_index] = chunk.indices.len();
        }
    }

    Ok(out)
}

/// Expands a gene chunk into a dense per-cell vector.
///
/// ### Params
///
/// * `chunk` - Gene chunk, already reindexed to the cell subset.
/// * `n_cells` - Length of the output.
///
/// ### Returns
///
/// The dense counts, zeros included.
fn densify(chunk: &CscGeneChunk, n_cells: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_cells];
    match &chunk.data_raw {
        RawCounts::U16(v) => {
            for (&i, &x) in chunk.indices.iter().zip(v.iter()) {
                out[i as usize] = x as f64;
            }
        }
        RawCounts::U32(v) => {
            for (&i, &x) in chunk.indices.iter().zip(v.iter()) {
                out[i as usize] = x as f64;
            }
        }
    }
    out
}

///////////////
// Gene pass //
///////////////

/// Everything the model fit needs from one pass over the gene-major store.
#[derive(Clone, Debug)]
pub struct SctGenePass {
    /// Per-gene statistics for every gene in the store, in store gene order.
    pub stats: SctGeneStats,
    /// Cells each gene is detected in, over the selected cells.
    pub detected: Vec<usize>,
    /// Store indices of the genes passing the `min_cells` filter, ascending.
    /// These are the genes the model covers.
    pub modelled: Vec<usize>,
    /// Median of every non-zero count across the modelled genes.
    pub median_nonzero: f64,
    /// Mean library size over the modelled genes, for the Poisson genes'
    /// closed-form intercept.
    pub mean_cell_sum: f64,
}

/// Per-gene accumulator for the sweep.
#[derive(Clone, Debug, Default)]
struct GeneAcc {
    /// Sum of the counts.
    sum: f64,
    /// Sum of squared deviations from the mean, zeros folded in analytically.
    ss: f64,
    /// Sum of `ln(count + eps)` over the non-zero entries.
    sum_log: f64,
    /// Number of non-zero entries.
    nnz: usize,
}

/// Sweeps one gene chunk into its accumulator.
///
/// Two passes over the gene's stored non-zeros, which is what R's
/// `row_var_dgcmatrix` does: the mean first, then the squared deviations from
/// it. The raw-moment form would be one pass cheaper and cancels badly on a
/// gene whose counts are large and tightly clustered, which is exactly the
/// highly expressed genes the regularisation leans on most.
///
/// ### Params
///
/// * `gene` - The gene chunk, already reindexed to the selected cells.
///
/// ### Returns
///
/// The accumulator.
fn sweep_gene(gene: &CscGeneChunk, n_cells: usize, gmean_eps: f64) -> GeneAcc {
    let values: Vec<f64> = match &gene.data_raw {
        RawCounts::U16(v) => v.iter().map(|&x| x as f64).collect(),
        RawCounts::U32(v) => v.iter().map(|&x| x as f64).collect(),
    };

    let nnz = values.len();
    let sum: f64 = values.iter().sum();
    let mean = sum / n_cells as f64;

    let ss_nonzero: f64 = values.iter().map(|&x| (x - mean) * (x - mean)).sum();
    let ss = ss_nonzero + mean * mean * (n_cells - nnz) as f64;

    let sum_log: f64 = values.iter().map(|&x| (x + gmean_eps).ln()).sum();

    GeneAcc {
        sum,
        ss,
        sum_log,
        nnz,
    }
}

/// One pass over every gene, producing the statistics the model fit needs.
///
/// Also decides the `min_cells` filter and, for the genes that pass it, folds
/// their counts into the global non-zero histogram and running total. Doing
/// both inside the sweep keeps this to a single read of the store: a gene's
/// eligibility depends only on its own detection count, which is available the
/// moment that gene is in hand.
///
/// ### Params
///
/// * `reader` - Gene-major reader.
/// * `cell_indices` - Cells to include, in the order results are indexed by.
/// * `params` - Tuning knobs; uses `min_cells` and `gmean_eps`.
/// * `opts` - Disk and reporting knobs.
///
/// ### Returns
///
/// The [`SctGenePass`], or a [`BixverseErrors`] from the reader.
pub fn sct_gene_pass<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    params: &SctParams,
    opts: SctStreamOpts,
) -> Result<SctGenePass, BixverseErrors> {
    if !reader.is_gene_based() {
        return Err(BixverseErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }
    if cell_indices.is_empty() {
        return Err(BixverseErrors::LengthMismatch {
            name: "cell_indices",
            expected: 1,
            found: 0,
        });
    }

    let verbosity = parse_verbosity_level(opts.verbose);
    let start = Instant::now();

    let n_cells = cell_indices.len();
    let n_genes = reader.get_header().total_genes;
    let cell_set = distinct_cell_set(cell_indices)?;
    let step = opts.gene_batch_size.unwrap_or(n_genes).max(1);

    let mut log_gmean = vec![0.0_f64; n_genes];
    let mut amean = vec![0.0_f64; n_genes];
    let mut var = vec![0.0_f64; n_genes];
    let mut detected = vec![0_usize; n_genes];

    let mut histogram: FxHashMap<u32, u64> = FxHashMap::default();
    let mut total_counts = 0.0_f64;

    for block_start in (0..n_genes).step_by(step) {
        let block_end = (block_start + step).min(n_genes);
        let block: Vec<usize> = (block_start..block_end).collect();
        let chunks = reader.read_gene_parallel_filtered(&block, &cell_set)?;

        // The per-gene work is independent, so fan out; the histogram folds in
        // sequentially afterwards, which is cheap relative to the sweep.
        let accs: Vec<GeneAcc> = chunks
            .par_iter()
            .map(|c| sweep_gene(c, n_cells, params.gmean_eps))
            .collect();

        for (chunk, acc) in chunks.iter().zip(accs.iter()) {
            let g = chunk.original_index;
            let n_f = n_cells as f64;

            amean[g] = acc.sum / n_f;
            // R's row_var is the sample variance, n - 1 in the denominator.
            var[g] = if n_cells > 1 {
                acc.ss / (n_f - 1.0)
            } else {
                0.0
            };
            // R: exp((sum(log(x + eps)) + log(eps) * n_zero) / n) - eps
            let gmean = ((acc.sum_log + params.gmean_eps.ln() * (n_cells - acc.nnz) as f64) / n_f)
                .exp()
                - params.gmean_eps;
            log_gmean[g] = gmean.log10();
            detected[g] = acc.nnz;

            if acc.nnz >= params.min_cells {
                total_counts += acc.sum;
                match &chunk.data_raw {
                    RawCounts::U16(v) => {
                        for &x in v {
                            *histogram.entry(x as u32).or_insert(0) += 1;
                        }
                    }
                    RawCounts::U32(v) => {
                        for &x in v {
                            *histogram.entry(x).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        if verbosity.detailed_verbosity() {
            report_decile_progress(block_end, block_start, n_genes, "genes", start.elapsed());
        }
    }

    let modelled: Vec<usize> = (0..n_genes)
        .filter(|&g| detected[g] >= params.min_cells)
        .collect();

    if modelled.is_empty() {
        return Err(BixverseErrors::SctNoGenesPassFilter {
            min_cells: params.min_cells,
            n_genes,
        });
    }

    let median_nonzero = histogram_median(&histogram);
    let mean_cell_sum = total_counts / n_cells as f64;

    if verbosity.normal_verbosity() {
        println!(
            "scTransform: swept {n_genes} genes in {:.2?}, {} pass the min_cells filter",
            start.elapsed(),
            modelled.len()
        );
    }

    Ok(SctGenePass {
        stats: SctGeneStats {
            log_gmean,
            amean,
            var,
        },
        detected,
        modelled,
        median_nonzero,
        mean_cell_sum,
    })
}

/// Median of the values a histogram describes, with R's even-length rule.
///
/// The counts are small integers, so a frequency table is exact and costs a
/// sort over the distinct values rather than over the billions of entries.
///
/// ### Params
///
/// * `histogram` - Value to frequency.
///
/// ### Returns
///
/// The median, or `0.0` for an empty histogram.
fn histogram_median(histogram: &FxHashMap<u32, u64>) -> f64 {
    let total: u64 = histogram.values().sum();
    if total == 0 {
        return 0.0;
    }

    let mut values: Vec<(u32, u64)> = histogram.iter().map(|(&v, &f)| (v, f)).collect();
    values.sort_unstable_by_key(|&(v, _)| v);

    // R averages the two central order statistics on an even count.
    let want = if total.is_multiple_of(2) {
        vec![total / 2 - 1, total / 2]
    } else {
        vec![total / 2]
    };

    let mut found = Vec::with_capacity(want.len());
    let mut seen = 0_u64;
    let mut wi = 0_usize;
    for &(value, freq) in &values {
        seen += freq;
        while wi < want.len() && want[wi] < seen {
            found.push(value as f64);
            wi += 1;
        }
        if wi == want.len() {
            break;
        }
    }

    found.iter().sum::<f64>() / found.len() as f64
}

//////////////////////
// Step-1 selection //
//////////////////////

/// Picks the step-1 subsample.
///
/// Cells are a uniform sample. Genes are filtered to those detected in at
/// least `min_cells` of the sampled cells and showing excess variance over
/// Poisson, then sampled with weight inversely proportional to the density of
/// `log10(geometric mean)`. That weighting is the point: a uniform sample would
/// be dominated by the crowded middle of the abundance spectrum and leave the
/// regularisation curve unconstrained at both ends, which is where it has to
/// extrapolate furthest.
///
/// ### Params
///
/// * `pass` - Output of [`sct_gene_pass`].
/// * `detected_step1` - Cells each gene is detected in among the sampled cells.
/// * `n_cells_total` - Number of cells available to sample from.
/// * `params` - Tuning knobs.
/// * `seed` - RNG seed.
///
/// ### Returns
///
/// The selection, or a [`BixverseErrors`] when nothing survives the filters.
fn select_step1_genes(
    pass: &SctGenePass,
    detected_step1: &[usize],
    params: &SctParams,
    rng: &mut StdRng,
) -> Result<Vec<usize>, BixverseErrors> {
    let eligible: Vec<usize> = pass
        .modelled
        .iter()
        .copied()
        .filter(|&g| {
            detected_step1[g] >= params.min_cells
                && pass.stats.var[g] - pass.stats.amean[g] > 0.0
                && pass.stats.log_gmean[g].is_finite()
        })
        .collect();

    if eligible.len() < 2 {
        return Err(BixverseErrors::SctTooFewGenesToRegularise {
            kept: eligible.len(),
            total: pass.modelled.len(),
        });
    }

    if eligible.len() <= params.n_genes {
        return Ok(eligible);
    }

    let x: Vec<f64> = eligible.iter().map(|&g| pass.stats.log_gmean[g]).collect();
    let weights = inverse_density_weights(&x)?;

    // Efraimidis-Spirakis: drawing `u^(1/w)` and taking the largest keys is a
    // weighted sample without replacement in one pass, no rejection loop.
    let mut keyed: Vec<(f64, usize)> = eligible
        .iter()
        .zip(weights.iter())
        .map(|(&g, &w)| {
            let u: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
            (u.powf(1.0 / w), g)
        })
        .collect();
    keyed.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut picked: Vec<usize> = keyed
        .into_iter()
        .take(params.n_genes)
        .map(|(_, g)| g)
        .collect();
    picked.sort_unstable();

    Ok(picked)
}

/// The interval R's `density()` lays its grid over.
///
/// ### Params
///
/// * `lo` - Smallest value in the data.
/// * `hi` - Largest value in the data.
/// * `bw` - The bandwidth.
///
/// ### Returns
///
/// The `(from, to)` span.
fn density_grid_span(lo: f64, hi: f64, bw: f64) -> (f64, f64) {
    (lo - DENSITY_CUTOFF * bw, hi + DENSITY_CUTOFF * bw)
}

/// Sampling weights inversely proportional to the density of `x`.
///
/// A Gaussian kernel density estimate on R's `bw.nrd` bandwidth, evaluated on
/// a grid and interpolated back, matching `density(bw = 'nrd')` followed by
/// `approx()`. Epsilon is added before inverting so a point in an empty
/// stretch of the spectrum gets a large weight rather than an infinite one.
///
/// ### Params
///
/// * `x` - The abundances to weight.
///
/// ### Returns
///
/// One weight per entry of `x`, or a [`BixverseErrors`] when the bandwidth
/// cannot be estimated.
fn inverse_density_weights(x: &[f64]) -> Result<Vec<f64>, BixverseErrors> {
    let bw = bw_nrd(x)?;
    let (lo, hi) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });

    let (from, to) = density_grid_span(lo, hi, bw);
    let step = (to - from) / (DENSITY_GRID_POINTS - 1) as f64;

    let grid: Vec<f64> = (0..DENSITY_GRID_POINTS)
        .map(|i| from + i as f64 * step)
        .collect();
    let norm = 1.0 / (x.len() as f64 * bw * (2.0 * std::f64::consts::PI).sqrt());
    let dens: Vec<f64> = grid
        .par_iter()
        .map(|&g| {
            norm * x
                .iter()
                .map(|&v| {
                    let u = (v - g) / bw;
                    (-0.5 * u * u).exp()
                })
                .sum::<f64>()
        })
        .collect();

    Ok(x.iter()
        .map(|&v| 1.0 / (interp_linear_at(&grid, &dens, v) + f64::EPSILON))
        .collect())
}

///////////////
// Model fit //
///////////////

/// Fits and regularises the scTransform v2 model.
///
/// Three stages: one pass over the store for the per-gene statistics, an
/// independent negative binomial fit on the step-1 subsample, and the
/// regularisation that carries those fits to every gene. Only the first is
/// linear in the data, and it holds nothing per cell beyond the gene in flight.
///
/// ### Params
///
/// * `reader` - Gene-major reader.
/// * `cell_indices` - Cells to model over.
/// * `library_sizes` - Total UMI per cell over the **whole** store, in
///   `cell_indices` order. Read cheaply from the cell-major store's chunk
///   headers via [`SingleCellReading::read_cell_library_sizes`]. This is the
///   offset, and sctransform takes it before any gene filtering, so it must not
///   be recomputed from a gene subset.
/// * `covariates` - Cell-level covariates to regress out alongside the library
///   size, sctransform's `latent_var` beyond `log_umi`. Pass
///   [`SctCovariates::default`] for the usual depth-only model.
/// * `params` - Tuning knobs.
/// * `step1_genes` - Store indices to fit on, overriding the sampling. Exists
///   so a parity fixture can pin R's own random subset, which no RNG here can
///   reproduce.
/// * `step1_cells` - Positions within `cell_indices` to fit on, same purpose.
/// * `opts` - Disk and reporting knobs.
///
/// ### Returns
///
/// The fitted [`SctModel`] alongside the [`SctGenePass`] it was built from, so
/// the caller can reuse the statistics rather than sweeping again.
#[allow(clippy::too_many_arguments)]
pub fn fit_sctransform<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    library_sizes: &[f64],
    covariates: &SctCovariates,
    params: &SctParams,
    step1_genes: Option<&[usize]>,
    step1_cells: Option<&[usize]>,
    opts: SctStreamOpts,
) -> Result<(SctModel, SctGenePass), BixverseErrors> {
    if library_sizes.len() != cell_indices.len() {
        return Err(BixverseErrors::LengthMismatch {
            name: "library_sizes",
            expected: cell_indices.len(),
            found: library_sizes.len(),
        });
    }
    covariates.validate(cell_indices.len())?;

    let verbosity = parse_verbosity_level(opts.verbose);
    let pass = sct_gene_pass(reader, cell_indices, params, opts)?;

    let mut rng = StdRng::seed_from_u64(opts.seed);
    let n_cells = cell_indices.len();

    let cells: Vec<usize> = match step1_cells {
        Some(c) => c.to_vec(),
        None if params.n_cells >= n_cells => (0..n_cells).collect(),
        None => {
            let mut idx: Vec<usize> = (0..n_cells).collect();
            // Partial Fisher-Yates: only the prefix that is kept gets shuffled.
            for i in 0..params.n_cells {
                let j = rng.random_range(i..n_cells);
                idx.swap(i, j);
            }
            let mut c = idx[..params.n_cells].to_vec();
            c.sort_unstable();
            c
        }
    };

    if let Some(&bad) = cells.iter().find(|&&c| c >= n_cells) {
        return Err(BixverseErrors::SctCellIndexOutOfRange {
            index: bad,
            n_cells,
        });
    }

    // Detection within the step-1 cells decides gene eligibility, so it needs
    // its own count. One extra read of the sampled cells only.
    let step1_cells_global: Vec<usize> = cells.iter().map(|&c| cell_indices[c]).collect();
    let step1_cell_set = distinct_cell_set(&step1_cells_global)?;

    let genes = match step1_genes {
        Some(g) => g.to_vec(),
        None => {
            let detected_step1 = count_detected(reader, &pass.modelled, &step1_cell_set, opts)?;
            select_step1_genes(&pass, &detected_step1, params, &mut rng)?
        }
    };

    if verbosity.normal_verbosity() {
        println!(
            "scTransform: step-1 fit on {} genes by {} cells",
            genes.len(),
            cells.len()
        );
    }

    let start_fit = Instant::now();
    let log_offset: Vec<f64> = cells.iter().map(|&c| library_sizes[c].ln()).collect();

    // The design is built over the sampled cells only, so its rows have to be
    // gathered rather than sliced: `cells` indexes into the full cell set.
    let n_coef = covariates.n_covariates() + 1;
    let mut design = Vec::with_capacity(cells.len() * n_coef);
    for &c in &cells {
        design.push(1.0);
        design.extend_from_slice(covariates.row(c));
    }

    let chunks = reader.read_gene_parallel_filtered(&genes, &step1_cell_set)?;
    let fits: Vec<NbOffsetFit> = chunks
        .par_iter()
        .map(|chunk| {
            let dense = densify(chunk, cells.len());
            fit_nb_offset_gene(&dense, &log_offset, &design, n_coef)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // `read_gene_parallel_filtered` preserves the requested order, but the
    // regularisation indexes the model by store position, so pair them up
    // explicitly rather than trusting that.
    let fit_idx: Vec<usize> = chunks.iter().map(|c| c.original_index).collect();

    let fits = flag_poisson_genes(fits, &fit_idx, &pass.stats, params.poisson_diff_theta);

    if verbosity.normal_verbosity() {
        println!(
            "scTransform: fitted {} genes with {} coefficient(s) in {:.2?}",
            fits.len(),
            n_coef,
            start_fit.elapsed()
        );
    }

    // The regularisation works over the modelled genes only, so translate store
    // indices into positions within `pass.modelled`.
    let position: FxHashMap<usize, usize> = pass
        .modelled
        .iter()
        .enumerate()
        .map(|(pos, &g)| (g, pos))
        .collect();

    let mut kept_fits = Vec::with_capacity(fits.len());
    let mut kept_idx = Vec::with_capacity(fits.len());
    for (fit, g) in fits.into_iter().zip(fit_idx.iter()) {
        if let Some(&pos) = position.get(g) {
            kept_fits.push(fit);
            kept_idx.push(pos);
        }
    }

    let sub_stats = SctGeneStats {
        log_gmean: pass
            .modelled
            .iter()
            .map(|&g| pass.stats.log_gmean[g])
            .collect(),
        amean: pass.modelled.iter().map(|&g| pass.stats.amean[g]).collect(),
        var: pass.modelled.iter().map(|&g| pass.stats.var[g]).collect(),
    };

    let mut model = regularise_sct_model(
        &kept_fits,
        &kept_idx,
        &sub_stats,
        &covariates.names,
        pass.mean_cell_sum,
        min_variance_from_umi_median(pass.median_nonzero),
        n_cells,
        params,
    )?;
    // The regularisation works positionally; hand the model the store indices
    // so a downstream caller holding an HVG set can look genes up directly.
    model.genes = pass.modelled.clone();

    Ok((model, pass))
}

///////////////////////
// Residual variance //
///////////////////////

/// Per-gene variance of the Pearson residuals, streaming.
///
/// This is sctransform's `gene_attr$residual_variance`, and it is what
/// scTransform uses to rank genes for feature selection: a gene whose residuals
/// still vary after the mean-variance relationship has been regressed out is
/// carrying signal the model does not explain.
///
/// The residual row is dense, so it is generated one gene at a time and reduced
/// to a scalar immediately. Peak memory is one row per worker.
///
/// ### Params
///
/// * `reader` - Gene-major reader.
/// * `model` - The fitted model; its `genes` field decides which genes are
///   visited.
/// * `cell_indices` - Cells to include, in the order `cells` is given in.
/// * `cells` - Per-cell library sizes and covariates.
/// * `opts` - Disk and reporting knobs.
///
/// ### Returns
///
/// One variance per modelled gene, in `model.genes` order, or a
/// [`BixverseErrors`] from the reader.
pub fn sct_residual_variance<S: SingleCellReading>(
    reader: &S,
    model: &SctModel,
    cell_indices: &[usize],
    cells: &SctCellContext<'_>,
    opts: SctStreamOpts,
) -> Result<Vec<f64>, BixverseErrors> {
    let source = SctResiduals::single(model, *cells)?;
    let mut per_group = residual_variance(reader, &source, cell_indices, opts)?;

    // One group by construction, so the single vector is the whole answer and
    // its gene order is `model.genes`.
    Ok(per_group.remove(0))
}

//////////////////////
// Corrected counts //
//////////////////////

/// Writes scTransform-corrected UMI counts as a gene-major store, with every
/// cell placed at the median library size:
///
/// ```text
/// r_c  = (y_c - mu_c) / sqrt(mu_c + mu_c^2 / theta)
/// mu'  = exp(b0 + log_umi_coef * median(log10_umi))
/// y'_c = max(round(mu' + r_c * sqrt(mu' + mu'^2 / theta)), 0)
/// ```
///
/// Unlike [`super::model::sct_residual_row`], the residual is neither clipped
/// nor floored by `min_variance` here, matching sctransform's `correct_counts`.
///
/// The normalised layer is `ln(1 + y')` with no library normalisation, and
/// `target_size` is left at the "unknown" sentinel.
///
/// The output's gene axis is `source.genes()`, not the store's: genes without
/// a fitted model are dropped rather than written as zero rows.
///
/// ### Params
///
/// * `reader` - Gene-major reader over the raw counts.
/// * `source` - The fitted models; its gene axis decides what is written.
/// * `cell_indices` - Cells to include, in the order `source` was built in.
/// * `out_path` - Gene-major store to write.
/// * `opts` - Disk and reporting knobs.
///
/// ### References
///
/// Hafemeister & Satija, Genome Biology, 2019, `correct_counts`
pub fn sct_corrected_counts<S: SingleCellReading, P: AsRef<Path>>(
    reader: &S,
    source: &SctResiduals<'_>,
    cell_indices: &[usize],
    out_path: P,
    opts: SctStreamOpts,
) -> Result<(), BixverseErrors> {
    if source.n_cells() != cell_indices.len() {
        return Err(BixverseErrors::LengthMismatch {
            name: "cell_indices",
            expected: source.n_cells(),
            found: cell_indices.len(),
        });
    }

    let verbosity = parse_verbosity_level(opts.verbose);
    let start = Instant::now();

    let cells = source.cells();
    let genes = source.genes();
    let n_genes = genes.len();
    let n_cells = cell_indices.len();
    let cell_set = distinct_cell_set(cell_indices)?;

    // Every latent variable is held at its median, the library size included,
    // so the target linear predictor is constant across the cells of a group.
    let median_log10_umi = median(cells.log10_umi).unwrap_or(0.0);
    let median_covariates = cells.covariates.medians(n_cells);

    // The corrected counts no longer carry library-size structure, so there is
    // no target size for the normalised layer to have been scaled to.
    let mut writer = CellGeneSparseWriter::new(out_path, false, n_cells, n_genes, 0.0)?;

    let step = opts.gene_batch_size.unwrap_or(n_genes).max(1);
    let mut done = 0_usize;

    for block in genes.chunks(step) {
        let chunks = reader.read_gene_parallel_filtered(block, &cell_set)?;

        let corrected: Vec<CscGeneChunk> = chunks
            .par_iter()
            .map(|chunk| {
                let pos = source.position(chunk.original_index).ok_or(
                    BixverseErrors::SctGeneNotModelled {
                        gene: chunk.original_index,
                    },
                )?;
                Ok(correct_one_gene(
                    chunk,
                    pos,
                    n_cells,
                    source,
                    median_log10_umi,
                    &median_covariates,
                ))
            })
            .collect::<Result<Vec<_>, BixverseErrors>>()?;

        for chunk in corrected {
            writer.write_gene_chunk(chunk)?;
        }

        let prev = done;
        done += block.len();
        if verbosity.detailed_verbosity() {
            report_decile_progress(done, prev, n_genes, "genes", start.elapsed());
        }
    }

    writer.finalise()?;

    if verbosity.normal_verbosity() {
        println!(
            "scTransform: wrote {n_genes} corrected genes in {:.2?}",
            start.elapsed()
        );
    }

    Ok(())
}

/////////////////
// Grouped fit //
/////////////////

/// One scTransform model per group, over a shared gene axis.
#[derive(Clone, Debug)]
pub struct SctGroupedFit {
    /// One fitted model per group, in group id order.
    pub models: Vec<SctModel>,
    /// Store gene indices modelled in every group, ascending.
    pub genes: Vec<usize>,
    /// Each group's statistics pass, kept for diagnostics.
    pub passes: Vec<SctGenePass>,
    /// Group id per selected cell, echoed back so a caller can build the
    /// residual source without recomputing it.
    pub group_of_cell: Vec<u32>,
}

/// Fits scTransform independently per group.
///
/// An experiment assembled from several files is several samples, each with its
/// own sequencing depth and composition. Fitting one model across them folds
/// the sample-level depth differences into the gene coefficients, and since the
/// residuals feed HVG selection and then PCA, that propagates all the way to
/// the embedding. Fitting per sample and scoring each cell under its own
/// sample's model is what Seurat v5 does for split layers.
///
/// Two things are shared rather than per group, both deliberately:
///
/// * **The clipping range.** Resolved once against the total selected cell
///   count and pushed into every group fit. Left per group, a small sample
///   would be clipped harder than a large one and the residuals would no longer
///   be on a common scale.
/// * **Nothing else.** The variance floor in particular stays per group: it
///   comes from that sample's median non-zero UMI count, which is a property of
///   its depth.
///
/// The gene axis is the intersection of the per-group modelled sets, Seurat's
/// "present in all layers" rule. A gene that one sample's `min_cells` filter
/// dropped has no model there, so it cannot carry a residual for that sample's
/// cells.
///
/// ### Params
///
/// * `reader` - Gene-major store.
/// * `cell_indices` - Global ids of the selected cells.
/// * `group_of_cell` - Group id per selected cell, densely covering
///   `0..n_groups`.
/// * `library_sizes` - Full library size per selected cell, in the same order.
/// * `covariates` - Cell-level covariates over the selected cells.
/// * `params` - Tuning knobs.
/// * `opts` - Disk batching and reporting.
///
/// ### Returns
///
/// The fit, or a [`BixverseErrors`] when the grouping is malformed, a group
/// fails to fit, or no gene is modelled in every group.
///
/// ### References
///
/// Seurat v5, `SCTransform.StdAssay`
pub fn fit_sctransform_grouped<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    group_of_cell: &[u32],
    library_sizes: &[f64],
    covariates: &SctCovariates,
    params: &SctParams,
    opts: SctStreamOpts,
) -> Result<SctGroupedFit, BixverseErrors> {
    let n_groups = validate_groups(group_of_cell, cell_indices.len())?;
    if library_sizes.len() != cell_indices.len() {
        return Err(BixverseErrors::LengthMismatch {
            name: "library_sizes",
            expected: cell_indices.len(),
            found: library_sizes.len(),
        });
    }
    covariates.validate(cell_indices.len())?;

    let verbosity = parse_verbosity_level(opts.verbose);
    let start = Instant::now();

    let shared_params = SctParams {
        clip_range: Some(params.resolve_clip_range(cell_indices.len())),
        ..*params
    };

    let mut models = Vec::with_capacity(n_groups);
    let mut passes = Vec::with_capacity(n_groups);

    for group in 0..n_groups {
        let slots: Vec<usize> = group_of_cell
            .iter()
            .enumerate()
            .filter(|&(_, &g)| g as usize == group)
            .map(|(slot, _)| slot)
            .collect();

        let group_cells: Vec<usize> = slots.iter().map(|&s| cell_indices[s]).collect();
        let group_sizes: Vec<f64> = slots.iter().map(|&s| library_sizes[s]).collect();
        let group_covariates = covariates.subset(&slots)?;

        // Without this a `SctNoGenesPassFilter` from group 7 of 8 is
        // indistinguishable from one from group 0, and the usual cause is one
        // specific shallow sample.
        let (model, pass) = fit_sctransform(
            reader,
            &group_cells,
            &group_sizes,
            &group_covariates,
            &shared_params,
            None,
            None,
            opts,
        )
        .map_err(|e| BixverseErrors::ResidualGroupFitFailed {
            group,
            reason: e.to_string(),
        })?;

        if verbosity.normal_verbosity() {
            println!(
                "scTransform: group {group} fitted over {} cells, {} genes ({:.2?})",
                group_cells.len(),
                model.len(),
                start.elapsed()
            );
        }

        models.push(model);
        passes.push(pass);
    }

    let sets: Vec<&[usize]> = models.iter().map(|m| m.genes.as_slice()).collect();
    let genes = intersect_gene_sets(&sets)?;

    if verbosity.normal_verbosity() {
        println!(
            "scTransform: {} gene(s) modelled in all {n_groups} group(s)",
            genes.len()
        );
    }

    Ok(SctGroupedFit {
        models,
        genes,
        passes,
        group_of_cell: group_of_cell.to_vec(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn stats(amean: Vec<f64>, var: Vec<f64>) -> SctGeneStats {
        SctGeneStats {
            log_gmean: amean.iter().map(|v| v.log10()).collect(),
            amean,
            var,
        }
    }

    /// A theta that has run far past what the second moment supports is pinned
    /// at infinity rather than left as a large finite number.
    ///
    /// This is v2's own post-fit check and it matters more than it looks:
    /// leaving it out moved the regularised theta a median 5.9e-3 away from
    /// sctransform on the parity fixture, where applying it brings that to
    /// 3.0e-5.
    #[test]
    fn test_flag_poisson_genes_overrides_a_confident_fit() {
        // Both genes have a moment estimate of amean^2 / (var - amean) = 1.
        // Gene 0's fit agrees, so it keeps its theta. Gene 1's fit ran off to
        // 10,000, four orders past what the second moment supports, so it is
        // pinned at infinity.
        let s = stats(vec![10.0, 10.0], vec![110.0, 110.0]);
        let fits = vec![
            NbOffsetFit {
                theta: 1.0,
                coefficients: vec![-5.0],
                converged: true,
            },
            NbOffsetFit {
                theta: 10_000.0,
                coefficients: vec![-5.0],
                converged: true,
            },
        ];

        let out = flag_poisson_genes(fits, &[0, 1], &s, 1e-3);

        assert_relative_eq!(out[0].theta, 1.0, max_relative = 1e-12);
        assert!(out[1].theta.is_infinite());
        // The coefficients are never touched, only the dispersion.
        assert_relative_eq!(out[1].intercept(), -5.0, max_relative = 1e-12);
    }

    /// A gene with no excess variance at all gives a negative moment estimate.
    /// R compares the ratio without guarding the sign, so it flags, which is
    /// the right answer for a gene whose variance is below its mean.
    #[test]
    fn test_flag_poisson_genes_handles_underdispersion() {
        let s = stats(vec![10.0], vec![5.0]);
        let fits = vec![NbOffsetFit {
            theta: 2.0,
            coefficients: vec![-5.0],
            converged: true,
        }];

        let out = flag_poisson_genes(fits, &[0], &s, 1e-3);

        assert!(out[0].theta.is_infinite());
    }

    /// An already-infinite theta stays infinite rather than becoming NaN
    /// through the ratio.
    #[test]
    fn test_flag_poisson_genes_leaves_infinite_theta_alone() {
        let s = stats(vec![10.0], vec![110.0]);
        let fits = vec![NbOffsetFit {
            theta: f64::INFINITY,
            coefficients: vec![-5.0],
            converged: true,
        }];

        let out = flag_poisson_genes(fits, &[0], &s, 1e-3);

        assert!(out[0].theta.is_infinite());
    }

    /// R's `median` averages the two central values on an even count, and the
    /// histogram path has to do the same or `min_variance` drifts.
    #[test]
    fn test_histogram_median_matches_r_on_an_even_count() {
        let mut h: FxHashMap<u32, u64> = FxHashMap::default();
        // 1, 1, 2, 3 -> R's median is 1.5
        h.insert(1, 2);
        h.insert(2, 1);
        h.insert(3, 1);

        assert_relative_eq!(histogram_median(&h), 1.5, max_relative = 1e-12);
    }

    #[test]
    fn test_histogram_median_on_an_odd_count() {
        let mut h: FxHashMap<u32, u64> = FxHashMap::default();
        // 1, 2, 2, 2, 9 -> 2
        h.insert(1, 1);
        h.insert(2, 3);
        h.insert(9, 1);

        assert_relative_eq!(histogram_median(&h), 2.0, max_relative = 1e-12);
    }

    #[test]
    fn test_histogram_median_of_nothing_is_zero() {
        assert_relative_eq!(
            histogram_median(&FxHashMap::default()),
            0.0,
            max_relative = 1e-12
        );
    }

    /// The sampling weight has to be largest where the abundance spectrum is
    /// emptiest, or the regularisation curve is left unconstrained exactly
    /// where it has to extrapolate furthest.
    /// The kernel density has to be R's `density()`, which is what the sampling
    /// weights are defined against.
    ///
    /// This is the only thing that pins [`DENSITY_CUTOFF`]: the parity fixture
    /// is small enough that both step-1 subsamples are skipped, so nothing else
    /// in the suite reaches this function at all. Reference from
    /// `density(x, bw = bw.nrd(x), n = 512)` then `approx()` at the data points,
    /// R 4.5.1.
    ///
    /// The tolerance is loose because R bins the data onto its grid and
    /// convolves with an FFT where this evaluates the kernel sum exactly at each
    /// grid point. The grid span, which is what `DENSITY_CUTOFF` controls, is
    /// identical; only the binning differs.
    // R emits seventeen significant digits; keeping them is the point of a
    // generated reference value.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn test_inverse_density_weights_match_r_density() {
        let x = [
            -2.5, -2.1, -1.9, -1.85, -1.8, -0.4, -0.35, -0.3, 0.0, 0.05, 0.1, 0.12, 0.9, 1.4, 2.2,
            3.1, 3.15, 4.0,
        ];
        let want = [
            1.03458082384436933e-01,
            1.25931894605516459e-01,
            1.36044862074708311e-01,
            1.38444272035506244e-01,
            1.40792693177528960e-01,
            1.82506541618378482e-01,
            1.82751321578175946e-01,
            1.82867864041406952e-01,
            1.80786853199413428e-01,
            1.79965686232037181e-01,
            1.79011125770661916e-01,
            1.78587221821085107e-01,
            1.48949556824229640e-01,
            1.24884348041357951e-01,
            9.70968112444874215e-02,
            7.94907425133158096e-02,
            7.84809737058110718e-02,
            5.60556936261453892e-02,
        ];

        // R's `bw.nrd`, which the grid span is measured in.
        let bw = bw_nrd(&x).unwrap();
        assert_relative_eq!(bw, 1.15072852580446972e+00, epsilon = 1e-12);

        // The grid span is what DENSITY_CUTOFF controls, and it is exact on both
        // sides, so this is the assertion that actually pins the constant.
        // R: `density(x, bw = bw.nrd(x), n = 512)$x[c(1, 512)]`.
        let (from, to) = density_grid_span(-2.5, 4.0, bw);
        assert_relative_eq!(from, -5.95218557741340959e+00, epsilon = 1e-12);
        assert_relative_eq!(to, 7.45218557741340959e+00, epsilon = 1e-12);

        let w = inverse_density_weights(&x).unwrap();
        for (i, &want_i) in want.iter().enumerate() {
            // The function returns one over the density.
            let got = 1.0 / w[i];
            assert_relative_eq!(got, want_i, max_relative = 1e-3);
        }
    }

    #[test]
    fn test_inverse_density_weights_favour_sparse_regions() {
        // A dense cluster near zero and one isolated point far out.
        let mut x: Vec<f64> = (0..200).map(|i| i as f64 * 0.001).collect();
        x.push(5.0);

        let w = inverse_density_weights(&x).unwrap();

        let isolated = w[w.len() - 1];
        let crowded = w[100];
        assert!(
            isolated > crowded * 10.0,
            "isolated weight {isolated} should dwarf the crowded one {crowded}"
        );
        assert!(w.iter().all(|v| v.is_finite()));
    }
}
