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

use indexmap::IndexSet;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;

use crate::core::base::kernel_smooth::bw_nrd;
use crate::core::math::vector_helpers::interp_linear_at;
use crate::errors::BixverseErrors;
use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{CscGeneChunk, RawCounts, SingleCellReading};

use super::sct_nb_fit::{NbOffsetFit, fit_nb_offset_gene};
use super::sctransform::{
    SctGeneStats, SctModel, SctParams, min_variance_from_umi_median, regularise_sct_model,
    sct_residual_row,
};

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

/// Kernel support cutoff for the density estimate, in bandwidths.
const DENSITY_CUTOFF: f64 = 4.0;

//////////////
// Options //
//////////////

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

////////////////
// Gene pass //
////////////////

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
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&c| c as u32).collect();
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

/// The genes and cells the step-1 fit runs on.
#[derive(Clone, Debug)]
pub struct SctStep1Selection {
    /// Store indices of the sampled genes, ascending.
    pub genes: Vec<usize>,
    /// Positions within `cell_indices` of the sampled cells, ascending.
    pub cells: Vec<usize>,
}

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

/// Sampling weights inversely proportional to the density of `x`.
///
/// A Gaussian kernel density estimate on R's `bw.nrd` bandwidth, evaluated on a
/// grid and interpolated back, matching `density(bw = 'nrd')` followed by
/// `approx()`. Machine epsilon is added before inverting so a gene in an empty
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

    // R's density() pads the grid by the kernel cutoff at both ends.
    let from = lo - DENSITY_CUTOFF * bw;
    let to = hi + DENSITY_CUTOFF * bw;
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

/////////////////
// Model fit //
/////////////////

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
pub fn fit_sctransform<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    library_sizes: &[f64],
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
        return Err(BixverseErrors::SctGeneIndexOutOfRange {
            index: bad,
            n_genes: n_cells,
        });
    }

    // Detection within the step-1 cells decides gene eligibility, so it needs
    // its own count. One extra read of the sampled cells only.
    let step1_cell_set: IndexSet<u32> = cells.iter().map(|&c| cell_indices[c] as u32).collect();

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

    let chunks = reader.read_gene_parallel_filtered(&genes, &step1_cell_set)?;
    let fits: Vec<NbOffsetFit> = chunks
        .par_iter()
        .map(|chunk| {
            let dense = densify(chunk, cells.len());
            fit_nb_offset_gene(&dense, &log_offset)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // `read_gene_parallel_filtered` preserves the requested order, but the
    // regularisation indexes the model by store position, so pair them up
    // explicitly rather than trusting that.
    let fit_idx: Vec<usize> = chunks.iter().map(|c| c.original_index).collect();

    if verbosity.normal_verbosity() {
        println!(
            "scTransform: fitted {} genes in {:.2?}",
            fits.len(),
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
    for (fit, g) in fits.iter().zip(fit_idx.iter()) {
        if let Some(&pos) = position.get(g) {
            kept_fits.push(*fit);
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
/// * `cell_indices` - Cells to include, in the order `log10_umi` is given in.
/// * `log10_umi` - `log10(total UMI)` per cell.
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
    log10_umi: &[f64],
    opts: SctStreamOpts,
) -> Result<Vec<f64>, BixverseErrors> {
    if log10_umi.len() != cell_indices.len() {
        return Err(BixverseErrors::LengthMismatch {
            name: "log10_umi",
            expected: cell_indices.len(),
            found: log10_umi.len(),
        });
    }

    let n_cells = cell_indices.len();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&c| c as u32).collect();
    let step = opts.gene_batch_size.unwrap_or(model.len()).max(1);
    let mut out = vec![0.0_f64; model.len()];

    for block in model.genes.chunks(step) {
        let chunks = reader.read_gene_parallel_filtered(block, &cell_set)?;

        let vars: Vec<(usize, f64)> = chunks
            .par_iter()
            .map(|chunk| {
                let pos = model.position(chunk.original_index).ok_or(
                    BixverseErrors::SctGeneIndexOutOfRange {
                        index: chunk.original_index,
                        n_genes: model.len(),
                    },
                )?;
                let counts = chunk_counts(chunk);
                let mut row = vec![0.0_f32; n_cells];
                sct_residual_row(
                    &counts,
                    &chunk.indices,
                    n_cells,
                    pos,
                    model,
                    log10_umi,
                    &mut row,
                )?;
                Ok((pos, sample_variance(&row)))
            })
            .collect::<Result<Vec<_>, BixverseErrors>>()?;

        for (pos, v) in vars {
            out[pos] = v;
        }
    }

    Ok(out)
}

/// Sample variance of a row, `n - 1` in the denominator, matching R's
/// `rowVars`.
///
/// Accumulated in `f64` off an `f32` row: the residuals span the clipping range
/// and the squared deviations of a strongly expressed gene lose `f32`
/// precision well before the sum completes.
///
/// ### Params
///
/// * `row` - The values.
///
/// ### Returns
///
/// The variance, or `0.0` for fewer than two values.
fn sample_variance(row: &[f32]) -> f64 {
    let n = row.len();
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;
    let mean = row.iter().map(|&x| x as f64).sum::<f64>() / n_f;

    row.iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / (n_f - 1.0)
}

/// A gene chunk's stored counts as `f64`.
///
/// ### Params
///
/// * `chunk` - The gene chunk.
///
/// ### Returns
///
/// The non-zero counts, in the chunk's own order.
fn chunk_counts(chunk: &CscGeneChunk) -> Vec<f64> {
    match &chunk.data_raw {
        RawCounts::U16(v) => v.iter().map(|&x| x as f64).collect(),
        RawCounts::U32(v) => v.iter().map(|&x| x as f64).collect(),
    }
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
