//! Implementations of highly variable gene detections in single cell. These
//! are based on the Seurat versions.
//!
//! Every entry point funnels into one of two drivers, [`run_hvg_vst`] and
//! [`run_hvg_dispersion`]. Streaming and non-streaming differ only in how many
//! genes are read per disk batch, and the batch-aware variants differ only in
//! how cells map onto accumulator slots. Both drivers read the store exactly
//! once and sweep each gene's entries exactly once, no matter how many batches
//! are requested.

use rayon::prelude::*;
use std::time::Instant;
use thousands::Separable;

use crate::core::base::info::{BinningStrategy, parse_bin_strategy_type};
use crate::core::base::loess::*;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Genes read per disk batch by the streaming HVG entry points.
///
/// Trades read-call overhead against how many decompressed gene chunks are
/// alive at once. A thousand chunks is enough to keep every core busy while
/// keeping peak occupancy proportional to the batch rather than the store.
const GENE_BATCH_SIZE: usize = 1000;

/// Largest raw count span summarised by counting sort.
///
/// The scratch grid is `n_slots * (max_count + 1)` `u32` counters, so at 1024
/// it stays a few KiB per slot and resident in L1. UMI counts for a single
/// gene sit far below this; the rare gene that does not falls back to the
/// gather-and-run-length path rather than forcing a huge sparse grid.
const MAX_HISTOGRAM_SPAN: usize = 1024;

/// Ceiling on retained [`RawCountSummary`] entries during the VST first pass.
///
/// Summaries have to survive until after the loess fit, so this is a whole-run
/// budget that shrinking the gene batch size cannot help with. At 8 bytes per
/// entry this caps them at 256 MiB, above which the run drops them and pays
/// for a second disk pass instead.
const HVG_SUMMARY_BUDGET_ENTRIES: usize = 32 * 1024 * 1024;

/////////////
// Results //
/////////////

/// Structure that stores HVG information from VST
#[derive(Clone, Debug)]
pub struct HvgRes {
    /// Mean expression of the gene.
    pub mean: Vec<f64>,
    /// Detected variance of the gene.
    pub var: Vec<f64>,
    /// Expected variance of the gene.
    pub var_exp: Vec<f64>,
    /// Standardised variance of the gene.
    pub var_std: Vec<f64>,
}

/// Result structure for dispersion / mean-variance-bin HVG selection
#[derive(Clone, Debug)]
pub struct HvgDispersionRes {
    /// ExpMean: log1p(mean(expm1(data_norm))) per gene
    pub mean: Vec<f64>,
    /// LogVMR: log(var(expm1(data_norm)) / mean(expm1(data_norm))) per gene
    pub dispersion: Vec<f64>,
    /// Dispersion z-scored within mean-bin; 0 for genes outside a bin
    pub dispersion_scaled: Vec<f64>,
    /// Bin assignment per gene; -1 for genes not binned (constant / NaN)
    pub bin: Vec<i32>,
}

/// Enum for the different methods
#[derive(Debug, Clone, Copy, Default)]
pub enum HvgMethod {
    /// Variance stabilising transformation
    #[default]
    Vst,
    /// Binned version by average expression
    MeanVarBin,
    /// Simple dispersion
    Dispersion,
}

/// Helper function to parse the HVG
///
/// ### Params
///
/// * `s` - Type of HVG calculation to do
///
/// ### Returns
///
/// Option of the HvgMethod (some not yet implemented)
pub fn parse_hvg_method(s: &str) -> Option<HvgMethod> {
    match s.to_lowercase().as_str() {
        "vst" => Some(HvgMethod::Vst),
        "meanvarbin" => Some(HvgMethod::MeanVarBin),
        "dispersion" => Some(HvgMethod::Dispersion),
        _ => None,
    }
}

/// Options shared by every HVG entry point.
#[derive(Clone, Copy, Debug)]
pub struct HvgRunOpts {
    /// Genes read per disk batch. `None` reads every gene in one go.
    pub gene_batch_size: Option<usize>,
    /// If `0` -> silent, `1` for normal verbosity, `2` for detailed verbosity.
    pub verbose: usize,
}

////////////////////
// CellBatchIndex //
////////////////////

/// Dense cell to accumulator-slot lookup over the whole store.
#[derive(Clone, Debug)]
pub struct CellBatchIndex {
    /// `batch_id + 1` for selected cells and `0` otherwise, indexed by global
    /// cell id. Length is the store's `total_cells`.
    lookup: Vec<u32>,
    /// Number of selected cells per batch, indexed by batch id.
    batch_sizes: Vec<usize>,
}

impl CellBatchIndex {
    /// Build the lookup, validating the selection against the store.
    ///
    /// Batch labels must densely cover `0..n_batches`.
    ///
    /// ### Params
    ///
    /// * `total_cells` - `SparseDataHeader::total_cells` of the store
    /// * `cell_indices` - Global ids of the cells to include
    /// * `batch_labels` - Batch id per entry of `cell_indices`. `None` puts
    ///   every selected cell into a single batch `0`.
    ///
    /// ### Returns
    ///
    /// The index, or the matching `BixverseErrors` variant if the selection is
    /// empty, contains a duplicate or out-of-range cell, disagrees in length
    /// with `batch_labels`, or leaves a batch empty.
    pub fn new(
        total_cells: usize,
        cell_indices: &[usize],
        batch_labels: Option<&[usize]>,
    ) -> Result<Self, BixverseErrors> {
        if cell_indices.is_empty() {
            return Err(BixverseErrors::HvgEmptySelection);
        }

        if let Some(labels) = batch_labels
            && labels.len() != cell_indices.len()
        {
            return Err(BixverseErrors::HvgBatchLabelLengthMismatch {
                n_labels: labels.len(),
                n_cells: cell_indices.len(),
            });
        }

        let n_batches = match batch_labels {
            Some(labels) => labels.iter().copied().max().map_or(1, |m| m + 1),
            None => 1,
        };

        // Densely labelled cells cannot reach a batch id above `n_cells - 1`,
        // so capping the tally both bounds the allocation against a nonsense
        // label and, by pigeonhole, guarantees a genuinely empty slot to
        // report when the cap bites. Tally before touching the lookup, so that
        // the density check below establishes `n_batches <= n_cells` before
        // any batch id is narrowed to `u32`.
        let tally_len = n_batches.min(cell_indices.len() + 1);
        let mut batch_sizes = vec![0usize; tally_len];

        match batch_labels {
            Some(labels) => {
                for &batch in labels {
                    if batch < tally_len {
                        batch_sizes[batch] += 1;
                    }
                }
            }
            None => batch_sizes[0] = cell_indices.len(),
        }

        if let Some(batch) = batch_sizes.iter().position(|&size| size == 0) {
            return Err(BixverseErrors::HvgEmptyBatch { batch, n_batches });
        }

        let mut lookup = vec![0u32; total_cells];

        for (i, &cell) in cell_indices.iter().enumerate() {
            if cell >= total_cells {
                return Err(BixverseErrors::HvgCellIndexOutOfRange {
                    index: cell,
                    total_cells,
                });
            }
            if lookup[cell] != 0 {
                return Err(BixverseErrors::HvgDuplicateCellIndex { index: cell });
            }

            // Safe to narrow: every batch id is below `n_batches`, which the
            // density check pinned to at most the number of selected cells,
            // and cell ids are `u32` in the store.
            let batch = batch_labels.map_or(0, |labels| labels[i]);
            lookup[cell] = batch as u32 + 1;
        }

        Ok(Self {
            lookup,
            batch_sizes,
        })
    }

    /// Number of batches represented.
    ///
    /// ### Returns
    ///
    /// The batch count.
    #[inline]
    pub fn n_batches(&self) -> usize {
        self.batch_sizes.len()
    }

    /// Number of accumulator slots, i.e. `n_batches + 1` including the discard
    /// bucket.
    ///
    /// ### Returns
    ///
    /// The slot count.
    #[inline]
    pub fn n_slots(&self) -> usize {
        self.batch_sizes.len() + 1
    }

    /// Number of selected cells per batch.
    ///
    /// ### Returns
    ///
    /// Slice of batch sizes, indexed by batch id.
    #[inline]
    pub fn batch_sizes(&self) -> &[usize] {
        &self.batch_sizes
    }

    /// Accumulator slot for a global cell id.
    ///
    /// ### Params
    ///
    /// * `cell_id` - Global cell id as stored in a gene chunk
    ///
    /// ### Returns
    ///
    /// `batch_id + 1` for selected cells and `0` otherwise. Ids beyond the
    /// store, which can only come from a corrupt file, land in the discard
    /// bucket rather than panicking.
    #[inline(always)]
    pub fn slot(&self, cell_id: u32) -> usize {
        self.lookup.get(cell_id as usize).copied().unwrap_or(0) as usize
    }
}

/////////////////////
// RawCountSummary //
/////////////////////

/// Run-length summary of one gene's filtered raw counts within one batch.
///
/// The standardised variance is a pure function of the multiset of a gene's
/// selected non-zero raw counts plus the mean, the expected standard
/// deviation, the clip and the cell count. Raw counts are small integers, so
/// that multiset compresses to distinct values and multiplicities. This makes
/// the summary a sufficient statistic for both VST passes, which is what lets
/// the second pass skip the disk entirely.
///
/// Worst case, when every selected value is distinct, it degenerates to
/// `(value, 1)` pairs and is no better than keeping the raw counts.
#[derive(Clone, Debug, Default)]
pub struct RawCountSummary {
    /// Distinct raw counts observed, ascending on the counting-sort path.
    values: Vec<u32>,
    /// Multiplicity of the matching entry of `values`.
    freqs: Vec<u32>,
    /// Selected non-zero count, i.e. the sum of `freqs`.
    nnz: usize,
}

impl RawCountSummary {
    /// Drop all entries, keeping the allocations for reuse.
    fn clear(&mut self) {
        self.values.clear();
        self.freqs.clear();
        self.nnz = 0;
    }

    /// Append one distinct value and its multiplicity.
    ///
    /// ### Params
    ///
    /// * `value` - The raw count
    /// * `freq` - How often it occurs among the selected cells
    #[inline]
    fn push(&mut self, value: u32, freq: u32) {
        self.values.push(value);
        self.freqs.push(freq);
        self.nnz += freq as usize;
    }

    /// Number of stored `(value, freq)` pairs, for budgeting.
    ///
    /// ### Returns
    ///
    /// The entry count.
    fn entries(&self) -> usize {
        self.values.len()
    }
}

//////////////////////
// HistogramScratch //
//////////////////////

/// Thread-local scratch for building [`RawCountSummary`] values, reused across
/// genes so the per-gene cost is a memset rather than an allocation.
pub struct HistogramScratch {
    /// Flat `n_slots * stride` counting-sort grid, slot major, where `stride`
    /// is `max_count + 1` for the gene in flight.
    counters: Vec<u32>,
    /// Per-slot gather buffers for the wide-range fallback path.
    gathered: Vec<Vec<u32>>,
}

impl HistogramScratch {
    /// Allocate scratch for a fixed slot count.
    ///
    /// ### Params
    ///
    /// * `n_slots` - Number of accumulator slots, see
    ///   [`CellBatchIndex::n_slots`]
    ///
    /// ### Returns
    ///
    /// The scratch.
    pub fn new(n_slots: usize) -> Self {
        Self {
            counters: Vec::new(),
            gathered: vec![Vec::new(); n_slots],
        }
    }
}

/// Summarise one gene's raw counts into one [`RawCountSummary`] per batch.
///
/// One sweep over the gene's stored entries regardless of the batch count: the
/// slot lookup dispatches each entry straight into its batch's counters. Two
/// build paths, picked from a cheap max-scan over the unfiltered counts:
/// counting sort while the value span stays within [`MAX_HISTOGRAM_SPAN`], and
/// gather-sort-run-length above it so a single very large count cannot force a
/// huge sparse grid.
///
/// ### Params
///
/// * `gene` - The gene chunk to summarise
/// * `index` - Cell to slot lookup
/// * `scratch` - Thread-local scratch, reused across genes
/// * `out` - One summary per batch, length `index.n_batches()`. Overwritten.
pub fn summarise_gene_raw(
    gene: &CscGeneChunk,
    index: &CellBatchIndex,
    scratch: &mut HistogramScratch,
    out: &mut [RawCountSummary],
) {
    for summary in out.iter_mut() {
        summary.clear();
    }

    if gene.indices.is_empty() {
        return;
    }

    let n_slots = index.n_slots();
    let max_count = gene.data_raw.iter().max().unwrap_or(0);
    let stride = max_count as usize + 1;

    if stride <= MAX_HISTOGRAM_SPAN {
        scratch.counters.clear();
        scratch.counters.resize(n_slots * stride, 0);

        for (i, &cell_id) in gene.indices.iter().enumerate() {
            let slot = index.slot(cell_id);
            scratch.counters[slot * stride + gene.data_raw.get(i) as usize] += 1;
        }

        for (batch, summary) in out.iter_mut().enumerate() {
            let base = (batch + 1) * stride;
            for (value, &freq) in scratch.counters[base..base + stride].iter().enumerate() {
                if freq > 0 {
                    summary.push(value as u32, freq);
                }
            }
        }
    } else {
        for buffer in scratch.gathered.iter_mut() {
            buffer.clear();
        }

        for (i, &cell_id) in gene.indices.iter().enumerate() {
            let slot = index.slot(cell_id);
            scratch.gathered[slot].push(gene.data_raw.get(i));
        }

        for (batch, summary) in out.iter_mut().enumerate() {
            let buffer = &mut scratch.gathered[batch + 1];
            buffer.sort_unstable();

            let mut run = 0u32;
            let mut current = 0u32;
            for &value in buffer.iter() {
                if run > 0 && value == current {
                    run += 1;
                } else {
                    if run > 0 {
                        summary.push(current, run);
                    }
                    current = value;
                    run = 1;
                }
            }
            if run > 0 {
                summary.push(current, run);
            }
        }
    }
}

/////////////
// Kernels //
/////////////

/// Mean and population variance of a gene within one batch.
///
/// Fuses the sums into a single accumulation via
/// `var = (sum(v^2) - n * mean^2) / n`, which folds the zero cells in
/// analytically. That one-pass form is only safe in `f64`.
///
/// ### Params
///
/// * `summary` - Run-length summary of the gene's selected raw counts
/// * `no_cells` - Number of selected cells in this batch
///
/// ### Returns
///
/// A tuple of `(mean, var)`.
#[inline]
pub fn mean_var_from_summary(summary: &RawCountSummary, no_cells: usize) -> (f32, f32) {
    if no_cells == 0 {
        return (0.0, 0.0);
    }

    let n = no_cells as f64;
    let mut sum = 0f64;
    let mut sum_sq = 0f64;

    for (&value, &freq) in summary.values.iter().zip(summary.freqs.iter()) {
        let value = value as f64;
        let freq = freq as f64;
        sum += freq * value;
        sum_sq += freq * value * value;
    }

    let mean = sum / n;
    let var = ((sum_sq - n * mean * mean) / n).max(0.0);

    (mean as f32, var as f32)
}

/// Standardised variance of a gene within one batch.
///
/// Clipping uses `.min(clip_max).max(-clip_max)` rather than `f32::clamp`, and
/// that is deliberate: `f32::min` discards NaN whereas `clamp` propagates it.
/// If `expected_var` underflows on an all-zero gene the ratio is `0/0`, and a
/// NaN reaching `var_std` panics the ranking in `select_hvg`.
///
/// ### Params
///
/// * `summary` - Run-length summary of the gene's selected raw counts
/// * `mean` - Mean of the gene in this batch
/// * `expected_var` - Expected variance from the loess fit
/// * `clip_max` - Symmetric clip applied to the standardised values
/// * `no_cells` - Number of selected cells in this batch
///
/// ### Returns
///
/// The standardised variance.
#[inline]
pub fn std_variance_from_summary(
    summary: &RawCountSummary,
    mean: f32,
    expected_var: f32,
    clip_max: f32,
    no_cells: usize,
) -> f32 {
    if no_cells == 0 {
        return 0.0;
    }

    let n = no_cells as f64;
    let expected_sd = expected_var.sqrt();

    let mut sum = 0f64;
    let mut sum_sq = 0f64;

    for (&value, &freq) in summary.values.iter().zip(summary.freqs.iter()) {
        let norm = ((value as f32 - mean) / expected_sd)
            .min(clip_max)
            .max(-clip_max) as f64;
        let freq = freq as f64;
        sum += freq * norm;
        sum_sq += freq * norm * norm;
    }

    // Saturating because a corrupt chunk listing the same cell twice would
    // otherwise underflow; a well-formed one always has nnz <= no_cells.
    let n_zeros = no_cells.saturating_sub(summary.nnz);
    if n_zeros > 0 {
        let norm = ((-mean) / expected_sd).min(clip_max).max(-clip_max) as f64;
        let n_zeros = n_zeros as f64;
        sum += n_zeros * norm;
        sum_sq += n_zeros * norm * norm;
    }

    let standardised_mean = sum / n;
    ((sum_sq / n) - standardised_mean * standardised_mean) as f32
}

/// Accumulate Seurat dispersion sums for one gene across every batch.
///
/// Zero cells contribute `expm1(0) = 0`, so the stored entries suffice. The
/// `expm1` is evaluated once per stored entry rather than once per entry per
/// batch, which is where the batch-aware dispersion path used to spend most of
/// its time. Unselected cells are skipped before the transcendental, since
/// here the branch guards real work.
///
/// ### Params
///
/// * `gene` - The gene chunk, read from the `data_norm` layer
/// * `index` - Cell to slot lookup
/// * `out` - Per-slot `(sum, sum_sq)` accumulators, length `index.n_slots()`
pub fn accumulate_disp_stats(gene: &CscGeneChunk, index: &CellBatchIndex, out: &mut [(f64, f64)]) {
    for (i, &cell_id) in gene.indices.iter().enumerate() {
        let slot = index.slot(cell_id);
        if slot == 0 {
            continue;
        }
        let value = (gene.data_norm[i].to_f32() as f64).exp_m1();
        out[slot].0 += value;
        out[slot].1 += value * value;
    }
}

/// Seurat `ExpMean` and `LogVMR` from accumulated dispersion sums.
///
/// Sample variance uses the `(n - 1)` denominator to match R's `var()`.
///
/// ### Params
///
/// * `sum` - Sum of `expm1(data_norm)` over the selected cells
/// * `sum_sq` - Sum of the squares of the same
/// * `no_cells` - Number of selected cells in this batch
///
/// ### Returns
///
/// `(exp_mean, log_vmr)`. `log_vmr` is NaN for constant or all-zero genes.
#[inline]
pub fn disp_stats_from_sums(sum: f64, sum_sq: f64, no_cells: usize) -> (f32, f32) {
    let n = no_cells as f64;

    let mean = if n > 0.0 { sum / n } else { 0.0 };
    let var = if n > 1.0 {
        ((sum_sq - n * mean * mean) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };

    let exp_mean = mean.ln_1p() as f32;
    let log_vmr = if mean > 0.0 && var > 0.0 {
        (var / mean).ln() as f32
    } else {
        f32::NAN
    };

    (exp_mean, log_vmr)
}

/////////////
// Binning //
/////////////

/// Assign genes to bins based on their mean expression
///
/// ### Params
///
/// * `means` - The mean values
/// * `method` - The binning strategy to apply
/// * `n_bins` - The number of bins to use
///
/// ### Returns
///
/// To which bin the given gene belongs
fn bin_features(means: &[f32], method: BinningStrategy, n_bins: usize) -> Vec<i32> {
    let n = means.len();
    let valid: Vec<bool> = means.iter().map(|v| v.is_finite()).collect();

    match method {
        BinningStrategy::EqualWidth => {
            let (mut vmin, mut vmax) = (f32::INFINITY, f32::NEG_INFINITY);
            for i in 0..n {
                if valid[i] {
                    if means[i] < vmin {
                        vmin = means[i];
                    }
                    if means[i] > vmax {
                        vmax = means[i];
                    }
                }
            }
            if !vmin.is_finite() || !vmax.is_finite() || vmax <= vmin {
                return vec![-1; n];
            }
            let width = (vmax - vmin) / n_bins as f32;
            (0..n)
                .map(|i| {
                    if !valid[i] {
                        -1
                    } else {
                        let idx = ((means[i] - vmin) / width) as i32;
                        idx.clamp(0, n_bins as i32 - 1)
                    }
                })
                .collect()
        }
        BinningStrategy::EqualFrequency => {
            let mut nonzero: Vec<f32> = means
                .iter()
                .zip(valid.iter())
                .filter_map(|(&v, &ok)| if ok && v > 0.0 { Some(v) } else { None })
                .collect();
            if nonzero.len() < 2 {
                return vec![-1; n];
            }
            nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let last = nonzero.len() - 1;
            let breaks: Vec<f32> = (0..=n_bins)
                .map(|i| {
                    let p = i as f32 / n_bins as f32;
                    let idx = (p * last as f32).round() as usize;
                    nonzero[idx.min(last)]
                })
                .collect();

            (0..n)
                .map(|i| {
                    if !valid[i] {
                        return -1;
                    }
                    let v = means[i];
                    if v < breaks[0] {
                        return 0;
                    }
                    for b in 0..n_bins {
                        if v <= breaks[b + 1] {
                            return b as i32;
                        }
                    }
                    (n_bins - 1) as i32
                })
                .collect()
        }
    }
}

/// Compute within-bin z-score of dispersion
///
/// ### Params
///
/// * `dispersion` - The dispersions
/// * `bins` - The bin assignments
/// * `n_bins` - The number of bins
///
/// ### Returns
///
/// Z-scores per bin
fn scale_within_bins(dispersion: &[f32], bins: &[i32], n_bins: usize) -> Vec<f32> {
    let mut sums = vec![0f64; n_bins];
    let mut sums_sq = vec![0f64; n_bins];
    let mut counts = vec![0usize; n_bins];

    for (&d, &b) in dispersion.iter().zip(bins.iter()) {
        if b >= 0 && d.is_finite() {
            let bi = b as usize;
            let df = d as f64;
            sums[bi] += df;
            sums_sq[bi] += df * df;
            counts[bi] += 1;
        }
    }

    let stats: Vec<(f64, f64)> = (0..n_bins)
        .map(|i| {
            if counts[i] < 2 {
                (0.0, 0.0)
            } else {
                let c = counts[i] as f64;
                let m = sums[i] / c;
                let var = ((sums_sq[i] - c * m * m) / (c - 1.0)).max(0.0);
                (m, var.sqrt())
            }
        })
        .collect();

    dispersion
        .iter()
        .zip(bins.iter())
        .map(|(&d, &b)| {
            if b < 0 || !d.is_finite() {
                return 0.0;
            }
            let (m, sd) = stats[b as usize];
            if sd > 0.0 {
                ((d as f64 - m) / sd) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Build the final [`HvgDispersionRes`] from raw per-gene means and dispersions
///
/// ### Params
///
/// * `means` - The mean expression of the gene
/// * `dispersions` - The dispersions of the gene
/// * `binning` - The binning strategy
/// * `n_bins` - Number of bins
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn build_disp_result(
    means: Vec<f32>,
    dispersions: Vec<f32>,
    binning: BinningStrategy,
    n_bins: usize,
) -> HvgDispersionRes {
    let bins = bin_features(&means, binning, n_bins);
    let scaled = scale_within_bins(&dispersions, &bins, n_bins);

    // replace NaN in means / dispersion with 0 (matches Seurat)
    let means_clean: Vec<f32> = means
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0.0 })
        .collect();
    let disp_clean: Vec<f32> = dispersions
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0.0 })
        .collect();

    HvgDispersionRes {
        mean: means_clean.r_float_convert(),
        dispersion: disp_clean.r_float_convert(),
        dispersion_scaled: scaled.r_float_convert(),
        bin: bins,
    }
}

/////////////
// Drivers //
/////////////

/// Gene index ranges for one run, given the requested disk batch size.
///
/// ### Params
///
/// * `no_genes` - Total genes in the store
/// * `gene_batch_size` - Genes per disk batch, `None` for a single batch
///
/// ### Returns
///
/// The `(start, end)` gene ranges, in ascending order.
fn gene_blocks(no_genes: usize, gene_batch_size: Option<usize>) -> Vec<(usize, usize)> {
    let step = gene_batch_size.unwrap_or(no_genes).max(1);
    (0..no_genes.div_ceil(step))
        .map(|i| (i * step, ((i + 1) * step).min(no_genes)))
        .collect()
}

/// Read a block of genes and summarise their raw counts per batch.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store
/// * `gene_indices` - Genes to read
/// * `index` - Cell to slot lookup
///
/// ### Returns
///
/// Gene-major summaries, `gene_indices.len() * index.n_batches()` entries.
fn summarise_gene_block<S: SingleCellReading>(
    reader: &S,
    gene_indices: &[usize],
    index: &CellBatchIndex,
) -> Result<Vec<RawCountSummary>, BixverseErrors> {
    let genes = reader.read_gene_parallel(gene_indices)?;
    let n_batches = index.n_batches();
    let n_slots = index.n_slots();

    let mut block = vec![RawCountSummary::default(); genes.len() * n_batches];

    genes
        .par_iter()
        .zip(block.par_chunks_mut(n_batches))
        .for_each_init(
            || HistogramScratch::new(n_slots),
            |scratch, (gene, out)| summarise_gene_raw(gene, index, scratch, out),
        );

    Ok(block)
}

/// Variance-stabilising HVG selection across every batch in `index`.
///
/// Single disk pass in the normal case: the first pass builds a
/// [`RawCountSummary`] per gene per batch, the loess is fit per batch, and the
/// standardised variance is then evaluated straight off those summaries. If
/// the summaries would exceed [`HVG_SUMMARY_BUDGET_ENTRIES`] they are dropped
/// and the second pass re-reads the store, reusing the same kernels.
///
/// Genes with no selected counts fall out with `mean = var = 0`, so `log10`
/// gives `-inf`. `LoessRegression::fit` drops non-finite points and leaves
/// their fitted value at `0.0`, which makes the expected variance `1.0` and
/// the standardised variance `0.0`. That behaviour is relied upon; do not
/// "fix" it into a NaN.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store
/// * `index` - Cell to slot lookup
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip, defaults to `sqrt(no_cells)` per batch
/// * `opts` - Disk batch size and verbosity
///
/// ### Returns
///
/// One `HvgRes` per batch, in batch id order.
pub fn run_hvg_vst<S: SingleCellReading>(
    reader: &S,
    index: &CellBatchIndex,
    loess_span: f32,
    clip_max: Option<f32>,
    opts: HvgRunOpts,
) -> Result<Vec<HvgRes>, BixverseErrors> {
    if !(loess_span > 0.0 && loess_span <= 1.0) {
        return Err(BixverseErrors::HvgInvalidLoessSpan { span: loess_span });
    }

    let verbosity = parse_verbosity_level(opts.verbose);
    let start_total = Instant::now();

    let no_genes = reader.get_header().total_genes;
    let n_batches = index.n_batches();
    let blocks = gene_blocks(no_genes, opts.gene_batch_size);

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST): {} genes, {} batches, {} disk block(s)",
            no_genes.separate_with_underscores(),
            n_batches,
            blocks.len()
        );
    }

    // pass 1 -> per-gene summaries, mean and variance
    let start_pass1 = Instant::now();

    let mut means: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];
    let mut vars: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];
    let mut summaries: Vec<RawCountSummary> = Vec::new();
    let mut retained = true;
    let mut budget_used = 0usize;

    for &(start_gene, end_gene) in blocks.iter() {
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();
        let block = summarise_gene_block(reader, &gene_indices, index)?;

        for gene_summaries in block.chunks(n_batches) {
            for (batch, summary) in gene_summaries.iter().enumerate() {
                let (mean, var) = mean_var_from_summary(summary, index.batch_sizes()[batch]);
                means[batch].push(mean);
                vars[batch].push(var);
            }
        }

        if retained {
            budget_used += block.iter().map(RawCountSummary::entries).sum::<usize>();
            if budget_used > HVG_SUMMARY_BUDGET_ENTRIES {
                retained = false;
                summaries = Vec::new();
                if verbosity.normal_verbosity() {
                    println!(
                        "HVG (VST): summary budget exceeded, falling back to a second disk pass"
                    );
                }
            } else {
                summaries.extend(block);
            }
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST): Calculated gene statistics in {:.2?}",
            start_pass1.elapsed()
        );
    }

    // loess per batch
    let start_loess = Instant::now();

    let mut loess_results = Vec::with_capacity(n_batches);
    let mut clips = Vec::with_capacity(n_batches);

    for batch in 0..n_batches {
        clips.push(clip_max.unwrap_or((index.batch_sizes()[batch] as f32).sqrt()));

        let means_log10: Vec<f32> = means[batch].iter().map(|x| x.log10()).collect();
        let vars_log10: Vec<f32> = vars[batch].iter().map(|x| x.log10()).collect();

        let loess = LoessRegression::new(loess_span, 2);
        loess_results.push(loess.fit(&means_log10, &vars_log10));
    }

    if verbosity.normal_verbosity() {
        println!("HVG (VST): Fitted Loess in {:.2?}", start_loess.elapsed());
    }

    // pass 2 -> standardised variance, off the summaries where possible
    let start_pass2 = Instant::now();

    let mut var_std: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];

    if retained {
        for batch in 0..n_batches {
            let values: Vec<f32> = summaries
                .par_chunks(n_batches)
                .enumerate()
                .map(|(gene, gene_summaries)| {
                    std_variance_from_summary(
                        &gene_summaries[batch],
                        means[batch][gene],
                        10_f32.powf(loess_results[batch].fitted_vals[gene]),
                        clips[batch],
                        index.batch_sizes()[batch],
                    )
                })
                .collect();
            var_std[batch] = values;
        }
    } else {
        for &(start_gene, end_gene) in blocks.iter() {
            let gene_indices: Vec<usize> = (start_gene..end_gene).collect();
            let block = summarise_gene_block(reader, &gene_indices, index)?;

            for (local_idx, gene_summaries) in block.chunks(n_batches).enumerate() {
                let gene = start_gene + local_idx;
                for (batch, summary) in gene_summaries.iter().enumerate() {
                    var_std[batch].push(std_variance_from_summary(
                        summary,
                        means[batch][gene],
                        10_f32.powf(loess_results[batch].fitted_vals[gene]),
                        clips[batch],
                        index.batch_sizes()[batch],
                    ));
                }
            }
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST): Standardised variance in {:.2?}",
            start_pass2.elapsed()
        );
        println!("HVG (VST): Total run time -> {:.2?}", start_total.elapsed());
    }

    // transform to f64 for R
    Ok(means
        .into_iter()
        .zip(vars)
        .zip(loess_results)
        .zip(var_std)
        .map(|(((mean, var), loess_res), std_var)| HvgRes {
            mean: mean.r_float_convert(),
            var: var.r_float_convert(),
            var_exp: loess_res.fitted_vals.r_float_convert(),
            var_std: std_var.r_float_convert(),
        })
        .collect())
}

/// Dispersion / mean-variance-bin HVG selection across every batch in `index`.
///
/// Single disk pass and a single sweep per gene: the slot lookup dispatches
/// each entry into its batch's accumulator, so `expm1` runs once per stored
/// entry rather than once per entry per batch.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store
/// * `index` - Cell to slot lookup
/// * `binning` - The binning strategy to apply
/// * `n_bins` - The number of bins to use
/// * `opts` - Disk batch size and verbosity
///
/// ### Returns
///
/// One `HvgDispersionRes` per batch, in batch id order.
pub fn run_hvg_dispersion<S: SingleCellReading>(
    reader: &S,
    index: &CellBatchIndex,
    binning: BinningStrategy,
    n_bins: usize,
    opts: HvgRunOpts,
) -> Result<Vec<HvgDispersionRes>, BixverseErrors> {
    if n_bins == 0 {
        return Err(BixverseErrors::HvgInvalidBinCount);
    }

    let verbosity = parse_verbosity_level(opts.verbose);
    let start_total = Instant::now();

    let no_genes = reader.get_header().total_genes;
    let n_batches = index.n_batches();
    let n_slots = index.n_slots();
    let blocks = gene_blocks(no_genes, opts.gene_batch_size);

    if verbosity.normal_verbosity() {
        println!(
            "HVG (dispersion): {} genes, {} batches, {} disk block(s)",
            no_genes.separate_with_underscores(),
            n_batches,
            blocks.len()
        );
    }

    let start_stats = Instant::now();

    let mut means: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];
    let mut dispersions: Vec<Vec<f32>> = vec![Vec::with_capacity(no_genes); n_batches];

    for &(start_gene, end_gene) in blocks.iter() {
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();
        let genes = reader.read_gene_parallel(&gene_indices)?;

        let mut acc = vec![(0f64, 0f64); genes.len() * n_slots];
        genes
            .par_iter()
            .zip(acc.par_chunks_mut(n_slots))
            .for_each(|(gene, out)| accumulate_disp_stats(gene, index, out));

        for gene_acc in acc.chunks(n_slots) {
            for batch in 0..n_batches {
                let (sum, sum_sq) = gene_acc[batch + 1];
                let (mean, dispersion) =
                    disp_stats_from_sums(sum, sum_sq, index.batch_sizes()[batch]);
                means[batch].push(mean);
                dispersions[batch].push(dispersion);
            }
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "HVG (dispersion): Calculated gene statistics in {:.2?}",
            start_stats.elapsed()
        );
    }

    let start_bin = Instant::now();
    let out: Vec<HvgDispersionRes> = means
        .into_iter()
        .zip(dispersions)
        .map(|(mean, dispersion)| build_disp_result(mean, dispersion, binning, n_bins))
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "HVG (dispersion): Binning and scaling in {:.2?}",
            start_bin.elapsed()
        );
        println!(
            "HVG (dispersion): Total run time -> {:.2?}",
            start_total.elapsed()
        );
    }

    Ok(out)
}

//////////////////
// Entry points //
//////////////////

/// Pull the single batch out of a driver result.
///
/// ### Params
///
/// * `results` - Driver output, which has exactly one element when the index
///   was built without batch labels
///
/// ### Returns
///
/// The only element.
fn single_batch<T>(results: Vec<T>) -> Result<T, BixverseErrors> {
    results
        .into_iter()
        .next()
        .ok_or(BixverseErrors::HvgEmptySelection)
}

/////////
// VST //
/////////

/// Implementation of the variance stabilised version of the HVG selection
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgRes`
pub fn get_hvg_vst<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: usize,
) -> Result<HvgRes, BixverseErrors> {
    let index = CellBatchIndex::new(reader.get_header().total_cells, cell_indices, None)?;
    let opts = HvgRunOpts {
        gene_batch_size: None,
        verbose,
    };

    single_batch(run_hvg_vst(reader, &index, loess_span, clip_max, opts)?)
}

/// Implementation of the variance stabilised version of the HVG selection
///
/// Genes are read in batches to keep peak memory proportional to the batch
/// rather than to the whole store.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgRes`
pub fn get_hvg_vst_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: usize,
) -> Result<HvgRes, BixverseErrors> {
    let index = CellBatchIndex::new(reader.get_header().total_cells, cell_indices, None)?;
    let opts = HvgRunOpts {
        gene_batch_size: Some(GENE_BATCH_SIZE),
        verbose,
    };

    single_batch(run_hvg_vst(reader, &index, loess_span, clip_max, opts)?)
}

/////////////////////////
// Dispersion versions //
/////////////////////////

/// Dispersion-based HVG detection (non-streaming)
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - The number of bins to use
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn get_hvg_dispersion<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<HvgDispersionRes, BixverseErrors> {
    let index = CellBatchIndex::new(reader.get_header().total_cells, cell_indices, None)?;
    let opts = HvgRunOpts {
        gene_batch_size: None,
        verbose,
    };
    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    single_batch(run_hvg_dispersion(reader, &index, binning, n_bins, opts)?)
}

/// Dispersion-based HVG detection (streaming)
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - The number of bins to use
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn get_hvg_dispersion_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<HvgDispersionRes, BixverseErrors> {
    let index = CellBatchIndex::new(reader.get_header().total_cells, cell_indices, None)?;
    let opts = HvgRunOpts {
        gene_batch_size: Some(GENE_BATCH_SIZE),
        verbose,
    };
    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    single_batch(run_hvg_dispersion(reader, &index, binning, n_bins, opts)?)
}

/// MVB is computationally identical to dispersion
///
/// Selection differs on the R side.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - The number of bins to use
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn get_hvg_mvb<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<HvgDispersionRes, BixverseErrors> {
    get_hvg_dispersion(reader, cell_indices, binning, n_bins, verbose)
}

/// MVB is computationally identical to dispersion (streaming)
///
/// Selection differs on the R side.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep.
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - The number of bins to use
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The `HvgDispersionRes`
pub fn get_hvg_mvb_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<HvgDispersionRes, BixverseErrors> {
    get_hvg_dispersion_streaming(reader, cell_indices, binning, n_bins, verbose)
}

/////////////////////
// HVG batch aware //
/////////////////////

/////////
// VST //
/////////

/// Batch-aware HVG selection using VST method
///
/// Calculates HVG statistics separately for each batch, returning per-batch
/// results. `batch_labels` must densely cover `0..n_batches`.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as
///   `cell_indices`)
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Vec<HvgRes>` - One HvgRes per batch
pub fn get_hvg_vst_batch_aware<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: usize,
) -> Result<Vec<HvgRes>, BixverseErrors> {
    let index = CellBatchIndex::new(
        reader.get_header().total_cells,
        cell_indices,
        Some(batch_labels),
    )?;
    let opts = HvgRunOpts {
        gene_batch_size: None,
        verbose,
    };

    run_hvg_vst(reader, &index, loess_span, clip_max, opts)
}

/// Batch-aware HVG selection using VST method with streaming
///
/// Calculates HVG statistics separately for each batch, reading genes in
/// batches. `batch_labels` must densely cover `0..n_batches`.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `loess_span` - Span parameter for the loess function
/// * `clip_max` - Optional clip max parameter
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Vec<HvgRes>` - One HvgRes per batch
pub fn get_hvg_vst_batch_aware_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    loess_span: f32,
    clip_max: Option<f32>,
    verbose: usize,
) -> Result<Vec<HvgRes>, BixverseErrors> {
    let index = CellBatchIndex::new(
        reader.get_header().total_cells,
        cell_indices,
        Some(batch_labels),
    )?;
    let opts = HvgRunOpts {
        gene_batch_size: Some(GENE_BATCH_SIZE),
        verbose,
    };

    run_hvg_vst(reader, &index, loess_span, clip_max, opts)
}

/////////////////////////
// Dispersion versions //
/////////////////////////

/// Dispersion-based HVG detection, batch-aware
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - Number of bins
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<HvgDispersionRes>` with each element being a batch.
pub fn get_hvg_dispersion_batch_aware<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<Vec<HvgDispersionRes>, BixverseErrors> {
    let index = CellBatchIndex::new(
        reader.get_header().total_cells,
        cell_indices,
        Some(batch_labels),
    )?;
    let opts = HvgRunOpts {
        gene_batch_size: None,
        verbose,
    };
    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    run_hvg_dispersion(reader, &index, binning, n_bins, opts)
}

/// Dispersion-based HVG detection, batch-aware with streaming
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - Number of bins
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<HvgDispersionRes>` with each element being a batch.
pub fn get_hvg_dispersion_batch_aware_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<Vec<HvgDispersionRes>, BixverseErrors> {
    let index = CellBatchIndex::new(
        reader.get_header().total_cells,
        cell_indices,
        Some(batch_labels),
    )?;
    let opts = HvgRunOpts {
        gene_batch_size: Some(GENE_BATCH_SIZE),
        verbose,
    };
    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    run_hvg_dispersion(reader, &index, binning, n_bins, opts)
}

/// MVB HVG, batch-aware
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - Number of bins
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<HvgDispersionRes>` with each element being a batch.
pub fn get_hvg_mvb_batch_aware<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<Vec<HvgDispersionRes>, BixverseErrors> {
    get_hvg_dispersion_batch_aware(reader, cell_indices, batch_labels, binning, n_bins, verbose)
}

/// MVB HVG, batch-aware (streaming)
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice with the cell indices to keep
/// * `batch_labels` - Batch assignment for each cell (same length as cell_
///   indices)
/// * `binning` - The binning strategy to use. One of
///   `"equal_width"` or `"equal_freq"`
/// * `n_bins` - Number of bins
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<HvgDispersionRes>` with each element being a batch.
pub fn get_hvg_mvb_batch_aware_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    batch_labels: &[usize],
    binning: &str,
    n_bins: usize,
    verbose: usize,
) -> Result<Vec<HvgDispersionRes>, BixverseErrors> {
    get_hvg_dispersion_batch_aware_streaming(
        reader,
        cell_indices,
        batch_labels,
        binning,
        n_bins,
        verbose,
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_data::data_io::CellGeneSparseWriter;
    use approx::assert_relative_eq;

    /// RAII guard that removes a test's temp store even if an assert fails.
    struct TempStore(std::path::PathBuf);

    /// Drop implementation for [`TempStore`]. Errors are ignored: the file may
    /// already be gone, and this runs during unwind.
    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempStore {
        /// Reserve a uniquely named scratch store in the system temp directory.
        ///
        /// ### Params
        ///
        /// * `name` - Test-unique suffix
        ///
        /// ### Returns
        ///
        /// The guard; the path is available via [`Self::path`].
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_hvg_{name}.bin")))
        }

        /// Path of the guarded store as a `&str`.
        ///
        /// ### Returns
        ///
        /// The path.
        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// Normalised layer value for a raw count.
    ///
    /// Quantised through `f16` exactly as the writer does, so the dense
    /// reference and the driver see identical inputs.
    ///
    /// ### Params
    ///
    /// * `count` - The raw count
    ///
    /// ### Returns
    ///
    /// The stored `data_norm` value.
    fn norm_of(count: u32) -> F16 {
        F16::from_f32((count as f32).ln_1p())
    }

    /// Write a dense count matrix out as a gene-based store.
    ///
    /// ### Params
    ///
    /// * `path` - Where to write
    /// * `dense` - `dense[gene][cell]` raw counts
    /// * `n_cells` - Number of cells, i.e. the row length of `dense`
    fn write_store(path: &str, dense: &[Vec<u32>], n_cells: usize) {
        let mut writer = CellGeneSparseWriter::new(path, false, n_cells, dense.len(), 1e4)
            .expect("writer opens");

        for (gene_idx, gene) in dense.iter().enumerate() {
            let mut raw = Vec::new();
            let mut indices = Vec::new();
            for (cell, &value) in gene.iter().enumerate() {
                if value > 0 {
                    raw.push(value);
                    indices.push(cell);
                }
            }
            let norms: Vec<F16> = raw.iter().map(|&v| norm_of(v)).collect();

            writer
                .write_gene_chunk(CscGeneChunk::from_conversion(
                    RawCounts::from_u32_auto(&raw),
                    &norms,
                    &indices,
                    gene_idx,
                    true,
                ))
                .expect("write gene chunk");
        }

        writer.finalise().expect("finalise");
    }

    /// Deterministic sparse count matrix, no RNG dependency.
    ///
    /// ### Params
    ///
    /// * `n_genes` - Number of genes
    /// * `n_cells` - Number of cells
    ///
    /// ### Returns
    ///
    /// `dense[gene][cell]` raw counts, roughly 40% dense.
    fn synthetic_counts(n_genes: usize, n_cells: usize) -> Vec<Vec<u32>> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut dense = vec![vec![0u32; n_cells]; n_genes];

        for (gene_idx, gene) in dense.iter_mut().enumerate() {
            for value in gene.iter_mut() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let draw = (state >> 33) as u32;
                if draw % 100 >= 60 {
                    *value = (draw % 17) + (gene_idx as u32 % 4) + 1;
                }
            }
        }

        dense
    }

    /// Dense reference implementation of the VST statistics.
    ///
    /// Iterates every selected cell including the zeros, so it shares no code
    /// with the summary-based kernels.
    ///
    /// ### Params
    ///
    /// * `dense` - `dense[gene][cell]` raw counts
    /// * `cells` - Selected cell ids
    /// * `loess_span` - Span parameter for the loess function
    /// * `clip_max` - Optional clip, defaults to `sqrt(n_cells)`
    ///
    /// ### Returns
    ///
    /// `(mean, var, var_exp, var_std)` per gene.
    fn reference_vst(
        dense: &[Vec<u32>],
        cells: &[usize],
        loess_span: f32,
        clip_max: Option<f32>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = cells.len() as f64;

        let mut means = Vec::with_capacity(dense.len());
        let mut vars = Vec::with_capacity(dense.len());

        for gene in dense {
            let mut sum = 0f64;
            let mut sum_sq = 0f64;
            for &cell in cells {
                let value = gene[cell] as f64;
                sum += value;
                sum_sq += value * value;
            }
            let mean = sum / n;
            means.push(mean as f32);
            vars.push(((sum_sq - n * mean * mean) / n).max(0.0) as f32);
        }

        let clip = clip_max.unwrap_or((cells.len() as f32).sqrt());
        let means_log10: Vec<f32> = means.iter().map(|x| x.log10()).collect();
        let vars_log10: Vec<f32> = vars.iter().map(|x| x.log10()).collect();
        let loess = LoessRegression::new(loess_span, 2).fit(&means_log10, &vars_log10);

        let mut var_std = Vec::with_capacity(dense.len());
        for (gene_idx, gene) in dense.iter().enumerate() {
            let expected_sd = 10_f32.powf(loess.fitted_vals[gene_idx]).sqrt();
            let mut sum = 0f64;
            let mut sum_sq = 0f64;
            for &cell in cells {
                let norm = ((gene[cell] as f32 - means[gene_idx]) / expected_sd)
                    .min(clip)
                    .max(-clip) as f64;
                sum += norm;
                sum_sq += norm * norm;
            }
            let standardised_mean = sum / n;
            var_std.push(((sum_sq / n) - standardised_mean * standardised_mean) as f32);
        }

        (means, vars, loess.fitted_vals, var_std)
    }

    /// Dense reference implementation of the Seurat dispersion statistics.
    ///
    /// ### Params
    ///
    /// * `dense` - `dense[gene][cell]` raw counts
    /// * `cells` - Selected cell ids
    ///
    /// ### Returns
    ///
    /// `(exp_mean, log_vmr)` per gene, NaN cleaned to zero as Seurat does.
    fn reference_dispersion(dense: &[Vec<u32>], cells: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let n = cells.len() as f64;

        let mut means = Vec::with_capacity(dense.len());
        let mut dispersions = Vec::with_capacity(dense.len());

        for gene in dense {
            let mut sum = 0f64;
            let mut sum_sq = 0f64;
            for &cell in cells {
                if gene[cell] == 0 {
                    continue;
                }
                let value = (norm_of(gene[cell]).to_f32() as f64).exp_m1();
                sum += value;
                sum_sq += value * value;
            }

            let mean = sum / n;
            let var = if n > 1.0 {
                ((sum_sq - n * mean * mean) / (n - 1.0)).max(0.0)
            } else {
                0.0
            };

            let exp_mean = mean.ln_1p() as f32;
            let log_vmr = if mean > 0.0 && var > 0.0 {
                (var / mean).ln() as f32
            } else {
                0.0
            };

            means.push(if exp_mean.is_finite() { exp_mean } else { 0.0 });
            dispersions.push(log_vmr);
        }

        (means, dispersions)
    }

    /// Open a reader over a freshly written synthetic store.
    ///
    /// ### Params
    ///
    /// * `temp` - Guard owning the path
    /// * `dense` - `dense[gene][cell]` raw counts
    /// * `n_cells` - Number of cells
    ///
    /// ### Returns
    ///
    /// The reader.
    fn reader_for(temp: &TempStore, dense: &[Vec<u32>], n_cells: usize) -> ParallelSparseReader {
        write_store(temp.path(), dense, n_cells);
        ParallelSparseReader::new(temp.path()).expect("reader opens")
    }

    #[test]
    fn test_vst_matches_dense_reference() {
        let (n_genes, n_cells) = (40, 120);
        let dense = synthetic_counts(n_genes, n_cells);
        let temp = TempStore::new("vst_reference");
        let reader = reader_for(&temp, &dense, n_cells);

        // deliberately a non-contiguous, descending subset
        let cells: Vec<usize> = (0..n_cells).rev().filter(|c| c % 3 != 0).collect();

        let res = get_hvg_vst(&reader, &cells, 0.3, None, 0).expect("vst runs");
        let (means, vars, var_exp, var_std) = reference_vst(&dense, &cells, 0.3, None);

        for gene in 0..n_genes {
            assert_relative_eq!(res.mean[gene], means[gene] as f64, epsilon = 1e-6);
            assert_relative_eq!(res.var[gene], vars[gene] as f64, epsilon = 1e-6);
            assert_relative_eq!(res.var_exp[gene], var_exp[gene] as f64, epsilon = 1e-6);
            assert_relative_eq!(res.var_std[gene], var_std[gene] as f64, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_dispersion_matches_dense_reference() {
        let (n_genes, n_cells) = (30, 90);
        let dense = synthetic_counts(n_genes, n_cells);
        let temp = TempStore::new("disp_reference");
        let reader = reader_for(&temp, &dense, n_cells);

        let cells: Vec<usize> = (0..n_cells).filter(|c| c % 2 == 0).collect();

        let res =
            get_hvg_dispersion(&reader, &cells, "equal_width", 10, 0).expect("dispersion runs");
        let (means, dispersions) = reference_dispersion(&dense, &cells);

        for gene in 0..n_genes {
            assert_relative_eq!(res.mean[gene], means[gene] as f64, epsilon = 1e-6);
            assert_relative_eq!(
                res.dispersion[gene],
                dispersions[gene] as f64,
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_batch_aware_single_batch_matches_plain() {
        let (n_genes, n_cells) = (25, 80);
        let dense = synthetic_counts(n_genes, n_cells);
        let temp = TempStore::new("single_batch");
        let reader = reader_for(&temp, &dense, n_cells);

        let cells: Vec<usize> = (0..n_cells).collect();
        let labels = vec![0usize; n_cells];

        let plain = get_hvg_vst(&reader, &cells, 0.3, None, 0).expect("plain vst");
        let batched =
            get_hvg_vst_batch_aware(&reader, &cells, &labels, 0.3, None, 0).expect("batch-aware");

        assert_eq!(batched.len(), 1);
        assert_eq!(plain.mean, batched[0].mean);
        assert_eq!(plain.var, batched[0].var);
        assert_eq!(plain.var_exp, batched[0].var_exp);
        assert_eq!(plain.var_std, batched[0].var_std);
    }

    #[test]
    fn test_batch_aware_matches_per_batch_runs() {
        let (n_genes, n_cells) = (25, 90);
        let dense = synthetic_counts(n_genes, n_cells);
        let temp = TempStore::new("batch_split");
        let reader = reader_for(&temp, &dense, n_cells);

        let cells: Vec<usize> = (0..n_cells).collect();
        let labels: Vec<usize> = (0..n_cells).map(|c| c % 3).collect();

        let batched =
            get_hvg_vst_batch_aware(&reader, &cells, &labels, 0.4, None, 0).expect("batch-aware");
        assert_eq!(batched.len(), 3);

        for (batch, res) in batched.iter().enumerate() {
            let subset: Vec<usize> = cells
                .iter()
                .zip(labels.iter())
                .filter_map(|(&c, &b)| if b == batch { Some(c) } else { None })
                .collect();
            let solo = get_hvg_vst(&reader, &subset, 0.4, None, 0).expect("solo vst");

            assert_eq!(solo.mean, res.mean);
            assert_eq!(solo.var, res.var);
            assert_eq!(solo.var_std, res.var_std);
        }
    }

    #[test]
    fn test_gene_batch_size_invariant() {
        let (n_genes, n_cells) = (37, 60);
        let dense = synthetic_counts(n_genes, n_cells);
        let temp = TempStore::new("batch_size");
        let reader = reader_for(&temp, &dense, n_cells);

        let cells: Vec<usize> = (0..n_cells).collect();
        let index = CellBatchIndex::new(n_cells, &cells, None).expect("index builds");

        let single = run_hvg_vst(
            &reader,
            &index,
            0.3,
            None,
            HvgRunOpts {
                gene_batch_size: None,
                verbose: 0,
            },
        )
        .expect("single block");

        // a size that does not divide the gene count evenly
        let chunked = run_hvg_vst(
            &reader,
            &index,
            0.3,
            None,
            HvgRunOpts {
                gene_batch_size: Some(7),
                verbose: 0,
            },
        )
        .expect("chunked");

        assert_eq!(single[0].mean, chunked[0].mean);
        assert_eq!(single[0].var, chunked[0].var);
        assert_eq!(single[0].var_std, chunked[0].var_std);
    }

    #[test]
    fn test_gene_without_selected_counts_gives_zero_var_std() {
        // gene 1 is only expressed in cells that are not selected
        let dense = vec![vec![3, 1, 4, 2], vec![0, 5, 0, 7], vec![2, 2, 2, 2]];
        let temp = TempStore::new("empty_gene");
        let reader = reader_for(&temp, &dense, 4);

        let cells = vec![0usize, 2];
        let res = get_hvg_vst(&reader, &cells, 1.0, None, 0).expect("vst runs");

        assert_eq!(res.mean[1], 0.0);
        assert_eq!(res.var[1], 0.0);
        assert!(res.var_std[1].is_finite(), "var_std must not be NaN");
        assert_eq!(res.var_std[1], 0.0);
    }

    #[test]
    fn test_wide_count_range_takes_sort_path() {
        // max count sits far above MAX_HISTOGRAM_SPAN and every value differs,
        // so this forces the gather-sort-run-length branch and the degenerate
        // freq-1 summary at once
        let n_cells = 32;
        let wide: Vec<u32> = (0..n_cells as u32).map(|c| 5_000 + c * 137).collect();
        let dense = vec![
            wide,
            vec![4u32; n_cells],
            (1..=n_cells as u32).collect::<Vec<u32>>(),
        ];

        let temp = TempStore::new("wide_range");
        let reader = reader_for(&temp, &dense, n_cells);

        let cells: Vec<usize> = (0..n_cells).collect();
        let res = get_hvg_vst(&reader, &cells, 1.0, None, 0).expect("vst runs");
        let (means, vars, _, var_std) = reference_vst(&dense, &cells, 1.0, None);

        for gene in 0..dense.len() {
            assert_relative_eq!(res.mean[gene], means[gene] as f64, epsilon = 1e-6);
            assert_relative_eq!(res.var[gene], vars[gene] as f64, max_relative = 1e-5);
            assert_relative_eq!(res.var_std[gene], var_std[gene] as f64, max_relative = 1e-4);
        }
    }

    #[test]
    fn test_counts_above_u16_max() {
        let n_cells = 8;
        let dense = vec![
            vec![70_000, 1, 0, 90_000, 2, 0, 3, 100_000],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        ];

        let temp = TempStore::new("u32_counts");
        let reader = reader_for(&temp, &dense, n_cells);

        let cells: Vec<usize> = (0..n_cells).collect();
        let res = get_hvg_vst(&reader, &cells, 1.0, None, 0).expect("vst runs");
        let (means, vars, _, var_std) = reference_vst(&dense, &cells, 1.0, None);

        for gene in 0..dense.len() {
            assert_relative_eq!(res.mean[gene], means[gene] as f64, epsilon = 1e-6);
            assert_relative_eq!(res.var[gene], vars[gene] as f64, max_relative = 1e-5);
            assert_relative_eq!(res.var_std[gene], var_std[gene] as f64, max_relative = 1e-4);
        }
    }

    #[test]
    fn test_clip_max_bites() {
        // one huge count against a flat background: without clipping the
        // standardised variance would be far larger
        let n_cells = 16;
        let mut gene = vec![1u32; n_cells];
        gene[0] = 50_000;
        let dense = vec![gene, vec![2u32; n_cells]];

        let temp = TempStore::new("clip_max");
        let reader = reader_for(&temp, &dense, n_cells);

        let cells: Vec<usize> = (0..n_cells).collect();

        let tight = get_hvg_vst(&reader, &cells, 1.0, Some(1.0), 0).expect("tight clip");
        let loose = get_hvg_vst(&reader, &cells, 1.0, Some(1e6), 0).expect("loose clip");

        assert!(
            tight.var_std[0] < loose.var_std[0],
            "clipping must shrink var_std: {} vs {}",
            tight.var_std[0],
            loose.var_std[0]
        );

        let (_, _, _, reference) = reference_vst(&dense, &cells, 1.0, Some(1.0));
        assert_relative_eq!(tight.var_std[0], reference[0] as f64, epsilon = 1e-4);
    }

    #[test]
    fn test_summarise_gene_raw_counts_per_batch() {
        let index = CellBatchIndex::new(6, &[0, 1, 2, 3], Some(&[0, 0, 1, 1])).expect("index");
        let gene = CscGeneChunk::from_conversion(
            RawCounts::from_u32_auto(&[5, 5, 5, 7, 9]),
            &[norm_of(5), norm_of(5), norm_of(5), norm_of(7), norm_of(9)],
            &[0, 1, 2, 3, 4],
            0,
            true,
        );

        let mut scratch = HistogramScratch::new(index.n_slots());
        let mut out = vec![RawCountSummary::default(); index.n_batches()];
        summarise_gene_raw(&gene, &index, &mut scratch, &mut out);

        // batch 0 holds cells 0 and 1, both at count 5
        assert_eq!(out[0].values, vec![5]);
        assert_eq!(out[0].freqs, vec![2]);
        assert_eq!(out[0].nnz, 2);

        // batch 1 holds cells 2 and 3, at counts 5 and 7; cell 4 is unselected
        assert_eq!(out[1].values, vec![5, 7]);
        assert_eq!(out[1].freqs, vec![1, 1]);
        assert_eq!(out[1].nnz, 2);
    }

    #[test]
    fn test_index_rejects_bad_input() {
        assert!(matches!(
            CellBatchIndex::new(10, &[], None),
            Err(BixverseErrors::HvgEmptySelection)
        ));

        assert!(matches!(
            CellBatchIndex::new(10, &[0, 11], None),
            Err(BixverseErrors::HvgCellIndexOutOfRange { index: 11, .. })
        ));

        assert!(matches!(
            CellBatchIndex::new(10, &[0, 3, 0], None),
            Err(BixverseErrors::HvgDuplicateCellIndex { index: 0 })
        ));

        assert!(matches!(
            CellBatchIndex::new(10, &[0, 1, 2], Some(&[0, 1])),
            Err(BixverseErrors::HvgBatchLabelLengthMismatch {
                n_labels: 2,
                n_cells: 3
            })
        ));

        // gapped labels: batch 1 is never used
        assert!(matches!(
            CellBatchIndex::new(10, &[0, 1, 2], Some(&[0, 2, 2])),
            Err(BixverseErrors::HvgEmptyBatch { batch: 1, .. })
        ));

        // a nonsense label must not force a huge allocation
        assert!(matches!(
            CellBatchIndex::new(10, &[0, 1], Some(&[0, usize::MAX / 2])),
            Err(BixverseErrors::HvgEmptyBatch { .. })
        ));
    }

    #[test]
    fn test_drivers_reject_bad_parameters() {
        let n_cells = 8;
        let dense = synthetic_counts(4, n_cells);
        let temp = TempStore::new("bad_params");
        let reader = reader_for(&temp, &dense, n_cells);

        let cells: Vec<usize> = (0..n_cells).collect();
        let index = CellBatchIndex::new(n_cells, &cells, None).expect("index builds");
        let opts = HvgRunOpts {
            gene_batch_size: None,
            verbose: 0,
        };

        assert!(matches!(
            run_hvg_vst(&reader, &index, 0.0, None, opts),
            Err(BixverseErrors::HvgInvalidLoessSpan { .. })
        ));

        assert!(matches!(
            run_hvg_vst(&reader, &index, 1.5, None, opts),
            Err(BixverseErrors::HvgInvalidLoessSpan { .. })
        ));

        assert!(matches!(
            run_hvg_dispersion(&reader, &index, BinningStrategy::EqualWidth, 0, opts),
            Err(BixverseErrors::HvgInvalidBinCount)
        ));
    }

    /// Unasserted timing sweep: 3000 genes x 6000 cells, 1 vs 8 batches.
    ///
    /// The point is that the batch-aware runtime should be roughly flat in the
    /// batch count, since every gene is swept once regardless.
    #[test]
    #[cfg(feature = "large_scale_diagnostics")]
    fn diagnostic_hvg_scaling_sweep() {
        let (n_genes, n_cells) = (3_000, 6_000);
        let dense = synthetic_counts(n_genes, n_cells);
        let temp = TempStore::new("scaling_sweep");
        let reader = reader_for(&temp, &dense, n_cells);

        let cells: Vec<usize> = (0..n_cells).collect();

        for &n_batches in &[1usize, 8] {
            let labels: Vec<usize> = (0..n_cells).map(|c| c % n_batches).collect();

            for &gene_batch_size in &[None, Some(GENE_BATCH_SIZE)] {
                let index = CellBatchIndex::new(n_cells, &cells, Some(&labels)).expect("index");
                let opts = HvgRunOpts {
                    gene_batch_size,
                    verbose: 0,
                };

                let start = Instant::now();
                let res = run_hvg_vst(&reader, &index, 0.3, None, opts).expect("vst");
                let vst = start.elapsed();

                let start = Instant::now();
                run_hvg_dispersion(&reader, &index, BinningStrategy::EqualWidth, 20, opts)
                    .expect("dispersion");
                let dispersion = start.elapsed();

                println!(
                    "batches {n_batches}, gene_batch_size {gene_batch_size:?} -> \
                     vst {vst:.2?}, dispersion {dispersion:.2?}"
                );
                assert_eq!(res.len(), n_batches);
            }
        }
    }
}
