//! Implementations of highly variable gene detections in single cell. These
//! are based on the Seurat versions.
//!
//! Every entry point funnels into one of two drivers, [`run_hvg_vst`] and
//! [`run_hvg_dispersion`]. Streaming and non-streaming differ only in how many
//! genes are read per disk batch, and the batch-aware variants differ only in
//! how cells map onto accumulator slots. Both drivers sweep each gene's
//! entries once per disk pass, no matter how many batches are requested, and
//! keep no per-gene state beyond a handful of scalars.
//!
//! The dispersion driver reads the store exactly once. The VST driver reads it
//! once and then re-reads the genes whose values the clip actually reaches,
//! which on droplet data is a fraction of a percent of them. See
//! [`clip_is_reachable`].

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

/// Largest raw count span tallied by counting sort.
///
/// The grid is `n_slots * (max_count + 1)` `u32` counters, so at 1024 it stays
/// a few KiB per slot and resident in L1. UMI counts for a single gene sit far
/// below this; the rare gene that does not accumulates per entry instead,
/// rather than forcing a huge sparse grid.
const MAX_HISTOGRAM_SPAN: usize = 1024;

/// Guard against a degenerate expected variance in the VST second pass.
///
/// The loess leaves dropped points at `0.0`, which exponentiates to an
/// expected variance of `1.0`, but a strongly negative fit can still underflow
/// to zero. A gene whose expected variance lands below this is sent down the
/// exact path, where the clip absorbs the resulting infinities the way the
/// dense reference does.
const MIN_EXPECTED_VAR: f32 = f32::MIN_POSITIVE;

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

///////////////
// GeneStats //
///////////////

/// First-pass statistics for one gene within one batch.
///
/// The standardised variance collapses to `var / expected_var` whenever the
/// clip reaches nothing, because the standardised values then sum to zero by
/// construction. Deciding that needs only the two extremes, so these five
/// scalars are all the second pass needs from the first. Nothing per-gene is
/// retained beyond them.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeneStats {
    /// Mean over the selected cells, zeros included.
    pub mean: f32,
    /// Population variance over the selected cells, zeros included.
    pub var: f32,
    /// Largest selected raw count. Zero when nothing is selected.
    pub max_count: u32,
    /// Smallest selected value, counting the implicit zeros. Zero unless the
    /// gene is expressed in every selected cell of the batch.
    pub min_count: u32,
}

/// Per-slot accumulator for the wide-range fallback.
#[derive(Clone, Copy, Debug)]
struct RawAccumulator {
    /// Sum of the selected raw counts.
    sum: f64,
    /// Sum of their squares.
    sum_sq: f64,
    /// Largest selected raw count.
    max: u32,
    /// Smallest selected non-zero raw count.
    min: u32,
    /// Number of selected non-zero entries.
    nnz: usize,
}

impl Default for RawAccumulator {
    /// Empty accumulator. `min` starts at the maximum so the first value wins.
    fn default() -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            max: 0,
            min: u32::MAX,
            nnz: 0,
        }
    }
}

impl RawAccumulator {
    /// Reduce to the batch's statistics.
    ///
    /// ### Params
    ///
    /// * `no_cells` - Number of selected cells in this batch
    ///
    /// ### Returns
    ///
    /// The statistics, or the default for an empty batch or gene.
    #[inline]
    fn finish(&self, no_cells: usize) -> GeneStats {
        if no_cells == 0 || self.nnz == 0 {
            return GeneStats::default();
        }

        let n = no_cells as f64;
        let mean = self.sum / n;
        let var = ((self.sum_sq - n * mean * mean) / n).max(0.0);

        GeneStats {
            mean: mean as f32,
            var: var as f32,
            max_count: self.max,
            // A gene expressed in every selected cell has no implicit zero, so
            // its lower extreme is the smallest stored count.
            min_count: if self.nnz < no_cells { 0 } else { self.min },
        }
    }
}

//////////////////////
// HistogramScratch //
//////////////////////

/// Thread-local scratch for the first pass, reused across genes so the
/// per-gene cost is a memset rather than an allocation.
pub struct HistogramScratch {
    /// Flat `n_slots * stride` counting-sort grid, slot major, where `stride`
    /// is `max_count + 1` for the gene in flight.
    counters: Vec<u32>,
    /// Per-slot accumulators for the wide-range fallback.
    accumulators: Vec<RawAccumulator>,
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
            accumulators: vec![RawAccumulator::default(); n_slots],
        }
    }
}

/// Tally one gene's entries into the counting-sort grid.
///
/// Generic over the stored width so the raw-count variant is matched once per
/// gene rather than once per entry. This is the hottest loop in the crate and
/// an enum match inside it stops the compiler vectorising anything.
///
/// Unselected cells land in slot 0 and are simply never read back, which keeps
/// the loop branch-free.
///
/// ### Params
///
/// * `values` - Raw counts, in chunk order
/// * `indices` - Global cell ids, in the same order
/// * `index` - Cell to slot lookup
/// * `counters` - Flat `n_slots * stride` grid, zeroed by the caller
/// * `stride` - `max_count + 1`
#[inline]
fn tally_counts<V: Copy + Into<u32>>(
    values: &[V],
    indices: &[u32],
    index: &CellBatchIndex,
    counters: &mut [u32],
    stride: usize,
) {
    for (&cell_id, &value) in indices.iter().zip(values.iter()) {
        let slot = index.slot(cell_id);
        counters[slot * stride + Into::<u32>::into(value) as usize] += 1;
    }
}

/// Accumulate one gene's entries directly, without a histogram.
///
/// The fallback for a gene whose count span would make the grid unreasonably
/// wide. Generic over the stored width for the same reason as [`tally_counts`].
///
/// ### Params
///
/// * `values` - Raw counts, in chunk order
/// * `indices` - Global cell ids, in the same order
/// * `index` - Cell to slot lookup
/// * `accumulators` - Per-slot accumulators, reset by the caller
#[inline]
fn accumulate_counts<V: Copy + Into<u32>>(
    values: &[V],
    indices: &[u32],
    index: &CellBatchIndex,
    accumulators: &mut [RawAccumulator],
) {
    for (&cell_id, &value) in indices.iter().zip(values.iter()) {
        let slot = index.slot(cell_id);
        if slot == 0 {
            continue;
        }

        let value = Into::<u32>::into(value);
        let as_f64 = value as f64;
        let acc = &mut accumulators[slot];

        acc.sum += as_f64;
        acc.sum_sq += as_f64 * as_f64;
        acc.max = acc.max.max(value);
        acc.min = acc.min.min(value);
        acc.nnz += 1;
    }
}

/// Largest raw count in a gene chunk.
///
/// Matches the storage variant once so the scan runs over a concrete slice.
///
/// ### Params
///
/// * `gene` - The gene chunk
///
/// ### Returns
///
/// The maximum, or `0` for an empty chunk.
#[inline]
fn max_raw_count(gene: &CscGeneChunk) -> u32 {
    match &gene.data_raw {
        RawCounts::U16(values) => values.iter().copied().max().unwrap_or(0) as u32,
        RawCounts::U32(values) => values.iter().copied().max().unwrap_or(0),
    }
}

/// Sweep one gene's stored entries into per-batch statistics.
///
/// One pass over the entries regardless of the batch count: the slot lookup
/// dispatches each entry into its batch. Raw counts are small integers, so the
/// default path tallies them into a counting-sort grid and then does the float
/// arithmetic once per *distinct* value rather than once per entry. On droplet
/// data the median gene has two distinct counts across hundreds of entries, so
/// that turns the inner loop into a single integer increment.
///
/// Genes whose count span would make the grid unreasonably wide fall back to
/// accumulating in `f64` per entry. That is the rare case, a couple of genes in
/// twenty thousand, and it needs no sorting because only sums and extremes come
/// out of it.
///
/// The zeros fold in analytically through `var = (sum(v^2) - n * mean^2) / n`,
/// which is why the accumulation is `f64` and not `f32`.
///
/// ### Params
///
/// * `gene` - The gene chunk to sweep
/// * `index` - Cell to slot lookup
/// * `scratch` - Thread-local scratch, reused across genes
/// * `out` - One entry per batch, length `index.n_batches()`. Overwritten.
pub fn gene_stats(
    gene: &CscGeneChunk,
    index: &CellBatchIndex,
    scratch: &mut HistogramScratch,
    out: &mut [GeneStats],
) {
    if gene.indices.is_empty() {
        out.fill(GeneStats::default());
        return;
    }

    let n_slots = index.n_slots();
    let stride = max_raw_count(gene) as usize + 1;

    if stride <= MAX_HISTOGRAM_SPAN {
        scratch.counters.clear();
        scratch.counters.resize(n_slots * stride, 0);

        match &gene.data_raw {
            RawCounts::U16(values) => {
                tally_counts(values, &gene.indices, index, &mut scratch.counters, stride)
            }
            RawCounts::U32(values) => {
                tally_counts(values, &gene.indices, index, &mut scratch.counters, stride)
            }
        }

        for (batch, stats) in out.iter_mut().enumerate() {
            let base = (batch + 1) * stride;
            let mut acc = RawAccumulator::default();

            // Bins are ascending by construction, so the first occupied one is
            // the minimum and the last is the maximum. No comparisons needed.
            for (value, &freq) in scratch.counters[base..base + stride].iter().enumerate() {
                if freq == 0 {
                    continue;
                }

                if acc.nnz == 0 {
                    acc.min = value as u32;
                }
                acc.max = value as u32;

                let value = value as f64;
                let freq = freq as f64;
                acc.sum += freq * value;
                acc.sum_sq += freq * value * value;
                acc.nnz += freq as usize;
            }

            *stats = acc.finish(index.batch_sizes()[batch]);
        }
    } else {
        scratch.accumulators.fill(RawAccumulator::default());

        match &gene.data_raw {
            RawCounts::U16(values) => {
                accumulate_counts(values, &gene.indices, index, &mut scratch.accumulators)
            }
            RawCounts::U32(values) => {
                accumulate_counts(values, &gene.indices, index, &mut scratch.accumulators)
            }
        }

        for (batch, stats) in out.iter_mut().enumerate() {
            *stats = scratch.accumulators[batch + 1].finish(index.batch_sizes()[batch]);
        }
    }
}

/////////////
// Kernels //
/////////////

/// Whether the clip can reach any value of a gene within one batch.
///
/// Checks both extremes: the largest stored count, and the smallest selected
/// value, which is zero for any gene left unexpressed in a selected cell. A
/// degenerate expected variance counts as reachable too, so that the exact path
/// absorbs the resulting infinities through the clip rather than the closed
/// form handing back one.
///
/// ### Params
///
/// * `mean` - Mean over the selected cells, zeros included
/// * `min` - Smallest selected value, zeros included
/// * `max` - Largest selected value
/// * `expected_var` - Expected variance from the loess fit
/// * `clip_max` - Symmetric clip applied to the standardised values
///
/// ### Returns
///
/// `true` when the gene needs the exact, distribution-aware treatment.
#[inline]
pub fn clip_is_reachable(mean: f32, min: f32, max: f32, expected_var: f32, clip_max: f32) -> bool {
    if !expected_var.is_finite() || expected_var < MIN_EXPECTED_VAR {
        return true;
    }

    let reach = clip_max * expected_var.sqrt();

    (max - mean) > reach || (mean - min) > reach
}

/// Exact standardised variance for one gene across every batch.
///
/// Only ever runs for the genes [`clip_is_reachable`] flags. Clipping uses
/// `.min(clip_max).max(-clip_max)` rather than `f32::clamp`, and that is
/// deliberate: `f32::min` discards NaN whereas `clamp` propagates it, and a NaN
/// reaching `var_std` panics the ranking in `select_hvg`.
///
/// ### Params
///
/// * `gene` - The gene chunk, re-read for this purpose
/// * `index` - Cell to slot lookup
/// * `params` - Per-batch `(mean, expected_sd, clip_max)`
/// * `scratch` - Per-slot `(sum, sum_sq, nnz)` accumulators, length
///   `index.n_slots()`. Overwritten.
/// * `out` - One standardised variance per batch, length `index.n_batches()`
pub fn std_variance_exact(
    gene: &CscGeneChunk,
    index: &CellBatchIndex,
    params: &[(f32, f32, f32)],
    scratch: &mut [(f64, f64, usize)],
    out: &mut [f32],
) {
    for slot in scratch.iter_mut() {
        *slot = (0.0, 0.0, 0);
    }

    for (i, &cell_id) in gene.indices.iter().enumerate() {
        let slot = index.slot(cell_id);
        if slot == 0 {
            continue;
        }

        let (mean, expected_sd, clip_max) = params[slot - 1];
        let norm = ((gene.data_raw.get(i) as f32 - mean) / expected_sd)
            .min(clip_max)
            .max(-clip_max) as f64;

        scratch[slot].0 += norm;
        scratch[slot].1 += norm * norm;
        scratch[slot].2 += 1;
    }

    for (batch, value) in out.iter_mut().enumerate() {
        let no_cells = index.batch_sizes()[batch];
        if no_cells == 0 {
            *value = 0.0;
            continue;
        }

        let (mean, expected_sd, clip_max) = params[batch];
        let (mut sum, mut sum_sq, nnz) = scratch[batch + 1];

        // Saturating because a corrupt chunk listing the same cell twice would
        // otherwise underflow; a well-formed one always has nnz <= no_cells.
        let n_zeros = no_cells.saturating_sub(nnz);
        if n_zeros > 0 {
            let norm = ((-mean) / expected_sd).min(clip_max).max(-clip_max) as f64;
            let n_zeros = n_zeros as f64;
            sum += n_zeros * norm;
            sum_sq += n_zeros * norm * norm;
        }

        let n = no_cells as f64;
        let standardised_mean = sum / n;
        *value = ((sum_sq / n) - standardised_mean * standardised_mean) as f32;
    }
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

/// Sweep a block of genes into gene-major statistics.
///
/// Everything happens inside the parallel closure: the entry sweep, the
/// mean/variance reduction and the extremes. There is no serial tail behind it,
/// and `out` is a slice of the run-wide array rather than a per-block
/// allocation, so nothing is appended or copied afterwards.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store
/// * `gene_indices` - Genes to read
/// * `index` - Cell to slot lookup
/// * `out` - Gene-major destination, `gene_indices.len() * index.n_batches()`
///   entries. Overwritten.
fn sweep_gene_block<S: SingleCellReading>(
    reader: &S,
    gene_indices: &[usize],
    index: &CellBatchIndex,
    out: &mut [GeneStats],
) -> Result<(), BixverseErrors> {
    let genes = reader.read_gene_parallel(gene_indices)?;
    let n_batches = index.n_batches();
    let n_slots = index.n_slots();

    genes
        .par_iter()
        .zip(out.par_chunks_mut(n_batches))
        .for_each_init(
            || HistogramScratch::new(n_slots),
            |scratch, (gene, stats)| gene_stats(gene, index, scratch, stats),
        );

    Ok(())
}

/// Variance-stabilising HVG selection across every batch in `index`.
///
/// One disk pass in the normal case, and no retained per-gene state. The first
/// pass keeps five scalars per gene per batch; after the loess, the
/// standardised variance falls out of `var / expected_var` for every gene the
/// clip cannot reach. That identity holds because unclipped standardised values
/// are `(x - mean) / sd`, which sum to zero by construction, leaving the
/// variance of the standardised values exactly `var / expected_var`.
///
/// The genes the clip does reach are re-read and evaluated exactly. On droplet
/// data that is a fraction of a percent of the store. It grows as `clip_max`
/// shrinks, so a small cell count re-reads more of the store, but a small cell
/// count is also a small store.
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
    let n_slots = index.n_slots();
    let blocks = gene_blocks(no_genes, opts.gene_batch_size);

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST): {} genes, {} batches, {} disk block(s)",
            no_genes.separate_with_underscores(),
            n_batches,
            blocks.len()
        );
    }

    // pass 1 -> mean, variance and the two extremes, gene-major
    let start_pass1 = Instant::now();

    let mut stats: Vec<GeneStats> = vec![GeneStats::default(); no_genes * n_batches];

    for (block, &(start_gene, end_gene)) in blocks.iter().enumerate() {
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();
        sweep_gene_block(
            reader,
            &gene_indices,
            index,
            &mut stats[start_gene * n_batches..end_gene * n_batches],
        )?;

        if verbosity.detailed_verbosity() {
            report_decile_progress(
                block + 1,
                block,
                blocks.len(),
                "gene blocks",
                start_pass1.elapsed(),
            );
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

        let means_log10: Vec<f32> = stats[batch..]
            .iter()
            .step_by(n_batches)
            .map(|s| s.mean.log10())
            .collect();
        let vars_log10: Vec<f32> = stats[batch..]
            .iter()
            .step_by(n_batches)
            .map(|s| s.var.log10())
            .collect();

        let loess = LoessRegression::new(loess_span, 2);
        loess_results.push(loess.fit(&means_log10, &vars_log10));
    }

    if verbosity.normal_verbosity() {
        println!("HVG (VST): Fitted Loess in {:.2?}", start_loess.elapsed());
    }

    // pass 2 -> closed form wherever the clip cannot reach
    let start_pass2 = Instant::now();

    let mut var_std: Vec<f32> = vec![0.0; no_genes * n_batches];

    let needs_exact: Vec<usize> = stats
        .par_chunks(n_batches)
        .zip(var_std.par_chunks_mut(n_batches))
        .enumerate()
        .filter_map(|(gene, (gene_stats, out))| {
            let mut reachable = false;
            for (batch, value) in out.iter_mut().enumerate() {
                let stats = &gene_stats[batch];
                let expected_var = 10_f32.powf(loess_results[batch].fitted_vals[gene]);
                if clip_is_reachable(
                    stats.mean,
                    stats.min_count as f32,
                    stats.max_count as f32,
                    expected_var,
                    clips[batch],
                ) {
                    reachable = true;
                } else {
                    *value = stats.var / expected_var;
                }
            }
            if reachable { Some(gene) } else { None }
        })
        .collect();

    if !needs_exact.is_empty() {
        let genes = reader.read_gene_parallel(&needs_exact)?;
        let mut exact = vec![0f32; needs_exact.len() * n_batches];

        genes
            .par_iter()
            .zip(needs_exact.par_iter())
            .zip(exact.par_chunks_mut(n_batches))
            .for_each_init(
                || vec![(0f64, 0f64, 0usize); n_slots],
                |scratch, ((gene, &gene_idx), out)| {
                    let params: Vec<(f32, f32, f32)> = (0..n_batches)
                        .map(|batch| {
                            let expected_var =
                                10_f32.powf(loess_results[batch].fitted_vals[gene_idx]);
                            (
                                stats[gene_idx * n_batches + batch].mean,
                                expected_var.sqrt(),
                                clips[batch],
                            )
                        })
                        .collect();
                    std_variance_exact(gene, index, &params, scratch, out);
                },
            );

        for (&gene, values) in needs_exact.iter().zip(exact.chunks(n_batches)) {
            var_std[gene * n_batches..(gene + 1) * n_batches].copy_from_slice(values);
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "HVG (VST): Standardised variance in {:.2?} ({} gene(s) re-read for the clip)",
            start_pass2.elapsed(),
            needs_exact.len().separate_with_underscores()
        );
        println!("HVG (VST): Total run time -> {:.2?}", start_total.elapsed());
    }

    // gather gene-major back into one result per batch
    Ok(loess_results
        .into_iter()
        .enumerate()
        .map(|(batch, loess_res)| HvgRes {
            mean: stats[batch..]
                .iter()
                .step_by(n_batches)
                .map(|s| s.mean as f64)
                .collect(),
            var: stats[batch..]
                .iter()
                .step_by(n_batches)
                .map(|s| s.var as f64)
                .collect(),
            var_exp: loess_res.fitted_vals.r_float_convert(),
            var_std: var_std[batch..]
                .iter()
                .step_by(n_batches)
                .map(|&v| v as f64)
                .collect(),
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

    // Gene-major, so the sums and the reduction both happen inside the
    // parallel closure and nothing is appended behind it.
    let mut stats: Vec<(f32, f32)> = vec![(0.0, 0.0); no_genes * n_batches];

    for (block, &(start_gene, end_gene)) in blocks.iter().enumerate() {
        let gene_indices: Vec<usize> = (start_gene..end_gene).collect();
        let genes = reader.read_gene_parallel(&gene_indices)?;

        genes
            .par_iter()
            .zip(stats[start_gene * n_batches..end_gene * n_batches].par_chunks_mut(n_batches))
            .for_each_init(
                || vec![(0f64, 0f64); n_slots],
                |acc, (gene, out)| {
                    acc.fill((0.0, 0.0));
                    accumulate_disp_stats(gene, index, acc);
                    for (batch, slot) in out.iter_mut().enumerate() {
                        let (sum, sum_sq) = acc[batch + 1];
                        *slot = disp_stats_from_sums(sum, sum_sq, index.batch_sizes()[batch]);
                    }
                },
            );

        if verbosity.detailed_verbosity() {
            report_decile_progress(
                block + 1,
                block,
                blocks.len(),
                "gene blocks",
                start_stats.elapsed(),
            );
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "HVG (dispersion): Calculated gene statistics in {:.2?}",
            start_stats.elapsed()
        );
    }

    let start_bin = Instant::now();
    let out: Vec<HvgDispersionRes> = (0..n_batches)
        .map(|batch| {
            let mean: Vec<f32> = stats[batch..]
                .iter()
                .step_by(n_batches)
                .map(|s| s.0)
                .collect();
            let dispersion: Vec<f32> = stats[batch..]
                .iter()
                .step_by(n_batches)
                .map(|s| s.1)
                .collect();
            build_disp_result(mean, dispersion, binning, n_bins)
        })
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
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_hvg_{name}.bin")))
        }

        /// Path of the guarded store as a `&str`.
        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// The stored `data_norm` value for a raw count, quantised through `f16`
    /// exactly as the writer does so the dense reference and the driver see
    /// identical inputs.
    fn norm_of(count: u32) -> F16 {
        F16::from_f32((count as f32).ln_1p())
    }

    /// Write a `dense[gene][cell]` count matrix out as a gene-based store.
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

    /// Deterministic `dense[gene][cell]` counts, roughly 40% dense, no RNG dependency.
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

    /// Dense reference for `(mean, var, var_exp, var_std)` per gene. Iterates
    /// every selected cell including the zeros, so it shares no code with the
    /// summary-based kernels. `clip_max` defaults to `sqrt(n_cells)`.
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

    /// Dense reference for the Seurat dispersion statistics, `(exp_mean,
    /// log_vmr)` per gene with NaN cleaned to zero as Seurat does.
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
    fn reader_for(temp: &TempStore, dense: &[Vec<u32>], n_cells: usize) -> ParallelSparseReader {
        write_store(temp.path(), dense, n_cells);
        ParallelSparseReader::new(temp.path()).expect("reader opens")
    }

    /// The summary-based VST kernels must match a dense pass over every selected cell.
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

    /// Same for the Seurat dispersion path, including its NaN cleaning.
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

    /// One batch has to reduce exactly to the plain path, hence `assert_eq!` on
    /// the `Vec<f64>`: the two are meant to be bit-identical, not merely close.
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

    /// Each batch has to reduce exactly to a solo run over its own cells, so
    /// `assert_eq!` on the `Vec<f64>` is deliberate rather than a missing tolerance.
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

    /// Chunking the gene sweep must be bit-neutral, so `assert_eq!` is the point here.
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

    /// A gene with no counts among the selected cells gives zero, not NaN.
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

    /// Genes the clip actually reaches take the re-read path and still match the reference.
    #[test]
    fn test_wide_count_range_takes_exact_path() {
        // counts spanning thousands against a clip of sqrt(32), so the clip
        // reaches these genes and they go down the re-read path rather than the
        // closed form
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

    /// Counts stored as u32 rather than u16 flow through the statistics unchanged.
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

    /// A tight `clip_max` shrinks `var_std` and still agrees with the dense reference.
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

    /// Per-batch statistics only see their own cells, and unselected cells count for nothing.
    #[test]
    fn test_gene_stats_per_batch() {
        let index = CellBatchIndex::new(6, &[0, 1, 2, 3], Some(&[0, 0, 1, 1])).expect("index");
        let gene = CscGeneChunk::from_conversion(
            RawCounts::from_u32_auto(&[5, 5, 5, 7, 9]),
            &[norm_of(5), norm_of(5), norm_of(5), norm_of(7), norm_of(9)],
            &[0, 1, 2, 3, 4],
            0,
            true,
        );

        let mut scratch = HistogramScratch::new(index.n_slots());
        let mut out = vec![GeneStats::default(); index.n_batches()];
        gene_stats(&gene, &index, &mut scratch, &mut out);

        // batch 0 holds cells 0 and 1, both at count 5, so the gene is dense
        // in this batch and has no implicit zero
        assert_eq!(out[0].mean, 5.0);
        assert_eq!(out[0].var, 0.0);
        assert_eq!(out[0].max_count, 5);
        assert_eq!(out[0].min_count, 5);

        // batch 1 holds cells 2 and 3, at counts 5 and 7; cell 4 is unselected
        assert_eq!(out[1].mean, 6.0);
        assert_eq!(out[1].var, 1.0);
        assert_eq!(out[1].max_count, 7);
        assert_eq!(out[1].min_count, 5);
    }

    /// A selected cell with no stored entry has to pull `min_count` down to zero.
    #[test]
    fn test_gene_stats_marks_the_implicit_zero() {
        // cell 2 is selected but carries no entry for this gene, so the lower
        // extreme has to be 0 rather than the smallest stored count
        let index = CellBatchIndex::new(4, &[0, 1, 2], None).expect("index");
        let gene = CscGeneChunk::from_conversion(
            RawCounts::from_u32_auto(&[4, 6]),
            &[norm_of(4), norm_of(6)],
            &[0, 1],
            0,
            true,
        );

        let mut scratch = HistogramScratch::new(index.n_slots());
        let mut out = vec![GeneStats::default(); index.n_batches()];
        gene_stats(&gene, &index, &mut scratch, &mut out);

        assert_eq!(out[0].max_count, 6);
        assert_eq!(out[0].min_count, 0);
        assert_relative_eq!(out[0].mean, 10.0 / 3.0, epsilon = 1e-6);
    }

    /// The gate routes to the exact path whenever the clip can bite or the variance is degenerate.
    #[test]
    fn test_clip_reachability_gates_the_exact_path() {
        // a tight clip cannot cover a wide gene, a generous one can
        let (mean, min, max) = (2.0, 0.0, 50.0);

        assert!(clip_is_reachable(mean, min, max, 1.0, 1.0));
        assert!(!clip_is_reachable(mean, min, max, 1.0, 100.0));

        // a degenerate expected variance always routes to the exact path
        assert!(clip_is_reachable(mean, min, max, 0.0, 1e9));
        assert!(clip_is_reachable(mean, min, max, f32::NAN, 1e9));
    }

    /// Every malformed selection or batch labelling is refused, including a label big enough to blow up the allocation.
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

    /// Out-of-range loess spans and bin counts error at the driver rather than downstream.
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
