//! Where the HVG wall clock actually goes, on realistic single-cell counts.
//!
//! The inline `diagnostic_hvg_scaling_sweep` runs on `synthetic_counts`, which
//! is 40% dense with every count in `1..21` and every gene distributed alike.
//! Real 10x data is 5-10% dense with a long tail in both nnz and max count, and
//! that tail is what drives the clip and the re-read. So this bench generates
//! counts from a gamma-Poisson model calibrated to 10x rather than from a
//! uniform draw.
//!
//! The measured cells decompose one VST run:
//!
//! * `read` - decompress and deserialise only. The floor everything sits on.
//! * `pass1` - plus the [`gene_stats`] sweep. Minus `read` this is the whole
//!   marginal cost of the statistics.
//! * `endtoend` - `run_hvg_vst` at `verbose = 1`, so the driver prints its own
//!   phase split next to ours.
//! * `clip` - unasserted: how many genes the clip actually reaches, which is
//!   how many get re-read, and whether `var / expected_var` agrees with the
//!   driver everywhere else.
//!
//! Every cell runs after a full warm-up read, so these are warm page-cache
//! numbers. A cold first run is a different measurement and not the one that
//! matters when the same store is scanned repeatedly.
//!
//! Run with:
//! ```
//! cargo bench --features single-cell --bench hvg_bench
//! ```
//!
//! The generated store is cached in the temp directory and keyed on shape and
//! seed, so only the first run pays for it. `HVG_BENCH_REGEN=1` forces a
//! rebuild. `HVG_BENCH_GENES` / `HVG_BENCH_CELLS` change the shape;
//! `HVG_BENCH_ONLY=read,pass1` runs a comma-separated subset of the cells.

#![cfg(feature = "single-cell")]

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, LogNormal, Poisson};
use rayon::prelude::*;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_data::data_io::CellGeneSparseWriter;
use bixverse_rs::single_cell::sc_processing::hvg::*;

////////////
// Shapes //
////////////

/// Default gene count. Roughly a filtered 10x feature set.
const DEFAULT_GENES: usize = 20_000;

/// Default cell count. Small enough that the store builds in well under a
/// minute, large enough that the per-gene work dominates the per-block
/// overhead. Push it to 200_000 via `HVG_BENCH_CELLS` for a production shape.
const DEFAULT_CELLS: usize = 20_000;

/// Batch counts swept by the batch-aware cells.
///
/// Anything that scales with the batch count rather than with nnz shows up
/// between 1 and 16.
const BATCH_SWEEP: [usize; 3] = [1, 4, 16];

/// Genes per disk block, mirroring the private `GENE_BATCH_SIZE` in `hvg.rs`.
const GENE_BATCH_SIZE: usize = 1_000;

/// Genes generated per write block. Bounds peak occupancy during generation to
/// this many chunks while still giving rayon enough to chew on.
const WRITE_BLOCK: usize = 512;

/// Repeats per timed phase. The fastest run is reported.
const REPEATS: usize = 5;

/// Loess span used by every VST call here. Matches the R-side default.
const LOESS_SPAN: f32 = 0.3;

/// Library size the `data_norm` layer is scaled against.
const TARGET_SIZE: f32 = 1e4;

/// Seed for the generator. Fixed so the cached store is reproducible.
const SEED: u64 = 0x5EED_0BEEF_u64;

/////////////////
// Count model //
/////////////////

/// Median library size, i.e. total UMIs per cell.
const MEDIAN_LIBRARY_SIZE: f64 = 5_000.0;

/// Log-scale spread of the library size. 0.35 gives roughly a 4x spread
/// between the 1st and 99th percentile cell, which is typical post-QC.
const LIBRARY_SIZE_LOG_SD: f64 = 0.35;

/// Log-scale spread of per-gene relative expression, before normalisation.
///
/// Relative shares are drawn log-normal and then rescaled to sum to one, so
/// only the spread matters, not the location. 2.0 puts the per-gene mean count
/// across roughly four orders of magnitude, which is what gives a realistic
/// sparsity and a realistic number of near-empty genes.
const EXPRESSION_LOG_SD: f64 = 2.0;

/// Number of dominant genes, standing in for MALAT1, the ribosomal proteins
/// and the mito genes.
const N_DOMINANT_GENES: usize = 40;

/// Share of every cell's library taken by the dominant genes, split across
/// them as `1 / rank`. The top gene therefore takes about 6% of a cell on its
/// own, which is what produces the raw counts in the hundreds that the clip
/// eventually reaches.
const DOMINANT_SHARE: f64 = 0.25;

/// Biological coefficient of variation. The gamma mixing weight has variance
/// `BCV^2`, so this sets the overdispersion on top of Poisson noise. 0.55 is
/// the usual figure quoted for droplet data.
const BCV: f64 = 0.55;

/// Below this expected count the gamma mixing weight is skipped and the draw
/// is plain Poisson.
///
/// At a fraction of a count per cell the mixture is indistinguishable from
/// Poisson in the only thing that survives into the store, namely whether the
/// entry is zero, one or two. Skipping it there removes the gamma from the
/// ~80% of genes that cannot contribute a tail anyway.
const OVERDISPERSION_CUTOFF: f64 = 0.25;

/// Poisson draws at or above this rate go through the rejection sampler.
/// Below it, inversion is cheaper than constructing a distribution.
const POISSON_INVERSION_LIMIT: f64 = 30.0;

//////////////////////
// Store generation //
//////////////////////

/// Per-gene expression share and per-cell library size for the count model.
struct CountModel {
    /// Fraction of a cell's library taken by each gene. Sums to one.
    gene_share: Vec<f64>,
    /// Expected total UMIs per cell.
    library_size: Vec<f64>,
}

/// Draw the gene shares and library sizes.
///
/// Shares are log-normal, rescaled so the non-dominant genes hold
/// `1 - DOMINANT_SHARE` between them and the dominant genes split
/// [`DOMINANT_SHARE`] as `1 / rank`.
///
/// ### Params
///
/// * `n_genes` - Number of genes
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// The model.
fn build_model(n_genes: usize, n_cells: usize) -> CountModel {
    let mut rng = SmallRng::seed_from_u64(SEED);

    let shares = LogNormal::new(0.0, EXPRESSION_LOG_SD).expect("valid log-normal");
    let mut gene_share: Vec<f64> = (0..n_genes).map(|_| shares.sample(&mut rng)).collect();

    let n_dominant = N_DOMINANT_GENES.min(n_genes);
    let background: f64 = gene_share[n_dominant..].iter().sum();
    let scale = if background > 0.0 {
        (1.0 - DOMINANT_SHARE) / background
    } else {
        0.0
    };
    for share in gene_share[n_dominant..].iter_mut() {
        *share *= scale;
    }

    let harmonic: f64 = (1..=n_dominant).map(|rank| 1.0 / rank as f64).sum();
    for (rank, share) in gene_share[..n_dominant].iter_mut().enumerate() {
        *share = DOMINANT_SHARE / (harmonic * (rank + 1) as f64);
    }

    let libraries =
        LogNormal::new(MEDIAN_LIBRARY_SIZE.ln(), LIBRARY_SIZE_LOG_SD).expect("valid log-normal");
    let library_size: Vec<f64> = (0..n_cells).map(|_| libraries.sample(&mut rng)).collect();

    CountModel {
        gene_share,
        library_size,
    }
}

/// Draw a Poisson count.
///
/// Inversion below [`POISSON_INVERSION_LIMIT`], where its expected iteration
/// count is small, and the rejection sampler above it.
///
/// ### Params
///
/// * `rng` - Thread-local generator
/// * `lambda` - The rate
///
/// ### Returns
///
/// The count, or `0` for a non-positive or non-finite rate.
#[inline]
fn poisson_draw(rng: &mut SmallRng, lambda: f64) -> u32 {
    if !lambda.is_finite() || lambda <= 0.0 {
        return 0;
    }

    if lambda < POISSON_INVERSION_LIMIT {
        let u: f64 = rng.random();
        let mut p = (-lambda).exp();
        let mut cumulative = p;
        let mut k = 0u32;
        // The guard is belt and braces against a cumulative that stalls below
        // `u` through rounding; it cannot bite for lambda in this range.
        while u > cumulative && k < 10_000 {
            k += 1;
            p *= lambda / k as f64;
            cumulative += p;
        }
        k
    } else {
        Poisson::new(lambda).map_or(0, |d| d.sample(rng) as u32)
    }
}

/// Generate one gene's counts across every cell.
///
/// ### Params
///
/// * `gene` - Gene index, which also seeds the generator so generation
///   parallelises without changing the output
/// * `model` - Expression shares and library sizes
///
/// ### Returns
///
/// The `CscGeneChunk`, holding only the non-zero cells.
fn draw_gene(gene: usize, model: &CountModel) -> CscGeneChunk {
    let mut rng = SmallRng::seed_from_u64(SEED ^ (gene as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

    let share = model.gene_share[gene];
    let overdispersed = share * MEDIAN_LIBRARY_SIZE > OVERDISPERSION_CUTOFF;
    let phi = BCV * BCV;
    // Shape 1/phi with scale phi has mean 1 and variance phi, so it perturbs
    // the rate without moving the gene's mean expression.
    let gamma = if overdispersed {
        Gamma::new(1.0 / phi, phi).ok()
    } else {
        None
    };

    let mut raw: Vec<u32> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut norms: Vec<F16> = Vec::new();

    for (cell, &library) in model.library_size.iter().enumerate() {
        let weight = gamma.as_ref().map_or(1.0, |g| g.sample(&mut rng));
        let count = poisson_draw(&mut rng, share * library * weight);
        if count == 0 {
            continue;
        }

        raw.push(count);
        indices.push(cell);
        norms.push(F16::from_f32(
            (count as f32 / library as f32 * TARGET_SIZE).ln_1p(),
        ));
    }

    CscGeneChunk::from_conversion(RawCounts::from_u32_auto(&raw), &norms, &indices, gene, true)
}

/// Path of the cached store for a given shape.
///
/// ### Params
///
/// * `n_genes` - Number of genes
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// The path, keyed on shape and seed so a shape change cannot silently reuse
/// the wrong store.
fn store_path(n_genes: usize, n_cells: usize) -> PathBuf {
    std::env::temp_dir().join(format!(
        "bixverse_hvg_bench_{n_genes}x{n_cells}_{SEED:x}.bin"
    ))
}

/// Write the store unless a matching one is already cached.
///
/// ### Params
///
/// * `n_genes` - Number of genes
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// The path of the store.
fn ensure_store(n_genes: usize, n_cells: usize) -> PathBuf {
    let path = store_path(n_genes, n_cells);
    let regenerate = std::env::var("HVG_BENCH_REGEN").is_ok();

    if path.exists() && !regenerate {
        println!("store: reusing {}", path.display());
        return path;
    }

    println!("store: generating {n_genes} genes x {n_cells} cells");
    let start = Instant::now();

    let model = build_model(n_genes, n_cells);
    let mut writer =
        CellGeneSparseWriter::new(&path, false, n_cells, n_genes, TARGET_SIZE).expect("writer");

    for block_start in (0..n_genes).step_by(WRITE_BLOCK) {
        let block_end = (block_start + WRITE_BLOCK).min(n_genes);
        let chunks: Vec<CscGeneChunk> = (block_start..block_end)
            .into_par_iter()
            .map(|gene| draw_gene(gene, &model))
            .collect();

        for chunk in chunks {
            writer.write_gene_chunk(chunk).expect("write gene chunk");
        }
    }

    writer.finalise().expect("finalise");

    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!(
        "store: written in {:.2?}, {:.1} MiB on disk",
        start.elapsed(),
        bytes as f64 / (1024.0 * 1024.0)
    );

    path
}

///////////////////
// Store profile //
///////////////////

/// Per-gene shape statistics of the generated store.
struct StoreProfile {
    /// Non-zero entries per gene.
    nnz: Vec<usize>,
    /// Largest raw count per gene.
    max_count: Vec<u32>,
}

/// Sweep the store once and record its shape.
///
/// ### Params
///
/// * `reader` - Reader over the generated store
/// * `blocks` - Gene blocks to read
///
/// ### Returns
///
/// The profile, gene-major.
fn profile_store<S: SingleCellReading + Sync>(reader: &S, blocks: &[Vec<usize>]) -> StoreProfile {
    let mut profile = StoreProfile {
        nnz: Vec::new(),
        max_count: Vec::new(),
    };

    for block in blocks {
        let genes = reader.read_gene_parallel(block).expect("read block");
        let stats: Vec<(usize, u32)> = genes
            .par_iter()
            .map(|gene| (gene.indices.len(), gene.data_raw.iter().max().unwrap_or(0)))
            .collect();

        for (nnz, max) in stats {
            profile.nnz.push(nnz);
            profile.max_count.push(max);
        }
    }

    profile
}

/// Quantile of a slice by rank, without interpolation.
///
/// ### Params
///
/// * `values` - The values, copied and sorted internally
/// * `q` - Quantile in `0..=1`
///
/// ### Returns
///
/// The value at that rank, or zero for an empty slice.
fn quantile<T: Copy + Ord + Default>(values: &[T], q: f64) -> T {
    if values.is_empty() {
        return T::default();
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

/// Print the shape of the generated store.
///
/// The point is to be able to tell at a glance whether the generator is
/// producing something that looks like 10x data before trusting any timing
/// taken on it.
///
/// ### Params
///
/// * `profile` - Per-gene statistics
/// * `n_cells` - Number of cells
fn report_profile(profile: &StoreProfile, n_cells: usize) {
    let n_genes = profile.nnz.len();
    let total_nnz: usize = profile.nnz.iter().sum();
    let empty = profile.nnz.iter().filter(|&&n| n == 0).count();

    println!();
    println!("store profile");
    println!(
        "  density              {:.2}%",
        100.0 * total_nnz as f64 / (n_genes as f64 * n_cells as f64)
    );
    println!("  empty genes          {empty}");
    println!(
        "  nnz per gene         median {}, p99 {}, max {}",
        quantile(&profile.nnz, 0.5),
        quantile(&profile.nnz, 0.99),
        profile.nnz.iter().max().copied().unwrap_or(0)
    );
    println!(
        "  max count per gene   median {}, p99 {}, max {}",
        quantile(&profile.max_count, 0.5),
        quantile(&profile.max_count, 0.99),
        profile.max_count.iter().max().copied().unwrap_or(0)
    );
    println!(
        "  first-pass state     {:.1} MiB of GeneStats, single batch, no allocations per gene",
        n_genes as f64 * std::mem::size_of::<GeneStats>() as f64 / (1024.0 * 1024.0)
    );
}

//////////////////
// Measurements //
//////////////////

/// Gene index blocks for a store.
///
/// ### Params
///
/// * `n_genes` - Total genes
/// * `block_size` - Genes per block
///
/// ### Returns
///
/// The blocks, ascending.
fn gene_blocks(n_genes: usize, block_size: usize) -> Vec<Vec<usize>> {
    (0..n_genes.div_ceil(block_size))
        .map(|i| ((i * block_size)..((i + 1) * block_size).min(n_genes)).collect())
        .collect()
}

/// Read every block and touch the result, without computing statistics.
///
/// ### Params
///
/// * `reader` - Reader over the store
/// * `blocks` - Gene blocks to read
///
/// ### Returns
///
/// Elapsed wall clock.
fn time_read<S: SingleCellReading + Sync>(reader: &S, blocks: &[Vec<usize>]) -> Duration {
    let start = Instant::now();
    let mut total = 0usize;
    for block in blocks {
        let genes = reader.read_gene_parallel(block).expect("read block");
        total += genes.par_iter().map(|g| g.indices.len()).sum::<usize>();
    }
    black_box(total);
    start.elapsed()
}

/// Read every block and run the first pass exactly as the driver does.
///
/// Minus [`time_read`] this is the entire marginal cost of the statistics:
/// there is no serial reduction behind the parallel sweep and no per-gene
/// allocation inside it.
///
/// ### Params
///
/// * `reader` - Reader over the store
/// * `blocks` - Gene blocks to read
/// * `index` - Cell to slot lookup
/// * `n_genes` - Total genes, for the output array
///
/// ### Returns
///
/// Elapsed wall clock.
fn time_pass1<S: SingleCellReading + Sync>(
    reader: &S,
    blocks: &[Vec<usize>],
    index: &CellBatchIndex,
    n_genes: usize,
) -> Duration {
    let n_batches = index.n_batches();
    let n_slots = index.n_slots();

    let start = Instant::now();
    let mut stats = vec![GeneStats::default(); n_genes * n_batches];

    for block in blocks {
        let genes = reader.read_gene_parallel(block).expect("read block");
        let first = block[0];
        let last = block[block.len() - 1] + 1;

        genes
            .par_iter()
            .zip(stats[first * n_batches..last * n_batches].par_chunks_mut(n_batches))
            .for_each_init(
                || HistogramScratch::new(n_slots),
                |scratch, (gene, out)| gene_stats(gene, index, scratch, out),
            );
    }

    let elapsed = start.elapsed();
    black_box(&stats);
    elapsed
}

/// Report how often the clip actually reaches a gene.
///
/// When no value is clipped, every standardised value is `(x - mean) / sd`, so
/// the standardised mean is zero and `var_std` collapses to
/// `var / expected_var`. Only the genes the clip reaches are re-read, so this
/// count is exactly the size of the driver's second read.
///
/// ### Params
///
/// * `res` - Result of a single-batch `run_hvg_vst`
/// * `max_count` - Largest raw count per gene
/// * `clip` - The clip applied
fn report_clip(res: &HvgRes, max_count: &[u32], clip: f32) {
    let n_genes = res.mean.len();
    let mut clipped = 0usize;
    let mut worst_relative = 0f64;
    let mut compared = 0usize;

    for (gene, &gene_max) in max_count.iter().enumerate().take(n_genes) {
        let expected_var = 10_f64.powf(res.var_exp[gene]);
        let expected_sd = expected_var.sqrt();
        if !expected_sd.is_finite() || expected_sd <= 0.0 {
            continue;
        }

        let mean = res.mean[gene];
        let reach = clip as f64 * expected_sd;
        let bites = (gene_max as f64 - mean) > reach || mean > reach;

        if bites {
            clipped += 1;
            continue;
        }

        let closed_form = res.var[gene] / expected_var;
        let actual = res.var_std[gene];
        if actual.abs() > 1e-9 {
            let relative = ((closed_form - actual) / actual).abs();
            worst_relative = worst_relative.max(relative);
            compared += 1;
        }
    }

    println!();
    println!("clip statistics (single batch, clip = {clip:.1})");
    println!(
        "  genes the clip reaches   {clipped} of {n_genes} ({:.2}%), i.e. the re-read size",
        100.0 * clipped as f64 / n_genes as f64
    );
    println!(
        "  genes on the closed form {} ({:.2}%)",
        n_genes - clipped,
        100.0 * (n_genes - clipped) as f64 / n_genes as f64
    );
    println!(
        "  var / expected_var vs driver var_std over {compared} unclipped genes: \
         worst relative error {worst_relative:.3e}"
    );
}

//////////
// Main //
//////////

/// Run a measurement [`REPEATS`] times and keep the fastest.
///
/// These phases land in the tens of milliseconds, where a single shot is at the
/// mercy of whatever else the machine is doing. The fastest run is the one with
/// the least interference, so it is the one worth comparing.
///
/// ### Params
///
/// * `run` - Whether the cell is selected at all
/// * `measure` - The measurement, called repeatedly
///
/// ### Returns
///
/// The fastest observed duration, or zero when the cell is not selected.
fn best_of(run: bool, mut measure: impl FnMut() -> Duration) -> Duration {
    if !run {
        return Duration::ZERO;
    }
    (0..REPEATS)
        .map(|_| measure())
        .min()
        .unwrap_or(Duration::ZERO)
}

/// Read a `usize` from the environment, falling back to a default.
///
/// ### Params
///
/// * `key` - Environment variable name
/// * `fallback` - Value to use when unset or unparseable
///
/// ### Returns
///
/// The resolved value.
fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Whether a cell should run, given `HVG_BENCH_ONLY`.
///
/// ### Params
///
/// * `label` - The cell's label
///
/// ### Returns
///
/// `true` when the filter is unset or matches.
fn selected(label: &str) -> bool {
    match std::env::var("HVG_BENCH_ONLY") {
        Ok(filter) => filter.split(',').any(|f| label.contains(f.trim())),
        Err(_) => true,
    }
}

fn main() {
    let n_genes = env_usize("HVG_BENCH_GENES", DEFAULT_GENES);
    let n_cells = env_usize("HVG_BENCH_CELLS", DEFAULT_CELLS);

    let path = ensure_store(n_genes, n_cells);
    let reader = ParallelSparseReader::new(path.to_str().expect("utf-8 path")).expect("reader");
    let blocks = gene_blocks(n_genes, GENE_BATCH_SIZE);

    // warm the page cache so every cell measures the same thing
    let warm = time_read(&reader, &blocks);
    println!("warm-up read: {warm:.2?}");

    let profile = profile_store(&reader, &blocks);
    report_profile(&profile, n_cells);

    let cells: Vec<usize> = (0..n_cells).collect();

    println!();
    println!("phase decomposition, all cells selected");
    println!("  {:<12} {:>10} {:>10}", "batches", "read", "pass1");

    for &n_batches in BATCH_SWEEP.iter() {
        let labels: Vec<usize> = (0..n_cells).map(|c| c % n_batches).collect();
        let index = CellBatchIndex::new(n_cells, &cells, Some(&labels)).expect("index");

        let read = best_of(selected("read"), || time_read(&reader, &blocks));
        let pass1 = best_of(selected("pass1"), || {
            time_pass1(&reader, &blocks, &index, n_genes)
        });

        println!("  {n_batches:<12} {read:>10.2?} {pass1:>10.2?}");
    }

    if selected("endtoend") {
        println!();
        println!("end to end, driver's own phase split");

        for &n_batches in BATCH_SWEEP.iter() {
            let labels: Vec<usize> = (0..n_cells).map(|c| c % n_batches).collect();
            let index = CellBatchIndex::new(n_cells, &cells, Some(&labels)).expect("index");

            for (name, gene_batch_size) in
                [("one block", None), ("streaming", Some(GENE_BATCH_SIZE))]
            {
                let opts = HvgRunOpts {
                    gene_batch_size,
                    verbose: 1,
                };
                println!();
                println!("  -- {n_batches} batch(es), {name}");
                let start = Instant::now();
                let res = run_hvg_vst(&reader, &index, LOESS_SPAN, None, opts).expect("vst");
                println!("  total {:.2?}", start.elapsed());
                black_box(&res);
            }
        }
    }

    if selected("clip") {
        let index = CellBatchIndex::new(n_cells, &cells, None).expect("index");
        let opts = HvgRunOpts {
            gene_batch_size: Some(GENE_BATCH_SIZE),
            verbose: 0,
        };
        let res = run_hvg_vst(&reader, &index, LOESS_SPAN, None, opts).expect("vst");
        report_clip(&res[0], &profile.max_count, (n_cells as f32).sqrt());
    }
}
