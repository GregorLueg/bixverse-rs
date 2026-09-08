//! Where the CellSweep wall clock actually goes.
//!
//! The fit is `nnz x max_iter` in the E-step and `n_genes x n_celltypes x
//! max_iter` in the M-step, so on any realistic shape the E-step is the whole
//! measurement. `e_step` and `fit_em` are private, so the cells go through
//! `run_cellsweep`, and each one runs the fit twice, at one iteration and at
//! `MAX_ITER`. The store read, `build_sample_csr`, the denoise pass and the
//! write do not scale with the iteration count and are the larger half of the
//! wall clock at this shape, so subtracting the one-iteration run is what
//! leaves the marginal cost of an EM iteration on its own.
//!
//! The measured cells:
//!
//! * `build` - synthesise and write the input store. Not part of the fit;
//!   reported so it can be subtracted by eye.
//! * `fit` - default `alpha_cap`. The ordinary path: most real barcodes stay
//!   inside the cap, so the cell-type reassignment runs on few of them.
//! * `reassign` - `alpha_cap` pushed low enough that nearly every real barcode
//!   is excluded, so the reassignment runs on all of them every iteration.
//!   `reassign` minus `fit` is the reassignment's marginal cost, which is
//!   `n_celltypes` passes over each row on top of the E-step.
//! * `one_type` - `fit` with a single cell type, so `p_numer` shrinks from
//!   `n_celltypes * n_genes` f64 to `n_genes`. Answers whether the scattered
//!   `p_numer` accumulate is missing L1. At the default shape it is worth about
//!   8%, so it is not where the time goes.
//! * `ln_kernel` - `ln_dot_simd` against the libm loop it replaces, plus the
//!   responsibility-split divide on its own. Both timed without the
//!   surrounding gathers.
//!
//! A caveat on `ln_kernel`, learnt the hard way. It times the `ln` in a
//! latency-bound accumulator loop with nothing else in it, and that overstates
//! its share of a real row: the E-step has enough independent work around the
//! `ln` for the out-of-order engine to hide most of the libm call behind it.
//! The kernel measures a clean 2x here and delivers roughly that on the
//! reassignment path, which really is `ln`-bound, but only a fraction of it on
//! the ordinary E-step row. Do not read a micro-benchmark ratio as an
//! end-to-end one.
//!
//! Iteration counts are pinned rather than left to the stopping rule: both
//! tolerances are set to zero so the loop always runs `max_iter` times and the
//! two runs being compared do the same work.
//!
//! Run with:
//! ```
//! cargo bench --features single-cell --bench cellsweep_bench
//! ```
//!
//! `CELLSWEEP_BENCH_ONLY=fit,reassign` runs a subset of the cells.
//! `RAYON_NUM_THREADS=1` is worth a run of its own: the default sweep chunks by
//! row count, and real barcodes carry an order of magnitude more non-zeros than
//! empty droplets and come first, so the parallel numbers carry a load
//! imbalance the single-threaded ones do not.

#![cfg(feature = "single-cell")]

use std::hint::black_box;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::rngs::SmallRng;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_data::data_io::CellGeneSparseWriter;
use bixverse_rs::single_cell::sc_processing::cellsweep::{
    CellSweepParams, CellSweepSample, run_cellsweep,
};

////////////
// Shapes //
////////////

/// Genes. Smaller than a filtered 10x feature set, because the per-gene work
/// is the M-step and the point of the bench is the E-step.
const N_GENES: usize = 5_000;

/// Real, annotated barcodes.
const N_REAL: usize = 4_000;

/// Empty droplets. Enough that the empty rows are a real share of nnz without
/// the store generation dominating the run.
const N_EMPTY: usize = 20_000;

/// Cell types. `best_celltype` is linear in this.
const N_CELLTYPES: usize = 8;

/// Non-zeros per real barcode.
const NNZ_REAL: usize = 400;

/// Non-zeros per empty droplet.
const NNZ_EMPTY: usize = 50;

/// EM iterations. Pinned, see the module docs.
const MAX_ITER: usize = 100;

/// Repeats per timed cell. The fastest run is reported.
const REPEATS: usize = 3;

/// Library size the `data_norm` layer is scaled against.
const TARGET_SIZE: f32 = 1e4;

/// Seed for the count generator.
const SEED: u64 = 0x5EED_C511_u64;

/// Elements per micro-benchmarked kernel call. A few tiles' worth.
const KERNEL_LEN: usize = 1_024;

/// Kernel calls per micro-benchmark repeat.
const KERNEL_CALLS: usize = 20_000;

/////////////
// Helpers //
/////////////

/// One `(gene indices, counts)` pair per barcode.
type SparseRows = Vec<(Vec<u32>, Vec<u32>)>;

/// RAII guard that removes a scratch file on drop.
struct TempBin(std::path::PathBuf);

/// Drop implementation for [`TempBin`]. Errors are ignored: the file may
/// already be gone.
impl Drop for TempBin {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempBin {
    /// Reserve a uniquely named scratch file in the system temp directory.
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!("bixverse_cellsweep_bench_{name}.bin")))
    }

    /// Path of the guarded file as a `&str`.
    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// Whether a named cell should run, given `CELLSWEEP_BENCH_ONLY`.
fn wanted(cell: &str) -> bool {
    match std::env::var("CELLSWEEP_BENCH_ONLY") {
        Ok(list) => list.split(',').any(|c| c.trim() == cell),
        Err(_) => true,
    }
}

/// Time a closure `REPEATS` times and report the fastest run.
fn time<T>(label: &str, extra: &str, mut f: impl FnMut() -> T) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..REPEATS {
        let start = Instant::now();
        black_box(f());
        best = best.min(start.elapsed());
    }
    println!("  {label:<12} {best:>10.3?}  {extra}");
    best
}

////////////////
// Count data //
////////////////

/// Generate one sample's counts, real barcodes first.
///
/// Cell types get disjoint marker blocks so the profiles are identifiable, and
/// every barcode carries a share of a skewed ambient profile so `alpha` has
/// something to fit. The exact model does not matter for the timing, only the
/// nnz per row and the fact that the EM does not immediately diverge.
///
/// ### Returns
///
/// One `(indices, counts)` pair per barcode, plus the cell-type label of each
/// real barcode.
fn build_counts() -> (SparseRows, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(SEED);

    // Skewed, so it is not absorbable into every profile at once. A flat
    // ambient profile collapses alpha to zero and the fit stops doing work.
    let ambient: Vec<f64> = (0..N_GENES)
        .map(|g| 1.0 + 9.0 * ((g % 97) as f64 / 97.0))
        .collect();
    let ambient_total: f64 = ambient.iter().sum();
    let ambient: Vec<f64> = ambient.iter().map(|a| a / ambient_total).collect();

    let block = N_GENES / N_CELLTYPES;

    let mut rows = Vec::with_capacity(N_REAL + N_EMPTY);
    let mut labels = Vec::with_capacity(N_REAL);

    for i in 0..N_REAL {
        let k = i % N_CELLTYPES;
        let marker = k * block..(k + 1) * block;

        // Half the non-zeros land in the barcode's own marker block, half are
        // ambient spillover spread across the whole gene space.
        let mut genes: Vec<u32> = Vec::with_capacity(NNZ_REAL);
        while genes.len() < NNZ_REAL / 2 {
            genes.push(rng.random_range(marker.clone()) as u32);
        }
        while genes.len() < NNZ_REAL {
            genes.push(rng.random_range(0..N_GENES) as u32);
        }
        genes.sort_unstable();
        genes.dedup();

        let counts: Vec<u32> = genes
            .iter()
            .map(|&g| {
                let g = g as usize;
                let signal = if marker.contains(&g) { 20.0 } else { 0.0 };
                1 + (signal + 2_000.0 * ambient[g]).round() as u32
            })
            .collect();

        rows.push((genes, counts));
        labels.push(k);
    }

    for _ in 0..N_EMPTY {
        let mut genes: Vec<u32> = (0..NNZ_EMPTY)
            .map(|_| rng.random_range(0..N_GENES) as u32)
            .collect();
        genes.sort_unstable();
        genes.dedup();
        let counts: Vec<u32> = genes
            .iter()
            .map(|&g| 1 + (500.0 * ambient[g as usize]).round() as u32)
            .collect();
        rows.push((genes, counts));
    }

    (rows, labels)
}

/// Write the generated counts to a cell-based store.
fn write_store(rows: &[(Vec<u32>, Vec<u32>)], path: &str) {
    let mut writer = CellGeneSparseWriter::new(path, true, rows.len(), N_GENES, TARGET_SIZE)
        .expect("writer opens");
    for (i, (indices, counts)) in rows.iter().enumerate() {
        writer
            .write_cell_chunk(CsrCellChunk::from_data(
                counts,
                indices,
                i,
                TARGET_SIZE,
                true,
            ))
            .expect("chunk writes");
    }
    writer.finalise().expect("store finalises");
}

/// Parameters pinned to a fixed iteration count.
///
/// Both stopping tolerances are zero, so neither the log-likelihood check nor
/// the parameter check ever fires and the loop always runs `max_iter` times.
fn pinned_params(alpha_cap: f64, max_iter: usize) -> CellSweepParams {
    CellSweepParams {
        alpha_cap,
        max_iter,
        del0_ll_tol: 0.0,
        min_ll_tol: 0.0,
        tol_p: 0.0,
        tol_f: 0.0,
        ..Default::default()
    }
}

/// Time a fit at one iteration and at [MAX_ITER], and report the difference.
///
/// The end-to-end call carries the store read, `build_sample_csr`, the denoise
/// pass and the write, and none of those scale with the iteration count. At
/// this shape they are the larger half of the wall clock, so a raw end-to-end
/// number buries whatever the EM itself did. Subtracting a one-iteration run
/// cancels all of it and leaves the marginal cost of an EM iteration, which is
/// the only thing worth comparing between two builds.
///
/// ### Params
///
/// * `label` - Cell name.
/// * `extra` - Note printed alongside the timing.
/// * `reader` - Store holding the counts.
/// * `sample` - The sample to fit.
/// * `alpha_cap` - Ceiling on `alpha`. Low values push most barcodes into the
///   excluded set and so into the cell-type reassignment.
/// * `nnz_total` - Non-zeros in the sample, for the per-non-zero figure.
fn time_fit(
    label: &str,
    extra: &str,
    reader: &ParallelSparseReader,
    sample: &CellSweepSample,
    alpha_cap: f64,
    nnz_total: usize,
) {
    let out = TempBin::new(&format!("{label}_out"));
    let run = |max_iter: usize| {
        run_cellsweep(
            reader,
            std::slice::from_ref(sample),
            pinned_params(alpha_cap, max_iter),
            out.path(),
            TARGET_SIZE,
            0,
        )
        .expect("cellsweep runs")
    };

    let fixed = time(
        &format!("{label} (1x)"),
        "one iteration, all the fixed cost",
        || run(1),
    );
    let full = time(label, extra, || run(MAX_ITER));

    let per_iter = full.saturating_sub(fixed) / (MAX_ITER - 1) as u32;
    println!(
        "               {per_iter:>10.3?} per EM iteration, {:>6.2} ns per non-zero",
        per_iter.as_nanos() as f64 / nnz_total as f64
    );
}

//////////
// Main //
//////////

fn main() {
    let nnz_total = NNZ_REAL * N_REAL + NNZ_EMPTY * N_EMPTY;

    println!(
        "CellSweep bench: {N_GENES} genes, {N_REAL} real, {N_EMPTY} empty, \
         {N_CELLTYPES} cell types, <={nnz_total} nnz, {MAX_ITER} iterations"
    );

    let (rows, labels) = build_counts();
    let raw = TempBin::new("raw");

    if wanted("build") {
        time("build", "store generation, not part of the fit", || {
            write_store(&rows, raw.path())
        });
    } else {
        write_store(&rows, raw.path());
    }

    let reader = ParallelSparseReader::new(raw.path()).expect("store opens");
    let sample = CellSweepSample {
        sample_id: "bench".to_string(),
        real_cells: (0..N_REAL).collect(),
        empty_cells: (N_REAL..N_REAL + N_EMPTY).collect(),
        celltype_idx: labels,
        n_celltypes: N_CELLTYPES,
    };

    if wanted("fit") {
        time_fit(
            "fit",
            "default alpha_cap, few reassignments",
            &reader,
            &sample,
            0.9,
            nnz_total,
        );
    }

    if wanted("reassign") {
        time_fit(
            "reassign",
            "alpha_cap 0.05, best_celltype on most rows",
            &reader,
            &sample,
            0.05,
            nnz_total,
        );
    }

    // Same non-zeros, same gathers, one cell type. The only thing that changes
    // is that `p_numer` drops from `n_celltypes * n_genes` f64 to `n_genes`,
    // i.e. from 320 KB per thread to 40 KB, which is the difference between
    // missing L1 on every scattered accumulate and hitting it.
    if wanted("one_type") {
        let flat = CellSweepSample {
            sample_id: "bench".to_string(),
            celltype_idx: vec![0; N_REAL],
            n_celltypes: 1,
            ..sample.clone()
        };
        time_fit(
            "one_type",
            "one cell type, so p_numer fits in L1",
            &reader,
            &flat,
            0.9,
            nnz_total,
        );
    }

    if wanted("ln_kernel") {
        bench_kernels();
    }
}

/// Micro-benchmark of the E-step's inner loops against their scalar forms.
///
/// `ln_dot_simd` is timed through its public dispatch, so the runtime feature
/// check and the tail handling are both in the measurement, which is how the
/// E-step calls it. `divide` is here to check the assumption the design rests
/// on: the responsibility split is element-wise over contiguous slices, so it
/// should already vectorise without a hand-written kernel.
fn bench_kernels() {
    use bixverse_rs::single_cell::sc_utils::simd::ln_dot_simd;

    let mut rng = SmallRng::seed_from_u64(SEED);

    // The range the E-step actually produces: `p` is a mixture of profile
    // entries, so it sits a few orders of magnitude below one.
    let counts: Vec<f64> = (0..KERNEL_LEN)
        .map(|_| rng.random_range(1.0..50.0))
        .collect();
    let p: Vec<f64> = (0..KERNEL_LEN)
        .map(|_| rng.random_range(1e-7..1e-3))
        .collect();
    let mut scale = vec![0.0_f64; KERNEL_LEN];

    let elements = (KERNEL_LEN * KERNEL_CALLS) as f64;
    let per_element = |elapsed: Duration| {
        println!(
            "               {:>8.2} ns per element",
            elapsed.as_nanos() as f64 / elements
        );
    };

    per_element(time("ln_dot", "ln_dot_simd", || {
        let mut acc = 0.0;
        for _ in 0..KERNEL_CALLS {
            acc += ln_dot_simd(black_box(&counts), black_box(&p), 1e-300);
        }
        acc
    }));

    per_element(time("ln_dot_ref", "the libm loop it replaces", || {
        let mut acc = 0.0;
        for _ in 0..KERNEL_CALLS {
            let counts = black_box(&counts);
            let p = black_box(&p);
            for j in 0..KERNEL_LEN {
                acc += counts[j] * p[j].max(1e-300).ln();
            }
        }
        acc
    }));

    per_element(time(
        "divide",
        "responsibility split, why it needs no kernel",
        || {
            for _ in 0..KERNEL_CALLS {
                let counts = black_box(&counts);
                let p = black_box(&p);
                let scale = black_box(&mut scale);
                for j in 0..KERNEL_LEN {
                    scale[j] = counts[j] / p[j].max(1e-12);
                }
            }
        },
    ));
}
