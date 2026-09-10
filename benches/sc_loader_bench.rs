//! Where the wall clock goes when a training loop pulls minibatches off disk.
//!
//! This is the read path the Python bindings expose for scVI-style training:
//! a cell-major store, a shuffled index vector, and a batch of cells decoded
//! per step. It is the path nothing else in the crate benchmarks, and the one
//! whose cost decides whether a GPU sits idle.
//!
//! The measured cells decompose one batch:
//!
//! * `decode` - [`SingleCellReading::read_cells_parallel`] alone. The floor.
//!   One lz4 frame and one `CsrCellChunk` per cell.
//! * `sparse` - `decode` plus concatenating the per-cell CSR fragments. What
//!   the loader does by default.
//! * `dense`  - `decode` plus a scatter into a zero-filled
//!   `(batch, n_genes)` buffer. The opt-in layout, measured so the default is
//!   an informed one rather than a guess.
//! * `epoch`  - a full pass driven the way the bindings drive it: one feeder
//!   thread, a bounded channel, and a consumer that only drains. Exposes
//!   whether the single feeder is the ceiling.
//!
//! Each of the first three runs twice, over a shuffled index vector and a
//! sequential one. The store is mapped with `Advice::Random`, so the gap
//! between the two is the part of the cost that is page-fault locality rather
//! than decompression.
//!
//! Every cell runs after a full warm-up pass, so these are warm page-cache
//! numbers. A cold first read is a different measurement, and the one that
//! decides whether `mmap` or explicit I/O is the right primitive; take it by
//! generating a fresh store and timing the warm-up line.
//!
//! Run with:
//! ```
//! cargo bench --features single-cell --bench sc_loader_bench
//! ```
//!
//! The generated store is cached in the temp directory and keyed on shape and
//! seed, so only the first run pays for it. `SC_LOADER_BENCH_REGEN=1` forces a
//! rebuild. `SC_LOADER_BENCH_CELLS` / `SC_LOADER_BENCH_GENES` /
//! `SC_LOADER_BENCH_BATCH` change the shape, and
//! `SC_LOADER_BENCH_ONLY=decode,sparse` runs a subset of the cells.

#![cfg(feature = "single-cell")]

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand::rngs::{SmallRng, StdRng};
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Gamma, LogNormal, Poisson};
use rayon::prelude::*;

use bixverse_rs::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CsrCellChunk, DataLayerReturn, ParallelSparseReader, SingleCellReading,
};

////////////
// Shapes //
////////////

/// Default cell count. Large enough that a shuffled pass misses the cache
/// between batches, small enough that the store builds in under a minute.
/// Push it to 200_000 via `SC_LOADER_BENCH_CELLS` for a production shape.
const DEFAULT_CELLS: usize = 20_000;

/// Default gene count. Roughly a filtered 10x feature set.
const DEFAULT_GENES: usize = 20_000;

/// Default cells per batch. The scVI default, and what the bindings default to.
const DEFAULT_BATCH: usize = 256;

/// Library size the `data_norm` layer is scaled against.
const TARGET_SIZE: f32 = 1e4;

/// Seed for the generator. Fixed so the cached store is reproducible.
const SEED: u64 = 0x5EED_0BEEF_u64;

/// Genes generated per write block. Bounds peak occupancy during generation
/// while still giving rayon enough to chew on.
const WRITE_BLOCK: usize = 1_024;

/////////////////
// Count model //
/////////////////

// Calibrated to 10x droplet data rather than to a uniform draw. The crate's
// own `create_celltype_sparse_csr_data` comes out around 65% dense, and the
// whole question this bench asks is how a 5-10% dense store behaves.

/// Median library size, i.e. total UMIs per cell.
const MEDIAN_LIBRARY_SIZE: f64 = 5_000.0;

/// Log-scale spread of the library size. 0.35 gives roughly a 4x spread
/// between the 1st and 99th percentile cell, which is typical post-QC.
const LIBRARY_SIZE_LOG_SD: f64 = 0.35;

/// Log-scale spread of per-gene relative expression, before normalisation.
///
/// Shares are drawn log-normal then rescaled to sum to one, so only the spread
/// matters. 2.0 puts the per-gene mean across roughly four orders of
/// magnitude, which is what gives realistic sparsity.
const EXPRESSION_LOG_SD: f64 = 2.0;

/// Number of dominant genes, standing in for MALAT1, the ribosomal proteins
/// and the mito genes.
const N_DOMINANT_GENES: usize = 40;

/// Share of every cell's library taken by the dominant genes, split as
/// `1 / rank`.
const DOMINANT_SHARE: f64 = 0.25;

/// Biological coefficient of variation. The gamma mixing weight has variance
/// `BCV^2`, so this sets the overdispersion on top of Poisson noise.
const BCV: f64 = 0.55;

/// Repeats per timed cell. The fastest run is reported, so a scheduler hiccup
/// cannot make a change look like a regression.
const REPEATS: usize = 5;

/// Batches held in flight by the feeder in the `epoch` cell. Matches the
/// bindings' default.
const PREFETCH: usize = 4;

//////////////////////
// Store generation //
//////////////////////

/// Per-gene expression shares and per-cell library sizes.
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
/// * `n_cells` - Number of cells
/// * `n_genes` - Number of genes
///
/// ### Returns
///
/// The model.
fn build_model(n_cells: usize, n_genes: usize) -> CountModel {
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

/// Draw one cell's counts as a chunk.
///
/// Each cell seeds its own generator from the cell index, so generation
/// parallelises without the draw depending on scheduling order.
///
/// ### Params
///
/// * `cell` - Cell index
/// * `model` - The shared count model
///
/// ### Returns
///
/// The cell's `CsrCellChunk`.
fn draw_cell(cell: usize, model: &CountModel) -> CsrCellChunk {
    let mut rng = SmallRng::seed_from_u64(SEED ^ (cell as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

    let library = model.library_size[cell];
    let phi = BCV * BCV;
    // Shape 1/phi with scale phi has mean 1 and variance phi, so it perturbs
    // the rate without moving the gene's mean expression.
    let gamma = Gamma::new(1.0 / phi, phi).expect("valid gamma");

    let mut counts: Vec<u32> = Vec::new();
    let mut cols: Vec<u32> = Vec::new();

    for (gene, &share) in model.gene_share.iter().enumerate() {
        let lambda = share * library * gamma.sample(&mut rng);
        if lambda <= 0.0 {
            continue;
        }
        let count = Poisson::new(lambda)
            .map(|p| p.sample(&mut rng) as u32)
            .unwrap_or(0);
        if count == 0 {
            continue;
        }
        counts.push(count);
        cols.push(gene as u32);
    }

    // A wholly empty cell is a different edge case and not what this measures.
    if counts.is_empty() {
        counts.push(1);
        cols.push(0);
    }

    CsrCellChunk::from_data(&counts, &cols, cell, TARGET_SIZE, true)
}

/// Path of the cached store for a given shape.
///
/// ### Params
///
/// * `n_cells` - Number of cells
/// * `n_genes` - Number of genes
///
/// ### Returns
///
/// The path, keyed on shape and seed so a shape change cannot silently reuse
/// the wrong store.
fn store_path(n_cells: usize, n_genes: usize) -> PathBuf {
    std::env::temp_dir().join(format!(
        "bixverse_sc_loader_bench_{n_cells}x{n_genes}_{SEED:x}.bin"
    ))
}

/// Write a cell-major store unless a matching one is already cached.
///
/// ### Params
///
/// * `n_cells` - Number of cells
/// * `n_genes` - Number of genes
///
/// ### Returns
///
/// Path to the store.
fn ensure_store(n_cells: usize, n_genes: usize) -> PathBuf {
    let path = store_path(n_cells, n_genes);
    let regenerate = std::env::var("SC_LOADER_BENCH_REGEN").is_ok();

    if path.exists() && !regenerate {
        println!("store: reusing {}", path.display());
        return path;
    }

    println!("store: generating {n_cells} cells x {n_genes} genes");
    let start = Instant::now();

    let model = build_model(n_cells, n_genes);
    let mut writer = CellGeneSparseWriter::new(&path, true, n_cells, n_genes, TARGET_SIZE)
        .expect("writer construction");

    for block_start in (0..n_cells).step_by(WRITE_BLOCK) {
        let block_end = (block_start + WRITE_BLOCK).min(n_cells);
        let chunks: Vec<CsrCellChunk> = (block_start..block_end)
            .into_par_iter()
            .map(|cell| draw_cell(cell, &model))
            .collect();

        for chunk in chunks {
            writer.write_cell_chunk(chunk).expect("write cell chunk");
        }
    }

    writer.finalise().expect("finalise");
    println!("store: written in {:.2?}", start.elapsed());

    path
}

////////////////////
// Materialisers  //
////////////////////

/// Concatenate per-cell CSR fragments, as the bindings' default path does.
///
/// Mirrors `bixverse-py`'s sparse materialiser rather than calling it: the
/// bindings are a separate crate and a bench must not depend on them.
///
/// ### Params
///
/// * `chunks` - Decompressed cell chunks in batch order
///
/// ### Returns
///
/// The `(data, indices, indptr)` triple, normalised layer.
fn materialise_sparse(chunks: &[CsrCellChunk]) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
    let nnz: usize = chunks.iter().map(|c| c.indices.len()).sum();
    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);
    let mut indptr = Vec::with_capacity(chunks.len() + 1);

    let mut offset = 0u32;
    indptr.push(offset);
    for chunk in chunks {
        data.extend(chunk.data_norm.iter().map(|v| v.to_f32()));
        indices.extend_from_slice(&chunk.indices);
        offset += chunk.indices.len() as u32;
        indptr.push(offset);
    }

    (data, indices, indptr)
}

/// Scatter the chunks into a zero-filled dense row-major buffer.
///
/// ### Params
///
/// * `chunks` - Decompressed cell chunks in batch order
/// * `n_genes` - Column stride
///
/// ### Returns
///
/// The flat `(chunks.len() * n_genes)` buffer.
fn materialise_dense(chunks: &[CsrCellChunk], n_genes: usize) -> Vec<f32> {
    let mut out = vec![0f32; chunks.len() * n_genes];
    for (row, chunk) in chunks.iter().enumerate() {
        let base = row * n_genes;
        for (value, &col) in chunk.data_norm.iter().zip(chunk.indices.iter()) {
            out[base + col as usize] = value.to_f32();
        }
    }
    out
}

///////////////
// Measuring //
///////////////

/// What one timed pass over the index vector cost, and how much it moved.
#[derive(Clone, Copy)]
struct Pass {
    /// Wall clock for the whole pass.
    elapsed: Duration,
    /// Non-zeros decoded across the pass.
    nnz: usize,
    /// Decompressed chunk bytes across the pass.
    bytes: usize,
}

/// Decompressed payload size of one chunk, per the on-disk layout.
///
/// Header plus raw counts at their stored element width, plus the f16 norm
/// layer, plus the u32 gene indices.
///
/// ### Params
///
/// * `chunk` - The decoded chunk
///
/// ### Returns
///
/// Bytes the chunk occupied after decompression.
fn chunk_bytes(chunk: &CsrCellChunk) -> usize {
    const HEADER: usize = 32;
    let raw = chunk.data_raw.len() * chunk.data_raw.elem_size().max(2) as usize;
    HEADER + raw + chunk.data_norm.len() * 2 + chunk.indices.len() * 4
}

/// Time one pass over the index vector in batches.
///
/// ### Params
///
/// * `reader` - The open store
/// * `order` - Cell indices, in the order the pass visits them
/// * `batch_size` - Cells per batch
/// * `materialise` - What to do with each decoded batch
///
/// ### Returns
///
/// The pass timing and volume.
fn time_pass(
    reader: &ParallelSparseReader,
    order: &[usize],
    batch_size: usize,
    mut materialise: impl FnMut(&[CsrCellChunk]),
) -> Pass {
    let mut nnz = 0usize;
    let mut bytes = 0usize;

    let start = Instant::now();
    for batch in order.chunks(batch_size) {
        let chunks = reader.read_cells_parallel(batch).expect("read batch");
        for chunk in &chunks {
            nnz += chunk.indices.len();
            bytes += chunk_bytes(chunk);
        }
        materialise(&chunks);
    }
    let elapsed = start.elapsed();

    Pass {
        elapsed,
        nnz,
        bytes,
    }
}

/// Time one pass over the index vector through the fused CSR reader.
///
/// The comparison for `sparse`: same output, but decompression and
/// materialisation are one walk over the bytes rather than two.
///
/// ### Params
///
/// * `reader` - The open store
/// * `order` - Cell indices, in the order the pass visits them
/// * `batch_size` - Cells per batch
///
/// ### Returns
///
/// The pass timing and volume.
fn time_fused(reader: &ParallelSparseReader, order: &[usize], batch_size: usize) -> Pass {
    let mut nnz = 0usize;

    let start = Instant::now();
    for batch in order.chunks(batch_size) {
        let csr = reader
            .read_cells_csr(batch, &DataLayerReturn::Norm)
            .expect("read batch");
        nnz += csr.indices.len();
        black_box(&csr);
    }
    let elapsed = start.elapsed();

    Pass {
        elapsed,
        nnz,
        bytes: 0,
    }
}

/// Run a pass `REPEATS` times and keep the fastest.
///
/// ### Params
///
/// * `run` - Whether this cell is selected at all
/// * `measure` - The pass
///
/// ### Returns
///
/// The fastest pass, or a zeroed one when the cell is skipped.
fn best_of(run: bool, mut measure: impl FnMut() -> Pass) -> Option<Pass> {
    if !run {
        return None;
    }
    (0..REPEATS).map(|_| measure()).min_by_key(|p| p.elapsed)
}

/// Print one result row as wall clock, cells/s, nnz/s and MB/s.
///
/// ### Params
///
/// * `label` - Cell name
/// * `pass` - The measured pass, or `None` when skipped
/// * `n_cells` - Cells visited in the pass
fn report(label: &str, pass: Option<Pass>, n_cells: usize) {
    let Some(pass) = pass else { return };
    let secs = pass.elapsed.as_secs_f64();
    println!(
        "  {:<20} {:>10.2?} {:>14.0} {:>16.0} {:>12.1}",
        label,
        pass.elapsed,
        n_cells as f64 / secs,
        pass.nnz as f64 / secs,
        pass.bytes as f64 / secs / (1024.0 * 1024.0),
    );
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

/// Whether a cell should run, given `SC_LOADER_BENCH_ONLY`.
///
/// ### Params
///
/// * `label` - The cell's label
///
/// ### Returns
///
/// `true` when the filter is unset or matches.
fn selected(label: &str) -> bool {
    match std::env::var("SC_LOADER_BENCH_ONLY") {
        Ok(filter) => filter.split(',').any(|f| label.contains(f.trim())),
        Err(_) => true,
    }
}

//////////
// Main //
//////////

fn main() {
    let n_cells = env_usize("SC_LOADER_BENCH_CELLS", DEFAULT_CELLS);
    let n_genes = env_usize("SC_LOADER_BENCH_GENES", DEFAULT_GENES);
    let batch_size = env_usize("SC_LOADER_BENCH_BATCH", DEFAULT_BATCH);

    let path = ensure_store(n_cells, n_genes);
    let reader = ParallelSparseReader::new(path.to_str().expect("utf-8 path")).expect("reader");

    let sequential: Vec<usize> = (0..n_cells).collect();
    let mut shuffled = sequential.clone();
    shuffled.shuffle(&mut StdRng::seed_from_u64(SEED));

    // Warm the page cache so every cell measures the same thing. On a freshly
    // generated store this line is the cold number.
    let warm = time_pass(&reader, &sequential, batch_size, |c| {
        black_box(c);
    });
    println!(
        "warm-up pass: {:.2?} over {} cells, {} nnz, {:.1} MB decompressed",
        warm.elapsed,
        n_cells,
        warm.nnz,
        warm.bytes as f64 / (1024.0 * 1024.0),
    );

    println!();
    println!("batch size {batch_size}, best of {REPEATS}");
    println!(
        "  {:<20} {:>10} {:>14} {:>16} {:>12}",
        "cell", "wall", "cells/s", "nnz/s", "MB/s"
    );

    for (pattern, order) in [("shuffled", &shuffled), ("sequential", &sequential)] {
        report(
            &format!("decode {pattern}"),
            best_of(selected("decode"), || {
                time_pass(&reader, order, batch_size, |c| {
                    black_box(c);
                })
            }),
            n_cells,
        );
        report(
            &format!("sparse {pattern}"),
            best_of(selected("sparse"), || {
                time_pass(&reader, order, batch_size, |c| {
                    black_box(materialise_sparse(c));
                })
            }),
            n_cells,
        );
        report(
            &format!("fused {pattern}"),
            best_of(selected("fused"), || time_fused(&reader, order, batch_size)),
            n_cells,
        );
        report(
            &format!("dense {pattern}"),
            best_of(selected("dense"), || {
                time_pass(&reader, order, batch_size, |c| {
                    black_box(materialise_dense(c, n_genes));
                })
            }),
            n_cells,
        );
    }

    if selected("epoch") {
        println!();
        println!("full epoch through a feeder thread, prefetch {PREFETCH}");

        let reader = Arc::new(reader);
        let elapsed = best_of(true, || {
            let order = shuffled.clone();
            let feeder_reader = Arc::clone(&reader);
            let (tx, rx) =
                std::sync::mpsc::sync_channel::<(Vec<f32>, Vec<u32>, Vec<u32>)>(PREFETCH);

            let handle = std::thread::spawn(move || {
                for batch in order.chunks(batch_size) {
                    let chunks = feeder_reader
                        .read_cells_parallel(batch)
                        .expect("read batch");
                    if tx.send(materialise_sparse(&chunks)).is_err() {
                        break;
                    }
                }
            });

            let start = Instant::now();
            let mut nnz = 0usize;
            for batch in rx {
                nnz += batch.1.len();
                black_box(&batch);
            }
            let elapsed = start.elapsed();
            handle.join().expect("feeder");

            Pass {
                elapsed,
                nnz,
                bytes: 0,
            }
        });

        if let Some(pass) = elapsed {
            let secs = pass.elapsed.as_secs_f64();
            println!(
                "  {:<20} {:>10.2?} {:>14.0} {:>16.0}",
                "epoch shuffled",
                pass.elapsed,
                n_cells as f64 / secs,
                pass.nnz as f64 / secs,
            );
        }
    }
}
