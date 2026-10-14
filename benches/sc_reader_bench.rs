//! Chunk path against the fused batch readers, on both store orientations.
//!
//! The chunk path is `read_*_parallel` handing back one owned chunk per cell
//! or gene, then whatever the consumer does with it. The fused path is
//! `read_cells_sparse` / `read_genes_sparse` appending straight into one
//! compressed sparse matrix. Each case runs both on the same input:
//!
//! * `gene_csc`     - every gene, a random half of the cells kept, into CSC
//!   (NMF, PCA, NEBULA). Chunk side: `read_gene_parallel_filtered` plus
//!   `from_gene_chunks`.
//! * `gene_csc_all` - same without a cell subset.
//! * `gene_stream`  - batches of [`GENE_BATCH`] genes, a per-gene sum of the
//!   normalised layer (HVG, scTransform, residuals).
//! * `cell_csr`     - every cell into CSR, raw `u32` (MetaCells2 piles).
//! * `cell_stream`  - batches of [`CELL_BATCH`] cells, a per-cell sum of raw
//!   counts (QC, VISION, module scores).
//!
//! Warm page cache, fastest of [`REPEATS`] runs.
//!
//! Run with:
//! ```
//! cargo bench --features single-cell --bench sc_reader_bench
//! ```
//!
//! `SC_READER_BENCH_CELLS` / `SC_READER_BENCH_GENES` change the shape. Stores
//! are cached in the temp directory, keyed on shape; `SC_READER_BENCH_REGEN=1`
//! rebuilds them.

#![cfg(feature = "single-cell")]

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use indexmap::IndexSet;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, LogNormal, Poisson};
use rayon::prelude::*;

use bixverse_rs::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CsrCellChunk, DataLayerReturn, ParallelSparseReader, SingleCellReading,
    from_cell_chunks, from_gene_chunks,
};
use bixverse_rs::single_cell::sc_data::gene_file_io::write_gene_file;

/// Default cell count.
const DEFAULT_CELLS: usize = 20_000;

/// Default gene count. Roughly a filtered 10x feature set.
const DEFAULT_GENES: usize = 20_000;

/// Genes per streaming batch, as in HVG, scTransform and MAGIC.
const GENE_BATCH: usize = 1_000;

/// Cells per streaming batch, as `CELL_BATCH_SIZE` in the crate.
const CELL_BATCH: usize = 50_000;

/// Median library size of a generated cell.
const MEDIAN_LIBRARY_SIZE: f64 = 5_000.0;

/// Log-scale spread of per-gene relative expression. 2.0 spreads gene means
/// over about four orders of magnitude, which gives 10x-like sparsity.
const EXPRESSION_LOG_SD: f64 = 2.0;

/// Library size the normalised layer is scaled against.
const TARGET_SIZE: f32 = 1e4;

/// Seed for the generator and the cell subset.
const SEED: u64 = 0x5EED_2EAD;

/// Repeats per timed case; the fastest is reported.
const REPEATS: usize = 5;

//////////////////////
// Store generation //
//////////////////////

/// Write the cell-major store and its gene-major twin, unless cached.
///
/// ### Params
///
/// * `n_cells` - Number of cells
/// * `n_genes` - Number of genes
///
/// ### Returns
///
/// `(cell_path, gene_path)`.
fn ensure_stores(n_cells: usize, n_genes: usize) -> (PathBuf, PathBuf) {
    let stem = format!("bixverse_sc_reader_bench_{n_cells}x{n_genes}_{SEED:x}");
    let cell_path = std::env::temp_dir().join(format!("{stem}_cells.bin"));
    let gene_path = std::env::temp_dir().join(format!("{stem}_genes.bin"));
    let regen = std::env::var("SC_READER_BENCH_REGEN").is_ok();

    if cell_path.exists() && gene_path.exists() && !regen {
        return (cell_path, gene_path);
    }

    let start = Instant::now();
    let mut rng = SmallRng::seed_from_u64(SEED);
    let shares = LogNormal::new(0.0, EXPRESSION_LOG_SD).expect("valid log-normal");
    let mut gene_share: Vec<f64> = (0..n_genes).map(|_| shares.sample(&mut rng)).collect();
    let total: f64 = gene_share.iter().sum();
    gene_share.iter_mut().for_each(|s| *s /= total);

    let mut writer = CellGeneSparseWriter::new(&cell_path, true, n_cells, n_genes, TARGET_SIZE)
        .expect("writer construction");
    for block in (0..n_cells).collect::<Vec<_>>().chunks(4_096) {
        let chunks: Vec<CsrCellChunk> = block
            .par_iter()
            .map(|&cell| {
                let mut rng = SmallRng::seed_from_u64(SEED ^ (cell as u64 + 1));
                let library = LogNormal::new(MEDIAN_LIBRARY_SIZE.ln(), 0.35)
                    .expect("valid log-normal")
                    .sample(&mut rng);
                let mut counts = Vec::new();
                let mut cols = Vec::new();
                for (gene, &share) in gene_share.iter().enumerate() {
                    let count = Poisson::new(share * library)
                        .map(|p| p.sample(&mut rng) as u32)
                        .unwrap_or(0);
                    if count > 0 {
                        counts.push(count);
                        cols.push(gene as u32);
                    }
                }
                CsrCellChunk::from_data(&counts, &cols, cell, TARGET_SIZE, true)
            })
            .collect();
        for chunk in chunks {
            writer.write_cell_chunk(chunk).expect("write cell chunk");
        }
    }
    writer.finalise().expect("finalise");

    write_gene_file(
        cell_path.to_str().expect("utf-8 path"),
        gene_path.to_str().expect("utf-8 path"),
        None,
        false,
    )
    .expect("gene file");

    println!("stores: generated in {:.2?}", start.elapsed());
    (cell_path, gene_path)
}

/////////////
// Harness //
/////////////

/// Fastest of [`REPEATS`] runs of `f`.
///
/// ### Params
///
/// * `f` - The case
///
/// ### Returns
///
/// The fastest wall time.
fn time(mut f: impl FnMut()) -> Duration {
    f();
    (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .min()
        .expect("REPEATS > 0")
}

/// Print one row of the results table.
///
/// ### Params
///
/// * `case` - Case name
/// * `chunk` - Chunk path time
/// * `fused` - Fused path time
fn report(case: &str, chunk: Duration, fused: Duration) {
    println!(
        "{case:<14} {:>10.1} {:>10.1} {:>8.2}x",
        chunk.as_secs_f64() * 1e3,
        fused.as_secs_f64() * 1e3,
        chunk.as_secs_f64() / fused.as_secs_f64()
    );
}

fn main() {
    let n_cells = std::env::var("SC_READER_BENCH_CELLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CELLS);
    let n_genes = std::env::var("SC_READER_BENCH_GENES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_GENES);

    let (cell_path, gene_path) = ensure_stores(n_cells, n_genes);
    let cells = ParallelSparseReader::new(cell_path.to_str().unwrap()).expect("cell store");
    let genes = ParallelSparseReader::new(gene_path.to_str().unwrap()).expect("gene store");

    let all_genes: Vec<usize> = (0..n_genes).collect();
    let all_cells: Vec<usize> = (0..n_cells).collect();
    let mut subset = all_cells.clone();
    subset.shuffle(&mut SmallRng::seed_from_u64(SEED));
    subset.truncate(n_cells / 2);
    subset.sort_unstable();
    let keep: IndexSet<u32> = subset.iter().map(|&c| c as u32).collect();

    let norm = DataLayerReturn::Norm;
    let raw = DataLayerReturn::Raw;

    println!(
        "{n_cells} cells x {n_genes} genes, {} threads",
        rayon::current_num_threads()
    );
    println!(
        "{:<14} {:>10} {:>10} {:>9}",
        "case", "chunk ms", "fused ms", "speedup"
    );

    let chunk = time(|| {
        let c = genes
            .read_gene_parallel_filtered(&all_genes, &keep)
            .unwrap();
        black_box(from_gene_chunks::<f32>(c, &norm, keep.len()).unwrap());
    });
    let fused = time(|| {
        black_box(
            genes
                .read_genes_sparse::<f32>(&all_genes, Some(&keep), &norm)
                .unwrap(),
        );
    });
    report("gene_csc", chunk, fused);

    let chunk = time(|| {
        let c = genes.read_gene_parallel(&all_genes).unwrap();
        black_box(from_gene_chunks::<f32>(c, &norm, n_cells).unwrap());
    });
    let fused = time(|| {
        black_box(
            genes
                .read_genes_sparse::<f32>(&all_genes, None, &norm)
                .unwrap(),
        );
    });
    report("gene_csc_all", chunk, fused);

    let chunk = time(|| {
        for batch in all_genes.chunks(GENE_BATCH) {
            let sums: Vec<f32> = genes
                .read_gene_parallel(batch)
                .unwrap()
                .par_iter()
                .map(|g| g.data_norm.iter().map(|v| v.to_f32()).sum())
                .collect();
            black_box(sums);
        }
    });
    let fused = time(|| {
        for batch in all_genes.chunks(GENE_BATCH) {
            let csc = genes.read_genes_sparse::<f32>(batch, None, &norm).unwrap();
            let values = csc.data_2.as_ref().unwrap();
            let sums: Vec<f32> = csc
                .indptr
                .par_windows(2)
                .map(|w| values[w[0] as usize..w[1] as usize].iter().sum())
                .collect();
            black_box(sums);
        }
    });
    report("gene_stream", chunk, fused);

    let chunk = time(|| {
        let c = cells.read_cells_parallel(&all_cells).unwrap();
        black_box(from_cell_chunks::<u32>(&c, &raw, n_genes).unwrap());
    });
    let fused = time(|| {
        black_box(cells.read_cells_sparse::<u32>(&all_cells, &raw).unwrap());
    });
    report("cell_csr", chunk, fused);

    let chunk = time(|| {
        for batch in all_cells.chunks(CELL_BATCH) {
            let sums: Vec<u64> = cells
                .read_cells_parallel(batch)
                .unwrap()
                .par_iter()
                .map(|c| c.data_raw.iter().map(|v| v as u64).sum())
                .collect();
            black_box(sums);
        }
    });
    let fused = time(|| {
        for batch in all_cells.chunks(CELL_BATCH) {
            let csr = cells.read_cells_sparse::<u32>(batch, &raw).unwrap();
            let sums: Vec<u64> = csr
                .indptr
                .par_windows(2)
                .map(|w| {
                    csr.data[w[0] as usize..w[1] as usize]
                        .iter()
                        .map(|&v| v as u64)
                        .sum()
                })
                .collect();
            black_box(sums);
        }
    });
    report("cell_stream", chunk, fused);
}
