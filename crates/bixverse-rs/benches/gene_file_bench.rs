//! Stage split of the CSR to CSC conversion and of the HVG gene statistics.
//!
//! `convert` mirrors `write_gene_file_impl` step by step on a cached synthetic
//! cell store and times each stage on its own:
//!
//! * `count` - the nnz histogram pass, full decode (as shipped), and the same
//!   pass over a raw-only fused decode.
//! * per phase: `decode` (`read_cells_range`, nothing done with the cells),
//!   `decode+scatter` (the shipped loop, with the per-task time split between
//!   the two), `raw decode` (`read_cells_sparse` raw only) and
//!   `raw decode+scatter` (raw and cell index only, no norm). The last two
//!   bound what dropping the owned chunks and the norm layer could buy.
//! * `encode` (chunk build plus `encode_gene`) and `write`
//!   (`write_compressed_gene_chunks`), summed over write batches.
//!
//! `e2e` runs the shipped `write_gene_file` with its own verbose split and
//! checks its output is byte identical to the mirror's.
//!
//! `hvg` sweeps the gene file in blocks of [`HVG_GENE_BLOCK`] genes, as
//! `sweep_gene_block` does, and times `read_gene_parallel`, `gene_stats` on the
//! chunks it returned, and a raw-only `read_genes_sparse` of the same block.
//!
//! `describe` scans the cell store for what drives the speed of both: nnz per
//! cell and per gene, raw count maxima, how many genes need `u32` raw counts
//! and how many chunks store their norm layer verbatim.
//!
//! With `GENE_FILE_BENCH_H5AD` set (needs the `h5` feature) the cell store is
//! built from that h5ad through `stream_h5_counts`, the path behind R's
//! `load_h5ad`, with the pilot QC ([`H5AD_QC`]), instead of being generated.
//! It is cached on the file name. The default stages are then
//! `describe,e2e,hvg`; on a large file set `GENE_FILE_BENCH_MAX_NNZ`, since a
//! single phase holds 10 bytes per nnz.
//!
//! One run per measurement. The stores were just written or just read, so the
//! numbers are warm page-cache unless the cache is purged in between.
//!
//! Run with:
//! ```text
//! cargo bench -p bixverse-rs --features single-cell --bench gene_file_bench
//! GENE_FILE_BENCH_H5AD=/path/to.h5ad \
//!   cargo bench -p bixverse-rs --features hdf5-static --bench gene_file_bench
//! ```
//!
//! Environment:
//!
//! * `GENE_FILE_BENCH_CELLS` / `GENE_FILE_BENCH_GENES` - shape
//! * `GENE_FILE_BENCH_LIBRARY` - median UMIs per cell, sets nnz per cell
//! * `GENE_FILE_BENCH_MAX_NNZ` - phase budget, unset for a single phase
//! * `GENE_FILE_BENCH_ONLY` - comma-separated subset of
//!   `describe,convert,e2e,hvg`
//! * `GENE_FILE_BENCH_H5AD` - build the cell store from this h5ad
//! * `GENE_FILE_BENCH_H5AD_SLOT` - raw count slot: `X` (default), `raw.X` or
//!   `layers.counts`
//! * `GENE_FILE_BENCH_DIR` - where the stores go, default the temp directory
//! * `GENE_FILE_BENCH_REGEN=1` - rebuild the cached cell store, synthetic or
//!   h5ad
//! * `RAYON_NUM_THREADS` - thread count

#![cfg(feature = "single-cell")]

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, LogNormal, Uniform};
use rayon::prelude::*;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CellPayload, CscGeneChunk, CsrCellChunk, DataLayerReturn,
    ParallelSparseReader, RawCounts, SingleCellReading,
};
use bixverse_rs::single_cell::sc_data::gene_file_io::write_gene_file;
use bixverse_rs::single_cell::sc_processing::hvg::{
    CellBatchIndex, GeneStats, HistogramScratch, gene_stats,
};

////////////
// Shapes //
////////////

/// Default cell count.
const DEFAULT_CELLS: usize = 500_000;

/// Default gene count. Roughly a filtered 10x feature set.
const DEFAULT_GENES: usize = 20_000;

/// Default median UMIs per cell. Chosen so the count model below lands near
/// 1,200 nnz per cell at 20k genes.
const DEFAULT_LIBRARY: f64 = 2_300.0;

/// Log-scale spread of the library size, a roughly 4x spread between the 1st
/// and 99th percentile cell.
const LIBRARY_SIZE_LOG_SD: f64 = 0.35;

/// Log-scale spread of per-gene relative expression.
const EXPRESSION_LOG_SD: f64 = 2.0;

/// Dominant genes (MALAT1, ribosomal, mito stand-ins). At the default library
/// size the top few are near ubiquitous with counts in the hundreds.
const N_DOMINANT_GENES: usize = 40;

/// Share of every library taken by the dominant genes, split as `1 / rank`.
const DOMINANT_SHARE: f64 = 0.25;

/// Library size of the normalised layer.
const TARGET_SIZE: f32 = 1e4;

/// Cells generated and encoded per write batch.
const GEN_BATCH: usize = 8_192;

/// Mirrors `GENE_FILE_READ_BATCH` in `gene_file_io.rs`.
const READ_BATCH: usize = 2_048;

/// Mirrors `GENE_FILE_WRITE_BATCH` in `gene_file_io.rs`.
const WRITE_BATCH: usize = 1_024;

/// Mirrors `GENE_BATCH_SIZE` in `hvg.rs`.
const HVG_GENE_BLOCK: usize = 1_000;

/// Above this file size the byte comparison of mirror and shipped output is
/// skipped, to keep the bench from reading both files back into memory.
const COMPARE_MAX_BYTES: u64 = 4 << 30;

/// Seed for the generator.
const SEED: u64 = 0x6E4E_F11E;

/// QC of the pilot run: `(min_unique_genes, min_lib_size, min_cells,
/// target_size)`.
#[cfg(feature = "h5")]
const H5AD_QC: (usize, usize, usize, f32) = (100, 250, 5, 1e4);

/// Largest raw count the `u16` raw layer holds.
const U16_LIMIT: u32 = u16::MAX as u32;

/////////////////
// Store setup //
/////////////////

/// Cumulative gene shares: log-normal background plus dominant genes.
///
/// ### Params
///
/// * `n_genes` - Number of genes
///
/// ### Returns
///
/// Ascending cumulative distribution, last entry 1.
fn gene_cdf(n_genes: usize) -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    let shares = LogNormal::new(0.0, EXPRESSION_LOG_SD).expect("valid log-normal");
    let mut share: Vec<f64> = (0..n_genes).map(|_| shares.sample(&mut rng)).collect();

    let n_dom = N_DOMINANT_GENES.min(n_genes / 10);
    let background: f64 = share[n_dom..].iter().sum();
    let dom_share = if n_dom > 0 { DOMINANT_SHARE } else { 0.0 };
    share[n_dom..]
        .iter_mut()
        .for_each(|s| *s *= (1.0 - dom_share) / background);
    let harmonic: f64 = (1..=n_dom).map(|r| 1.0 / r as f64).sum();
    for (rank, s) in share[..n_dom].iter_mut().enumerate() {
        *s = dom_share / (harmonic * (rank + 1) as f64);
    }

    // spread the dominant genes over the index range rather than leaving them
    // all in the first phase
    let stride = (n_genes / n_dom.max(1)).max(1);
    for rank in 0..n_dom {
        share.swap(rank, rank * stride);
    }

    let mut acc = 0.0;
    let mut cdf: Vec<f64> = share
        .iter()
        .map(|&s| {
            acc += s;
            acc
        })
        .collect();
    *cdf.last_mut().expect("n_genes > 0") = 1.0;
    cdf
}

/// Draw one cell as a multinomial over the gene shares.
///
/// ### Params
///
/// * `cell` - Cell index, seeds the draw
/// * `cdf` - Cumulative gene shares
/// * `library` - Median library size
///
/// ### Returns
///
/// The cell chunk.
fn draw_cell(cell: usize, cdf: &[f64], library: f64) -> CsrCellChunk {
    let mut rng = SmallRng::seed_from_u64(SEED ^ (cell as u64).wrapping_mul(0x9E37_79B9));
    let lib = LogNormal::new(library.ln(), LIBRARY_SIZE_LOG_SD)
        .expect("valid log-normal")
        .sample(&mut rng)
        .max(1.0) as usize;
    let unit = Uniform::new(0.0, 1.0).expect("valid range");
    let mut genes: Vec<u32> = (0..lib)
        .map(|_| {
            let u: f64 = unit.sample(&mut rng);
            cdf.partition_point(|&c| c < u).min(cdf.len() - 1) as u32
        })
        .collect();
    genes.sort_unstable();

    let mut cols = Vec::new();
    let mut counts = Vec::new();
    for &g in &genes {
        if cols.last() == Some(&g) {
            *counts.last_mut().expect("paired with cols") += 1u32;
        } else {
            cols.push(g);
            counts.push(1u32);
        }
    }
    CsrCellChunk::from_data(&counts, &cols, cell, TARGET_SIZE, true)
}

/// Write the cell store unless cached.
///
/// ### Params
///
/// * `path` - Store path
/// * `n_cells` - Number of cells
/// * `n_genes` - Number of genes
/// * `library` - Median library size
fn ensure_cell_store(path: &Path, n_cells: usize, n_genes: usize, library: f64) {
    if path.exists() && std::env::var("GENE_FILE_BENCH_REGEN").is_err() {
        return;
    }
    let start = Instant::now();
    let cdf = gene_cdf(n_genes);
    let mut writer = CellGeneSparseWriter::new(
        path.to_str().expect("utf-8 path"),
        true,
        n_cells,
        n_genes,
        TARGET_SIZE,
    )
    .expect("writer");
    let format = writer.format();

    for b0 in (0..n_cells).step_by(GEN_BATCH) {
        let b1 = (b0 + GEN_BATCH).min(n_cells);
        let payloads: Vec<CellPayload> = (b0..b1)
            .into_par_iter()
            .map(|cell| {
                let chunk = draw_cell(cell, &cdf, library);
                CellPayload {
                    original_index: cell,
                    library_size: chunk.library_size,
                    bytes: format.encode_cell(&chunk).expect("encode cell"),
                }
            })
            .collect();
        writer
            .write_compressed_cell_chunks(&payloads)
            .expect("write cells");
    }
    writer.finalise().expect("finalise");
    eprintln!("cell store generated in {:.1?}", start.elapsed());
}

/// Build the cell store from an h5ad unless cached.
///
/// Same call as R's `load_h5ad` (streaming, its default). Format and shape
/// come from the `shape` attribute of the count group and the length of its
/// `indptr`, the fallback `get_h5ad_dimensions` uses on the R side.
///
/// ### Params
///
/// * `h5ad` - Input file
/// * `cell_path` - Store path
/// * `slot` - Raw count slot, as the R `slot` argument
#[cfg(feature = "h5")]
fn ensure_h5ad_store(h5ad: &Path, cell_path: &Path, slot: &str) {
    use bixverse_rs::single_cell::sc_data::data_io::MinCellQuality;
    use bixverse_rs::single_cell::sc_data::h5ad_io::{parse_raw_slot, stream_h5_counts};

    if cell_path.exists() && std::env::var("GENE_FILE_BENCH_REGEN").is_err() {
        return;
    }
    let raw_slot =
        parse_raw_slot(slot).expect("GENE_FILE_BENCH_H5AD_SLOT: X, raw.X or layers.counts");
    let (no_cells, no_genes, indptr_len) = {
        let file = hdf5::File::open(h5ad).expect("open h5ad");
        let shape: Vec<i64> = file
            .group(raw_slot.get_dense_path())
            .and_then(|g| g.attr("shape"))
            .and_then(|a| a.read_raw::<i64>())
            .expect("sparse count group with a shape attribute (dense X is not supported here)");
        let indptr_len = file.dataset(raw_slot.get_indptr()).expect("indptr").size();
        (shape[0] as usize, shape[1] as usize, indptr_len)
    };
    let format = if no_cells == no_genes {
        panic!("square matrix, cannot tell CSR from CSC by indptr length");
    } else if indptr_len == no_cells + 1 {
        "csr"
    } else if indptr_len == no_genes + 1 {
        "csc"
    } else {
        panic!("indptr length {indptr_len} fits neither {no_cells} cells nor {no_genes} genes");
    };
    eprintln!("h5ad {no_cells} x {no_genes}, {format}, slot {slot}");

    let (min_unique_genes, min_lib_size, min_cells, target_size) = H5AD_QC;
    let qc = MinCellQuality {
        min_unique_genes,
        min_lib_size,
        min_cells,
        target_size,
    };
    let (res, t) = timed(|| {
        stream_h5_counts(
            h5ad, cell_path, format, no_cells, no_genes, qc, raw_slot, true,
        )
    });
    let (kept_cells, kept_genes, _) = res.expect("stream_h5_counts");
    eprintln!(
        "h5ad store: {kept_cells} cells x {kept_genes} genes kept, stream_h5_counts {:.1} s \
         (pass and write split on stdout)",
        s(t)
    );
}

/// Value at quantile `q` of `v`, nearest rank.
///
/// ### Params
///
/// * `v` - Values, sorted in place
/// * `q` - Quantile in `[0, 1]`
///
/// ### Returns
///
/// The value, 0 for an empty slice.
fn quantile(v: &mut [u64], q: f64) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Format a stored-norm chunk count, or why there is none.
///
/// ### Params
///
/// * `reader` - Store reader
///
/// ### Returns
///
/// The count as text.
fn norm_stored(reader: &ParallelSparseReader) -> String {
    reader
        .count_norm_stored_chunks()
        .map_or_else(|e| format!("n/a ({e})"), |n| n.to_string())
}

/// Print the data descriptors that drive conversion and HVG speed.
///
/// One raw-only pass over the cell store.
///
/// ### Params
///
/// * `cell_path` - Cell store
fn describe_store(cell_path: &str) {
    let reader = ParallelSparseReader::new(cell_path).expect("cell reader");
    let header = reader.get_header();
    let (n_cells, n_genes) = (header.total_cells, header.total_genes);
    let raw_layer = DataLayerReturn::Raw;

    // per worker: nnz per cell, nnz per gene, max raw per gene
    let ((mut cell_nnz, mut gene_nnz, gene_max, u32_cells), t) = timed(|| {
        thread_chunks(n_cells)
            .into_par_iter()
            .map(|(start, end)| {
                let mut cell_nnz = Vec::with_capacity(end - start);
                let mut gene_nnz = vec![0u64; n_genes];
                let mut gene_max = vec![0u32; n_genes];
                let mut u32_cells = 0usize;
                for b0 in (start..end).step_by(READ_BATCH) {
                    let b1 = (b0 + READ_BATCH).min(end);
                    let idx: Vec<usize> = (b0..b1).collect();
                    let csr = reader
                        .read_cells_sparse::<u32>(&idx, &raw_layer)
                        .expect("read");
                    for w in csr.indptr.windows(2) {
                        let (a, b) = (w[0] as usize, w[1] as usize);
                        cell_nnz.push((b - a) as u64);
                        u32_cells += csr.data[a..b].iter().any(|&v| v > U16_LIMIT) as usize;
                        for (&g, &v) in csr.indices[a..b].iter().zip(&csr.data[a..b]) {
                            gene_nnz[g as usize] += 1;
                            gene_max[g as usize] = gene_max[g as usize].max(v);
                        }
                    }
                }
                (cell_nnz, gene_nnz, gene_max, u32_cells)
            })
            .reduce(
                || (Vec::new(), vec![0u64; n_genes], vec![0u32; n_genes], 0),
                |mut a, b| {
                    a.0.extend(b.0);
                    a.1.iter_mut().zip(&b.1).for_each(|(x, y)| *x += y);
                    a.2.iter_mut()
                        .zip(&b.2)
                        .for_each(|(x, &y)| *x = (*x).max(y));
                    a.3 += b.3;
                    a
                },
            )
    });

    let nnz: u64 = cell_nnz.iter().sum();
    let u32_genes = gene_max.iter().filter(|&&m| m > U16_LIMIT).count();
    let ubiquitous = gene_nnz.iter().filter(|&&n| n * 2 > n_cells as u64).count();
    let mut gene_max: Vec<u64> = gene_max.into_iter().map(u64::from).collect();

    eprintln!("describe ({:.1} s scan)", s(t));
    eprintln!("  cells x genes                {n_cells} x {n_genes}, nnz {nnz}");
    eprintln!(
        "  nnz per cell                 mean {:.0}, p50 {}, p99 {}, max {}",
        nnz as f64 / n_cells.max(1) as f64,
        quantile(&mut cell_nnz, 0.5),
        quantile(&mut cell_nnz, 0.99),
        quantile(&mut cell_nnz, 1.0)
    );
    eprintln!(
        "  nnz per gene                 p50 {}, p99 {}, max {}; in > 50% of cells: {ubiquitous}",
        quantile(&mut gene_nnz, 0.5),
        quantile(&mut gene_nnz, 0.99),
        quantile(&mut gene_nnz, 1.0)
    );
    eprintln!(
        "  max raw per gene             p50 {}, p99 {}, max {}",
        quantile(&mut gene_max, 0.5),
        quantile(&mut gene_max, 0.99),
        quantile(&mut gene_max, 1.0)
    );
    eprintln!("  genes needing u32 raw        {u32_genes} of {n_genes}");
    eprintln!("  cells needing u32 raw        {u32_cells} of {n_cells}");
    eprintln!(
        "  cell chunks with stored norm {} of {n_cells}",
        norm_stored(&reader)
    );
}

/////////////
// Helpers //
/////////////

/// Shared mutable pointer for disjoint scatter writes across rayon tasks.
#[derive(Clone, Copy)]
struct Scatter<T>(*mut T);

// SAFETY: every user writes disjoint positions reserved by the count pass.
unsafe impl<T: Send> Send for Scatter<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for Scatter<T> {}

impl<T> Scatter<T> {
    /// Write `v` at `pos`.
    ///
    /// ### Params
    ///
    /// * `pos` - Element offset, unique to the caller
    /// * `v` - Value
    ///
    /// ### Safety
    ///
    /// `pos` is in bounds and no other thread writes it concurrently.
    #[inline(always)]
    unsafe fn write(self, pos: usize, v: T) {
        unsafe { *self.0.add(pos) = v };
    }
}

/// Seconds as f64.
///
/// ### Params
///
/// * `d` - Duration
///
/// ### Returns
///
/// Seconds.
fn s(d: Duration) -> f64 {
    d.as_secs_f64()
}

/// Print one table row.
///
/// ### Params
///
/// * `stage` - Stage name
/// * `secs` - Wall seconds
/// * `note` - Free text
fn row(stage: &str, secs: f64, note: &str) {
    eprintln!("  {stage:<28} {secs:>9.3}  {note}");
}

/// Time `f`.
///
/// ### Params
///
/// * `f` - Work
///
/// ### Returns
///
/// Result and wall time.
fn timed<R>(f: impl FnOnce() -> R) -> (R, Duration) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed())
}

/// Phase ranges, the same cut as `plan_phases` in `gene_file_io.rs`.
///
/// ### Params
///
/// * `gene_nnz` - Non-zeros per gene
/// * `max_nnz` - Budget per phase
///
/// ### Returns
///
/// Half-open gene ranges.
fn plan_phases(gene_nnz: &[usize], max_nnz: Option<usize>) -> Vec<(usize, usize)> {
    let n = gene_nnz.len();
    let Some(max_nnz) = max_nnz else {
        return vec![(0, n)];
    };
    let mut phases = Vec::new();
    let (mut start, mut acc) = (0, 0usize);
    for (g, &nnz) in gene_nnz.iter().enumerate() {
        if g > start && acc + nnz > max_nnz {
            phases.push((start, g));
            start = g;
            acc = 0;
        }
        acc += nnz;
    }
    if start < n || phases.is_empty() {
        phases.push((start, n));
    }
    phases
}

/// Per-chunk write cursors for one phase.
///
/// ### Params
///
/// * `counts` - Chunk-major histogram
/// * `gene_start` - Run start per phase gene
/// * `n_chunks` - Number of cell chunks
/// * `n_genes` - Genes in the file
/// * `g0` - First gene of the phase
/// * `g1` - One past the last gene of the phase
///
/// ### Returns
///
/// Chunk-major cursors, `n_chunks * (g1 - g0)`.
fn phase_cursors(
    counts: &[usize],
    gene_start: &[usize],
    n_chunks: usize,
    n_genes: usize,
    g0: usize,
    g1: usize,
) -> Vec<usize> {
    let n_phase = g1 - g0;
    let mut cursors = vec![0usize; n_chunks * n_phase];
    let mut running = gene_start[..n_phase].to_vec();
    for (c, row) in cursors.chunks_mut(n_phase.max(1)).enumerate() {
        for ((cur, run), &cnt) in row
            .iter_mut()
            .zip(running.iter_mut())
            .zip(&counts[c * n_genes + g0..c * n_genes + g1])
        {
            *cur = *run;
            *run += cnt;
        }
    }
    cursors
}

/////////////
// Convert //
/////////////

/// Mirror of `write_gene_file_impl` (v4) with every stage timed.
///
/// ### Params
///
/// * `cell_path` - Cell store
/// * `gene_path` - Gene store to write
/// * `max_nnz` - Phase budget
fn bench_convert(cell_path: &str, gene_path: &str, max_nnz: Option<usize>) {
    let total = Instant::now();
    let reader = ParallelSparseReader::new(cell_path).expect("cell reader");
    let header = reader.get_header();
    let (n_cells, n_genes) = (header.total_cells, header.total_genes);
    let chunks = thread_chunks(n_cells);
    let raw_layer = DataLayerReturn::Raw;

    eprintln!("convert ({} cell chunks)", chunks.len());
    eprintln!("  {:<28} {:>9}  note", "stage", "wall s");

    // count pass, as shipped
    let (counts, t) = timed(|| {
        let mut counts = vec![0usize; chunks.len() * n_genes];
        chunks
            .par_iter()
            .zip(counts.par_chunks_mut(n_genes))
            .for_each(|(&(start, end), counts)| {
                for b0 in (start..end).step_by(READ_BATCH) {
                    let b1 = (b0 + READ_BATCH).min(end);
                    for cell in reader.read_cells_range(b0, b1).expect("read") {
                        for &g in &cell.indices {
                            counts[g as usize] += 1;
                        }
                    }
                }
            });
        counts
    });
    row("count", s(t), "read_cells_range, full decode");

    // count pass over the raw-only fused decode
    let (_, t) = timed(|| {
        let mut counts = vec![0usize; chunks.len() * n_genes];
        chunks
            .par_iter()
            .zip(counts.par_chunks_mut(n_genes))
            .for_each(|(&(start, end), counts)| {
                for b0 in (start..end).step_by(READ_BATCH) {
                    let b1 = (b0 + READ_BATCH).min(end);
                    let idx: Vec<usize> = (b0..b1).collect();
                    let csr = reader
                        .read_cells_sparse::<u32>(&idx, &raw_layer)
                        .expect("read");
                    for &g in &csr.indices {
                        counts[g as usize] += 1;
                    }
                }
            });
        black_box(counts)
    });
    row("count (raw fused)", s(t), "read_cells_sparse raw only");

    let mut gene_nnz = vec![0usize; n_genes];
    for c in counts.chunks(n_genes) {
        gene_nnz.iter_mut().zip(c).for_each(|(t, &v)| *t += v);
    }
    let nnz_total: usize = gene_nnz.iter().sum();
    let phases = plan_phases(&gene_nnz, max_nnz);
    let max_phase_nnz = phases
        .iter()
        .map(|&(a, b)| gene_nnz[a..b].iter().sum::<usize>())
        .max()
        .unwrap_or(0);
    eprintln!(
        "  nnz {nnz_total} ({:.0} per cell), {} phase(s), phase buffers {:.2} GB",
        nnz_total as f64 / n_cells as f64,
        phases.len(),
        (max_phase_nnz * 10) as f64 / 1e9
    );

    let (cell_lib_sizes, t) = timed(|| {
        reader
            .read_cell_library_sizes(&(0..n_cells).collect::<Vec<_>>())
            .expect("lib sizes")
            .into_iter()
            .map(|l| l as u64)
            .collect::<Vec<u64>>()
    });
    row("library sizes", s(t), "");
    let cell_lib: Vec<f32> = cell_lib_sizes.iter().map(|&l| l as f32).collect();

    let mut writer = CellGeneSparseWriter::new(
        gene_path,
        false,
        n_cells,
        n_genes,
        reader.target_size().unwrap_or(0.0),
    )
    .expect("gene writer");
    let format = writer.format();

    let mut raw = vec![0u32; max_phase_nnz];
    let mut norm = vec![F16::default(); max_phase_nnz];
    let mut cell_idx = vec![0u32; max_phase_nnz];

    for (pi, &(g0, g1)) in phases.iter().enumerate() {
        let n_phase = g1 - g0;
        eprintln!("  phase {}/{}: genes {g0}-{g1}", pi + 1, phases.len());
        let mut gene_start = vec![0usize; n_phase + 1];
        for g in g0..g1 {
            gene_start[g - g0 + 1] = gene_start[g - g0] + gene_nnz[g];
        }

        // decode only
        let (_, t) = timed(|| {
            chunks.par_iter().for_each(|&(start, end)| {
                for b0 in (start..end).step_by(READ_BATCH) {
                    let b1 = (b0 + READ_BATCH).min(end);
                    black_box(reader.read_cells_range(b0, b1).expect("read"));
                }
            })
        });
        row("  decode", s(t), "read_cells_range, no scatter");

        // raw-only fused decode
        let (_, t) = timed(|| {
            chunks.par_iter().for_each(|&(start, end)| {
                for b0 in (start..end).step_by(READ_BATCH) {
                    let b1 = (b0 + READ_BATCH).min(end);
                    let idx: Vec<usize> = (b0..b1).collect();
                    black_box(
                        reader
                            .read_cells_sparse::<u32>(&idx, &raw_layer)
                            .expect("read"),
                    );
                }
            })
        });
        row("  raw decode", s(t), "read_cells_sparse raw only");

        // raw-only fused decode plus a raw + cell index scatter
        let mut cursors = phase_cursors(&counts, &gene_start, chunks.len(), n_genes, g0, g1);
        let raw_ptr = Scatter(raw.as_mut_ptr());
        let idx_ptr = Scatter(cell_idx.as_mut_ptr());
        let (_, t) = timed(|| {
            chunks
                .par_iter()
                .zip(cursors.par_chunks_mut(n_phase))
                .for_each(|(&(start, end), cursor)| {
                    for b0 in (start..end).step_by(READ_BATCH) {
                        let b1 = (b0 + READ_BATCH).min(end);
                        let idx: Vec<usize> = (b0..b1).collect();
                        let csr = reader
                            .read_cells_sparse::<u32>(&idx, &raw_layer)
                            .expect("read");
                        for (off, w) in csr.indptr.windows(2).enumerate() {
                            let cell_id = (b0 + off) as u32;
                            let genes = &csr.indices[w[0] as usize..w[1] as usize];
                            let vals = &csr.data[w[0] as usize..w[1] as usize];
                            let lo = genes.partition_point(|&g| (g as usize) < g0);
                            let hi = genes.partition_point(|&g| (g as usize) < g1);
                            for k in lo..hi {
                                let local = genes[k] as usize - g0;
                                let pos = cursor[local];
                                cursor[local] += 1;
                                // SAFETY: run reserved by the count pass
                                unsafe {
                                    raw_ptr.write(pos, vals[k]);
                                    idx_ptr.write(pos, cell_id);
                                }
                            }
                        }
                    }
                })
        });
        row("  raw decode+scatter", s(t), "raw + cell idx, no norm");

        // the shipped loop, with the per-task split
        let mut cursors = phase_cursors(&counts, &gene_start, chunks.len(), n_genes, g0, g1);
        let norm_ptr = Scatter(norm.as_mut_ptr());
        let (splits, t) = timed(|| {
            chunks
                .par_iter()
                .zip(cursors.par_chunks_mut(n_phase))
                .map(|(&(start, end), cursor)| {
                    let (mut t_dec, mut t_sc) = (Duration::ZERO, Duration::ZERO);
                    for b0 in (start..end).step_by(READ_BATCH) {
                        let b1 = (b0 + READ_BATCH).min(end);
                        let (cells, td) = timed(|| reader.read_cells_range(b0, b1).expect("read"));
                        t_dec += td;
                        let ts = Instant::now();
                        for (off, cell) in cells.iter().enumerate() {
                            let cell_id = (b0 + off) as u32;
                            let lo = cell.indices.partition_point(|&g| (g as usize) < g0);
                            let hi = cell.indices.partition_point(|&g| (g as usize) < g1);
                            for k in lo..hi {
                                let local = cell.indices[k] as usize - g0;
                                let pos = cursor[local];
                                cursor[local] += 1;
                                // SAFETY: run reserved by the count pass
                                unsafe {
                                    raw_ptr.write(pos, cell.data_raw.get(k));
                                    norm_ptr.write(pos, cell.data_norm[k]);
                                    idx_ptr.write(pos, cell_id);
                                }
                            }
                        }
                        t_sc += ts.elapsed();
                        let td = Instant::now();
                        drop(cells);
                        t_dec += td.elapsed();
                    }
                    (s(t_dec), s(t_sc))
                })
                .collect::<Vec<_>>()
        });
        let n = splits.len() as f64;
        let dec: f64 = splits.iter().map(|x| x.0).sum::<f64>() / n;
        let sc: f64 = splits.iter().map(|x| x.1).sum::<f64>() / n;
        row(
            "  decode+scatter",
            s(t),
            &format!("shipped; mean per task: decode {dec:.3} s, scatter {sc:.3} s"),
        );

        // encode and write, summed over write batches
        let (mut t_enc, mut t_wr) = (Duration::ZERO, Duration::ZERO);
        for b0 in (g0..g1).step_by(WRITE_BATCH) {
            let b1 = (b0 + WRITE_BATCH).min(g1);
            let (payloads, te) = timed(|| {
                (b0..b1)
                    .into_par_iter()
                    .map(|g| {
                        let (a, b) = (gene_start[g - g0], gene_start[g - g0 + 1]);
                        let data_norm = norm[a..b].to_vec();
                        let avg_exp =
                            F16::from_f32(data_norm.iter().map(|v| v.to_f32()).sum::<f32>());
                        let chunk = CscGeneChunk {
                            data_raw: RawCounts::from_u32_auto(&raw[a..b]),
                            data_norm,
                            avg_exp,
                            nnz: b - a,
                            indices: cell_idx[a..b].to_vec(),
                            original_index: g,
                            to_keep: true,
                        };
                        (g, format.encode_gene(&chunk, &cell_lib).expect("encode"))
                    })
                    .collect::<Vec<_>>()
            });
            t_enc += te;
            let (_, tw) = timed(|| writer.write_compressed_gene_chunks(&payloads));
            t_wr += tw;
        }
        row("  encode", s(t_enc), "chunk build + encode_gene");
        row("  write", s(t_wr), "write_compressed_gene_chunks");
    }

    let (_, t) = timed(|| {
        writer
            .set_cell_library_table(cell_lib_sizes)
            .expect("lib table");
        writer.finalise().expect("finalise");
    });
    row("finalise", s(t), "");
    row("total (all variants)", s(total.elapsed()), "");
}

/////////
// HVG //
/////////

/// Per-block split of the HVG statistics sweep.
///
/// ### Params
///
/// * `gene_path` - Gene store
fn bench_hvg(gene_path: &str) {
    let reader = ParallelSparseReader::new(gene_path).expect("gene reader");
    let header = reader.get_header();
    let (n_cells, n_genes) = (header.total_cells, header.total_genes);
    let cells: Vec<usize> = (0..n_cells).collect();
    let index = CellBatchIndex::new(n_cells, &cells, None).expect("index");
    let (n_batches, n_slots) = (index.n_batches(), index.n_slots());
    let raw_layer = DataLayerReturn::Raw;
    let mut out = vec![GeneStats::default(); HVG_GENE_BLOCK * n_batches];

    let (mut t_read, mut t_stats, mut t_raw) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let mut n_blocks = 0usize;
    for g0 in (0..n_genes).step_by(HVG_GENE_BLOCK) {
        let g1 = (g0 + HVG_GENE_BLOCK).min(n_genes);
        let block: Vec<usize> = (g0..g1).collect();

        let (genes, t) = timed(|| reader.read_gene_parallel(&block).expect("read"));
        t_read += t;

        let (_, t) = timed(|| {
            genes
                .par_iter()
                .zip(out.par_chunks_mut(n_batches))
                .for_each_init(
                    || HistogramScratch::new(n_slots),
                    |scratch, (gene, stats)| gene_stats(gene, &index, scratch, stats),
                )
        });
        t_stats += t;
        black_box(&out);
        drop(genes);

        let (csc, t) = timed(|| {
            reader
                .read_genes_sparse::<u32>(&block, None, &raw_layer)
                .expect("read")
        });
        t_raw += t;
        black_box(csc);
        n_blocks += 1;
    }

    eprintln!("hvg ({n_blocks} blocks of {HVG_GENE_BLOCK} genes)");
    eprintln!("  {:<28} {:>9}  note", "stage", "wall s");
    let per = |t: Duration| format!("{:.1} ms per block", s(t) * 1e3 / n_blocks as f64);
    row("read_gene_parallel", s(t_read), &per(t_read));
    row("gene_stats", s(t_stats), &per(t_stats));
    row("raw read_genes_sparse", s(t_raw), &per(t_raw));
}

//////////
// Main //
//////////

/// Read an env var as `T`.
///
/// ### Params
///
/// * `key` - Variable name
///
/// ### Returns
///
/// The parsed value, if set and valid.
fn env<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

/// Compare two files byte for byte, unless they are large.
///
/// ### Params
///
/// * `a` - First file
/// * `b` - Second file
///
/// ### Returns
///
/// A one-word verdict.
fn compare_files(a: &Path, b: &Path) -> &'static str {
    let (la, lb) = (
        std::fs::metadata(a).expect("stat").len(),
        std::fs::metadata(b).expect("stat").len(),
    );
    if la != lb {
        return "DIFFERENT (length)";
    }
    if la > COMPARE_MAX_BYTES {
        return "same length, bytes not compared";
    }
    if std::fs::read(a).expect("read") == std::fs::read(b).expect("read") {
        "identical"
    } else {
        "DIFFERENT (bytes)"
    }
}

fn main() {
    let n_cells: usize = env("GENE_FILE_BENCH_CELLS").unwrap_or(DEFAULT_CELLS);
    let n_genes: usize = env("GENE_FILE_BENCH_GENES").unwrap_or(DEFAULT_GENES);
    let library: f64 = env("GENE_FILE_BENCH_LIBRARY").unwrap_or(DEFAULT_LIBRARY);
    let max_nnz: Option<usize> = env("GENE_FILE_BENCH_MAX_NNZ");
    let h5ad: Option<PathBuf> = std::env::var("GENE_FILE_BENCH_H5AD")
        .ok()
        .map(PathBuf::from);
    let only = std::env::var("GENE_FILE_BENCH_ONLY").unwrap_or_else(|_| match h5ad {
        Some(_) => "describe,e2e,hvg".to_string(),
        None => String::new(),
    });
    let run = |name: &str| only.is_empty() || only.split(',').any(|o| o.trim() == name);
    let dir: PathBuf = std::env::var("GENE_FILE_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());

    let stem = match &h5ad {
        Some(p) => format!(
            "bixverse_gene_file_bench_h5ad_{}",
            p.file_stem().expect("h5ad file name").to_string_lossy()
        ),
        None => format!("bixverse_gene_file_bench_{n_cells}x{n_genes}_{library}_{SEED:x}"),
    };
    let cell_path = dir.join(format!("{stem}_cells.bin"));
    let mirror_path = dir.join(format!("{stem}_genes_mirror.bin"));
    let gene_path = dir.join(format!("{stem}_genes.bin"));

    eprintln!(
        "{} threads, max_nnz {max_nnz:?}, stages {}",
        rayon::current_num_threads(),
        if only.is_empty() { "all" } else { &only }
    );
    match &h5ad {
        #[cfg(feature = "h5")]
        Some(p) => {
            let slot = std::env::var("GENE_FILE_BENCH_H5AD_SLOT").unwrap_or_else(|_| "X".into());
            ensure_h5ad_store(p, &cell_path, &slot);
        }
        #[cfg(not(feature = "h5"))]
        Some(_) => panic!("GENE_FILE_BENCH_H5AD needs the h5 feature, e.g. --features hdf5-static"),
        None => {
            eprintln!("synthetic {n_cells} cells x {n_genes} genes, median library {library}");
            ensure_cell_store(&cell_path, n_cells, n_genes, library);
        }
    }
    eprintln!(
        "cell store {:.2} GB at {}",
        std::fs::metadata(&cell_path).expect("stat").len() as f64 / 1e9,
        cell_path.display()
    );

    if run("describe") {
        describe_store(cell_path.to_str().expect("utf-8"));
    }

    if run("convert") {
        bench_convert(
            cell_path.to_str().expect("utf-8"),
            mirror_path.to_str().expect("utf-8"),
            max_nnz,
        );
    }

    if run("e2e") || (run("hvg") && !gene_path.exists()) {
        eprintln!("e2e write_gene_file (its own split on stdout)");
        let (res, t) = timed(|| {
            write_gene_file(
                cell_path.to_str().expect("utf-8"),
                gene_path.to_str().expect("utf-8"),
                max_nnz,
                true,
            )
        });
        res.expect("write_gene_file");
        row("write_gene_file", s(t), "");
        if run("convert") {
            eprintln!(
                "  mirror vs shipped output: {}",
                compare_files(&mirror_path, &gene_path)
            );
        }
        let genes =
            ParallelSparseReader::new(gene_path.to_str().expect("utf-8")).expect("gene reader");
        eprintln!(
            "  gene store {:.2} GB, gene chunks with stored norm {} of {}",
            std::fs::metadata(&gene_path).expect("stat").len() as f64 / 1e9,
            norm_stored(&genes),
            genes.get_header().total_genes
        );
    }

    if run("hvg") {
        bench_hvg(gene_path.to_str().expect("utf-8"));
    }
}
