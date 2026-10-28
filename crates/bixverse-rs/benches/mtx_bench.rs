//! Stage split of the Matrix Market (.mtx) loader.
//!
//! Generates (and caches) a synthetic count matrix as `.mtx` in the two
//! layouts `Matrix::writeMM` produces, since it always writes column-major:
//!
//! * `genes` - genes as rows, entries in cell-major order (CellRanger, or
//!   `writeMM` of a genes x cells `dgCMatrix`). A cell's entries are
//!   contiguous.
//! * `cells` - cells as rows, entries in gene-major order (`writeMM` of a
//!   cells x genes `dgCMatrix`). Every cell is scattered across the file,
//!   which is the case the bucketing transpose exists for.
//!
//! Then times, per layout:
//!
//! * `read` - raw bytes, sequential (8 MB reads) and parallel (one positional
//!   read per 64 MB chunk). The I/O floor.
//! * `split` - line splitting only: the shipped `BufRead::read_until` loop
//!   (copies every line), a newline count over the mmap single threaded, and
//!   the same in parallel chunks.
//! * `parse` - a mirror of `parse_mtx_line` over the mmap, no per-line copy,
//!   single threaded and in parallel chunks. The parse ceiling for an
//!   mmap-and-split design.
//! * `quality` - `MtxReader::parse_mtx_quality` (two full text passes: gene
//!   nnz, then cell stats). Its verbose output splits the two.
//! * `stream` - `process_mtx_and_write_bin_streaming` (third text pass into
//!   bucket files, then sort and write). Verbose output splits bucketing from
//!   writing.
//! * `inmem` - `process_mtx_and_write_bin`, the non-streaming writer. Holds
//!   every kept entry at 8 bytes plus a `Vec` per cell, so opt-in only.
//! * `gz` - with `MTX_BENCH_GZ=1`, gzips the file once (`gzip -1`, cached) and
//!   times `gzip -dc` to `/dev/null`: the single-threaded inflate floor that
//!   any gz path pays at least once.
//!
//! `quality` plus `stream` is what R's `load_mtx` runs (streaming is its
//! default), on one `MtxReader`, before the CSR to CSC conversion that
//! `gene_file_bench` covers. The `stream` stage prints an `FxHasher` hash of
//! the store, so two builds of the loader can be checked for identical output.
//!
//! With `MTX_BENCH_MTX` set, only the loader stages run, on that file (plain
//! or gzipped, if the loader reads gz), with the pilot QC ([`QC`]). Nothing
//! is cached: the store is removed after hashing.
//!
//! One run per measurement. The file was just written or just read, so the
//! numbers are warm page cache unless the cache is purged in between.
//!
//! Run with:
//! ```text
//! cargo bench -p bixverse-rs --features single-cell --bench mtx_bench
//! ```
//!
//! Environment:
//!
//! * `MTX_BENCH_CELLS` / `MTX_BENCH_GENES` - shape
//! * `MTX_BENCH_LIBRARY` - median UMIs per cell, sets nnz per cell
//! * `MTX_BENCH_LAYOUT` - comma-separated subset of `genes,cells`
//! * `MTX_BENCH_ONLY` - comma-separated subset of
//!   `read,split,parse,quality,stream,inmem,gz`; default all but `inmem` and
//!   `gz`
//! * `MTX_BENCH_GZ=1` - add the `gz` stage
//! * `MTX_BENCH_GEN_MEM_GB` - memory for the gene-major generator's per-block
//!   transpose, at 8 bytes per entry
//! * `MTX_BENCH_DIR` - where the files go, default the temp directory
//! * `MTX_BENCH_REGEN=1` - rebuild the cached files
//! * `MTX_BENCH_MTX` - run the loader stages on this real file instead
//! * `MTX_BENCH_CELLS_AS_ROWS` - orientation of that file, `1` (default) or
//!   `0`
//! * `RAYON_NUM_THREADS` - thread count

#![cfg(feature = "single-cell")]

use std::fs::File;
use std::hash::Hasher;
use std::hint::black_box;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use memmap2::Mmap;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, LogNormal, Uniform};
use rayon::prelude::*;
use rustc_hash::FxHasher;

use bixverse_rs::single_cell::sc_data::data_io::MinCellQuality;
use bixverse_rs::single_cell::sc_data::mtx_io::MtxReader;

////////////
// Shapes //
////////////

/// Default cell count. Small enough for a quick run; scale with the env.
const DEFAULT_CELLS: usize = 100_000;

/// Default gene count. A full 10x feature set.
const DEFAULT_GENES: usize = 30_000;

/// Default median UMIs per cell, chosen to land near 1,200 nnz per cell.
const DEFAULT_LIBRARY: f64 = 2_300.0;

/// Log-scale spread of the library size.
const LIBRARY_SIZE_LOG_SD: f64 = 0.35;

/// Log-scale spread of per-gene relative expression.
const EXPRESSION_LOG_SD: f64 = 2.0;

/// Dominant genes (MALAT1, ribosomal, mito stand-ins).
const N_DOMINANT_GENES: usize = 40;

/// Share of every library taken by the dominant genes, split as `1 / rank`.
const DOMINANT_SHARE: f64 = 0.25;

/// Cells drawn and formatted per parallel generation batch.
const GEN_BATCH: usize = 8_192;

/// Default memory for the gene-major generator's per-block transpose.
const DEFAULT_GEN_MEM_GB: f64 = 4.0;

/// Mirrors `MTX_PARSE_CHUNK_BYTES` in `mtx_io.rs`: the parallel stages cut
/// the file at the same granularity as the loader.
const CHUNK_BYTES: usize = 64 * 1024 * 1024;

/// Buffer of the sequential read stage.
const SEQ_READ_BYTES: usize = 8 * 1024 * 1024;

/// Mirrors the per-task `BufReader` capacity in `mtx_io.rs`.
const LOADER_BUF_BYTES: usize = 256 * 1024;

/// Seed for the generator.
const SEED: u64 = 0x6D74_7842;

/// QC for the loader stages, the pilot settings: `(min_unique_genes,
/// min_lib_size, min_cells, target_size)`. On the synthetic data nearly
/// everything survives, so the write pass sees the full matrix.
const QC: (usize, usize, usize, f32) = (100, 250, 5, 1e4);

///////////////
// Generator //
///////////////

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

    // spread the dominant genes over the index range
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
/// `(gene, count)` pairs sorted by gene.
fn draw_cell(cell: usize, cdf: &[f64], library: f64) -> Vec<(u32, u32)> {
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

    let mut row: Vec<(u32, u32)> = Vec::new();
    for g in genes {
        match row.last_mut() {
            Some((last, c)) if *last == g => *c += 1,
            _ => row.push((g, 1)),
        }
    }
    row
}

/// Append the decimal form of `v` to `out`.
///
/// ### Params
///
/// * `out` - Output buffer
/// * `v` - Value
#[inline]
fn push_u32(out: &mut Vec<u8>, mut v: u32) {
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    out.extend_from_slice(&tmp[i..]);
}

/// Append one 1-based coordinate line.
///
/// ### Params
///
/// * `out` - Output buffer
/// * `row` - 0-based row
/// * `col` - 0-based column
/// * `value` - Count
#[inline]
fn push_line(out: &mut Vec<u8>, row: u32, col: u32, value: u32) {
    push_u32(out, row + 1);
    out.push(b' ');
    push_u32(out, col + 1);
    out.push(b' ');
    push_u32(out, value);
    out.push(b'\n');
}

/// Write the Matrix Market banner and shape line.
///
/// ### Params
///
/// * `out` - Destination
/// * `rows` - Row count
/// * `cols` - Column count
/// * `nnz` - Entry count
fn write_header(out: &mut impl Write, rows: usize, cols: usize, nnz: usize) {
    write!(
        out,
        "%%MatrixMarket matrix coordinate integer general\n%metadata_json: {{\"generator\": \"mtx_bench\"}}\n{rows} {cols} {nnz}\n"
    )
    .expect("header written");
}

/// Non-zeros per gene over the whole matrix, one full generation sweep.
///
/// ### Params
///
/// * `n_cells` - Number of cells
/// * `cdf` - Cumulative gene shares
/// * `library` - Median library size
///
/// ### Returns
///
/// nnz per gene.
fn gene_nnz(n_cells: usize, cdf: &[f64], library: f64) -> Vec<usize> {
    (0..n_cells)
        .into_par_iter()
        .fold(
            || vec![0usize; cdf.len()],
            |mut acc, cell| {
                for (g, _) in draw_cell(cell, cdf, library) {
                    acc[g as usize] += 1;
                }
                acc
            },
        )
        .reduce(
            || vec![0usize; cdf.len()],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        )
}

/// Write the genes-as-rows file in cell-major entry order.
///
/// ### Params
///
/// * `path` - Output path
/// * `cdf` - Cumulative gene shares
/// * `n_cells` - Number of cells
/// * `library` - Median library size
/// * `nnz` - Total entries, for the shape line
fn write_genes_as_rows(path: &Path, cdf: &[f64], n_cells: usize, library: f64, nnz: usize) {
    let mut out = BufWriter::with_capacity(SEQ_READ_BYTES, File::create(path).expect("create"));
    write_header(&mut out, cdf.len(), n_cells, nnz);
    for b0 in (0..n_cells).step_by(GEN_BATCH) {
        let b1 = (b0 + GEN_BATCH).min(n_cells);
        let texts: Vec<Vec<u8>> = (b0..b1)
            .into_par_iter()
            .map(|cell| {
                let mut text = Vec::new();
                for (g, c) in draw_cell(cell, cdf, library) {
                    push_line(&mut text, g, cell as u32, c);
                }
                text
            })
            .collect();
        for t in texts {
            out.write_all(&t).expect("write");
        }
    }
    out.flush().expect("flush");
}

/// Write the cells-as-rows file in gene-major entry order.
///
/// Transposes in gene blocks sized to `mem_bytes`, regenerating every cell
/// once per block, so memory stays bounded at the cost of extra sweeps.
///
/// ### Params
///
/// * `path` - Output path
/// * `cdf` - Cumulative gene shares
/// * `n_cells` - Number of cells
/// * `library` - Median library size
/// * `per_gene` - nnz per gene
/// * `mem_bytes` - Budget for one block at 8 bytes per entry
fn write_cells_as_rows(
    path: &Path,
    cdf: &[f64],
    n_cells: usize,
    library: f64,
    per_gene: &[usize],
    mem_bytes: usize,
) {
    let n_genes = cdf.len();
    let budget = (mem_bytes / 8).max(1);
    let mut blocks = Vec::new();
    let (mut g0, mut acc) = (0, 0);
    for (g, &n) in per_gene.iter().enumerate() {
        if acc > 0 && acc + n > budget {
            blocks.push(g0..g);
            g0 = g;
            acc = 0;
        }
        acc += n;
    }
    blocks.push(g0..n_genes);
    eprintln!("  gene-major transpose in {} block(s)", blocks.len());

    let mut out = BufWriter::with_capacity(SEQ_READ_BYTES, File::create(path).expect("create"));
    write_header(&mut out, n_cells, n_genes, per_gene.iter().sum());
    for block in blocks {
        let mut per: Vec<Vec<(u32, u32)>> = block
            .clone()
            .map(|g| Vec::with_capacity(per_gene[g]))
            .collect();
        for b0 in (0..n_cells).step_by(GEN_BATCH) {
            let b1 = (b0 + GEN_BATCH).min(n_cells);
            let drawn: Vec<Vec<(u32, u32)>> = (b0..b1)
                .into_par_iter()
                .map(|cell| draw_cell(cell, cdf, library))
                .collect();
            for (k, row) in drawn.into_iter().enumerate() {
                for (g, c) in row {
                    if block.contains(&(g as usize)) {
                        per[g as usize - block.start].push(((b0 + k) as u32, c));
                    }
                }
            }
        }
        let texts: Vec<Vec<u8>> = per
            .par_iter()
            .enumerate()
            .map(|(k, entries)| {
                let g = (block.start + k) as u32;
                let mut text = Vec::with_capacity(entries.len() * 16);
                for &(cell, c) in entries {
                    push_line(&mut text, cell, g, c);
                }
                text
            })
            .collect();
        drop(per);
        for t in texts {
            out.write_all(&t).expect("write");
        }
    }
    out.flush().expect("flush");
}

/// gzip `src` to `dst` with `gzip -1` unless cached.
///
/// ### Params
///
/// * `src` - Plain mtx
/// * `dst` - Output `.mtx.gz`
/// * `regen` - Rebuild even if cached
fn ensure_gz(src: &Path, dst: &Path, regen: bool) {
    if dst.exists() && !regen {
        return;
    }
    let start = Instant::now();
    let status = Command::new("gzip")
        .args(["-1", "-c"])
        .arg(src)
        .stdout(File::create(dst).expect("create gz"))
        .status()
        .expect("gzip on PATH");
    assert!(status.success(), "gzip failed");
    eprintln!("  gzipped in {:.1?}", start.elapsed());
}

/////////////////
// Line parser //
/////////////////

/// Mirror of the private `parse_mtx_line` in `mtx_io.rs`.
///
/// ### Params
///
/// * `line` - One line without its terminator
///
/// ### Returns
///
/// `(row, col, value)` as stored, `None` for a malformed line.
#[inline]
fn parse_mtx_line(line: &[u8]) -> Option<(u32, u32, u32)> {
    let mut i = 0;
    let len = line.len();

    let mut row = 0u32;
    while i < len && line[i].is_ascii_digit() {
        row = row * 10 + (line[i] - b'0') as u32;
        i += 1;
    }
    if i == 0 {
        return None;
    }
    while i < len && (line[i] == b' ' || line[i] == b'\t') {
        i += 1;
    }
    if i >= len {
        return None;
    }
    let mut col = 0u32;
    while i < len && line[i].is_ascii_digit() {
        col = col * 10 + (line[i] - b'0') as u32;
        i += 1;
    }
    while i < len && (line[i] == b' ' || line[i] == b'\t') {
        i += 1;
    }
    if i >= len {
        return None;
    }
    let mut val = 0u32;
    while i < len && line[i].is_ascii_digit() {
        val = val * 10 + (line[i] - b'0') as u32;
        i += 1;
    }
    Some((row, col, val))
}

/// Parse every line of a newline-aligned byte range.
///
/// ### Params
///
/// * `bytes` - Whole lines
///
/// ### Returns
///
/// `(entries parsed, checksum)`; the checksum keeps the parse alive.
fn parse_range(bytes: &[u8]) -> (usize, u64) {
    let (mut n, mut sum) = (0usize, 0u64);
    for line in bytes.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if let Some((r, c, v)) = parse_mtx_line(line) {
            n += 1;
            sum = sum.wrapping_add(r as u64 ^ (c as u64) << 20 ^ (v as u64) << 40);
        }
    }
    (n, sum)
}

/////////////
// Helpers //
/////////////

/// Offset of the first entry line: past the banner, comments and shape line.
///
/// ### Params
///
/// * `bytes` - Whole file
///
/// ### Returns
///
/// Byte offset.
fn data_start(bytes: &[u8]) -> usize {
    let mut pos = 0;
    loop {
        let end = bytes[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(bytes.len(), |e| pos + e + 1);
        let is_comment = bytes[pos] == b'%';
        pos = end;
        if !is_comment {
            return pos;
        }
    }
}

/// Cut `[start, len)` into ranges of about `chunk` bytes ending on newlines.
///
/// ### Params
///
/// * `bytes` - Whole file
/// * `start` - First byte
/// * `chunk` - Target chunk size
///
/// ### Returns
///
/// Half-open byte ranges covering every line once.
fn line_chunks(bytes: &[u8], start: usize, chunk: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut a = start;
    while a < bytes.len() {
        let target = (a + chunk).min(bytes.len());
        let b = bytes[target..]
            .iter()
            .position(|&c| c == b'\n')
            .map_or(bytes.len(), |e| target + e + 1);
        out.push((a, b));
        a = b;
    }
    out
}

/// Time a closure.
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

/// Parse an env var.
///
/// ### Params
///
/// * `key` - Variable name
///
/// ### Returns
///
/// The parsed value, `None` if unset or unparseable.
fn env<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

/// Print one result row.
///
/// ### Params
///
/// * `stage` - Stage name
/// * `t` - Wall time
/// * `bytes` - Bytes the stage covered, for GB/s
/// * `lines` - Entries the stage covered, for ns per entry
fn row(stage: &str, t: Duration, bytes: u64, lines: usize) {
    let s = t.as_secs_f64();
    println!(
        "  {stage:<34} {s:>9.3} s {:>8.2} GB/s {:>8.2} ns/entry",
        bytes as f64 / s / 1e9,
        s * 1e9 / lines.max(1) as f64
    );
}

/// Shape line of an mtx file, through `gzip -dc` when it is gzipped.
///
/// ### Params
///
/// * `path` - Plain or gzipped mtx
///
/// ### Returns
///
/// The three numbers of the shape line.
fn shape_line(path: &Path) -> [usize; 3] {
    let mut magic = [0u8; 2];
    File::open(path)
        .expect("open mtx")
        .read_exact(&mut magic)
        .expect("read magic");
    let mut child = None;
    let reader: Box<dyn BufRead> = if magic == [0x1f, 0x8b] {
        let mut c = Command::new("gzip")
            .arg("-dc")
            .arg(path)
            .stdout(Stdio::piped())
            .spawn()
            .expect("gzip on PATH");
        let out = c.stdout.take().expect("piped stdout");
        child = Some(c);
        Box::new(BufReader::new(out))
    } else {
        Box::new(BufReader::new(File::open(path).expect("open mtx")))
    };
    let line = reader
        .lines()
        .map(|l| l.expect("header line"))
        .find(|l| !l.starts_with('%'))
        .expect("shape line");
    if let Some(mut c) = child {
        let _ = c.kill();
        let _ = c.wait();
    }
    let v: Vec<usize> = line
        .split_whitespace()
        .map(|t| t.parse().expect("numeric shape line"))
        .collect();
    [v[0], v[1], v[2]]
}

/// Time the loader as R's `load_mtx` drives it: one `MtxReader`,
/// `parse_mtx_quality`, then the streaming write. The `inmem` stage gets a
/// fresh reader and the same quality.
///
/// ### Params
///
/// * `path` - Input mtx, plain or gzipped (the latter only if the loader
///   reads gz)
/// * `cells_as_rows` - Orientation
/// * `qc` - Quality thresholds
/// * `bin` - Store path, removed afterwards
/// * `run` - Stage filter
/// * `data_bytes` - Bytes behind the GB/s column
/// * `entries` - Entries behind the ns/entry column
fn run_loader(
    path: &Path,
    cells_as_rows: bool,
    qc: &MinCellQuality,
    bin: &Path,
    run: &dyn Fn(&str) -> bool,
    data_bytes: u64,
    entries: usize,
) {
    let bin_str = bin.to_str().expect("utf-8 path");
    if !(run("quality") || run("stream") || run("inmem")) {
        return;
    }
    let mut reader = MtxReader::new(path, qc.clone(), cells_as_rows).expect("reader");
    println!("  -- parse_mtx_quality (verbose) --");
    let (q, t_quality) = timed(|| reader.parse_mtx_quality(true).expect("quality"));
    row("loader quality", t_quality, data_bytes, entries);

    if run("stream") {
        println!("  -- process_mtx_and_write_bin_streaming (verbose) --");
        let (res, t) = timed(|| {
            reader
                .process_mtx_and_write_bin_streaming(bin_str, &q, true)
                .expect("streaming write")
        });
        row("loader stream write", t, data_bytes, entries);
        row(
            "loader e2e (quality + stream)",
            t_quality + t,
            data_bytes,
            entries,
        );
        let bytes = std::fs::read(bin).expect("read store");
        let mut h = FxHasher::default();
        h.write(&bytes);
        println!(
            "  kept {} cells x {} genes, store {:.2} GB, hash {:016x}",
            res.no_cells,
            res.no_genes,
            bytes.len() as f64 / 1e9,
            h.finish()
        );
    }
    if run("inmem") {
        println!("  -- process_mtx_and_write_bin (verbose) --");
        let (_, t) = timed(|| {
            MtxReader::new(path, qc.clone(), cells_as_rows)
                .expect("reader")
                .process_mtx_and_write_bin(bin_str, &q, true)
                .expect("in-memory write")
        });
        row("loader in-memory write", t, data_bytes, entries);
    }
    let _ = std::fs::remove_file(bin);
}

//////////
// Main //
//////////

fn main() {
    let n_cells: usize = env("MTX_BENCH_CELLS").unwrap_or(DEFAULT_CELLS);
    let n_genes: usize = env("MTX_BENCH_GENES").unwrap_or(DEFAULT_GENES);
    let library: f64 = env("MTX_BENCH_LIBRARY").unwrap_or(DEFAULT_LIBRARY);
    let gen_mem = (env::<f64>("MTX_BENCH_GEN_MEM_GB").unwrap_or(DEFAULT_GEN_MEM_GB) * 1e9) as usize;
    let regen = std::env::var("MTX_BENCH_REGEN").is_ok();
    let gz = std::env::var("MTX_BENCH_GZ").is_ok_and(|v| v == "1");
    let layouts = std::env::var("MTX_BENCH_LAYOUT").unwrap_or_else(|_| "genes,cells".into());
    let only = std::env::var("MTX_BENCH_ONLY").unwrap_or_default();
    let run = |name: &str| {
        if only.is_empty() {
            !matches!(name, "inmem" | "gz") || (name == "gz" && gz)
        } else {
            only.split(',').any(|o| o.trim() == name)
        }
    };
    let dir: PathBuf = std::env::var("MTX_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());

    println!(
        "mtx_bench: {n_cells} cells x {n_genes} genes, library {library}, {} threads",
        rayon::current_num_threads()
    );

    let stem = format!("bixverse_mtx_bench_{n_cells}x{n_genes}_{library}_{SEED:x}");
    let path_of = |layout: &str| dir.join(format!("{stem}_{layout}_as_rows.mtx"));

    // shared generation state, built only if some file is missing
    let needs_gen = layouts
        .split(',')
        .any(|l| regen || !path_of(l.trim()).exists());
    if needs_gen && std::env::var("MTX_BENCH_MTX").is_err() {
        let (per_gene, t) = timed(|| {
            let cdf = gene_cdf(n_genes);
            (gene_nnz(n_cells, &cdf, library), cdf)
        });
        let (per_gene, cdf) = per_gene;
        let nnz: usize = per_gene.iter().sum();
        eprintln!(
            "nnz sweep {:.1?}: {nnz} entries, {:.0} per cell",
            t,
            nnz as f64 / n_cells as f64
        );
        for layout in layouts.split(',').map(str::trim) {
            let path = path_of(layout);
            if path.exists() && !regen {
                continue;
            }
            let t = Instant::now();
            match layout {
                "genes" => write_genes_as_rows(&path, &cdf, n_cells, library, nnz),
                "cells" => write_cells_as_rows(&path, &cdf, n_cells, library, &per_gene, gen_mem),
                other => panic!("MTX_BENCH_LAYOUT: unknown layout {other}"),
            }
            eprintln!("{layout}-as-rows file written in {:.1?}", t.elapsed());
        }
    }

    let (min_unique_genes, min_lib_size, min_cells, target_size) = QC;
    let qc = MinCellQuality {
        min_unique_genes,
        min_lib_size,
        min_cells,
        target_size,
    };

    if let Ok(real) = std::env::var("MTX_BENCH_MTX") {
        let path = PathBuf::from(real);
        let cells_as_rows = std::env::var("MTX_BENCH_CELLS_AS_ROWS").map_or(true, |v| v != "0");
        let size = std::fs::metadata(&path).expect("stat mtx").len();
        let [rows, cols, entries] = shape_line(&path);
        println!(
            "\n{}: {rows} x {cols}, {entries} entries, {:.2} GB on disk, cells as rows: {cells_as_rows}",
            path.display(),
            size as f64 / 1e9
        );
        let bin = dir.join(format!(
            "bixverse_mtx_bench_real_{}_cells.bin",
            path.file_name().expect("file name").to_string_lossy()
        ));
        run_loader(&path, cells_as_rows, &qc, &bin, &run, size, entries);
        return;
    }

    for layout in layouts.split(',').map(str::trim) {
        let path = path_of(layout);
        let cells_as_rows = layout == "cells";
        let file = File::open(&path).expect("open mtx");
        let size = file.metadata().expect("stat").len();
        // SAFETY: the bench owns the file and nothing writes it while mapped.
        let map = unsafe { Mmap::map(&file).expect("mmap") };
        let start = data_start(&map);
        let entries: usize = std::str::from_utf8(&map[..start])
            .expect("ascii header")
            .lines()
            .last()
            .and_then(|l| l.split_whitespace().nth(2))
            .and_then(|n| n.parse().ok())
            .expect("shape line");
        let data_bytes = size - start as u64;
        let chunks = line_chunks(&map, start, CHUNK_BYTES);

        println!(
            "\n{layout} as rows: {} ({:.2} GB, {entries} entries, {:.1} bytes/entry)",
            path.display(),
            size as f64 / 1e9,
            data_bytes as f64 / entries as f64
        );

        if run("read") {
            let (n, t) = timed(|| {
                let mut f = File::open(&path).expect("open");
                let mut buf = vec![0u8; SEQ_READ_BYTES];
                let mut n = 0u64;
                loop {
                    let k = f.read(&mut buf).expect("read");
                    if k == 0 {
                        break;
                    }
                    n += black_box(&buf[..k]).len() as u64;
                }
                n
            });
            row("read seq (8 MB reads)", t, n, entries);

            let (n, t) = timed(|| {
                chunks
                    .par_iter()
                    .map_init(Vec::new, |buf: &mut Vec<u8>, &(a, b)| {
                        let mut f = File::open(&path).expect("open");
                        f.seek(SeekFrom::Start(a as u64)).expect("seek");
                        buf.resize(b - a, 0);
                        f.read_exact(buf).expect("read");
                        black_box(&buf[..]).len() as u64
                    })
                    .sum::<u64>()
            });
            row("read par (64 MB chunks)", t, n, entries);
        }

        if run("split") {
            let (n, t) = timed(|| {
                chunks
                    .par_iter()
                    .map(|&(a, b)| {
                        let mut f = File::open(&path).expect("open");
                        f.seek(SeekFrom::Start(a as u64)).expect("seek");
                        let mut r = BufReader::with_capacity(LOADER_BUF_BYTES, f);
                        let mut line = Vec::with_capacity(64);
                        let (mut pos, mut n) = (a, 0usize);
                        while pos < b {
                            line.clear();
                            let k = r.read_until(b'\n', &mut line).expect("read");
                            if k == 0 {
                                break;
                            }
                            pos += k;
                            n += black_box(&line).len().min(1);
                        }
                        n
                    })
                    .sum::<usize>()
            });
            assert_eq!(n, entries, "read_until line count");
            row("split read_until par (as shipped)", t, data_bytes, entries);

            let (n, t) = timed(|| map[start..].iter().filter(|&&b| b == b'\n').count());
            assert_eq!(n, entries, "newline count");
            row("split mmap count seq", t, data_bytes, entries);

            let (n, t) = timed(|| {
                chunks
                    .par_iter()
                    .map(|&(a, b)| map[a..b].iter().filter(|&&c| c == b'\n').count())
                    .sum::<usize>()
            });
            assert_eq!(n, entries, "newline count");
            row("split mmap count par", t, data_bytes, entries);
        }

        if run("parse") {
            let ((n, sum), t) = timed(|| parse_range(&map[start..]));
            assert_eq!(n, entries, "parsed entries");
            black_box(sum);
            row("parse mmap seq", t, data_bytes, entries);

            let ((n, sum), t) = timed(|| {
                chunks
                    .par_iter()
                    .map(|&(a, b)| parse_range(&map[a..b]))
                    .reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1.wrapping_add(y.1)))
            });
            assert_eq!(n, entries, "parsed entries");
            black_box(sum);
            row("parse mmap par", t, data_bytes, entries);
        }

        drop(map);

        let bin = dir.join(format!("{stem}_{layout}_cells.bin"));
        run_loader(&path, cells_as_rows, &qc, &bin, &run, data_bytes, entries);

        if run("gz") {
            let gz_path = path.with_extension("mtx.gz");
            ensure_gz(&path, &gz_path, regen);
            let gz_size = std::fs::metadata(&gz_path).expect("stat gz").len();
            let (status, t) = timed(|| {
                Command::new("gzip")
                    .arg("-dc")
                    .arg(&gz_path)
                    .stdout(Stdio::null())
                    .status()
                    .expect("gzip on PATH")
            });
            assert!(status.success(), "gzip -dc failed");
            println!(
                "  gz file {:.2} GB (ratio {:.2})",
                gz_size as f64 / 1e9,
                size as f64 / gz_size as f64
            );
            row("gzip -dc > /dev/null (1 thread)", t, size, entries);
        }
    }
}
