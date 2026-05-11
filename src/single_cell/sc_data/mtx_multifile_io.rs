//! Multi-mtx loading: scan multiple mtx files, apply global gene QC,
//! and write all cells into a single binary file. Mirrors the h5ad
//! multi-loader design.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::*;

////////////////
// File tasks //
////////////////

/// Per-file task descriptor for multi-mtx loading
pub struct MtxFileTask {
    /// Experimental identifier
    pub exp_id: String,
    /// Path to the (decompressed) .mtx file
    pub mtx_path: String,
    /// file-local gene idx -> universe gene idx; None if gene not in
    /// the (intersected) universe
    pub gene_local_to_universe: Vec<Option<usize>>,
    /// Are cells the rows in this file
    pub cells_as_rows: bool,
}

/// Per-file QC output returned to the caller
pub struct MtxFileQcResult {
    /// Experimental identifier
    pub exp_id: String,
    /// Cells to keep (file-local 0-indexed indices)
    pub cells_to_keep: Vec<usize>,
    /// Library sizes of the kept cells
    pub lib_size: Vec<usize>,
    /// Number of features per kept cell
    pub nnz: Vec<usize>,
}

/// Final result from multi-mtx loading
pub struct MultiMtxResult {
    /// Universe indices of the genes that survived the global QC
    pub global_gene_indices: Vec<usize>,
    /// Total cells ingested
    pub total_cells: usize,
    /// Total genes/features ingested
    pub total_genes: usize,
    /// Per file QC information
    pub per_file: Vec<MtxFileQcResult>,
}

/////////////
// Helpers //
/////////////

/// Skip comment lines and read the dimension header from an mtx file.
///
/// ### Params
///
/// * `reader` - Mutable reference to the buffered reader
///
/// ### Returns
///
/// The dimension header line (first non-comment line)
fn read_mtx_header(reader: &mut BufReader<File>) -> std::io::Result<String> {
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        if !line.starts_with('%') {
            break;
        }
    }
    Ok(line)
}

/// Find byte boundaries for parallel chunk scanning of an mtx file.
///
/// Skips the comment block, then splits the data region into `num_chunks`
/// roughly equal ranges aligned to newline boundaries.
///
/// ### Params
///
/// * `path` - Path to the mtx file
/// * `num_chunks` - Desired number of chunks
///
/// ### Returns
///
/// Vector of `(start, end)` byte offsets, one per chunk
fn find_chunk_boundaries(path: &Path, num_chunks: usize) -> std::io::Result<Vec<(u64, u64)>> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(256 * 1024, file);

    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        if !line.starts_with('%') {
            break;
        }
    }
    let data_start = reader.stream_position()?;
    let data_size = file_size - data_start;
    let chunk_size = data_size / num_chunks as u64;

    let mut boundaries = vec![(data_start, data_start)];

    for i in 1..num_chunks {
        let target_pos = data_start + chunk_size * i as u64;
        if target_pos >= file_size {
            break;
        }
        reader.seek(std::io::SeekFrom::Start(target_pos))?;
        let mut byte = [0u8; 1];
        while reader.read(&mut byte)? > 0 {
            if byte[0] == b'\n' {
                break;
            }
        }
        let boundary = reader.stream_position()?;
        boundaries.push((boundary, boundary));
    }
    boundaries.push((file_size, file_size));

    for i in 0..boundaries.len() - 1 {
        boundaries[i].1 = boundaries[i + 1].0;
    }
    boundaries.pop();
    Ok(boundaries)
}

/// Read a line into a byte buffer with trailing newline and CR stripped.
///
/// ### Params
///
/// * `reader` - Mutable reference to the buffered reader
/// * `buf` - Buffer to read into; cleared before each read
///
/// ### Returns
///
/// `Ok(true)` if a line was read, `Ok(false)` at EOF
#[inline]
fn read_trimmed_line(reader: &mut BufReader<File>, buf: &mut Vec<u8>) -> std::io::Result<bool> {
    buf.clear();
    let n = reader.read_until(b'\n', buf)?;
    if n == 0 {
        return Ok(false);
    }
    if buf.last() == Some(&b'\n') {
        buf.pop();
    }
    if buf.last() == Some(&b'\r') {
        buf.pop();
    }
    Ok(true)
}

/// Accumulate per-universe-gene NNZ counts for a single byte chunk of an mtx
/// file.
///
/// ### Params
///
/// * `path` - Path to the mtx file
/// * `start` - Start byte offset of the chunk
/// * `end` - End byte offset of the chunk
/// * `universe_size` - Total number of genes in the universe
/// * `gene_local_to_universe` - Mapping from file-local gene index to universe
///   index
/// * `cells_as_rows` - Whether cells occupy rows (true) or columns (false)
///
/// ### Returns
///
/// Per-universe-gene NNZ counts as `Vec<u32>`
fn scan_gene_nnz_chunk(
    path: &Path,
    start: u64,
    end: u64,
    universe_size: usize,
    gene_local_to_universe: &[Option<usize>],
    cells_as_rows: bool,
) -> Vec<u32> {
    let mut local = vec![0u32; universe_size];

    let Ok(file) = File::open(path) else {
        return local;
    };
    let mut reader = BufReader::with_capacity(256 * 1024, file);
    if reader.seek(std::io::SeekFrom::Start(start)).is_err() {
        return local;
    }

    let mut buf = Vec::with_capacity(64);
    let mut read = 0u64;
    while read < (end - start) {
        let len = match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        read += len as u64;

        let blen = buf.len();
        let trim = if blen > 0 && buf[blen - 1] == b'\n' {
            if blen > 1 && buf[blen - 2] == b'\r' {
                blen - 2
            } else {
                blen - 1
            }
        } else {
            blen
        };

        if trim < 3 {
            buf.clear();
            continue;
        }

        if let Some((row, col, _)) = parse_mtx_coord(&buf[..trim]) {
            let local_gene = if cells_as_rows {
                (col - 1) as usize
            } else {
                (row - 1) as usize
            };
            if let Some(&Some(u_idx)) = gene_local_to_universe.get(local_gene) {
                local[u_idx] += 1;
            }
        }
        buf.clear();
    }
    local
}

/// Parse a single coordinate-format mtx line from raw bytes.
///
/// Values exceeding `u16::MAX` are saturated.
///
/// ### Params
///
/// * `line` - Raw bytes of a single trimmed mtx line
///
/// ### Returns
///
/// `Some((row, col, value))` on success, `None` if the line is malformed
fn scan_gene_nnz(task: &MtxFileTask, universe_size: usize) -> Result<Vec<usize>, BixverseErrors> {
    let path = PathBuf::from(&task.mtx_path);
    let file_size = std::fs::metadata(&path)?.len();
    let num_chunks = ((file_size / (64 * 1024 * 1024)) as usize).max(1);
    let boundaries = find_chunk_boundaries(&path, num_chunks)?;

    let per_chunk: Vec<Vec<u32>> = boundaries
        .par_iter()
        .map(|&(s, e)| {
            scan_gene_nnz_chunk(
                &path,
                s,
                e,
                universe_size,
                &task.gene_local_to_universe,
                task.cells_as_rows,
            )
        })
        .collect();

    let mut total = vec![0usize; universe_size];
    for c in per_chunk {
        for (i, &v) in c.iter().enumerate() {
            total[i] += v as usize;
        }
    }
    Ok(total)
}

/// Parse a single coordinate-format mtx line from raw bytes.
///
/// Values exceeding `u16::MAX` are saturated.
///
/// ### Params
///
/// * `line` - Raw bytes of a single trimmed mtx line
///
/// ### Returns
///
/// `Some((row, col, value))` on success, `None` if the line is malformed
#[inline]
fn parse_mtx_coord(line: &[u8]) -> Option<(u32, u32, u16)> {
    let mut i = 0;
    let len = line.len();

    let mut row = 0u32;
    let start = i;
    while i < len && line[i].is_ascii_digit() {
        row = row * 10 + (line[i] - b'0') as u32;
        i += 1;
    }
    if i == start {
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
    Some((row, col, val.min(u16::MAX as u32) as u16))
}

/// Read the cell and gene counts from an mtx file header.
///
/// ### Params
///
/// * `path` - Path to the mtx file
/// * `cells_as_rows` - Whether cells occupy rows (true) or columns (false)
///
/// ### Returns
///
/// `(no_cells, no_genes)` parsed from the dimension header
fn read_mtx_dims(path: &Path, cells_as_rows: bool) -> std::io::Result<(usize, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let header = read_mtx_header(&mut reader)?;
    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid mtx header",
        ));
    }
    let a: usize = parts[0]
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid count"))?;
    let b: usize = parts[1]
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid count"))?;
    Ok(if cells_as_rows { (a, b) } else { (b, a) })
}

/// Scan a single mtx file for per-cell NNZ and library size over the final gene
/// set.
///
/// Splits the file into chunks and scans them in parallel, ignoring genes not
/// present in `gene_local_to_final`.
///
/// ### Params
///
/// * `task` - The file task descriptor
/// * `gene_local_to_final` - Mapping from file-local gene index to final gene
///   index
///
/// ### Returns
///
/// Per-cell `(nnz, lib_size)` pairs indexed by file-local cell index
fn scan_cell_stats(
    task: &MtxFileTask,
    gene_local_to_final: &[Option<usize>],
) -> Result<Vec<(usize, f32)>, BixverseErrors> {
    let path = PathBuf::from(&task.mtx_path);
    let (no_cells, _) = read_mtx_dims(&path, task.cells_as_rows)?;

    let file_size = std::fs::metadata(&path)?.len();
    let num_chunks = ((file_size / (64 * 1024 * 1024)) as usize).max(1);
    let boundaries = find_chunk_boundaries(&path, num_chunks)?;

    let per_chunk: Vec<(Vec<u32>, Vec<f32>)> = boundaries
        .par_iter()
        .map(|&(s, e)| {
            let mut unique = vec![0u32; no_cells];
            let mut lib = vec![0.0f32; no_cells];
            let Ok(file) = File::open(&path) else {
                return (unique, lib);
            };
            let mut reader = BufReader::with_capacity(256 * 1024, file);
            if reader.seek(std::io::SeekFrom::Start(s)).is_err() {
                return (unique, lib);
            }
            let mut buf = Vec::with_capacity(64);
            let mut read = 0u64;
            while read < (e - s) {
                let len = match reader.read_until(b'\n', &mut buf) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(_) => break,
                };
                read += len as u64;
                let blen = buf.len();
                let trim = if blen > 0 && buf[blen - 1] == b'\n' {
                    if blen > 1 && buf[blen - 2] == b'\r' {
                        blen - 2
                    } else {
                        blen - 1
                    }
                } else {
                    blen
                };
                if trim < 3 {
                    buf.clear();
                    continue;
                }
                if let Some((row, col, value)) = parse_mtx_coord(&buf[..trim]) {
                    let (cell_idx, gene_idx) = if task.cells_as_rows {
                        ((row - 1) as usize, (col - 1) as usize)
                    } else {
                        ((col - 1) as usize, (row - 1) as usize)
                    };
                    if let Some(&Some(_)) = gene_local_to_final.get(gene_idx)
                        && cell_idx < no_cells
                    {
                        unique[cell_idx] += 1;
                        lib[cell_idx] += value as f32;
                    }
                }
                buf.clear();
            }
            (unique, lib)
        })
        .collect();

    let mut unique = vec![0usize; no_cells];
    let mut lib = vec![0.0f32; no_cells];
    for (u, l) in per_chunk {
        for i in 0..no_cells {
            unique[i] += u[i] as usize;
            lib[i] += l[i];
        }
    }
    Ok(unique.into_iter().zip(lib).collect())
}

/// Write the kept cells of a single mtx file into the unified sparse writer.
///
/// Reads the file sequentially, filters to kept cells and final genes, sorts
/// each cell's entries by gene index, and flushes them via `CsrCellChunk`.
///
/// ### Params
///
/// * `task` - The file task descriptor
/// * `cells_to_keep` - File-local 0-indexed cell indices to include
/// * `gene_local_to_final` - Mapping from file-local gene index to final gene
///   index
/// * `target_size` - Target library size for normalisation
/// * `cell_offset` - Global cell offset for this file's cells in the unified
///   output
/// * `writer` - Mutable reference to the unified sparse writer
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Per-file QC statistics for the written cells
fn write_mtx_file_cells(
    task: &MtxFileTask,
    cells_to_keep: &[usize],
    gene_local_to_final: &[Option<usize>],
    target_size: f32,
    cell_offset: usize,
    writer: &mut CellGeneSparseWriter,
    verbose: bool,
) -> Result<MtxFileQcResult, BixverseErrors> {
    if verbose {
        println!(
            "  Writing {} ({} cells)...",
            task.exp_id,
            cells_to_keep.len().separate_with_underscores()
        );
    }

    let path = PathBuf::from(&task.mtx_path);
    let kept_set: rustc_hash::FxHashSet<usize> = cells_to_keep.iter().copied().collect();
    let cell_old_to_new: FxHashMap<usize, usize> = cells_to_keep
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect();

    // (gene_final_idx, raw_count) per kept cell
    let mut cell_data: Vec<Vec<(u32, u16)>> = vec![Vec::new(); cells_to_keep.len()];

    let file = File::open(&path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let _ = read_mtx_header(&mut reader)?;

    let mut buf = Vec::with_capacity(64);
    while read_trimmed_line(&mut reader, &mut buf)? {
        if buf.is_empty() {
            continue;
        }
        let Some((row, col, value)) = parse_mtx_coord(&buf) else {
            continue;
        };
        let (old_cell, old_gene) = if task.cells_as_rows {
            ((row - 1) as usize, (col - 1) as usize)
        } else {
            ((col - 1) as usize, (row - 1) as usize)
        };
        if !kept_set.contains(&old_cell) {
            continue;
        }
        let Some(&Some(final_gene)) = gene_local_to_final.get(old_gene) else {
            continue;
        };
        let new_cell = cell_old_to_new[&old_cell];
        cell_data[new_cell].push((final_gene as u32, value));
    }

    let mut lib_size = Vec::with_capacity(cells_to_keep.len());
    let mut nnz = Vec::with_capacity(cells_to_keep.len());

    for (i, mut data) in cell_data.into_iter().enumerate() {
        data.sort_unstable_by_key(|(g, _)| *g);
        let gene_indices: Vec<u32> = data.iter().map(|(g, _)| *g).collect();
        let gene_counts: Vec<u16> = data.iter().map(|(_, c)| *c).collect();

        let chunk = CsrCellChunk::from_data(
            &gene_counts,
            &gene_indices,
            cell_offset + i,
            target_size,
            true,
        );
        let (nnz_i, lib_i) = chunk.get_qc_info();
        nnz.push(nnz_i);
        lib_size.push(lib_i);
        writer.write_cell_chunk(chunk)?;
    }

    Ok(MtxFileQcResult {
        exp_id: task.exp_id.clone(),
        cells_to_keep: cells_to_keep.to_vec(),
        lib_size,
        nnz,
    })
}

//////////
// Main //
//////////

/// Load multiple mtx files into a single binary.
///
/// 1. Parallel per-file scan of gene NNZ against the intersected universe
/// 2. Apply global `min_cells` to determine the final gene set
/// 3. Parallel per-file scan of cell stats against the final gene set
/// 4. Apply per-cell `min_unique_genes` / `min_lib_size`
/// 5. Stream kept cells into the unified binary
///
/// ### Params
///
/// * `tasks` - Slice of per-file task descriptors
/// * `bin_path` - Output path for the unified binary
/// * `universe_size` - Total number of genes in the shared universe
/// * `cell_qc` - Cell and gene quality thresholds and target library size
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Summary of the global gene set, total cell and gene counts, and per-file QC
/// results
pub fn multi_mtx_to_file<P: AsRef<Path>>(
    tasks: &[MtxFileTask],
    bin_path: P,
    universe_size: usize,
    cell_qc: &MinCellQuality,
    verbose: bool,
) -> Result<MultiMtxResult, BixverseErrors> {
    let total_start = Instant::now();

    if verbose {
        println!("Scan 1/2: gene NNZ across {} mtx files...", tasks.len());
    }
    let completed = Arc::new(AtomicUsize::new(0));
    let report_interval = (tasks.len() / 10).max(1);

    let per_file_nnz: Vec<Vec<usize>> = tasks
        .par_iter()
        .map(|task| {
            let res = scan_gene_nnz(task, universe_size);
            if verbose {
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(report_interval) || done == tasks.len() {
                    println!("  Scanned {}/{} files", done, tasks.len());
                }
            }
            res
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut global_gene_nnz = vec![0usize; universe_size];
    for f in &per_file_nnz {
        for (i, &c) in f.iter().enumerate() {
            global_gene_nnz[i] += c;
        }
    }

    let mut universe_to_final = vec![None; universe_size];
    let mut final_idx = 0usize;
    for (u, &nnz) in global_gene_nnz.iter().enumerate() {
        if nnz >= cell_qc.min_cells {
            universe_to_final[u] = Some(final_idx);
            final_idx += 1;
        }
    }
    let global_gene_indices: Vec<usize> = (0..universe_size)
        .filter(|&i| universe_to_final[i].is_some())
        .collect();
    let total_genes = global_gene_indices.len();

    if verbose {
        println!(
            "  Genes passing global QC: {} / {}",
            total_genes.separate_with_underscores(),
            universe_size.separate_with_underscores()
        );
    }

    let composed: Vec<Vec<Option<usize>>> = tasks
        .iter()
        .map(|t| {
            t.gene_local_to_universe
                .iter()
                .map(|opt| opt.and_then(|u| universe_to_final[u]))
                .collect()
        })
        .collect();

    if verbose {
        println!("Scan 2/2: cell stats with final gene set...");
    }
    let completed = Arc::new(AtomicUsize::new(0));

    let per_file_cell_stats: Vec<Vec<(usize, f32)>> = tasks
        .par_iter()
        .zip(composed.par_iter())
        .map(|(task, mapping)| {
            let res = scan_cell_stats(task, mapping);
            if verbose {
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(report_interval) || done == tasks.len() {
                    println!("  Scanned {}/{} files", done, tasks.len());
                }
            }
            res
        })
        .collect::<Result<Vec<_>, _>>()?;

    let per_file_cells: Vec<Vec<usize>> = per_file_cell_stats
        .iter()
        .map(|stats| {
            stats
                .iter()
                .enumerate()
                .filter(|(_, (u, l))| {
                    *u >= cell_qc.min_unique_genes && *l >= cell_qc.min_lib_size as f32
                })
                .map(|(i, _)| i)
                .collect()
        })
        .collect();

    let total_cells: usize = per_file_cells.iter().map(|v| v.len()).sum();

    if verbose {
        for (i, t) in tasks.iter().enumerate() {
            println!(
                "  {}: {} cells passing QC",
                t.exp_id,
                per_file_cells[i].len().separate_with_underscores()
            );
        }
        println!("  Total cells: {}", total_cells.separate_with_underscores());
    }

    if verbose {
        println!("Writing cells to binary...");
    }
    let mut writer = CellGeneSparseWriter::new(&bin_path, true, total_cells, total_genes)?;
    let mut cell_offset = 0usize;
    let mut per_file_results = Vec::with_capacity(tasks.len());

    for (idx, task) in tasks.iter().enumerate() {
        let res = write_mtx_file_cells(
            task,
            &per_file_cells[idx],
            &composed[idx],
            cell_qc.target_size,
            cell_offset,
            &mut writer,
            verbose,
        )?;
        cell_offset += per_file_cells[idx].len();
        per_file_results.push(res);
    }

    writer.finalise()?;

    if verbose {
        println!("Multi-mtx loading complete: {:.2?}", total_start.elapsed());
    }

    Ok(MultiMtxResult {
        global_gene_indices,
        total_cells,
        total_genes,
        per_file: per_file_results,
    })
}
