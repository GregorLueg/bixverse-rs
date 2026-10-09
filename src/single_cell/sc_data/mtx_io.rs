//! Contains the MTX file-related parts to do read in data from mtx files and
//! transform them into the binarised files for usage in bixverse-rs

use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Result as IoResult, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CellOnFileQuality, INDEX_DROPPED, compress_cell_row, dense_index_map,
    write_cell_rows,
};

/////////
// MTX //
/////////

/// Byte width of one record in a temporary bucket spill file:
/// `cell_idx: u32`, `gene_idx: u32`, `count: u32`.
///
/// These files live and die inside a single
/// [`MtxReader::process_mtx_and_write_bin_streaming`] call, so the layout
/// carries no on-disk compatibility obligation.
const BUCKET_RECORD_LEN: usize = 12;

/// Bytes of mtx text parsed per parallel task in the bucketing pass.
const MTX_PARSE_CHUNK_BYTES: u64 = 64 * 1024 * 1024;

/// Size at which a worker's per-bucket record buffer is appended to the shared
/// bucket file. Caps what the bucketing pass holds at
/// `n_threads * n_buckets * BUCKET_FLUSH_BYTES`: 128 MB at 16 threads and the
/// 128-bucket ceiling.
const BUCKET_FLUSH_BYTES: usize = 64 * 1024;

/// MTX file metadata
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MtxHeader {
    /// Number of cells identified in the .mtx header.
    pub total_cells: usize,
    /// Number of genes identified in the .mtx header.
    pub total_genes: usize,
    /// Number of entries identified in the .mtx header.
    pub total_entries: usize,
}

/// MTX final data
///
/// Structure to store final results after reading in the .mtx file
#[derive(Debug, Clone)]
pub struct MtxFinalData {
    /// Structure containing the information on which cells/genes to keep and
    /// library size and NNZ for cells.
    pub cell_qc: CellQuality,
    /// No of genes that were read in
    pub no_genes: usize,
    /// No of cells that were read in
    pub no_cells: usize,
}

/// RAII guard that removes temporary bucket files on drop.
///
/// Ensures temp files are cleaned up even if the bucketing or writing pass
/// returns early due to an I/O error or panic.
struct TempFileGuard(Vec<PathBuf>);

/// Drop implementation for `TempFileGuard`
///
/// Attempts to remove each tracked temp file. Errors are silently ignored since
/// this runs during unwind and the files may already have been removed by the
/// writing pass.
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// MTX Reader for bixverse
pub struct MtxReader {
    /// Path to the mtx file
    path: PathBuf,
    /// Buffered reader of the mtx file
    reader: BufReader<File>,
    /// Header of the MtxFile
    header: MtxHeader,
    /// Cell and gene quality parameters to apply
    qc_params: MinCellQuality,
    /// Are the cells stored as rows
    cells_as_rows: bool,
}

impl MtxReader {
    /// Generate a new instance of the reader
    ///
    /// ### Params
    ///
    /// * `path` - Path to the mtx file.
    /// * `qc_params` - The min quality parameters that genes and cells have to
    ///   reach and the target library size
    /// * `cells_as_rows` - Boolean. Are the cells the rows (= true) or columns in
    ///   the mtx file.
    pub fn new<P: AsRef<Path>>(
        path: P,
        qc_params: MinCellQuality,
        cells_as_rows: bool,
    ) -> Result<Self, BixverseErrors> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mut reader = BufReader::with_capacity(1024 * 1024, file);

        let header = Self::parse_header(&mut reader, cells_as_rows)?;

        Ok(Self {
            path,
            reader,
            header,
            qc_params,
            cells_as_rows,
        })
    }

    /// Parse the header of the mtx file
    ///
    /// ### Returns
    ///
    /// The `MtxHeader`
    fn parse_header(
        reader: &mut BufReader<File>,
        cells_as_rows: bool,
    ) -> Result<MtxHeader, BixverseErrors> {
        let mut line = String::new();

        loop {
            line.clear();
            reader.read_line(&mut line)?;
            if !line.starts_with('%') {
                break;
            }
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() != 3 {
            return Err(BixverseErrors::MtxHeaderInvalid(
                "expected three whitespace-separated fields on the shape line",
            ));
        }

        let parse_field = |s: &str, field: &'static str| -> Result<usize, BixverseErrors> {
            s.parse()
                .map_err(|_| BixverseErrors::MtxParseError { field })
        };

        // Header is either `cells genes entries` or `genes cells entries`.
        let (cell_pos, gene_pos) = if cells_as_rows { (0, 1) } else { (1, 0) };

        Ok(MtxHeader {
            total_cells: parse_field(parts[cell_pos], "cell count")?,
            total_genes: parse_field(parts[gene_pos], "gene count")?,
            total_entries: parse_field(parts[2], "entry count")?,
        })
    }

    /// Helper to parse the file to understand which cells to keep
    ///
    /// ### Params
    ///
    /// * `verbose` - Controls verbosity of the function.
    ///
    /// ### Returns
    ///
    /// The CellOnFileQuality file containing the indices and mappings
    /// for the cells/genes to keep.
    pub fn parse_mtx_quality(&mut self, verbose: bool) -> IoResult<CellOnFileQuality> {
        const CHUNK_SIZE: u64 = 64 * 1024 * 1024;
        let file_size = self.reader.get_ref().metadata()?.len();
        let num_chunks = ((file_size / CHUNK_SIZE) as usize).max(1);

        if verbose {
            println!("First file pass - getting gene statistics:");
        }

        let first_scan_time = Instant::now();

        let boundaries = self.find_chunk_boundaries(num_chunks)?;
        let completed_chunks = Arc::new(AtomicUsize::new(0));
        let report_interval = (num_chunks / 10).max(1);

        // per-thread gene counts. MTX coordinate format guarantees each
        // (row, col) pair appears at most once, so counts can be summed across
        // chunks without double-counting.
        let results: Vec<Vec<u32>> = boundaries
            .par_iter()
            .map(|&(start, end)| {
                let mut local_gene_counts = vec![0u32; self.header.total_genes];

                if let Ok(file) = File::open(&self.path) {
                    let mut reader = BufReader::with_capacity(256 * 1024, file);
                    if reader.seek(std::io::SeekFrom::Start(start)).is_ok() {
                        let mut line_buffer = Vec::with_capacity(64);
                        let mut bytes_read = 0u64;

                        while bytes_read < (end - start) {
                            line_buffer.clear();
                            if let Ok(n) = reader.read_until(b'\n', &mut line_buffer) {
                                if n == 0 {
                                    break;
                                }
                                bytes_read += n as u64;

                                let len = line_buffer.len();
                                if len < 3 {
                                    continue;
                                }
                                let trim_end = if line_buffer[len - 1] == b'\n' {
                                    if len > 1 && line_buffer[len - 2] == b'\r' {
                                        len - 2
                                    } else {
                                        len - 1
                                    }
                                } else {
                                    len
                                };

                                if let Some((row, col, _)) =
                                    parse_mtx_line(&line_buffer[..trim_end])
                                {
                                    let gene_idx = if self.cells_as_rows {
                                        (col - 1) as usize
                                    } else {
                                        (row - 1) as usize
                                    };

                                    if gene_idx < self.header.total_genes {
                                        local_gene_counts[gene_idx] += 1;
                                    }
                                }
                            }
                        }
                    }
                }

                if verbose {
                    let completed = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                    if completed.is_multiple_of(report_interval) || completed == num_chunks {
                        let progress = (completed as f64 / num_chunks as f64 * 100.0) as usize;
                        println!(
                            "  Processed {}% of chunks ({}/{})",
                            progress, completed, num_chunks
                        );
                    }
                }

                local_gene_counts
            })
            .collect();

        // Sum counts across threads, then filter genes.
        let genes_to_keep: Vec<usize> = (0..self.header.total_genes)
            .into_par_iter()
            .filter(|&i| {
                let total: u32 = results.iter().map(|local| local[i]).sum();
                total as usize >= self.qc_params.min_cells
            })
            .collect();
        let mut genes_to_keep_lookup = vec![false; self.header.total_genes];
        for &g in &genes_to_keep {
            genes_to_keep_lookup[g] = true;
        }

        drop(results);

        let first_scan_end = first_scan_time.elapsed();

        if verbose {
            println!("First pass done: {:.2?}", first_scan_end);
            println!("Second pass - cell statistics:");
        }

        let second_scan_time = Instant::now();
        let completed_chunks = Arc::new(AtomicUsize::new(0));

        // Parallel second pass - cell stats with filtered genes
        // folded per worker rather than per chunk, so the n_cells-sized
        // accumulators do not all live until the merge
        let merged = boundaries
            .par_iter()
            .fold(
                || vec![(0u32, 0u32); self.header.total_cells],
                |mut local_cell_stats, &(start, end)| {
                    if let Ok(file) = File::open(&self.path) {
                        let mut reader = BufReader::with_capacity(256 * 1024, file);
                        if reader.seek(std::io::SeekFrom::Start(start)).is_ok() {
                            let mut line_buffer = Vec::with_capacity(64);
                            let mut bytes_read = 0u64;

                            while bytes_read < (end - start) {
                                line_buffer.clear();
                                if let Ok(n) = reader.read_until(b'\n', &mut line_buffer) {
                                    if n == 0 {
                                        break;
                                    }
                                    bytes_read += n as u64;

                                    let len = line_buffer.len();
                                    if len < 3 {
                                        continue;
                                    }
                                    let trim_end = if line_buffer[len - 1] == b'\n' {
                                        if len > 1 && line_buffer[len - 2] == b'\r' {
                                            len - 2
                                        } else {
                                            len - 1
                                        }
                                    } else {
                                        len
                                    };

                                    if let Some((row, col, value)) =
                                        parse_mtx_line(&line_buffer[..trim_end])
                                    {
                                        let (cell_idx, gene_idx) = if self.cells_as_rows {
                                            ((row - 1) as usize, (col - 1) as usize)
                                        } else {
                                            ((col - 1) as usize, (row - 1) as usize)
                                        };

                                        if genes_to_keep_lookup.get(gene_idx) == Some(&true)
                                            && cell_idx < self.header.total_cells
                                        {
                                            local_cell_stats[cell_idx].0 += 1;
                                            local_cell_stats[cell_idx].1 += value;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if verbose {
                        let completed = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                        if completed.is_multiple_of(report_interval) || completed == num_chunks {
                            let progress = (completed as f64 / num_chunks as f64 * 100.0) as usize;
                            println!(
                                "  Processed {}% of chunks ({}/{})",
                                progress, completed, num_chunks
                            );
                        }
                    }

                    local_cell_stats
                },
            )
            .reduce_with(|mut a, b| {
                for (acc, add) in a.iter_mut().zip(b) {
                    acc.0 += add.0;
                    acc.1 += add.1;
                }
                a
            })
            .unwrap_or_else(|| vec![(0u32, 0u32); self.header.total_cells]);

        let cell_gene_count: Vec<u32> = merged.iter().map(|&(count, _)| count).collect();
        let cell_lib_size: Vec<u32> = merged.iter().map(|&(_, size)| size).collect();

        // Filter cells
        let cells_to_keep: Vec<usize> = (0..self.header.total_cells)
            .filter(|&i| {
                cell_gene_count[i] as usize >= self.qc_params.min_unique_genes
                    && cell_lib_size[i] as f32 >= self.qc_params.min_lib_size as f32
            })
            .collect();

        let mut quality = CellOnFileQuality::new(cells_to_keep, genes_to_keep);
        quality.generate_maps_sets();

        let second_scan_end = second_scan_time.elapsed();

        if verbose {
            println!("Second pass done: {:.2?}", second_scan_end);
            println!(
                "Genes passing QC: {} / {}",
                quality.genes_to_keep.len().separate_with_underscores(),
                self.header.total_genes.separate_with_underscores()
            );
            println!(
                "Cells passing QC: {} / {}",
                quality.cells_to_keep.len().separate_with_underscores(),
                self.header.total_cells.separate_with_underscores()
            );
        }

        Ok(quality)
    }

    /// Process the mtx file and write to binarised Rust file
    ///
    /// Gathers every kept cell in memory, then writes them via
    /// [`write_cell_rows`]. See
    /// [`Self::process_mtx_and_write_bin_streaming`] for the bounded-memory
    /// variant.
    ///
    /// ### Params
    ///
    /// * `bin_path` - Where to save the binarised file.
    /// * `quality` - Structure indicating which cells and genes to keep from
    ///   the mtx file.
    /// * `verbose` - Controls verbosity of the function.
    ///
    /// ### Returns
    ///
    /// The `MtxFinalData` with information how many cells were written to
    /// file, how many genes were included in which cells did not parse
    /// the thresholds.
    pub fn process_mtx_and_write_bin(
        mut self,
        bin_path: &str,
        quality: &CellOnFileQuality,
        verbose: bool,
    ) -> Result<MtxFinalData, BixverseErrors> {
        let start_read = Instant::now();

        if verbose {
            println!(
                "Starting to write cells passing quality thresholds in a cell I/O-friendly format to disk."
            )
        }

        let cell_map = dense_index_map(&quality.cells_to_keep);
        let gene_map = dense_index_map(&quality.genes_to_keep);

        // (gene_index, raw_count), both u32: gene index to support >65k
        // features, count to avoid saturating high-expression genes.
        let mut cell_data: Vec<Vec<(u32, u32)>> = vec![Vec::new(); quality.cells_to_keep.len()];

        let cells_as_rows = self.cells_as_rows;
        for (start, end) in self.find_chunk_boundaries(1)? {
            for_each_mtx_entry(&self.path, start, end, |row, col, value| {
                if let Some((cell, gene)) =
                    map_mtx_entry(row, col, cells_as_rows, &cell_map, &gene_map)
                {
                    cell_data[cell as usize].push((gene, value));
                }
            })?;
        }

        let mut writer = CellGeneSparseWriter::new(
            bin_path,
            true,
            quality.cells_to_keep.len(),
            quality.genes_to_keep.len(),
            self.qc_params.target_size,
        )?;
        let (nnz, lib_size) =
            write_cell_rows(&mut cell_data, 0, self.qc_params.target_size, &mut writer)?;
        writer.finalise()?;

        if verbose {
            println!("Reading in cell data done: {:.2?}", start_read.elapsed());
        }

        Ok(MtxFinalData {
            cell_qc: CellQuality {
                cell_indices: quality.cells_to_keep.to_vec(),
                gene_indices: quality.genes_to_keep.to_vec(),
                lib_size,
                nnz,
            },
            no_genes: quality.genes_to_keep.len(),
            no_cells: quality.cells_to_keep.len(),
        })
    }

    /// Process the mtx file and write to binarised Rust file
    ///
    /// Streams the mtx file in two passes via temp file bucketing, supporting
    /// both `cells_as_rows = true` and `cells_as_rows = false` without requiring
    /// the input to be sorted. Memory usage is bounded by the size of a single
    /// bucket rather than the total kept entries.
    ///
    /// The bucketing pass parses byte ranges of the file in parallel. Each
    /// worker buffers records per bucket and appends them to the shared bucket
    /// file under its lock, so record order inside a bucket is arbitrary; the
    /// writing pass sorts each bucket on the full `(cell, gene, count)` tuple,
    /// which makes the output deterministic. Every kept cell is written, empty
    /// or not.
    ///
    /// ### Params
    ///
    /// * `bin_path` - Where to save the binarised file.
    /// * `quality` - Structure indicating which cells and genes to keep from
    ///   the mtx file.
    /// * `verbose` - Controls verbosity of the function.
    ///
    /// ### Returns
    ///
    /// The `MtxFinalData` with information how many cells were written to
    /// file, how many genes were included and the per-cell library size and
    /// NNZ statistics.
    pub fn process_mtx_and_write_bin_streaming(
        mut self,
        bin_path: &str,
        quality: &CellOnFileQuality,
        verbose: bool,
    ) -> Result<MtxFinalData, BixverseErrors> {
        let n_kept_cells = quality.cells_to_keep.len();
        let n_kept_genes = quality.genes_to_keep.len();
        let target_size = self.qc_params.target_size;

        if n_kept_cells == 0 {
            let writer = CellGeneSparseWriter::new(bin_path, true, 0, n_kept_genes, target_size)?;
            writer.finalise()?;
            return Ok(MtxFinalData {
                cell_qc: CellQuality {
                    cell_indices: vec![],
                    gene_indices: quality.genes_to_keep.to_vec(),
                    lib_size: vec![],
                    nnz: vec![],
                },
                no_genes: n_kept_genes,
                no_cells: 0,
            });
        }

        // Bucket layout. Cap at 128 buckets to stay well below macOS fd limits.
        let n_buckets = n_kept_cells.div_ceil(5_000).clamp(1, 128);
        let cells_per_bucket = n_kept_cells.div_ceil(n_buckets);

        let bin_path_buf = PathBuf::from(bin_path);
        let temp_dir = bin_path_buf
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let stem = bin_path_buf
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "mtx".into());

        let temp_paths: Vec<PathBuf> = (0..n_buckets)
            .map(|i| temp_dir.join(format!(".{}.bucket_{:04}.tmp", stem, i)))
            .collect();
        let _guard = TempFileGuard(temp_paths.clone());

        let pass1 = Instant::now();
        if verbose {
            println!(
                "Bucketing pass: {} buckets, ~{} cells/bucket",
                n_buckets, cells_per_bucket
            );
        }

        let cell_map = dense_index_map(&quality.cells_to_keep);
        let gene_map = dense_index_map(&quality.genes_to_keep);

        let file_size = self.reader.get_ref().metadata()?.len();
        let num_chunks = ((file_size / MTX_PARSE_CHUNK_BYTES) as usize).max(1);
        let boundaries = self.find_chunk_boundaries(num_chunks)?;

        let bucket_files: Vec<Mutex<BufWriter<File>>> = temp_paths
            .iter()
            .map(|p| File::create(p).map(|f| Mutex::new(BufWriter::with_capacity(256 * 1024, f))))
            .collect::<IoResult<Vec<_>>>()?;

        let (path, cells_as_rows) = (&self.path, self.cells_as_rows);
        boundaries
            .par_iter()
            .try_for_each(|&(start, end)| -> IoResult<()> {
                let flush = |bucket: usize, buf: &mut Vec<u8>| -> IoResult<()> {
                    bucket_files[bucket]
                        .lock()
                        .expect("bucket writer lock is never held across a panic")
                        .write_all(buf)?;
                    buf.clear();
                    Ok(())
                };

                let mut bufs: Vec<Vec<u8>> = vec![Vec::new(); n_buckets];
                let mut flush_err = Ok(());
                for_each_mtx_entry(path, start, end, |row, col, value| {
                    let Some((cell, gene)) =
                        map_mtx_entry(row, col, cells_as_rows, &cell_map, &gene_map)
                    else {
                        return;
                    };
                    let bucket = cell as usize / cells_per_bucket;
                    let buf = &mut bufs[bucket];
                    buf.extend_from_slice(&cell.to_le_bytes());
                    buf.extend_from_slice(&gene.to_le_bytes());
                    buf.extend_from_slice(&value.to_le_bytes());
                    if buf.len() >= BUCKET_FLUSH_BYTES && flush_err.is_ok() {
                        flush_err = flush(bucket, buf);
                    }
                })?;
                flush_err?;

                for (bucket, buf) in bufs.iter_mut().enumerate() {
                    if !buf.is_empty() {
                        flush(bucket, buf)?;
                    }
                }
                Ok(())
            })?;

        for bucket_file in bucket_files {
            bucket_file
                .into_inner()
                .expect("bucket writer lock is never held across a panic")
                .flush()?;
        }

        if verbose {
            println!("Bucketing done: {:.2?}", pass1.elapsed());
        }

        let mut writer =
            CellGeneSparseWriter::new(bin_path, true, n_kept_cells, n_kept_genes, target_size)?;
        let mut lib_size = Vec::with_capacity(n_kept_cells);
        let mut nnz = Vec::with_capacity(n_kept_cells);

        let pass2 = Instant::now();
        if verbose {
            println!("Writing pass: streaming buckets to output");
        }

        for (bucket_idx, temp_path) in temp_paths.iter().enumerate() {
            let bucket_bytes = std::fs::read(temp_path)?;
            let mut entries: Vec<(u32, u32, u32)> = bucket_bytes
                .par_chunks_exact(BUCKET_RECORD_LEN)
                .map(|r| {
                    let word = |k: usize| {
                        u32::from_le_bytes(r[k..k + 4].try_into().expect("4-byte slice"))
                    };
                    (word(0), word(4), word(8))
                })
                .collect();
            drop(bucket_bytes);
            entries.par_sort_unstable();

            let first_cell = bucket_idx * cells_per_bucket;
            let end_cell = (first_cell + cells_per_bucket).min(n_kept_cells);
            let run_starts: Vec<usize> = (first_cell..=end_cell)
                .map(|cell| entries.partition_point(|e| (e.0 as usize) < cell))
                .collect();

            let built = (first_cell..end_cell)
                .into_par_iter()
                .map(|cell| {
                    let k = cell - first_cell;
                    let mut row: Vec<(u32, u32)> = entries[run_starts[k]..run_starts[k + 1]]
                        .iter()
                        .map(|&(_, g, v)| (g, v))
                        .collect();
                    Ok(compress_cell_row(&mut row, cell, target_size)?)
                })
                .collect::<Result<Vec<_>, BixverseErrors>>()?;

            let mut payloads = Vec::with_capacity(built.len());
            for (nnz_i, payload) in built {
                nnz.push(nnz_i);
                lib_size.push(payload.library_size);
                payloads.push(payload);
            }
            writer.write_compressed_cell_chunks(&payloads)?;

            let _ = std::fs::remove_file(temp_path);

            if verbose
                && ((bucket_idx + 1) % (n_buckets / 10).max(1) == 0 || bucket_idx + 1 == n_buckets)
            {
                let progress = ((bucket_idx + 1) as f64 / n_buckets as f64 * 100.0) as usize;
                println!(
                    "  Wrote bucket {}/{} ({}%)",
                    bucket_idx + 1,
                    n_buckets,
                    progress
                );
            }
        }

        writer.finalise()?;

        if verbose {
            println!("Writing pass done: {:.2?}", pass2.elapsed());
        }

        Ok(MtxFinalData {
            cell_qc: CellQuality {
                cell_indices: quality.cells_to_keep.to_vec(),
                gene_indices: quality.genes_to_keep.to_vec(),
                lib_size,
                nnz,
            },
            no_genes: n_kept_genes,
            no_cells: n_kept_cells,
        })
    }

    /// Generate chunk boundaries for parallel processing
    ///
    /// ### Params
    ///
    /// * `num_chunks` - Number of desired chunks
    ///
    /// ### Results
    ///
    /// A vector of tuples indicating the chunk boundaries.
    fn find_chunk_boundaries(&mut self, num_chunks: usize) -> IoResult<Vec<(u64, u64)>> {
        let file_size = self.reader.get_ref().metadata()?.len();

        self.reader.rewind()?;
        let mut line = String::new();
        loop {
            line.clear();
            self.reader.read_line(&mut line)?;
            if !line.starts_with('%') {
                break;
            }
        }

        let data_start = self.reader.stream_position()?;

        let data_size = file_size - data_start;
        let chunk_size = data_size / num_chunks as u64;

        let mut boundaries = vec![(data_start, data_start)];

        for i in 1..num_chunks {
            let target_pos = data_start + (chunk_size * i as u64);
            if target_pos >= file_size {
                break;
            }

            self.reader.seek(std::io::SeekFrom::Start(target_pos))?;

            let mut byte = [0u8; 1];
            while self.reader.read(&mut byte)? > 0 {
                if byte[0] == b'\n' {
                    break;
                }
            }

            let boundary = self.reader.stream_position()?;
            boundaries.push((boundary, boundary));
        }

        boundaries.push((file_size, file_size));

        for i in 0..boundaries.len() - 1 {
            boundaries[i].1 = boundaries[i + 1].0;
        }
        boundaries.pop();

        Ok(boundaries)
    }
}

/////////////
// Helpers //
/////////////

/// Parse mtx line from bytes
///
/// ### Params
///
/// * `line` - The file line as bytes
///
/// ### Return
///
/// Returns an Option of a tuple representing
/// `<row_index, col_index, raw_count>`. Counts are kept at full `u32` width;
/// the on-disk chunk format narrows to u16 only when every value fits.
#[inline]
fn parse_mtx_line(line: &[u8]) -> Option<(u32, u32, u32)> {
    let mut i = 0;
    let len = line.len();

    // Parse first number (row)
    let mut row = 0u32;
    while i < len && line[i].is_ascii_digit() {
        row = row * 10 + (line[i] - b'0') as u32;
        i += 1;
    }
    if i == 0 {
        return None;
    }

    // Skip whitespace
    while i < len && (line[i] == b' ' || line[i] == b'\t') {
        i += 1;
    }
    if i >= len {
        return None;
    }

    // Parse second number (col)
    let mut col = 0u32;
    while i < len && line[i].is_ascii_digit() {
        col = col * 10 + (line[i] - b'0') as u32;
        i += 1;
    }

    // Skip whitespace
    while i < len && (line[i] == b' ' || line[i] == b'\t') {
        i += 1;
    }
    if i >= len {
        return None;
    }

    // Parse third number (value)
    let mut val = 0u32;
    while i < len && line[i].is_ascii_digit() {
        val = val * 10 + (line[i] - b'0') as u32;
        i += 1;
    }

    Some((row, col, val))
}

/// Call `f(row, col, value)` for every entry line in a byte range of an mtx
/// file.
///
/// ### Params
///
/// * `path` - Path to the mtx file
/// * `start` - First byte of the range, at the start of a line
/// * `end` - End of the range; the line straddling it is read in full
/// * `f` - Callback per parsed entry, with the 1-based coordinates as stored
///
/// ### Returns
///
/// `Ok(())` once the range is read, an I/O error otherwise.
fn for_each_mtx_entry<F: FnMut(u32, u32, u32)>(
    path: &Path,
    start: u64,
    end: u64,
    mut f: F,
) -> IoResult<()> {
    let mut reader = BufReader::with_capacity(256 * 1024, File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(start))?;

    let mut line = Vec::with_capacity(64);
    let mut pos = start;
    while pos < end {
        line.clear();
        let n = reader.read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        pos += n as u64;

        let mut len = line.len();
        while len > 0 && (line[len - 1] == b'\n' || line[len - 1] == b'\r') {
            len -= 1;
        }
        if let Some((row, col, value)) = parse_mtx_line(&line[..len]) {
            f(row, col, value);
        }
    }
    Ok(())
}

/// Map a 1-based mtx entry to its output cell and gene.
///
/// ### Params
///
/// * `row` - 1-based row as stored
/// * `col` - 1-based column as stored
/// * `cells_as_rows` - Whether rows are cells
/// * `cell_map` - Dense old-to-new cell map, see [`dense_index_map`]
/// * `gene_map` - Dense old-to-new gene map
///
/// ### Returns
///
/// `Some((cell, gene))` in output indices, `None` if either is dropped or the
/// coordinate is out of range.
#[inline]
fn map_mtx_entry(
    row: u32,
    col: u32,
    cells_as_rows: bool,
    cell_map: &[u32],
    gene_map: &[u32],
) -> Option<(u32, u32)> {
    let (cell, gene) = if cells_as_rows {
        (row, col)
    } else {
        (col, row)
    };
    let cell = *cell_map.get((cell as usize).checked_sub(1)?)?;
    let gene = *gene_map.get((gene as usize).checked_sub(1)?)?;
    (cell != INDEX_DROPPED && gene != INDEX_DROPPED).then_some((cell, gene))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    ///////////////
    // Fixtures //
    ///////////////

    /// Three cells x four genes, cells as rows, five entries. All indices are
    /// base-1 as the Matrix Market spec demands.
    ///
    /// ```text
    ///        g0 g1 g2 g3
    ///   c0 [  5  .  7  . ]
    ///   c1 [  .  3  .  . ]
    ///   c2 [  2  .  .  9 ]
    /// ```
    const CELLS_AS_ROWS_MTX: &str = "%%MatrixMarket matrix coordinate integer general\n\
%\n\
3 4 5\n\
1 1 5\n\
1 3 7\n\
2 2 3\n\
3 1 2\n\
3 4 9\n";

    /// The transpose of [`CELLS_AS_ROWS_MTX`], i.e. genes as rows.
    const GENES_AS_ROWS_MTX: &str = "%%MatrixMarket matrix coordinate integer general\n\
%\n\
4 3 5\n\
1 1 5\n\
3 1 7\n\
2 2 3\n\
1 3 2\n\
4 3 9\n";

    /// The `(gene indices, raw counts)` the two fixtures both encode.
    fn expected_cells() -> Vec<(Vec<u32>, Vec<u32>)> {
        vec![
            (vec![0, 2], vec![5, 7]),
            (vec![1], vec![3]),
            (vec![0, 3], vec![2, 9]),
        ]
    }

    /// RAII guard that removes a test's scratch file even if an assert fails.
    struct TempPath(PathBuf);

    /// Drop implementation for [`TempPath`]. Errors are ignored: the file may
    /// already be gone, and this runs during unwind.
    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempPath {
        /// Reserve a scratch file named after the calling test. The test name
        /// is the uniqueness guarantee: `cargo test` runs the whole module in
        /// one process, so a PID alone would collide across threads.
        fn new(name: &str, ext: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_mtx_io_{name}.{ext}")))
        }

        /// Path of the guarded file as a `&str`.
        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// Write `body` to a scratch `.mtx` file and hand back its guard.
    fn write_mtx(name: &str, body: &str) -> TempPath {
        let temp = TempPath::new(name, "mtx");
        std::fs::write(&temp.0, body).expect("mtx file written");
        temp
    }

    /// QC thresholds that keep every cell and gene of the fixtures.
    fn keep_all_qc() -> MinCellQuality {
        MinCellQuality {
            min_unique_genes: 1,
            min_lib_size: 1,
            min_cells: 1,
            target_size: 1e4,
        }
    }

    /// Read the first `n_cells` cells of a written store back as
    /// `(gene indices, raw counts)`.
    fn read_back(bin_path: &str, n_cells: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
        let reader = ParallelSparseReader::new(bin_path).expect("reader opens");
        let indices: Vec<usize> = (0..n_cells).collect();
        reader
            .read_cells_parallel(&indices)
            .expect("cells read back")
            .into_iter()
            .map(|c| (c.indices.clone(), c.data_raw.iter().collect()))
            .collect()
    }

    /////////////////
    // Line parser //
    /////////////////

    /// Regression: the parser used to saturate counts at `u16::MAX`.
    #[test]
    fn test_parse_mtx_line_keeps_full_u32_counts() {
        assert_eq!(parse_mtx_line(b"1 2 3"), Some((1, 2, 3)));
        assert_eq!(parse_mtx_line(b"7\t9\t65535"), Some((7, 9, 65_535)));
        assert_eq!(parse_mtx_line(b"4 5 70000"), Some((4, 5, 70_000)));
        assert_eq!(parse_mtx_line(b"1 2 4294967295"), Some((1, 2, u32::MAX)));
    }

    /// Empty, non-numeric and short lines give `None`, not a partial triplet.
    #[test]
    fn test_parse_mtx_line_rejects_malformed_input() {
        assert_eq!(parse_mtx_line(b""), None);
        assert_eq!(parse_mtx_line(b"abc"), None);
        assert_eq!(parse_mtx_line(b"1"), None);
        assert_eq!(parse_mtx_line(b"1 2"), None);
    }

    ////////////
    // Header //
    ////////////

    /// The shape line is split by orientation, so both readings of the same
    /// three numbers have to be pinned.
    #[test]
    fn mtx_header_shape_line_is_read_per_orientation() {
        let mtx = write_mtx("header_shape", CELLS_AS_ROWS_MTX);

        let cells_as_rows = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        assert_eq!(cells_as_rows.header.total_cells, 3);
        assert_eq!(cells_as_rows.header.total_genes, 4);
        assert_eq!(cells_as_rows.header.total_entries, 5);

        // Same file read the other way round: the first two fields swap roles.
        let genes_as_rows = MtxReader::new(mtx.path(), keep_all_qc(), false).expect("reader opens");
        assert_eq!(genes_as_rows.header.total_cells, 4);
        assert_eq!(genes_as_rows.header.total_genes, 3);
        assert_eq!(genes_as_rows.header.total_entries, 5);
    }

    /// A shape line that is not three fields must raise `MtxHeaderInvalid`
    /// rather than index past the end of the split.
    #[test]
    fn mtx_header_with_wrong_field_count_is_rejected() {
        let mtx = write_mtx(
            "header_two_fields",
            "%%MatrixMarket matrix coordinate integer general\n3 4\n1 1 5\n",
        );

        assert!(matches!(
            MtxReader::new(mtx.path(), keep_all_qc(), true),
            Err(BixverseErrors::MtxHeaderInvalid(_))
        ));
    }

    /// `MtxParseError` has to name the field that failed, and the name depends
    /// on the orientation because the first two fields swap.
    #[test]
    fn mtx_header_parse_error_names_the_failing_field() {
        let bad_second = write_mtx(
            "header_bad_second",
            "%%MatrixMarket matrix coordinate integer general\n3 x 5\n",
        );
        assert!(matches!(
            MtxReader::new(bad_second.path(), keep_all_qc(), true),
            Err(BixverseErrors::MtxParseError {
                field: "gene count"
            })
        ));
        // Cells are the columns now, so the same broken field is the cell count.
        assert!(matches!(
            MtxReader::new(bad_second.path(), keep_all_qc(), false),
            Err(BixverseErrors::MtxParseError {
                field: "cell count"
            })
        ));

        let bad_third = write_mtx(
            "header_bad_third",
            "%%MatrixMarket matrix coordinate integer general\n3 4 x\n",
        );
        assert!(matches!(
            MtxReader::new(bad_third.path(), keep_all_qc(), true),
            Err(BixverseErrors::MtxParseError {
                field: "entry count"
            })
        ));
    }

    /// A missing file must surface as an I/O error, not a panic on unwrap.
    #[test]
    fn mtx_reader_reports_a_missing_file() {
        let missing = std::env::temp_dir().join("bixverse_mtx_io_does_not_exist.mtx");
        assert!(matches!(
            MtxReader::new(&missing, keep_all_qc(), true),
            Err(BixverseErrors::BinaryIo(_))
        ));
    }

    /////////////////
    // Quality pass //
    /////////////////

    /// `parse_mtx_quality` drives every downstream index remap, so pin the
    /// gene-then-cell filter order and the old-to-new maps it builds.
    #[test]
    fn parse_mtx_quality_filters_genes_then_cells() {
        let mtx = write_mtx("quality_filters", CELLS_AS_ROWS_MTX);

        // Everything permissive: nothing is dropped and the maps are identities.
        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        assert_eq!(quality.cells_to_keep, vec![0, 1, 2]);
        assert_eq!(quality.genes_to_keep, vec![0, 1, 2, 3]);

        // min_cells = 2 keeps only gene 0, which is seen in cells 0 and 2.
        // Cell 1 then has zero kept entries and fails min_unique_genes = 1.
        let qc = MinCellQuality {
            min_unique_genes: 1,
            min_lib_size: 1,
            min_cells: 2,
            target_size: 1e4,
        };
        let mut reader = MtxReader::new(mtx.path(), qc, true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        assert_eq!(quality.genes_to_keep, vec![0]);
        assert_eq!(quality.cells_to_keep, vec![0, 2]);
        assert_eq!(quality.cell_old_to_new[&0], 0);
        assert_eq!(quality.cell_old_to_new[&2], 1);
        assert_eq!(quality.gene_old_to_new[&0], 0);
    }

    /////////////////
    // Round trips //
    /////////////////

    /// The only mtx -> bin -> reader path in the crate. Counts and gene indices
    /// have to survive the conversion untouched.
    #[test]
    fn mtx_round_trip_preserves_counts_and_indices() {
        let mtx = write_mtx("round_trip", CELLS_AS_ROWS_MTX);
        let bin = TempPath::new("round_trip", "bin");

        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        let final_data = reader
            .process_mtx_and_write_bin(bin.path(), &quality, false)
            .expect("conversion");

        assert_eq!(final_data.no_cells, 3);
        assert_eq!(final_data.no_genes, 4);
        // Library sizes: 5 + 7 = 12, 3, 2 + 9 = 11.
        assert_eq!(final_data.cell_qc.lib_size, vec![12, 3, 11]);
        assert_eq!(final_data.cell_qc.nnz, vec![2, 1, 2]);

        assert_eq!(read_back(bin.path(), 3), expected_cells());
    }

    /// Matrix Market is base-1. The lowest legal index has to land on 0 and the
    /// highest on `n - 1`, with nothing shifted in between.
    #[test]
    fn mtx_indices_shift_from_base_one_to_base_zero() {
        let mtx = write_mtx(
            "base_one",
            "%%MatrixMarket matrix coordinate integer general\n2 3 2\n1 1 42\n2 3 7\n",
        );
        let bin = TempPath::new("base_one", "bin");

        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        reader
            .process_mtx_and_write_bin(bin.path(), &quality, false)
            .expect("conversion");

        // Only genes 1 and 3 (base-1) carry counts, so the kept genes are the
        // old 0 and 2, renumbered to 0 and 1.
        assert_eq!(quality.genes_to_keep, vec![0, 2]);
        assert_eq!(
            read_back(bin.path(), 2),
            vec![(vec![0], vec![42]), (vec![1], vec![7])]
        );
    }

    /// The two orientations parse different files but must produce the same
    /// store, which is the only cross-check on the row/column swap.
    #[test]
    fn mtx_gene_as_rows_matches_cell_as_rows() {
        let mtx = write_mtx("orientation", GENES_AS_ROWS_MTX);
        let bin = TempPath::new("orientation", "bin");

        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), false).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        let final_data = reader
            .process_mtx_and_write_bin(bin.path(), &quality, false)
            .expect("conversion");

        assert_eq!(final_data.no_cells, 3);
        assert_eq!(final_data.no_genes, 4);
        assert_eq!(final_data.cell_qc.lib_size, vec![12, 3, 11]);
        assert_eq!(read_back(bin.path(), 3), expected_cells());
    }

    /// The streaming writer buckets to temp files and re-sorts; it must land on
    /// exactly the same store as the in-memory writer.
    #[test]
    fn mtx_streaming_matches_non_streaming() {
        let mtx = write_mtx("streaming", CELLS_AS_ROWS_MTX);
        let bin_plain = TempPath::new("streaming_plain", "bin");
        let bin_stream = TempPath::new("streaming_stream", "bin");

        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        let plain = reader
            .process_mtx_and_write_bin(bin_plain.path(), &quality, false)
            .expect("conversion");

        let reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let streamed = reader
            .process_mtx_and_write_bin_streaming(bin_stream.path(), &quality, false)
            .expect("streaming conversion");

        assert_eq!(streamed.no_cells, plain.no_cells);
        assert_eq!(streamed.no_genes, plain.no_genes);
        assert_eq!(streamed.cell_qc.lib_size, plain.cell_qc.lib_size);
        assert_eq!(streamed.cell_qc.nnz, plain.cell_qc.nnz);
        assert_eq!(
            read_back(bin_stream.path(), 3),
            read_back(bin_plain.path(), 3)
        );
    }

    /// An empty input has no cells to keep; the streaming path short-circuits
    /// there and still has to leave a readable, empty store behind.
    #[test]
    fn mtx_streaming_writes_a_readable_store_when_nothing_is_kept() {
        let bin = TempPath::new("streaming_empty", "bin");
        let mtx = write_mtx("streaming_empty", CELLS_AS_ROWS_MTX);

        let reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let mut quality = CellOnFileQuality::new(vec![], vec![0, 1, 2, 3]);
        quality.generate_maps_sets();

        let final_data = reader
            .process_mtx_and_write_bin_streaming(bin.path(), &quality, false)
            .expect("streaming conversion");

        assert_eq!(final_data.no_cells, 0);
        assert_eq!(final_data.no_genes, 4);
        assert!(final_data.cell_qc.lib_size.is_empty());

        let reader = ParallelSparseReader::new(bin.path()).expect("reader opens");
        assert!(reader.is_cell_based());
        assert_eq!(reader.get_header().total_cells, 0);
    }

    /// A kept cell with no kept entries is still written, as an empty chunk,
    /// so `lib_size` / `nnz` line up with `cell_indices` in both writers.
    #[test]
    fn mtx_kept_cell_without_entries_is_written_empty() {
        for streaming in [false, true] {
            let name = format!("empty_cell_{streaming}");
            let mtx = write_mtx(&name, CELLS_AS_ROWS_MTX);
            let bin = TempPath::new(&name, "bin");

            // Keep every cell but only gene 0, which cell 1 does not express.
            let reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
            let mut quality = CellOnFileQuality::new(vec![0, 1, 2], vec![0]);
            quality.generate_maps_sets();

            let final_data = if streaming {
                reader.process_mtx_and_write_bin_streaming(bin.path(), &quality, false)
            } else {
                reader.process_mtx_and_write_bin(bin.path(), &quality, false)
            }
            .expect("conversion");

            assert_eq!(final_data.cell_qc.cell_indices.len(), 3);
            assert_eq!(final_data.cell_qc.lib_size, vec![5, 0, 2]);
            assert_eq!(final_data.cell_qc.nnz, vec![1, 0, 1]);

            let store = ParallelSparseReader::new(bin.path()).expect("reader opens");
            let cell = store
                .read_cells_parallel(&[1])
                .expect("empty cell is on disk");
            assert!(cell[0].indices.is_empty());
        }
    }
}
