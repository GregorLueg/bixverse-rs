//! Contains the MTX file-related parts to do read in data from mtx files and
//! transform them into the binarised files for usage in bixverse-rs

use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Result as IoResult, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{CellGeneSparseWriter, CellOnFileQuality};

/////////
// MTX //
/////////

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
    /// Optional set of cell indices to admit, 0-based in the mtx file's own
    /// column (or row) order. `None` admits everything, which is the
    /// behaviour every caller had before the allowlist existed.
    cell_allowlist: Option<FxHashSet<usize>>,
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
    ) -> IoResult<Self> {
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
            cell_allowlist: None,
        })
    }

    /// Restrict the reader to a subset of the cells in the file.
    ///
    /// The filter bites in [`MtxReader::parse_mtx_quality`], before the gene
    /// statistics are accumulated, so `min_cells` is computed against the
    /// admitted cells only. Excluded cells never reach the writer either,
    /// since the write passes gate on `quality.cells_to_keep_set`.
    ///
    /// Index space: 0-based positions in the mtx file's own cell axis, which
    /// is the row axis when `cells_as_rows` and the column axis otherwise.
    /// For 10x output that is the order of `barcodes.tsv.gz` and nothing else;
    /// `tissue_positions.csv` is in array raster order and must be joined on
    /// the barcode string first.
    ///
    /// ### Params
    ///
    /// * `allowlist` - Cell indices to admit. An empty list admits nothing.
    ///
    /// ### Returns
    ///
    /// Self, for chaining off [`MtxReader::new`].
    pub fn with_cell_allowlist(mut self, allowlist: impl IntoIterator<Item = usize>) -> Self {
        self.cell_allowlist = Some(allowlist.into_iter().collect());
        self
    }

    /// Parse the header of the mtx file
    ///
    /// ### Returns
    ///
    /// The `MtxHeader`
    fn parse_header(reader: &mut BufReader<File>, cells_as_rows: bool) -> IoResult<MtxHeader> {
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
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid MTX header format",
            ));
        }

        let (total_cells, total_genes) = if cells_as_rows {
            // Header format: cells genes entries
            (
                parts[0].parse().map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid cell count")
                })?,
                parts[1].parse().map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid gene count")
                })?,
            )
        } else {
            // Header format: genes cells entries
            (
                parts[1].parse().map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid cell count")
                })?,
                parts[0].parse().map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid gene count")
                })?,
            )
        };

        let total_entries = parts[2].parse().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid entry count")
        })?;

        Ok(MtxHeader {
            total_cells,
            total_genes,
            total_entries,
        })
    }

    /// Helper to parse the file to understand which cells to keep
    ///
    /// Both passes honour [`MtxReader::with_cell_allowlist`]. That ordering is
    /// the point of the allowlist: the gene statistics in the first pass, and
    /// hence `min_cells`, see only the admitted cells.
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
        // Bound after the last `&mut self` call, so the immutable borrow does
        // not collide with `find_chunk_boundaries`.
        let allowlist = self.cell_allowlist.as_ref();
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
                                    let (cell_idx, gene_idx) = if self.cells_as_rows {
                                        ((row - 1) as usize, (col - 1) as usize)
                                    } else {
                                        ((col - 1) as usize, (row - 1) as usize)
                                    };

                                    let admitted =
                                        allowlist.is_none_or(|set| set.contains(&cell_idx));

                                    if admitted && gene_idx < self.header.total_genes {
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
        let genes_to_keep_set: FxHashSet<usize> = (0..self.header.total_genes)
            .into_par_iter()
            .filter(|&i| {
                let total: u32 = results.iter().map(|local| local[i]).sum();
                total as usize >= self.qc_params.min_cells
            })
            .collect();

        drop(results);

        let first_scan_end = first_scan_time.elapsed();

        if verbose {
            println!("First pass done: {:.2?}", first_scan_end);
            println!("Second pass - cell statistics:");
        }

        let second_scan_time = Instant::now();
        let completed_chunks = Arc::new(AtomicUsize::new(0));

        // Parallel second pass - cell stats with filtered genes
        let results: Vec<_> = boundaries
            .par_iter()
            .map(|&(start, end)| {
                let mut local_cell_stats = vec![(0u32, 0u32); self.header.total_cells];

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

                                    let admitted =
                                        allowlist.is_none_or(|set| set.contains(&cell_idx));

                                    if admitted
                                        && genes_to_keep_set.contains(&gene_idx)
                                        && cell_idx < self.header.total_cells
                                    {
                                        local_cell_stats[cell_idx].0 += 1;
                                        local_cell_stats[cell_idx].1 += value as u32;
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
            })
            .collect();

        // Merge cell results
        let mut cell_gene_count = vec![0u32; self.header.total_cells];
        let mut cell_lib_size = vec![0u32; self.header.total_cells];

        for local_cells in results {
            for (i, (count, size)) in local_cells.into_iter().enumerate() {
                cell_gene_count[i] += count;
                cell_lib_size[i] += size;
            }
        }

        // Filter cells. The allowlist is checked here as well as in the scan
        // above: with `min_unique_genes` and `min_lib_size` both at zero the
        // thresholds admit everything, so a zeroed-out excluded cell would
        // otherwise survive.
        let cells_to_keep: Vec<usize> = (0..self.header.total_cells)
            .filter(|&i| {
                allowlist.is_none_or(|set| set.contains(&i))
                    && cell_gene_count[i] as usize >= self.qc_params.min_unique_genes
                    && cell_lib_size[i] as f32 >= self.qc_params.min_lib_size as f32
            })
            .collect();

        let mut genes_to_keep: Vec<usize> = genes_to_keep_set.into_iter().collect();
        genes_to_keep.sort_unstable();

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
        let mut writer = CellGeneSparseWriter::new(
            bin_path,
            true,
            quality.cells_to_keep.len(),
            quality.genes_to_keep.len(),
        )?;

        // (gene_index, raw_count) - gene index as u32 to support >65k features
        let mut cell_data: Vec<Vec<(u32, u16)>> = vec![Vec::new(); quality.cells_to_keep.len()];
        let mut line_buffer = Vec::with_capacity(64);

        let start_read = Instant::now();

        self.reader.rewind()?;
        Self::skip_header(&mut self.reader)?;

        if verbose {
            println!(
                "Starting to write cells passing quality thresholds in a cell I/O-friendly format to disk."
            )
        }

        let mut lines_read = 0usize;
        let report_interval = (self.header.total_entries / 10).max(1);

        while {
            line_buffer.clear();
            self.reader.read_until(b'\n', &mut line_buffer)? > 0
        } {
            if line_buffer.last() == Some(&b'\n') {
                line_buffer.pop();
            }
            if line_buffer.last() == Some(&b'\r') {
                line_buffer.pop();
            }
            if line_buffer.is_empty() {
                continue;
            }

            let (row, col, value) = match parse_mtx_line(&line_buffer) {
                Some(parsed) => parsed,
                None => continue,
            };

            let (old_cell_idx, old_gene_idx) = if self.cells_as_rows {
                ((row - 1) as usize, (col - 1) as usize)
            } else {
                ((col - 1) as usize, (row - 1) as usize)
            };

            if !quality.genes_to_keep_set.contains(&old_gene_idx)
                || !quality.cells_to_keep_set.contains(&old_cell_idx)
            {
                continue;
            }

            let new_cell_idx = quality.cell_old_to_new[&old_cell_idx];
            let new_gene_idx = quality.gene_old_to_new[&old_gene_idx] as u32;

            cell_data[new_cell_idx].push((new_gene_idx, value));

            lines_read += 1;
            if verbose && lines_read.is_multiple_of(report_interval) {
                let progress =
                    (lines_read as f64 / self.header.total_entries as f64 * 100.0) as usize;
                println!("  Processed {}% of entries", progress);
            }
        }

        let mut lib_size = Vec::with_capacity(quality.cells_to_keep.len());
        let mut nnz = Vec::with_capacity(quality.cells_to_keep.len());

        for (cell_idx, data) in cell_data.iter_mut().enumerate() {
            if data.is_empty() {
                continue;
            }

            data.sort_by_key(|(gene_idx, _)| *gene_idx);

            let gene_indices: Vec<u32> = data.iter().map(|(g, _)| *g).collect();
            let gene_counts: Vec<u16> = data.iter().map(|(_, c)| *c).collect();

            let total_umi: u32 = gene_counts.iter().map(|&x| x as u32).sum();
            let n_genes = gene_counts.len();

            lib_size.push(total_umi as usize);
            nnz.push(n_genes);

            let cell_chunk = CsrCellChunk::from_data(
                &gene_counts,
                &gene_indices,
                cell_idx,
                self.qc_params.target_size,
                true,
            );
            writer.write_cell_chunk(cell_chunk)?;
        }

        writer.finalise()?;

        let end_read = start_read.elapsed();

        if verbose {
            println!("Reading in cell data done: {:.2?}", end_read);
        }

        let cell_quality = CellQuality {
            cell_indices: quality.cells_to_keep.to_vec(),
            gene_indices: quality.genes_to_keep.to_vec(),
            lib_size,
            nnz,
        };

        Ok(MtxFinalData {
            cell_qc: cell_quality,
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

        if n_kept_cells == 0 {
            let writer = CellGeneSparseWriter::new(bin_path, true, 0, n_kept_genes)?;
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

        let mut bucket_writers: Vec<BufWriter<File>> = temp_paths
            .iter()
            .map(|p| File::create(p).map(|f| BufWriter::with_capacity(256 * 1024, f)))
            .collect::<IoResult<Vec<_>>>()?;

        self.reader.rewind()?;
        Self::skip_header(&mut self.reader)?;

        let mut line_buffer = Vec::with_capacity(64);
        let mut lines_read = 0usize;
        let report_interval = (self.header.total_entries / 10).max(1);

        while {
            line_buffer.clear();
            self.reader.read_until(b'\n', &mut line_buffer)? > 0
        } {
            if line_buffer.last() == Some(&b'\n') {
                line_buffer.pop();
            }
            if line_buffer.last() == Some(&b'\r') {
                line_buffer.pop();
            }
            if line_buffer.is_empty() {
                continue;
            }

            let (row, col, value) = match parse_mtx_line(&line_buffer) {
                Some(parsed) => parsed,
                None => continue,
            };

            let (old_cell_idx, old_gene_idx) = if self.cells_as_rows {
                ((row - 1) as usize, (col - 1) as usize)
            } else {
                ((col - 1) as usize, (row - 1) as usize)
            };

            if !quality.genes_to_keep_set.contains(&old_gene_idx)
                || !quality.cells_to_keep_set.contains(&old_cell_idx)
            {
                continue;
            }

            let new_cell_idx = quality.cell_old_to_new[&old_cell_idx] as u32;
            let new_gene_idx = quality.gene_old_to_new[&old_gene_idx] as u32;
            let bucket = (new_cell_idx as usize) / cells_per_bucket;

            let mut buf = [0u8; 10];
            buf[0..4].copy_from_slice(&new_cell_idx.to_le_bytes());
            buf[4..8].copy_from_slice(&new_gene_idx.to_le_bytes());
            buf[8..10].copy_from_slice(&value.to_le_bytes());
            bucket_writers[bucket].write_all(&buf)?;

            lines_read += 1;
            if verbose && lines_read.is_multiple_of(report_interval) {
                let progress =
                    (lines_read as f64 / self.header.total_entries as f64 * 100.0) as usize;
                println!("  Bucketed {}% of entries", progress);
            }
        }

        for w in bucket_writers.iter_mut() {
            w.flush()?;
        }
        drop(bucket_writers);

        if verbose {
            println!("Bucketing done: {:.2?}", pass1.elapsed());
        }

        let mut writer = CellGeneSparseWriter::new(bin_path, true, n_kept_cells, n_kept_genes)?;
        let mut lib_size = Vec::with_capacity(n_kept_cells);
        let mut nnz = Vec::with_capacity(n_kept_cells);

        let pass2 = Instant::now();
        if verbose {
            println!("Writing pass: streaming buckets to output");
        }

        for (bucket_idx, temp_path) in temp_paths.iter().enumerate() {
            let bucket_bytes = std::fs::read(temp_path)?;
            let n_entries = bucket_bytes.len() / 10;

            let mut entries: Vec<(u32, u32, u16)> = Vec::with_capacity(n_entries);
            for i in 0..n_entries {
                let off = i * 10;
                let cell = u32::from_le_bytes([
                    bucket_bytes[off],
                    bucket_bytes[off + 1],
                    bucket_bytes[off + 2],
                    bucket_bytes[off + 3],
                ]);
                let gene = u32::from_le_bytes([
                    bucket_bytes[off + 4],
                    bucket_bytes[off + 5],
                    bucket_bytes[off + 6],
                    bucket_bytes[off + 7],
                ]);
                let value = u16::from_le_bytes([bucket_bytes[off + 8], bucket_bytes[off + 9]]);
                entries.push((cell, gene, value));
            }
            drop(bucket_bytes);

            entries.sort_unstable_by_key(|&(c, g, _)| (c, g));

            let mut i = 0;
            while i < entries.len() {
                let cell = entries[i].0;
                let mut j = i;
                while j < entries.len() && entries[j].0 == cell {
                    j += 1;
                }

                let gene_indices: Vec<u32> = entries[i..j].iter().map(|&(_, g, _)| g).collect();
                let gene_counts: Vec<u16> = entries[i..j].iter().map(|&(_, _, v)| v).collect();
                let total_umi: u32 = gene_counts.iter().map(|&x| x as u32).sum();

                lib_size.push(total_umi as usize);
                nnz.push(gene_counts.len());

                let cell_chunk = CsrCellChunk::from_data(
                    &gene_counts,
                    &gene_indices,
                    cell as usize,
                    self.qc_params.target_size,
                    true,
                );
                writer.write_cell_chunk(cell_chunk)?;

                i = j;
            }

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

        let cell_quality = CellQuality {
            cell_indices: quality.cells_to_keep.to_vec(),
            gene_indices: quality.genes_to_keep.to_vec(),
            lib_size,
            nnz,
        };

        Ok(MtxFinalData {
            cell_qc: cell_quality,
            no_genes: n_kept_genes,
            no_cells: n_kept_cells,
        })
    }

    /// Helper function to skip the header
    fn skip_header(reader: &mut BufReader<File>) -> IoResult<()> {
        let mut line = String::new();
        loop {
            line.clear();
            reader.read_line(&mut line)?;
            if !line.starts_with('%') {
                break;
            }
        }
        Ok(())
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
/// `<row_index, col_index, raw_count>` where the count is u16 (sufficient
/// for single cell data; values exceeding u16::MAX will saturate).
#[inline]
fn parse_mtx_line(line: &[u8]) -> Option<(u32, u32, u16)> {
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

    // Parse third number (value) - parse as u32 then saturate to u16
    let mut val = 0u32;
    while i < len && line[i].is_ascii_digit() {
        val = val * 10 + (line[i] - b'0') as u32;
        i += 1;
    }

    Some((row, col, val.min(u16::MAX as u32) as u16))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a tiny coordinate-format mtx to a temp path.
    ///
    /// Layout is genes-as-rows, so `cells_as_rows` is false, matching 10x.
    ///
    /// ### Params
    ///
    /// * `name` - File stem, to keep parallel tests off each other's files.
    /// * `n_genes` - Row count in the header.
    /// * `n_cells` - Column count in the header.
    /// * `entries` - `(gene, cell, value)` triples, all 1-based.
    ///
    /// ### Returns
    ///
    /// The path written to.
    fn write_mtx(
        name: &str,
        n_genes: usize,
        n_cells: usize,
        entries: &[(usize, usize, u16)],
    ) -> PathBuf {
        let path = std::env::temp_dir().join(format!("bixverse_mtx_test_{name}.mtx"));
        let mut out = String::from("%%MatrixMarket matrix coordinate integer general\n");
        out.push_str(&format!("{n_genes} {n_cells} {}\n", entries.len()));
        for &(g, c, v) in entries {
            out.push_str(&format!("{g} {c} {v}\n"));
        }
        std::fs::write(&path, out).unwrap();
        path
    }

    /// Three genes, four cells. Gene 3 only ever fires in cells 3 and 4, so an
    /// allowlist of cells 1 and 2 must drop it under `min_cells = 2`.
    fn fixture() -> Vec<(usize, usize, u16)> {
        vec![
            (1, 1, 5),
            (1, 2, 5),
            (1, 3, 5),
            (1, 4, 5),
            (2, 1, 3),
            (2, 2, 3),
            (2, 3, 3),
            (2, 4, 3),
            (3, 3, 7),
            (3, 4, 7),
        ]
    }

    fn qc() -> MinCellQuality {
        MinCellQuality {
            min_unique_genes: 1,
            min_lib_size: 1,
            min_cells: 2,
            target_size: 1e4,
        }
    }

    #[test]
    fn test_mtx_without_allowlist_keeps_everything() {
        let path = write_mtx("no_allowlist", 3, 4, &fixture());
        let mut reader = MtxReader::new(&path, qc(), false).unwrap();
        let quality = reader.parse_mtx_quality(false).unwrap();

        assert_eq!(quality.cells_to_keep, vec![0, 1, 2, 3]);
        assert_eq!(quality.genes_to_keep, vec![0, 1, 2]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mtx_allowlist_restricts_cells_and_gene_qc() {
        let path = write_mtx("allowlist", 3, 4, &fixture());
        let mut reader = MtxReader::new(&path, qc(), false)
            .unwrap()
            .with_cell_allowlist([0_usize, 1]);
        let quality = reader.parse_mtx_quality(false).unwrap();

        assert_eq!(quality.cells_to_keep, vec![0, 1]);
        // Gene 3 (index 2) fires only in the excluded cells, so it must not
        // reach `min_cells = 2`. This is the whole point of the allowlist.
        assert_eq!(quality.genes_to_keep, vec![0, 1]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mtx_allowlist_survives_zero_thresholds() {
        let path = write_mtx("allowlist_zero", 3, 4, &fixture());
        let params = MinCellQuality {
            min_unique_genes: 0,
            min_lib_size: 0,
            min_cells: 0,
            target_size: 1e4,
        };
        let mut reader = MtxReader::new(&path, params, false)
            .unwrap()
            .with_cell_allowlist([2_usize]);
        let quality = reader.parse_mtx_quality(false).unwrap();

        assert_eq!(quality.cells_to_keep, vec![2]);

        let _ = std::fs::remove_file(&path);
    }
}
