//! Multi-mtx loading: scan multiple mtx files, apply global gene QC,
//! and write all cells into a single binary file. Mirrors the h5ad
//! multi-loader design.

use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::Separable;

use crate::data_io::*;
use crate::mtx_io::{MtxSource, parse_block};
use crate::prelude::*;

////////////////
// File tasks //
////////////////

/// Per-file task descriptor for multi-mtx loading
pub struct MtxFileTask {
    /// Experimental identifier
    pub exp_id: String,
    /// Path to the .mtx file, plain or gzipped
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

/// Open one task's mtx file and read its shape in cell and gene terms.
///
/// ### Params
///
/// * `task` - The file task descriptor
///
/// ### Returns
///
/// `(source, no_cells, no_genes)`.
fn open_task(task: &MtxFileTask) -> Result<(MtxSource, usize, usize), ScDataErrors> {
    let src = MtxSource::open(Path::new(&task.mtx_path), task.cells_as_rows)?;
    let [rows, cols, _] = src.shape;
    let (no_cells, no_genes) = if task.cells_as_rows {
        (rows, cols)
    } else {
        (cols, rows)
    };
    Ok((src, no_cells, no_genes))
}

/// Split a 1-based mtx coordinate into 0-based `(cell, gene)`.
///
/// ### Params
///
/// * `row` - 1-based row as stored
/// * `col` - 1-based column as stored
/// * `cells_as_rows` - Whether rows are cells
///
/// ### Returns
///
/// `None` for a 0 coordinate, which the format does not allow.
#[inline]
fn cell_gene(row: u32, col: u32, cells_as_rows: bool) -> Option<(usize, usize)> {
    let (cell, gene) = if cells_as_rows {
        (row, col)
    } else {
        (col, row)
    };
    Some((
        (cell as usize).checked_sub(1)?,
        (gene as usize).checked_sub(1)?,
    ))
}

/// Count, per universe gene, the number of cells expressing it in one mtx file.
///
/// The blocks of the file are parsed in parallel, one tally per worker.
///
/// ### Params
///
/// * `task` - The mtx file to scan plus its local-to-universe gene mapping
/// * `universe_size` - Number of genes in the intersection universe
///
/// ### Returns
///
/// Per-gene non-zero counts, indexed by universe gene index.
fn scan_gene_nnz(task: &MtxFileTask, universe_size: usize) -> Result<Vec<usize>, ScDataErrors> {
    let (src, _, _) = open_task(task)?;
    let field = src.field;
    let per_worker = src.par_fold_blocks(
        || vec![0u32; universe_size],
        |acc, _, bytes| {
            parse_block(bytes, field, |row, col, _| {
                if let Some((_, gene)) = cell_gene(row, col, task.cells_as_rows)
                    && let Some(&Some(u)) = task.gene_local_to_universe.get(gene)
                {
                    acc[u] += 1;
                }
            })
        },
    )?;

    let mut total = vec![0usize; universe_size];
    for acc in per_worker {
        total
            .iter_mut()
            .zip(acc)
            .for_each(|(t, n)| *t += n as usize);
    }
    Ok(total)
}

/// Scan a single mtx file for per-cell NNZ and library size over the final gene
/// set.
///
/// The blocks of the file are parsed in parallel, ignoring genes not present
/// in `gene_local_to_final`.
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
) -> Result<Vec<(usize, u64)>, ScDataErrors> {
    let (src, no_cells, _) = open_task(task)?;
    let field = src.field;
    let per_worker = src.par_fold_blocks(
        || vec![(0u32, 0u64); no_cells],
        |acc, _, bytes| {
            parse_block(bytes, field, |row, col, value| {
                if let Some((cell, gene)) = cell_gene(row, col, task.cells_as_rows)
                    && cell < no_cells
                    && let Some(&Some(_)) = gene_local_to_final.get(gene)
                {
                    acc[cell].0 += 1;
                    acc[cell].1 += value as u64;
                }
            })
        },
    )?;

    let mut stats = vec![(0usize, 0u64); no_cells];
    for acc in per_worker {
        stats.iter_mut().zip(acc).for_each(|(t, (u, l))| {
            t.0 += u as usize;
            t.1 += l;
        });
    }
    Ok(stats)
}

/// Write the kept cells of a single mtx file into the unified sparse writer.
///
/// Reads the file block by block, filters to kept cells and final genes, and
/// writes each cell's entries sorted by gene via [`write_cell_rows`].
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
) -> Result<MtxFileQcResult, ScDataErrors> {
    if verbose {
        println!(
            "  Writing {} ({} cells)...",
            task.exp_id,
            cells_to_keep.len().separate_with_underscores()
        );
    }

    let (src, _, _) = open_task(task)?;
    let field = src.field;
    let cell_map = dense_index_map(cells_to_keep);
    let gene_map = dense_gene_map_from_options(gene_local_to_final);

    // (gene_final_idx, raw_count) per kept cell
    let mut cell_data: Vec<Vec<(u32, u32)>> = vec![Vec::new(); cells_to_keep.len()];

    src.for_each_block_seq(|_, block| {
        parse_block(&block, field, |row, col, value| {
            let Some((cell, gene)) = cell_gene(row, col, task.cells_as_rows) else {
                return;
            };
            let new_cell = cell_map.get(cell).copied().unwrap_or(INDEX_DROPPED);
            let final_gene = gene_map.get(gene).copied().unwrap_or(INDEX_DROPPED);
            if new_cell != INDEX_DROPPED && final_gene != INDEX_DROPPED {
                cell_data[new_cell as usize].push((final_gene, value));
            }
        })
    })?;

    let (nnz, lib_size) = write_cell_rows(&mut cell_data, cell_offset, target_size, writer)?;

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
) -> Result<MultiMtxResult, ScDataErrors> {
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

    let per_file_cell_stats: Vec<Vec<(usize, u64)>> = tasks
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
                    *u >= cell_qc.min_unique_genes && *l as f32 >= cell_qc.min_lib_size as f32
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
    let mut writer = CellGeneSparseWriter::new(
        &bin_path,
        true,
        total_cells,
        total_genes,
        cell_qc.target_size,
    )?;
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    /// Removes a test's scratch file even if an assert fails.
    struct TempPath(PathBuf);

    /// Drop implementation for [`TempPath`]. Errors are ignored: the file may
    /// already be gone, and this runs during unwind.
    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempPath {
        /// Scratch file named after the calling test.
        fn new(name: &str, ext: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_mtx_multi_{name}.{ext}")))
        }

        /// Path as a `String`.
        fn string(&self) -> String {
            self.0.to_str().expect("utf-8 temp path").to_string()
        }
    }

    /// Three cells x four genes as integer counts, cells as rows.
    const INTEGER_CELLS_AS_ROWS: &str = "%%MatrixMarket matrix coordinate integer general\n\
3 4 5\n1 1 5\n1 3 7\n2 2 3\n3 1 2\n3 4 9\n";

    /// Two cells x four genes as integral reals, genes as rows.
    const REAL_GENES_AS_ROWS: &str = "%%MatrixMarket matrix coordinate real general\n\
4 2 3\n1 1 4.0\n2 1 1.1e+01\n4 2 6\n";

    /// One cell x four genes as a pattern, cells as rows.
    const PATTERN_CELLS_AS_ROWS: &str = "%%MatrixMarket matrix coordinate pattern general\n\
1 4 2\n1 2\n1 4\n";

    /// Task over an identity gene universe of four genes.
    fn task(name: &str, path: &TempPath, cells_as_rows: bool) -> MtxFileTask {
        MtxFileTask {
            exp_id: name.to_string(),
            mtx_path: path.string(),
            gene_local_to_universe: (0..4).map(Some).collect(),
            cells_as_rows,
        }
    }

    /// QC that keeps everything.
    fn keep_all() -> MinCellQuality {
        MinCellQuality {
            min_unique_genes: 1,
            min_lib_size: 1,
            min_cells: 1,
            target_size: 1e4,
        }
    }

    /// Integer, gzipped real and pattern files load into one store with the
    /// values each field type implies.
    #[test]
    fn multi_mtx_reads_every_field_type_and_gzip() {
        let integer = TempPath::new("integer", "mtx");
        std::fs::write(&integer.0, INTEGER_CELLS_AS_ROWS).expect("written");
        let real = TempPath::new("real", "mtx.gz");
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(REAL_GENES_AS_ROWS.as_bytes())
            .expect("gzipped");
        std::fs::write(&real.0, enc.finish().expect("gzipped")).expect("written");
        let pattern = TempPath::new("pattern", "mtx");
        std::fs::write(&pattern.0, PATTERN_CELLS_AS_ROWS).expect("written");
        let bin = TempPath::new("store", "bin");

        let tasks = [
            task("integer", &integer, true),
            task("real", &real, false),
            task("pattern", &pattern, true),
        ];
        let res = multi_mtx_to_file(&tasks, &bin.0, 4, &keep_all(), false).expect("loaded");
        assert_eq!(res.total_cells, 6);
        assert_eq!(res.per_file[1].lib_size, vec![15, 6]);
        assert_eq!(res.per_file[2].lib_size, vec![2]);

        let reader = ParallelSparseReader::new(&bin.string()).expect("reader opens");
        let cells: Vec<(Vec<u32>, Vec<u32>)> = reader
            .read_cells_parallel(&[3, 4, 5])
            .expect("cells read back")
            .into_iter()
            .map(|c| (c.indices.clone(), c.data_raw.iter().collect()))
            .collect();
        assert_eq!(
            cells,
            vec![
                (vec![0, 1], vec![4, 11]),
                (vec![3], vec![6]),
                (vec![1, 3], vec![1, 1]),
            ]
        );
    }

    /// A fractional value fails the load rather than truncating.
    #[test]
    fn multi_mtx_rejects_a_fractional_real() {
        let real = TempPath::new("fraction", "mtx");
        std::fs::write(
            &real.0,
            "%%MatrixMarket matrix coordinate real general\n1 4 1\n1 1 2.5\n",
        )
        .expect("written");
        let bin = TempPath::new("fraction_store", "bin");
        assert!(matches!(
            multi_mtx_to_file(
                &[task("fraction", &real, true)],
                &bin.0,
                4,
                &keep_all(),
                false
            ),
            Err(ScDataErrors::MtxNotACount { .. })
        ));
    }
}
