//! Multi-10x loading: scan multiple 10x CellRanger h5 files, apply global gene
//! QC, and write all cells into a single binary file. Mirrors the h5ad and mtx
//! multi-loader designs and reuses the 10x layout helpers from `h5_10x_io`.
//!
//! 10x stores the matrix as `genes x cells` in CSC, with `indptr` over cells
//! and `indices` holding gene rows, so each cell's entries are contiguous and
//! can be processed per-cell exactly like the h5ad CSR path.
//!
//! V3 modality filtering (default "Gene Expression") is applied internally by
//! reading the per-file `feature_type` dataset and masking out non-target
//! features. Callers therefore only need to supply `gene_local_to_universe`
//! based on feature identifiers; non-gene modalities (e.g. Antibody Capture)
//! are dropped automatically.

use hdf5::File;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::*;
use crate::single_cell::sc_data::h5_10x_io::{TenxVersion, validate_feature_types_tenx};

////////////////
// File tasks //
////////////////

/// Per-file task descriptor for multi-10x loading
pub struct TenxFileTask {
    /// Experimental identifier
    pub exp_id: String,
    /// Path to the 10x CellRanger h5 file
    pub h5_path: String,
    /// CellRanger layout version (resolved by the caller)
    pub version: TenxVersion,
    /// Number of cells/spots (columns) in this file
    pub no_cells: usize,
    /// Number of features (rows, incl. non-gene modalities) in this file
    pub no_genes: usize,
    /// file-local feature idx -> universe gene idx; `None` if the feature is
    /// not in the (intersected) universe. V3 modality filtering is applied
    /// internally by the loader; callers may map all features here regardless
    /// of modality.
    pub gene_local_to_universe: Vec<Option<usize>>,
    /// Target modality for V3 files (e.g. "Gene Expression", "Antibody
    /// Capture"). Defaults to "Gene Expression" when `None`. Ignored for V2,
    /// which is single-modality.
    pub feature_type: Option<String>,
}

/// Per-file QC output returned to the caller
pub struct TenxFileQcResult {
    /// Experimental identifier
    pub exp_id: String,
    /// Cells to keep (file-local 0-indexed indices)
    pub cells_to_keep: Vec<usize>,
    /// Library sizes of the kept cells
    pub lib_size: Vec<usize>,
    /// Number of features per kept cell
    pub nnz: Vec<usize>,
}

/// Final result from multi-10x loading
pub struct MultiTenxResult {
    /// Universe indices of the genes that survived the global QC
    pub global_gene_indices: Vec<usize>,
    /// Total cells ingested
    pub total_cells: usize,
    /// Total genes/features ingested
    pub total_genes: usize,
    /// Per file QC information
    pub per_file: Vec<TenxFileQcResult>,
}

/////////////
// Helpers //
/////////////

/// Number of cells read per HDF5 slice while scanning a single file.
const CELL_CHUNK_SIZE: usize = 10_000;

/// Number of kept cells buffered per HDF5 slice while writing a single file.
const CELL_BATCH_SIZE: usize = 1_000;

/// Build the effective file-local -> universe mapping for a single task.
///
/// For V3, reads `matrix/features/feature_type` and zeroes out the mapping for
/// features whose modality does not match `task.feature_type` (default
/// "Gene Expression"). For V2 the caller's mapping is returned unchanged.
///
/// ### Params
///
/// * `task` - The file task descriptor
///
/// ### Returns
///
/// The effective mapping: file-local feature idx -> universe gene idx, with
/// non-target modalities replaced by `None`.
fn build_effective_mapping(task: &TenxFileTask) -> Result<Vec<Option<usize>>, BixverseErrors> {
    match task.version {
        TenxVersion::V2 => Ok(task.gene_local_to_universe.clone()),
        TenxVersion::V3 => {
            let target = task.feature_type.as_deref();
            let valid = validate_feature_types_tenx(&task.h5_path, task.version, target)?;
            let mut mask = vec![false; task.no_genes];
            for i in valid {
                if i < task.no_genes {
                    mask[i] = true;
                }
            }
            Ok(task
                .gene_local_to_universe
                .iter()
                .enumerate()
                .map(|(local, opt)| {
                    if mask.get(local).copied().unwrap_or(false) {
                        *opt
                    } else {
                        None
                    }
                })
                .collect())
        }
    }
}

/// Scan a single 10x file for per-universe-gene NNZ counts.
///
/// Reads `indptr` once, then streams `indices` in cell-sized chunks, mapping
/// each file-local feature index onto the universe via the precomputed
/// effective mapping. Features outside the universe (incl. non-target V3
/// modalities) are skipped.
///
/// ### Params
///
/// * `task` - The file task descriptor
/// * `gene_local_to_universe` - Effective mapping for this file (modality
///   filter already applied)
/// * `universe_size` - Total number of genes in the shared universe
///
/// ### Returns
///
/// Per-universe-gene NNZ counts
fn scan_gene_nnz(
    task: &TenxFileTask,
    gene_local_to_universe: &[Option<usize>],
    universe_size: usize,
) -> Result<Vec<usize>, BixverseErrors> {
    let file = File::open(&task.h5_path)?;
    let indptr: Vec<usize> = file.dataset(task.version.get_indptr())?.read_1d()?.to_vec();
    let indices_ds = file.dataset(task.version.get_indices())?;

    let mut gene_nnz = vec![0usize; universe_size];

    for chunk_start in (0..task.no_cells).step_by(CELL_CHUNK_SIZE) {
        let chunk_end = (chunk_start + CELL_CHUNK_SIZE).min(task.no_cells);
        let data_start = indptr[chunk_start];
        let data_end = indptr[chunk_end];
        if data_start >= data_end {
            continue;
        }

        let chunk_indices: Vec<usize> = indices_ds.read_slice_1d(data_start..data_end)?.to_vec();

        for cell_idx in chunk_start..chunk_end {
            let start = indptr[cell_idx] - data_start;
            let end = indptr[cell_idx + 1] - data_start;
            for local_idx in start..end {
                let local_gene = chunk_indices[local_idx];
                if let Some(&Some(u_idx)) = gene_local_to_universe.get(local_gene) {
                    gene_nnz[u_idx] += 1;
                }
            }
        }
    }

    Ok(gene_nnz)
}

/// Scan a single 10x file for per-cell NNZ and library size over the final gene
/// set.
///
/// Reads `indptr` once, then streams `data` and `indices` in cell-sized chunks,
/// counting only features present in `gene_local_to_final`.
///
/// ### Params
///
/// * `task` - The file task descriptor
/// * `gene_local_to_final` - Mapping from file-local feature index to final
///   gene index (composed with the modality filter and global gene QC)
///
/// ### Returns
///
/// Per-cell `(nnz, lib_size)` pairs indexed by file-local cell index
fn scan_cell_stats(
    task: &TenxFileTask,
    gene_local_to_final: &[Option<usize>],
) -> Result<Vec<(usize, f32)>, BixverseErrors> {
    let file = File::open(&task.h5_path)?;
    let indptr: Vec<usize> = file.dataset(task.version.get_indptr())?.read_1d()?.to_vec();
    let data_ds = file.dataset(task.version.get_data())?;
    let indices_ds = file.dataset(task.version.get_indices())?;

    let mut cell_stats = Vec::with_capacity(task.no_cells);

    for chunk_start in (0..task.no_cells).step_by(CELL_CHUNK_SIZE) {
        let chunk_end = (chunk_start + CELL_CHUNK_SIZE).min(task.no_cells);
        let data_start = indptr[chunk_start];
        let data_end = indptr[chunk_end];
        if data_start >= data_end {
            cell_stats.extend((chunk_start..chunk_end).map(|_| (0usize, 0.0f32)));
            continue;
        }

        let chunk_data: Vec<f32> = data_ds.read_slice_1d(data_start..data_end)?.to_vec();
        let chunk_indices: Vec<usize> = indices_ds.read_slice_1d(data_start..data_end)?.to_vec();

        for cell_idx in chunk_start..chunk_end {
            let start = indptr[cell_idx] - data_start;
            let end = indptr[cell_idx + 1] - data_start;

            let mut unique = 0usize;
            let mut lib_size = 0.0f32;
            for local_idx in start..end {
                if let Some(&Some(_)) = gene_local_to_final.get(chunk_indices[local_idx]) {
                    unique += 1;
                    lib_size += chunk_data[local_idx];
                }
            }
            cell_stats.push((unique, lib_size));
        }
    }

    Ok(cell_stats)
}

/// Write the kept cells of a single 10x file into the unified sparse writer.
///
/// Reads kept cells in batches, remaps their feature indices to the final gene
/// set, sorts each cell's entries by gene index, and flushes them via
/// `CsrCellChunk`. Cells are processed in `cells_to_keep` order, so the global
/// position is simply `cell_offset + written`.
///
/// ### Params
///
/// * `task` - The file task descriptor
/// * `cells_to_keep` - File-local 0-indexed cell indices to include
/// * `gene_local_to_final` - Mapping from file-local feature index to final
///   gene index (composed with the modality filter and global gene QC)
/// * `target_size` - Target library size for normalisation
/// * `cell_offset` - Global cell offset for this file's cells in the unified
///   output
/// * `writer` - Mutable reference to the unified sparse writer
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Per-file QC statistics for the written cells
fn write_tenx_file_cells(
    task: &TenxFileTask,
    cells_to_keep: &[usize],
    gene_local_to_final: &[Option<usize>],
    target_size: f32,
    cell_offset: usize,
    writer: &mut CellGeneSparseWriter,
    verbose: bool,
) -> Result<TenxFileQcResult, BixverseErrors> {
    if verbose {
        println!(
            "  Writing {} ({} cells)...",
            task.exp_id,
            cells_to_keep.len().separate_with_underscores()
        );
    }

    let file = File::open(&task.h5_path)?;
    let data_ds = file.dataset(task.version.get_data())?;
    let indices_ds = file.dataset(task.version.get_indices())?;
    let indptr: Vec<usize> = file.dataset(task.version.get_indptr())?.read_1d()?.to_vec();

    let mut lib_size = Vec::with_capacity(cells_to_keep.len());
    let mut nnz = Vec::with_capacity(cells_to_keep.len());

    // (final_gene_index, raw_count)
    let mut cell_buf: Vec<(usize, u32)> = Vec::with_capacity(10_000);
    let mut gene_idx_buf: Vec<u32> = Vec::with_capacity(10_000);
    let mut count_buf: Vec<u32> = Vec::with_capacity(10_000);

    let mut written = 0usize;

    for cell_batch in cells_to_keep.chunks(CELL_BATCH_SIZE) {
        let start_pos = cell_batch.iter().map(|&c| indptr[c]).min().unwrap_or(0);
        let end_pos = cell_batch.iter().map(|&c| indptr[c + 1]).max().unwrap_or(0);

        if start_pos >= end_pos {
            for _ in cell_batch {
                let empty = CsrCellChunk::from_data(
                    &[] as &[u32],
                    &[] as &[u32],
                    cell_offset + written,
                    target_size,
                    true,
                );
                lib_size.push(0);
                nnz.push(0);
                writer.write_cell_chunk(empty)?;
                written += 1;
            }
            continue;
        }

        let chunk_data: Vec<f32> = data_ds.read_slice_1d(start_pos..end_pos)?.to_vec();
        let chunk_indices: Vec<usize> = indices_ds.read_slice_1d(start_pos..end_pos)?.to_vec();

        for &old_cell in cell_batch {
            let start = indptr[old_cell] - start_pos;
            let end = indptr[old_cell + 1] - start_pos;

            cell_buf.clear();
            gene_idx_buf.clear();
            count_buf.clear();

            for local_idx in start..end {
                let old_gene = chunk_indices[local_idx];
                if let Some(&Some(final_gene)) = gene_local_to_final.get(old_gene) {
                    cell_buf.push((final_gene, chunk_data[local_idx] as u32));
                }
            }

            if !cell_buf.is_empty() {
                if cell_buf.windows(2).any(|w| w[0].0 > w[1].0) {
                    cell_buf.sort_unstable_by_key(|&(g, _)| g);
                }
                gene_idx_buf.extend(cell_buf.iter().map(|(g, _)| *g as u32));
                count_buf.extend(cell_buf.iter().map(|(_, c)| *c));
            }

            let chunk = CsrCellChunk::from_data(
                &count_buf,
                &gene_idx_buf,
                cell_offset + written,
                target_size,
                true,
            );

            let (nnz_i, lib_i) = chunk.get_qc_info();
            nnz.push(nnz_i);
            lib_size.push(lib_i);
            writer.write_cell_chunk(chunk)?;
            written += 1;
        }
    }

    Ok(TenxFileQcResult {
        exp_id: task.exp_id.clone(),
        cells_to_keep: cells_to_keep.to_vec(),
        lib_size,
        nnz,
    })
}

//////////
// Main //
//////////

/// Load multiple 10x CellRanger h5 files into a single binary.
///
/// 1. Resolve per-file V3 modality filters into effective local - >universe
///    maps
/// 2. Parallel per-file scan of gene NNZ against the intersected universe
/// 3. Apply global `min_cells` to determine the final gene set
/// 4. Parallel per-file scan of cell stats against the final gene set
/// 5. Apply per-cell `min_unique_genes` / `min_lib_size`
/// 6. Stream kept cells into the unified binary
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
pub fn multi_10x_h5_to_file<P: AsRef<Path>>(
    tasks: &[TenxFileTask],
    bin_path: P,
    universe_size: usize,
    cell_qc: &MinCellQuality,
    verbose: bool,
) -> Result<MultiTenxResult, BixverseErrors> {
    let total_start = Instant::now();

    if verbose {
        println!(
            "Resolving modality filters across {} 10x files...",
            tasks.len()
        );
    }

    let effective_mappings: Vec<Vec<Option<usize>>> = tasks
        .par_iter()
        .map(build_effective_mapping)
        .collect::<Result<Vec<_>, _>>()?;

    if verbose {
        for (task, mapping) in tasks.iter().zip(effective_mappings.iter()) {
            let kept = mapping.iter().filter(|o| o.is_some()).count();
            let in_universe = task
                .gene_local_to_universe
                .iter()
                .filter(|o| o.is_some())
                .count();
            println!(
                "  {}: {} / {} features kept after modality filter",
                task.exp_id,
                kept.separate_with_underscores(),
                in_universe.separate_with_underscores()
            );
        }
        println!("Scan 1/2: gene NNZ across {} 10x files...", tasks.len());
    }

    let completed = Arc::new(AtomicUsize::new(0));
    let report_interval = (tasks.len() / 10).max(1);

    let per_file_nnz: Vec<Vec<usize>> = tasks
        .par_iter()
        .zip(effective_mappings.par_iter())
        .map(|(task, mapping)| {
            let res = scan_gene_nnz(task, mapping, universe_size);
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

    let composed: Vec<Vec<Option<usize>>> = effective_mappings
        .iter()
        .map(|m| {
            m.iter()
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
        let res = write_tenx_file_cells(
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
        println!("Multi-10x loading complete: {:.2?}", total_start.elapsed());
    }

    Ok(MultiTenxResult {
        global_gene_indices,
        total_cells,
        total_genes,
        per_file: per_file_results,
    })
}
