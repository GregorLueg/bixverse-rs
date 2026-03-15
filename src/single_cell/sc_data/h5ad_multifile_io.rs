//! Multi-h5ad loading: scan multiple h5ad files, apply global gene QC,
//! and write all cells into a single binary file.

use hdf5::File;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::*;

//////////////////
// H5File tasks //
//////////////////

/// Per-file task descriptor for multi-h5ad loading
pub struct H5adFileTask {
    /// Experimental identifier
    pub exp_id: String,
    /// Path to the h5ad file
    pub h5_path: String,
    /// Type of storage in this specific h5ad file
    pub cs_type: CompressedSparseFormat,
    /// Number of cells/spots in this h5ad file
    pub no_cells: usize,
    /// Number of genes/features in this h5ad file
    pub no_genes: usize,
    /// file-local gene idx -> universe gene idx; None if gene not in universe
    pub gene_local_to_universe: Vec<Option<usize>>,
}

//////////////////
// H5File scans //
//////////////////

//////////////
// Features //
//////////////

/// Scan per given task the number of NNZ for the features
///
/// For CSR-stored h5ad files.
///
/// ### Params
///
/// * `task` - The H5adFileTask
/// * `universe_size` - Total number of features in this universe
///
/// ### Returns
///
/// Number of NNZ per given task/feature
fn scan_gene_nnz_csr(task: &H5adFileTask, universe_size: usize) -> hdf5::Result<Vec<usize>> {
    let file = File::open(&task.h5_path)?;
    let indptr: Vec<u32> = file.dataset("X/indptr")?.read_1d()?.to_vec();
    let indices_ds = file.dataset("X/indices")?;

    let mut gene_nnz = vec![0usize; universe_size];

    const CHUNK_SIZE: usize = 10_000;

    for chunk_start in (0..task.no_cells).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(task.no_cells);
        let data_start = indptr[chunk_start] as usize;
        let data_end = indptr[chunk_end] as usize;

        if data_start >= data_end {
            continue;
        }

        let chunk_indices: Vec<u32> = indices_ds.read_slice_1d(data_start..data_end)?.to_vec();

        for cell_idx in chunk_start..chunk_end {
            let start = indptr[cell_idx] as usize - data_start;
            let end = indptr[cell_idx + 1] as usize - data_start;

            for local_idx in start..end {
                let gene_idx = chunk_indices[local_idx] as usize;
                if let Some(&Some(u_idx)) = task.gene_local_to_universe.get(gene_idx) {
                    gene_nnz[u_idx] += 1;
                }
            }
        }
    }

    Ok(gene_nnz)
}

/// Scan per given task the number of NNZ for the features
///
/// For CSC-stored h5ad files.
///
/// ### Params
///
/// * `task` - The H5adFileTask
/// * `universe_size` - Total number of features in this universe
///
/// ### Returns
///
/// Number of NNZ per given task/feature
fn scan_gene_nnz_csc(task: &H5adFileTask, universe_size: usize) -> hdf5::Result<Vec<usize>> {
    let file = File::open(&task.h5_path)?;
    let indptr: Vec<u32> = file.dataset("X/indptr")?.read_1d()?.to_vec();

    let mut gene_nnz = vec![0usize; universe_size];

    for (local, opt) in task.gene_local_to_universe.iter().enumerate() {
        if let Some(u_idx) = opt {
            gene_nnz[*u_idx] = (indptr[local + 1] - indptr[local]) as usize;
        }
    }

    Ok(gene_nnz)
}

/// Scan per given task the number of NNZ for the features
///
/// ### Params
///
/// * `task` - The H5adFileTask
/// * `universe_size` - Total number of features in this universe
///
/// ### Returns
///
/// Number of NNZ per given task/feature
fn scan_gene_nnz(task: &H5adFileTask, universe_size: usize) -> hdf5::Result<Vec<usize>> {
    match task.cs_type {
        CompressedSparseFormat::Csr => scan_gene_nnz_csr(task, universe_size),
        CompressedSparseFormat::Csc => scan_gene_nnz_csc(task, universe_size),
    }
}

///////////
// Cells //
///////////

/// Scan per given task the cell/spot quality measures
///
/// For CSR-stored h5ad files.
///
/// ### Params
///
/// * `task` - The H5adFileTask
/// * `gene_local_to_final` - Genes to include
///
/// ### Returns
///
/// Vector of (unique_genes, library_size) per cell
fn scan_cell_stats_csr(
    task: &H5adFileTask,
    gene_local_to_final: &[Option<usize>],
) -> hdf5::Result<Vec<(usize, f32)>> {
    let file = File::open(&task.h5_path)?;
    let indptr: Vec<u32> = file.dataset("X/indptr")?.read_1d()?.to_vec();
    let data_ds = file.dataset("X/data")?;
    let indices_ds = file.dataset("X/indices")?;

    let mut cell_stats = Vec::with_capacity(task.no_cells);

    const CHUNK_SIZE: usize = 10_000;

    for chunk_start in (0..task.no_cells).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(task.no_cells);
        let data_start = indptr[chunk_start] as usize;
        let data_end = indptr[chunk_end] as usize;

        if data_start >= data_end {
            cell_stats.extend((chunk_start..chunk_end).map(|_| (0usize, 0.0f32)));
            continue;
        }

        let chunk_data: Vec<f32> = data_ds.read_slice_1d(data_start..data_end)?.to_vec();
        let chunk_indices: Vec<u32> = indices_ds.read_slice_1d(data_start..data_end)?.to_vec();

        for cell_idx in chunk_start..chunk_end {
            let start = indptr[cell_idx] as usize - data_start;
            let end = indptr[cell_idx + 1] as usize - data_start;

            let mut unique = 0usize;
            let mut lib_size = 0.0f32;

            for local_idx in start..end {
                let gene_idx = chunk_indices[local_idx] as usize;
                if let Some(&Some(_)) = gene_local_to_final.get(gene_idx) {
                    unique += 1;
                    lib_size += chunk_data[local_idx];
                }
            }

            cell_stats.push((unique, lib_size));
        }
    }

    Ok(cell_stats)
}

/// Scan per given task the cell/spot quality measures
///
/// For CSC-stored h5ad files.
///
/// ### Params
///
/// * `task` - The H5adFileTask
/// * `gene_local_to_final` - Genes to include
///
/// ### Returns
///
/// Vector of (unique_genes, library_size) per cell
fn scan_cell_stats_csc(
    task: &H5adFileTask,
    gene_local_to_final: &[Option<usize>],
) -> hdf5::Result<Vec<(usize, f32)>> {
    let file = File::open(&task.h5_path)?;
    let indptr: Vec<u32> = file.dataset("X/indptr")?.read_1d()?.to_vec();
    let data_ds = file.dataset("X/data")?;
    let indices_ds = file.dataset("X/indices")?;

    let mut cell_unique = vec![0usize; task.no_cells];
    let mut cell_lib_size = vec![0.0f32; task.no_cells];

    let genes_with_final: Vec<usize> = gene_local_to_final
        .iter()
        .enumerate()
        .filter_map(|(local, opt)| opt.map(|_| local))
        .collect();

    const GENE_CHUNK_SIZE: usize = 5_000;

    for gene_chunk in genes_with_final.chunks(GENE_CHUNK_SIZE) {
        let start_pos = gene_chunk
            .iter()
            .map(|&g| indptr[g] as usize)
            .min()
            .unwrap_or(0);
        let end_pos = gene_chunk
            .iter()
            .map(|&g| indptr[g + 1] as usize)
            .max()
            .unwrap_or(0);

        if start_pos >= end_pos {
            continue;
        }

        let chunk_data: Vec<f32> = data_ds.read_slice_1d(start_pos..end_pos)?.to_vec();
        let chunk_indices: Vec<u32> = indices_ds.read_slice_1d(start_pos..end_pos)?.to_vec();

        for &local_gene in gene_chunk {
            let gene_start = indptr[local_gene] as usize - start_pos;
            let gene_end = indptr[local_gene + 1] as usize - start_pos;

            for idx in gene_start..gene_end {
                let cell_idx = chunk_indices[idx] as usize;
                if cell_idx < task.no_cells {
                    cell_unique[cell_idx] += 1;
                    cell_lib_size[cell_idx] += chunk_data[idx];
                }
            }
        }
    }

    Ok(cell_unique.into_iter().zip(cell_lib_size).collect())
}

/// Scan per given task the cell/spot quality measures
///
/// Dispatches to the respective helper
///
/// ### Params
///
/// * `task` - The H5adFileTask
/// * `gene_local_to_final` - Genes to include
///
/// ### Returns
///
/// Vector of (unique_genes, library_size) per cell
fn scan_cell_stats(
    task: &H5adFileTask,
    gene_local_to_final: &[Option<usize>],
) -> hdf5::Result<Vec<(usize, f32)>> {
    match task.cs_type {
        CompressedSparseFormat::Csr => scan_cell_stats_csr(task, gene_local_to_final),
        CompressedSparseFormat::Csc => scan_cell_stats_csc(task, gene_local_to_final),
    }
}

///////////////////
// H5file writes //
///////////////////

/// Per-file QC output returned to caller
pub struct H5FileQcResult {
    /// Experimental identifier
    pub exp_id: String,
    /// Cells/spots to keep
    pub cells_to_keep: Vec<usize>,
    /// Library sizes of the cells
    pub lib_size: Vec<usize>,
    /// Number of cells/spots expressing the gene
    pub nnz: Vec<usize>,
}

/// Final result from multi-h5ad loading
pub struct MultiH5adResult {
    /// Global gene indices
    pub global_gene_indices: Vec<usize>,
    /// Total cells/spots ingested
    pub total_cells: usize,
    /// Total genes/features/ingested
    pub total_genes: usize,
    /// Per file QC information as a vector
    pub per_file: Vec<H5FileQcResult>,
}

/// Write the cells from an H5adFileTask (in CSR)
///
/// ### Params
///
/// * `task` - The H5adFileTask
/// * `cells_to_keep` - Which cells to keep from this file
/// * `gene_mapping` - Informations on the gene mapping
/// * `target_size` - The target size to normalise to
/// * `cell_offset` - Current offset in the binary file
/// * `writer` - The CellGeneSparseWriter to write to the binary file
///
/// ### Returns
///
/// An H5FileQcResult
fn write_h5_csr_cells(
    task: &H5adFileTask,
    cells_to_keep: &[usize],
    gene_mapping: &[Option<usize>],
    target_size: f32,
    cell_offset: usize,
    writer: &mut CellGeneSparseWriter,
) -> std::io::Result<H5FileQcResult> {
    let file = hdf5::File::open(&task.h5_path)?;
    let data_ds = file.dataset("X/data")?;
    let indices_ds = file.dataset("X/indices")?;
    let indptr: Vec<u32> = file.dataset("X/indptr")?.read_1d()?.to_vec();

    let mut lib_size = Vec::with_capacity(cells_to_keep.len());
    let mut nnz = Vec::with_capacity(cells_to_keep.len());

    // (final_gene_index, raw_count) - gene index as usize, count as u16
    let mut cell_buf: Vec<(usize, u16)> = Vec::with_capacity(10_000);
    let mut gene_idx_buf: Vec<u32> = Vec::with_capacity(10_000);
    let mut count_buf: Vec<u16> = Vec::with_capacity(10_000);

    const BATCH_SIZE: usize = 1_000;
    let mut written = 0usize;

    for cell_batch in cells_to_keep.chunks(BATCH_SIZE) {
        let start_pos = cell_batch
            .iter()
            .map(|&c| indptr[c] as usize)
            .min()
            .unwrap_or(0);
        let end_pos = cell_batch
            .iter()
            .map(|&c| indptr[c + 1] as usize)
            .max()
            .unwrap_or(0);

        if start_pos >= end_pos {
            for _ in cell_batch {
                let empty = CsrCellChunk::from_data(
                    &[] as &[u16],
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
        let chunk_indices: Vec<u32> = indices_ds.read_slice_1d(start_pos..end_pos)?.to_vec();

        for &old_cell in cell_batch {
            let start = indptr[old_cell] as usize - start_pos;
            let end = indptr[old_cell + 1] as usize - start_pos;

            cell_buf.clear();
            gene_idx_buf.clear();
            count_buf.clear();

            for local_idx in start..end {
                let old_gene = chunk_indices[local_idx] as usize;
                if let Some(&Some(final_gene)) = gene_mapping.get(old_gene) {
                    cell_buf.push((final_gene, chunk_data[local_idx] as u16));
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

    Ok(H5FileQcResult {
        exp_id: task.exp_id.clone(),
        cells_to_keep: cells_to_keep.to_vec(),
        lib_size,
        nnz,
    })
}

/// Write the cells from an H5adFileTask (in CSC)
///
/// ### Params
///
/// * `task` - The H5adFileTask
/// * `cells_to_keep` - Which cells to keep from this file
/// * `gene_mapping` - Informations on the gene mapping
/// * `target_size` - The target size to normalise to
/// * `cell_offset` - Current offset in the binary file
/// * `writer` - The CellGeneSparseWriter to write to the binary file
///
/// ### Returns
///
/// An H5FileQcResult
fn write_h5_csc_cells(
    task: &H5adFileTask,
    cells_to_keep: &[usize],
    gene_mapping: &[Option<usize>],
    target_size: f32,
    cell_offset: usize,
    writer: &mut CellGeneSparseWriter,
) -> std::io::Result<H5FileQcResult> {
    let file = hdf5::File::open(&task.h5_path)?;
    let data_ds = file.dataset("X/data")?;
    let indices_ds = file.dataset("X/indices")?;
    let indptr: Vec<u32> = file.dataset("X/indptr")?.read_1d()?.to_vec();

    let cell_old_to_new: FxHashMap<usize, usize> = cells_to_keep
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect();

    // (gene_index, raw_count) - gene index as u32 to support >65k features
    let mut cell_data: Vec<Vec<(u32, u16)>> = vec![Vec::new(); cells_to_keep.len()];

    let genes_with_final: Vec<(usize, usize)> = gene_mapping
        .iter()
        .enumerate()
        .filter_map(|(local, opt)| opt.map(|f| (local, f)))
        .collect();

    const GENE_CHUNK_SIZE: usize = 5_000;

    for gene_chunk in genes_with_final.chunks(GENE_CHUNK_SIZE) {
        let start_pos = gene_chunk
            .iter()
            .map(|&(local, _)| indptr[local] as usize)
            .min()
            .unwrap_or(0);
        let end_pos = gene_chunk
            .iter()
            .map(|&(local, _)| indptr[local + 1] as usize)
            .max()
            .unwrap_or(0);

        if start_pos >= end_pos {
            continue;
        }

        let chunk_data: Vec<f32> = data_ds.read_slice_1d(start_pos..end_pos)?.to_vec();
        let chunk_indices: Vec<u32> = indices_ds.read_slice_1d(start_pos..end_pos)?.to_vec();

        for &(local_gene, final_gene) in gene_chunk {
            let gene_start = indptr[local_gene] as usize - start_pos;
            let gene_end = indptr[local_gene + 1] as usize - start_pos;

            for idx in gene_start..gene_end {
                let old_cell = chunk_indices[idx] as usize;
                if let Some(&new_cell) = cell_old_to_new.get(&old_cell) {
                    cell_data[new_cell].push((final_gene as u32, chunk_data[idx] as u16));
                }
            }
        }
    }

    let mut lib_size = Vec::with_capacity(cells_to_keep.len());
    let mut nnz = Vec::with_capacity(cells_to_keep.len());

    for (i, mut data) in cell_data.into_iter().enumerate() {
        data.sort_by_key(|(g, _)| *g);

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

    Ok(H5FileQcResult {
        exp_id: task.exp_id.clone(),
        cells_to_keep: cells_to_keep.to_vec(),
        lib_size,
        nnz,
    })
}

/// Write the cells from an H5adFileTask (dispatch to respective helper)
///
/// ### Params
///
/// * `task` - The H5adFileTask
/// * `cells_to_keep` - Which cells to keep from this file
/// * `gene_mapping` - Informations on the gene mapping
/// * `target_size` - The target size to normalise to
/// * `cell_offset` - Current offset in the binary file
/// * `writer` - The CellGeneSparseWriter to write to the binary file
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// An H5FileQcResult
fn write_h5_file_cells(
    task: &H5adFileTask,
    cells_to_keep: &[usize],
    gene_mapping: &[Option<usize>],
    target_size: f32,
    cell_offset: usize,
    writer: &mut CellGeneSparseWriter,
    verbose: bool,
) -> std::io::Result<H5FileQcResult> {
    if verbose {
        println!(
            "  Writing {} ({} cells)...",
            task.exp_id,
            cells_to_keep.len().separate_with_underscores()
        );
    }
    match task.cs_type {
        CompressedSparseFormat::Csr => write_h5_csr_cells(
            task,
            cells_to_keep,
            gene_mapping,
            target_size,
            cell_offset,
            writer,
        ),
        CompressedSparseFormat::Csc => write_h5_csc_cells(
            task,
            cells_to_keep,
            gene_mapping,
            target_size,
            cell_offset,
            writer,
        ),
    }
}

//////////
// Main //
//////////

/// Write multiple h5ad files to disk
///
/// ### Params
///
/// * `tasks` - Reference to the H5adFileTasks to ingest.
/// * `bin_path` - Path to the cells.bin file to write all the h5ad files to.
/// * `universe` - Size of the feature universe
/// * `cell_qc` - The minimum quality filters to apply
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// The MultiH5adResult results
pub fn multi_h5ad_to_file<P: AsRef<Path>>(
    tasks: &[H5adFileTask],
    bin_path: P,
    universe_size: usize,
    cell_qc: &MinCellQuality,
    verbose: bool,
) -> MultiH5adResult {
    let total_start = Instant::now();

    if verbose {
        println!(
            "Scan 1/2: Counting gene expression across {} files...",
            tasks.len()
        );
    }

    // first parallel scan to check features
    let per_file_nnz: Vec<Vec<usize>> = tasks
        .par_iter()
        .map(|task| {
            if verbose {
                println!("  Scanning genes in {}...", task.exp_id);
            }
            scan_gene_nnz(task, universe_size).unwrap()
        })
        .collect();

    // aggregate
    let mut global_gene_nnz = vec![0usize; universe_size];
    for file_nnz in &per_file_nnz {
        for (i, &count) in file_nnz.iter().enumerate() {
            global_gene_nnz[i] += count;
        }
    }

    // determine final gene set
    let mut universe_to_final = vec![None; universe_size];
    let mut final_idx = 0usize;
    for (u_idx, &nnz) in global_gene_nnz.iter().enumerate() {
        if nnz >= cell_qc.min_cells {
            universe_to_final[u_idx] = Some(final_idx);
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

    // compose per-file local -> final mappings
    let composed_mappings: Vec<Vec<Option<usize>>> = tasks
        .iter()
        .map(|task| {
            task.gene_local_to_universe
                .iter()
                .map(|opt| opt.and_then(|u| universe_to_final[u]))
                .collect()
        })
        .collect();

    // second parallel scan across cells
    if verbose {
        println!("Scan 2/2: Computing cell QC with final gene set...");
    }

    let per_file_cell_stats: Vec<Vec<(usize, f32)>> = tasks
        .par_iter()
        .zip(composed_mappings.par_iter())
        .map(|(task, mapping)| {
            if verbose {
                println!("  Scanning cells in {}...", task.exp_id);
            }
            scan_cell_stats(task, mapping).unwrap()
        })
        .collect();

    let per_file_cells: Vec<Vec<usize>> = per_file_cell_stats
        .iter()
        .map(|stats| {
            stats
                .iter()
                .enumerate()
                .filter(|(_, (unique, lib))| {
                    *unique >= cell_qc.min_unique_genes && *lib >= cell_qc.min_lib_size as f32
                })
                .map(|(idx, _)| idx)
                .collect()
        })
        .collect();

    let total_cells: usize = per_file_cells.iter().map(|v| v.len()).sum();

    if verbose {
        for (i, task) in tasks.iter().enumerate() {
            println!(
                "  {}: {} / {} cells passing QC",
                task.exp_id,
                per_file_cells[i].len().separate_with_underscores(),
                task.no_cells.separate_with_underscores()
            );
        }
        println!("  Total cells: {}", total_cells.separate_with_underscores());
    }

    // write all of the cells to file
    if verbose {
        println!("Writing cells to binary...");
    }

    let mut writer = CellGeneSparseWriter::new(&bin_path, true, total_cells, total_genes).unwrap();

    let mut cell_offset = 0usize;
    let mut per_file_results = Vec::with_capacity(tasks.len());

    for (file_idx, task) in tasks.iter().enumerate() {
        let qc_result = write_h5_file_cells(
            task,
            &per_file_cells[file_idx],
            &composed_mappings[file_idx],
            cell_qc.target_size,
            cell_offset,
            &mut writer,
            verbose,
        )
        .unwrap();

        cell_offset += per_file_cells[file_idx].len();
        per_file_results.push(qc_result);
    }

    writer.finalise().unwrap();

    let elapsed = total_start.elapsed();
    if verbose {
        println!("Multi-h5ad loading complete: {:.2?}", elapsed);
    }

    MultiH5adResult {
        global_gene_indices,
        total_cells,
        total_genes,
        per_file: per_file_results,
    }
}
