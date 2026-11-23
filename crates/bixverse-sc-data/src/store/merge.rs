//! Helpers for multi-file binary file merging. This is used to combine multiple
//! SingleCells experiments on the R side. It enables the user to combine and
//! merge several experiments.

use rayon::prelude::*;
use serde::Deserialize;
use std::path::Path;
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::store::{
    chunk::{CsrCellChunk, RawCounts},
    reader::{ParallelSparseReader, SingleCellReading},
    v5::{FeatureKind, NormKind},
    writer::{CellGeneSparseWriter, write_cell_chunks_parallel},
};

////////////
// Consts //
////////////

/// Number of cells to merge in one go per batch
const MERGE_CELL_BATCH_SIZE: usize = 10_000;

////////////////
// Structures //
////////////////

/// A single input source for the merge operation
///
/// Deserialises from the keys of one R `merge_sc_experiments()` merge task.
#[derive(Debug, Clone, Deserialize)]
pub struct BinMergeTask {
    /// Identifier for this experiment, becomes a column in obs.
    pub exp_id: String,
    /// Path to the input cells .bin file.
    pub bin_cells_path: String,
    /// 0-indexed original cell indices to include from this input (typically
    /// the cells with `to_keep = TRUE` in the input's obs).
    pub cells_to_keep: Vec<usize>,
    /// Vector indexed by the input's local gene index. `Some(universe_idx)` if
    /// the gene survives the intersection, `None` if it does not.
    pub gene_local_to_universe: Vec<Option<u32>>,
}

/// Per-file output of the merge
///
/// ### Notes
///
/// `lib_size` semantics depend on `renormalise`:
/// - `renormalise = false`: the original library size from the input bin
///   (i.e. sum over the input's gene set, pre-intersection).
/// - `renormalise = true`: recomputed sum over the surviving (intersected)
///   gene set.
#[derive(Debug, Clone)]
pub struct PerFileMergeResult {
    /// Experiment id string
    pub exp_id: String,
    /// The library size per cell per file
    pub lib_size: Vec<usize>,
    /// The number with non-zero expression genes per cells
    pub nnz: Vec<usize>,
}

/// Total result of merging multiple SingleCells bin files
#[derive(Debug, Clone)]
pub struct MultiSingleCellsResult {
    /// The total number of cells across the experiments
    pub total_cells: usize,
    /// The total number of genes across the experiments
    pub total_genes: usize,
    /// A vectors of [PerFileMergeResult]
    pub per_file: Vec<PerFileMergeResult>,
}

/////////////
// Helpers //
/////////////

/// Remap a single cell onto the universe gene indexing
///
/// Surviving entries are sorted by universe index so the resulting chunk has
/// indices in ascending order, matching the invariant of the rest of the
/// pipeline.
///
/// ### Params
///
/// * `cell` - Reference to the [CsrCellChunk]
/// * `gene_local_to_universe` - The mapping between the local genes to the
///   universe.
/// * `new_cell_idx` - New cell index
/// * `renormalise` - Renormalise the counts.
/// * `target_size` - The target size to normalise to
///
/// ### Returns
///
/// The new chunk plus `(library_size, nnz)` for reporting.
fn remap_cell(
    cell: &CsrCellChunk,
    gene_local_to_universe: &[Option<u32>],
    new_cell_idx: usize,
    renormalise: bool,
    target_size: f32,
) -> (CsrCellChunk, usize, usize) {
    // a store without a derivable norm reads an empty norm layer, and the
    // renormalising path never needs it
    let norm_at = |i: usize| if renormalise { 0.0 } else { cell.data_norm[i] };
    let mut entries: Vec<(u32, u32, f32)> = Vec::with_capacity(cell.indices.len());
    for (i, &local_idx) in cell.indices.iter().enumerate() {
        if let Some(universe_idx) = gene_local_to_universe[local_idx as usize] {
            entries.push((universe_idx, cell.data_raw.get(i), norm_at(i)));
        }
    }
    entries.sort_unstable_by_key(|&(u, _, _)| u);

    let nnz = entries.len();

    if renormalise {
        let raw: Vec<u32> = entries.iter().map(|&(_, r, _)| r).collect();
        let indices: Vec<u32> = entries.iter().map(|&(u, _, _)| u).collect();
        let new_chunk = CsrCellChunk::from_data(&raw, &indices, new_cell_idx, target_size, true);
        let lib_size = new_chunk.library_size;
        (new_chunk, lib_size, nnz)
    } else {
        let raw: Vec<u32> = entries.iter().map(|&(_, r, _)| r).collect();
        let norm: Vec<f32> = entries.iter().map(|&(_, _, n)| n).collect();
        let indices: Vec<u32> = entries.iter().map(|&(u, _, _)| u).collect();
        let new_chunk = CsrCellChunk {
            data_raw: RawCounts::from_u32_auto(&raw),
            data_norm: norm,
            library_size: cell.library_size,
            indices,
            original_index: new_cell_idx,
            to_keep: true,
        };
        (new_chunk, cell.library_size, nnz)
    }
}

//////////
// Main //
//////////

/// Merge multiple existing bin files into a single new cells .bin file
///
/// Streams cells from each input bin, remaps gene indices onto the
/// intersection universe, and writes a unified cell-based output. The
/// gene-based companion file must be generated separately afterwards via
/// `generate_gene_based_data_streaming`.
///
/// ### Params
///
/// * `tasks` - One `BinMergeTask` per input source.
/// * `output_bin_path` - Path to the new cells .bin file to write.
/// * `universe_size` - Number of genes in the intersection universe.
/// * `renormalise` - If `true`, recompute `data_norm` as log-CPM against
///   `target_size` using each cell's surviving raw counts. If `false`, pass
///   `data_norm` through untouched; the inputs must then be log-CPM and agree
///   on the `target_size` recorded in their headers, else
///   [`ScDataErrors::StoreKindMismatch`] or
///   [`ScDataErrors::TargetSizeMismatch`]. Inputs predating that header field
///   report nothing and are skipped by the target check.
/// * `target_size` - Target library size for renormalisation. Ignored when
///   `renormalise = false`.
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// `MultiSingleCellsResult` summarising the merge, or
/// [`ScDataErrors::StoreKindMismatch`] for inputs that differ in feature kind,
/// or that carry more than one raw layer (the merge would drop the others).
/// The output keeps the inputs' feature kind and is log-CPM; a TF-IDF store
/// needs its IDF recomputed on the merged cells.
pub fn merge_sc_bin_files<P: AsRef<Path>>(
    tasks: &[BinMergeTask],
    output_bin_path: P,
    universe_size: usize,
    renormalise: bool,
    target_size: f32,
    verbose: bool,
) -> Result<MultiSingleCellsResult, ScDataErrors> {
    let total_cells: usize = tasks.iter().map(|t| t.cells_to_keep.len()).sum();

    let start = Instant::now();

    if verbose {
        println!(
            "Merging {} input files, {} total cells, {} universe genes (renormalise = {})",
            tasks.len(),
            total_cells.separate_with_underscores(),
            universe_size.separate_with_underscores(),
            renormalise
        );
    }

    let readers = tasks
        .iter()
        .map(|t| ParallelSparseReader::new(&t.bin_cells_path))
        .collect::<Result<Vec<_>, _>>()?;
    let kind = readers
        .first()
        .map_or(FeatureKind::Gene, ParallelSparseReader::feature_kind);
    for r in &readers {
        if r.feature_kind() != kind {
            return Err(ScDataErrors::StoreKindMismatch(format!(
                "feature kind ({} and {})",
                kind.name(),
                r.feature_kind().name()
            )));
        }
        if r.n_layers() > 1 {
            return Err(ScDataErrors::StoreKindMismatch(format!(
                "raw layers: an input carries {}, and the merge keeps only one",
                r.n_layers()
            )));
        }
        if !renormalise && r.norm_kind() != NormKind::LogCpm {
            return Err(ScDataErrors::StoreKindMismatch(
                "normalisation: passing the norm through needs log-CPM inputs; merge with \
                 renormalise = true"
                    .into(),
            ));
        }
    }

    // Pass-through mode inherits the inputs' normalisation, so they have to
    // agree on it. Renormalise mode imposes `target_size` regardless.
    let output_target_size = if renormalise {
        target_size
    } else {
        let mut agreed: Option<f32> = None;
        for r in &readers {
            let Some(found) = r.target_size() else {
                continue;
            };
            match agreed {
                None => agreed = Some(found),
                Some(expected) if expected != found => {
                    return Err(ScDataErrors::TargetSizeMismatch {
                        header: expected,
                        requested: found,
                    });
                }
                Some(_) => {}
            }
        }
        agreed.unwrap_or(0.0)
    };

    let mut writer = CellGeneSparseWriter::new(
        output_bin_path,
        true,
        total_cells,
        universe_size,
        output_target_size,
    )?
    .with_kinds(kind, NormKind::LogCpm)?;

    let mut per_file: Vec<PerFileMergeResult> = Vec::with_capacity(tasks.len());
    let mut new_cell_idx: usize = 0;

    for (task_idx, (task, reader)) in tasks.iter().zip(&readers).enumerate() {
        if verbose {
            println!(
                "  Input {}/{} ({}): {} cells",
                task_idx + 1,
                tasks.len(),
                task.exp_id,
                task.cells_to_keep.len().separate_with_underscores()
            );
        }

        let mut lib_size_out: Vec<usize> = Vec::with_capacity(task.cells_to_keep.len());
        let mut nnz_out: Vec<usize> = Vec::with_capacity(task.cells_to_keep.len());

        for batch in task.cells_to_keep.chunks(MERGE_CELL_BATCH_SIZE) {
            let cells = reader.read_cells_parallel(batch)?;

            let remapped: Vec<(CsrCellChunk, usize, usize)> = cells
                .par_iter()
                .enumerate()
                .map(|(offset, cell)| {
                    let idx = new_cell_idx + offset;
                    remap_cell(
                        cell,
                        &task.gene_local_to_universe,
                        idx,
                        renormalise,
                        target_size,
                    )
                })
                .collect();

            let mut chunks = Vec::with_capacity(remapped.len());
            for (chunk, lib_size_i, nnz_i) in remapped {
                lib_size_out.push(lib_size_i);
                nnz_out.push(nnz_i);
                chunks.push(chunk);
            }
            write_cell_chunks_parallel(&chunks, &mut writer)?;

            new_cell_idx += batch.len();
        }

        per_file.push(PerFileMergeResult {
            exp_id: task.exp_id.clone(),
            lib_size: lib_size_out,
            nnz: nnz_out,
        });

        if verbose {
            println!("   Task done in {:.2?}", start.elapsed());
        }
    }

    writer.finalise()?;

    if verbose {
        println!(
            "Merge complete: {} cells written in {:.2?}",
            total_cells.separate_with_underscores(),
            start.elapsed()
        );
    }

    Ok(MultiSingleCellsResult {
        total_cells,
        total_genes: universe_size,
        per_file,
    })
}

////////////////////////
// Gene store to cell //
////////////////////////

/// Rebuilds the cell-major companion of a gene-major store.
///
/// Memory is bounded by phasing over cells. Each phase holds the entries for
/// one window of cells and re-reads the gene file to fill it, so peak memory is
/// the window rather than the matrix, at the cost of reading the source once
/// per phase. That is the same trade the downstream CSR to CSC conversion
/// makes.
///
/// The output inherits the input's identity (feature kind, norm kind and its
/// scales, IDF, cell-ID hash, metadata), so the normalised layer keeps meaning
/// the same thing on both sides. A store whose norm is not log-CPM is carried
/// as raw counts and its norm derived on read, as in the source.
///
/// ### Params
///
/// * `in_path` - Gene-major source store.
/// * `out_path` - Cell-major store to write.
/// * `cells_per_phase` - Cells held in memory at once. Peak memory is roughly
///   `cells_per_phase * mean_genes_per_cell * 12` bytes.
/// * `gene_batch_size` - Genes read from disk per batch within a phase.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// `()`, or a [`ScDataErrors`] when the source is not gene-major, carries more
/// than one raw layer, or a read or write fails.
pub fn gene_store_to_cell_store<P: AsRef<Path>>(
    in_path: P,
    out_path: P,
    cells_per_phase: usize,
    gene_batch_size: usize,
    verbose: usize,
) -> Result<(), ScDataErrors> {
    let start = Instant::now();

    let reader = ParallelSparseReader::new(in_path.as_ref().to_str().ok_or(
        ScDataErrors::InvalidArgument("input path is not valid UTF-8".to_string()),
    )?)?;

    if !reader.is_gene_based() {
        return Err(ScDataErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }

    let header = reader.dims();
    let (no_cells, no_genes) = (header.total_cells, header.total_genes);
    let cells_per_phase = cells_per_phase.max(1);
    let gene_batch_size = gene_batch_size.max(1);

    if reader.n_layers() > 1 {
        return Err(ScDataErrors::StoreKindMismatch(format!(
            "raw layers: the source carries {}, and the transpose keeps only one",
            reader.n_layers()
        )));
    }
    let mut writer =
        CellGeneSparseWriter::new(out_path, true, no_cells, no_genes, reader.header_scales().0)?;
    writer.inherit_from(&reader)?;
    // only a log-CPM norm can be stored verbatim where it does not recompute;
    // every other kind is derived on read from the raw counts
    let raw_only = reader.norm_kind() != NormKind::LogCpm;

    let n_phases = no_cells.div_ceil(cells_per_phase);

    if verbose >= 1 {
        println!(
            "Transpose: {} genes by {} cells, {} phase(s)",
            no_genes.separate_with_underscores(),
            no_cells.separate_with_underscores(),
            n_phases
        );
    }

    for phase in 0..n_phases {
        let cell_start = phase * cells_per_phase;
        let cell_end = ((phase + 1) * cells_per_phase).min(no_cells);
        let width = cell_end - cell_start;

        // One bucket per cell in the window. Entries accumulate in gene order
        // because the gene batches are walked ascending, so no sort is needed
        // before writing.
        let mut buckets: Vec<Vec<(u32, u32, f32)>> = vec![Vec::new(); width];

        for block_start in (0..no_genes).step_by(gene_batch_size) {
            let block_end = (block_start + gene_batch_size).min(no_genes);
            let block: Vec<usize> = (block_start..block_end).collect();

            let chunks = if raw_only {
                reader.read_gene_parallel_raw(&block)?
            } else {
                reader.read_gene_parallel(&block)?
            };
            for chunk in chunks {
                let gene = chunk.original_index as u32;
                for (slot, &cell) in chunk.indices.iter().enumerate() {
                    let cell = cell as usize;
                    if cell >= cell_start && cell < cell_end {
                        let norm = if raw_only { 0.0 } else { chunk.data_norm[slot] };
                        buckets[cell - cell_start].push((gene, chunk.data_raw.get(slot), norm));
                    }
                }
            }
        }

        for (offset, entries) in buckets.into_iter().enumerate() {
            let library_size: u64 = entries.iter().map(|&(_, raw, _)| raw as u64).sum();
            let raw: Vec<u32> = entries.iter().map(|&(_, r, _)| r).collect();
            if raw_only {
                let genes: Vec<u32> = entries.iter().map(|&(g, _, _)| g).collect();
                writer.write_cell_layers(cell_start + offset, &genes, &[&raw], true)?;
                continue;
            }

            writer.write_cell_chunk(CsrCellChunk {
                data_raw: RawCounts::from_u32_auto(&raw),
                data_norm: entries.iter().map(|&(_, _, n)| n).collect(),
                library_size: library_size as usize,
                indices: entries.iter().map(|&(g, _, _)| g).collect(),
                original_index: cell_start + offset,
                to_keep: true,
            })?;
        }

        if verbose >= 2 {
            println!(
                "Transpose: phase {}/{} done in {:.2?}",
                phase + 1,
                n_phases,
                start.elapsed()
            );
        }
    }

    writer.finalise()?;

    if verbose >= 1 {
        println!(
            "Transpose: wrote the cell-major store in {:.2?}",
            start.elapsed()
        );
    }

    Ok(())
}
