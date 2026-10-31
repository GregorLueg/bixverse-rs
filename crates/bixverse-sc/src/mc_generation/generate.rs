//! End-to-end meta cell generation over a cell-major store: resolve the cells,
//! find the meta cells and aggregate their counts.
//!
//! Two index spaces meet here and conflating them silently aggregates the
//! wrong cells. *Working space* is the row space of the embedding or kNN graph
//! the caller hands over, one row per kept cell. *Original space* is the row
//! space of the store, which [`aggregate_meta_cells`] reads and which every
//! membership returned here is given in. [`CellMapping`] translates between
//! the two.

use faer::{Mat, MatRef};
use rustc_hash::FxHashMap;
use std::time::Instant;

use crate::mc_generation::cell_aggregation_utils::{
    aggregate_meta_cells, remap_metacells_to_original,
};
use crate::mc_generation::hdwgcna_meta_cells::{
    BootstrappedMetaCellParams, assign_bootstrapped_meta_cells, identify_meta_cells,
};
use crate::mc_generation::metacells2::{DacParams, MetacellsParams, compute_dac_metacells};
use crate::mc_generation::seacells::{
    SEACells, SEACellsParams, SeacellsFitResult, assignments_to_metacells,
};
use crate::mc_generation::super_cells::{SuperCellParams, supercell};
use crate::prelude::*;

///////////
// Types //
///////////

/// A kNN graph handed over by the caller: neighbour indices and distances,
/// one row per working cell, self excluded.
pub type KnnRef<'a> = (&'a [Vec<usize>], &'a [Vec<f32>]);

/// Resolved mapping between working space and the store.
#[derive(Clone, Debug)]
pub struct CellMapping {
    /// Working rows to retain, in output order. `None` means every working row
    /// is used and the embedding needs no subsetting.
    pub rows_to_use: Option<Vec<usize>>,
    /// Maps a retained working row onto its row in the store.
    pub subset_to_orig: Vec<usize>,
    /// Number of cells in the store.
    pub n_total: usize,
}

/// Meta cells and their aggregated counts.
#[derive(Clone, Debug)]
pub struct MetaCellsRes {
    /// Store cell indices per meta cell, in the order the method lists them.
    pub metacells: Vec<Vec<usize>>,
    /// Meta cells x genes, CSR: summed raw counts in `data`, the counts
    /// re-normalised to the target size and `log1p`'d in `data_2`.
    pub aggregated: CompressedSparseData2<u32, f32>,
    /// Number of cells in the store.
    pub n_total: usize,
}

/// SEACells meta cells plus the fit diagnostics.
#[derive(Clone, Debug)]
pub struct SeaCellsRes {
    /// The meta cells; empty archetypes are dropped and the rest renumbered.
    pub base: MetaCellsRes,
    /// RSS per iteration.
    pub rss: Vec<f32>,
    /// Store cell index of the archetype of each retained meta cell.
    pub archetypes: Vec<usize>,
}

/// MetaCells2 meta cells plus the per-cell and per-gene annotations.
#[derive(Clone, Debug)]
pub struct Metacells2Res {
    /// The meta cells.
    pub base: MetaCellsRes,
    /// Per included cell: was its candidate dissolved in its last pile.
    pub dissolved: Vec<bool>,
    /// Per included cell: the outlier round that placed it.
    pub level: Vec<i32>,
    /// Per included cell: 0-based rare gene module, `-1` for none.
    pub rare_module_of_cell: Vec<i32>,
    /// Per store gene: 0-based rare gene module, `-1` for none.
    pub rare_module_of_gene: Vec<i32>,
    /// Genes selected in any pile plus the rare module genes, ascending.
    pub selected_genes: Vec<usize>,
}

/////////////
// Helpers //
/////////////

/// Work out how working-space rows map back onto the store.
///
/// Resolves the identity case (no QC filtering), the filtered case, and the
/// narrowed case where only some of the kept cells are requested. Requested
/// cells absent from `cells_to_keep` are dropped silently, since a caller may
/// legitimately ask for cells that failed QC.
///
/// ### Params
///
/// * `cells_to_keep` - Store indices of the working rows, in row order. `None`
///   assumes the working rows are the store one-to-one.
/// * `cells_to_use` - Optional narrowing, as store indices.
/// * `n_working_rows` - Rows of the embedding or kNN graph.
/// * `n_total` - Cells in the store.
///
/// ### Returns
///
/// The [`CellMapping`], or an error when `cells_to_keep` disagrees with
/// `n_working_rows` or no requested cell survives.
pub fn resolve_cell_mapping(
    cells_to_keep: Option<&[usize]>,
    cells_to_use: Option<&[usize]>,
    n_working_rows: usize,
    n_total: usize,
) -> Result<CellMapping, ScErrors> {
    let keep: Vec<usize> = match cells_to_keep {
        Some(keep) if keep.len() != n_working_rows => {
            return Err(ScErrors::MetaCellMappingMismatch {
                n_keep: keep.len(),
                n_rows: n_working_rows,
            });
        }
        Some(keep) => keep.to_vec(),
        None => (0..n_working_rows).collect(),
    };

    let Some(use_cells) = cells_to_use else {
        return Ok(CellMapping {
            rows_to_use: None,
            subset_to_orig: keep,
            n_total,
        });
    };

    let orig_to_working: FxHashMap<usize, usize> = keep
        .iter()
        .enumerate()
        .map(|(row, &orig)| (orig, row))
        .collect();
    let (rows_to_use, subset_to_orig): (Vec<usize>, Vec<usize>) = use_cells
        .iter()
        .filter_map(|&orig| orig_to_working.get(&orig).map(|&row| (row, orig)))
        .unzip();
    if subset_to_orig.is_empty() {
        return Err(ScErrors::MetaCellNoCellsToUse);
    }
    Ok(CellMapping {
        rows_to_use: Some(rows_to_use),
        subset_to_orig,
        n_total,
    })
}

/// Gather rows of an embedding.
///
/// ### Params
///
/// * `embd` - The embedding, one row per working cell
/// * `rows` - Rows to gather, in output order
///
/// ### Returns
///
/// The `rows.len() x embd.ncols()` matrix.
fn subset_rows(embd: MatRef<f32>, rows: &[usize]) -> Mat<f32> {
    Mat::from_fn(rows.len(), embd.ncols(), |i, j| embd[(rows[i], j)])
}

/// Map working-space meta cells to the store and aggregate their counts.
///
/// ### Params
///
/// * `reader` - The cell-major store
/// * `metacells` - Working rows per meta cell
/// * `mapping` - The resolved mapping
/// * `target_size` - Library size to normalise the aggregates to
///
/// ### Returns
///
/// The [`MetaCellsRes`].
fn aggregate(
    reader: &ParallelSparseReader,
    metacells: &[Vec<usize>],
    mapping: &CellMapping,
    target_size: f32,
) -> Result<MetaCellsRes, ScErrors> {
    let refs: Vec<&[usize]> = metacells.iter().map(Vec::as_slice).collect();
    let metacells = remap_metacells_to_original(&refs, &mapping.subset_to_orig);
    let refs: Vec<&[usize]> = metacells.iter().map(Vec::as_slice).collect();
    let n_genes = reader.get_header().total_genes;
    let aggregated = aggregate_meta_cells(reader, &refs, target_size, n_genes)?;
    Ok(MetaCellsRes {
        metacells,
        aggregated,
        n_total: mapping.n_total,
    })
}

/// A kNN graph with distances over an embedding.
///
/// ### Params
///
/// * `embd` - The embedding
/// * `params` - The kNN parameters
/// * `seed` - Seed for the index build
/// * `verbose` - Report the run time
///
/// ### Returns
///
/// Indices and distances, self excluded.
fn knn_with_dist(
    embd: MatRef<f32>,
    params: &KnnParams,
    seed: usize,
    verbose: usize,
) -> ScKnnResults {
    let start = Instant::now();
    let (idx, dist) = generate_knn_with_dist(embd, params, true, false, seed, verbose > 1)?;
    if verbose > 0 {
        println!(
            "kNN generation done in: {:.2?} with {}",
            start.elapsed(),
            params.knn_method
        );
    }
    Ok((idx, dist.expect("distances were asked for")))
}

////////////////
// Generators //
////////////////

/// Bootstrapped (hdWGCNA) meta cells over a cell-major store.
///
/// The kNN graph is the supplied one, or built on `embd`. With `cells_to_use`
/// it is always rebuilt on those cells, which needs `embd`. Meta cells can
/// overlap.
///
/// ### Params
///
/// * `f_path_cell` - Path to the `counts_cells.bin` file
/// * `knn` - Optional kNN graph over the working rows, self excluded
/// * `embd` - Optional embedding, one row per working cell
/// * `cells_to_keep` - Store index of each working row; `None` for one-to-one
/// * `cells_to_use` - Optional narrowing, as store indices
/// * `params` - The meta cell parameters
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [`MetaCellsRes`].
///
/// ### References
///
/// Morabito, et al., Cell Rep. Methods, 2023
#[allow(clippy::too_many_arguments)]
pub fn bootstrapped_metacells_on_store(
    f_path_cell: &str,
    knn: Option<&[Vec<usize>]>,
    embd: Option<MatRef<f32>>,
    cells_to_keep: Option<&[usize]>,
    cells_to_use: Option<&[usize]>,
    params: &BootstrappedMetaCellParams,
    target_size: f32,
    seed: usize,
    verbose: usize,
) -> Result<MetaCellsRes, ScErrors> {
    let reader = ParallelSparseReader::new(f_path_cell)?;
    if cells_to_use.is_some() && embd.is_none() {
        return Err(ScErrors::MetaCellMissingInput {
            what: "'cells_to_use' needs the embedding to rebuild the kNN graph",
        });
    }
    let n_rows = match (knn, embd) {
        (Some(k), _) => k.len(),
        (None, Some(e)) => e.nrows(),
        (None, None) => {
            return Err(ScErrors::MetaCellMissingInput {
                what: "provide a kNN graph or an embedding",
            });
        }
    };
    let mapping = resolve_cell_mapping(
        cells_to_keep,
        cells_to_use,
        n_rows,
        reader.get_header().total_cells,
    )?;

    let rebuilt;
    let knn_graph: &[Vec<usize>] = match (&mapping.rows_to_use, knn, embd) {
        (Some(rows), _, Some(e)) => {
            if verbose > 0 {
                println!(
                    "Subsetting to {} cells (from {} kept cells) and regenerating the kNN graph",
                    rows.len(),
                    n_rows
                );
            }
            let sub = subset_rows(e, rows);
            rebuilt = generate_knn_with_dist(
                sub.as_ref(),
                &params.knn_params,
                false,
                false,
                seed,
                verbose > 1,
            )?
            .0;
            &rebuilt
        }
        (None, Some(k), _) => k,
        (None, None, Some(e)) => {
            rebuilt =
                generate_knn_with_dist(e, &params.knn_params, false, false, seed, verbose > 1)?.0;
            &rebuilt
        }
        _ => unreachable!("inputs checked above"),
    };

    let nn_map = build_nn_map(knn_graph);
    if verbose > 0 {
        println!("Identifying meta cells.");
    }
    let centres = identify_meta_cells(
        &nn_map,
        params.max_shared,
        params.target_no_metacells,
        params.max_iter,
        seed,
        verbose,
    );
    let metacells = assign_bootstrapped_meta_cells(&centres, &nn_map);
    if verbose > 0 {
        println!("Aggregating meta cells.");
    }
    aggregate(&reader, &metacells, &mapping, target_size)
}

/// SEACells over a cell-major store.
///
/// The kernel is built on `embd`; the kNN graph is the supplied one unless
/// `cells_to_use` narrows the cells, in which case it is rebuilt on them.
/// Archetypes that end up with no cells are dropped and the rest renumbered.
///
/// ### Params
///
/// * `f_path_cell` - Path to the `counts_cells.bin` file
/// * `embd` - Embedding, one row per working cell
/// * `knn` - Optional kNN indices and distances over the working rows
/// * `cells_to_keep` - Store index of each working row; `None` for one-to-one
/// * `cells_to_use` - Optional narrowing, as store indices
/// * `params` - The SEACells parameters
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [`SeaCellsRes`].
///
/// ### References
///
/// Persad, et al., Nat. Biotechnol., 2023
#[allow(clippy::too_many_arguments)]
pub fn seacells_on_store(
    f_path_cell: &str,
    embd: MatRef<f32>,
    knn: Option<KnnRef<'_>>,
    cells_to_keep: Option<&[usize]>,
    cells_to_use: Option<&[usize]>,
    params: &SEACellsParams,
    target_size: f32,
    seed: usize,
    verbose: usize,
) -> Result<SeaCellsRes, ScErrors> {
    seacells_on_store_with(
        f_path_cell,
        embd,
        knn,
        cells_to_keep,
        cells_to_use,
        params,
        target_size,
        seed,
        verbose,
        |embd, knn_idx, knn_dist| seacells_fit(embd, knn_idx, knn_dist, params, seed, verbose),
    )
}

/// SEACells over a cell-major store, with the fit supplied by the caller.
///
/// Everything around the fit (cell mapping, kNN, dropping empty archetypes,
/// aggregation) is shared, so a device backend only supplies `fit`. Generic
/// over the backend's error, as the other device seams are.
///
/// ### Params
///
/// * `f_path_cell` - Path to the `counts_cells.bin` file
/// * `embd` - Embedding, one row per working cell
/// * `knn` - Optional kNN indices and distances over the working rows
/// * `cells_to_keep` - Store index of each working row; `None` for one-to-one
/// * `cells_to_use` - Optional narrowing, as store indices
/// * `params` - The SEACells parameters
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed for a kNN rebuild
/// * `verbose` - `0` silent, `1` normal, `2` detailed
/// * `fit` - The fit over the embedding and the kNN actually used, see
///   [`seacells_fit`]
///
/// ### Returns
///
/// The [`SeaCellsRes`].
#[allow(clippy::too_many_arguments)]
pub fn seacells_on_store_with<E, F>(
    f_path_cell: &str,
    embd: MatRef<f32>,
    knn: Option<KnnRef<'_>>,
    cells_to_keep: Option<&[usize]>,
    cells_to_use: Option<&[usize]>,
    params: &SEACellsParams,
    target_size: f32,
    seed: usize,
    verbose: usize,
    fit: F,
) -> Result<SeaCellsRes, E>
where
    E: From<ScErrors>,
    F: FnOnce(MatRef<f32>, &[Vec<usize>], &[Vec<f32>]) -> Result<SeacellsFitResult, E>,
{
    let start = Instant::now();
    let reader = ParallelSparseReader::new(f_path_cell).map_err(ScErrors::from)?;
    let mapping = resolve_cell_mapping(
        cells_to_keep,
        cells_to_use,
        embd.nrows(),
        reader.get_header().total_cells,
    )?;

    let subset;
    let embd_mat = match &mapping.rows_to_use {
        Some(rows) => {
            subset = subset_rows(embd, rows);
            subset.as_ref()
        }
        None => embd,
    };

    let rebuilt;
    let (knn_idx, knn_dist): (&[Vec<usize>], &[Vec<f32>]) = match knn {
        Some((idx, dist)) if mapping.rows_to_use.is_none() => {
            if idx.len() != embd_mat.nrows() {
                return Err(ScErrors::MetaCellKnnEmbeddingMismatch {
                    knn_rows: idx.len(),
                    embd_rows: embd_mat.nrows(),
                }
                .into());
            }
            (idx, dist)
        }
        _ => {
            rebuilt = knn_with_dist(embd_mat, &params.knn_params, seed, verbose)?;
            (&rebuilt.0, &rebuilt.1)
        }
    };

    let (_, groups_raw, archetypes_raw, rss) = fit(embd_mat, knn_idx, knn_dist)?;
    let k = params.n_sea_cells;

    let (groups, kept_archetypes): (Vec<Vec<usize>>, Vec<usize>) = groups_raw
        .into_iter()
        .zip(archetypes_raw)
        .filter(|(group, _)| !group.is_empty())
        .unzip();
    if verbose > 0 && groups.len() < k {
        println!(
            "Dropped {} empty archetype(s); keeping {} of {} requested",
            k - groups.len(),
            groups.len(),
            k
        );
    }
    let archetypes = kept_archetypes
        .iter()
        .map(|&row| mapping.subset_to_orig[row])
        .collect();

    if verbose > 0 {
        println!("Aggregating meta cells.");
    }
    let base = aggregate(&reader, &groups, &mapping, target_size)?;
    if verbose > 0 {
        println!("SEACells found in: {:.2?}", start.elapsed());
    }
    Ok(SeaCellsRes {
        base,
        rss,
        archetypes,
    })
}

/// The SEACells fit on the CPU.
///
/// ### Params
///
/// * `embd` - Embedding, one row per cell
/// * `knn_idx` - kNN indices per cell
/// * `knn_dist` - kNN distances per cell
/// * `params` - The SEACells parameters
/// * `seed` - Seed
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `(hard assignments per cell, metacell groupings, archetype cell indices, RSS
/// history)`, one grouping and archetype per archetype the initialisation
/// selected.
pub fn seacells_fit(
    embd: MatRef<f32>,
    knn_idx: &[Vec<usize>],
    knn_dist: &[Vec<f32>],
    params: &SEACellsParams,
    seed: usize,
    verbose: usize,
) -> Result<SeacellsFitResult, ScErrors> {
    let mut seacell = SEACells::new(embd.nrows(), params);
    seacell.construct_kernel_mat(embd, knn_idx, knn_dist, verbose);
    match params.n_landmarks {
        Some(n_landmarks) => seacell.initialise_archetypes_landmark(
            embd,
            knn_idx,
            knn_dist,
            n_landmarks,
            verbose,
            seed as u64,
        )?,
        None => seacell.initialise_archetypes(knn_idx, knn_dist, verbose, seed as u64)?,
    }
    seacell.fit(seed, verbose)?;

    let assignments = seacell.get_hard_assignments()?;
    let archetypes = seacell.get_archetypes()?;
    let groups = assignments_to_metacells(&assignments, archetypes.len());
    Ok((
        assignments,
        groups,
        archetypes,
        seacell.get_rss_history().to_vec(),
    ))
}

/// SuperCells over a cell-major store.
///
/// Walktrap on the supplied kNN graph, or on one built on `embd`; with
/// `cells_to_use` it is always rebuilt on those cells, which needs `embd`.
/// The number of meta cells is `ceil(n_cells / graining_factor)`.
///
/// ### Params
///
/// * `f_path_cell` - Path to the `counts_cells.bin` file
/// * `knn` - Optional kNN indices and distances over the working rows
/// * `embd` - Optional embedding, one row per working cell
/// * `cells_to_keep` - Store index of each working row; `None` for one-to-one
/// * `cells_to_use` - Optional narrowing, as store indices
/// * `params` - The SuperCell parameters
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed for a kNN build
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [`MetaCellsRes`].
///
/// ### References
///
/// Bilous, et al., BMC Bioinform., 2022
#[allow(clippy::too_many_arguments)]
pub fn supercells_on_store(
    f_path_cell: &str,
    knn: Option<KnnRef<'_>>,
    embd: Option<MatRef<f32>>,
    cells_to_keep: Option<&[usize]>,
    cells_to_use: Option<&[usize]>,
    params: &SuperCellParams,
    target_size: f32,
    seed: usize,
    verbose: usize,
) -> Result<MetaCellsRes, ScErrors> {
    let reader = ParallelSparseReader::new(f_path_cell)?;
    if cells_to_use.is_some() && embd.is_none() {
        return Err(ScErrors::MetaCellMissingInput {
            what: "'cells_to_use' needs the embedding to rebuild the kNN graph",
        });
    }
    let n_rows = match (embd, knn) {
        (Some(e), _) => e.nrows(),
        (None, Some((idx, _))) => idx.len(),
        (None, None) => {
            return Err(ScErrors::MetaCellMissingInput {
                what: "provide a kNN graph or an embedding",
            });
        }
    };
    let mapping = resolve_cell_mapping(
        cells_to_keep,
        cells_to_use,
        n_rows,
        reader.get_header().total_cells,
    )?;

    let rebuilt;
    let (knn_idx, knn_dist): (&[Vec<usize>], &[Vec<f32>]) = match (&mapping.rows_to_use, knn, embd)
    {
        (Some(rows), _, Some(e)) => {
            if verbose > 0 {
                println!(
                    "Subsetting to {} cells and regenerating the kNN graph",
                    rows.len()
                );
            }
            let sub = subset_rows(e, rows);
            rebuilt = knn_with_dist(sub.as_ref(), &params.knn_params, seed, verbose)?;
            (&rebuilt.0, &rebuilt.1)
        }
        (None, Some((idx, dist)), _) => {
            if let Some(e) = embd
                && idx.len() != e.nrows()
            {
                return Err(ScErrors::MetaCellKnnEmbeddingMismatch {
                    knn_rows: idx.len(),
                    embd_rows: e.nrows(),
                });
            }
            (idx, dist)
        }
        (None, None, Some(e)) => {
            rebuilt = knn_with_dist(e, &params.knn_params, seed, verbose)?;
            (&rebuilt.0, &rebuilt.1)
        }
        _ => unreachable!("inputs checked above"),
    };

    let n_meta_cells = (knn_idx.len() as f64 / params.graining_factor).ceil() as usize;
    if verbose > 0 {
        println!("Running SuperCell with Walktrap");
    }
    let membership = supercell(knn_idx, knn_dist, params, n_meta_cells, verbose)?;
    let metacells = assignments_to_metacells(&membership, n_meta_cells);
    if verbose > 0 {
        println!("Aggregating meta cells.");
    }
    aggregate(&reader, &metacells, &mapping, target_size)
}

/// MetaCells2 (divide and conquer) over a cell-major store.
///
/// Works on the raw counts directly, so no embedding or kNN graph is needed.
/// Cells left as outliers belong to no meta cell.
///
/// ### Params
///
/// * `f_path_cell` - Path to the `counts_cells.bin` file
/// * `cell_indices` - Store indices of the cells to include
/// * `params` - The per-pile parameters
/// * `dac` - The divide and conquer parameters
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The [`Metacells2Res`], per-cell annotations aligned to `cell_indices`.
///
/// ### References
///
/// Ben-Kiki, et al., Genome Biol., 2022
pub fn metacells2_on_store(
    f_path_cell: &str,
    cell_indices: &[usize],
    params: &MetacellsParams,
    dac: &DacParams,
    target_size: f32,
    seed: usize,
    verbose: bool,
) -> Result<Metacells2Res, ScErrors> {
    let reader = ParallelSparseReader::new(f_path_cell)?;
    let header = reader.get_header();
    let res = compute_dac_metacells(&reader, cell_indices, None, params, dac, seed, verbose)?;

    let mut metacells: Vec<Vec<usize>> = vec![Vec::new(); res.n_metacells];
    for (&mc, &cell) in res.metacell_of_cell.iter().zip(cell_indices) {
        if let Ok(mc) = usize::try_from(mc) {
            metacells[mc].push(cell);
        }
    }
    let refs: Vec<&[usize]> = metacells.iter().map(Vec::as_slice).collect();
    let aggregated = aggregate_meta_cells(&reader, &refs, target_size, header.total_genes)?;

    Ok(Metacells2Res {
        base: MetaCellsRes {
            metacells,
            aggregated,
            n_total: header.total_cells,
        },
        dissolved: res.dissolved_of_cell,
        level: res.level_of_cell,
        rare_module_of_cell: res.rare_module_of_cell,
        rare_module_of_gene: res.rare_module_of_gene,
        selected_genes: res.selected_genes,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_cell_mapping_identity_and_narrowed() {
        let m = resolve_cell_mapping(Some(&[2, 5, 7]), None, 3, 10).unwrap();
        assert!(m.rows_to_use.is_none());
        assert_eq!(m.subset_to_orig, vec![2, 5, 7]);

        // 3 is not kept and drops silently; order follows the request
        let m = resolve_cell_mapping(Some(&[2, 5, 7]), Some(&[7, 3, 2]), 3, 10).unwrap();
        assert_eq!(m.rows_to_use, Some(vec![2, 0]));
        assert_eq!(m.subset_to_orig, vec![7, 2]);
    }

    #[test]
    fn test_resolve_cell_mapping_errors() {
        assert!(matches!(
            resolve_cell_mapping(Some(&[1, 2]), None, 3, 10),
            Err(ScErrors::MetaCellMappingMismatch {
                n_keep: 2,
                n_rows: 3
            })
        ));
        assert!(matches!(
            resolve_cell_mapping(Some(&[1, 2]), Some(&[4]), 2, 10),
            Err(ScErrors::MetaCellNoCellsToUse)
        ));
    }
}
