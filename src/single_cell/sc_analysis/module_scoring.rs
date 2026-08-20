//! The module scoring approach used in Seurat, originally used in Tirosh,
//! et. al., Science, 2016

use indexmap::IndexSet;
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;

use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Structure holding gene binning information
///
/// ### Fields
///
/// * `gene_to_bin` - HashMap that contains gene to bin mapping
/// * `bins` - Bin to gene lookup
struct GeneBins {
    gene_to_bin: FxHashMap<usize, usize>,
    bins: Vec<Vec<usize>>,
}

/// Helper function to get the average gene expression across the cells
///
/// ### Params
///
/// * `gene_reader` - Reader for the gene-based store.
/// * `cell_set` - IndexSet that stores which cells to include in the analysis.
/// * `streaming` - Boolean. If set to TRUE, the chunks will be loaded in groups
///   of 500 gene.
///
/// ### Returns
///
/// A vector of `(gene_index, avg expression)`
fn get_average_expression<S: SingleCellReading>(
    gene_reader: &S,
    cell_set: &IndexSet<u32>,
    streaming: bool,
) -> Result<Vec<(usize, f32)>, BixverseErrors> {
    let total_genes = gene_reader.get_header().total_genes;

    if streaming {
        const CHUNK_SIZE: usize = 500;
        let gene_indices: Vec<usize> = (0..total_genes).collect();
        let mut results: Vec<(usize, f32)> = Vec::with_capacity(total_genes);
        for chunk in gene_indices.chunks(CHUNK_SIZE) {
            let gene_chunks = gene_reader.read_gene_parallel(chunk)?;
            let chunk_results: Vec<(usize, f32)> = gene_chunks
                .par_iter()
                .map(|gene| gene.calculate_avg_exp(cell_set))
                .collect();
            results.extend(chunk_results);
        }
        Ok(results)
    } else {
        let gene_chunks = gene_reader.get_all_genes()?;
        let results: Vec<(usize, f32)> = gene_chunks
            .par_iter()
            .map(|gene| gene.calculate_avg_exp(cell_set))
            .collect();
        Ok(results)
    }
}

/// Create expression bins for genes
///
/// Bins genes into equal-sized groups based on average expression.
/// Follows Seurat's approach using quantile-based binning.
///
/// ### Params
///
/// * `gene_means` - Slice of tuples (gene_index, average_expression)
/// * `nbin` - Number of bins to create
///
/// ### Returns
///
/// `GeneBins` structure with gene->bin mapping and bin->genes lookup
fn create_expression_bins(gene_means: &[(usize, f32)], nbin: usize, seed: &usize) -> GeneBins {
    let mut sorted_genes = gene_means.to_vec();

    // add tiny random noise to break ties
    // seurat does this
    let mut rng = StdRng::seed_from_u64(*seed as u64);
    for (_, exp) in sorted_genes.iter_mut() {
        *exp += rng.random::<f32>() / 1e30;
    }

    // Sort by expression
    sorted_genes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Create equal-sized bins
    let total_genes = sorted_genes.len();
    let genes_per_bin = (total_genes as f32 / nbin as f32).ceil() as usize;

    let mut gene_to_bin = FxHashMap::default();
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); nbin];

    for (i, (gene_idx, _)) in sorted_genes.iter().enumerate() {
        let bin_id = (i / genes_per_bin).min(nbin - 1);
        gene_to_bin.insert(*gene_idx, bin_id);
        bins[bin_id].push(*gene_idx);
    }

    GeneBins { gene_to_bin, bins }
}

/// Sample control genes for a gene set
///
/// For each gene in the set, samples `ctrl` genes from the same expression bin.
///
/// ### Params
///
/// * `gene_set` - Slice of gene indices in the set
/// * `gene_bins` - Gene binning structure
/// * `ctrl` - Number of control genes per feature
/// * `rng` - Random number generator
///
/// ### Returns
///
/// Vec of unique control gene indices
fn sample_control_genes(
    gene_set: &[usize],
    gene_bins: &GeneBins,
    ctrl: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let mut controls = IndexSet::new();

    for &gene_idx in gene_set {
        if let Some(&bin_id) = gene_bins.gene_to_bin.get(&gene_idx) {
            let bin_genes = &gene_bins.bins[bin_id];
            let sampled = bin_genes.choose_multiple(rng, ctrl.min(bin_genes.len()));
            controls.extend(sampled.copied());
        }
    }

    controls.into_iter().collect()
}

/// Check that every gene index addresses a column of the dense scratch row.
///
/// The scoring kernel indexes the scratch directly, so an index past the gene
/// axis panics inside a rayon worker where the old binary search returned
/// `0.0`. Both the caller's gene sets and the control genes sampled from the
/// expression bins go through here.
///
/// ### Params
///
/// * `gene_sets` - The index sets to validate
/// * `n_genes` - Number of genes, i.e. the exclusive bound
///
/// ### Returns
///
/// `Ok(())`, or [`BixverseErrors::SliceIndexOutOfBounds`] for the first
/// offending index.
fn validate_gene_indices(gene_sets: &[Vec<usize>], n_genes: usize) -> Result<(), BixverseErrors> {
    for gene_set in gene_sets {
        for &gene in gene_set {
            if gene >= n_genes {
                return Err(BixverseErrors::SliceIndexOutOfBounds {
                    index: gene,
                    len: n_genes,
                });
            }
        }
    }

    Ok(())
}

/// Mean of a dense expression row over a set of gene indices.
///
/// ### Params
///
/// * `dense` - Dense expression row, length `n_genes`
/// * `genes` - Gene indices to average over
///
/// ### Returns
///
/// The mean, or `0.0` for an empty set.
#[inline]
fn mean_over_genes(dense: &[f32], genes: &[usize]) -> f32 {
    if genes.is_empty() {
        return 0.0;
    }

    genes.iter().map(|&idx| dense[idx]).sum::<f32>() / genes.len() as f32
}

/// Calculate module scores for a single cell, one per gene set
///
/// `scratch` is a dense `n_genes` buffer owned by the calling thread. The cell
/// is scattered into it, every gene set and its control set gather by direct
/// index, and only the touched slots are cleared again. That replaces a binary
/// search per gene lookup, which each gene set paid separately: with `m` sets
/// the old cost was `m * (set + ctrl) * log(nnz)` against `nnz` plus the total
/// gathered size now. Control sets are typically `ctrl` times larger than the
/// sets themselves, so the gap widens with the number of modules.
///
/// ### Params
///
/// * `cell` - Reference to a CsrCellChunk
/// * `gene_sets` - Indices of the genes in each module
/// * `control_sets` - Indices of the control genes for each module, in the same
///   order as `gene_sets`
/// * `scratch` - Thread-local dense row, zeroed on entry and on exit
///
/// ### Returns
///
/// One score per module, defined as `mean(genes_of_interest) - mean(controls)`.
fn calculate_cell_module_scores(
    cell: &CsrCellChunk,
    gene_sets: &[Vec<usize>],
    control_sets: &[Vec<usize>],
    scratch: &mut [f32],
) -> Vec<f32> {
    for (&idx, value) in cell.indices.iter().zip(cell.data_norm.iter()) {
        scratch[idx as usize] = value.to_f32();
    }

    let scores: Vec<f32> = gene_sets
        .iter()
        .zip(control_sets.iter())
        .map(|(gene_set, control_set)| {
            mean_over_genes(scratch, gene_set) - mean_over_genes(scratch, control_set)
        })
        .collect();

    for &idx in &cell.indices {
        scratch[idx as usize] = 0.0;
    }

    scores
}

/// Calculate the module scores
///
/// ### Params
///
/// * `cell_reader` - Reader for the cell-based store
/// * `gene_sets` - Slice of indices of the gene sets
/// * `control_sets` - Control genes per gene set, sampled and bounds-checked by
///   [`calculate_module_scores_main`]
/// * `cells_to_keep` - Slice of indices of the cells to keep
/// * `n_genes` - Size of the dense scratch row. Every index in `gene_sets` and
///   `control_sets` must be below it.
///
/// ### Returns
///
/// Vec of vec with outer vector representing the gene sets and the inner ones
/// the cells.
fn calculate_module_scores<S: SingleCellReading>(
    cell_reader: &S,
    gene_sets: &[Vec<usize>],
    control_sets: &[Vec<usize>],
    cells_to_keep: &[usize],
    n_genes: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let cell_chunks = cell_reader.read_cells_parallel(cells_to_keep)?;

    let all_scores: Vec<Vec<f32>> = cell_chunks
        .par_iter()
        .map_init(
            || vec![0.0_f32; n_genes],
            |scratch, cell| calculate_cell_module_scores(cell, gene_sets, control_sets, scratch),
        )
        .collect();

    // Transpose: cells x modules -> modules x cells
    let mut results: Vec<Vec<f32>> = vec![Vec::with_capacity(cell_chunks.len()); gene_sets.len()];
    for cell_scores in all_scores {
        for (module_idx, score) in cell_scores.into_iter().enumerate() {
            results[module_idx].push(score);
        }
    }

    Ok(results)
}

/// Calculate the module scores (in a streaming fashion)
///
/// ### Params
///
/// * `cell_reader` - Reader for the cell-based store
/// * `gene_sets` - Slice of indices of the gene sets
/// * `control_sets` - Control genes per gene set, sampled and bounds-checked by
///   [`calculate_module_scores_main`]
/// * `cells_to_keep` - Slice of indices of the cells to keep
/// * `n_genes` - Size of the dense scratch row. Every index in `gene_sets` and
///   `control_sets` must be below it.
/// * `verbose` - Print per-chunk progress
///
/// ### Returns
///
/// Vec of vec with outer vector representing the gene sets and the inner ones
/// the cells. Identical to [`calculate_module_scores`] for the same inputs;
/// only the read granularity differs.
fn calculate_module_scores_streaming<S: SingleCellReading>(
    cell_reader: &S,
    gene_sets: &[Vec<usize>],
    control_sets: &[Vec<usize>],
    cells_to_keep: &[usize],
    n_genes: usize,
    verbose: bool,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    const CHUNK_SIZE: usize = 50000;

    let total_chunks = cells_to_keep.len().div_ceil(CHUNK_SIZE);
    let mut results: Vec<Vec<f32>> = vec![Vec::with_capacity(cells_to_keep.len()); gene_sets.len()];

    for (chunk_idx, cell_indices_chunk) in cells_to_keep.chunks(CHUNK_SIZE).enumerate() {
        let start = Instant::now();

        let cell_chunks = cell_reader.read_cells_parallel(cell_indices_chunk)?;

        // Calculate scores in parallel (cells x modules)
        let chunk_scores: Vec<Vec<f32>> = cell_chunks
            .par_iter()
            .map_init(
                || vec![0.0_f32; n_genes],
                |scratch, cell| {
                    calculate_cell_module_scores(cell, gene_sets, control_sets, scratch)
                },
            )
            .collect();

        // Transpose and append: cells x modules -> modules x cells
        for cell_scores in chunk_scores {
            for (module_idx, score) in cell_scores.into_iter().enumerate() {
                results[module_idx].push(score);
            }
        }

        if verbose {
            let elapsed = start.elapsed();
            let pct = ((chunk_idx + 1) as f32 / total_chunks as f32) * 100.0;
            println!(
                "Chunk {} of {} (took {:.2?}, {:.1}% complete)",
                chunk_idx + 1,
                total_chunks,
                elapsed,
                pct
            );
        }
    }

    Ok(results)
}

/// Calculate the module scores
///
/// ### Params
///
/// * `gene_reader` - Reader for the gene-based store.
/// * `cell_reader` - Reader for the cell-based store.
/// * `gene_sets` - Slice of indices of the gene sets. Every index must address
///   a gene of the store.
/// * `cells_to_use` - Slice of indices of the cells to use.
/// * `nbin` - Number of bins to use
/// * `ctrl` - Number of control genes to use.
/// * `streaming` - Shall streaming be used. Useful for larger data sets.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
///   detailed verbosity.
///
/// ### Returns
///
/// Vec of vec with outer vector representing the gene sets and the inner ones
/// the cells.
#[allow(clippy::too_many_arguments)]
pub fn calculate_module_scores_main<S: SingleCellReading>(
    gene_reader: &S,
    cell_reader: &S,
    gene_sets: &[Vec<usize>],
    cells_to_use: &[usize],
    nbin: usize,
    ctrl: usize,
    streaming: bool,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);

    let n_genes = cell_reader.get_header().total_genes;
    let n_genes_gene_store = gene_reader.get_header().total_genes;
    if n_genes != n_genes_gene_store {
        return Err(BixverseErrors::GeneAxisMismatch {
            gene_store: n_genes_gene_store,
            cell_store: n_genes,
        });
    }

    // the scoring kernel indexes a dense row directly, so a gene past the axis
    // has to error here rather than panic per cell
    validate_gene_indices(gene_sets, n_genes)?;

    let cell_set: IndexSet<u32> = cells_to_use.iter().map(|&x| x as u32).collect();

    let start_total = Instant::now();

    if verbosity.normal_verbosity() {
        println!("Calculating the average expression across the cells.")
    }

    let start_avg_exp = Instant::now();

    let avg_exp = get_average_expression(gene_reader, &cell_set, streaming)?;

    let end_evg_exp = start_avg_exp.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Finished the calculation of the avg gene expression in {:.2?}",
            end_evg_exp
        );
        println!("Calculating the module scores now.")
    }

    let start_modules = Instant::now();

    let gene_bins = create_expression_bins(&avg_exp, nbin, &seed);

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let control_sets: Vec<Vec<usize>> = gene_sets
        .iter()
        .map(|gene_set| sample_control_genes(gene_set, &gene_bins, ctrl, &mut rng))
        .collect();

    validate_gene_indices(&control_sets, n_genes)?;

    let module_scores = if streaming {
        calculate_module_scores_streaming(
            cell_reader,
            gene_sets,
            &control_sets,
            cells_to_use,
            n_genes,
            verbosity.detailed_verbosity(),
        )
    } else {
        calculate_module_scores(cell_reader, gene_sets, &control_sets, cells_to_use, n_genes)
    }?;

    let end_modules = start_modules.elapsed();
    let end_total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Finished the calculation of the modules in {:.2?}",
            end_modules
        );
        println!("Total runtime: {:.2?}", end_total)
    }

    Ok(module_scores)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a cell chunk from a dense row, keeping only the non-zeros.
    fn chunk_from_dense(row: &[u32], original_index: usize) -> CsrCellChunk {
        let idx: Vec<u32> = (0..row.len() as u32)
            .filter(|&g| row[g as usize] > 0)
            .collect();
        let data: Vec<u32> = idx.iter().map(|&g| row[g as usize]).collect();

        CsrCellChunk::from_data(&data, &idx, original_index, 1e4, true)
    }

    /// Densify a chunk the slow, obvious way.
    fn densify(cell: &CsrCellChunk, n_genes: usize) -> Vec<f32> {
        let mut dense = vec![0.0_f32; n_genes];
        for (&idx, value) in cell.indices.iter().zip(cell.data_norm.iter()) {
            dense[idx as usize] = value.to_f32();
        }
        dense
    }

    /// Reference score: the definition, straight off a dense row.
    fn reference_score(dense: &[f32], gene_set: &[usize], control_set: &[usize]) -> f32 {
        let mean = |genes: &[usize]| -> f32 {
            if genes.is_empty() {
                return 0.0;
            }
            genes.iter().map(|&g| dense[g]).sum::<f32>() / genes.len() as f32
        };

        mean(gene_set) - mean(control_set)
    }

    fn toy_cells() -> Vec<CsrCellChunk> {
        vec![
            chunk_from_dense(&[5, 0, 3, 0, 11, 0, 2, 0], 0),
            chunk_from_dense(&[0, 7, 0, 1, 0, 4, 0, 9], 1),
            chunk_from_dense(&[1, 1, 0, 0, 6, 0, 0, 2], 2),
        ]
    }

    /// Both the caller's gene sets and the sampled control sets go through this,
    /// which is the fix for control genes reaching the scratch unchecked.
    #[test]
    fn test_validate_gene_indices_bounds_every_set() {
        let ok = vec![vec![0, 3], vec![7]];
        assert!(validate_gene_indices(&ok, 8).is_ok());

        // the offending index sits in the SECOND set, which is where the
        // control genes land
        let bad = vec![vec![0, 3], vec![8]];
        assert!(matches!(
            validate_gene_indices(&bad, 8),
            Err(BixverseErrors::SliceIndexOutOfBounds { index: 8, len: 8 })
        ));

        assert!(validate_gene_indices(&[], 8).is_ok());
        assert!(validate_gene_indices(&[vec![]], 8).is_ok());
    }

    #[test]
    fn test_module_scores_match_the_dense_reference() {
        let n_genes = 8;
        let cells = toy_cells();

        let gene_sets = vec![vec![0, 2, 4], vec![1, 7]];
        let control_sets = vec![vec![1, 3, 5, 6, 7], vec![0, 2, 4, 6]];

        let mut scratch = vec![0.0_f32; n_genes];
        for cell in &cells {
            let got = calculate_cell_module_scores(cell, &gene_sets, &control_sets, &mut scratch);
            let dense = densify(cell, n_genes);

            assert_eq!(got.len(), gene_sets.len());
            for (module, &score) in got.iter().enumerate() {
                assert_relative_eq!(
                    score,
                    reference_score(&dense, &gene_sets[module], &control_sets[module]),
                    epsilon = 1e-6
                );
            }
        }
    }

    /// The scratch is shared across cells, so a cell must not see the values of
    /// the one scored before it. This is the failure mode the dense buffer
    /// introduces over the binary-search lookup it replaced.
    #[test]
    fn test_module_scores_scratch_is_left_clean() {
        let n_genes = 8;
        let cells = vec![
            chunk_from_dense(&[9, 9, 9, 9, 0, 0, 0, 0], 0),
            chunk_from_dense(&[0, 0, 0, 0, 1, 2, 3, 4], 1),
        ];

        // scores only genes the second cell does not express
        let gene_sets = vec![vec![0, 1, 2, 3]];
        let control_sets = vec![vec![4, 5]];

        let mut scratch = vec![0.0_f32; n_genes];
        let mut scores = Vec::new();
        for cell in &cells {
            scores.push(calculate_cell_module_scores(
                cell,
                &gene_sets,
                &control_sets,
                &mut scratch,
            ));
        }

        assert!(scratch.iter().all(|&v| v == 0.0));

        let dense = densify(&cells[1], n_genes);
        assert_relative_eq!(
            scores[1][0],
            reference_score(&dense, &gene_sets[0], &control_sets[0]),
            epsilon = 1e-6
        );
    }

    /// An empty set contributes a zero mean rather than a NaN.
    #[test]
    fn test_module_scores_handle_empty_sets() {
        let n_genes = 8;
        let cells = toy_cells();

        let gene_sets = vec![vec![], vec![0, 4]];
        let control_sets = vec![vec![1, 3], vec![]];

        let mut scratch = vec![0.0_f32; n_genes];
        let got = calculate_cell_module_scores(&cells[0], &gene_sets, &control_sets, &mut scratch);
        let dense = densify(&cells[0], n_genes);

        assert_relative_eq!(
            got[0],
            -reference_score(&dense, &control_sets[0], &[]),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            got[1],
            reference_score(&dense, &gene_sets[1], &[]),
            epsilon = 1e-6
        );
    }
}
