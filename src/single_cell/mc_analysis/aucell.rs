//! AUCell approach for meta cells. Compared to the single cell version, the
//! data is actually kept in memory

use rayon::prelude::*;
use std::borrow::Cow;
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_analysis::dge_pathway_scores::*;
use crate::single_cell::sc_analysis::fast_ranking::rank_within_rows_f32;

//////////
// Main //
//////////

/// Calculate AUCell on metacell data
///
/// In-memory version of AUCell that operates directly on
/// `CompressedSparseData2`. Reads normalised counts from `data_2`.
///
/// ### Params
///
/// * `matrix` - Sparse matrix in CSR or CSC format, shape (cells, genes).
///   `data_2` must contain normalised counts.
/// * `gene_sets` - Slice of Vecs of gene indices into the matrix columns.
/// * `auc_type` - One of `"auroc"` or `"wilcox"`.
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// AUCell values in form gene set x cells.
pub fn calculate_aucell_metacells<T>(
    matrix: &CompressedSparseData2<T, f32>,
    gene_sets: &[Vec<usize>],
    auc_type: &str,
    verbose: bool,
) -> Vec<Vec<f32>>
where
    T: BixverseNumeric,
{
    let auc_type = parse_auc_type(auc_type).unwrap_or_default();

    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csc => Cow::Owned(matrix.transform()),
    };

    let (n_cells, n_genes) = csr.shape;
    let data_2 = csr
        .data_2
        .as_ref()
        .expect("calculate_aucell_metacells requires normalised counts in data_2");

    let start_ranking = Instant::now();
    let ranks = rank_within_rows_f32(&csr.indptr, &csr.indices, data_2, n_cells, n_genes);
    if verbose {
        println!(
            "Ranked gene expression within metacells: {:.2?}",
            start_ranking.elapsed()
        );
    }

    let start_auc = Instant::now();
    let mut all_results: Vec<Vec<f32>> = vec![Vec::with_capacity(n_cells); gene_sets.len()];

    for cell_ranks in ranks {
        let aucs: Vec<f32> = gene_sets
            .par_iter()
            .map(|gene_set| match auc_type {
                AucType::ClassicalAuc => calculate_auc_for_cell_auroc(&cell_ranks, gene_set),
                AucType::MannWhitney => calculate_auc_per_cell_mw(&cell_ranks, gene_set),
            })
            .collect();

        for (gene_set_idx, auc) in aucs.into_iter().enumerate() {
            all_results[gene_set_idx].push(auc);
        }
    }

    if verbose {
        println!("Calculated AUCs: {:.2?}", start_auc.elapsed());
    }

    all_results
}
