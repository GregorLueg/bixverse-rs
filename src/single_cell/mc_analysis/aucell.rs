//! AUCell approach for meta cells. Compared to the single cell version, the
//! data is actually kept in memory

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
/// * `params` - Optional [AucellParams]. Defaults to the recovery-curve AUC
///   with an auto-picked rank cutoff.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// AUCell values in form gene set x cells.
///
/// ### References
///
/// Aibar, et al., Nat Methods, 2017
pub fn calculate_aucell_metacells<T>(
    matrix: &CompressedSparseData2<T, f32>,
    gene_sets: &[Vec<usize>],
    params: Option<AucellParams>,
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors>
where
    T: BixverseNumeric,
{
    let verbosity = parse_verbosity_level(verbose);
    let params = params.unwrap_or_default();

    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csc => Cow::Owned(matrix.transform()),
    };

    let (n_cells, n_genes) = csr.shape;
    let data_2 = csr
        .data_2
        .as_ref()
        .ok_or(BixverseErrors::Data2NotAvailable)?;

    let start_ranking = Instant::now();
    let indptr_usize = csr.indptr.as_slice().index_cast();
    let indices_usize = csr.indices.as_slice().index_cast();
    let ranks = rank_within_rows_f32(&indptr_usize, &indices_usize, data_2, n_cells, n_genes);
    if verbosity.normal_verbosity() {
        println!(
            "Ranked gene expression within metacells: {:.2?}",
            start_ranking.elapsed()
        );
    }

    let start_auc = Instant::now();
    let max_rank = resolve_max_rank(params.max_rank, n_genes);
    let mut all_results: Vec<Vec<f32>> = vec![Vec::with_capacity(n_cells); gene_sets.len()];

    for cell_ranks in ranks {
        let aucs = score_cell(&cell_ranks, gene_sets, params.auc_type, max_rank);

        for (gene_set_idx, auc) in aucs.into_iter().enumerate() {
            all_results[gene_set_idx].push(auc);
        }
    }

    if params.standardise {
        standardise_rows(&mut all_results);
    }

    if verbosity.normal_verbosity() {
        println!("Calculated AUCs: {:.2?}", start_auc.elapsed());
    }

    Ok(all_results)
}
