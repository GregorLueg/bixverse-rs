//! Bonsai trees over metacells.
//!
//! The metacell raw counts are integer sums over their cells, and a sum of
//! independent Poisson counts is Poisson whatever the per-cell rates, so
//! Sanity's model holds for a metacell as it does for a cell: it estimates the
//! metacell's library-weighted expression, with the metacell's total UMIs as
//! the library size. Bigger metacells get tighter error bars, which Bonsai
//! weights natively.
//!
//! The counts are already in memory, so genes are sliced out of the CSC matrix
//! chunk by chunk instead of read from disk; the chunk loop, the keep test and
//! everything after Sanity are the single-cell path's.

use bonsai_rs::sanity_sc_rs::input::CountMatrix;
use bonsai_rs::sanity_sc_rs::sanity_select;
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::mc_analysis::as_csc;
use crate::single_cell::sc_analysis::bonsai::{
    BonsaiScParams, BonsaiScResult, SANITY_GENE_CHUNK, chunk_sanity_params, keep_for_bonsai,
    run_bonsai_sc, stream_chunks,
};

//////////////////
// Sanity input //
//////////////////

/// Total counts of every metacell over all genes.
///
/// ### Params
///
/// * `csc` - Metacell counts, metacells by genes, CSC
///
/// ### Returns
///
/// One total per metacell, `f64` because Sanity takes it that way.
pub fn mc_totals(csc: &CompressedSparseData2<u32, f32>) -> Vec<f64> {
    let mut totals = vec![0.0f64; csc.nrows()];
    for (&row, &count) in csc.indices.iter().zip(&csc.data) {
        totals[row as usize] += count as f64;
    }
    totals
}

/// One chunk of genes as a Sanity count matrix.
///
/// A CSC column of a metacells-by-genes matrix is one gene across metacells,
/// which is the gene-major layout Sanity reads, so each gene is a slice.
/// Genes with no counts are left out, since Sanity cannot fit them.
///
/// ### Params
///
/// * `csc` - Metacell counts, metacells by genes, CSC
/// * `genes` - Genes to take, 0-indexed
///
/// ### Returns
///
/// The count matrix and the input gene index of each of its columns; `None`
/// if every gene in the chunk was empty.
pub fn mc_count_chunk(
    csc: &CompressedSparseData2<u32, f32>,
    genes: &[usize],
) -> Result<Option<(CountMatrix, Vec<usize>)>, BixverseErrors> {
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut indptr: Vec<usize> = vec![0];
    let mut kept: Vec<usize> = Vec::with_capacity(genes.len());

    for &gene in genes {
        let (lo, hi) = (csc.indptr[gene] as usize, csc.indptr[gene + 1] as usize);
        let counts = &csc.data[lo..hi];
        if counts.iter().all(|&k| k == 0) {
            continue;
        }
        indices.extend_from_slice(&csc.indices[lo..hi]);
        values.extend_from_slice(counts);
        indptr.push(indices.len());
        kept.push(gene);
    }

    if kept.is_empty() {
        return Ok(None);
    }
    Ok(Some((
        CountMatrix::new(indices, values, indptr, csc.nrows())?,
        kept,
    )))
}

////////////
// Bonsai //
////////////

/// Metacell counts to a laid-out Bonsai tree, Sanity on the CPU.
///
/// ### Params
///
/// * `counts` - Metacell raw counts, metacells by genes, either orientation
/// * `gene_indices` - Genes to consider, 0-indexed; only the ones passing
///   Bonsai's ingest filters are kept
/// * `params` - Sanity, Bonsai and layout parameters
/// * `verbosity` - How much to print
///
/// ### Returns
///
/// The tree over the metacells, in their row order, its layout, and the genes
/// it was built on.
pub fn sanity_bonsai_mc(
    counts: &CompressedSparseData2<u32, f32>,
    gene_indices: &[usize],
    params: &BonsaiScParams,
    verbosity: Verbosity,
) -> Result<BonsaiScResult, BixverseErrors> {
    let started = Instant::now();
    let csc = as_csc(counts);
    let totals = mc_totals(&csc);
    let sanity_params = chunk_sanity_params(params);
    let keep = keep_for_bonsai(params);

    let post = stream_chunks(
        gene_indices,
        SANITY_GENE_CHUNK,
        &totals,
        verbosity,
        |genes| mc_count_chunk(&csc, genes),
        |chunk, totals| Ok(sanity_select(chunk, totals, Some(sanity_params), &keep)?),
    )?;
    let t_sanity = started.elapsed().as_secs_f64();

    let mut res = run_bonsai_sc(post, gene_indices.len(), params, verbosity)?;
    res.timings.insert(0, ("sanity".to_string(), t_sanity));
    Ok(res)
}
