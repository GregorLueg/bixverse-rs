//! Direct (single-pile) MC2 metacell pipeline, upstream's
//! `compute_direct_metacells` (`metacells/pipeline/direct.py`).
//!
//! ### Stages
//!
//! 1. Early out to a single candidate if the pile is too small to split.
//! 2. Downsample (per-row) and select feature genes.
//! 3. Choose `knn_k` (size / median / quantile heuristic); early out to a
//!    single candidate if it is zero or the median estimate covers the pile.
//! 4. Cell-cell similarity over the selected genes and a balanced kNN graph.
//! 5. Candidate metacells (seeds, SA, cut/split/merge).
//! 6. Unless `must_complete_cover`: deviant cells, then dissolve.
//!
//! Stage 2 mutates the `Pile` (caching downsampled and selected matrices).

use std::time::Instant;

use crate::prelude::*;

use super::candidates::{compute_candidate_metacells, make_incoming_view};
use super::deviants::find_deviant_cells;
use super::dissolve::dissolve_metacells;
use super::downsample::downsample_pile;
use super::knn::build_knn_graph;
use super::params::MetacellsParams;
use super::pile::Pile;
use super::select::select_features;
use super::similarity::compute_similarity;

////////////
// Consts //
////////////

/// Upstream rejects target sizes at or above this, as they look like UMIs.
const MAX_TARGET_METACELL_SIZE: usize = 1000;

///////////
// Types //
///////////

/// Output of the direct pipeline. Per-cell vectors are length `n_cells`.
#[derive(Debug, Clone)]
pub struct DirectMetacellsResult {
    /// Metacell assignment per cell. `-1` means outlier (deviant or
    /// dissolved). Surviving metacells have dense IDs `[0, n_metacells)`.
    pub metacell_of_cell: Vec<i32>,
    /// True if this cell was flagged as a deviant.
    pub deviant_of_cell: Vec<bool>,
    /// True if this cell was in a candidate that got dissolved. A cell can be
    /// a deviant or dissolved, never both.
    pub dissolved_of_cell: Vec<bool>,
    /// Number of surviving metacells.
    pub n_metacells: usize,
    /// Genes used for the similarity, in the pile's gene space. Empty when
    /// the pile took the single-candidate early out.
    pub selected_genes: Vec<usize>,
}

/////////////
// Helpers //
/////////////

/// `knn_k` estimates, as upstream's `direct.py`.
#[derive(Debug, Clone, Copy)]
struct KnnK {
    /// The `k` to use: the override, or the maximum of the estimates.
    k: usize,
    /// `round(target_metacell_umis / median)`; the pile is a single candidate
    /// when this reaches the cell count.
    of_median: usize,
}

/// Python's `int(round(x))` for a non-negative estimate.
///
/// ### Params
///
/// * `x` - Value.
///
/// ### Returns
///
/// `x` rounded half to even, `0` for non-finite or negative input.
fn round_count(x: f64) -> usize {
    if x.is_finite() && x > 0.0 {
        x.round_ties_even() as usize
    } else {
        0
    }
}

/// Compute `knn_k` with upstream's heuristic.
///
/// Takes the maximum of the estimate by cell count, by median mass scaled by
/// `k_size_factor`, by a mass quantile, and the configured minimum.
///
/// ### Params
///
/// * `mass_per_cell` - Per-cell mass.
/// * `params` - MC2 parameters.
///
/// ### Returns
///
/// The [`KnnK`] estimates.
fn compute_knn_k(mass_per_cell: &[f32], params: &MetacellsParams) -> KnnK {
    let n_cells = mass_per_cell.len();
    let target_umis = params.target_metacell_umis as f64;

    let mut sorted: Vec<f64> = mass_per_cell.iter().map(|&v| v as f64).collect();
    sorted.sort_unstable_by(f64::total_cmp);
    let quantile = |q: f64| {
        let pos = q.clamp(0.0, 1.0) * (n_cells - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n_cells - 1);
        sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
    };
    let median = quantile(0.5);

    let of_median = round_count(target_umis / median);
    let k = match params.knn.knn_k_override {
        Some(k) => k,
        None => {
            let by_size = round_count(n_cells as f64 / params.target_metacell_size as f64);
            let by_median = round_count(params.knn.k_size_factor * target_umis / median);
            let by_quantile = round_count(target_umis / quantile(params.knn.k_umis_quantile));
            by_size
                .max(by_median)
                .max(by_quantile)
                .max(params.knn.min_knn_k.unwrap_or(0))
        }
    };
    KnnK { k, of_median }
}

/// Whether the pile is too small to split, upstream's first early out.
///
/// ### Params
///
/// * `n_cells` - Cells in the pile.
/// * `total_mass` - Summed mass.
/// * `params` - MC2 parameters.
///
/// ### Returns
///
/// `true` if every cell should go into one candidate.
fn too_small_to_split(n_cells: usize, total_mass: f64, params: &MetacellsParams) -> bool {
    let merge = params.partition.max_merge_size_factor;
    let size_floor =
        (params.min_metacell_size as f64).min(params.target_metacell_size as f64 * merge);
    n_cells as f64 <= 2.0 * size_floor
        || total_mass <= 2.0 * params.target_metacell_umis as f64 * merge
}

/// Print a stage time when verbose.
///
/// ### Params
///
/// * `verbose` - Whether to print.
/// * `stage` - Stage name.
/// * `start` - Pile start time.
fn report(verbose: bool, stage: &str, start: Instant) {
    if verbose {
        println!(" {stage} done at {:.2?}", start.elapsed());
    }
}

//////////
// Main //
//////////

/// Run the direct MC2 pipeline on one pile.
///
/// ### Params
///
/// * `pile` - Mutable pile; downsampled and selected matrices will be
///   populated as side effects.
/// * `params` - Top-level metacells parameters.
/// * `seed` - Master RNG seed.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// [`DirectMetacellsResult`] with per-cell assignments and flags, indexed by
/// pile row (`pile.cell_indices[i]` gives the reader index).
///
/// ### Errors
///
/// `Mc2EmptyPile`, `Mc2InvalidParams`, and anything the stages raise.
pub fn compute_direct_metacells(
    pile: &mut Pile,
    params: &MetacellsParams,
    seed: usize,
    verbose: bool,
) -> Result<DirectMetacellsResult, ScErrors> {
    let start = Instant::now();
    let n_cells = pile.n_cells();
    if n_cells == 0 {
        return Err(ScErrors::Mc2EmptyPile);
    }
    if params.target_metacell_size >= MAX_TARGET_METACELL_SIZE {
        return Err(ScErrors::Mc2InvalidParams {
            reason: format!(
                "target_metacell_size {} looks like UMIs, should be cells",
                params.target_metacell_size
            ),
        });
    }

    let total_mass: f64 = pile.mass_per_cell.iter().map(|&m| m as f64).sum();
    let mut selected_genes = Vec::new();

    let candidate_of_cell = if too_small_to_split(n_cells, total_mass, params) {
        vec![0; n_cells]
    } else {
        downsample_pile(pile, &params.select, seed as u64);
        select_features(pile, &params.select)?;
        report(verbose, "downsampling and feature selection", start);

        let knn_k = compute_knn_k(&pile.mass_per_cell, params);
        if knn_k.k == 0 || knn_k.of_median >= n_cells {
            vec![0; n_cells]
        } else {
            let similarity = compute_similarity(pile, &params.similarity)?;
            report(verbose, "similarity", start);

            let outgoing = build_knn_graph(&similarity, knn_k.k, &params.knn);
            drop(similarity);
            let incoming = make_incoming_view(&outgoing);
            report(verbose, "kNN graph", start);

            let candidates = compute_candidate_metacells(
                &outgoing,
                &incoming,
                &pile.mass_per_cell,
                params,
                seed as u64,
            )?;
            report(verbose, "candidates", start);
            selected_genes = pile.selected_gene_indices.clone().unwrap_or_default();
            candidates
        }
    };

    if params.must_complete_cover {
        let n_metacells = (candidate_of_cell.iter().copied().max().unwrap_or(-1) + 1) as usize;
        return Ok(DirectMetacellsResult {
            metacell_of_cell: candidate_of_cell,
            deviant_of_cell: vec![false; n_cells],
            dissolved_of_cell: vec![false; n_cells],
            n_metacells,
            selected_genes,
        });
    }

    let deviant_of_cell = find_deviant_cells(
        &pile.raw,
        &pile.umis_per_cell,
        &candidate_of_cell,
        &params.deviants,
    )?;
    let (metacell_of_cell, dissolved_of_cell) = dissolve_metacells(
        &pile.raw,
        &pile.umis_per_cell,
        &pile.mass_per_cell,
        &candidate_of_cell,
        &deviant_of_cell,
        params.target_metacell_size,
        params.target_metacell_umis as f64,
        params.min_metacell_size,
        &params.dissolve,
    );
    report(verbose, "deviants and dissolve", start);

    let n_metacells = (metacell_of_cell.iter().copied().max().unwrap_or(-1) + 1) as usize;
    Ok(DirectMetacellsResult {
        metacell_of_cell,
        deviant_of_cell,
        dissolved_of_cell,
        n_metacells,
        selected_genes,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_count_is_half_to_even() {
        assert_eq!(round_count(2.5), 2);
        assert_eq!(round_count(3.5), 4);
        assert_eq!(round_count(f64::INFINITY), 0);
        assert_eq!(round_count(-1.0), 0);
    }

    #[test]
    fn test_too_small_to_split_matches_upstream_bounds() {
        let params = MetacellsParams::default();
        // 2 * min(12, 48 * 0.5) = 24 cells; 2 * 160000 * 0.5 = 160000 UMIs
        assert!(too_small_to_split(24, 1e9, &params));
        assert!(!too_small_to_split(25, 1e9, &params));
        assert!(too_small_to_split(1000, 160_000.0, &params));
        assert!(!too_small_to_split(1000, 160_001.0, &params));
    }

    #[test]
    fn test_knn_k_takes_max_of_estimates() {
        let mut params = MetacellsParams::default();
        params.knn.min_knn_k = None;
        // 960 cells / 48 = 20; median 4000: 2 * 160000 / 4000 = 80; 160000 /
        // 4000 = 40 by quantile
        let mass = vec![4000.0_f32; 960];
        let k = compute_knn_k(&mass, &params);
        assert_eq!(k.k, 80);
        assert_eq!(k.of_median, 40);
    }
}
