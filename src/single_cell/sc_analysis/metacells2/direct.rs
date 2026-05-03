//! Direct (single-pile) MC2 metacell pipeline.
//!
//! Runs the full pipeline on one pile of cells and returns metacell
//! assignments plus deviant/dissolved flags. This is the user-facing entry
//! point for the "no divide-and-conquer" mode.
//!
//! ### Stages
//!
//! 1. Downsample (per-row binomial).
//! 2. Select feature genes.
//! 3. Compute cell-cell similarity over selected genes.
//! 4. Choose `knn_k` from upstream heuristic (size / median / quantile).
//! 5. Build balanced KNN graph.
//! 6. Build incoming view.
//! 7. Compute candidate metacells (seeds + SA + cut/split/merge).
//! 8. Find deviant cells.
//! 9. Dissolve too-small candidates.
//! 10. Pack into result.
//!
//! Stages 1-2 mutate the `Pile` (caching downsampled and selected matrices).
//! All others are pure functions.

use std::time::Instant;

use crate::core::math::vector_helpers::{median, quantile};
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

/////////////
// Helpers //
/////////////

/// Output of the direct pipeline. All vectors are length `n_cells`.
#[derive(Debug, Clone)]
pub struct DirectMetacellsResult {
    /// Metacell assignment per cell. `-1` means outlier (deviant or
    /// dissolved). Surviving metacells have dense IDs `[0, k)` where `k =
    /// n_metacells`.
    pub metacell_of_cell: Vec<i32>,
    /// True if this cell was flagged as a deviant during deviant detection.
    pub deviant_of_cell: Vec<bool>,
    /// True if this cell was in a candidate that got dissolved (small metacells
    /// folded back into outliers). Distinct from `deviant`: a cell can be a
    /// deviant XOR dissolved, never both.
    pub dissolved_of_cell: Vec<bool>,
    /// Number of surviving metacells.
    pub n_metacells: usize,
}

/////////////
// Helpers //
/////////////

/// Compute `knn_k` matching upstream's `direct.py` heuristic.
///
/// Takes the maximum of four estimates: by cell count relative to target
/// metacell size, by median cell UMIs scaled by `knn_k_size_factor`, by a
/// UMI quantile, and the configured minimum. Returns `0` if all estimates
/// are non-positive, which the caller treats as a single-metacell early-out.
///
/// ### Params
///
/// * `n_cells` - Total number of cells.
/// * `umis_per_cell` - Per-cell UMI counts.
/// * `target_metacell_size` - Ideal number of cells per metacell.
/// * `target_metacell_umis` - Ideal UMI total per metacell.
/// * `knn_k_size_factor` - Scaling factor applied to the median-UMI estimate.
/// * `knn_k_umis_quantile` - Quantile of `umis_per_cell` used for the quantile
///   estimate.
/// * `min_knn_k` - Optional hard lower bound on the returned value.
/// * `knn_k_override` - If `Some`, returned directly without any computation.
///
/// ### Returns
///
/// The computed `k` as `usize`.
#[allow(clippy::too_many_arguments)]
fn compute_knn_k(
    n_cells: usize,
    umis_per_cell: &[f32],
    target_metacell_size: usize,
    target_metacell_umis: f64,
    knn_k_size_factor: f32,
    knn_k_umis_quantile: f32,
    min_knn_k: Option<usize>,
    knn_k_override: Option<usize>,
) -> usize {
    if let Some(k) = knn_k_override {
        return k;
    }

    let by_size = ((n_cells as f64) / (target_metacell_size as f64)).round() as i64;

    // Median cell UMIs.
    let median = median(umis_per_cell).unwrap();
    let by_median = if median > 0.0 {
        (knn_k_size_factor as f64 * target_metacell_umis / median as f64).round() as i64
    } else {
        0
    };
    let _by_median_check = if median > 0.0 {
        (target_metacell_umis / median as f64).round() as i64
    } else {
        0
    };

    let q_umis = quantile(umis_per_cell, knn_k_umis_quantile);
    let by_quantile = if q_umis > 0.0 {
        (target_metacell_umis / q_umis as f64).round() as i64
    } else {
        0
    };

    let floor = min_knn_k.unwrap_or(0) as i64;

    by_size.max(by_median).max(by_quantile).max(floor).max(0) as usize
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
/// `DirectMetacellsResult` with per-cell assignments and flags. The
/// indices in the returned vectors correspond to row indices in
/// `pile.raw` (i.e., `pile.cell_indices[i]` gives the global cell index).
pub fn compute_direct_metacells(
    pile: &mut Pile,
    params: &MetacellsParams,
    seed: usize,
    verbose: bool,
) -> Result<DirectMetacellsResult, BixverseErrors> {
    let start_pile = Instant::now();

    let n_cells = pile.raw.shape.0;

    if verbose {
        println!("Starting UMI downsampling and feature selection for the pile...")
    }
    downsample_pile(pile, &params.select, seed as u64);
    select_features(pile, &params.select);
    let end_downsampling_features = start_pile.elapsed();
    if verbose {
        println!(
            " Finished downsampling and feature selection in {:.2?}...",
            end_downsampling_features
        )
    }

    if verbose {
        println!("Computing similarities for the pile...")
    }
    let similarity = compute_similarity(pile, &params.similarity)?;
    let end_similarity = start_pile.elapsed();

    if verbose {
        println!(
            " Finished similarity calculations in {:.2?}...",
            end_similarity
        )
    }

    if verbose {
        println!("Computing balanced kNN for the pile...")
    }
    let knn_k = compute_knn_k(
        n_cells,
        &pile.umis_per_cell,
        params.target_metacell_size,
        params.target_metacell_umis as f64,
        params.knn.k_size_factor,
        params.knn.k_umis_quantile,
        params.knn.min_knn_k,
        params.knn.knn_k_override,
    );

    // early-out: too few cells, or too dense → single metacell.
    if knn_k == 0 || knn_k >= n_cells {
        return Ok(DirectMetacellsResult {
            metacell_of_cell: vec![0; n_cells],
            deviant_of_cell: vec![false; n_cells],
            dissolved_of_cell: vec![false; n_cells],
            n_metacells: 1,
        });
    }

    let outgoing = build_knn_graph(&similarity, knn_k, &params.knn);
    drop(similarity);

    let incoming = make_incoming_view(&outgoing);

    let end_knn = start_pile.elapsed();

    if verbose {
        println!(" Finished kNN generations in {:.2?}...", end_knn)
    }

    if verbose {
        println!("Identifying candidates for the pile...")
    }

    let candidate_of_cell = compute_candidate_metacells(
        &outgoing,
        &incoming,
        &pile.umis_per_cell,
        params,
        seed as u64,
    );
    drop(outgoing);
    drop(incoming);

    let end_candidates = start_pile.elapsed();

    if verbose {
        println!(
            " Finished candidates identification in {:.2?}...",
            end_candidates
        )
    }

    if verbose {
        println!("Identifying deviant cells and dissolve too small metacells for the pile...")
    }
    let deviant_of_cell = find_deviant_cells(
        &pile.raw,
        &pile.umis_per_cell,
        &candidate_of_cell,
        &params.deviants,
    );

    let (metacell_of_cell, dissolved_of_cell) = dissolve_metacells(
        &pile.raw,
        &pile.umis_per_cell,
        &candidate_of_cell,
        &deviant_of_cell,
        params.target_metacell_size,
        params.target_metacell_umis as f64,
        params.min_metacell_size,
        &params.dissolve,
    );

    let end_deviant_dissolving = start_pile.elapsed();

    if verbose {
        println!(
            " Finished deviant identification and dissolution in {:.2?}...",
            end_deviant_dissolving
        )
    }

    let n_metacells = (metacell_of_cell.iter().copied().max().unwrap_or(-1) + 1) as usize;

    let end_pile = start_pile.elapsed();

    if verbose {
        println!(" Finished the pile processing in {:.2?}...", end_pile)
    }

    Ok(DirectMetacellsResult {
        metacell_of_cell,
        deviant_of_cell,
        dissolved_of_cell,
        n_metacells,
    })
}
