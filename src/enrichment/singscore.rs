//! Implementation of singscore from Foroutan, Bhuva, et al.,
//! BMC Bioinformatics, 2018.

use faer::{ColRef, Mat, MatRef};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use rayon::prelude::*;

use crate::core::math::vector_helpers::{mad, median};
use crate::prelude::{BixverseFloat, MAD_SCALE};

///////////
// Enums //
///////////

/// Type of ranks fed into singscore. Affects only the theoretical score bounds.
#[derive(Debug, Clone, Copy, Default)]
pub enum SingscoreRankType {
    /// Column-wise ranks across the full background (output of
    /// `rank_matrix_col`). Bounds are derived from gene-set size and background
    /// size.
    #[default]
    Standard,
    /// Unit-normalised ranks computed relative to a set of stable genes (output
    /// of `rank_matrix_col_stable`). Bounds are fixed to [0, 1].
    Stable,
}

//////////////////
// Result types //
//////////////////

/// singscore result for one up set (+ optional down set) across many samples.
///
/// Up/down components are populated only when `down_set` was provided.
#[derive(Debug, Clone)]
pub struct SingscoreSingleResult<T: BixverseFloat> {
    /// Total score
    pub total_score: Vec<T>,
    /// Total dispersion
    pub total_dispersion: Vec<T>,
    /// Up scores
    pub up_score: Option<Vec<T>>,
    /// Dispersion of up scores
    pub up_dispersion: Option<Vec<T>>,
    /// Down scores
    pub down_score: Option<Vec<T>>,
    /// Dispersion of down scores
    pub down_dispersion: Option<Vec<T>>,
}

/// singscore result across many gene sets and many samples. Each matrix has
/// shape `gene_sets × samples`.
#[derive(Debug, Clone)]
pub struct SingscoreMultiResult<T: BixverseFloat> {
    /// The scores
    pub scores: Mat<T>,
    /// The dispersion of the scores
    pub dispersions: Mat<T>,
}

/// singscore permutation-test result.
#[derive(Debug, Clone)]
pub struct SingscorePermutationResult<T: BixverseFloat> {
    /// Per-sample total score from the real gene set
    pub observed_scores: Vec<T>,
    /// `B × n_samples` matrix of null total scores
    pub null_distribution: Mat<T>,
    /// Per-sample empirical one-tailed p-value, lower-bounded by `1 / B`
    pub pvals: Vec<T>,
}

/////////////
// Helpers //
/////////////

/// Theoretical score bounds.
///
/// ### Params
///
/// * `gs_size` - Gene set size.
/// * `bg_size` - Background size.
/// * `rank_type` - Selects the bounds formula.
///
/// ### Returns
///
/// `(lower, upper)` raw mean-rank bounds.
#[inline]
fn calc_bounds(gs_size: usize, bg_size: usize, rank_type: SingscoreRankType) -> (f64, f64) {
    match rank_type {
        SingscoreRankType::Stable => (0.0, 1.0),
        SingscoreRankType::Standard => {
            let low = (gs_size as f64 + 1.0) / 2.0;
            let up = (2.0 * bg_size as f64 - gs_size as f64 + 1.0) / 2.0;
            (low, up)
        }
    }
}

/// Bounds adjusted for the unknown-direction transform.
///
/// Halves `gs_size` and rounds `bg_size` up to reflect the transformed rank space.
///
/// ### Params
///
/// * `gs_size` - Gene set size.
/// * `bg_size` - Background size.
/// * `rank_type` - Selects the bounds formula.
///
/// ### Returns
///
/// `(lower, upper)` bounds in the transformed rank space.
#[inline]
fn calc_bounds_unknown_dir(
    gs_size: usize,
    bg_size: usize,
    rank_type: SingscoreRankType,
) -> (f64, f64) {
    calc_bounds(gs_size / 2, bg_size.div_ceil(2), rank_type)
}

/// Unknown-direction rank transform for one sample column.
///
/// Applies `|x - ceil(median(x))|` to each element.
///
/// ### Params
///
/// * `col` - Rank column for one sample.
///
/// ### Returns
///
/// Transformed ranks as a `Vec<T>`.
fn unknown_dir_transform<T: BixverseFloat>(col: ColRef<T>) -> Vec<T> {
    let v: Vec<T> = col.iter().copied().collect();
    let med = median(&v).expect("singscore: empty rank column");
    // Median can be x.5 under average ties; R's `ceiling` is what we want.
    let med_ceil = T::from_f64(med.to_f64().unwrap().ceil()).unwrap();
    v.into_iter().map(|x| (x - med_ceil).abs()).collect()
}

/// Apply the unknown-direction transform to every column in parallel.
///
/// ### Params
///
/// * `ranks` - Rank matrix (genes × samples).
///
/// ### Returns
///
/// Transformed matrix with the same shape as `ranks`.
fn build_unknown_dir_matrix<T: BixverseFloat>(ranks: &MatRef<T>) -> Mat<T> {
    let (n_genes, n_samples) = ranks.shape();
    let cols: Vec<Vec<T>> = (0..n_samples)
        .into_par_iter()
        .map(|j| unknown_dir_transform(ranks.col(j)))
        .collect();

    let mut out: Mat<T> = Mat::zeros(n_genes, n_samples);
    for (j, col_data) in cols.into_iter().enumerate() {
        for (i, v) in col_data.into_iter().enumerate() {
            out[(i, j)] = v;
        }
    }
    out
}

/// Score one gene set against one sample.
///
/// ### Params
///
/// * `col` - Rank column for one sample.
/// * `gene_set` - Row indices of the gene set.
/// * `bounds` - `(lower, upper)` bounds for normalisation.
/// * `score_offset` - Added to the normalised score (e.g. `-0.5` to centre).
///
/// ### Returns
///
/// `(score, dispersion)` for this sample.
#[inline]
fn score_at_sample<T: BixverseFloat>(
    col: ColRef<T>,
    gene_set: &[usize],
    bounds: (f64, f64),
    score_offset: f64,
) -> (T, T) {
    let (low, up) = bounds;
    let range_inv = 1.0 / (up - low);

    let gs_ranks: Vec<T> = gene_set.iter().map(|&i| col[i]).collect();
    let mean = gs_ranks.iter().map(|&x| x.to_f64().unwrap()).sum::<f64>() / gs_ranks.len() as f64;
    let score = T::from_f64((mean - low) * range_inv + score_offset).unwrap();
    let disp = mad(&gs_ranks, Some(T::from_f64(MAD_SCALE).unwrap())).unwrap_or_else(T::zero);

    (score, disp)
}

/// Score one gene set across all samples, parallel over samples.
///
/// ### Params
///
/// * `ranks` - Rank matrix (genes × samples).
/// * `gene_set` - Row indices of the gene set.
/// * `bounds` - `(lower, upper)` bounds for normalisation.
/// * `score_offset` - Added to each normalised score.
///
/// ### Returns
///
/// `(scores, dispersions)`, one value per sample.
fn score_one_set_par<T: BixverseFloat>(
    ranks: &MatRef<T>,
    gene_set: &[usize],
    bounds: (f64, f64),
    score_offset: f64,
) -> (Vec<T>, Vec<T>) {
    let (_, n_samples) = ranks.shape();
    let results: Vec<(T, T)> = (0..n_samples)
        .into_par_iter()
        .map(|j| score_at_sample(ranks.col(j), gene_set, bounds, score_offset))
        .collect();
    let mut scores = Vec::with_capacity(n_samples);
    let mut disps = Vec::with_capacity(n_samples);
    for (s, d) in results {
        scores.push(s);
        disps.push(d);
    }
    (scores, disps)
}

/// Score one gene set across all samples, serial over samples.
///
/// Use from contexts already parallelised at a higher level.
///
/// ### Params
///
/// * `ranks` - Rank matrix (genes × samples).
/// * `gene_set` - Row indices of the gene set.
/// * `bounds` - `(lower, upper)` bounds for normalisation.
/// * `score_offset` - Added to each normalised score.
///
/// ### Returns
///
/// `(scores, dispersions)`, one value per sample.
fn score_one_set_serial<T: BixverseFloat>(
    ranks: &MatRef<T>,
    gene_set: &[usize],
    bounds: (f64, f64),
    score_offset: f64,
) -> (Vec<T>, Vec<T>) {
    let (_, n_samples) = ranks.shape();
    let mut scores = Vec::with_capacity(n_samples);
    let mut disps = Vec::with_capacity(n_samples);
    for j in 0..n_samples {
        let (s, d) = score_at_sample(ranks.col(j), gene_set, bounds, score_offset);
        scores.push(s);
        disps.push(d);
    }
    (scores, disps)
}

/// Combine up-set and down-set scores into a single result.
///
/// Mirrors R's `calcScoresUpDn`: `dn_score = 1 - dn_raw + center_offset`,
/// `total = up + dn_score`, `dispersion = (up_disp + dn_disp) / 2`.
///
/// ### Params
///
/// * `up_scores` - Normalised scores for the up set.
/// * `up_disp` - Dispersions for the up set.
/// * `dn_raw` - Raw normalised scores for the down set (before inversion).
/// * `dn_disp` - Dispersions for the down set.
/// * `center_offset` - Offset applied when centering (e.g. `-0.5`).
///
/// ### Returns
///
/// [`SingscoreSingleResult`] with all up/down components populated.
fn combine_up_down<T: BixverseFloat>(
    up_scores: Vec<T>,
    up_disp: Vec<T>,
    dn_raw: Vec<T>,
    dn_disp: Vec<T>,
    center_offset: f64,
) -> SingscoreSingleResult<T> {
    let one_plus_offset = T::from_f64(1.0 + center_offset).unwrap();
    let two = T::from_f64(2.0).unwrap();

    let dn_scores: Vec<T> = dn_raw.into_iter().map(|s| one_plus_offset - s).collect();
    let total_score: Vec<T> = up_scores
        .iter()
        .zip(dn_scores.iter())
        .map(|(&u, &d)| u + d)
        .collect();
    let total_disp: Vec<T> = up_disp
        .iter()
        .zip(dn_disp.iter())
        .map(|(&a, &b)| (a + b) / two)
        .collect();

    SingscoreSingleResult {
        total_score,
        total_dispersion: total_disp,
        up_score: Some(up_scores),
        up_dispersion: Some(up_disp),
        down_score: Some(dn_scores),
        down_dispersion: Some(dn_disp),
    }
}

///////////////////////
// Stable-gene ranks //
///////////////////////

/// Unit-normalised column ranks relative to a stable gene set.
///
/// For each sample, counts how many stable genes have strictly lower expression
/// than each gene, adds 1, then divides by `n_stable + 1`. Yields ranks in
/// `(0, 1]`. Use with [`SingscoreRankType::Stable`] for correct score bounds.
///
/// ### Params
///
/// * `expr_matrix` - Expression matrix (genes × samples).
/// * `stable_gene_indices` - Row indices of the stable reference genes.
///
/// ### Returns
///
/// Stable rank matrix with the same shape as `expr_matrix`.
pub fn rank_matrix_col_stable<T: BixverseFloat>(
    expr_matrix: &MatRef<T>,
    stable_gene_indices: &[usize],
) -> Mat<T> {
    let (n_genes, n_samples) = expr_matrix.shape();
    let n_stable = stable_gene_indices.len();
    assert!(
        n_stable > 0,
        "rank_matrix_col_stable: empty stable_gene_indices"
    );

    let denom = T::from_f64(n_stable as f64 + 1.0).unwrap();
    let mut out: Mat<T> = Mat::zeros(n_genes, n_samples);

    out.par_col_iter_mut()
        .enumerate()
        .for_each(|(j, mut col_out)| {
            let col_in = expr_matrix.col(j);
            let stable_vals: Vec<T> = stable_gene_indices.iter().map(|&i| col_in[i]).collect();

            for i in 0..n_genes {
                let xi = col_in[i];
                let count = stable_vals.iter().filter(|&&sg| xi > sg).count();
                let rank = T::from_usize(count + 1).unwrap();
                col_out[i] = rank / denom;
            }
        });

    out
}

////////////////////
// Main functions //
////////////////////

/// singscore for a single gene set.
///
/// Scores one up set with an optional paired down set across all samples.
///
/// ### Params
///
/// * `rank_matrix` - Ranked expression matrix (genes × samples).
/// * `up_set` - Row indices of the up-regulated gene set.
/// * `down_set` - Row indices of the paired down-regulated set, if any.
/// * `center_score` - Shift scores by `-0.5`. Ignored when
///   `known_direction = false`.
/// * `known_direction` - If false, applies `|x - ceil(median(x))|` transform
///   per sample before scoring.
/// * `rank_type` - Selects the score bounds formula.
///
/// ### Returns
///
/// [`SingscoreSingleResult`] with total scores, dispersions, and optional
/// up/down components.
pub fn singscore_single<T: BixverseFloat>(
    rank_matrix: &MatRef<T>,
    up_set: &[usize],
    down_set: Option<&[usize]>,
    center_score: bool,
    known_direction: bool,
    rank_type: SingscoreRankType,
) -> SingscoreSingleResult<T> {
    let (n_genes, _) = rank_matrix.shape();

    if known_direction {
        let center_offset = if center_score { -0.5 } else { 0.0 };
        let up_bounds = calc_bounds(up_set.len(), n_genes, rank_type);
        let (up_scores, up_disp) = score_one_set_par(rank_matrix, up_set, up_bounds, center_offset);

        match down_set {
            None => SingscoreSingleResult {
                total_score: up_scores,
                total_dispersion: up_disp,
                up_score: None,
                up_dispersion: None,
                down_score: None,
                down_dispersion: None,
            },
            Some(dn) => {
                let dn_bounds = calc_bounds(dn.len(), n_genes, rank_type);
                let (dn_raw, dn_disp) = score_one_set_par(rank_matrix, dn, dn_bounds, 0.0);
                combine_up_down(up_scores, up_disp, dn_raw, dn_disp, center_offset)
            }
        }
    } else {
        let transformed = build_unknown_dir_matrix(rank_matrix);
        let transformed_ref = transformed.as_ref();
        let up_bounds = calc_bounds_unknown_dir(up_set.len(), n_genes, rank_type);
        let (up_scores, up_disp) = score_one_set_par(&transformed_ref, up_set, up_bounds, 0.0);

        match down_set {
            None => SingscoreSingleResult {
                total_score: up_scores,
                total_dispersion: up_disp,
                up_score: None,
                up_dispersion: None,
                down_score: None,
                down_dispersion: None,
            },
            Some(dn) => {
                // R uses regular calc_bounds (not unknown_dir) for the down set
                // here. Not sure why ... ?
                let dn_bounds = calc_bounds(dn.len(), n_genes, rank_type);
                let (dn_raw, dn_disp) = score_one_set_par(&transformed_ref, dn, dn_bounds, 0.0);
                combine_up_down(up_scores, up_disp, dn_raw, dn_disp, 0.0)
            }
        }
    }
}

/// singscore for many gene sets.
///
/// Parallelises over gene sets; each set is scored serially across samples.
/// `up_sets` and `down_sets`, when provided, must be the same length with
/// paired indexing.
///
/// ### Params
///
/// * `rank_matrix` - Ranked expression matrix (genes × samples).
/// * `up_sets` - Row indices for each up-regulated gene set.
/// * `down_sets` - Row indices for each paired down-regulated set, if any.
/// * `center_score` - Shift scores by `-0.5`. Ignored when
///   `known_direction = false`.
/// * `known_direction` - If false, applies the unknown-direction transform per
///   sample before scoring.
/// * `rank_type` - Selects the score bounds formula.
///
/// ### Returns
///
/// [`SingscoreMultiResult`] with score and dispersion matrices of shape
/// `gene_sets × samples`.
pub fn singscore_multi<T: BixverseFloat>(
    rank_matrix: &MatRef<T>,
    up_sets: &[Vec<usize>],
    down_sets: Option<&[Vec<usize>]>,
    center_score: bool,
    known_direction: bool,
    rank_type: SingscoreRankType,
) -> SingscoreMultiResult<T> {
    let (n_genes, n_samples) = rank_matrix.shape();
    let n_sets = up_sets.len();

    if let Some(dn) = down_sets {
        assert_eq!(
            up_sets.len(),
            dn.len(),
            "singscore_multi: up_sets and down_sets must have equal length"
        );
    }

    let working_mat: Option<Mat<T>> = if known_direction {
        None
    } else {
        Some(build_unknown_dir_matrix(rank_matrix))
    };
    let working_ref: MatRef<T> = match &working_mat {
        Some(m) => m.as_ref(),
        None => *rank_matrix,
    };
    let center_offset = if known_direction && center_score {
        -0.5
    } else {
        0.0
    };

    let per_set: Vec<(Vec<T>, Vec<T>)> = (0..n_sets)
        .into_par_iter()
        .map(|i| {
            let up = &up_sets[i];
            let up_bounds = if known_direction {
                calc_bounds(up.len(), n_genes, rank_type)
            } else {
                calc_bounds_unknown_dir(up.len(), n_genes, rank_type)
            };
            let (up_s, up_d) = score_one_set_serial(&working_ref, up, up_bounds, center_offset);

            if let Some(dns) = down_sets {
                let dn = &dns[i];
                let dn_bounds = calc_bounds(dn.len(), n_genes, rank_type);
                let (dn_raw, dn_d) = score_one_set_serial(&working_ref, dn, dn_bounds, 0.0);
                let combined = combine_up_down(up_s, up_d, dn_raw, dn_d, center_offset);
                (combined.total_score, combined.total_dispersion)
            } else {
                (up_s, up_d)
            }
        })
        .collect();

    let mut scores: Mat<T> = Mat::zeros(n_sets, n_samples);
    let mut dispersions: Mat<T> = Mat::zeros(n_sets, n_samples);
    for (i, (s_row, d_row)) in per_set.into_iter().enumerate() {
        for (j, &v) in s_row.iter().enumerate() {
            scores[(i, j)] = v;
        }
        for (j, &v) in d_row.iter().enumerate() {
            dispersions[(i, j)] = v;
        }
    }

    SingscoreMultiResult {
        scores,
        dispersions,
    }
}

/// Permutation test for singscore.
///
/// For each permutation, draws `n_up + n_down` random gene indices without
/// replacement and scores them with the same settings as the real call.
/// Computes per-sample empirical p-values as `max(1/B, mean(null > observed))`.
/// Parameters `center_score`, `known_direction`, and `rank_type` must match the
/// corresponding `singscore_single` call.
///
/// ### Params
///
/// * `rank_matrix` - Ranked expression matrix (genes × samples).
/// * `up_set` - Row indices of the up-regulated gene set.
/// * `down_set` - Row indices of the paired down-regulated set, if any.
/// * `center_score` - Must match the real scoring call.
/// * `known_direction` - Must match the real scoring call.
/// * `rank_type` - Must match the real scoring call.
/// * `n_permutations` - Number of permutations (`B`).
/// * `seed` - Master RNG seed; each permutation derives a deterministic
///   sub-seed from it.
///
/// ### Returns
///
/// [`SingscorePermutationResult`] with observed scores, the null distribution,
/// and p-values.
#[allow(clippy::too_many_arguments)]
pub fn singscore_permutation_test<T: BixverseFloat>(
    rank_matrix: &MatRef<T>,
    up_set: &[usize],
    down_set: Option<&[usize]>,
    center_score: bool,
    known_direction: bool,
    rank_type: SingscoreRankType,
    n_permutations: usize,
    seed: u64,
) -> SingscorePermutationResult<T> {
    let (n_genes, n_samples) = rank_matrix.shape();
    let n_up = up_set.len();
    let n_down = down_set.map_or(0, |d| d.len());
    let total_size = n_up + n_down;

    assert!(
        total_size > 0,
        "singscore_permutation_test: empty gene sets"
    );
    assert!(
        total_size <= n_genes,
        "singscore_permutation_test: combined gene set larger than background"
    );
    assert!(
        n_permutations > 0,
        "singscore_permutation_test: n_permutations must be > 0"
    );

    // Observed (uses the parallel single-set path).
    let observed = singscore_single(
        rank_matrix,
        up_set,
        down_set,
        center_score,
        known_direction,
        rank_type,
    );

    // Pre-build the working matrix once; permutations re-use it.
    let working_mat: Option<Mat<T>> = if known_direction {
        None
    } else {
        Some(build_unknown_dir_matrix(rank_matrix))
    };
    let working_ref: MatRef<T> = match &working_mat {
        Some(m) => m.as_ref(),
        None => *rank_matrix,
    };
    let center_offset = if known_direction && center_score {
        -0.5
    } else {
        0.0
    };

    let up_bounds = if known_direction {
        calc_bounds(n_up, n_genes, rank_type)
    } else {
        calc_bounds_unknown_dir(n_up, n_genes, rank_type)
    };
    let dn_bounds = if n_down > 0 {
        Some(calc_bounds(n_down, n_genes, rank_type))
    } else {
        None
    };

    let null_rows: Vec<Vec<T>> = (0..n_permutations)
        .into_par_iter()
        .map(|b| {
            // Per-permutation deterministic seed derived from the master seed.
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(b as u64));
            let mut pool: Vec<usize> = (0..n_genes).collect();
            pool.partial_shuffle(&mut rng, total_size);
            let chosen: &[usize] = &pool[..total_size];

            let random_up = &chosen[..n_up];
            let (up_s, up_d) =
                score_one_set_serial(&working_ref, random_up, up_bounds, center_offset);

            if let Some(dn_b) = dn_bounds {
                let random_dn = &chosen[n_up..];
                let (dn_raw, dn_d) = score_one_set_serial(&working_ref, random_dn, dn_b, 0.0);
                let combined = combine_up_down(up_s, up_d, dn_raw, dn_d, center_offset);
                combined.total_score
            } else {
                up_s
            }
        })
        .collect();

    let mut null_distribution: Mat<T> = Mat::zeros(n_permutations, n_samples);
    for (b, row) in null_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            null_distribution[(b, j)] = v;
        }
    }

    let b_inv = 1.0 / n_permutations as f64;
    let pvals: Vec<T> = (0..n_samples)
        .map(|j| {
            let obs = observed.total_score[j];
            let count = (0..n_permutations)
                .filter(|&b| null_distribution[(b, j)] > obs)
                .count();
            let p = ((count as f64) * b_inv).max(b_inv);
            T::from_f64(p).unwrap()
        })
        .collect();

    SingscorePermutationResult {
        observed_scores: observed.total_score,
        null_distribution,
        pvals,
    }
}
