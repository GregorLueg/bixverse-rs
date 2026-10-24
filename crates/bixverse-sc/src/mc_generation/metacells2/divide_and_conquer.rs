//! Divide-and-conquer MC2, upstream's `compute_divide_and_conquer_metacells`
//! (`metacells/pipeline/divide_and_conquer.py`).
//!
//! ### Phases (one subset)
//!
//! 1. **Preliminary**: random piles of about the target pile size, each run
//!    directly with complete cover; outliers of multi-pile rounds are re-piled
//!    until every cell is in a preliminary metacell.
//! 2. **Groups**: the preliminary metacells become "cells" (their profiles
//!    from [`super::collect`], mass = member count) and are themselves
//!    divided and conquered with the `piles_*` parameters. Each resulting
//!    group of metacells is one final pile.
//! 3. **Final**: the cells of each group are run directly. Unless
//!    `quick_and_dirty`, the outliers are then run through the whole subset
//!    procedure again one level up.
//!
//! A subset that fits one pile skips the groups phase: it is run directly,
//! then its outliers once more.
//!
//! Elements are cells at the top level and metacells inside the groups phase.
//! Cells are streamed from the reader one pile at a time, so memory is bounded
//! by the pile size times `max_parallel_piles`, not by the dataset.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::prelude::*;

use super::collect::collect_metacell_profiles;
use super::direct::{DirectMetacellsResult, compute_direct_metacells};
use super::params::{DacParams, MetacellsParams};
use super::pile::Pile;

////////////
// Consts //
////////////

/// Upstream's `max_metacells_level` for "until the outliers run out".
const UNBOUNDED_LEVEL: usize = 1000;

///////////
// Types //
///////////

/// Output of the divide-and-conquer pipeline. Per-cell vectors follow the
/// order of the `cell_indices` passed in.
#[derive(Debug, Clone)]
pub struct DacMetacellsResult {
    /// Metacell per cell, `-1` for outliers; dense IDs `[0, n_metacells)`.
    pub metacell_of_cell: Vec<i32>,
    /// True if the cell's candidate was dissolved in the last pile it was in.
    pub dissolved_of_cell: Vec<bool>,
    /// Level (number of outlier rounds) of the run that placed the cell.
    pub level_of_cell: Vec<i32>,
    /// Number of metacells.
    pub n_metacells: usize,
    /// Union of the genes selected in any pile, ascending.
    pub selected_genes: Vec<usize>,
}

/// Where the elements of one divide-and-conquer run come from.
enum Source<'a, S> {
    /// Cells streamed from a reader.
    Cells {
        /// The cell-based reader.
        reader: &'a S,
        /// Reader index of each element.
        cell_indices: &'a [usize],
        /// Mass per element.
        mass: &'a [f32],
        /// Genes in the store.
        n_genes: usize,
    },
    /// Metacell profiles held in memory (the groups phase).
    Groups {
        /// Profiles, one row per element.
        counts: &'a CompressedSparseData2<u32, f32>,
        /// Mass per element (member counts).
        mass: &'a [f32],
    },
}

impl<S: SingleCellReading> Source<'_, S> {
    /// Mass per element.
    ///
    /// ### Returns
    ///
    /// The masses.
    fn mass(&self) -> &[f32] {
        match self {
            Source::Cells { mass, .. } | Source::Groups { mass, .. } => mass,
        }
    }

    /// Genes per element.
    ///
    /// ### Returns
    ///
    /// The gene count.
    fn n_genes(&self) -> usize {
        match self {
            Source::Cells { n_genes, .. } => *n_genes,
            Source::Groups { counts, .. } => counts.shape.1,
        }
    }

    /// Raw rows of a set of elements, in order.
    ///
    /// ### Params
    ///
    /// * `elements` - Element indices.
    ///
    /// ### Returns
    ///
    /// Cell-major counts, one row per element.
    fn rows(&self, elements: &[usize]) -> Result<CompressedSparseData2<u32, f32>, ScErrors> {
        match self {
            Source::Cells {
                reader,
                cell_indices,
                ..
            } => {
                let globals: Vec<usize> = elements.iter().map(|&e| cell_indices[e]).collect();
                Ok(reader.read_cells_sparse(&globals, &DataLayerReturn::Raw)?)
            }
            Source::Groups { counts, .. } => Ok(counts.slice_rows(elements)?),
        }
    }

    /// A pile of elements, `cell_indices` set to the element indices.
    ///
    /// ### Params
    ///
    /// * `elements` - Element indices.
    ///
    /// ### Returns
    ///
    /// The [`Pile`].
    fn pile(&self, elements: &[usize]) -> Result<Pile, ScErrors> {
        let mass = elements.iter().map(|&e| self.mass()[e]).collect();
        Pile::from_raw(self.rows(elements)?, elements.to_vec(), Some(mass))
    }
}

/// Per-element results being assembled.
struct State {
    /// Metacell per element.
    metacell: Vec<i32>,
    /// Dissolved flag per element.
    dissolved: Vec<bool>,
    /// Level per element.
    level: Vec<i32>,
    /// Union of selected genes.
    selected: Vec<bool>,
}

impl State {
    /// Fresh state.
    ///
    /// ### Params
    ///
    /// * `n` - Element count.
    /// * `n_genes` - Gene count.
    ///
    /// ### Returns
    ///
    /// All elements unassigned.
    fn new(n: usize, n_genes: usize) -> Self {
        Self {
            metacell: vec![-1; n],
            dissolved: vec![false; n],
            level: vec![-1; n],
            selected: vec![false; n_genes],
        }
    }
}

/// Parameters of one divide-and-conquer run.
#[derive(Clone)]
struct Ctx {
    /// Per-pile parameters.
    direct: MetacellsParams,
    /// Divide-and-conquer parameters.
    dac: DacParams,
    /// Target elements per pile.
    target_pile_size: usize,
    /// RNG seed.
    seed: usize,
    /// Print progress.
    verbose: bool,
}

impl Ctx {
    /// The same run with complete cover forced on.
    ///
    /// ### Returns
    ///
    /// The modified context.
    fn must_cover(&self) -> Self {
        let mut ctx = self.clone();
        ctx.direct.must_complete_cover = true;
        ctx
    }
}

/////////////
// Helpers //
/////////////

/// Python's `int(round(x))`, half to even.
///
/// ### Params
///
/// * `x` - Value.
///
/// ### Returns
///
/// The rounded count, `0` for non-finite or negative input.
fn round_count(x: f64) -> usize {
    if x.is_finite() && x > 0.0 {
        x.round_ties_even() as usize
    } else {
        0
    }
}

/// Upstream's `compute_target_pile_size`.
///
/// The larger of `target_metacell_size * target_metacells_in_pile` and the
/// cells needed for `target_metacells_in_pile` metacells of
/// `target_metacell_umis` at the mean mass, clamped to the configured range.
///
/// ### Params
///
/// * `mass` - Mass per element.
/// * `target_metacell_size` - Target elements per metacell.
/// * `target_metacell_umis` - Target mass per metacell.
/// * `dac` - Divide-and-conquer parameters.
///
/// ### Returns
///
/// The target pile size.
pub fn compute_target_pile_size(
    mass: &[f32],
    target_metacell_size: usize,
    target_metacell_umis: f64,
    dac: &DacParams,
) -> usize {
    let by_cells = target_metacell_size * dac.target_metacells_in_pile;
    let mean = mass.iter().map(|&m| m as f64).sum::<f64>() / mass.len().max(1) as f64;
    let by_umis = round_count(target_metacell_umis / mean * dac.target_metacells_in_pile as f64);
    by_cells
        .max(by_umis)
        .max(dac.min_target_pile_size)
        .min(dac.max_target_pile_size)
}

/// Upstream's `random_piles`: split `n` elements into piles of about
/// `target_pile_size`, choosing whichever of the floor or ceiling pile count
/// deviates less from the target, then shuffle.
///
/// ### Params
///
/// * `n` - Element count, positive.
/// * `target_pile_size` - Target size, positive.
/// * `seed` - RNG seed.
///
/// ### Returns
///
/// The pile index of each element.
pub fn random_piles(n: usize, target_pile_size: usize, seed: u64) -> Vec<usize> {
    let ratio = n as f64 / target_pile_size as f64;
    let few = (ratio.floor() as usize).max(1);
    let many = ratio.ceil() as usize;
    let piles_count = if few == many {
        few
    } else {
        let few_factor = (n as f64 / few as f64) / target_pile_size as f64;
        let many_factor = target_pile_size as f64 / (n as f64 / many as f64);
        if few_factor < many_factor { few } else { many }
    };

    let minimal = n / piles_count;
    let extra = n - minimal * piles_count;
    let mut pile_of: Vec<usize> = (0..extra)
        .chain((0..piles_count).flat_map(|p| std::iter::repeat_n(p, minimal)))
        .collect();
    pile_of.shuffle(&mut StdRng::seed_from_u64(seed));
    pile_of
}

/// Group element lists by pile id.
///
/// ### Params
///
/// * `elements` - Elements, ascending.
/// * `pile_of` - Pile id per entry of `elements`.
///
/// ### Returns
///
/// Elements per pile, ascending within each pile.
fn bucket(elements: &[usize], pile_of: &[usize]) -> Vec<Vec<usize>> {
    let n_piles = pile_of.iter().copied().max().map_or(0, |m| m + 1);
    let mut piles = vec![Vec::new(); n_piles];
    for (&e, &p) in elements.iter().zip(pile_of) {
        piles[p].push(e);
    }
    piles
}

/// Random piles over a subset of elements.
///
/// ### Params
///
/// * `subset` - Elements, ascending.
/// * `ctx` - Run parameters.
///
/// ### Returns
///
/// Elements per pile.
fn random_subset_piles(subset: &[usize], ctx: &Ctx) -> Vec<Vec<usize>> {
    bucket(
        subset,
        &random_piles(subset.len(), ctx.target_pile_size, ctx.seed as u64),
    )
}

/// Run every pile directly and record the results, upstream's
/// `_compute_metacells_of_piles`.
///
/// With several piles, complete cover is switched off per pile, as upstream.
/// Metacell IDs continue from `counts`.
///
/// ### Params
///
/// * `source` - Element source.
/// * `state` - Results; updated in place.
/// * `piles` - Elements per pile.
/// * `counts` - Next free metacell ID; advanced in place.
/// * `level` - Level recorded for the placed elements.
/// * `ctx` - Run parameters.
fn run_piles<S: SingleCellReading>(
    source: &Source<S>,
    state: &mut State,
    piles: &[Vec<usize>],
    counts: &mut usize,
    level: usize,
    ctx: &Ctx,
) -> Result<(), ScErrors> {
    let mut direct = ctx.direct.clone();
    if piles.len() > 1 {
        direct.must_complete_cover = false;
    }

    let run = |elements: &Vec<usize>| -> Result<DirectMetacellsResult, ScErrors> {
        let mut pile = source.pile(elements)?;
        compute_direct_metacells(&mut pile, &direct, ctx.seed, false)
    };

    for chunk in piles.chunks(ctx.dac.max_parallel_piles.max(1)) {
        let results: Vec<DirectMetacellsResult> =
            chunk.par_iter().map(run).collect::<Result<_, _>>()?;
        for (elements, result) in chunk.iter().zip(results) {
            for (pos, &e) in elements.iter().enumerate() {
                let m = result.metacell_of_cell[pos];
                state.metacell[e] = if m >= 0 { m + *counts as i32 } else { -1 };
                state.dissolved[e] = result.dissolved_of_cell[pos];
                state.level[e] = level as i32;
            }
            for &g in &result.selected_genes {
                state.selected[g] = true;
            }
            *counts += result.n_metacells;
        }
    }
    if ctx.verbose {
        println!(
            " level {level}: {} piles of {} elements, {} metacells so far",
            piles.len(),
            piles.iter().map(|p| p.len()).sum::<usize>(),
            *counts
        );
    }
    Ok(())
}

/// Upstream's `_compute_metacells_in_levels`: run the piles, then keep
/// re-piling and re-running the outliers.
///
/// The initial pile count decides the stopping rule: one pile means one extra
/// round on the outliers; several mean rounds until no outliers remain or
/// `max_level` is reached.
///
/// ### Params
///
/// * `source` - Element source.
/// * `state` - Results; updated in place.
/// * `subset` - Elements of this run, ascending.
/// * `piles` - Initial piles.
/// * `counts` - Next free metacell ID.
/// * `level` - Starting level.
/// * `max_level` - Last level to run.
/// * `ctx` - Run parameters.
#[allow(clippy::too_many_arguments)]
fn run_in_levels<S: SingleCellReading>(
    source: &Source<S>,
    state: &mut State,
    subset: &[usize],
    mut piles: Vec<Vec<usize>>,
    counts: &mut usize,
    mut level: usize,
    max_level: usize,
    ctx: &Ctx,
) -> Result<(), ScErrors> {
    let single_pile = piles.len() == 1;
    let mut was_last = false;
    let mut subset = subset.to_vec();
    loop {
        let is_last = if level + 1 > max_level {
            true
        } else if single_pile {
            std::mem::replace(&mut was_last, true)
        } else {
            false
        };

        run_piles(source, state, &piles, counts, level + 1, ctx)?;
        if is_last {
            return Ok(());
        }

        level += 1;
        subset.retain(|&e| state.metacell[e] < 0);
        if subset.is_empty() {
            return Ok(());
        }
        piles = random_subset_piles(&subset, ctx);
    }
}

/// Upstream's `_compute_metacell_groups`: group the preliminary metacells of
/// a subset into final piles by dividing and conquering their profiles.
///
/// ### Params
///
/// * `source` - Element source.
/// * `state` - Results holding the preliminary metacells of `subset`.
/// * `subset` - Elements, ascending.
/// * `ctx` - Run parameters (complete cover).
///
/// ### Returns
///
/// Elements per final pile.
fn group_piles<S: SingleCellReading>(
    source: &Source<S>,
    state: &mut State,
    subset: &[usize],
    ctx: &Ctx,
) -> Result<Vec<Vec<usize>>, ScErrors> {
    // preliminary IDs within the subset are dense from zero
    let n_metacells = subset
        .iter()
        .map(|&e| state.metacell[e])
        .max()
        .map_or(0, |m| (m + 1).max(0) as usize);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_metacells];
    for &e in subset {
        let m = state.metacell[e];
        debug_assert!(m >= 0, "preliminary phase must cover every element");
        members[m as usize].push(e);
    }

    let (profiles, sizes) = collect_metacell_profiles(
        |elements| source.rows(elements),
        &members,
        source.n_genes(),
        ctx.target_pile_size,
    )?;

    let dac = &ctx.dac;
    let mut direct = ctx.direct.clone();
    direct.target_metacell_umis = ctx.target_pile_size as u64;
    direct.target_metacell_size = dac.target_metacells_in_pile;
    direct.knn.k_size_factor = dac.piles_knn_k_size_factor;
    direct.partition.min_split_size_factor = dac.piles_min_split_size_factor;
    direct.partition.max_merge_size_factor = dac.piles_max_merge_size_factor;
    direct.similarity = dac.groups_similarity.clone();
    direct.dissolve.min_robust_size_factor = dac.piles_min_robust_size_factor;
    direct.must_complete_cover = true;
    let group_ctx = Ctx {
        target_pile_size: compute_target_pile_size(
            &sizes,
            dac.target_metacells_in_pile,
            ctx.target_pile_size as f64,
            dac,
        ),
        direct,
        dac: dac.clone(),
        seed: ctx.seed,
        verbose: ctx.verbose,
    };

    let group_source: Source<S> = Source::Groups {
        counts: &profiles,
        mass: &sizes,
    };
    let mut group_state = State::new(n_metacells, source.n_genes());
    let all: Vec<usize> = (0..n_metacells).collect();
    let mut group_counts = 0;
    run_subset(
        &group_source,
        &mut group_state,
        &all,
        0,
        &mut group_counts,
        &group_ctx,
    )?;

    for (s, g) in state.selected.iter_mut().zip(&group_state.selected) {
        *s |= *g;
    }
    let pile_of: Vec<usize> = subset
        .iter()
        .map(|&e| group_state.metacell[state.metacell[e] as usize] as usize)
        .collect();
    Ok(bucket(subset, &pile_of))
}

/// Upstream's `_compute_divide_and_conquer_subset`.
///
/// ### Params
///
/// * `source` - Element source.
/// * `state` - Results; updated in place.
/// * `subset` - Elements of this run, ascending.
/// * `level` - Level of this run.
/// * `counts` - Next free metacell ID.
/// * `ctx` - Run parameters.
fn run_subset<S: SingleCellReading>(
    source: &Source<S>,
    state: &mut State,
    subset: &[usize],
    level: usize,
    counts: &mut usize,
    ctx: &Ctx,
) -> Result<(), ScErrors> {
    if subset.is_empty() {
        return Ok(());
    }
    for &e in subset {
        state.metacell[e] = -1;
    }

    let preliminary = random_subset_piles(subset, ctx);
    if preliminary.len() == 1 {
        return run_in_levels(
            source,
            state,
            subset,
            preliminary,
            counts,
            level,
            UNBOUNDED_LEVEL,
            ctx,
        );
    }

    let cover = ctx.must_cover();
    let mut preliminary_counts = 0;
    run_in_levels(
        source,
        state,
        subset,
        preliminary,
        &mut preliminary_counts,
        level,
        UNBOUNDED_LEVEL,
        &cover,
    )?;

    let final_piles = group_piles(source, state, subset, &cover)?;

    if ctx.dac.quick_and_dirty {
        let max_level = if ctx.direct.must_complete_cover {
            UNBOUNDED_LEVEL
        } else {
            1
        };
        return run_in_levels(
            source,
            state,
            subset,
            final_piles,
            counts,
            level,
            max_level,
            ctx,
        );
    }

    run_in_levels(
        source,
        state,
        subset,
        final_piles,
        counts,
        level,
        level,
        &cover,
    )?;
    if level == 0 || ctx.direct.must_complete_cover {
        let outliers: Vec<usize> = subset
            .iter()
            .copied()
            .filter(|&e| state.metacell[e] < 0)
            .collect();
        run_subset(source, state, &outliers, level + 1, counts, ctx)?;
    }
    Ok(())
}

//////////
// Main //
//////////

/// Compute metacells by divide and conquer.
///
/// ### Params
///
/// * `reader` - Cell-based reader.
/// * `cell_indices` - Reader indices of the cells to include.
/// * `mass_per_cell` - Optional mass per included cell (upstream's
///   `cell_umis`); the library sizes if `None`.
/// * `params` - Per-pile parameters.
/// * `dac` - Divide-and-conquer parameters.
/// * `seed` - RNG seed.
/// * `verbose` - Print progress.
///
/// ### Returns
///
/// [`DacMetacellsResult`] in the order of `cell_indices`.
///
/// ### Errors
///
/// `Mc2EmptyPile`, `Mc2LengthMismatch`, reader errors and anything a pile
/// raises.
pub fn compute_dac_metacells<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    mass_per_cell: Option<&[f32]>,
    params: &MetacellsParams,
    dac: &DacParams,
    seed: usize,
    verbose: bool,
) -> Result<DacMetacellsResult, ScErrors> {
    let n = cell_indices.len();
    if n == 0 {
        return Err(ScErrors::Mc2EmptyPile);
    }
    let mass: Vec<f32> = match mass_per_cell {
        Some(m) if m.len() != n => {
            return Err(ScErrors::Mc2LengthMismatch {
                what: "mass_per_cell",
                expected: n,
                got: m.len(),
            });
        }
        Some(m) => m.to_vec(),
        None => reader
            .read_cell_library_sizes(cell_indices)?
            .into_iter()
            .map(|l| l as f32)
            .collect(),
    };
    let n_genes = reader.get_header().total_genes;

    let target_pile_size = dac.target_pile_size.unwrap_or_else(|| {
        compute_target_pile_size(
            &mass,
            params.target_metacell_size,
            params.target_metacell_umis as f64,
            dac,
        )
    });
    if verbose {
        println!("Divide and conquer over {n} cells, target pile size {target_pile_size}");
    }

    let source = Source::Cells {
        reader,
        cell_indices,
        mass: &mass,
        n_genes,
    };
    let ctx = Ctx {
        direct: params.clone(),
        dac: dac.clone(),
        target_pile_size,
        seed,
        verbose,
    };
    let mut state = State::new(n, n_genes);
    let mut counts = 0;
    let all: Vec<usize> = (0..n).collect();
    run_subset(&source, &mut state, &all, 0, &mut counts, &ctx)?;

    Ok(DacMetacellsResult {
        metacell_of_cell: state.metacell,
        dissolved_of_cell: state.dissolved,
        level_of_cell: state.level,
        n_metacells: counts,
        selected_genes: (0..n_genes).filter(|&g| state.selected[g]).collect(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_piles_sizes_and_reproducibility() {
        // 25 / 10: floor 2 piles of 12.5 (factor 1.25) vs ceil 3 piles of
        // 8.33 (factor 1.2), so 3 piles of sizes 9, 8, 8
        let piles = random_piles(25, 10, 7);
        let mut sizes = [0usize; 3];
        for &p in &piles {
            sizes[p] += 1;
        }
        assert_eq!(sizes, [9, 8, 8]);
        assert_eq!(piles, random_piles(25, 10, 7));
    }

    #[test]
    fn test_random_piles_small_input_is_one_pile() {
        assert!(random_piles(5, 100, 1).iter().all(|&p| p == 0));
    }

    #[test]
    fn test_target_pile_size_clamps() {
        let dac = DacParams::default();
        // 48 * 100 = 4800 by cells; 160000 / 2000 * 100 = 8000 by UMIs
        let mass = vec![2000.0_f32; 10];
        assert_eq!(compute_target_pile_size(&mass, 48, 160_000.0, &dac), 8000);
        // tiny libraries push past the upper clamp
        let mass = vec![100.0_f32; 10];
        assert_eq!(compute_target_pile_size(&mass, 48, 160_000.0, &dac), 24000);
    }
}
