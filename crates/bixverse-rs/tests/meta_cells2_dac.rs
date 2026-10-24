//! Divide-and-conquer MetaCells2 over a cell store on disk.
//!
//! Eight planted clusters of 400 cells, rebuilt from the same LCG as
//! `dev/gen_metacells2_dac_fixtures.py`, with the target pile size forced to
//! 800 so the run goes through the preliminary, groups and final phases.
//! Divide and conquer is stochastic, so the gate is a band: summary statistics
//! over five seeds must sit within the spread upstream shows over its five
//! seeds. Upstream mixes planted clusters too: preliminary leftovers too small
//! to form their own metacell are glued onto others under complete cover, and
//! the gap rule only removes a few cells per metacell.

#![cfg(feature = "single-cell")]

mod meta_cells2_dac_fixtures;
mod meta_cells2_rare_dac_fixtures;

use std::path::PathBuf;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::mc_generation::metacells2::{
    DacMetacellsResult, DacParams, MC2KnnParams, MetacellsParams, RareParams, compute_dac_metacells,
};
use bixverse_rs::single_cell::sc_data::data_io::CellGeneSparseWriter;
use meta_cells2_dac_fixtures as fx;
use meta_cells2_rare_dac_fixtures as rfx;

///////////////
// Constants //
///////////////

/// Total cells.
#[cfg(feature = "large-test")]
const N_CELLS: usize = fx::N_CLUSTERS * fx::CELLS_PER_CLUSTER;
/// Total genes.
const N_GENES: usize = fx::N_CLUSTERS * fx::MARKERS + fx::NOISE_GENES;
/// Seeds run on the Rust side.
#[cfg(feature = "large-test")]
const SEEDS: [usize; 5] = [1, 2, 3, 4, 5];
/// Store normalisation target, irrelevant to MC2 which reads raw counts.
const TARGET_SIZE: f32 = 1e4;

/////////////
// Helpers //
/////////////

/// Removes the scratch store when the test ends, pass or panic.
struct TempStore(PathBuf);

impl Drop for TempStore {
    /// Removes the file, ignoring a missing one.
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempStore {
    /// The store path as the writer and reader take it.
    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// Write the generator's counts as a cell-major store, for the first
/// `n_clusters` clusters.
///
/// ### Params
///
/// * `name` - Distinguishes the scratch file.
/// * `n_clusters` - Clusters to write; [`fx::N_CLUSTERS`] for the reference.
///
/// ### Returns
///
/// The store guard.
fn build_store(name: &str, n_clusters: usize) -> TempStore {
    let store = TempStore(std::env::temp_dir().join(format!("bixverse_mc2_dac_{name}.bin")));
    let mut x = fx::LCG_SEED;
    let mut next = || {
        x = (1_664_525 * x + 1_013_904_223) % (1 << 32);
        x >> 16
    };

    let n_cells = n_clusters * fx::CELLS_PER_CLUSTER;
    let mut writer = CellGeneSparseWriter::new(store.path(), true, n_cells, N_GENES, TARGET_SIZE)
        .expect("writer opens");
    for i in 0..n_cells {
        let cluster = i / fx::CELLS_PER_CLUSTER;
        let mut indices = Vec::new();
        let mut counts = Vec::new();
        for g in 0..N_GENES {
            let hi = next();
            let v = if (cluster * fx::MARKERS..(cluster + 1) * fx::MARKERS).contains(&g) {
                hi % 41
            } else {
                u64::from(hi.is_multiple_of(4))
            } as u32;
            if v > 0 {
                indices.push(g as u32);
                counts.push(v);
            }
        }
        writer
            .write_cell_chunk(CsrCellChunk::from_data(
                &counts,
                &indices,
                i,
                TARGET_SIZE,
                true,
            ))
            .expect("chunk writes");
    }
    writer.finalise().expect("store finalises");
    store
}

/// The generator's parameters.
fn params() -> (MetacellsParams, DacParams) {
    let params = MetacellsParams {
        target_metacell_size: 25,
        target_metacell_umis: 25_000,
        min_metacell_size: 8,
        knn: MC2KnnParams {
            min_knn_k: Some(10),
            ..Default::default()
        },
        ..Default::default()
    };
    let dac = DacParams {
        min_target_pile_size: fx::PILE_SIZE,
        max_target_pile_size: fx::PILE_SIZE,
        target_metacells_in_pile: 20,
        ..Default::default()
    };
    (params, dac)
}

/// `(n_metacells, outlier cells, mixed metacells, misplaced cells)`, as the
/// generator's `stats`.
///
/// ### Params
///
/// * `result` - A run.
///
/// ### Returns
///
/// The statistics.
#[cfg(feature = "large-test")]
fn stats(result: &DacMetacellsResult) -> (usize, usize, usize, usize) {
    let n = result.n_metacells;
    let mut composition = vec![[0usize; fx::N_CLUSTERS]; n];
    let mut outliers = 0;
    for (i, &m) in result.metacell_of_cell.iter().enumerate() {
        if m < 0 {
            outliers += 1;
        } else {
            composition[m as usize][i / fx::CELLS_PER_CLUSTER] += 1;
        }
    }
    let mixed = composition
        .iter()
        .filter(|c| c.iter().filter(|&&v| v > 0).count() > 1)
        .count();
    let misplaced = composition
        .iter()
        .map(|c| c.iter().sum::<usize>() - c.iter().max().unwrap())
        .sum();
    (n, outliers, mixed, misplaced)
}

/// Median of a few counts.
#[cfg(feature = "large-test")]
fn median(mut v: Vec<usize>) -> usize {
    v.sort_unstable();
    v[v.len() / 2]
}

///////////
// Tests //
///////////

/// Dense IDs, placed cells never flagged dissolved, every cell placed by some
/// level.
///
/// ### Params
///
/// * `result` - A run.
fn assert_structure(result: &DacMetacellsResult) {
    let mut seen = vec![false; result.n_metacells];
    for (i, &m) in result.metacell_of_cell.iter().enumerate() {
        assert!(result.level_of_cell[i] >= 1, "cell {i} never ran");
        if m >= 0 {
            seen[m as usize] = true;
            assert!(
                !result.dissolved_of_cell[i],
                "cell {i} placed yet dissolved"
            );
        }
    }
    assert!(seen.iter().all(|&s| s), "metacell ids are not dense");
}

/// Two clusters, three preliminary piles: every phase runs on a small input,
/// and the same seed reproduces the run, rare detection on or off.
#[test]
fn test_dac_structure_and_reproducibility() {
    let n_clusters = 2;
    let store = build_store("small", n_clusters);
    let reader = ParallelSparseReader::new(store.path()).expect("store opens");
    let cells: Vec<usize> = (0..n_clusters * fx::CELLS_PER_CLUSTER).collect();
    let (params, mut dac) = params();
    dac.min_target_pile_size = 250;
    dac.max_target_pile_size = 250;

    let a =
        compute_dac_metacells(&reader, &cells, None, &params, &dac, 3, false).expect("dac runs");
    // no rare candidates here, so detection off must not change the run
    dac.rare = None;
    let b =
        compute_dac_metacells(&reader, &cells, None, &params, &dac, 3, false).expect("dac runs");
    assert_structure(&a);
    assert!(a.rare_module_of_cell.iter().all(|&m| m == -1));
    assert!(a.n_metacells >= n_clusters);
    assert_eq!(a.metacell_of_cell, b.metacell_of_cell);
    assert_eq!(a.level_of_cell, b.level_of_cell);
}

/// Statistics over five seeds within upstream's band.
#[test]
// 3200 cells, six divide-and-conquer runs
#[cfg(feature = "large-test")]
fn test_dac_matches_upstream_statistics() {
    let store = build_store("band", fx::N_CLUSTERS);
    let reader = ParallelSparseReader::new(store.path()).expect("store opens");
    let cells: Vec<usize> = (0..N_CELLS).collect();
    let (params, dac) = params();

    let runs: Vec<DacMetacellsResult> = SEEDS
        .iter()
        .map(|&seed| {
            compute_dac_metacells(&reader, &cells, None, &params, &dac, seed, false)
                .expect("dac runs")
        })
        .collect();
    runs.iter().for_each(assert_structure);

    let ours: Vec<(usize, usize, usize, usize)> = runs.iter().map(stats).collect();
    let column = |rows: &[(usize, usize, usize, usize)], k: usize| -> Vec<usize> {
        rows.iter().map(|r| [r.0, r.1, r.2, r.3][k]).collect()
    };
    let names = [
        "metacells",
        "outliers",
        "mixed metacells",
        "misplaced cells",
    ];
    for (k, name) in names.iter().enumerate() {
        let upstream = column(fx::UPSTREAM_STATS, k);
        let lo = *upstream.iter().min().unwrap();
        let hi = *upstream.iter().max().unwrap();
        let ours_k = column(&ours, k);
        let m = median(ours_k.clone());
        // a band of upstream's spread, widened by a tenth and a few units of
        // slack for the small counts
        let slack = (hi - lo) / 10 + 5;
        assert!(
            m + slack >= lo && m <= hi + slack,
            "{name}: median {m} over {ours_k:?}, upstream {upstream:?}"
        );
    }
}

////////////////////////////
// Rare gene module cells //
////////////////////////////

/// Planted cluster of a rare-fixture cell and its rare population, `None`
/// for the common cells.
///
/// ### Params
///
/// * `i` - Cell.
/// * `n_common` - Common cells, `n_clusters * CELLS_PER_CLUSTER`.
///
/// ### Returns
///
/// `(cluster, population)`.
fn rare_population(i: usize, n_common: usize) -> (usize, Option<usize>) {
    if i < n_common {
        return (i / rfx::CELLS_PER_CLUSTER, None);
    }
    let mut start = n_common;
    let mut r = 0;
    while i >= start + rfx::RARE_SIZES[r] {
        start += rfx::RARE_SIZES[r];
        r += 1;
    }
    (rfx::RARE_HOSTS[r], Some(r))
}

/// Write the rare generator's counts as a cell-major store: the first
/// `n_clusters` planted clusters, then the rare populations.
///
/// ### Params
///
/// * `name` - Distinguishes the scratch file.
/// * `n_clusters` - Planted clusters; [`rfx::N_CLUSTERS`] for the reference.
///   Must exceed every host cluster.
///
/// ### Returns
///
/// The store guard and the cell count.
fn build_rare_store(name: &str, n_clusters: usize) -> (TempStore, usize) {
    let store = TempStore(std::env::temp_dir().join(format!("bixverse_mc2_rare_{name}.bin")));
    let n_genes = rfx::N_CLUSTERS * rfx::MARKERS
        + rfx::NOISE_GENES
        + rfx::RARE_SIZES.len() * rfx::RARE_MARKERS;
    let rare_start = rfx::N_CLUSTERS * rfx::MARKERS + rfx::NOISE_GENES;
    let n_common = n_clusters * rfx::CELLS_PER_CLUSTER;
    let n_cells = n_common + rfx::RARE_SIZES.iter().sum::<usize>();
    let mut x = rfx::LCG_SEED;
    let mut next = || {
        x = (1_664_525 * x + 1_013_904_223) % (1 << 32);
        x >> 16
    };

    let mut writer = CellGeneSparseWriter::new(store.path(), true, n_cells, n_genes, TARGET_SIZE)
        .expect("writer opens");
    for i in 0..n_cells {
        let (cluster, population) = rare_population(i, n_common);
        let mut indices = Vec::new();
        let mut counts = Vec::new();
        for g in 0..n_genes {
            let hi = next();
            let v = if (cluster * rfx::MARKERS..(cluster + 1) * rfx::MARKERS).contains(&g) {
                hi % 41
            } else if g >= rare_start {
                let own = population.is_some_and(|r| {
                    (rare_start + r * rfx::RARE_MARKERS..rare_start + (r + 1) * rfx::RARE_MARKERS)
                        .contains(&g)
                });
                if own { hi % 11 } else { 0 }
            } else {
                u64::from(hi.is_multiple_of(4))
            } as u32;
            if v > 0 {
                indices.push(g as u32);
                counts.push(v);
            }
        }
        writer
            .write_cell_chunk(CsrCellChunk::from_data(
                &counts,
                &indices,
                i,
                TARGET_SIZE,
                true,
            ))
            .expect("chunk writes");
    }
    writer.finalise().expect("store finalises");
    (store, n_cells)
}

/// Cells of each rare population in metacells it is the majority of, as the
/// rare generator's `recovered`.
///
/// ### Params
///
/// * `result` - A run.
/// * `n_common` - Common cells.
///
/// ### Returns
///
/// Recovered cells per population.
fn rare_recovered(result: &DacMetacellsResult, n_common: usize) -> Vec<usize> {
    let truth: Vec<Option<usize>> = (0..result.metacell_of_cell.len())
        .map(|i| rare_population(i, n_common).1)
        .collect();
    let mut members = vec![Vec::new(); result.n_metacells];
    for (i, &m) in result.metacell_of_cell.iter().enumerate() {
        if m >= 0 {
            members[m as usize].push(i);
        }
    }
    (0..rfx::RARE_SIZES.len())
        .map(|r| {
            (0..truth.len())
                .filter(|&i| truth[i] == Some(r))
                .filter(|&i| {
                    let m = result.metacell_of_cell[i];
                    m >= 0 && {
                        let own = members[m as usize]
                            .iter()
                            .filter(|&&j| truth[j] == Some(r))
                            .count();
                        own * 2 > members[m as usize].len()
                    }
                })
                .count()
        })
        .collect()
}

/// Every rare cell, and only those, sits in the module of its population.
///
/// ### Params
///
/// * `result` - A run.
/// * `n_common` - Common cells.
fn assert_rare_modules(result: &DacMetacellsResult, n_common: usize) {
    let n_populations = rfx::RARE_SIZES.len();
    let mut module_of_population = vec![None; n_populations];
    for (i, &m) in result.rare_module_of_cell.iter().enumerate() {
        match rare_population(i, n_common).1 {
            None => assert_eq!(m, -1, "common cell {i} in rare module {m}"),
            Some(r) => {
                assert!(m >= 0, "rare cell {i} has no module");
                assert_eq!(*module_of_population[r].get_or_insert(m), m);
            }
        }
    }
    let mut seen: Vec<i32> = module_of_population.iter().flatten().copied().collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), n_populations, "populations share a module");
}

/// Dense IDs, placed cells never flagged dissolved, every cell run (rare
/// piles are level 0).
///
/// ### Params
///
/// * `result` - A run.
fn assert_rare_structure(result: &DacMetacellsResult) {
    let mut seen = vec![false; result.n_metacells];
    for (i, &m) in result.metacell_of_cell.iter().enumerate() {
        assert!(result.level_of_cell[i] >= 0, "cell {i} never ran");
        if m >= 0 {
            seen[m as usize] = true;
            assert!(
                !result.dissolved_of_cell[i],
                "cell {i} placed yet dissolved"
            );
        }
    }
    assert!(seen.iter().all(|&s| s), "metacell ids are not dense");
}

/// The rare generator's parameters.
///
/// ### Params
///
/// * `pile_size` - Target pile size, fixed.
/// * `max_gene_cell_fraction` - Rare candidate fraction cap.
///
/// ### Returns
///
/// The per-pile and divide-and-conquer parameters.
fn rare_params(pile_size: usize, max_gene_cell_fraction: f64) -> (MetacellsParams, DacParams) {
    let (params, mut dac) = params();
    dac.min_target_pile_size = pile_size;
    dac.max_target_pile_size = pile_size;
    dac.rare = Some(RareParams {
        max_gene_cell_fraction,
        ..Default::default()
    });
    (params, dac)
}

/// Four clusters and both rare populations: the modules pick out exactly the
/// rare cells, their piles keep them together, and rare genes join the
/// selected genes.
#[test]
fn test_dac_rare_modules_small() {
    let n_clusters = 4;
    let n_common = n_clusters * rfx::CELLS_PER_CLUSTER;
    let (store, n_cells) = build_rare_store("small", n_clusters);
    let reader = ParallelSparseReader::new(store.path()).expect("store opens");
    let cells: Vec<usize> = (0..n_cells).collect();
    // 48 of 1664 cells is above the 0.02 the reference uses on 3264 cells
    let (params, dac) = rare_params(400, 0.03);

    let result =
        compute_dac_metacells(&reader, &cells, None, &params, &dac, 1, false).expect("dac runs");
    assert_rare_structure(&result);
    assert_rare_modules(&result, n_common);
    let recovered = rare_recovered(&result, n_common);
    println!("rare recovered {recovered:?} of {:?}", rfx::RARE_SIZES);
    for (r, (&got, &size)) in recovered.iter().zip(&rfx::RARE_SIZES).enumerate() {
        assert!(got + 2 >= size, "population {r}: {got} of {size}");
    }
    for (g, &m) in result.rare_module_of_gene.iter().enumerate() {
        if m >= 0 {
            assert!(
                result.selected_genes.contains(&g),
                "rare gene {g} not selected"
            );
        }
    }
}

/// Rare recovery on the reference dataset over upstream's seeds.
#[test]
// 3264 cells, three divide-and-conquer runs
#[cfg(feature = "large-test")]
fn test_dac_rare_recovery_matches_upstream() {
    let n_common = rfx::N_CLUSTERS * rfx::CELLS_PER_CLUSTER;
    let (store, n_cells) = build_rare_store("band", rfx::N_CLUSTERS);
    let reader = ParallelSparseReader::new(store.path()).expect("store opens");
    let cells: Vec<usize> = (0..n_cells).collect();
    let (params, dac) = rare_params(rfx::PILE_SIZE, 0.02);

    for (seed, upstream) in (1..).zip(rfx::UPSTREAM_RECOVERED) {
        let result = compute_dac_metacells(&reader, &cells, None, &params, &dac, seed, false)
            .expect("dac runs");
        assert_rare_structure(&result);
        assert_rare_modules(&result, n_common);
        let recovered = rare_recovered(&result, n_common);
        println!("seed {seed}: recovered {recovered:?}, upstream {upstream:?}");
        for (r, (&got, &want)) in recovered.iter().zip(upstream).enumerate() {
            // a couple of cells of slack: the rare piles' annealing is seeded
            // differently from upstream's
            assert!(
                got + 2 >= want,
                "seed {seed}, population {r}: {got} vs {want}"
            );
        }
    }
}
