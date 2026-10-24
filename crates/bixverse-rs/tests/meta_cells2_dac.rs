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

use std::path::PathBuf;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::mc_generation::metacells2::{
    DacMetacellsResult, DacParams, MC2KnnParams, MetacellsParams, compute_dac_metacells,
};
use bixverse_rs::single_cell::sc_data::data_io::CellGeneSparseWriter;
use meta_cells2_dac_fixtures as fx;

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
/// and the same seed reproduces the run.
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
    let b =
        compute_dac_metacells(&reader, &cells, None, &params, &dac, 3, false).expect("dac runs");
    assert_structure(&a);
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
