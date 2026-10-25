//! Parity tests against metacells 0.9.5, fixtures in [`super::parity_fixtures`].
//!
//! Every input is rebuilt from the same LCG as `dev/gen_metacells2_fixtures.py`.
//! Deterministic stages are compared exactly or to float tolerance; stochastic
//! ones (downsampled counts, seeds, the annealing) are not covered here.

use faer::Mat;

use crate::prelude::*;

use bixverse_core::core::math::sparse::transpose_sparse;

use super::candidates::min_cut;
use super::deviants::find_deviant_cells;
use super::dissolve::dissolve_metacells;
use super::downsample::compute_downsample_target;
use super::knn::build_knn_graph;
use super::params::{
    DeviantsParams, DissolveParams, MC2KnnParams, RareParams, SelectParams, SimilarityParams,
};
use super::parity_fixtures as fx;
use super::parity_rare_fixtures as rfx;
use super::pile::Pile;
use super::rare::find_rare_gene_modules;
use super::seeds::seeds_count_for;
use super::select::{gene_stats, relative_variance, select_features};
use super::similarity::compute_similarity;

/////////////
// Helpers //
/////////////

/// Numerical Recipes LCG, mirrored from the generator.
struct Lcg(u64);

impl Lcg {
    /// Next 32-bit state.
    fn next(&mut self) -> u64 {
        self.0 = (1_664_525 * self.0 + 1_013_904_223) % (1 << 32);
        self.0
    }
}

/// The generator's `make_counts`, as CSR.
fn make_counts() -> CompressedSparseData2<u32, f32> {
    let mut rng = Lcg(fx::LCG_SEED);
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0usize];
    for i in 0..fx::N_CELLS {
        let c = i / fx::CELLS_PER_CLUSTER;
        for g in 0..fx::N_GENES {
            let x = rng.next();
            let hi = x >> 16;
            let mid = (x >> 8) & 0xFF;
            let mut v = if (c * fx::MARKERS..(c + 1) * fx::MARKERS).contains(&g) {
                if !mid.is_multiple_of(4) {
                    hi % 30 + 5
                } else {
                    0
                }
            } else if (fx::HOUSEKEEPING.0..fx::HOUSEKEEPING.1).contains(&g) {
                hi % 50 + 20
            } else if mid.is_multiple_of(3) {
                hi % 4
            } else {
                0
            } as u32;
            if g == fx::DEVIANT_GENE && fx::DEVIANT_CELLS.contains(&i) {
                v += fx::DEVIANT_SPIKE;
            }
            if v > 0 {
                data.push(v);
                indices.push(g);
            }
        }
        indptr.push(data.len());
    }
    CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (fx::N_CELLS, fx::N_GENES),
    }
}

/// A pile whose "downsampled" layer is the raw counts, as the generator's
/// no-op downsampling.
fn undownsampled_pile() -> Pile {
    let raw = make_counts();
    let mut pile = Pile::from_raw(raw, (0..fx::N_CELLS).collect(), None).unwrap();
    pile.downsampled = Some(pile.raw.clone());
    pile
}

/// The generator's `make_similarity`.
fn make_similarity(n: usize, n_hubs: usize, seed: u64) -> Mat<f32> {
    let mut rng = Lcg(seed);
    let mut s = Mat::<f32>::zeros(n, n);
    for i in 0..n {
        for j in 0..i {
            let mut v = (rng.next() >> 8) as f32 / 16_777_216.0;
            if i < n_hubs || j < n_hubs {
                v += 1.0;
            }
            s[(i, j)] = v;
            s[(j, i)] = v;
        }
        s[(i, i)] = 2.0;
    }
    s
}

/// The generator's `make_cut_graph`, lower triangle only, so that the
/// symmetrised graph `W + W^T` equals the generator's matrix.
fn make_cut_graph(
    n: usize,
    seed: u64,
    split: usize,
    cross_mod: u64,
) -> CompressedSparseData2<f32, f32> {
    let mut rng = Lcg(seed);
    let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..i {
            let x = rng.next();
            let same = (i < split) == (j < split);
            let v = if same {
                ((x >> 16) % 100 + 1) as f32 / 128.0
            } else if (x >> 24).is_multiple_of(cross_mod) {
                ((x >> 16) % 10 + 1) as f32 / 1024.0
            } else {
                0.0
            };
            if v > 0.0 {
                rows[i].push((j, v));
            }
        }
    }
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0usize];
    for row in rows {
        for (j, v) in row {
            indices.push(j);
            data.push(v);
        }
        indptr.push(data.len());
    }
    CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (n, n),
    }
}

/// Selection parameters for the parity runs.
fn select_params(min_rel_var: f32, min_genes: usize) -> SelectParams {
    SelectParams {
        min_gene_relative_variance: Some(min_rel_var),
        min_genes,
        ..SelectParams::default()
    }
}

/// The candidates the generator used for deviants and dissolve.
fn fixture_candidates() -> Vec<i32> {
    (0..fx::N_CELLS)
        .map(|i| if i >= 230 { 8 } else { (i / 30) as i32 })
        .collect()
}

/// Relative closeness.
fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

///////////
// Tests //
///////////

#[test]
fn test_parity_downsample_samples() {
    let pile = undownsampled_pile();
    let defaults = SelectParams::default();
    let samples = compute_downsample_target(
        &pile.umis_per_cell,
        defaults.downsample_min_samples,
        defaults.downsample_min_cell_quantile,
        defaults.downsample_max_cell_quantile,
    );
    assert_eq!(samples as u64, fx::DOWNSAMPLE_SAMPLES);
}

#[test]
fn test_parity_gene_statistics() {
    let pile = undownsampled_pile();
    let csc = transpose_sparse(&pile.raw);
    let stats = gene_stats(&csc);
    for g in 0..fx::N_GENES {
        assert_eq!(stats.top3[g] as f32, fx::TOP3[g], "top3 of gene {g}");
    }
    // upstream's own order: ties between equal means are numpy's to break
    let rv = relative_variance(&stats.log_norm_var, fx::MEAN_ORDER, 100);
    let off: Vec<(usize, f64, f64)> = (0..fx::N_GENES)
        .filter(|&g| !close(rv[g], fx::RELATIVE_VARIANCE[g], 1e-5))
        .map(|g| (g, rv[g], fx::RELATIVE_VARIANCE[g]))
        .collect();
    assert!(
        off.is_empty(),
        "{} genes off: {:?}",
        off.len(),
        &off[..off.len().min(10)]
    );
}

#[test]
fn test_parity_select_default() {
    let mut pile = undownsampled_pile();
    select_features(&mut pile, &select_params(0.1, 100)).unwrap();
    assert_eq!(pile.selected_gene_indices.unwrap(), fx::SELECTED_DEFAULT);
}

#[test]
fn test_parity_select_relaxed() {
    let mut pile = undownsampled_pile();
    select_features(&mut pile, &select_params(1.5, 120)).unwrap();
    assert_eq!(pile.selected_gene_indices.unwrap(), fx::SELECTED_RELAXED);
}

#[test]
fn test_parity_similarity() {
    let mut pile = undownsampled_pile();
    select_features(&mut pile, &select_params(0.1, 100)).unwrap();
    let sim = compute_similarity(&pile, &SimilarityParams::default()).unwrap();
    for i in 0..fx::N_CELLS {
        let row_sum: f64 = (0..fx::N_CELLS).map(|j| sim[(i, j)] as f64).sum();
        assert!(
            close(row_sum, fx::SIMILARITY_ROW_SUMS[i], 1e-4),
            "row sum {i}: {row_sum} vs {}",
            fx::SIMILARITY_ROW_SUMS[i]
        );
    }
    for (r, expected) in [
        (0, fx::SIMILARITY_ROW_0),
        (61, fx::SIMILARITY_ROW_61),
        (239, fx::SIMILARITY_ROW_239),
    ] {
        for j in 0..fx::N_CELLS {
            let got = sim[(r, j)];
            assert!(
                (got - expected[j]).abs() < 1e-4,
                "similarity ({r}, {j}): {got} vs {}",
                expected[j]
            );
        }
    }
}

/// Compare one kNN case.
///
/// Upstream's `f32` offset quantises the stored ranks, so equal values are
/// common and its `nth_element` keeps an arbitrary one at a prune cut. A tie
/// broken differently at a column prune swaps an entry of another row, which
/// rescales that row's weights. So: row degrees must match exactly, and the
/// edges that disagree (absent on one side, or with another weight) must stay
/// rare. Everything else matches to float tolerance.
fn assert_knn_case(
    n: usize,
    hubs: usize,
    seed: u64,
    k: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
) {
    let sim = make_similarity(n, hubs, seed);
    let g = build_knn_graph(&sim, k, &MC2KnnParams::default());
    let got_indptr: Vec<usize> = g.indptr.iter().map(|&v| v as usize).collect();
    let got_indices: Vec<usize> = g.indices.iter().map(|&v| v as usize).collect();
    assert_eq!(got_indptr, indptr, "indptr");

    let same = |a: f32, b: f32| (a - b).abs() <= 1e-6 * b.abs().max(1e-3);
    let mut disagree = 0usize;
    for row in 0..n {
        let r = indptr[row]..indptr[row + 1];
        for idx in r.clone() {
            let c = got_indices[idx];
            let matched = indices[r.clone()]
                .iter()
                .position(|&uc| uc == c)
                .is_some_and(|p| same(g.data[idx], data[indptr[row] + p]));
            if !matched {
                disagree += 1;
            }
        }
    }
    assert!(
        disagree * 50 <= indices.len(),
        "{disagree} of {} edges disagree with upstream",
        indices.len()
    );
}

#[test]
fn test_parity_knn_plain() {
    assert_knn_case(
        fx::KNN_PLAIN_N,
        fx::KNN_PLAIN_HUBS,
        fx::KNN_PLAIN_SEED,
        fx::KNN_PLAIN_K,
        fx::KNN_PLAIN_INDPTR,
        fx::KNN_PLAIN_INDICES,
        fx::KNN_PLAIN_DATA,
    );
}

#[test]
fn test_parity_knn_hubs_retry() {
    assert_knn_case(
        fx::KNN_HUBS_N,
        fx::KNN_HUBS_HUBS,
        fx::KNN_HUBS_SEED,
        fx::KNN_HUBS_K,
        fx::KNN_HUBS_INDPTR,
        fx::KNN_HUBS_INDICES,
        fx::KNN_HUBS_DATA,
    );
}

#[test]
fn test_parity_min_cut() {
    for &(n, split, seed, cross_mod, small, strength) in fx::MIN_CUT_CASES {
        let graph = make_cut_graph(n, seed, split, cross_mod);
        let cells: Vec<usize> = (0..n).collect();
        let cut = min_cut(&graph, &cells, &mut vec![-1; n]);
        let got_small = cut.first.len().min(cut.second.len());
        assert_eq!(got_small, small, "small side of case {seed}");
        match cut.strength {
            None => assert!(strength.is_nan(), "case {seed}: expected {strength}"),
            Some(s) => assert!(close(s, strength, 1e-9), "case {seed}: {s} vs {strength}"),
        }
    }
}

#[test]
fn test_parity_seeds_count() {
    for &(size, umis, min_size, max_size, min_umis, max_umis, expected) in fx::SEEDS_COUNT_CASES {
        let got = seeds_count_for(size, umis, min_size, max_size, min_umis, max_umis);
        assert_eq!(got, expected, "seeds count for size {size}, umis {umis}");
    }
}

#[test]
fn test_parity_deviants() {
    let raw = make_counts();
    let pile = Pile::from_raw(raw, (0..fx::N_CELLS).collect(), None).unwrap();
    let deviants = find_deviant_cells(
        &pile.raw,
        &pile.umis_per_cell,
        &fixture_candidates(),
        &DeviantsParams::default(),
    )
    .unwrap();
    let got: Vec<usize> = (0..fx::N_CELLS).filter(|&i| deviants[i]).collect();
    assert_eq!(got, fx::DEVIANTS);
}

#[test]
fn test_parity_dissolve() {
    let raw = make_counts();
    let pile = Pile::from_raw(raw, (0..fx::N_CELLS).collect(), None).unwrap();
    let mut deviants = vec![false; fx::N_CELLS];
    for &i in fx::DEVIANTS {
        deviants[i] = true;
    }
    let candidates = fixture_candidates();
    let run = |target_size: usize, target_umis: f64| {
        dissolve_metacells(
            &pile.raw,
            &pile.umis_per_cell,
            &pile.mass_per_cell,
            &candidates,
            &deviants,
            target_size,
            target_umis,
            12,
            &DissolveParams::default(),
        )
    };

    let (metacells, dissolved) = run(48, 160_000.0);
    assert_eq!(metacells, fx::DISSOLVE_METACELL);
    let got: Vec<usize> = (0..fx::N_CELLS).filter(|&i| dissolved[i]).collect();
    assert_eq!(got, fx::DISSOLVED);

    let (metacells, _) = run(60, 200_000.0);
    assert_eq!(metacells, fx::DISSOLVE_METACELL_BIG_TARGET);
}

///////////////////////
// Rare gene modules //
///////////////////////

/// One planted case of `dev/gen_metacells2_rare_fixtures.py`.
struct RareCase {
    /// LCG seed.
    seed: u64,
    /// Cells.
    n_cells: usize,
    /// Leading background genes.
    n_background: usize,
    /// `(gene start, gene end, cell start, cell end)` per block.
    blocks: &'static [(usize, usize, usize, usize)],
    /// `(gene, cell start, cell end)` per related gene.
    related: &'static [(usize, usize, usize)],
    /// Sporadic gene range; its end is the gene count.
    sporadic: (usize, usize),
    /// LCG bits that must be zero for a sporadic gene to be on.
    sporadic_mask: u64,
}

/// The generator's `make_counts` for a rare case, as CSR.
fn make_rare_counts(case: &RareCase) -> CompressedSparseData2<u32, f32> {
    let n_genes = case.sporadic.1;
    let mut rng = Lcg(case.seed);
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0usize];
    for i in 0..case.n_cells {
        for g in 0..n_genes {
            let x = rng.next();
            let hi = x >> 16;
            let mid = (x >> 8) & 0xFF;
            let block = case.blocks.iter().find(|b| (b.0..b.1).contains(&g));
            let related = case.related.iter().find(|r| r.0 == g);
            let v = if g < case.n_background {
                if mid.is_multiple_of(4) { hi % 3 + 1 } else { 0 }
            } else if let Some(&(_, _, c0, c1)) = block {
                if (c0..c1).contains(&i) && !mid.is_multiple_of(5) {
                    hi % 9
                } else {
                    0
                }
            } else if let Some(&(_, c0, c1)) = related {
                if (c0..c1).contains(&i) {
                    if mid.is_multiple_of(2) {
                        hi % 10 + 1
                    } else {
                        0
                    }
                } else {
                    u64::from(mid.is_multiple_of(64))
                }
            } else if (x >> 8) & case.sporadic_mask == 0 {
                hi % 12
            } else {
                0
            } as u32;
            if v > 0 {
                data.push(v);
                indices.push(g);
            }
        }
        indptr.push(data.len());
    }
    CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (case.n_cells, n_genes),
    }
}

/// Run the port on a rare case and compare with the reference exactly,
/// labels included.
///
/// ### Params
///
/// * `case` - The planted case.
/// * `params` - Rare module parameters, as the generator's kwargs.
/// * `target_pile_size` - As the generator's kwargs.
/// * `gene_module` - Reference module per gene.
/// * `rare_cells` - Reference cells in a module.
/// * `rare_cell_module` - Their modules.
fn assert_rare_case(
    case: &RareCase,
    params: &RareParams,
    target_pile_size: usize,
    gene_module: &[i32],
    rare_cells: &[usize],
    rare_cell_module: &[i32],
) {
    let counts = make_rare_counts(case);
    let n_genes = counts.shape.1;
    let got = find_rare_gene_modules(
        |cells| Ok(counts.slice_rows(cells)?),
        case.n_cells,
        n_genes,
        None,
        48,
        target_pile_size,
        params,
        false,
    )
    .unwrap();
    assert_eq!(got.module_of_gene, gene_module);
    let cells: Vec<usize> = (0..case.n_cells)
        .filter(|&i| got.module_of_cell[i] >= 0)
        .collect();
    assert_eq!(cells, rare_cells);
    let modules: Vec<i32> = cells.iter().map(|&i| got.module_of_cell[i]).collect();
    assert_eq!(modules, rare_cell_module);
    assert_eq!(
        got.rare_gene_mask,
        gene_module.iter().map(|&m| m >= 0).collect::<Vec<_>>()
    );
}

/// Overlapping modules (strongest wins), a related gene, modules dropped for
/// too few genes or cells, at a raised `max_gene_cell_fraction`.
#[test]
fn test_parity_rare_planted() {
    let case = RareCase {
        seed: rfx::PLANTED_SEED,
        n_cells: rfx::PLANTED_N_CELLS,
        n_background: rfx::PLANTED_N_BACKGROUND,
        blocks: rfx::PLANTED_BLOCKS,
        related: rfx::PLANTED_RELATED,
        sporadic: rfx::PLANTED_SPORADIC,
        sporadic_mask: rfx::PLANTED_SPORADIC_MASK,
    };
    let params = RareParams {
        max_gene_cell_fraction: 0.02,
        ..Default::default()
    };
    assert_rare_case(
        &case,
        &params,
        400,
        rfx::PLANTED_GENE_MODULE,
        rfx::PLANTED_RARE_CELLS,
        rfx::PLANTED_RARE_CELL_MODULE,
    );
}

/// Upstream defaults throughout, 20000 cells.
#[test]
fn test_parity_rare_defaults() {
    let case = RareCase {
        seed: rfx::DEFAULT_SEED,
        n_cells: rfx::DEFAULT_N_CELLS,
        n_background: rfx::DEFAULT_N_BACKGROUND,
        blocks: rfx::DEFAULT_BLOCKS,
        related: rfx::DEFAULT_RELATED,
        sporadic: rfx::DEFAULT_SPORADIC,
        sporadic_mask: rfx::DEFAULT_SPORADIC_MASK,
    };
    assert_rare_case(
        &case,
        &RareParams::default(),
        8000,
        rfx::DEFAULT_GENE_MODULE,
        rfx::DEFAULT_RARE_CELLS,
        rfx::DEFAULT_RARE_CELL_MODULE,
    );
}
