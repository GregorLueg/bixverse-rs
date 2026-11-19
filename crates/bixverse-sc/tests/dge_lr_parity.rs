//! Logistic regression DGE against Seurat's `LRDETest`, fixture written by
//! `dev/atac/gen_atac_lr.R`.

mod dge_lr_fixtures;

use bixverse_core::core::math::stats::TestAlternative;
use bixverse_sc::sc_analysis::dge_lr::{
    calculate_dge_grps_lr, calculate_dge_one_vs_many_lr, calculate_dge_one_vs_rest_lr,
    logistic_lr_test, lr_p_value,
};
use bixverse_sc_data::prelude::*;
use bixverse_sc_data::store::gene_file::write_gene_file;
use bixverse_sc_data::store::writer::CellGeneSparseWriter;
use dge_lr_fixtures::*;

/// Relative tolerance on the p-values with the feature in f64, as R has it.
/// Both sides run the same IRLS iterates to the same stopping rule, so what
/// is left is rounding in the solves (QR in R, Cholesky on standardised
/// columns here) and in the deviance sums. Observed: 3e-13.
const P_TOL_F64: f64 = 1e-10;

/// Relative tolerance through the store, whose norm layer is f32: a relative
/// error of 6e-8 in every `x` moves each statistic by about that relative
/// amount times its size. Observed: 3e-7.
const P_TOL_STORE: f64 = 1e-5;

/// The perfectly separated last feature. R (`glm.fit: algorithm did not
/// converge`) and we both run into the iteration cap on the same iterates, so
/// even its p-value matches (observed: 2e-14 relative), but it is the
/// deviance of an arbitrary stopping point.
const SEPARATED: usize = N_FEAT - 1;

/// Removes the scratch store on drop.
struct TempStore(std::path::PathBuf);

impl Drop for TempStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempStore {
    fn new(name: &str) -> Self {
        Self(
            std::env::temp_dir().join(format!("bixverse_dge_lr_{name}_{}.bin", std::process::id())),
        )
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// The fixture counts, `counts[feature][cell]`, rebuilt from the LCG exactly
/// as the R script draws them.
fn fixture_counts() -> Vec<Vec<u32>> {
    let mut state = LCG_SEED;
    let mut next = || {
        state = (1_664_525 * state + 1_013_904_223) % 4_294_967_296;
        state
    };
    let group_1 = |c: usize| c < N_GRP_1;
    let depth: Vec<u64> = (0..N_CELLS)
        .map(|c| 1 + next() % 4 + u64::from(group_1(c)))
        .collect();

    (0..N_FEAT)
        .map(|f| {
            (0..N_CELLS)
                .map(|c| {
                    let r = next();
                    if f == N_FEAT - 1 {
                        return if group_1(c) { 1 + (r % 3) as u32 } else { 0 };
                    }
                    let up = if group_1(c) && f % 3 == 0 { 8 } else { 0 };
                    let modulus = 5 + (f as u64 % 4) + 2 * depth[c] + up;
                    (r % modulus).saturating_sub(4) as u32
                })
                .collect()
        })
        .collect()
}

fn library_sizes(counts: &[Vec<u32>]) -> Vec<f64> {
    (0..N_CELLS)
        .map(|c| counts.iter().map(|f| f[c] as f64).sum())
        .collect()
}

fn assert_p_close(ours: f64, theirs: f64, tol: f64, feature: usize) {
    let rel = (ours - theirs).abs() / theirs;
    assert!(
        rel <= tol,
        "feature {feature}: ours {ours:e}, Seurat {theirs:e}, relative {rel:e}"
    );
}

/// Write the fixture as a cell store and its gene store.
fn gene_reader(cells: &TempStore, genes: &TempStore, counts: &[Vec<u32>]) -> ParallelSparseReader {
    let mut writer =
        CellGeneSparseWriter::new(cells.path(), true, N_CELLS, N_FEAT, 1e4).expect("writer opens");
    for c in 0..N_CELLS {
        let mut raw = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for (f, feature) in counts.iter().enumerate() {
            if feature[c] > 0 {
                raw.push(feature[c]);
                indices.push(f as u32);
            }
        }
        writer
            .write_cell_chunk(CsrCellChunk::from_data(&raw, &indices, c, 1e4, true))
            .expect("write cell chunk");
    }
    writer.finalise().expect("finalise");
    write_gene_file(cells.path(), genes.path(), None, false).expect("gene file");
    ParallelSparseReader::new(genes.path()).expect("reader opens")
}

/// The single-feature test on the f64 layer R sees.
#[test]
fn test_lr_matches_seurat() {
    let counts = fixture_counts();
    let lib = library_sizes(&counts);
    let y: Vec<bool> = (0..N_CELLS).map(|c| c < N_GRP_1).collect();

    let mut max_rel: f64 = 0.0;
    for (f, feature) in counts.iter().enumerate() {
        let x: Vec<f64> = feature
            .iter()
            .zip(&lib)
            .map(|(&n, &l)| (n as f64 / l * 1e4).ln_1p())
            .collect();
        let test = logistic_lr_test(&y, &x, &lib);
        let p = lr_p_value(&test, &TestAlternative::TwoSided);

        assert_eq!(test.converged, f != SEPARATED, "feature {f}");
        max_rel = max_rel.max((p - P_VALUE[f]).abs() / P_VALUE[f]);
        assert_p_close(p, P_VALUE[f], P_TOL_F64, f);
    }
    println!("max relative p-value difference against Seurat: {max_rel:e}");
}

/// The store path: gene-major reads, f32 norm layer, library sizes as the
/// latent. Two groups through one-vs-rest give the pairwise arm and its
/// mirror.
#[test]
fn test_lr_store_matches_seurat() {
    let counts = fixture_counts();
    let lib = library_sizes(&counts);
    let cells = TempStore::new("parity_cells");
    let genes = TempStore::new("parity_genes");
    let reader = gene_reader(&cells, &genes, &counts);

    let grp_1: Vec<usize> = (0..N_GRP_1).collect();
    let grp_2: Vec<usize> = (N_GRP_1..N_CELLS).collect();
    let res =
        calculate_dge_grps_lr(&reader, &grp_1, &grp_2, &lib, 0.0, "twosided", 0).expect("LR runs");

    assert!(res.genes_to_keep.iter().all(|&k| k));
    let mut max_rel: f64 = 0.0;
    for (f, (&ours, &theirs)) in res.p_vals.iter().zip(P_VALUE).enumerate() {
        max_rel = max_rel.max((ours - theirs).abs() / theirs);
        assert_p_close(ours, theirs, P_TOL_STORE, f);
    }
    println!("store path, max relative p-value difference: {max_rel:e}");

    let rest = calculate_dge_one_vs_rest_lr(
        &reader,
        &[grp_1.clone(), grp_2.clone()],
        &lib,
        0.0,
        "twosided",
        0,
    )
    .expect("one-vs-rest runs");
    assert_eq!(rest.len(), 2);
    assert_eq!(rest[0].p_vals, res.p_vals);
    for f in 0..SEPARATED {
        assert!((rest[1].p_vals[f] - res.p_vals[f]).abs() <= 1e-9 * res.p_vals[f]);
        assert!((rest[1].z_scores[f] + res.z_scores[f]).abs() <= 1e-6);
    }

    // the one rival of a two-group one-vs-many is the pairwise test
    let many =
        calculate_dge_one_vs_many_lr(&reader, &[grp_1, grp_2], &[0], &lib, 0.0, "twosided", 0)
            .expect("one-vs-many runs");
    assert_eq!(many[0].p_vals[0], res.p_vals);
    assert_eq!(many[0].max_p, res.p_vals);
}
