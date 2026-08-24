//! End-to-end DIALOGUE on synthetic data with a planted multicellular
//! programme.
//!
//! Three cell types share a per-sample latent value. Each carries a handful of
//! sample-varying features, one of which tracks that latent, plus genes whose
//! expression follows it. DIALOGUE should recover the shared factor, call the
//! programme significant, and land on the planted genes.

#![cfg(feature = "single-cell")]
// Matches the crate-level allow in lib.rs; the loop index drives the whole
// body here, not just the one subscript clippy notices.
#![allow(clippy::needless_range_loop)]

use bixverse_rs::core::math::vector_helpers::pearson_correlation;
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::mc_analysis::dialogue_mc::dialogue_metacells;
use bixverse_rs::single_cell::sc_analysis::dialogue::{DialogueParams, DialogueResult, PmdParams};
use bixverse_rs::single_cell::sc_data::data_io::CellGeneSparseWriter;
use bixverse_rs::single_cell::sc_data::sc_synthetic_data::{
    DialogueSyntheticData, DialogueSyntheticParams, create_dialogue_synthetic_data,
};

use faer::{Mat, MatRef};

/////////////////////
// Synthetic input //
/////////////////////

/// How many samples the synthetic experiment has.
const N_SAMPLES: usize = 14;
/// Cells per sample per cell type.
const CELLS_PER_SAMPLE: usize = 25;
/// Cell types.
const N_TYPES: usize = 3;
/// Feature columns per cell type.
const N_FEATURES: usize = 8;
/// Feature columns carrying a per-sample component. The rest are pure noise and
/// exist so the ANOVA filter has something to reject.
const N_SAMPLE_FEATURES: usize = 5;
/// Genes in the synthetic store.
const N_GENES: usize = 90;
/// Planted genes per cell type, all tracking the shared latent.
const N_PLANTED: usize = 8;

/// The synthetic shape, sized so the test constants above stay the source of
/// truth for the assertions.
fn shape() -> DialogueSyntheticParams {
    DialogueSyntheticParams::new(
        N_SAMPLES,
        CELLS_PER_SAMPLE,
        N_TYPES,
        N_FEATURES,
        N_SAMPLE_FEATURES,
        N_GENES,
        N_PLANTED,
    )
}

/// Builds the synthetic experiment.
fn build(seed: u64) -> DialogueSyntheticData {
    create_dialogue_synthetic_data(&shape(), seed).expect("the fixture shape is buildable")
}

/// Parameters sized for the synthetic data: a small permutation null, since the
/// point is the pipeline rather than the tail of the p-value.
fn params(n_permutations: usize) -> DialogueParams {
    DialogueParams {
        pmd: PmdParams {
            k: 2,
            n_permutations,
            abn_c: 10,
            n_genes: 12,
            seed: 7,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Runs the pipeline over the synthetic store.
fn run(data: &DialogueSyntheticData, n_permutations: usize) -> DialogueResult {
    let feature_refs: Vec<MatRef<f64>> = data.features.iter().map(|m| m.as_ref()).collect();
    let genes: Vec<usize> = (0..N_GENES).collect();
    dialogue_metacells(
        &data.matrix,
        &data.cell_type_indices,
        &feature_refs,
        &data.sample_ids,
        &data.quality,
        &genes,
        &params(n_permutations),
        0,
    )
    .expect("DIALOGUE should run on well-formed synthetic data")
}

/// Weakest agreement with the planted latent across cell types, for one
/// programme. A programme only counts as recovered if *every* cell type tracks
/// the latent, so the minimum is the right summary.
fn worst_latent_agreement(
    result: &DialogueResult,
    data: &DialogueSyntheticData,
    programme: usize,
) -> f64 {
    (0..N_TYPES)
        .map(|t| {
            let means = sample_means(
                &result.cca_scores[t],
                &data.cell_type_indices[t],
                &data.sample_ids,
                programme,
            );
            pearson_correlation(&means, &data.latent)
                .unwrap_or(0.0)
                .abs()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Which programme actually tracks the planted latent, by ground truth.
fn planted_programme(result: &DialogueResult, data: &DialogueSyntheticData) -> usize {
    if worst_latent_agreement(result, data, 0) >= worst_latent_agreement(result, data, 1) {
        0
    } else {
        1
    }
}

/// Sample-averages a cell type's scores for one programme.
fn sample_means(
    scores: &Mat<f64>,
    cells: &[usize],
    sample_ids: &[usize],
    programme: usize,
) -> Vec<f64> {
    let mut sums = [0.0_f64; N_SAMPLES];
    let mut counts = [0.0_f64; N_SAMPLES];
    for (row, &cell) in cells.iter().enumerate() {
        let s = sample_ids[cell];
        sums[s] += scores[(row, programme)];
        counts[s] += 1.0;
    }
    sums.iter()
        .zip(counts.iter())
        .map(|(s, c)| if *c > 0.0 { s / c } else { 0.0 })
        .collect()
}

///////////
// Tests //
///////////

/// The decomposition recovers the planted latent.
///
/// For at least one programme, every cell type's sample-averaged score tracks
/// the latent the data was built around. That is the property the whole method
/// rests on, and it does not depend on any threshold downstream.
#[test]
fn test_dialogue_recovers_the_planted_latent() {
    let data = build(11);
    let result = run(&data, 20);

    let best = (0..2)
        .map(|programme| worst_latent_agreement(&result, &data, programme))
        .fold(0.0_f64, f64::max);
    assert!(
        best > 0.7,
        "no programme tracked the latent in every cell type; best worst-case |r| was {best:.3}"
    );
}

/// The programme that tracks the latent is called significant, and it spans
/// every cell type.
#[test]
fn test_dialogue_calls_the_planted_programme_significant() {
    let data = build(11);
    let result = run(&data, 20);

    // Identify the programme by the ground truth, then check what DIALOGUE
    // said about it.
    let planted_programme = planted_programme(&result, &data);

    let n_pairs = N_TYPES * (N_TYPES - 1) / 2;
    for pair in 0..n_pairs {
        let p = result.emp_p[(planted_programme, pair)];
        assert!(
            p < 0.1,
            "pair {pair} of the planted programme had an empirical p of {p:.4}"
        );
    }
    assert_eq!(
        result.mcp_cell_types[planted_programme].len(),
        N_TYPES,
        "the planted programme should span every cell type"
    );
}

/// The refined signatures land on the planted genes rather than the noise.
#[test]
fn test_dialogue_signatures_favour_the_planted_genes() {
    let data = build(11);
    let result = run(&data, 20);

    let programme = planted_programme(&result, &data);

    // The planted programme must produce a signature in every cell type, and
    // each must be dominated by that cell type's planted genes. Chance would be
    // N_PLANTED / N_GENES, under a tenth. Asserting per slot rather than "at
    // least one slot somewhere" is the difference between this test noticing
    // two of three signatures turning to noise and not noticing.
    for t in 0..N_TYPES {
        let sig = &result.permissive[t][programme];
        let all: Vec<usize> = sig.up.iter().chain(sig.down.iter()).copied().collect();
        assert!(
            !all.is_empty(),
            "cell type {t} produced no signature for the planted programme"
        );
        let hits = all.iter().filter(|g| data.planted[t].contains(g)).count();
        let rate = hits as f64 / all.len() as f64;
        assert!(
            rate > 0.8,
            "cell type {t} signature was only {rate:.2} planted ({hits}/{})",
            all.len()
        );
    }

    // Any signature the other programme produced must not be pure noise:
    // nothing else in the data is shared across cell types.
    let other = 1 - programme;
    for t in 0..N_TYPES {
        let sig = &result.permissive[t][other];
        let all: Vec<usize> = sig.up.iter().chain(sig.down.iter()).copied().collect();
        if all.is_empty() {
            continue;
        }
        let hits = all.iter().filter(|g| data.planted[t].contains(g)).count();
        assert!(
            hits > 0,
            "cell type {t} programme {other} returned {} genes, none of them planted",
            all.len()
        );
    }
}

/// The *final* scores, not just the canonical ones, must track the planted
/// latent.
///
/// This is the headline output of the pipeline and it was previously
/// constrained by nothing but shape and finiteness. Reversing the projection
/// vector in stage three, so every cell received another cell's score, left all
/// seven tests passing.
#[test]
fn test_dialogue_final_scores_track_the_planted_latent() {
    let data = build(11);
    let result = run(&data, 20);
    let programme = planted_programme(&result, &data);

    for t in 0..N_TYPES {
        let means = sample_means(
            &result.scores[t],
            &data.cell_type_indices[t],
            &data.sample_ids,
            programme,
        );
        let r = pearson_correlation(&means, &data.latent)
            .unwrap_or(0.0)
            .abs();
        assert!(
            r > 0.7,
            "cell type {t} final scores tracked the latent at only |r| = {r:.3}"
        );
    }
}

/// Stage three's refit stays anchored to the programme stage one found.
///
/// `refit_fidelity` is a correlation, so `|r| <= 1` is an identity and
/// asserting it tests nothing. What matters is that it is *high* for the
/// planted programme.
#[test]
fn test_dialogue_refit_tracks_the_canonical_score() {
    let data = build(11);
    let result = run(&data, 20);
    let programme = planted_programme(&result, &data);

    for t in 0..N_TYPES {
        let fidelity = result.refit_fidelity[(t, programme)];
        assert!(
            fidelity.abs() > 0.5,
            "cell type {t} refit drifted from the canonical score: {fidelity:.3}"
        );
    }
    for t in 0..N_TYPES {
        assert_eq!(result.scores[t].nrows(), data.cell_type_indices[t].len());
        assert_eq!(result.scores[t].ncols(), 2);
        for i in 0..result.scores[t].nrows() {
            for j in 0..2 {
                assert!(result.scores[t][(i, j)].is_finite());
            }
        }
    }
}

/// Stage two must actually fit something.
///
/// The determinism and equivalence tests compare verdict *counts*, which pass
/// at `0 == 0`. Nothing else asserted the association table was non-empty, so a
/// regression silencing stage two entirely would have slipped through.
#[test]
fn test_dialogue_produces_associations_and_verdicts() {
    let data = build(11);
    let result = run(&data, 20);

    assert!(
        !result.verdicts.is_empty(),
        "stage two produced no gene associations at all"
    );
    let programme = planted_programme(&result, &data);
    assert!(
        result.verdicts.iter().any(|v| v.programme == programme),
        "no verdict for the planted programme"
    );
    assert!(
        result.verdicts.iter().any(|v| v.coefficient > 0.0),
        "the staged refit dropped every gene"
    );
}

/// Same input, same output.
#[test]
fn test_dialogue_is_deterministic() {
    let data = build(11);
    let a = run(&data, 12);
    let b = run(&data, 12);

    for t in 0..N_TYPES {
        for i in 0..a.scores[t].nrows() {
            for j in 0..a.scores[t].ncols() {
                assert_eq!(a.scores[t][(i, j)], b.scores[t][(i, j)]);
            }
        }
    }
    for programme in 0..2 {
        for pair in 0..(N_TYPES * (N_TYPES - 1) / 2) {
            assert_eq!(a.emp_p[(programme, pair)], b.emp_p[(programme, pair)]);
        }
    }
    assert_eq!(a.verdicts.len(), b.verdicts.len());
}

/// A scratch store that cleans up after itself.
struct TempStore(std::path::PathBuf);

impl Drop for TempStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempStore {
    /// Reserves a uniquely named store in the system temp directory.
    ///
    /// The process id is part of the name: two `cargo test` invocations on one
    /// machine, which the CI feature matrix does routinely, would otherwise
    /// collide on the same file and read a half-written store.
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!(
            "bixverse_dialogue_{name}_{}.bin",
            std::process::id()
        )))
    }

    /// Path as a `&str`.
    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// Writes the synthetic matrix out as a gene-major bixverse store.
///
/// The in-memory reader narrows `data_2` to `f16` when it builds a chunk, and
/// so does the writer, so both paths see bit-identical values and the
/// comparison below can be exact rather than approximate.
fn write_store(path: &str, data: &DialogueSyntheticData) {
    let n_cells = data.matrix.shape.0;
    let mut writer =
        CellGeneSparseWriter::new(path, false, n_cells, N_GENES, 1e4).expect("writer opens");

    let norm = data.matrix.data_2.as_ref().expect("normalised layer");
    for gene in 0..N_GENES {
        let lo = data.matrix.indptr[gene] as usize;
        let hi = data.matrix.indptr[gene + 1] as usize;
        let raw: Vec<u32> = data.matrix.data[lo..hi].to_vec();
        let indices: Vec<usize> = data.matrix.indices[lo..hi]
            .iter()
            .map(|v| *v as usize)
            .collect();
        let norms: Vec<F16> = norm[lo..hi].iter().map(|v| F16::from_f32(*v)).collect();
        writer
            .write_gene_chunk(CscGeneChunk::from_conversion(
                RawCounts::from_u32_auto(&raw),
                &norms,
                &indices,
                gene,
                true,
            ))
            .expect("write gene chunk");
    }
    writer.finalise().expect("finalise");
}

/// The metacell shim and the streaming path must agree exactly.
///
/// This is what keeps the shim from drifting into a second implementation. Both
/// go through the same core; if this ever fails, one of them has grown a
/// special case.
#[test]
fn test_dialogue_in_memory_matches_the_streamed_store() {
    let data = build(11);
    let store = TempStore::new("equivalence");
    write_store(store.path(), &data);

    let feature_refs: Vec<MatRef<f64>> = data.features.iter().map(|m| m.as_ref()).collect();
    let genes: Vec<usize> = (0..N_GENES).collect();
    let settings = params(12);

    let in_memory = run(&data, 12);

    let reader = ParallelSparseReader::new(store.path()).expect("store opens");
    let streamed = bixverse_rs::single_cell::sc_analysis::dialogue::dialogue_run(
        &reader,
        &data.cell_type_indices,
        &feature_refs,
        &data.sample_ids,
        &data.quality,
        &genes,
        &settings,
        0,
    )
    .expect("DIALOGUE runs off the store");

    for programme in 0..2 {
        for pair in 0..(N_TYPES * (N_TYPES - 1) / 2) {
            assert_eq!(
                in_memory.emp_p[(programme, pair)],
                streamed.emp_p[(programme, pair)],
                "empirical p diverged at programme {programme}, pair {pair}"
            );
        }
    }
    assert_eq!(in_memory.mcp_cell_types, streamed.mcp_cell_types);
    assert_eq!(in_memory.verdicts.len(), streamed.verdicts.len());

    for t in 0..N_TYPES {
        assert_eq!(in_memory.permissive[t].len(), streamed.permissive[t].len());
        for programme in 0..2 {
            assert_eq!(
                in_memory.permissive[t][programme].up,
                streamed.permissive[t][programme].up
            );
            assert_eq!(
                in_memory.permissive[t][programme].down,
                streamed.permissive[t][programme].down
            );
            assert_eq!(
                in_memory.strict[t][programme].up,
                streamed.strict[t][programme].up
            );
        }
        for i in 0..in_memory.scores[t].nrows() {
            for j in 0..2 {
                assert_eq!(
                    in_memory.scores[t][(i, j)],
                    streamed.scores[t][(i, j)],
                    "score diverged at cell type {t}, cell {i}, programme {j}"
                );
            }
        }
    }
}

/// Malformed input is rejected rather than half-processed.
#[test]
fn test_dialogue_rejects_malformed_input() {
    let data = build(11);
    let feature_refs: Vec<MatRef<f64>> = data.features.iter().map(|m| m.as_ref()).collect();
    let genes: Vec<usize> = (0..N_GENES).collect();

    // One cell type is not enough to correlate anything against.
    let single: Vec<Vec<usize>> = vec![data.cell_type_indices[0].clone()];
    let err = dialogue_metacells(
        &data.matrix,
        &single,
        &feature_refs[..1],
        &data.sample_ids,
        &data.quality,
        &genes,
        &params(5),
        0,
    )
    .unwrap_err();
    assert!(
        matches!(err, BixverseErrors::DialogueTooFewCellTypes { .. }),
        "expected the cell-type check to fire, got {err}"
    );
    // Feature rows that do not line up with the cell list.
    let short = Mat::<f64>::zeros(3, N_FEATURES);
    let mismatched: Vec<MatRef<f64>> = vec![short.as_ref(), feature_refs[1], feature_refs[2]];
    assert!(
        dialogue_metacells(
            &data.matrix,
            &data.cell_type_indices,
            &mismatched,
            &data.sample_ids,
            &data.quality,
            &genes,
            &params(5),
            0,
        )
        .is_err()
    );

    // A cell index past the end of the store must error rather than panic on
    // the dense position table: this crosses an FFI boundary.
    let mut out_of_range = data.cell_type_indices.clone();
    let n_cells = data.matrix.shape.0;
    out_of_range[0].push(n_cells + 5);
    let mut padded_sample_ids = data.sample_ids.clone();
    let mut padded_quality = data.quality.clone();
    padded_sample_ids.resize(n_cells + 10, 0);
    padded_quality.resize(n_cells + 10, 0.0);
    let wide_features: Vec<Mat<f64>> = data
        .features
        .iter()
        .enumerate()
        .map(|(t, m)| {
            let extra = usize::from(t == 0);
            Mat::<f64>::from_fn(m.nrows() + extra, m.ncols(), |i, j| {
                if i < m.nrows() { m[(i, j)] } else { 0.0 }
            })
        })
        .collect();
    let wide_refs: Vec<MatRef<f64>> = wide_features.iter().map(|m| m.as_ref()).collect();
    assert!(
        dialogue_metacells(
            &data.matrix,
            &out_of_range,
            &wide_refs,
            &padded_sample_ids,
            &padded_quality,
            &genes,
            &params(5),
            0,
        )
        .is_err()
    );
}
