//! Gates the pseudobulk join, not edgeR.
//!
//! `tests/edger_bulk.rs` already checks the quasi-likelihood chain against
//! edgeR 4.8.2. What is left to pin here is that the cells are summed into the
//! samples the caller asked for, in the orientation the chain wants, and that
//! the design rows line up with the columns that come out. Getting the
//! transpose wrong still fits a model, it just fits the wrong one.

#![cfg(feature = "single-cell")]

use bixverse_rs::methods::dge_bulk::{EdgeRQlParams, run_edger_ql};
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::pseudobulk_dge::pseudobulk_dge;
use bixverse_rs::single_cell::sc_data::in_memory_io::InMemorySparseReader;

use edge_rs::glm::test::Tested;

use rand::prelude::*;

////////////////
// Dimensions //
////////////////

/// Genes in the fixture.
const N_GENES: usize = 120;
/// Pseudobulk samples, six per group.
const N_SAMPLES: usize = 12;
/// Cells per sample.
const CELLS_PER_SAMPLE: usize = 40;
/// Total cells.
const N_CELLS: usize = N_SAMPLES * CELLS_PER_SAMPLE;
/// Intercept and group.
const N_COEF: usize = 2;

/////////////
// Fixture //
/////////////

/// A synthetic experiment plus the aggregate computed by hand.
struct Fixture {
    /// Counts as CSC over `(cells, genes)`.
    matrix: CompressedSparseData2<u32, f32>,
    /// Cell indices per pseudobulk sample.
    sample_cells: Vec<Vec<usize>>,
    /// Hand-summed counts, gene-major and row-major.
    aggregate: Vec<f64>,
    /// Design, row-major `N_SAMPLES * N_COEF`.
    design: Vec<f64>,
}

/// Builds the fixture.
///
/// Cells are handed to samples out of order, so a join that assumed the cells
/// were already grouped would sum the wrong ones.
///
/// ### Params
///
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// The [Fixture].
fn build(seed: u64) -> Fixture {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut order: Vec<usize> = (0..N_CELLS).collect();
    order.shuffle(&mut rng);
    let sample_cells: Vec<Vec<usize>> = order
        .chunks(CELLS_PER_SAMPLE)
        .map(|chunk| chunk.to_vec())
        .collect();

    let sample_of: Vec<usize> = {
        let mut out = vec![0; N_CELLS];
        for (sample, cells) in sample_cells.iter().enumerate() {
            for &cell in cells {
                out[cell] = sample;
            }
        }
        out
    };

    let mut dense = vec![0.0_f64; N_GENES * N_CELLS];
    for gene in 0..N_GENES {
        // Every twentieth gene is higher in the second group, the rest is
        // noise. Six of a hundred and twenty, deliberately: plant the effect in
        // a fifth of the panel instead and the extra counts move the library
        // sizes enough that TMM cannot fully undo it, at which point unaffected
        // genes come out looking down in the second group. That is real
        // compositional bias rather than a bug, but it is not what this test is
        // for.
        let base = 2 + (gene % 9);
        for cell in 0..N_CELLS {
            let group = sample_of[cell] / (N_SAMPLES / 2);
            let ceiling = if gene % 20 == 0 && group == 1 {
                base * 5
            } else {
                base
            };
            dense[gene * N_CELLS + cell] = rng.random_range(0..ceiling) as f64;
        }
    }

    let mut data: Vec<u32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut indptr: Vec<u32> = vec![0];
    for gene in 0..N_GENES {
        for cell in 0..N_CELLS {
            let count = dense[gene * N_CELLS + cell];
            if count > 0.0 {
                data.push(count as u32);
                indices.push(cell as u32);
            }
        }
        indptr.push(data.len() as u32);
    }

    let library: Vec<f64> = (0..N_CELLS)
        .map(|cell| (0..N_GENES).map(|g| dense[g * N_CELLS + cell]).sum())
        .collect();
    let data_2: Vec<f32> = indices
        .iter()
        .zip(data.iter())
        .map(|(&cell, &count)| {
            let lib = library[cell as usize].max(1.0);
            ((count as f64 / lib) * 1e4).ln_1p() as f32
        })
        .collect();

    let matrix = CompressedSparseData2::from_parts(
        data,
        indices,
        indptr,
        Some(data_2),
        CompressedSparseFormat::Csc,
        (N_CELLS, N_GENES),
    );

    let mut aggregate = vec![0.0_f64; N_GENES * N_SAMPLES];
    for gene in 0..N_GENES {
        for (sample, cells) in sample_cells.iter().enumerate() {
            aggregate[gene * N_SAMPLES + sample] =
                cells.iter().map(|&cell| dense[gene * N_CELLS + cell]).sum();
        }
    }

    let design: Vec<f64> = (0..N_SAMPLES)
        .flat_map(|s| [1.0, (s / (N_SAMPLES / 2)) as f64])
        .collect();

    Fixture {
        matrix,
        sample_cells,
        aggregate,
        design,
    }
}

///////////
// Tests //
///////////

/// The join is the aggregation and nothing else.
#[test]
fn test_pseudobulk_dge_matches_the_hand_aggregate() {
    let f = build(3);
    let reader = InMemorySparseReader::new(&f.matrix, None).expect("reader failed");
    let genes: Vec<usize> = (0..N_GENES).collect();
    let params = EdgeRQlParams::default();
    let tested = Tested::Coef(vec![1]);

    let got = pseudobulk_dge(
        &reader,
        &genes,
        &f.sample_cells,
        &f.design,
        N_COEF,
        &tested,
        &params,
        0,
    )
    .expect("pseudobulk_dge failed");

    let want = run_edger_ql(
        &f.aggregate,
        N_GENES,
        N_SAMPLES,
        &f.design,
        N_COEF,
        &tested,
        &params,
    )
    .expect("run_edger_ql failed");

    assert_eq!(got.genes_to_keep, want.genes_to_keep);
    assert_eq!(got.log_fc, want.log_fc);
    assert_eq!(got.log_cpm, want.log_cpm);
    assert_eq!(got.f_stat, want.f_stat);
    assert_eq!(got.p_val, want.p_val);
    assert_eq!(got.fdr, want.fdr);
}

/// The planted genes are the ones that come out.
///
/// Not a power check, just enough to catch a transposed aggregate: a design
/// applied to the wrong axis would find nothing at all.
#[test]
fn test_pseudobulk_dge_finds_the_planted_genes() {
    let f = build(3);
    let reader = InMemorySparseReader::new(&f.matrix, None).expect("reader failed");
    let genes: Vec<usize> = (0..N_GENES).collect();

    let got = pseudobulk_dge(
        &reader,
        &genes,
        &f.sample_cells,
        &f.design,
        N_COEF,
        &Tested::Coef(vec![1]),
        &EdgeRQlParams::default(),
        0,
    )
    .expect("pseudobulk_dge failed");

    let kept: Vec<usize> = got
        .genes_to_keep
        .iter()
        .enumerate()
        .filter(|(_, k)| **k)
        .map(|(gene, _)| gene)
        .collect();

    let hits: Vec<usize> = (0..kept.len())
        .filter(|&i| got.fdr[i] < 0.05)
        .map(|i| kept[i])
        .collect();

    let planted = [0, 20, 40, 60, 80, 100];
    for gene in planted {
        assert!(
            hits.contains(&gene),
            "planted gene {gene} was missed: {hits:?}"
        );
    }
    // A five per cent false discovery rate is a promise about the proportion of
    // hits that are wrong, not a promise that none of them are, so a stray gene
    // or two here is the method working rather than failing.
    assert!(
        hits.len() <= planted.len() + 2,
        "too many genes came out for the effect planted: {hits:?}"
    );
}

/// A design that does not match the samples has to say so.
#[test]
fn test_pseudobulk_dge_rejects_a_mismatched_design() {
    let f = build(3);
    let reader = InMemorySparseReader::new(&f.matrix, None).expect("reader failed");
    let genes: Vec<usize> = (0..N_GENES).collect();

    assert!(matches!(
        pseudobulk_dge(
            &reader,
            &genes,
            &f.sample_cells,
            &f.design[..N_COEF],
            N_COEF,
            &Tested::Coef(vec![1]),
            &EdgeRQlParams::default(),
            0,
        ),
        Err(BixverseErrors::DgeShapeMismatch { name: "design", .. })
    ));
}
