//! Gates the NEBULA adapter, not the NEBULA numerics.
//!
//! `edge-rs` already checks its fits against the `nebula` R package on
//! committed fixtures, so there is nothing to gain from repeating that here.
//! What this file pins is everything the crate does around the call: the
//! subject reordering, the sparse concatenation, the gene batching and the
//! metacell shim. Every assertion is against
//! [`edge_rs::sc::nebula::nebula`] run on the same data laid out by hand, and
//! every one of them is exact rather than approximate, because the adapter is
//! supposed to change nothing.
//!
//! Ten genes over 150 cells and six subjects, so 25 cells per subject, which
//! is below the thirty NEBULA-LN needs and puts every gene on the HL path.

#![cfg(feature = "single-cell")]

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::mc_analysis::nebula_mc::nebula_metacells;
use bixverse_rs::single_cell::sc_analysis::nebula::{NebulaScParams, NebulaScRes, run_nebula};
use bixverse_rs::single_cell::sc_data::in_memory_io::InMemorySparseReader;

use edge_rs::sc::nebula::{NebulaFit, nebula};
use edge_rs::sc::test::ScTested;

use rand::prelude::*;
use rand_distr::{Distribution, Gamma, Normal, Poisson};

////////////////
// Dimensions //
////////////////

/// Genes in the fixture. Gene zero is deliberately near-empty.
const N_GENES: usize = 10;
/// Subjects, the random effect NEBULA fits.
const N_SUBJECTS: usize = 6;
/// Cells per subject. Below thirty, so the LN method is downgraded to HL.
const CELLS_PER_SUBJECT: usize = 25;
/// Total cells.
const N_CELLS: usize = N_SUBJECTS * CELLS_PER_SUBJECT;
/// Intercept, group and one continuous covariate.
const N_COEF: usize = 3;
/// Library size every cell is drawn around.
const LIBRARY_SIZE: f64 = 3000.0;
/// Negative binomial size, the inverse of the cell-level overdispersion.
const NB_SIZE: f64 = 8.0;
/// Subject-level random effect on the log scale.
const SUBJECT_SD: f64 = 0.4;

/////////////
// Fixture //
/////////////

/// A synthetic experiment plus everything the two call paths need.
struct Fixture {
    /// Counts as CSC over `(cells, genes)`, raw in `data` and normalised in
    /// `data_2`, which is what [`InMemorySparseReader`] takes.
    matrix: CompressedSparseData2<u32, f32>,
    /// Dense counts, gene-major and row-major, for the reference call.
    dense: Vec<f64>,
    /// Subject per cell, interleaved so the adapter has real work to do.
    subject_ids: Vec<usize>,
    /// Design, row-major `N_CELLS * N_COEF`, rows in cell order.
    design: Vec<f64>,
    /// Offset per cell, in cell order.
    offset: Vec<f64>,
}

/// Draws from `NegBin(size, mu)` through a gamma-Poisson mixture.
///
/// ### Params
///
/// * `size` - Negative binomial size
/// * `mu` - Mean
/// * `rng` - Source of randomness
///
/// ### Returns
///
/// One count.
fn nb_sample(size: f64, mu: f64, rng: &mut StdRng) -> u32 {
    if mu <= 1e-12 {
        return 0;
    }
    let scale = mu / size;
    let lambda = Gamma::new(size, scale).expect("bad gamma").sample(rng);
    if lambda <= 0.0 {
        return 0;
    }
    Poisson::new(lambda).expect("bad poisson").sample(rng) as u32
}

/// Builds the fixture.
///
/// Subjects are assigned round-robin, so the natural cell order interleaves
/// them and `run_nebula` has to sort before `edge-rs` will accept the labels.
/// The group covariate varies between subjects and not within, which is the
/// design NEBULA is actually for.
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
    let normal = Normal::new(0.0, 1.0).expect("bad normal");

    let subject_ids: Vec<usize> = (0..N_CELLS).map(|c| c % N_SUBJECTS).collect();
    let covariate: Vec<f64> = (0..N_CELLS).map(|_| normal.sample(&mut rng)).collect();

    let mut design = Vec::with_capacity(N_CELLS * N_COEF);
    for cell in 0..N_CELLS {
        design.push(1.0);
        design.push((subject_ids[cell] % 2) as f64);
        design.push(covariate[cell]);
    }

    let subject_effect: Vec<Vec<f64>> = (0..N_GENES)
        .map(|_| {
            (0..N_SUBJECTS)
                .map(|_| normal.sample(&mut rng) * SUBJECT_SD)
                .collect()
        })
        .collect();

    let offset: Vec<f64> = (0..N_CELLS)
        .map(|_| LIBRARY_SIZE * (1.0 + 0.2 * normal.sample(&mut rng)).max(0.5))
        .collect();

    let mut dense = vec![0.0_f64; N_GENES * N_CELLS];
    for gene in 0..N_GENES {
        // Gene zero sits under NEBULA's own `cpc` and `mincp` filter, so it
        // never reaches a fit and the batch holding it alone comes back empty.
        let base = if gene == 0 {
            -14.0
        } else {
            -8.0 + 0.3 * gene as f64
        };
        let group_effect = 0.4 * (gene % 3) as f64;
        let cov_effect = 0.2 * (gene % 2) as f64;

        for cell in 0..N_CELLS {
            let eta = base
                + group_effect * design[cell * N_COEF + 1]
                + cov_effect * covariate[cell]
                + subject_effect[gene][subject_ids[cell]];
            let mu = offset[cell] * eta.exp();
            dense[gene * N_CELLS + cell] = nb_sample(NB_SIZE, mu, &mut rng) as f64;
        }
    }

    // CSC over (cells, genes): `indptr` walks the genes, `indices` holds cells.
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

    // The normalised layer is never read by NEBULA, but the reader insists on
    // one, so it gets the usual ln1p of the depth-scaled count.
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

    Fixture {
        matrix,
        dense,
        subject_ids,
        design,
        offset,
    }
}

/// The knobs both paths run with.
///
/// ### Returns
///
/// [NebulaScParams] testing the group coefficient, shrinkage off so the
/// comparison is against the raw fit.
fn params() -> NebulaScParams {
    NebulaScParams {
        shrink_dispersion: false,
        tested: ScTested::Coef(1),
        ..NebulaScParams::default()
    }
}

/// Runs `edge-rs` directly on the fixture, cells sorted by subject.
///
/// ### Params
///
/// * `f` - The fixture
///
/// ### Returns
///
/// The [NebulaFit], and the original cell order the sort produced.
fn reference(f: &Fixture) -> NebulaFit {
    let mut order: Vec<usize> = (0..N_CELLS).collect();
    order.sort_by_key(|&c| f.subject_ids[c]);

    let mut dense = vec![0.0_f64; N_GENES * N_CELLS];
    for gene in 0..N_GENES {
        for (slot, &cell) in order.iter().enumerate() {
            dense[gene * N_CELLS + slot] = f.dense[gene * N_CELLS + cell];
        }
    }
    let subject: Vec<usize> = order.iter().map(|&c| f.subject_ids[c]).collect();
    let mut design = Vec::with_capacity(N_CELLS * N_COEF);
    for &cell in &order {
        design.extend_from_slice(&f.design[cell * N_COEF..(cell + 1) * N_COEF]);
    }
    let offset: Vec<f64> = order.iter().map(|&c| f.offset[c]).collect();

    nebula(
        &dense,
        N_GENES,
        N_CELLS,
        &subject,
        &design,
        N_COEF,
        Some(&offset),
        Some(params().nebula),
    )
    .expect("the reference nebula call failed")
}

/// Runs the crate's adapter over the in-memory store.
///
/// ### Params
///
/// * `f` - The fixture
/// * `cells` - Cell selection, in whatever order the test wants
/// * `p` - The knobs
///
/// ### Returns
///
/// The [NebulaScRes].
fn adapter(f: &Fixture, cells: &[usize], p: &NebulaScParams) -> NebulaScRes {
    let reader = InMemorySparseReader::new(&f.matrix, None).expect("reader failed");
    let genes: Vec<usize> = (0..N_GENES).collect();
    let offset: Vec<f64> = cells.iter().map(|&c| f.offset[c]).collect();
    let mut design = Vec::with_capacity(cells.len() * N_COEF);
    for &cell in cells {
        design.extend_from_slice(&f.design[cell * N_COEF..(cell + 1) * N_COEF]);
    }

    run_nebula(
        &reader,
        &reader,
        cells,
        &genes,
        &f.subject_ids,
        &design,
        N_COEF,
        Some(&offset),
        p,
        0,
    )
    .expect("run_nebula failed")
}

///////////
// Tests //
///////////

/// The adapter is the reference call with the plumbing hoisted out.
///
/// The cells arrive interleaved by subject, so this also pins the sort, the
/// design permutation and the offset permutation in one go.
#[test]
fn test_run_nebula_matches_the_direct_call() {
    let f = build(7);
    let want = reference(&f);
    let cells: Vec<usize> = (0..N_CELLS).collect();

    let got = adapter(&f, &cells, &params());

    assert_eq!(got.gene_idx, want.gene_index, "surviving genes");
    assert_eq!(got.coefficients, want.coefficients);
    assert_eq!(got.se, want.se);
    assert_eq!(got.subject_overdispersion, want.subject_overdispersion);
    assert_eq!(got.cell_overdispersion, want.cell_overdispersion);
    assert_eq!(got.convergence, want.convergence);
    assert_eq!(got.sigma_at_bound, want.sigma_at_bound);
    assert_eq!(got.n_coef, N_COEF);
}

/// The gene batch size is a memory knob and nothing else.
///
/// One gene per batch and every gene in one batch have to agree exactly, which
/// is only true because NEBULA's expression filter is per gene.
#[test]
fn test_gene_batching_changes_nothing() {
    let f = build(7);
    let cells: Vec<usize> = (0..N_CELLS).collect();

    let one_at_a_time = adapter(
        &f,
        &cells,
        &NebulaScParams {
            gene_batch_size: 1,
            ..params()
        },
    );
    let all_at_once = adapter(
        &f,
        &cells,
        &NebulaScParams {
            gene_batch_size: 10_000,
            ..params()
        },
    );

    assert_eq!(one_at_a_time.gene_idx, all_at_once.gene_idx);
    assert_eq!(one_at_a_time.coefficients, all_at_once.coefficients);
    assert_eq!(one_at_a_time.se, all_at_once.se);
    assert_eq!(one_at_a_time.p_val, all_at_once.p_val);
}

/// A batch that loses every gene to the filter is skipped, not fatal.
///
/// Gene zero is drawn far below `cpc`, so with one gene per batch its batch
/// comes back as `NoGenesAfterFiltering`. The sweep has to carry on.
#[test]
fn test_a_fully_filtered_batch_is_skipped() {
    let f = build(7);
    let cells: Vec<usize> = (0..N_CELLS).collect();

    let got = adapter(
        &f,
        &cells,
        &NebulaScParams {
            gene_batch_size: 1,
            ..params()
        },
    );

    assert!(
        !got.gene_idx.contains(&0),
        "gene zero should not survive the expression filter"
    );
    assert!(
        got.gene_idx.len() > 1,
        "the rest of the sweep should still be there"
    );
}

/// The cell order the caller uses does not reach the fit.
///
/// A shuffle is undone by the same sort that handles the interleaving, so the
/// two runs are the same fit gene for gene.
#[test]
fn test_cell_order_does_not_change_the_fit() {
    let f = build(7);
    let natural: Vec<usize> = (0..N_CELLS).collect();
    let mut shuffled = natural.clone();
    shuffled.shuffle(&mut StdRng::seed_from_u64(11));

    let straight = adapter(&f, &natural, &params());
    let jumbled = adapter(&f, &shuffled, &params());

    assert_eq!(straight.gene_idx, jumbled.gene_idx);
    assert_eq!(straight.coefficients, jumbled.coefficients);
    assert_eq!(straight.se, jumbled.se);
}

/// Leaving out the offset falls back to the library sizes.
#[test]
fn test_no_offset_uses_the_library_sizes() {
    let f = build(7);
    let reader = InMemorySparseReader::new(&f.matrix, None).expect("reader failed");
    let cells: Vec<usize> = (0..N_CELLS).collect();
    let genes: Vec<usize> = (0..N_GENES).collect();

    let library: Vec<f64> = (0..N_CELLS)
        .map(|cell| (0..N_GENES).map(|g| f.dense[g * N_CELLS + cell]).sum())
        .collect();

    let implicit = run_nebula(
        &reader,
        &reader,
        &cells,
        &genes,
        &f.subject_ids,
        &f.design,
        N_COEF,
        None,
        &params(),
        0,
    )
    .expect("run_nebula failed");
    let explicit = run_nebula(
        &reader,
        &reader,
        &cells,
        &genes,
        &f.subject_ids,
        &f.design,
        N_COEF,
        Some(&library),
        &params(),
        0,
    )
    .expect("run_nebula failed");

    assert_eq!(implicit.coefficients, explicit.coefficients);
}

/// The metacell shim only coerces the layout.
///
/// Feeding it the CSR twin exercises `as_csc`, which is the one thing the shim
/// does beyond building a reader.
#[test]
fn test_metacell_shim_accepts_either_orientation() {
    let f = build(7);
    let cells: Vec<usize> = (0..N_CELLS).collect();
    let genes: Vec<usize> = (0..N_GENES).collect();

    let csc = nebula_metacells(
        &f.matrix,
        &cells,
        &genes,
        &f.subject_ids,
        &f.design,
        N_COEF,
        Some(&f.offset),
        &params(),
        0,
    )
    .expect("nebula_metacells failed");

    let csr = nebula_metacells(
        &f.matrix.transform(),
        &cells,
        &genes,
        &f.subject_ids,
        &f.design,
        N_COEF,
        Some(&f.offset),
        &params(),
        0,
    )
    .expect("nebula_metacells failed");

    assert_eq!(csc.gene_idx, csr.gene_idx);
    assert_eq!(csc.coefficients, csr.coefficients);
}

/// The shrinkage runs once over the whole sweep, so it cannot depend on the
/// batching either.
#[test]
fn test_dispersion_shrinkage_is_batch_independent() {
    let f = build(7);
    let cells: Vec<usize> = (0..N_CELLS).collect();

    let split = adapter(
        &f,
        &cells,
        &NebulaScParams {
            gene_batch_size: 2,
            shrink_dispersion: true,
            ..params()
        },
    );
    let whole = adapter(
        &f,
        &cells,
        &NebulaScParams {
            shrink_dispersion: true,
            ..params()
        },
    );

    let split_shrunk = split
        .cell_overdispersion_shrunk
        .expect("shrinkage was asked for");
    let whole_shrunk = whole
        .cell_overdispersion_shrunk
        .expect("shrinkage was asked for");
    assert_eq!(split_shrunk, whole_shrunk);
    assert_eq!(split_shrunk.len(), split.gene_idx.len());
}

/// Shapes that cannot line up have to say so rather than fit something else.
#[test]
fn test_run_nebula_rejects_mismatched_inputs() {
    let f = build(7);
    let reader = InMemorySparseReader::new(&f.matrix, None).expect("reader failed");
    let cells: Vec<usize> = (0..N_CELLS).collect();
    let genes: Vec<usize> = (0..N_GENES).collect();
    let p = params();

    let short_design = &f.design[..(N_CELLS - 1) * N_COEF];
    assert!(matches!(
        run_nebula(
            &reader,
            &reader,
            &cells,
            &genes,
            &f.subject_ids,
            short_design,
            N_COEF,
            None,
            &p,
            0,
        ),
        Err(BixverseErrors::DgeShapeMismatch { name: "design", .. })
    ));

    assert!(matches!(
        run_nebula(
            &reader,
            &reader,
            &cells,
            &genes,
            &f.subject_ids[..10],
            &f.design,
            N_COEF,
            None,
            &p,
            0,
        ),
        Err(BixverseErrors::DgeShapeMismatch {
            name: "subject_ids",
            ..
        })
    ));

    assert!(matches!(
        run_nebula(
            &reader,
            &reader,
            &cells,
            &genes,
            &f.subject_ids,
            &f.design,
            N_COEF,
            Some(&f.offset[..10]),
            &p,
            0,
        ),
        Err(BixverseErrors::DgeShapeMismatch { name: "offset", .. })
    ));

    assert!(matches!(
        run_nebula(
            &reader,
            &reader,
            &[N_CELLS + 1],
            &genes,
            &f.subject_ids,
            &f.design[..N_COEF],
            N_COEF,
            None,
            &p,
            0,
        ),
        Err(BixverseErrors::ChunkIndexNotFound(_))
    ));

    // A repeated cell shortens the cell axis without shortening the design,
    // which would otherwise fit a model against the wrong rows.
    let mut duplicated = cells.clone();
    duplicated[3] = duplicated[2];
    assert!(matches!(
        run_nebula(
            &reader,
            &reader,
            &duplicated,
            &genes,
            &f.subject_ids,
            &f.design,
            N_COEF,
            None,
            &p,
            0,
        ),
        Err(BixverseErrors::DgeShapeMismatch { .. })
    ));
}
