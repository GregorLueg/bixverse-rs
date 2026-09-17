#![cfg(feature = "single-cell")]
//! Parity for the analytic Pearson residuals against scanpy 1.11.5.
//!
//! The count matrix is rebuilt from the same LCG the fixture generator used, so
//! nothing but integers and reference answers crosses as text. See
//! `dev/gen_analytic_pearson_fixtures.py`.
//!
//! The model is closed form, so the bar is tight throughout: both sides
//! evaluate the same expression in `f64` and the only slack is the order the
//! sums accumulate in. Residual *rows* are `f32` on this side, which is where
//! the looser probe tolerance comes from.

use approx::assert_relative_eq;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_data::bin_merge_io::gene_store_to_cell_store;
use bixverse_rs::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CscGeneChunk, ParallelSparseReader, RawCounts,
};
use bixverse_rs::single_cell::sc_processing::analytic_pearson::model::{AprParams, AprResiduals};
use bixverse_rs::single_cell::sc_processing::analytic_pearson::stream::{
    apr_gene_pass, build_apr_model, cell_totals_over_genes, fit_analytic_pearson_grouped,
};
use bixverse_rs::single_cell::sc_processing::hvg::select_residual_hvg;
use bixverse_rs::single_cell::sc_processing::residuals::residual_variance;
use bixverse_rs::single_cell::sc_processing::sctransform::stream::SctStreamOpts;

mod analytic_pearson_fixtures;
use analytic_pearson_fixtures as fx;

/////////////
// Helpers //
/////////////

/// Removes the scratch stores when the test ends, pass or panic.
struct TempStore(std::path::PathBuf);

impl Drop for TempStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempStore {
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!("bixverse_apr_{name}.bin")))
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// Rebuilds the count matrix `dev/gen_analytic_pearson_fixtures.py` generated.
///
/// The draw order mirrors the Python script call for call: every library scale,
/// then every gene rate, then the counts gene-major, then the second half
/// thinned by integer division.
fn fixture_counts() -> Vec<Vec<u32>> {
    let mut state = fx::LCG_SEED;
    let mut next = || {
        state = (1_664_525_u64
            .wrapping_mul(state)
            .wrapping_add(1_013_904_223))
            % 4_294_967_296;
        state as f64 / 4_294_967_296.0
    };

    let lib_scale: Vec<f64> = (0..fx::N_CELLS).map(|_| 2000.0 + 6000.0 * next()).collect();
    let gene_rate: Vec<f64> = (0..fx::N_GENES)
        .map(|_| 10.0_f64.powf(-5.0 + 3.2 * next()))
        .collect();

    let mut counts = vec![vec![0_u32; fx::N_CELLS]; fx::N_GENES];
    for (g, row) in counts.iter_mut().enumerate() {
        for (c, slot) in row.iter_mut().enumerate() {
            let u = next();
            *slot = (u * u * u * gene_rate[g] * lib_scale[c] * 12.0).floor() as u32;
        }
    }

    for row in counts.iter_mut() {
        for value in row.iter_mut().skip(fx::SPLIT) {
            *value /= fx::THIN;
        }
    }

    counts
}

/// Writes a `dense[gene][cell]` matrix out as a gene-major store.
fn write_store(path: &str, dense: &[Vec<u32>], n_cells: usize) {
    let mut writer =
        CellGeneSparseWriter::new(path, false, n_cells, dense.len(), 1e4).expect("writer opens");

    for (gene_idx, gene) in dense.iter().enumerate() {
        let mut raw = Vec::new();
        let mut indices = Vec::new();
        for (cell, &value) in gene.iter().enumerate() {
            if value > 0 {
                raw.push(value);
                indices.push(cell);
            }
        }
        let norms: Vec<F16> = raw
            .iter()
            .map(|&v| F16::from_f32((v as f32).ln_1p()))
            .collect();

        writer
            .write_gene_chunk(CscGeneChunk::from_conversion(
                RawCounts::from_u32_auto(&raw),
                &norms,
                &indices,
                gene_idx,
                true,
            ))
            .expect("write gene chunk");
    }

    writer.finalise().expect("finalise");
}

/// A gene-major store plus its cell-major twin, both cleaned up on drop.
struct Stores {
    gene: TempStore,
    cell: TempStore,
}

fn build_stores(name: &str, counts: &[Vec<u32>]) -> Stores {
    let gene = TempStore::new(&format!("{name}_gene"));
    let cell = TempStore::new(&format!("{name}_cell"));
    write_store(gene.path(), counts, fx::N_CELLS);
    gene_store_to_cell_store(gene.path(), cell.path(), 20_000, 512, 0).expect("transpose");
    Stores { gene, cell }
}

fn params() -> AprParams {
    AprParams {
        theta: fx::THETA,
        min_cells: fx::MIN_CELLS,
        clip_range: None,
    }
}

///////////
// Tests //
///////////

/// The rebuilt matrix has to agree with Python's before anything downstream
/// means anything. The retained gene set is derived from the counts on the
/// Python side, so it doubles as a checksum on the whole matrix.
#[test]
fn test_fixture_counts_round_trip() {
    let counts = fixture_counts();
    let retained: Vec<usize> = (0..fx::N_GENES)
        .filter(|&g| counts[g].iter().filter(|&&v| v > 0).count() >= fx::MIN_CELLS)
        .collect();

    assert_eq!(retained, fx::POOLED_RETAINED.to_vec());
}

/// The three marginals the closed form is built from.
#[test]
fn test_model_marginals_match_scanpy() {
    let counts = fixture_counts();
    let stores = build_stores("marginals", &counts);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();

    let pass = apr_gene_pass(&gene_reader, &cells, &params(), SctStreamOpts::default())
        .expect("gene pass");
    assert_eq!(pass.retained, fx::POOLED_RETAINED.to_vec());

    let totals = cell_totals_over_genes(&cell_reader, &cells, &pass.retained).expect("cell totals");
    let model = build_apr_model(&pass, &totals, &params()).expect("model");

    assert_relative_eq!(model.total, fx::POOLED_TOTAL, epsilon = 1e-12);
    for (pos, &want) in fx::POOLED_GENE_SUMS.iter().enumerate() {
        assert_relative_eq!(model.gene_sums[pos], want, epsilon = 1e-12);
    }

    // The two marginals have to sum to the same grand total, or `mu` is not the
    // maximum likelihood solution of anything.
    let from_cells: f64 = totals.iter().sum();
    assert_relative_eq!(from_cells, fx::POOLED_TOTAL, epsilon = 1e-9);

    // The clip resolves to sqrt(n), scanpy's default.
    assert_relative_eq!(
        model.clip_range.1,
        (fx::N_CELLS as f64).sqrt(),
        epsilon = 1e-12
    );
}

/// Individual residuals, not just their variance.
#[test]
fn test_residuals_match_scanpy() {
    let counts = fixture_counts();
    let stores = build_stores("residuals", &counts);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();

    let pass = apr_gene_pass(&gene_reader, &cells, &params(), SctStreamOpts::default())
        .expect("gene pass");
    let totals = cell_totals_over_genes(&cell_reader, &cells, &pass.retained).expect("cell totals");
    let model = build_apr_model(&pass, &totals, &params()).expect("model");
    let source = AprResiduals::single(&model, &totals).expect("residual source");

    let mut probe = 0;
    for &gene_pos in fx::PROBE_GENES.iter() {
        let gene = source.genes()[gene_pos];
        let chunk = gene_reader.read_gene(gene).expect("gene reads");
        let nz: Vec<f64> = match &chunk.data_raw {
            RawCounts::U16(v) => v.iter().map(|&x| x as f64).collect(),
            RawCounts::U32(v) => v.iter().map(|&x| x as f64).collect(),
        };

        let mut row = vec![0.0_f32; fx::N_CELLS];
        source
            .residual_row(&nz, &chunk.indices, gene_pos, &mut row)
            .expect("residual row");

        for &cell in fx::PROBE_CELLS.iter() {
            let want = fx::PROBE_RESIDUALS[probe];
            // The row is f32, so seven significant digits is the ceiling.
            assert_relative_eq!(row[cell] as f64, want, epsilon = 1e-5, max_relative = 1e-5);
            probe += 1;
        }
    }
}

/// Residual variance over every retained gene, pooled.
#[test]
fn test_residual_variance_matches_scanpy() {
    let counts = fixture_counts();
    let stores = build_stores("variance", &counts);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let opts = SctStreamOpts::default();

    let pass = apr_gene_pass(&gene_reader, &cells, &params(), opts).expect("gene pass");
    let totals = cell_totals_over_genes(&cell_reader, &cells, &pass.retained).expect("cell totals");
    let model = build_apr_model(&pass, &totals, &params()).expect("model");
    let source = AprResiduals::single(&model, &totals).expect("residual source");

    let got = residual_variance(&gene_reader, &source, &cells, opts).expect("residual variance");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].len(), fx::POOLED_RESIDUAL_VARIANCE.len());

    for (pos, &want) in fx::POOLED_RESIDUAL_VARIANCE.iter().enumerate() {
        assert_relative_eq!(got[0][pos], want, epsilon = 1e-4, max_relative = 1e-4);
    }
}

/// scanpy's own HVG path divides by `n`, this crate by `n - 1`. Pin the
/// relationship rather than assume it.
#[test]
fn test_scanpy_hvg_variance_differs_only_by_the_denominator() {
    let n = fx::N_CELLS as f64;
    for (pos, &want_n) in fx::POOLED_RESIDUAL_VARIANCE_N.iter().enumerate() {
        let from_n_minus_1 = fx::POOLED_RESIDUAL_VARIANCE[pos] * (n - 1.0) / n;
        assert_relative_eq!(
            from_n_minus_1,
            want_n,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }
}

/// Sharing the clip range across samples is a real choice, not a no-op.
///
/// If it were a no-op the grouped parity test above would pass either way and
/// would be pinning nothing.
#[test]
fn test_shared_clip_changes_the_per_sample_variance() {
    let differs = fx::A_RESIDUAL_VARIANCE
        .iter()
        .zip(fx::A_RESIDUAL_VARIANCE_SHARED_CLIP.iter())
        .filter(|(own, shared)| (*own - *shared).abs() > 1e-6)
        .count();

    assert!(
        differs > 0,
        "clipping at sqrt(400) rather than sqrt(200) should move some genes"
    );
}

/// Each sample's model is the model scanpy gives that sample on its own.
#[test]
fn test_grouped_fit_matches_scanpy_per_sample() {
    let counts = fixture_counts();
    let stores = build_stores("grouped", &counts);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let groups: Vec<u32> = (0..fx::N_CELLS)
        .map(|c| if c < fx::SPLIT { 0 } else { 1 })
        .collect();
    let opts = SctStreamOpts::default();

    let fit =
        fit_analytic_pearson_grouped(&gene_reader, &cell_reader, &cells, &groups, &params(), opts)
            .expect("grouped fit");

    assert_eq!(fit.models.len(), 2);
    assert_eq!(fit.models[0].genes, fx::A_RETAINED.to_vec());
    assert_eq!(fit.models[1].genes, fx::B_RETAINED.to_vec());
    assert_relative_eq!(fit.models[0].total, fx::A_TOTAL, epsilon = 1e-12);
    assert_relative_eq!(fit.models[1].total, fx::B_TOTAL, epsilon = 1e-12);

    for (pos, &want) in fx::A_GENE_SUMS.iter().enumerate() {
        assert_relative_eq!(fit.models[0].gene_sums[pos], want, epsilon = 1e-12);
    }
    for (pos, &want) in fx::B_GENE_SUMS.iter().enumerate() {
        assert_relative_eq!(fit.models[1].gene_sums[pos], want, epsilon = 1e-12);
    }

    // Thinning the second sample really does change the model, otherwise this
    // whole path would be pointless.
    assert!(fit.models[1].total < fit.models[0].total);
    assert!(fit.models[1].genes.len() < fit.models[0].genes.len());

    // One clip range across groups, resolved against the total cell count.
    assert_eq!(fit.models[0].clip_range, fit.models[1].clip_range);
    assert_relative_eq!(
        fit.models[0].clip_range.1,
        (fx::N_CELLS as f64).sqrt(),
        epsilon = 1e-12
    );
}

/// Per-group residual variance, against scanpy run on each sample separately.
#[test]
fn test_grouped_residual_variance_matches_scanpy_per_sample() {
    let counts = fixture_counts();
    let stores = build_stores("grouped_var", &counts);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let groups: Vec<u32> = (0..fx::N_CELLS)
        .map(|c| if c < fx::SPLIT { 0 } else { 1 })
        .collect();
    let opts = SctStreamOpts::default();

    let fit =
        fit_analytic_pearson_grouped(&gene_reader, &cell_reader, &cells, &groups, &params(), opts)
            .expect("grouped fit");

    let source = AprResiduals::new(&fit.models, &fit.cell_totals, fit.group_of_cell.clone())
        .expect("residual source");
    let got = residual_variance(&gene_reader, &source, &cells, opts).expect("residual variance");
    assert_eq!(got.len(), 2);

    // The shared axis is the intersection, so each group's reference has to be
    // looked up by store gene index rather than by position. The reference is
    // the shared-clip variant: the grouped path resolves one clip range against
    // the total cell count, where scanpy's default would give each sample its
    // own.
    let references: [(&[usize], &[f64]); 2] = [
        (&fx::A_RETAINED, &fx::A_RESIDUAL_VARIANCE_SHARED_CLIP),
        (&fx::B_RETAINED, &fx::B_RESIDUAL_VARIANCE_SHARED_CLIP),
    ];

    for (group, (retained, reference)) in references.iter().enumerate() {
        for (pos, &gene) in source.genes().iter().enumerate() {
            let want_pos = retained
                .binary_search(&gene)
                .expect("a shared gene is retained in every group");
            assert_relative_eq!(
                got[group][pos],
                reference[want_pos],
                epsilon = 1e-4,
                max_relative = 1e-4
            );
        }
    }
}

/// The shared gene axis is the intersection, and HVG selection unions the
/// per-sample tops over it.
#[test]
fn test_grouped_hvg_unions_the_per_sample_tops() {
    let counts = fixture_counts();
    let stores = build_stores("grouped_hvg", &counts);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let groups: Vec<u32> = (0..fx::N_CELLS)
        .map(|c| if c < fx::SPLIT { 0 } else { 1 })
        .collect();
    let opts = SctStreamOpts::default();

    let fit =
        fit_analytic_pearson_grouped(&gene_reader, &cell_reader, &cells, &groups, &params(), opts)
            .expect("grouped fit");
    let source = AprResiduals::new(&fit.models, &fit.cell_totals, fit.group_of_cell.clone())
        .expect("residual source");

    // Every shared gene is retained in both samples, and the axis is ascending.
    for &gene in source.genes() {
        assert!(fx::A_RETAINED.binary_search(&gene).is_ok());
        assert!(fx::B_RETAINED.binary_search(&gene).is_ok());
    }
    assert!(source.genes().windows(2).all(|w| w[0] < w[1]));

    let per_group = residual_variance(&gene_reader, &source, &cells, opts).expect("variance");
    let n_hvg = 30;
    let hvg = select_residual_hvg(&per_group, source.genes(), n_hvg).expect("hvg");

    // Between n_hvg and 2 * n_hvg: the two samples agree on some genes but not
    // all, which is the whole reason for unioning rather than pooling.
    assert!(hvg.len() >= n_hvg);
    assert!(hvg.len() <= 2 * n_hvg);
    assert!(hvg.windows(2).all(|w| w[0] < w[1]));

    // Every selected gene is top-n_hvg in at least one sample.
    for &gene in &hvg {
        let pos = source
            .position(gene)
            .expect("selected genes are on the axis");
        let top_in_some = per_group.iter().any(|variance| {
            let rank = variance.iter().filter(|&&v| v > variance[pos]).count();
            rank < n_hvg
        });
        assert!(top_in_some, "gene {gene} is not top-{n_hvg} in any sample");
    }
}

/// A gene-major store is refused for the cell-major pass and vice versa, rather
/// than silently summing the wrong axis.
#[test]
fn test_readers_are_checked_for_orientation() {
    let counts = fixture_counts();
    let stores = build_stores("orientation", &counts);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();

    assert!(matches!(
        apr_gene_pass(&cell_reader, &cells, &params(), SctStreamOpts::default()),
        Err(BixverseErrors::ReaderModeMismatch { .. })
    ));
    assert!(matches!(
        cell_totals_over_genes(&gene_reader, &cells, &[0, 1, 2]),
        Err(BixverseErrors::ReaderModeMismatch { .. })
    ));
}
