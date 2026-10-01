#![cfg(feature = "bonsai")]
//! Bonsai over the single cell binary stores.
//!
//! The counts come from sanity-sc-rs's own simulator and are written through
//! the normal store writer, so the reader path is checked against Sanity run
//! directly on the same `CountMatrix`.

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::bonsai::{
    BonsaiLayout, BonsaiScParams, SANITY_GENE_CHUNK, bonsai_layout, keep_for_bonsai,
    read_count_chunk, sanity_bonsai_sc, stream_sanity,
};
use bixverse_rs::single_cell::sc_data::bin_merge_io::gene_store_to_cell_store;
use bixverse_rs::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CscGeneChunk, ParallelSparseReader, RawCounts,
};
use bonsai_rs::sanity_sc_rs::simulate::{Simulation, SimulationParams, simulate};
use bonsai_rs::sanity_sc_rs::{SanityOutput, sanity, sanity_select};
use bonsai_rs::tree::NO_NODE;
use indexmap::IndexSet;

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
        Self(std::env::temp_dir().join(format!("bixverse_bonsai_{name}.bin")))
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// A gene-major store plus its cell-major twin.
struct Stores {
    gene: TempStore,
    cell: TempStore,
}

/// Small simulated dataset: 64 cells, up to 40 genes.
fn fixture() -> Simulation {
    simulate(Some(SimulationParams {
        n_genes: 40,
        n_cells: 64,
        seed: 11,
        ..SimulationParams::default()
    }))
    .expect("simulate")
}

/// Write the simulated counts, plus `extra_empty` all-zero genes at the end,
/// as a gene-major store and its cell-major twin.
fn build_stores(name: &str, sim: &Simulation, extra_empty: usize) -> Stores {
    let n_cells = sim.counts.n_cells();
    let n_genes = sim.counts.n_genes();
    let gene = TempStore::new(&format!("{name}_gene"));
    let cell = TempStore::new(&format!("{name}_cell"));

    let mut writer =
        CellGeneSparseWriter::new(gene.path(), false, n_cells, n_genes + extra_empty, 1e4)
            .expect("writer opens");
    for g in 0..n_genes + extra_empty {
        let (idx, raw): (Vec<usize>, Vec<u32>) = if g < n_genes {
            let (i, v) = sim.counts.gene(g);
            (i.iter().map(|&c| c as usize).collect(), v.to_vec())
        } else {
            (Vec::new(), Vec::new())
        };
        let norms: Vec<F16> = raw
            .iter()
            .map(|&v| F16::from_f32((v as f32).ln_1p()))
            .collect();
        writer
            .write_gene_chunk(CscGeneChunk::from_conversion(
                RawCounts::from_u32_auto(&raw),
                &norms,
                &idx,
                g,
                true,
            ))
            .expect("write gene chunk");
    }
    writer.finalise().expect("finalise");
    gene_store_to_cell_store(gene.path(), cell.path(), 20_000, 512, 0).expect("transpose");
    Stores { gene, cell }
}

///////////
// Tests //
///////////

#[test]
fn test_chunked_stream_matches_sanity_direct() {
    let sim = fixture();
    let stores = build_stores("direct", &sim, 0);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");

    let cells: Vec<usize> = (0..sim.counts.n_cells()).collect();
    let genes: Vec<usize> = (0..sim.counts.n_genes()).collect();
    // seven genes per chunk, so the stream crosses several chunk boundaries
    let streamed = stream_sanity(
        &gene_reader,
        &cell_reader,
        &cells,
        &genes,
        7,
        Verbosity::Quiet,
        |counts, totals| {
            assert_eq!(totals, sim.cell_totals.as_slice());
            Ok(sanity::<f32>(counts, totals, None)?)
        },
    )
    .expect("stream");

    let direct = sanity::<f32>(&sim.counts, &sim.cell_totals, None).expect("sanity");
    assert_eq!(streamed.genes, genes);
    assert_eq!(streamed.n_genes, genes.len());
    assert_eq!(streamed.log_fold_changes, direct.log_fold_changes);
    assert_eq!(streamed.error_bars, direct.error_bars);
    assert_eq!(streamed.variance, direct.variance);
}

#[test]
fn test_read_count_chunk_subsets_cells_and_skips_empty_genes() {
    let sim = fixture();
    let n_genes = sim.counts.n_genes();
    let stores = build_stores("subset", &sim, 2);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");

    let cells: Vec<usize> = (1..sim.counts.n_cells()).step_by(2).collect();
    let cell_set: IndexSet<u32> = cells.iter().map(|&c| c as u32).collect();
    let genes: Vec<usize> = (0..n_genes + 2).collect();
    let (counts, kept) = read_count_chunk(&gene_reader, &genes, &cell_set)
        .expect("read")
        .expect("not every gene is empty");

    assert_eq!(counts.n_cells(), cells.len());
    assert!(kept.iter().all(|&g| g < n_genes), "an empty gene was kept");

    // every stored count must be the original count of that gene in that cell
    for (col, &g) in kept.iter().enumerate() {
        let (orig_idx, orig_val) = sim.counts.gene(g);
        let (idx, val) = counts.gene(col);
        for (&local, &v) in idx.iter().zip(val) {
            let pos = orig_idx
                .iter()
                .position(|&c| c as usize == cells[local as usize])
                .expect("count in a cell the original gene has no count for");
            assert_eq!(v, orig_val[pos]);
        }
    }

    let empty_only =
        read_count_chunk(&gene_reader, &[n_genes, n_genes + 1], &cell_set).expect("read");
    assert!(empty_only.is_none());
}

#[test]
fn test_filtered_stream_does_not_depend_on_the_chunk_size() {
    let sim = fixture();
    let stores = build_stores("filtered", &sim, 2);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");

    let cells: Vec<usize> = (0..sim.counts.n_cells()).collect();
    let genes: Vec<usize> = (0..sim.counts.n_genes() + 2).collect();
    let params = BonsaiScParams::default();
    let keep = keep_for_bonsai(&params);
    let run = |chunk: usize| -> SanityOutput<f32> {
        stream_sanity(
            &gene_reader,
            &cell_reader,
            &cells,
            &genes,
            chunk,
            Verbosity::Quiet,
            |counts, totals| Ok(sanity_select(counts, totals, None, &keep)?),
        )
        .expect("stream")
    };

    let small = run(5);
    let whole = run(SANITY_GENE_CHUNK);
    assert!(small.n_genes > 0, "the filter kept nothing");
    assert!(
        small.n_genes < sim.counts.n_genes(),
        "the filter kept everything"
    );
    assert_eq!(small.genes, whole.genes);
    assert_eq!(small.log_fold_changes, whole.log_fold_changes);
    assert_eq!(small.error_bars, whole.error_bars);
}

#[test]
fn test_sanity_bonsai_sc_returns_a_single_rooted_tree() {
    let sim = fixture();
    let stores = build_stores("tree", &sim, 0);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");

    let cells: Vec<usize> = (0..sim.counts.n_cells()).collect();
    let genes: Vec<usize> = (0..sim.counts.n_genes()).collect();
    let res = sanity_bonsai_sc(
        &gene_reader,
        &cell_reader,
        &cells,
        &genes,
        &BonsaiScParams::default(),
        Verbosity::Quiet,
    )
    .expect("bonsai");

    let n_nodes = res.parent.len();
    assert_eq!(res.n_leaves, cells.len());
    assert_eq!(res.branch.len(), n_nodes);
    assert_eq!(res.x.len(), n_nodes);
    assert_eq!(res.y.len(), n_nodes);
    assert_eq!(res.parent.iter().filter(|&&p| p == NO_NODE).count(), 1);
    assert!(res.x.iter().chain(&res.y).all(|v| v.is_finite()));
    assert!(res.loglik.is_finite());
    assert!(!res.genes_used.is_empty());
    let stages: Vec<&str> = res.timings.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(stages, ["sanity", "ingest", "bonsai", "layout"]);
    assert!(res.timings.iter().all(|(_, t)| t.is_finite() && *t >= 0.0));
    assert_eq!(res.steps.len(), 7, "linkage start reports seven steps");
    assert!(res.genes_used.windows(2).all(|w| w[0] < w[1]));

    // relaying out the returned tree keeps its leaves and node count
    let (parent, _, coords) = bonsai_layout(
        res.parent.clone(),
        res.branch.clone(),
        res.n_leaves,
        BonsaiLayout::Dendrogram,
        true,
    )
    .expect("layout");
    assert_eq!(parent.len(), n_nodes);
    assert_eq!(coords.x.len(), n_nodes);
    assert!(
        coords
            .x
            .iter()
            .zip(&coords.y)
            .all(|(x, y)| x * x + y * y <= 1.0),
        "hyperbolic layout left the unit disk"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_sanity_bonsai_sc_gpu_returns_a_single_rooted_tree() {
    use bixverse_rs::gpu::sc_gpu::sanity_bonsai_gpu::sanity_bonsai_sc_gpu;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    let sim = fixture();
    let stores = build_stores("tree_gpu", &sim, 0);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");

    let cells: Vec<usize> = (0..sim.counts.n_cells()).collect();
    let genes: Vec<usize> = (0..sim.counts.n_genes()).collect();
    let res = sanity_bonsai_sc_gpu::<WgpuRuntime, _, _>(
        &gene_reader,
        &cell_reader,
        &cells,
        &genes,
        &BonsaiScParams::default(),
        WgpuDevice::default(),
        Verbosity::Quiet,
    )
    .expect("bonsai");

    assert_eq!(res.n_leaves, cells.len());
    assert_eq!(res.parent.iter().filter(|&&p| p == NO_NODE).count(), 1);
    assert!(res.x.iter().chain(&res.y).all(|v| v.is_finite()));
}

/// The simulated counts as an in-memory metacell matrix, cells as rows.
fn as_metacell_matrix(sim: &Simulation) -> CompressedSparseData2<u32, f32> {
    let (mut data, mut indices, mut indptr) = (Vec::new(), Vec::new(), vec![0u32]);
    for g in 0..sim.counts.n_genes() {
        let (idx, val) = sim.counts.gene(g);
        indices.extend_from_slice(idx);
        data.extend_from_slice(val);
        indptr.push(indices.len() as u32);
    }
    CompressedSparseData2::new_csc(
        &data,
        &indices,
        &indptr,
        None,
        (sim.counts.n_cells(), sim.counts.n_genes()),
    )
}

#[test]
fn test_metacell_path_matches_the_disk_path() {
    use bixverse_rs::single_cell::mc_analysis::bonsai_mc::sanity_bonsai_mc;

    let sim = fixture();
    let stores = build_stores("mc", &sim, 0);
    let gene_reader = ParallelSparseReader::new(stores.gene.path()).expect("gene reader");
    let cell_reader = ParallelSparseReader::new(stores.cell.path()).expect("cell reader");
    let cells: Vec<usize> = (0..sim.counts.n_cells()).collect();
    let genes: Vec<usize> = (0..sim.counts.n_genes()).collect();
    let params = BonsaiScParams::default();

    let disk = sanity_bonsai_sc(
        &gene_reader,
        &cell_reader,
        &cells,
        &genes,
        &params,
        Verbosity::Quiet,
    )
    .expect("disk");

    let csc = as_metacell_matrix(&sim);
    // the same counts from memory, in both orientations
    for counts in [csc.clone(), csc.transform()] {
        let mc = sanity_bonsai_mc(&counts, &genes, &params, Verbosity::Quiet).expect("mc");
        assert_eq!(mc.genes_used, disk.genes_used);
        assert_eq!(mc.parent, disk.parent);
        assert_eq!(mc.branch, disk.branch);
        assert_eq!(mc.loglik, disk.loglik);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_metacell_gpu_path_returns_a_single_rooted_tree() {
    use bixverse_rs::gpu::sc_gpu::sanity_bonsai_gpu::sanity_bonsai_mc_gpu;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    let sim = fixture();
    let genes: Vec<usize> = (0..sim.counts.n_genes()).collect();
    let res = sanity_bonsai_mc_gpu::<WgpuRuntime>(
        &as_metacell_matrix(&sim),
        &genes,
        &BonsaiScParams::default(),
        WgpuDevice::default(),
        Verbosity::Quiet,
    )
    .expect("bonsai");

    assert_eq!(res.n_leaves, sim.counts.n_cells());
    assert_eq!(res.parent.iter().filter(|&&p| p == NO_NODE).count(), 1);
}
