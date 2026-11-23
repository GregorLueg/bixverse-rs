//! The doublet callers' store entry points against the on-disk drivers: a
//! small selection takes the in-memory cell subset and must give the same
//! result at the same seed.

use bixverse_sc::sc_processing::doublet_detection::*;
use bixverse_sc::sc_processing::scdblfinder::*;
use bixverse_sc::sc_processing::scrublet::*;
use bixverse_sc_data::prelude::*;
use bixverse_sc_data::store::in_memory::prefer_cell_subset;
use bixverse_sc_data::store::writer::CellGeneSparseWriter;
use tempfile::{NamedTempFile, TempPath};

const N_CELLS: usize = 3_000;
const N_GENES: usize = 120;
const SIZE_FACTOR: f32 = 1e4;
const SEED: usize = 42;

/// Deterministic value in `[0, 1)`.
fn hash01(a: usize, b: usize) -> f32 {
    let mut h = (a as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// `dense[cell][gene]` counts, three cell types, ~30 % dropout.
fn synthetic_counts() -> Vec<Vec<u32>> {
    (0..N_CELLS)
        .map(|cell| {
            (0..N_GENES)
                .map(|gene| {
                    let base = if gene % 3 == cell % 3 { 30.0 } else { 3.0 };
                    if hash01(gene, cell + 7) < 0.3 {
                        0
                    } else {
                        (base * (0.5 + hash01(cell, gene))) as u32
                    }
                })
                .collect()
        })
        .collect()
}

/// Write the gene-major and cell-major stores for the same counts.
fn write_stores(dense: &[Vec<u32>], genes: &TempPath, cells: &TempPath) {
    let lib: Vec<f32> = dense
        .iter()
        .map(|row| row.iter().map(|&v| v as f32).sum::<f32>().max(1.0))
        .collect();

    let mut gene_writer = CellGeneSparseWriter::new(
        genes.to_str().unwrap(),
        false,
        N_CELLS,
        N_GENES,
        SIZE_FACTOR,
    )
    .expect("gene writer opens");
    for gene in 0..N_GENES {
        let (mut raw, mut idx, mut norms) = (Vec::new(), Vec::new(), Vec::new());
        for (cell, row) in dense.iter().enumerate() {
            if row[gene] > 0 {
                raw.push(row[gene]);
                idx.push(cell);
                norms.push((row[gene] as f32 / lib[cell] * SIZE_FACTOR).ln_1p());
            }
        }
        gene_writer
            .write_gene_chunk(CscGeneChunk::from_conversion(
                RawCounts::from_u32_auto(&raw),
                &norms,
                &idx,
                gene,
                true,
            ))
            .expect("write gene chunk");
    }
    gene_writer.finalise().expect("finalise genes");

    let mut cell_writer =
        CellGeneSparseWriter::new(cells.to_str().unwrap(), true, N_CELLS, N_GENES, SIZE_FACTOR)
            .expect("cell writer opens");
    for (cell, row) in dense.iter().enumerate() {
        let (mut raw, mut idx) = (Vec::new(), Vec::new());
        for (gene, &v) in row.iter().enumerate() {
            if v > 0 {
                raw.push(v);
                idx.push(gene as u32);
            }
        }
        cell_writer
            .write_cell_chunk(CsrCellChunk::from_data(&raw, &idx, cell, SIZE_FACTOR, true))
            .expect("write cell chunk");
    }
    cell_writer.finalise().expect("finalise cells");
}

/// Stores plus a selection small enough to take the in-memory path.
fn fixture() -> (TempPath, TempPath, Vec<usize>) {
    let genes = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
    let cells = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
    write_stores(&synthetic_counts(), &genes, &cells);
    let selection: Vec<usize> = (5..N_CELLS).step_by(12).collect();
    assert!(prefer_cell_subset(selection.len(), N_CELLS));
    (genes, cells, selection)
}

fn readers(genes: &TempPath, cells: &TempPath) -> (ParallelSparseReader, ParallelSparseReader) {
    (
        ParallelSparseReader::new(genes.to_str().unwrap()).unwrap(),
        ParallelSparseReader::new(cells.to_str().unwrap()).unwrap(),
    )
}

#[test]
fn test_scrublet_on_store_matches_disk() {
    let (genes, cells, selection) = fixture();
    let (gene_reader, cell_reader) = readers(&genes, &cells);

    let (disk, disk_pca, disk_p1, _) =
        Scrublet::new(&gene_reader, &cell_reader, scrublet_params(), &selection)
            .run_scrublet(false, SEED, 0, true, true)
            .unwrap();
    let (mem, mem_pca, mem_p1, _) = run_scrublet_on_store(
        genes.to_str().unwrap(),
        cells.to_str().unwrap(),
        &selection,
        scrublet_params(),
        false,
        SEED,
        0,
        true,
        true,
    )
    .unwrap();

    assert_eq!(disk.doublet_scores_obs, mem.doublet_scores_obs);
    assert_eq!(disk.doublet_scores_sim, mem.doublet_scores_sim);
    assert_eq!(disk.predicted_doublets, mem.predicted_doublets);
    assert_eq!(disk.threshold, mem.threshold);
    assert_eq!(disk_p1, mem_p1);
    assert_eq!(disk_pca.unwrap(), mem_pca.unwrap());
}

#[test]
fn test_boost_on_store_matches_disk() {
    let (genes, cells, selection) = fixture();
    let (gene_reader, cell_reader) = readers(&genes, &cells);

    let disk = BoostClassifier::new(&gene_reader, &cell_reader, boost_params(), &selection)
        .run_boost(false, SEED, 0)
        .unwrap();
    let mem = run_boost_on_store(
        genes.to_str().unwrap(),
        cells.to_str().unwrap(),
        &selection,
        boost_params(),
        false,
        SEED,
        0,
    )
    .unwrap();

    assert_eq!(disk.doublet_scores, mem.doublet_scores);
    assert_eq!(disk.voting_average, mem.voting_average);
    assert_eq!(disk.predicted_doublets, mem.predicted_doublets);
}

#[test]
fn test_scdblfinder_on_store_matches_disk() {
    let (genes, cells, selection) = fixture();
    let (gene_reader, cell_reader) = readers(&genes, &cells);

    let disk = ScDblFinder::new(
        &gene_reader,
        &cell_reader,
        ScDblFinderParams::default(),
        &selection,
    )
    .run(SEED, false, 0)
    .unwrap();
    let mem = run_scdblfinder_on_store(
        genes.to_str().unwrap(),
        cells.to_str().unwrap(),
        &selection,
        ScDblFinderParams::default(),
        SEED,
        false,
        0,
    )
    .unwrap();

    assert_eq!(disk.selected_genes, mem.selected_genes);
    assert_eq!(disk.doublet_scores, mem.doublet_scores);
    assert_eq!(disk.cluster_labels, mem.cluster_labels);
    assert_eq!(disk.predicted_doublets, mem.predicted_doublets);
    assert_eq!(disk.threshold, mem.threshold);
}

/// The fixture has 18 HVGs, under the default 30 PCs.
fn scrublet_params() -> ScrubletParams {
    ScrubletParams {
        no_pcs: 10,
        ..Default::default()
    }
}

fn boost_params() -> BoostParams {
    BoostParams {
        no_pcs: 10,
        ..Default::default()
    }
}
