//! Frozen stores: a v4 and a v5 store written once and committed.
//!
//! Every build must keep reading them. A failure here means the on-disk format
//! changed, which is a compatibility break: fix the reader, never the fixture.
//! The only legitimate edit is adding a new fixture for a new format feature
//! (an `incompat` bit, a section) next to these. `regenerate_golden_fixtures`
//! wrote them and is kept to document how; it is ignored and must not be run
//! to "fix" a failure.

use std::path::PathBuf;

use bixverse_sc_data::store::chunk::CsrCellChunk;
use bixverse_sc_data::store::gene_file::write_gene_file;
use bixverse_sc_data::store::reader::{ParallelSparseReader, SingleCellReading};
use bixverse_sc_data::store::v5::{
    FeatureKind, NormKind, set_cell_ids, set_store_metadata, verify_store,
};
use bixverse_sc_data::store::writer::CellGeneSparseWriter;

/// Target size of both stores.
const TARGET: f32 = 1e4;

/// The matrix: cells by genes, raw counts.
const COUNTS: [[u32; 6]; 5] = [
    [3, 0, 1, 0, 0, 7],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 250, 0, 0, 2, 0],
    [9, 0, 0, 70_000, 0, 4],
];

/// Directory of the fixtures.
fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden")
}

/// Path of one fixture file as a string.
fn path(name: &str) -> String {
    dir().join(name).to_str().unwrap().to_string()
}

/// Indices and raw counts of a row.
fn row(c: usize) -> (Vec<u32>, Vec<u32>) {
    (0..6u32)
        .filter(|&g| COUNTS[c][g as usize] > 0)
        .map(|g| (g, COUNTS[c][g as usize]))
        .unzip()
}

/// The log-CPM norm of an entry, as every build must recompute it.
fn norm(c: usize, g: usize) -> f32 {
    let lib: u32 = COUNTS[c].iter().sum();
    libm::log1p((COUNTS[c][g] as f32 / lib as f32 * TARGET) as f64) as f32
}

/// Check one store pair against [`COUNTS`].
fn check(cells: &str, genes: &str, version: u32) {
    let cr = ParallelSparseReader::new(cells).unwrap();
    let gr = ParallelSparseReader::new(genes).unwrap();
    assert_eq!(cr.file_version(), version);
    assert_eq!(gr.file_version(), version);
    assert_eq!((cr.dims().total_cells, cr.dims().total_genes), (5, 6));
    assert_eq!(cr.target_size(), Some(TARGET));
    assert_eq!(cr.norm_kind(), NormKind::LogCpm);
    assert_eq!(cr.feature_kind(), FeatureKind::Gene);
    for c in 0..5 {
        let cell = cr.read_cell(c).unwrap();
        let (idx, raw) = row(c);
        assert_eq!(cell.indices, idx, "cell {c}");
        assert_eq!(cell.data_raw.iter().collect::<Vec<_>>(), raw, "cell {c}");
        for (k, &g) in idx.iter().enumerate() {
            assert_eq!(
                cell.data_norm[k].to_bits(),
                norm(c, g as usize).to_bits(),
                "cell {c}"
            );
        }
    }
    for (g, _) in COUNTS[0].iter().enumerate() {
        let gene = gr.read_gene(g).unwrap();
        let cells: Vec<u32> = (0..5u32).filter(|&c| COUNTS[c as usize][g] > 0).collect();
        assert_eq!(gene.indices, cells, "gene {g}");
        for (k, &c) in cells.iter().enumerate() {
            assert_eq!(gene.data_raw.get(k), COUNTS[c as usize][g]);
            assert_eq!(gene.data_norm[k].to_bits(), norm(c as usize, g).to_bits());
        }
    }
}

#[test]
fn test_golden_v4_store_reads() {
    check(&path("v4_cells.bin"), &path("v4_genes.bin"), 4);
}

#[test]
fn test_golden_v5_store_reads() {
    let (cells, genes) = (path("v5_cells.bin"), path("v5_genes.bin"));
    check(&cells, &genes, 5);
    let r = ParallelSparseReader::new(&cells).unwrap();
    assert!(r.cell_id_hash().is_some());
    assert_eq!(
        r.store_metadata(),
        vec![("fixture".to_string(), "golden".to_string())]
    );
    assert!(verify_store(&cells).unwrap());
    assert!(verify_store(&genes).unwrap());
}

/// Wrote the fixtures. Do not run to repair a failing test, see the module docs.
#[test]
#[ignore]
fn regenerate_golden_fixtures() {
    std::fs::create_dir_all(dir()).unwrap();
    for (version, cells, genes) in [
        (4, "v4_cells.bin", "v4_genes.bin"),
        (5, "v5_cells.bin", "v5_genes.bin"),
    ] {
        let (cells, genes) = (path(cells), path(genes));
        let mut w = if version == 4 {
            CellGeneSparseWriter::new_v4(&cells, true, 5, 6, TARGET).unwrap()
        } else {
            CellGeneSparseWriter::new(&cells, true, 5, 6, TARGET).unwrap()
        };
        for c in 0..5 {
            let (idx, raw) = row(c);
            w.write_cell_chunk(CsrCellChunk::from_data(&raw, &idx, c, TARGET, true))
                .unwrap();
        }
        w.finalise().unwrap();
        write_gene_file(&cells, &genes, None, false).unwrap();
        if version == 5 {
            let ids = ["c0", "c1", "c2", "c3", "c4"];
            set_cell_ids(&[&cells, &genes], &ids).unwrap();
            set_store_metadata(&[&cells, &genes], &[("fixture", "golden")]).unwrap();
        }
    }
}
