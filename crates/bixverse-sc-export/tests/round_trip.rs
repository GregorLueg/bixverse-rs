//! Export a store built from an anndata-written h5ad and read the result back
//! through the ingest reader: `X`, obs and var must come back as anndata wrote
//! them.

use bixverse_sc_data::h5::ann_store::AnnFile;
use bixverse_sc_data::h5::h5ad::{RawDataSlot, write_h5_counts};
use bixverse_sc_data::store::{quality::MinCellQuality, reader::ParallelSparseReader};
use bixverse_sc_export::ad_export::{Column, ColumnData, ObsTable, VarTable};
use bixverse_sc_export::prelude::*;
use std::path::{Path, PathBuf};
use tempfile::{NamedTempFile, TempPath, tempdir};

const N_CELLS: usize = 30;
const N_GENES: usize = 12;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/tests/fixtures/anndata_csr.h5ad")
}

/// Dense cells x genes from a CSR `X` (h5ad or Zarr).
fn dense_x(f: &AnnFile, n_genes: usize) -> Vec<f64> {
    let indptr = f.indptr("X/indptr").unwrap();
    let indices: Vec<usize> = f.read_1d("X/indices").unwrap();
    let data: Vec<f64> = f.read_1d("X/data").unwrap();
    let mut out = vec![0.0; (indptr.len() - 1) * n_genes];
    for r in 0..indptr.len() - 1 {
        for k in indptr[r]..indptr[r + 1] {
            out[r * n_genes + indices[k]] = data[k];
        }
    }
    out
}

/// The cell file of the fixture, every cell and gene kept.
fn store() -> TempPath {
    let bin = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
    let qc = MinCellQuality {
        min_unique_genes: 0,
        min_lib_size: 0,
        min_cells: 0,
        target_size: 1e4,
    };
    let (n, g, _) = write_h5_counts(
        fixture(),
        bin.to_path_buf(),
        "csr",
        N_CELLS,
        N_GENES,
        qc,
        RawDataSlot::DataX,
        false,
    )
    .unwrap();
    assert_eq!((n, g), (N_CELLS, N_GENES));
    bin
}

fn parts(cells: &[usize], genes: &[usize]) -> AdParts {
    AdParts {
        obs: ObsTable {
            index: cells.iter().map(|c| format!("cell_{c}")).collect(),
            columns: vec![Column::new(
                "n",
                ColumnData::Int(cells.iter().map(|&c| c as i32).collect()),
            )],
        },
        var: VarTable {
            index: genes.iter().map(|g| format!("gene_{g}")).collect(),
            columns: Vec::new(),
        },
        ..Default::default()
    }
}

/// Export a cell subset, and optionally a gene subset, and compare with the
/// fixture.
fn check(out: &Path, format: AdFormat, genes: Option<&[usize]>) {
    let bin = store();
    let reader = ParallelSparseReader::new(bin.to_str().unwrap()).unwrap();
    // a subset in a new order, over several batches
    let cells: Vec<usize> = (0..N_CELLS).rev().step_by(2).collect();
    let all: Vec<usize> = (0..N_GENES).collect();
    let cols = genes.unwrap_or(&all);
    let p = parts(&cells, cols);
    save_store(&reader, out, format, &cells, genes, false, p, 4).unwrap();

    let src = dense_x(&AnnFile::open(fixture()).unwrap(), N_GENES);
    let got_file = AnnFile::open(out).unwrap();
    let got = dense_x(&got_file, cols.len());
    for (r, &c) in cells.iter().enumerate() {
        for (j, &g) in cols.iter().enumerate() {
            assert_eq!(
                got[r * cols.len() + j],
                src[c * N_GENES + g],
                "{format:?} cell {c} gene {g}"
            );
        }
    }
    let obs = got_file.read_strings("obs/_index").unwrap();
    assert_eq!(obs, parts(&cells, cols).obs.index);
    let var = got_file.read_strings("var/_index").unwrap();
    assert_eq!(var, parts(&cells, cols).var.index);
}

/// A gene subset out of store order.
const GENES: [usize; 5] = [7, 3, 0, 11, 5];

#[test]
fn test_save_store_h5ad_round_trip() {
    let dir = tempdir().unwrap();
    let h5ad = AdFormat::H5ad {
        compression: Some(6),
    };
    check(&dir.path().join("out.h5ad"), h5ad, None);
    check(&dir.path().join("genes.h5ad"), h5ad, Some(&GENES));
}

#[cfg(feature = "zarr")]
#[test]
fn test_save_store_zarr_round_trip() {
    let dir = tempdir().unwrap();
    check(&dir.path().join("out.zarr"), AdFormat::Zarr, None);
    check(&dir.path().join("genes.zarr"), AdFormat::Zarr, Some(&GENES));
}

#[test]
fn test_bad_parts_fail_before_the_file_exists() {
    let bin = store();
    let reader = ParallelSparseReader::new(bin.to_str().unwrap()).unwrap();
    let dir = tempdir().unwrap();
    let out = dir.path().join("bad.h5ad");
    let cells = [0, 1, 2];
    let all: Vec<usize> = (0..N_GENES).collect();
    let fmt = AdFormat::H5ad { compression: None };
    let mut p = parts(&cells, &all);
    p.var.index.pop();
    let err = save_store(&reader, &out, fmt, &cells, None, false, p, 2).unwrap_err();
    assert!(err.to_string().contains("var index"), "{err}");
    let p = parts(&[N_CELLS], &all);
    let err = save_store(&reader, &out, fmt, &[N_CELLS], None, false, p, 2).unwrap_err();
    assert!(err.to_string().contains("outside"), "{err}");
    let p = parts(&cells, &[1, 1]);
    let err = save_store(&reader, &out, fmt, &cells, Some(&[1, 1]), false, p, 2).unwrap_err();
    assert!(err.to_string().contains("repeated"), "{err}");
    assert!(!out.exists());
}
