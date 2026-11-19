//! Export a store built from an anndata-written h5ad and read the result back
//! through the ingest reader: `X`, obs and var must come back as anndata wrote
//! them.

use bixverse_sc_data::ann_store::AnnFile;
use bixverse_sc_data::data_io::{MinCellQuality, ParallelSparseReader};
use bixverse_sc_data::h5ad_io::{RawDataSlot, write_h5_counts};
use bixverse_sc_export::ad_export::{Column, ColumnData, ObsTable, VarTable};
use bixverse_sc_export::prelude::*;
use std::path::{Path, PathBuf};

const N_CELLS: usize = 30;
const N_GENES: usize = 12;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/tests/fixtures/anndata_csr.h5ad")
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("bixverse_export_{}_{name}", std::process::id()))
}

/// Dense cells x genes from a CSR `X` (h5ad or Zarr).
fn dense_x(f: &AnnFile) -> Vec<f64> {
    let indptr = f.indptr("X/indptr").unwrap();
    let indices: Vec<usize> = f.read_1d("X/indices").unwrap();
    let data: Vec<f64> = f.read_1d("X/data").unwrap();
    let n_genes = N_GENES;
    let mut out = vec![0.0; (indptr.len() - 1) * n_genes];
    for r in 0..indptr.len() - 1 {
        for k in indptr[r]..indptr[r + 1] {
            out[r * n_genes + indices[k]] = data[k];
        }
    }
    out
}

/// The cell file of the fixture, every cell and gene kept.
fn store() -> PathBuf {
    let bin = tmp("cells.bin");
    let qc = MinCellQuality {
        min_unique_genes: 0,
        min_lib_size: 0,
        min_cells: 0,
        target_size: 1e4,
    };
    let (n, g, _) = write_h5_counts(
        fixture(),
        bin.clone(),
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

fn parts(cells: &[usize]) -> AdParts {
    AdParts {
        obs: ObsTable {
            index: cells.iter().map(|c| format!("cell_{c}")).collect(),
            columns: vec![Column::new(
                "n",
                ColumnData::Int(cells.iter().map(|&c| c as i32).collect()),
            )],
        },
        var: VarTable {
            index: (0..N_GENES).map(|g| format!("gene_{g}")).collect(),
            columns: Vec::new(),
        },
        ..Default::default()
    }
}

fn check(out: &Path, format: AdFormat) {
    let reader = ParallelSparseReader::new(store().to_str().unwrap()).unwrap();
    // a subset in a new order, over several batches
    let cells: Vec<usize> = (0..N_CELLS).rev().step_by(2).collect();
    save_store(&reader, out, format, &cells, false, parts(&cells), 4).unwrap();

    let src = dense_x(&AnnFile::open(fixture()).unwrap());
    let got_file = AnnFile::open(out).unwrap();
    let got = dense_x(&got_file);
    for (r, &c) in cells.iter().enumerate() {
        assert_eq!(
            got[r * N_GENES..(r + 1) * N_GENES],
            src[c * N_GENES..(c + 1) * N_GENES],
            "{format:?} cell {c}"
        );
    }
    let obs = got_file.read_strings("obs/_index").unwrap();
    assert_eq!(obs, parts(&cells).obs.index);
}

#[test]
fn test_save_store_h5ad_round_trip() {
    let out = tmp("out.h5ad");
    check(
        &out,
        AdFormat::H5ad {
            compression: Some(6),
        },
    );
    let _ = std::fs::remove_file(out);
}

#[cfg(feature = "zarr")]
#[test]
fn test_save_store_zarr_round_trip() {
    let out = tmp("out.zarr");
    let _ = std::fs::remove_dir_all(&out);
    check(&out, AdFormat::Zarr);
    let _ = std::fs::remove_dir_all(out);
}

#[test]
fn test_bad_parts_fail_before_the_file_exists() {
    let reader = ParallelSparseReader::new(store().to_str().unwrap()).unwrap();
    let out = tmp("bad.h5ad");
    let _ = std::fs::remove_file(&out);
    let cells = [0, 1, 2];
    let mut p = parts(&cells);
    p.var.index.pop();
    let err = save_store(
        &reader,
        &out,
        AdFormat::H5ad { compression: None },
        &cells,
        false,
        p,
        2,
    )
    .unwrap_err();
    assert!(err.to_string().contains("var index"), "{err}");
    let err = save_store(
        &reader,
        &out,
        AdFormat::H5ad { compression: None },
        &[N_CELLS],
        false,
        parts(&[N_CELLS]),
        2,
    )
    .unwrap_err();
    assert!(err.to_string().contains("outside"), "{err}");
    assert!(!out.exists());
}
