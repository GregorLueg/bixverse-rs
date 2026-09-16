#![cfg(all(feature = "single-cell", feature = "large_scale_diagnostics"))]
//! Print-only scaling sweep for scTransform v2.
//!
//! The claim this exists to check is that every stage after the step-1 fit is a
//! per-gene reduction, so the algorithm's own memory is flat in the cell count
//! where the R implementation's is linear in it (`vst.R:370` preallocates a
//! dense genes-by-cells residual matrix). Nothing here asserts; read the table
//! and pair it with an external peak-RSS measurement:
//!
//! ```text
//! /usr/bin/time -l ./target/release/deps/sctransform_scaling-* --nocapture
//! ```
//!
//! Read that peak with care. [`ParallelSparseReader`] populates its mmap for a
//! store under 8 GiB, so resident file pages dominate RSS and say nothing about
//! what the code allocates. Measured at a fixed 160,000 cells on an M-series
//! Mac, varying only `SCT_GENES` so the store size moves and the cell count
//! does not:
//!
//! ```text
//! genes    store      peak RSS
//!   500   326 MB       1.92 GB
//!  1000   649 MB       3.79 GB
//!  4000  2602 MB       8.31 GB
//! ```
//!
//! Peak RSS tracks the store, not the cells, and at the top end it is within a
//! few percent of the three stores this sweep holds open at once (the input,
//! the corrected output and its cell-major transpose). The algorithm's own
//! allocations are the per-worker residual row plus a handful of gene-length
//! vectors, which at 160,000 cells is tens of megabytes.

use std::time::Instant;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_data::bin_merge_io::gene_store_to_cell_store;
use bixverse_rs::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CscGeneChunk, ParallelSparseReader, RawCounts,
};
use bixverse_rs::single_cell::sctransform::model::SctParams;
use bixverse_rs::single_cell::sctransform::stream::{
    SctStreamOpts, fit_sctransform, sct_corrected_counts, sct_residual_variance,
};

/// Cell counts swept. The step-1 fit is pinned at 2000 cells throughout, so if
/// anything downstream were quadratic in the cell count it would show here.
///
/// Overridable with `SCT_CELLS`, as a comma-separated list, so that a peak-RSS
/// run can isolate a single configuration instead of reporting the largest of
/// three.
const CELL_COUNTS: [usize; 3] = [10_000, 40_000, 160_000];

/// Genes in every sweep, held fixed so only the cell axis varies.
///
/// Overridable with `SCT_GENES`. Varying this at a fixed cell count is what
/// separates the algorithm's own memory from the store's: the reader populates
/// its mmap, so resident file pages track the store size rather than anything
/// the code allocates.
const N_GENES: usize = 4_000;

/// Resolves the swept cell counts, honouring `SCT_CELLS`.
///
/// ### Returns
///
/// The cell counts to sweep.
fn cell_counts() -> Vec<usize> {
    match std::env::var("SCT_CELLS") {
        Ok(v) => v.split(',').filter_map(|p| p.trim().parse().ok()).collect(),
        Err(_) => CELL_COUNTS.to_vec(),
    }
}

/// Resolves the gene count, honouring `SCT_GENES`.
///
/// ### Returns
///
/// The number of genes to generate.
fn gene_count() -> usize {
    std::env::var("SCT_GENES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(N_GENES)
}

/// Removes the scratch stores when the sweep ends.
struct TempStore(std::path::PathBuf);

impl Drop for TempStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempStore {
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!("bixverse_sct_scale_{name}.bin")))
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }

    fn size_mb(&self) -> f64 {
        std::fs::metadata(&self.0).map(|m| m.len()).unwrap_or(0) as f64 / 1e6
    }
}

/// Writes a synthetic gene-major store without ever holding the matrix.
///
/// Counts come from an LCG through `u^3`, the same heavy-tailed draw the parity
/// fixture uses, so the genes are genuinely overdispersed and the model has
/// something to fit.
fn write_synthetic_store(path: &str, n_genes: usize, n_cells: usize) -> Vec<f64> {
    let mut state = 20_260_101_u64;
    let mut next = || {
        state = (1_664_525_u64
            .wrapping_mul(state)
            .wrapping_add(1_013_904_223))
            % 4_294_967_296;
        state as f64 / 4_294_967_296.0
    };

    let lib_scale: Vec<f64> = (0..n_cells).map(|_| 2000.0 + 6000.0 * next()).collect();
    let gene_rate: Vec<f64> = (0..n_genes)
        .map(|_| 10.0_f64.powf(-5.0 + 3.2 * next()))
        .collect();

    let mut writer =
        CellGeneSparseWriter::new(path, false, n_cells, n_genes, 1e4).expect("writer opens");
    let mut library_sizes = vec![0.0_f64; n_cells];

    for (gene_idx, &rate) in gene_rate.iter().enumerate() {
        let mut raw = Vec::new();
        let mut indices = Vec::new();
        for (cell, &lib) in lib_scale.iter().enumerate() {
            let u = next();
            let value = (u * u * u * rate * lib * 12.0).floor() as u32;
            if value > 0 {
                raw.push(value);
                indices.push(cell);
                library_sizes[cell] += value as f64;
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
    library_sizes
}

/// Times each stage across a range of cell counts.
#[test]
// 4000 genes by up to 160,000 cells, written to disk three times over.
fn diagnostic_sctransform_scaling() {
    println!(
        "\n{:>9} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "cells", "store_MB", "write_s", "fit_s", "resvar_s", "correct_s", "transp_s"
    );

    let n_genes = gene_count();

    for &n_cells in cell_counts().iter() {
        let store = TempStore::new(&format!("in_{n_cells}"));
        let corrected = TempStore::new(&format!("corr_{n_cells}"));
        let cell_major = TempStore::new(&format!("cells_{n_cells}"));

        let t = Instant::now();
        let library_sizes = write_synthetic_store(store.path(), n_genes, n_cells);
        let write_s = t.elapsed().as_secs_f64();

        let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
        let cells: Vec<usize> = (0..n_cells).collect();
        let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.max(1.0).log10()).collect();
        let params = SctParams::default();
        let opts = SctStreamOpts::default();

        let t = Instant::now();
        let (model, _) =
            fit_sctransform(&reader, &cells, &library_sizes, &params, None, None, opts)
                .expect("fit");
        let fit_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let rv = sct_residual_variance(&reader, &model, &cells, &log10_umi, opts)
            .expect("residual variance");
        let resvar_s = t.elapsed().as_secs_f64();
        assert_eq!(rv.len(), model.len());

        let t = Instant::now();
        sct_corrected_counts(&reader, &model, &cells, &log10_umi, corrected.path(), opts)
            .expect("corrected counts");
        let correct_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        gene_store_to_cell_store(corrected.path(), cell_major.path(), 20_000, 512, 0)
            .expect("transpose");
        let transp_s = t.elapsed().as_secs_f64();

        println!(
            "{:>9} {:>10.1} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            n_cells,
            store.size_mb(),
            write_s,
            fit_s,
            resvar_s,
            correct_s,
            transp_s
        );
    }

    println!(
        "\nThe step-1 fit is pinned at {} cells, so `fit_s` should flatten out \
         once the sweep passes it while the per-gene stages stay linear.\n",
        SctParams::default().n_cells
    );
}
