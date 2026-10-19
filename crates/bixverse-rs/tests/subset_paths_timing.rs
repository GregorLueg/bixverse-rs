//! Gene-major vs cell-subset timings for HVG and the scTransform fit.
//!
//! Reads a real store from `BIXVERSE_TIMING_STORE` (a directory holding
//! `counts_cells.bin` and `counts_genes.bin`) and is skipped when it is unset.
//! Sizes come from `BIXVERSE_TIMING_SIZES`, comma separated cell counts.
//! Prints only, so it sits behind `large_scale_diagnostics`.

#![cfg(all(feature = "single-cell", feature = "large_scale_diagnostics"))]

use std::path::PathBuf;
use std::time::Instant;

use indexmap::IndexSet;
use rand::prelude::*;

use bixverse_rs::single_cell::sc_data::data_io::{ParallelSparseReader, SingleCellReading};
use bixverse_rs::single_cell::sc_data::in_memory_io::{InMemorySparseReader, read_cell_subset_csc};
use bixverse_rs::single_cell::sc_processing::hvg::get_hvg_vst;
use bixverse_rs::single_cell::sc_processing::sctransform::model::{SctCovariates, SctParams};
use bixverse_rs::single_cell::sc_processing::sctransform::stream::{
    SctStreamOpts, fit_sctransform, sct_gene_pass,
};

/// Genes per block for the decompression-only sweep.
const BLOCK: usize = 2000;

/// Sorted random sample of `n` cells out of `total`.
///
/// ### Params
///
/// * `total` - Cells in the store
/// * `n` - Cells to draw
///
/// ### Returns
///
/// Ascending cell indices.
fn sample_cells(total: usize, n: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(1);
    let mut cells = rand::seq::index::sample(&mut rng, total, n).into_vec();
    cells.sort_unstable();
    cells
}

#[test]
// Real 850k-cell store, minutes per size.
fn timing_subset_paths() {
    let Ok(dir) = std::env::var("BIXVERSE_TIMING_STORE") else {
        eprintln!("BIXVERSE_TIMING_STORE unset, skipping");
        return;
    };
    let dir = PathBuf::from(dir);
    let sizes: Vec<usize> = std::env::var("BIXVERSE_TIMING_SIZES")
        .unwrap_or_else(|_| "100,1000,6000".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    let genes = ParallelSparseReader::new(dir.join("counts_genes.bin").to_str().unwrap()).unwrap();
    let cells = ParallelSparseReader::new(dir.join("counts_cells.bin").to_str().unwrap()).unwrap();
    let header = genes.get_header();
    let n_total = header.total_cells;
    let n_genes = header.total_genes;
    println!(
        "store: {} cells, {} genes, {} threads",
        n_total,
        n_genes,
        rayon::current_num_threads()
    );

    let params = SctParams::default();
    let opts = SctStreamOpts::default();
    let covariates = SctCovariates::default();
    let all_genes: Vec<usize> = (0..n_genes).collect();

    for &n in &sizes {
        let subset = sample_cells(n_total, n);
        let cell_set: IndexSet<u32> = subset.iter().map(|&c| c as u32).collect();
        let lib: Vec<f64> = cells
            .read_cell_library_sizes(&subset)
            .unwrap()
            .into_iter()
            .map(|s| s as f64)
            .collect();

        // gene path, stage split
        let t = Instant::now();
        for block in all_genes.chunks(BLOCK) {
            std::hint::black_box(genes.read_gene_parallel(block).unwrap());
        }
        let g_decomp = t.elapsed();

        let t = Instant::now();
        for block in all_genes.chunks(BLOCK) {
            std::hint::black_box(genes.read_gene_parallel_filtered(block, &cell_set).unwrap());
        }
        let g_filtered = t.elapsed();

        let t = Instant::now();
        let hvg_gene = get_hvg_vst(&genes, &subset, 0.3, None, 0).unwrap();
        let g_hvg = t.elapsed();

        let t = Instant::now();
        std::hint::black_box(sct_gene_pass(&genes, &subset, &params, opts).unwrap());
        let g_pass = t.elapsed();

        let t = Instant::now();
        let (model_gene, _) = fit_sctransform(
            &genes,
            &subset,
            &lib,
            &covariates,
            &params,
            None,
            None,
            opts,
        )
        .unwrap();
        let g_fit = t.elapsed();

        // cell path
        let t = Instant::now();
        let csc = read_cell_subset_csc(&cells, &subset).unwrap();
        let c_load = t.elapsed();
        let mem = InMemorySparseReader::new(&csc, cells.target_size()).unwrap();
        let local: Vec<usize> = (0..n).collect();

        let t = Instant::now();
        let hvg_cell = get_hvg_vst(&mem, &local, 0.3, None, 0).unwrap();
        let c_hvg = t.elapsed();

        let t = Instant::now();
        let (model_cell, _) =
            fit_sctransform(&mem, &local, &lib, &covariates, &params, None, None, opts).unwrap();
        let c_fit = t.elapsed();

        let lib_mem = mem.read_cell_library_sizes(&local).unwrap();
        let lib_agree = lib_mem
            .iter()
            .zip(lib.iter())
            .filter(|(a, b)| **a as f64 == **b)
            .count();

        println!(
            "n={n:>7} nnz={:>11} | gene: decomp {:>7.2?} filtered {:>7.2?} hvg {:>7.2?} pass {:>7.2?} fit {:>7.2?} | cell: load {:>7.2?} hvg {:>7.2?} fit {:>7.2?}",
            csc.data.len(),
            g_decomp,
            g_filtered,
            g_hvg,
            g_pass,
            g_fit,
            c_load,
            c_hvg,
            c_fit
        );
        println!(
            "          parity: hvg var_std identical {} | model genes identical {} | theta identical {} | lib sizes agree {}/{}",
            hvg_gene.var_std == hvg_cell.var_std,
            model_gene.genes == model_cell.genes,
            format!("{:?}", model_gene) == format!("{:?}", model_cell),
            lib_agree,
            n
        );
    }
}
