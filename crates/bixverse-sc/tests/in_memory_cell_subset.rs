//! Cell subsets read from the in-memory store against the gene-major path,
//! including the HVG and scTransform results that sit on top.

use half::f16;

use bixverse_sc::sc_processing::hvg::get_hvg_vst;
use bixverse_sc::sc_processing::sctransform::model::{SctCovariates, SctParams};
use bixverse_sc::sc_processing::sctransform::stream::{SctStreamOpts, fit_sctransform};
use bixverse_sc_data::data_io::CellGeneSparseWriter;
use bixverse_sc_data::in_memory_io::*;
use bixverse_sc_data::prelude::*;

const N_CELLS: usize = 300;
const N_GENES: usize = 60;
const SIZE_FACTOR: f32 = 1e4;

/// Removes the scratch store on drop; errors ignored, this runs during
/// unwind.
struct TempStore(std::path::PathBuf);

impl Drop for TempStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempStore {
    /// Reserve a uniquely named scratch store.
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!(
            "bixverse_cell_subset_{name}_{}.bin",
            std::process::id()
        )))
    }

    /// Path of the guarded store.
    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// Sparse `dense[gene][cell]` counts from an LCG, mean rising with the
/// gene index so the mean-variance trend has range.
fn synthetic_counts() -> Vec<Vec<u32>> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..N_GENES)
        .map(|gene| {
            (0..N_CELLS)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let draw = (state >> 33) as u32;
                    if draw % 10 < 6 {
                        0
                    } else {
                        draw % (gene as u32 + 3)
                    }
                })
                .collect()
        })
        .collect()
}

/// Write the gene-major and cell-major stores for the same counts.
fn write_stores(dense: &[Vec<u32>], genes: &TempStore, cells: &TempStore) {
    let lib: Vec<f32> = (0..N_CELLS)
        .map(|c| dense.iter().map(|g| g[c] as f32).sum::<f32>().max(1.0))
        .collect();

    let mut gene_writer =
        CellGeneSparseWriter::new(genes.path(), false, N_CELLS, N_GENES, SIZE_FACTOR)
            .expect("gene writer opens");
    for (gene_idx, gene) in dense.iter().enumerate() {
        let (mut raw, mut idx, mut norms) = (Vec::new(), Vec::new(), Vec::new());
        for (cell, &value) in gene.iter().enumerate() {
            if value > 0 {
                raw.push(value);
                idx.push(cell);
                let norm = (value as f32 / lib[cell] * SIZE_FACTOR).ln_1p();
                norms.push(F16::from(f16::from_f32(norm)));
            }
        }
        gene_writer
            .write_gene_chunk(CscGeneChunk::from_conversion(
                RawCounts::from_u32_auto(&raw),
                &norms,
                &idx,
                gene_idx,
                true,
            ))
            .expect("write gene chunk");
    }
    gene_writer.finalise().expect("finalise genes");

    let mut cell_writer =
        CellGeneSparseWriter::new(cells.path(), true, N_CELLS, N_GENES, SIZE_FACTOR)
            .expect("cell writer opens");
    for cell in 0..N_CELLS {
        let (mut raw, mut idx) = (Vec::new(), Vec::new());
        for (gene_idx, gene) in dense.iter().enumerate() {
            if gene[cell] > 0 {
                raw.push(gene[cell]);
                idx.push(gene_idx as u32);
            }
        }
        cell_writer
            .write_cell_chunk(CsrCellChunk::from_data(&raw, &idx, cell, SIZE_FACTOR, true))
            .expect("write cell chunk");
    }
    cell_writer.finalise().expect("finalise cells");
}

/// An unordered subset, so the row order of both paths is exercised.
fn subset() -> Vec<usize> {
    let mut cells: Vec<usize> = (0..N_CELLS).filter(|c| c % 3 != 1).collect();
    cells.reverse();
    cells.swap(0, 50);
    cells
}

/// The in-memory subset holds exactly what the filtered gene read sees.
#[test]
fn test_cell_subset_matches_filtered_gene_read() {
    let (genes, cells) = (TempStore::new("read_g"), TempStore::new("read_c"));
    let dense = synthetic_counts();
    write_stores(&dense, &genes, &cells);
    let gene_reader = ParallelSparseReader::new(genes.path()).unwrap();
    let cell_reader = ParallelSparseReader::new(cells.path()).unwrap();

    let subset = subset();
    let csc = read_cell_subset_csc(&cell_reader, &subset).unwrap();
    assert_eq!(csc.shape, (subset.len(), N_GENES));

    let keep: indexmap::IndexSet<u32> = subset.iter().map(|&c| c as u32).collect();
    let all: Vec<usize> = (0..N_GENES).collect();
    for chunk in gene_reader
        .read_gene_parallel_filtered(&all, &keep)
        .unwrap()
    {
        let g = chunk.original_index;
        let (lo, hi) = (csc.indptr[g] as usize, csc.indptr[g + 1] as usize);
        assert_eq!(&csc.indices[lo..hi], &chunk.indices[..]);
        assert_eq!(
            csc.data[lo..hi].to_vec(),
            chunk.data_raw.iter().collect::<Vec<u32>>()
        );
    }
}

/// VST HVG and the scTransform fit give identical results on both paths.
#[test]
fn test_cell_subset_hvg_and_sct_match_gene_path() {
    let (genes, cells) = (TempStore::new("fit_g"), TempStore::new("fit_c"));
    let dense = synthetic_counts();
    write_stores(&dense, &genes, &cells);
    let gene_reader = ParallelSparseReader::new(genes.path()).unwrap();
    let cell_reader = ParallelSparseReader::new(cells.path()).unwrap();

    let subset = subset();
    let local: Vec<usize> = (0..subset.len()).collect();
    let csc = read_cell_subset_csc(&cell_reader, &subset).unwrap();
    let mem = InMemorySparseReader::new(&csc, cell_reader.target_size()).unwrap();

    let hvg_gene = get_hvg_vst(&gene_reader, &subset, 0.3, None, 0).unwrap();
    let hvg_cell = get_hvg_vst(&mem, &local, 0.3, None, 0).unwrap();
    assert_eq!(hvg_gene.var_std, hvg_cell.var_std);
    assert_eq!(hvg_gene.mean, hvg_cell.mean);

    let lib: Vec<f64> = cell_reader
        .read_cell_library_sizes(&subset)
        .unwrap()
        .into_iter()
        .map(|s| s as f64)
        .collect();
    let (params, opts, cov) = (
        SctParams::default(),
        SctStreamOpts::default(),
        SctCovariates::default(),
    );
    let (m_gene, _) =
        fit_sctransform(&gene_reader, &subset, &lib, &cov, &params, None, None, opts).unwrap();
    let (m_cell, _) = fit_sctransform(&mem, &local, &lib, &cov, &params, None, None, opts).unwrap();
    assert_eq!(format!("{m_gene:?}"), format!("{m_cell:?}"));
}

/// Both limits bind: the fraction on small stores, the cap on big ones.
#[test]
fn test_prefer_cell_subset_boundaries() {
    assert!(prefer_cell_subset(100, 1_000));
    assert!(!prefer_cell_subset(101, 1_000));
    assert!(prefer_cell_subset(CELL_SUBSET_MAX_CELLS, 10_000_000));
    assert!(!prefer_cell_subset(CELL_SUBSET_MAX_CELLS + 1, 10_000_000));
}
