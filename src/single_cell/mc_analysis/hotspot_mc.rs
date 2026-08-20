//! HotSpot for meta cells, see DeTomaso and Yosef, Cell Syst., 2021.
//!
//! The algorithm is the single-cell one verbatim: everything in
//! [`crate::single_cell::sc_analysis::hotspot`] is generic over
//! [`SingleCellReading`] and only needs per-cell library sizes and gene chunks,
//! both of which [`InMemorySparseReader`] serves straight out of a
//! [`CompressedSparseData2`]. These entry points are the thin shim that builds
//! that reader and hands it over.
//!
//! Only the non-streaming drivers are exposed. The streaming variants exist to
//! bound disk re-reads, which is not a problem an in-memory matrix has.
//!
//! Gene module clustering needs no metacell twin:
//! [`hotspot_gene_clusters`](crate::single_cell::sc_analysis::hotspot::hotspot_gene_clusters)
//! takes a plain Z matrix and works unchanged.

use std::borrow::Cow;

use crate::prelude::*;
use crate::single_cell::sc_analysis::hotspot::{
    HotSpotGeneRes, HotSpotPairRes, HotSpotParams, Hotspot,
};
use crate::single_cell::sc_data::in_memory_io::InMemorySparseReader;

/////////////
// Helpers //
/////////////

/// Coerce the input to the gene-major orientation the reader needs.
///
/// ### Params
///
/// * `matrix` - The counts, shape (cells, genes)
///
/// ### Returns
///
/// The matrix as CSC, borrowed when it already was.
fn as_csc(matrix: &CompressedSparseData2<u32, f32>) -> Cow<'_, CompressedSparseData2<u32, f32>> {
    match matrix.cs_type {
        CompressedSparseFormat::Csc => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csr => Cow::Owned(matrix.transform()),
    }
}

/////////////////////
// Autocorrelation //
/////////////////////

/// Spatial autocorrelation of every requested gene over a metacell graph.
///
/// Geary's C and its Z-score under the chosen null model, as in
/// [`Hotspot::compute_all_genes`].
///
/// ### Params
///
/// * `matrix` - The metacell counts, shape (metacells, genes). Raw counts in
///   `data`, normalised counts in `data_2`. Either orientation is accepted.
/// * `neighbours` - Neighbour indices per metacell, self excluded
/// * `distances` - Neighbour distances matching `neighbours`, ascending
/// * `metacells_to_keep` - Indices of the metacells to analyse. Must be in the
///   same order and of the same length as `neighbours`.
/// * `genes_to_use` - Indices of the genes to analyse
/// * `params` - See [HotSpotParams]. `knn_params` is ignored here, since the
///   graph is supplied rather than built.
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [HotSpotGeneRes].
///
/// ### References
///
/// DeTomaso and Yosef, Cell Systems, 2021
pub fn hotspot_autocor_metacells(
    matrix: &CompressedSparseData2<u32, f32>,
    neighbours: &[Vec<usize>],
    distances: &[Vec<f32>],
    metacells_to_keep: &[usize],
    genes_to_use: &[usize],
    params: &HotSpotParams,
    verbose: usize,
) -> Result<HotSpotGeneRes, BixverseErrors> {
    let csc = as_csc(matrix);
    let reader = InMemorySparseReader::new(csc.as_ref(), None)?;

    let mut hotspot = Hotspot::new(
        &reader,
        &reader,
        metacells_to_keep,
        neighbours,
        distances,
        Some(params.graph_params),
    )?;

    hotspot.compute_all_genes(genes_to_use, &params.model, params.normalise, verbose)
}

//////////////
// Pairwise //
//////////////

/// Gene to gene local correlations over a metacell graph.
///
/// As in [`Hotspot::compute_gene_cor`], and with the same memory caveat: three
/// dense `n_metacells x n_genes` blocks are live at once, so keep
/// `genes_to_use` to the panel actually of interest rather than the whole
/// transcriptome.
///
/// ### Params
///
/// * `matrix` - The metacell counts, shape (metacells, genes). Raw counts in
///   `data`, normalised counts in `data_2`. Either orientation is accepted.
/// * `neighbours` - Neighbour indices per metacell, self excluded
/// * `distances` - Neighbour distances matching `neighbours`, ascending
/// * `metacells_to_keep` - Indices of the metacells to analyse. Must be in the
///   same order and of the same length as `neighbours`.
/// * `genes_to_use` - Indices of the genes to analyse
/// * `params` - See [HotSpotParams]. `knn_params` and `normalise` are both
///   unused on this path, matching the single-cell entry point.
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [HotSpotPairRes], holding the correlation and Z-score matrices.
///
/// ### References
///
/// DeTomaso and Yosef, Cell Systems, 2021
pub fn hotspot_gene_cor_metacells(
    matrix: &CompressedSparseData2<u32, f32>,
    neighbours: &[Vec<usize>],
    distances: &[Vec<f32>],
    metacells_to_keep: &[usize],
    genes_to_use: &[usize],
    params: &HotSpotParams,
    verbose: usize,
) -> Result<HotSpotPairRes, BixverseErrors> {
    let csc = as_csc(matrix);
    let reader = InMemorySparseReader::new(csc.as_ref(), None)?;

    let mut hotspot = Hotspot::new(
        &reader,
        &reader,
        metacells_to_keep,
        neighbours,
        distances,
        Some(params.graph_params),
    )?;

    hotspot.compute_gene_cor(genes_to_use, &params.model, verbose)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;
    use half::f16;

    use crate::single_cell::sc_analysis::hotspot::{HotSpotGraphParams, hotspot_gene_clusters};
    use crate::single_cell::sc_data::data_io::{
        CellGeneSparseWriter, CscGeneChunk, CsrCellChunk, ParallelSparseReader, RawCounts,
    };
    use crate::single_cell::sc_traits::F16;

    const N_CELLS: usize = 24;
    const N_GENES: usize = 12;
    const SIZE_FACTOR: f32 = 1e4;

    /// Removes the scratch stores on drop. Errors are ignored: the file may
    /// already be gone, and this runs during unwind.
    struct TempStore(std::path::PathBuf);

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempStore {
        /// Reserve a uniquely named scratch store in the system temp directory.
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_hotspot_mc_{name}.bin")))
        }

        /// Path of the guarded store.
        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// Deterministic `dense[gene][cell]` counts in the hundreds, so the metacell
    /// regime rather than the single-cell one gets exercised.
    fn synthetic_counts() -> Vec<Vec<u32>> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut dense = vec![vec![0u32; N_CELLS]; N_GENES];

        for (gene, row) in dense.iter_mut().enumerate() {
            for value in row.iter_mut() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let draw = (state >> 33) as u32;
                // every third gene is sparse, the rest dense like a metacell
                *value = if gene % 3 == 0 && !draw.is_multiple_of(4) {
                    0
                } else {
                    draw % 400 + 1
                };
            }
        }

        dense
    }

    /// The same counts as a (cells, genes) CSC matrix with both layers. The
    /// normalised layer is quantised through `f16` exactly as the writer does,
    /// so the on-disk and in-memory paths see identical inputs.
    fn to_csc(dense: &[Vec<u32>], lib_sizes: &[f32]) -> CompressedSparseData2<u32, f32> {
        let mut data: Vec<u32> = Vec::new();
        let mut data_2: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = vec![0];

        for gene in dense.iter() {
            for (cell, &value) in gene.iter().enumerate() {
                if value > 0 {
                    data.push(value);
                    let norm = (value as f32 / lib_sizes[cell] * SIZE_FACTOR).ln_1p();
                    data_2.push(f16::from_f32(norm).to_f32());
                    indices.push(cell as u32);
                }
            }
            indptr.push(data.len() as u32);
        }

        CompressedSparseData2::new_csc(&data, &indices, &indptr, Some(&data_2), (N_CELLS, N_GENES))
    }

    /// Write the gene-major and cell-major stores the disk-backed reference needs.
    fn write_stores(dense: &[Vec<u32>], genes: &TempStore, cells: &TempStore) {
        let mut gene_writer =
            CellGeneSparseWriter::new(genes.path(), false, N_CELLS, N_GENES, SIZE_FACTOR)
                .expect("gene writer opens");

        let lib_sizes = library_sizes(dense);

        for (gene_idx, gene) in dense.iter().enumerate() {
            let mut raw = Vec::new();
            let mut idx = Vec::new();
            let mut norms = Vec::new();
            for (cell, &value) in gene.iter().enumerate() {
                if value > 0 {
                    raw.push(value);
                    idx.push(cell);
                    let norm = (value as f32 / lib_sizes[cell] * SIZE_FACTOR).ln_1p();
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
            let mut raw = Vec::new();
            let mut idx = Vec::new();
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

    /// Total counts per cell.
    fn library_sizes(dense: &[Vec<u32>]) -> Vec<f32> {
        (0..N_CELLS)
            .map(|cell| dense.iter().map(|gene| gene[cell] as f32).sum())
            .collect()
    }

    /// A ring graph over the cells, so every node has two neighbours and every
    /// edge is reciprocal. Distances ascend within each row, as every kNN
    /// producer in the crate guarantees.
    fn ring_graph() -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let neighbours: Vec<Vec<usize>> = (0..N_CELLS)
            .map(|i| vec![(i + 1) % N_CELLS, (i + N_CELLS - 1) % N_CELLS])
            .collect();
        let distances: Vec<Vec<f32>> = (0..N_CELLS)
            .map(|i| vec![0.5 + (i % 3) as f32 * 0.25, 1.25])
            .collect();

        (neighbours, distances)
    }

    /// Parameters for a given null model, on an unweighted graph.
    fn params_for(model: &str) -> HotSpotParams {
        HotSpotParams {
            model: model.to_string(),
            normalise: true,
            knn_params: KnnParams::default(),
            graph_params: HotSpotGraphParams::default(),
        }
    }

    /// The whole point of the adapter: identical numbers whether the counts
    /// come off disk or out of memory, for every null model.
    #[test]
    fn test_metacell_autocor_matches_the_disk_backed_path() {
        let dense = synthetic_counts();
        let lib_sizes = library_sizes(&dense);
        let matrix = to_csc(&dense, &lib_sizes);

        let gene_store = TempStore::new("autocor_genes");
        let cell_store = TempStore::new("autocor_cells");
        write_stores(&dense, &gene_store, &cell_store);

        let gene_reader = ParallelSparseReader::new(gene_store.path()).expect("gene reader opens");
        let cell_reader = ParallelSparseReader::new(cell_store.path()).expect("cell reader opens");

        let cells: Vec<usize> = (0..N_CELLS).collect();
        let genes: Vec<usize> = (0..N_GENES).collect();
        let (neighbours, distances) = ring_graph();

        for model in ["danb", "bernoulli", "normal"] {
            for graph_params in [
                HotSpotGraphParams::default(),
                HotSpotGraphParams::new(true, 3.0, false),
            ] {
                let params = HotSpotParams {
                    model: model.to_string(),
                    normalise: true,
                    knn_params: KnnParams::default(),
                    graph_params,
                };

                let expected = {
                    let mut hotspot = Hotspot::new(
                        &gene_reader,
                        &cell_reader,
                        &cells,
                        &neighbours,
                        &distances,
                        Some(graph_params),
                    )
                    .expect("disk hotspot builds");
                    hotspot
                        .compute_all_genes(&genes, model, params.normalise, 0)
                        .expect("disk autocor runs")
                };

                let actual = hotspot_autocor_metacells(
                    &matrix,
                    &neighbours,
                    &distances,
                    &cells,
                    &genes,
                    &params,
                    0,
                )
                .expect("metacell autocor runs");

                assert_eq!(actual.gene_idx, expected.gene_idx, "model {model}");
                for i in 0..expected.gene_idx.len() {
                    assert_relative_eq!(actual.c[i], expected.c[i], epsilon = 1e-5);
                    assert_relative_eq!(actual.z[i], expected.z[i], epsilon = 1e-5);
                    assert_relative_eq!(actual.fdr[i], expected.fdr[i], epsilon = 1e-5);
                }
            }
        }
    }

    /// Same equivalence for the pair path, plus the clustering it feeds.
    #[test]
    fn test_metacell_gene_cor_matches_the_disk_backed_path() {
        let dense = synthetic_counts();
        let lib_sizes = library_sizes(&dense);
        let matrix = to_csc(&dense, &lib_sizes);

        let gene_store = TempStore::new("genecor_genes");
        let cell_store = TempStore::new("genecor_cells");
        write_stores(&dense, &gene_store, &cell_store);

        let gene_reader = ParallelSparseReader::new(gene_store.path()).expect("gene reader opens");
        let cell_reader = ParallelSparseReader::new(cell_store.path()).expect("cell reader opens");

        let cells: Vec<usize> = (0..N_CELLS).collect();
        let genes: Vec<usize> = (0..N_GENES).collect();
        let (neighbours, distances) = ring_graph();
        let params = HotSpotParams {
            model: "danb".to_string(),
            normalise: true,
            knn_params: KnnParams::default(),
            graph_params: HotSpotGraphParams::new(true, 3.0, false),
        };

        let expected = {
            let mut hotspot = Hotspot::new(
                &gene_reader,
                &cell_reader,
                &cells,
                &neighbours,
                &distances,
                Some(params.graph_params),
            )
            .expect("disk hotspot builds");
            hotspot
                .compute_gene_cor(&genes, &params.model, 0)
                .expect("disk gene cor runs")
        };

        let actual = hotspot_gene_cor_metacells(
            &matrix,
            &neighbours,
            &distances,
            &cells,
            &genes,
            &params,
            0,
        )
        .expect("metacell gene cor runs");

        for i in 0..N_GENES {
            for j in 0..N_GENES {
                assert_relative_eq!(actual.cor[(i, j)], expected.cor[(i, j)], epsilon = 1e-5);
                assert_relative_eq!(
                    actual.z_scores[(i, j)],
                    expected.z_scores[(i, j)],
                    epsilon = 1e-5
                );
            }
        }

        // the clustering takes the Z matrix as-is, no metacell twin needed
        let z = Mat::<f64>::from_fn(N_GENES, N_GENES, |i, j| actual.z_scores[(i, j)] as f64);
        let assignment = hotspot_gene_clusters(z.as_ref(), 0.5, 2);
        assert_eq!(assignment.len(), N_GENES);
    }

    /// A CSR input has to give the same answer as its CSC twin.
    #[test]
    fn test_csr_input_matches_csc_input() {
        let dense = synthetic_counts();
        let lib_sizes = library_sizes(&dense);
        let csc = to_csc(&dense, &lib_sizes);
        let csr = csc.transform();

        let cells: Vec<usize> = (0..N_CELLS).collect();
        let genes: Vec<usize> = (0..N_GENES).collect();
        let (neighbours, distances) = ring_graph();
        let params = params_for("danb");

        let from_csc =
            hotspot_autocor_metacells(&csc, &neighbours, &distances, &cells, &genes, &params, 0)
                .expect("csc runs");

        let from_csr =
            hotspot_autocor_metacells(&csr, &neighbours, &distances, &cells, &genes, &params, 0)
                .expect("csr runs");

        assert_eq!(from_csr.gene_idx, from_csc.gene_idx);
        for i in 0..from_csc.gene_idx.len() {
            assert_relative_eq!(from_csr.z[i], from_csc.z[i], epsilon = 1e-5);
        }
    }
}
