use faer::Mat;
use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::pca_svd::randomised_sparse_svd;
use crate::core::math::pca_svd::*;
use crate::core::math::sparse::sparse_svd_lanczos;
use crate::prelude::*;

/////////
// PCA //
/////////

/// Scales the data in a CSC chunk
///
/// ### Params
///
/// * `chunk` - The CscGeneChunk for which to scale the data
/// * `no_cells` - Number of cells represented
///
/// ### Returns
///
/// A densified, scaled vector per gene basis.
#[inline]
pub fn scale_csc_chunk(chunk: &CscGeneChunk, no_cells: usize) -> (Vec<f32>, f32, f32) {
    let mut dense_data = vec![0.0f32; no_cells];
    for (idx, &row_idx) in chunk.indices.iter().enumerate() {
        dense_data[row_idx as usize] = chunk.data_norm[idx].to_f32();
    }
    let mean = dense_data.iter().sum::<f32>() / no_cells as f32;
    let variance = dense_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / no_cells as f32;
    let std_dev = variance.sqrt();

    let scaled = if std_dev < 1e-8 {
        // Zero variance gene - just return centered data (all zeros after centering)
        vec![0.0f32; no_cells]
    } else {
        dense_data.iter().map(|&x| (x - mean) / std_dev).collect()
    };

    (scaled, mean, std_dev)
}

/// Calculate the PCs for single cell data
///
/// ### Params
///
/// * `f_path` - Path to the gene-based binary file.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `random_svd` - Shall randomised singular value decompostion be used. This
///   has the advantage of speed-ups, but loses precision.
/// * `return_scaled` - Return the scaled data.
/// * `seed` - Seed for randomised SVD.
///
/// ### Return
///
/// A tuple of the samples projected on thePC space, gene loadings and singular
/// values.
#[allow(clippy::too_many_arguments)]
pub fn pca_on_sc(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    random_svd: bool,
    seed: usize,
    return_scaled: bool,
    verbose: bool,
) -> (Mat<f32>, Mat<f32>, Vec<f32>, Option<Mat<f32>>) {
    let start_total = Instant::now();

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let start_reading = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices);

    let end_reading = start_reading.elapsed();

    if verbose {
        println!("Loaded in data : {:.2?}", end_reading);
    }

    let start_scaling = Instant::now();

    gene_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    let scaled_data: Vec<Vec<f32>> = gene_chunks
        .par_iter()
        .map(|chunk| {
            let (scaled, _, _) = scale_csc_chunk(chunk, cell_indices.len());
            scaled
        })
        .collect();

    let num_genes = scaled_data.len();
    let scaled_data = Mat::from_fn(cell_indices.len(), num_genes, |row, col| {
        scaled_data[col][row]
    });

    let end_scaling = start_scaling.elapsed();

    if verbose {
        println!("Finished scaling : {:.2?}", end_scaling);
    }

    let start_svd = Instant::now();

    let (scores, loadings, s) = if random_svd {
        let res: RandomSvdResults<f32> =
            randomised_svd(scaled_data.as_ref(), no_pcs, seed, Some(100_usize), None);
        // Take first no_pcs components and compute scores as X * V
        let loadings = res.v.submatrix(0, 0, num_genes, no_pcs).to_owned();
        let scores = &scaled_data * &loadings;
        (scores, loadings, res.s)
    } else {
        let res = scaled_data.thin_svd().unwrap();
        // Take only the first no_pcs components
        let loadings = res.V().submatrix(0, 0, num_genes, no_pcs).to_owned();
        let scores = &scaled_data * &loadings;
        let s: Vec<f32> = res
            .S()
            .column_vector()
            .iter()
            .take(no_pcs)
            .copied()
            .collect();
        (scores, loadings, s)
    };

    let end_svd = start_svd.elapsed();

    if verbose {
        println!("Finished PCA calculations : {:.2?}", end_svd);
    }

    let end_total = start_total.elapsed();

    if verbose {
        println!("Total run time PCA detection: {:.2?}", end_total);
    }

    let scaled = if return_scaled {
        Some(scaled_data)
    } else {
        None
    };

    (scores, loadings, s, scaled)
}

/// Calculate the PCs for single cell data
///
/// ### Params
///
/// * `f_path` - Path to the gene-based binary file.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `random_svd` - Shall randomised singular value decompostion be used. This
///   has the advantage of speed-ups, but loses precision.
/// * `return_scaled` - Return the scaled data.
/// * `seed` - Seed for randomised SVD.
///
/// ### Return
///
/// A tuple of the samples projected on thePC space, gene loadings and singular
/// values.
#[allow(clippy::too_many_arguments)]
pub fn pca_on_sc_sparse(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    random_svd: bool,
    seed: usize,
    verbose: bool,
) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
    let start_total = Instant::now();

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let start_reading = Instant::now();

    let reader = ParallelSparseReader::new(f_path).unwrap();
    let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(gene_indices);

    let end_reading = start_reading.elapsed();

    if verbose {
        println!("Loaded in data : {:.2?}", end_reading);
    }

    let n_cells = cell_set.len();

    let start_svd = Instant::now();

    gene_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    let csc = from_gene_chunks::<f32>(&gene_chunks, n_cells);

    let (scores, loadings, s) = if random_svd {
        let svd_res =
            randomised_sparse_svd::<f32, f32>(&csc, no_pcs, seed as u64, true, None, None);
        let scores = compute_pc_scores(&svd_res);
        (scores, svd_res.u().to_owned(), svd_res.s().to_vec())
    } else {
        let svd_res = sparse_svd_lanczos::<f32, f32, f32>(&csc, no_pcs, seed as u64, true);
        let scores = compute_pc_scores(&svd_res);
        (scores, svd_res.u().to_owned(), svd_res.s().to_vec())
    };

    let end_svd = start_svd.elapsed();

    if verbose {
        println!("Finished sparse PCA calculations : {:.2?}", end_svd);
    }

    let end_total = start_total.elapsed();

    if verbose {
        println!("Total run time sparse PCA detection: {:.2?}", end_total);
    }

    (scores, loadings, s)
}
