use faer::Mat;
use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::pca_svd::randomised_sparse_svd;
use crate::core::math::pca_svd::*;
use crate::core::math::sparse::sparse_svd_lanczos;
use crate::prelude::*;
use crate::utils::simd::*;

/////////////
// Helpers //
/////////////

/// Enum representing the type of SVD to use for PCA analysis in single cell
#[derive(Clone, Debug)]
pub enum SvdType {
    /// Dense SVD solving with scaling
    Dense { randomised: bool },
    /// Sparse SVD solving without scaling
    Sparse { randomised: bool },
}

/// Default implementation for SvdType
impl Default for SvdType {
    fn default() -> SvdType {
        SvdType::Dense { randomised: true }
    }
}

/// Parse the SVD type to use
///
/// ### Params
///
/// * `s` - The string representation of the SVD type
/// * `randomised` - Whether to use randomised SVD
///
/// ### Returns
///
/// An Option containing the parsed SVD type, or None if the input is invalid
pub fn parse_svd_type(s: &str, randomised: bool) -> Option<SvdType> {
    match s.to_lowercase().as_str() {
        "dense" => Some(SvdType::Dense { randomised }),
        "sparse" => Some(SvdType::Sparse { randomised }),
        _ => None,
    }
}

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
    let mut dense_data = vec![0_f32; no_cells];
    for (idx, &row_idx) in chunk.indices.iter().enumerate() {
        dense_data[row_idx as usize] = chunk.data_norm[idx].to_f32();
    }
    let mean = sum_simd_f32(&dense_data) / no_cells as f32;
    let variance = variance_simd_f32(&dense_data, mean) / (no_cells as f32 - 1.0);
    let std_dev = variance.sqrt();

    let scaled = if std_dev < 1e-8 {
        // Zero variance gene - just return centered data (all zeros after centering)
        vec![0_f32; no_cells]
    } else {
        dense_data.iter().map(|&x| (x - mean) / std_dev).collect()
    };

    (scaled, mean, std_dev)
}

/// Compute column means from a CSC sparse matrix using SIMD-accelerated
/// summation.
///
/// Divides the sum of non-zero values per column by `n_rows`,
/// accounting for structural zeros.
///
/// ### Params
///
/// * `csc` - The CSC sparse matrix.
/// * `use_second_layer` - Whether to use the second layer of data.
///
/// ### Returns
///
/// The column means.
pub fn sparse_csc_column_means(
    csc: &CompressedSparseData<f32>,
    use_second_layer: bool,
) -> Vec<f32> {
    assert!(
        matches!(csc.cs_type, CompressedSparseFormat::Csc),
        "Expected CSC format"
    );
    let (n, m) = csc.shape;
    let n_f = n as f32;

    let values: &[f32] = if use_second_layer {
        csc.data_2
            .as_ref()
            .expect("data_2 is None but use_second_layer is true")
    } else {
        &csc.data
    };

    (0..m)
        .into_par_iter()
        .map(|j| {
            let start = csc.indptr[j];
            let end = csc.indptr[j + 1];
            sum_simd_f32(&values[start..end]) / n_f
        })
        .collect()
}

/// Calculates the standard deviation of a CSC sparse matrix chunk
///
/// ### Params
///
/// * `csc` - The CSC sparse matrix.
/// * `col_means` - The column means.
/// * `use_second_layer` - Whether to use the second layer of data.
///
/// ### Returns
///
/// The standard deviation.
pub fn sparse_csc_column_stds(
    csc: &CompressedSparseData<f32>,
    col_means: &[f32],
    use_second_layer: bool,
) -> Vec<f32> {
    assert!(
        matches!(csc.cs_type, CompressedSparseFormat::Csc),
        "Expected CSC format"
    );
    let (n, m) = csc.shape;
    let n_f = n as f32;
    let values: &[f32] = if use_second_layer {
        csc.data_2
            .as_ref()
            .expect("data_2 is None but use_second_layer is true")
    } else {
        &csc.data
    };
    (0..m)
        .into_par_iter()
        .map(|j| {
            let start = csc.indptr[j];
            let end = csc.indptr[j + 1];
            let slice = &values[start..end];
            let sq_sum: f32 = sum_squares_simd_f32(slice);
            let variance = (sq_sum - n_f * col_means[j] * col_means[j]) / (n_f - 1.0);
            variance.max(0.0).sqrt().max(f32::EPSILON)
        })
        .collect()
}

///////////////
// Dense PCA //
///////////////

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

////////////////
// Sparse PCA //
////////////////

/// Calculate the PCs for single cell data (sparse)
///
/// This version does NOT scale the data and avoids densifying the data at any
/// point, avoiding holding a large matrix in memory. This comes at the cost of
/// the first principal component being largely driven by average gene
/// expression, but makes analysing large datasets actually tractable. For
/// the non random version a sparse SVD Lanczos algorithm is used. For the
/// randomised version, it uses the randomised SVD approach in which the dense
/// matrix is of much smaller size than the potentially massive scaled, dense
/// data.
///
/// ### Params
///
/// * `f_path` - Path to the gene-based binary file.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `random_svd` - Shall randomised sparse singular value decompostion be
///   used. This has the advantage of speed-ups, but loses precision.
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

    let start_data_prep = Instant::now();

    let n_cells = cell_set.len();

    gene_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    let csc = from_gene_chunks::<f32>(&gene_chunks, n_cells);

    let end_data_prep = start_data_prep.elapsed();

    let col_means = sparse_csc_column_means(&csc, true);
    let col_stds = sparse_csc_column_stds(&csc, &col_means, true);

    if verbose {
        println!("Finished the data preparations : {:.2?}", end_data_prep);
    }

    let start_svd = Instant::now();

    let (scores, loadings, s) = if random_svd {
        let svd_res = randomised_sparse_svd::<f32, f32>(
            &csc,
            no_pcs,
            seed as u64,
            true,
            Some(100_usize),
            None,
            Some(&col_means),
            Some(&col_stds),
        );
        let scores = compute_pc_scores(&svd_res);
        (
            scores.submatrix(0, 0, cell_set.len(), no_pcs).to_owned(),
            svd_res
                .v()
                .submatrix(0, 0, gene_indices.len(), no_pcs)
                .to_owned(),
            svd_res.s().to_vec(),
        )
    } else {
        let svd_res = sparse_svd_lanczos::<f32, f32, f32>(
            &csc,
            no_pcs,
            seed as u64,
            true,
            Some(&col_means),
            Some(&col_stds),
        );
        let scores = compute_pc_scores(&svd_res);
        (scores, svd_res.v().to_owned(), svd_res.s().to_vec())
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
