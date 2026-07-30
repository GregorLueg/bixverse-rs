//! Single cell-related PCA functions. Implements dense, sparse version of the
//! normal SVD and also randomised SVD. The upcasts to f64 are needed due to
//! nasty floating operation errors that can accumulate over time with large
//! data sets.

use faer::Mat;
use half::f16;
use indexmap::IndexSet;
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::pca_svd::randomised_sparse_svd;
use crate::core::math::pca_svd::*;
use crate::core::math::sparse::sparse_svd_lanczos;
use crate::prelude::*;

///////////
// Types //
///////////

/// Single cell related PCA result
///
/// ### Fields
///
/// * `0` - Scores
/// * `1` - Loadings
/// * `2` - Eigenvalues
/// * `3` - Optional scaled values (input into the SVD)
pub type SingleCellPcaResScaled =
    Result<(Mat<f32>, Mat<f32>, Vec<f32>, Option<Mat<f32>>), BixverseErrors>;

/// Single cell related PCA result
///
/// ### Fields
///
/// * `0` - Scores
/// * `1` - Loadings
/// * `2` - Eigenvalues
/// * `3` - Optional scaled values (input into the SVD)
/// * `4` - The feature means
/// * `5` - The feature SDs
pub type SingleCellPcaResScaledStats = Result<
    (
        Mat<f32>,
        Mat<f32>,
        Vec<f32>,
        Option<Mat<f32>>,
        Vec<f32>,
        Vec<f32>,
    ),
    BixverseErrors,
>;

/// Single cell related PCA result
///
/// ### Fields
///
/// * `0` - Scores
/// * `1` - Loadings
/// * `2` - Eigenvalues
pub type SingleCellPcaRes = Result<(Mat<f32>, Mat<f32>, Vec<f32>), BixverseErrors>;

/// Single cell related PCA result
///
/// ### Fields
///
/// * `0` - Scores
/// * `1` - Loadings
/// * `2` - Eigenvalues
/// * `3` - The feature means
/// * `4` - The feature SDs
pub type SingleCellPcaResStats =
    Result<(Mat<f32>, Mat<f32>, Vec<f32>, Vec<f32>, Vec<f32>), BixverseErrors>;

////////////
// Params //
////////////

/// Parameters for the main single cell PCA around normalisations and if the
/// randomised, approximate path shall be used.
#[derive(Clone, Debug)]
pub struct SingleCellPcaParams {
    /// Mean center the data
    pub mean_center: bool,
    /// Normalise the variance
    pub normalise_variance: bool,
    /// Shall an approximate, randomised SVD be used
    pub randomised: bool,
    /// Apply the CLR transformation
    pub clr: bool,
    /// Size factor
    pub size_factor: f32,
}

impl SingleCellPcaParams {
    /// Generate a new instance of [SingleCellPcaParams]
    ///
    /// ### Params
    ///
    /// * `mean_center` - Shall the data be mean centered
    /// * `normalise_variance` - Shall the variance be normalised
    /// * `randomised` - Shall fast approximate randomised SVD be used
    /// * `clr` - Shall the CLR transformation be used, see Booeshaghi, et al.,
    ///   bioRxive, 2026.
    /// * `size_factor` - The used size factor for preparing everything for the
    ///   clr transformation
    ///
    /// ### Returns
    ///
    /// Initialised self.
    pub fn new(
        mean_center: bool,
        normalise_variance: bool,
        randomised: bool,
        clr: bool,
        size_factor: f32,
    ) -> Self {
        Self {
            mean_center,
            normalise_variance,
            randomised,
            clr,
            size_factor,
        }
    }
}

/// Default implementation for [SingleCellPcaParams]
impl Default for SingleCellPcaParams {
    fn default() -> Self {
        Self {
            mean_center: true,
            normalise_variance: true,
            randomised: true,
            clr: false,
            size_factor: 1e4,
        }
    }
}

/////////////
// Helpers //
/////////////

impl CscGeneChunk {
    /// Convert the normalised counts from the `log1p(u * sf)` scale to the
    /// `log1p(u)` scale, enabling downstream use of the shifted CLR (PFlog1pPF)
    /// transformation. Mutates `data_norm` in place and refreshes `avg_exp`.
    ///
    /// ### Params
    ///
    /// * `size_factor` - The size factor used in the original normalisation
    ///   (e.g. 1e4 for CP10k).
    pub fn transform_to_clr(&mut self, size_factor: f32) {
        let sf_inv = 1.0_f32 / size_factor;
        let mut new_sum = 0.0_f32;
        for v in self.data_norm.iter_mut() {
            let new_val = (v.to_f32().exp_m1() * sf_inv).ln_1p();
            new_sum += new_val;
            *v = F16::from(f16::from_f32(new_val));
        }
        self.avg_exp = F16::from(f16::from_f32(new_sum));
    }
}

/// Scales the data in a CSC chunk
///
/// ### Params
///
/// * `chunk` - The CscGeneChunk for which to scale the data
/// * `no_cells` - Number of cells represented
/// * `mean_center` - Boolean. Shall the mean be subtracted (mean = 0).
/// * `normalise_variance` - Boolean. Shall the genes be scaled to unit
///   variance.
/// * `row_offsets` - Optional per-row offsets used by the shifted CLR
///   transformation.
///
/// ### Returns
///
/// Tuple of (densified vector, mean, standard deviation). The latter two will
/// be 0.0 pending if you set `mean_center=true` and/or
/// `normalise_variance=true`.
pub fn scale_csc_chunk(
    chunk: &CscGeneChunk,
    no_cells: usize,
    mean_center: bool,
    normalise_variance: bool,
    row_offsets: Option<&[f64]>,
) -> (Vec<f32>, f32, f32) {
    let mut dense_data = vec![0_f32; no_cells];
    for (idx, &row_idx) in chunk.indices.iter().enumerate() {
        dense_data[row_idx as usize] = chunk.data_norm[idx].to_f32();
    }

    if let Some(off) = row_offsets {
        for i in 0..no_cells {
            dense_data[i] -= off[i] as f32;
        }
    }

    let n = no_cells as f64;

    match (mean_center, normalise_variance) {
        (false, false) => (dense_data, 0.0, 0.0),
        (true, false) => {
            let mean = (dense_data.iter().map(|&x| x as f64).sum::<f64>() / n) as f32;
            let scaled = dense_data.iter().map(|&x| x - mean).collect();
            (scaled, mean, 0.0)
        }
        (false, true) => {
            let mean_f64 = dense_data.iter().map(|&x| x as f64).sum::<f64>() / n;
            let mean = mean_f64 as f32;
            let std_dev = (dense_data
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean_f64;
                    d * d
                })
                .sum::<f64>()
                / (n - 1.0))
                .sqrt() as f32;
            let scaled = if std_dev < 1e-8 {
                vec![0_f32; no_cells]
            } else {
                dense_data.iter().map(|&x| x / std_dev).collect()
            };
            (scaled, mean, std_dev)
        }
        (true, true) => {
            let mean_f64 = dense_data.iter().map(|&x| x as f64).sum::<f64>() / n;
            let mean = mean_f64 as f32;
            let std_dev = (dense_data
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean_f64;
                    d * d
                })
                .sum::<f64>()
                / (n - 1.0))
                .sqrt() as f32;
            let scaled = if std_dev < 1e-8 {
                vec![0_f32; no_cells]
            } else {
                dense_data.iter().map(|&x| (x - mean) / std_dev).collect()
            };
            (scaled, mean, std_dev)
        }
    }
}

/// Compute column means from a CSC sparse matrix using SIMD-accelerated
/// summation. When `row_offsets` is provided, returns column means of the
/// row-offset-corrected (CLR) matrix, i.e. `μ_j(A) - mean(offsets)`.
///
/// ### Params
///
/// * `csc` - The CSC sparse matrix.
/// * `use_second_layer` - Whether to use the second layer of data.
/// * `row_offsets` - Optional per-row offsets used by the shifted CLR
///   transformation.
///
/// ### Returns
///
/// The column means.
pub fn sparse_csc_column_means(
    csc: &CompressedSparseData2<f32>,
    use_second_layer: bool,
    row_offsets: Option<&[f64]>,
) -> Result<Vec<f64>, BixverseErrors> {
    if !csc.cs_type.is_csc() {
        return Err(BixverseErrors::SparseMatrixMustBeCsc);
    }

    let (n, m) = csc.shape;
    let n_f = n as f64;
    let values: &[f32] = if use_second_layer {
        csc.data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?
            .as_slice()
    } else {
        &csc.data
    };

    let m_bar = row_offsets
        .map(|off| off.iter().sum::<f64>() / n_f)
        .unwrap_or(0.0);

    let res = (0..m)
        .into_par_iter()
        .map(|j| {
            let start = csc.indptr[j] as usize;
            let end = csc.indptr[j + 1] as usize;
            let sum: f64 = values[start..end].iter().map(|&x| x as f64).sum();
            sum / n_f - m_bar
        })
        .collect();

    Ok(res)
}

/// Calculate column standard deviations of a CSC sparse matrix. When
/// `row_offsets` is provided, returns standard deviations of the
/// row-offset-corrected (CLR) matrix. `col_means` must be on the same
/// scale (i.e. computed with the same `row_offsets`).
///
/// ### Params
///
/// * `csc` - The CSC sparse matrix.
/// * `col_means` - The column means (CLR-adjusted if `row_offsets` is set).
/// * `use_second_layer` - Whether to use the second layer of data.
/// * `row_offsets` - Optional per-row offsets used by the shifted CLR
///   transformation.
///
/// ### Returns
///
/// The standard deviations.
pub fn sparse_csc_column_stds(
    csc: &CompressedSparseData2<f32>,
    col_means: &[f64],
    use_second_layer: bool,
    row_offsets: Option<&[f64]>,
) -> Result<Vec<f64>, BixverseErrors> {
    if !csc.cs_type.is_csc() {
        return Err(BixverseErrors::SparseMatrixMustBeCsc);
    }

    let (n, m) = csc.shape;
    let n_f = n as f64;
    let values: &[f32] = if use_second_layer {
        csc.data_2
            .as_ref()
            .ok_or(BixverseErrors::Data2NotAvailable)?
            .as_slice()
    } else {
        &csc.data
    };

    let (m_bar, var_m, m_centered) = if let Some(off) = row_offsets {
        let m_bar = off.iter().sum::<f64>() / n_f;
        let m_centered: Vec<f64> = off.iter().map(|&v| v - m_bar).collect();
        let var_m = m_centered.iter().map(|&v| v * v).sum::<f64>() / (n_f - 1.0);
        (m_bar, var_m, Some(m_centered))
    } else {
        (0.0, 0.0, None)
    };

    let res = (0..m)
        .into_par_iter()
        .map(|j| {
            let start = csc.indptr[j] as usize;
            let end = csc.indptr[j + 1] as usize;
            let nnz = end - start;

            // col_means[j] is μ_j(CLR) = μ_j(A) - m_bar; recover μ_j(A).
            let mu_a = col_means[j] + m_bar;

            let slice = &values[start..end];
            let indices = &csc.indices[start..end];

            let ss_nonzero: f64 = slice
                .iter()
                .map(|&x| {
                    let d = x as f64 - mu_a;
                    d * d
                })
                .sum();
            let ss_zeros = (n - nnz) as f64 * mu_a * mu_a;
            let var_a = (ss_nonzero + ss_zeros) / (n_f - 1.0);

            let variance = if let Some(mc) = &m_centered {
                let cov_term: f64 = slice
                    .iter()
                    .zip(indices.iter())
                    .map(|(&x, &i)| x as f64 * mc[i as usize])
                    .sum();
                var_a - (2.0 / (n_f - 1.0)) * cov_term + var_m
            } else {
                var_a
            };

            variance.max(0.0).sqrt().max(f64::EPSILON)
        })
        .collect();

    Ok(res)
}

///////////////
// Dense PCA //
///////////////

/// Worker for dense PCA
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `params_pca` - The parameters for this single cell PCA run, see
///   [SingleCellPcaParams].
/// * `params_pca` - Parameters for this PCA run, see [SingleCellPcaParams]
/// * `clr_offsets` - Pre-computed CLR offsets if you want to use the CLR
///   transformation here.
/// * `return_scaled` - Return the scaled data.
/// * `seed` - Seed for randomised SVD.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Return
///
/// The [SingleCellPcaResScaledStats]
#[allow(clippy::too_many_arguments)]
fn dense_pca<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    clr_offsets: Option<&[f64]>,
    seed: usize,
    return_scaled: bool,
    verbose: usize,
) -> SingleCellPcaResScaledStats {
    if params_pca.clr && clr_offsets.is_none() {
        return Err(BixverseErrors::OffsetsNotProvidedForClrPCA);
    }
    if params_pca.clr
        && let Some(offs) = clr_offsets
        && offs.len() != cell_indices.len()
    {
        return Err(BixverseErrors::OffsetsLengthDoesNotMatchNCells {
            len_offset: offs.len(),
            n_cells: cell_indices.len(),
        });
    }

    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let start_reading = Instant::now();

    let mut gene_chunks: Vec<CscGeneChunk> =
        reader.read_gene_parallel_filtered(gene_indices, &cell_set)?;

    if params_pca.clr {
        gene_chunks
            .par_iter_mut()
            .for_each(|chunk| chunk.transform_to_clr(params_pca.size_factor));
    }

    let end_reading = start_reading.elapsed();

    if verbosity.normal_verbosity() {
        println!("PCA: Loaded in data in {:.2?}", end_reading);
    }

    let start_scaling = Instant::now();

    let scaled_data: Vec<(Vec<f32>, f32, f32)> = gene_chunks
        .par_iter()
        .map(|chunk| {
            let (scaled, mean, sd) = scale_csc_chunk(
                chunk,
                cell_indices.len(),
                params_pca.mean_center,
                params_pca.normalise_variance,
                clr_offsets,
            );
            (scaled, mean, sd)
        })
        .collect();

    let mut feature_means = Vec::with_capacity(scaled_data.len());
    let mut feature_sds = Vec::with_capacity(scaled_data.len());

    for (_, mean, sd) in scaled_data.iter() {
        feature_means.push(*mean);
        feature_sds.push(*sd);
    }

    // manual drop
    drop(gene_chunks);

    let num_genes = scaled_data.len();
    let n_cells = cell_indices.len();

    // Build f64 matrix for numerically stable SVD
    let scaled_f64 = Mat::<f64>::from_fn(n_cells, num_genes, |row, col| {
        scaled_data[col].0[row] as f64
    });

    // Also build f32 if needed for return or score computation
    let scaled_f32 = if return_scaled {
        Some(Mat::<f32>::from_fn(n_cells, num_genes, |row, col| {
            scaled_data[col].0[row]
        }))
    } else {
        None
    };

    drop(scaled_data);

    let end_scaling = start_scaling.elapsed();

    if verbosity.normal_verbosity() {
        println!("PCA: Finished scaling in {:.2?}", end_scaling);
    }

    let start_svd = Instant::now();

    let (scores, loadings, s) = if params_pca.randomised {
        let res: RandomSvdResults<f64> =
            randomised_svd(scaled_f64.as_ref(), no_pcs, seed, Some(100_usize), None)?;
        let loadings = Mat::<f32>::from_fn(num_genes, no_pcs, |i, j| res.v[(i, j)] as f32);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| (res.u[(i, j)] * res.s[j]) as f32);
        let s: Vec<f32> = res.s[..no_pcs].iter().map(|&x| x as f32).collect();
        (scores, loadings, s)
    } else {
        let res = scaled_f64
            .thin_svd()
            .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;
        let loadings = Mat::<f32>::from_fn(num_genes, no_pcs, |i, j| res.V()[(i, j)] as f32);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| {
            (res.U()[(i, j)] * res.S().column_vector()[j]) as f32
        });
        let s: Vec<f32> = res
            .S()
            .column_vector()
            .iter()
            .take(no_pcs)
            .map(|&x| x as f32)
            .collect();
        (scores, loadings, s)
    };

    let end_svd = start_svd.elapsed();

    if verbosity.normal_verbosity() {
        println!("PCA: Finished calculations in {:.2?}", end_svd);
    }

    let end_total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!("PCA: Total run time -> {:.2?}", end_total);
    }

    Ok((scores, loadings, s, scaled_f32, feature_means, feature_sds))
}

/// Calculate the PCs for single cell data
///
/// This uses a dense path under the hood that can be faster than the implicit
/// centering, scaling via matrix algebra.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `params_pca` - The parameters for this single cell PCA run, see
///   [SingleCellPcaParams].
/// * `params_pca` - Parameters for this PCA run, see [SingleCellPcaParams]
/// * `clr_offsets` - Pre-computed CLR offsets if you want to use the CLR
///   transformation here.
/// * `return_scaled` - Return the scaled data.
/// * `seed` - Seed for randomised SVD.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Return
///
/// The [SingleCellPcaResScaled]
#[allow(clippy::too_many_arguments)]
pub fn pca_on_sc<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    clr_offsets: Option<&[f64]>,
    seed: usize,
    return_scaled: bool,
    verbose: usize,
) -> SingleCellPcaResScaled {
    let (scores, loadings, s, scaled_f32, _, _) = dense_pca(
        reader,
        cell_indices,
        gene_indices,
        no_pcs,
        params_pca,
        clr_offsets,
        seed,
        return_scaled,
        verbose,
    )?;

    Ok((scores, loadings, s, scaled_f32))
}

/// Calculate the PCs for single cell data (with stats)
///
/// This uses a dense path under the hood that can be faster than the implicit
/// centering, scaling via matrix algebra. Additionally, it returns the
/// feature means and standard deviations for downstream consumption.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `params_pca` - The parameters for this single cell PCA run, see
///   [SingleCellPcaParams].
/// * `params_pca` - Parameters for this PCA run, see [SingleCellPcaParams]
/// * `clr_offsets` - Pre-computed CLR offsets if you want to use the CLR
///   transformation here.
/// * `seed` - Seed for randomised SVD.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Return
///
/// The [SingleCellPcaResScaled]
#[allow(clippy::too_many_arguments)]
pub fn pca_on_sc_stats<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    clr_offsets: Option<&[f64]>,
    seed: usize,
    verbose: usize,
) -> SingleCellPcaResStats {
    let (scores, loadings, s, _, feature_means, feature_sds) = dense_pca(
        reader,
        cell_indices,
        gene_indices,
        no_pcs,
        params_pca,
        clr_offsets,
        seed,
        false,
        verbose,
    )?;

    Ok((scores, loadings, s, feature_means, feature_sds))
}

///////////////////////
// Streaming version //
///////////////////////

/// Calculate the PCs for single cell data using a streaming approach
///
/// Processes genes in batches to reduce peak memory usage. This is particularly
/// useful when working with a large number of genes but a small cell subsample,
/// as it avoids loading all sparse gene chunks simultaneously.
///
/// The dense scaled matrix is still fully held in memory for SVD, but the
/// sparse data is loaded and discarded batch by batch.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate.
/// * `params_pca` - The parameters for this single cell PCA run, see
///   [SingleCellPcaParams].
/// * `params_pca` - Parameters for this PCA run, see [SingleCellPcaParams]
/// * `clr_offsets` - Pre-computed CLR offsets if you want to use the CLR
///   transformation here.
/// * `seed` - Seed for randomised SVD.
/// * `return_scaled` - Return the scaled data.
/// * `gene_batch_size` - Number of genes to load per batch.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Return
///
/// A tuple of the samples projected on the PC space, gene loadings, singular
/// values, and optionally the scaled data.
#[allow(clippy::too_many_arguments)]
pub fn pca_on_sc_streaming<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    clr_offsets: Option<&[f64]>,
    seed: usize,
    return_scaled: bool,
    gene_batch_size: usize,
    verbose: usize,
) -> SingleCellPcaResScaled {
    if params_pca.clr && clr_offsets.is_none() {
        return Err(BixverseErrors::OffsetsNotProvidedForClrPCA);
    }
    if params_pca.clr
        && let Some(offs) = clr_offsets
        && offs.len() != cell_indices.len()
    {
        return Err(BixverseErrors::OffsetsLengthDoesNotMatchNCells {
            len_offset: offs.len(),
            n_cells: cell_indices.len(),
        });
    }
    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = cell_indices.len();
    let n_genes = gene_indices.len();
    let num_batches = n_genes.div_ceil(gene_batch_size);


    let mut scaled_matrix = Mat::<f64>::zeros(n_cells, n_genes);

    let start_scaling = Instant::now();

    for batch_idx in 0..num_batches {
        if verbosity.normal_verbosity() {
            println!(
                "PCA (streaming): Scaling batch {}/{} ({} genes each)",
                batch_idx + 1,
                num_batches,
                gene_batch_size
            );
        }

        let start_gene = batch_idx * gene_batch_size;
        let end_gene = ((batch_idx + 1) * gene_batch_size).min(n_genes);
        let batch_gene_indices = &gene_indices[start_gene..end_gene];

        let start_loading = Instant::now();
        let mut gene_chunks = reader.read_gene_parallel_filtered(batch_gene_indices, &cell_set)?;

        if params_pca.clr {
            gene_chunks
                .par_iter_mut()
                .for_each(|chunk| chunk.transform_to_clr(params_pca.size_factor));
        }

        if verbosity.detailed_verbosity() {
            println!("  Loaded batch in: {:.2?}", start_loading.elapsed());
        }

        let batch_scaled: Vec<Vec<f32>> = gene_chunks
            .par_iter()
            .map(|chunk| {
                let (scaled, _, _) = scale_csc_chunk(
                    chunk,
                    n_cells,
                    params_pca.mean_center,
                    params_pca.normalise_variance,
                    clr_offsets,
                );
                scaled
            })
            .collect();

        for (local_col, scaled_col) in batch_scaled.iter().enumerate() {
            let global_col = start_gene + local_col;
            for (row, &val) in scaled_col.iter().enumerate() {
                scaled_matrix[(row, global_col)] = val as f64;
            }
        }

        drop(gene_chunks);
    }

    if verbosity.normal_verbosity() {
        println!(
            "PCA (streaming): finished scaling in {:.2?}",
            start_scaling.elapsed()
        );
    }

    let start_svd = Instant::now();

    let (scores, loadings, s) = if params_pca.randomised {
        let res: RandomSvdResults<f64> =
            randomised_svd(scaled_matrix.as_ref(), no_pcs, seed, Some(100_usize), None)?;
        let loadings = Mat::<f32>::from_fn(n_genes, no_pcs, |i, j| res.v[(i, j)] as f32);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| (res.u[(i, j)] * res.s[j]) as f32);
        let s: Vec<f32> = res.s[..no_pcs].iter().map(|&x| x as f32).collect();
        (scores, loadings, s)
    } else {
        let res = scaled_matrix
            .thin_svd()
            .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;
        let loadings = Mat::<f32>::from_fn(n_genes, no_pcs, |i, j| res.V()[(i, j)] as f32);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| {
            (res.U()[(i, j)] * res.S().column_vector()[j]) as f32
        });
        let s: Vec<f32> = res
            .S()
            .column_vector()
            .iter()
            .take(no_pcs)
            .map(|&x| x as f32)
            .collect();
        (scores, loadings, s)
    };

    if verbosity.normal_verbosity() {
        println!(
            "PCA (streaming): finished calculations in {:.2?}",
            start_svd.elapsed()
        );
        println!(
            "PCA (streaming): total run time -> {:.2?}",
            start_total.elapsed()
        );
    }

    let scaled = if return_scaled {
        Some(Mat::<f32>::from_fn(n_cells, n_genes, |i, j| {
            scaled_matrix[(i, j)] as f32
        }))
    } else {
        None
    };

    Ok((scores, loadings, s, scaled))
}

////////////////
// Sparse PCA //
////////////////

/// Worker for sparse PCA
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `params_pca` - The parameters for this single cell PCA run, see
///   [SingleCellPcaParams].
/// * `clr_offsets` - Pre-computed CLR offsets if you want to use the CLR
///   transformation here.
/// * `seed` - Seed for randomised SVD.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Return
///
/// The [SingleCellPcaResStats]
#[allow(clippy::too_many_arguments)]
fn sparse_pca<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    clr_offsets: Option<&[f64]>,
    seed: usize,
    verbose: usize,
) -> SingleCellPcaResStats {
    if params_pca.clr && clr_offsets.is_none() {
        return Err(BixverseErrors::OffsetsNotProvidedForClrPCA);
    }
    if params_pca.clr
        && let Some(offs) = clr_offsets
        && offs.len() != cell_indices.len()
    {
        return Err(BixverseErrors::OffsetsLengthDoesNotMatchNCells {
            len_offset: offs.len(),
            n_cells: cell_indices.len(),
        });
    }

    let verbosity = parse_verbosity_level(verbose);

    let start_total = Instant::now();

    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let start_reading = Instant::now();

    let mut gene_chunks: Vec<CscGeneChunk> =
        reader.read_gene_parallel_filtered(gene_indices, &cell_set)?;

    if params_pca.clr {
        gene_chunks
            .par_iter_mut()
            .for_each(|chunk| chunk.transform_to_clr(params_pca.size_factor));
    }

    let end_reading = start_reading.elapsed();

    if verbosity.normal_verbosity() {
        println!("Sparse PCA: Loaded in data in {:.2?}", end_reading);
    }

    let start_data_prep = Instant::now();

    let n_cells = cell_set.len();

    let csc = from_gene_chunks::<f32>(gene_chunks, &DataLayerReturn::Norm, n_cells);

    let end_data_prep = start_data_prep.elapsed();

    let col_means: Vec<f64> = sparse_csc_column_means(&csc, true, clr_offsets)?;
    let col_stds: Vec<f64> = sparse_csc_column_stds(&csc, &col_means, true, clr_offsets)?;

    if verbosity.normal_verbosity() {
        println!(
            "Sparse PCA: finished the data preparations in {:.2?}",
            end_data_prep
        );
    }

    let start_svd = Instant::now();

    let (scores, loadings, s) = if params_pca.randomised {
        let svd_res = randomised_sparse_svd::<f32, f64>(
            csc,
            no_pcs,
            seed as u64,
            true,
            Some(100_usize),
            None,
            Some(&col_means),
            Some(&col_stds),
            clr_offsets,
        )?;
        let scores_f64 = compute_pc_scores(&svd_res);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| scores_f64[(i, j)] as f32);
        let loadings = Mat::<f32>::from_fn(gene_indices.len(), no_pcs, |i, j| {
            svd_res.v()[(i, j)] as f32
        });
        let s: Vec<f32> = svd_res.s()[..no_pcs].iter().map(|&x| x as f32).collect();
        (scores, loadings, s)
    } else {
        let svd_res = sparse_svd_lanczos::<f32, f32, f64>(
            &csc,
            no_pcs,
            seed as u64,
            true,
            Some(&col_means),
            Some(&col_stds),
            clr_offsets,
        )?;
        let scores_f64 = compute_pc_scores(&svd_res);
        let scores = Mat::<f32>::from_fn(scores_f64.nrows(), scores_f64.ncols(), |i, j| {
            scores_f64[(i, j)] as f32
        });
        let loadings = Mat::<f32>::from_fn(svd_res.v().nrows(), svd_res.v().ncols(), |i, j| {
            svd_res.v()[(i, j)] as f32
        });
        let s: Vec<f32> = svd_res.s().iter().map(|&x| x as f32).collect();
        (scores, loadings, s)
    };

    let end_svd = start_svd.elapsed();

    if verbosity.normal_verbosity() {
        println!("Sparse PCA: finished calculations in {:.2?}", end_svd);
    }

    let end_total = start_total.elapsed();

    if verbosity.normal_verbosity() {
        println!("Sparse PCA: total run time -> {:.2?}", end_total);
    }

    let col_means = col_means.iter().map(|x| *x as f32).collect();
    let col_stds = col_stds.iter().map(|x| *x as f32).collect();

    Ok((scores, loadings, s, col_means, col_stds))
}

/// Calculate the PCs for single cell data (sparse)
///
/// This version does NOT scale the data and avoids densifying the data at any
/// point, avoiding holding a large matrix in memory.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `params_pca` - The parameters for this single cell PCA run, see
///   [SingleCellPcaParams].
/// * `clr_offsets` - Pre-computed CLR offsets if you want to use the CLR
///   transformation here.
/// * `seed` - Seed for randomised SVD.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Return
///
/// The [SingleCellPcaRes]
#[allow(clippy::too_many_arguments)]
pub fn pca_on_sc_sparse<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    clr_offsets: Option<&[f64]>,
    seed: usize,
    verbose: usize,
) -> SingleCellPcaRes {
    let (scores, loadings, s, _, _) = sparse_pca(
        reader,
        cell_indices,
        gene_indices,
        no_pcs,
        params_pca,
        clr_offsets,
        seed,
        verbose,
    )?;

    Ok((scores, loadings, s))
}

/// Calculate the PCs for single cell data (sparse)
///
/// This version does NOT scale the data and avoids densifying the data at any
/// point, avoiding holding a large matrix in memory.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `cell_indices` - Slice of indices for the cells.
/// * `gene_indices` - Slice of indices for the genes.
/// * `no_pcs` - Number of principal components to calculate
/// * `params_pca` - The parameters for this single cell PCA run, see
///   [SingleCellPcaParams].
/// * `clr_offsets` - Pre-computed CLR offsets if you want to use the CLR
///   transformation here.
/// * `seed` - Seed for randomised SVD.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Return
///
/// The [SingleCellPcaResStats]
#[allow(clippy::too_many_arguments)]
pub fn pca_on_sc_sparse_stats<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    clr_offsets: Option<&[f64]>,
    seed: usize,
    verbose: usize,
) -> SingleCellPcaResStats {
    let res = sparse_pca(
        reader,
        cell_indices,
        gene_indices,
        no_pcs,
        params_pca,
        clr_offsets,
        seed,
        verbose,
    )?;

    Ok(res)
}
