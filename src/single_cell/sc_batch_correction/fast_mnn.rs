//! Implementation of the (fast)MNN approach from Haghverdi, et al, Nat
//! Biotechnol, 2018. The neighbour search is passed in as a closure, so the
//! CPU path here and the GPU path in `gpu::sc_gpu::fast_mnn_gpu` share every
//! other step.

use faer::{Col, Mat, MatMut, MatRef};
use rayon::prelude::*;
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_batch_correction::batch_utils::batch_knn_search;
use crate::single_cell::sc_batch_correction::batch_utils::cosine_normalise;
use crate::single_cell::sc_batch_correction::batch_utils::process_batch_labels;
use crate::single_cell::sc_processing::pca::*;

///////////////
// Constants //
///////////////

/// Below this L2 norm a batch vector has no direction to centre along, and
/// the data passes through unchanged.
const MIN_BATCH_VEC_NORM: f64 = 1e-15;

/// Distances (and tricube bandwidths) below this count as zero. A cell whose
/// neighbours all sit on top of it gets the unweighted mean of the zero
/// distance corrections instead of the kernel.
const MIN_DIST: f32 = 1e-15;

////////////
// Params //
////////////

/// Parameters for fastMNN batch correction
#[derive(Clone, Debug)]
pub struct FastMnnParams {
    /// Number of median distances for tricube kernel bandwidth (default 3.0)
    pub ndist: f32,
    /// Apply cosine normalisation before computing distances.
    pub cos_norm: bool,
    /// Number of PCs to use for the MNN calculations
    pub no_pcs: usize,
    /// Shall sparse SVD be utilised -> reduces memory pressure
    pub sparse_svd: bool,
    /// [KnnParams] for the various approximate nearest neighbour searches
    /// in ann-search-rs
    pub knn_params: KnnParams,
    /// [SingleCellPcaParams] specifying the to-be-applied normalisations and
    /// the PCA solver.
    pub pca_params: SingleCellPcaParams,
}

/////////////
// Helpers //
/////////////

/// Find mutual nearest neighbours from two KNN graphs
///
/// `k` is small, so a linear scan of the neighbour row beats hashing it.
///
/// ### Params
///
/// * `left_knn` - KNN indices for left batch (each row is a left cell's
///   neighbours in the right batch)
/// * `right_knn` - KNN indices for right batch (each row is a right cell's
///   neighbours in the left batch)
///
/// ### Returns
///
/// (left_indices, right_indices) of MNN pairs, ordered by left cell
pub fn find_mutual_nns(
    left_knn: &[Vec<usize>],
    right_knn: &[Vec<usize>],
) -> (Vec<usize>, Vec<usize>) {
    left_knn
        .par_iter()
        .enumerate()
        .flat_map_iter(|(left_idx, left_neighbours)| {
            left_neighbours
                .iter()
                .filter(move |&&right_idx| right_knn[right_idx].contains(&left_idx))
                .map(move |&right_idx| (left_idx, right_idx))
        })
        .unzip()
}

/// Compute raw correction vectors from MNN pairs
///
/// Pairs are bucketed by their right cell first, so every target is averaged
/// independently and in parallel.
///
/// ### Params
///
/// * `data_1` - Left batch data (cells x features)
/// * `data_2` - Right batch data (cells x features)
/// * `mnn_1` - MNN indices in left batch
/// * `mnn_2` - MNN indices in right batch
///
/// ### Returns
///
/// Matrix of correction vectors averaged per unique cell in right batch
/// (n_unique x features) and the sorted right batch indices of its rows
pub fn compute_correction_vecs(
    data_1: &MatRef<f32>,
    data_2: &MatRef<f32>,
    mnn_1: &[usize],
    mnn_2: &[usize],
) -> (Mat<f32>, Vec<usize>) {
    let n_features = data_1.ncols();
    let n_right = data_2.nrows();

    let mut indptr = vec![0_usize; n_right + 1];
    for &r in mnn_2 {
        indptr[r + 1] += 1;
    }
    for r in 0..n_right {
        indptr[r + 1] += indptr[r];
    }
    let mut fill = indptr.clone();
    let mut partners = vec![0_usize; mnn_1.len()];
    for (&l, &r) in mnn_1.iter().zip(mnn_2) {
        partners[fill[r]] = l;
        fill[r] += 1;
    }

    let targets: Vec<usize> = (0..n_right)
        .filter(|&r| indptr[r + 1] > indptr[r])
        .collect();

    let mut averaged = vec![0_f32; targets.len() * n_features];
    averaged
        .par_chunks_mut(n_features)
        .zip(targets.par_iter())
        .for_each(|(row, &r)| {
            let left = &partners[indptr[r]..indptr[r + 1]];
            for &l in left {
                for (g, sum) in row.iter_mut().enumerate() {
                    *sum += data_1[(l, g)] - data_2[(r, g)];
                }
            }
            let n = left.len() as f32;
            row.iter_mut().for_each(|x| *x /= n);
        });

    (
        Mat::from_fn(targets.len(), n_features, |i, g| {
            averaged[i * n_features + g]
        }),
        targets,
    )
}

/// Centre data along the batch vector direction, in place.
///
/// Projects all cells onto the unit batch vector, computes mean projection,
/// then shifts each cell so variation along the batch direction is removed.
/// This is `.center_along_batch_vector` from the R code. The projection is a
/// faer matvec and the shift a column-wise rank-1 update, so both run along
/// the contiguous axis of the column-major matrix.
///
/// ### Params
///
/// * `data` - Cell data matrix (cells x features), modified in place
/// * `batch_vec` - The batch direction vector (length = n_features). If its
///   L2 norm is below [`MIN_BATCH_VEC_NORM`], data is left unchanged.
fn center_along_batch_vector(data: MatMut<f32>, batch_vec: &[f32]) {
    let n_cells = data.nrows();

    let l2 = batch_vec
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if l2 < MIN_BATCH_VEC_NORM {
        return;
    }
    let unit: Col<f32> = Col::from_fn(batch_vec.len(), |g| (batch_vec[g] as f64 / l2) as f32);

    let projections: Col<f32> = data.as_ref() * unit.as_ref();
    // f64 so the mean over many cells does not drift
    let mean_proj = projections.iter().map(|&p| p as f64).sum::<f64>() / n_cells as f64;
    let shift: Vec<f32> = projections
        .iter()
        .map(|&p| (mean_proj - p as f64) as f32)
        .collect();

    data.par_col_iter_mut().enumerate().for_each(|(g, col)| {
        let u = unit[g];
        for (x, &s) in col.iter_mut().zip(&shift) {
            *x += s * u;
        }
    });
}

/// Median of a slice, reordering it in place.
///
/// ### Params
///
/// * `x` - Non-empty slice of values. Reordered on return.
///
/// ### Returns
///
/// The median; the mean of the two middle values for an even length.
fn median_in_place(x: &mut [f32]) -> f32 {
    let len = x.len();
    let (lower, &mut upper, _) = x.select_nth_unstable_by(len / 2, f32::total_cmp);
    if len.is_multiple_of(2) {
        let lower_max = lower.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (lower_max + upper) * 0.5
    } else {
        upper
    }
}

/// Compute tricube-weighted correction for all cells in the target batch.
///
/// For each cell, finds k nearest neighbours among MNN-involved cells,
/// computes tricube-weighted average of their correction vectors, and
/// returns the corrected data (data + weighted_correction). Parallel over
/// cells, with the corrections copied row-major once so each cell reads its
/// neighbours' vectors contiguously.
///
/// ### Params
///
/// * `data` - Target batch cell coordinates (cells x features)
/// * `corrections` - Averaged correction vectors, one row per unique MNN cell
///   (n_unique_mnn x features)
/// * `mnn_indices` - Row indices into `data` identifying cells that
///   participated in MNN pairs; used to build the neighbour search reference
/// * `k` - Number of nearest MNN neighbours to use for weighting
/// * `ndist` - Bandwidth multiplier; bandwidth = ndist * median distance to
///   the k neighbours
/// * `search` - Neighbour search, called as `search(query, reference, k)` and
///   returning true (not squared) distances
///
/// ### Returns
///
/// Corrected matrix (cells x features); each cell's coordinates shifted by
/// its tricube-weighted average correction vector
pub fn tricube_weighted_correction<F>(
    data: &MatRef<f32>,
    corrections: &MatRef<f32>,
    mnn_indices: &[usize],
    k: usize,
    ndist: f32,
    search: &F,
) -> Result<Mat<f32>, BixverseErrors>
where
    F: Fn(MatRef<f32>, MatRef<f32>, usize) -> ScKnnResults,
{
    let n_cells = data.nrows();
    let n_features = data.ncols();
    let n_mnn = mnn_indices.len();

    let mnn_data = Mat::from_fn(n_mnn, n_features, |r, c| *data.get(mnn_indices[r], c));
    let safe_k = k.min(n_mnn);

    let (knn_idx, knn_dist) = search(*data, mnn_data.as_ref(), safe_k)?;

    let corr: Vec<f32> = (0..n_mnn)
        .flat_map(|r| (0..n_features).map(move |g| corrections[(r, g)]))
        .collect();

    let mut shift = vec![0_f32; n_cells * n_features];
    shift
        .par_chunks_mut(n_features)
        .zip(knn_idx.par_iter().zip(knn_dist.par_iter()))
        .for_each_init(
            || (Vec::with_capacity(safe_k), Vec::with_capacity(safe_k)),
            |(scratch, weights), (row, (indices, dists))| {
                if dists.is_empty() {
                    return;
                }

                scratch.clear();
                scratch.extend_from_slice(dists);
                let bandwidth = ndist * median_in_place(scratch);

                weights.clear();
                if bandwidth < MIN_DIST {
                    weights.extend(dists.iter().map(|&d| if d < MIN_DIST { 1.0 } else { 0.0 }));
                } else {
                    weights.extend(dists.iter().map(|&d| {
                        let ratio = d / bandwidth;
                        if ratio < 1.0 {
                            let t = 1.0 - ratio * ratio * ratio;
                            t * t * t
                        } else {
                            0.0
                        }
                    }));
                }

                let weight_sum: f32 = weights.iter().sum();
                if weight_sum <= 0.0 {
                    return;
                }
                let inv_w = 1.0 / weight_sum;

                for (&nn, &w) in indices.iter().zip(weights.iter()) {
                    if w > 0.0 {
                        let w = w * inv_w;
                        let c = &corr[nn * n_features..(nn + 1) * n_features];
                        for (out, &c_g) in row.iter_mut().zip(c) {
                            *out += c_g * w;
                        }
                    }
                }
            },
        );

    Ok(Mat::from_fn(n_cells, n_features, |i, g| {
        data.get(i, g) + shift[i * n_features + g]
    }))
}

/// Merge two batches following the fastMNN algorithm.
///
/// Method
///
/// 1. Orthogonalise batch2 against accumulated batch vectors
/// 2. Find MNN pairs
/// 3. Compute average correction and overall batch vector
/// 4. Centre both batches along the batch vector (removes variation along batch
///    direction)
/// 5. Recompute corrections with centred data (same MNN pairs)
/// 6. Apply tricube-weighted correction to batch2
/// 7. Stack and return
///
/// ### Params
///
/// * `data_1` - Left (reference) batch coordinates (cells x features)
/// * `data_2` - Right (target) batch coordinates (cells x features)
/// * `k` - Number of neighbours for the MNN and tricube searches
/// * `ndist` - Tricube bandwidth multiplier, see
///   [`tricube_weighted_correction`]
/// * `search` - Neighbour search, called as `search(query, reference, k)` and
///   returning true (not squared) distances
/// * `batch_vecs` - Accumulated batch direction vectors from all prior merges;
///   `data_2` is orthogonalised against each before MNN search, and the new
///   batch vector is appended on return
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Stacked matrix of left (centred) and corrected right batch (n_total x
/// features)
pub fn merge_two_batches<F>(
    data_1: &MatRef<f32>,
    data_2: &MatRef<f32>,
    k: usize,
    ndist: f32,
    search: &F,
    batch_vecs: &mut Vec<Vec<f32>>,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors>
where
    F: Fn(MatRef<f32>, MatRef<f32>, usize) -> ScKnnResults,
{
    let verbosity = parse_verbosity_level(verbose);

    let n_features = data_1.ncols();
    let n_left = data_1.nrows();

    // Step 1: Orthogonalise new batch against all previous batch vectors.
    // In a progressive merge the left (already merged) carries previous
    // orthogonalisations baked in, so only the right needs this.
    let mut right = data_2.to_owned();
    for vec in batch_vecs.iter() {
        center_along_batch_vector(right.as_mut(), vec);
    }

    // Step 2: Find MNN pairs
    let (knn_1_to_2, _) = search(*data_1, right.as_ref(), k)?;
    let (knn_2_to_1, _) = search(right.as_ref(), *data_1, k)?;

    let (mnn_1, mnn_2) = find_mutual_nns(&knn_1_to_2, &knn_2_to_1);

    if mnn_1.is_empty() {
        if verbosity.normal_verbosity() {
            eprintln!("Warning: No MNN pairs found, skipping correction");
        }
        let n_total = n_left + right.nrows();
        return Ok(Mat::from_fn(n_total, n_features, |row, col| {
            if row < n_left {
                *data_1.get(row, col)
            } else {
                right[(row - n_left, col)]
            }
        }));
    }

    if verbosity.normal_verbosity() {
        println!(
            "Found {} MNN pairs",
            mnn_1.len().separate_with_underscores()
        );
    }

    // Step 3: Compute average correction vectors and overall batch vector
    let (averaged, _) = compute_correction_vecs(data_1, &right.as_ref(), &mnn_1, &mnn_2);

    let n_unique = averaged.nrows() as f64;
    let overall_batch: Vec<f32> = (0..n_features)
        .map(|g| (averaged.col(g).iter().map(|&x| x as f64).sum::<f64>() / n_unique) as f32)
        .collect();

    // Step 4: Centre both batches along the overall batch vector
    let mut left = data_1.to_owned();
    center_along_batch_vector(left.as_mut(), &overall_batch);
    center_along_batch_vector(right.as_mut(), &overall_batch);

    // Step 5: Recompute correction vectors with centred coordinates (same MNN pairs)
    let (re_averaged, re_unique_mnn) =
        compute_correction_vecs(&left.as_ref(), &right.as_ref(), &mnn_1, &mnn_2);

    // Step 6: Tricube-weighted correction applied to every cell in batch2
    let right_corrected = tricube_weighted_correction(
        &right.as_ref(),
        &re_averaged.as_ref(),
        &re_unique_mnn,
        k,
        ndist,
        search,
    )?;

    // Record this batch vector for future orthogonalisation steps
    batch_vecs.push(overall_batch);

    // Step 7: Stack left (centred) and right (corrected)
    let n_total = n_left + right_corrected.nrows();

    Ok(Mat::from_fn(n_total, n_features, |row, col| {
        if row < n_left {
            left[(row, col)]
        } else {
            right_corrected[(row - n_left, col)]
        }
    }))
}

/// Fast MNN with cell order tracking, generic over the neighbour search
///
/// ### Params
///
/// * `batches` - Vec of PCA matrices per batch (cells x n_pcs)
/// * `original_indices` - Vec of original cell indices per batch
/// * `k` - Number of neighbours for the MNN and tricube searches
/// * `ndist` - Tricube bandwidth multiplier
/// * `search` - Neighbour search, called as `search(query, reference, k)` and
///   returning true (not squared) distances
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// (corrected_pca, output_to_original_mapping)
pub fn fast_mnn_with_search<F>(
    batches: Vec<Mat<f32>>,
    original_indices: Vec<Vec<usize>>,
    k: usize,
    ndist: f32,
    search: &F,
    verbose: usize,
) -> Result<(Mat<f32>, Vec<usize>), BixverseErrors>
where
    F: Fn(MatRef<f32>, MatRef<f32>, usize) -> ScKnnResults,
{
    if batches.len() != original_indices.len() {
        return Err(BixverseErrors::LengthMismatch {
            name: "original_indices",
            expected: batches.len(),
            found: original_indices.len(),
        });
    }

    let verbosity = parse_verbosity_level(verbose);
    let total_batches = batches.len();

    let mut batches = batches.into_iter();
    let mut original_indices = original_indices.into_iter();
    let (Some(mut merged), Some(mut index_map)) = (batches.next(), original_indices.next()) else {
        return Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches: 0 });
    };

    let mut batch_vecs: Vec<Vec<f32>> = Vec::new();

    for (batch_num, (batch, batch_indices)) in batches.zip(original_indices).enumerate() {
        let start = Instant::now();
        merged = merge_two_batches(
            &merged.as_ref(),
            &batch.as_ref(),
            k,
            ndist,
            search,
            &mut batch_vecs,
            verbose,
        )?;

        if verbosity.normal_verbosity() {
            println!(
                "Merged {} of {} batches in {:.2}s",
                batch_num + 2,
                total_batches,
                start.elapsed().as_secs_f64()
            );
        }

        index_map.extend(batch_indices);
    }

    Ok((merged, index_map))
}

/// Fast MNN with cell order tracking, using the CPU neighbour searches
///
/// ### Params
///
/// * `batches` - Vec of PCA matrices per batch (cells x n_pcs)
/// * `original_indices` - Vec of original cell indices per batch
/// * `params` - `FastMnnParams` params with all of the parameters for this
///   run
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// (corrected_pca, output_to_original_mapping)
pub fn fast_mnn(
    batches: Vec<Mat<f32>>,
    original_indices: Vec<Vec<usize>>,
    params: &FastMnnParams,
    seed: usize,
    verbose: usize,
) -> Result<(Mat<f32>, Vec<usize>), BixverseErrors> {
    let search = |query: MatRef<f32>, reference: MatRef<f32>, k: usize| {
        batch_knn_search(query, reference, k, &params.knn_params, seed, verbose)
    };
    fast_mnn_with_search(
        batches,
        original_indices,
        params.knn_params.k,
        params.ndist,
        &search,
        verbose,
    )
}

/// Batch correct a full embedding with fastMNN
///
/// Splits by batch, optionally cosine normalises, merges and puts the cells
/// back in their original order. Shared by the CPU and GPU entry points.
///
/// ### Params
///
/// * `embd` - Embedding of all cells (cells x n_pcs), usually PCA
/// * `batch_indices` - Batch assignment for each cell
/// * `cos_norm` - Cosine normalise each batch before merging
/// * `k` - Number of neighbours for the MNN and tricube searches
/// * `ndist` - Tricube bandwidth multiplier
/// * `search` - Neighbour search, called as `search(query, reference, k)` and
///   returning true (not squared) distances
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Batch-corrected embedding (cells x n_pcs) in original cell order
pub fn fast_mnn_embedding<F>(
    embd: &Mat<f32>,
    batch_indices: &[usize],
    cos_norm: bool,
    k: usize,
    ndist: f32,
    search: &F,
    verbose: usize,
) -> Result<Mat<f32>, BixverseErrors>
where
    F: Fn(MatRef<f32>, MatRef<f32>, usize) -> ScKnnResults,
{
    let verbosity = parse_verbosity_level(verbose);
    let (_, n_batches) = process_batch_labels(batch_indices);
    if n_batches < 2 {
        return Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches });
    }

    let (mut batches, original_indices) = split_pca_by_batch(embd, batch_indices);

    if cos_norm {
        if verbosity.normal_verbosity() {
            println!("Applying cosine normalisation to the dimension reduction.")
        }
        for batch in batches.iter_mut() {
            *batch = cosine_normalise(batch);
        }
    }

    let (corrected, index_map) =
        fast_mnn_with_search(batches, original_indices, k, ndist, search, verbose)?;

    Ok(reorder_to_original(&corrected, &index_map))
}

/// Reorder corrected PCA back to original cell order
///
/// ### Params
///
/// * `corrected_pca` - Output from fast_mnn (cells x n_pcs)
/// * `output_to_original` - Mapping from output row -> original index
///
/// ### Returns
///
/// Reordered matrix matching original cell order
pub fn reorder_to_original(corrected_pca: &Mat<f32>, output_to_original: &[usize]) -> Mat<f32> {
    let n_pcs = corrected_pca.ncols();

    let n_original_cells = output_to_original.iter().copied().max().unwrap_or(0) + 1;

    let mut original_to_output = vec![0; n_original_cells];
    for (output_idx, &original_idx) in output_to_original.iter().enumerate() {
        original_to_output[original_idx] = output_idx;
    }

    Mat::from_fn(n_original_cells, n_pcs, |row, col| {
        *corrected_pca.get(original_to_output[row], col)
    })
}

/// Split PCA matrix by batch indices
///
/// ### Params
///
/// * `pca_all` - Full PCA matrix (all cells x n_pcs)
/// * `batch_indices` - Batch assignment for each cell
///
/// ### Returns
///
/// (matrices per batch, original cell indices per batch)
pub fn split_pca_by_batch(
    pca_all: &Mat<f32>,
    batch_indices: &[usize],
) -> (Vec<Mat<f32>>, Vec<Vec<usize>>) {
    let n_features = pca_all.ncols();
    let n_batches = batch_indices.iter().copied().max().unwrap_or(0) + 1;

    let mut batch_cells: Vec<Vec<usize>> = vec![Vec::new(); n_batches];
    for (cell_idx, &batch) in batch_indices.iter().enumerate() {
        batch_cells[batch].push(cell_idx);
    }

    let batches: Vec<Mat<f32>> = batch_cells
        .iter()
        .map(|cells| {
            Mat::from_fn(cells.len(), n_features, |row, col| {
                pca_all[(cells[row], col)]
            })
        })
        .collect();

    (batches, batch_cells)
}

///////////////////
// Main function //
///////////////////

/// Perform fastMNN batch correction on single cell data
///
/// Computes PCA on all cells, splits by batch, applies fastMNN correction,
/// and returns the corrected PCA matrix in the original cell order.
///
/// ### Params
///
/// * `reader` - Reader over the single cell count store
/// * `cell_indices` - Indices of cells to include
/// * `gene_indices` - Indices of genes to include
/// * `batch_indices` - Batch assignment for each cell
/// * `pre_computed_pca` - Pre-computed PCA matrix (optional)
/// * `clr_offsets` - Pre-computed CLR offsets if the PCA is recomputed with
///   the CLR transformation. Ignored with `pre_computed_pca`.
/// * `params` - FastMNN parameters
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Batch-corrected PCA matrix (cells x n_pcs) in original cell order
#[allow(clippy::too_many_arguments)]
pub fn fast_mnn_main<S: SingleCellReading>(
    reader: &S,
    cell_indices: &[usize],
    gene_indices: &[usize],
    batch_indices: &[usize],
    pre_computed_pca: Option<Mat<f32>>,
    clr_offsets: Option<&[f64]>,
    params: &FastMnnParams,
    verbose: usize,
    seed: usize,
) -> Result<Mat<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let (_, n_batches) = process_batch_labels(batch_indices);

    // checked here as well, so a single batch errors before the PCA runs
    if n_batches < 2 {
        return Err(BixverseErrors::NeedAtLeastTwoBatches { n_batches });
    }

    let pca_all = if let Some(pca) = pre_computed_pca {
        if verbosity.normal_verbosity() {
            println!("Using pre-computed PCA")
        }
        pca
    } else {
        if verbosity.detailed_verbosity() {
            println!("Re-computing PCA")
        }
        if params.sparse_svd {
            let (pca, _, _) = pca_on_sc_sparse(
                reader,
                cell_indices,
                gene_indices,
                params.no_pcs,
                &params.pca_params,
                clr_offsets,
                seed,
                verbose,
            )?;

            pca
        } else {
            let (pca, _, _, _) = pca_on_sc(
                reader,
                cell_indices,
                gene_indices,
                params.no_pcs,
                &params.pca_params,
                clr_offsets,
                seed,
                false,
                verbose,
            )?;

            pca
        }
    };

    let search = |query: MatRef<f32>, reference: MatRef<f32>, k: usize| {
        batch_knn_search(query, reference, k, &params.knn_params, seed, verbose)
    };

    fast_mnn_embedding(
        &pca_all,
        batch_indices,
        params.cos_norm,
        params.knn_params.k,
        params.ndist,
        &search,
        verbose,
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::mat;

    /// Centred copy of `data`, leaving the input untouched.
    fn centred_copy(data: &Mat<f32>, batch_vec: &[f32]) -> Mat<f32> {
        let mut out = data.clone();
        center_along_batch_vector(out.as_mut(), batch_vec);
        out
    }

    /// The median handles odd and even lengths without needing sorted input.
    #[test]
    fn test_median_in_place() {
        assert_relative_eq!(median_in_place(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_relative_eq!(median_in_place(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_relative_eq!(median_in_place(&mut [5.0]), 5.0);
    }

    /// A pair only counts as mutual when both sides list each other.
    #[test]
    fn test_find_mutual_nns_simple() {
        // Cell 0 in left has right neighbour 0; cell 0 in right has left neighbour 0
        let left_knn = vec![vec![0], vec![1]];
        let right_knn = vec![vec![0], vec![1]];

        let (mnn_l, mnn_r) = find_mutual_nns(&left_knn, &right_knn);

        assert_eq!(mnn_l.len(), mnn_r.len());
        assert!(mnn_l.contains(&0));
        assert!(mnn_r.contains(&0));
        assert!(mnn_l.contains(&1));
        assert!(mnn_r.contains(&1));
    }

    /// One-directional neighbour links are dropped; only the mutual pair survives.
    #[test]
    fn test_find_mutual_nns_asymmetric() {
        // Left 0 -> right 0, but right 0 -> left 1 (not 0)
        let left_knn = vec![vec![0], vec![0]];
        let right_knn = vec![vec![1]];

        let (mnn_l, mnn_r) = find_mutual_nns(&left_knn, &right_knn);

        // Only left 1 -> right 0 and right 0 -> left 1 is mutual
        assert_eq!(mnn_l.len(), 1);
        assert_eq!(mnn_l[0], 1);
        assert_eq!(mnn_r[0], 0);
    }

    /// No mutual pair at all yields empty index vectors rather than an error.
    #[test]
    fn test_find_mutual_nns_empty() {
        let left_knn: Vec<Vec<usize>> = vec![vec![0]];
        let right_knn: Vec<Vec<usize>> = vec![vec![1]]; // points to left 1 which doesn't exist in left_knn... but left 0 -> right 0

        // left 0 -> right 0, right 0 -> left 1. Not mutual.
        let (mnn_l, mnn_r) = find_mutual_nns(&left_knn, &right_knn);
        assert!(mnn_l.is_empty());
        assert!(mnn_r.is_empty());
    }

    /// With k > 1 every mutual pair is emitted, not just the first one per cell.
    #[test]
    fn test_find_mutual_nns_multiple_neighbours() {
        let left_knn = vec![vec![0, 1], vec![0, 1], vec![1]];
        let right_knn = vec![vec![0, 1], vec![1, 2]];

        let (mnn_l, mnn_r) = find_mutual_nns(&left_knn, &right_knn);

        assert!(mnn_l.len() >= 3);

        let pairs: Vec<(usize, usize)> = mnn_l
            .iter()
            .zip(mnn_r.iter())
            .map(|(&l, &r)| (l, r))
            .collect();
        assert!(pairs.contains(&(0, 0)));
        assert!(pairs.contains(&(1, 1)));
        assert!(pairs.contains(&(2, 1)));
    }

    /// A single MNN pair gives the plain difference between the two cells.
    #[test]
    fn test_correction_vecs_single_pair() {
        let data_1 = mat![[1.0f32, 2.0, 3.0]];
        let data_2 = mat![[0.5f32, 1.0, 1.5]];
        let mnn_1 = vec![0];
        let mnn_2 = vec![0];

        let (averaged, indices) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        assert_eq!(indices, vec![0]);
        assert_eq!(averaged.nrows(), 1);
        assert_eq!(averaged.ncols(), 3);
        assert_relative_eq!(averaged[(0, 0)], 0.5, epsilon = 1e-6);
        assert_relative_eq!(averaged[(0, 1)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(averaged[(0, 2)], 1.5, epsilon = 1e-6);
    }

    /// Several pairs hitting the same target cell collapse into one averaged vector.
    #[test]
    fn test_correction_vecs_averaging() {
        // Two pairs map to the same cell in batch 2
        let data_1 = mat![[2.0f32, 0.0], [4.0, 0.0],];
        let data_2 = mat![[1.0f32, 0.0]];
        let mnn_1 = vec![0, 1];
        let mnn_2 = vec![0, 0];

        let (averaged, indices) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        assert_eq!(indices, vec![0]);
        assert_relative_eq!(averaged[(0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(averaged[(0, 1)], 0.0, epsilon = 1e-6);
    }

    /// Distinct target cells keep their own correction vectors.
    #[test]
    fn test_correction_vecs_multiple_targets() {
        let data_1 = mat![[3.0f32, 0.0], [0.0, 3.0],];
        let data_2 = mat![[1.0f32, 0.0], [0.0, 1.0],];
        let mnn_1 = vec![0, 1];
        let mnn_2 = vec![0, 1];

        let (averaged, indices) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        assert_eq!(indices.len(), 2);
        assert_eq!(averaged.nrows(), 2);

        let idx_0 = indices.iter().position(|&x| x == 0).unwrap();
        let idx_1 = indices.iter().position(|&x| x == 1).unwrap();
        assert_relative_eq!(averaged[(idx_0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(averaged[(idx_1, 1)], 2.0, epsilon = 1e-6);
    }

    /// Centring collapses the spread along the batch vector onto the batch mean.
    #[test]
    fn test_center_diagonal_batch_vector() {
        // batch_vec along (1,1), data offset along that diagonal
        let data = mat![[0.0f32, 0.0], [2.0, 2.0],];
        let batch_vec = vec![1.0f32, 1.0];

        let centred = centred_copy(&data, &batch_vec);

        assert_relative_eq!(centred[(0, 0)], 1.0, epsilon = 1e-5);
        assert_relative_eq!(centred[(0, 1)], 1.0, epsilon = 1e-5);
        assert_relative_eq!(centred[(1, 0)], 1.0, epsilon = 1e-5);
        assert_relative_eq!(centred[(1, 1)], 1.0, epsilon = 1e-5);
    }

    /// A zero batch vector has no direction to centre along, so the data passes through.
    #[test]
    fn test_center_zero_batch_vector() {
        let data = mat![[1.0f32, 2.0], [3.0, 4.0],];
        let batch_vec = vec![0.0f32, 0.0];

        let centred = centred_copy(&data, &batch_vec);

        // Should return data unchanged
        for i in 0..2 {
            for g in 0..2 {
                assert_relative_eq!(centred[(i, g)], data[(i, g)], epsilon = 1e-6);
            }
        }
    }

    /// Centring twice along the same vector matches centring once.
    #[test]
    fn test_center_idempotent() {
        let data = mat![[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0],];
        let batch_vec = vec![1.0f32, 0.5, -0.3];

        let centred_once = centred_copy(&data, &batch_vec);
        let centred_twice = centred_copy(&centred_once, &batch_vec);

        for i in 0..3 {
            for g in 0..3 {
                assert_relative_eq!(centred_once[(i, g)], centred_twice[(i, g)], epsilon = 1e-5);
            }
        }
    }

    /// Splitting by batch keeps row order within a batch and records the original indices.
    #[test]
    fn test_split_pca_basic() {
        let pca = mat![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let batch_indices = vec![0, 1, 0, 1];

        let (batches, indices) = split_pca_by_batch(&pca, &batch_indices);

        assert_eq!(batches.len(), 2);
        assert_eq!(indices.len(), 2);

        assert_eq!(indices[0], vec![0, 2]);
        assert_eq!(indices[1], vec![1, 3]);

        assert_eq!(batches[0].nrows(), 2);
        assert_relative_eq!(batches[0][(0, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(batches[0][(1, 0)], 5.0, epsilon = 1e-6);

        assert_eq!(batches[1].nrows(), 2);
        assert_relative_eq!(batches[1][(0, 0)], 3.0, epsilon = 1e-6);
        assert_relative_eq!(batches[1][(1, 0)], 7.0, epsilon = 1e-6);
    }

    /// Batches come back ordered by label, not by order of first appearance.
    #[test]
    fn test_split_pca_three_batches() {
        let pca = mat![[1.0f32], [2.0], [3.0], [4.0], [5.0], [6.0],];
        let batch_indices = vec![2, 0, 1, 0, 2, 1];

        let (batches, indices) = split_pca_by_batch(&pca, &batch_indices);

        assert_eq!(batches.len(), 3);
        assert_eq!(indices[0], vec![1, 3]);
        assert_eq!(indices[1], vec![2, 5]);
        assert_eq!(indices[2], vec![0, 4]);

        assert_relative_eq!(batches[0][(0, 0)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(batches[0][(1, 0)], 4.0, epsilon = 1e-6);
        assert_relative_eq!(batches[2][(0, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(batches[2][(1, 0)], 5.0, epsilon = 1e-6);
    }

    /// An identity mapping leaves the matrix untouched.
    #[test]
    fn test_reorder_identity() {
        let data = mat![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0],];
        let mapping = vec![0, 1, 2];

        let reordered = reorder_to_original(&data, &mapping);

        for i in 0..3 {
            for g in 0..2 {
                assert_relative_eq!(reordered[(i, g)], data[(i, g)], epsilon = 1e-6);
            }
        }
    }

    /// The mapping is read as working row to original row, not the inverse.
    #[test]
    fn test_reorder_reversed() {
        let data = mat![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0],];
        let mapping = vec![2, 1, 0]; // output row 0 -> original 2, etc.

        let reordered = reorder_to_original(&data, &mapping);

        assert_relative_eq!(reordered[(0, 0)], 5.0, epsilon = 1e-6);
        assert_relative_eq!(reordered[(1, 0)], 3.0, epsilon = 1e-6);
        assert_relative_eq!(reordered[(2, 0)], 1.0, epsilon = 1e-6);
    }

    /// Two identical batches must produce zero correction, not drift.
    #[test]
    fn test_correction_vecs_symmetric() {
        // If batches are identical, corrections should be zero
        let data = mat![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0],];
        let mnn_1 = vec![0, 1, 2];
        let mnn_2 = vec![0, 1, 2];

        let (averaged, _) = compute_correction_vecs(&data.as_ref(), &data.as_ref(), &mnn_1, &mnn_2);

        for i in 0..averaged.nrows() {
            for g in 0..averaged.ncols() {
                assert_relative_eq!(averaged[(i, g)], 0.0, epsilon = 1e-6);
            }
        }
    }

    /// The averaged correction points from the target batch back to the reference.
    #[test]
    fn test_overall_batch_vector_direction() {
        // Batch 2 is shifted +2 along feature 0 relative to batch 1
        let data_1 = mat![[0.0f32, 0.0], [0.0, 1.0], [0.0, -1.0],];
        let data_2 = mat![[2.0f32, 0.0], [2.0, 1.0], [2.0, -1.0],];
        let mnn_1 = vec![0, 1, 2];
        let mnn_2 = vec![0, 1, 2];

        let (averaged, _) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        let n_features = 2;
        let n_unique = averaged.nrows();
        let mut overall = vec![0.0f32; n_features];
        for g in 0..n_features {
            for i in 0..n_unique {
                overall[g] += averaged[(i, g)];
            }
            overall[g] /= n_unique as f32;
        }

        // Overall batch vector should point in -x direction (ref - target = 0 - 2 = -2)
        assert!(
            overall[0] < -1.5,
            "Expected strong negative x component, got {}",
            overall[0]
        );
        assert!(
            overall[1].abs() < 1e-6,
            "Expected near-zero y component, got {}",
            overall[1]
        );
    }

    /// Centring strips spread along the batch vector and leaves orthogonal features alone.
    #[test]
    fn test_center_then_recompute_reduces_within_batch_spread() {
        // Batch 1 at x~0, batch 2 at x~4, with some x-spread within each
        let data_1 = mat![[-1.0f32, 0.0], [0.0, 1.0], [1.0, 2.0],];
        let data_2 = mat![[3.0f32, 0.0], [4.0, 1.0], [5.0, 2.0],];
        let mnn_1 = vec![0, 1, 2];
        let mnn_2 = vec![0, 1, 2];

        let (averaged, _) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        let n_unique = averaged.nrows();
        let mut overall = vec![0.0f32; 2];
        for g in 0..2 {
            for i in 0..n_unique {
                overall[g] += averaged[(i, g)];
            }
            overall[g] /= n_unique as f32;
        }

        let left_c = centred_copy(&data_1, &overall);
        let right_c = centred_copy(&data_2, &overall);

        // Within each batch, all x-coordinates should now be equal (collapsed to batch mean)
        let mean_x_left: f32 = (0..3).map(|i| left_c[(i, 0)]).sum::<f32>() / 3.0;
        let mean_x_right: f32 = (0..3).map(|i| right_c[(i, 0)]).sum::<f32>() / 3.0;

        for i in 0..3 {
            assert_relative_eq!(left_c[(i, 0)], mean_x_left, epsilon = 1e-5);
            assert_relative_eq!(right_c[(i, 0)], mean_x_right, epsilon = 1e-5);
        }

        // y-coordinates should be preserved
        for i in 0..3 {
            assert_relative_eq!(left_c[(i, 1)], data_1[(i, 1)], epsilon = 1e-5);
            assert_relative_eq!(right_c[(i, 1)], data_2[(i, 1)], epsilon = 1e-5);
        }
    }

    /// No cell is dropped, duplicated or misplaced when the batches are recombined.
    #[test]
    fn test_split_and_reorder_preserves_all_data() {
        // Property test: split then stack then reorder must reconstruct original
        let n_cells = 10;
        let n_features = 3;
        let pca = Mat::from_fn(n_cells, n_features, |i, j| (i * n_features + j) as f32);
        let batch_indices = vec![0, 2, 1, 0, 2, 1, 0, 1, 2, 0];

        let (batches, original_indices) = split_pca_by_batch(&pca, &batch_indices);

        let mut merged = Mat::zeros(n_cells, n_features);
        let mut index_map = Vec::new();
        let mut row = 0;
        for (batch, indices) in batches.iter().zip(original_indices.iter()) {
            for i in 0..batch.nrows() {
                for g in 0..n_features {
                    merged[(row, g)] = batch[(i, g)];
                }
                row += 1;
            }
            index_map.extend(indices.iter().copied());
        }

        let reordered = reorder_to_original(&merged, &index_map);

        for i in 0..n_cells {
            for g in 0..n_features {
                assert_relative_eq!(reordered[(i, g)], pca[(i, g)], epsilon = 1e-6,);
            }
        }
    }

    /// Target indices come back sorted whatever order the MNN pairs arrive in.
    #[test]
    fn test_correction_vecs_indices_are_sorted() {
        let data_1 = mat![[1.0f32, 0.0], [2.0, 0.0], [3.0, 0.0],];
        let data_2 = mat![[0.0f32, 0.0], [0.0, 0.0], [0.0, 0.0],];
        // Map to targets in reverse order
        let mnn_1 = vec![0, 1, 2];
        let mnn_2 = vec![2, 0, 1];

        let (_, indices) =
            compute_correction_vecs(&data_1.as_ref(), &data_2.as_ref(), &mnn_1, &mnn_2);

        // Indices should be sorted
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(indices, sorted);
    }

    /// Only the component along the batch vector moves; the other dimensions stay put.
    #[test]
    fn test_center_high_dimensional() {
        let n_cells = 5;
        let n_features = 10;
        let data = Mat::from_fn(n_cells, n_features, |i, _| i as f32);

        // batch_vec only in first dimension
        let mut batch_vec = vec![0.0f32; n_features];
        batch_vec[0] = 1.0;

        let centred = centred_copy(&data, &batch_vec);

        // All cells should have same projection onto feature 0
        let mean_proj: f32 = (0..n_cells).map(|i| data[(i, 0)]).sum::<f32>() / n_cells as f32;
        for i in 0..n_cells {
            assert_relative_eq!(centred[(i, 0)], mean_proj, epsilon = 1e-5);
        }
        // All other features unchanged (data[i, g>0] = i for all g, which is original)
        for i in 0..n_cells {
            for g in 1..n_features {
                assert_relative_eq!(centred[(i, g)], data[(i, g)], epsilon = 1e-5);
            }
        }
    }
}
