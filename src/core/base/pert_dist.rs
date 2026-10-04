//! E-distance and related perturbation distances.
//!
//! E-distance compares two cell distributions in an embedding via
//! `2 * mean(d(X, Y)) - mean(d(X, X')) - mean(d(Y, Y'))`.

use crate::utils::gemm;
use faer::Accum;
use faer::linalg::matmul::triangular::BlockStructure;
use faer::{Mat, MatRef, Par};
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::faer_parallelism;

///////////
// Enums //
///////////

/// Cell-cell distance used in the E-distance formula.
#[derive(Debug, Clone, Copy, Default)]
pub enum PertDistance {
    /// Energy distance with Euclidean cell-cell distance.
    #[default]
    Euclidean,
    /// Energy distance with squared Euclidean cell-cell distance.
    ///
    /// Reduces algebraically to `2 * ||mu_X - mu_Y||^2`; within-group
    /// dispersion does not affect the result.
    SquaredEuclidean,
}

/// Parse the perturbation distance variant from a string.
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// The Option of [PertDistance]
pub fn parse_perturbation_distance(s: &str) -> Option<PertDistance> {
    match s.to_lowercase().as_str() {
        "euclidean" => Some(PertDistance::Euclidean),
        "squared_euclidean" | "sqeuclidean" => Some(PertDistance::SquaredEuclidean),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Row chunk of the within-group gram product
const GRAM_ROW_TILE: usize = 512;

/// Calculate the squared norms of the rows
///
/// Accumulated feature by feature over contiguous column slices; the per-row
/// summation order over features is unchanged.
///
/// ### Params
///
/// * `x` - The matrix for which to calculate the squared row norms
///
/// ### Returns
///
/// Vector of squared row norms
fn row_norms_squared<T>(x: MatRef<T>) -> Vec<T>
where
    T: BixverseFloat,
{
    let mut out = vec![T::zero(); x.nrows()];
    out.par_chunks_mut(GRAM_ROW_TILE)
        .enumerate()
        .for_each(|(c, chunk)| {
            let start = c * GRAM_ROW_TILE;
            for j in 0..x.ncols() {
                let col = x.col(j).subrows(start, chunk.len());
                for (o, &v) in chunk.iter_mut().zip(col.iter()) {
                    *o += v * v;
                }
            }
        });
    out
}

/// Cell-cell distance from the squared distance
#[inline(always)]
fn finish_dist<T: BixverseFloat>(d_sq: T, dist: PertDistance) -> T {
    match dist {
        PertDistance::Euclidean => d_sq.sqrt(),
        PertDistance::SquaredEuclidean => d_sq,
    }
}

/// Mean pairwise distance within a group.
///
/// Returns zero if `x` has fewer than two rows.
///
/// ### Params
///
/// * `x` - The matrix for which to calculate the within distance
/// * `dist` - Which [PertDistance] to calculate
///
/// ### Returns
///
/// The within pert distance
pub fn mean_pairwise_within<T>(x: MatRef<T>, dist: PertDistance) -> T
where
    T: BixverseFloat + std::iter::Sum,
{
    mean_pairwise_within_impl(x, dist, true)
}

/// Within-group mean distance, optionally single-threaded
///
/// ### Params
///
/// * `x` - The matrix for which to calculate the within distance
/// * `dist` - Which [PertDistance] to calculate
/// * `parallel` - Use rayon; false for calls made inside an outer parallel loop
///
/// ### Returns
///
/// The within pert distance
fn mean_pairwise_within_impl<T>(x: MatRef<T>, dist: PertDistance, parallel: bool) -> T
where
    T: BixverseFloat + std::iter::Sum,
{
    let n = x.nrows();
    if n < 2 {
        return T::zero();
    }

    // X * X^T, lower triangle: diagonal is ||x_i||^2, sub-diagonal is <x_i, x_j>.
    let mut gram = Mat::<T>::zeros(n, n);
    gemm::gram(
        gram.as_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        x,
        x.transpose(),
        T::one(),
        if parallel {
            faer_parallelism()
        } else {
            Par::Seq
        },
    );

    let norms_sq: Vec<T> = (0..n).map(|i| *gram.get(i, i)).collect();
    let two = T::from_f64(2.0).unwrap();

    // Column j of the lower triangle is contiguous.
    let col_sum = |j: usize| -> T {
        let mut s = T::zero();
        for (i, &g) in gram.col(j).iter().enumerate().skip(j + 1) {
            // clamp: floating point can produce tiny negatives here.
            let d_sq = (norms_sq[i] + norms_sq[j] - two * g).max(T::zero());
            s += finish_dist(d_sq, dist);
        }
        s
    };
    let total: T = if parallel {
        (0..n).into_par_iter().map(col_sum).sum()
    } else {
        (0..n).map(col_sum).sum()
    };

    let n_pairs = T::from_usize(n * (n - 1) / 2).unwrap();

    total / n_pairs
}

/// Mean pairwise distance between two groups (rows are samples).
///
/// ### Params
///
/// * `x` - First matrix
/// * `y` - Second matrix
///
/// ### Returns
///
/// The pairwise mean distance between the two matrices
pub fn mean_pairwise_between<T>(
    x: MatRef<T>,
    y: MatRef<T>,
    dist: PertDistance,
) -> Result<T, BixverseErrors>
where
    T: BixverseFloat + std::iter::Sum,
{
    mean_pairwise_between_impl(x, y, dist, true)
}

/// Between-group mean distance, optionally single-threaded
///
/// Tiled over row blocks of `x`, so the full `n_x x n_y` gram is never held.
///
/// ### Params
///
/// * `x` - First matrix
/// * `y` - Second matrix
/// * `dist` - Which [PertDistance] to calculate
/// * `parallel` - Use rayon; false for calls made inside an outer parallel loop
///
/// ### Returns
///
/// The pairwise mean distance between the two matrices
fn mean_pairwise_between_impl<T>(
    x: MatRef<T>,
    y: MatRef<T>,
    dist: PertDistance,
    parallel: bool,
) -> Result<T, BixverseErrors>
where
    T: BixverseFloat + std::iter::Sum,
{
    if x.ncols() != y.ncols() {
        return Err(BixverseErrors::NonMatchingFeatureDim {
            dim_x: x.ncols(),
            dim_y: y.ncols(),
        });
    }

    let n_x = x.nrows();
    let n_y = y.nrows();
    // special case, return zero
    if n_x == 0 || n_y == 0 {
        return Ok(T::zero());
    }

    let x_norms = row_norms_squared(x);
    let y_norms = row_norms_squared(y);
    let two = T::from_f64(2.0).unwrap();
    let par = if parallel {
        faer_parallelism()
    } else {
        Par::Seq
    };

    let mut gram = Mat::<T>::zeros(GRAM_ROW_TILE.min(n_x), n_y);
    let mut total = T::zero();
    for start in (0..n_x).step_by(GRAM_ROW_TILE) {
        let rows = GRAM_ROW_TILE.min(n_x - start);
        let mut tile = gram.as_mut().submatrix_mut(0, 0, rows, n_y);
        gemm::gemm(
            tile.as_mut(),
            Accum::Replace,
            x.subrows(start, rows),
            y.transpose(),
            T::one(),
            par,
        );
        let tile = tile.as_ref();
        let x_tile = &x_norms[start..start + rows];

        let col_sum = |j: usize| -> T {
            let mut s = T::zero();
            for (&g, &xn) in tile.col(j).iter().zip(x_tile) {
                let d_sq = (xn + y_norms[j] - two * g).max(T::zero());
                s += finish_dist(d_sq, dist);
            }
            s
        };
        let tile_total: T = if parallel {
            (0..n_y).into_par_iter().map(col_sum).sum()
        } else {
            (0..n_y).map(col_sum).sum()
        };
        total += tile_total;
    }

    let n_pairs = T::from_usize(n_x * n_y).unwrap();

    Ok(total / n_pairs)
}

/// Pack rows of a (samples x features) matrix into one `Mat<T>` per label.
///
/// ### Params
///
/// * `embedding` - The original embedding matrix of shape samples x features
/// * `labels` - A grouping vector to use. Needs to have the same length as
///   nrows embedding.
///
/// ### Returns
///
/// A vector of matrices per label
fn group_rows_by_label<T>(
    embedding: MatRef<T>,
    labels: &[usize],
) -> Result<Vec<Mat<T>>, BixverseErrors>
where
    T: BixverseFloat,
{
    if embedding.nrows() != labels.len() {
        return Err(BixverseErrors::NumberLabelsNotEqualSampleNumber {
            label_length: labels.len(),
            n_samples: embedding.nrows(),
        });
    }

    let d = embedding.ncols();
    let n_groups = labels.iter().max().map(|&x| x + 1).unwrap_or(0);

    let mut indices: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (sample, &g) in labels.iter().enumerate() {
        indices[g].push(sample);
    }

    let res = indices
        .into_par_iter()
        .map(|rows| Mat::from_fn(rows.len(), d, |i, j| *embedding.get(rows[i], j)))
        .collect();

    Ok(res)
}

////////////////////
// Main functions //
////////////////////

/// E-distance between two groups, each given as (samples x features).
///
/// `2 * mean(d(X, Y)) - mean(d(X, X')) - mean(d(Y, Y'))`
///
/// ### Params
///
/// * `x` - First matrix
/// * `y` - Second matrix
/// * `dist` - The perturbation distance to calculate
///
/// ### Returns
///
/// The E-distance between the two groups
pub fn edistance_two_matrices<T>(
    x: MatRef<T>,
    y: MatRef<T>,
    dist: PertDistance,
) -> Result<T, BixverseErrors>
where
    T: BixverseFloat + std::iter::Sum,
{
    let within_x = mean_pairwise_within(x, dist);
    let within_y = mean_pairwise_within(y, dist);
    let between = mean_pairwise_between(x, y, dist)?;
    let two = T::from_f64(2.0).unwrap();

    Ok(two * between - within_x - within_y)
}

/// E-distance between two groups, single-threaded
///
/// For calls made from inside an outer parallel loop, where nested rayon or
/// faer parallelism would only oversubscribe.
///
/// ### Params
///
/// * `x` - First matrix
/// * `y` - Second matrix
/// * `dist` - The perturbation distance to calculate
///
/// ### Returns
///
/// The E-distance between the two groups
pub(crate) fn edistance_two_matrices_seq<T>(
    x: MatRef<T>,
    y: MatRef<T>,
    dist: PertDistance,
) -> Result<T, BixverseErrors>
where
    T: BixverseFloat + std::iter::Sum,
{
    let within_x = mean_pairwise_within_impl(x, dist, false);
    let within_y = mean_pairwise_within_impl(y, dist, false);
    let between = mean_pairwise_between_impl(x, y, dist, false)?;
    let two = T::from_f64(2.0).unwrap();

    Ok(two * between - within_x - within_y)
}

/// Pairwise E-distances between all groups defined by `labels`.
///
/// ### Params
///
/// * `embedding` - Data matrix (samples x features)
/// * `labels` - Per-sample group label (values in `0..n_groups`)
/// * `dist` - Distance variant
///
/// ### Returns
///
/// Symmetric `n_groups x n_groups` E-distance matrix with a zero diagonal.
pub fn pairwise_edistance<T>(
    embedding: MatRef<T>,
    labels: &[usize],
    dist: PertDistance,
) -> Result<Mat<T>, BixverseErrors>
where
    T: BixverseFloat + std::iter::Sum,
{
    let groups = group_rows_by_label(embedding, labels)?;
    let n_groups = groups.len();

    // Only nest faer/rayon inside the outer loop when it cannot fill the pool.
    let inner_par = groups.len() < rayon::current_num_threads();
    let withins: Vec<T> = groups
        .par_iter()
        .map(|g| mean_pairwise_within_impl(g.as_ref(), dist, inner_par))
        .collect();

    let pairs: Vec<(usize, usize)> = (0..n_groups)
        .flat_map(|i| (i + 1..n_groups).map(move |j| (i, j)))
        .collect();

    let two = T::from_f64(2.0).unwrap();
    let results: Vec<(usize, usize, T)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let between = mean_pairwise_between_impl(
                groups[i].as_ref(),
                groups[j].as_ref(),
                dist,
                inner_par,
            )?;
            let ed = two * between - withins[i] - withins[j];

            Ok((i, j, ed))
        })
        .collect::<Result<Vec<_>, BixverseErrors>>()?;

    let mut result = Mat::<T>::zeros(n_groups, n_groups);
    for (i, j, val) in results {
        result[(i, j)] = val;
        result[(j, i)] = val;
    }

    Ok(result)
}

/// One-sided E-distances from each group to a reference group.
///
/// ### Params
///
/// * `embedding` - Data matrix (samples x features)
/// * `labels` - Per-sample group label
/// * `reference` - Label of the reference group (e.g. control)
/// * `dist` - Distance variant
///
/// ### Returns
///
/// Vector of length `n_groups`; the entry at `reference` is zero.
pub fn onesided_edistance<T>(
    embedding: MatRef<T>,
    labels: &[usize],
    reference: usize,
    dist: PertDistance,
) -> Result<Vec<T>, BixverseErrors>
where
    T: BixverseFloat + std::iter::Sum,
{
    let groups = group_rows_by_label(embedding, labels)?;
    let n_groups = groups.len();
    assert!(reference < n_groups, "reference label out of range");

    let inner_par = n_groups < rayon::current_num_threads();
    let within_ref = mean_pairwise_within(groups[reference].as_ref(), dist);
    let two = T::from_f64(2.0).unwrap();

    let res = (0..n_groups)
        .into_par_iter()
        .map(|g| {
            if g == reference {
                Ok(T::zero())
            } else {
                let within_g = mean_pairwise_within_impl(groups[g].as_ref(), dist, inner_par);
                let between = mean_pairwise_between_impl(
                    groups[g].as_ref(),
                    groups[reference].as_ref(),
                    dist,
                    inner_par,
                )?;

                Ok(two * between - within_g - within_ref)
            }
        })
        .collect::<Result<Vec<_>, BixverseErrors>>()?;

    Ok(res)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    /// The R-facing parser is case insensitive, accepts both aliases and rejects the rest.
    #[test]
    fn parse_pert_distance_variants() {
        assert!(matches!(
            parse_perturbation_distance("euclidean"),
            Some(PertDistance::Euclidean)
        ));
        assert!(matches!(
            parse_perturbation_distance("Euclidean"),
            Some(PertDistance::Euclidean)
        ));
        assert!(matches!(
            parse_perturbation_distance("squared_euclidean"),
            Some(PertDistance::SquaredEuclidean)
        ));
        assert!(matches!(
            parse_perturbation_distance("sqeuclidean"),
            Some(PertDistance::SquaredEuclidean)
        ));
        assert!(parse_perturbation_distance("manhattan").is_none());
        assert!(parse_perturbation_distance("").is_none());
    }

    /// With no pair to average over the mean is defined as zero, not NaN from a zero divisor.
    #[test]
    fn within_lt_two_rows_is_zero() {
        let single = Mat::<f64>::from_fn(1, 3, |_, j| j as f64);
        assert_eq!(
            mean_pairwise_within(single.as_ref(), PertDistance::Euclidean),
            0.0
        );
        let empty = Mat::<f64>::zeros(0, 3);
        assert_eq!(
            mean_pairwise_within(empty.as_ref(), PertDistance::Euclidean),
            0.0
        );
    }

    /// Pins the within-group mean against a hand-computed value for both metrics.
    #[test]
    fn within_three_points_known() {
        // Right triangle (0,0), (3,0), (0,4): pairwise 3, 4, 5; mean 4.
        let m = Mat::<f64>::from_fn(3, 2, |i, j| match (i, j) {
            (1, 0) => 3.0,
            (2, 1) => 4.0,
            _ => 0.0,
        });
        let d = mean_pairwise_within(m.as_ref(), PertDistance::Euclidean);
        assert!(close(d, 4.0));
        let d_sq = mean_pairwise_within(m.as_ref(), PertDistance::SquaredEuclidean);
        assert!(close(d_sq, (9.0 + 16.0 + 25.0) / 3.0));
    }

    /// Mismatched feature counts error rather than reading past the shorter row.
    #[test]
    fn between_dim_mismatch_errors() {
        let x = Mat::<f64>::zeros(2, 3);
        let y = Mat::<f64>::zeros(2, 4);
        assert!(matches!(
            mean_pairwise_between(x.as_ref(), y.as_ref(), PertDistance::Euclidean),
            Err(BixverseErrors::NonMatchingFeatureDim { .. })
        ));
    }

    /// An empty group on one side yields zero instead of dividing by an empty product.
    #[test]
    fn between_empty_returns_zero() {
        let x = Mat::<f64>::zeros(0, 2);
        let y = Mat::<f64>::from_fn(2, 2, |i, _| i as f64);
        assert_eq!(
            mean_pairwise_between(x.as_ref(), y.as_ref(), PertDistance::Euclidean).unwrap(),
            0.0
        );
    }

    /// Pins the between-group mean against a hand-computed value.
    #[test]
    fn between_known_value() {
        // X = {(0,0)}, Y = {(3,4), (6,8)}. Distances 5, 10; mean 7.5.
        let x = Mat::<f64>::zeros(1, 2);
        let y = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 3.0,
            (0, 1) => 4.0,
            (1, 0) => 6.0,
            (1, 1) => 8.0,
            _ => 0.0,
        });
        let d = mean_pairwise_between(x.as_ref(), y.as_ref(), PertDistance::Euclidean).unwrap();
        assert!(close(d, 7.5));
    }

    /// The energy distance is symmetric in its two groups despite the asymmetric argument order.
    #[test]
    fn edistance_symmetric() {
        let x = Mat::<f64>::from_fn(4, 3, |i, j| (i + j) as f64);
        let y = Mat::<f64>::from_fn(5, 3, |i, j| (i * 2 + j) as f64 - 1.0);
        let exy = edistance_two_matrices(x.as_ref(), y.as_ref(), PertDistance::Euclidean).unwrap();
        let eyx = edistance_two_matrices(y.as_ref(), x.as_ref(), PertDistance::Euclidean).unwrap();
        assert!(close(exy, eyx));
    }

    /// The one closed form the energy distance has, so it pins the constant factor.
    #[test]
    fn edistance_singletons_squared_identity() {
        // Singletons: within = 0, so E = 2*||mu_x - mu_y||^2 exactly.
        let x = Mat::<f64>::zeros(1, 2);
        let y = Mat::<f64>::from_fn(1, 2, |_, j| if j == 0 { 3.0 } else { 4.0 });
        let e =
            edistance_two_matrices(x.as_ref(), y.as_ref(), PertDistance::SquaredEuclidean).unwrap();
        assert!(close(e, 50.0));
    }

    /// The pairwise matrix is group by group, symmetric, and zero on the diagonal.
    #[test]
    fn pairwise_edistance_shape_and_symmetry() {
        let emb = Mat::<f64>::from_fn(6, 2, |i, j| (i + j) as f64);
        let labels = vec![0, 0, 1, 1, 2, 2];
        let m = pairwise_edistance(emb.as_ref(), &labels, PertDistance::Euclidean).unwrap();
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
        for i in 0..3 {
            assert_eq!(*m.get(i, i), 0.0);
            for j in (i + 1)..3 {
                assert!(close(*m.get(i, j), *m.get(j, i)));
            }
        }
    }

    /// Fewer labels than rows errors rather than silently dropping the tail.
    #[test]
    fn pairwise_edistance_label_count_mismatch() {
        let emb = Mat::<f64>::zeros(4, 2);
        let labels = vec![0_usize, 0, 1];
        assert!(matches!(
            pairwise_edistance(emb.as_ref(), &labels, PertDistance::Euclidean),
            Err(BixverseErrors::NumberLabelsNotEqualSampleNumber { .. })
        ));
    }

    /// The reference group keeps its slot in the output and scores zero against itself.
    #[test]
    fn onesided_reference_entry_is_zero() {
        let emb = Mat::<f64>::from_fn(6, 2, |i, j| (i + j) as f64);
        let labels = vec![0_usize, 0, 1, 1, 2, 2];
        let res = onesided_edistance(emb.as_ref(), &labels, 1, PertDistance::Euclidean).unwrap();
        assert_eq!(res.len(), 3);
        assert_eq!(res[1], 0.0);
    }

    /// The cheap one-sided path must agree with the column it replaces in the full matrix.
    #[test]
    fn onesided_matches_pairwise_column() {
        let emb = Mat::<f64>::from_fn(8, 3, |i, j| ((i + 1) * (j + 1)) as f64);
        let labels = vec![0_usize, 0, 0, 1, 1, 2, 2, 2];
        let pw = pairwise_edistance(emb.as_ref(), &labels, PertDistance::Euclidean).unwrap();
        let os = onesided_edistance(emb.as_ref(), &labels, 0, PertDistance::Euclidean).unwrap();
        for g in 0..3 {
            assert!(close(*pw.get(g, 0), os[g]));
        }
    }
}
