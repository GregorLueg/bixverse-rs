use faer::{Mat, MatRef};
use rayon::iter::*;

use crate::core::math::matrix_helpers::*;
use crate::prelude::*;
use crate::utils::utils_mat::*;

///////////
// Enums //
///////////

/// Enum for the RBF function
#[derive(Debug, Default)]
pub enum RbfType {
    #[default]
    Gaussian,
    Bump,
    InverseQuadratic,
}

/// Parsing the RBF function
///
/// ### Params
///
/// * `s` - String to transform into `RbfType`
///
/// ### Returns
///
/// Returns the `RbfType`
pub fn parse_rbf_types(s: &str) -> Option<RbfType> {
    match s.to_lowercase().as_str() {
        "gaussian" => Some(RbfType::Gaussian),
        "bump" => Some(RbfType::Bump),
        "inverse_quadratic" => Some(RbfType::InverseQuadratic),
        _ => None,
    }
}

//////////////
// Gaussian //
//////////////

/// Gaussian Radial Basis function
///
/// Applies a Gaussian Radial Basis function on a vector of distances with the
/// following formula:
/// `φ(r) = e^(-(εr)²)`
///
/// ### Params
///
/// * `dist` - Vector of distances
/// * `epsilon` - Shape parameter controlling function width
///
/// ### Returns
///
/// The resulting affinity vector
pub fn rbf_gaussian<T>(dist: &[T], epsilon: &T) -> Vec<T>
where
    T: BixverseFloat,
{
    dist.par_iter()
        .map(|x| T::exp(-((*x * *epsilon).powi(2))))
        .collect()
}

/// Gaussian Radial Basis function for matrices.
///
/// Applies a Gaussian Radial Basis function on a matrix of distances with the
/// following formula:
///
/// `φ(r) = e^(-(εr)²)`
///
/// ### Params
///
/// * `dist` - Matrix of distances
/// * `epsilon` - Shape parameter controlling function width
///
/// ### Returns
///
/// The affinity matrix
pub fn rbf_gaussian_mat<T>(dist: MatRef<T>, epsilon: &T) -> Mat<T>
where
    T: BixverseFloat,
{
    let ncol = dist.ncols();
    let nrow = dist.nrows();
    Mat::from_fn(nrow, ncol, |i, j| {
        let x = *dist.get(i, j);
        T::exp(-((x * *epsilon).powi(2)))
    })
}

//////////
// Bump //
//////////

/// Bump Radial Basis function
///
/// Applies a Bump Radial Basis function on a vector of distances with the
/// following formula:
/// `
/// φ(r) = { exp(-1/(1-(εr)²)) + 1,  if εr < 1
///        { 0,                      if εr ≥ 1
/// `
///
/// ### Params
///
/// * `dist` - Vector of distances
/// * `epsilon` - Shape parameter controlling function width
///
/// ### Returns
///
/// The resulting affinity vector
pub fn rbf_bump<T>(dist: &[T], epsilon: &T) -> Vec<T>
where
    T: BixverseFloat,
{
    dist.par_iter()
        .map(|x| {
            if *x < (T::one() / *epsilon) {
                T::exp(-(T::one() / (T::one() - (*epsilon * *x).powi(2))) + T::one())
            } else {
                T::zero()
            }
        })
        .collect()
}

/// Bump Radial Basis function for matrices
///
/// Applies a Bump Radial Basis function on a matrix of distances with the
/// following formula:
/// `
/// φ(r) = { exp(-1/(1-(εr)²)) + 1,  if εr < 1
///        { 0,                      if εr ≥ 1
/// `
///
/// ### Params
///
/// * `dist` - Matrix of distances
/// * `epsilon` - Shape parameter controlling function width
///
/// ### Returns
///
/// The resulting affinity matrix
pub fn rbf_bump_mat<T>(dist: MatRef<T>, epsilon: &T) -> Mat<T>
where
    T: BixverseFloat,
{
    let ncol = dist.ncols();
    let nrow = dist.nrows();
    Mat::from_fn(nrow, ncol, |i, j| {
        let x = dist.get(i, j);
        if *x < (T::one() / *epsilon) {
            T::exp(-(T::one() / (T::one() - (*epsilon * *x).powi(2))) + T::one())
        } else {
            T::zero()
        }
    })
}

///////////////////////
// Inverse quadratic //
///////////////////////

/// Inverse quadratic RBF
///
/// Applies a Inverse Quadratic Radial Basis function on a vector of distances
/// with the following formula:
/// `
/// φ(r) = 1/(1 + (εr)²)
/// `
///
/// ### Params
///
/// * `dist` - Vector of distances
/// * `epsilon` - Shape parameter controlling function width
///
/// ### Return
///
/// The resulting affinity vector
pub fn rbf_inverse_quadratic<T>(dist: &[T], epsilon: &T) -> Vec<T>
where
    T: BixverseFloat,
{
    dist.par_iter()
        .map(|x| T::one() / (T::one() + (*epsilon * *x).powi(2)))
        .collect()
}

/// Inverse quadratic RBF for matrices
///
/// Applies a Inverse Quadratic Radial Basis function on a matrix of distances
/// with the following formula:
/// `
/// φ(r) = 1/(1 + (εr)²)
/// `
///
/// ### Params
///
/// * `dist` - Matrix of distances
/// * `epsilon` - Shape parameter controlling function width
///
/// ### Returns
///
/// The resulting affinity matrix
pub fn rbf_inverse_quadratic_mat<T>(dist: MatRef<T>, epsilon: &T) -> Mat<T>
where
    T: BixverseFloat,
{
    let ncol = dist.ncols();
    let nrow = dist.nrows();
    Mat::from_fn(nrow, ncol, |i, j| {
        let x = dist.get(i, j);
        T::one() / (T::one() + (*epsilon * *x).powi(2))
    })
}

////////////
// Others //
////////////

/// Test different epsilons over a distance vector
///
/// ### Params
///
/// * `dist` - The distance vector on which to apply the specified RBF function.
///   Assumes that these are the values of upper triangle of the distance
///   matrix.
/// * `epsilons` - Vector of epsilons to test.
/// * `n` - Original dimensions of the distance matrix from which `dist` was
///   derived.
/// * `shift` - Was a shift applied during the generation of the vector, i.e., was
///   the diagonal included or not.
/// * `rbf_type` - Which RBF function to apply on the distance vector.
///
/// ### Returns
///
/// The column sums of the resulting adjacency matrices after application of the
/// RBF function to for example check if these are following power law distributions.
pub fn rbf_iterate_epsilons<T>(
    dist: &[T],
    epsilons: &[T],
    n: usize,
    shift: usize,
    rbf_type: &str,
) -> Mat<T>
where
    T: BixverseFloat,
{
    // Now specifying String as the error type
    let rbf_fun = parse_rbf_types(rbf_type).unwrap_or_default();

    let k_res: Vec<Vec<T>> = epsilons
        .par_iter()
        .map(|epsilon| {
            let affinity_adj = match rbf_fun {
                RbfType::Gaussian => rbf_gaussian(dist, epsilon),
                RbfType::Bump => rbf_bump(dist, epsilon),
                RbfType::InverseQuadratic => rbf_inverse_quadratic(dist, epsilon),
            };
            let affinity_adj_mat = upper_triangle_to_sym_faer(&affinity_adj, shift, n);
            col_sums(affinity_adj_mat.as_ref())
        })
        .collect();

    nested_vector_to_faer_mat(k_res, true)
}
