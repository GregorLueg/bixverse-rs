//! Radial basis function to transform distances

use faer::{Mat, MatRef};
use rayon::iter::*;

use crate::prelude::*;
use crate::utils::matrix_utils::*;

////////////
// Consts //
////////////

/// Matrix size (elements) from which an elementwise map fans out over rayon
const PAR_MAP_MIN_ELEMS: usize = 1 << 16;

///////////
// Enums //
///////////

/// Enum for the RBF function
#[derive(Debug, Default)]
pub enum RbfType {
    /// Gaussian radial basis function
    #[default]
    Gaussian,
    /// Bump radial basis function
    Bump,
    /// Inverse quadratic radial basis function
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

/// Apply a scalar function to every entry of a matrix
///
/// ### Params
///
/// * `mat` - Input matrix
/// * `f` - Function applied per entry
///
/// ### Returns
///
/// The mapped matrix, same shape.
fn map_mat_par<T, F>(mat: MatRef<T>, f: F) -> Mat<T>
where
    T: BixverseFloat,
    F: Fn(T) -> T + Sync,
{
    let mut out = Mat::<T>::zeros(mat.nrows(), mat.ncols());
    let fill = |j: usize, col: faer::ColMut<T>| {
        for (o, &x) in col.iter_mut().zip(mat.col(j).iter()) {
            *o = f(x);
        }
    };
    if mat.nrows() * mat.ncols() >= PAR_MAP_MIN_ELEMS {
        out.par_col_iter_mut()
            .enumerate()
            .for_each(|(j, col)| fill(j, col));
    } else {
        for j in 0..mat.ncols() {
            fill(j, out.col_mut(j));
        }
    }
    out
}

//////////////
// Gaussian //
//////////////

/// Gaussian Radial Basis function
///
/// Applies a Gaussian Radial Basis function on distances with the formula:
/// y = exp(-(eps * r)^2)
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
/// Applies a Gaussian Radial Basis function on distances with the formula:
/// y = exp(-(eps * r)^2)
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
    map_mat_par(dist, |x| T::exp(-((x * *epsilon).powi(2))))
}

//////////
// Bump //
//////////

/// Bump Radial Basis function
///
/// Applies a Bump Radial Basis function on distances with the formula:
/// y = exp(-1 / (1 - (eps * r)^2) + 1)   if eps * r < 1
/// y = 0                                 if eps * r >= 1
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
/// Applies a Bump Radial Basis function on distances with the formula:
/// y = exp(-1 / (1 - (eps * r)^2) + 1)   if eps * r < 1
/// y = 0                                 if eps * r >= 1
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
    map_mat_par(dist, |x| {
        if x < (T::one() / *epsilon) {
            T::exp(-(T::one() / (T::one() - (*epsilon * x).powi(2))) + T::one())
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
/// Applies an Inverse Quadratic Radial Basis function on distances with the
/// formula:
/// y = 1 / (1 + (eps * r)^2)
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
/// Applies an Inverse Quadratic Radial Basis function on distances with the
/// formula:
/// y = 1 / (1 + (eps * r)^2)
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
    map_mat_par(dist, |x| T::one() / (T::one() + (*epsilon * x).powi(2)))
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
            // Sequential maps: the outer loop over epsilons already fills the pool.
            let affinity_adj: Vec<T> = match rbf_fun {
                RbfType::Gaussian => dist
                    .iter()
                    .map(|x| T::exp(-((*x * *epsilon).powi(2))))
                    .collect(),
                RbfType::Bump => dist
                    .iter()
                    .map(|x| {
                        if *x < (T::one() / *epsilon) {
                            T::exp(-(T::one() / (T::one() - (*epsilon * *x).powi(2))) + T::one())
                        } else {
                            T::zero()
                        }
                    })
                    .collect(),
                RbfType::InverseQuadratic => dist
                    .iter()
                    .map(|x| T::one() / (T::one() + (*epsilon * *x).powi(2)))
                    .collect(),
            };
            let affinity_adj_mat = upper_triangle_to_sym_faer(&affinity_adj, shift, n);
            affinity_adj_mat
                .col_iter()
                .map(|col| col.iter().fold(T::zero(), |acc, &v| acc + v))
                .collect()
        })
        .collect();

    nested_vector_to_faer_mat(k_res, true)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel name parser ignores case and rejects anything it does not know.
    #[test]
    fn test_parse_rbf_types() {
        assert!(matches!(
            parse_rbf_types("gaussian"),
            Some(RbfType::Gaussian)
        ));
        assert!(matches!(parse_rbf_types("BUMP"), Some(RbfType::Bump)));
        assert!(matches!(
            parse_rbf_types("inverse_quadratic"),
            Some(RbfType::InverseQuadratic)
        ));
        assert!(parse_rbf_types("unknown").is_none());
    }

    /// The Gaussian kernel `exp(-(eps d)^2)` at zero and unit distance.
    #[test]
    fn test_rbf_gaussian() {
        let dist: Vec<f64> = vec![0.0, 1.0];
        let eps = 1.0;
        let res = rbf_gaussian(&dist, &eps);

        assert!((res[0] - 1.0).abs() < 1e-6);
        assert!((res[1] - std::f64::consts::E.powi(-1)).abs() < 1e-6);
    }

    /// The inverse quadratic kernel `1 / (1 + (eps d)^2)` at distances zero, one and two.
    #[test]
    fn test_rbf_inverse_quadratic() {
        let dist: Vec<f64> = vec![0.0, 1.0, 2.0];
        let eps = 1.0;
        let res = rbf_inverse_quadratic(&dist, &eps);

        assert!((res[0] - 1.0).abs() < 1e-6);
        assert!((res[1] - 0.5).abs() < 1e-6);
        assert!((res[2] - 0.2).abs() < 1e-6);
    }

    /// The bump kernel peaks at one and is exactly zero outside its `1 / eps` support.
    #[test]
    fn test_rbf_bump() {
        let dist: Vec<f64> = vec![0.0, 2.0];
        let eps = 1.0;
        let res = rbf_bump(&dist, &eps);

        // The implementation scales via +1 inside the exp: exp(-1/(1 - 0) + 1) = exp(0) = 1.0
        assert!((res[0] - 1.0).abs() < 1e-6);
        // 2.0 > 1.0/eps, should return 0.0
        assert!((res[1] - 0.0).abs() < 1e-6);
    }
}
