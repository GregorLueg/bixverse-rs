use faer::{
    Mat, MatRef, Scale,
    linalg::solvers::{PartialPivLu, Solve},
};
use rand::prelude::*;
use rand_distr::Distribution;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::core::math::matrix_helpers::scale_matrix_col;
use crate::core::math::pca_svd::randomised_svd;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Enum for the ICA types
#[derive(Clone, Debug, Default)]
pub enum IcaType {
    /// Use the `LogCosh` type implementation of fastICA
    #[default]
    LogCosh,
    /// Use the `Exp` type implementation of fastICA
    Exp,
}

/// Parsing the ICA types
///
/// ### Params
///
/// * `s` - string defining the ICA type
///
/// ### Returns
///
/// The `IcaType`.
pub fn parse_ica_type(s: &str) -> Option<IcaType> {
    match s.to_lowercase().as_str() {
        "exp" => Some(IcaType::Exp),
        "logcosh" => Some(IcaType::LogCosh),
        _ => None,
    }
}

/// Type alias of the ICA results
///
/// ### Fields
///
/// * `0` - Mixing matrix w
/// * `1` - Tolerance
type IcaRes<T> = (Mat<T>, T);

/// Structure to save ICA parameters
///
/// ### Fields
///
/// * `maxit` - Maximum number of iterations to run ICA for.
/// * `alpha` - Alpha parameter for the `logcosh` variant.
/// * `tol` - Tolerance parameter.
/// * `verbose` - Shall print messages be returned.
#[derive(Clone, Debug)]
pub struct IcaParams<T: BixverseFloat> {
    pub maxit: usize,
    pub alpha: T,
    pub tol: T,
    pub verbose: bool,
}

/// Prepare the whitening.
///
/// This is needed pre-processing for ICA. Has the option to use randomised SVD
/// for faster computations.
///
/// ### Params
///
/// * `x` - The pre-processed matrix on which to apply ICA.
/// * `fast_svd` - Shall the faster version of SVD be used.
/// * `seed` - Random seed for reproducibility purposes
/// * `rank` - The target rank of the approximation (number of singular values,
///   vectors to compute).
/// * `oversampling` - Additional samples beyond the target rank to improve accuracy.
///   Defaults to 10 if not specified.
/// * `n_power_iter` - Number of power iterations to perform for better approximation
///   quality. More iterations generally improve accuracy but increase computation time.
///   Defaults to 2 if not specified.
///
/// ### Returns
///
/// A tuple of the processed matrix (pre-whitening) and the whitening matrix K.
pub fn prepare_whitening<T: BixverseFloat>(
    x: MatRef<T>,
    fast_svd: bool,
    seed: usize,
    rank: usize,
    oversampling: Option<usize>,
    n_power_iter: Option<usize>,
) -> (Mat<T>, Mat<T>) {
    let n = x.nrows();

    let centered = scale_matrix_col(&x, false);

    let centered = centered.transpose();

    let n_recip = T::from_usize(n).unwrap().recip();
    let v = Scale(n_recip) * (centered * centered.transpose());

    let k = if fast_svd {
        let svd_res = randomised_svd(v.as_ref(), rank, seed, oversampling, n_power_iter);
        let s: Vec<T> = svd_res.s.iter().map(|x| x.sqrt().recip()).collect();
        let d = faer_diagonal_from_vec(s);
        d * svd_res.u.transpose()
    } else {
        let svd_res = v.thin_svd().unwrap();
        let s = svd_res
            .S()
            .column_vector()
            .iter()
            .map(|x| x.sqrt().recip())
            .collect::<Vec<_>>();
        let d = faer_diagonal_from_vec(s);
        let u = svd_res.U();
        let u_t = u.transpose();
        d * u_t
    };

    (centered.cloned(), k)
}

/// Helper function to update the mixing matrix for ICA
///
/// ### Params
///
/// * `w` The mixing matrix
///
/// ### Returns
///
/// The updated mixing matrix.
fn update_mix_mat<T: BixverseFloat>(w: MatRef<T>) -> faer::Mat<T> {
    // SVD
    let svd_res = w.thin_svd().unwrap();

    let s = svd_res.S();
    let u = svd_res.U();
    let s = s
        .column_vector()
        .iter()
        .map(|x| x.recip())
        .collect::<Vec<_>>();
    let d = faer_diagonal_from_vec(s);

    u * d * u.transpose() * w
}

/// Generate a random mixing matrix
///
/// ### Params
///
/// * `n_comp` - Number of independent components. This will influence the dimensionality
///   of the randomly initialised mixing matrix.
/// * `seed` - Random seed for reproducibility purposes
///
/// ### Returns
///
/// Mixing matrix of dimensions `n_comp` x `n_comp`
fn create_w_init<T: BixverseFloat>(n_comp: usize, seed: u64) -> faer::Mat<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let vec_size = n_comp.pow(2);
    let data: Vec<T> = (0..vec_size)
        .map(|_| T::from_f64(normal.sample(&mut rng)).unwrap())
        .collect();

    Mat::from_fn(n_comp, n_comp, |i, j| data[i + j * n_comp])
}

///////////////////////////
// Cross validation data //
///////////////////////////

/// Structure to save ICA CV results
///
/// ### Fields
///
/// * `pre_white_matrices` - Vector of pre-processed matrices, ready for whitening
/// * `k_matrices` - Vector of pre-whitening matrices
pub struct IcaCvData<T: BixverseFloat> {
    pub pre_white_matrices: Vec<Mat<T>>,
    pub k_matrices: Vec<Mat<T>>,
}

impl<T: BixverseFloat> IcaCvData<T> {
    /// Generate the `IcaCvData` from data
    ///
    /// This function will default to the faster randomised SVD for the whitening
    /// process.
    ///
    /// ### Params
    ///
    /// * `x` - The matrix for which to generate a cross-validation like set
    /// * `num_folds` - In how many folds to split the data
    /// * `seed` - Random seed for reproducibility purposes
    /// * `rank` - Optional rank for the randomised SVD that is used to generate the
    ///   the pre-whitened matrix and whitening matrix k.
    ///
    /// ### Returns
    ///
    /// An `IcaCvData` structure.
    pub fn create_from_data(
        x: MatRef<T>,
        num_folds: usize,
        seed: usize,
        rank: Option<usize>,
    ) -> IcaCvData<T> {
        let no_samples = x.nrows();
        let no_features = x.ncols();
        let mut indices: Vec<usize> = (0..no_samples).collect();
        let mut rng = StdRng::seed_from_u64(seed as u64);
        indices.shuffle(&mut rng);

        let svd_rank = rank.unwrap_or(no_features);

        let fold_size = no_samples / num_folds;
        let remainder = no_samples % num_folds;

        let mut folds = Vec::with_capacity(num_folds);
        let mut start = 0;

        for idx in 0..num_folds {
            let current_fold_size = if idx < remainder {
                fold_size + 1
            } else {
                fold_size
            };

            let end = start + current_fold_size;
            folds.push(indices[start..end].to_vec());
            start = end;
        }

        let k_x_matrices: Vec<(Mat<T>, Mat<T>)> = folds
            .par_iter()
            .map(|test_indices| {
                let train_indices: Vec<usize> = indices
                    .iter()
                    .filter(|&idx| !test_indices.contains(idx))
                    .cloned()
                    .collect();
                let mut x_i = Mat::<T>::zeros(train_indices.len(), no_features);

                for (new_row, old_row) in train_indices.iter().enumerate() {
                    for j in 0..no_features {
                        x_i[(new_row, j)] = x[(*old_row, j)];
                    }
                }

                prepare_whitening(x_i.as_ref(), true, seed + 1, svd_rank, None, None)
            })
            .collect();

        let mut pre_white_matrices = Vec::with_capacity(num_folds);
        let mut k_matrices = Vec::with_capacity(num_folds);

        for (x_i, k_i) in k_x_matrices {
            pre_white_matrices.push(x_i);
            k_matrices.push(k_i);
        }

        Self {
            pre_white_matrices,
            k_matrices,
        }
    }
}

////////////////////
// Main functions //
////////////////////

/// Fast ICA implementation based on logcosh.
///
/// ### Params
///
/// * `x` - Whitened matrix
/// * `w_init` - Initial, random mixing matrix
/// * `maxit` - Maximum number of iterations to run ICA for.
/// * `alpha` - Alpha parameter for this variant
/// * `tol` - Tolerance parameter.
/// * `verbose` - Shall print messages be returned for each iteration.
///
/// ### Returns
///
/// A tuple of the final identified mixing matrix w and the reached tolerance
/// value.
pub fn fast_ica_logcosh<T: BixverseFloat>(
    x: MatRef<T>,
    w_init: MatRef<T>,
    tol: T,
    alpha: T,
    maxit: usize,
    verbose: bool,
) -> IcaRes<T> {
    let p = x.ncols();
    let mut w = update_mix_mat(w_init);
    let mut lim = vec![T::from_f64(1000.0).unwrap(); maxit];

    let mut it = 0;

    while it < maxit && lim[it] > tol {
        let wx: Mat<T> = w.clone() * x;

        let gwx = Mat::from_fn(wx.nrows(), wx.ncols(), |i, j| {
            let x = wx.get(i, j);
            (alpha * *x).tanh()
        });

        let p_recip = T::from_usize(p).unwrap().recip();
        let v1 = Scale(p_recip) * (gwx.clone() * x.transpose());

        let gwx_2 = Mat::from_fn(gwx.nrows(), gwx.ncols(), |i, j| {
            let x = gwx.get(i, j);
            alpha * (T::one() - *x * *x)
        });

        let ones = Mat::from_fn(p, 1, |_, _| T::one());
        let row_means = Scale(p_recip) * (gwx_2.clone() * ones);

        let row_means_vec: Vec<T> = row_means
            .as_ref()
            .col_iter()
            .flat_map(|col| col.iter())
            .copied()
            .collect();

        let v2 = faer_diagonal_from_vec(row_means_vec) * w.clone();

        let w1 = update_mix_mat((v1 - v2).as_ref());

        let w1_up = w1.clone() * w.transpose();

        let tol_it = w1_up
            .diagonal()
            .column_vector()
            .iter()
            .map(|x| (x.abs() - T::one()).abs())
            .fold(T::neg_infinity(), |a, b| a.max(b));

        if it + 1 < maxit {
            lim[it + 1] = tol_it
        }

        if verbose {
            println!("Iteration: {:?}, tol: {:?}", it + 1, tol_it)
        }

        w = w1;

        it += 1;
    }

    let min_tol = array_min(&lim);

    (w, min_tol)
}

/// Fast ICA implementation based on exp algorithm.
///
/// ### Params
///
/// * `x` - Whitened matrix
/// * `w_init` - Initial, random mixing matrix
/// * `maxit` - Maximum number of iterations to run ICA for.
/// * `tol` - Tolerance parameter.
/// * `verbose` - Shall print messages be returned for each iteration.
///
/// ### Returns
///
/// A tuple of the final identified mixing matrix w and the reached tolerance
/// value.
pub fn fast_ica_exp<T: BixverseFloat>(
    x: MatRef<T>,
    w_init: MatRef<T>,
    tol: T,
    maxit: usize,
    verbose: bool,
) -> IcaRes<T> {
    let p = x.ncols();
    let mut w = update_mix_mat(w_init);
    let mut lim = vec![T::from_f64(1000.0).unwrap(); maxit];

    let mut it = 0;
    while it < maxit && lim[it] > tol {
        let wx: Mat<T> = w.clone() * x;

        let neg_half = T::from_f64(-0.5).unwrap();
        let gwx = Mat::from_fn(wx.nrows(), wx.ncols(), |i, j| {
            let x = wx.get(i, j);
            *x * (neg_half * *x * *x).exp()
        });

        let p_recip = T::from_usize(p).unwrap().recip();
        let v1 = Scale(p_recip) * (gwx.clone() * x.transpose());

        let gwx_2 = Mat::from_fn(wx.nrows(), wx.ncols(), |i, j| {
            let x = wx.get(i, j);
            (T::one() - *x * *x) * (neg_half * *x * *x).exp()
        });

        let ones = Mat::from_fn(p, 1, |_, _| T::one());
        let row_means = Scale(p_recip) * (gwx_2.clone() * ones);

        let row_means_vec: Vec<T> = row_means
            .as_ref()
            .col_iter()
            .flat_map(|col| col.iter())
            .copied()
            .collect();

        let v2 = faer_diagonal_from_vec(row_means_vec) * w.clone();

        let w1 = update_mix_mat((v1 - v2).as_ref());

        let w1_up = w1.clone() * w.transpose();

        let tol_it = w1_up
            .diagonal()
            .column_vector()
            .iter()
            .map(|x| (x.abs() - T::one()).abs())
            .fold(T::neg_infinity(), |a, b| a.max(b));

        if it + 1 < maxit {
            lim[it + 1] = tol_it
        }

        if verbose {
            println!("Iteration: {:?}, tol: {:?}", it + 1, tol_it)
        }

        w = w1;

        it += 1;
    }

    let min_tol = array_min(&lim);

    (w, min_tol)
}

////////////////////
// Stabilised ICA //
////////////////////

/// Stabilised ICA iteration implementation
///
/// Iterate through a set of random initialisations with a given pre-whitened
/// matrix, the whitening matrix k and the respective ICA parameters. It will
/// generate `no_iters` random seeds and generate the S matrix for all of them.
///
/// ### Params
///
/// * `x_pre_whiten` - he pre-processed, but not yet whitened matrix.
/// * `k` - Whitening matrix k.
/// * `no_comp` - Number of independent components to test for.
/// * `no_iters` - Number of random iterations.
/// * `ica_type` - Which of the implemented versions of ICA to test.
/// * `ica_params` - `IcaParams` structure with the parameters for the
///   individual runs.
/// * `random_seed` - Seed for reproducibility purposes.
///
/// ### Returns
///
/// Returns a tuple of the column bound S matrices and a vector of the final tolerances
/// each individual run achieved.
pub fn stabilised_ica_iters<T: BixverseFloat>(
    x_pre_whiten: MatRef<T>,
    k: MatRef<T>,
    no_comp: usize,
    no_iters: usize,
    ica_type: &str,
    ica_params: IcaParams<T>,
    random_seed: usize,
) -> (Mat<T>, Vec<bool>) {
    // generate the random w_inits
    let w_inits: Vec<Mat<T>> = (0..no_iters)
        .map(|iter| create_w_init(no_comp, (random_seed + iter) as u64))
        .collect();
    let k_ncol = k.ncols();
    let k_red = k.get(0..no_comp, 0..k_ncol);
    let x_whiten = k_red * x_pre_whiten;

    let ica_type = parse_ica_type(ica_type).unwrap();
    let iter_res: Vec<(Mat<T>, T)> = w_inits
        .par_iter()
        .map(|w_init| match ica_type {
            IcaType::Exp => fast_ica_exp(
                x_whiten.as_ref(),
                w_init.as_ref(),
                ica_params.tol,
                ica_params.maxit,
                ica_params.verbose,
            ),
            IcaType::LogCosh => fast_ica_logcosh(
                x_whiten.as_ref(),
                w_init.as_ref(),
                ica_params.tol,
                ica_params.alpha,
                ica_params.maxit,
                ica_params.verbose,
            ),
        })
        .collect();

    let mut convergence = Vec::new();
    let mut a_matrices = Vec::new();

    for (a, final_tol) in iter_res {
        a_matrices.push(a);
        convergence.push(final_tol < ica_params.tol);
    }

    let s_matrices: Vec<Mat<T>> = a_matrices
        .par_iter()
        .map(|a| {
            let w = a.clone() * k_red;
            let to_solve = w.clone() * w.transpose();
            let identity = Mat::<T>::identity(to_solve.nrows(), to_solve.ncols());
            let lu = PartialPivLu::new(to_solve.as_ref());
            let solved = lu.solve(&identity);
            w.transpose() * solved
        })
        .collect();

    let s_combined = colbind_matrices(&s_matrices);

    (s_combined, convergence)
}

/// Run stabilised ICA iterations over the CV-like data
///
/// ### Params
///
/// * `x` - The matrix for which to run the stabilised ICA over CV-like subsets.
/// * `no_comp` - Number of independent components to test for.
/// * `no_folds` - Number of folds to use for the cross-validation.
/// * `no_iters` - Number of random iterations.
/// * `ica_type` - Which of the implemented versions of ICA to test.
/// * `ica_params` - `IcaParams` structure with the parameters for the individual
///   runs.
/// * `ica_cv_data` - Optional pre-processed `IcaCvData` structure. If not provided,
///   these will be automatically generated.
/// * `random_seed` - Seed for reproducibility purposes.
///
/// ### Returns
///
/// Returns a tuple of the column bound S matrices (in this case across the different
/// folds and random initialisations of the mixing matrix) and a vector of the final tolerances
/// each individual run achieved.
#[allow(clippy::too_many_arguments)]
pub fn stabilised_ica_cv<T: BixverseFloat>(
    x: MatRef<T>,
    no_comp: usize,
    no_folds: usize,
    no_iters: usize,
    ica_type: &str,
    ica_params: IcaParams<T>,
    ica_cv_data: Option<IcaCvData<T>>,
    seed: usize,
) -> (Mat<T>, Vec<bool>) {
    // Generate cross-validation data if not provided
    let cv_data = match ica_cv_data {
        Some(data) => data, // Use the provided data
        None => IcaCvData::create_from_data(x, no_folds, seed, Some(no_comp)), // Generate new data
    };

    // Iterate through bootstrapped samples
    let cv_res: Vec<(Mat<T>, Vec<bool>)> = cv_data
        .k_matrices
        .par_iter()
        .zip(cv_data.pre_white_matrices)
        .map(|(k_i, x_i)| {
            let (s_i, converged_i) = stabilised_ica_iters(
                x_i.as_ref(),
                k_i.as_ref(),
                no_comp,
                no_iters,
                ica_type,
                ica_params.clone(),
                seed + 2,
            );

            (s_i, converged_i)
        })
        .collect();

    let mut s_final = Vec::new();
    let mut converged_final = Vec::new();

    for (s_i, converged_i) in cv_res {
        s_final.push(s_i);
        converged_final.push(converged_i);
    }

    let s_final = colbind_matrices(&s_final);
    let converged_final = flatten_vector(converged_final);

    (s_final, converged_final)
}
