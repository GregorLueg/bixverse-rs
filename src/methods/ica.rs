//! Implementation of independent component analysis, based on the fastICA from
//! Hyvrinen and Oja, Neural Comput., 1997

use faer::{
    Accum, Mat, MatRef, Par, Scale,
    linalg::solvers::{PartialPivLu, Solve},
};
use rand::prelude::*;
use rand_distr::Distribution;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::core::math::matrix_helpers::scale_matrix_col;
use crate::core::math::pca_svd::randomised_svd;
use crate::prelude::*;
use crate::utils::faer_parallelism;
use crate::utils::gemm::gemm;

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
#[derive(Clone, Debug)]
pub struct IcaParams<T: BixverseFloat> {
    /// Maximum number of iterations for ICA
    pub maxit: usize,
    /// Alpha parameter for the `logcosh` variant.
    pub alpha: T,
    /// Tolerance parameter.
    pub tol: T,
    /// Controls ICA internal verbosity
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
) -> Result<(Mat<T>, Mat<T>), BixverseErrors> {
    let n = x.nrows();

    let centered = scale_matrix_col(&x, false);

    let centered = centered.transpose();

    let n_recip = T::from_usize(n).unwrap().recip();
    let v = Scale(n_recip) * (centered * centered.transpose());

    let k = if fast_svd {
        let svd_res = randomised_svd(v.as_ref(), rank, seed, oversampling, n_power_iter)?;
        let s: Vec<T> = svd_res.s.iter().map(|x| x.sqrt().recip()).collect();
        let d = faer_diagonal_from_vec(s);
        d * svd_res.u.transpose()
    } else {
        let svd_res = v
            .thin_svd()
            .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;
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

    Ok((centered.cloned(), k))
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
fn update_mix_mat<T: BixverseFloat>(w: MatRef<T>) -> Result<Mat<T>, BixverseErrors> {
    // SVD
    let svd_res = w
        .thin_svd()
        .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;

    let s = svd_res.S();
    let u = svd_res.U();
    let s = s
        .column_vector()
        .iter()
        .map(|x| x.recip())
        .collect::<Vec<_>>();
    let d = faer_diagonal_from_vec(s);

    Ok(u * d * u.transpose() * w)
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
fn create_w_init<T: BixverseFloat>(n_comp: usize, seed: u64) -> Mat<T> {
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
pub struct IcaCvData<T: BixverseFloat> {
    /// Vector of pre-processed matrices, ready for whitening
    pub pre_white_matrices: Vec<Mat<T>>,
    /// Vector of pre-whitening matrices
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
    ) -> Result<IcaCvData<T>, BixverseErrors> {
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
                let mut is_test = vec![false; no_samples];
                for &idx in test_indices {
                    is_test[idx] = true;
                }
                let train_indices: Vec<usize> = indices
                    .iter()
                    .filter(|&&idx| !is_test[idx])
                    .cloned()
                    .collect();
                let x_i = Mat::<T>::from_fn(train_indices.len(), no_features, |new_row, j| {
                    x[(train_indices[new_row], j)]
                });
                prepare_whitening(x_i.as_ref(), true, seed + 1, svd_rank, None, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut pre_white_matrices = Vec::with_capacity(num_folds);
        let mut k_matrices = Vec::with_capacity(num_folds);

        for (x_i, k_i) in k_x_matrices {
            pre_white_matrices.push(x_i);
            k_matrices.push(k_i);
        }

        Ok(Self {
            pre_white_matrices,
            k_matrices,
        })
    }
}

////////////////////
// Main functions //
////////////////////

/// Shared fastICA fixed-point loop
///
/// `contrast` maps a projection `u` to `(g(u), g'(u))`. The products run with
/// the given parallelism, so callers inside an outer parallel loop pass
/// `Par::Seq`.
///
/// ### Params
///
/// * `par` - Parallelism of the matrix products
/// * `x` - Whitened matrix
/// * `w_init` - Initial, random mixing matrix
/// * `tol` - Tolerance parameter.
/// * `maxit` - Maximum number of iterations to run ICA for.
/// * `verbose` - Shall print messages be returned for each iteration.
/// * `contrast` - The contrast function and its derivative
///
/// ### Returns
///
/// A tuple of the final identified mixing matrix w and the reached tolerance
/// value.
fn fast_ica_core<T: BixverseFloat>(
    par: Par,
    x: MatRef<T>,
    w_init: MatRef<T>,
    tol: T,
    maxit: usize,
    verbose: bool,
    contrast: impl Fn(T) -> (T, T),
) -> Result<IcaRes<T>, BixverseErrors> {
    let p = x.ncols();
    let mut w = update_mix_mat(w_init)?;
    let n_comp = w.nrows();
    let mut lim = vec![T::from_f64(1000.0).unwrap(); maxit];

    let p_recip = T::from_usize(p).unwrap().recip();
    let ones = Mat::<T>::from_fn(p, 1, |_, _| T::one());

    // scratch reused across iterations
    let mut wx = Mat::<T>::zeros(n_comp, p);
    let mut gwx = Mat::<T>::zeros(n_comp, p);
    let mut gwx_2 = Mat::<T>::zeros(n_comp, p);
    let mut v1 = Mat::<T>::zeros(n_comp, n_comp);
    let mut row_sums = Mat::<T>::zeros(n_comp, 1);

    let mut it = 0;

    while it < maxit && lim[it] > tol {
        gemm(wx.as_mut(), Accum::Replace, w.as_ref(), x, T::one(), par);

        for j in 0..p {
            let wx_col = wx.col_as_slice(j);
            let gwx_col = gwx.col_as_slice_mut(j);
            let gwx_2_col = gwx_2.col_as_slice_mut(j);
            for ((&u, g), g2) in wx_col.iter().zip(gwx_col).zip(gwx_2_col) {
                (*g, *g2) = contrast(u);
            }
        }

        gemm(
            v1.as_mut(),
            Accum::Replace,
            gwx.as_ref(),
            x.transpose(),
            p_recip,
            par,
        );
        gemm(
            row_sums.as_mut(),
            Accum::Replace,
            gwx_2.as_ref(),
            ones.as_ref(),
            T::one(),
            par,
        );

        let v2 = Mat::<T>::from_fn(n_comp, n_comp, |i, j| {
            row_sums[(i, 0)] * p_recip * w[(i, j)]
        });

        let w1 = update_mix_mat((v1.as_ref() - v2.as_ref()).as_ref())?;

        // only the diagonal of w1 * w^T is read
        let tol_it = (0..n_comp)
            .map(|i| {
                let d = (0..n_comp).fold(T::zero(), |acc, c| acc + w1[(i, c)] * w[(i, c)]);
                (d.abs() - T::one()).abs()
            })
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

    Ok((w, min_tol))
}

/// Contrast function and derivative for the logcosh variant
///
/// ### Params
///
/// * `alpha` - Alpha parameter for this variant
///
/// ### Returns
///
/// A closure mapping `u` to `(tanh(alpha u), alpha (1 - tanh(alpha u)^2))`.
fn logcosh_contrast<T: BixverseFloat>(alpha: T) -> impl Fn(T) -> (T, T) {
    move |u| {
        let g = (alpha * u).tanh();
        (g, alpha * (T::one() - g * g))
    }
}

/// Contrast function and derivative for the exp variant
///
/// ### Returns
///
/// A closure mapping `u` to `(u e, (1 - u^2) e)` with `e = exp(-u^2 / 2)`.
fn exp_contrast<T: BixverseFloat>() -> impl Fn(T) -> (T, T) {
    let neg_half = T::from_f64(-0.5).unwrap();
    move |u| {
        let e = (neg_half * u * u).exp();
        (u * e, (T::one() - u * u) * e)
    }
}

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
) -> Result<IcaRes<T>, BixverseErrors> {
    fast_ica_core(
        faer_parallelism(),
        x,
        w_init,
        tol,
        maxit,
        verbose,
        logcosh_contrast(alpha),
    )
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
) -> Result<IcaRes<T>, BixverseErrors> {
    fast_ica_core(
        faer_parallelism(),
        x,
        w_init,
        tol,
        maxit,
        verbose,
        exp_contrast(),
    )
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
) -> Result<(Mat<T>, Vec<bool>), BixverseErrors> {
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
            IcaType::Exp => fast_ica_core(
                Par::Seq,
                x_whiten.as_ref(),
                w_init.as_ref(),
                ica_params.tol,
                ica_params.maxit,
                ica_params.verbose,
                exp_contrast(),
            ),
            IcaType::LogCosh => fast_ica_core(
                Par::Seq,
                x_whiten.as_ref(),
                w_init.as_ref(),
                ica_params.tol,
                ica_params.maxit,
                ica_params.verbose,
                logcosh_contrast(ica_params.alpha),
            ),
        })
        .collect::<Result<Vec<(Mat<T>, T)>, BixverseErrors>>()?;

    let mut convergence = Vec::new();
    let mut a_matrices = Vec::new();

    for (a, final_tol) in iter_res {
        a_matrices.push(a);
        convergence.push(final_tol < ica_params.tol);
    }

    let s_matrices: Vec<Mat<T>> = a_matrices
        .par_iter()
        .map(|a| {
            let mut w = Mat::<T>::zeros(a.nrows(), k_red.ncols());
            gemm(
                w.as_mut(),
                Accum::Replace,
                a.as_ref(),
                k_red,
                T::one(),
                Par::Seq,
            );
            let mut to_solve = Mat::<T>::zeros(w.nrows(), w.nrows());
            gemm(
                to_solve.as_mut(),
                Accum::Replace,
                w.as_ref(),
                w.transpose(),
                T::one(),
                Par::Seq,
            );
            let identity = Mat::<T>::identity(to_solve.nrows(), to_solve.ncols());
            let lu = PartialPivLu::new(to_solve.as_ref());
            let solved = lu.solve(&identity);
            let mut out = Mat::<T>::zeros(w.ncols(), solved.ncols());
            gemm(
                out.as_mut(),
                Accum::Replace,
                w.transpose(),
                solved.as_ref(),
                T::one(),
                Par::Seq,
            );
            out
        })
        .collect();

    let s_combined = colbind_matrices(&s_matrices);

    Ok((s_combined, convergence))
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
) -> Result<(Mat<T>, Vec<bool>), BixverseErrors> {
    // generate cross-validation data if not provided
    let cv_data = match ica_cv_data {
        Some(data) => Ok(data), // Use the provided data
        None => IcaCvData::create_from_data(x, no_folds, seed, Some(no_comp)),
    }?;

    // Iterate through bootstrapped samples
    let cv_res: Vec<(Mat<T>, Vec<bool>)> = cv_data
        .k_matrices
        .par_iter()
        .zip(cv_data.pre_white_matrices)
        .map(|(k_i, x_i)| {
            stabilised_ica_iters(
                x_i.as_ref(),
                k_i.as_ref(),
                no_comp,
                no_iters,
                ica_type,
                ica_params.clone(),
                seed + 2,
            )
        })
        .collect::<Result<Vec<(Mat<T>, Vec<bool>)>, BixverseErrors>>()?;

    let mut s_final = Vec::new();
    let mut converged_final = Vec::new();

    for (s_i, converged_i) in cv_res {
        s_final.push(s_i);
        converged_final.push(converged_i);
    }

    let s_final = colbind_matrices(&s_final);
    let converged_final = flatten_vector(converged_final);

    Ok((s_final, converged_final))
}
