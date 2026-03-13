//! Implementation of the dual graph regularised sparse dictionary learning
//! based on the work of Pan, et al., Cell Syst, 2022

use faer::{
    ColRef, Mat, MatRef, Scale,
    linalg::solvers::{PartialPivLu, Solve},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;

use crate::core::base::cors_similarity::column_pairwise_cos;
use crate::core::math::linear_algebra::sylvester_solver;
use crate::graph::graph_structures::{adjacency_to_laplacian, get_knn_graph_adj};
use crate::prelude::*;

////////////
// Params //
////////////

/// Structure to store the dual graph regularised dictionary learning params
#[derive(Debug, Clone, Default)]
pub struct DgrdlParams<T> {
    /// Sparsity constraint (max non-zero coefficients per signal)
    pub sparsity: usize,
    /// Dictionary size
    pub dict_size: usize,
    /// Sample context regularisation weight
    pub alpha: T,
    /// Feature effect regularisation weight
    pub beta: T,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Number of neighbours in the KNN graph
    pub k_neighbours: usize,
    /// ADMM iterations for sparse coding
    pub admm_iter: usize,
    /// ADMM step size
    pub rho: T,
}

impl<T> DgrdlParams<T>
where
    T: BixverseFloat,
{
    /// Create default DGRDL parameters
    ///
    /// ### Returns
    ///
    /// `DgrdlParams` with default parameter configuration
    #[allow(dead_code)]
    fn default() -> Self {
        Self {
            sparsity: 5,
            dict_size: 50,
            alpha: T::from_f32(1.0).unwrap(),
            beta: T::from_f32(1.0).unwrap(),
            max_iter: 20,
            k_neighbours: 5,
            admm_iter: 5,
            rho: T::from_f32(1.0).unwrap(),
        }
    }
}

/////////////
// Results //
/////////////

/// DGRDL algorithm results
#[derive(Debug)]
pub struct DgrdlResults<T> {
    /// Learned dictionary of size n x k
    pub dictionary: Mat<T>,
    /// Sparse coefficients of size k x m
    pub coefficients: Mat<T>,
    /// Sample context Laplacian n × n
    pub sample_laplacian: Mat<T>,
    /// Feature context laplacian of size m x m
    pub feature_laplacian: Mat<T>,
}

/////////////////
// DGRDL Cache //
/////////////////

/// DGRDL cache for hyperparameter tuning
#[derive(Debug)]
pub struct DgrdlCache<T> {
    /// Hash of the underlying data
    pub data_hash: u64,
    /// HashMap of the feature and sample Laplacian matrix to avoid
    /// recomputations
    pub laplacian_cache: FxHashMap<usize, (Mat<T>, Mat<T>)>,
    /// HashMap of the dictionary cashe with (dict_size, seed) as keys to avoid
    /// recomputation
    pub dictionary_cache: FxHashMap<(usize, usize), Mat<T>>,
    /// The distance matrix of the data as a flat vector
    pub distance_matrix: Option<Vec<T>>,
}

impl<T: BixverseFloat> DgrdlCache<T> {
    /// Generate a new instance of the Cache
    ///
    /// ### Returns
    ///
    /// Returns self with empty fields
    pub fn new() -> Self {
        Self {
            data_hash: 0,
            laplacian_cache: FxHashMap::default(),
            dictionary_cache: FxHashMap::default(),
            distance_matrix: None,
        }
    }

    /// Clear cache if data has changed
    ///
    /// Will clear the cash if the Hash changed
    ///
    /// ### Params
    ///
    /// * `data` The data for which to calculate the Hash
    fn validate_data(&mut self, data: &MatRef<T>) {
        let new_hash = self.compute_data_hash(data);
        if new_hash != self.data_hash {
            self.clear_cache();
            self.data_hash = new_hash;
        }
    }

    /// Compute the hash of the data. This will test just some elements
    ///
    /// ### Params
    ///
    /// * `data` - The data for the DGRDL algorithm
    ///
    /// ### Returns
    ///
    /// Hash usize
    fn compute_data_hash(&self, data: &MatRef<T>) -> u64 {
        let mut hash = data.nrows() as u64;
        hash = hash.wrapping_mul(31).wrapping_add(data.ncols() as u64);
        if data.nrows() > 0 && data.ncols() > 0 {
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(data[(0, 0)].to_f64().unwrap().to_bits());
            let last_row = data.nrows() - 1;
            let last_col = data.ncols() - 1;
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(data[(last_row, last_col)].to_f64().unwrap().to_bits());
        }
        hash
    }

    /// Clears the internal Cache
    fn clear_cache(&mut self) {
        self.laplacian_cache.clear();
        self.dictionary_cache.clear();
        self.distance_matrix = None;
    }
}

/// Default implementation for DgrdlCache
impl<T> Default for DgrdlCache<T>
where
    T: BixverseFloat,
{
    fn default() -> Self {
        Self {
            data_hash: 0,
            laplacian_cache: FxHashMap::default(),
            dictionary_cache: FxHashMap::default(),
            distance_matrix: None,
        }
    }
}

///////////////////
// DGRDL objects //
///////////////////

/// Calculate individual components of DGRDL objective function
#[derive(Debug, Clone)]
pub struct DgrdlObjectives<T> {
    /// Calculates the approximation error in form of squared Frobenius norm
    pub approximation_error: T,
    /// Measures the similiarity of coefficients for the feature Laplacian
    pub feature_laplacian_objective: T,
    /// Measures the smoothness of the dictionary in respect to sample Laplacian
    pub sample_laplacian_objective: T,
    /// Random seed for reproducibility
    pub seed: usize,
    /// Number of neighbours
    pub k_neighbours: usize,
    /// The size of the dictionary
    pub dict_size: usize,
}

impl<T: BixverseFloat> DgrdlObjectives<T> {
    /// Calculate the different objectives and reconstruction error for a given DGRDL run
    ///
    /// ### Params
    ///
    /// * `dat` - Input matrix
    /// * `res` - The `DgrdlResults` of that run
    /// * `alpha` - Sample context regularisation weight
    /// * `beta` - Feature context regularisation weight
    ///
    /// ### Returns
    ///
    /// The `DgrdlObjectives` object
    pub fn calculate(
        dat: &MatRef<T>,
        res: &DgrdlResults<T>,
        alpha: T,
        beta: T,
        seed: usize,
        k_neighbours: usize,
        dict_size: usize,
    ) -> Self {
        let approximation_error =
            Self::calculate_approximation_error(dat, &res.dictionary, &res.coefficients);

        let feature_laplacian_objective =
            beta * Self::calculate_trace_xlx(&res.coefficients, &res.feature_laplacian);

        let sample_laplacian_objective =
            alpha * Self::calculate_trace_ddl(&res.dictionary, &res.sample_laplacian);

        Self {
            approximation_error,
            feature_laplacian_objective,
            sample_laplacian_objective,
            seed,
            k_neighbours,
            dict_size,
        }
    }

    /// Calculate the reconstruction error
    ///
    /// ### Params
    ///
    /// * `x` - The input matrix
    /// * `dictionary` - The yielded dictionary matrix from the algorithm
    /// * `coefficients` - The yielded coefficient matrix from the algorithm
    ///
    /// ### Returns
    ///
    /// The Frobenius norm squared. The lower, the better.
    fn calculate_approximation_error(
        x: &MatRef<T>,
        dictionary: &Mat<T>,
        coefficients: &Mat<T>,
    ) -> T {
        let reconstruction = dictionary * coefficients;
        let residual = x.as_ref() - &reconstruction;
        residual.norm_l2().powi(2)
    }

    /// Calculate gene Laplacian term: tr(X·L_f·X^T)
    ///
    /// ### Params
    ///
    /// * `coefficients` - The yielded coefficient matrix from the algorithm
    /// * `feature_laplacian` - The Laplacian matrix of the features
    ///
    /// ### Returns
    ///
    /// The trace of that part of the equation. The lower, the better.
    fn calculate_trace_xlx(coefficients: &Mat<T>, feature_laplacian: &Mat<T>) -> T {
        let xl = coefficients * feature_laplacian;

        let mut trace = T::zero();
        for i in 0..coefficients.nrows() {
            for j in 0..coefficients.ncols() {
                trace += xl[(i, j)] * coefficients[(i, j)];
            }
        }
        trace
    }

    /// Calculate cell Laplacian term: tr(D^T·D·L_s)
    ///
    /// ### Params
    ///
    /// * `dictionary` - The yielded dictionary matrix from the algorithm
    /// * `sample_laplacian` - The Laplacian matrix of the samples
    ///
    /// ### Returns
    ///
    /// The trace of that part of equation. The lower, the better.
    fn calculate_trace_ddl(dictionary: &Mat<T>, sample_laplacian: &Mat<T>) -> T {
        let dt_l = dictionary.transpose() * sample_laplacian;

        let mut trace = T::zero();
        for i in 0..dictionary.ncols() {
            for j in 0..dictionary.nrows() {
                trace += dt_l[(i, j)] * dictionary[(j, i)];
            }
        }
        trace
    }
}

/////////////
// Helpers //
/////////////

/////////////
// Helpers //
/////////////

/// Get the upper triangle indices as a pair for rapid distance calculations
///
/// Stores the indices of a 3 x 3 matrix for example in a pattern of
/// `(0,1)`, `(0,2)`, `(0,3)`, `(1,2)`, `(1,3)`, `(2,3)` (assuming a linear
/// matrix).
///
/// ### Params
///
/// * `pair_idx` - Linear index in compressed storage, i.e., `0 to n (n - 1) / 2 - 1)`
/// * `n` - Shape of the original column
///
/// ### Returns
///
/// Tuple of `(i, j)`
fn triangle_to_indices(linear_idx: usize, n: usize) -> (usize, usize) {
    let mut idx = linear_idx;
    let mut i = 0;

    while idx >= n - i - 1 {
        idx -= n - i - 1;
        i += 1;
    }

    (i, i + 1 + idx)
}

/// Retrieve distance from compressed upper-triangle storage
///
/// ### Params
///
/// * `distances` - Pre-computed distance array in compressed format
/// * `i` - First matrix index
/// * `j` - Second matrix index
/// * `n` - Original matrix size
///
/// ### Returns
///
/// Distance value between the two values
fn get_distance<T: BixverseFloat>(distances: &[T], i: usize, j: usize, n: usize) -> T {
    if i == j {
        return T::zero();
    }

    let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
    let linear_idx = (min_idx * (2 * n - min_idx - 1)) / 2 + (max_idx - min_idx - 1);
    distances[linear_idx]
}

/// Create the Laplacian matrix for the features or samples
///
/// ### Fields
///
/// * `data` - The original data matrix with rows = samples and columns =
///   features
/// * `k` - Number of neighbours for the KNN graph
/// * `features` - If `true` generated the Laplacian for the features, otherwise
///   for the samples
///
/// ### Returns
///
/// The Laplacian matrix L = D - A where D is the degree matrix and A is the
/// adjacency matrix
fn get_dgrdl_laplacian<T: BixverseFloat>(data: &MatRef<T>, k: usize, features: bool) -> Mat<T> {
    let cosine_sim = if features {
        column_pairwise_cos(data)
    } else {
        column_pairwise_cos(&data.transpose())
    };

    let knn_adjacency = get_knn_graph_adj(&cosine_sim.as_ref(), k);

    adjacency_to_laplacian(&knn_adjacency.as_ref(), false)
}

/// Function to do the ADMM on the GRSC
///
/// Solves the optimization problem:
///
/// min_X ||Y - DX||²_F + β·tr(X·L_f·X^T) + λ||X||_0
///
/// subject to sparsity constraint using alternating direction method of
/// multipliers (ADMM).
///
/// ### Params
///
/// * `dictionary` - Current dictionary matrix D
/// * `data` - Input data matrix Y
/// * `feature_laplacian` - Feature graph Laplacian L_f
/// * `sparsity` - Maximum number of non-zero coefficients per column
/// * `beta` - Graph regularization weight
/// * `max_iter` - Maximum ADMM iterations
/// * `rho` - ADMM penalty parameter
///
/// ### Returns
///
/// Sparse coefficient matrix X satisfying the sparsity constraint
pub fn grsc_admm<T: BixverseFloat>(
    dictionary: &Mat<T>,
    data: &MatRef<T>,
    feature_laplacian: &Mat<T>,
    sparsity: usize,
    beta: T,
    max_iter: usize,
    rho: T,
) -> Mat<T> {
    let k = dictionary.ncols();
    let m = data.ncols();

    let gram = dictionary.transpose() * dictionary;
    let dt_y = dictionary.transpose() * data;

    #[allow(unused_assignments)]
    let mut x: Mat<T> = Mat::zeros(k, m);
    let mut z: Mat<T> = Mat::zeros(k, m);
    let mut u: Mat<T> = Mat::zeros(k, m);

    for _ in 0..max_iter {
        x = admm_solve_x(&gram, &dt_y, &z, &u, feature_laplacian, beta, rho);

        z = sparse_projection(&(&x + &u), sparsity);

        u = &u + &x - &z;
    }

    z
}

/// Solve the X-update step in ADMM
///
/// Solves the Sylvester equation: AX + XB = C
/// where A = G + ρI, B = βL_f, C = D^T Y + ρ(Z - U)
///
/// ### Params
///
/// * `gram` - Gram matrix G = D^T D
/// * `dt_y` - Data projection D^T Y
/// * `z` - Current Z variable from ADMM
/// * `u` - Current dual variable from ADMM
/// * `feature_laplacian` - Feature Laplacian L_f
/// * `beta` - Graph regularization weight
/// * `rho` - ADMM penalty parameter
///
/// ### Returns
///
/// Updated X matrix solving the regularized least squares problem
fn admm_solve_x<T: BixverseFloat>(
    gram: &Mat<T>,
    dt_y: &Mat<T>,
    z: &Mat<T>,
    u: &Mat<T>,
    feature_laplacian: &Mat<T>,
    beta: T,
    rho: T,
) -> Mat<T> {
    let k = gram.nrows();

    let identity_mat: Mat<T> = Mat::identity(k, k);
    let a: Mat<T> = gram + &(Scale(rho) * identity_mat);
    let b = Scale(beta) * feature_laplacian;
    let c = dt_y + &(Scale(rho) * (z - u));

    sylvester_solver(&a.as_ref(), &b.as_ref(), &c.as_ref())
}

/// Apply sparsity constraint via hard thresholding
///
/// For each column, keeps only the largest (in absolute value) 'sparsity'
/// number of coefficients and sets the rest to zero.
///
/// ### Params
///
/// * `x` - Input coefficient matrix
/// * `sparsity` - Number of non-zero coefficients to keep per column
///
/// ### Returns
///
/// Sparse matrix with at most 'sparsity' non-zero entries per column
fn sparse_projection<T: BixverseFloat>(x: &Mat<T>, sparsity: usize) -> Mat<T> {
    let (k, m) = x.shape();
    let mut result = Mat::zeros(k, m);

    let columns: Vec<Vec<T>> = (0..m)
        .into_par_iter()
        .map(|j| {
            let x_col = x.col(j);
            let mut indexed_vals: Vec<(usize, T)> =
                x_col.iter().enumerate().map(|(i, &val)| (i, val)).collect();

            let take_count = sparsity.min(k);
            if take_count < indexed_vals.len() {
                indexed_vals
                    .select_nth_unstable_by(take_count, |a, b| b.1.abs().total_cmp(&a.1.abs()));
            }

            let mut col_result = vec![T::zero(); k];
            for &(idx, val) in indexed_vals.iter().take(sparsity) {
                col_result[idx] = val;
            }
            col_result
        })
        .collect();

    for (j, col_data) in columns.iter().enumerate() {
        for (i, &val) in col_data.iter().enumerate() {
            result[(i, j)] = val;
        }
    }

    result
}

/// Compute Euclidean distance between two columns
///
/// ### Params
///
/// * `col_i` - First column vector
/// * `col_j` - Second column vector
///
/// ### Returns
///
/// Euclidean distance between the two columns
fn column_distance<T: BixverseFloat>(col_i: ColRef<T>, col_j: ColRef<T>) -> T {
    col_i
        .iter()
        .zip(col_j.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt()
}

/// Update the entire dictionary
///
/// Updates all dictionary atoms sequentially using graph regularisation.
/// Each atom is updated to minimise the reconstruction error while
/// maintaining smoothness according to the sample graph structure.
///
/// ### Params
///
/// * `data` - Input data matrix Y
/// * `coefficients` - Current coefficient matrix X
/// * `sample_laplacian` - Sample graph Laplacian L_s
/// * `alpha` - Sample regularization weight
///
/// ### Returns
///
/// Updated dictionary matrix with normalised columns
fn update_dictionary<T: BixverseFloat>(
    data: &MatRef<T>,
    coefficients: &Mat<T>,
    sample_laplacian: &Mat<T>,
    alpha: T,
) -> Mat<T> {
    let (n_contexts, _) = data.shape();
    let k = coefficients.nrows();
    let identity: Mat<T> = Mat::identity(n_contexts, n_contexts);
    let reg_term = (Scale(alpha) * sample_laplacian).to_owned();

    let atom_columns: Vec<Vec<T>> = (0..k)
        .into_par_iter()
        .map(|atom_idx| {
            let x_j = coefficients.row(atom_idx);
            let active_signals: Vec<(usize, T)> = x_j
                .iter()
                .enumerate()
                .filter_map(|(signal_idx, &coeff)| {
                    if coeff.abs() > T::from_f64(1e-12).unwrap() {
                        Some((signal_idx, coeff))
                    } else {
                        None
                    }
                })
                .collect();
            if active_signals.is_empty() {
                return vec![T::zero(); n_contexts];
            }
            let mut rhs = Mat::zeros(n_contexts, 1);
            for &(signal_idx, coeff) in &active_signals {
                let signal = data.col(signal_idx);
                for i in 0..n_contexts {
                    rhs[(i, 0)] += signal[i] * coeff;
                }
            }
            let x_j_norm_sq = active_signals
                .iter()
                .map(|&(_, coeff)| coeff * coeff)
                .fold(T::zero(), |acc, x| acc + x);
            if x_j_norm_sq > T::from_f64(1e-12).unwrap() {
                let scaled_identity = (Scale(x_j_norm_sq) * identity.as_ref()).to_owned();
                let system_matrix = &scaled_identity + &reg_term;
                let lu = PartialPivLu::new(system_matrix.as_ref());
                let solution = lu.solve(&rhs);
                let norm = solution.norm_l2();
                if norm > T::from_f64(1e-12).unwrap() {
                    (0..n_contexts).map(|i| solution[(i, 0)] / norm).collect()
                } else {
                    vec![T::zero(); n_contexts]
                }
            } else {
                vec![T::zero(); n_contexts]
            }
        })
        .collect();
    let mut dictionary = Mat::zeros(n_contexts, k);
    for (atom_idx, atom_col) in atom_columns.iter().enumerate() {
        for (i, &val) in atom_col.iter().enumerate() {
            dictionary[(i, atom_idx)] = val;
        }
    }
    dictionary
}

//////////////////////////
// DGRDL implementation //
//////////////////////////

/// Main DGRDL implementation
///
/// Implements the Dual Graph Regularized Dictionary Learning algorithm which
/// learns a sparse dictionary representation while preserving local geometric
/// structure in both sample and feature spaces through graph regularization.
///
/// ### Fields
///
/// * `params` - Configuration parameters for the algorithm
#[derive(Debug)]
pub struct Dgrdl<T> {
    params: DgrdlParams<T>,
    cache: DgrdlCache<T>,
}

impl<T: BixverseFloat> Dgrdl<T> {
    /// Create a new DGRDL instance
    ///
    /// ### Params
    ///
    /// * `params` - Configuration parameters for the algorithm
    ///
    /// ### Returns
    ///
    /// New DGRDL instance ready for training
    pub fn new(params: DgrdlParams<T>) -> Self {
        Self {
            params,
            cache: DgrdlCache::new(),
        }
    }

    /// Run DGRDL algorithm on input data (with initial hyper parameters)
    ///
    /// Implements the main optimization loop alternating between sparse coding
    /// and dictionary update steps. The algorithm minimizes:
    /// ||Y - DX||²_F + α·tr(D^T·D·L_s) + β·tr(X·L_f·X^T) + sparsity constraint
    /// where Y is data, D is dictionary, X is coefficients, L_s and L_f are
    /// sample and feature Laplacians respectively.
    ///
    /// ### Params
    ///
    /// * `data` - Input data matrix of size n_samples × n_features
    /// * `seed` - Seed for dictionary initialisation.
    /// * `verbose` - Whether to print iteration progress
    ///
    /// ### Returns
    ///
    /// `DgrdlResults` containing the learned dictionary, coefficients, and Laplacians
    pub fn fit(&mut self, data: &MatRef<T>, seed: usize, verbose: bool) -> DgrdlResults<T> {
        self.fit_with_params(data, self.params.clone(), seed, verbose)
    }

    /// Run DGRDL algorithm on input data (with provided hyperparameters)
    ///
    /// Implements the main optimization loop alternating between sparse coding
    /// and dictionary update steps. The algorithm minimizes:
    /// ||Y - DX||²_F + α·tr(D^T·D·L_s) + β·tr(X·L_f·X^T) + sparsity constraint
    /// where Y is data, D is dictionary, X is coefficients, L_s and L_f are
    /// sample and feature Laplacians respectively.
    ///
    /// ### Params
    ///
    /// * `data` - Input data matrix of size n_samples × n_features
    /// * `params` - The `DgrdlParams` containing the parameters you wish to use.
    /// * `seed` - Seed for dictionary initialisation.
    /// * `verbose` - Whether to print iteration progress
    ///
    /// ### Returns
    ///
    /// `DgrdlResults` containing the learned dictionary, coefficients, and
    /// Laplacians
    pub fn fit_with_params(
        &mut self,
        data: &MatRef<T>,
        params: DgrdlParams<T>,
        seed: usize,
        verbose: bool,
    ) -> DgrdlResults<T> {
        let (n_samples, n_features) = data.shape();

        if verbose {
            println!(
                "Starting DGRDL run with {:?} samples and {:?} features.",
                n_samples, n_features
            )
        }

        let start_total = Instant::now();

        let start_dictionary_gen = Instant::now();
        let mut dictionary = self.get_dictionary(data, params.dict_size, seed);
        let end_dictionary_gen = start_dictionary_gen.elapsed();

        if verbose {
            println!(
                "DGRDL dictionary obtained in {:.2?} (cached: {}).",
                end_dictionary_gen,
                self.cache
                    .dictionary_cache
                    .contains_key(&(self.params.dict_size, seed))
            );
        }

        let start_laplacian_gen = Instant::now();
        let (feature_laplacian, sample_laplacian) = self.get_laplacians(data, params.k_neighbours);
        let end_laplacian_gen = start_laplacian_gen.elapsed();

        if verbose {
            println!(
                "DGRDL graph laplacians obtained in {:.2?} (cached: {}).",
                end_laplacian_gen,
                self.cache
                    .laplacian_cache
                    .contains_key(&params.k_neighbours)
            );
        }

        let mut coefficients: Mat<T> = Mat::zeros(params.dict_size, n_features);

        for iter in 0..self.params.max_iter {
            let start_iter = Instant::now();

            coefficients = grsc_admm(
                &dictionary,
                data,
                &feature_laplacian,
                params.sparsity,
                params.beta,
                params.admm_iter,
                params.rho,
            );

            dictionary = update_dictionary(data, &coefficients, &sample_laplacian, params.alpha);

            let end_iter = start_iter.elapsed();

            if verbose {
                println!(
                    " DGRDL iteration {}/{} in {:.2?}.",
                    iter + 1,
                    params.max_iter,
                    end_iter
                );
            }
        }

        let end_time = start_total.elapsed();

        if verbose {
            println!(
                "Total time elapsed for fitting the DGRDL run: {:.2?}.",
                end_time
            )
        }

        DgrdlResults {
            dictionary,
            coefficients,
            sample_laplacian,
            feature_laplacian,
        }
    }

    /// Run a hyperparameter search over DGRDL
    ///
    /// Helper function to sweep over a set of hyperparameters and return
    /// different objectives from the algorithm for subsequent visualisation.
    ///
    /// ### Params
    ///
    /// * `data` - Input data matrix of size n_samples × n_features
    /// * `dict_sizes` - Slice of different dictionary sizes you wish to test
    ///   for.
    /// * `k_neighbours_iters` - Slice of different k neighbours to iterate
    ///   over.
    /// * `seeds` - Slice of random initialisations to test.
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// `DgrdlResults` containing the learned dictionary, coefficients, and
    /// Laplacians
    pub fn grid_search(
        &mut self,
        data: &MatRef<T>,
        dict_sizes: &[usize],
        k_neighbours_iters: &[usize],
        seeds: &[usize],
        verbose: bool,
    ) -> Vec<DgrdlObjectives<T>> {
        let (n_samples, n_features) = data.shape();

        let n_dict = dict_sizes.len();
        let n_k_neighbours = k_neighbours_iters.len();
        let n_seeds = seeds.len();

        let total_permutations = n_dict * n_k_neighbours * n_seeds;

        if verbose {
            println!(
                "Starting DGRDL hyperparameter grid search with {:?} samples and {:?} features and {:?} permutations.",
                n_samples, n_features, total_permutations
            )
        }

        let start_total = Instant::now();
        let mut iteration: usize = 1;

        let mut grid_search_res: Vec<DgrdlObjectives<T>> = Vec::with_capacity(total_permutations);

        for seed in seeds {
            for dict_size in dict_sizes {
                for &k_neighbours in k_neighbours_iters {
                    let iter_total = Instant::now();

                    let mut params = self.params.clone();
                    params.dict_size = *dict_size;
                    params.k_neighbours = k_neighbours;

                    if verbose {
                        println!(
                            " Iter ({}|{}) - seed = {} | dict_size = {} | k_neighbours = {}",
                            iteration, total_permutations, seed, dict_size, k_neighbours
                        );
                    }

                    let res = self.fit_with_params(data, params.clone(), *seed, false);

                    let metrics = DgrdlObjectives::calculate(
                        data,
                        &res,
                        params.alpha,
                        params.beta,
                        *seed,
                        k_neighbours,
                        *dict_size,
                    );

                    grid_search_res.push(metrics);

                    let iter_time = iter_total.elapsed();

                    if verbose {
                        println!(
                            " Iter ({}|{}) - finalised in {:.2?}",
                            iteration, total_permutations, iter_time
                        );
                    }

                    iteration += 1;
                }
            }
        }

        let total_time = start_total.elapsed();

        if verbose {
            println!("Finished hyperparameter grid search in {:.2?}", total_time);
        }

        grid_search_res
    }

    /// Get or compute dictionary for given dict_size and seed
    ///
    /// ### Params
    ///
    /// * `data` - Input data into the algorithm
    /// * `dict_size` - The size of the dictionary
    /// * `seed` - Random seed for the initialisation
    ///
    /// ### Returns
    ///
    /// The initial dictionary matrix
    fn get_dictionary(&mut self, data: &MatRef<T>, dict_size: usize, seed: usize) -> Mat<T> {
        self.cache.validate_data(data);

        let key = (dict_size, seed);
        if let Some(dictionary) = self.cache.dictionary_cache.get(&key) {
            return dictionary.clone();
        }

        if self.cache.distance_matrix.is_none() {
            self.cache.distance_matrix = Some(self.precompute_distances(data));
        }

        let dictionary = self.initialise_dictionary(data, dict_size, seed);

        self.cache.dictionary_cache.insert(key, dictionary.clone());

        dictionary
    }

    /// Initialise dictionary using k-medoids clustering
    ///
    /// Selects initial dictionary atoms by finding representative samples
    /// (medoids) that minimise the total distance to all other samples.
    /// This provides a good initialisation that captures data diversity. Will
    /// leverage the pre-computed distances.
    ///
    /// ### Params
    ///
    /// * `data` - Input data matrix
    /// * `dict_size` - The size of the dictionary.
    /// * `seed` - Seed for the random initilisation of the first medoid.
    ///
    /// ### Returns
    ///
    /// Initialised dictionary matrix with normalized columns
    fn initialise_dictionary(&self, data: &MatRef<T>, dict_size: usize, seed: usize) -> Mat<T> {
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let k = dict_size;
        let (n_samples, n_features) = data.shape();

        let mut selected = Vec::with_capacity(k);

        selected.push(rng.random_range(0..n_features));

        let distances = self.cache.distance_matrix.as_ref().unwrap();

        for _ in 1..k {
            let min_distances: Vec<T> = (0..n_features)
                .into_par_iter()
                .map(|i| {
                    if selected.contains(&i) {
                        T::zero()
                    } else {
                        selected
                            .iter()
                            .map(|&j| get_distance(distances, i, j, n_features))
                            .fold(T::infinity(), |a, b| a.min(b))
                    }
                })
                .collect();

            let total_weight = min_distances
                .iter()
                .copied()
                .map(|d| d * d)
                .fold(T::zero(), |acc, x| acc + x);

            if total_weight > T::zero() {
                let mut target = T::from_f64(rng.random::<f64>()).unwrap() * total_weight;
                let mut chosen = 0;

                for (i, &dist) in min_distances.iter().enumerate() {
                    target -= dist * dist;
                    if target <= T::zero() || i == n_features - 1 {
                        chosen = i;
                        break;
                    }
                }

                if !selected.contains(&chosen) {
                    selected.push(chosen);
                }
            } else {
                let unselected: Vec<usize> =
                    (0..n_features).filter(|i| !selected.contains(i)).collect();
                if !unselected.is_empty() {
                    selected.push(unselected[rng.random_range(0..unselected.len())]);
                }
            }
        }

        let mut dictionary = Mat::zeros(n_samples, k);
        for (dict_idx, &feature_idx) in selected.iter().enumerate() {
            let col = data.col(feature_idx);
            let norm = col.norm_l2();
            if norm > T::from_f64(1e-12).unwrap() {
                for i in 0..n_samples {
                    dictionary[(i, dict_idx)] = col[i] / norm;
                }
            }
        }
        dictionary
    }

    /// Helper function to pre-calculate all the distances
    ///
    /// This function will calculate the cosine distance between all the
    /// columns in a fast, sparse way
    ///
    /// ### Params
    ///
    /// * `data` - The input data
    ///
    /// ### Returns
    ///
    /// The flattened distances
    fn precompute_distances(&self, data: &MatRef<T>) -> Vec<T> {
        let n_features = data.ncols();
        let n_pairs = n_features * (n_features - 1) / 2;

        (0..n_pairs)
            .into_par_iter()
            .map(|pair_idx| {
                let (i, j) = triangle_to_indices(pair_idx, n_features);
                column_distance(data.col(i), data.col(j))
            })
            .collect()
    }

    /// Pull out the Laplacians out of the cache or recompute for given k_neighbours
    ///
    /// ### Params
    ///
    /// * `data` - Input data
    /// * `k_neighbours` - Number of neighbours in the KNN graph
    ///
    /// ### Returns
    ///
    /// Tuple of `(feature_laplacian, sample_laplacian)`
    fn get_laplacians(&mut self, data: &MatRef<T>, k_neighbours: usize) -> (Mat<T>, Mat<T>) {
        self.cache.validate_data(data);

        if let Some((feature_lap, sample_lap)) = self.cache.laplacian_cache.get(&k_neighbours) {
            return (feature_lap.clone(), sample_lap.clone());
        }

        let feature_laplacian = get_dgrdl_laplacian(&data.as_ref(), k_neighbours, true);
        let sample_laplacian = get_dgrdl_laplacian(&data.as_ref(), k_neighbours, false);

        self.cache.laplacian_cache.insert(
            k_neighbours,
            (feature_laplacian.clone(), sample_laplacian.clone()),
        );

        (feature_laplacian, sample_laplacian)
    }
}
