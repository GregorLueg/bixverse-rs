//! The variational Bayes solver: E-step, M-step, bound and the outer loop.
//!
//! Both steps are fan-outs over disjoint output slices, so neither needs a
//! reduction. The E-step gives each document its own column of `gamma`; the
//! M-step gives each term its own column of the sufficient statistics.
//!
//! ### Not caching `phinorm`
//!
//! The M-step recomputes the per-token normaliser the E-step already had.
//! Caching it would cost a buffer the size of the corpus, which on a real
//! scATAC matrix is gigabytes, to save one `k`-length dot product per non-zero
//! when the E-step has already done `inner_max_iter` of them. Recomputing is
//! the cheaper side of that trade, and it is what lets the M-step parallelise
//! over terms instead of documents, which is what removes the per-thread
//! `k x n_terms` accumulator the naive arrangement would need.
//!
//! ### References
//!
//! Hoffman, Blei and Bach, Online Learning for Latent Dirichlet Allocation,
//! NIPS, 2010

use faer::Mat;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;

use crate::core::math::special::digamma;
use crate::prelude::*;

use super::{
    LDA_DOC_SEED_MULT, LDA_INIT_SCALE, LDA_INIT_SHAPE, LDA_PHI_EPS, LdaCorpus, LdaLearning,
    LdaParams, LdaResult,
};

///////////////
// Utilities //
///////////////

/// Column-major dense buffer with a contiguous column accessor.
///
/// `faer::Mat` cannot hand out a `&mut [F]` column and is not mutably `Sync`,
/// and both hot loops here want exactly that: a contiguous `k`-slice per
/// document or per term, written from a `rayon` task that owns it. A flat `Vec`
/// with an explicit leading dimension is the smaller thing that does the job.
pub(crate) struct ColMajor<F> {
    /// Flat buffer, column `j` at `j * rows .. (j + 1) * rows`.
    data: Vec<F>,
    /// Leading dimension, i.e. the length of one column.
    rows: usize,
}

impl<F: BixverseFloat + Send + Sync> ColMajor<F> {
    /// Allocate a `rows x cols` buffer filled with `value`.
    ///
    /// ### Params
    ///
    /// * `rows` - Column length.
    /// * `cols` - Number of columns.
    /// * `value` - Fill value.
    ///
    /// ### Returns
    ///
    /// The initialised [ColMajor].
    fn filled(rows: usize, cols: usize, value: F) -> Self {
        Self {
            data: vec![value; rows * cols],
            rows,
        }
    }

    /// Immutable view of column `j`.
    ///
    /// ### Params
    ///
    /// * `j` - Column index.
    ///
    /// ### Returns
    ///
    /// The contiguous column slice.
    #[inline]
    pub(crate) fn col(&self, j: usize) -> &[F] {
        &self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// Iterator over columns, for `rayon` fan-out over disjoint slices.
    ///
    /// ### Returns
    ///
    /// A parallel iterator yielding each column as a mutable slice.
    fn par_cols_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [F]> {
        self.data.par_chunks_mut(self.rows)
    }

    /// Convert to a `faer::Mat` of the same shape.
    ///
    /// ### Returns
    ///
    /// A `rows x cols` matrix holding a copy of the buffer.
    fn to_mat(&self) -> Mat<F> {
        let cols = self.data.len() / self.rows;
        Mat::from_fn(self.rows, cols, |i, j| self.data[j * self.rows + i])
    }
}

/// `exp(digamma(x) - digamma(sum x))` over a Dirichlet parameter vector.
///
/// The expected value under `q` of `log theta`, exponentiated, which is the
/// only form either step ever needs.
///
/// ### Params
///
/// * `param` - Dirichlet parameters, strictly positive.
/// * `out` - Destination, same length as `param`.
#[inline]
fn exp_expected_log<F: BixverseFloat>(param: &[F], out: &mut [F]) {
    let total = param.iter().fold(F::zero(), |a, b| a + *b);
    let norm = digamma(total);
    for (o, p) in out.iter_mut().zip(param) {
        *o = (digamma(*p) - norm).exp();
    }
}

////////////
// E-step //
////////////

/// Run the per-document fixed point for one document.
///
/// Iterates `gamma_d = alpha + exp_elog_theta_d * sum_j (n_dj / phinorm_j) *
/// exp_elog_beta[:, j]` to convergence, which is the coordinate ascent on the
/// document's variational parameters with `phi` marginalised out.
///
/// ### Params
///
/// * `ids` - Term indices of this document's non-zeros.
/// * `cts` - Counts matching `ids`.
/// * `exp_elog_beta` - `k x n_terms` topic-term buffer.
/// * `gamma_d` - Document's Dirichlet parameters, updated in place.
/// * `exp_elog_theta_d` - Scratch of length `k`, left holding the converged
///   value.
/// * `alpha` - Document-topic prior.
/// * `inner_max_iter` - Iteration budget.
/// * `inner_tol` - Mean absolute change in `gamma_d` to stop at.
#[allow(clippy::too_many_arguments)]
fn e_step_document<F: BixverseFloat + BixverseSimd + Send + Sync>(
    ids: &[u32],
    cts: &[F],
    exp_elog_beta: &ColMajor<F>,
    gamma_d: &mut [F],
    exp_elog_theta_d: &mut [F],
    alpha: F,
    inner_max_iter: usize,
    inner_tol: F,
) {
    let k = gamma_d.len();
    let eps = F::from_f64(LDA_PHI_EPS).unwrap();
    let k_f = F::from_usize(k).unwrap();

    let mut acc = vec![F::zero(); k];
    exp_expected_log(gamma_d, exp_elog_theta_d);

    for _ in 0..inner_max_iter {
        acc.iter_mut().for_each(|v| *v = F::zero());
        for (&id, &ct) in ids.iter().zip(cts) {
            let beta_col = exp_elog_beta.col(id as usize);
            let phinorm = F::bxv_dot_simd(exp_elog_theta_d, beta_col) + eps;
            F::bxv_axpy_simd(&mut acc, ct / phinorm, beta_col);
        }

        let mut mean_change = F::zero();
        for ((g, a), t) in gamma_d
            .iter_mut()
            .zip(&acc)
            .zip(exp_elog_theta_d.iter().copied())
        {
            let updated = alpha + t * *a;
            mean_change += (updated - *g).abs();
            *g = updated;
        }

        exp_expected_log(gamma_d, exp_elog_theta_d);

        if mean_change / k_f < inner_tol {
            break;
        }
    }
}

/// Run the E-step over a set of documents.
///
/// Each document owns its own column of `gamma` and `exp_elog_theta`, so the
/// fan-out needs no synchronisation. Every document draws its initial `gamma`
/// from a seed derived from its own index, which is what keeps a run
/// reproducible independent of how `rayon` schedules it.
///
/// ### Params
///
/// * `corpus` - The corpus, read through its CSR view.
/// * `docs` - Document indices to process. Must contain no duplicates: the
///   scattered writes below are only disjoint because of that, and it is
///   asserted in debug builds.
/// * `exp_elog_beta` - `k x n_terms` topic-term buffer.
/// * `gamma` - `k x n_docs` destination, written at the `docs` columns.
/// * `exp_elog_theta` - `k x n_docs` destination, written at the `docs`
///   columns.
/// * `alpha` - Document-topic prior.
/// * `params` - Solver options, for the inner loop budget and the seed.
fn e_step<F: BixverseFloat + BixverseSimd + BixverseNumeric>(
    corpus: &LdaCorpus<F>,
    docs: &[usize],
    exp_elog_beta: &ColMajor<F>,
    gamma: &mut ColMajor<F>,
    exp_elog_theta: &mut ColMajor<F>,
    alpha: F,
    params: &LdaParams<F>,
) {
    debug_assert!(
        {
            let mut seen = rustc_hash::FxHashSet::default();
            docs.iter().all(|d| seen.insert(*d))
        },
        "e_step received duplicate document indices, which would alias the writes below"
    );

    let k = gamma.rows;
    let csr = &corpus.csr;
    let seed = params.seed;

    // Collect the raw pointers once: each task writes only its own columns, so
    // the aliasing the borrow checker cannot see here is genuinely absent.
    let gamma_ptr = gamma.data.as_mut_ptr() as usize;
    let theta_ptr = exp_elog_theta.data.as_mut_ptr() as usize;

    docs.par_iter().for_each(|&d| {
        let start = csr.indptr[d] as usize;
        let end = csr.indptr[d + 1] as usize;
        let ids = &csr.indices[start..end];
        let cts = &csr.data[start..end];

        let mut rng = SmallRng::seed_from_u64(seed ^ (d as u64).wrapping_mul(LDA_DOC_SEED_MULT));
        let dist = Gamma::new(LDA_INIT_SHAPE, LDA_INIT_SCALE).unwrap();
        let mut gamma_d: Vec<F> = (0..k)
            .map(|_| F::from_f64(dist.sample(&mut rng)).unwrap())
            .collect();
        let mut theta_d = vec![F::zero(); k];

        e_step_document(
            ids,
            cts,
            exp_elog_beta,
            &mut gamma_d,
            &mut theta_d,
            alpha,
            params.inner_max_iter,
            params.inner_tol,
        );

        // SAFETY: `d` is unique within `docs`, so the two `k`-length windows
        // written here are disjoint from every other task's.
        unsafe {
            let g = (gamma_ptr as *mut F).add(d * k);
            let t = (theta_ptr as *mut F).add(d * k);
            std::ptr::copy_nonoverlapping(gamma_d.as_ptr(), g, k);
            std::ptr::copy_nonoverlapping(theta_d.as_ptr(), t, k);
        }
    });
}

////////////
// M-step //
////////////

/// Accumulate the sufficient statistics over a set of documents.
///
/// Walks terms rather than documents, so each task owns one column of `sstats`
/// and the fan-out again needs no reduction. Returns the token part of the
/// bound as a by-product, since it is exactly the `sum n_dw log phinorm_dw`
/// this scan already forms.
///
/// ### Params
///
/// * `corpus` - The corpus, read through its CSC view.
/// * `doc_mask` - `None` to use every document, or a per-document flag marking
///   the mini-batch.
/// * `exp_elog_beta` - `k x n_terms` topic-term buffer.
/// * `exp_elog_theta` - `k x n_docs` document-topic buffer from the E-step.
/// * `sstats` - `k x n_terms` destination, overwritten.
///
/// ### Returns
///
/// `sum_{d, w} n_dw log phinorm_dw` over the selected documents.
fn m_step_sstats<F: BixverseFloat + BixverseSimd + BixverseNumeric>(
    corpus: &LdaCorpus<F>,
    doc_mask: Option<&[bool]>,
    exp_elog_beta: &ColMajor<F>,
    exp_elog_theta: &ColMajor<F>,
    sstats: &mut ColMajor<F>,
) -> f64 {
    let csc = &corpus.csc;
    let eps = F::from_f64(LDA_PHI_EPS).unwrap();

    sstats
        .par_cols_mut()
        .enumerate()
        .map(|(w, out)| {
            out.iter_mut().for_each(|v| *v = F::zero());
            let start = csc.indptr[w] as usize;
            let end = csc.indptr[w + 1] as usize;
            let beta_col = exp_elog_beta.col(w);

            let mut token_ll = 0.0_f64;
            for idx in start..end {
                let d = csc.indices[idx] as usize;
                if let Some(mask) = doc_mask
                    && !mask[d]
                {
                    continue;
                }
                let ct = csc.data[idx];
                let theta_col = exp_elog_theta.col(d);
                let phinorm = F::bxv_dot_simd(theta_col, beta_col) + eps;
                F::bxv_axpy_simd(out, ct / phinorm, theta_col);
                token_ll += ct.to_f64().unwrap_or(0.0) * phinorm.to_f64().unwrap_or(0.0).ln();
            }
            token_ll
        })
        .sum()
}

/// Turn sufficient statistics into the topic-term Dirichlet parameters.
///
/// `lambda = eta + sstats * exp_elog_beta` elementwise, optionally blended into
/// the previous `lambda` for the online variant.
///
/// ### Params
///
/// * `sstats` - Sufficient statistics from [m_step_sstats], consumed in place.
/// * `exp_elog_beta` - The buffer the statistics were accumulated against.
/// * `lambda` - `k x n_terms` Dirichlet parameters, updated in place.
/// * `eta` - Topic-term prior.
/// * `blend` - `None` for a full replacement, or `Some((rho, scale))` to move
///   `lambda` a step `rho` towards `eta + scale * sstats * exp_elog_beta`.
fn apply_sstats<F: BixverseFloat + Send + Sync>(
    sstats: &ColMajor<F>,
    exp_elog_beta: &ColMajor<F>,
    lambda: &mut ColMajor<F>,
    eta: F,
    blend: Option<(F, F)>,
) {
    match blend {
        None => lambda
            .data
            .par_iter_mut()
            .zip(sstats.data.par_iter())
            .zip(exp_elog_beta.data.par_iter())
            .for_each(|((l, s), b)| *l = eta + *s * *b),
        Some((rho, scale)) => {
            let one_minus = F::one() - rho;
            lambda
                .data
                .par_iter_mut()
                .zip(sstats.data.par_iter())
                .zip(exp_elog_beta.data.par_iter())
                .for_each(|((l, s), b)| *l = one_minus * *l + rho * (eta + scale * *s * *b));
        }
    }
}

///////////
// Bound //
///////////

/// Dirichlet part of the variational bound for one set of parameters.
///
/// `sum_j [(prior - q_j) E[log x_j] + lgamma(q_j) - lgamma(prior)] +
/// lgamma(prior * n) - lgamma(sum_j q_j)`, i.e. the `E[log p] - E[log q]`
/// contribution of one Dirichlet-distributed variable.
///
/// Accumulates in `f64` regardless of `F`: this is a sum of `n_terms` terms of
/// mixed sign, and in `f32` the cancellation shows up directly in the reported
/// bound.
///
/// ### Params
///
/// * `param` - Variational Dirichlet parameters.
/// * `prior` - The corresponding symmetric Dirichlet prior.
///
/// ### Returns
///
/// The contribution to the bound.
fn dirichlet_bound<F: BixverseFloat>(param: &[F], prior: F) -> f64 {
    let n = param.len();
    let prior_f = prior.to_f64().unwrap_or(0.0);
    let total = param.iter().fold(F::zero(), |a, b| a + *b);
    let norm = digamma(total).to_f64().unwrap_or(0.0);

    let mut acc = 0.0_f64;
    for p in param {
        let p_f = p.to_f64().unwrap_or(0.0);
        let e_log = digamma(*p).to_f64().unwrap_or(0.0) - norm;
        acc += (prior_f - p_f) * e_log + ln_gamma(p_f) - ln_gamma(prior_f);
    }

    acc + ln_gamma(prior_f * n as f64) - ln_gamma(total.to_f64().unwrap_or(0.0))
}

/// The variational bound (ELBO) for the current parameters.
///
/// The token term is handed in rather than recomputed, because the M-step scan
/// already formed it.
///
/// ### Params
///
/// * `token_ll` - `sum_{d, w} n_dw log phinorm_dw` from [m_step_sstats].
/// * `gamma` - `k x n_docs` document-topic parameters.
/// * `lambda` - `k x n_terms` topic-term parameters.
/// * `docs` - Documents contributing to `token_ll`.
/// * `n_terms` - Vocabulary size.
/// * `alpha` - Document-topic prior.
/// * `eta` - Topic-term prior.
/// * `doc_scale` - `n_docs / |docs|`, scaling the document part up to the whole
///   corpus for a mini-batch estimate. One for a batch sweep.
///
/// ### Returns
///
/// The bound, in `f64`.
#[allow(clippy::too_many_arguments)]
fn variational_bound<F: BixverseFloat + Send + Sync>(
    token_ll: f64,
    gamma: &ColMajor<F>,
    lambda: &ColMajor<F>,
    docs: &[usize],
    n_terms: usize,
    alpha: F,
    eta: F,
    doc_scale: f64,
) -> f64 {
    let doc_part: f64 = docs
        .par_iter()
        .map(|&d| dirichlet_bound(gamma.col(d), alpha))
        .sum();

    let topic_part = topic_dirichlet_bound(lambda, eta, n_terms);

    (token_ll + doc_part) * doc_scale + topic_part
}

/// Topic-term Dirichlet contribution to the bound.
///
/// The Dirichlet here runs over terms within a topic, i.e. along the rows of
/// the `k x n_terms` layout. Gathering those rows would stride the whole
/// buffer `k` times, so this instead carries `k` accumulators across one
/// contiguous sweep of the columns.
///
/// ### Params
///
/// * `lambda` - `k x n_terms` topic-term parameters.
/// * `eta` - Topic-term prior.
/// * `n_terms` - Vocabulary size.
///
/// ### Returns
///
/// The contribution to the bound, summed over topics.
fn topic_dirichlet_bound<F: BixverseFloat + Send + Sync>(
    lambda: &ColMajor<F>,
    eta: F,
    n_terms: usize,
) -> f64 {
    let k = lambda.rows;
    let eta_f = eta.to_f64().unwrap_or(0.0);

    let mut row_sums = vec![F::zero(); k];
    for w in 0..n_terms {
        for (s, v) in row_sums.iter_mut().zip(lambda.col(w)) {
            *s += *v;
        }
    }
    let norms: Vec<f64> = row_sums
        .iter()
        .map(|s| digamma(*s).to_f64().unwrap_or(0.0))
        .collect();

    let mut acc = vec![0.0_f64; k];
    for w in 0..n_terms {
        for ((a, v), norm) in acc.iter_mut().zip(lambda.col(w)).zip(&norms) {
            let p = v.to_f64().unwrap_or(0.0);
            *a += (eta_f - p) * (digamma(p) - norm) + ln_gamma(p);
        }
    }

    let const_part = ln_gamma(eta_f * n_terms as f64) - n_terms as f64 * ln_gamma(eta_f);
    acc.iter()
        .zip(&row_sums)
        .map(|(a, total)| a + const_part - ln_gamma(total.to_f64().unwrap_or(0.0)))
        .sum()
}

//////////
// Main //
//////////

/// Initialise the topic-term Dirichlet parameters.
///
/// Draws `Gamma(100, 0.01)` per entry, seeded per term so the draw does not
/// depend on the fan-out order.
///
/// ### Params
///
/// * `k` - Number of topics.
/// * `n_terms` - Vocabulary size.
/// * `seed` - Run seed.
///
/// ### Returns
///
/// The `k x n_terms` initial `lambda`.
fn init_lambda<F: BixverseFloat + Send + Sync>(k: usize, n_terms: usize, seed: u64) -> ColMajor<F> {
    let mut lambda = ColMajor::filled(k, n_terms, F::zero());
    lambda.par_cols_mut().enumerate().for_each(|(w, col)| {
        let mut rng =
            SmallRng::seed_from_u64(seed ^ (w as u64).wrapping_mul(LDA_DOC_SEED_MULT) ^ 0xA5A5);
        let dist = Gamma::new(LDA_INIT_SHAPE, LDA_INIT_SCALE).unwrap();
        for v in col.iter_mut() {
            *v = F::from_f64(dist.sample(&mut rng)).unwrap();
        }
    });
    lambda
}

/// Normalise a set of columns to sum to one.
///
/// ### Params
///
/// * `buf` - Buffer whose columns are normalised in place.
fn normalise_columns<F: BixverseFloat + Send + Sync>(buf: &mut ColMajor<F>) {
    buf.par_cols_mut().for_each(|col| {
        let total = col.iter().fold(F::zero(), |a, b| a + *b);
        if total > F::zero() {
            col.iter_mut().for_each(|v| *v /= total);
        }
    });
}

/// Fit an LDA model by variational Bayes.
///
/// ### Params
///
/// * `corpus` - The corpus in both orientations.
/// * `k` - Number of topics.
/// * `params` - Solver options.
/// * `verbosity` - Controls the per-iteration reporting.
///
/// ### Returns
///
/// The fitted [LdaResult].
pub(crate) fn fit_vb<F>(
    corpus: &LdaCorpus<F>,
    k: usize,
    params: &LdaParams<F>,
    verbosity: Verbosity,
) -> Result<LdaResult<F>, BixverseErrors>
where
    F: BixverseFloat + BixverseNumeric + BixverseSimd,
{
    if k == 0 || k > corpus.n_terms {
        return Err(BixverseErrors::LdaInvalidTopicCount {
            requested: k,
            max_available: corpus.n_terms,
        });
    }
    let (alpha, eta) = params.resolve_priors(k)?;

    let n_docs = corpus.n_docs;
    let n_terms = corpus.n_terms;

    let mut lambda = init_lambda::<F>(k, n_terms, params.seed);
    let mut exp_elog_beta = ColMajor::filled(k, n_terms, F::zero());
    let mut sstats = ColMajor::filled(k, n_terms, F::zero());
    let mut gamma = ColMajor::filled(k, n_docs, alpha);
    let mut exp_elog_theta = ColMajor::filled(k, n_docs, F::zero());

    // A zero cadence would divide by zero below, and it arrives from R as a
    // plausible way of spelling "never check".
    let check_every = params.check_every.max(1);

    let all_docs: Vec<usize> = (0..n_docs).collect();
    let mut previous_bound = f64::NEG_INFINITY;
    let mut bound = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    match params.learning {
        LdaLearning::Batch => {
            for iter in 0..params.max_iter {
                n_iter = iter + 1;
                update_exp_elog_beta(&lambda, &mut exp_elog_beta, n_terms);
                e_step(
                    corpus,
                    &all_docs,
                    &exp_elog_beta,
                    &mut gamma,
                    &mut exp_elog_theta,
                    alpha,
                    params,
                );
                let token_ll =
                    m_step_sstats(corpus, None, &exp_elog_beta, &exp_elog_theta, &mut sstats);
                apply_sstats(&sstats, &exp_elog_beta, &mut lambda, eta, None);

                let due = n_iter % check_every == 0 || n_iter == params.max_iter;
                if due {
                    bound = variational_bound(
                        token_ll, &gamma, &lambda, &all_docs, n_terms, alpha, eta, 1.0,
                    );
                    if verbosity.normal_verbosity() {
                        println!("LDA: iteration {n_iter}, bound {bound:.4}");
                    }
                    if previous_bound.is_finite() {
                        let rel = (bound - previous_bound).abs() / previous_bound.abs().max(1.0);
                        if rel < params.tol.to_f64().unwrap_or(0.0) {
                            converged = true;
                            break;
                        }
                    }
                    previous_bound = bound;
                }
            }
        }
        LdaLearning::Online {
            batch_size,
            tau0,
            kappa,
            n_epochs,
        } => {
            let batch_size = batch_size.clamp(1, n_docs);
            let mut order = all_docs.clone();
            let mut rng = SmallRng::seed_from_u64(params.seed);
            let mut step = 0_u64;

            for epoch in 0..n_epochs {
                order.shuffle(&mut rng);
                for batch in order.chunks(batch_size) {
                    n_iter += 1;
                    let mut mask = vec![false; n_docs];
                    batch.iter().for_each(|&d| mask[d] = true);

                    update_exp_elog_beta(&lambda, &mut exp_elog_beta, n_terms);
                    e_step(
                        corpus,
                        batch,
                        &exp_elog_beta,
                        &mut gamma,
                        &mut exp_elog_theta,
                        alpha,
                        params,
                    );
                    let token_ll = m_step_sstats(
                        corpus,
                        Some(&mask),
                        &exp_elog_beta,
                        &exp_elog_theta,
                        &mut sstats,
                    );

                    let rho = F::from_f64((tau0 + step as f64).powf(-kappa)).unwrap();
                    let scale =
                        F::from_usize(n_docs).unwrap() / F::from_usize(batch.len()).unwrap();
                    apply_sstats(
                        &sstats,
                        &exp_elog_beta,
                        &mut lambda,
                        eta,
                        Some((rho, scale)),
                    );
                    step += 1;

                    bound = variational_bound(
                        token_ll,
                        &gamma,
                        &lambda,
                        batch,
                        n_terms,
                        alpha,
                        eta,
                        n_docs as f64 / batch.len() as f64,
                    );
                }

                if verbosity.normal_verbosity() {
                    println!("LDA: epoch {}, bound estimate {bound:.4}", epoch + 1);
                }
                if previous_bound.is_finite() {
                    let rel = (bound - previous_bound).abs() / previous_bound.abs().max(1.0);
                    if rel < params.tol.to_f64().unwrap_or(0.0) {
                        converged = true;
                        break;
                    }
                }
                previous_bound = bound;
            }

            // The online sweep only ever touched the sampled documents, so the
            // untouched columns of gamma still hold the prior. One batch E-step
            // over the whole corpus makes the returned model complete.
            update_exp_elog_beta(&lambda, &mut exp_elog_beta, n_terms);
            e_step(
                corpus,
                &all_docs,
                &exp_elog_beta,
                &mut gamma,
                &mut exp_elog_theta,
                alpha,
                params,
            );
        }
    }

    if lambda.data.iter().any(|v| !v.is_finite()) || gamma.data.iter().any(|v| !v.is_finite()) {
        return Err(BixverseErrors::LdaNonFinite);
    }

    normalise_columns(&mut gamma);
    let cell_topic = gamma.to_mat();

    // lambda is k x n_terms; the public result is terms x k so a topic's term
    // vector is the contiguous direction for the metrics and any binarisation.
    let topic_region = topic_region_from_lambda(&lambda, k, n_terms);

    let bound_f = F::from_f64(bound).unwrap_or(F::neg_infinity());
    let perplexity = if corpus.total_tokens > 0.0 && bound.is_finite() {
        F::from_f64((-bound / corpus.total_tokens).exp()).unwrap_or(F::infinity())
    } else {
        F::infinity()
    };

    Ok(LdaResult {
        cell_topic,
        topic_region,
        bound: bound_f,
        perplexity,
        n_iter,
        converged,
    })
}

/// Refresh `exp(E[log beta])` from the current `lambda`.
///
/// The per-topic normaliser runs across terms, i.e. across columns of the
/// `k x n_terms` layout, so it is formed once up front rather than per column.
///
/// ### Params
///
/// * `lambda` - `k x n_terms` topic-term parameters.
/// * `out` - `k x n_terms` destination.
/// * `n_terms` - Vocabulary size.
fn update_exp_elog_beta<F: BixverseFloat + Send + Sync>(
    lambda: &ColMajor<F>,
    out: &mut ColMajor<F>,
    n_terms: usize,
) {
    let k = lambda.rows;
    let mut row_sums = vec![F::zero(); k];
    for w in 0..n_terms {
        let col = lambda.col(w);
        for (s, v) in row_sums.iter_mut().zip(col) {
            *s += *v;
        }
    }
    let norms: Vec<F> = row_sums.iter().map(|s| digamma(*s)).collect();

    let lambda_data = &lambda.data;
    out.par_cols_mut().enumerate().for_each(|(w, col)| {
        let src = &lambda_data[w * k..(w + 1) * k];
        for ((o, v), n) in col.iter_mut().zip(src).zip(&norms) {
            *o = (digamma(*v) - *n).exp();
        }
    });
}

/// Build the terms x topics probability matrix from `lambda`.
///
/// ### Params
///
/// * `lambda` - `k x n_terms` topic-term parameters.
/// * `k` - Number of topics.
/// * `n_terms` - Vocabulary size.
///
/// ### Returns
///
/// A `n_terms x k` matrix whose columns sum to one.
fn topic_region_from_lambda<F: BixverseFloat + Send + Sync>(
    lambda: &ColMajor<F>,
    k: usize,
    n_terms: usize,
) -> Mat<F> {
    let mut totals = vec![F::zero(); k];
    for w in 0..n_terms {
        for (t, v) in totals.iter_mut().zip(lambda.col(w)) {
            *t += *v;
        }
    }
    Mat::from_fn(n_terms, k, |w, topic| {
        let total = totals[topic];
        if total > F::zero() {
            lambda.col(w)[topic] / total
        } else {
            F::zero()
        }
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::methods::lda::{LdaCorpus, lda_fit};
    use approx::assert_relative_eq;

    /// Build a CSR documents x terms matrix from dense rows.
    ///
    /// ### Params
    ///
    /// * `rows` - One row per document, each of length `n_terms`.
    ///
    /// ### Returns
    ///
    /// The CSR matrix with the zeros dropped.
    pub(crate) fn csr_from_rows(rows: &[Vec<f64>]) -> CompressedSparseData2<f64> {
        let n_terms = rows[0].len();
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0_u32];
        for row in rows {
            for (w, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    data.push(v);
                    indices.push(w as u32);
                }
            }
            indptr.push(data.len() as u32);
        }
        CompressedSparseData2::from_parts(
            data,
            indices,
            indptr,
            None,
            CompressedSparseFormat::Csr,
            (rows.len(), n_terms),
        )
    }

    /// Three disjoint vocabulary blocks, each document drawing from one block.
    ///
    /// ### Params
    ///
    /// * `docs_per_block` - Documents generated per block.
    /// * `block_size` - Terms per block.
    ///
    /// ### Returns
    ///
    /// A binary documents x terms matrix with `3 * block_size` terms.
    pub(crate) fn block_corpus(
        docs_per_block: usize,
        block_size: usize,
    ) -> CompressedSparseData2<f64> {
        let n_terms = 3 * block_size;
        let mut rows = Vec::new();
        for block in 0..3 {
            for d in 0..docs_per_block {
                let mut row = vec![0.0; n_terms];
                // Drop one term per document so documents within a block differ
                for t in 0..block_size {
                    if (t + d) % block_size != 0 {
                        row[block * block_size + t] = 1.0;
                    }
                }
                rows.push(row);
            }
        }
        csr_from_rows(&rows)
    }

    /// Shared solver options: tight tolerance, checked every iteration.
    ///
    /// ### Returns
    ///
    /// The [LdaParams] every test below starts from.
    fn test_params() -> LdaParams<f64> {
        LdaParams {
            // The cisTopic default of 50/k is far too strong for corpora this
            // small: a four-token document cannot outvote a prior of 16 per
            // topic, so every document would come back uniform. Real scATAC
            // documents carry thousands of regions and are unaffected.
            alpha: 0.1,
            alpha_by_topic: false,
            max_iter: 60,
            check_every: 1,
            tol: 1e-9,
            seed: 42,
            ..Default::default()
        }
    }

    /// The batch bound must never decrease: that is the guarantee coordinate
    /// ascent on the ELBO buys, and the cheapest thing that catches a sign or
    /// indexing slip in either step.
    #[test]
    fn test_batch_bound_is_monotone() {
        let matrix = block_corpus(6, 5);
        let corpus = LdaCorpus::<f64>::new(&matrix).unwrap();

        let mut previous = f64::NEG_INFINITY;
        for iter in 1..=25 {
            let p = LdaParams {
                max_iter: iter,
                ..test_params()
            };
            let bound = fit_vb(&corpus, 3, &p, Verbosity::Quiet).unwrap().bound;
            assert!(
                bound >= previous - 1e-6,
                "bound decreased at iteration {iter}: {previous} -> {bound}"
            );
            previous = bound;
        }
    }

    /// Each recovered topic should concentrate on exactly one vocabulary block.
    #[test]
    fn test_recovers_disjoint_blocks() {
        let block_size = 5;
        let matrix = block_corpus(8, block_size);
        let model = lda_fit(&matrix, 3, Some(test_params()), 0).unwrap();

        let mut claimed = [false; 3];
        for topic in 0..3 {
            let col = model.topic_region.col_as_slice(topic);
            let mass: Vec<f64> = (0..3)
                .map(|b| col[b * block_size..(b + 1) * block_size].iter().sum())
                .collect();
            let best = (0..3).max_by(|&a, &b| mass[a].total_cmp(&mass[b])).unwrap();
            assert!(
                mass[best] > 0.9,
                "topic {topic} spread across blocks: {mass:?}"
            );
            assert!(!claimed[best], "two topics claimed block {best}");
            claimed[best] = true;
        }
    }

    /// The same seed must give the same fit, whatever the thread count. This is
    /// what the per-document seed derivation exists for.
    #[test]
    fn test_reproducible_across_thread_counts() {
        let matrix = block_corpus(6, 5);
        let params = test_params();

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| lda_fit(&matrix, 3, Some(params), 0).unwrap())
        };

        let single = run(1);
        let multi = run(4);
        for topic in 0..3 {
            for (a, b) in single
                .topic_region
                .col_as_slice(topic)
                .iter()
                .zip(multi.topic_region.col_as_slice(topic))
            {
                assert_eq!(a, b);
            }
        }
        // The parameters are bit-identical, but the bound is a rayon
        // reduction whose association order follows the thread count.
        assert_relative_eq!(single.bound, multi.bound, max_relative = 1e-12);
    }

    /// A single topic puts every document wholly in it.
    #[test]
    fn test_single_topic_is_degenerate() {
        let matrix = block_corpus(4, 4);
        let model = lda_fit(&matrix, 1, Some(test_params()), 0).unwrap();
        assert_eq!(model.cell_topic.nrows(), 1);
        for d in 0..model.cell_topic.ncols() {
            assert_relative_eq!(model.cell_topic[(0, d)], 1.0, epsilon = 1e-12);
        }
        let total: f64 = model.topic_region.col_as_slice(0).iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }

    /// An empty document must not divide by zero; with no evidence it keeps the
    /// prior, which is uniform over topics.
    #[test]
    fn test_empty_document_falls_back_to_prior() {
        let rows = vec![
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0; 4],
        ];
        let matrix = csr_from_rows(&rows);
        let model = lda_fit(&matrix, 2, Some(test_params()), 0).unwrap();

        let empty = model.cell_topic.col_as_slice(3);
        assert!(empty.iter().all(|v| v.is_finite()));
        assert_relative_eq!(empty[0], 0.5, epsilon = 1e-9);
        assert_relative_eq!(empty[1], 0.5, epsilon = 1e-9);
    }

    /// A term nobody uses still gets a well-formed, tiny probability.
    #[test]
    fn test_unused_term_keeps_prior_mass() {
        let rows = vec![
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ];
        let matrix = csr_from_rows(&rows);
        let model = lda_fit(&matrix, 2, Some(test_params()), 0).unwrap();

        for topic in 0..2 {
            let col = model.topic_region.col_as_slice(topic);
            assert!(col[3] > 0.0 && col[3] < 0.05, "unused term got {}", col[3]);
            assert_relative_eq!(col.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
        }
    }

    /// Columns of both output matrices are probability vectors.
    #[test]
    fn test_outputs_are_normalised() {
        let matrix = block_corpus(5, 4);
        let model = lda_fit(&matrix, 3, Some(test_params()), 0).unwrap();

        for d in 0..model.cell_topic.ncols() {
            let s: f64 = model.cell_topic.col_as_slice(d).iter().sum();
            assert_relative_eq!(s, 1.0, epsilon = 1e-10);
        }
        for topic in 0..model.topic_region.ncols() {
            let s: f64 = model.topic_region.col_as_slice(topic).iter().sum();
            assert_relative_eq!(s, 1.0, epsilon = 1e-10);
        }
    }

    /// The online sweep lands in the same place as the batch one on a corpus
    /// this separable, which is what keeps the mini-batch arm honest.
    #[test]
    fn test_online_recovers_blocks() {
        let block_size = 5;
        let matrix = block_corpus(10, block_size);
        let params = LdaParams {
            learning: LdaLearning::Online {
                batch_size: 6,
                tau0: 10.0,
                kappa: 0.7,
                n_epochs: 40,
            },
            ..test_params()
        };
        let model = lda_fit(&matrix, 3, Some(params), 0).unwrap();

        let mut claimed = [false; 3];
        for topic in 0..3 {
            let col = model.topic_region.col_as_slice(topic);
            let mass: Vec<f64> = (0..3)
                .map(|b| col[b * block_size..(b + 1) * block_size].iter().sum())
                .collect();
            let best = (0..3).max_by(|&a, &b| mass[a].total_cmp(&mass[b])).unwrap();
            assert!(mass[best] > 0.8, "online topic {topic} spread: {mass:?}");
            claimed[best] = true;
        }
        assert!(claimed.iter().all(|c| *c), "online run collapsed topics");
    }

    /// Guard rails on the topic count and the hyperparameters.
    #[test]
    fn test_invalid_inputs_error() {
        let matrix = block_corpus(3, 3);
        assert!(matches!(
            lda_fit(&matrix, 0, Some(test_params()), 0),
            Err(BixverseErrors::LdaInvalidTopicCount { .. })
        ));
        assert!(matches!(
            lda_fit(&matrix, 1000, Some(test_params()), 0),
            Err(BixverseErrors::LdaInvalidTopicCount { .. })
        ));

        let bad = LdaParams {
            eta: 0.0,
            ..test_params()
        };
        assert!(matches!(
            lda_fit(&matrix, 2, Some(bad), 0),
            Err(BixverseErrors::LdaInvalidHyperparameter { .. })
        ));
    }

    /// A CSC input describes the same corpus as its CSR twin, so both fit
    /// identically. Guards the orientation handling in the corpus builder.
    #[test]
    fn test_csc_input_matches_csr() {
        let rows = vec![
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ];
        let csr = csr_from_rows(&rows);
        let csc = csr.transform();

        let a = lda_fit(&csr, 2, Some(test_params()), 0).unwrap();
        let b = lda_fit(&csc, 2, Some(test_params()), 0).unwrap();
        assert_relative_eq!(a.bound, b.bound, epsilon = 1e-9);
    }

    /// An all-zero matrix is rejected rather than producing a degenerate fit.
    #[test]
    fn test_empty_matrix_errors() {
        let matrix = csr_from_rows(&[vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert!(matches!(
            lda_fit(&matrix, 1, Some(test_params()), 0),
            Err(BixverseErrors::LdaEmptyMatrix)
        ));
    }

    /// The solver is generic over the float, and `f32` is the one that matters
    /// for a real cells x regions matrix, where three dense `k x n_terms`
    /// buffers is the binding memory cost.
    #[test]
    fn test_f32_recovers_blocks() {
        let block_size = 5;
        let dense = block_corpus(8, block_size);
        let data: Vec<f32> = dense.data.iter().map(|v| *v as f32).collect();
        let matrix: CompressedSparseData2<f32> = CompressedSparseData2::from_parts(
            data,
            dense.indices.clone(),
            dense.indptr.clone(),
            None,
            CompressedSparseFormat::Csr,
            dense.shape,
        );

        let params = LdaParams::<f32> {
            alpha: 0.1,
            alpha_by_topic: false,
            max_iter: 60,
            check_every: 1,
            tol: 1e-7,
            seed: 42,
            ..Default::default()
        };
        let model = lda_fit(&matrix, 3, Some(params), 0).unwrap();

        assert!(model.bound.is_finite());
        for topic in 0..3 {
            let col = model.topic_region.col_as_slice(topic);
            let mass: Vec<f32> = (0..3)
                .map(|b| col[b * block_size..(b + 1) * block_size].iter().sum())
                .collect();
            let best = (0..3).max_by(|&a, &b| mass[a].total_cmp(&mass[b])).unwrap();
            assert!(mass[best] > 0.9, "f32 topic {topic} spread: {mass:?}");
        }
    }

    /// The bound must agree with scikit-learn's when both are evaluated on the
    /// *same* parameters.
    ///
    /// This is the check that pins the ELBO formula down. Comparing two fits
    /// would only ever show that both found some optimum; feeding scikit-learn's
    /// own `lambda` and `gamma` through this crate's bound isolates the formula
    /// from the solver.
    #[test]
    fn test_bound_matches_sklearn_on_shared_params() {
        use crate::methods::lda::sklearn_fixture::*;

        let matrix = sklearn_corpus();
        let corpus = LdaCorpus::<f64>::new(&matrix).unwrap();
        let (k, n_terms, n_docs) = (SKLEARN_K, SKLEARN_N_TERMS, SKLEARN_N_DOCS);

        let mut lambda = ColMajor::filled(k, n_terms, 0.0f64);
        lambda.data.copy_from_slice(&SKLEARN_LAMBDA);
        let mut gamma = ColMajor::filled(k, n_docs, 0.0f64);
        gamma.data.copy_from_slice(&SKLEARN_GAMMA);

        let mut exp_elog_beta = ColMajor::filled(k, n_terms, 0.0f64);
        update_exp_elog_beta(&lambda, &mut exp_elog_beta, n_terms);

        let mut exp_elog_theta = ColMajor::filled(k, n_docs, 0.0f64);
        for d in 0..n_docs {
            let g = gamma.col(d).to_vec();
            let mut out = vec![0.0f64; k];
            exp_expected_log(&g, &mut out);
            exp_elog_theta.data[d * k..(d + 1) * k].copy_from_slice(&out);
        }

        let mut sstats = ColMajor::filled(k, n_terms, 0.0f64);
        let token_ll = m_step_sstats(&corpus, None, &exp_elog_beta, &exp_elog_theta, &mut sstats);

        let all: Vec<usize> = (0..n_docs).collect();
        let bound = variational_bound(
            token_ll,
            &gamma,
            &lambda,
            &all,
            n_terms,
            SKLEARN_PRIOR,
            SKLEARN_PRIOR,
            1.0,
        );

        assert_relative_eq!(bound, SKLEARN_BOUND, max_relative = 1e-12);
    }

    /// Fitting the reference corpus must not land below scikit-learn's optimum.
    ///
    /// Deliberately one-sided. The bound formula is already pinned by
    /// [test_bound_matches_sklearn_on_shared_params], so the only thing left to
    /// check is that the solver does at least as well, and on this corpus it
    /// does better: scikit-learn settles into a local optimum where two topics
    /// absorb the bleed terms unevenly. Asserting equality of the fitted
    /// distributions would be asserting that we reproduce that local optimum.
    #[test]
    fn test_fit_reaches_sklearn_optimum() {
        use crate::methods::lda::sklearn_fixture::*;

        let matrix = sklearn_corpus();
        let params = LdaParams {
            alpha: SKLEARN_PRIOR,
            alpha_by_topic: false,
            eta: SKLEARN_PRIOR,
            eta_by_topic: false,
            max_iter: 200,
            tol: 1e-12,
            inner_max_iter: 200,
            inner_tol: 1e-5,
            check_every: 25,
            learning: LdaLearning::Batch,
            seed: 0,
        };
        let model = lda_fit(&matrix, SKLEARN_K, Some(params), 0).unwrap();

        assert!(
            model.bound >= SKLEARN_BOUND,
            "bound {} fell below the scikit-learn reference {SKLEARN_BOUND}",
            model.bound
        );

        // Perplexity must stay consistent with the bound it is derived from.
        let n_tokens = matrix.get_nnz() as f64;
        assert_relative_eq!(
            model.perplexity,
            (-model.bound / n_tokens).exp(),
            max_relative = 1e-12
        );

        // Each topic should own one of the four term blocks.
        let block = SKLEARN_N_TERMS / SKLEARN_K;
        let mut claimed = [false; SKLEARN_K];
        for topic in 0..SKLEARN_K {
            let col = model.topic_region.col_as_slice(topic);
            let mass: Vec<f64> = (0..SKLEARN_K)
                .map(|b| col[b * block..(b + 1) * block].iter().sum())
                .collect();
            let best = (0..SKLEARN_K)
                .max_by(|&a, &b| mass[a].total_cmp(&mass[b]))
                .unwrap();
            assert!(mass[best] > 0.8, "topic {topic} spread: {mass:?}");
            assert!(!claimed[best], "two topics claimed block {best}");
            claimed[best] = true;
        }
    }
}
