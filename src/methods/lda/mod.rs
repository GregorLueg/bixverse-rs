//! Latent Dirichlet allocation by variational Bayes, for sparse count and
//! binary matrices.
//!
//! The topic model behind cisTopic: given a binarised cells x regions scATAC
//! matrix, recover a cell-topic distribution for clustering and a topic-region
//! distribution for region set discovery. Nothing here is ATAC-specific, so any
//! documents x terms count matrix works.
//!
//! ### Why variational Bayes and not collapsed Gibbs
//!
//! cisTopic and pycisTopic both run collapsed Gibbs sampling, which is
//! inherently sequential over tokens and parallelises only across models. This
//! implementation follows Hoffman, Blei and Bach instead: the E-step factorises
//! over documents and the M-step over terms, so both fan out with `rayon`, and
//! the result is deterministic given a seed. It converges to the same structure
//! but is not bit-comparable with a Gibbs run.
//!
//! ### Layout
//!
//! The solver holds the corpus in both orientations, CSR for the document-major
//! E-step and CSC for the term-major M-step, the same trick the sparse NMF
//! backend uses. Every dense buffer is column-major `faer::Mat`, laid out so
//! that the `k`-vector each hot loop touches is contiguous.
//!
//! Note that [CompressedSparseData2] indexes with `u32`, which caps a corpus at
//! roughly 4.3e9 non-zeros.
//!
//! ### References
//!
//! Hoffman, Blei and Bach, Online Learning for Latent Dirichlet Allocation,
//! NIPS, 2010
//!
//! Bravo Gonzalez-Blas et al., cisTopic, Nature Methods, 2019

use faer::Mat;
use num_traits::ToPrimitive;

use crate::prelude::*;

pub mod metrics;
pub mod vb;

#[cfg(test)]
pub(crate) mod sklearn_fixture;

use metrics::{LdaMetrics, lda_metrics};
use vb::fit_vb;

////////////
// Consts //
////////////

/// Multiplier turning a document index into a per-document RNG seed.
///
/// The E-step initialises every document's `gamma` from a random draw. Deriving
/// each document's seed from the run seed rather than sharing one stream is
/// what makes a run reproducible regardless of how `rayon` schedules the
/// documents. A large odd multiplier keeps neighbouring documents from landing
/// on correlated streams.
const LDA_DOC_SEED_MULT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Shape of the gamma draw that initialises the variational parameters.
///
/// `Gamma(100, 0.01)` has mean one and a standard deviation of a tenth, so it
/// breaks the symmetry between topics without starting any of them far from the
/// uniform. Matches the initialisation used by Hoffman's reference code.
const LDA_INIT_SHAPE: f64 = 100.0;

/// Scale of the gamma draw that initialises the variational parameters. See
/// [LDA_INIT_SHAPE].
const LDA_INIT_SCALE: f64 = 0.01;

/// Floor added to the per-token normaliser in the E-step and M-step.
///
/// `phinorm` is a sum of strictly positive terms, but a document whose terms
/// all sit in topics that have collapsed can drive it to zero and turn the
/// following division into an infinity. Hoffman's reference uses the same
/// guard.
const LDA_PHI_EPS: f64 = 1e-100;

/// Default number of top-scoring topics averaged into the reported coherence.
///
/// pycisTopic's `top_topics_coh`. Coherence over all topics is dragged down by
/// the ones that never specialised, which makes it useless for comparing across
/// `k`; averaging the best few is what makes the metric monotone enough to
/// select on.
pub const DEFAULT_TOP_TOPICS_COH: usize = 5;

/// Smallest topic count admitted into a coherence-based model selection.
///
/// pycisTopic's `min_topics_coh`. Below five topics the top-`n` region sets
/// overlap so heavily that coherence saturates and would always win.
pub const DEFAULT_MIN_TOPICS_COH: usize = 5;

///////////
// Enums //
///////////

/// How the variational parameters are updated across the corpus.
#[derive(Clone, Copy, Debug, Default)]
pub enum LdaLearning {
    /// One sweep over every document per iteration, then a full replacement of
    /// `lambda`. Deterministic and monotone in the bound.
    #[default]
    Batch,
    /// Stochastic updates from shuffled mini-batches, with `lambda` moved a
    /// decaying step towards each batch's estimate. Reaches a usable fit in far
    /// fewer passes when the corpus is large, at the cost of the monotonicity
    /// guarantee.
    Online {
        /// Documents per mini-batch.
        batch_size: usize,
        /// Learning-rate offset `tau0`, damping the first few steps.
        tau0: f64,
        /// Learning-rate decay `kappa`, in `(0.5, 1.0]`.
        kappa: f64,
        /// Passes over the whole corpus.
        n_epochs: usize,
    },
}

/// Parse the LDA learning strategy.
///
/// ### Params
///
/// * `s` - String to parse. One of `"batch"` or `"online"`.
/// * `batch_size` - Documents per mini-batch, used only by the online variant.
/// * `n_epochs` - Passes over the corpus, used only by the online variant.
///
/// ### Returns
///
/// The option of [LdaLearning], with the online variant carrying the default
/// `tau0` and `kappa`.
pub fn parse_lda_learning(s: &str, batch_size: usize, n_epochs: usize) -> Option<LdaLearning> {
    match s.to_lowercase().as_str() {
        "batch" | "full" => Some(LdaLearning::Batch),
        "online" | "stochastic" | "minibatch" => Some(LdaLearning::Online {
            batch_size,
            tau0: 10.0,
            kappa: 0.7,
            n_epochs,
        }),
        _ => None,
    }
}

////////////
// Params //
////////////

/// Options for the variational Bayes LDA solver.
///
/// Defaults follow pycisTopic so the knobs mean the same thing on both sides.
#[derive(Clone, Copy, Debug)]
pub struct LdaParams<F: BixverseFloat> {
    /// Dirichlet prior on the document-topic distributions.
    pub alpha: F,
    /// Whether `alpha` is divided by the topic count. `true` gives the
    /// Griffiths and Steyvers `50 / k` heuristic that cisTopic defaults to.
    pub alpha_by_topic: bool,
    /// Dirichlet prior on the topic-term distributions.
    pub eta: F,
    /// Whether `eta` is divided by the topic count.
    pub eta_by_topic: bool,
    /// Maximum outer iterations (batch), ignored by the online variant which
    /// counts epochs instead.
    pub max_iter: usize,
    /// Relative change in the bound below which the solver stops.
    pub tol: F,
    /// Maximum fixed-point iterations of the per-document E-step.
    pub inner_max_iter: usize,
    /// Mean absolute change in `gamma` below which the E-step stops.
    pub inner_tol: F,
    /// Cadence, in iterations, for evaluating the bound and testing `tol`.
    pub check_every: usize,
    /// How the corpus is swept, see [LdaLearning].
    pub learning: LdaLearning,
    /// Seed for the `lambda` and `gamma` initialisation.
    pub seed: u64,
}

impl<F: BixverseFloat> LdaParams<F> {
    /// Generate a new instance of [LdaParams]
    ///
    /// ### Params
    ///
    /// * `alpha` - Document-topic Dirichlet prior.
    /// * `alpha_by_topic` - Divide `alpha` by the topic count.
    /// * `eta` - Topic-term Dirichlet prior.
    /// * `eta_by_topic` - Divide `eta` by the topic count.
    /// * `max_iter` - Maximum outer iterations.
    /// * `tol` - Relative bound change for convergence.
    /// * `inner_max_iter` - Maximum per-document E-step iterations.
    /// * `inner_tol` - Mean absolute `gamma` change for E-step convergence.
    /// * `check_every` - Iterations between bound evaluations.
    /// * `learning` - Batch or online, see [LdaLearning].
    /// * `seed` - Seed for the initialisation.
    ///
    /// ### Returns
    ///
    /// The initialised [LdaParams].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        alpha: F,
        alpha_by_topic: bool,
        eta: F,
        eta_by_topic: bool,
        max_iter: usize,
        tol: F,
        inner_max_iter: usize,
        inner_tol: F,
        check_every: usize,
        learning: LdaLearning,
        seed: u64,
    ) -> Self {
        Self {
            alpha,
            alpha_by_topic,
            eta,
            eta_by_topic,
            max_iter,
            tol,
            inner_max_iter,
            inner_tol,
            check_every,
            learning,
            seed,
        }
    }

    /// Resolve the effective Dirichlet priors for a given topic count.
    ///
    /// ### Params
    ///
    /// * `k` - Number of topics.
    ///
    /// ### Returns
    ///
    /// The `(alpha, eta)` actually handed to the solver, or an error if either
    /// is not strictly positive.
    pub fn resolve_priors(&self, k: usize) -> Result<(F, F), BixverseErrors> {
        let k_f = F::from_usize(k).unwrap();
        let alpha = if self.alpha_by_topic {
            self.alpha / k_f
        } else {
            self.alpha
        };
        let eta = if self.eta_by_topic {
            self.eta / k_f
        } else {
            self.eta
        };

        for (name, value) in [("alpha", alpha), ("eta", eta)] {
            if value <= F::zero() || !value.is_finite() {
                return Err(BixverseErrors::LdaInvalidHyperparameter {
                    name: name.to_string(),
                    value: value.to_f64().unwrap_or(f64::NAN),
                });
            }
        }

        Ok((alpha, eta))
    }
}

impl<F: BixverseFloat> Default for LdaParams<F> {
    fn default() -> Self {
        Self {
            alpha: F::from_f64(50.0).unwrap(),
            alpha_by_topic: true,
            eta: F::from_f64(0.1).unwrap(),
            eta_by_topic: false,
            max_iter: 150,
            tol: F::from_f64(1e-3).unwrap(),
            inner_max_iter: 100,
            inner_tol: F::from_f64(1e-3).unwrap(),
            check_every: 10,
            learning: LdaLearning::default(),
            seed: 555,
        }
    }
}

/////////////
// Results //
/////////////

/// A fitted LDA model.
#[derive(Clone, Debug)]
pub struct LdaResult<F: BixverseFloat> {
    /// Topic proportions per document, shape `(k, n_docs)`. Columns sum to one,
    /// so a document's topic vector is one contiguous column.
    pub cell_topic: Mat<F>,
    /// Term probabilities per topic, shape `(n_terms, k)`. Columns sum to one,
    /// so a topic's term vector is one contiguous column.
    pub topic_region: Mat<F>,
    /// Final variational bound (ELBO). Higher is better.
    pub bound: F,
    /// Per-token perplexity, `exp(-bound / total_tokens)`. Lower is better.
    pub perplexity: F,
    /// Outer iterations actually run.
    pub n_iter: usize,
    /// Whether the relative bound change fell below `tol`.
    pub converged: bool,
}

/// One fitted model plus its evaluation, as produced by a topic-count sweep.
#[derive(Clone, Debug)]
pub struct LdaSweepEntry<F: BixverseFloat> {
    /// Topic count this entry was fitted with.
    pub k: usize,
    /// The fitted model.
    pub model: LdaResult<F>,
    /// Model selection metrics for it.
    pub metrics: LdaMetrics<F>,
}

/// The outcome of a topic-count sweep.
#[derive(Clone, Debug)]
pub struct LdaSweepResult<F: BixverseFloat> {
    /// One entry per requested topic count, in the order requested.
    pub entries: Vec<LdaSweepEntry<F>>,
    /// Combined rescaled score per entry. `NaN` for entries excluded from
    /// selection by the coherence topic-count floor.
    pub combined_score: Vec<F>,
    /// Topic count with the highest combined score.
    pub best_k: usize,
}

////////////
// Corpus //
////////////

/// A corpus held in both orientations, with counts cast to the solver's float.
///
/// The E-step walks documents and the M-step walks terms, so keeping both
/// layouts turns each into a contiguous scan. Cast once here rather than per
/// touch, since the solver revisits every non-zero on every iteration.
pub struct LdaCorpus<F: BixverseFloat> {
    /// Documents x terms, CSR. Drives the E-step.
    pub(crate) csr: CompressedSparseData2<F>,
    /// Documents x terms, CSC. Drives the M-step.
    pub(crate) csc: CompressedSparseData2<F>,
    /// Number of documents.
    pub(crate) n_docs: usize,
    /// Number of terms.
    pub(crate) n_terms: usize,
    /// Total count mass, used to turn the bound into a perplexity.
    pub(crate) total_tokens: f64,
    /// Per-document count mass, needed by the Arun metric.
    pub(crate) doc_lengths: Vec<F>,
}

impl<F: BixverseFloat + BixverseNumeric> LdaCorpus<F> {
    /// Build a corpus from a documents x terms sparse matrix.
    ///
    /// Accepts either orientation of compressed storage and re-expresses it,
    /// but the logical shape must already be documents x terms. A caller
    /// holding terms x documents wants
    /// [CompressedSparseData2::transpose_and_convert] first, which is a clone
    /// and a relabel when the input is CSC.
    ///
    /// ### Params
    ///
    /// * `matrix` - Documents x terms counts. Negative entries are rejected.
    ///
    /// ### Returns
    ///
    /// The [LdaCorpus], or an error if the matrix is empty or holds a negative
    /// or non-finite count.
    pub fn new<T, U>(matrix: &CompressedSparseData2<T, U>) -> Result<Self, BixverseErrors>
    where
        T: BixverseNumeric + ToPrimitive,
        U: BixverseNumeric,
    {
        let (n_docs, n_terms) = matrix.shape();
        if matrix.get_nnz() == 0 || n_docs == 0 || n_terms == 0 {
            return Err(BixverseErrors::LdaEmptyMatrix);
        }

        // Cast once, then derive the missing orientation, so the counting sort
        // runs on the narrower of the two buffers only.
        let cast = cast_counts::<T, U, F>(matrix)?;
        let (csr, csc) = match cast.cs_type {
            CompressedSparseFormat::Csr => {
                let csc = cast.transform();
                (cast, csc)
            }
            CompressedSparseFormat::Csc => {
                let csr = cast.transform();
                (csr, cast)
            }
        };

        let total_tokens = csr.data.iter().map(|v| v.to_f64().unwrap_or(0.0)).sum();
        let doc_lengths = (0..n_docs)
            .map(|d| {
                let start = csr.indptr[d] as usize;
                let end = csr.indptr[d + 1] as usize;
                csr.data[start..end].iter().fold(F::zero(), |a, b| a + *b)
            })
            .collect();

        Ok(Self {
            csr,
            csc,
            n_docs,
            n_terms,
            total_tokens,
            doc_lengths,
        })
    }
}

/// Cast a sparse matrix's primary counts into the solver's float type.
///
/// The second data layer, if any, is dropped: LDA reads counts and nothing
/// else.
///
/// ### Params
///
/// * `matrix` - Source matrix, whose layout is carried over unchanged.
///
/// ### Returns
///
/// The same matrix with `F` data, or an error on a negative or non-finite
/// count.
fn cast_counts<T, U, F>(
    matrix: &CompressedSparseData2<T, U>,
) -> Result<CompressedSparseData2<F>, BixverseErrors>
where
    T: BixverseNumeric + ToPrimitive,
    U: BixverseNumeric,
    F: BixverseFloat + BixverseNumeric,
{
    let mut data = Vec::with_capacity(matrix.data.len());
    for v in &matrix.data {
        let f = F::from_f64(v.to_f64().ok_or(BixverseErrors::LdaNonFinite)?)
            .ok_or(BixverseErrors::LdaNonFinite)?;
        if !f.is_finite() || f < F::zero() {
            return Err(BixverseErrors::LdaNonFinite);
        }
        data.push(f);
    }

    Ok(CompressedSparseData2::from_parts(
        data,
        matrix.indices.clone(),
        matrix.indptr.clone(),
        None,
        matrix.cs_type,
        matrix.shape,
    ))
}

//////////
// Main //
//////////

/// Fit an LDA model to a documents x terms count matrix.
///
/// Variational Bayes throughout, see the module documentation for the layout
/// and the deviation from the collapsed Gibbs sampler cisTopic uses.
///
/// ### Params
///
/// * `matrix` - Documents x terms counts, either compressed orientation. For a
///   cisTopic-style run this is the binarised cells x regions matrix.
/// * `k` - Number of topics.
/// * `params` - Solver options, `None` for [LdaParams::default].
/// * `verbose` - Verbosity level, parsed by [parse_verbosity_level].
///
/// ### Returns
///
/// The fitted [LdaResult].
pub fn lda_fit<T, U, F>(
    matrix: &CompressedSparseData2<T, U>,
    k: usize,
    params: Option<LdaParams<F>>,
    verbose: usize,
) -> Result<LdaResult<F>, BixverseErrors>
where
    T: BixverseNumeric + ToPrimitive,
    U: BixverseNumeric,
    F: BixverseFloat + BixverseNumeric + BixverseSimd,
{
    let params = params.unwrap_or_default();
    let corpus = LdaCorpus::new(matrix)?;
    fit_vb(&corpus, k, &params, parse_verbosity_level(verbose))
}

/// Fit LDA across a range of topic counts and score each fit.
///
/// Each fit is already parallel over documents, so the sweep runs the topic
/// counts sequentially rather than nesting `rayon` pools. The corpus is built
/// once and shared.
///
/// ### Params
///
/// * `matrix` - Documents x terms counts, as for [lda_fit].
/// * `ks` - Topic counts to try.
/// * `params` - Solver options, `None` for [LdaParams::default].
/// * `top_topics_coh` - Top-scoring topics averaged into the reported
///   coherence, `None` for [DEFAULT_TOP_TOPICS_COH].
/// * `verbose` - Verbosity level, parsed by [parse_verbosity_level].
///
/// ### Returns
///
/// The [LdaSweepResult], carrying every fit, its metrics, and the selected
/// topic count.
pub fn lda_k_sweep<T, U, F>(
    matrix: &CompressedSparseData2<T, U>,
    ks: &[usize],
    params: Option<LdaParams<F>>,
    top_topics_coh: Option<usize>,
    verbose: usize,
) -> Result<LdaSweepResult<F>, BixverseErrors>
where
    T: BixverseNumeric + ToPrimitive,
    U: BixverseNumeric,
    F: BixverseFloat + BixverseNumeric + BixverseSimd,
{
    if ks.is_empty() {
        return Err(BixverseErrors::LdaInvalidTopicCount {
            requested: 0,
            max_available: 0,
        });
    }

    let params = params.unwrap_or_default();
    let verbosity = parse_verbosity_level(verbose);
    let top_topics_coh = top_topics_coh.unwrap_or(DEFAULT_TOP_TOPICS_COH);
    let corpus = LdaCorpus::new(matrix)?;

    let mut entries = Vec::with_capacity(ks.len());
    for &k in ks {
        if verbosity.normal_verbosity() {
            println!("LDA: fitting k = {k}");
        }
        let model = fit_vb(&corpus, k, &params, verbosity)?;
        let metrics = lda_metrics(&corpus, &model, top_topics_coh)?;
        entries.push(LdaSweepEntry { k, model, metrics });
    }

    let (combined_score, best_k) = metrics::select_best_k(&entries, DEFAULT_MIN_TOPICS_COH);

    Ok(LdaSweepResult {
        entries,
        combined_score,
        best_k,
    })
}
