//! Model selection metrics for LDA, and the combination that picks a topic
//! count.
//!
//! The same four pycisTopic reports, with one substitution. Its fourth metric
//! is a Griffiths and Steyvers joint log-likelihood whose implementation
//! assigns rather than accumulates in both inner loops, so it returns a value
//! derived from the last topic and last document only, and it is additionally
//! called with the unscaled `alpha` rather than the `alpha / k` the model was
//! fitted with. We report the variational bound instead, which is the quantity
//! this solver actually maximises.
//!
//! ### References
//!
//! Arun et al., On Finding the Natural Number of Topics with Latent Dirichlet
//! Allocation, PAKDD, 2010
//!
//! Cao et al., A density-based method for adaptive LDA model selection,
//! Neurocomputing, 2009
//!
//! Mimno et al., Optimizing Semantic Coherence in Topic Models, EMNLP, 2011

use faer::{Accum, Mat, linalg::matmul::matmul};
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::faer_parallelism;

use super::{LdaCorpus, LdaResult, LdaSweepEntry};

////////////
// Consts //
////////////

/// Top terms per topic entering the coherence sum.
///
/// Mimno's own choice, and what pycisTopic passes. The pair count grows as the
/// square, so this is already 190 co-document lookups per topic.
const COHERENCE_TOP_N: usize = 20;

/// Smoothing added to the co-document count in the coherence sum.
///
/// Keeps a pair that never co-occurs from taking the logarithm of zero. Small
/// enough that such a pair still dominates the sum, which is the point: the
/// metric is meant to punish it.
const COHERENCE_EPS: f64 = 1e-12;

/// Floor applied to both distributions before the Arun symmetric KL.
///
/// A singular value or a topic mass can round to zero, and the divergence takes
/// the logarithm of both vectors and their ratio.
const ARUN_FLOOR: f64 = 1e-12;

/////////////
// Results //
/////////////

/// Model selection metrics for one fitted model.
#[derive(Clone, Debug)]
pub struct LdaMetrics<F: BixverseFloat> {
    /// Arun 2010 symmetric KL. Lower is better.
    pub arun_2010: F,
    /// Cao Juan 2009 mean pairwise topic cosine similarity. Lower is better.
    pub cao_juan_2009: F,
    /// Mean coherence of the highest-scoring topics. Higher is better.
    pub mimno_2011: F,
    /// Mimno UMass coherence per topic, in topic order.
    pub coherence_per_topic: Vec<F>,
    /// The model's variational bound. Higher is better.
    pub bound: F,
    /// The model's per-token perplexity. Lower is better.
    pub perplexity: F,
}

/////////////
// Metrics //
/////////////

/// Cao Juan 2009: mean pairwise cosine similarity between topics.
///
/// Topics that duplicate one another push this up, so a topic count that has
/// started splitting a single signal scores worse. One GEMM of the
/// L2-normalised `n_terms x k` matrix against itself gives every pair at once.
///
/// ### Params
///
/// * `topic_region` - `n_terms x k` topic-term probabilities, columns summing
///   to one.
///
/// ### Returns
///
/// The mean cosine similarity over the `k (k - 1) / 2` distinct topic pairs, or
/// zero when there is only one topic.
pub fn cao_juan_2009<F>(topic_region: &Mat<F>) -> F
where
    F: BixverseFloat + Send + Sync,
{
    let n_terms = topic_region.nrows();
    let k = topic_region.ncols();
    if k < 2 {
        return F::zero();
    }

    let mut normalised = topic_region.cloned();
    for topic in 0..k {
        let norm = (0..n_terms)
            .map(|w| normalised[(w, topic)] * normalised[(w, topic)])
            .fold(F::zero(), |a, b| a + b)
            .sqrt();
        if norm > F::zero() {
            for w in 0..n_terms {
                normalised[(w, topic)] /= norm;
            }
        }
    }

    let mut gram = Mat::<F>::zeros(k, k);
    matmul(
        gram.as_mut(),
        Accum::Replace,
        normalised.as_ref().transpose(),
        normalised.as_ref(),
        F::one(),
        faer_parallelism(),
    );

    let mut total = F::zero();
    for i in 0..k {
        for j in (i + 1)..k {
            total += gram[(i, j)];
        }
    }
    total / F::from_usize(k * (k - 1) / 2).unwrap()
}

/// Arun 2010: symmetric KL between the topic-term spectrum and the
/// length-weighted topic mass.
///
/// `cm1` is the singular value spectrum of the topic-term matrix,
/// sum-normalised; `cm2` is the document lengths projected onto the topic
/// proportions, sorted descending and sum-normalised. The two agree when the
/// topic count matches the rank the corpus actually supports and diverge either
/// side of it.
///
/// The thin SVD runs on the `n_terms x k` matrix directly rather than on the
/// `k x k` Gram, which would halve the precision of every singular value before
/// the logarithms see them.
///
/// Note this follows the tmtoolkit formulation pycisTopic uses, which
/// sum-normalises both vectors and sorts `cm2` descending. The R `ldatuning`
/// implementation differs.
///
/// ### Params
///
/// * `topic_region` - `n_terms x k` topic-term probabilities.
/// * `cell_topic` - `k x n_docs` topic proportions, columns summing to one.
/// * `doc_lengths` - Per-document count mass, length `n_docs`.
///
/// ### Returns
///
/// The symmetric KL divergence, or an error if the SVD fails or the dimensions
/// disagree.
pub fn arun_2010<F>(
    topic_region: &Mat<F>,
    cell_topic: &Mat<F>,
    doc_lengths: &[F],
) -> Result<F, BixverseErrors>
where
    F: BixverseFloat + BixverseSimd + Send + Sync,
{
    let k = topic_region.ncols();
    let n_docs = cell_topic.ncols();
    if cell_topic.nrows() != k {
        return Err(BixverseErrors::LdaDimensionMismatch {
            expected: k,
            got: cell_topic.nrows(),
        });
    }
    if doc_lengths.len() != n_docs {
        return Err(BixverseErrors::LdaDimensionMismatch {
            expected: n_docs,
            got: doc_lengths.len(),
        });
    }

    let svd = topic_region
        .thin_svd()
        .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;
    let mut cm1: Vec<f64> = svd
        .S()
        .column_vector()
        .iter()
        .map(|v| v.to_f64().unwrap_or(0.0).max(ARUN_FLOOR))
        .collect();

    // cm2 = doc_lengths @ cell_topic^T. cell_topic is k x n_docs, so walking
    // documents keeps the accumulation on contiguous columns.
    let mut cm2_f = vec![F::zero(); k];
    for d in 0..n_docs {
        let col = cell_topic.col_as_slice(d);
        F::bxv_axpy_simd(&mut cm2_f, doc_lengths[d], col);
    }
    let mut cm2: Vec<f64> = cm2_f
        .iter()
        .map(|v| v.to_f64().unwrap_or(0.0).max(ARUN_FLOOR))
        .collect();
    cm2.sort_by(|a, b| b.total_cmp(a));

    normalise_to_sum(&mut cm1);
    normalise_to_sum(&mut cm2);

    let divergence: f64 = cm1
        .iter()
        .zip(&cm2)
        .map(|(a, b)| a * (a / b).ln() + b * (b / a).ln())
        .sum();

    Ok(F::from_f64(divergence).unwrap_or(F::infinity()))
}

/// Scale a vector to sum to one.
///
/// ### Params
///
/// * `v` - Vector, normalised in place. A zero-sum vector is left alone.
fn normalise_to_sum(v: &mut [f64]) {
    let total: f64 = v.iter().sum();
    if total > 0.0 {
        v.iter_mut().for_each(|x| *x /= total);
    }
}

/// Mimno 2011 UMass coherence, per topic.
///
/// For each topic, takes the `top_n` highest-probability terms and sums
/// `log((D(v_m, v_l) + eps) / D(v_l))` over ordered pairs, where `D` counts
/// documents. The denominator is always the more probable term of the pair.
/// Normalised by the pair count so topics stay comparable.
///
/// Document frequencies come off the term-major view, where `D(v)` is a column
/// length and a co-document count is a merge intersection of two sorted index
/// runs.
///
/// ### Params
///
/// * `corpus` - The corpus, read through its CSC view.
/// * `topic_region` - `n_terms x k` topic-term probabilities.
/// * `top_n` - Terms per topic entering the sum.
///
/// ### Returns
///
/// One coherence value per topic, in topic order.
pub fn mimno_2011_coherence<F>(
    corpus: &LdaCorpus<F>,
    topic_region: &Mat<F>,
    top_n: usize,
) -> Result<Vec<F>, BixverseErrors>
where
    F: BixverseFloat + BixverseNumeric + Send + Sync,
{
    let n_terms = corpus.n_terms;
    if top_n > n_terms {
        return Err(BixverseErrors::LdaTopNTooLarge {
            requested: top_n,
            vocab_size: n_terms,
        });
    }
    if topic_region.nrows() != n_terms {
        return Err(BixverseErrors::LdaDimensionMismatch {
            expected: n_terms,
            got: topic_region.nrows(),
        });
    }

    let csc = &corpus.csc;
    let pair_norm = 2.0 / (top_n * (top_n - 1)) as f64;

    Ok((0..topic_region.ncols())
        .into_par_iter()
        .map(|topic| {
            let col = topic_region.col_as_slice(topic);
            let mut ranked: Vec<usize> = (0..n_terms).collect();
            ranked.sort_by(|&a, &b| col[b].total_cmp(&col[a]));
            let top: Vec<usize> = ranked.into_iter().take(top_n).collect();

            let docs: Vec<&[u32]> = top
                .iter()
                .map(|&w| {
                    let start = csc.indptr[w] as usize;
                    let end = csc.indptr[w + 1] as usize;
                    &csc.indices[start..end]
                })
                .collect();

            let mut total = 0.0_f64;
            for m in 1..top_n {
                for l in 0..m {
                    let co = sorted_intersection_len(docs[m], docs[l]) as f64;
                    let df = docs[l].len() as f64;
                    if df > 0.0 {
                        total += ((co + COHERENCE_EPS) / df).ln();
                    }
                }
            }
            F::from_f64(total * pair_norm).unwrap_or(F::zero())
        })
        .collect())
}

/// Size of the intersection of two ascending index runs.
///
/// ### Params
///
/// * `a` - First run, ascending.
/// * `b` - Second run, ascending.
///
/// ### Returns
///
/// How many indices appear in both.
fn sorted_intersection_len(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut count) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

/// Score a fitted model on every metric.
///
/// ### Params
///
/// * `corpus` - The corpus the model was fitted on.
/// * `model` - The fitted model.
/// * `top_topics_coh` - Highest-scoring topics averaged into the reported
///   coherence. Falls back to every topic when the model has no more than this
///   many.
///
/// ### Returns
///
/// The [LdaMetrics] for the model.
pub fn lda_metrics<F>(
    corpus: &LdaCorpus<F>,
    model: &LdaResult<F>,
    top_topics_coh: usize,
) -> Result<LdaMetrics<F>, BixverseErrors>
where
    F: BixverseFloat + BixverseNumeric + BixverseSimd + Send + Sync,
{
    let top_n = COHERENCE_TOP_N.min(corpus.n_terms);
    let coherence_per_topic = mimno_2011_coherence(corpus, &model.topic_region, top_n)?;

    let mut sorted = coherence_per_topic.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let take = top_topics_coh.clamp(1, sorted.len());
    let mimno_2011 =
        sorted[..take].iter().fold(F::zero(), |a, b| a + *b) / F::from_usize(take).unwrap();

    Ok(LdaMetrics {
        arun_2010: arun_2010(&model.topic_region, &model.cell_topic, &corpus.doc_lengths)?,
        cao_juan_2009: cao_juan_2009(&model.topic_region),
        mimno_2011,
        coherence_per_topic,
        bound: model.bound,
        perplexity: model.perplexity,
    })
}

///////////////
// Selection //
///////////////

/// Min-max rescale a vector into `[0, 1]`.
///
/// A constant vector maps to all zeros, which drops it out of the sum rather
/// than letting an arbitrary ordering decide the winner.
///
/// ### Params
///
/// * `values` - Values to rescale.
///
/// ### Returns
///
/// The rescaled values.
fn rescale(values: &[f64]) -> Vec<f64> {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if !range.is_finite() || range <= 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| (v - min) / range).collect()
}

/// Combine the metrics across a topic-count sweep and pick a winner.
///
/// Arun and Cao Juan are negated so every metric points the same way, all four
/// are min-max rescaled across the candidates, and the unweighted sum is
/// maximised. Candidates below `min_topics_coh` are excluded, because coherence
/// saturates on very small topic counts and would always take them.
///
/// One deliberate deviation from pycisTopic: it rescales Arun, Cao Juan and its
/// log-likelihood over the full sweep while scoring only the surviving subset,
/// and rescales coherence over the subset alone. That asymmetry changes which
/// model wins. Here the subset is taken first and everything is rescaled over
/// it.
///
/// ### Params
///
/// * `entries` - The sweep, one entry per topic count.
/// * `min_topics_coh` - Smallest topic count eligible to win.
///
/// ### Returns
///
/// The combined score per entry, `NaN` where an entry was excluded, and the
/// winning topic count.
pub fn select_best_k<F>(entries: &[LdaSweepEntry<F>], min_topics_coh: usize) -> (Vec<F>, usize)
where
    F: BixverseFloat,
{
    let eligible: Vec<usize> = (0..entries.len())
        .filter(|&i| entries[i].k >= min_topics_coh)
        .collect();
    // Every candidate below the floor means the floor cannot be honoured;
    // scoring them all beats returning nothing.
    let eligible = if eligible.is_empty() {
        (0..entries.len()).collect()
    } else {
        eligible
    };

    let pull = |f: &dyn Fn(&LdaSweepEntry<F>) -> f64| -> Vec<f64> {
        eligible.iter().map(|&i| f(&entries[i])).collect()
    };

    let arun = rescale(&pull(&|e| -e.metrics.arun_2010.to_f64().unwrap_or(0.0)));
    let cao = rescale(&pull(&|e| -e.metrics.cao_juan_2009.to_f64().unwrap_or(0.0)));
    let coh = rescale(&pull(&|e| e.metrics.mimno_2011.to_f64().unwrap_or(0.0)));
    let bound = rescale(&pull(&|e| e.metrics.bound.to_f64().unwrap_or(0.0)));

    let mut combined = vec![F::nan(); entries.len()];
    let mut best = (f64::NEG_INFINITY, entries[0].k);
    for (slot, &i) in eligible.iter().enumerate() {
        let score = arun[slot] + cao[slot] + coh[slot] + bound[slot];
        combined[i] = F::from_f64(score).unwrap_or(F::nan());
        if score > best.0 {
            best = (score, entries[i].k);
        }
    }

    (combined, best.1)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::lda::vb::tests::{block_corpus, csr_from_rows};
    use crate::methods::lda::{DEFAULT_MIN_TOPICS_COH, LdaParams, lda_fit, lda_k_sweep};
    use approx::assert_relative_eq;

    /// Build a `n_terms x k` topic-term matrix from per-topic columns.
    ///
    /// ### Params
    ///
    /// * `cols` - One column per topic, each of length `n_terms`.
    ///
    /// ### Returns
    ///
    /// The dense matrix.
    fn topic_matrix(cols: &[Vec<f64>]) -> Mat<f64> {
        Mat::from_fn(cols[0].len(), cols.len(), |w, t| cols[t][w])
    }

    /// Shared solver options for the sweep tests.
    ///
    /// ### Returns
    ///
    /// The [LdaParams] used below.
    fn test_params() -> LdaParams<f64> {
        LdaParams {
            alpha: 0.1,
            alpha_by_topic: false,
            max_iter: 40,
            check_every: 1,
            tol: 1e-9,
            seed: 42,
            ..Default::default()
        }
    }

    /// Disjoint topics are orthogonal, so every pairwise cosine is zero.
    #[test]
    fn test_cao_juan_orthogonal_topics() {
        let m = topic_matrix(&[vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.0, 0.5, 0.5]]);
        assert_relative_eq!(cao_juan_2009(&m), 0.0, epsilon = 1e-12);
    }

    /// Identical topics are perfectly correlated, so every cosine is one.
    #[test]
    fn test_cao_juan_identical_topics() {
        let col = vec![0.4, 0.3, 0.2, 0.1];
        let m = topic_matrix(&[col.clone(), col.clone(), col]);
        assert_relative_eq!(cao_juan_2009(&m), 1.0, epsilon = 1e-12);
    }

    /// A single topic has no pairs to average over.
    #[test]
    fn test_cao_juan_single_topic() {
        let m = topic_matrix(&[vec![0.6, 0.4]]);
        assert_eq!(cao_juan_2009(&m), 0.0);
    }

    /// Half-overlapping topics: the closed form is the shared mass over the
    /// product of the norms.
    #[test]
    fn test_cao_juan_partial_overlap() {
        let m = topic_matrix(&[vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.5, 0.5, 0.0]]);
        // cos = 0.25 / (sqrt(0.5) * sqrt(0.5)) = 0.5
        assert_relative_eq!(cao_juan_2009(&m), 0.5, epsilon = 1e-12);
    }

    /// Two topics whose spectra match the length-weighted topic mass exactly
    /// give a divergence of zero.
    #[test]
    fn test_arun_is_zero_when_spectra_agree() {
        // Orthogonal topics of equal mass give equal singular values, and two
        // documents of equal length split evenly across topics match them.
        let topic_region = topic_matrix(&[vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.0, 0.5, 0.5]]);
        let cell_topic = Mat::from_fn(2, 2, |t, d| if t == d { 1.0 } else { 0.0 });
        let doc_lengths = vec![1.0, 1.0];
        let arun = arun_2010(&topic_region, &cell_topic, &doc_lengths).unwrap();
        assert_relative_eq!(arun, 0.0, epsilon = 1e-10);
    }

    /// Divergence is non-negative and grows once the two vectors disagree.
    #[test]
    fn test_arun_grows_with_imbalance() {
        let topic_region = topic_matrix(&[vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.0, 0.5, 0.5]]);
        let balanced = Mat::from_fn(2, 2, |t, d| if t == d { 1.0 } else { 0.0 });
        let lengths_even = vec![1.0, 1.0];
        let lengths_skewed = vec![50.0, 1.0];

        let even = arun_2010(&topic_region, &balanced, &lengths_even).unwrap();
        let skewed = arun_2010(&topic_region, &balanced, &lengths_skewed).unwrap();
        assert!(skewed > even, "skewed {skewed} did not exceed even {even}");
        assert!(even >= 0.0);
    }

    /// Dimension checks fire rather than reading past the end.
    #[test]
    fn test_arun_dimension_mismatch() {
        let topic_region = topic_matrix(&[vec![0.5, 0.5], vec![0.5, 0.5]]);
        let cell_topic = Mat::from_fn(3, 2, |_, _| 0.5);
        assert!(matches!(
            arun_2010(&topic_region, &cell_topic, &[1.0, 1.0]),
            Err(BixverseErrors::LdaDimensionMismatch { .. })
        ));

        let cell_topic = Mat::from_fn(2, 2, |_, _| 0.5);
        assert!(matches!(
            arun_2010(&topic_region, &cell_topic, &[1.0]),
            Err(BixverseErrors::LdaDimensionMismatch { .. })
        ));
    }

    /// Terms that always co-occur give the maximum coherence of zero, since
    /// every co-document count equals the document frequency.
    #[test]
    fn test_coherence_perfect_co_occurrence() {
        // Both terms appear in both documents, so D(a, b) = D(b) and log 1 = 0.
        let matrix = csr_from_rows(&[vec![1.0, 1.0], vec![1.0, 1.0]]);
        let corpus = LdaCorpus::<f64>::new(&matrix).unwrap();
        let topic_region = topic_matrix(&[vec![0.5, 0.5]]);

        let coh = mimno_2011_coherence(&corpus, &topic_region, 2).unwrap();
        assert_relative_eq!(coh[0], 0.0, epsilon = 1e-9);
    }

    /// Terms that never co-occur are punished hard: the log of the smoothing
    /// constant over the document frequency.
    #[test]
    fn test_coherence_never_co_occurring() {
        let matrix = csr_from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let corpus = LdaCorpus::<f64>::new(&matrix).unwrap();
        let topic_region = topic_matrix(&[vec![0.5, 0.5]]);

        let coh = mimno_2011_coherence(&corpus, &topic_region, 2).unwrap();
        // One pair, D(a, b) = 0, D(b) = 1, normalised by 2 / (2 * 1) = 1.
        assert_relative_eq!(coh[0], (COHERENCE_EPS / 1.0).ln(), epsilon = 1e-9);
    }

    /// A coherent topic scores above an incoherent one on the same corpus.
    #[test]
    fn test_coherence_orders_topics() {
        // Terms 0 and 1 always co-occur; terms 2 and 3 never do.
        let matrix = csr_from_rows(&[
            vec![1.0, 1.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0, 0.0],
        ]);
        let corpus = LdaCorpus::<f64>::new(&matrix).unwrap();
        let topic_region = topic_matrix(&[vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.0, 0.5, 0.5]]);

        let coh = mimno_2011_coherence(&corpus, &topic_region, 2).unwrap();
        assert!(
            coh[0] > coh[1],
            "co-occurring topic {} scored below the disjoint one {}",
            coh[0],
            coh[1]
        );
    }

    /// Asking for more top terms than the vocabulary holds is an error.
    #[test]
    fn test_coherence_top_n_too_large() {
        let matrix = csr_from_rows(&[vec![1.0, 1.0]]);
        let corpus = LdaCorpus::<f64>::new(&matrix).unwrap();
        let topic_region = topic_matrix(&[vec![0.5, 0.5]]);
        assert!(matches!(
            mimno_2011_coherence(&corpus, &topic_region, 5),
            Err(BixverseErrors::LdaTopNTooLarge { .. })
        ));
    }

    /// Rescaling maps the extremes to zero and one, and a flat vector to zeros.
    #[test]
    fn test_rescale() {
        let r = rescale(&[2.0, 4.0, 6.0]);
        assert_relative_eq!(r[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(r[1], 0.5, epsilon = 1e-12);
        assert_relative_eq!(r[2], 1.0, epsilon = 1e-12);

        assert_eq!(rescale(&[3.0, 3.0, 3.0]), vec![0.0, 0.0, 0.0]);
    }

    /// The reported coherence averages the best topics, not all of them.
    #[test]
    fn test_metrics_average_top_topics_only() {
        let matrix = block_corpus(6, 5);
        let model = lda_fit(&matrix, 3, Some(test_params()), 0).unwrap();
        let corpus = LdaCorpus::<f64>::new(&matrix).unwrap();

        let all = lda_metrics(&corpus, &model, 3).unwrap();
        let best_one = lda_metrics(&corpus, &model, 1).unwrap();
        assert!(
            best_one.mimno_2011 >= all.mimno_2011,
            "top-1 coherence {} below the all-topic mean {}",
            best_one.mimno_2011,
            all.mimno_2011
        );
        assert_eq!(all.coherence_per_topic.len(), 3);
    }

    /// A sweep returns one scored entry per topic count and picks one of them.
    #[test]
    fn test_k_sweep_scores_every_entry() {
        let matrix = block_corpus(8, 6);
        let ks = [2, 3, 4, 5];
        let sweep = lda_k_sweep(&matrix, &ks, Some(test_params()), Some(2), 0).unwrap();

        assert_eq!(sweep.entries.len(), ks.len());
        assert_eq!(sweep.combined_score.len(), ks.len());
        for (entry, &k) in sweep.entries.iter().zip(&ks) {
            assert_eq!(entry.k, k);
            assert!(entry.metrics.arun_2010.is_finite());
            assert!(entry.metrics.cao_juan_2009.is_finite());
            assert!(entry.metrics.bound.is_finite());
        }
        assert!(ks.contains(&sweep.best_k));
    }

    /// The topic-count floor keeps small models out of the selection, unless
    /// every candidate is below it.
    #[test]
    fn test_selection_respects_topic_floor() {
        let matrix = block_corpus(8, 6);
        let ks = [2, 3, 5, 6];
        let sweep = lda_k_sweep(&matrix, &ks, Some(test_params()), Some(2), 0).unwrap();

        assert!(sweep.combined_score[0].is_nan(), "k = 2 was scored");
        assert!(sweep.combined_score[1].is_nan(), "k = 3 was scored");
        assert!(sweep.combined_score[2].is_finite());
        assert!(sweep.best_k >= DEFAULT_MIN_TOPICS_COH);

        // With nothing above the floor, everything is scored instead.
        let small = lda_k_sweep(&matrix, &[2, 3], Some(test_params()), Some(2), 0).unwrap();
        assert!(small.combined_score.iter().all(|v| v.is_finite()));
        assert!([2, 3].contains(&small.best_k));
    }

    /// An empty topic-count list is rejected rather than silently returning
    /// nothing.
    #[test]
    fn test_k_sweep_rejects_empty_ks() {
        let matrix = block_corpus(4, 4);
        assert!(matches!(
            lda_k_sweep(&matrix, &[], Some(test_params()), None, 0),
            Err(BixverseErrors::LdaInvalidTopicCount { .. })
        ));
    }
}
