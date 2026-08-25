//! End-to-end parity checks against scikit-learn's `LatentDirichletAllocation`.
//!
//! Two separate claims, and the distinction is the point of this file.
//!
//! `test_bound_matches_sklearn_on_shared_params` pins the *objective*: it runs
//! scikit-learn's own fitted `lambda` and `gamma` through `lda_bound` and
//! requires the same number back. Fitting twice and comparing would only show
//! that both solvers reached some optimum, whereas this isolates the formula
//! from the solver.
//!
//! `test_fit_reaches_sklearn_optimum` then pins the *solver*, one-sidedly: with
//! the objective already established, all that is left is that this crate does
//! at least as well. On this corpus it does better, because scikit-learn settles
//! into a local optimum where two topics absorb the overlap terms unevenly.
//! Asserting the fitted distributions match would be asserting that we reproduce
//! that local optimum.
//!
//! ### Regenerating the fixture
//!
//! Run, under `uv run --with numpy --with scipy --with scikit-learn python`,
//! `LatentDirichletAllocation(n_components=4, doc_topic_prior=0.1,
//! topic_word_prior=0.1, learning_method="batch", max_iter=200,
//! mean_change_tol=1e-5, max_doc_update_iter=200, evaluate_every=-1,
//! random_state=0)` on `sklearn_corpus`, then dump `components_`,
//! `_unnormalized_transform(X)` and `score(X)`. Emit the floats with Python
//! `repr()`, the shortest round-tripping form; `%.17g` trips clippy's
//! `excessive_precision`.

use approx::assert_relative_eq;
use faer::Mat;

use bixverse_rs::methods::lda::{LdaCorpus, LdaLearning, LdaParams, lda_bound, lda_fit};
use bixverse_rs::prelude::*;

/// Documents in the reference corpus.
const SKLEARN_N_DOCS: usize = 60;

/// Vocabulary size of the reference corpus.
const SKLEARN_N_TERMS: usize = 40;

/// Topics the reference model was fitted with.
const SKLEARN_K: usize = 4;

/// Both Dirichlet priors of the reference fit.
const SKLEARN_PRIOR: f64 = 0.1;

/// The bound scikit-learn reports for the parameters below.
const SKLEARN_BOUND: f64 = -1899.5186748694439;

/// Reference corpus: four term blocks, each document drawn from one block with
/// a few terms bleeding in from the next so the topics are not trivially
/// separable.
///
/// ### Returns
///
/// A binary documents x terms CSR matrix.
fn sklearn_corpus() -> CompressedSparseData2<f64> {
    let block = SKLEARN_N_TERMS / SKLEARN_K;
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0u32];
    for d in 0..SKLEARN_N_DOCS {
        let mut row = vec![0.0f64; SKLEARN_N_TERMS];
        let primary = d % SKLEARN_K;
        for t in 0..block {
            if (t + d) % block != 0 {
                row[primary * block + t] = 1.0;
            }
        }
        let next = (primary + 1) % SKLEARN_K;
        for t in 0..block {
            if (t + d * 3) % block == 0 {
                row[next * block + t] = 1.0;
            }
        }
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
        (SKLEARN_N_DOCS, SKLEARN_N_TERMS),
    )
}

/// scikit-learn's fitted `lambda`, flattened column-major over the
/// `k x n_terms` layout this crate uses.
#[rustfmt::skip]
const SKLEARN_LAMBDA: [f64; 160] = [
    0.10000359377886044, 0.10000000009604207, 0.10000000009606966, 12.09999640602899,
    0.1000038076952954, 0.10000000020193706, 3.099946934720859, 15.100049257381832,
    12.099798747793312, 0.1000000006500502, 0.1000000006502454, 0.10020125090614437,
    0.10000380769529542, 0.10000000020193706, 3.0999469347186777, 15.100049257384013,
    0.1000035937788601, 0.10000000009604207, 0.10000000009606966, 12.09999640602899,
    0.1000038076952954, 0.10000000020193706, 3.099946934724966, 15.100049257377728,
    0.10000359377886044, 0.10000000009604207, 0.10000000009606966, 12.09999640602899,
    0.1000038076952954, 0.10000000020193706, 3.0999469347176802, 15.100049257385017,
    0.10000359377886012, 0.10000000009604207, 0.10000000009606966, 12.09999640602899,
    0.10000380769529542, 0.10000000020193706, 3.099946934719373, 15.100049257383318,
    18.09996739459699, 0.10000000019792414, 0.10000000019777222, 0.10003260500724596,
    12.099999999649453, 0.10000000011018798, 0.10000000010996458, 0.10000000013035348,
    18.09996739459699, 0.10000000019792414, 0.10000000019777225, 0.1000326050072369,
    12.099999999649453, 0.10000000011018798, 0.10000000010996458, 0.10000000013035348,
    18.099967394596998, 0.10000000019792414, 0.10000000019777225, 0.10003260500723701,
    12.099999999649455, 0.10000000011018798, 0.10000000010996458, 0.10000000013035348,
    15.100026461967753, 0.10000000019842145, 0.10000000019822072, 3.0999735376355253,
    12.099999999649453, 0.10000000011018798, 0.10000000010996458, 0.10000000013035348,
    18.099967394596984, 0.10000000019792414, 0.10000000019777222, 0.10003260500724596,
    12.099999999649455, 0.10000000011018798, 0.10000000010996458, 0.10000000013035348,
    0.10000000008225862, 12.099999999709722, 0.10000000009526352, 0.10000000011271995,
    3.099947142801519, 15.100052856725295, 0.1000000002165764, 0.10000000025652725,
    0.10000000008225862, 12.099999999709722, 0.10000000009526352, 0.10000000011271995,
    3.099947142801858, 15.10005285672495, 0.1000000002165764, 0.10000000025652725,
    0.10000000008225862, 12.099999999709722, 0.10000000009526352, 0.10000000011271995,
    3.0999471427995693, 15.100052856727242, 0.1000000002165764, 0.10000000025652725,
    0.10000000008225862, 12.09999999970972, 0.10000000009526352, 0.10000000011271995,
    3.099947142801138, 15.100052856725672, 0.1000000002165764, 0.10000000025652725,
    0.10000000008225862, 12.09999999970972, 0.10000000009526352, 0.10000000011271995,
    3.099947142799886, 15.100052856726931, 0.1000000002165764, 0.10000000025652725,
    0.10000000017291226, 3.0999564008374745, 15.10004359875233, 0.10000000023720416,
    0.1000000000822386, 0.10000000009512314, 12.099999999709649, 0.10000000011295476,
    0.10000000017291226, 3.0999564008384386, 15.10004359875137, 0.10000000023720416,
    0.1000000000822386, 0.10000000009512314, 12.099999999709649, 0.10000000011295476,
    0.10000000017291226, 3.0999564008380087, 15.100043598751798, 0.10000000023720416,
    0.1000000000822386, 0.10000000009512314, 12.099999999709649, 0.10000000011295476,
    0.10000000017291226, 3.09995640083867, 15.100043598751139, 0.10000000023720416,
    0.1000000000822386, 0.10000000009512314, 12.099999999709649, 0.10000000011295476,
    0.10000000017291226, 3.0999564008379625, 15.100043598751844, 0.10000000023720416,
    0.1000000000822386, 0.10000000009512314, 12.099999999709649, 0.10000000011295476
];

/// scikit-learn's fitted `gamma`, flattened column-major over the
/// `k x n_docs` layout this crate uses.
#[rustfmt::skip]
const SKLEARN_GAMMA: [f64; 240] = [
    2.099975145027993, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553442, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893633, 0.10002052200221768,
    2.0999751450279924, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553443, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893631, 0.1000205220022177,
    0.10001261942958325, 0.10000000008296393, 0.10000233850803018, 10.09998504197939,
    10.099979221932024, 0.10002001811553442, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885065, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893633, 0.10002052200221771,
    2.099975145027993, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553442, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893631, 0.10002052200221764,
    2.099975145027993, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553443, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893631, 0.1000205220022177,
    2.099975145027993, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553442, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893633, 0.10002052200221768,
    2.0999751450279924, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553443, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893631, 0.1000205220022177,
    0.10001261942958325, 0.10000000008296393, 0.10000233850803018, 10.09998504197939,
    10.099979221932024, 0.10002001811553442, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885065, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893633, 0.10002052200221771,
    2.099975145027993, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553442, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893631, 0.10002052200221764,
    2.099975145027993, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553443, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893631, 0.1000205220022177,
    2.099975145027993, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553442, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893633, 0.10002052200221768,
    2.0999751450279924, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553443, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893631, 0.1000205220022177,
    0.10001261942958325, 0.10000000008296393, 0.10000233850803018, 10.09998504197939,
    10.099979221932024, 0.10002001811553442, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885065, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893633, 0.10002052200221771,
    2.099975145027993, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553442, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893631, 0.10002052200221764,
    2.099975145027993, 0.10000000014875209, 0.10000295333116892, 8.10002190149203,
    10.099979221932024, 0.10002001811553443, 0.10000000010875125, 0.1000007598436498,
    0.1000023971191676, 10.099980297145379, 0.10001730561885064, 0.10000000011656361,
    0.10000000008505466, 0.10000277201905794, 10.099976705893631, 0.1000205220022177
];

///////////
// Tests //
///////////

/// The bound must agree with scikit-learn's when both are evaluated on the same
/// parameters. This is the check that pins the ELBO formula down.
#[test]
fn test_bound_matches_sklearn_on_shared_params() {
    let matrix = sklearn_corpus();
    let corpus = LdaCorpus::<f64>::new(&matrix).unwrap();

    let lambda = Mat::from_fn(SKLEARN_K, SKLEARN_N_TERMS, |t, w| {
        SKLEARN_LAMBDA[w * SKLEARN_K + t]
    });
    let gamma = Mat::from_fn(SKLEARN_K, SKLEARN_N_DOCS, |t, d| {
        SKLEARN_GAMMA[d * SKLEARN_K + t]
    });

    let bound = lda_bound(
        &corpus,
        lambda.as_ref(),
        gamma.as_ref(),
        SKLEARN_PRIOR,
        SKLEARN_PRIOR,
    )
    .unwrap();

    assert_relative_eq!(bound, SKLEARN_BOUND, max_relative = 1e-12);
}

/// Shapes that disagree with the corpus are rejected rather than read past.
#[test]
fn test_bound_rejects_bad_shapes() {
    let matrix = sklearn_corpus();
    let corpus = LdaCorpus::<f64>::new(&matrix).unwrap();
    let lambda = Mat::<f64>::from_fn(SKLEARN_K, SKLEARN_N_TERMS, |_, _| 0.1);
    let gamma = Mat::<f64>::from_fn(SKLEARN_K, SKLEARN_N_DOCS, |_, _| 0.1);

    let wrong_k = Mat::<f64>::from_fn(SKLEARN_K + 1, SKLEARN_N_DOCS, |_, _| 0.1);
    assert!(matches!(
        lda_bound(&corpus, lambda.as_ref(), wrong_k.as_ref(), 0.1, 0.1),
        Err(BixverseErrors::LdaDimensionMismatch { .. })
    ));

    let wrong_terms = Mat::<f64>::from_fn(SKLEARN_K, SKLEARN_N_TERMS - 1, |_, _| 0.1);
    assert!(matches!(
        lda_bound(&corpus, wrong_terms.as_ref(), gamma.as_ref(), 0.1, 0.1),
        Err(BixverseErrors::LdaDimensionMismatch { .. })
    ));

    assert!(matches!(
        lda_bound(&corpus, lambda.as_ref(), gamma.as_ref(), 0.0, 0.1),
        Err(BixverseErrors::LdaInvalidHyperparameter { .. })
    ));
}

/// Fitting the reference corpus must not land below scikit-learn's optimum, and
/// each topic must still own one of the four term blocks.
#[test]
fn test_fit_reaches_sklearn_optimum() {
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

/// The public result matches the shapes the module documents.
#[test]
fn test_result_shapes() {
    let matrix = sklearn_corpus();
    let params = LdaParams {
        alpha: SKLEARN_PRIOR,
        alpha_by_topic: false,
        max_iter: 30,
        seed: 0,
        ..Default::default()
    };
    let model = lda_fit(&matrix, SKLEARN_K, Some(params), 0).unwrap();

    assert!(model.bound.is_finite());
    assert_eq!(model.cell_topic.nrows(), SKLEARN_K);
    assert_eq!(model.cell_topic.ncols(), SKLEARN_N_DOCS);
    assert_eq!(model.topic_region.nrows(), SKLEARN_N_TERMS);
    assert_eq!(model.topic_region.ncols(), SKLEARN_K);
}
