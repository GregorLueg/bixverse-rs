//! ULM / MLM parity against decoupler. Fixtures from
//! `dev/gen_decoupler_fixtures.py`, which rebuilds the same inputs from the
//! same LCG.

mod decoupler_fixtures;

use approx::assert_relative_eq;
use faer::Mat;

use bixverse_rs::enrichment::lm_activity::*;
use decoupler_fixtures::*;

/// Numerical Recipes LCG, mirrors the generator.
///
/// ### Params
///
/// * `n` - Number of draws
/// * `seed` - Seed
///
/// ### Returns
///
/// The draws
fn lcg(n: usize, seed: u64) -> Vec<u64> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = (1664525 * x + 1013904223) % (1 << 32);
            x
        })
        .collect()
}

/// The features x observations input matrix.
///
/// ### Returns
///
/// The matrix
fn data() -> Mat<f64> {
    let vals = lcg(N_OBS * N_FEATURES, LCG_SEED);
    Mat::from_fn(N_FEATURES, N_OBS, |i, o| {
        vals[o * N_FEATURES + i] as f64 / 4294967296.0 * 6.0 - 3.0
    })
}

/// The feature sets, weighted or binary.
///
/// ### Params
///
/// * `weighted` - Use the generator's weights rather than 1
///
/// ### Returns
///
/// The feature sets
fn net(weighted: bool) -> FeatureSetWeights {
    let sets: Vec<Vec<usize>> = vec![
        SET_0.to_vec(),
        SET_1.to_vec(),
        SET_2.to_vec(),
        SET_3.to_vec(),
    ];
    let n_entries: usize = sets.iter().map(|s| s.len()).sum();
    let mut raw = lcg(n_entries, LCG_SEED + 1).into_iter();
    let weights: Vec<Vec<f64>> = sets
        .iter()
        .map(|s| {
            s.iter()
                .map(|_| {
                    let k = (raw.next().unwrap() % 8) as f64;
                    if k < 4.0 {
                        (k - 4.0) / 4.0
                    } else {
                        (k - 3.0) / 4.0
                    }
                })
                .collect()
        })
        .collect();
    let weights = weighted.then_some(weights.as_slice());
    FeatureSetWeights::new(&sets, weights, N_FEATURES).unwrap()
}

/// Compare one method on one network against the fixtures.
///
/// ### Params
///
/// * `method` - ULM or MLM
/// * `weighted` - Which network
/// * `t` - Reference t values
/// * `coef` - Reference slopes
/// * `p` - Reference p-values
fn check(
    method: LmMethod,
    weighted: bool,
    t: &[[f64; 4]; 6],
    coef: &[[f64; 4]; 6],
    p: &[[f64; 4]; 6],
) {
    let mat = data();
    let net = net(weighted);
    let t_res = lm_activity(
        mat.as_ref(),
        &net,
        Some(LmActivityParams::new(method, LmStat::TValue, true)),
    )
    .unwrap();
    let coef_res = lm_activity(
        mat.as_ref(),
        &net,
        Some(LmActivityParams::new(method, LmStat::Coefficient, false)),
    )
    .unwrap();
    let p_res = t_res.p_values.unwrap();
    for o in 0..N_OBS {
        for s in 0..4 {
            assert_relative_eq!(t_res.es[(o, s)], t[o][s], max_relative = 1e-10);
            assert_relative_eq!(coef_res.es[(o, s)], coef[o][s], max_relative = 1e-10);
            assert_relative_eq!(p_res[(o, s)], p[o][s], max_relative = 1e-8);
        }
    }
}

#[test]
fn test_ulm_binary_matches_decoupler() {
    check(LmMethod::Ulm, false, &ULM_BIN_T, &ULM_BIN_COEF, &ULM_BIN_P);
}

#[test]
fn test_ulm_weighted_matches_decoupler() {
    check(LmMethod::Ulm, true, &ULM_W_T, &ULM_W_COEF, &ULM_W_P);
}

#[test]
fn test_mlm_binary_matches_decoupler() {
    check(LmMethod::Mlm, false, &MLM_BIN_T, &MLM_BIN_COEF, &MLM_BIN_P);
}

#[test]
fn test_mlm_weighted_matches_decoupler() {
    check(LmMethod::Mlm, true, &MLM_W_T, &MLM_W_COEF, &MLM_W_P);
}
