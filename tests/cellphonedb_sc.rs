//! Parity against the CellPhoneDB v5 statistical method.
//!
//! The means are deterministic and match to rounding. The p-values come from a
//! different RNG, so they are checked against the Monte Carlo error of 1000
//! permutations, and the significance calls must agree away from the
//! threshold.

#![cfg(feature = "single-cell")]

mod cpdb_fixtures;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::cellphonedb::{
    CellPhoneDbParams, LrInteraction, cellphonedb_statistical,
};
use bixverse_rs::single_cell::sc_data::in_memory_io::InMemorySparseReader;

use cpdb_fixtures::*;

/// Numerical Recipes LCG, mirroring `dev/gen_cpdb_fixtures.py`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = (1664525 * self.0 + 1013904223) % (1 << 32);
        self.0
    }
}

/// The fixture matrix as CSC (cells, genes), norm values in `data_2`.
fn fixture_matrix() -> CompressedSparseData2<u32, f32> {
    let labels: Vec<usize> = CLUSTER_SIZES
        .iter()
        .enumerate()
        .flat_map(|(k, &n)| std::iter::repeat_n(k, n))
        .collect();
    let mut rng = Lcg(LCG_SEED);
    let mut data_2: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut indptr: Vec<u32> = vec![0];
    for g in 0..GENES.len() {
        for (c, &k) in labels.iter().enumerate() {
            let u = rng.next();
            let v = rng.next();
            if (u >> 8) % 1000 < DENSITY[(g * 7 + k * 5) % DENSITY.len()] {
                data_2.push(1.0 + ((v >> 16) % 24) as f32 / 8.0);
                indices.push(c as u32);
            }
        }
        indptr.push(data_2.len() as u32);
    }
    let data: Vec<u32> = data_2.iter().map(|&v| (v * 8.0) as u32).collect();
    CompressedSparseData2::new_csc(
        &data,
        &indices,
        &indptr,
        Some(&data_2),
        (labels.len(), GENES.len()),
    )
}

#[test]
fn test_cpdb_parity_with_reference() {
    let matrix = fixture_matrix();
    let reader = InMemorySparseReader::new(&matrix, None).unwrap();
    let mut start = 0;
    let clusters: Vec<Vec<usize>> = CLUSTER_SIZES
        .iter()
        .map(|&n| {
            start += n;
            (start - n..start).collect()
        })
        .collect();
    let interactions: Vec<LrInteraction> = FIXTURES
        .iter()
        .map(|f| LrInteraction {
            partner_a: f.partner_a.to_vec(),
            partner_b: f.partner_b.to_vec(),
        })
        .collect();

    let params = CellPhoneDbParams::new(ITERATIONS, 0.1, 42, None);
    let res = cellphonedb_statistical::<f64, _>(
        &reader,
        &interactions,
        &clusters,
        Some(PAIRS),
        Some(params),
    )
    .unwrap();

    let n = ITERATIONS as f64;
    let mut n_tested = 0;
    for (i, f) in FIXTURES.iter().enumerate() {
        for (p, pair) in PAIRS.iter().enumerate() {
            let ctx = format!("{} pair {pair:?}", f.id);
            let got_mean = res.obs.means[(i, p)];
            // the reference casts counts to f32 before averaging
            assert!(
                (got_mean - f.means[p]).abs() <= 1e-6 * f.means[p].max(1.0),
                "{ctx}: mean {got_mean} vs {}",
                f.means[p]
            );

            let tested = res.obs.gate[i * PAIRS.len() + p] && got_mean > 0.0;
            if !tested {
                assert_eq!(f.pvals[p], 1.0, "{ctx}: reference tested an untested entry");
                assert_eq!(res.pvals[(i, p)], 1.0, "{ctx}");
                continue;
            }
            n_tested += 1;

            let (got, want) = (res.pvals[(i, p)], f.pvals[p]);
            // both sides are Monte Carlo estimates; pool them so p = 0 on one
            // side does not collapse the standard error
            let pooled = (got + want) / 2.0;
            let tol = 4.0 * (2.0 * pooled * (1.0 - pooled) / n).sqrt() + 1.0 / n;
            assert!((got - want).abs() <= tol, "{ctx}: p {got} vs {want}");
            if want < 0.01 {
                assert!(got <= 0.05, "{ctx}: p {got} vs {want}");
            } else if want > 0.2 {
                assert!(got > 0.05, "{ctx}: p {got} vs {want}");
            }
        }
    }
    assert!(n_tested > 0);
}
