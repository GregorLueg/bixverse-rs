#![allow(clippy::excessive_precision)]
//
// The reference emits seventeen significant digits. Trimming them by hand would
// mean editing generated values, so the lint is turned off for this file only.

//! blitzGSEA parity fixtures, generated from the reference implementation.
//!
//! DO NOT EDIT. Regenerate with
//! `dev/gen_blitzgsea_fixtures.py` against
//! <https://github.com/MaayanLab/blitzgsea> and paste the output here.
//!
//! Signature: 2000 genes from a Numerical Recipes LCG seeded at
//! 12345, mapped by `x / 2^32 * 6 - 3`, sorted descending, mean
//! centred. Every step is exactly representable, so the Rust side rebuilds
//! the identical input without any float data crossing as text.

/// Number of genes in the fixture signature.
pub const N_GENES: usize = 2000;

/// Seed for the fixture LCG.
pub const LCG_SEED: u64 = 12345;

/////////////////////////
// Enrichment scores //
/////////////////////////

/// One gene set and what the reference computes for it.
pub struct EsFixture {
    /// Name of the gene set, for assertion messages
    pub name: &'static str,
    /// Index positions into the sorted, centred signature
    pub indices: &'static [i32],
    /// `blitzgsea.enrichment_score` on those indices
    pub es: f64,
    /// Index positions of the leading edge genes, ascending
    pub leading_edge: &'static [i32],
}

/// Gene sets covering an enriched, a depleted and two mixed case.
pub const ES_FIXTURES: &[EsFixture] = &[
    EsFixture {
        name: "top_50",
        indices: &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49,
        ],
        es: 9.99999999999999778e-01,
        leading_edge: &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48,
        ],
    },
    EsFixture {
        name: "bottom_50",
        indices: &[
            1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963,
            1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977,
            1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991,
            1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
        ],
        es: -1.00000000000002864e+00,
        leading_edge: &[
            1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963,
            1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977,
            1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991,
            1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
        ],
    },
    EsFixture {
        name: "strided_50",
        indices: &[
            0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680,
            720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280,
            1320, 1360, 1400, 1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840,
            1880, 1920, 1960,
        ],
        es: 1.58176455113078374e-01,
        leading_edge: &[0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440],
    },
    EsFixture {
        name: "small_7",
        indices: &[3, 17, 40, 91, 220, 501, 1300],
        es: 7.49304665836865835e-01,
        leading_edge: &[3, 17, 40, 91],
    },
    EsFixture {
        name: "head_heavy_30",
        indices: &[
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 1000, 1030,
            1060, 1090, 1120, 1150, 1180, 1210, 1240, 1270,
        ],
        es: 9.27496992750968197e-01,
        leading_edge: &[
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
        ],
    },
];

///////////////////
// Gamma fitting //
///////////////////

/// Sample size for the gamma fitting fixture.
pub const GAMMA_FIT_N: usize = 500;

/// `scipy.stats.gamma.fit(x, floc=0)` shape on the fixture sample.
pub const GAMMA_FIT_SHAPE: f64 = 1.50757109856112459e+00;

/// `scipy.stats.gamma.fit(x, floc=0)` scale on the fixture sample.
pub const GAMMA_FIT_SCALE: f64 = 3.17854594358669473e+01;

////////////////////////
// p-value and NES //
////////////////////////

/// Gamma tail parameters used by [`PVAL_FIXTURES`] and [`ES_FIXTURES`].
pub struct TailFixture {
    /// Positive-tail shape
    pub shape_pos: f64,
    /// Positive-tail scale
    pub scale_pos: f64,
    /// Negative-tail shape
    pub shape_neg: f64,
    /// Negative-tail scale
    pub scale_neg: f64,
    /// Fraction of the null mass above zero
    pub pos_ratio: f64,
}

/// The tail the p-value fixtures were generated against.
pub const FIXTURE_TAIL: TailFixture = TailFixture {
    shape_pos: 4.00000000000000000e+00,
    scale_pos: 5.00000000000000028e-02,
    shape_neg: 4.00000000000000000e+00,
    scale_neg: 5.00000000000000028e-02,
    pos_ratio: 5.00000000000000000e-01,
};

/// One enrichment score and the p-value and NES the reference maps it to.
pub struct PvalFixture {
    /// The enrichment score
    pub es: f64,
    /// Two-sided p-value from `blitzgsea.gsea`
    pub pval: f64,
    /// Normalised enrichment score as the reference reports it
    pub nes: f64,
}

/// A sweep across both signs and out into the tail scipy can still resolve.
pub const PVAL_FIXTURES: &[PvalFixture] = &[
    PvalFixture {
        es: -9.00000000000000022e-01,
        pval: 1.75601666456692840e-05,
        nes: -4.29384933051243145e+00,
    },
    PvalFixture {
        es: -6.99999999999999956e-01,
        pval: 4.74248546128652748e-04,
        nes: -3.49489485353944973e+00,
    },
    PvalFixture {
        es: -5.00000000000000000e-01,
        pval: 1.03360506759258008e-02,
        nes: -2.56437935139955986e+00,
    },
    PvalFixture {
        es: -3.49999999999999978e-01,
        pval: 8.17654162447216670e-02,
        nes: -1.74053331061823879e+00,
    },
    PvalFixture {
        es: -2.00000000000000011e-01,
        pval: 4.33470120366708844e-01,
        nes: -7.83267369969811944e-01,
    },
    PvalFixture {
        es: -1.00000000000000006e-01,
        pval: 8.57123460498546930e-01,
        nes: -1.80037077021895697e-01,
    },
    PvalFixture {
        es: 1.00000000000000006e-01,
        pval: 8.57123460498546930e-01,
        nes: 1.80037077021895697e-01,
    },
    PvalFixture {
        es: 2.00000000000000011e-01,
        pval: 4.33470120366708844e-01,
        nes: 7.83267369969811944e-01,
    },
    PvalFixture {
        es: 3.49999999999999978e-01,
        pval: 8.17654162447216670e-02,
        nes: 1.74053331061823879e+00,
    },
    PvalFixture {
        es: 5.00000000000000000e-01,
        pval: 1.03360506759258008e-02,
        nes: 2.56437935139955986e+00,
    },
    PvalFixture {
        es: 6.99999999999999956e-01,
        pval: 4.74248546128652748e-04,
        nes: 3.49489485353944973e+00,
    },
    PvalFixture {
        es: 9.00000000000000022e-01,
        pval: 1.75601666456692840e-05,
        nes: 4.29384933051243145e+00,
    },
];

/// The p-value and NES for each [`ES_FIXTURES`] entry under
/// [`FIXTURE_TAIL`], so the whole scoring path can be checked end to end.
pub const ES_FIXTURE_PVALS: &[PvalFixture] = &[
    PvalFixture {
        es: 9.99999999999999778e-01,
        pval: 3.20371978057565343e-06,
        nes: 4.65730735180018574e+00,
    },
    PvalFixture {
        es: -1.00000000000002864e+00,
        pval: 3.20371978057565343e-06,
        nes: -4.65730735180018574e+00,
    },
    PvalFixture {
        es: 1.58176455113078374e-01,
        pval: 6.10647458347385541e-01,
        nes: 5.09149470188736131e-01,
    },
    PvalFixture {
        es: 7.49304665836865835e-01,
        pval: 2.13784786740944810e-04,
        nes: 3.70214450942350037e+00,
    },
    PvalFixture {
        es: 9.27496992750968197e-01,
        pval: 1.10321314430450457e-05,
        nes: 4.39588700723865866e+00,
    },
];

/////////////////////////
// Anchor calibration //
/////////////////////////

/// Gamma parameters the reference fits at one anchor size.
///
/// Statistical, not exact. The reference seeds Python's `random` but draws
/// through `numpy.random`, so its per-anchor samples are not seed
/// controlled and no bit-exact comparison is possible. These come from
/// 20000 permutations, which pins each parameter to well inside the
/// tolerance the Rust side asserts.
pub struct AnchorFixture {
    /// Gene set size the anchor was fitted at
    pub size: usize,
    /// Shape of the pooled gamma over the absolute null scores
    pub shape: f64,
    /// Scale of the pooled gamma over the absolute null scores
    pub scale: f64,
    /// Fraction of non-zero null scores that were positive
    pub pos_ratio: f64,
}

/// Anchors spanning the range where the gamma parameters move fastest.
pub const ANCHOR_FIXTURES: &[AnchorFixture] = &[
    AnchorFixture {
        size: 5,
        shape: 1.14868210150497809e+01,
        scale: 3.96183593573386297e-02,
        pos_ratio: 5.02700000000000036e-01,
    },
    AnchorFixture {
        size: 20,
        shape: 1.39942472117653267e+01,
        scale: 2.04983818408584519e-02,
        pos_ratio: 5.03000000000000003e-01,
    },
    AnchorFixture {
        size: 100,
        shape: 2.90604459253386160e+01,
        scale: 6.72596354706142714e-03,
        pos_ratio: 5.10700000000000043e-01,
    },
    AnchorFixture {
        size: 500,
        shape: 7.78429195101405327e+01,
        scale: 2.02670550161572960e-03,
        pos_ratio: 5.17950000000000021e-01,
    },
];
