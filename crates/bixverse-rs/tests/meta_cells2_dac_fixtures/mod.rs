#![cfg_attr(not(feature = "large-test"), allow(dead_code))]

//! Divide-and-conquer reference statistics from metacells 0.9.5.
//!
//! DO NOT EDIT. Regenerate with `dev/gen_metacells2_dac_fixtures.py`.

pub const LCG_SEED: u64 = 4242;
pub const N_CLUSTERS: usize = 8;
pub const CELLS_PER_CLUSTER: usize = 400;
pub const MARKERS: usize = 30;
pub const NOISE_GENES: usize = 400;
pub const PILE_SIZE: usize = 800;
/// Upstream seeds run: [1, 2, 3, 4, 5].
/// Per seed: (n_metacells, outlier cells, mixed metacells, misplaced cells).
pub const UPSTREAM_STATS: &[(usize, usize, usize, usize)] = &[
    (138, 5, 2, 20),
    (136, 27, 22, 193),
    (137, 23, 2, 28),
    (139, 20, 10, 89),
    (125, 7, 2, 16),
];
