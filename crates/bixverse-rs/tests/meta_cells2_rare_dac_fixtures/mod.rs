#![cfg_attr(not(feature = "large-test"), allow(dead_code))]

//! Rare-population recovery of divide-and-conquer MetaCells2, metacells 0.9.5.
//!
//! DO NOT EDIT. Regenerate with `dev/gen_metacells2_rare_dac_fixtures.py`.

pub const LCG_SEED: u64 = 9191;
pub const N_CLUSTERS: usize = 8;
pub const CELLS_PER_CLUSTER: usize = 400;
pub const MARKERS: usize = 30;
pub const NOISE_GENES: usize = 400;
pub const RARE_SIZES: [usize; 2] = [16, 48];
pub const RARE_HOSTS: [usize; 2] = [0, 3];
pub const RARE_MARKERS: usize = 12;
pub const PILE_SIZE: usize = 800;
/// Upstream seeds run: [1, 2, 3].
/// Per seed: cells of each rare population in metacells it is the majority of.
pub const UPSTREAM_RECOVERED: &[[usize; 2]] = &[[16, 46], [16, 46], [16, 46]];
