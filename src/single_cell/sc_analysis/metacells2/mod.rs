//! Port of the MetaCells2 workflow over into `biverse-rs`. The original
//! implementations were in R/C++ for the original implementation by Baran,
//! et al., Genome Biol. 2019; this was expanded in the MetaCells2 framework
//! into Python and C++, please refer to Ben‑Kiki, et al., Genome Biol. 2022.
//! Due to complexity of the algorithm, this lives in its own module.

pub mod downsample;
pub mod params;
pub mod pile;
pub mod select;
