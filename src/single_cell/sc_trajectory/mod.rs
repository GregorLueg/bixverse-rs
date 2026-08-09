//! Trajectory inference on single cells and meta cells.
//!
//! Palantir (Setty, et al., Nat. Biotechnol., 2019) ported from the reference
//! Python package. The algorithm consumes a multiscale diffusion embedding plus
//! a kNN graph and never touches counts, so one implementation serves the
//! disk-backed single cell store and in-memory meta cells identically.

pub mod diffusion;
pub mod markov;
pub mod palantir;
pub mod pseudotime;
