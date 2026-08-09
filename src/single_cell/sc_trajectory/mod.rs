//! Trajectory inference on single cells and meta cells.
//!
//! Two methods, both consuming a kNN graph and never touching counts, so one
//! implementation serves the disk-backed single cell store and in-memory meta
//! cells identically.
//!
//! Palantir (Setty, et al., Nat. Biotechnol., 2019) works per cell: it builds a
//! multiscale diffusion embedding and returns a pseudotime plus fate
//! probabilities. PAGA (Wolf, et al., Genome Biol., 2019) works the other way
//! up, coarse-graining the kNN graph over a clustering into an abstracted graph
//! of cluster-to-cluster confidences. The two answer different questions and
//! share nothing but their input.

pub mod diffusion;
pub mod markov;
pub mod paga;
pub mod palantir;
pub mod pseudotime;
