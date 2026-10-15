//! GPU-accelerated single cell methods, on top of `bixverse-gpu` and
//! `bixverse-sc`.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod bbknn_gpu;
pub mod errors;
pub mod fast_clusters_gpu;
pub mod fast_mnn_gpu;
pub mod harmony_gpu;
pub mod kernels;
pub mod knn_gpu;
pub mod nebula_gpu;
pub mod pca_gpu;
pub mod prelude;
#[cfg(feature = "bonsai")]
pub mod sanity_bonsai_gpu;
pub mod scenic_gpu;
pub mod scrublet_gpu;
pub mod seacells_gpu;
