//! Single cell ATAC-seq for bixverse.
//!
//! Peaks and tiles live in the regular v5 store (`FeatureKind::Peak` /
//! `Tile`, TF-IDF as the norm, coordinates in the interval index), so every
//! store-level tool reads them. This crate adds what is ATAC specific:
//! fragment ingest and QC, counting, gene activity, feature selection and LSI. The numerics follow Signac
//! (Stuart et al., Nature Methods, 2021) unless a module says otherwise.
//!
//! ## Features
#![doc = document_features::document_features!()]

pub mod annotation;
pub mod counting;
pub mod errors;
pub mod features;
pub mod fragments;
pub mod lsi;
pub mod prelude;
pub mod qc;
pub mod regions;
pub mod rlsi;
#[cfg(feature = "h5")]
pub mod tenx_peaks;
