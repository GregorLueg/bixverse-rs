//! Single cell ATAC-seq for bixverse.
//!
//! Peaks and tiles live in the regular v5 store (`FeatureKind::Peak` /
//! `Tile`, TF-IDF as the norm, coordinates in the interval index), so every
//! store-level tool reads them. This crate adds what is ATAC specific:
//! fragment ingest and QC, the 10x multiome `Peaks` ingest, counting, gene
//! activity, feature selection, LSI, rLSI integration, peak calling (a MACS3
//! port) and AMULET multiplets. The numerics follow Signac (Stuart et al.,
//! Nature Methods, 2021) unless a module says otherwise.
//!
//! ## Features
#![doc = document_features::document_features!()]

pub mod errors;
pub mod integration;
pub mod io;
pub mod peaks;
pub mod prelude;
pub mod processing;
