//! Methods for the calculation of gene set enrichment statistics. The module
//! contains overenrichment analysis, gene set variation analysis, gene set
//! enrichment analysis and mitch multi-dimensional enrichment. Additionally,
//! wrappers for R are provided

pub mod enrichment_r_wrapper;
pub mod gsea;
pub mod gsva;
pub mod mitch;
pub mod oae;
