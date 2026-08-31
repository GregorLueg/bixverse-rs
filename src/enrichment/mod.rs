//! Methods for the calculation of gene set enrichment statistics. The module
//! contains overenrichment analysis, gene set variation analysis, gene set
//! enrichment analysis (fgsea and blitzGSEA) and mitch multi-dimensional
//! enrichment. Additionally, wrappers for R are provided

pub mod blitzgsea;
pub mod enrichment_r_wrapper;
pub mod gsea;
pub mod gsva;
pub mod mitch;
pub mod oae;
pub mod singscore;
