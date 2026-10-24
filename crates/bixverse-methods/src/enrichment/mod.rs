//! Methods for the calculation of gene set enrichment statistics. The module
//! contains overenrichment analysis, gene set variation analysis, gene set
//! enrichment analysis (fgsea and blitzGSEA), mitch multi-dimensional
//! enrichment and decoupler-style linear model activity scores (ULM, MLM).

pub mod blitzgsea;
pub mod gsea;
pub mod gsva;
pub mod lm_activity;
pub mod mitch;
pub mod oae;
pub mod singscore;
