//! This is the main module that contains various analysis methods for single
//! cell applications. It ranges from scoring gene sets in single cell, to
//! differential gene expression, pseudo-bulking cells and various different
//! metacell approaches.

pub mod dge_pathway_scores;
pub mod fast_clusters;
pub mod fast_ranking;
pub mod hotspot;
pub mod meld;
pub mod milo_r;
pub mod module_scoring;
pub mod scenic;
pub mod vision;
