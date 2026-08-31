//! This module contains Rust-based methods of various common bioinformatics
//! methods, accelerated and optimised were possible via multi-threading and
//! data structures with good cache locality

pub mod cis_target;
pub mod coremo;
#[cfg(feature = "dge")]
pub mod dge_bulk;
pub mod dgrdl;
pub mod diffcor;
pub mod graph_diffusions;
pub mod ica;
pub mod lda;
pub mod methods_r_wrapper;
pub mod multi_cca;
pub mod nmf_bulk;
pub mod nmf_hals;
pub mod rbh;
pub mod snf;
