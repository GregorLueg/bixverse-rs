//! Reading what an ATAC experiment ships: fragment files (BGZF, tabix), GTF
//! gene annotation and the 10x multiome peak matrix.

pub mod annotation;
pub mod fragments;
#[cfg(feature = "h5")]
pub mod tenx_peaks;
