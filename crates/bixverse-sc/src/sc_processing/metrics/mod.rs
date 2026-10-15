//! Metrics for assessing a processed or integrated single-cell dataset.
//!
//! Split along the axis the scores actually measure: [`batch`] for how well an
//! integration erased the batch covariate, [`bio`] for how much cell type
//! structure survived it, and [`shared`] for the label-generic engines both
//! sit on. [`gene_cor`] is the odd one out and holds gene-gene correlations.
//!
//! cLISI has no function of its own: run [`lisi`] on cell type labels and read
//! [`LisiResult::clisi_norm`], the same way iLISI is [`lisi`] on batch labels
//! read through [`LisiResult::ilisi_norm`].
//!
//! ### References
//!
//! Büttner et al., Nat. Methods, 2019; Korsunsky et al., Nat. Methods, 2019;
//! Luecken et al., Nat. Methods, 2022

pub mod batch;
pub mod bio;
pub mod gene_cor;
pub mod shared;

pub use batch::*;
pub use bio::*;
pub use gene_cor::*;
pub use shared::*;
