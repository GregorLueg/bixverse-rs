//! scTransform v2: regularised negative binomial normalisation.
//!
//! A per-gene negative binomial model with the cell's total UMI count as a
//! fixed offset, fitted on a bounded subsample and then regularised across
//! genes against their abundance before being applied to everything.
//!
//! ```text
//! mu_gc  = exp(b0_g + ln(10) * log10_umi_c + x_c . beta_g)
//! var_gc = max(mu_gc + mu_gc^2 / theta_g, min_variance)
//! r_gc   = clamp((y_gc - mu_gc) / sqrt(var_gc), clip_range)
//! ```
//!
//! The split across files follows the stages:
//!
//! * [`nb_fit`] - the per-gene fit, one gene's counts against the offset and
//!   any covariates. Built on `edge-rs`.
//! * [`model`] - parameters, the fitted model, the regularisation that carries
//!   the subsample's fits to every gene, and the residual row.
//! * [`stream`] - the reader-driven stages over the gene-major store: the
//!   statistics sweep, the subsample selection, residual variance and the
//!   corrected counts.
//!
//! Two pieces deliberately live elsewhere, with their subject rather than with
//! this method. PCA on the residuals is a PCA entry point and sits in
//! [`crate::single_cell::sc_processing::pca`]; regenerating a cell-major store
//! from a gene-major one is a store-format operation, useful beyond this
//! method, and sits in [`crate::single_cell::sc_data::bin_merge_io`].
//!
//! Every stage bar the subsample fit is a per-gene reduction, so memory is flat
//! in the cell count. That is the difference from the R implementation, which
//! preallocates a dense genes-by-cells residual matrix before it starts.
//!
//! ### References
//!
//! Hafemeister & Satija, Genome Biology, 2019; Choudhary & Satija, Genome
//! Biology, 2022

pub mod model;
pub mod nb_fit;
pub mod stream;
