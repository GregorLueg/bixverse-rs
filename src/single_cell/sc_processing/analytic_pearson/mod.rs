//! Analytic Pearson residuals for UMI normalisation.
//!
//! The one-parameter offset model of Lause, Berens & Kobak. Where scTransform
//! fits a negative binomial GLM per gene and then regularises the estimates
//! across genes, this drops the free slope, keeps sequencing depth as a fixed
//! offset, and shares one overdispersion across every gene. What is left has a
//! closed-form maximum likelihood solution and no fitting at all:
//!
//! ```text
//! mu_cg = (sum_j X_cj * sum_i X_ig) / sum_ij X_ij
//! z_cg  = clamp((X_cg - mu_cg) / sqrt(mu_cg + mu_cg^2 / theta), +/- sqrt(n))
//! ```
//!
//! with `theta = 100`, which the paper reads off negative control datasets as
//! the technical overdispersion of UMI counts. `theta -> inf` gives the Poisson
//! limit. The clipping is Hafemeister & Satija's, and it matters: without it a
//! marker gene confined to a handful of cells produces an enormous residual
//! variance and dominates the leading components.
//!
//! The split across files follows the stages:
//!
//! * [`model`] - parameters, the closed-form model and the residual row.
//! * [`stream`] - the reader-driven passes that collect the three sums, and the
//!   grouped driver that fits one model per sample.
//!
//! Everything downstream is shared with scTransform through
//! [`ResidualSource`](crate::single_cell::sc_processing::residuals::ResidualSource):
//! residual variance for HVG selection, then PCA on the selected genes.
//!
//! ### References
//!
//! Lause, Berens & Kobak, Genome Biology, 2021, 22:258

pub mod model;
pub mod stream;
