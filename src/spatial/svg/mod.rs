//! Spatially variable gene (SVG) detection methods.
//!
//! Exposes Moran's I via [`morans_i`] (analytical normality null on a
//! pre-built spatial graph) and SPARK-X via [`sparkx`] (kernel-bank score
//! test on raw coordinates).

pub mod morans_i;
pub mod shared;
pub mod sparkx;
