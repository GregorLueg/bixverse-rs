//! Highly optimised code for computational biology: statistics, enrichment,
//! graphs, matrix factorisation, a single cell suite and GPU backends.
//!
//! This crate is a facade. The code lives in the `bixverse-*` crates; this one
//! re-exports them under the module paths the crate has always had, so a
//! consumer that wants everything only needs one dependency and a consumer
//! that wants one layer can depend on that crate directly.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]

pub use bixverse_core::{VERSION, core, ml, utils};
pub use bixverse_methods::{enrichment, graph, methods, ontology};

/// The error enums of every crate compiled in. Each upper enum wraps the ones
/// below it, so `?` converts upwards.
pub mod errors {
    pub use bixverse_core::errors::*;
    #[cfg(feature = "gpu")]
    pub use bixverse_gpu::errors::*;
    pub use bixverse_methods::errors::*;
    #[cfg(feature = "mlx")]
    pub use bixverse_mlx::errors::*;
    #[cfg(feature = "single-cell")]
    pub use bixverse_sc::errors::*;
    #[cfg(feature = "single-cell")]
    pub use bixverse_sc_data::errors::*;
    #[cfg(feature = "sc-gpu")]
    pub use bixverse_sc_gpu::errors::*;
}

/// The union of the preludes of every crate compiled in.
///
/// Each upper prelude already re-exports the ones below it, so some of these
/// globs are redundant for a given feature set.
#[allow(unused_imports)]
pub mod prelude {
    #[cfg(feature = "gpu")]
    pub use bixverse_gpu::prelude::*;
    pub use bixverse_methods::prelude::*;
    #[cfg(feature = "single-cell")]
    pub use bixverse_sc::prelude::*;
    #[cfg(feature = "sc-gpu")]
    pub use bixverse_sc_gpu::prelude::*;
}

/// Single cell and metacell analysis.
#[cfg(feature = "single-cell")]
pub mod single_cell {
    pub use bixverse_sc::*;
    pub use bixverse_sc_data as sc_data;
    pub use bixverse_sc_data::{CELL_BATCH_SIZE, sc_traits};
}

/// GPU methods via cubecl.
#[cfg(feature = "gpu")]
pub mod gpu {
    pub use bixverse_gpu::*;
    #[cfg(feature = "sc-gpu")]
    pub use bixverse_sc_gpu as sc_gpu;
}

#[cfg(feature = "mlx")]
pub use bixverse_mlx as mlx;

/// Whether the MLX backend was compiled in: the `mlx` feature is on and the
/// build script found `libmlxc`.
#[cfg(feature = "mlx")]
pub const MLX_AVAILABLE: bool = bixverse_mlx::MLX_AVAILABLE;
/// Whether the MLX backend was compiled in: the `mlx` feature is on and the
/// build script found `libmlxc`.
#[cfg(not(feature = "mlx"))]
pub const MLX_AVAILABLE: bool = false;
