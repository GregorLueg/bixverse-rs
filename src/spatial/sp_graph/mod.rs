//! Construction of the spatial neighbourhood graph.
//!
//! One entry point, [`builder::build_spatial_graph`], which maps per-spot
//! coordinates onto the `(neighbours, weights)` adjacency pair that the
//! spatial statistics in `sp_processing` and `sp_analysis` consume.

pub mod builder;

pub use builder::{
    SpatialAdjacency, SpatialGraphLayout, SpatialGraphParams, SpatialWeighting,
    build_spatial_graph, build_spatial_graph_csr,
};
