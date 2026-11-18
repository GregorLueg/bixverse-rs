//! Python bindings for the bixverse GPU single cell methods.
//!
//! The GPU twins of the CPU functions in `bixverse._bixverse`, on wgpu via
//! cubecl. Where a twin keeps its CPU signature, the Python layer runs the
//! CPU code path with this module swapped in. With the `mlx` feature, the
//! `*_mlx` twins run the same calls on MLX (Apple Silicon, via mlx-c). The
//! two extensions share no Rust objects: stores are read from their paths and
//! arrays cross as NumPy.

use pyo3::prelude::*;

mod de;
mod device;
mod doublets;
mod error;
mod graph;
mod integrate;
mod metacells;
mod mlx;
mod nmf;
mod pathways;
mod pp;

/// Register the extension module.
///
/// ### Params
///
/// * `m` - The module object pyo3 hands us.
///
/// ### Returns
///
/// `Ok(())` once every function is registered.
#[pymodule]
fn _bixverse_gpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    for f in [
        wrap_pyfunction!(device::gpu_available, m)?,
        wrap_pyfunction!(device::gpu_plane_ops, m)?,
        wrap_pyfunction!(pp::pca, m)?,
        wrap_pyfunction!(graph::knn_with_dist, m)?,
        wrap_pyfunction!(graph::leiden, m)?,
        wrap_pyfunction!(graph::fast_cluster, m)?,
        wrap_pyfunction!(integrate::harmony_v2, m)?,
        wrap_pyfunction!(integrate::bbknn_graph, m)?,
        wrap_pyfunction!(integrate::fast_mnn, m)?,
        wrap_pyfunction!(doublets::scrublet, m)?,
        wrap_pyfunction!(metacells::seacells, m)?,
        wrap_pyfunction!(metacells::sc_bonsai, m)?,
        wrap_pyfunction!(metacells::mc_bonsai, m)?,
        wrap_pyfunction!(nmf::nmf_single_sc, m)?,
        wrap_pyfunction!(nmf::nmf_multi_sc, m)?,
        wrap_pyfunction!(nmf::nmf_consensus_sc, m)?,
        wrap_pyfunction!(nmf::nmf_k_sweep_sc, m)?,
        wrap_pyfunction!(nmf::nmf_single_mc, m)?,
        wrap_pyfunction!(nmf::nmf_multi_mc, m)?,
        wrap_pyfunction!(nmf::nmf_consensus_mc, m)?,
        wrap_pyfunction!(nmf::nmf_k_sweep_mc, m)?,
        wrap_pyfunction!(pathways::scenic_grn, m)?,
        wrap_pyfunction!(pathways::mc_scenic_grn, m)?,
        wrap_pyfunction!(de::nebula, m)?,
        wrap_pyfunction!(mlx::mlx_available, m)?,
    ] {
        m.add_function(f)?;
    }
    #[cfg(feature = "mlx")]
    for f in [
        wrap_pyfunction!(mlx::twins::pca_mlx, m)?,
        wrap_pyfunction!(mlx::twins::knn_with_dist_mlx, m)?,
        wrap_pyfunction!(mlx::twins::fast_cluster_mlx, m)?,
        wrap_pyfunction!(mlx::twins::harmony_v2_mlx, m)?,
        wrap_pyfunction!(mlx::twins::seacells_mlx, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
