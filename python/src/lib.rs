//! Python bindings for `bixverse-rs`.
//!
//! Open a cell-major sparse store and stream minibatches out of it fast enough
//! to keep a GPU training loop fed, and ingest h5ad, mtx and 10x h5 data into
//! such stores. Everything numerical happens in the crates; this layer only
//! moves buffers across the FFI boundary and maps errors onto exceptions.

use pyo3::prelude::*;

mod annotate;
mod batch;
mod de;
mod doublets;
mod error;
mod graph;
mod h5ad_write;
mod integrate;
mod io;
mod loader;
mod metacells;
mod mm;
mod nmf;
mod pathways;
mod pp;
mod qc;
mod reader;
mod synthetic;
mod testing;
mod traj;
mod trees;

/// Register the extension module.
///
/// ### Params
///
/// * `m` - The module object pyo3 hands us.
///
/// ### Returns
///
/// `Ok(())` once every class, exception and constant is registered.
#[pymodule]
fn _bixverse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__core_version__", bixverse_sc_data::VERSION)?;

    m.add("BixverseError", m.py().get_type::<error::BixverseError>())?;
    m.add(
        "SparseFileError",
        m.py().get_type::<error::SparseFileError>(),
    )?;

    m.add_class::<reader::PyCellReader>()?;
    m.add_class::<loader::PyCellLoader>()?;
    m.add_class::<loader::PyCellLoaderIter>()?;
    m.add_class::<pp::PyResidualFit>()?;

    m.add_function(wrap_pyfunction!(testing::write_synthetic_store, m)?)?;

    for f in [
        wrap_pyfunction!(io::csr_to_file, m)?,
        wrap_pyfunction!(io::h5ad_to_file, m)?,
        wrap_pyfunction!(io::norm_h5ad_to_file, m)?,
        wrap_pyfunction!(io::mtx_to_file, m)?,
        wrap_pyfunction!(io::tenx_h5_to_file, m)?,
        wrap_pyfunction!(io::multi_h5ad_to_file, m)?,
        wrap_pyfunction!(io::multi_mtx_to_file, m)?,
        wrap_pyfunction!(io::multi_tenx_h5_to_file, m)?,
        wrap_pyfunction!(io::merge_sc_files, m)?,
        wrap_pyfunction!(io::generate_gene_based_data, m)?,
        wrap_pyfunction!(io::gene_nnz, m)?,
        wrap_pyfunction!(io::archive_cells, m)?,
        wrap_pyfunction!(io::restore_cells_archive, m)?,
        wrap_pyfunction!(io::file_version, m)?,
        wrap_pyfunction!(io::upgrade_to_v4, m)?,
        wrap_pyfunction!(io::h5ad_obs_var_to_parquet, m)?,
        wrap_pyfunction!(io::tenx_obs_var_to_parquet, m)?,
        wrap_pyfunction!(io::snake_case, m)?,
        wrap_pyfunction!(io::h5ad_dimensions, m)?,
        wrap_pyfunction!(io::raw_count_slot, m)?,
        wrap_pyfunction!(io::h5_strings, m)?,
        wrap_pyfunction!(io::tenx_h5_metadata, m)?,
        wrap_pyfunction!(io::write_tenx, m)?,
        wrap_pyfunction!(h5ad_write::save_h5ad, m)?,
        wrap_pyfunction!(h5ad_write::write_h5ad, m)?,
        wrap_pyfunction!(synthetic::synthetic_sc_data, m)?,
        wrap_pyfunction!(synthetic::synthetic_adt_data, m)?,
        wrap_pyfunction!(synthetic::synthetic_dialogue_data, m)?,
        wrap_pyfunction!(synthetic::synthetic_cellsweep_data, m)?,
        wrap_pyfunction!(qc::mad_outlier, m)?,
        wrap_pyfunction!(qc::top_genes_perc, m)?,
        wrap_pyfunction!(qc::gene_set_perc, m)?,
        wrap_pyfunction!(pp::hvg, m)?,
        wrap_pyfunction!(pp::hvg_batch_aware, m)?,
        wrap_pyfunction!(pp::pca, m)?,
        wrap_pyfunction!(pp::fit_residuals, m)?,
        wrap_pyfunction!(pp::residual_variance, m)?,
        wrap_pyfunction!(pp::pca_residuals, m)?,
        wrap_pyfunction!(pp::sct_corrected_counts, m)?,
        wrap_pyfunction!(pp::gene_store_to_cell_store_py, m)?,
        wrap_pyfunction!(pp::magic_impute, m)?,
        wrap_pyfunction!(graph::knn_with_dist, m)?,
        wrap_pyfunction!(graph::knn_metrics, m)?,
        wrap_pyfunction!(graph::knn_graph, m)?,
        wrap_pyfunction!(graph::leiden, m)?,
        wrap_pyfunction!(graph::fast_cluster, m)?,
        wrap_pyfunction!(integrate::harmony, m)?,
        wrap_pyfunction!(integrate::harmony_v2, m)?,
        wrap_pyfunction!(integrate::bbknn_graph, m)?,
        wrap_pyfunction!(integrate::fast_mnn, m)?,
        wrap_pyfunction!(integrate::seurat_cca, m)?,
        wrap_pyfunction!(integrate::seurat_rpca, m)?,
        wrap_pyfunction!(integrate::kbet, m)?,
        wrap_pyfunction!(integrate::lisi, m)?,
        wrap_pyfunction!(integrate::batch_asw, m)?,
        wrap_pyfunction!(integrate::cell_type_asw, m)?,
        wrap_pyfunction!(integrate::pcr, m)?,
        wrap_pyfunction!(integrate::graph_connectivity, m)?,
        wrap_pyfunction!(doublets::scrublet, m)?,
        wrap_pyfunction!(doublets::boost, m)?,
        wrap_pyfunction!(doublets::scdblfinder, m)?,
        wrap_pyfunction!(doublets::empty_droplets, m)?,
        wrap_pyfunction!(doublets::cellsweep, m)?,
        wrap_pyfunction!(de::dge_mann_whitney, m)?,
        wrap_pyfunction!(de::dge_one_vs_rest, m)?,
        wrap_pyfunction!(de::dge_one_vs_many, m)?,
        wrap_pyfunction!(de::pseudobulk, m)?,
        wrap_pyfunction!(de::edger_ql, m)?,
        wrap_pyfunction!(de::nebula, m)?,
        wrap_pyfunction!(de::milor_nhoods, m)?,
        wrap_pyfunction!(de::spatial_fdr, m)?,
        wrap_pyfunction!(de::meld, m)?,
        wrap_pyfunction!(annotate::sc_type, m)?,
        wrap_pyfunction!(annotate::sc_type_cluster_assignment, m)?,
        wrap_pyfunction!(annotate::sc_type_assign_cells, m)?,
        wrap_pyfunction!(annotate::build_symphony_ref, m)?,
        wrap_pyfunction!(annotate::symphony_map_query, m)?,
        wrap_pyfunction!(annotate::transfer_labels_symphony, m)?,
        wrap_pyfunction!(metacells::bootstrapped_metacells, m)?,
        wrap_pyfunction!(metacells::seacells, m)?,
        wrap_pyfunction!(metacells::supercells, m)?,
        wrap_pyfunction!(metacells::metacells2, m)?,
        wrap_pyfunction!(metacells::metacell_density, m)?,
        wrap_pyfunction!(metacells::metacell_compactness, m)?,
        wrap_pyfunction!(metacells::metacell_separation, m)?,
        wrap_pyfunction!(metacells::mc_hvg, m)?,
        wrap_pyfunction!(metacells::mc_pca, m)?,
        wrap_pyfunction!(metacells::mc_fit_residuals, m)?,
        wrap_pyfunction!(metacells::mc_residual_variance, m)?,
        wrap_pyfunction!(metacells::mc_pca_residuals, m)?,
        wrap_pyfunction!(traj::palantir, m)?,
        wrap_pyfunction!(traj::paga, m)?,
        wrap_pyfunction!(traj::gene_trends, m)?,
        wrap_pyfunction!(trees::sc_bonsai, m)?,
        wrap_pyfunction!(trees::mc_bonsai, m)?,
        wrap_pyfunction!(trees::bonsai_layout, m)?,
        wrap_pyfunction!(pathways::module_scores, m)?,
        wrap_pyfunction!(pathways::aucell, m)?,
        wrap_pyfunction!(pathways::mc_aucell, m)?,
        wrap_pyfunction!(pathways::regulon_thresholds, m)?,
        wrap_pyfunction!(pathways::vision, m)?,
        wrap_pyfunction!(pathways::mc_vision, m)?,
        wrap_pyfunction!(pathways::vision_autocor, m)?,
        wrap_pyfunction!(pathways::mc_vision_autocor, m)?,
        wrap_pyfunction!(pathways::lm_activity, m)?,
        wrap_pyfunction!(pathways::mc_lm_activity, m)?,
        wrap_pyfunction!(pathways::hotspot_autocor, m)?,
        wrap_pyfunction!(pathways::hotspot_gene_cor, m)?,
        wrap_pyfunction!(pathways::mc_hotspot_autocor, m)?,
        wrap_pyfunction!(pathways::mc_hotspot_gene_cor, m)?,
        wrap_pyfunction!(pathways::hotspot_cluster_genes, m)?,
        wrap_pyfunction!(pathways::scenic_filter, m)?,
        wrap_pyfunction!(pathways::scenic_grn, m)?,
        wrap_pyfunction!(pathways::mc_scenic_grn, m)?,
        wrap_pyfunction!(pathways::top_k_targets, m)?,
        wrap_pyfunction!(pathways::importance_threshold, m)?,
        wrap_pyfunction!(pathways::pairwise_gene_cors, m)?,
        wrap_pyfunction!(pathways::mc_pairwise_gene_cors, m)?,
        wrap_pyfunction!(pathways::cistarget, m)?,
        wrap_pyfunction!(nmf::nmf_single_sc, m)?,
        wrap_pyfunction!(nmf::nmf_multi_sc, m)?,
        wrap_pyfunction!(nmf::nmf_consensus_sc, m)?,
        wrap_pyfunction!(nmf::nmf_k_sweep_sc, m)?,
        wrap_pyfunction!(nmf::nmf_single_mc, m)?,
        wrap_pyfunction!(nmf::nmf_multi_mc, m)?,
        wrap_pyfunction!(nmf::nmf_consensus_mc, m)?,
        wrap_pyfunction!(nmf::nmf_k_sweep_mc, m)?,
        wrap_pyfunction!(mm::adt_clr, m)?,
        wrap_pyfunction!(mm::dsb, m)?,
        wrap_pyfunction!(mm::adt_pca, m)?,
        wrap_pyfunction!(mm::wnn, m)?,
    ] {
        m.add_function(f)?;
    }

    Ok(())
}
