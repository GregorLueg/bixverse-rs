//! The MLX twins (Apple Silicon, via mlx-c) of the wgpu functions in this
//! module, behind the `mlx` feature.
//!
//! Each `*_mlx` function keeps the signature of its wgpu namesake, so the
//! Python layer picks one by the `backend` argument. MLX is f32 only and has
//! no device handle: every call runs on the default MLX GPU stream.

use pyo3::prelude::*;

/// Whether the MLX twins were compiled in
///
/// The `mlx` feature is only on in a source build against an installed
/// mlx-c, so a wheel from PyPI always answers `False`.
///
/// ### Returns
///
/// `True` when the module carries the `*_mlx` functions.
#[pyfunction]
pub fn mlx_available() -> bool {
    cfg!(feature = "mlx")
}

#[cfg(feature = "mlx")]
pub(crate) mod twins {
    use ann_search_rs::prelude::KMeansTrainingParams;
    use bixverse_core::ml::clustering::k_means::parse_kmeans_init;
    use bixverse_mlx::errors::MlxErrors;
    use bixverse_mlx::sc::fast_clusters::MlxKMeans;
    use bixverse_mlx::sc::harmony as mlx_harmony;
    use bixverse_mlx::sc::knn as mlx_knn;
    use bixverse_mlx::sc::pca::pca_on_sc_sparse_mlx;
    use bixverse_mlx::sc::seacells::seacells_fit_mlx;
    use bixverse_sc::mc_generation::generate::seacells_on_store_with;
    use bixverse_sc::mc_generation::seacells::SEACellsParams;
    use bixverse_sc::sc_analysis::fast_clusters::{FastClusterParams, fast_clusters};
    use bixverse_sc::sc_batch_correction::harmony_v2::HarmonyParamsV2;
    use bixverse_sc::sc_processing::knn::pad_knn_rows;
    use bixverse_sc::sc_processing::pca::SingleCellPcaParams;
    use bixverse_sc_data::store::reader::{ParallelSparseReader, SingleCellReading};
    use bixverse_sc_gpu::fast_clusters_gpu::KMeansGpuSpec;
    use bixverse_sc_gpu::harmony_gpu::HarmonyParamsV2Gpu;
    use bixverse_sc_gpu::knn_gpu::KnnParamsGpu;
    use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
    use py_common::convert::{flat_array, from_py, index_vec, knn_pair, mat_array, mat_view};
    use py_common::error::{BvErr, data_err, sc_err};
    use py_common::results::{FastClusterOut, fast_cluster_out, seacells_dict};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use crate::error::{BixverseError, SparseFileError};

    ////////////
    // Errors //
    ////////////

    /// Carries a [`MlxErrors`] to the FFI boundary, as `GpuErr` does for the
    /// wgpu errors.
    struct MlxErr(MlxErrors);

    impl From<MlxErrors> for MlxErr {
        fn from(e: MlxErrors) -> Self {
            Self(e)
        }
    }

    impl From<MlxErr> for PyErr {
        fn from(e: MlxErr) -> PyErr {
            match e.0 {
                MlxErrors::Sc(e) => sc_err::<BixverseError, SparseFileError>(e),
                MlxErrors::ScData(e) => data_err::<BixverseError, SparseFileError>(e),
                other => BixverseError::new_err(other.to_string()),
            }
        }
    }

    /////////
    // PCA //
    /////////

    /// Sparse PCA over a cell and gene selection on MLX
    ///
    /// The twin of `pca`, same signature and solvers.
    ///
    /// ### Params
    ///
    /// * `f_path_gene` - The gene-based binary
    /// * `f_path_cell` - The cell-based binary, for the PFlogPF offsets
    /// * `cell_indices` - 0-based cells
    /// * `gene_indices` - 0-based genes
    /// * `no_pcs` - Number of components
    /// * `params` - `to_rust(PcaParams)`
    /// * `sparse` - Ignored, always sparse
    /// * `seed` - Seed for the sketch or the Lanczos starting block
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// `(scores, loadings, singular_values)`, `float32`.
    #[pyfunction]
    #[pyo3(signature = (
        f_path_gene, f_path_cell, cell_indices, gene_indices, no_pcs, params, sparse = true,
        seed = 42, verbose = 1
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn pca_mlx<'py>(
        py: Python<'py>,
        f_path_gene: &str,
        f_path_cell: &str,
        cell_indices: PyReadonlyArray1<'py, u32>,
        gene_indices: PyReadonlyArray1<'py, u32>,
        no_pcs: usize,
        params: &Bound<'py, PyAny>,
        sparse: bool,
        seed: usize,
        verbose: usize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let _ = sparse;
        let params: SingleCellPcaParams = from_py(params, "pca params")?;
        let cells = index_vec(&cell_indices)?;
        let genes = index_vec(&gene_indices)?;
        let offsets = if params.clr {
            let reader = ParallelSparseReader::new(f_path_cell).map_err(BvErr::from)?;
            let offsets = reader.get_clr_offsets(&cells, None).map_err(BvErr::from)?;
            Some(offsets.iter().map(|&x| x as f32).collect::<Vec<f32>>())
        } else {
            None
        };
        let reader = ParallelSparseReader::new(f_path_gene).map_err(BvErr::from)?;
        let (scores, loadings, s) = py
            .detach(|| {
                pca_on_sc_sparse_mlx(
                    &reader,
                    &cells,
                    &genes,
                    no_pcs,
                    offsets.as_deref(),
                    &params,
                    seed,
                    verbose,
                )
            })
            .map_err(MlxErr::from)?;
        Ok((
            mat_array(py, &scores)?,
            mat_array(py, &loadings)?,
            s.into_pyarray(py).into_any(),
        ))
    }

    /////////
    // kNN //
    /////////

    /// MLX kNN over an embedding, with the distances
    ///
    /// The twin of `knn_with_dist`, same methods and padding.
    ///
    /// ### Params
    ///
    /// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
    /// * `params` - `to_rust(KnnGpuDefaults)`; `k` sets the neighbours
    /// * `seed` - Seed for the index builds
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` as `(n_cells, k)` `uint32` and `float32`, self
    /// excluded, Euclidean as true L2.
    #[pyfunction]
    #[pyo3(signature = (embd, params, seed = 42, verbose = 1))]
    pub fn knn_with_dist_mlx<'py>(
        py: Python<'py>,
        embd: PyReadonlyArray2<'py, f32>,
        params: &Bound<'py, PyAny>,
        seed: usize,
        verbose: usize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let params: KnnParamsGpu = from_py(params, "GPU knn params")?;
        let view = mat_view(&embd)?;
        let (n, k) = (view.nrows(), params.k);
        if k == 0 || k >= n {
            return Err(PyValueError::new_err(format!(
                "k must be in 1..{n} for {n} cells, got {k}"
            )));
        }
        let (mut idx, mut dist) = py
            .detach(|| mlx_knn::knn_with_dist_mlx(view, k, &params, seed, verbose))
            .map_err(MlxErr::from)?;
        let padded = pad_knn_rows(&mut idx, Some(&mut dist), k);
        if padded > 0 && verbose > 0 {
            println!(
                "{padded} of {n} rows came back with fewer than {k} neighbours and were \
                 padded with duplicate edges."
            );
        }
        let idx: Vec<u32> = idx.into_iter().flatten().map(|j| j as u32).collect();
        let dist: Vec<f32> = dist.into_iter().flatten().collect();
        Ok((flat_array(py, idx, n, k)?, flat_array(py, dist, n, k)?))
    }

    ///////////////////
    // Fast clusters //
    ///////////////////

    /// k-means coarsening on MLX, a centroid graph and Louvain / Leiden on it
    ///
    /// The twin of `fast_cluster`. MLX runs every Lloyd's iteration without a
    /// convergence check and holds the data at f32, so `fixed` and `quantise`
    /// are ignored.
    ///
    /// ### Params
    ///
    /// * `embd` - `(n_cells, n_dims)` `float32` embedding, C-contiguous
    /// * `resolutions` - The resolutions, in output order
    /// * `params` - `to_rust(ScFastClusterGpuParams)` with `n_centroids` and
    ///   `n_iterations` resolved
    /// * `n_seeds` - Community-detection seeds per resolution
    /// * `seed` - Seed
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// See [`fast_cluster_out`].
    #[pyfunction]
    #[pyo3(signature = (embd, resolutions, params, n_seeds = 1, seed = 42, verbose = 1))]
    pub fn fast_cluster_mlx<'py>(
        py: Python<'py>,
        embd: PyReadonlyArray2<'py, f32>,
        resolutions: Vec<f32>,
        params: &Bound<'py, PyAny>,
        n_seeds: usize,
        seed: usize,
        verbose: usize,
    ) -> PyResult<FastClusterOut<'py>> {
        let fc: FastClusterParams = from_py(params, "fast cluster params")?;
        let km: KMeansGpuSpec = from_py(params, "GPU k-means params")?;
        let init = km
            .k_means_init
            .as_deref()
            .map(|s| {
                parse_kmeans_init(s).ok_or_else(|| {
                    PyValueError::new_err(format!("Invalid k-means initialisation: {s}"))
                })
            })
            .transpose()?;
        let km = KMeansTrainingParams::new(km.k_means_iter, init, None);
        let view = mat_view(&embd)?;
        let n = view.nrows();
        let res = py
            .detach(|| {
                fast_clusters(
                    view,
                    &MlxKMeans::new(Some(km)),
                    &fc,
                    &resolutions,
                    n_seeds,
                    seed,
                    verbose,
                )
            })
            .map_err(MlxErr::from)?;
        fast_cluster_out(py, res, n, n_seeds)
    }

    /////////////
    // Harmony //
    /////////////

    /// Harmony v2 on MLX, one batch covariate
    ///
    /// The twin of `harmony_v2`; takes the GPU parameters and converts them.
    ///
    /// ### Params
    ///
    /// * `pca` - `(n_cells, n_pcs)` `float32` PCA, C-contiguous
    /// * `params` - `to_rust(ScHarmonyV2GpuParams)` with `k` resolved
    /// * `batch_labels` - One `uint32` code vector per covariate; MLX takes
    ///   exactly one
    /// * `seed` - Seed
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// The `(n_cells, n_pcs)` `float32` corrected embedding.
    #[pyfunction]
    #[pyo3(signature = (pca, params, batch_labels, seed = 42, verbose = 1))]
    pub fn harmony_v2_mlx<'py>(
        py: Python<'py>,
        pca: PyReadonlyArray2<'py, f32>,
        params: &Bound<'py, PyAny>,
        batch_labels: Vec<PyReadonlyArray1<'py, u32>>,
        seed: usize,
        verbose: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params: HarmonyParamsV2Gpu = from_py(params, "GPU harmony v2 params")?;
        let params = HarmonyParamsV2::from(&params);
        let labels = batch_labels
            .iter()
            .map(index_vec)
            .collect::<PyResult<Vec<_>>>()?;
        let view = mat_view(&pca)?;
        let res = py
            .detach(|| mlx_harmony::harmony_v2_mlx(view, &labels, &params, seed, verbose))
            .map_err(MlxErr::from)?;
        mat_array(py, &res)
    }

    //////////////
    // SEACells //
    //////////////

    /// SEACells over a cell-major store, the fit on MLX
    ///
    /// The twin of `seacells`, same signature.
    ///
    /// ### Params
    ///
    /// * `f_path_cell` - The cell-based binary
    /// * `embd` - `(n, d)` embedding over the working rows, C-contiguous
    /// * `knn_indices` - Optional `(n, k)` kNN over the working rows
    /// * `knn_distances` - Its distances
    /// * `cells_to_keep` - Store index of each working row
    /// * `cells_to_use` - Optional narrowing, as store indices
    /// * `params` - `to_rust(SeacellsParams)`
    /// * `target_size` - Library size to normalise the aggregates to
    /// * `seed` - Seed
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// See [`seacells_dict`].
    #[pyfunction]
    #[pyo3(signature = (
        f_path_cell, embd, knn_indices, knn_distances, cells_to_keep, cells_to_use, params,
        target_size, seed, verbose = 1
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn seacells_mlx<'py>(
        py: Python<'py>,
        f_path_cell: &str,
        embd: PyReadonlyArray2<'py, f32>,
        knn_indices: Option<PyReadonlyArray2<'py, u32>>,
        knn_distances: Option<PyReadonlyArray2<'py, f32>>,
        cells_to_keep: PyReadonlyArray1<'py, u32>,
        cells_to_use: Option<PyReadonlyArray1<'py, u32>>,
        params: &Bound<'py, PyAny>,
        target_size: f32,
        seed: usize,
        verbose: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let params: SEACellsParams = from_py(params, "SEACells params")?;
        let knn = knn_pair(knn_indices.as_ref(), knn_distances.as_ref())?;
        let embd = mat_view(&embd)?;
        let keep = index_vec(&cells_to_keep)?;
        let to_use = cells_to_use.as_ref().map(index_vec).transpose()?;
        let v = verbose;
        let res = py
            .detach(|| {
                seacells_on_store_with::<MlxErrors, _>(
                    f_path_cell,
                    embd,
                    knn.as_ref().map(|(i, d)| (i.as_slice(), d.as_slice())),
                    Some(&keep),
                    to_use.as_deref(),
                    &params,
                    target_size,
                    seed,
                    v,
                    |e, i, d| seacells_fit_mlx(e, i, d, &params, seed, v),
                )
            })
            .map_err(MlxErr::from)?;
        seacells_dict(py, res)
    }
}
