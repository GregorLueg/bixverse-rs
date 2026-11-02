//! HVG, PCA, residual models and MAGIC, the Python twins of `rs_sc_hvg`,
//! `rs_sc_hvg_batch_aware`, `rs_sc_pca`, `rs_sc_pca_sparse`,
//! `rs_sc_fit_residuals`, `rs_sc_residual_variance`, `rs_sc_pca_residuals`,
//! `rs_sct_corrected_counts`, `rs_sc_gene_store_to_cell_store` and
//! `rs_magic_impute`.
//!
//! The numerics and the store-level dispatch live in `bixverse-sc`; this only
//! moves buffers. Every index is 0-based. A fitted residual model stays in
//! Rust as a [`PyResidualFit`] rather than round-tripping through a dict.

use bixverse_sc::sc_processing::analytic_pearson::model::AprParams;
use bixverse_sc::sc_processing::hvg::{
    HvgParams, HvgStats, hvg_batch_aware_on_store, hvg_on_store,
};
use bixverse_sc::sc_processing::magic::{MagicOperator, MagicParams, magic_impute_genes};
use bixverse_sc::sc_processing::pca::{SingleCellPcaParams, pca_on_store};
use bixverse_sc::sc_processing::residual_fit::{
    ResidualFit, ResidualParams, fit_residuals_on_store, pca_residuals_on_store,
    residual_variance_on_store, sct_corrected_counts_on_store,
};
use bixverse_sc::sc_processing::sctransform::model::{SctCovariates, SctParams};
use bixverse_sc::sc_processing::sctransform::stream::SctStreamOpts;
use bixverse_sc_data::bin_merge_io::gene_store_to_cell_store;
use bixverse_sc_data::data_io::{ParallelSparseReader, SingleCellReading};
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use py_common::convert::{from_py, index_vec, mat_array, u32_array};
use py_common::error::BvErr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/////////////
// Helpers //
/////////////

/// Disk batching, seed and verbosity of the residual stages
///
/// ### Params
///
/// * `gene_batch_size` - Genes per disk batch; `None` keeps the crate default
/// * `seed` - Seed for the step-1 subsampling
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The options.
fn stream_opts(gene_batch_size: Option<usize>, seed: u64, verbose: usize) -> SctStreamOpts {
    let defaults = SctStreamOpts::default();
    SctStreamOpts {
        gene_batch_size: gene_batch_size.or(defaults.gene_batch_size),
        seed,
        verbose,
    }
}

/// HVG statistics, one batch after another, as a column dict
///
/// Column names are R's, so the dict writes straight to var. With
/// `batch_cols` it also carries `batch` and `gene_idx` per row, as
/// `rs_sc_hvg_batch_aware` returns them.
///
/// ### Params
///
/// * `py` - GIL token
/// * `stats` - One set of statistics per batch, all of one method
/// * `batch_cols` - Add the `batch` and `gene_idx` columns
///
/// ### Returns
///
/// Column name to `float64` array, `bin` and `batch` as `int32`, `gene_idx`
/// as `uint32`.
pub(crate) fn stats_dict<'py>(
    py: Python<'py>,
    stats: Vec<HvgStats>,
    batch_cols: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut cols: Vec<(&'static str, Vec<f64>)> = Vec::new();
    let mut bin: Option<Vec<i32>> = None;
    let mut batch: Vec<i32> = Vec::new();
    let mut gene_idx: Vec<u32> = Vec::new();

    for (b, s) in stats.into_iter().enumerate() {
        let (named, n_genes) = match s {
            HvgStats::Vst(r) => {
                let n = r.mean.len();
                let named = vec![
                    ("mean", r.mean),
                    ("var", r.var),
                    ("var_exp", r.var_exp),
                    ("var_std", r.var_std),
                ];
                (named, n)
            }
            HvgStats::Dispersion(r) => {
                let n = r.mean.len();
                bin.get_or_insert_with(Vec::new).extend(r.bin);
                let named = vec![
                    ("mean", r.mean),
                    ("dispersion", r.dispersion),
                    ("dispersion_scaled", r.dispersion_scaled),
                ];
                (named, n)
            }
            HvgStats::Scran(r) => {
                let n = r.mean.len();
                let named = vec![
                    ("scran_mean", r.mean),
                    ("scran_var", r.var),
                    ("scran_fitted", r.fitted),
                    ("scran_residual", r.residual),
                ];
                (named, n)
            }
        };
        if cols.is_empty() {
            cols = named;
        } else {
            for ((_, acc), (_, v)) in cols.iter_mut().zip(named) {
                acc.extend(v);
            }
        }
        batch.extend(std::iter::repeat_n(b as i32, n_genes));
        gene_idx.extend(0..n_genes as u32);
    }

    let d = PyDict::new(py);
    for (name, v) in cols {
        d.set_item(name, v.into_pyarray(py))?;
    }
    if let Some(bin) = bin {
        d.set_item("bin", bin.into_pyarray(py))?;
    }
    if batch_cols {
        d.set_item("batch", batch.into_pyarray(py))?;
        d.set_item("gene_idx", gene_idx.into_pyarray(py))?;
    }
    Ok(d)
}

/////////
// HVG //
/////////

/// HVG statistics over a cell selection
///
/// The twin of `rs_sc_hvg`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `cell_indices` - 0-based cells, unique
/// * `params` - `to_rust(HvgParams)`, method other than `"residual"`
/// * `streaming` - Read the genes in batches
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Column name to one value per store gene, see [stats_dict].
#[pyfunction]
#[pyo3(signature = (f_path_gene, f_path_cell, cell_indices, params, streaming, verbose = 1))]
pub fn hvg<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    streaming: bool,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params: HvgParams = from_py(params, "hvg params")?;
    let cells = index_vec(&cell_indices)?;
    let res = py
        .detach(|| {
            hvg_on_store(
                f_path_gene,
                f_path_cell,
                &cells,
                &params,
                streaming,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    stats_dict(py, vec![res], false)
}

/// Per-batch HVG statistics over a cell selection
///
/// The twin of `rs_sc_hvg_batch_aware`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `cell_indices` - 0-based cells, unique
/// * `batch_labels` - 0-based batch per cell, densely covering the batches
/// * `params` - `to_rust(HvgParams)`, method other than `"residual"`
/// * `streaming` - Read the genes in batches
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The statistics stacked batch after batch, with `batch` and `gene_idx`.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cell_indices, batch_labels, params, streaming, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn hvg_batch_aware<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    batch_labels: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    streaming: bool,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let params: HvgParams = from_py(params, "hvg params")?;
    let cells = index_vec(&cell_indices)?;
    let batches = index_vec(&batch_labels)?;
    let res = py
        .detach(|| {
            hvg_batch_aware_on_store(
                f_path_gene,
                f_path_cell,
                &cells,
                &batches,
                &params,
                streaming,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    stats_dict(py, res, true)
}

/////////
// PCA //
/////////

/// PCA over a cell and gene selection
///
/// The twin of `rs_sc_pca` (`sparse = false`) and `rs_sc_pca_sparse`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary, for the CLR offsets
/// * `cell_indices` - 0-based cells, in score row order
/// * `gene_indices` - 0-based genes, in loading row order
/// * `no_pcs` - Number of components
/// * `params` - `to_rust(PcaParams)`
/// * `sparse` - Solve on the sparse matrix
/// * `seed` - Seed for the randomised SVD
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(scores, loadings, singular_values)`, `float32`.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, cell_indices, gene_indices, no_pcs, params, sparse, seed,
    verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn pca<'py>(
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
    let params: SingleCellPcaParams = from_py(params, "pca params")?;
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;
    let (scores, loadings, s) = py
        .detach(|| {
            pca_on_store(
                f_path_gene,
                f_path_cell,
                &cells,
                &genes,
                no_pcs,
                &params,
                sparse,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    Ok((
        mat_array(py, &scores)?,
        mat_array(py, &loadings)?,
        s.into_pyarray(py).into_any(),
    ))
}

///////////////
// Residuals //
///////////////

/// A fitted scTransform or analytic Pearson model, held in Rust
///
/// R's `ScResidualFit` minus the parameters, which the Python side keeps.
#[pyclass(frozen, name = "ResidualFit", module = "bixverse._bixverse")]
pub struct PyResidualFit {
    /// The fit and its cells.
    pub inner: ResidualFit,
}

#[pymethods]
impl PyResidualFit {
    /// `"sctransform"` or `"analytic_pearson"`.
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method()
    }

    /// Number of groups, one model each.
    #[getter]
    fn n_groups(&self) -> usize {
        self.inner.n_groups()
    }

    /// 0-based store indices of the fitted cells, in fitting order.
    #[getter]
    fn cell_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        u32_array(py, &self.inner.cell_indices)
    }

    /// 0-based store genes modelled in every group, ascending.
    #[getter]
    fn genes<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        u32_array(py, &self.inner.genes())
    }

    /// Group id per fitted cell.
    #[getter]
    fn group_of_cell<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner
            .group_of_cell()
            .to_vec()
            .into_pyarray(py)
            .into_any()
    }

    fn __repr__(&self) -> String {
        format!(
            "ResidualFit(method={:?}, n_groups={}, cells={}, genes={})",
            self.inner.method(),
            self.inner.n_groups(),
            self.inner.cell_indices.len(),
            self.inner.genes().len()
        )
    }
}

/// Fit a residual model over a cell selection
///
/// The twin of `rs_sc_fit_residuals`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `method` - `"sctransform"` or `"analytic_pearson"`
/// * `cell_indices` - 0-based cells, unique, in fitting order
/// * `group_of_cell` - Optional 0-based group per cell, dense
/// * `covariates` - `(name, values)` per covariate, aligned to the cells
///   (scTransform only)
/// * `params` - `to_rust(SctransformParams)` or `to_rust(AprParams)`
/// * `gene_batch_size` - Genes per disk batch, `None` for the default
/// * `seed` - Seed for the step-1 subsampling
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The [PyResidualFit].
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, method, cell_indices, group_of_cell, covariates, params,
    gene_batch_size = None, seed = 42, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn fit_residuals(
    py: Python<'_>,
    f_path_gene: &str,
    f_path_cell: &str,
    method: &str,
    cell_indices: PyReadonlyArray1<'_, u32>,
    group_of_cell: Option<PyReadonlyArray1<'_, u32>>,
    covariates: Vec<(String, PyReadonlyArray1<'_, f64>)>,
    params: &Bound<'_, PyAny>,
    gene_batch_size: Option<usize>,
    seed: u64,
    verbose: usize,
) -> PyResult<PyResidualFit> {
    let params = match method {
        "sctransform" => ResidualParams::Sct(from_py::<SctParams>(params, "sctransform params")?),
        "analytic_pearson" => ResidualParams::Apr(from_py::<AprParams>(params, "apr params")?),
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'sctransform' or 'analytic_pearson', got '{other}'"
            )));
        }
    };
    let cells = index_vec(&cell_indices)?;
    let groups: Option<Vec<u32>> = group_of_cell
        .map(|g| g.as_slice().map(<[u32]>::to_vec))
        .transpose()?;
    let columns = covariates
        .iter()
        .map(|(name, v)| Ok((name.clone(), v.as_slice()?.to_vec())))
        .collect::<PyResult<Vec<(String, Vec<f64>)>>>()?;
    let opts = stream_opts(gene_batch_size, seed, verbose);

    let fit = py
        .detach(|| {
            let covariates = SctCovariates::from_columns(&columns)?;
            fit_residuals_on_store(
                f_path_gene,
                f_path_cell,
                &cells,
                groups.as_deref(),
                covariates,
                params,
                opts,
            )
        })
        .map_err(BvErr::from)?;
    Ok(PyResidualFit { inner: fit })
}

/// Residual variance per gene and group, and the residual HVGs
///
/// The twin of `rs_sc_residual_variance`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `fit` - The fitted model
/// * `cell_indices` - 0-based cells; must be the fitted ones
/// * `n_hvg` - Variable features per group
/// * `gene_batch_size` - Genes per disk batch, `None` for the default
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(genes, variance, hvg)`: the model's genes, a `(n_genes, n_groups)`
/// `float64` variance matrix and the selected genes, all 0-based.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, fit, cell_indices, n_hvg, gene_batch_size = None, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn residual_variance<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    fit: &PyResidualFit,
    cell_indices: PyReadonlyArray1<'py, u32>,
    n_hvg: usize,
    gene_batch_size: Option<usize>,
    verbose: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let cells = index_vec(&cell_indices)?;
    let opts = stream_opts(gene_batch_size, SctStreamOpts::default().seed, verbose);
    let res = py
        .detach(|| {
            residual_variance_on_store(f_path_gene, f_path_cell, &fit.inner, &cells, n_hvg, opts)
        })
        .map_err(BvErr::from)?;

    let (n_genes, n_groups) = (res.genes.len(), res.variance.len());
    let flat: Vec<f64> = (0..n_genes)
        .flat_map(|g| res.variance.iter().map(move |v| v[g]))
        .collect();
    Ok((
        u32_array(py, &res.genes),
        flat.into_pyarray(py)
            .reshape([n_genes, n_groups])?
            .into_any(),
        u32_array(py, &res.hvg),
    ))
}

/// PCA on the residuals of a fitted model
///
/// The twin of `rs_sc_pca_residuals`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `f_path_cell` - The cell-based binary
/// * `fit` - The fitted model
/// * `cell_indices` - 0-based cells; must be the fitted ones
/// * `gene_indices` - 0-based genes, all covered by the model
/// * `no_pcs` - Number of components
/// * `params` - `to_rust(PcaParams)` with `clr` and `normalise_variance` off
/// * `seed` - Seed for the randomised SVD
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(scores, loadings, singular_values)`, `float32`.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, f_path_cell, fit, cell_indices, gene_indices, no_pcs, params, seed,
    verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn pca_residuals<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    f_path_cell: &str,
    fit: &PyResidualFit,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    no_pcs: usize,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let params: SingleCellPcaParams = from_py(params, "pca params")?;
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;
    let (scores, loadings, s) = py
        .detach(|| {
            pca_residuals_on_store(
                f_path_gene,
                f_path_cell,
                &fit.inner,
                &cells,
                &genes,
                no_pcs,
                &params,
                seed,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    Ok((
        mat_array(py, &scores)?,
        mat_array(py, &loadings)?,
        s.into_pyarray(py).into_any(),
    ))
}

/// Write scTransform corrected counts as a gene-major store
///
/// The twin of `rs_sct_corrected_counts`.
///
/// ### Params
///
/// * `f_path_gene` - The source gene-based binary
/// * `fit` - The fitted scTransform model
/// * `cell_indices` - 0-based cells; must be the fitted ones
/// * `f_path_out` - The gene-based binary to write
/// * `gene_batch_size` - Genes per disk batch, `None` for the default
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The 0-based source genes written, the new store's gene axis.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, fit, cell_indices, f_path_out, gene_batch_size = None, verbose = 1
))]
pub fn sct_corrected_counts<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    fit: &PyResidualFit,
    cell_indices: PyReadonlyArray1<'py, u32>,
    f_path_out: &str,
    gene_batch_size: Option<usize>,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let cells = index_vec(&cell_indices)?;
    let opts = stream_opts(gene_batch_size, SctStreamOpts::default().seed, verbose);
    let genes = py
        .detach(|| sct_corrected_counts_on_store(f_path_gene, &fit.inner, &cells, f_path_out, opts))
        .map_err(BvErr::from)?;
    Ok(u32_array(py, &genes))
}

/// Build the cell-based twin of a gene-based binary
///
/// The twin of `rs_sc_gene_store_to_cell_store`.
///
/// ### Params
///
/// * `f_path_in` - The gene-based binary
/// * `f_path_out` - The cell-based binary to write
/// * `cells_per_phase` - Cells held in memory per phase
/// * `gene_batch_size` - Genes read per batch
/// * `verbose` - Controls verbosity
#[pyfunction]
#[pyo3(name = "gene_store_to_cell_store", signature = (f_path_in, f_path_out, cells_per_phase, gene_batch_size, verbose = 1))]
pub fn gene_store_to_cell_store_py(
    py: Python<'_>,
    f_path_in: &str,
    f_path_out: &str,
    cells_per_phase: usize,
    gene_batch_size: usize,
    verbose: usize,
) -> PyResult<()> {
    py.detach(|| {
        gene_store_to_cell_store(
            f_path_in,
            f_path_out,
            cells_per_phase,
            gene_batch_size,
            verbose,
        )
    })
    .map_err(BvErr::from)?;
    Ok(())
}

///////////
// MAGIC //
///////////

/// Impute genes with MAGIC over a kNN graph
///
/// The twin of `rs_magic_impute`.
///
/// ### Params
///
/// * `f_path_gene` - The gene-based binary
/// * `knn_indices` - `(n_cells, k)` neighbours, positions within
///   `cell_indices`
/// * `knn_distances` - `(n_cells, k)` distances
/// * `cell_indices` - 0-based cells the kNN rows are, in row order
/// * `gene_indices` - 0-based genes to impute
/// * `params` - `to_rust(MagicParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The `(n_cells, n_genes)` `float32` imputed values.
#[pyfunction]
#[pyo3(signature = (
    f_path_gene, knn_indices, knn_distances, cell_indices, gene_indices, params, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn magic_impute<'py>(
    py: Python<'py>,
    f_path_gene: &str,
    knn_indices: PyReadonlyArray2<'py, u32>,
    knn_distances: PyReadonlyArray2<'py, f32>,
    cell_indices: PyReadonlyArray1<'py, u32>,
    gene_indices: PyReadonlyArray1<'py, u32>,
    params: &Bound<'py, PyAny>,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let params: MagicParams = from_py(params, "magic params")?;
    let k = knn_indices.shape()[1];
    let indices: Vec<Vec<usize>> = knn_indices
        .as_slice()?
        .chunks(k.max(1))
        .map(|row| row.iter().map(|&x| x as usize).collect())
        .collect();
    let distances: Vec<Vec<f32>> = knn_distances
        .as_slice()?
        .chunks(k.max(1))
        .map(<[f32]>::to_vec)
        .collect();
    let cells = index_vec(&cell_indices)?;
    let genes = index_vec(&gene_indices)?;

    let res = py
        .detach(|| {
            let reader = ParallelSparseReader::new(f_path_gene)?;
            let total_cells = reader.get_header().total_cells;
            let operator = MagicOperator::from_knn(&indices, &distances, &cells, total_cells)?;
            magic_impute_genes(&reader, &operator, &genes, Some(params), verbose)
        })
        .map_err(BvErr::from)?;
    let n_genes = res.n_genes();
    Ok(res
        .data
        .into_pyarray(py)
        .reshape([res.n_cells, n_genes])?
        .into_any())
}
