//! h5ad export through scx-core, the Python twin of `rs_save_h5ad` in the R
//! package.
//!
//! Two entry points share one builder: [save_h5ad] streams the counts out of
//! a cell-based store batch by batch, [write_h5ad] writes an in-memory CSR
//! matrix (the `write_h5ad_sc()` fixture writer). The Python side decides how
//! each obs / var column is encoded; this only maps the types. Every input
//! is checked before the file is created.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bixverse_sc_data::data_io::{DataLayerReturn, ParallelSparseReader};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::CSlice;
use py_common::error::{BixverseError, BvErr};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scx_core::api::write::{H5AdBuilder, H5AdOptions};
use scx_core::ir::{
    Column, ColumnData, DenseMatrix, Embeddings, ObsTable, UnsTable, VarTable, Varm,
};
use sprs::CsMatI;

/// gzip level of the numeric datasets, the one the rhdf5 writers used.
const H5AD_COMPRESSION: u8 = 6;

/////////////
// Helpers //
/////////////

/// Map any scx error onto the package's base exception
///
/// ### Params
///
/// * `e` - The error
///
/// ### Returns
///
/// A `BixverseError` carrying its message.
fn scx_err(e: impl std::fmt::Display) -> PyErr {
    BixverseError::new_err(e.to_string())
}

/// Error unless a length matches the axis it belongs to
///
/// ### Params
///
/// * `what` - Name of the input, for the message
/// * `got` - Its length
/// * `expected` - The length of the axis
///
/// ### Returns
///
/// `Ok(())` if the lengths match, a `ValueError` otherwise.
fn check_len(what: &str, got: usize, expected: usize) -> PyResult<()> {
    if got == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{what} has length {got}, expected {expected}."
        )))
    }
}

/// Translate `(name, kind, payload)` tuples into scx columns
///
/// ### Params
///
/// * `cols` - One tuple per column. `kind` is `"categorical"` (payload
///   `(codes int32 with -1 for missing, levels)`), `"int"` (int32 array),
///   `"float"` (float64 array), `"bool"` (bool array) or `"string"` (list).
/// * `n` - Length every column must have
///
/// ### Returns
///
/// The columns in order.
fn to_columns(cols: Vec<(String, String, Bound<'_, PyAny>)>, n: usize) -> PyResult<Vec<Column>> {
    cols.into_iter()
        .map(|(name, kind, payload)| {
            let data = match kind.as_str() {
                "categorical" => {
                    let (codes, levels): (PyReadonlyArray1<i32>, Vec<String>) =
                        payload.extract()?;
                    // -1 becomes u32::MAX, which scx narrows to -1, the anndata
                    // code for a missing value, in every code width
                    let codes = codes
                        .c_slice()?
                        .iter()
                        .map(|&c| if c < 0 { u32::MAX } else { c as u32 })
                        .collect();
                    ColumnData::Categorical { codes, levels }
                }
                "int" => ColumnData::Int(payload.extract::<PyReadonlyArray1<i32>>()?.to_vec()?),
                "float" => ColumnData::Float(payload.extract::<PyReadonlyArray1<f64>>()?.to_vec()?),
                "bool" => ColumnData::Bool(payload.extract::<PyReadonlyArray1<bool>>()?.to_vec()?),
                "string" => ColumnData::String(payload.extract()?),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Column '{name}' has kind '{other}', which has no h5ad encoding."
                    )));
                }
            };
            check_len(&format!("Column '{name}'"), data.len(), n)?;
            Ok(Column::new(name, data))
        })
        .collect()
}

/// Translate named 2-D arrays into row-major scx matrices
///
/// ### Params
///
/// * `mats` - Name to `float64` array with one row per obs / var
/// * `n` - Number of rows every matrix must have
///
/// ### Returns
///
/// The map from name to matrix.
fn to_dense(
    mats: HashMap<String, PyReadonlyArray2<'_, f64>>,
    n: usize,
) -> PyResult<HashMap<String, DenseMatrix>> {
    mats.into_iter()
        .map(|(name, arr)| {
            let shape = (arr.shape()[0], arr.shape()[1]);
            check_len(&format!("Matrix '{name}'"), shape.0, n)?;
            let data = arr.as_array().iter().copied().collect();
            Ok((name, DenseMatrix { shape, data }))
        })
        .collect()
}

/// Translate named CSR triples into square sprs matrices
///
/// ### Params
///
/// * `mats` - Name to `(data float32, indices uint32, indptr uint32)`
/// * `n` - Number of rows and columns
///
/// ### Returns
///
/// The validated matrices, in arbitrary order.
#[allow(clippy::type_complexity)]
fn to_csr(
    mats: HashMap<
        String,
        (
            PyReadonlyArray1<'_, f32>,
            PyReadonlyArray1<'_, u32>,
            PyReadonlyArray1<'_, u32>,
        ),
    >,
    n: usize,
) -> PyResult<Vec<(String, CsMatI<f32, u32>)>> {
    mats.into_iter()
        .map(|(name, (data, indices, indptr))| {
            let mat = CsMatI::try_new((n, n), indptr.to_vec()?, indices.to_vec()?, data.to_vec()?)
                .map_err(|(_, _, _, e)| PyValueError::new_err(format!("obsp '{name}': {e}")))?;
            Ok((name, mat))
        })
        .collect()
}

/// Everything an h5ad file holds besides `X`, converted and checked
struct H5adParts {
    /// The obs table
    obs: ObsTable,
    /// The var table
    var: VarTable,
    /// Cell embeddings
    obsm: Embeddings,
    /// Gene loadings
    varm: Varm,
    /// Cell x cell graphs
    obsp: Vec<(String, CsMatI<f32, u32>)>,
    /// Unstructured metadata
    uns: UnsTable,
}

/// Write an h5ad file, `X` pushed in row chunks by `push_x`
///
/// ### Params
///
/// * `h5_path` - File to create
/// * `n_obs` - Rows of `X`
/// * `n_vars` - Columns of `X`
/// * `chunk_size` - Rows per streaming chunk
/// * `parts` - The tables and annotations
/// * `push_x` - Pushes every row chunk of `X` into the builder, in order
///
/// ### Returns
///
/// `Ok(())`, or the first error raised.
fn write_parts(
    h5_path: &Path,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
    parts: H5adParts,
    push_x: impl FnOnce(&mut H5AdBuilder) -> PyResult<()>,
) -> PyResult<()> {
    let mut opts = H5AdOptions::default();
    opts.compression = Some(H5AD_COMPRESSION);
    opts.chunk_size = Some(chunk_size);
    let mut builder = H5AdBuilder::new(h5_path, n_obs, n_vars, &opts).map_err(scx_err)?;

    push_x(&mut builder)?;

    builder.obs(parts.obs).map_err(scx_err)?;
    builder.var(parts.var).map_err(scx_err)?;
    builder.add_obsm(parts.obsm).map_err(scx_err)?;
    builder.add_varm(parts.varm).map_err(scx_err)?;
    for (name, mat) in &parts.obsp {
        builder.add_obsp_csr(name, mat.view()).map_err(scx_err)?;
    }
    builder.add_uns(parts.uns).map_err(scx_err)?;
    builder.finalize().map_err(scx_err)?;
    Ok(())
}

/// Convert and check the non-`X` parts of an h5ad file
///
/// ### Params
///
/// * `n_obs` - Number of cells
/// * `obs_index` - One name per cell
/// * `obs` - obs columns, see [to_columns]
/// * `var_index` - One name per gene
/// * `var` - var columns, see [to_columns]
/// * `obsm` - Cell embeddings, see [to_dense]
/// * `varm` - Gene loadings, see [to_dense]
/// * `obsp` - Cell graphs, see [to_csr]
/// * `uns_json` - JSON object for `uns`
///
/// ### Returns
///
/// The [H5adParts].
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_parts(
    n_obs: usize,
    obs_index: Vec<String>,
    obs: Vec<(String, String, Bound<'_, PyAny>)>,
    var_index: Vec<String>,
    var: Vec<(String, String, Bound<'_, PyAny>)>,
    obsm: HashMap<String, PyReadonlyArray2<'_, f64>>,
    varm: HashMap<String, PyReadonlyArray2<'_, f64>>,
    obsp: HashMap<
        String,
        (
            PyReadonlyArray1<'_, f32>,
            PyReadonlyArray1<'_, u32>,
            PyReadonlyArray1<'_, u32>,
        ),
    >,
    uns_json: &str,
) -> PyResult<H5adParts> {
    let n_vars = var_index.len();
    check_len("obs_index", obs_index.len(), n_obs)?;
    Ok(H5adParts {
        obs: ObsTable {
            index: obs_index,
            columns: to_columns(obs, n_obs)?,
        },
        var: VarTable {
            index: var_index,
            columns: to_columns(var, n_vars)?,
        },
        obsm: Embeddings {
            map: to_dense(obsm, n_obs)?,
        },
        varm: Varm {
            map: to_dense(varm, n_vars)?,
        },
        obsp: to_csr(obsp, n_obs)?,
        uns: UnsTable {
            raw: serde_json::from_str(uns_json)
                .map_err(|e| PyValueError::new_err(format!("uns_json: {e}")))?,
        },
    })
}

///////////////
// Functions //
///////////////

/// Write a single cell experiment to h5ad, streaming the counts from the store
///
/// ### Params
///
/// * `f_path_cells` - The cell-based binary
/// * `h5_path` - File to create
/// * `cell_indices` - 0-based cells to write, in file order
/// * `norm` - Write the normalised instead of the raw counts
/// * `obs_index` - One name per written cell
/// * `obs` - obs columns, see [to_columns]
/// * `var_index` - One name per gene
/// * `var` - var columns
/// * `obsm` - Cell embeddings
/// * `varm` - Gene loadings
/// * `obsp` - Cell graphs as `(data, indices, indptr)`
/// * `uns_json` - JSON object for `uns`
/// * `chunk_size` - Cells per streaming batch
///
/// ### Returns
///
/// `None`; raises on failure before or during the write.
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, h5_path, cell_indices, norm, obs_index, obs, var_index, var, obsm, varm,
    obsp, uns_json, chunk_size = 10_000
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn save_h5ad(
    py: Python<'_>,
    f_path_cells: PathBuf,
    h5_path: PathBuf,
    cell_indices: PyReadonlyArray1<'_, u32>,
    norm: bool,
    obs_index: Vec<String>,
    obs: Vec<(String, String, Bound<'_, PyAny>)>,
    var_index: Vec<String>,
    var: Vec<(String, String, Bound<'_, PyAny>)>,
    obsm: HashMap<String, PyReadonlyArray2<'_, f64>>,
    varm: HashMap<String, PyReadonlyArray2<'_, f64>>,
    obsp: HashMap<
        String,
        (
            PyReadonlyArray1<'_, f32>,
            PyReadonlyArray1<'_, u32>,
            PyReadonlyArray1<'_, u32>,
        ),
    >,
    uns_json: &str,
    chunk_size: usize,
) -> PyResult<()> {
    let cells: Vec<usize> = cell_indices
        .c_slice()?
        .iter()
        .map(|&c| c as usize)
        .collect();
    let (n_obs, n_vars) = (cells.len(), var_index.len());
    let chunk_size = chunk_size.max(1);
    let path = f_path_cells
        .to_str()
        .ok_or_else(|| PyValueError::new_err("f_path_cells is not valid UTF-8"))?;

    // convert and check everything before the file exists, so that bad input
    // fails here and not after the counts have been streamed
    let reader = py
        .detach(|| ParallelSparseReader::new(path))
        .map_err(BvErr::from)?;
    let parts = build_parts(
        n_obs, obs_index, obs, var_index, var, obsm, varm, obsp, uns_json,
    )?;
    let layer = if norm {
        DataLayerReturn::Norm
    } else {
        DataLayerReturn::Raw
    };

    py.detach(|| {
        write_parts(&h5_path, n_obs, n_vars, chunk_size, parts, |builder| {
            for (i, batch) in cells.chunks(chunk_size).enumerate() {
                let csr = reader.read_cells_csr(batch, &layer).map_err(BvErr::from)?;
                let data = if norm { csr.data_norm } else { csr.data_raw }
                    .expect("the requested layer is always materialised");
                let indptr: Vec<u64> = csr.indptr.iter().map(|&p| p as u64).collect();
                builder
                    .push_x_csr_chunk(i * chunk_size, indptr, csr.indices, data)
                    .map_err(scx_err)?;
            }
            Ok(())
        })
    })
}

/// Write an in-memory cells x genes CSR matrix to h5ad
///
/// ### Params
///
/// * `h5_path` - File to create
/// * `data` - Values, one per non-zero
/// * `indices` - 0-based gene index per non-zero
/// * `indptr` - Row pointers over cells
/// * `obs_index` - One name per cell
/// * `obs` - obs columns, see [to_columns]
/// * `var_index` - One name per gene
/// * `var` - var columns
///
/// ### Returns
///
/// `None`; raises on inconsistent input.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn write_h5ad(
    py: Python<'_>,
    h5_path: PathBuf,
    data: PyReadonlyArray1<'_, f32>,
    indices: PyReadonlyArray1<'_, u32>,
    indptr: PyReadonlyArray1<'_, u64>,
    obs_index: Vec<String>,
    obs: Vec<(String, String, Bound<'_, PyAny>)>,
    var_index: Vec<String>,
    var: Vec<(String, String, Bound<'_, PyAny>)>,
) -> PyResult<()> {
    let (n_obs, n_vars) = (obs_index.len(), var_index.len());
    check_len("indptr", indptr.len(), n_obs + 1)?;
    let (data, indices, indptr) = (data.to_vec()?, indices.to_vec()?, indptr.to_vec()?);
    if indices.iter().any(|&j| j as usize >= n_vars) {
        return Err(PyValueError::new_err("indices hold a column outside var"));
    }
    let empty = PyDict::new(py);
    let parts = build_parts(
        n_obs,
        obs_index,
        obs,
        var_index,
        var,
        empty.extract()?,
        empty.extract()?,
        empty.extract()?,
        "{}",
    )?;
    py.detach(|| {
        write_parts(&h5_path, n_obs, n_vars, n_obs.max(1), parts, |builder| {
            builder
                .push_x_csr_chunk(0, indptr, indices, data)
                .map_err(scx_err)?;
            Ok(())
        })
    })
}
