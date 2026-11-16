//! h5ad and AnnData Zarr export through `bixverse-sc-export`, the Python twin
//! of `rs_save_h5ad` in the R package.
//!
//! [save_h5ad] streams the counts out of a cell-based store batch by batch,
//! [write_h5ad] writes an in-memory CSR matrix (the `write_h5ad_sc()` fixture
//! writer). The Python side decides how each obs / var column is encoded;
//! this only maps the types. The export crate checks every input before the
//! output is created.

use std::collections::HashMap;
use std::path::PathBuf;

use bixverse_sc_data::data_io::ParallelSparseReader;
use bixverse_sc_export::ad_export::{
    AdFormat, AdParts, Column, ColumnData, CsMatI, DenseMatrix, Embeddings, LevelType, ObsTable,
    UnsTable, VarTable, Varm, save_store, write_csr,
};
use bixverse_sc_export::errors::ExportErrors;
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use py_common::convert::CSlice;
use py_common::error::{BixverseError, BvErr};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// gzip level of the numeric datasets, the one the rhdf5 writers used.
const H5AD_COMPRESSION: u8 = 6;

/////////////
// Helpers //
/////////////

/// Map an export error onto the Python exceptions
///
/// ### Params
///
/// * `e` - The error
///
/// ### Returns
///
/// The store error as raised elsewhere, `ValueError` for inputs that do not
/// fit together, `BixverseError` for a writer failure.
fn export_err(e: ExportErrors) -> PyErr {
    match e {
        ExportErrors::ScData(e) => BvErr::from(e).into(),
        ExportErrors::InvalidArgument(m) => PyValueError::new_err(m),
        e => BixverseError::new_err(e.to_string()),
    }
}

/// Translate `(name, kind, payload)` tuples into scx columns
///
/// ### Params
///
/// * `cols` - One tuple per column. `kind` is `"categorical"` (payload
///   `(codes int32 with -1 for missing, levels)`), `"int"` (int32 array),
///   `"float"` (float64 array), `"bool"` (bool array) or `"string"` (list).
///
/// ### Returns
///
/// The columns in order.
fn to_columns(cols: Vec<(String, String, Bound<'_, PyAny>)>) -> PyResult<Vec<Column>> {
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
                    ColumnData::Categorical {
                        codes,
                        levels,
                        ordered: false,
                        level_type: LevelType::Str,
                    }
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
            Ok(Column::new(name, data))
        })
        .collect()
}

/// Translate named 2-D arrays into row-major scx matrices
///
/// ### Params
///
/// * `mats` - Name to `float64` array with one row per obs / var
///
/// ### Returns
///
/// The map from name to matrix.
fn to_dense(mats: HashMap<String, PyReadonlyArray2<'_, f64>>) -> HashMap<String, DenseMatrix> {
    mats.into_iter()
        .map(|(name, arr)| {
            let shape = (arr.shape()[0], arr.shape()[1]);
            let data = arr.as_array().iter().copied().collect();
            (name, DenseMatrix { shape, data })
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

/// Convert the non-`X` parts of an AnnData object
///
/// ### Params
///
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
/// The [AdParts]; the export checks their lengths.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_parts(
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
) -> PyResult<AdParts> {
    let n_obs = obs_index.len();
    Ok(AdParts {
        obs: ObsTable {
            index: obs_index,
            columns: to_columns(obs)?,
        },
        var: VarTable {
            index: var_index,
            columns: to_columns(var)?,
        },
        obsm: Embeddings {
            map: to_dense(obsm),
        },
        varm: Varm {
            map: to_dense(varm),
        },
        obsp: to_csr(obsp, n_obs)?,
        uns: UnsTable {
            raw: serde_json::from_str(uns_json)
                .map_err(|e| PyValueError::new_err(format!("uns_json: {e}")))?,
            ..Default::default()
        },
    })
}

/// The export format
///
/// ### Params
///
/// * `zarr` - AnnData Zarr rather than h5ad
///
/// ### Returns
///
/// The format; h5ad at the gzip level the rhdf5 writers used.
fn ad_format(zarr: bool) -> AdFormat {
    if zarr {
        AdFormat::Zarr
    } else {
        AdFormat::H5ad {
            compression: Some(H5AD_COMPRESSION),
        }
    }
}

///////////////
// Functions //
///////////////

/// Write a single cell experiment to h5ad or AnnData Zarr, streaming the
/// counts from the store
///
/// ### Params
///
/// * `f_path_cells` - The cell-based binary
/// * `h5_path` - Output to create
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
/// * `zarr` - Write an AnnData Zarr store instead of an h5ad file
///
/// ### Returns
///
/// `None`; raises on failure before or during the write.
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, h5_path, cell_indices, norm, obs_index, obs, var_index, var, obsm, varm,
    obsp, uns_json, chunk_size = 10_000, zarr = false
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
    zarr: bool,
) -> PyResult<()> {
    let cells: Vec<usize> = cell_indices
        .c_slice()?
        .iter()
        .map(|&c| c as usize)
        .collect();
    let path = f_path_cells
        .to_str()
        .ok_or_else(|| PyValueError::new_err("f_path_cells is not valid UTF-8"))?;
    let parts = build_parts(obs_index, obs, var_index, var, obsm, varm, obsp, uns_json)?;
    let format = ad_format(zarr);
    py.detach(|| {
        let reader = ParallelSparseReader::new(path).map_err(ExportErrors::from)?;
        save_store(&reader, &h5_path, format, &cells, norm, parts, chunk_size)
    })
    .map_err(export_err)
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
    let (data, indices, indptr) = (data.to_vec()?, indices.to_vec()?, indptr.to_vec()?);
    let parts = AdParts {
        obs: ObsTable {
            index: obs_index,
            columns: to_columns(obs)?,
        },
        var: VarTable {
            index: var_index,
            columns: to_columns(var)?,
        },
        ..Default::default()
    };
    py.detach(|| write_csr(&h5_path, ad_format(false), indptr, indices, data, parts))
        .map_err(export_err)
}
