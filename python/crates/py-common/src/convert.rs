//! Moving buffers across the FFI boundary: NumPy in and out, parameter dicts
//! into crate structs, kNN and CSR parts into the crate layouts.
//!
//! Every index is 0-based. Matrices cross row-major and C-contiguous.

use bixverse_sc_data::prelude::{BixverseNumeric, CompressedSparseData2, CompressedSparseFormat};
use faer::{Mat, MatRef};
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pythonize::depythonize;
use serde::de::DeserializeOwned;

///////////
// Types //
///////////

/// The CSR parts of a meta cell matrix: data, column indices, row pointers and
/// shape.
pub type CsrParts<'a, 'py> = (
    &'a PyReadonlyArray1<'py, f32>,
    &'a PyReadonlyArray1<'py, u32>,
    &'a PyReadonlyArray1<'py, u32>,
    (usize, usize),
);

/// kNN indices and distances, one row per cell.
pub type KnnRows = (Vec<Vec<usize>>, Vec<Vec<f32>>);

////////////
// Params //
////////////

/// Deserialise a parameter dict into a crate struct
///
/// ### Params
///
/// * `obj` - The dict, as `to_rust(params)` builds it
/// * `what` - What it is, for the error message
///
/// ### Returns
///
/// The struct, or a `ValueError` naming the offending key.
pub fn from_py<T: DeserializeOwned>(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<T> {
    depythonize(obj).map_err(|e| PyValueError::new_err(format!("invalid {what}: {e}")))
}

///////////////
// NumPy in  //
///////////////

/// `uint32` indices as `usize`
///
/// ### Params
///
/// * `idx` - 0-based indices
///
/// ### Returns
///
/// The indices.
pub fn index_vec(idx: &PyReadonlyArray1<'_, u32>) -> PyResult<Vec<usize>> {
    Ok(idx.as_slice()?.iter().map(|&x| x as usize).collect())
}

/// A C-contiguous `float32` array as a faer view
///
/// ### Params
///
/// * `m` - The `(n, d)` array
///
/// ### Returns
///
/// The view, or an error if the array is not contiguous.
pub fn mat_view<'a>(m: &'a PyReadonlyArray2<'_, f32>) -> PyResult<MatRef<'a, f32>> {
    let (n, d) = (m.shape()[0], m.shape()[1]);
    Ok(MatRef::from_row_major_slice(m.as_slice()?, n, d))
}

/// kNN indices as one row per cell
///
/// ### Params
///
/// * `indices` - `(n, k)` `uint32` neighbour indices
///
/// ### Returns
///
/// The rows.
pub fn knn_rows(indices: &PyReadonlyArray2<'_, u32>) -> PyResult<Vec<Vec<usize>>> {
    let k = indices.shape()[1];
    Ok(indices
        .as_slice()?
        .chunks_exact(k.max(1))
        .map(|row| row.iter().map(|&j| j as usize).collect())
        .collect())
}

/// kNN distances as one row per cell
///
/// ### Params
///
/// * `dist` - `(n, k)` `float32` distances
/// * `shape` - The shape of the matching indices
///
/// ### Returns
///
/// The rows, or an error if the shapes disagree.
pub fn dist_rows(dist: &PyReadonlyArray2<'_, f32>, shape: &[usize]) -> PyResult<Vec<Vec<f32>>> {
    if dist.shape() != shape {
        return Err(PyValueError::new_err(
            "knn indices and distances must have the same shape",
        ));
    }
    Ok(dist
        .as_slice()?
        .chunks_exact(shape[1].max(1))
        .map(<[f32]>::to_vec)
        .collect())
}

/// An optional kNN, indices and distances given together or not at all
///
/// ### Params
///
/// * `indices` - Optional `(n, k)` indices
/// * `distances` - Optional `(n, k)` distances
///
/// ### Returns
///
/// The rows, `None` without a kNN, or an error if only one half was given.
pub fn knn_pair(
    indices: Option<&PyReadonlyArray2<'_, u32>>,
    distances: Option<&PyReadonlyArray2<'_, f32>>,
) -> PyResult<Option<KnnRows>> {
    match (indices, distances) {
        (Some(idx), Some(dist)) => Ok(Some((knn_rows(idx)?, dist_rows(dist, idx.shape())?))),
        (None, None) => Ok(None),
        _ => Err(PyValueError::new_err(
            "pass the kNN indices and distances together",
        )),
    }
}

/// A CSR matrix from its NumPy parts
///
/// The primary layer is the data cast by `cast`, the secondary layer the data
/// as given, as the meta cell matrices carry raw and normalised counts.
///
/// ### Params
///
/// * `parts` - Data, indices, row pointers and shape
/// * `cast` - Conversion of the data into the primary layer type
///
/// ### Returns
///
/// The matrix, or an error if the parts do not describe one CSR matrix.
pub fn csr_from_parts<T: BixverseNumeric>(
    parts: CsrParts<'_, '_>,
    cast: impl Fn(f32) -> T,
) -> PyResult<CompressedSparseData2<T, f32>> {
    let (data, indices, indptr, shape) = parts;
    let (data, indices, indptr) = (data.as_slice()?, indices.as_slice()?, indptr.as_slice()?);
    if indptr.len() != shape.0 + 1
        || data.len() != indices.len()
        || indptr.last().copied().unwrap_or(0) as usize != data.len()
    {
        return Err(PyValueError::new_err(
            "data, indices, indptr and shape do not describe one CSR matrix",
        ));
    }
    Ok(CompressedSparseData2::from_parts(
        data.iter().map(|&x| cast(x)).collect(),
        indices.to_vec(),
        indptr.to_vec(),
        Some(data.to_vec()),
        CompressedSparseFormat::Csr,
        shape,
    ))
}

///////////////
// NumPy out //
///////////////

/// `usize` values as a numpy `uint32` array
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The values, store indices or counts by construction
///
/// ### Returns
///
/// The array.
pub fn u32_array<'py>(py: Python<'py>, v: &[usize]) -> Bound<'py, PyAny> {
    v.iter()
        .map(|&x| x as u32)
        .collect::<Vec<u32>>()
        .into_pyarray(py)
        .into_any()
}

/// `f32` values widened to a numpy `float64` array
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The values
///
/// ### Returns
///
/// The array.
pub fn f64_array<'py>(py: Python<'py>, v: &[f32]) -> Bound<'py, PyAny> {
    v.iter()
        .map(|&x| x as f64)
        .collect::<Vec<f64>>()
        .into_pyarray(py)
        .into_any()
}

/// A flat row-major buffer as an `(n, k)` numpy array
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The values, row-major
/// * `n` - Rows
/// * `k` - Columns
///
/// ### Returns
///
/// The array.
pub fn flat_array<'py, T: numpy::Element>(
    py: Python<'py>,
    v: Vec<T>,
    n: usize,
    k: usize,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(v.into_pyarray(py).reshape([n, k])?.into_any())
}

/// A faer matrix as a row-major numpy array
///
/// ### Params
///
/// * `py` - GIL token
/// * `m` - The matrix
///
/// ### Returns
///
/// The `(nrows, ncols)` `float32` array.
pub fn mat_array<'py>(py: Python<'py>, m: &Mat<f32>) -> PyResult<Bound<'py, PyAny>> {
    let (r, c) = (m.nrows(), m.ncols());
    let data: Vec<f32> = (0..r)
        .flat_map(|i| (0..c).map(move |j| m[(i, j)]))
        .collect();
    flat_array(py, data, r, c)
}

/// A faer matrix widened to a row-major `float64` numpy array
///
/// ### Params
///
/// * `py` - GIL token
/// * `m` - The matrix
///
/// ### Returns
///
/// The `(nrows, ncols)` `float64` array.
pub fn mat_f64<'py, T: Copy + Into<f64>>(
    py: Python<'py>,
    m: &Mat<T>,
) -> PyResult<Bound<'py, PyAny>> {
    let (r, c) = (m.nrows(), m.ncols());
    let data: Vec<f64> = (0..r)
        .flat_map(|i| (0..c).map(move |j| m[(i, j)].into()))
        .collect();
    flat_array(py, data, r, c)
}
