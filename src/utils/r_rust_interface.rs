//! Various helpers to transfer between Rust and R via the rextendr interface

use extendr_api::prelude::*;
use faer::{Mat, MatRef};
use num_traits::NumCast;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::collections::{BTreeMap, HashMap};
use std::ops::{Add, Mul};

use crate::prelude::*;
use crate::utils::traits::FaerRType;

//////////////////
// Type aliases //
//////////////////

/// Type alias for a double nested HashMap
pub type NestedHashMap = FxHashMap<String, FxHashMap<String, FxHashSet<String>>>;

/// Type alias for double nested BtreeMap
pub type NestedBtreeMap = BTreeMap<String, BTreeMap<String, FxHashSet<String>>>;

////////////
// Consts //
////////////

/// Largest count [r_list_count] accepts.
///
/// `2^53` is where `f64` stops being able to tell consecutive integers apart,
/// so past it the "is this a whole number" test stops meaning anything.
const MAX_R_COUNT: f64 = 9_007_199_254_740_992.0;

/// Largest index R's own sparse matrices can hold.
///
/// `dgCMatrix` stores `i` and `p` as INTSXP, so R cannot represent a pointer at
/// or above `2^31 - 1` whatever this crate's `u32` buffers manage. Truncating
/// silently hands R negative column pointers, so the writer refuses instead.
const MAX_R_SPARSE_INDEX: u32 = i32::MAX as u32;

/////////////
// Helpers //
/////////////

/// Read an R integer vector of non-negative indices.
///
/// `NA_integer_` is `i32::MIN` and a plain `as usize` turns it into 2147483648,
/// while `-1` becomes 4294967295 after the later `index_cast`. Both build a
/// plausible-looking structure whose row spans are nonsense and whose panic
/// lands somewhere deep in a slicing routine naming neither R nor the input.
///
/// ### Params
///
/// * `value` - The list element, if present.
/// * `missing` - Error message when the slot is absent or not an integer vector.
/// * `invalid` - Error message when an entry is `NA` or negative.
///
/// ### Returns
///
/// The indices, or an error.
fn r_index_slice(
    value: Option<&Robj>,
    missing: &'static str,
    invalid: &'static str,
) -> Result<Vec<usize>, BixverseErrors> {
    let raw = value
        .and_then(|v| v.as_integer_slice())
        .ok_or(BixverseErrors::RListParse(missing))?;

    raw.iter()
        .map(|&x| {
            if x < 0 {
                Err(BixverseErrors::RListParse(invalid))
            } else {
                Ok(x as usize)
            }
        })
        .collect()
}

/// Accept an explicit zero for a matrix dimension.
///
/// [r_list_count] floors at one, since a count of zero is a mistake for every
/// parameter it serves. A sparse matrix may legitimately have no rows or no
/// columns, so that one case is read separately.
///
/// ### Params
///
/// * `value` - The list element, if present.
///
/// ### Returns
///
/// `Some(0)` when the slot holds an exact zero, `None` otherwise.
fn r_dimension_zero(value: Option<&Robj>) -> Option<usize> {
    let value = value?;
    let is_zero = value.as_integer().map(|v| v == 0).unwrap_or(false)
        || value.as_real().map(|v| v == 0.0).unwrap_or(false);
    is_zero.then_some(0)
}

/// Check that a parsed sparse structure is internally consistent.
///
/// ### Params
///
/// * `indptr` - Row (CSR) or column (CSC) pointers.
/// * `indices` - Column (CSR) or row (CSC) indices.
/// * `nnz` - Length of the data buffer.
/// * `shape` - `(nrow, ncol)`.
/// * `cs_type` - Which of the two orientations the buffers describe.
///
/// ### Returns
///
/// Nothing on success, or an error naming the violated invariant.
fn validate_sparse_layout(
    indptr: &[usize],
    indices: &[usize],
    nnz: usize,
    shape: (usize, usize),
    cs_type: CompressedSparseFormat,
) -> Result<(), BixverseErrors> {
    let (major, minor) = match cs_type {
        CompressedSparseFormat::Csr => shape,
        CompressedSparseFormat::Csc => (shape.1, shape.0),
    };

    if indices.len() != nnz {
        return Err(BixverseErrors::RListParse(
            "indices and data must have the same length",
        ));
    }
    if indptr.len() != major + 1 {
        return Err(BixverseErrors::RListParse(
            "indptr length must be the major dimension plus one",
        ));
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(BixverseErrors::RListParse("indptr must be non-decreasing"));
    }
    // `indptr` is non-empty here: `major + 1 >= 1`.
    if indptr[indptr.len() - 1] != nnz {
        return Err(BixverseErrors::RListParse(
            "the last indptr entry must equal the number of stored values",
        ));
    }
    if indices.iter().any(|&j| j >= minor) {
        return Err(BixverseErrors::RListParse(
            "an index sits outside the minor dimension",
        ));
    }

    Ok(())
}

////////////
// Errors //
////////////

/// Error handling for named numeric conversion
#[derive(Debug)]
pub enum NamedVecError {
    /// Not numeric error
    NotNumeric,
    /// No Names provided error
    NoNames,
    /// Missing values in the data error
    MissingValues,
}

impl std::fmt::Display for NamedVecError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NamedVecError::NotNumeric => write!(f, "Input is not a numeric vector"),
            NamedVecError::NoNames => write!(f, "Vector has no names attribute"),
            NamedVecError::MissingValues => write!(f, "Vector contains missing values"),
        }
    }
}

impl std::error::Error for NamedVecError {}

impl From<NamedVecError> for extendr_api::Error {
    fn from(err: NamedVecError) -> Self {
        extendr_api::Error::Other(err.to_string())
    }
}

///////////
// Lists //
///////////

/// Flatten a named R list into a name-keyed map, rejecting unnamed lists.
///
/// extendr's `TryFrom<List> for HashMap<&str, Robj>` is infallible and falls
/// back to a names vector of `NA` when the list carries no names attribute. The
/// whole list then collapses onto a single `"NA"` key, every lookup misses, and
/// a parameter block silently resolves to its defaults. `do.call(f,
/// unname(params))` is the realistic way into that, so it is checked here once
/// rather than at each of the four dozen call sites.
///
/// An empty list is allowed through: `names(list())` is `NULL` in R, and an
/// empty parameter block genuinely does mean "take every default".
///
/// ### Params
///
/// * `r_list` - The R list to flatten.
///
/// ### Returns
///
/// The name-keyed map, or an error when a non-empty list has no names.
pub fn r_list_to_map<'a>(r_list: List) -> extendr_api::Result<HashMap<&'a str, Robj>> {
    if !r_list.is_empty() && r_list.names().is_none() {
        return Err(Error::Other(
            "The parameter list has no names; did an `unname()` slip in?".to_string(),
        ));
    }
    r_list.try_into()
}

/// Read a count-like parameter out of a flattened R list.
///
/// R writes `10` as a double and only `10L` as an integer, and extendr's
/// `as_integer()` matches INTSXP alone without coercing. A caller writing
/// `list(knn = 50)` therefore hands over a double that the plain accessor drops
/// on the floor, and the default runs instead. This accepts either storage mode
/// as long as the value is a whole number.
///
/// Missing keys, `NULL` and `NA` all read as absent, so the caller's default
/// applies. Values > min are accepted and give more fine grained control if
/// a `0` value is allowed.
///
/// ### Params
///
/// * `params` - Parsed R list contents, already flattened to a map.
/// * `key` - The list element to read.
/// * `min` - Minimum value.
///
/// ### Returns
///
/// `Some(count)` when the key holds a positive whole number, `None` when it is
/// absent or `NA`, or an error naming the key.
fn r_list_bounded_count(
    params: &HashMap<&str, Robj>,
    key: &str,
    min: usize,
) -> extendr_api::Result<Option<usize>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }

    let raw = if let Some(v) = value.as_integer() {
        if v == i32::MIN {
            return Ok(None);
        }
        v as f64
    } else if let Some(v) = value.as_real() {
        if v.is_nan() {
            return Ok(None);
        }
        v
    } else {
        return Err(Error::Other(format!(
            "'{key}' must be a single number, got something else"
        )));
    };

    if raw < min as f64 || raw.fract() != 0.0 || raw > MAX_R_COUNT {
        return Err(Error::Other(format!(
            "'{key}' must be a whole number >= {min}, got {raw}"
        )));
    }

    Ok(Some(raw as usize))
}

/// Read a count-like parameter out of a flattened R list.
///
/// R writes `10` as a double and only `10L` as an integer, and extendr's
/// `as_integer()` matches INTSXP alone without coercing. A caller writing
/// `list(knn = 50)` therefore hands over a double that the plain accessor drops
/// on the floor, and the default runs instead. This accepts either storage mode
/// as long as the value is a whole number.
///
/// Missing keys, `NULL` and `NA` all read as absent, so the caller's default
/// applies. Anything present but unusable (zero, negative, fractional, or the
/// wrong type entirely) is an error rather than a silent fallback: a typo and a
/// deliberate omission should not resolve to the same run. If you want zero
/// to pass through, use [r_list_count_allow_zero].
///
/// ### Params
///
/// * `params` - Parsed R list contents, already flattened to a map.
/// * `key` - The list element to read.
///
/// ### Returns
///
/// `Some(count)` when the key holds a positive whole number, `None` when it is
/// absent or `NA`, or an error naming the key.
pub fn r_list_count(params: &HashMap<&str, Robj>, key: &str) -> extendr_api::Result<Option<usize>> {
    r_list_bounded_count(params, key, 1)
}

/// As [r_list_count], but admits `0`, which several callers use as a
/// "derive it from the data" sentinel.
///
/// ### Params
///
/// * `params` - Parsed R list contents, already flattened to a map.
/// * `key` - The list element to read.
///
/// ### Returns
///
/// `Some(count)` when the key holds a positive whole number or zero, `None`
/// when it is absent or `NA`, or an error naming the key.
pub fn r_list_count_allow_zero(
    params: &HashMap<&str, Robj>,
    key: &str,
) -> extendr_api::Result<Option<usize>> {
    r_list_bounded_count(params, key, 0)
}

/// Transforms a Robj List into a Hashmap
///
/// This function assumes that the R list contains string vector!
///
/// ### Params
///
/// * `r_list` - R list that has names and contains string vectors.
///
/// ### Returns
///
/// A HashMap with as keys the names of the list and values the string vectors.
pub fn r_list_to_hashmap(r_list: List) -> extendr_api::Result<FxHashMap<String, Vec<String>>> {
    let mut result = FxHashMap::with_capacity_and_hasher(r_list.len(), FxBuildHasher);

    for (n, s) in r_list {
        let s_vec = s.as_string_vector().ok_or_else(|| {
            Error::Other(format!(
                "Failed to convert value for key '{}' to string vector",
                n
            ))
        })?;
        result.insert(n.to_string(), s_vec);
    }

    Ok(result)
}

/// Transforms a Robj List into a Hashmap with HashSet values
///
/// This function assumes that the R list contains string vector!
///
/// ### Params
///
/// * `r_list` - R list that has names and contains string vectors.
///
/// ### Returns
///
/// A HashMap with as keys the names of the list and values as HashSets.
pub fn r_list_to_hashmap_set(
    r_list: List,
) -> extendr_api::Result<FxHashMap<String, FxHashSet<String>>> {
    let mut result = FxHashMap::with_capacity_and_hasher(r_list.len(), FxBuildHasher);

    for (n, s) in r_list {
        let s_vec = s.as_string_vector().ok_or_else(|| {
            Error::Other(format!(
                "Failed to convert value for key '{}' to string vector",
                n
            ))
        })?;
        let mut s_hash = FxHashSet::with_capacity_and_hasher(s_vec.len(), FxBuildHasher);
        for item in s_vec {
            s_hash.insert(item);
        }
        result.insert(n.to_string(), s_hash);
    }

    Ok(result)
}

/// Transforms an Robj nested list into a nested HashMap containing further HashMap
///
/// A helper that generates a nested HashMap from a nested R list.
///
/// ### Params
///
/// * `r_nested_list` - A named R list that contains named lists with String vectors.
///
/// ### Returns
///
/// Returns a `NestedHashMap`
#[allow(dead_code)]
pub fn r_nested_list_to_nested_hashmap(r_nested_list: List) -> extendr_api::Result<NestedHashMap> {
    let mut result = FxHashMap::with_capacity_and_hasher(r_nested_list.len(), FxBuildHasher);
    for (n, obj) in r_nested_list {
        let inner_list = obj.as_list().ok_or_else(|| {
            Error::Other(format!("Failed to convert value for key '{}' to list", n))
        })?;
        let inner_hashmap = r_list_to_hashmap_set(inner_list)?;
        result.insert(n.to_string(), inner_hashmap);
    }
    Ok(result)
}

/// Transforms an R list to a vector of HashSets
///
/// ### Params
///
/// * `r_list` - A named R list that contains named lists with String vectors.
///
/// ### Returns
///
/// Returns a Vector of FxHashSets
pub fn r_list_to_hash_vec(r_list: List) -> extendr_api::Result<Vec<FxHashSet<String>>> {
    let mut res = Vec::with_capacity(r_list.len());
    for (n, s) in r_list {
        let s_vec = s.as_string_vector().ok_or_else(|| {
            Error::Other(format!(
                "Failed to convert value for key '{}' to string vector",
                n
            ))
        })?;
        let mut s_hash = FxHashSet::with_capacity_and_hasher(s_vec.len(), FxBuildHasher);
        for item in s_vec {
            s_hash.insert(item);
        }
        res.push(s_hash)
    }

    Ok(res)
}

/// Transform a Robj List into a BTreeMap with the values as HashSet
///
/// Use where ordering of the values matters as the HashMaps have non-deterministic
/// ordering
///
/// ### Params
///
/// * `r_list` - R list that has names and contains string vectors.
///
/// ### Returns
///
/// A BTreeMap with as keys the names of the list and values as HashSets.
pub fn r_list_to_btree_set(
    r_list: List,
) -> extendr_api::Result<BTreeMap<String, FxHashSet<String>>> {
    let mut result = BTreeMap::new();
    for (n, s) in r_list {
        let s_vec = s.as_string_vector().ok_or_else(|| {
            Error::Other(format!(
                "Failed to convert value for key '{}' to string vector",
                n
            ))
        })?;
        let mut s_hash = FxHashSet::with_capacity_and_hasher(s_vec.len(), FxBuildHasher);
        for item in s_vec {
            s_hash.insert(item);
        }
        result.insert(n.to_string(), s_hash);
    }
    Ok(result)
}

/// Transform an Robj nested list into a nested BtreeMap
///
/// A helper that generates a nested BTreeMap from a nested R list.
///
/// ### Params
///
/// * `r_nested_list` - A named R list that contains named lists with String vectors.
///
/// ### Returns
///
/// Returns a `NestedBtreeMap`
pub fn r_nested_list_to_btree_nest(r_nested_list: List) -> extendr_api::Result<NestedBtreeMap> {
    let mut result = BTreeMap::new();

    for (n, obj) in r_nested_list {
        let inner_list = obj.as_list().ok_or_else(|| {
            Error::Other(format!("Failed to convert value for key '{}' to list", n))
        })?;
        let inner_tree = r_list_to_btree_set(inner_list)?;
        result.insert(n.to_string(), inner_tree);
    }

    Ok(result)
}

/////////////
// Vectors //
/////////////

/// Type alias for named numeric vectors
///
/// ### Fields
///
/// * `0` The names of the vector
/// * `1` The values of the vector
pub type NamedNumericVec = (Vec<String>, Vec<f64>);

/// Transforms a Robj List into an array of String arrays.
///
/// ### Params
///
/// * `r_list` - R list that has names and contains string vectors.
///
/// ### Returns
///
/// A vector of vectors with Strings
pub fn r_list_to_str_vec(r_list: List) -> extendr_api::Result<Vec<Vec<String>>> {
    let mut result = Vec::with_capacity(r_list.len());

    for (n, s) in r_list.into_iter() {
        let s_vec = s.as_string_vector().ok_or_else(|| {
            Error::Other(format!(
                "Failed to convert value to string vector at key '{}'",
                n
            ))
        })?;
        result.push(s_vec);
    }

    Ok(result)
}

/// Get the names and numeric values from a named R vector
///
/// ### Params
///
/// * `named_vec` - Robj that represents a named numeric in R
///
/// ### Returns
///
/// The `NamedNumericVec` type alias, or [NamedVecError::MissingValues] when any
/// entry is `NA` or `NaN`. Both arrive as a NaN payload and would otherwise
/// propagate into whatever statistic consumes the vector, surfacing much later
/// with no provenance.
pub fn r_named_vec_data(named_vec: Robj) -> extendr_api::Result<NamedNumericVec> {
    let values = named_vec
        .as_real_vector()
        .ok_or(NamedVecError::NotNumeric)?;

    if values.iter().any(|v| v.is_nan()) {
        return Err(NamedVecError::MissingValues.into());
    }

    let names_attr = named_vec.names().ok_or(NamedVecError::NoNames)?;

    let names: Vec<String> = names_attr.into_iter().map(|s| s.to_string()).collect();

    Ok((names, values))
}

//////////////
// Matrices //
//////////////

/// Transform an R matrix to a Faer one
///
/// ### Params
///
/// * `x` - The R matrix to transform into a faer MatRef (with `f64`)
///
/// ### Returns
///
/// The faer `MatRef` from the original R matrix.
pub fn r_matrix_to_faer<T>(x: &RMatrix<T>) -> MatRef<'_, T>
where
    T: Copy + Clone,
    extendr_api::Robj: for<'a> extendr_api::AsTypedSlice<'a, T>,
{
    let ncol = x.ncols();
    let nrow = x.nrows();
    let data = x.data();
    MatRef::from_column_major_slice(data, nrow, ncol)
}

/// Transform an R matrix into a nested vector of booleans
///
/// ### Params
///
/// * `x` - The R matrix to transform into a vector of vectors with booleans
///
/// ### Returns
///
/// The nested vector with the outer vector representing the columns.
pub fn r_matrix_to_vec_bool(x: &RMatrix<Rbool>) -> Vec<Vec<bool>> {
    let ncol = x.ncols();
    let nrow = x.nrows();
    let data = x.data();

    (0..ncol)
        .map(|j| (0..nrow).map(|i| data[i + j * nrow].to_bool()).collect())
        .collect()
}

/// Transform a faer into an R matrix
///
/// ### Params
///
/// * `x` - faer `MatRef` matrix to transform into an R matrix
///
/// ###
///
/// The R matrix based on the faer matrix.
pub fn faer_to_r_matrix<T>(x: MatRef<T>) -> extendr_api::RArray<T::RType, 2>
where
    T: FaerRType,
{
    T::to_r_matrix(x)
}

/// Transform an R matrix into a f32 one
///
/// ### Params
///
/// * `x` - R matrix with f64.
///
/// ### Returns
///
/// A faer Mat with f32
pub fn r_matrix_to_faer_fp32(x: &RMatrix<f64>) -> Mat<f32> {
    let ncol = x.ncols();
    let nrow = x.nrows();
    let data = x.data();
    let data_fp32 = data.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    Mat::from_fn(nrow, ncol, |i, j| data_fp32[i + j * nrow])
}

/// Transform a [CompressedSparseData2] into an R list
///
/// ### Params
///
/// * `sparse` - The compressed sparse structure to export.
///
/// ### Returns
///
/// R list with the following slots, or an error when the index buffers cannot
/// be represented in R:
/// * `data` - The values, as a double vector
/// * `indptr` - The row (CSR) or column (CSC) pointers, as an integer vector
/// * `indices` - The column (CSR) or row (CSC) indices, as an integer vector
/// * `nrow` - Number of rows, as an integer
/// * `ncol` - Number of columns, as an integer
/// * `cs_type` - `"csr"` or `"csc"`
///
/// The key names are exactly the ones [list_to_sparse_matrix] reads, so a list
/// survives a round trip through both.
pub fn sparse_data_to_list<T>(sparse: CompressedSparseData2<T>) -> Result<List, BixverseErrors>
where
    T: Into<Robj> + Clone + Default + Into<f64> + Sync + Add + PartialEq + Mul,
{
    let data: Vec<f64> = sparse.data.into_iter().map(Into::into).collect();
    let indptr = index_buffer_to_r(
        &sparse.indptr,
        "indptr exceeds what R's integer vectors can hold",
    )?;
    let indices = index_buffer_to_r(
        &sparse.indices,
        "indices exceed what R's integer vectors can hold",
    )?;
    let nrow = i32::try_from(sparse.shape.0)
        .map_err(|_| BixverseErrors::RListParse("nrow exceeds what R can represent"))?;
    let ncol = i32::try_from(sparse.shape.1)
        .map_err(|_| BixverseErrors::RListParse("ncol exceeds what R can represent"))?;
    let cs_type = match sparse.cs_type {
        CompressedSparseFormat::Csr => "csr",
        CompressedSparseFormat::Csc => "csc",
    };

    Ok(list!(
        data = data,
        indptr = indptr,
        indices = indices,
        nrow = nrow,
        ncol = ncol,
        cs_type = cs_type.to_string()
    ))
}

/// Narrow a `u32` index buffer to the `i32` R stores sparse indices in.
///
/// ### Params
///
/// * `buffer` - The `indptr` or `indices` buffer.
/// * `message` - Error message naming the offending slot.
///
/// ### Returns
///
/// The buffer as `i32`, or an error when any entry is past
/// [MAX_R_SPARSE_INDEX].
fn index_buffer_to_r(buffer: &[u32], message: &'static str) -> Result<Vec<i32>, BixverseErrors> {
    if buffer.iter().any(|&x| x > MAX_R_SPARSE_INDEX) {
        return Err(BixverseErrors::RListParse(message));
    }
    Ok(buffer.iter().map(|&x| x as i32).collect())
}

/// Transform an R list storing CSR/C data into CompressedSparseData2
///
/// ### Params
///
/// * `r_list` - R list that has the following elements: `indptr`, `indices`,
///   `data`, `nrow`, `ncol` and `cs_type`. The key names are exactly the ones
///   [sparse_data_to_list] writes, so a list survives a round trip through both.
///   `nrow` and `ncol` are accepted as either an integer or a whole double,
///   because R writes `5` as a double and only `5L` as an integer.
/// * `populate_data_2` - Boolean. If set to `true`, the data will be also
///   copied into data_2 of the `CompressedSparseData2`.
///
/// ### Returns
///
/// The [CompressedSparseData2]
pub fn list_to_sparse_matrix<T>(
    r_list: List,
    populate_data_2: bool,
) -> Result<CompressedSparseData2<T>, BixverseErrors>
where
    T: Clone + Default + NumCast,
{
    if !r_list.is_empty() && r_list.names().is_none() {
        return Err(BixverseErrors::RListParse("not a named list"));
    }
    let r_data: HashMap<&str, Robj> = r_list
        .try_into()
        .map_err(|_| BixverseErrors::RListParse("not a named list"))?;

    let indptr = r_index_slice(
        r_data.get("indptr"),
        "indptr missing or not integer",
        "indptr holds NA or negative values",
    )?;

    let indices = r_index_slice(
        r_data.get("indices"),
        "indices missing or not integer",
        "indices hold NA or negative values",
    )?;

    let data: Vec<T> = r_data
        .get("data")
        .and_then(|v| v.as_real_slice())
        .ok_or(BixverseErrors::RListParse("data missing or not double"))?
        .iter()
        .map(|&x| T::from(x).ok_or(BixverseErrors::RListParse("data value out of range")))
        .collect::<Result<Vec<T>, _>>()?;

    let nrow = r_list_count(&r_data, "nrow")
        .ok()
        .flatten()
        .or_else(|| r_dimension_zero(r_data.get("nrow")))
        .ok_or(BixverseErrors::RListParse(
            "nrow missing or not a non-negative whole number",
        ))?;
    let ncol = r_list_count(&r_data, "ncol")
        .ok()
        .flatten()
        .or_else(|| r_dimension_zero(r_data.get("ncol")))
        .ok_or(BixverseErrors::RListParse(
            "ncol missing or not a non-negative whole number",
        ))?;

    let cs_type = r_data
        .get("cs_type")
        .and_then(|v| v.as_str())
        .and_then(parse_compressed_sparse_format)
        .ok_or(BixverseErrors::RListParse(
            "cs_type missing or not one of 'csr' / 'csc'",
        ))?;

    validate_sparse_layout(&indptr, &indices, data.len(), (nrow, ncol), cs_type)?;

    let data_2 = if populate_data_2 {
        Some(data.clone())
    } else {
        None
    };

    Ok(CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type,
        data_2,
        shape: (nrow, ncol),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_sparse_layout_accepts_a_sound_csr() {
        // 3x4 CSR with two entries in row 0 and one in row 2.
        let indptr = [0usize, 2, 2, 3];
        let indices = [1usize, 3, 0];

        assert!(
            validate_sparse_layout(&indptr, &indices, 3, (3, 4), CompressedSparseFormat::Csr)
                .is_ok()
        );
    }

    #[test]
    fn test_validate_sparse_layout_reads_csc_the_other_way_round() {
        // The same buffers describe a 4x3 CSC: the major axis is the column
        // count and the indices are row indices.
        let indptr = [0usize, 2, 2, 3];
        let indices = [1usize, 3, 0];

        assert!(
            validate_sparse_layout(&indptr, &indices, 3, (4, 3), CompressedSparseFormat::Csc)
                .is_ok()
        );
        // As a CSR the same buffers want three rows and four columns, so a 4x3
        // shape has both the pointer length and the index range wrong.
        assert!(
            validate_sparse_layout(&indptr, &indices, 3, (4, 3), CompressedSparseFormat::Csr)
                .is_err()
        );
    }

    #[test]
    fn test_validate_sparse_layout_catches_each_fault() {
        // indices and data disagree.
        assert!(
            validate_sparse_layout(
                &[0usize, 2],
                &[0usize, 1],
                1,
                (1, 2),
                CompressedSparseFormat::Csr
            )
            .is_err()
        );
        // indptr the wrong length for the row count.
        assert!(
            validate_sparse_layout(
                &[0usize, 1],
                &[0usize],
                1,
                (3, 2),
                CompressedSparseFormat::Csr
            )
            .is_err()
        );
        // Non-monotonic pointers: the Rust range comes out empty and the row is
        // silently skipped.
        assert!(
            validate_sparse_layout(
                &[0usize, 5, 2, 7],
                &[0usize; 7],
                7,
                (3, 3),
                CompressedSparseFormat::Csr
            )
            .is_err()
        );
        // The last pointer must account for every stored value.
        assert!(
            validate_sparse_layout(
                &[0usize, 1, 1],
                &[0usize, 1],
                2,
                (2, 2),
                CompressedSparseFormat::Csr
            )
            .is_err()
        );
        // A column index past the declared width.
        assert!(
            validate_sparse_layout(
                &[0usize, 2],
                &[0usize, 9],
                2,
                (1, 2),
                CompressedSparseFormat::Csr
            )
            .is_err()
        );
    }

    #[test]
    fn test_validate_sparse_layout_accepts_an_empty_matrix() {
        assert!(
            validate_sparse_layout(&[0usize], &[], 0, (0, 0), CompressedSparseFormat::Csr).is_ok()
        );
    }
}
