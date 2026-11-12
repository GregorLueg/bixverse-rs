//! Decoding of h5ad dataframes and plain HDF5 datasets into memory, plus a
//! minimal h5ad writer. `obs_var_io` builds its Parquet export on top of the
//! decoders.

use hdf5::types::{TypeDescriptor, VarLenUnicode};
use hdf5::{Container, Dataset, File, Group, IndexType, IterationOrder, Location, LocationType};
use ndarray::ArrayView2;
use std::ops::Deref;
use std::path::Path;
use std::str::FromStr;

use crate::h5_10x_io::read_string_dataset;
use crate::prelude::*;

///////////
// Types //
///////////

/// A decoded h5ad dataframe column (or any 1-D dataset)
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnData {
    /// Integers, `None` for missing or out of `i32` range
    Integer(Vec<Option<i32>>),
    /// Floats widened to `f64`; NaN stays NaN, `None` is NULL
    Double(Vec<Option<f64>>),
    /// Booleans
    Boolean(Vec<Option<bool>>),
    /// Strings
    String(Vec<Option<String>>),
    /// Categorical; a code outside `0..categories.len()` is NULL
    Categorical {
        /// Per-row level codes
        codes: Vec<i32>,
        /// Levels
        categories: Vec<String>,
    },
}

impl ColumnData {
    /// Number of rows
    ///
    /// ### Returns
    ///
    /// The column length.
    pub(crate) fn len(&self) -> usize {
        match self {
            ColumnData::Integer(v) => v.len(),
            ColumnData::Double(v) => v.len(),
            ColumnData::Boolean(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::Categorical { codes, .. } => codes.len(),
        }
    }

    /// Keep the rows at `keep`, in that order
    ///
    /// ### Params
    ///
    /// * `keep` - 0-based row positions, all `< self.len()`
    ///
    /// ### Returns
    ///
    /// The filtered column.
    #[cfg(feature = "parquet")]
    pub(crate) fn gather(self, keep: &[usize]) -> Self {
        fn pick<T: Clone>(v: &[T], keep: &[usize]) -> Vec<T> {
            keep.iter().map(|&i| v[i].clone()).collect()
        }
        match self {
            ColumnData::Integer(v) => ColumnData::Integer(pick(&v, keep)),
            ColumnData::Double(v) => ColumnData::Double(pick(&v, keep)),
            ColumnData::Boolean(v) => ColumnData::Boolean(pick(&v, keep)),
            ColumnData::String(v) => ColumnData::String(pick(&v, keep)),
            ColumnData::Categorical { codes, categories } => ColumnData::Categorical {
                codes: pick(&codes, keep),
                categories,
            },
        }
    }

    /// Set the rows flagged in `mask` to NULL (anndata nullable encoding)
    ///
    /// ### Params
    ///
    /// * `mask` - `Some(true)` marks a missing value
    fn apply_mask(&mut self, mask: &[Option<bool>]) {
        let missing = |i: usize| mask.get(i).copied().flatten().unwrap_or(false);
        match self {
            ColumnData::Integer(v) => mask_vec(v, missing),
            ColumnData::Double(v) => mask_vec(v, missing),
            ColumnData::Boolean(v) => mask_vec(v, missing),
            ColumnData::String(v) => mask_vec(v, missing),
            ColumnData::Categorical { codes, .. } => {
                for (i, c) in codes.iter_mut().enumerate() {
                    if missing(i) {
                        *c = -1;
                    }
                }
            }
        }
    }

    /// Render the values as categorical levels, the way R's `factor()` turns
    /// non-character levels into strings
    ///
    /// ### Returns
    ///
    /// The levels, or `None` for a categorical of categoricals.
    pub(crate) fn into_levels(self) -> Option<Vec<String>> {
        fn render<T>(v: Vec<Option<T>>, f: impl Fn(T) -> String) -> Vec<String> {
            v.into_iter()
                .map(|x| x.map(&f).unwrap_or_else(|| "NA".to_string()))
                .collect()
        }
        match self {
            ColumnData::Integer(v) => Some(render(v, |x| x.to_string())),
            ColumnData::Double(v) => Some(render(v, |x| x.to_string())),
            ColumnData::Boolean(v) => {
                Some(render(v, |x| if x { "TRUE" } else { "FALSE" }.to_string()))
            }
            ColumnData::String(v) => Some(render(v, |x| x)),
            ColumnData::Categorical { .. } => None,
        }
    }
}

/// Null out the masked entries of an optional vector
///
/// ### Params
///
/// * `v` - Values
/// * `missing` - Whether row `i` is missing
fn mask_vec<T>(v: &mut [Option<T>], missing: impl Fn(usize) -> bool) {
    for (i, x) in v.iter_mut().enumerate() {
        if missing(i) {
            *x = None;
        }
    }
}

/// An h5ad dataframe group as read, before renaming and filtering
pub struct H5adFrame {
    /// Columns in R's order (groups, then datasets, each by name)
    pub cols: Vec<(String, ColumnData)>,
    /// The index column, if the group has one
    pub idx: Option<ColumnData>,
    /// Columns left out for an unsupported encoding
    pub skipped: Vec<String>,
}

/////////////
// Readers //
/////////////

/// Convert an HDF5 integer to an R integer: out of range, or R's NA sentinel
/// `i32::MIN`, becomes missing
///
/// ### Params
///
/// * `x` - The value
///
/// ### Returns
///
/// The value as `i32`, or `None`.
#[inline]
fn to_r_int<T: TryInto<i32>>(x: T) -> Option<i32> {
    x.try_into().ok().filter(|&v| v != i32::MIN)
}

/// Read a plain 1-D dataset
///
/// HDF5 enums with only `FALSE` / `TRUE` members are booleans (anndata writes
/// no other enums).
///
/// ### Params
///
/// * `ds` - The dataset
///
/// ### Returns
///
/// The column, or `None` for a dtype or shape R would not turn into a vector.
fn read_dataset(ds: &Dataset) -> Result<Option<ColumnData>, ScDataErrors> {
    if ds.ndim() != 1 {
        return Ok(None);
    }
    read_values(ds)
}

/// Dispatch a dataset or attribute on its dtype, ignoring its shape
///
/// ### Params
///
/// * `ds` - The dataset or attribute
///
/// ### Returns
///
/// The values flattened, or `None` for an unsupported dtype.
fn read_values(ds: &impl Deref<Target = Container>) -> Result<Option<ColumnData>, ScDataErrors> {
    let col = match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(_) => {
            ColumnData::Integer(ds.read_raw::<i64>()?.into_iter().map(to_r_int).collect())
        }
        TypeDescriptor::Unsigned(_) => {
            ColumnData::Integer(ds.read_raw::<u64>()?.into_iter().map(to_r_int).collect())
        }
        TypeDescriptor::Float(_) => {
            ColumnData::Double(ds.read_raw::<f64>()?.into_iter().map(Some).collect())
        }
        TypeDescriptor::Boolean => {
            ColumnData::Boolean(ds.read_raw::<bool>()?.into_iter().map(Some).collect())
        }
        TypeDescriptor::Enum(e)
            if e.members
                .iter()
                .all(|m| m.name == "FALSE" || m.name == "TRUE") =>
        {
            ColumnData::Boolean(ds.read_raw::<bool>()?.into_iter().map(Some).collect())
        }
        TypeDescriptor::VarLenUnicode
        | TypeDescriptor::VarLenAscii
        | TypeDescriptor::FixedAscii(_)
        | TypeDescriptor::FixedUnicode(_) => {
            ColumnData::String(read_string_dataset(ds)?.into_iter().map(Some).collect())
        }
        _ => return Ok(None),
    };
    Ok(Some(col))
}

/// Names of the members of `group` of one type, in HDF5 name order
///
/// ### Params
///
/// * `group` - The group
/// * `loc_type` - Keep members of this type
///
/// ### Returns
///
/// The member names.
fn members_of_type(group: &Group, loc_type: LocationType) -> Result<Vec<String>, ScDataErrors> {
    let mut out = Vec::new();
    for name in group.member_names_by(IndexType::Name, IterationOrder::Increasing)? {
        if group.loc_type_by_name(&name)? == loc_type {
            out.push(name);
        }
    }
    Ok(out)
}

/// Read one dataframe column: a plain
/// dataset, a categorical group (`categories` + `codes`, negative code is
/// NA) or a nullable group (`values` + optional `mask`)
///
/// ### Params
///
/// * `group` - The dataframe group
/// * `name` - Column name inside it
///
/// ### Returns
///
/// The column, or `None` for an unsupported encoding.
fn read_column(group: &Group, name: &str) -> Result<Option<ColumnData>, ScDataErrors> {
    match group.loc_type_by_name(name)? {
        LocationType::Dataset => read_dataset(&group.dataset(name)?),
        LocationType::Group => {
            let g = group.group(name)?;
            let subs = members_of_type(&g, LocationType::Dataset)?;
            let has = |s: &str| subs.iter().any(|x| x == s);

            if has("categories") && has("codes") {
                let Some(categories) =
                    read_dataset(&g.dataset("categories")?)?.and_then(ColumnData::into_levels)
                else {
                    return Ok(None);
                };
                let Some(ColumnData::Integer(codes)) = read_dataset(&g.dataset("codes")?)? else {
                    return Ok(None);
                };
                let codes = codes.into_iter().map(|c| c.unwrap_or(-1)).collect();
                return Ok(Some(ColumnData::Categorical { codes, categories }));
            }

            if has("values") {
                let Some(mut values) = read_dataset(&g.dataset("values")?)? else {
                    return Ok(None);
                };
                if has("mask") {
                    let Some(ColumnData::Boolean(mask)) = read_dataset(&g.dataset("mask")?)? else {
                        return Ok(None);
                    };
                    values.apply_mask(&mask);
                }
                return Ok(Some(values));
            }

            Ok(None)
        }
        _ => Ok(None),
    }
}

/// Read an h5ad dataframe group
///
/// The index is the column named by the `_index` attribute, else an `_index`
/// dataset. Old-style categoricals (integer codes in a dataset, levels under
/// `__categories/<name>`) are resolved.
///
/// ### Params
///
/// * `group` - The dataframe group
///
/// ### Returns
///
/// The [`H5adFrame`].
pub(crate) fn read_frame(group: &Group) -> Result<H5adFrame, ScDataErrors> {
    let mut skip: Vec<String> = vec!["_index".into(), "__categories".into()];

    let idx = if group.attr_names()?.iter().any(|a| a == "_index") {
        let idx_col = read_string_dataset(&group.attr("_index")?)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                ScDataErrors::UnsupportH5ADFormat("empty _index attribute".to_string())
            })?;
        let idx = read_column(group, &idx_col)?;
        skip.push(idx_col);
        idx
    } else if group.link_exists("_index")
        && group.loc_type_by_name("_index")? == LocationType::Dataset
    {
        read_dataset(&group.dataset("_index")?)?
    } else {
        None
    };

    let old_cats: Vec<String> = if group.link_exists("__categories")
        && group.loc_type_by_name("__categories")? == LocationType::Group
    {
        members_of_type(&group.group("__categories")?, LocationType::Dataset)?
    } else {
        Vec::new()
    };

    let mut names = members_of_type(group, LocationType::Group)?;
    names.extend(members_of_type(group, LocationType::Dataset)?);
    names.retain(|n| !skip.contains(n));

    let mut cols = Vec::with_capacity(names.len());
    let mut skipped = Vec::new();
    for name in names {
        let Some(mut col) = read_column(group, &name)? else {
            skipped.push(name);
            continue;
        };
        if old_cats.contains(&name)
            && let ColumnData::Integer(codes) = &col
        {
            let categories = read_dataset(&group.dataset(&format!("__categories/{name}"))?)?
                .and_then(ColumnData::into_levels);
            if let Some(categories) = categories {
                col = ColumnData::Categorical {
                    codes: codes.iter().map(|c| c.unwrap_or(-1)).collect(),
                    categories,
                };
            }
        }
        cols.push((name, col));
    }

    Ok(H5adFrame { cols, idx, skipped })
}

/////////////
// Exports //
/////////////

/// Split an HDF5 path into the parent group and the member name
///
/// ### Params
///
/// * `file` - The open file
/// * `path` - Path of the member, with or without a leading `/`
///
/// ### Returns
///
/// The parent group and the member name.
fn parent_and_name<'a>(file: &File, path: &'a str) -> Result<(Group, &'a str), ScDataErrors> {
    let path = path.trim_start_matches('/');
    match path.rsplit_once('/') {
        Some((parent, name)) => Ok((file.group(parent)?, name)),
        None => Ok((file.group("/")?, path)),
    }
}

/// Read an h5ad dataframe group (`obs`, `var` or any other) into memory
///
/// ### Params
///
/// * `h5_path` - The h5ad file
/// * `group` - Path of the dataframe group
///
/// ### Returns
///
/// The decoded [`H5adFrame`].
pub fn read_h5ad_frame<P: AsRef<Path>>(h5_path: P, group: &str) -> Result<H5adFrame, ScDataErrors> {
    let file = File::open(h5_path)?;
    read_frame(&file.group(group)?)
}

/// Read one 1-D dataset or encoded column (categorical, nullable) at any path
///
/// ### Params
///
/// * `h5_path` - The HDF5 file
/// * `path` - Path of the dataset or column group
///
/// ### Returns
///
/// The column, or `None` for an unsupported dtype, shape or encoding.
pub fn read_h5_column<P: AsRef<Path>>(
    h5_path: P,
    path: &str,
) -> Result<Option<ColumnData>, ScDataErrors> {
    let file = File::open(h5_path)?;
    let (parent, name) = parent_and_name(&file, path)?;
    if parent.loc_type_by_name(name)? == LocationType::Dataset {
        let ds = parent.dataset(name)?;
        if ds.ndim() == 0 {
            return read_values(&ds);
        }
    }
    read_column(&parent, name)
}

/// One member of an HDF5 group
#[derive(Clone, Debug)]
pub struct H5Member {
    /// Member name
    pub name: String,
    /// Group (`true`) or dataset (`false`)
    pub is_group: bool,
    /// Dataset shape; empty for groups and scalars
    pub shape: Vec<usize>,
}

/// List the groups and datasets directly inside a group
///
/// ### Params
///
/// * `h5_path` - The HDF5 file
/// * `group` - Path of the group; `"/"` for the root
///
/// ### Returns
///
/// The members in HDF5 name order. Other link types are left out.
pub fn list_h5_group<P: AsRef<Path>>(
    h5_path: P,
    group: &str,
) -> Result<Vec<H5Member>, ScDataErrors> {
    let file = File::open(h5_path)?;
    let g = file.group(group)?;
    let mut out = Vec::new();
    for name in g.member_names_by(IndexType::Name, IterationOrder::Increasing)? {
        match g.loc_type_by_name(&name)? {
            LocationType::Group => out.push(H5Member {
                name,
                is_group: true,
                shape: Vec::new(),
            }),
            LocationType::Dataset => {
                let shape = g.dataset(&name)?.shape();
                out.push(H5Member {
                    name,
                    is_group: false,
                    shape,
                });
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Read every attribute of a group or dataset
///
/// Scalar attributes come back as length-one columns. Attributes of an
/// unsupported dtype are left out.
///
/// ### Params
///
/// * `h5_path` - The HDF5 file
/// * `path` - Path of the group or dataset; `"/"` for the root
///
/// ### Returns
///
/// `(name, values)` pairs in HDF5 name order.
pub fn read_h5_attrs<P: AsRef<Path>>(
    h5_path: P,
    path: &str,
) -> Result<Vec<(String, ColumnData)>, ScDataErrors> {
    let file = File::open(h5_path)?;
    let trimmed = path.trim_start_matches('/');
    let loc: Location = if trimmed.is_empty() {
        file.group("/")?.deref().clone()
    } else {
        match file.loc_type_by_name(trimmed)? {
            LocationType::Dataset => file.dataset(trimmed)?.deref().deref().clone(),
            _ => file.group(trimmed)?.deref().clone(),
        }
    };
    let mut out = Vec::new();
    for name in loc.attr_names()? {
        if let Some(col) = read_values(&loc.attr(&name)?)? {
            out.push((name, col));
        }
    }
    Ok(out)
}

/// Read a 2-D numeric dataset as `f64`
///
/// ### Params
///
/// * `h5_path` - The HDF5 file
/// * `path` - Path of the dataset
///
/// ### Returns
///
/// The values in row-major order with the number of rows and columns.
pub fn read_h5_matrix<P: AsRef<Path>>(
    h5_path: P,
    path: &str,
) -> Result<(Vec<f64>, usize, usize), ScDataErrors> {
    let file = File::open(h5_path)?;
    let ds = file.dataset(path.trim_start_matches('/'))?;
    let shape = ds.shape();
    if shape.len() != 2 {
        return Err(ScDataErrors::InvalidArgument(format!(
            "{path} has {} dimensions, expected 2",
            shape.len()
        )));
    }
    Ok((ds.read_raw::<f64>()?, shape[0], shape[1]))
}

////////////
// Writer //
////////////

/// The `X` matrix for [`write_h5ad_plain`]
pub enum PlainX<'a> {
    /// Cell-major sparse: `indptr` over cells, `indices` are genes
    Csr {
        /// Non-zero values
        data: &'a [f64],
        /// 0-based minor indices
        indices: &'a [i32],
        /// 0-based major offsets
        indptr: &'a [i32],
    },
    /// Gene-major sparse: `indptr` over genes, `indices` are cells
    Csc {
        /// Non-zero values
        data: &'a [f64],
        /// 0-based minor indices
        indices: &'a [i32],
        /// 0-based major offsets
        indptr: &'a [i32],
    },
    /// Dense, cells x genes in row-major order
    Dense(&'a [f64]),
}

/// Write one dataframe column as a plain dataset, or a categorical group
///
/// ### Params
///
/// * `g` - The dataframe group
/// * `name` - Column name
/// * `col` - The values
fn write_column(g: &Group, name: &str, col: &ColumnData) -> Result<(), ScDataErrors> {
    let strings = |v: &[Option<String>]| -> Result<Vec<VarLenUnicode>, ScDataErrors> {
        v.iter()
            .map(|s| {
                VarLenUnicode::from_str(s.as_deref().unwrap_or("NA"))
                    .map_err(|e| ScDataErrors::InvalidArgument(format!("{name}: {e}")))
            })
            .collect()
    };
    match col {
        ColumnData::Integer(v) => {
            let v: Vec<i32> = v.iter().map(|x| x.unwrap_or(i32::MIN)).collect();
            g.new_dataset_builder().with_data(&v).create(name)?;
        }
        ColumnData::Double(v) => {
            let v: Vec<f64> = v.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
            g.new_dataset_builder().with_data(&v).create(name)?;
        }
        ColumnData::Boolean(v) => {
            let v = v
                .iter()
                .map(|x| {
                    x.ok_or_else(|| {
                        ScDataErrors::InvalidArgument(format!(
                            "{name}: missing values in a logical column are not supported"
                        ))
                    })
                })
                .collect::<Result<Vec<bool>, _>>()?;
            g.new_dataset_builder().with_data(&v).create(name)?;
        }
        ColumnData::String(v) => {
            g.new_dataset_builder()
                .with_data(&strings(v)?)
                .create(name)?;
        }
        ColumnData::Categorical { codes, categories } => {
            let cg = g.create_group(name)?;
            let cats: Vec<Option<String>> = categories.iter().cloned().map(Some).collect();
            cg.new_dataset_builder()
                .with_data(&strings(&cats)?)
                .create("categories")?;
            cg.new_dataset_builder().with_data(codes).create("codes")?;
        }
    }
    Ok(())
}

/// Write a minimal h5ad file: `X` plus `obs` / `var` as plain datasets
///
/// No anndata encoding attributes are written; this is the bare layout older
/// tools produce, used for test fixtures. The first column of `obs` and `var`
/// becomes `_index`.
///
/// ### Params
///
/// * `h5_path` - Output file, overwritten
/// * `x` - The count matrix
/// * `n_obs` - Number of cells
/// * `n_var` - Number of genes
/// * `obs` - Cell columns, identifier first
/// * `var` - Gene columns, identifier first
pub fn write_h5ad_plain<P: AsRef<Path>>(
    h5_path: P,
    x: PlainX,
    n_obs: usize,
    n_var: usize,
    obs: &[(String, ColumnData)],
    var: &[(String, ColumnData)],
) -> Result<(), ScDataErrors> {
    let file = File::create(h5_path)?;

    match x {
        PlainX::Csr {
            data,
            indices,
            indptr,
        }
        | PlainX::Csc {
            data,
            indices,
            indptr,
        } => {
            let g = file.create_group("X")?;
            g.new_dataset_builder().with_data(data).create("data")?;
            g.new_dataset_builder()
                .with_data(indices)
                .create("indices")?;
            g.new_dataset_builder().with_data(indptr).create("indptr")?;
        }
        PlainX::Dense(data) => {
            let view = ArrayView2::from_shape((n_obs, n_var), data)
                .map_err(|e| ScDataErrors::InvalidArgument(format!("X: {e}")))?;
            file.new_dataset_builder().with_data(&view).create("X")?;
        }
    }

    for (group, cols, n) in [("obs", obs, n_obs), ("var", var, n_var)] {
        let g = file.create_group(group)?;
        for (i, (name, col)) in cols.iter().enumerate() {
            if col.len() != n {
                return Err(ScDataErrors::InvalidArgument(format!(
                    "{group}/{name} has {} rows, expected {n}",
                    col.len()
                )));
            }
            write_column(&g, if i == 0 { "_index" } else { name }, col)?;
        }
    }
    Ok(())
}
