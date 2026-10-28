//! Parquet export of the cell and feature metadata: the h5ad `obs` / `var`
//! dataframes and the 10x barcodes and features. DuckDB ingests the file with
//! `read_parquet`, which replaces reading every column into R.
//!
//! The table matches what the R package built before: an `*_idx` column
//! (1-based, after filtering), the `*_id` column, then every other column
//! snake_cased, in HDF5 name order with column groups ahead of plain datasets.
//! Integers land as INT32 (out of range values as NULL, as rhdf5 does),
//! floats as DOUBLE, booleans as BOOLEAN, strings as UTF-8. Categoricals are
//! written as strings; their levels come back in [`ParquetColumnInfo`] so the
//! caller can cast them to a DuckDB ENUM.

use hdf5::{Dataset, File, Group, IndexType, IterationOrder, LocationType, types::TypeDescriptor};
use parquet::basic::{Compression, LogicalType, Repetition, Type as PhysicalType, ZstdLevel};
use parquet::data_type::{BoolType, ByteArray, ByteArrayType, DataType, DoubleType, Int32Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::{SerializedColumnWriter, SerializedFileWriter};
use parquet::schema::types::Type;
use rustc_hash::FxHashSet;
use std::path::Path;
use std::sync::Arc;

use crate::h5_10x_io::{TenxVersion, read_string_dataset};
use crate::prelude::*;

/// Rows per Parquet row group. DuckDB parallelises a scan over row groups, so
/// a multi-million row table should not land in one. Not tuned.
const ROW_GROUP_ROWS: usize = 1 << 20;

/// zstd level for the column chunks. The file is read once and deleted, so the
/// fastest level is the right trade.
const ZSTD_LEVEL: i32 = 1;

///////////
// Types //
///////////

/// Which metadata table is being exported. Fixes the HDF5 group and the names
/// of the index and identifier columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    /// Cells: `obs`, `cell_idx`, `cell_id`
    Obs,
    /// Features: `var`, `gene_idx`, `gene_id`
    Var,
}

impl FrameKind {
    /// HDF5 group of the h5ad dataframe
    ///
    /// ### Returns
    ///
    /// `"obs"` or `"var"`.
    pub fn group(&self) -> &'static str {
        match self {
            FrameKind::Obs => "obs",
            FrameKind::Var => "var",
        }
    }

    /// Name of the 1-based row index column
    ///
    /// ### Returns
    ///
    /// `"cell_idx"` or `"gene_idx"`.
    pub fn idx_name(&self) -> &'static str {
        match self {
            FrameKind::Obs => "cell_idx",
            FrameKind::Var => "gene_idx",
        }
    }

    /// Name of the identifier column
    ///
    /// ### Returns
    ///
    /// `"cell_id"` or `"gene_id"`.
    pub fn id_name(&self) -> &'static str {
        match self {
            FrameKind::Obs => "cell_id",
            FrameKind::Var => "gene_id",
        }
    }
}

/// Parse a frame kind
///
/// ### Params
///
/// * `s` - `"obs"` or `"var"`
///
/// ### Returns
///
/// The [`FrameKind`], or `None` for anything else.
pub fn parse_frame_kind(s: &str) -> Option<FrameKind> {
    match s {
        "obs" => Some(FrameKind::Obs),
        "var" => Some(FrameKind::Var),
        _ => None,
    }
}

/// Type of a column as written to Parquet
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParquetColumnType {
    /// INT32
    Integer,
    /// DOUBLE
    Double,
    /// BOOLEAN
    Boolean,
    /// UTF-8 string
    String,
    /// UTF-8 string holding a categorical; levels in
    /// [`ParquetColumnInfo::categories`]
    Categorical,
}

/// One column of the written table
#[derive(Clone, Debug)]
pub struct ParquetColumnInfo {
    /// Column name in the Parquet file
    pub name: String,
    /// Column type
    pub col_type: ParquetColumnType,
    /// Levels in order for a categorical (unused ones included), else `None`
    pub categories: Option<Vec<String>>,
}

/// Summary of a written table
#[derive(Clone, Debug)]
pub struct ParquetFrameInfo {
    /// Columns in file order
    pub columns: Vec<ParquetColumnInfo>,
    /// Columns of the source group left out for an unsupported encoding
    pub skipped: Vec<String>,
    /// Number of rows written
    pub n_rows: usize,
}

/// A column held in memory between the HDF5 read and the Parquet write
#[derive(Clone, Debug, PartialEq)]
enum ColumnData {
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
    fn len(&self) -> usize {
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
    fn gather(self, keep: &[usize]) -> Self {
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
    fn into_levels(self) -> Option<Vec<String>> {
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

    /// Type of the column as written
    ///
    /// ### Returns
    ///
    /// The matching [`ParquetColumnType`].
    fn col_type(&self) -> ParquetColumnType {
        match self {
            ColumnData::Integer(_) => ParquetColumnType::Integer,
            ColumnData::Double(_) => ParquetColumnType::Double,
            ColumnData::Boolean(_) => ParquetColumnType::Boolean,
            ColumnData::String(_) => ParquetColumnType::String,
            ColumnData::Categorical { .. } => ParquetColumnType::Categorical,
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
struct RawFrame {
    /// Columns in R's order (groups, then datasets, each by name)
    cols: Vec<(String, ColumnData)>,
    /// The index column, if the group has one
    idx: Option<ColumnData>,
    /// Columns left out for an unsupported encoding
    skipped: Vec<String>,
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

/// Read a plain 1-D dataset, mirroring `.read_h5ad_dataset` in R
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

/// Names of the members of `group` of one type, in HDF5 name order (what
/// `rhdf5::h5ls` lists)
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

/// Read one dataframe column, mirroring `.read_h5ad_column` in R: a plain
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

/// Read an h5ad dataframe group, mirroring `.read_h5ad_frame` and
/// `.resolve_h5_index` in R
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
/// The [`RawFrame`].
fn read_frame(group: &Group) -> Result<RawFrame, ScDataErrors> {
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

    Ok(RawFrame { cols, idx, skipped })
}

//////////////
// Assembly //
//////////////

/// R's `to_snake_case`: split lower-upper boundaries, lowercase, collapse
/// every run outside `[a-z0-9]` to one underscore, trim underscores
///
/// ### Params
///
/// * `x` - The name
///
/// ### Returns
///
/// The snake_cased name.
pub fn to_snake_case(x: &str) -> String {
    let chars: Vec<char> = x.chars().collect();
    let mut split = String::with_capacity(x.len() + 4);
    let mut i = 0;
    while i < chars.len() {
        split.push(chars[i]);
        // non-overlapping like gsub("([a-z])([A-Z])", "\\1_\\2"): the upper
        // case letter is consumed with its partner
        if chars[i].is_ascii_lowercase() && chars.get(i + 1).is_some_and(|c| c.is_ascii_uppercase())
        {
            split.push('_');
            split.push(chars[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    let mut out = String::with_capacity(split.len());
    let mut in_gap = false;
    for c in split.to_lowercase().chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            in_gap = false;
        } else if !in_gap {
            out.push('_');
            in_gap = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// Rows of a set of columns, checking they agree
///
/// ### Params
///
/// * `cols` - The columns
///
/// ### Returns
///
/// The common length (0 for no columns).
fn common_len(cols: &[(String, ColumnData)]) -> Result<usize, ScDataErrors> {
    let n = cols.first().map(|(_, c)| c.len()).unwrap_or(0);
    if let Some((name, c)) = cols.iter().find(|(_, c)| c.len() != n) {
        return Err(ScDataErrors::UnsupportH5ADFormat(format!(
            "column '{}' has {} rows, expected {}",
            name,
            c.len(),
            n
        )));
    }
    Ok(n)
}

/// Filter the rows, then put a fresh 1-based `idx_name` column in front
///
/// A column already called `idx_name` is replaced, as `dt[, cell_idx := .I]`
/// does in R. The identifier must already be the first column.
///
/// ### Params
///
/// * `cols` - Columns, identifier first
/// * `keep` - Optional 0-based row positions to keep, in output order
/// * `idx_name` - Name of the index column
///
/// ### Returns
///
/// The final columns and the row count.
fn filter_and_index(
    cols: Vec<(String, ColumnData)>,
    keep: Option<&[usize]>,
    idx_name: &str,
) -> Result<(Vec<(String, ColumnData)>, usize), ScDataErrors> {
    let n = common_len(&cols)?;
    let mut cols = match keep {
        None => cols,
        Some(keep) => {
            if let Some(&bad) = keep.iter().find(|&&i| i >= n) {
                return Err(ScDataErrors::InvalidArgument(format!(
                    "row index {} out of range for {} rows",
                    bad, n
                )));
            }
            cols.into_iter()
                .map(|(name, c)| (name, c.gather(keep)))
                .collect()
        }
    };
    let n_out = keep.map_or(n, <[usize]>::len);

    cols.retain(|(name, _)| name != idx_name);
    let idx = (1..=n_out).map(|i| i32::try_from(i).ok()).collect();
    cols.insert(0, (idx_name.to_string(), ColumnData::Integer(idx)));
    Ok((cols, n_out))
}

/// Rename and order an h5ad dataframe like `populate_obs_from_h5ad` /
/// `populate_vars_from_h5ad` in R
///
/// With an index, it becomes the identifier (replacing any column of that
/// name). Without one, `id_col` or else the first column is renamed to the
/// identifier. Every other column is snake_cased.
///
/// ### Params
///
/// * `frame` - The frame as read
/// * `kind` - Obs or var
/// * `id_col` - Optional column to use as identifier when there is no index
///
/// ### Returns
///
/// The columns, identifier first.
fn assemble_h5ad(
    frame: RawFrame,
    kind: FrameKind,
    id_col: Option<&str>,
) -> Result<Vec<(String, ColumnData)>, ScDataErrors> {
    let id_name = kind.id_name();
    let mut cols = frame.cols;

    if cols.is_empty() && frame.idx.is_none() {
        return Err(ScDataErrors::UnsupportH5ADFormat(format!(
            "no {} data found in the h5 file",
            kind.group()
        )));
    }

    let id = match frame.idx {
        Some(idx) => {
            cols.retain(|(name, _)| name != id_name);
            idx
        }
        None => {
            let pos = match id_col {
                Some(c) => cols.iter().position(|(name, _)| name == c).ok_or_else(|| {
                    ScDataErrors::InvalidArgument(format!(
                        "column '{}' not found in {}",
                        c,
                        kind.group()
                    ))
                })?,
                None => 0,
            };
            cols.remove(pos).1
        }
    };

    let mut out = Vec::with_capacity(cols.len() + 1);
    out.push((id_name.to_string(), id));
    for (name, col) in cols {
        let name = if name == id_name {
            name
        } else {
            to_snake_case(&name)
        };
        out.push((name, col));
    }
    Ok(out)
}

////////////
// Writer //
////////////

/// Write one column chunk of optional values with definition levels
///
/// ### Params
///
/// * `cw` - The column writer
/// * `vals` - Values, `None` for NULL
fn write_optional<T: DataType>(
    cw: &mut SerializedColumnWriter<'_>,
    vals: impl Iterator<Item = Option<T::T>>,
) -> Result<(), ScDataErrors> {
    let (lo, _) = vals.size_hint();
    let mut values = Vec::with_capacity(lo);
    let mut def = Vec::with_capacity(lo);
    for v in vals {
        match v {
            Some(x) => {
                values.push(x);
                def.push(1i16);
            }
            None => def.push(0),
        }
    }
    cw.typed::<T>().write_batch(&values, Some(&def), None)?;
    Ok(())
}

/// Write the columns to a Parquet file, all OPTIONAL, zstd, row groups of
/// [`ROW_GROUP_ROWS`]
///
/// ### Params
///
/// * `cols` - Final columns, all of length `n_rows`
/// * `n_rows` - Row count
/// * `out_path` - Output file
fn write_parquet(
    cols: &[(String, ColumnData)],
    n_rows: usize,
    out_path: &Path,
) -> Result<(), ScDataErrors> {
    let fields = cols
        .iter()
        .map(|(name, col)| {
            let (physical, logical) = match col.col_type() {
                ParquetColumnType::Integer => (PhysicalType::INT32, None),
                ParquetColumnType::Double => (PhysicalType::DOUBLE, None),
                ParquetColumnType::Boolean => (PhysicalType::BOOLEAN, None),
                ParquetColumnType::String | ParquetColumnType::Categorical => {
                    (PhysicalType::BYTE_ARRAY, Some(LogicalType::String))
                }
            };
            Type::primitive_type_builder(name, physical)
                .with_repetition(Repetition::OPTIONAL)
                .with_logical_type(logical)
                .build()
                .map(Arc::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let schema = Arc::new(
        Type::group_type_builder("schema")
            .with_fields(fields)
            .build()?,
    );
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(ZSTD_LEVEL)?))
            .build(),
    );

    // byte arrays of the levels once; `ByteArray` clones are refcounted
    let levels: Vec<Option<Vec<ByteArray>>> = cols
        .iter()
        .map(|(_, c)| match c {
            ColumnData::Categorical { categories, .. } => Some(
                categories
                    .iter()
                    .map(|s| ByteArray::from(s.as_str()))
                    .collect(),
            ),
            _ => None,
        })
        .collect();

    let file = std::fs::File::create(out_path)?;
    let mut writer = SerializedFileWriter::new(file, schema, props)?;
    let mut start = 0;
    // one (possibly empty) row group even for zero rows, so the schema is read
    while start < n_rows || start == 0 {
        let end = (start + ROW_GROUP_ROWS).min(n_rows);
        let mut rg = writer.next_row_group()?;
        for ((_, col), lv) in cols.iter().zip(&levels) {
            let mut cw = rg.next_column()?.expect("schema has one leaf per column");
            match col {
                ColumnData::Integer(v) => {
                    write_optional::<Int32Type>(&mut cw, v[start..end].iter().copied())?
                }
                ColumnData::Double(v) => {
                    write_optional::<DoubleType>(&mut cw, v[start..end].iter().copied())?
                }
                ColumnData::Boolean(v) => {
                    write_optional::<BoolType>(&mut cw, v[start..end].iter().copied())?
                }
                ColumnData::String(v) => write_optional::<ByteArrayType>(
                    &mut cw,
                    v[start..end]
                        .iter()
                        .map(|s| s.as_deref().map(ByteArray::from)),
                )?,
                ColumnData::Categorical { codes, .. } => {
                    let lv = lv.as_ref().expect("levels built for every categorical");
                    write_optional::<ByteArrayType>(
                        &mut cw,
                        codes[start..end]
                            .iter()
                            .map(|&c| usize::try_from(c).ok().and_then(|c| lv.get(c)).cloned()),
                    )?
                }
            }
            cw.close()?;
        }
        rg.close()?;
        if end == n_rows {
            break;
        }
        start = end;
    }
    writer.close()?;
    Ok(())
}

/// Check names are unique, write the file and describe it
///
/// ### Params
///
/// * `cols` - Final columns
/// * `n_rows` - Row count
/// * `skipped` - Source columns left out
/// * `out_path` - Output file
///
/// ### Returns
///
/// The [`ParquetFrameInfo`].
fn finish(
    cols: Vec<(String, ColumnData)>,
    n_rows: usize,
    skipped: Vec<String>,
    out_path: &Path,
) -> Result<ParquetFrameInfo, ScDataErrors> {
    // DuckDB identifiers are case-insensitive
    let mut seen = FxHashSet::default();
    if let Some((name, _)) = cols.iter().find(|(n, _)| !seen.insert(n.to_lowercase())) {
        return Err(ScDataErrors::DuplicateColumnName(name.clone()));
    }

    write_parquet(&cols, n_rows, out_path)?;

    let columns = cols
        .into_iter()
        .map(|(name, col)| ParquetColumnInfo {
            name,
            col_type: col.col_type(),
            categories: match col {
                ColumnData::Categorical { categories, .. } => Some(categories),
                _ => None,
            },
        })
        .collect();
    Ok(ParquetFrameInfo {
        columns,
        skipped,
        n_rows,
    })
}

/////////////
// Exports //
/////////////

/// Write the `obs` or `var` dataframe of an h5ad file to Parquet
///
/// Reproduces the table `populate_obs_from_h5ad` / `populate_vars_from_h5ad`
/// in the R package built: `cell_idx` / `gene_idx` (1-based after
/// filtering), `cell_id` / `gene_id`, then the remaining columns snake_cased.
///
/// ### Params
///
/// * `h5_path` - The h5ad file
/// * `kind` - Obs or var
/// * `keep` - Optional 0-based rows to keep, in output order (the cell or
///   gene indices the count write kept)
/// * `id_col` - Optional column to use as identifier when the group has no
///   index; the first column otherwise. Ignored when there is an index
/// * `out_path` - Parquet file to write
///
/// ### Returns
///
/// The written columns with their types and categorical levels, the columns
/// skipped for an unsupported encoding, and the row count.
pub fn h5ad_frame_to_parquet<P: AsRef<Path>, Q: AsRef<Path>>(
    h5_path: P,
    kind: FrameKind,
    keep: Option<&[usize]>,
    id_col: Option<&str>,
    out_path: Q,
) -> Result<ParquetFrameInfo, ScDataErrors> {
    let file = File::open(h5_path)?;
    let frame = read_frame(&file.group(kind.group())?)?;
    let skipped = frame.skipped.clone();
    let cols = assemble_h5ad(frame, kind, id_col)?;
    let (cols, n_rows) = filter_and_index(cols, keep, kind.idx_name())?;
    finish(cols, n_rows, skipped, out_path.as_ref())
}

/// Write the barcodes (obs) or features (var) of a 10x CellRanger h5 file to
/// Parquet
///
/// Obs is `cell_idx`, `cell_id`. Var is `gene_idx`, `gene_id`, `gene_name`,
/// plus `feature_type` for v3. Features are read in file order across all
/// modalities, so `keep` indexes into that order.
///
/// ### Params
///
/// * `h5_path` - The 10x h5 file
/// * `version` - The [`TenxVersion`]
/// * `kind` - Obs or var
/// * `keep` - Optional 0-based rows to keep, in output order
/// * `out_path` - Parquet file to write
///
/// ### Returns
///
/// The written columns with their types and the row count.
pub fn tenx_frame_to_parquet<P: AsRef<Path>, Q: AsRef<Path>>(
    h5_path: P,
    version: TenxVersion,
    kind: FrameKind,
    keep: Option<&[usize]>,
    out_path: Q,
) -> Result<ParquetFrameInfo, ScDataErrors> {
    let file = File::open(h5_path)?;
    let strings = |path: &str| -> Result<ColumnData, ScDataErrors> {
        Ok(ColumnData::String(
            read_string_dataset(&file.dataset(path)?)?
                .into_iter()
                .map(Some)
                .collect(),
        ))
    };

    let mut cols = Vec::new();
    match kind {
        FrameKind::Obs => cols.push((kind.id_name().to_string(), strings(version.get_barcodes())?)),
        FrameKind::Var => {
            cols.push((
                kind.id_name().to_string(),
                strings(version.get_feature_id())?,
            ));
            if let Some(p) = version.get_feature_name() {
                cols.push(("gene_name".to_string(), strings(p)?));
            }
            if let Some(p) = version.get_feature_type() {
                cols.push(("feature_type".to_string(), strings(p)?));
            }
        }
    }

    let (cols, n_rows) = filter_and_index(cols, keep, kind.idx_name())?;
    finish(cols, n_rows, Vec::new(), out_path.as_ref())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::types::{FixedAscii, VarLenUnicode};
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::Field;
    use std::path::PathBuf;
    use std::str::FromStr;

    /// Unique temp path for a test file
    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("bixverse_obs_var_{}_{}", std::process::id(), name))
    }

    /// Variable-length UTF-8 strings
    fn vlen(xs: &[&str]) -> Vec<VarLenUnicode> {
        xs.iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect()
    }

    /// Write a vlen string dataset
    fn write_strings(g: &Group, name: &str, xs: &[&str]) {
        g.new_dataset_builder()
            .with_data(&vlen(xs))
            .create(name)
            .unwrap();
    }

    /// Read a Parquet file back into column names and rows of fields
    fn read_back(path: &Path) -> (Vec<String>, Vec<Vec<Field>>) {
        let reader = SerializedFileReader::try_from(path).unwrap();
        let names = reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        let rows = reader
            .get_row_iter(None)
            .unwrap()
            .map(|r| {
                r.unwrap()
                    .get_column_iter()
                    .map(|(_, f)| f.clone())
                    .collect()
            })
            .collect();
        (names, rows)
    }

    /// One column of the read-back rows
    fn column(names: &[String], rows: &[Vec<Field>], name: &str) -> Vec<Field> {
        let j = names.iter().position(|n| n == name).unwrap();
        rows.iter().map(|r| r[j].clone()).collect()
    }

    fn s(x: &str) -> Field {
        Field::Str(x.to_string())
    }

    /// An obs group with every encoding: `_index` attribute, new and old
    /// categoricals, nullable, float32, int32, int64 out of range, bool,
    /// plain and long fixed-length strings, and one unsupported group
    fn write_obs_fixture(path: &Path) {
        let file = File::create(path).unwrap();
        let obs = file.create_group("obs").unwrap();
        obs.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&VarLenUnicode::from_str("barcode").unwrap())
            .unwrap();
        write_strings(&obs, "barcode", &["c1", "c2", "c3", "c4"]);

        let ct = obs.create_group("cellType").unwrap();
        write_strings(&ct, "categories", &["T", "B", "NK"]);
        ct.new_dataset_builder()
            .with_data(&[0i8, -1, 2, 0])
            .create("codes")
            .unwrap();

        let ni = obs.create_group("nullable_int").unwrap();
        ni.new_dataset_builder()
            .with_data(&[1i32, 2, 3, 4])
            .create("values")
            .unwrap();
        ni.new_dataset_builder()
            .with_data(&[false, true, false, false])
            .create("mask")
            .unwrap();

        obs.create_group("weird")
            .unwrap()
            .new_dataset_builder()
            .with_data(&[1i32, 2, 3, 4])
            .create("foo")
            .unwrap();

        obs.new_dataset_builder()
            .with_data(&[1i32, 0, 1, 5])
            .create("batch")
            .unwrap();
        let cats = obs.create_group("__categories").unwrap();
        write_strings(&cats, "batch", &["a", "b"]);

        obs.new_dataset_builder()
            .with_data(&[1i64, 1 << 40, -3, 7])
            .create("big")
            .unwrap();

        let long = "x".repeat(100);
        let fixed: Vec<FixedAscii<100>> = [long.as_str(), "short", "", "y"]
            .iter()
            .map(|s| FixedAscii::<100>::from_ascii(s.as_bytes()).unwrap())
            .collect();
        obs.new_dataset_builder()
            .with_data(&fixed)
            .create("fixed_long")
            .unwrap();

        obs.new_dataset_builder()
            .with_data(&[true, false, true, true])
            .create("is_doublet")
            .unwrap();
        write_strings(&obs, "label", &["p", "q", "r", "s"]);
        obs.new_dataset_builder()
            .with_data(&[1.5f32, f32::NAN, 2.0, 3.0])
            .create("nCount_RNA")
            .unwrap();
        obs.new_dataset_builder()
            .with_data(&[10i32, 20, 30, 40])
            .create("nFeature")
            .unwrap();
    }

    #[test]
    fn test_snake_case_matches_r() {
        // reference: bixverse::to_snake_case() on the same inputs
        assert_eq!(to_snake_case("Gene Name"), "gene_name");
        assert_eq!(to_snake_case("someRNAValue"), "some_rnavalue");
        assert_eq!(to_snake_case("Foo-Bar"), "foo_bar");
        assert_eq!(to_snake_case("nCount_RNA"), "n_count_rna");
        assert_eq!(to_snake_case("__x__y__"), "x_y");
        assert_eq!(to_snake_case("aBcD"), "a_bc_d");
        assert_eq!(to_snake_case("pct.mito (%)"), "pct_mito");
    }

    #[test]
    fn test_h5ad_obs_all_encodings() {
        let h5 = tmp("obs_all.h5ad");
        let pq = tmp("obs_all.parquet");
        write_obs_fixture(&h5);

        let info = h5ad_frame_to_parquet(&h5, FrameKind::Obs, None, None, &pq).unwrap();
        let (names, rows) = read_back(&pq);
        let _ = std::fs::remove_file(&h5);
        let _ = std::fs::remove_file(&pq);

        assert_eq!(
            names,
            [
                "cell_idx",
                "cell_id",
                "cell_type",
                "nullable_int",
                "batch",
                "big",
                "fixed_long",
                "is_doublet",
                "label",
                "n_count_rna",
                "n_feature"
            ]
        );
        assert_eq!(info.skipped, ["weird"]);
        assert_eq!(info.n_rows, 4);

        let types: Vec<_> = info.columns.iter().map(|c| c.col_type).collect();
        use ParquetColumnType as P;
        assert_eq!(
            types,
            [
                P::Integer,
                P::String,
                P::Categorical,
                P::Integer,
                P::Categorical,
                P::Integer,
                P::String,
                P::Boolean,
                P::String,
                P::Double,
                P::Integer
            ]
        );
        assert_eq!(
            info.columns[2].categories.as_deref(),
            Some(&["T".to_string(), "B".into(), "NK".into()][..])
        );

        assert_eq!(
            column(&names, &rows, "cell_idx"),
            [Field::Int(1), Field::Int(2), Field::Int(3), Field::Int(4)]
        );
        assert_eq!(
            column(&names, &rows, "cell_id"),
            [s("c1"), s("c2"), s("c3"), s("c4")]
        );
        assert_eq!(
            column(&names, &rows, "cell_type"),
            [s("T"), Field::Null, s("NK"), s("T")]
        );
        assert_eq!(
            column(&names, &rows, "nullable_int"),
            [Field::Int(1), Field::Null, Field::Int(3), Field::Int(4)]
        );
        // old-style categorical; code 5 is past the levels, so NA as in R
        assert_eq!(
            column(&names, &rows, "batch"),
            [s("b"), s("a"), s("b"), Field::Null]
        );
        assert_eq!(
            column(&names, &rows, "big"),
            [Field::Int(1), Field::Null, Field::Int(-3), Field::Int(7)]
        );
        assert_eq!(
            column(&names, &rows, "fixed_long"),
            [s(&"x".repeat(100)), s("short"), s(""), s("y")]
        );
        assert_eq!(
            column(&names, &rows, "is_doublet"),
            [
                Field::Bool(true),
                Field::Bool(false),
                Field::Bool(true),
                Field::Bool(true)
            ]
        );
        let n_count = column(&names, &rows, "n_count_rna");
        assert_eq!(n_count[0], Field::Double(1.5));
        assert!(matches!(n_count[1], Field::Double(x) if x.is_nan()));
        assert_eq!(n_count[3], Field::Double(3.0));
    }

    #[test]
    fn test_h5ad_obs_filter_reindexes() {
        let h5 = tmp("obs_filter.h5ad");
        let pq = tmp("obs_filter.parquet");
        write_obs_fixture(&h5);

        let info = h5ad_frame_to_parquet(&h5, FrameKind::Obs, Some(&[3, 0]), None, &pq).unwrap();
        let (names, rows) = read_back(&pq);
        let _ = std::fs::remove_file(&h5);
        let _ = std::fs::remove_file(&pq);

        assert_eq!(info.n_rows, 2);
        assert_eq!(
            column(&names, &rows, "cell_idx"),
            [Field::Int(1), Field::Int(2)]
        );
        assert_eq!(column(&names, &rows, "cell_id"), [s("c4"), s("c1")]);
        assert_eq!(column(&names, &rows, "cell_type"), [s("T"), s("T")]);
        assert_eq!(
            column(&names, &rows, "n_feature"),
            [Field::Int(40), Field::Int(10)]
        );
        // levels survive the filter, unused ones included
        assert_eq!(info.columns[2].categories.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_h5ad_filter_out_of_range() {
        let h5 = tmp("obs_oob.h5ad");
        let pq = tmp("obs_oob.parquet");
        write_obs_fixture(&h5);
        let res = h5ad_frame_to_parquet(&h5, FrameKind::Obs, Some(&[4]), None, &pq);
        let _ = std::fs::remove_file(&h5);
        let _ = std::fs::remove_file(&pq);
        assert!(matches!(res, Err(ScDataErrors::InvalidArgument(_))));
    }

    #[test]
    fn test_h5ad_var_old_index_dataset() {
        let h5 = tmp("var_old.h5ad");
        let pq = tmp("var_old.parquet");
        {
            let file = File::create(&h5).unwrap();
            let var = file.create_group("var").unwrap();
            write_strings(&var, "_index", &["g1", "g2", "g3"]);
            write_strings(&var, "Gene Symbol", &["A", "B", "C"]);
        }
        let info = h5ad_frame_to_parquet(&h5, FrameKind::Var, Some(&[2, 1]), None, &pq).unwrap();
        let (names, rows) = read_back(&pq);
        let _ = std::fs::remove_file(&h5);
        let _ = std::fs::remove_file(&pq);

        assert_eq!(names, ["gene_idx", "gene_id", "gene_symbol"]);
        assert!(info.skipped.is_empty());
        assert_eq!(column(&names, &rows, "gene_id"), [s("g3"), s("g2")]);
        assert_eq!(column(&names, &rows, "gene_symbol"), [s("C"), s("B")]);
    }

    #[test]
    fn test_h5ad_no_index_id_col_and_first_column() {
        let h5 = tmp("obs_noidx.h5ad");
        let pq = tmp("obs_noidx.parquet");
        {
            let file = File::create(&h5).unwrap();
            let obs = file.create_group("obs").unwrap();
            write_strings(&obs, "aaa", &["x", "y"]);
            write_strings(&obs, "myBarcode", &["b1", "b2"]);
        }

        h5ad_frame_to_parquet(&h5, FrameKind::Obs, None, Some("myBarcode"), &pq).unwrap();
        let (names, rows) = read_back(&pq);
        assert_eq!(names, ["cell_idx", "cell_id", "aaa"]);
        assert_eq!(column(&names, &rows, "cell_id"), [s("b1"), s("b2")]);

        h5ad_frame_to_parquet(&h5, FrameKind::Obs, None, None, &pq).unwrap();
        let (names, rows) = read_back(&pq);
        assert_eq!(names, ["cell_idx", "cell_id", "my_barcode"]);
        assert_eq!(column(&names, &rows, "cell_id"), [s("x"), s("y")]);

        let missing = h5ad_frame_to_parquet(&h5, FrameKind::Obs, None, Some("nope"), &pq);
        let _ = std::fs::remove_file(&h5);
        let _ = std::fs::remove_file(&pq);
        assert!(matches!(missing, Err(ScDataErrors::InvalidArgument(_))));
    }

    #[test]
    fn test_h5ad_duplicate_snake_case_errors() {
        let h5 = tmp("obs_dup.h5ad");
        let pq = tmp("obs_dup.parquet");
        {
            let file = File::create(&h5).unwrap();
            let obs = file.create_group("obs").unwrap();
            write_strings(&obs, "_index", &["c1"]);
            write_strings(&obs, "Sample ID", &["s"]);
            write_strings(&obs, "sample_id", &["s"]);
        }
        let res = h5ad_frame_to_parquet(&h5, FrameKind::Obs, None, None, &pq);
        let _ = std::fs::remove_file(&h5);
        let _ = std::fs::remove_file(&pq);
        assert!(matches!(res, Err(ScDataErrors::DuplicateColumnName(n)) if n == "sample_id"));
    }

    #[test]
    fn test_tenx_v3_and_v2() {
        let h5 = tmp("tenx_v3.h5");
        let pq = tmp("tenx_v3.parquet");
        {
            let file = File::create(&h5).unwrap();
            let m = file.create_group("matrix").unwrap();
            write_strings(&m, "barcodes", &["AA-1", "CC-1", "GG-1"]);
            let f = m.create_group("features").unwrap();
            write_strings(&f, "id", &["E1", "E2", "AB1"]);
            write_strings(&f, "name", &["n1", "n2", "CD3"]);
            write_strings(
                &f,
                "feature_type",
                &["Gene Expression", "Gene Expression", "Antibody Capture"],
            );
        }
        tenx_frame_to_parquet(&h5, TenxVersion::V3, FrameKind::Obs, Some(&[0, 2]), &pq).unwrap();
        let (names, rows) = read_back(&pq);
        assert_eq!(names, ["cell_idx", "cell_id"]);
        assert_eq!(column(&names, &rows, "cell_id"), [s("AA-1"), s("GG-1")]);

        let info =
            tenx_frame_to_parquet(&h5, TenxVersion::V3, FrameKind::Var, Some(&[1]), &pq).unwrap();
        let (names, rows) = read_back(&pq);
        assert_eq!(names, ["gene_idx", "gene_id", "gene_name", "feature_type"]);
        assert_eq!(info.n_rows, 1);
        assert_eq!(
            rows,
            [vec![Field::Int(1), s("E2"), s("n2"), s("Gene Expression")]]
        );
        let _ = std::fs::remove_file(&h5);

        let h5 = tmp("tenx_v2.h5");
        {
            let file = File::create(&h5).unwrap();
            let fixed = |xs: &[&str]| -> Vec<FixedAscii<18>> {
                xs.iter()
                    .map(|s| FixedAscii::<18>::from_ascii(s.as_bytes()).unwrap())
                    .collect()
            };
            file.new_dataset_builder()
                .with_data(&fixed(&["E1", "E2"]))
                .create("genes")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&fixed(&["n1", "n2"]))
                .create("gene_names")
                .unwrap();
        }
        tenx_frame_to_parquet(&h5, TenxVersion::V2, FrameKind::Var, None, &pq).unwrap();
        let (names, rows) = read_back(&pq);
        let _ = std::fs::remove_file(&h5);
        let _ = std::fs::remove_file(&pq);
        assert_eq!(names, ["gene_idx", "gene_id", "gene_name"]);
        assert_eq!(column(&names, &rows, "gene_id"), [s("E1"), s("E2")]);
    }
}
