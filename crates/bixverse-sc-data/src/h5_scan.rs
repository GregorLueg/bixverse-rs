//! Cheap scans of an h5ad file ahead of ingestion: the obs / var lengths, how
//! `X` is stored, and which slot holds raw counts. Ports of the rhdf5 helpers
//! of the R package (`get_h5ad_dimensions()`, `detect_raw_count_slot()`), so
//! a front end without its own HDF5 binding can prescan through the crate.
//! Nothing here reads more than a sample of the counts.

use hdf5::{File, Group, IndexType, IterationOrder, LocationType};
use ndarray::s;
use std::path::Path;

use crate::h5_10x_io::read_string_dataset;
use crate::h5ad_io::{H5ADFormat, RawDataSlot};
use crate::prelude::*;

////////////
// Consts //
////////////

/// Values sampled per slot by [detect_raw_count_slot] in the R package.
pub const RAW_SLOT_N_SAMPLE: usize = 10_000;

/// Fraction of sampled non-zeros that must be whole numbers for a slot to
/// count as raw, as in the R package.
pub const RAW_SLOT_THRESHOLD: f64 = 0.99;

////////////////
// Structures //
////////////////

/// Shape and storage layout of an h5ad file
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H5adDimensions {
    /// Rows of the obs dataframe
    pub n_obs: usize,
    /// Rows of the var dataframe
    pub n_var: usize,
    /// How `X` is stored
    pub format: H5ADFormat,
}

/////////////
// Helpers //
/////////////

/// Names of the members of a group of one location type, in name order
///
/// ### Params
///
/// * `group` - The group
/// * `loc_type` - Keep members of this type
///
/// ### Returns
///
/// The member names, as `h5ls` lists them.
fn members_of_type(group: &Group, loc_type: LocationType) -> Result<Vec<String>, ScDataErrors> {
    let mut out = Vec::new();
    for name in group.member_names_by(IndexType::Name, IterationOrder::Increasing)? {
        if group.loc_type_by_name(&name)? == loc_type {
            out.push(name);
        }
    }
    Ok(out)
}

/// Whether a path exists and is of the given location type
///
/// ### Params
///
/// * `group` - Where to look
/// * `name` - Path relative to `group`
/// * `loc_type` - The expected type
///
/// ### Returns
///
/// `true` if the link exists and has that type.
fn is_type(group: &Group, name: &str, loc_type: LocationType) -> Result<bool, ScDataErrors> {
    Ok(group.link_exists(name) && group.loc_type_by_name(name)? == loc_type)
}

/// Read a string attribute of a group, if present
///
/// ### Params
///
/// * `group` - The group
/// * `name` - Attribute name
///
/// ### Returns
///
/// The first string of the attribute, or `None` if it is absent.
fn string_attr(group: &Group, name: &str) -> Result<Option<String>, ScDataErrors> {
    if !group.attr_names()?.iter().any(|a| a == name) {
        return Ok(None);
    }
    Ok(read_string_dataset(&group.attr(name)?)?.into_iter().next())
}

/// Number of rows of an h5ad dataframe group
///
/// Mirrors `.h5ad_frame_length()` in R: the length of the column named by the
/// `_index` attribute (a plain dataset, or a group holding `codes` or
/// `values`), else the length of the first plain dataset in the group.
///
/// ### Params
///
/// * `file` - The open h5ad file
/// * `group` - `"obs"` or `"var"`
///
/// ### Returns
///
/// The number of rows, or [ScDataErrors::UnsupportH5ADFormat] if none can be
/// resolved.
fn frame_length(file: &File, group: &str) -> Result<usize, ScDataErrors> {
    let g = file.group(group)?;
    let unresolved =
        || ScDataErrors::UnsupportH5ADFormat(format!("cannot resolve the length of /{group}"));

    let Some(index) = string_attr(&g, "_index")? else {
        let first = members_of_type(&g, LocationType::Dataset)?
            .into_iter()
            .next()
            .ok_or_else(unresolved)?;
        return Ok(g.dataset(&first)?.shape().first().copied().unwrap_or(0));
    };

    if is_type(&g, &index, LocationType::Dataset)? {
        return Ok(g.dataset(&index)?.shape().first().copied().unwrap_or(0));
    }
    if is_type(&g, &index, LocationType::Group)? {
        let sub = g.group(&index)?;
        for name in members_of_type(&sub, LocationType::Dataset)? {
            if name == "codes" || name == "values" {
                return Ok(sub.dataset(&name)?.shape().first().copied().unwrap_or(0));
            }
        }
    }
    Err(unresolved())
}

/// Read a sample of the values of a count slot
///
/// Mirrors `.read_slot_value_sample()` in R: for a sparse slot (a group with
/// `data`) the first `n_sample` stored values, for a dense one the top left
/// `ceil(sqrt(n_sample))` square.
///
/// ### Params
///
/// * `file` - The open h5ad file
/// * `slot` - Which slot to sample
/// * `n_sample` - Maximum number of values to read
///
/// ### Returns
///
/// The sampled values, or `None` if the slot does not exist.
fn sample_slot(
    file: &File,
    slot: &RawDataSlot,
    n_sample: usize,
) -> Result<Option<Vec<f64>>, ScDataErrors> {
    let path = slot.get_dense_path();
    if is_type(file, path, LocationType::Group)? {
        if !is_type(file, slot.get_data(), LocationType::Dataset)? {
            return Ok(None);
        }
        let ds = file.dataset(slot.get_data())?;
        let n = ds.size().min(n_sample);
        return Ok(Some(ds.read_slice_1d::<f64, _>(s![0..n])?.to_vec()));
    }
    if is_type(file, path, LocationType::Dataset)? {
        let ds = file.dataset(path)?;
        let shape = ds.shape();
        if shape.len() != 2 {
            return Ok(None);
        }
        let side = (n_sample as f64).sqrt().ceil().max(1.0) as usize;
        let block = ds.read_slice_2d::<f64, _>(s![0..shape[0].min(side), 0..shape[1].min(side)])?;
        return Ok(Some(block.iter().copied().collect()));
    }
    Ok(None)
}

///////////////
// Functions //
///////////////

/// Read the dimensions and the storage layout of an h5ad file
///
/// Mirrors `get_h5ad_dimensions()` in R. Sparse `X` takes its layout from the
/// `encoding-type` attribute, falling back to comparing the `indptr` length
/// against the var length. Dense `X` is `DENSE_ROW` when its on-disk shape is
/// `(n_obs, n_var)`, `DENSE_COL` when it is `(n_var, n_obs)`, and
/// `DENSE_ROW` on a tie, the AnnData convention.
///
/// ### Params
///
/// * `h5_path` - Path to the h5ad file
///
/// ### Returns
///
/// The [H5adDimensions], or an error if `X` is missing or its shape fits
/// neither ordering.
pub fn get_h5ad_dimensions<P: AsRef<Path>>(h5_path: P) -> Result<H5adDimensions, ScDataErrors> {
    let file = File::open(h5_path)?;
    let n_obs = frame_length(&file, "obs")?;
    let n_var = frame_length(&file, "var")?;

    let format = if is_type(&file, "X", LocationType::Group)? {
        if !is_type(&file, "X/indptr", LocationType::Dataset)? {
            return Err(ScDataErrors::UnsupportH5ADFormat(
                "/X is a group but contains no indptr dataset".to_string(),
            ));
        }
        match string_attr(&file.group("X")?, "encoding-type")?.as_deref() {
            Some("csr_matrix") => H5ADFormat::Csr,
            Some("csc_matrix") => H5ADFormat::Csc,
            _ if file.dataset("X/indptr")?.size() == n_var + 1 => H5ADFormat::Csc,
            _ => H5ADFormat::Csr,
        }
    } else if is_type(&file, "X", LocationType::Dataset)? {
        let shape = file.dataset("X")?.shape();
        match shape.as_slice() {
            [r, c] if *r == n_obs && *c == n_var => H5ADFormat::DenseRow,
            [r, c] if *r == n_var && *c == n_obs => H5ADFormat::DenseCol,
            [r, c] => {
                return Err(ScDataErrors::UnsupportH5ADFormat(format!(
                    "/X dense dims ({r} x {c}) match neither obs/var ({n_obs} / {n_var}) ordering"
                )));
            }
            other => {
                return Err(ScDataErrors::UnsupportH5ADFormat(format!(
                    "unexpected /X rank: {}",
                    other.len()
                )));
            }
        }
    } else {
        return Err(ScDataErrors::UnsupportH5ADFormat(
            "could not locate /X in the h5ad file".to_string(),
        ));
    };

    Ok(H5adDimensions {
        n_obs,
        n_var,
        format,
    })
}

/// Detect which slot of an h5ad file holds raw integer counts
///
/// Mirrors `detect_raw_count_slot()` in R: samples each candidate in turn and
/// returns the first whose non-zero values are whole numbers at a rate of at
/// least `threshold`. A missing slot, or one sampled to all zeros, is
/// skipped.
///
/// ### Params
///
/// * `h5_path` - Path to the h5ad file
/// * `candidates` - Slots to test, in priority order
/// * `n_sample` - Values to sample per slot
/// * `threshold` - Minimum fraction of whole numbers among the non-zeros
///
/// ### Returns
///
/// The first qualifying slot, or `None`.
pub fn detect_raw_count_slot<P: AsRef<Path>>(
    h5_path: P,
    candidates: &[RawDataSlot],
    n_sample: usize,
    threshold: f64,
) -> Result<Option<RawDataSlot>, ScDataErrors> {
    let file = File::open(h5_path)?;
    for slot in candidates {
        let Some(values) = sample_slot(&file, slot, n_sample)? else {
            continue;
        };
        let nz: Vec<f64> = values.into_iter().filter(|&v| v != 0.0).collect();
        if nz.is_empty() {
            continue;
        }
        let whole = nz.iter().filter(|v| v.fract() == 0.0).count();
        if whole as f64 / nz.len() as f64 >= threshold {
            return Ok(Some(*slot));
        }
    }
    Ok(None)
}

/// Read a 1-D string dataset of an HDF5 file
///
/// ### Params
///
/// * `h5_path` - Path to the file
/// * `dataset` - Path of the dataset inside it, e.g. `"var/_index"`
///
/// ### Returns
///
/// The strings, in any of the encodings [read_string_dataset] handles.
pub fn read_h5_strings<P: AsRef<Path>>(
    h5_path: P,
    dataset: &str,
) -> Result<Vec<String>, ScDataErrors> {
    read_string_dataset(&File::open(h5_path)?.dataset(dataset)?)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::types::VarLenUnicode;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("bixverse_h5_scan_{}_{name}", std::process::id()))
    }

    fn strings(xs: &[&str]) -> Vec<VarLenUnicode> {
        xs.iter().map(|x| x.parse().unwrap()).collect()
    }

    /// An h5ad with a 3 x 2 obs / var, `X` as given by `write_x`.
    fn write_h5ad(path: &Path, write_x: impl Fn(&File)) {
        let f = File::create(path).unwrap();
        for (grp, ids) in [("obs", vec!["c1", "c2", "c3"]), ("var", vec!["g1", "g2"])] {
            let g = f.create_group(grp).unwrap();
            g.new_dataset_builder()
                .with_data(&strings(&ids))
                .create("_index")
                .unwrap();
            let idx: VarLenUnicode = "_index".parse().unwrap();
            g.new_attr::<VarLenUnicode>()
                .create("_index")
                .unwrap()
                .write_scalar(&idx)
                .unwrap();
        }
        write_x(&f);
    }

    #[test]
    fn test_h5ad_dimensions_sparse_dense_and_raw_slot() {
        let csr = tmp("csr.h5ad");
        write_h5ad(&csr, |f| {
            let x = f.create_group("X").unwrap();
            x.new_dataset_builder()
                .with_data(&[1f32, 2.0, 3.0])
                .create("data")
                .unwrap();
            x.new_dataset_builder()
                .with_data(&[0i32, 1, 0])
                .create("indices")
                .unwrap();
            x.new_dataset_builder()
                .with_data(&[0i32, 1, 2, 3])
                .create("indptr")
                .unwrap();
        });
        let dims = get_h5ad_dimensions(&csr).unwrap();
        assert_eq!((dims.n_obs, dims.n_var), (3, 2));
        // no encoding attribute, indptr is n_obs + 1 long
        assert_eq!(dims.format, H5ADFormat::Csr);
        let slots = [
            RawDataSlot::LayerCounts,
            RawDataSlot::RawX,
            RawDataSlot::DataX,
        ];
        let found = detect_raw_count_slot(&csr, &slots, 10, 0.99).unwrap();
        assert!(matches!(found, Some(RawDataSlot::DataX)));

        let dense = tmp("dense.h5ad");
        write_h5ad(&dense, |f| {
            f.new_dataset_builder()
                .with_data(&ndarray::arr2(&[[0.5f32, 1.0], [2.0, 0.0], [1.5, 3.0]]))
                .create("X")
                .unwrap();
        });
        let dims = get_h5ad_dimensions(&dense).unwrap();
        assert_eq!(dims.format, H5ADFormat::DenseRow);
        // two of four non-zeros are fractional
        assert!(
            detect_raw_count_slot(&dense, &slots, 10, 0.99)
                .unwrap()
                .is_none()
        );
        assert_eq!(read_h5_strings(&dense, "var/_index").unwrap(), ["g1", "g2"]);

        let _ = std::fs::remove_file(csr);
        let _ = std::fs::remove_file(dense);
    }
}
