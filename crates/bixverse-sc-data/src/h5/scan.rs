//! Cheap scans of an h5ad file or Zarr store ahead of ingestion: the obs / var
//! lengths, how `X` is stored, and which slot holds raw counts. Ports of the rhdf5 helpers
//! of the R package (`get_h5ad_dimensions()`, `detect_raw_count_slot()`), so
//! a front end without its own HDF5 binding can prescan through the crate.
//! Nothing here reads more than a sample of the counts.

use std::path::Path;

use crate::h5::ann_store::{AnnFile, NodeKind};
use crate::h5::h5ad::{H5ADFormat, RawDataSlot};
use crate::prelude::*;

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

/// Length of the 1-D array at `path`.
///
/// ### Params
///
/// * `file` - The open file
/// * `path` - Path of the array
///
/// ### Returns
///
/// The first dimension, 0 for a scalar.
fn len_of(file: &AnnFile, path: &str) -> Result<usize, ScDataErrors> {
    Ok(file.shape(path)?.first().copied().unwrap_or(0))
}

/// Number of rows of an anndata dataframe group
///
/// Mirrors `.h5ad_frame_length()` in R: the length of the column named by the
/// `_index` attribute (a plain array, or a group holding `codes` or
/// `values`), else the length of the first plain array in the group.
///
/// ### Params
///
/// * `file` - The open file
/// * `group` - `"obs"` or `"var"`
///
/// ### Returns
///
/// The number of rows, or [ScDataErrors::UnsupportedH5adFormat] if none can be
/// resolved.
fn frame_length(file: &AnnFile, group: &str) -> Result<usize, ScDataErrors> {
    let unresolved =
        || ScDataErrors::UnsupportedH5adFormat(format!("cannot resolve the length of /{group}"));

    let Some(index) = file.attr_str(group, "_index")? else {
        let first = file
            .members(group)?
            .into_iter()
            .find(|(_, k)| *k == NodeKind::Array)
            .ok_or_else(unresolved)?;
        return len_of(file, &format!("{group}/{}", first.0));
    };

    let path = format!("{group}/{index}");
    match file.node(&path) {
        Some(NodeKind::Array) => len_of(file, &path),
        Some(NodeKind::Group) => {
            for (name, kind) in file.members(&path)? {
                if kind == NodeKind::Array && (name == "codes" || name == "values") {
                    return len_of(file, &format!("{path}/{name}"));
                }
            }
            Err(unresolved())
        }
        None => Err(unresolved()),
    }
}

/// Read a sample of the values of a count slot
///
/// Mirrors `.read_slot_value_sample()` in R: for a sparse slot (a group with
/// `data`) the first `n_sample` stored values, for a dense one the top left
/// `ceil(sqrt(n_sample))` square.
///
/// ### Params
///
/// * `file` - The open file
/// * `slot` - Which slot to sample
/// * `n_sample` - Maximum number of values to read
///
/// ### Returns
///
/// The sampled values, or `None` if the slot does not exist.
fn sample_slot(
    file: &AnnFile,
    slot: &RawDataSlot,
    n_sample: usize,
) -> Result<Option<Vec<f64>>, ScDataErrors> {
    let path = slot.get_dense_path();
    match file.node(path) {
        Some(NodeKind::Group) => {
            if file.node(slot.get_data()) != Some(NodeKind::Array) {
                return Ok(None);
            }
            let n = len_of(file, slot.get_data())?.min(n_sample);
            Ok(Some(file.array(slot.get_data())?.read::<f64>(0..n)?))
        }
        Some(NodeKind::Array) => {
            let shape = file.shape(path)?;
            if shape.len() != 2 {
                return Ok(None);
            }
            let side = (n_sample as f64).sqrt().ceil().max(1.0) as usize;
            let block = file.read_2d::<f64>(path, 0..shape[0].min(side), 0..shape[1].min(side))?;
            Ok(Some(block.iter().copied().collect()))
        }
        None => Ok(None),
    }
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
/// * `h5_path` - Path to the h5ad file or Zarr store
///
/// ### Returns
///
/// The [H5adDimensions], or an error if `X` is missing or its shape fits
/// neither ordering.
pub fn get_h5ad_dimensions<P: AsRef<Path>>(h5_path: P) -> Result<H5adDimensions, ScDataErrors> {
    let file = AnnFile::open(h5_path)?;
    let n_obs = frame_length(&file, "obs")?;
    let n_var = frame_length(&file, "var")?;

    let format = match file.node("X") {
        Some(NodeKind::Group) => {
            if file.node("X/indptr") != Some(NodeKind::Array) {
                return Err(ScDataErrors::UnsupportedH5adFormat(
                    "/X is a group but contains no indptr dataset".to_string(),
                ));
            }
            match file.attr_str("X", "encoding-type")?.as_deref() {
                Some("csr_matrix") => H5ADFormat::Csr,
                Some("csc_matrix") => H5ADFormat::Csc,
                _ if len_of(&file, "X/indptr")? == n_var + 1 => H5ADFormat::Csc,
                _ => H5ADFormat::Csr,
            }
        }
        Some(NodeKind::Array) => {
            let shape = file.shape("X")?;
            match shape.as_slice() {
                [r, c] if *r == n_obs && *c == n_var => H5ADFormat::DenseRow,
                [r, c] if *r == n_var && *c == n_obs => H5ADFormat::DenseCol,
                [r, c] => {
                    return Err(ScDataErrors::UnsupportedH5adFormat(format!(
                        "/X dense dims ({r} x {c}) match neither obs/var ({n_obs} / {n_var}) ordering"
                    )));
                }
                other => {
                    return Err(ScDataErrors::UnsupportedH5adFormat(format!(
                        "unexpected /X rank: {}",
                        other.len()
                    )));
                }
            }
        }
        None => {
            return Err(ScDataErrors::UnsupportedH5adFormat(
                "could not locate /X in the h5ad file".to_string(),
            ));
        }
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
/// * `h5_path` - Path to the h5ad file or Zarr store
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
    let file = AnnFile::open(h5_path)?;
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

/// Read a 1-D string array of an h5ad file or Zarr store
///
/// ### Params
///
/// * `h5_path` - Path to the h5ad file or Zarr store
/// * `dataset` - Path of the dataset inside it, e.g. `"var/_index"`
///
/// ### Returns
///
/// The strings.
pub fn read_h5_strings<P: AsRef<Path>>(
    h5_path: P,
    dataset: &str,
) -> Result<Vec<String>, ScDataErrors> {
    AnnFile::open(h5_path)?.read_strings(dataset)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::File;
    use hdf5::types::VarLenUnicode;
    use tempfile::NamedTempFile;

    /// Variable-length UTF-8 strings for a dataset.
    ///
    /// ### Params
    ///
    /// * `xs` - The strings
    ///
    /// ### Returns
    ///
    /// The HDF5 strings.
    fn strings(xs: &[&str]) -> Vec<VarLenUnicode> {
        xs.iter().map(|x| x.parse().unwrap()).collect()
    }

    /// An h5ad with a 3 x 2 obs / var, `X` as given by `write_x`.
    ///
    /// ### Params
    ///
    /// * `path` - Output file
    /// * `write_x` - Writes the `X` group or dataset into the file
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
        let csr = NamedTempFile::with_suffix(".h5ad")
            .unwrap()
            .into_temp_path();
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

        let dense = NamedTempFile::with_suffix(".h5ad")
            .unwrap()
            .into_temp_path();
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
    }
}
