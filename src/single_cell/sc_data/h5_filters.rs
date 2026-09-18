//! Filter pipeline checks for HDF5 inputs.
//!
//! An HDF5 library can only decode the filters compiled into it. Everything
//! else is a dynamically loaded plugin, looked up in `H5_DEFAULT_PLUGINDIR`,
//! which for a `hdf5-metno-src` build is the cargo `OUT_DIR` of the build that
//! produced the binary and is long gone by the time anything runs. A dataset
//! written by `hdf5plugin` therefore dies inside `H5Dread` complaining about a
//! missing directory rather than about compression. These checks run before
//! the first read and name the filter instead.
//!
//! The `hdf5-filters` feature adds LZF (32000) and Blosc (32001) to the build.
//! Both register themselves through `H5Zregister` at library init, so they
//! never touch the plugin path.

use hdf5::File;
use hdf5::filters::Filter;

use crate::errors::BixverseErrors;
use crate::single_cell::sc_data::h5_10x_io::TenxVersion;
use crate::single_cell::sc_data::h5ad_io::{H5ADFormat, RawDataSlot};

/// Name of an HDF5 filter id
///
/// Covers the six built-ins plus the registered third-party ids that turn up
/// in single cell data. The full registry lives at
/// <https://support.hdfgroup.org/services/filters.html>.
///
/// ### Params
///
/// * `id` - The HDF5 filter identifier.
///
/// ### Returns
///
/// A human readable name, or `"unknown"` for an unregistered id.
pub fn h5_filter_name(id: i32) -> &'static str {
    match id {
        1 => "deflate/gzip",
        2 => "shuffle",
        3 => "fletcher32",
        4 => "szip",
        5 => "nbit",
        6 => "scaleoffset",
        307 => "bzip2",
        32000 => "lzf",
        32001 => "blosc",
        32004 => "lz4",
        32008 => "bitshuffle",
        32013 => "zfp",
        32015 => "zstd",
        32017 => "sz",
        32026 => "blosc2",
        _ => "unknown",
    }
}

/// Filters in a pipeline that this build cannot decode
///
/// ### Params
///
/// * `filters` - The filter pipeline as read off a dataset.
///
/// ### Returns
///
/// The ids that are missing, in pipeline order.
pub fn unsupported_filters(filters: &[Filter]) -> Vec<i32> {
    filters
        .iter()
        .filter(|f| !Filter::get_info(f.id()).decode_enabled)
        .map(|f| f.id())
        .collect()
}

/// Check that every dataset in `paths` can be decoded
///
/// Missing datasets are skipped: whichever reader follows will raise the
/// better error for those.
///
/// ### Params
///
/// * `file` - The open HDF5 file.
/// * `paths` - Dataset paths inside the file.
///
/// ### Returns
///
/// Nothing if okay, [`BixverseErrors::UnsupportedH5Filter`] otherwise.
pub fn check_h5_filters(file: &File, paths: &[&str]) -> Result<(), BixverseErrors> {
    for path in paths {
        let Ok(ds) = file.dataset(path) else {
            continue;
        };
        let Ok(dcpl) = ds.dcpl() else {
            continue;
        };
        if dcpl.all_filters_avail() {
            continue;
        }
        if let Some(&id) = unsupported_filters(&dcpl.filters()).first() {
            return Err(BixverseErrors::UnsupportedH5Filter {
                dataset: (*path).to_string(),
                filter_id: id,
                filter_name: h5_filter_name(id).to_string(),
            });
        }
    }

    Ok(())
}

/// Check the filter pipelines on the count datasets of an h5ad file
///
/// ### Params
///
/// * `file` - The open h5ad file.
/// * `raw_slot` - Where the raw counts live, see [`RawDataSlot`].
/// * `format` - How the counts are stored, see [`H5ADFormat`].
///
/// ### Returns
///
/// Nothing if okay, [`BixverseErrors::UnsupportedH5Filter`] otherwise.
pub fn check_h5ad_filters(
    file: &File,
    raw_slot: &RawDataSlot,
    format: &H5ADFormat,
) -> Result<(), BixverseErrors> {
    match format {
        H5ADFormat::Csr | H5ADFormat::Csc => check_h5_filters(
            file,
            &[
                raw_slot.get_data(),
                raw_slot.get_indices(),
                raw_slot.get_indptr(),
            ],
        ),
        H5ADFormat::DenseRow | H5ADFormat::DenseCol => {
            check_h5_filters(file, &[raw_slot.get_dense_path()])
        }
    }
}

/// Check the filter pipelines on the count datasets of a 10x h5 file
///
/// ### Params
///
/// * `file` - The open 10x h5 file.
/// * `version` - The CellRanger layout, see [`TenxVersion`].
///
/// ### Returns
///
/// Nothing if okay, [`BixverseErrors::UnsupportedH5Filter`] otherwise.
pub fn check_tenx_filters(file: &File, version: &TenxVersion) -> Result<(), BixverseErrors> {
    check_h5_filters(
        file,
        &[
            version.get_data(),
            version.get_indices(),
            version.get_indptr(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RAII guard that removes a test's temp file even if an assert fails.
    struct TempH5(std::path::PathBuf);

    impl Drop for TempH5 {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempH5 {
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_h5_filters_{name}.h5")))
        }
    }

    /// An unregistered id is flagged. `Filter::User` is the only way to build a
    /// pipeline entry for a filter this build does not have, since the dataset
    /// builder refuses to write one.
    #[test]
    fn test_unsupported_filters_flags_unavailable_ids() {
        let pipeline = vec![Filter::Shuffle, Filter::User(32015, vec![])];

        assert_eq!(unsupported_filters(&pipeline), vec![32015]);
    }

    /// Everything that ships with the build passes.
    #[test]
    fn test_unsupported_filters_accepts_built_ins() {
        let pipeline = vec![Filter::Shuffle, Filter::Deflate(4), Filter::Fletcher32];

        assert!(unsupported_filters(&pipeline).is_empty());
    }

    /// Ids we expect to meet in the wild all resolve to a name.
    #[test]
    fn test_filter_names_cover_the_hdf5plugin_codecs() {
        for id in [32000, 32001, 32004, 32008, 32015, 32026] {
            assert_ne!(h5_filter_name(id), "unknown", "filter {id}");
        }
        assert_eq!(h5_filter_name(31999), "unknown");
    }

    /// A gzip + shuffle dataset, the anndata default, passes the file level
    /// check.
    #[test]
    fn test_check_h5_filters_accepts_gzip() {
        let temp = TempH5::new("gzip");
        {
            let file = File::create(&temp.0).expect("create");
            file.new_dataset_builder()
                .shuffle()
                .deflate(4)
                .chunk(16)
                .with_data(&(0..64_i32).collect::<Vec<_>>())
                .create("X")
                .expect("write dataset");
        }

        let file = File::open(&temp.0).expect("open");
        assert!(check_h5_filters(&file, &["X"]).is_ok());
    }

    /// A path that is not in the file is left to the reader behind us.
    #[test]
    fn test_check_h5_filters_skips_missing_datasets() {
        let temp = TempH5::new("missing");
        {
            let file = File::create(&temp.0).expect("create");
            file.new_dataset_builder()
                .with_data(&[1_i32, 2, 3])
                .create("X")
                .expect("write dataset");
        }

        let file = File::open(&temp.0).expect("open");
        assert!(check_h5_filters(&file, &["layers/counts/data"]).is_ok());
    }
}
