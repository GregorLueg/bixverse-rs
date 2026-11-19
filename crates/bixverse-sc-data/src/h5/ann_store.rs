//! One handle over the two anndata backends: h5ad (HDF5) and Zarr (v2, v3).
//!
//! The ingest loops only ever open a file, take 1-D arrays (`indptr`,
//! `indices`, `data`) and read ranges of them, read rows of a dense `X`, and
//! look at attributes and group members. [`AnnFile`] offers exactly that over
//! both backends, so the loops in `h5::h5ad` serve both. HDF5 reads keep going
//! through [`H5ChunkedReader`], Zarr reads through `zarrs`, which decodes
//! chunks in parallel itself. Values convert through [`H5Elem`] whatever the
//! stored type.

use hdf5::{File, Group, IndexType, IterationOrder, LocationType};
use ndarray::{Array2, s};
use std::ops::Range;
use std::path::Path;

#[cfg(feature = "zarr")]
use std::sync::Arc;
#[cfg(feature = "zarr")]
use zarrs::array::{Array, data_type as zdt};
#[cfg(feature = "zarr")]
use zarrs::filesystem::FilesystemStore;
#[cfg(feature = "zarr")]
use zarrs::group::Group as ZGroup;

use crate::h5::chunked::{H5ChunkedReader, H5Elem};
#[cfg(feature = "zarr")]
use crate::h5::h5ad_frame::{ColumnData, to_r_int};
use crate::h5::tenx::read_string_dataset;
use crate::prelude::*;

/// A Zarr array on the local filesystem.
#[cfg(feature = "zarr")]
type ZArray = Array<FilesystemStore>;

/// What sits at a path of the store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum NodeKind {
    /// A group (h5ad and Zarr alike).
    Group,
    /// An array (an HDF5 dataset).
    Array,
}

/// Whether `path` is a Zarr store: a directory with v3 (`zarr.json`) or v2
/// (`.zgroup`) group metadata.
///
/// ### Params
///
/// * `path` - Path to test
///
/// ### Returns
///
/// `true` for a Zarr store.
pub fn is_zarr<P: AsRef<Path>>(path: P) -> bool {
    let p = path.as_ref();
    p.is_dir() && (p.join("zarr.json").is_file() || p.join(".zgroup").is_file())
}

/// An open anndata file.
pub enum AnnFile {
    /// An h5ad file.
    H5(File),
    /// A Zarr store.
    #[cfg(feature = "zarr")]
    Zarr(Arc<FilesystemStore>),
}

impl AnnFile {
    /// Open an h5ad file or a Zarr store, by what `path` is.
    ///
    /// ### Params
    ///
    /// * `path` - Path to the `.h5ad` file or the Zarr directory
    ///
    /// ### Returns
    ///
    /// The handle, or [`ScDataErrors::UnsupportH5ADFormat`] for a Zarr store
    /// in a build without the `zarr` feature.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ScDataErrors> {
        let path = path.as_ref();
        if is_zarr(path) {
            #[cfg(feature = "zarr")]
            {
                let store = FilesystemStore::new(path).map_err(zarr_err)?;
                return Ok(Self::Zarr(Arc::new(store)));
            }
            #[cfg(not(feature = "zarr"))]
            return Err(ScDataErrors::UnsupportH5ADFormat(format!(
                "{} is a Zarr store; this build lacks the `zarr` feature",
                path.display()
            )));
        }
        Ok(Self::H5(File::open(path)?))
    }

    /// The HDF5 file, for checks that only apply to h5ad.
    ///
    /// ### Returns
    ///
    /// The file, `None` for Zarr.
    pub fn h5(&self) -> Option<&File> {
        match self {
            Self::H5(f) => Some(f),
            #[cfg(feature = "zarr")]
            Self::Zarr(_) => None,
        }
    }

    /// What sits at `path`.
    ///
    /// ### Params
    ///
    /// * `path` - Path inside the file, e.g. `"X/indptr"`
    ///
    /// ### Returns
    ///
    /// The kind, `None` when nothing is there.
    pub fn node(&self, path: &str) -> Option<NodeKind> {
        match self {
            Self::H5(f) => match f.loc_type_by_name(path) {
                Ok(LocationType::Group) => Some(NodeKind::Group),
                Ok(LocationType::Dataset) => Some(NodeKind::Array),
                _ => None,
            },
            #[cfg(feature = "zarr")]
            Self::Zarr(store) => {
                if ZArray::open(store.clone(), &zpath(path)).is_ok() {
                    Some(NodeKind::Array)
                } else if ZGroup::open(store.clone(), &zpath(path)).is_ok() {
                    Some(NodeKind::Group)
                } else {
                    None
                }
            }
        }
    }

    /// A string attribute of the group or array at `path`.
    ///
    /// ### Params
    ///
    /// * `path` - Path of the node
    /// * `name` - Attribute name, e.g. `"encoding-type"`
    ///
    /// ### Returns
    ///
    /// The value, `None` when the attribute is absent or not a string.
    pub fn attr_str(&self, path: &str, name: &str) -> Result<Option<String>, ScDataErrors> {
        match self {
            Self::H5(f) => {
                let attrs = match self.node(path) {
                    Some(NodeKind::Group) => f.group(path)?.attr_names()?,
                    Some(NodeKind::Array) => f.dataset(path)?.attr_names()?,
                    None => return Ok(None),
                };
                if !attrs.iter().any(|a| a == name) {
                    return Ok(None);
                }
                let attr = match self.node(path) {
                    Some(NodeKind::Group) => f.group(path)?.attr(name)?,
                    _ => f.dataset(path)?.attr(name)?,
                };
                Ok(read_string_dataset(&attr)?.into_iter().next())
            }
            #[cfg(feature = "zarr")]
            Self::Zarr(store) => {
                let p = zpath(path);
                let value = if let Ok(a) = ZArray::open(store.clone(), &p) {
                    a.attributes().get(name).cloned()
                } else if let Ok(g) = ZGroup::open(store.clone(), &p) {
                    g.attributes().get(name).cloned()
                } else {
                    None
                };
                Ok(value.and_then(|v| v.as_str().map(str::to_string)))
            }
        }
    }

    /// Members of the group at `path`, sorted by name, with their kinds.
    ///
    /// ### Params
    ///
    /// * `path` - Path of the group
    ///
    /// ### Returns
    ///
    /// `(name, kind)` pairs.
    pub fn members(&self, path: &str) -> Result<Vec<(String, NodeKind)>, ScDataErrors> {
        match self {
            Self::H5(f) => {
                let g: Group = f.group(path)?;
                let mut out = Vec::new();
                for name in g.member_names_by(IndexType::Name, IterationOrder::Increasing)? {
                    let kind = match g.loc_type_by_name(&name)? {
                        LocationType::Group => NodeKind::Group,
                        LocationType::Dataset => NodeKind::Array,
                        _ => continue,
                    };
                    out.push((name, kind));
                }
                Ok(out)
            }
            #[cfg(feature = "zarr")]
            Self::Zarr(store) => {
                let g = ZGroup::open(store.clone(), &zpath(path)).map_err(zarr_err)?;
                let name_of = |p: &zarrs::node::NodePath| {
                    p.as_str()
                        .rsplit('/')
                        .next()
                        .unwrap_or_default()
                        .to_string()
                };
                let mut out: Vec<(String, NodeKind)> = g
                    .child_array_paths()
                    .map_err(zarr_err)?
                    .iter()
                    .map(|p| (name_of(p), NodeKind::Array))
                    .chain(
                        g.child_group_paths()
                            .map_err(zarr_err)?
                            .iter()
                            .map(|p| (name_of(p), NodeKind::Group)),
                    )
                    .collect();
                out.sort();
                Ok(out)
            }
        }
    }

    /// Shape of the array at `path`.
    ///
    /// ### Params
    ///
    /// * `path` - Path of the array
    ///
    /// ### Returns
    ///
    /// The shape.
    pub fn shape(&self, path: &str) -> Result<Vec<usize>, ScDataErrors> {
        match self {
            Self::H5(f) => Ok(f.dataset(path)?.shape()),
            #[cfg(feature = "zarr")]
            Self::Zarr(store) => Ok(open_zarray(store, path)?
                .shape()
                .iter()
                .map(|&n| n as usize)
                .collect()),
        }
    }

    /// A 1-D array to read ranges of. Build it once per dataset, outside any
    /// parallel loop.
    ///
    /// ### Params
    ///
    /// * `path` - Path of the array, e.g. `"X/data"`
    ///
    /// ### Returns
    ///
    /// The array.
    pub fn array(&self, path: &str) -> Result<AnnArray, ScDataErrors> {
        match self {
            Self::H5(f) => Ok(AnnArray::H5(H5ChunkedReader::new(f.dataset(path)?))),
            #[cfg(feature = "zarr")]
            Self::Zarr(store) => Ok(AnnArray::Zarr(Box::new(open_zarray(store, path)?))),
        }
    }

    /// A whole 1-D array, converted to `T`.
    ///
    /// ### Params
    ///
    /// * `path` - Path of the array
    ///
    /// ### Returns
    ///
    /// The values.
    pub fn read_1d<T: H5Elem>(&self, path: &str) -> Result<Vec<T>, ScDataErrors> {
        match self {
            Self::H5(f) => Ok(f.dataset(path)?.read_1d::<T>()?.to_vec()),
            #[cfg(feature = "zarr")]
            Self::Zarr(store) => {
                let arr = open_zarray(store, path)?;
                zread(&arr, whole(&arr))
            }
        }
    }

    /// An `indptr` array as `usize`.
    ///
    /// AnnData switches `indptr` to int64 past 2^31 stored values, so it is
    /// never read narrower than `usize`.
    ///
    /// ### Params
    ///
    /// * `path` - Path of the `indptr` array
    ///
    /// ### Returns
    ///
    /// The offsets.
    pub fn indptr(&self, path: &str) -> Result<Vec<usize>, ScDataErrors> {
        self.read_1d::<usize>(path)
    }

    /// A block of a 2-D array, converted to `T`.
    ///
    /// ### Params
    ///
    /// * `path` - Path of the array
    /// * `rows` - Rows to read
    /// * `cols` - Columns to read
    ///
    /// ### Returns
    ///
    /// The block, `rows.len() x cols.len()`.
    pub fn read_2d<T: H5Elem>(
        &self,
        path: &str,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<Array2<T>, ScDataErrors> {
        match self {
            Self::H5(f) => Ok(f
                .dataset(path)?
                .read_slice_2d::<T, _>(s![rows.clone(), cols.clone()])?),
            #[cfg(feature = "zarr")]
            Self::Zarr(store) => {
                let arr = open_zarray(store, path)?;
                let (r, c) = (rows.len(), cols.len());
                let vals = zread::<T>(
                    &arr,
                    vec![
                        rows.start as u64..rows.end as u64,
                        cols.start as u64..cols.end as u64,
                    ],
                )?;
                Array2::from_shape_vec((r, c), vals)
                    .map_err(|e| ScDataErrors::ZarrError(format!("{path}: {e}")))
            }
        }
    }

    /// A 1-D string array (obs / var index, categories).
    ///
    /// ### Params
    ///
    /// * `path` - Path of the array
    ///
    /// ### Returns
    ///
    /// The strings.
    pub fn read_strings(&self, path: &str) -> Result<Vec<String>, ScDataErrors> {
        match self {
            Self::H5(f) => read_string_dataset(&f.dataset(path)?),
            #[cfg(feature = "zarr")]
            Self::Zarr(store) => {
                let arr = open_zarray(store, path)?;
                if *arr.data_type() != zdt::string() {
                    return Err(ScDataErrors::ZarrError(format!(
                        "{path} is not a string array"
                    )));
                }
                arr.retrieve_array_subset::<Vec<String>>(&whole(&arr))
                    .map_err(zarr_err)
            }
        }
    }

    /// A 1-D Zarr array as a dataframe column.
    ///
    /// ### Params
    ///
    /// * `path` - Path of the array
    ///
    /// ### Returns
    ///
    /// The column, or `None` for a dtype R would not turn into a vector.
    #[cfg(feature = "zarr")]
    pub(crate) fn read_zarr_column(&self, path: &str) -> Result<Option<ColumnData>, ScDataErrors> {
        let Self::Zarr(store) = self else {
            return Ok(None);
        };
        let arr = open_zarray(store, path)?;
        let dt = arr.data_type();
        let ints = [
            zdt::int8(),
            zdt::int16(),
            zdt::int32(),
            zdt::int64(),
            zdt::uint8(),
            zdt::uint16(),
            zdt::uint32(),
            zdt::uint64(),
        ];
        Ok(Some(if ints.contains(dt) {
            ColumnData::Integer(
                zread::<i64>(&arr, whole(&arr))?
                    .into_iter()
                    .map(to_r_int)
                    .collect(),
            )
        } else if *dt == zdt::float32() || *dt == zdt::float64() {
            ColumnData::Double(
                zread::<f64>(&arr, whole(&arr))?
                    .into_iter()
                    .map(Some)
                    .collect(),
            )
        } else if *dt == zdt::bool() {
            ColumnData::Boolean(
                arr.retrieve_array_subset::<Vec<bool>>(&whole(&arr))
                    .map_err(zarr_err)?
                    .into_iter()
                    .map(Some)
                    .collect(),
            )
        } else if *dt == zdt::string() {
            ColumnData::String(self.read_strings(path)?.into_iter().map(Some).collect())
        } else {
            return Ok(None);
        }))
    }
}

/// A 1-D array of an [`AnnFile`].
pub enum AnnArray {
    /// An HDF5 dataset behind the chunk-parallel reader.
    H5(H5ChunkedReader),
    /// A Zarr array.
    #[cfg(feature = "zarr")]
    Zarr(Box<ZArray>),
}

impl AnnArray {
    /// Read `range`, converted to `T`.
    ///
    /// ### Params
    ///
    /// * `range` - Element range
    ///
    /// ### Returns
    ///
    /// The values.
    pub fn read<T: H5Elem>(&self, range: Range<usize>) -> Result<Vec<T>, ScDataErrors> {
        match self {
            Self::H5(r) => r.read(range),
            #[cfg(feature = "zarr")]
            Self::Zarr(a) => zread(
                a,
                std::iter::once(range.start as u64..range.end as u64).collect(),
            ),
        }
    }
}

//////////
// Zarr //
//////////

/// zarrs paths are absolute.
///
/// ### Params
///
/// * `path` - Path relative to the store root, h5ad style
///
/// ### Returns
///
/// The absolute path.
#[cfg(feature = "zarr")]
fn zpath(path: &str) -> String {
    format!("/{}", path.trim_start_matches('/'))
}

/// Map a zarrs error.
///
/// ### Params
///
/// * `e` - The error
///
/// ### Returns
///
/// [`ScDataErrors::ZarrError`].
#[cfg(feature = "zarr")]
fn zarr_err(e: impl std::fmt::Display) -> ScDataErrors {
    ScDataErrors::ZarrError(e.to_string())
}

/// Open the array at `path`.
///
/// ### Params
///
/// * `store` - The store
/// * `path` - Path of the array
///
/// ### Returns
///
/// The array.
#[cfg(feature = "zarr")]
fn open_zarray(store: &Arc<FilesystemStore>, path: &str) -> Result<ZArray, ScDataErrors> {
    ZArray::open(store.clone(), &zpath(path))
        .map_err(|e| ScDataErrors::ZarrError(format!("{path}: {e}")))
}

/// The whole of a Zarr array as a subset.
///
/// ### Params
///
/// * `arr` - The array
///
/// ### Returns
///
/// One full range per dimension.
#[cfg(feature = "zarr")]
fn whole(arr: &ZArray) -> Vec<Range<u64>> {
    arr.shape().iter().map(|&n| 0..n).collect()
}

/// Read a subset of a Zarr array in its stored type and convert to `T`.
///
/// ### Params
///
/// * `arr` - The array
/// * `subset` - One range per dimension
///
/// ### Returns
///
/// The values in row-major order, or [`ScDataErrors::ZarrError`] for a type
/// that is not a number or a bool.
#[cfg(feature = "zarr")]
fn zread<T: H5Elem>(arr: &ZArray, subset: Vec<Range<u64>>) -> Result<Vec<T>, ScDataErrors> {
    macro_rules! via {
        ($t:ty, $conv:ident, $wide:ty) => {
            arr.retrieve_array_subset::<Vec<$t>>(&subset)
                .map_err(zarr_err)?
                .into_iter()
                .map(|v| T::$conv(v as $wide))
                .collect()
        };
    }
    let dt = arr.data_type();
    Ok(if *dt == zdt::float32() {
        via!(f32, from_f64, f64)
    } else if *dt == zdt::float64() {
        via!(f64, from_f64, f64)
    } else if *dt == zdt::int8() {
        via!(i8, from_i64, i64)
    } else if *dt == zdt::int16() {
        via!(i16, from_i64, i64)
    } else if *dt == zdt::int32() {
        via!(i32, from_i64, i64)
    } else if *dt == zdt::int64() {
        via!(i64, from_i64, i64)
    } else if *dt == zdt::uint8() {
        via!(u8, from_u64, u64)
    } else if *dt == zdt::uint16() {
        via!(u16, from_u64, u64)
    } else if *dt == zdt::uint32() {
        via!(u32, from_u64, u64)
    } else if *dt == zdt::uint64() {
        via!(u64, from_u64, u64)
    } else if *dt == zdt::bool() {
        arr.retrieve_array_subset::<Vec<bool>>(&subset)
            .map_err(zarr_err)?
            .into_iter()
            .map(|v| T::from_u64(v as u64))
            .collect()
    } else {
        return Err(ScDataErrors::ZarrError(format!(
            "{}: unsupported data type {:?}",
            arr.path(),
            dt
        )));
    })
}

///////////
// Tests //
///////////

#[cfg(all(test, feature = "zarr"))]
mod tests {
    use super::*;
    use crate::h5::h5ad::{RawDataSlot, write_h5_counts};
    use crate::h5::h5ad_frame::read_h5ad_frame;
    use crate::h5::scan::{detect_raw_count_slot, get_h5ad_dimensions};
    use std::path::PathBuf;

    /// anndata-written fixtures, `python/dev/gen_anndata_fixture.py`.
    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/tests/fixtures")
            .join(name)
    }

    /// The Zarr store reads like its h5ad twin: scan, obs / var, and the cell
    /// file built from it, byte for byte.
    fn assert_matches_h5ad(zarr: &str, h5ad: &str, format: &str) {
        let (z, h) = (fixture(zarr), fixture(h5ad));
        assert!(is_zarr(&z) && !is_zarr(&h));

        let dims = get_h5ad_dimensions(&z).unwrap();
        assert_eq!(dims, get_h5ad_dimensions(&h).unwrap(), "{zarr}");

        let slots = [
            RawDataSlot::LayerCounts,
            RawDataSlot::RawX,
            RawDataSlot::DataX,
        ];
        let slot = detect_raw_count_slot(&z, &slots, 100, 0.99).unwrap();
        assert!(matches!(slot, Some(RawDataSlot::DataX)), "{zarr}");

        for group in ["obs", "var"] {
            let (fz, fh) = (
                read_h5ad_frame(&z, group).unwrap(),
                read_h5ad_frame(&h, group).unwrap(),
            );
            assert_eq!(fz.cols, fh.cols, "{zarr} {group}");
            assert_eq!(fz.idx, fh.idx, "{zarr} {group}");
            assert_eq!(fz.skipped, fh.skipped, "{zarr} {group}");
        }
        let obs = read_h5ad_frame(&z, "obs").unwrap();
        assert!(obs.skipped.is_empty(), "{zarr}: {:?}", obs.skipped);
        assert_eq!(obs.cols.len(), 7, "{zarr}");

        let dir = std::env::temp_dir().join(format!("bixverse_zarr_{}_{zarr}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let qc = MinCellQuality {
            min_unique_genes: 0,
            min_lib_size: 0,
            min_cells: 0,
            target_size: 1e4,
        };
        let mut bytes = Vec::new();
        for (src, out) in [(&z, "zarr.bin"), (&h, "h5ad.bin")] {
            let out = dir.join(out);
            write_h5_counts(
                src.as_path(),
                out.as_path(),
                format,
                dims.n_obs,
                dims.n_var,
                qc.clone(),
                RawDataSlot::DataX,
                false,
            )
            .unwrap();
            bytes.push(std::fs::read(&out).unwrap());
        }
        assert_eq!(bytes[0], bytes[1], "{zarr}: cell file differs");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_zarr_v3_matches_h5ad() {
        assert_matches_h5ad("anndata_csr_v3.zarr", "anndata_csr.h5ad", "csr");
        assert_matches_h5ad("anndata_csc_v3.zarr", "anndata_csc.h5ad", "csc");
        assert_matches_h5ad("anndata_dense_v3.zarr", "anndata_dense.h5ad", "dense_row");
        assert_matches_h5ad("anndata_csr_v3_sharded.zarr", "anndata_csr.h5ad", "csr");
    }

    // the R build leaves blosc out: a blosc store must refuse, not misread
    #[cfg(not(feature = "zarr-blosc"))]
    #[test]
    fn test_zarr_v2_without_blosc_names_the_codec() {
        let z = fixture("anndata_csr_v2.zarr");
        let err = AnnFile::open(&z)
            .unwrap()
            .array("X/data")
            .err()
            .expect("blosc array opened without the codec");
        assert!(err.to_string().contains("blosc"), "{err}");
    }

    // zarr v2 defaults to blosc
    #[cfg(feature = "zarr-blosc")]
    #[test]
    fn test_zarr_v2_matches_h5ad() {
        assert_matches_h5ad("anndata_csr_v2.zarr", "anndata_csr.h5ad", "csr");
        assert_matches_h5ad("anndata_csc_v2.zarr", "anndata_csc.h5ad", "csc");
        assert_matches_h5ad("anndata_dense_v2.zarr", "anndata_dense.h5ad", "dense_row");
    }
}
