#![cfg(feature = "single-cell")]

//! Round trips of compressed h5ad inputs.
//!
//! An HDF5 dataset written with a filter the library does not have fails
//! inside `H5Dread` with a message about a missing plugin directory, because
//! `hdf5-metno-src` bakes `H5_DEFAULT_PLUGINDIR` to the cargo `OUT_DIR` of the
//! build. These tests pin what the build can actually decode: gzip always, LZF
//! and Blosc under the `hdf5-filters` feature, and a named error otherwise.

use hdf5::File;
use hdf5::filters::Filter;

use bixverse_rs::single_cell::sc_data::data_io::{
    MinCellQuality, ParallelSparseReader, SingleCellReading,
};
use bixverse_rs::single_cell::sc_data::h5_filters::unsupported_filters;
use bixverse_rs::single_cell::sc_data::h5ad_io::{RawDataSlot, stream_h5_counts, write_h5_counts};

const NO_CELLS: usize = 40;
const NO_GENES: usize = 30;

/// Filter pipelines the fixture writer can apply.
#[derive(Clone, Copy, Debug)]
enum Compression {
    /// Contiguous, no filter at all.
    Raw,
    /// Shuffle plus deflate, what `anndata` writes by default.
    Gzip,
    /// Filter 32000, needs `hdf5-filters`.
    #[cfg(feature = "hdf5-filters")]
    Lzf,
    /// Filter 32001 with the zstd codec inside, needs `hdf5-filters`.
    #[cfg(feature = "hdf5-filters")]
    Blosc,
}

impl Compression {
    fn pipeline(self) -> Vec<Filter> {
        match self {
            Self::Raw => vec![],
            Self::Gzip => vec![Filter::Shuffle, Filter::Deflate(4)],
            #[cfg(feature = "hdf5-filters")]
            Self::Lzf => vec![Filter::LZF],
            #[cfg(feature = "hdf5-filters")]
            Self::Blosc => vec![Filter::Blosc(
                hdf5::filters::Blosc::ZStd,
                5,
                hdf5::filters::BloscShuffle::Byte,
            )],
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Gzip => "gzip",
            #[cfg(feature = "hdf5-filters")]
            Self::Lzf => "lzf",
            #[cfg(feature = "hdf5-filters")]
            Self::Blosc => "blosc",
        }
    }
}

/// RAII guard that removes a test's temp files even if an assert fails.
struct TempPath(std::path::PathBuf);

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempPath {
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!("bixverse_h5ad_compression_{name}")))
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// Deterministic CSR counts from a seeded LCG.
///
/// Every cell gets at least one count so no cell is dropped by QC and the
/// index mapping stays the identity.
///
/// ### Returns
///
/// `(data, indices, indptr)` in the h5ad dtypes: f32 counts, i32 structure.
fn reference_csr() -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let mut state: u64 = 0x5EED_1234;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (state >> 33) as u32
    };

    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0_i32];

    for _ in 0..NO_CELLS {
        let mut written = 0;
        for gene in 0..NO_GENES {
            if next() % 3 == 0 {
                data.push((next() % 50 + 1) as f32);
                indices.push(gene as i32);
                written += 1;
            }
        }
        if written == 0 {
            data.push(1.0);
            indices.push(0);
        }
        indptr.push(data.len() as i32);
    }

    (data, indices, indptr)
}

/// Write a minimal CSR h5ad with the given filter pipeline on every count
/// dataset.
fn write_fixture(path: &str, compression: Compression) {
    let (data, indices, indptr) = reference_csr();
    let filters = compression.pipeline();

    let file = File::create(path).expect("create h5ad");
    let group = file.create_group("X").expect("create X");

    group
        .new_dataset_builder()
        .set_filters(&filters)
        .chunk(64.min(data.len()))
        .with_data(&data)
        .create("data")
        .expect("write data");
    group
        .new_dataset_builder()
        .set_filters(&filters)
        .chunk(64.min(indices.len()))
        .with_data(&indices)
        .create("indices")
        .expect("write indices");
    group
        .new_dataset_builder()
        .set_filters(&filters)
        .chunk(16)
        .with_data(&indptr)
        .create("indptr")
        .expect("write indptr");
}

fn qc_params() -> MinCellQuality {
    MinCellQuality {
        min_unique_genes: 0,
        min_lib_size: 0,
        min_cells: 0,
        target_size: 1e4,
    }
}

/// Read every cell back out of the binarised file as `(indices, counts)`.
fn read_back(bin_path: &str) -> Vec<(Vec<u32>, Vec<u32>)> {
    let reader = ParallelSparseReader::new(bin_path).expect("reader opens");
    let indices: Vec<usize> = (0..NO_CELLS).collect();

    reader
        .read_cells_parallel(&indices)
        .expect("read cells")
        .into_iter()
        .map(|chunk| (chunk.indices, chunk.data_raw.iter().collect()))
        .collect()
}

/// The reference matrix in the same shape as [`read_back`] returns.
fn expected_cells() -> Vec<(Vec<u32>, Vec<u32>)> {
    let (data, indices, indptr) = reference_csr();

    (0..NO_CELLS)
        .map(|cell| {
            let start = indptr[cell] as usize;
            let end = indptr[cell + 1] as usize;
            (
                indices[start..end].iter().map(|&i| i as u32).collect(),
                data[start..end].iter().map(|&v| v as u32).collect(),
            )
        })
        .collect()
}

/// Both ingest paths must reproduce the source matrix exactly, whatever the
/// filter pipeline on the way in.
fn assert_round_trip(compression: Compression) {
    let h5 = TempPath::new(&format!("{}.h5ad", compression.label()));
    write_fixture(h5.path(), compression);

    for mode in ["write", "stream"] {
        let bin = TempPath::new(&format!("{}_{mode}.bin", compression.label()));

        let ingest = if mode == "write" {
            write_h5_counts
        } else {
            stream_h5_counts
        };

        let (cells, genes, _) = ingest(
            h5.path(),
            bin.path(),
            "csr",
            NO_CELLS,
            NO_GENES,
            qc_params(),
            RawDataSlot::DataX,
            false,
        )
        .unwrap_or_else(|e| panic!("{} ingest of {} failed: {e}", mode, compression.label()));

        assert_eq!(cells, NO_CELLS, "{mode} / {}", compression.label());
        assert_eq!(genes, NO_GENES, "{mode} / {}", compression.label());
        assert_eq!(
            read_back(bin.path()),
            expected_cells(),
            "{mode} / {}",
            compression.label()
        );
    }
}

/// Uncompressed input, the baseline the other cases are measured against.
#[test]
fn test_round_trip_uncompressed() {
    assert_round_trip(Compression::Raw);
}

/// Shuffle plus deflate, the `anndata` default and the only compression a
/// stock build can decode.
#[test]
fn test_round_trip_gzip() {
    assert_round_trip(Compression::Gzip);
}

/// LZF, filter 32000. Registered by `hdf5-filters` through `H5Zregister`.
#[cfg(feature = "hdf5-filters")]
#[test]
fn test_round_trip_lzf() {
    assert_round_trip(Compression::Lzf);
}

/// Blosc, filter 32001, carrying zstd as the inner codec.
#[cfg(feature = "hdf5-filters")]
#[test]
fn test_round_trip_blosc() {
    assert_round_trip(Compression::Blosc);
}

/// What the feature buys: with it the two filters are decodable, without it
/// they are reported rather than hit as a missing plugin directory.
#[test]
fn test_lzf_and_blosc_availability_tracks_the_feature() {
    let pipeline = vec![Filter::User(32000, vec![]), Filter::User(32001, vec![])];
    let missing = unsupported_filters(&pipeline);

    if cfg!(feature = "hdf5-filters") {
        assert!(
            missing.is_empty(),
            "expected both filters, missing {missing:?}"
        );
    } else {
        assert_eq!(missing, vec![32000, 32001]);
    }
}

/// A filter nothing in this crate can provide is always reported.
#[test]
fn test_zstd_filter_is_reported_as_unsupported() {
    let pipeline = vec![Filter::User(32015, vec![])];

    assert_eq!(unsupported_filters(&pipeline), vec![32015]);
}
