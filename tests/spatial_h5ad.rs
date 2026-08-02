//! End-to-end reads of the spatial extras out of h5ad files.
//!
//! The fixtures are written here rather than shipped, and they are shaped so a
//! transposed read is *detectable*: the two axes have different ranges and the
//! point cloud is directional, so swapping the columns moves every spot rather
//! than producing a mirror image that happens to still look like tissue.
//!
//! One test runs against the two real collections when they are present. They
//! are not fixtures and the test skips cleanly when the paths do not exist.

#![cfg(feature = "spatial")]

use std::path::{Path, PathBuf};

use bixverse_rs::spatial::sp_data::{
    OrientationEvidence, SpatialH5adParams, SpatialOrientation, read_spatial_h5ad,
};

/// Spots along a diagonal ramp. `y` spans 0..770 in steps of 10, `x` spans
/// 0..7700 in steps of 100, so the two axes are an order of magnitude apart and
/// a swap cannot hide.
const N_SPOTS: usize = 78;

/// Build the `(pxl_row, pxl_col)` pair the fixtures are made of.
fn fixture_pixels() -> (Vec<i64>, Vec<i64>) {
    let pxl_row: Vec<i64> = (0..N_SPOTS).map(|i| (i as i64) * 10).collect();
    let pxl_col: Vec<i64> = (0..N_SPOTS).map(|i| (i as i64) * 100).collect();
    (pxl_row, pxl_col)
}

/// Write a minimal spatial h5ad.
///
/// Only what the reader looks at: `obsm/spatial`, optionally the `obs` pixel
/// and array columns, and optionally a `uns/spatial` library.
struct Fixture {
    /// Column order written into `obsm/spatial`.
    orientation: SpatialOrientation,
    /// Whether to write `pxl_row_in_fullres` / `pxl_col_in_fullres`.
    with_pixel_cols: bool,
    /// Write those two columns under scanpy's names, which are swapped relative
    /// to Space Ranger. What every Li et al file looks like.
    scanpy_labels: bool,
    /// Whether to write `array_row` / `array_col`.
    with_array_cols: bool,
    /// Optional `(image_height, image_width, scale_factor)` for a `lowres`
    /// image under `uns/spatial/<lib>`.
    image: Option<(usize, usize, f64)>,
    /// Paint a block of "tissue" into the image, given as
    /// `(row_start, row_end, col_start, col_end)` in image pixels.
    tissue_block: Option<(usize, usize, usize, usize)>,
    /// Extra `obsm` members, to check the reader keys on the exact name.
    extra_obsm: bool,
}

impl Default for Fixture {
    fn default() -> Self {
        Self {
            orientation: SpatialOrientation::Yx,
            with_pixel_cols: true,
            scanpy_labels: false,
            with_array_cols: true,
            image: None,
            tissue_block: None,
            extra_obsm: false,
        }
    }
}

impl Fixture {
    fn write(&self, path: &Path) {
        let file = hdf5::File::create(path).expect("create fixture");
        let (pxl_row, pxl_col) = fixture_pixels();

        let flat: Vec<i64> = (0..N_SPOTS)
            .flat_map(|i| match self.orientation {
                SpatialOrientation::Yx => [pxl_row[i], pxl_col[i]],
                SpatialOrientation::Xy => [pxl_col[i], pxl_row[i]],
            })
            .collect();

        let obsm = file.create_group("obsm").unwrap();
        obsm.new_dataset::<i64>()
            .shape((N_SPOTS, 2))
            .create("spatial")
            .unwrap()
            .write_raw(&flat)
            .unwrap();

        if self.extra_obsm {
            // The Vanderbilt shape: a cropped copy sitting next to the real
            // thing. Deliberately transposed so keying on the wrong member is
            // loud rather than subtle.
            let trimmed: Vec<i64> = flat.iter().rev().cloned().collect();
            obsm.new_dataset::<i64>()
                .shape((N_SPOTS, 2))
                .create("spatial_trim")
                .unwrap()
                .write_raw(&trimmed)
                .unwrap();
        }

        let obs = file.create_group("obs").unwrap();
        if self.with_pixel_cols {
            let (under_row, under_col) = if self.scanpy_labels {
                (&pxl_col, &pxl_row)
            } else {
                (&pxl_row, &pxl_col)
            };
            obs.new_dataset::<i64>()
                .shape(N_SPOTS)
                .create("pxl_row_in_fullres")
                .unwrap()
                .write_raw(under_row)
                .unwrap();
            obs.new_dataset::<i64>()
                .shape(N_SPOTS)
                .create("pxl_col_in_fullres")
                .unwrap()
                .write_raw(under_col)
                .unwrap();
        }
        if self.with_array_cols {
            let idx: Vec<i64> = (0..N_SPOTS as i64).collect();
            obs.new_dataset::<i64>()
                .shape(N_SPOTS)
                .create("array_row")
                .unwrap()
                .write_raw(&idx)
                .unwrap();
            obs.new_dataset::<i64>()
                .shape(N_SPOTS)
                .create("array_col")
                .unwrap()
                .write_raw(&idx)
                .unwrap();
        }

        if let Some((height, width, scalef)) = self.image {
            let lib = file.create_group("uns/spatial/test_library").unwrap();
            let sf = lib.create_group("scalefactors").unwrap();
            for (key, value) in [
                ("tissue_lowres_scalef", scalef),
                ("spot_diameter_fullres", 14.0_f64),
            ] {
                sf.new_dataset::<f64>()
                    .shape(())
                    .create(key)
                    .unwrap()
                    .write_scalar(&value)
                    .unwrap();
            }
            // White slide, optionally with a dark block of tissue on it.
            let mut pixels = vec![1.0_f32; height * width * 3];
            if let Some((r0, r1, c0, c1)) = self.tissue_block {
                for r in r0..r1.min(height) {
                    for c in c0..c1.min(width) {
                        for ch in 0..3 {
                            pixels[(r * width + c) * 3 + ch] = 0.2;
                        }
                    }
                }
            }
            let images = lib.create_group("images").unwrap();
            images
                .new_dataset::<f32>()
                .shape((height, width, 3))
                .create("lowres")
                .unwrap()
                .write_raw(&pixels)
                .unwrap();
        }
    }
}

/// A throwaway path under the OS temp directory.
fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "bixverse_sp_h5ad_{}_{}.h5ad",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_file(&p);
    p
}

///////////////////////
// Orientation, e2e //
///////////////////////

#[test]
fn test_reader_swaps_the_scanpy_order_into_xy() {
    let path = temp_path("scanpy_order");
    Fixture::default().write(&path);

    let data = read_spatial_h5ad(&path, None).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(data.orientation, SpatialOrientation::Yx);
    assert_eq!(data.evidence, OrientationEvidence::ObsPixelColumns);
    assert_eq!(data.n_spots(), N_SPOTS);

    // The contract is (x, y). `pxl_col` is x and spans 0..7700, `pxl_row` is y
    // and spans 0..770. A failure to swap puts the small range first.
    let (pxl_row, pxl_col) = fixture_pixels();
    for i in 0..N_SPOTS {
        assert_eq!(data.coordinates[i].0, pxl_col[i] as f64, "x at spot {i}");
        assert_eq!(data.coordinates[i].1, pxl_row[i] as f64, "y at spot {i}");
    }

    let x_range = data.coordinates.iter().map(|c| c.0).fold(0.0, f64::max);
    let y_range = data.coordinates.iter().map(|c| c.1).fold(0.0, f64::max);
    assert!(
        x_range > y_range * 5.0,
        "x should span an order of magnitude more than y, got {x_range} vs {y_range}"
    );
}

#[test]
fn test_reader_leaves_the_xy_order_alone() {
    let path = temp_path("xy_order");
    Fixture {
        orientation: SpatialOrientation::Xy,
        ..Fixture::default()
    }
    .write(&path);

    let data = read_spatial_h5ad(&path, None).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(data.orientation, SpatialOrientation::Xy);
    assert_eq!(data.evidence, OrientationEvidence::ObsPixelColumns);

    let (pxl_row, pxl_col) = fixture_pixels();
    for i in 0..N_SPOTS {
        assert_eq!(data.coordinates[i].0, pxl_col[i] as f64, "x at spot {i}");
        assert_eq!(data.coordinates[i].1, pxl_row[i] as f64, "y at spot {i}");
    }
}

#[test]
fn test_both_orders_land_on_the_same_coordinates() {
    // The whole point of the detection: the same tissue written two ways comes
    // out identical. If the detection is inverted this is the test that dies,
    // because the two files then disagree by a transpose.
    let a = temp_path("agree_yx");
    let b = temp_path("agree_xy");
    Fixture::default().write(&a);
    Fixture {
        orientation: SpatialOrientation::Xy,
        ..Fixture::default()
    }
    .write(&b);

    let da = read_spatial_h5ad(&a, None).unwrap();
    let db = read_spatial_h5ad(&b, None).unwrap();
    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);

    assert_eq!(da.coordinates, db.coordinates);
}

#[test]
fn test_tissue_overrules_the_obs_labels() {
    // The Li et al shape: the file is (x, y) on disk but carries scanpy's
    // swapped labels, which claim (y, x). The tissue in the image is where the
    // spots are under (x, y), so the image wins and the labels lose.
    let path = temp_path("tissue_vs_labels");
    Fixture {
        orientation: SpatialOrientation::Xy,
        scanpy_labels: true,
        image: Some((200, 900, 0.1)),
        tissue_block: Some((0, 80, 0, 780)),
        ..Fixture::default()
    }
    .write(&path);

    let data = read_spatial_h5ad(&path, None).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(data.evidence, OrientationEvidence::ImageTissue);
    assert_eq!(data.orientation, SpatialOrientation::Xy);

    let (pxl_row, pxl_col) = fixture_pixels();
    for i in 0..N_SPOTS {
        assert_eq!(data.coordinates[i].0, pxl_col[i] as f64, "x at spot {i}");
        assert_eq!(data.coordinates[i].1, pxl_row[i] as f64, "y at spot {i}");
    }
}

#[test]
fn test_labels_alone_would_have_got_it_wrong() {
    // The same file with the image stripped out. Nothing is left but the
    // swapped labels, so the reader takes them and lands on the transpose. This
    // is what the image evidence is for, and it is why the labels rank last.
    let path = temp_path("labels_only");
    Fixture {
        orientation: SpatialOrientation::Xy,
        scanpy_labels: true,
        ..Fixture::default()
    }
    .write(&path);

    let data = read_spatial_h5ad(&path, None).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(data.evidence, OrientationEvidence::ObsPixelColumns);
    assert_eq!(data.orientation, SpatialOrientation::Yx);
    assert!(data.evidence.is_uncertain());
}

#[test]
fn test_image_frame_settles_it_without_the_pixel_columns() {
    // A slide 800 tall and 8000 wide. `pxl_col` runs to 7700, so reading the
    // file as (y, x) puts spots 6900 px below the bottom edge.
    let path = temp_path("image_frame");
    Fixture {
        orientation: SpatialOrientation::Xy,
        with_pixel_cols: false,
        image: Some((80, 800, 0.1)),
        ..Fixture::default()
    }
    .write(&path);

    let data = read_spatial_h5ad(&path, None).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(data.evidence, OrientationEvidence::ImageFrame);
    assert_eq!(data.orientation, SpatialOrientation::Xy);

    let (pxl_row, pxl_col) = fixture_pixels();
    assert_eq!(data.coordinates[N_SPOTS - 1].0, pxl_col[N_SPOTS - 1] as f64);
    assert_eq!(data.coordinates[N_SPOTS - 1].1, pxl_row[N_SPOTS - 1] as f64);
}

#[test]
fn test_reader_admits_when_nothing_settles_it() {
    let path = temp_path("assumed");
    Fixture {
        with_pixel_cols: false,
        ..Fixture::default()
    }
    .write(&path);

    // The default is what `scanpy.read_visium` actually produces, which is
    // (x, y), and what all 236 survey files hold.
    let default = read_spatial_h5ad(&path, None).unwrap();
    assert_eq!(default.evidence, OrientationEvidence::Assumed);
    assert_eq!(default.orientation, SpatialOrientation::Xy);

    // ... and the parameter is honoured rather than ignored.
    let forced = read_spatial_h5ad(
        &path,
        Some(SpatialH5adParams::new(None, SpatialOrientation::Yx)),
    )
    .unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(forced.evidence, OrientationEvidence::Assumed);
    assert_eq!(forced.orientation, SpatialOrientation::Yx);
    assert_ne!(default.coordinates, forced.coordinates);
}

////////////////////
// What is around //
////////////////////

#[test]
fn test_reader_keys_on_the_exact_obsm_name() {
    let path = temp_path("extra_obsm");
    Fixture {
        extra_obsm: true,
        ..Fixture::default()
    }
    .write(&path);

    let data = read_spatial_h5ad(&path, None).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(data.obsm_keys, vec!["spatial", "spatial_trim"]);
    let (pxl_row, pxl_col) = fixture_pixels();
    assert_eq!(data.coordinates[0].0, pxl_col[0] as f64);
    assert_eq!(data.coordinates[0].1, pxl_row[0] as f64);
}

#[test]
fn test_reader_reports_the_missing_array_indices() {
    let path = temp_path("no_array");
    Fixture {
        with_array_cols: false,
        ..Fixture::default()
    }
    .write(&path);

    let data = read_spatial_h5ad(&path, None).unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(!data.has_array_indices);
    assert!(data.has_pixel_columns);
}

#[test]
fn test_reader_survives_without_uns_spatial() {
    let path = temp_path("no_uns");
    Fixture::default().write(&path);

    let data = read_spatial_h5ad(&path, None).unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(data.library_id.is_none());
    assert!(data.scale_factors.is_empty());
    assert!(data.images.is_empty());
    assert_eq!(data.n_spots(), N_SPOTS);
}

#[test]
fn test_reader_passes_the_scale_factors_through() {
    let path = temp_path("scalefactors");
    Fixture {
        image: Some((60, 600, 0.1)),
        ..Fixture::default()
    }
    .write(&path);

    let data = read_spatial_h5ad(&path, None).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(data.library_id.as_deref(), Some("test_library"));
    assert_eq!(data.scale_factors.len(), 2);
    assert_eq!(data.images.len(), 1);
    assert_eq!(data.images[0].key, "lowres");
    assert_eq!((data.images[0].height, data.images[0].width), (60, 600));
}

#[test]
fn test_reader_says_so_when_there_are_no_coordinates() {
    // The bare `filtered_feature_bc_matrix.h5` case: nothing to read, and the
    // message has to say where the coordinates actually live.
    let path = temp_path("no_obsm");
    {
        let file = hdf5::File::create(&path).unwrap();
        file.create_group("obs").unwrap();
    }

    let err = read_spatial_h5ad(&path, None).unwrap_err();
    let _ = std::fs::remove_file(&path);

    let msg = err.to_string();
    assert!(msg.contains("obsm/spatial"), "got: {msg}");
    assert!(msg.contains("load_visium"), "got: {msg}");
}

////////////////////////
// Real files, if any //
////////////////////////

/// Read the two academic collections when they happen to be on this machine.
///
/// Gregor's data, not a fixture. Skips loudly when absent.
#[test]
fn test_real_collections_when_present() {
    let home = match std::env::var("HOME") {
        Ok(h) => PathBuf::from(h),
        Err(_) => return,
    };
    let roots = [
        home.join("Documents/Mestag_Tx/public_data/li_et_al/individual_cancers"),
        home.join("Documents/Mestag_Tx/public_data/vanderbilt_crc_atlas/h5ad"),
    ];

    let mut checked = 0usize;
    for root in roots.iter().filter(|p| p.is_dir()) {
        let mut files: Vec<PathBuf> = std::fs::read_dir(root)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "h5ad"))
            .collect();
        files.sort();

        for path in files.iter().take(3) {
            let data = read_spatial_h5ad(path, None)
                .unwrap_or_else(|e| panic!("{} failed: {e}", path.display()));
            assert!(data.n_spots() > 0);
            assert!(
                data.coordinates
                    .iter()
                    .all(|c| c.0.is_finite() && c.1.is_finite()),
                "{} produced a non-finite coordinate",
                path.display()
            );
            // Every file in both collections that the image evidence could
            // separate came back (x, y), and none contradicted. A file coming
            // back (y, x) means something changed.
            assert_eq!(
                data.orientation,
                SpatialOrientation::Xy,
                "{} was read as (y, x) via {}",
                path.display(),
                data.evidence.as_str()
            );
            checked += 1;
        }
    }

    if checked == 0 {
        eprintln!("skipping: neither public_data collection is on this machine");
    }
}
