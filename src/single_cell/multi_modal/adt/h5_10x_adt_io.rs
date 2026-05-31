//! Helper functions to specifically extract ADT counts from the h5 file.
//! Can be theoretically used for other modalities, but it does return dense
//! data!

use hdf5::File;
use std::path::Path;

use crate::prelude::*;
use crate::single_cell::sc_data::h5_10x_io::*;

/////////////
// Structs //
/////////////

/// Dense modality matrix extracted from a 10x h5 file.
pub struct TenxDenseModality {
    /// Column-major, n_cells x n_features.
    pub counts: Vec<f64>,
    /// Cell barcodes, file order.
    pub barcodes: Vec<String>,
    /// Feature labels, full-array order (ascending).
    pub features: Vec<String>,
    /// Number of cells
    pub n_cells: usize,
    /// Number of features
    pub n_features: usize,
}

////////////////////////
// Extract ADT counts //
////////////////////////

/// Extract one modality (e.g. "Antibody Capture") as a dense matrix.
///
/// ADT-style modalities are small and near-dense, so the streaming binary
/// path is unnecessary.
///
/// ### Params
///
/// * `file_path` - The path to the h5 file.
/// * `version` - The optional [TenxVersion]
/// * `feature_type` - The feature to extract
///
/// ### Returns
///
/// The [TenxDenseModality]
pub fn read_tenx_h5_modality<P: AsRef<Path>>(
    file_path: P,
    version: Option<TenxVersion>,
    feature_type: &str,
) -> Result<TenxDenseModality, BixverseErrors> {
    let file_path = file_path.as_ref();
    let version = resolve_tenx_version(file_path, version)?;

    let ft_path = version.get_feature_type().ok_or_else(|| {
        BixverseErrors::InvalidArgument(
            "modality extraction requires a v3 file (v2 has no feature types)".to_string(),
        )
    })?;

    let file = File::open(file_path)?;

    let feature_types = read_string_dataset(&file.dataset(ft_path)?)?;
    let feature_names = read_string_dataset(&file.dataset(version.get_feature_name().unwrap())?)?;
    let barcodes = read_string_dataset(&file.dataset(version.get_barcodes())?)?;

    let adt_full_idx: Vec<usize> = feature_types
        .iter()
        .enumerate()
        .filter(|(_, ft)| ft.trim() == feature_type)
        .map(|(i, _)| i)
        .collect();

    if adt_full_idx.is_empty() {
        return Err(BixverseErrors::FeatureTypeNotFound {
            requested: feature_type.to_string(),
            found: feature_types.iter().map(|s| s.trim().to_string()).collect(),
        });
    }

    let n_features = adt_full_idx.len();
    let n_cells = barcodes.len();

    let mut full_to_dense = vec![usize::MAX; feature_types.len()];
    for (dense, &full) in adt_full_idx.iter().enumerate() {
        full_to_dense[full] = dense;
    }
    let features: Vec<String> = adt_full_idx
        .iter()
        .map(|&i| feature_names[i].clone())
        .collect();

    let mut counts = vec![0.0f64; n_cells * n_features];

    let indptr: Vec<usize> = file.dataset(version.get_indptr())?.read_1d()?.to_vec();
    let data_ds = file.dataset(version.get_data())?;
    let indices_ds = file.dataset(version.get_indices())?;

    const CELL_CHUNK: usize = 10000;
    for chunk_start in (0..n_cells).step_by(CELL_CHUNK) {
        let chunk_end = (chunk_start + CELL_CHUNK).min(n_cells) - 1;
        let data_start = indptr[chunk_start];
        let data_end = indptr[chunk_end + 1];
        if data_start >= data_end {
            continue;
        }

        let chunk_data: Vec<f32> = data_ds.read_slice_1d(data_start..data_end)?.to_vec();
        let chunk_indices: Vec<usize> = indices_ds.read_slice_1d(data_start..data_end)?.to_vec();

        for cell in chunk_start..=chunk_end {
            let s = indptr[cell] - data_start;
            let e = indptr[cell + 1] - data_start;
            for k in s..e {
                let dense_col = full_to_dense[chunk_indices[k]];
                if dense_col != usize::MAX {
                    // column-major: feature-major blocks of n_cells
                    counts[dense_col * n_cells + cell] = chunk_data[k] as f64;
                }
            }
        }
    }

    Ok(TenxDenseModality {
        counts,
        barcodes,
        features,
        n_cells,
        n_features,
    })
}
