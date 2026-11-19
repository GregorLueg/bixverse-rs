//! The `Peaks` modality of a 10x multiome h5 into a peak store.
//!
//! Cell Ranger ARC writes the peak counts next to the gene counts, with the
//! peak coordinates in `matrix/features/interval` as `chrom:start-end` in BED
//! convention (0-based start, exclusive end). The store is written for a given
//! list of barcodes, in that order, so it shares its cell axis with the RNA
//! store of the same file; the cell-ID hash records it. Every peak is kept, in
//! file order: no feature filter, so feature indices match the h5.

use std::path::Path;

use rustc_hash::FxHashMap;

use bixverse_sc_data::h5::ann_store::AnnFile;
use bixverse_sc_data::h5::tenx::{
    TenxVersion, resolve_tenx_version, validate_feature_types_tenx, write_h5_tenx_streaming,
};
use bixverse_sc_data::store::gene_file::write_gene_file;
use bixverse_sc_data::store::interval_index::{
    Arm, FeatureInterval, GenomeIntervals, attach_interval_index,
};
use bixverse_sc_data::store::quality::{CellOnFileQuality, MinCellQuality};
use bixverse_sc_data::store::v5::{FeatureKind, set_cell_ids, set_feature_kind};

use crate::prelude::*;
use crate::processing::lsi::TF_IDF_SCALE;

/// Feature type of the ATAC modality in a Cell Ranger ARC h5.
pub const TENX_PEAKS_FEATURE_TYPE: &str = "Peaks";

/// Peak coordinates in a 10x v3 h5.
const TENX_INTERVAL_PATH: &str = "matrix/features/interval";

///////////
// Types //
///////////

/// What [`tenx_peaks_to_store`] wrote.
#[derive(Clone, Debug)]
pub struct TenxPeakIngest {
    /// Peak names (`features/id`), store order.
    pub peak_names: Vec<String>,
    /// Peak coordinates, store order.
    pub intervals: Vec<FeatureInterval>,
    /// Total peak counts per cell, store order.
    pub lib_size: Vec<usize>,
    /// Peaks with a count per cell, store order.
    pub nnz: Vec<usize>,
}

///////////////
// Functions //
///////////////

/// Parse a `chrom:start-end` interval.
///
/// The chromosome may itself contain colons (`HLA-A*01:01`), so the split is
/// at the last one.
///
/// ### Params
///
/// * `s` - The interval string
///
/// ### Returns
///
/// `(chrom, start, end)` as written, or [`AtacErrors::IntervalParse`].
pub fn parse_interval(s: &str) -> Result<(String, u32, u32), AtacErrors> {
    let err = || AtacErrors::IntervalParse {
        interval: s.to_string(),
    };
    let (chrom, range) = s.rsplit_once(':').ok_or_else(err)?;
    let (start, end) = range.split_once('-').ok_or_else(err)?;
    let start: u32 = start.trim().parse().map_err(|_| err())?;
    let end: u32 = end.trim().parse().map_err(|_| err())?;
    if chrom.is_empty() || end <= start {
        return Err(err());
    }
    Ok((chrom.to_string(), start, end))
}

/// Map barcodes to their column in the h5.
///
/// ### Params
///
/// * `h5_barcodes` - Barcodes in the h5, column order
/// * `barcodes` - Barcodes to look up
///
/// ### Returns
///
/// One column per barcode, or [`AtacErrors::BarcodesNotFound`].
fn barcode_columns(h5_barcodes: &[String], barcodes: &[String]) -> Result<Vec<usize>, AtacErrors> {
    let lookup: FxHashMap<&str, usize> = h5_barcodes
        .iter()
        .enumerate()
        .map(|(i, b)| (b.as_str(), i))
        .collect();
    let mut missing = barcodes.iter().filter(|b| !lookup.contains_key(b.as_str()));
    if let Some(first) = missing.next() {
        return Err(AtacErrors::BarcodesNotFound {
            n_missing: 1 + missing.count(),
            n_requested: barcodes.len(),
            example: first.clone(),
        });
    }
    Ok(barcodes.iter().map(|b| lookup[b.as_str()]).collect())
}

/// Write the `Peaks` modality of a 10x multiome h5 as a v5 peak store.
///
/// Both files are written: the cell file streamed from the h5, the gene file
/// converted from it. Then the feature kind (`Peak`), the interval index and
/// the cell IDs go into the headers. The norm stays log-CPM until
/// [`crate::processing::lsi::set_tf_idf_on_store`] switches it, since the IDF depends on
/// the final cell set.
///
/// ### Params
///
/// * `h5_path` - The Cell Ranger ARC h5 (v3 layout)
/// * `cell_path` - Cell-based file to write
/// * `gene_path` - Gene-based (peak-major) file to write
/// * `barcodes` - Cells to write, in store order; typically the cell IDs of
///   the RNA store of the same experiment
/// * `max_nnz_in_memory` - Non-zeros held per phase of the gene file
///   conversion, `None` for one phase
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The [`TenxPeakIngest`], or an error for a v2 file, a file without
/// `Peaks`, unknown barcodes or a malformed interval.
pub fn tenx_peaks_to_store<P: AsRef<Path>>(
    h5_path: P,
    cell_path: &str,
    gene_path: &str,
    barcodes: &[String],
    max_nnz_in_memory: Option<usize>,
    verbose: bool,
) -> Result<TenxPeakIngest, AtacErrors> {
    let h5_path = h5_path.as_ref();
    let version = resolve_tenx_version(h5_path, None)?;
    if version != TenxVersion::V3 {
        return Err(AtacErrors::InvalidArgument(
            "a 10x h5 with peaks has the v3 layout (feature types)".into(),
        ));
    }

    let file = AnnFile::open(h5_path)?;
    let columns = barcode_columns(&file.read_strings(version.get_barcodes())?, barcodes)?;
    let peak_rows = validate_feature_types_tenx(h5_path, version, Some(TENX_PEAKS_FEATURE_TYPE))?;

    let all_ids = file.read_strings(version.get_feature_id())?;
    let all_intervals = file.read_strings(TENX_INTERVAL_PATH)?;
    let peak_names: Vec<String> = peak_rows.iter().map(|&r| all_ids[r].clone()).collect();
    let intervals = peak_rows
        .iter()
        .map(|&r| {
            let (chrom, start, end) = parse_interval(all_intervals[r].trim())?;
            Ok(FeatureInterval {
                chrom,
                start,
                end,
                strand: 0,
                arm: Arm::Unknown,
            })
        })
        .collect::<Result<Vec<_>, AtacErrors>>()?;
    drop(file);

    let mut quality = CellOnFileQuality::new(columns, peak_rows);
    quality.generate_maps_sets();
    let cell_qc = MinCellQuality {
        min_unique_genes: 0,
        min_lib_size: 0,
        min_cells: 0,
        target_size: TF_IDF_SCALE,
    };
    let written = write_h5_tenx_streaming(
        h5_path,
        Path::new(cell_path),
        version,
        &quality,
        cell_qc,
        verbose,
    )?;
    write_gene_file(cell_path, gene_path, max_nnz_in_memory, verbose)?;

    let paths = [cell_path, gene_path];
    set_feature_kind(&paths, FeatureKind::Peak)?;
    attach_interval_index(&paths, &GenomeIntervals::new(&intervals)?)?;
    set_cell_ids(&paths, barcodes)?;

    Ok(TenxPeakIngest {
        peak_names,
        intervals,
        lib_size: written.lib_size,
        nnz: written.nnz,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_interval_bed_convention() {
        assert_eq!(
            parse_interval("chr1:9782-10672").unwrap(),
            ("chr1".to_string(), 9782, 10672)
        );
        assert_eq!(
            parse_interval("HLA-A*01:01:1-50").unwrap(),
            ("HLA-A*01:01".to_string(), 1, 50)
        );
    }

    #[test]
    fn test_parse_interval_rejects_malformed() {
        for s in ["chr1", "chr1:10", "chr1:a-b", ":1-2", "chr1:5-5", "NA"] {
            assert!(parse_interval(s).is_err(), "{s}");
        }
    }

    #[test]
    fn test_barcode_columns_reports_missing() {
        let h5: Vec<String> = ["A", "B", "C"].map(String::from).to_vec();
        let want: Vec<String> = ["C", "A"].map(String::from).to_vec();
        assert_eq!(barcode_columns(&h5, &want).unwrap(), vec![2, 0]);
        let bad: Vec<String> = ["A", "X", "Y"].map(String::from).to_vec();
        match barcode_columns(&h5, &bad) {
            Err(AtacErrors::BarcodesNotFound {
                n_missing,
                n_requested,
                example,
            }) => {
                assert_eq!((n_missing, n_requested, example.as_str()), (2, 3, "X"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
