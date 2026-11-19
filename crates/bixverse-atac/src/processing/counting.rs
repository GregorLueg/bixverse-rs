//! Counting fragments into features (peaks, tiles, gene regions) and writing
//! the result as a v5 store.
//!
//! One pass over the fragment file. Each worker collects, per cell, the ids of
//! the features hit; the per-cell lists are merged, sorted and run-length
//! encoded into counts, then written cell by cell. The cell file's gene-major
//! twin comes from the regular conversion. Memory is one `u32` per hit until
//! the cells are written.

use std::path::Path;

use rayon::prelude::*;

use bixverse_sc_data::data_io::CellGeneSparseWriter;
use bixverse_sc_data::gene_file_io::write_gene_file;
use bixverse_sc_data::interval_index::{FeatureInterval, GenomeIntervals, attach_interval_index};
use bixverse_sc_data::v5_io::{FeatureKind, set_cell_ids, set_feature_kind};

use crate::io::fragments::{BarcodeIndex, Fragment, fold_fragments};
use crate::peaks::regions::{ChromRegions, RegionIndex};
use crate::prelude::*;

///////////
// Types //
///////////

/// What a fragment contributes to a feature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CountMode {
    /// Each Tn5 insertion in a feature counts once, so a fragment with both
    /// ends in one feature counts twice (ArchR, Cell Ranger ATAC). The
    /// insertions are the raw start and end of the fragment line, Cell
    /// Ranger's convention, which reproduces its `Peaks` matrix exactly
    /// (SnapATAC2 and ArchR put the right one at `end - 1`).
    #[default]
    Insertion,
    /// Like `Insertion`, but a fragment counts at most once per feature
    /// (SnapATAC2's default).
    PairedInsertion,
    /// Each fragment overlapping a feature counts once (Signac
    /// `FeatureMatrix`).
    Fragment,
}

/// Parse a count mode.
///
/// ### Params
///
/// * `s` - `"insertion"`, `"paired-insertion"` or `"fragment"`
///
/// ### Returns
///
/// The mode, or `None` for anything else.
pub fn parse_count_mode(s: &str) -> Option<CountMode> {
    match s.to_lowercase().replace('_', "-").as_str() {
        "insertion" => Some(CountMode::Insertion),
        "paired-insertion" => Some(CountMode::PairedInsertion),
        "fragment" => Some(CountMode::Fragment),
        _ => None,
    }
}

/// Counts of the requested cells over a feature set, cells x features CSR
/// in raw form.
#[derive(Clone, Debug, Default)]
pub struct FeatureCounts {
    /// Number of features.
    pub n_features: usize,
    /// Per cell, the feature ids with a count, ascending.
    pub indices: Vec<Vec<u32>>,
    /// Per cell, the counts, parallel to `indices`.
    pub counts: Vec<Vec<u32>>,
}

///////////////
// Functions //
///////////////

/// Feature ids one fragment hits under a count mode.
///
/// ### Params
///
/// * `regions` - Features on the fragment's chromosome
/// * `f` - The fragment
/// * `mode` - See [`CountMode`]
/// * `out` - Hits are appended here, one id per count
#[inline]
fn fragment_hits(regions: &ChromRegions, f: &Fragment, mode: CountMode, out: &mut Vec<u32>) {
    let right = f.end;
    match mode {
        CountMode::Fragment => regions.for_each_overlap(f.start, f.end, |id| out.push(id)),
        CountMode::Insertion => {
            regions.for_each_overlap(f.start, f.start + 1, |id| out.push(id));
            regions.for_each_overlap(right, right + 1, |id| out.push(id));
        }
        CountMode::PairedInsertion => {
            let from = out.len();
            regions.for_each_overlap(f.start, f.start + 1, |id| out.push(id));
            let left_end = out.len();
            regions.for_each_overlap(right, right + 1, |id| {
                if !out[from..left_end].contains(&id) {
                    out.push(id);
                }
            });
        }
    }
}

/// Count the fragments of the requested cells into features.
///
/// ### Params
///
/// * `path` - The fragment file
/// * `barcodes` - Cells, in output order
/// * `features` - The features; ids are positions in this slice
/// * `mode` - See [`CountMode`]
///
/// ### Returns
///
/// The [`FeatureCounts`], or an I/O or parse error.
pub fn count_fragments(
    path: &Path,
    barcodes: &[String],
    features: &[FeatureInterval],
    mode: CountMode,
) -> Result<FeatureCounts, AtacErrors> {
    let index = BarcodeIndex::new(barcodes)?;
    let regions = RegionIndex::new(features);
    let n = barcodes.len();

    let hits: Vec<Vec<u32>> = fold_fragments(
        path,
        &index,
        || vec![Vec::new(); n],
        |acc: &mut Vec<Vec<u32>>, chrom: &str, frags: &[Fragment]| {
            let Some(chrom_regions) = regions.chrom(chrom) else {
                return;
            };
            for f in frags {
                fragment_hits(chrom_regions, f, mode, &mut acc[f.cell as usize]);
            }
        },
        |mut a, b| {
            for (x, mut y) in a.iter_mut().zip(b) {
                if x.len() < y.len() {
                    std::mem::swap(x, &mut y);
                }
                x.extend_from_slice(&y);
            }
            a
        },
    )?;

    let (indices, counts): (Vec<Vec<u32>>, Vec<Vec<u32>>) = hits
        .into_par_iter()
        .map(|mut h| {
            h.sort_unstable();
            let mut idx = Vec::new();
            let mut cnt: Vec<u32> = Vec::new();
            for id in h {
                if idx.last() == Some(&id) {
                    *cnt.last_mut().expect("parallel to idx") += 1;
                } else {
                    idx.push(id);
                    cnt.push(1);
                }
            }
            (idx, cnt)
        })
        .unzip();

    Ok(FeatureCounts {
        n_features: features.len(),
        indices,
        counts,
    })
}

/// Write counts as a v5 store with feature kind, interval index and cell IDs.
///
/// The norm stays log-CPM until [`crate::processing::lsi::set_tf_idf_on_store`] switches
/// it.
///
/// ### Params
///
/// * `counts` - From [`count_fragments`]
/// * `features` - The features the counts refer to, store order
/// * `barcodes` - Cell IDs, store order
/// * `kind` - [`FeatureKind::Peak`] or [`FeatureKind::Tile`] (or `Gene` for
///   gene activity)
/// * `target_size` - Target size of the log-CPM norm the store starts
///   with; irrelevant once TF-IDF is set
/// * `cell_path` - Cell-based file to write
/// * `gene_path` - Gene-based file to write
/// * `max_nnz_in_memory` - Non-zeros per phase of the gene file conversion
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The library size of every cell, or a write error.
#[allow(clippy::too_many_arguments)]
pub fn write_counts_store(
    counts: &FeatureCounts,
    features: &[FeatureInterval],
    barcodes: &[String],
    kind: FeatureKind,
    target_size: f32,
    cell_path: &str,
    gene_path: &str,
    max_nnz_in_memory: Option<usize>,
    verbose: bool,
) -> Result<Vec<u64>, AtacErrors> {
    if counts.indices.len() != barcodes.len() || counts.n_features != features.len() {
        return Err(AtacErrors::InvalidArgument(format!(
            "counts are {} x {}, barcodes and features {} x {}",
            counts.indices.len(),
            counts.n_features,
            barcodes.len(),
            features.len()
        )));
    }
    let mut writer =
        CellGeneSparseWriter::new(cell_path, true, barcodes.len(), features.len(), target_size)?;
    let mut lib = Vec::with_capacity(barcodes.len());
    for (cell, (idx, cnt)) in counts.indices.iter().zip(&counts.counts).enumerate() {
        lib.push(cnt.iter().map(|&c| c as u64).sum());
        writer.write_cell_chunk(CsrCellChunk::from_data(cnt, idx, cell, target_size, true))?;
    }
    writer.finalise()?;
    write_gene_file(cell_path, gene_path, max_nnz_in_memory, verbose)?;

    let paths = [cell_path, gene_path];
    set_feature_kind(&paths, kind)?;
    attach_interval_index(&paths, &GenomeIntervals::new(features)?)?;
    set_cell_ids(&paths, barcodes)?;
    Ok(lib)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peaks::regions::region;

    /// Hits of one fragment against `[100, 200)`, `[150, 300)`, `[400, 500)`.
    ///
    /// ### Params
    ///
    /// * `start` - Fragment start
    /// * `end` - Fragment end
    /// * `mode` - Count mode
    ///
    /// ### Returns
    ///
    /// Sorted feature ids, one per count.
    fn hits(start: u32, end: u32, mode: CountMode) -> Vec<u32> {
        let feats = vec![
            region("chr1", 100, 200),
            region("chr1", 150, 300),
            region("chr1", 400, 500),
        ];
        let idx = RegionIndex::new(&feats);
        let mut out = Vec::new();
        let f = Fragment {
            start,
            end,
            cell: 0,
            count: 1,
        };
        fragment_hits(idx.chrom("chr1").unwrap(), &f, mode, &mut out);
        out.sort_unstable();
        out
    }

    #[test]
    fn test_count_modes() {
        // both ends inside feature 0 only
        assert_eq!(hits(110, 141, CountMode::Insertion), vec![0, 0]);
        assert_eq!(hits(110, 141, CountMode::PairedInsertion), vec![0]);
        assert_eq!(hits(110, 141, CountMode::Fragment), vec![0]);
        // spans feature 2 without an end inside it
        assert_eq!(hits(350, 600, CountMode::Insertion), Vec::<u32>::new());
        assert_eq!(hits(350, 600, CountMode::Fragment), vec![2]);
        // left end in 0 and 1, right end in 1
        assert_eq!(hits(160, 251, CountMode::Insertion), vec![0, 1, 1]);
        assert_eq!(hits(160, 251, CountMode::PairedInsertion), vec![0, 1]);
        // right insertion is the raw end: end 100 is inside [100, 200)
        assert_eq!(hits(10, 100, CountMode::Insertion), vec![0]);
        assert_eq!(hits(10, 99, CountMode::Insertion), Vec::<u32>::new());
    }

    #[test]
    fn test_parse_count_mode() {
        assert_eq!(
            parse_count_mode("paired_insertion"),
            Some(CountMode::PairedInsertion)
        );
        assert_eq!(parse_count_mode("Fragment"), Some(CountMode::Fragment));
        assert_eq!(parse_count_mode("cut"), None);
    }
}
