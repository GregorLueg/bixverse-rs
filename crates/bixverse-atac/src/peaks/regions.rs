//! Genomic regions: BED input, genome tiles, merging and a fast overlap
//! index for the fragment sweeps.
//!
//! Coordinates are 0-based half-open throughout (BED). The overlap index
//! answers "which regions overlap `[s, e)`" without allocating, which the
//! counting pass asks once or twice per fragment.

use std::io::BufRead;
use std::path::Path;

use rustc_hash::FxHashMap;

use bixverse_sc_data::store::interval_index::{Arm, FeatureInterval};

use crate::io::open_text;
use crate::prelude::*;

///////////
// Input //
///////////

/// A region with no strand or arm.
///
/// ### Params
///
/// * `chrom` - Chromosome
/// * `start` - 0-based start
/// * `end` - Exclusive end
///
/// ### Returns
///
/// The [`FeatureInterval`].
pub fn region(chrom: &str, start: u32, end: u32) -> FeatureInterval {
    FeatureInterval {
        chrom: chrom.to_string(),
        start,
        end,
        strand: 0,
        arm: Arm::Unknown,
    }
}

/// Read the first three columns of a BED file (plain, gzip or BGZF, by
/// magic bytes).
///
/// `#`, `track` and `browser` lines are skipped. Fields may be separated by
/// tabs or spaces. Zero-length records (`end == start`, as some insertion and
/// blacklist files carry) hold no base and are skipped.
///
/// ### Params
///
/// * `path` - The BED file
///
/// ### Returns
///
/// The regions in file order, or an error for a malformed line.
pub fn read_bed(path: &Path) -> Result<Vec<FeatureInterval>, AtacErrors> {
    let mut out = Vec::new();
    for line in open_text(path)?.lines() {
        let line = line?;
        let line = line.trim_end();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with("track")
            || line.starts_with("browser")
        {
            continue;
        }
        let bad = || AtacErrors::InvalidArgument(format!("malformed BED line '{line}'"));
        let mut f = line.split_whitespace();
        let (Some(chrom), Some(start), Some(end)) = (f.next(), f.next(), f.next()) else {
            return Err(bad());
        };
        let start: u32 = start.parse().map_err(|_| bad())?;
        let end: u32 = end.parse().map_err(|_| bad())?;
        if end < start {
            return Err(bad());
        }
        if end > start {
            out.push(region(chrom, start, end));
        }
    }
    Ok(out)
}

/// Fixed-width tiles over chromosomes, the last one per chromosome cut at
/// its end (GenomicRanges `tileGenome(cut.last.tile.in.chrom = TRUE)`).
///
/// ### Params
///
/// * `chrom_sizes` - `(chromosome, length)` in the order the tiles should
///   follow
/// * `width` - Tile width in base pairs
///
/// ### Returns
///
/// The tiles in genomic order.
pub fn genome_tiles(chrom_sizes: &[(String, u32)], width: u32) -> Vec<FeatureInterval> {
    let width = width.max(1);
    chrom_sizes
        .iter()
        .flat_map(|(chrom, len)| {
            (0..len.div_ceil(width))
                .map(move |i| region(chrom, i * width, ((i + 1) * width).min(*len)))
        })
        .collect()
}

/// Merge overlapping and abutting regions (GenomicRanges `reduce`).
///
/// ### Params
///
/// * `regions` - Regions in any order
///
/// ### Returns
///
/// Disjoint regions sorted by chromosome (first appearance order) and start,
/// plus for each the indices of the input regions it absorbed.
pub fn reduce_regions(regions: &[FeatureInterval]) -> (Vec<FeatureInterval>, Vec<Vec<usize>>) {
    let mut chrom_order: FxHashMap<&str, usize> = FxHashMap::default();
    for r in regions {
        let n = chrom_order.len();
        chrom_order.entry(r.chrom.as_str()).or_insert(n);
    }
    let mut idx: Vec<usize> = (0..regions.len()).collect();
    idx.sort_by_key(|&i| {
        (
            chrom_order[regions[i].chrom.as_str()],
            regions[i].start,
            regions[i].end,
        )
    });

    let mut merged: Vec<FeatureInterval> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for i in idx {
        let r = &regions[i];
        match merged.last_mut() {
            Some(m) if m.chrom == r.chrom && r.start <= m.end => {
                m.end = m.end.max(r.end);
                members.last_mut().expect("parallel to merged").push(i);
            }
            _ => {
                merged.push(region(&r.chrom, r.start, r.end));
                members.push(vec![i]);
            }
        }
    }
    (merged, members)
}

///////////////////
// Overlap index //
///////////////////

/// Regions on one chromosome, sorted by start.
#[derive(Clone, Debug, Default)]
pub struct ChromRegions {
    /// Starts, ascending.
    starts: Vec<u32>,
    /// Ends, parallel to `starts`.
    ends: Vec<u32>,
    /// Feature id of each region, parallel to `starts`.
    ids: Vec<u32>,
    /// Longest region, bounds the backward scan.
    max_len: u32,
}

impl ChromRegions {
    /// Visit every region overlapping `[s, e)`.
    ///
    /// Regions starting before `e` are found by binary search; scanning back
    /// stops once even the longest region could not reach `s`.
    ///
    /// ### Params
    ///
    /// * `s` - Query start
    /// * `e` - Query end (exclusive), `e > s`
    /// * `f` - Called with the feature id of each overlapping region
    #[inline]
    pub fn for_each_overlap(&self, s: u32, e: u32, mut f: impl FnMut(u32)) {
        let hi = self.starts.partition_point(|&x| x < e);
        let floor = s.saturating_sub(self.max_len);
        for j in (0..hi).rev() {
            if self.starts[j] < floor {
                break;
            }
            if self.ends[j] > s {
                f(self.ids[j]);
            }
        }
    }
}

/// Overlap index over a feature set, by chromosome name.
///
/// Kept apart from the store's `GenomeIntervals`, which answers the same
/// question: the fragment sweeps resolve the chromosome once per run of
/// fragments here (a hash lookup) and then visit overlaps through a callback
/// without allocating, where `GenomeIntervals` looks the chromosome up by a
/// linear scan and returns a `Vec` on every query.
#[derive(Clone, Debug, Default)]
pub struct RegionIndex {
    /// Regions per chromosome.
    chroms: FxHashMap<String, ChromRegions>,
}

impl RegionIndex {
    /// Index regions; feature ids are their positions in `regions`.
    ///
    /// ### Params
    ///
    /// * `regions` - The features, any order, overlaps allowed
    ///
    /// ### Returns
    ///
    /// The index.
    pub fn new(regions: &[FeatureInterval]) -> Self {
        let mut chroms: FxHashMap<String, Vec<(u32, u32, u32)>> = FxHashMap::default();
        for (i, r) in regions.iter().enumerate() {
            chroms
                .entry(r.chrom.clone())
                .or_default()
                .push((r.start, r.end, i as u32));
        }
        let chroms = chroms
            .into_iter()
            .map(|(name, mut v)| {
                v.sort_unstable();
                let cr = ChromRegions {
                    max_len: v.iter().map(|&(s, e, _)| e - s).max().unwrap_or(0),
                    starts: v.iter().map(|x| x.0).collect(),
                    ends: v.iter().map(|x| x.1).collect(),
                    ids: v.iter().map(|x| x.2).collect(),
                };
                (name, cr)
            })
            .collect();
        Self { chroms }
    }

    /// Regions of one chromosome.
    ///
    /// ### Params
    ///
    /// * `chrom` - Chromosome name
    ///
    /// ### Returns
    ///
    /// The regions, or `None` for a chromosome without features.
    #[inline]
    pub fn chrom(&self, chrom: &str) -> Option<&ChromRegions> {
        self.chroms.get(chrom)
    }
}

/// Whether each region overlaps any of another set, e.g. a blacklist.
///
/// ### Params
///
/// * `regions` - The regions to test
/// * `other` - The set to test against
///
/// ### Returns
///
/// One flag per region, in region order.
pub fn overlaps_any(regions: &[FeatureInterval], other: &[FeatureInterval]) -> Vec<bool> {
    let index = RegionIndex::new(other);
    regions
        .iter()
        .map(|r| {
            let mut hit = false;
            if let Some(c) = index.chrom(&r.chrom) {
                c.for_each_overlap(r.start, r.end, |_| hit = true);
            }
            hit
        })
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlaps_any() {
        let black = vec![region("chr1", 100, 200), region("chr2", 0, 50)];
        let peaks = vec![
            region("chr1", 50, 101),
            region("chr1", 200, 300),
            region("chr2", 10, 20),
            region("chr3", 0, 10),
        ];
        assert_eq!(overlaps_any(&peaks, &black), vec![true, false, true, false]);
    }

    #[test]
    fn test_read_bed_takes_spaces_and_skips_empty_records() {
        let p = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        std::fs::write(
            &p,
            "track name=x\nchr1 10 20\nchr1\t30\t30\tzero\nchr2\t5\t9\tname\t0\t+\n",
        )
        .unwrap();
        let r = read_bed(&p).unwrap();
        assert_eq!(r, vec![region("chr1", 10, 20), region("chr2", 5, 9)]);
        std::fs::write(&p, "chr1\t20\t10\n").unwrap();
        assert!(read_bed(&p).is_err());
    }

    #[test]
    fn test_overlap_index_matches_brute_force() {
        let regions = vec![
            region("chr1", 100, 200),
            region("chr1", 150, 400),
            region("chr1", 500, 510),
            region("chr1", 0, 1000),
            region("chr2", 100, 200),
        ];
        let idx = RegionIndex::new(&regions);
        for (s, e) in [
            (0, 1),
            (99, 100),
            (100, 101),
            (199, 201),
            (450, 505),
            (999, 1500),
        ] {
            let mut got = Vec::new();
            idx.chrom("chr1")
                .unwrap()
                .for_each_overlap(s, e, |i| got.push(i));
            got.sort_unstable();
            let want: Vec<u32> = regions
                .iter()
                .enumerate()
                .filter(|(_, r)| r.chrom == "chr1" && r.start < e && r.end > s)
                .map(|(i, _)| i as u32)
                .collect();
            assert_eq!(got, want, "query [{s}, {e})");
        }
        assert!(idx.chrom("chrX").is_none());
    }

    #[test]
    fn test_genome_tiles_cut_the_last_tile() {
        let tiles = genome_tiles(&[("chr1".into(), 12), ("chr2".into(), 5)], 5);
        let got: Vec<(&str, u32, u32)> = tiles
            .iter()
            .map(|t| (t.chrom.as_str(), t.start, t.end))
            .collect();
        assert_eq!(
            got,
            vec![
                ("chr1", 0, 5),
                ("chr1", 5, 10),
                ("chr1", 10, 12),
                ("chr2", 0, 5)
            ]
        );
    }

    #[test]
    fn test_reduce_merges_overlapping_and_abutting() {
        let regions = vec![
            region("chr1", 50, 60),
            region("chr1", 10, 20),
            region("chr1", 20, 30),
            region("chr1", 25, 40),
            region("chr2", 1, 2),
        ];
        let (merged, members) = reduce_regions(&regions);
        let got: Vec<(&str, u32, u32)> = merged
            .iter()
            .map(|t| (t.chrom.as_str(), t.start, t.end))
            .collect();
        assert_eq!(
            got,
            vec![("chr1", 10, 40), ("chr1", 50, 60), ("chr2", 1, 2)]
        );
        assert_eq!(members, vec![vec![1, 2, 3], vec![0], vec![4]]);
    }
}
