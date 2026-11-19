//! Genome interval index: feature to chromosome, start, end, strand and arm.
//!
//! Coordinates are 0-based and half-open (`[start, end)`, the BED
//! convention). Features keep the store's order; the index adds a
//! permutation that sorts them by `(chromosome, start)` and the longest
//! feature per chromosome, so an overlap query is a binary search plus a short
//! scan. A store written in genomic order (ATAC tiles and peaks) has the
//! identity permutation, flagged and not stored, and a region is a contiguous
//! feature range.
//!
//! Lives in a v5 tail section ([`SectionKind::IntervalIndex`]) and can be
//! attached to a finished store, since gene coordinates arrive from a GTF
//! after ingest. Section layout (little endian):
//!
//! ```text
//! u32 n_chroms, u32 flags (bit 0: identity order), u64 n_features
//! per chromosome: u16 name length, name bytes (UTF-8)
//! padding to 4 bytes
//! per feature, store order: u16 chrom, i8 strand, u8 arm, u32 start, u32 end
//! u32 order[n_features]            (absent with the identity flag)
//! u32 chrom_offsets[n_chroms + 1]  into the sorted order
//! u32 max_len[n_chroms]
//! ```

use rustc_hash::FxHashMap;
use std::ops::Range;

use crate::prelude::*;
use crate::store::reader::ParallelSparseReader;
use crate::store::v5::{SectionKind, SectionWriter, rewrite_sections};

/// Bytes per feature record in the section.
const FEATURE_RECORD_LEN: usize = 12;

/// Section flag: features are already in `(chromosome, start)` order.
const FLAG_IDENTITY_ORDER: u32 = 1;

///////////
// Types //
///////////

/// Chromosome arm of a feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// Short arm.
    P,
    /// Long arm.
    Q,
    /// Not annotated (or acrocentric short arm without features).
    Unknown,
}

impl Arm {
    /// The byte stored per feature.
    ///
    /// ### Returns
    ///
    /// 0 unknown, 1 p, 2 q.
    fn to_u8(self) -> u8 {
        match self {
            Arm::Unknown => 0,
            Arm::P => 1,
            Arm::Q => 2,
        }
    }

    /// Parse the stored byte.
    ///
    /// ### Params
    ///
    /// * `b` - The byte
    ///
    /// ### Returns
    ///
    /// The arm, or [`ScDataErrors::V5HeaderInvalid`] for an unknown byte.
    fn from_u8(b: u8) -> Result<Self, ScDataErrors> {
        match b {
            0 => Ok(Arm::Unknown),
            1 => Ok(Arm::P),
            2 => Ok(Arm::Q),
            other => Err(ScDataErrors::V5HeaderInvalid(format!(
                "unknown arm {other}"
            ))),
        }
    }
}

/// One feature's genomic location, the input of [`GenomeIntervals::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureInterval {
    /// Chromosome name, as in the reference (`"chr1"`, `"1"`, ...).
    pub chrom: String,
    /// 0-based start, inclusive.
    pub start: u32,
    /// 0-based end, exclusive.
    pub end: u32,
    /// `1` forward, `-1` reverse, `0` unstranded.
    pub strand: i8,
    /// Chromosome arm.
    pub arm: Arm,
}

/// The genome interval index of a store's features.
#[derive(Clone, Debug, PartialEq)]
pub struct GenomeIntervals {
    /// Chromosome names, in first-seen order of the sorted features.
    chroms: Vec<String>,
    /// Chromosome of every feature, an index into `chroms`.
    chrom: Vec<u16>,
    /// Start of every feature.
    start: Vec<u32>,
    /// End of every feature.
    end: Vec<u32>,
    /// Strand of every feature.
    strand: Vec<i8>,
    /// Arm of every feature.
    arm: Vec<Arm>,
    /// Feature indices sorted by `(chromosome, start, end)`. Empty when the
    /// features are already in that order.
    order: Vec<u32>,
    /// Where each chromosome's run starts in the sorted order.
    chrom_offsets: Vec<u32>,
    /// Longest feature per chromosome, which bounds the overlap scan.
    max_len: Vec<u32>,
}

impl GenomeIntervals {
    /// Build the index from one location per feature, in store order.
    ///
    /// Chromosomes are numbered in the order they first appear; sort the
    /// features (or pass them sorted) for a natural chromosome order.
    ///
    /// ### Params
    ///
    /// * `features` - One location per feature
    ///
    /// ### Returns
    ///
    /// The index, or [`ScDataErrors::InvalidArgument`] for an empty interval
    /// or more than `u16::MAX` chromosomes.
    pub fn new(features: &[FeatureInterval]) -> Result<Self, ScDataErrors> {
        let mut chroms: Vec<String> = Vec::new();
        let mut lookup: FxHashMap<&str, u16> = FxHashMap::default();
        let mut chrom = Vec::with_capacity(features.len());
        for (i, f) in features.iter().enumerate() {
            if f.end <= f.start {
                return Err(ScDataErrors::InvalidArgument(format!(
                    "feature {i} has an empty interval [{}, {})",
                    f.start, f.end
                )));
            }
            let id = match lookup.get(f.chrom.as_str()) {
                Some(&id) => id,
                None => {
                    let id = u16::try_from(chroms.len()).map_err(|_| {
                        ScDataErrors::InvalidArgument("more than 65535 chromosomes".into())
                    })?;
                    chroms.push(f.chrom.clone());
                    lookup.insert(f.chrom.as_str(), id);
                    id
                }
            };
            chrom.push(id);
        }

        let mut index = Self {
            chroms,
            chrom,
            start: features.iter().map(|f| f.start).collect(),
            end: features.iter().map(|f| f.end).collect(),
            strand: features.iter().map(|f| f.strand).collect(),
            arm: features.iter().map(|f| f.arm).collect(),
            order: Vec::new(),
            chrom_offsets: Vec::new(),
            max_len: Vec::new(),
        };
        index.build_order();
        Ok(index)
    }

    /// Sort the features, then derive the per-chromosome offsets and maxima.
    fn build_order(&mut self) {
        let n = self.start.len();
        let key = |i: usize| (self.chrom[i], self.start[i], self.end[i]);
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&i| key(i as usize));

        let n_chroms = self.chroms.len();
        let mut chrom_offsets = vec![0u32; n_chroms + 1];
        let mut max_len = vec![0u32; n_chroms];
        for &i in &order {
            let c = self.chrom[i as usize] as usize;
            chrom_offsets[c + 1] += 1;
            max_len[c] = max_len[c].max(self.end[i as usize] - self.start[i as usize]);
        }
        for c in 0..n_chroms {
            chrom_offsets[c + 1] += chrom_offsets[c];
        }

        let identity = order.iter().enumerate().all(|(k, &i)| k == i as usize);
        self.order = if identity { Vec::new() } else { order };
        self.chrom_offsets = chrom_offsets;
        self.max_len = max_len;
    }

    /// Number of features.
    ///
    /// ### Returns
    ///
    /// The feature count.
    pub fn len(&self) -> usize {
        self.start.len()
    }

    /// Whether the index holds no features.
    ///
    /// ### Returns
    ///
    /// `true` for an empty index.
    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }

    /// Whether the features are stored in genomic order, so a region is a
    /// contiguous feature range.
    ///
    /// ### Returns
    ///
    /// `true` for the identity order.
    pub fn is_genomic_order(&self) -> bool {
        self.order.is_empty()
    }

    /// The location of feature `i`.
    ///
    /// ### Params
    ///
    /// * `i` - Feature index
    ///
    /// ### Returns
    ///
    /// The location, `None` past the last feature.
    pub fn feature(&self, i: usize) -> Option<FeatureInterval> {
        (i < self.len()).then(|| FeatureInterval {
            chrom: self.chroms[self.chrom[i] as usize].clone(),
            start: self.start[i],
            end: self.end[i],
            strand: self.strand[i],
            arm: self.arm[i],
        })
    }

    /// The chromosome names.
    ///
    /// ### Returns
    ///
    /// The names, in index order.
    pub fn chroms(&self) -> &[String] {
        &self.chroms
    }

    /// Feature at sorted position `k`.
    ///
    /// ### Params
    ///
    /// * `k` - Position in the sorted order
    ///
    /// ### Returns
    ///
    /// The feature index.
    #[inline]
    fn sorted(&self, k: usize) -> usize {
        if self.order.is_empty() {
            k
        } else {
            self.order[k] as usize
        }
    }

    /// Sorted positions `lo..hi` of the features on `chrom` that may overlap
    /// `[start, end)`: start below `end`, and not so far left that even the
    /// longest feature of the chromosome ends before `start`.
    ///
    /// ### Params
    ///
    /// * `chrom` - Chromosome name
    /// * `start` - Query start
    /// * `end` - Query end
    ///
    /// ### Returns
    ///
    /// The candidate range, empty for an unknown chromosome.
    fn candidates(&self, chrom: &str, start: u32, end: u32) -> Range<usize> {
        let Some(c) = self.chroms.iter().position(|n| n == chrom) else {
            return 0..0;
        };
        let (from, to) = (
            self.chrom_offsets[c] as usize,
            self.chrom_offsets[c + 1] as usize,
        );
        let starts = |k: usize| self.start[self.sorted(k)];
        let floor = start.saturating_sub(self.max_len[c]);
        let lo = from + partition(to - from, |k| starts(from + k) < floor);
        let hi = from + partition(to - from, |k| starts(from + k) < end);
        lo..hi.max(lo)
    }

    /// Features overlapping `[start, end)` on `chrom`, in genomic order.
    ///
    /// ### Params
    ///
    /// * `chrom` - Chromosome name
    /// * `start` - Query start, 0-based inclusive
    /// * `end` - Query end, exclusive
    ///
    /// ### Returns
    ///
    /// Feature indices, empty for an unknown chromosome or an empty query.
    pub fn overlapping(&self, chrom: &str, start: u32, end: u32) -> Vec<usize> {
        if end <= start {
            return Vec::new();
        }
        self.candidates(chrom, start, end)
            .map(|k| self.sorted(k))
            .filter(|&i| self.end[i] > start)
            .collect()
    }

    /// Features overlapping a window of `half_width` either side of `centre`.
    ///
    /// ### Params
    ///
    /// * `chrom` - Chromosome name
    /// * `centre` - Window centre
    /// * `half_width` - Distance either side
    ///
    /// ### Returns
    ///
    /// Feature indices in genomic order, see [`Self::overlapping`].
    pub fn window(&self, chrom: &str, centre: u32, half_width: u32) -> Vec<usize> {
        self.overlapping(
            chrom,
            centre.saturating_sub(half_width),
            centre.saturating_add(half_width),
        )
    }

    /// The features overlapping `[start, end)` as one index range.
    ///
    /// Only defined in genomic order, where the overlap set is contiguous.
    ///
    /// ### Params
    ///
    /// * `chrom` - Chromosome name
    /// * `start` - Query start
    /// * `end` - Query end
    ///
    /// ### Returns
    ///
    /// The range, `None` unless the index is in genomic order.
    pub fn overlapping_range(&self, chrom: &str, start: u32, end: u32) -> Option<Range<usize>> {
        if !self.is_genomic_order() {
            return None;
        }
        let hits = self.overlapping(chrom, start, end);
        Some(match (hits.first(), hits.last()) {
            (Some(&a), Some(&b)) => a..b + 1,
            _ => 0..0,
        })
    }

    /// Serialise into the section layout, see the module docs.
    ///
    /// ### Returns
    ///
    /// The section bytes.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(16 + self.len() * (FEATURE_RECORD_LEN + 4));
        b.extend_from_slice(&(self.chroms.len() as u32).to_le_bytes());
        let flags = if self.is_genomic_order() {
            FLAG_IDENTITY_ORDER
        } else {
            0
        };
        b.extend_from_slice(&flags.to_le_bytes());
        b.extend_from_slice(&(self.len() as u64).to_le_bytes());
        for name in &self.chroms {
            b.extend_from_slice(&(name.len() as u16).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
        }
        b.resize(b.len().next_multiple_of(4), 0);
        for i in 0..self.len() {
            b.extend_from_slice(&self.chrom[i].to_le_bytes());
            b.push(self.strand[i] as u8);
            b.push(self.arm[i].to_u8());
            b.extend_from_slice(&self.start[i].to_le_bytes());
            b.extend_from_slice(&self.end[i].to_le_bytes());
        }
        for v in self
            .order
            .iter()
            .chain(&self.chrom_offsets)
            .chain(&self.max_len)
        {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    /// Parse the section layout, see the module docs.
    ///
    /// ### Params
    ///
    /// * `b` - The section bytes
    ///
    /// ### Returns
    ///
    /// The index, or [`ScDataErrors::V5HeaderInvalid`] for a malformed
    /// section.
    pub(crate) fn decode(b: &[u8]) -> Result<Self, ScDataErrors> {
        let bad = || ScDataErrors::V5HeaderInvalid("malformed interval index".into());
        let take = |at: usize, n: usize| b.get(at..at + n).ok_or_else(bad);
        let u32_at = |at: usize| -> Result<u32, ScDataErrors> {
            Ok(u32::from_le_bytes(
                take(at, 4)?.try_into().expect("4 bytes"),
            ))
        };

        let n_chroms = u32_at(0)? as usize;
        let identity = u32_at(4)? & FLAG_IDENTITY_ORDER != 0;
        let n = u64::from_le_bytes(take(8, 8)?.try_into().expect("8 bytes")) as usize;
        let mut at = 16;
        let mut chroms = Vec::with_capacity(n_chroms);
        for _ in 0..n_chroms {
            let len = u16::from_le_bytes(take(at, 2)?.try_into().expect("2 bytes")) as usize;
            let name = std::str::from_utf8(take(at + 2, len)?).map_err(|_| bad())?;
            chroms.push(name.to_string());
            at += 2 + len;
        }
        at = at.next_multiple_of(4);

        let records = take(at, n.checked_mul(FEATURE_RECORD_LEN).ok_or_else(bad)?)?;
        let mut idx = Self {
            chroms,
            chrom: Vec::with_capacity(n),
            start: Vec::with_capacity(n),
            end: Vec::with_capacity(n),
            strand: Vec::with_capacity(n),
            arm: Vec::with_capacity(n),
            order: Vec::new(),
            chrom_offsets: Vec::new(),
            max_len: Vec::new(),
        };
        for r in records.as_chunks::<FEATURE_RECORD_LEN>().0 {
            let c = u16::from_le_bytes([r[0], r[1]]);
            if c as usize >= n_chroms {
                return Err(bad());
            }
            idx.chrom.push(c);
            idx.strand.push(r[2] as i8);
            idx.arm.push(Arm::from_u8(r[3])?);
            idx.start
                .push(u32::from_le_bytes(r[4..8].try_into().expect("4 bytes")));
            idx.end
                .push(u32::from_le_bytes(r[8..12].try_into().expect("4 bytes")));
        }
        at += records.len();

        let tail_len = if identity { 0 } else { n } + (n_chroms + 1) + n_chroms;
        let tail: Vec<u32> = take(at, tail_len * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect();
        let (order, rest) = tail.split_at(if identity { 0 } else { n });
        let (offsets, max_len) = rest.split_at(n_chroms + 1);
        if order.iter().any(|&i| i as usize >= n) || offsets.last().copied() != Some(n as u32) {
            return Err(bad());
        }
        idx.order = order.to_vec();
        idx.chrom_offsets = offsets.to_vec();
        idx.max_len = max_len.to_vec();
        Ok(idx)
    }
}

/// First position in `0..n` where `below` turns false; `below` must be
/// monotone (true then false).
///
/// ### Params
///
/// * `n` - Length of the range
/// * `below` - The predicate
///
/// ### Returns
///
/// The partition point.
fn partition(n: usize, below: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if below(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

///////////
// Store //
///////////

/// Attach (or replace) the interval index of a v5 store.
///
/// ### Params
///
/// * `paths` - The store's files (cell file, gene file)
/// * `index` - One interval per feature, in the store's feature order
///
/// ### Returns
///
/// `Ok(())`, or an error for a file that is not v5 or a feature count other
/// than the store's.
pub fn attach_interval_index(paths: &[&str], index: &GenomeIntervals) -> Result<(), ScDataErrors> {
    for &p in paths {
        let n = ParallelSparseReader::new(p)?.dims().total_genes;
        if n != index.len() {
            return Err(ScDataErrors::InvalidArgument(format!(
                "interval index has {} features for a store of {n}",
                index.len()
            )));
        }
    }
    let bytes = index.encode();
    for &p in paths {
        let mut sections = SectionWriter::default();
        sections.push_bytes(SectionKind::IntervalIndex, bytes.clone());
        rewrite_sections(p, sections, |_| {})?;
    }
    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A spread of features on three chromosomes, deliberately out of
    /// genomic order, with one long feature that spans many short ones.
    fn features() -> Vec<FeatureInterval> {
        let mut out = Vec::new();
        let mut state = 7u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for i in 0..400 {
            let chrom = ["chr2", "chr1", "chrX"][i % 3];
            let start = next() % 1_000_000;
            let len = 1 + next() % 5_000;
            out.push(FeatureInterval {
                chrom: chrom.to_string(),
                start,
                end: start + len,
                strand: [1, -1, 0][i % 3],
                arm: if start < 500_000 { Arm::P } else { Arm::Q },
            });
        }
        out.push(FeatureInterval {
            chrom: "chr1".into(),
            start: 100,
            end: 900_000,
            strand: 1,
            arm: Arm::Unknown,
        });
        out
    }

    /// Brute force: every feature overlapping the query, in genomic order.
    fn brute(f: &[FeatureInterval], chrom: &str, start: u32, end: u32) -> Vec<usize> {
        let mut hits: Vec<usize> = (0..f.len())
            .filter(|&i| f[i].chrom == chrom && f[i].start < end && f[i].end > start)
            .collect();
        hits.sort_by_key(|&i| (f[i].start, f[i].end, i));
        hits
    }

    #[test]
    fn test_overlap_matches_brute_force() {
        let f = features();
        let idx = GenomeIntervals::new(&f).unwrap();
        assert!(!idx.is_genomic_order());
        for (chrom, s, e) in [
            ("chr1", 0, 50),
            ("chr1", 120_000, 125_000),
            ("chr2", 999_000, 2_000_000),
            ("chrX", 500_000, 500_001),
            ("chr1", 899_999, 900_000),
            ("chr1", 900_000, 900_500),
            ("chr7", 0, 1_000),
        ] {
            assert_eq!(
                idx.overlapping(chrom, s, e),
                brute(&f, chrom, s, e),
                "{chrom}:{s}-{e}"
            );
        }
        assert_eq!(
            idx.window("chr2", 10_000, 2_000),
            brute(&f, "chr2", 8_000, 12_000)
        );
        assert!(idx.overlapping("chr1", 10, 10).is_empty());
        assert_eq!(idx.overlapping_range("chr1", 0, 10), None);
    }

    /// Tiles in genomic order: identity permutation, contiguous ranges.
    #[test]
    fn test_genomic_order_gives_contiguous_ranges() {
        let tiles: Vec<FeatureInterval> = ["chr1", "chr2"]
            .iter()
            .flat_map(|c| {
                (0..100u32).map(move |t| FeatureInterval {
                    chrom: c.to_string(),
                    start: t * 500,
                    end: (t + 1) * 500,
                    strand: 0,
                    arm: Arm::Unknown,
                })
            })
            .collect();
        let idx = GenomeIntervals::new(&tiles).unwrap();
        assert!(idx.is_genomic_order());
        assert_eq!(idx.overlapping_range("chr2", 1_200, 2_600), Some(102..106));
        assert_eq!(idx.overlapping_range("chr3", 0, 10), Some(0..0));
    }

    #[test]
    fn test_encode_decode_round_trip() {
        for f in [features(), vec![]] {
            let idx = GenomeIntervals::new(&f).unwrap();
            assert_eq!(GenomeIntervals::decode(&idx.encode()).unwrap(), idx);
        }
        let idx = GenomeIntervals::new(&features()).unwrap();
        let bytes = idx.encode();
        assert!(GenomeIntervals::decode(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn test_empty_interval_is_refused() {
        let mut f = features();
        f[3].end = f[3].start;
        assert!(GenomeIntervals::new(&f).is_err());
    }

    /// Attaching appends a section to a finished store: the index reads back
    /// and the chunks still read as before.
    #[test]
    fn test_attach_to_a_store_and_read_back() {
        use crate::store::{
            chunk::CsrCellChunk, reader::SingleCellReading, writer::CellGeneSparseWriter,
        };

        let f = features();
        let path = std::env::temp_dir().join(format!("bx_intervals_{}.bin", std::process::id()));
        let p = path.to_str().unwrap();
        let mut w = CellGeneSparseWriter::new(p, true, 3, f.len(), 1e4).unwrap();
        for c in 0..3 {
            let idx = [c as u32, 100 + c as u32];
            w.write_cell_chunk(CsrCellChunk::from_data(&[2u32, 5], &idx, c, 1e4, true))
                .unwrap();
        }
        w.finalise().unwrap();
        let before = ParallelSparseReader::new(p)
            .unwrap()
            .read_cells_parallel(&[0, 1, 2])
            .unwrap();

        let idx = GenomeIntervals::new(&f).unwrap();
        attach_interval_index(&[p], &idx).unwrap();
        let reader = ParallelSparseReader::new(p).unwrap();
        assert_eq!(reader.genome_intervals().unwrap(), Some(idx.clone()));
        let after = reader.read_cells_parallel(&[0, 1, 2]).unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.indices, b.indices);
            assert_eq!(a.data_norm, b.data_norm);
        }

        // replacing keeps one section
        attach_interval_index(&[p], &idx).unwrap();
        assert_eq!(
            ParallelSparseReader::new(p)
                .unwrap()
                .genome_intervals()
                .unwrap(),
            Some(idx)
        );

        let short = GenomeIntervals::new(&f[..10]).unwrap();
        assert!(attach_interval_index(&[p], &short).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
