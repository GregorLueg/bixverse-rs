//! Multiplet detection from fragment overlaps, after AMULET (Thibodeau et
//! al., Genome Biology, 2021). Clean room from the paper; behaviour the paper
//! leaves open matches the reference implementation run as a black box.
//!
//! A diploid nucleus holds at most two copies of a locus, so it should never
//! yield more than two overlapping fragments at one position. Per cell, the
//! loci where more than `max_overlap` valid fragments overlap are collected
//! (autosomes only, restriction regions such as repeats and blacklists
//! dropped) and merged across cells. Sites shared by unexpectedly many cells
//! are taken for unannotated repeats and removed (Poisson, Benjamini-Hochberg).
//! Each cell's count of the remaining sites is tested against the mean count
//! (Poisson, Benjamini-Hochberg); cells above the cutoff are multiplets.
//!
//! The fragment file is read once, in chromosome batches (see
//! [`crate::io::fragments::plan_batches`]); only the sites and one count per
//! cell outlive a batch.

use std::path::Path;

use bixverse_core::core::math::stats::p_adjust_fdr;
use bixverse_sc_data::store::interval_index::FeatureInterval;
use rayon::prelude::*;
use statrs::distribution::{DiscreteCDF, Poisson};

use crate::io::fragments::{
    ChromAcc, extend_larger, fold_batch, last_used_chromosome, plan_batches,
};
use crate::peaks::regions::{ChromRegions, RegionIndex};
use crate::prelude::*;

/// Memory per compressed fragment file byte in flight: 12 bytes per fragment
/// kept and roughly 10 compressed bytes per fragment line, plus vector
/// growth and the per-worker buffers of the parallel read. Not tuned.
const MEM_PER_COMPRESSED_BYTE: f64 = 2.5;

///////////
// Types //
///////////

/// Parameters of [`amulet`]. The defaults are the paper's.
#[derive(Clone, Debug)]
pub struct AmuletParams {
    /// Overlapping fragments a diploid nucleus can explain; a locus counts
    /// when more than this many overlap. Raise it for polyploid cells.
    pub max_overlap: u32,
    /// Longest valid fragment.
    pub max_fragment_len: u32,
    /// q-value under which a site is a repeat and dropped.
    pub repeat_q: f64,
    /// q-value under which a cell is a multiplet.
    pub multiplet_q: f64,
    /// Chromosomes to use; `None` for the autosomes (names that are a number,
    /// with or without a `chr` prefix).
    pub chroms: Option<Vec<String>>,
    /// Memory budget of the fragments read at once, in GB; `None` reads one
    /// chromosome at a time. Without a tabix index the whole file is read at
    /// once.
    pub max_mem_gb: Option<f64>,
    /// Count a cell's trailing loci on the last chromosome used, those no
    /// later fragment of the cell starts beyond. The reference
    /// implementation loses them there (and only there: a probe with chr21,
    /// chr22, chrX and a contig keeps chr21's and drops chr22's), so `false`
    /// reproduces its counts. The order comes from the tabix index, or from
    /// reading the file backwards; a plain gzip file without an index has
    /// neither, and its trailing loci are counted.
    pub count_trailing_loci: bool,
}

impl Default for AmuletParams {
    /// The paper's defaults, trailing loci as the reference implementation.
    ///
    /// ### Returns
    ///
    /// The parameters.
    fn default() -> Self {
        Self {
            max_overlap: 2,
            max_fragment_len: 900,
            repeat_q: 0.01,
            multiplet_q: 0.01,
            chroms: None,
            max_mem_gb: None,
            count_trailing_loci: false,
        }
    }
}

impl AmuletParams {
    /// Check the parameters.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the first invalid parameter.
    fn validate(&self) -> Result<(), AtacErrors> {
        let bad = |msg: &str| Err(AtacErrors::InvalidArgument(msg.into()));
        if self.max_overlap == 0 {
            return bad("max_overlap must be positive");
        }
        for q in [self.repeat_q, self.multiplet_q] {
            if q.is_nan() || q <= 0.0 || q > 1.0 {
                return bad("q-value cutoffs must lie in (0, 1]");
            }
        }
        if self.max_mem_gb.is_some_and(|m| m.is_nan() || m <= 0.0) {
            return bad("max_mem_gb must be positive");
        }
        Ok(())
    }

    /// Whether a chromosome is used.
    ///
    /// ### Params
    ///
    /// * `chrom` - Chromosome name
    ///
    /// ### Returns
    ///
    /// `true` when it is listed, or an autosome without a list.
    fn uses(&self, chrom: &str) -> bool {
        match &self.chroms {
            Some(list) => list.iter().any(|c| c == chrom),
            None => {
                let n = chrom.strip_prefix("chr").unwrap_or(chrom);
                !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
            }
        }
    }
}

/// Per-cell results of [`amulet`], in barcode order, plus the site totals.
#[derive(Clone, Debug)]
pub struct AmuletResult {
    /// Valid fragments on the chromosomes used.
    pub n_fragments: Vec<u32>,
    /// Loci with more than `max_overlap` overlapping fragments, before any
    /// filter.
    pub n_overlaps: Vec<u32>,
    /// Merged sites the cell is part of after the restriction and repeat
    /// filters: the tested count.
    pub n_sites: Vec<u32>,
    /// Poisson upper tail `P(X > n_sites)` at the mean count.
    pub p_value: Vec<f64>,
    /// Benjamini-Hochberg of `p_value`.
    pub q_value: Vec<f64>,
    /// `q_value < multiplet_q`.
    pub multiplet: Vec<bool>,
    /// Merged sites after the restriction filter.
    pub n_merged_sites: usize,
    /// Of those, the sites dropped as repeats.
    pub n_repeat_sites: usize,
}

/// One locus of one cell: `(start, end, cell)`.
type CellSite = (u32, u32, u32);

/// The fragments of one read, per chromosome, and per-cell valid counts.
#[derive(Default)]
struct OverlapAcc {
    /// Valid fragments as `(cell, start, end)`, per chromosome.
    frags: ChromAcc<Vec<(u32, u32, u32)>>,
    /// Valid fragments per cell; empty until the first one.
    n_valid: Vec<u32>,
}

impl OverlapAcc {
    /// Add another worker's fragments.
    ///
    /// ### Params
    ///
    /// * `other` - The other accumulator
    ///
    /// ### Returns
    ///
    /// The union.
    fn merge(mut self, other: OverlapAcc) -> OverlapAcc {
        self.frags.merge(other.frags, extend_larger);
        if self.n_valid.len() < other.n_valid.len() {
            self.n_valid.resize(other.n_valid.len(), 0);
        }
        for (a, b) in self.n_valid.iter_mut().zip(other.n_valid) {
            *a += b;
        }
        self
    }
}

//////////////
// Overlaps //
//////////////

/// The loci of one cell where more than `max_overlap` fragments overlap.
///
/// Depth is per base over half-open fragments: a fragment `[s, e)` covers
/// the bases `s..e`, so fragments that only touch share no base. The sweep
/// applies all starts and ends at a position together, so a fragment ending
/// where another starts does not split a locus. Each locus is a maximal run
/// of bases with excess depth.
///
/// ### Params
///
/// * `frags` - The cell's fragments as `(start, end)`, sorted by start
/// * `max_overlap` - Depth a diploid cell can explain
/// * `ends` - Scratch for the sorted ends
///
/// ### Returns
///
/// The loci as `(start, end)`, in order.
fn cell_overlaps(frags: &[(u32, u32)], max_overlap: u32, ends: &mut Vec<u32>) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    if frags.len() <= max_overlap as usize {
        return out;
    }
    ends.clear();
    ends.extend(frags.iter().map(|f| f.1));
    ends.sort_unstable();
    let (mut i_s, mut i_e) = (0, 0);
    let mut depth = 0u32;
    let mut open = 0u32;
    while i_e < ends.len() {
        let pos = match frags.get(i_s) {
            Some(f) => f.0.min(ends[i_e]),
            None => ends[i_e],
        };
        let before = depth;
        while i_s < frags.len() && frags[i_s].0 == pos {
            depth += 1;
            i_s += 1;
        }
        while i_e < ends.len() && ends[i_e] == pos {
            depth -= 1;
            i_e += 1;
        }
        if before <= max_overlap && depth > max_overlap {
            open = pos;
        } else if before > max_overlap && depth <= max_overlap {
            out.push((open, pos));
        }
    }
    out
}

/// The overlap loci of every cell on one chromosome.
///
/// Each locus is then checked against the restriction regions: the cell's
/// fragments overlapping the locus that touch a region (the region's end
/// counts as inside) are set aside, and the locus survives when the rest
/// still overlap more than `max_overlap` deep. A region therefore removes a
/// locus through its reads, not its coordinates; this is the reference
/// implementation's behaviour, the paper only says sites were filtered.
///
/// ### Params
///
/// * `frags` - The chromosome's valid fragments as `(cell, start, end)`
/// * `max_overlap` - Depth a diploid cell can explain
/// * `drop_trailing` - Drop loci that no later fragment of the cell starts
///   at or beyond
/// * `restrict` - The chromosome's restriction regions
///
/// ### Returns
///
/// The loci as `(start, end, cell, survives the restriction)`, any order.
fn chromosome_overlaps(
    mut frags: Vec<(u32, u32, u32)>,
    max_overlap: u32,
    drop_trailing: bool,
    restrict: Option<&ChromRegions>,
) -> Vec<(u32, u32, u32, bool)> {
    frags.par_sort_unstable();
    let mut bounds = vec![0];
    for i in 1..frags.len() {
        if frags[i].0 != frags[i - 1].0 {
            bounds.push(i);
        }
    }
    bounds.push(frags.len());
    let restricted = |s: u32, e: u32| {
        let mut hit = false;
        if let Some(r) = restrict {
            r.for_each_overlap(s.saturating_sub(1), e, |_| hit = true);
        }
        hit
    };
    bounds
        .par_windows(2)
        .flat_map_iter(|w| {
            let cell = &frags[w[0]..w[1]];
            let id = cell.first().map_or(0, |f| f.0);
            let se: Vec<(u32, u32)> = cell.iter().map(|f| (f.1, f.2)).collect();
            let last_start = se.last().map_or(0, |f| f.0);
            let loci = cell_overlaps(&se, max_overlap, &mut Vec::new());
            loci.into_iter()
                .filter(move |&(_, e)| !drop_trailing || last_start >= e)
                .map(|(s, e)| {
                    // fragments starting before the locus end, ending after its start
                    let hi = se.partition_point(|f| f.0 < e);
                    let rest: Vec<(u32, u32)> = se[..hi]
                        .iter()
                        .filter(|f| f.1 > s && !restricted(f.0, f.1))
                        .copied()
                        .collect();
                    let keep = rest.len() > max_overlap as usize
                        && !cell_overlaps(&rest, max_overlap, &mut Vec::new()).is_empty();
                    (s, e, id, keep)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Merge the loci of all cells on one chromosome into sites.
///
/// Loci that overlap or touch join one site, as the reference implementation
/// merges them (the paper says "at least one base pair").
///
/// ### Params
///
/// * `loci` - `(start, end, cell)` of every cell
///
/// ### Returns
///
/// The cells of each site, sorted and distinct.
fn merge_sites(mut loci: Vec<CellSite>) -> Vec<Vec<u32>> {
    loci.sort_unstable();
    let mut sites: Vec<Vec<u32>> = Vec::new();
    let mut end = 0u32;
    for (s, e, cell) in loci {
        match sites.last_mut() {
            Some(cells) if s <= end => {
                cells.push(cell);
                end = end.max(e);
            }
            _ => {
                sites.push(vec![cell]);
                end = e;
            }
        }
    }
    for cells in &mut sites {
        cells.sort_unstable();
        cells.dedup();
    }
    sites
}

////////////////
// Statistics //
////////////////

/// Poisson upper tails `P(X > x)` at the mean of `counts`, and their
/// Benjamini-Hochberg q-values.
///
/// ### Params
///
/// * `counts` - The observations
///
/// ### Returns
///
/// `(p-values, q-values)`; all 1 when every count is 0.
fn poisson_upper(counts: &[u32]) -> (Vec<f64>, Vec<f64>) {
    let n = counts.len().max(1) as f64;
    let lambda = counts.iter().map(|&c| c as f64).sum::<f64>() / n;
    let p: Vec<f64> = match Poisson::new(lambda) {
        Ok(dist) => counts.iter().map(|&c| dist.sf(c as u64)).collect(),
        Err(_) => vec![1.0; counts.len()],
    };
    let q = p_adjust_fdr(&p);
    (p, q)
}

/// Detect multiplets from fragment overlaps.
///
/// ### Params
///
/// * `path` - Fragment file
/// * `barcodes` - Cells to test, the result's order
/// * `restrict` - Regions whose overlap loci are dropped: repeats,
///   segmental duplications, blacklists
/// * `params` - See [`AmuletParams`]; `None` for the defaults
///
/// ### Returns
///
/// The per-cell results, or an I/O, parse or argument error.
pub fn amulet(
    path: &Path,
    barcodes: &[String],
    restrict: &[FeatureInterval],
    params: Option<AmuletParams>,
) -> Result<AmuletResult, AtacErrors> {
    let params = params.unwrap_or_default();
    params.validate()?;
    let n_cells = barcodes.len();
    let index = BarcodeIndex::new(barcodes)?;
    let mut restrict_chroms: Vec<String> = restrict.iter().map(|r| r.chrom.clone()).collect();
    restrict_chroms.sort_unstable();
    restrict_chroms.dedup();
    let restrict = RegionIndex::new(restrict);
    let (batches, order) = plan_batches(path, params.max_mem_gb, MEM_PER_COMPRESSED_BYTE, |c| {
        params.uses(c)
    })?;
    // the reference flushes a cell's pending loci when the chromosome
    // changes but not at the end of the file, so trailing loci are lost on
    // the last chromosome used
    let last_used = match batches.first() {
        Some(Some(_)) => order.iter().rev().find(|c| params.uses(c)).cloned(),
        _ if params.count_trailing_loci => None,
        _ => last_used_chromosome(path, |c| params.uses(c))?,
    };

    let mut n_fragments = vec![0u32; n_cells];
    let mut n_overlaps = vec![0u32; n_cells];
    let mut sites: Vec<Vec<u32>> = Vec::new();
    let mut used_seen: Vec<String> = Vec::new();
    for batch in &batches {
        let acc = fold_batch(
            path,
            batch,
            &index,
            OverlapAcc::default,
            |acc: &mut OverlapAcc, chrom: &str, frags: &[Fragment]| {
                if !params.uses(chrom) {
                    return;
                }
                if acc.n_valid.is_empty() {
                    acc.n_valid = vec![0; n_cells];
                }
                let chrom_frags = acc.frags.entry(chrom, Vec::new);
                for f in frags.iter().filter(|f| f.len() <= params.max_fragment_len) {
                    chrom_frags.push((f.cell, f.start, f.end));
                    acc.n_valid[f.cell as usize] += 1;
                }
            },
            OverlapAcc::merge,
        )?;
        for (a, b) in n_fragments.iter_mut().zip(&acc.n_valid) {
            *a += b;
        }
        used_seen.extend(acc.frags.chroms.iter().cloned());
        for (chrom, frags) in acc.frags.chroms.into_iter().zip(acc.frags.values) {
            let drop_trailing =
                !params.count_trailing_loci && last_used.as_deref() == Some(chrom.as_str());
            let loci = chromosome_overlaps(
                frags,
                params.max_overlap,
                drop_trailing,
                restrict.chrom(&chrom),
            );
            for &(_, _, cell, _) in &loci {
                n_overlaps[cell as usize] += 1;
            }
            let kept: Vec<CellSite> = loci
                .into_iter()
                .filter(|l| l.3)
                .map(|(s, e, c, _)| (s, e, c))
                .collect();
            sites.extend(merge_sites(kept));
        }
    }

    // an empty result would call every cell a singlet without a word
    if n_cells > 0 && n_fragments.iter().all(|&n| n == 0) {
        return Err(AtacErrors::InvalidArgument(
            "no fragment of the requested cells lies on a chromosome AMULET uses (autosomes \
             named 1 to 22, with or without 'chr', or the `chroms` list); check the barcodes \
             and the chromosome naming"
                .into(),
        ));
    }
    if !restrict_chroms.is_empty() && !restrict_chroms.iter().any(|c| used_seen.contains(c)) {
        return Err(AtacErrors::InvalidArgument(format!(
            "no restriction region lies on a chromosome AMULET used (regions on {}, fragments on \
             {}); check the chromosome naming",
            restrict_chroms.iter().take(3).cloned().collect::<Vec<_>>().join(", "),
            used_seen.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
        )));
    }

    // sites shared by too many cells are repeats
    let row_sums: Vec<u32> = sites.iter().map(|c| c.len() as u32).collect();
    let (_, repeat_q) = poisson_upper(&row_sums);
    let mut n_sites = vec![0u32; n_cells];
    let mut n_repeat_sites = 0;
    for (cells, &q) in sites.iter().zip(&repeat_q) {
        if q < params.repeat_q {
            n_repeat_sites += 1;
            continue;
        }
        for &c in cells {
            n_sites[c as usize] += 1;
        }
    }

    let (p_value, q_value) = poisson_upper(&n_sites);
    let multiplet = q_value.iter().map(|&q| q < params.multiplet_q).collect();
    Ok(AmuletResult {
        n_fragments,
        n_overlaps,
        n_sites,
        p_value,
        q_value,
        multiplet,
        n_merged_sites: sites.len(),
        n_repeat_sites,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_three_nested_fragments_make_one_locus() {
        let f = [(0, 100), (10, 50), (20, 30)];
        assert_eq!(cell_overlaps(&f, 2, &mut Vec::new()), vec![(20, 30)]);
    }

    #[test]
    fn test_touching_fragments_do_not_overlap() {
        let f = [(0, 10), (0, 10), (10, 20)];
        assert!(cell_overlaps(&f, 2, &mut Vec::new()).is_empty());
    }

    #[test]
    fn test_locus_spans_the_whole_excess() {
        // depth 3 on [20, 25), 4 on [25, 28), 3 on [28, 30): one locus
        let f = [(0, 30), (10, 40), (20, 50), (25, 28)];
        assert_eq!(cell_overlaps(&f, 2, &mut Vec::new()), vec![(20, 30)]);
    }

    #[test]
    fn test_threshold_follows_max_overlap() {
        let f = [(0, 10), (5, 15)];
        assert_eq!(cell_overlaps(&f, 1, &mut Vec::new()), vec![(5, 10)]);
        assert!(cell_overlaps(&f, 2, &mut Vec::new()).is_empty());
    }

    #[test]
    fn test_sites_merge_cells_that_overlap() {
        let sites = merge_sites(vec![
            (0, 10, 1),
            (5, 15, 0),
            (15, 20, 2),
            (5, 8, 1),
            (21, 30, 3),
        ]);
        assert_eq!(sites, vec![vec![0, 1, 2], vec![3]]);
    }

    #[test]
    fn test_poisson_upper_without_overlaps_is_one() {
        let (p, q) = poisson_upper(&[0, 0, 0]);
        assert_eq!(p, vec![1.0; 3]);
        assert_eq!(q, vec![1.0; 3]);
    }

    #[test]
    fn test_autosomes_by_name() {
        let p = AmuletParams::default();
        assert!(p.uses("chr1") && p.uses("22") && p.uses("chr19"));
        assert!(!p.uses("chrX") && !p.uses("chrM") && !p.uses("GL000194.1") && !p.uses("chr"));
    }
}
