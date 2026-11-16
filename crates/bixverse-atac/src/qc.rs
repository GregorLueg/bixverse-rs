//! Per-cell QC from the fragment file, in one pass.
//!
//! Counts per cell: fragments, read pairs, nucleosome-free and
//! mononucleosomal fragments (Signac `CountFragments` bins), fragments on the
//! mitochondrial genome, and Tn5 cut sites around TSSs for the TSS
//! enrichment. The scores follow Signac:
//!
//! * nucleosome signal: `mononucleosomal / nucleosome_free`
//!   (`NucleosomeSignal`; Signac reads only the first `5000 * n_cells` lines,
//!   this reads all of them)
//! * TSS enrichment, Signac's fast mode (`TSSEnrichment(fast = TRUE)`): cut
//!   sites in the 1001 bp centred on each TSS, over the mean per base of the
//!   two 100 bp flanks at `+-(901..=1000)`, summed over all TSSs, divided by
//!   1001. Cells with no flank cut get the mean flank of all cells.
//!
//! Cut sites are the raw start and end of a fragment and count once for
//! every TSS window they fall in, with Signac's window arithmetic replicated
//! exactly (see [`SIGNAC_CUT_SHIFT_LO`]).

use std::path::Path;

use rustc_hash::FxHashMap;

use crate::fragments::{BarcodeIndex, Fragment, fold_fragments};
use crate::prelude::*;

/// Chromosome names treated as the mitochondrial genome by default.
pub const MITO_CHROMS: [&str; 4] = ["chrM", "MT", "chrMT", "M"];

///////////
// Types //
///////////

/// Parameters of [`fragment_qc`].
#[derive(Clone, Debug)]
pub struct FragmentQcParams {
    /// Half width of the TSS centre window; the window is `2 * c + 1` bp.
    pub tss_centre: u32,
    /// Inner edge of each TSS flank (distance from the TSS).
    pub tss_flank_inner: u32,
    /// Outer edge of each TSS flank (distance from the TSS).
    pub tss_flank_outer: u32,
    /// Fragments shorter than this are nucleosome free.
    pub nucleosome_free_below: u32,
    /// Fragments strictly between `nucleosome_free_below` and this are
    /// mononucleosomal.
    pub mononucleosome_below: u32,
    /// Chromosome names of the mitochondrial genome.
    pub mito_chroms: Vec<String>,
}

impl Default for FragmentQcParams {
    /// Signac's windows and bins.
    fn default() -> Self {
        Self {
            tss_centre: 500,
            tss_flank_inner: 901,
            tss_flank_outer: 1000,
            nucleosome_free_below: 147,
            mononucleosome_below: 294,
            mito_chroms: MITO_CHROMS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Counters of one cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CellCounts {
    /// Fragments.
    fragments: u64,
    /// Read pairs (column 5).
    reads: u64,
    /// Fragments shorter than the nucleosome-free bound.
    nucleosome_free: u64,
    /// Mononucleosomal fragments.
    mononucleosomal: u64,
    /// Fragments on the mitochondrial genome.
    mito: u64,
    /// Cut sites in the TSS centre windows.
    tss_centre: u64,
    /// Cut sites in the TSS flanks.
    tss_flank: u64,
}

impl CellCounts {
    /// Add another cell's counts.
    ///
    /// ### Params
    ///
    /// * `o` - The other counts
    fn add(&mut self, o: &CellCounts) {
        self.fragments += o.fragments;
        self.reads += o.reads;
        self.nucleosome_free += o.nucleosome_free;
        self.mononucleosomal += o.mononucleosomal;
        self.mito += o.mito;
        self.tss_centre += o.tss_centre;
        self.tss_flank += o.tss_flank;
    }
}

/// Per-cell QC counts, in the order of the requested barcodes.
#[derive(Clone, Debug, Default)]
pub struct FragmentQc {
    /// Fragments per cell.
    pub n_fragments: Vec<u64>,
    /// Read pairs per cell (sum of column 5).
    pub n_reads: Vec<u64>,
    /// Nucleosome-free fragments per cell.
    pub nucleosome_free: Vec<u64>,
    /// Mononucleosomal fragments per cell.
    pub mononucleosomal: Vec<u64>,
    /// Fragments on the mitochondrial genome per cell.
    pub n_mito: Vec<u64>,
    /// Cut sites in the TSS centre windows per cell.
    pub tss_centre: Vec<u64>,
    /// Cut sites in the TSS flanks per cell.
    pub tss_flank: Vec<u64>,
    /// Width of the centre window, for the score.
    centre_width: f64,
    /// Width of both flanks together, for the score.
    flank_width: f64,
}

impl FragmentQc {
    /// Nucleosome signal per cell, `mononucleosomal / nucleosome_free`.
    ///
    /// Infinite or `NaN` for a cell without nucleosome-free fragments, as in
    /// Signac.
    ///
    /// ### Returns
    ///
    /// One score per cell.
    pub fn nucleosome_signal(&self) -> Vec<f64> {
        self.mononucleosomal
            .iter()
            .zip(&self.nucleosome_free)
            .map(|(&m, &n)| m as f64 / n as f64)
            .collect()
    }

    /// TSS enrichment per cell, Signac's fast definition.
    ///
    /// ### Returns
    ///
    /// One score per cell; 0 where the score is undefined.
    pub fn tss_enrichment(&self) -> Vec<f64> {
        let flank_mean: Vec<f64> = self
            .tss_flank
            .iter()
            .map(|&f| f as f64 / self.flank_width)
            .collect();
        let fill = flank_mean.iter().sum::<f64>() / flank_mean.len().max(1) as f64;
        self.tss_centre
            .iter()
            .zip(&self.tss_flank)
            .zip(&flank_mean)
            .map(|((&c, &f), &fm)| {
                let fm = if f == 0 { fill } else { fm };
                let score = c as f64 / fm / self.centre_width;
                if score.is_nan() { 0.0 } else { score }
            })
            .collect()
    }

    /// Fraction of mitochondrial fragments per cell.
    ///
    /// ### Returns
    ///
    /// One fraction per cell, `NaN` for a cell without fragments.
    pub fn mito_fraction(&self) -> Vec<f64> {
        self.n_mito
            .iter()
            .zip(&self.n_fragments)
            .map(|(&m, &n)| m as f64 / n as f64)
            .collect()
    }
}

///////////////
// Functions //
///////////////

/// Sorted TSS positions per chromosome.
///
/// ### Params
///
/// * `tss` - `(chromosome, 0-based position)`
/// * `mito` - Chromosomes to leave out
///
/// ### Returns
///
/// Positions per chromosome, ascending.
fn index_tss(tss: &[(String, u32)], mito: &[String]) -> FxHashMap<String, Vec<u32>> {
    let mut by_chrom: FxHashMap<String, Vec<u32>> = FxHashMap::default();
    for (c, p) in tss {
        if !mito.iter().any(|m| m == c) {
            by_chrom.entry(c.clone()).or_default().push(*p);
        }
    }
    for v in by_chrom.values_mut() {
        v.sort_unstable();
    }
    by_chrom
}

/// Lower edge of Signac's counted range, relative to the 0-based window
/// start. Signac's cut matrix converts between coordinate systems such that a
/// cut site at raw position `c` lands in a window `[ws, we)` when
/// `ws + 3 <= c <= we + 2`; kept as is for parity with its TSS enrichment.
const SIGNAC_CUT_SHIFT_LO: i64 = 3;

/// Upper edge of Signac's counted range, relative to the 0-based window end
/// (inclusive). See [`SIGNAC_CUT_SHIFT_LO`].
const SIGNAC_CUT_SHIFT_HI: i64 = 2;

/// Cut sites of one fragment that Signac counts in one window.
///
/// Signac only sees fragments overlapping the window (a tabix query) and
/// counts their raw start and end inside the shifted range.
///
/// ### Params
///
/// * `s` - Fragment start
/// * `e` - Fragment end
/// * `ws` - Window start (0-based)
/// * `we` - Window end (exclusive)
///
/// ### Returns
///
/// 0, 1 or 2.
#[inline]
fn window_cuts(s: i64, e: i64, ws: i64, we: i64) -> u64 {
    if s >= we || e <= ws {
        return 0;
    }
    let inside = |c: i64| (ws + SIGNAC_CUT_SHIFT_LO..=we + SIGNAC_CUT_SHIFT_HI).contains(&c);
    inside(s) as u64 + inside(e) as u64
}

/// Add the TSS window hits of one fragment.
///
/// Windows around a TSS `t`: centre `[t - c, t + c + 1)`, flanks
/// `[t - outer, t - inner + 1)` and `[t + inner, t + outer + 1)`; the union
/// is symmetric, so the strand of the TSS does not matter.
///
/// ### Params
///
/// * `f` - The fragment
/// * `tss` - TSS positions on the chromosome, ascending
/// * `p` - Window parameters
/// * `c` - Counters of the cell
#[inline]
fn tss_hits(f: &Fragment, tss: &[u32], p: &FragmentQcParams, c: &mut CellCounts) {
    let (s, e) = (f.start as i64, f.end as i64);
    let reach = p.tss_flank_outer as i64 + SIGNAC_CUT_SHIFT_LO + 1;
    let from = tss.partition_point(|&t| (t as i64) < s - reach);
    let (centre, inner, outer) = (
        p.tss_centre as i64,
        p.tss_flank_inner as i64,
        p.tss_flank_outer as i64,
    );
    for &t in &tss[from..] {
        let t = t as i64;
        if t > e + reach {
            break;
        }
        c.tss_centre += window_cuts(s, e, t - centre, t + centre + 1);
        c.tss_flank += window_cuts(s, e, t - outer, t - inner + 1)
            + window_cuts(s, e, t + inner, t + outer + 1);
    }
}

/// Per-cell QC counts from a fragment file.
///
/// ### Params
///
/// * `path` - The fragment file
/// * `barcodes` - Cells, in output order
/// * `tss` - TSS positions `(chromosome, 0-based position)`; mitochondrial
///   ones are dropped as in Signac
/// * `params` - See [`FragmentQcParams`]
///
/// ### Returns
///
/// The [`FragmentQc`], or an I/O or parse error.
pub fn fragment_qc(
    path: &Path,
    barcodes: &[String],
    tss: &[(String, u32)],
    params: &FragmentQcParams,
) -> Result<FragmentQc, AtacErrors> {
    let index = BarcodeIndex::new(barcodes)?;
    let tss = index_tss(tss, &params.mito_chroms);
    let n = barcodes.len();
    let empty: Vec<u32> = Vec::new();

    let counts = fold_fragments(
        path,
        &index,
        || vec![CellCounts::default(); n],
        |acc: &mut Vec<CellCounts>, chrom: &str, frags: &[Fragment]| {
            let is_mito = params.mito_chroms.iter().any(|m| m == chrom);
            let tss_chrom = tss.get(chrom).unwrap_or(&empty);
            for f in frags {
                let c = &mut acc[f.cell as usize];
                c.fragments += 1;
                c.reads += f.count as u64;
                let len = f.len();
                if len < params.nucleosome_free_below {
                    c.nucleosome_free += 1;
                } else if len > params.nucleosome_free_below && len < params.mononucleosome_below {
                    c.mononucleosomal += 1;
                }
                if is_mito {
                    c.mito += 1;
                }
                if !tss_chrom.is_empty() {
                    tss_hits(f, tss_chrom, params, c);
                }
            }
        },
        |mut a, b| {
            for (x, y) in a.iter_mut().zip(&b) {
                x.add(y);
            }
            a
        },
    )?;

    let pick = |g: fn(&CellCounts) -> u64| counts.iter().map(g).collect::<Vec<u64>>();
    Ok(FragmentQc {
        n_fragments: pick(|c| c.fragments),
        n_reads: pick(|c| c.reads),
        nucleosome_free: pick(|c| c.nucleosome_free),
        mononucleosomal: pick(|c| c.mononucleosomal),
        n_mito: pick(|c| c.mito),
        tss_centre: pick(|c| c.tss_centre),
        tss_flank: pick(|c| c.tss_flank),
        centre_width: (2 * params.tss_centre + 1) as f64,
        flank_width: (2 * (params.tss_flank_outer - params.tss_flank_inner + 1)) as f64,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_cuts_follow_signac_shift() {
        // window [100, 200): counted range 103..=202, fragment must overlap
        assert_eq!(window_cuts(103, 150, 100, 200), 2);
        assert_eq!(window_cuts(102, 150, 100, 200), 1);
        assert_eq!(window_cuts(150, 202, 100, 200), 2);
        assert_eq!(window_cuts(150, 203, 100, 200), 1);
        // start at 200 does not overlap [100, 200) although 200 <= 202
        assert_eq!(window_cuts(200, 260, 100, 200), 0);
        assert_eq!(window_cuts(40, 100, 100, 200), 0);
    }

    #[test]
    fn test_tss_hits_sum_over_tss() {
        let p = FragmentQcParams::default();
        let f = Fragment {
            start: 10_200,
            end: 10_400,
            cell: 0,
            count: 1,
        };
        let mut c = CellCounts::default();
        // both cuts in the centre of both TSSs
        tss_hits(&f, &[10_000, 10_300], &p, &mut c);
        assert_eq!((c.tss_centre, c.tss_flank), (4, 0));
        // TSS at 9_300: downstream flank [10_201, 10_301) counts
        // 10_204..=10_302, which neither 10_200 nor 10_400 is in
        let mut c = CellCounts::default();
        tss_hits(&f, &[9_300], &p, &mut c);
        assert_eq!((c.tss_centre, c.tss_flank), (0, 0));
        let mut c = CellCounts::default();
        tss_hits(&f, &[9_250], &p, &mut c);
        // flank [10_151, 10_251) counts 10_154..=10_252: the start
        assert_eq!((c.tss_centre, c.tss_flank), (0, 1));
    }

    #[test]
    fn test_tss_enrichment_fills_empty_flanks_with_the_mean() {
        let qc = FragmentQc {
            tss_centre: vec![1001, 2002, 0],
            tss_flank: vec![200, 0, 0],
            centre_width: 1001.0,
            flank_width: 200.0,
            ..Default::default()
        };
        // flank means 1, 0, 0 -> fill 1/3 for the empty ones
        let s = qc.tss_enrichment();
        assert_eq!(s[0], 1.0);
        assert!((s[1] - 2002.0 / (1.0 / 3.0) / 1001.0).abs() < 1e-12);
        assert_eq!(s[2], 0.0);
    }
}
