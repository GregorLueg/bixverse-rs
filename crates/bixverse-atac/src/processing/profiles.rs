//! Per-group fragment distributions for plotting: the Tn5 insertion profile
//! around TSSs and the fragment length histogram, in one pass.
//!
//! Both follow SnapATAC2 (`metrics.tsse`, `metrics.frag_size_distr`): the
//! insertions of a fragment are its start and `end - 1`, an insertion's
//! offset is its distance to each TSS within `half_width`, downstream
//! positive on either strand, and a fragment longer than the histogram goes
//! to bin 0.

use std::path::Path;

use rustc_hash::FxHashMap;

use crate::io::fragments::{BarcodeIndex, Coverage, Fragment, fold_fragments_checked};
use crate::prelude::*;

/// Insertion profiles and length histograms per group of cells.
#[derive(Clone, Debug)]
pub struct FragmentProfiles {
    /// Per group, insertions at offsets `-half_width..=half_width` from the
    /// TSSs, summed over TSSs.
    pub tss: Vec<Vec<u64>>,
    /// Per group, fragments of length `l` in bin `l` for `1..=max_len`, and
    /// longer fragments in bin 0.
    pub lengths: Vec<Vec<u64>>,
    /// What the pass saw; off target means on a chromosome without a TSS.
    pub coverage: Coverage,
}

/// TSS profiles and length histograms per group of cells.
///
/// ### Params
///
/// * `path` - The fragment file
/// * `barcodes` - Cells
/// * `groups` - Group of each cell, `0..n_groups`
/// * `tss` - TSSs as `(chromosome, 0-based position, strand)`; strand `-1`
///   is minus, anything else plus
/// * `half_width` - Profile half width (SnapATAC2: 2000)
/// * `max_len` - Longest fragment with its own histogram bin (SnapATAC2:
///   1000)
///
/// ### Returns
///
/// The [`FragmentProfiles`], or an I/O, parse, argument or coverage error.
pub fn fragment_profiles(
    path: &Path,
    barcodes: &[String],
    groups: &[u32],
    tss: &[(String, u32, i8)],
    half_width: u32,
    max_len: u32,
) -> Result<FragmentProfiles, AtacErrors> {
    if barcodes.len() != groups.len() {
        return Err(AtacErrors::InvalidArgument(format!(
            "{} barcodes but {} group labels",
            barcodes.len(),
            groups.len()
        )));
    }
    let n_groups = groups.iter().max().map_or(0, |&g| g as usize + 1);
    let width = 2 * half_width as usize + 1;
    let bins = max_len as usize + 1;

    let mut by_chrom: FxHashMap<&str, Vec<(u32, bool)>> = FxHashMap::default();
    for (c, p, s) in tss {
        by_chrom.entry(c.as_str()).or_default().push((*p, *s < 0));
    }
    for v in by_chrom.values_mut() {
        v.sort_unstable();
    }
    let tss_chroms: Vec<&str> = by_chrom.keys().copied().collect();
    let targets = (!tss_chroms.is_empty()).then_some(&tss_chroms[..]);
    let index = BarcodeIndex::new(barcodes)?;
    let half = half_width as i64;
    let empty = Vec::new();

    let ((tss_acc, len_acc), coverage) = fold_fragments_checked(
        path,
        barcodes,
        &index,
        targets,
        "TSSs",
        || (vec![0u64; n_groups * width], vec![0u64; n_groups * bins]),
        |(t_acc, l_acc): &mut (Vec<u64>, Vec<u64>), chrom: &str, frags: &[Fragment]| {
            let sites = by_chrom.get(chrom).unwrap_or(&empty);
            for f in frags {
                let g = groups[f.cell as usize] as usize;
                let len = f.len() as usize;
                l_acc[g * bins + if len < bins { len } else { 0 }] += 1;
                if sites.is_empty() {
                    continue;
                }
                for x in [f.start, f.end - 1] {
                    let lo = sites.partition_point(|s| (s.0 as i64) < x as i64 - half);
                    for &(t, minus) in sites[lo..]
                        .iter()
                        .take_while(|s| s.0 as i64 <= x as i64 + half)
                    {
                        let d = if minus {
                            t as i64 - x as i64
                        } else {
                            x as i64 - t as i64
                        };
                        t_acc[g * width + (d + half) as usize] += 1;
                    }
                }
            }
        },
        |(mut ta, mut la), (tb, lb)| {
            ta.iter_mut().zip(tb).for_each(|(a, b)| *a += b);
            la.iter_mut().zip(lb).for_each(|(a, b)| *a += b);
            (ta, la)
        },
    )?;

    Ok(FragmentProfiles {
        tss: tss_acc.chunks(width).map(<[u64]>::to_vec).collect(),
        lengths: len_acc.chunks(bins).map(<[u64]>::to_vec).collect(),
        coverage,
    })
}
