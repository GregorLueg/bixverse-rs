//! Peak to gene links: correlation of a peak's accessibility with a gene's
//! expression over groups of cells, for peaks near the gene's TSS (ArchR
//! `addPeak2GeneLinks`).
//!
//! Both modalities are summed per group (metacells, or fine clusters) by the
//! caller, then log-normalised per group, `log2(x / total * scale + 1)`. A
//! link is a peak whose span reaches within `max_distance` of the TSS; its
//! score is the Pearson correlation over the groups, its p-value the
//! two-sided t-test of that correlation with `n - 2` degrees of freedom, and
//! the FDR Benjamini-Hochberg over every tested pair. Unlike Signac's
//! `LinkPeaks` there is no GC-matched background, which needs the genome
//! sequence.

use faer::{Mat, MatRef};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};

use bixverse_core::core::math::stats::p_adjust_fdr;
use bixverse_sc_data::store::interval_index::FeatureInterval;

use crate::peaks::regions::RegionIndex;
use crate::prelude::*;

/// Parameters of [`peak_to_gene_links`].
#[derive(Clone, Debug)]
pub struct LinkParams {
    /// Largest distance between a peak and a TSS (ArchR: 250 kb).
    pub max_distance: u32,
    /// Normalisation scale (ArchR: 1e4).
    pub scale: f64,
}

impl Default for LinkParams {
    /// ArchR's defaults.
    ///
    /// ### Returns
    ///
    /// The parameters.
    fn default() -> Self {
        Self {
            max_distance: 250_000,
            scale: 1e4,
        }
    }
}

/// One tested peak and gene pair.
#[derive(Clone, Debug, PartialEq)]
pub struct PeakGeneLink {
    /// Peak index.
    pub peak: u32,
    /// Gene index.
    pub gene: u32,
    /// Peak centre minus TSS, in bases.
    pub distance: i64,
    /// Pearson correlation over the groups.
    pub r: f64,
    /// Two-sided p-value of the correlation.
    pub p: f64,
    /// Benjamini-Hochberg FDR over all tested pairs.
    pub fdr: f64,
}

/// Log-normalise each group (row) and standardise each feature (column).
///
/// ### Params
///
/// * `x` - Groups x features, summed counts
/// * `scale` - Normalisation scale
///
/// ### Returns
///
/// The standardised matrix (column sd over `n - 1`), and per column whether
/// it varies; a constant column is all zero.
fn standardise(x: MatRef<f64>, scale: f64) -> (Mat<f64>, Vec<bool>) {
    let (n, m) = (x.nrows(), x.ncols());
    let totals: Vec<f64> = (0..n).map(|i| (0..m).map(|j| x[(i, j)]).sum()).collect();
    let mut out = Mat::<f64>::zeros(n, m);
    let varies: Vec<bool> = out
        .par_col_iter_mut()
        .enumerate()
        .map(|(j, mut col)| {
            for i in 0..n {
                let t = totals[i];
                col[i] = if t > 0.0 {
                    (x[(i, j)] / t * scale + 1.0).log2()
                } else {
                    0.0
                };
            }
            let mean = (0..n).map(|i| col[i]).sum::<f64>() / n as f64;
            let ss: f64 = (0..n).map(|i| (col[i] - mean).powi(2)).sum();
            let sd = (ss / (n as f64 - 1.0)).sqrt();
            let varies = sd > 0.0 && sd.is_finite();
            for i in 0..n {
                col[i] = if varies { (col[i] - mean) / sd } else { 0.0 };
            }
            varies
        })
        .collect();
    (out, varies)
}

/// Correlate peaks with nearby genes over groups of cells.
///
/// ### Params
///
/// * `peak_counts` - Groups x peaks, summed counts
/// * `gene_counts` - Groups x genes, summed counts, same groups
/// * `peaks` - Peak coordinates, one per column of `peak_counts`
/// * `tss` - TSS `(chromosome, 0-based position)` per gene column
/// * `params` - See [`LinkParams`]
///
/// ### Returns
///
/// The tested pairs, by gene then peak position. Pairs with a constant peak
/// or gene are not tested. An error for mismatched shapes or fewer than three
/// groups.
pub fn peak_to_gene_links(
    peak_counts: MatRef<f64>,
    gene_counts: MatRef<f64>,
    peaks: &[FeatureInterval],
    tss: &[(String, u32)],
    params: &LinkParams,
) -> Result<Vec<PeakGeneLink>, AtacErrors> {
    let n = peak_counts.nrows();
    if gene_counts.nrows() != n
        || peak_counts.ncols() != peaks.len()
        || gene_counts.ncols() != tss.len()
    {
        return Err(AtacErrors::InvalidArgument(format!(
            "peak counts {} x {} and gene counts {} x {} do not fit {} peaks and {} genes",
            n,
            peak_counts.ncols(),
            gene_counts.nrows(),
            gene_counts.ncols(),
            peaks.len(),
            tss.len()
        )));
    }
    if n < 3 {
        return Err(AtacErrors::InvalidArgument(format!(
            "{n} groups; a correlation test needs at least three"
        )));
    }
    let (zp, peak_varies) = standardise(peak_counts, params.scale);
    let (zg, gene_varies) = standardise(gene_counts, params.scale);
    let index = RegionIndex::new(peaks);
    let t_dist = StudentsT::new(0.0, 1.0, (n - 2) as f64)
        .map_err(|e| AtacErrors::InvalidArgument(format!("t distribution: {e}")))?;
    let df = (n - 2) as f64;

    let mut links: Vec<PeakGeneLink> = tss
        .par_iter()
        .enumerate()
        .filter(|&(g, _)| gene_varies[g])
        .flat_map_iter(|(g, (chrom, pos))| {
            let mut hits: Vec<u32> = Vec::new();
            if let Some(c) = index.chrom(chrom) {
                let lo = pos.saturating_sub(params.max_distance);
                let hi = pos.saturating_add(params.max_distance).saturating_add(1);
                c.for_each_overlap(lo, hi, |p| hits.push(p));
            }
            hits.sort_unstable_by_key(|&p| (peaks[p as usize].start, p));
            let gcol = zg.col(g);
            let (zp, peak_varies) = (&zp, &peak_varies);
            hits.into_iter()
                .filter(move |&p| peak_varies[p as usize])
                .map(move |p| {
                    let pcol = zp.col(p as usize);
                    let dot: f64 = (0..n).map(|i| pcol[i] * gcol[i]).sum();
                    let r = (dot / (n as f64 - 1.0)).clamp(-1.0, 1.0);
                    let t = r * (df / (1.0 - r * r).max(f64::MIN_POSITIVE)).sqrt();
                    let p_val = (2.0 * t_dist.sf(t.abs())).min(1.0);
                    let peak = &peaks[p as usize];
                    let centre = (peak.start as i64 + peak.end as i64) / 2;
                    PeakGeneLink {
                        peak: p,
                        gene: g as u32,
                        distance: centre - *pos as i64,
                        r,
                        p: p_val,
                        fdr: f64::NAN,
                    }
                })
        })
        .collect();
    let p: Vec<f64> = links.iter().map(|l| l.p).collect();
    for (l, q) in links.iter_mut().zip(p_adjust_fdr(&p)) {
        l.fdr = q;
    }
    Ok(links)
}
