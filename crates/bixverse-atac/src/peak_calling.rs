//! Peak calling on pseudobulk Tn5 tags, a port of MACS3 `callpeak` without a
//! control track (Zhang et al., Genome Biology, 2008).
//!
//! Cells are pooled per group (a cluster, a sample) and every group is called
//! on its own, as Signac's `CallPeaks(group.by = ...)` does with one MACS run
//! per group. The steps are those of `callpeak --nomodel` on single end tags:
//!
//! 1. Tags: both Tn5 insertions of every fragment ([`TagSource::Insertion`])
//!    or the fragment start only ([`TagSource::FragmentStart`], what MACS sees
//!    when Signac hands it the fragment file), all on the plus strand,
//!    optionally deduplicated per position.
//! 2. Treatment pileup: each tag shifted by `shift` and extended to
//!    `extsize`.
//! 3. Local lambda from the tags themselves: the pileup over `llocal` centred
//!    windows, scaled by `extsize / llocal`, floored at the genome background
//!    `extsize * n_tags / genome_size`.
//! 4. Per segment, the Poisson upper tail `-log10 P(X > pileup)`; q-scores by
//!    Benjamini-Hochberg over all segments of the group, weighted by length.
//! 5. Segments above the q cutoff, merged across gaps up to `max_gap`, kept
//!    from `min_length`; the summit is the middle of the highest segments.
//!
//! The arithmetic follows MACS3 3.1.0 to the bit where it matters for the
//! peak set: pileups and lambdas in f32, the tail in f64 and rounded to five
//! decimals, p-scores and q-scores as f32. That is why this module fixes its
//! float types rather than being generic.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use flate2::read::MultiGzDecoder;

use bixverse_sc_data::interval_index::FeatureInterval;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::prelude::*;

/// Fragments per group whose mean length MACS takes as the tag size, which
/// becomes the default `max_gap` for [`TagSource::FragmentStart`].
const TAG_SIZE_SAMPLE: usize = 10;

/// MACS stops summing the Poisson tail once a term moves the natural log of
/// the sum by less than this.
const POISSON_TAIL_TOLERANCE: f64 = 1e-5;

/// Starting value of the running q-score minimum, MACS's `INT_MAX` as f32.
const Q_START: f32 = 2_147_483_647.0;

/// Pseudocount of the fold enrichment at the summit.
const FOLD_PSEUDOCOUNT: f64 = 1.0;

///////////
// Types //
///////////

/// What becomes a tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagSource {
    /// Both insertions, raw start and raw end (the Cell Ranger convention).
    Insertion,
    /// The fragment start only. Matches Signac's `CallPeaks`, whose fragment
    /// file reaches MACS without a strand column, so every line is a plus
    /// strand read at its start and the end is ignored.
    FragmentStart,
}

/// Duplicate tags at one position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeepDup {
    /// Keep every tag. Tags of different cells at one base are signal in a
    /// pseudobulk.
    All,
    /// Keep at most this many tags per position (MACS `--keep-dup n`; its
    /// default, and so Signac's, is 1).
    AtMost(u32),
}

/// Parse a tag source name.
///
/// ### Params
///
/// * `s` - `"insertion"` or `"fragment_start"`
///
/// ### Returns
///
/// The source, or `None` for an unknown name.
pub fn parse_tag_source(s: &str) -> Option<TagSource> {
    match s {
        "insertion" => Some(TagSource::Insertion),
        "fragment_start" => Some(TagSource::FragmentStart),
        _ => None,
    }
}

/// Parameters of [`call_peaks`]. The defaults are MACS3 `callpeak` with
/// Signac's `--nomodel --extsize 200 --shift -100`, both insertions as tags
/// and every duplicate kept.
#[derive(Clone, Copy, Debug)]
pub struct PeakCallParams {
    /// What becomes a tag.
    pub tag_source: TagSource,
    /// Duplicate handling.
    pub keep_dup: KeepDup,
    /// Shift of each tag before extension (MACS `--shift`).
    pub shift: i32,
    /// Extension of each tag, also the fragment size `d` (`--extsize`).
    pub extsize: u32,
    /// Effective genome size (`-g`).
    pub genome_size: f64,
    /// Window of the local lambda (`--llocal`).
    pub llocal: u32,
    /// q-value cutoff (`-q`).
    pub q_value: f64,
    /// Minimum peak length; `None` takes `extsize`, as MACS.
    pub min_length: Option<u32>,
    /// Largest gap merged into one peak; `None` takes MACS's tag size: 1 for
    /// insertions, the truncated mean length of the group's first
    /// [`TAG_SIZE_SAMPLE`] fragments on the file's first chromosome for
    /// fragment starts.
    pub max_gap: Option<u32>,
}

impl Default for PeakCallParams {
    /// MACS3 defaults with Signac's shift and extension.
    ///
    /// ### Returns
    ///
    /// The parameters.
    fn default() -> Self {
        Self {
            tag_source: TagSource::Insertion,
            keep_dup: KeepDup::All,
            shift: -100,
            extsize: 200,
            genome_size: 2.7e9,
            llocal: 10_000,
            q_value: 0.05,
            min_length: None,
            max_gap: None,
        }
    }
}

impl PeakCallParams {
    /// Construct the parameters.
    ///
    /// ### Params
    ///
    /// * `tag_source` - What becomes a tag
    /// * `keep_dup` - Duplicate handling
    /// * `shift` - Tag shift before extension
    /// * `extsize` - Tag extension, the fragment size
    /// * `genome_size` - Effective genome size
    /// * `llocal` - Local lambda window
    /// * `q_value` - q-value cutoff
    /// * `min_length` - Minimum peak length, `None` for `extsize`
    /// * `max_gap` - Largest merged gap, `None` for MACS's tag size
    ///
    /// ### Returns
    ///
    /// The parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tag_source: TagSource,
        keep_dup: KeepDup,
        shift: i32,
        extsize: u32,
        genome_size: f64,
        llocal: u32,
        q_value: f64,
        min_length: Option<u32>,
        max_gap: Option<u32>,
    ) -> Self {
        Self {
            tag_source,
            keep_dup,
            shift,
            extsize,
            genome_size,
            llocal,
            q_value,
            min_length,
            max_gap,
        }
    }

    /// Check the parameters.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the first invalid parameter.
    fn validate(&self) -> Result<(), AtacErrors> {
        let bad = |msg: &str| Err(AtacErrors::InvalidArgument(msg.into()));
        if self.extsize == 0 {
            return bad("extsize must be positive");
        }
        if self.llocal == 0 {
            return bad("llocal must be positive");
        }
        if self.genome_size.is_nan() || self.genome_size <= 0.0 {
            return bad("genome_size must be positive");
        }
        if self.q_value.is_nan() || self.q_value <= 0.0 || self.q_value > 1.0 {
            return bad("q_value must lie in (0, 1]");
        }
        if self.keep_dup == KeepDup::AtMost(0) {
            return bad("keep_dup must keep at least one tag per position");
        }
        Ok(())
    }
}

/// One called peak, MACS's narrowPeak plus the summit pileup.
#[derive(Clone, Debug, PartialEq)]
pub struct NarrowPeak {
    /// Chromosome.
    pub chrom: String,
    /// 0-based start, inclusive.
    pub start: u32,
    /// 0-based end, exclusive.
    pub end: u32,
    /// Absolute summit position.
    pub summit: u32,
    /// Treatment pileup at the summit.
    pub pileup: f32,
    /// `-log10` p-value at the summit.
    pub pscore: f32,
    /// Fold enrichment over the local lambda at the summit.
    pub fold: f32,
    /// `-log10` q-value at the summit.
    pub qscore: f32,
}

impl NarrowPeak {
    /// The peak as a feature interval.
    ///
    /// ### Returns
    ///
    /// The unstranded interval.
    pub fn interval(&self) -> FeatureInterval {
        crate::regions::region(&self.chrom, self.start, self.end)
    }
}

/// The tags of one group.
struct GroupTags {
    /// Per chromosome, sorted by name: the tag positions (unsorted until
    /// [`call_group`]).
    chroms: Vec<(String, Vec<u32>)>,
    /// Per chromosome: the first fragments by start, as `(start, length)`,
    /// at most [`TAG_SIZE_SAMPLE`].
    first: FxHashMap<String, Vec<(u32, u32)>>,
}

/// A paired treatment and control pileup of one chromosome. Segment `i`
/// spans `[pos[i - 1], pos[i])`, segment 0 starts at 0.
struct PairedTrack {
    /// Segment ends.
    pos: Vec<i32>,
    /// Treatment pileup per segment.
    treat: Vec<f32>,
    /// Local lambda per segment.
    ctrl: Vec<f32>,
}

/// Group-wide values every chromosome of a call needs.
struct CallContext {
    /// Shift to the start of an extended tag (MACS `five_shift`).
    treat_five: i64,
    /// Shift to the end of an extended tag (MACS `three_shift`).
    treat_three: i64,
    /// Same for the local lambda window.
    ctrl_five: i64,
    /// Same for the local lambda window.
    ctrl_three: i64,
    /// `extsize / llocal`, f32 as in MACS.
    ctrl_scale: f32,
    /// Genome background, the floor of the local lambda.
    lambda_bg: f32,
}

/// Histogram key: observed pileup (truncated) and the bits of the lambda.
type PKey = (u32, u32);

/////////////
// Tagging //
/////////////

/// Pool the tags of each group in one pass over the fragment file.
///
/// ### Params
///
/// * `path` - Fragment file
/// * `barcodes` - Cells
/// * `groups` - Group of each cell
/// * `n_groups` - Number of groups
/// * `source` - What becomes a tag
///
/// ### Returns
///
/// The tags per group, or an I/O or parse error.
fn collect_tags(
    path: &Path,
    barcodes: &[String],
    groups: &[u32],
    n_groups: usize,
    source: TagSource,
) -> Result<Vec<GroupTags>, AtacErrors> {
    type Acc = Vec<(
        FxHashMap<String, Vec<u32>>,
        FxHashMap<String, Vec<(u32, u32)>>,
    )>;
    let index = BarcodeIndex::new(barcodes)?;
    let acc: Acc = fold_fragments(
        path,
        &index,
        || vec![Default::default(); n_groups],
        |acc: &mut Acc, chrom: &str, frags: &[Fragment]| {
            for f in frags {
                let (tags, first) = &mut acc[groups[f.cell as usize] as usize];
                let t = match tags.get_mut(chrom) {
                    Some(t) => t,
                    None => tags.entry(chrom.to_string()).or_default(),
                };
                t.push(f.start);
                if source == TagSource::Insertion {
                    t.push(f.end);
                }
                let fs = match first.get_mut(chrom) {
                    Some(fs) => fs,
                    None => first.entry(chrom.to_string()).or_default(),
                };
                // Runs reach a worker in any order, so keep the smallest
                // starts seen, not the first ones.
                fs.push((f.start, f.len()));
                if fs.len() > 2 * TAG_SIZE_SAMPLE {
                    fs.sort_unstable();
                    fs.truncate(TAG_SIZE_SAMPLE);
                }
            }
        },
        |mut a, b| {
            for ((ta, fa), (tb, fb)) in a.iter_mut().zip(b) {
                for (chrom, mut t) in tb {
                    let e = ta.entry(chrom).or_default();
                    if e.len() < t.len() {
                        std::mem::swap(e, &mut t);
                    }
                    e.extend_from_slice(&t);
                }
                for (chrom, f) in fb {
                    let e = fa.entry(chrom).or_default();
                    e.extend(f);
                    e.sort_unstable();
                    e.truncate(TAG_SIZE_SAMPLE);
                }
            }
            a
        },
    )?;

    Ok(acc
        .into_iter()
        .map(|(tags, first)| {
            let mut chroms: Vec<(String, Vec<u32>)> = tags.into_iter().collect();
            chroms.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            GroupTags { chroms, first }
        })
        .collect())
}

/// The chromosome of the first fragment line of a file.
///
/// ### Params
///
/// * `path` - Fragment file, gzip, BGZF or plain
///
/// ### Returns
///
/// The chromosome, `None` for a file without fragment lines, or an I/O
/// error.
fn first_chromosome(path: &Path) -> Result<Option<String>, AtacErrors> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 2];
    let gz = file.read(&mut magic)? == 2 && magic == [0x1f, 0x8b];
    file.seek(SeekFrom::Start(0))?;
    let reader: Box<dyn BufRead> = if gz {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        return Ok(line.split('\t').next().map(str::to_string));
    }
    Ok(None)
}

/// MACS's tag size of a fragment start group: the truncated mean length of
/// its first fragments.
///
/// MACS takes the first lines of the file it reads, which for a group is the
/// group's first fragments on the file's first chromosome. A group without
/// fragments there falls back to its chromosome first in name order.
///
/// ### Params
///
/// * `group` - The group's tags
/// * `file_first` - The file's first chromosome
///
/// ### Returns
///
/// The tag size, 0 for an empty group.
fn fragment_tag_size(group: &GroupTags, file_first: Option<&str>) -> u32 {
    let chrom = match file_first.filter(|c| group.first.contains_key(*c)) {
        Some(c) => c,
        None => match group.chroms.first() {
            Some((c, _)) => c.as_str(),
            None => return 0,
        },
    };
    let mut first = group.first.get(chrom).cloned().unwrap_or_default();
    first.sort_unstable();
    first.truncate(TAG_SIZE_SAMPLE);
    if first.is_empty() {
        return 0;
    }
    let sum: u64 = first.iter().map(|&(_, l)| l as u64).sum();
    (sum as f64 / first.len() as f64) as u32
}

/// Sort tags and drop duplicates above `keep`.
///
/// ### Params
///
/// * `tags` - Tag positions, any order
/// * `keep` - Duplicate handling
fn sort_and_dedup(tags: &mut Vec<u32>, keep: KeepDup) {
    tags.par_sort_unstable();
    let KeepDup::AtMost(max) = keep else {
        return;
    };
    let mut w = 0;
    let mut run = 0u32;
    for r in 0..tags.len() {
        run = if r > 0 && tags[r] == tags[r - 1] {
            run + 1
        } else {
            1
        };
        if run <= max {
            tags[w] = tags[r];
            w += 1;
        }
    }
    tags.truncate(w);
}

////////////
// Pileup //
////////////

/// Pileup of tags extended to `[tag - five, tag + three)`, as MACS's
/// `pileup_from_PN_shifted` builds it for plus strand tags.
///
/// Ends are clipped at 0. A segment's value is the count times `scale`,
/// floored at `baseline`; the leading uncovered segment `[0, first)` takes
/// `max(0, baseline)`.
///
/// ### Params
///
/// * `tags` - Sorted tag positions
/// * `five` - Distance from the tag to the start of its extension
/// * `three` - Distance from the tag to the end of its extension
/// * `scale` - Factor on the count
/// * `baseline` - Floor of every value
///
/// ### Returns
///
/// `(segment ends, values)`.
fn pileup(tags: &[u32], five: i64, three: i64, scale: f32, baseline: f32) -> (Vec<i32>, Vec<f32>) {
    let clip = |x: i64| x.max(0) as i32;
    let starts: Vec<i32> = tags.iter().map(|&t| clip(t as i64 - five)).collect();
    let ends: Vec<i32> = tags.iter().map(|&t| clip(t as i64 + three)).collect();
    let n = starts.len();
    let mut pos = Vec::with_capacity(2 * n);
    let mut val = Vec::with_capacity(2 * n);
    if n == 0 {
        return (pos, val);
    }

    let mut pre_p = starts[0].min(ends[0]);
    if pre_p != 0 {
        pos.push(pre_p);
        val.push(0f32.max(baseline));
    }
    let (mut i_s, mut i_e) = (0, 0);
    let mut count: i32 = 0;
    let mut emit = |p: i32, count: i32, pos: &mut Vec<i32>, val: &mut Vec<f32>| {
        if p != pre_p {
            pos.push(p);
            val.push((count as f32 * scale).max(baseline));
            pre_p = p;
        }
    };
    while i_s < n && i_e < n {
        if starts[i_s] < ends[i_e] {
            emit(starts[i_s], count, &mut pos, &mut val);
            count += 1;
            i_s += 1;
        } else if starts[i_s] > ends[i_e] {
            emit(ends[i_e], count, &mut pos, &mut val);
            count -= 1;
            i_e += 1;
        } else {
            i_s += 1;
            i_e += 1;
        }
    }
    while i_e < n {
        emit(ends[i_e], count, &mut pos, &mut val);
        count -= 1;
        i_e += 1;
    }
    (pos, val)
}

/// Treatment and local lambda pileups of one chromosome, paired on the
/// union of their breakpoints up to the shorter one's end.
///
/// ### Params
///
/// * `tags` - Sorted tags of the chromosome
/// * `ctx` - Group-wide shifts, scale and background
///
/// ### Returns
///
/// The paired track.
fn paired_track(tags: &[u32], ctx: &CallContext) -> PairedTrack {
    let (tp, tv) = pileup(tags, ctx.treat_five, ctx.treat_three, 1.0, 0.0);
    let (cp, cv) = pileup(
        tags,
        ctx.ctrl_five,
        ctx.ctrl_three,
        ctx.ctrl_scale,
        ctx.lambda_bg,
    );
    let mut out = PairedTrack {
        pos: Vec::with_capacity(tp.len() + cp.len()),
        treat: Vec::with_capacity(tp.len() + cp.len()),
        ctrl: Vec::with_capacity(tp.len() + cp.len()),
    };
    let (mut it, mut ic) = (0, 0);
    while it < tp.len() && ic < cp.len() {
        out.pos.push(tp[it].min(cp[ic]));
        out.treat.push(tv[it]);
        out.ctrl.push(cv[ic]);
        let (t, c) = (tp[it], cp[ic]);
        it += (t <= c) as usize;
        ic += (c <= t) as usize;
    }
    out
}

/// Histogram key of a segment.
///
/// ### Params
///
/// * `treat` - Treatment pileup
/// * `ctrl` - Local lambda
///
/// ### Returns
///
/// The key; the pileup is truncated to an integer as MACS does.
#[inline]
fn pkey(treat: f32, ctrl: f32) -> PKey {
    (treat as i32 as u32, ctrl.to_bits())
}

////////////
// Scores //
////////////

/// Round to five decimals, ties to even, as Python's `round(x, 5)` (MACS's
/// `round5`).
///
/// ### Params
///
/// * `x` - Value
///
/// ### Returns
///
/// The double nearest to the rounded decimal.
fn round5(x: f64) -> f64 {
    let mut r = (x * 1e5).round_ties_even();
    let d = x.mul_add(1e5, -(r + 0.5));
    if d > 0.0 {
        r += 1.0;
    } else if d == 0.0 {
        if r % 2.0 != 0.0 {
            r += 1.0;
        }
    } else {
        let d = x.mul_add(1e5, -(r - 0.5));
        if d < 0.0 || (d == 0.0 && r % 2.0 != 0.0) {
            r -= 1.0;
        }
    }
    if r == 0.0 {
        return 0f64.copysign(x);
    }
    r / 1e5
}

/// `log10 P(X > k)` for `X ~ Poisson(lambda)`, summed in log space as MACS's
/// `log10_poisson_cdf_Q_large_lambda` does.
///
/// ### Params
///
/// * `k` - Observed count
/// * `lambda` - Expectation, positive
/// * `ln_fact` - `ln_fact[m] = ln(1) + ... + ln(m)`, summed in ascending
///   order, covering at least `k + 1`
///
/// ### Returns
///
/// The log10 tail, rounded to five decimals.
fn log10_poisson_upper(k: u32, lambda: f64, ln_fact: &[f64]) -> f64 {
    let ln_lbd = lambda.ln();
    let mut m = k as u64 + 1;
    let mut logx = m as f64 * ln_lbd - ln_fact[m as usize];
    let mut residue = logx;
    loop {
        m += 1;
        let logy = logx + ln_lbd - (m as f64).ln();
        let pre = residue;
        residue = if pre > logy {
            pre + (logy - pre).exp().ln_1p()
        } else {
            logy + (pre - logy).exp().ln_1p()
        };
        if (pre - residue).abs() < POISSON_TAIL_TOLERANCE {
            break;
        }
        logx = logy;
    }
    round5((residue - lambda) / 10f64.ln())
}

/// Key of a p-score in the q-score table: its bits, with `-0.0` as `0.0`.
///
/// ### Params
///
/// * `p` - p-score
///
/// ### Returns
///
/// The key.
#[inline]
fn qkey(p: f32) -> u32 {
    if p == 0.0 { 0 } else { p.to_bits() }
}

/// p-scores of every histogram key and the p-score to q-score table.
///
/// q-scores are Benjamini-Hochberg over segments weighted by length: the
/// distinct p-scores in descending order, each `p + log10(rank) - log10(N)`
/// with `rank` the bases above it plus one and `N` all bases, kept monotone
/// and floored at 0. The arithmetic and its f32 roundings are MACS's, which
/// includes giving the lowest p-score a q-score of 0 even when the loop runs
/// to the end.
///
/// ### Params
///
/// * `hist` - Summed segment length per key
///
/// ### Returns
///
/// `(p-score per key, q-score per p-score key)`.
fn score_tables(hist: &FxHashMap<PKey, i64>) -> (FxHashMap<PKey, f32>, FxHashMap<u32, f32>) {
    let max_k = hist.keys().map(|&(k, _)| k).max().unwrap_or(0) as usize;
    let mut ln_fact = vec![0f64; max_k + 2];
    for m in 1..ln_fact.len() {
        ln_fact[m] = ln_fact[m - 1] + (m as f64).ln();
    }
    let pscores: FxHashMap<PKey, f32> = hist
        .par_iter()
        .map(|(&key, _)| {
            let lam = f32::from_bits(key.1) as f64;
            (key, (-log10_poisson_upper(key.0, lam, &ln_fact)) as f32)
        })
        .collect();

    let mut by_p: FxHashMap<u32, (f32, i64)> = FxHashMap::default();
    for (key, &len) in hist {
        let p = pscores[key];
        by_p.entry(qkey(p)).or_insert((p, 0)).1 += len;
    }
    let mut uniq: Vec<(f32, i64)> = by_p.into_values().collect();
    uniq.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let mut qtable = FxHashMap::default();
    let n_bases: i64 = uniq.iter().map(|&(_, l)| l).sum();
    if uniq.is_empty() {
        return (pscores, qtable);
    }
    let f = if n_bases > 0 {
        -(n_bases as f64).log10() as f32
    } else {
        0.0
    };
    let mut pre_q = if n_bases > 0 { Q_START } else { 0.0 };
    let mut k: i64 = 1;
    let mut stop = uniq.len() - 1;
    for (i, &(v, l)) in uniq.iter().enumerate() {
        let mut q = (v as f64 + ((k as f64).log10() + f as f64)) as f32;
        if q > pre_q {
            q = pre_q;
        }
        if q <= 0.0 {
            stop = i;
            break;
        }
        qtable.insert(qkey(v), q);
        pre_q = q;
        k += l;
    }
    for &(v, _) in &uniq[stop..] {
        qtable.insert(qkey(v), 0.0);
    }
    (pscores, qtable)
}

/////////////
// Calling //
/////////////

/// Call the peaks of one group.
///
/// Chromosomes run in parallel twice: once for the p-score histogram, once
/// to call. The pileups are rebuilt in the second pass rather than kept, so
/// memory stays at the tags plus one track per thread.
///
/// ### Params
///
/// * `group` - The group's tags
/// * `params` - Validated parameters
/// * `file_first` - The fragment file's first chromosome, for the default
///   `max_gap` of fragment start tags
///
/// ### Returns
///
/// The peaks, chromosomes in name order, peaks by position.
fn call_group(
    mut group: GroupTags,
    params: &PeakCallParams,
    file_first: Option<&str>,
) -> Vec<NarrowPeak> {
    let tag_size = match params.tag_source {
        TagSource::Insertion => 1,
        TagSource::FragmentStart => fragment_tag_size(&group, file_first),
    };
    group
        .chroms
        .par_iter_mut()
        .for_each(|(_, t)| sort_and_dedup(t, params.keep_dup));
    let n_tags: usize = group.chroms.iter().map(|(_, t)| t.len()).sum();
    if n_tags == 0 {
        return Vec::new();
    }

    let d = params.extsize as i64;
    let llocal = params.llocal as i64;
    let ctx = CallContext {
        treat_five: -(params.shift as i64),
        treat_three: params.shift as i64 + d,
        ctrl_five: llocal / 2,
        ctrl_three: llocal - llocal / 2,
        ctrl_scale: (params.extsize as f64 / params.llocal as f64) as f32,
        lambda_bg: (params.extsize as f64 * n_tags as f64 / params.genome_size) as f32,
    };

    let hist: FxHashMap<PKey, i64> = group
        .chroms
        .par_iter()
        .map(|(_, tags)| {
            let track = paired_track(tags, &ctx);
            let mut h: FxHashMap<PKey, i64> = FxHashMap::default();
            let mut pre = 0i64;
            for i in 0..track.pos.len() {
                *h.entry(pkey(track.treat[i], track.ctrl[i])).or_insert(0) +=
                    track.pos[i] as i64 - pre;
                pre = track.pos[i] as i64;
            }
            h
        })
        .reduce(FxHashMap::default, |mut a, b| {
            for (k, l) in b {
                *a.entry(k).or_insert(0) += l;
            }
            a
        });
    let (pscores, qtable) = score_tables(&hist);

    // MACS takes Python's log(q, 10), not log10(q)
    let cutoff = -(params.q_value.ln() / 10f64.ln());
    let min_length = params.min_length.unwrap_or(params.extsize) as i64;
    let max_gap = params.max_gap.unwrap_or(tag_size) as i64;

    group
        .chroms
        .par_iter()
        .flat_map_iter(|(chrom, tags)| {
            let track = paired_track(tags, &ctx);
            let above: Vec<usize> = (0..track.pos.len())
                .filter(|&i| {
                    let p = pscores[&pkey(track.treat[i], track.ctrl[i])];
                    qtable[&qkey(p)] as f64 > cutoff
                })
                .collect();
            let mut peaks = Vec::new();
            let mut close = |seg: &[usize]| {
                if let Some(p) = close_peak(chrom, seg, &track, min_length, &pscores, &qtable) {
                    peaks.push(p);
                }
            };
            if !above.is_empty() {
                let mut g0 = 0;
                let mut lastp = track.pos[above[0]] as i64;
                for k in 1..above.len() {
                    let ti = above[k];
                    if track.pos[ti - 1] as i64 - lastp > max_gap {
                        close(&above[g0..k]);
                        g0 = k;
                    }
                    lastp = track.pos[ti] as i64;
                }
                close(&above[g0..]);
            }
            peaks
        })
        .collect()
}

/// Close one region of segments above the cutoff into a peak.
///
/// The summit is the middle one, in order, of the segments holding the
/// highest treatment pileup.
///
/// ### Params
///
/// * `chrom` - Chromosome
/// * `seg` - Indices of the region's segments, increasing
/// * `track` - The chromosome's paired track
/// * `min_length` - Shorter regions are dropped
/// * `pscores` - p-score per histogram key
/// * `qtable` - q-score per p-score
///
/// ### Returns
///
/// The peak, or `None` for a region shorter than `min_length`.
fn close_peak(
    chrom: &str,
    seg: &[usize],
    track: &PairedTrack,
    min_length: i64,
    pscores: &FxHashMap<PKey, f32>,
    qtable: &FxHashMap<u32, f32>,
) -> Option<NarrowPeak> {
    let seg_start = |i: usize| if i > 0 { track.pos[i - 1] as i64 } else { 0 };
    let start = seg_start(seg[0]);
    let end = track.pos[*seg.last().expect("a region has a segment")] as i64;
    if end - start < min_length {
        return None;
    }

    let mut summit_value = 0f64;
    let mut first = 0;
    let mut n_tie: usize = 0;
    for (k, &i) in seg.iter().enumerate() {
        let t = track.treat[i] as f64;
        if summit_value == 0.0 || summit_value < t {
            first = k;
            n_tie = 1;
            summit_value = t;
        } else if summit_value == t {
            n_tie += 1;
        }
    }
    let mid = n_tie.div_ceil(2) - 1;
    let ti = seg[first..]
        .iter()
        .copied()
        .filter(|&i| track.treat[i] as f64 == summit_value)
        .nth(mid)
        .expect("the tie count covers the middle");

    let treat = track.treat[ti];
    let ctrl = track.ctrl[ti];
    let pscore = pscores[&pkey(treat, ctrl)];
    Some(NarrowPeak {
        chrom: chrom.to_string(),
        start: start as u32,
        end: end as u32,
        summit: ((seg_start(ti) + track.pos[ti] as i64) / 2) as u32,
        pileup: treat,
        pscore,
        fold: ((treat as f64 + FOLD_PSEUDOCOUNT) / (ctrl as f64 + FOLD_PSEUDOCOUNT)) as f32,
        qscore: qtable[&qkey(pscore)],
    })
}

/// Call peaks per group of cells.
///
/// One pass over the fragment file pools the tags of every group, then the
/// groups are called in parallel. Merge the groups' peaks with
/// [`crate::regions::reduce_regions`] for one peak set to count against.
///
/// ### Params
///
/// * `path` - Fragment file
/// * `barcodes` - Cells to use
/// * `groups` - Group of each cell, `0..n_groups`
/// * `params` - See [`PeakCallParams`]; `None` for the defaults
///
/// ### Returns
///
/// The peaks of each group, indexed by group, or an I/O, parse or argument
/// error.
pub fn call_peaks(
    path: &Path,
    barcodes: &[String],
    groups: &[u32],
    params: Option<PeakCallParams>,
) -> Result<Vec<Vec<NarrowPeak>>, AtacErrors> {
    let params = params.unwrap_or_default();
    params.validate()?;
    if barcodes.len() != groups.len() {
        return Err(AtacErrors::InvalidArgument(format!(
            "{} barcodes but {} group labels",
            barcodes.len(),
            groups.len()
        )));
    }
    let n_groups = groups.iter().max().map_or(0, |&g| g as usize + 1);
    let tags = collect_tags(path, barcodes, groups, n_groups, params.tag_source)?;
    let file_first = match params.tag_source {
        TagSource::FragmentStart if params.max_gap.is_none() => first_chromosome(path)?,
        _ => None,
    };
    Ok(tags
        .into_par_iter()
        .map(|g| call_group(g, &params, file_first.as_deref()))
        .collect())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values from CPython's `round(x, 5)`.
    #[test]
    fn test_round5_matches_python_round() {
        assert_eq!(round5(1.234565), 1.23456);
        assert_eq!(round5(-2.000004), -2.0);
        assert_eq!(round5(0.000015), 2e-5);
        assert_eq!(round5(0.000025), 3e-5);
        assert_eq!(round5(0.000001), 0.0);
        assert!(round5(-0.000001).is_sign_negative());
    }

    #[test]
    fn test_poisson_tail_matches_direct_sum() {
        let mut ln_fact = vec![0f64; 256];
        for m in 1..ln_fact.len() {
            ln_fact[m] = ln_fact[m - 1] + (m as f64).ln();
        }
        for &(k, lam) in &[(0u32, 0.5f64), (3, 1.2), (10, 2.0), (25, 4.0)] {
            let tail: f64 = (k as usize + 1..ln_fact.len())
                .map(|i| (i as f64 * lam.ln() - lam - ln_fact[i]).exp())
                .sum();
            let direct = tail.log10();
            let ours = log10_poisson_upper(k, lam, &ln_fact);
            assert!(
                (ours - direct).abs() < 1e-4,
                "k {k} lambda {lam}: {ours} vs {direct}"
            );
        }
    }

    #[test]
    fn test_pileup_counts_overlaps() {
        // Tags 10 and 15 extended to [t - 5, t + 5): [5, 15) and [10, 20).
        let (pos, val) = pileup(&[10, 15], 5, 5, 1.0, 0.0);
        assert_eq!(pos, vec![5, 10, 15, 20]);
        assert_eq!(val, vec![0.0, 1.0, 2.0, 1.0]);
    }

    #[test]
    fn test_pileup_clips_at_zero_and_floors_at_baseline() {
        let (pos, val) = pileup(&[2], 5, 5, 0.5, 0.75);
        assert_eq!(pos, vec![7]);
        assert_eq!(val, vec![0.75]);
    }

    #[test]
    fn test_dedup_keeps_at_most_n() {
        let mut t = vec![5, 1, 5, 5, 3, 1];
        sort_and_dedup(&mut t, KeepDup::AtMost(2));
        assert_eq!(t, vec![1, 1, 3, 5, 5]);
        let mut t = vec![5, 1, 5];
        sort_and_dedup(&mut t, KeepDup::All);
        assert_eq!(t, vec![1, 5, 5]);
    }

    #[test]
    fn test_invalid_params_are_refused() {
        let p = PeakCallParams {
            keep_dup: KeepDup::AtMost(0),
            ..Default::default()
        };
        assert!(p.validate().is_err());
        let p = PeakCallParams {
            q_value: 0.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }
}
