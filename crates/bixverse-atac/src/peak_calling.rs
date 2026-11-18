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
//!
//! Memory: three passes. Pass 1 reads the fragment file in chromosome
//! batches (through its tabix index, one chromosome at a time by default),
//! sorts and deduplicates the tags and spills them to a scratch file, 4 bytes
//! per tag; the background lambda needs the tag count of the whole group
//! before any segment can be scored. Pass 2 sums segment lengths per score
//! from the spill, pass 3 calls from it. The pileups are never stored: they
//! are swept straight off the sorted tags, in position windows that run in
//! parallel. Peak memory is one batch's tags in pass 1 and one chromosome's
//! after. Without an index pass 1 holds the whole file's tags.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::iter::Peekable;
use std::path::{Path, PathBuf};

use flate2::read::MultiGzDecoder;
use memmap2::{Mmap, MmapOptions};

use bixverse_sc_data::interval_index::FeatureInterval;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::fragments::{ChromSpan, chrom_spans, fold_fragment_spans};
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

/// Bytes of memory per compressed fragment file byte in flight, to turn
/// `max_mem_gb` into chromosome batches. Tags are 4 bytes and a fragment line
/// compresses to roughly 10 bytes, giving up to two tags; the factor leaves
/// room for vector growth and the per-worker buffers of the parallel read.
/// Not tuned.
const MEM_PER_COMPRESSED_BYTE: f64 = 2.0;

/// Tags per position window of a chromosome sweep. Windows overlap by the
/// local lambda window on each side, so they must be much wider than it;
/// beyond that, smaller windows balance the threads better. Not tuned.
const WINDOW_TAGS: usize = 1 << 20;

/// How far under the cutoff a p-score still gets an exact q-score. A q-score
/// never exceeds its p-score in exact arithmetic, but both pass through f32,
/// so a p-score a rounding step under the cutoff could in principle land
/// over it; the margin is orders of magnitude wider than that step.
const CANDIDATE_MARGIN: f64 = 1e-3;

/// Ratio between the p-score levels that bound which keys can pass. Finer
/// levels exclude slightly more keys for one more threshold walk each. Not
/// tuned.
const LEVEL_RATIO: f64 = 1.25;

/// Highest p-score level. Past it a level would exclude only keys with
/// astronomically small p-values, which are few.
const LEVEL_MAX: f64 = 1e3;

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
#[derive(Clone, Debug)]
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
    /// Memory budget of the tags read at once in pass 1, in GB. With a tabix
    /// index, chromosomes are read in batches that fit it; `None` reads one
    /// chromosome at a time. Without an index the whole file is read at once.
    pub max_mem_gb: Option<f64>,
    /// Directory of the scratch file holding the tags between passes;
    /// `None` for the system temporary directory.
    pub tmp_dir: Option<PathBuf>,
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
            max_mem_gb: None,
            tmp_dir: None,
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
    /// * `max_mem_gb` - Memory budget of pass 1, `None` for one chromosome
    ///   at a time
    /// * `tmp_dir` - Scratch directory, `None` for the system one
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
        max_mem_gb: Option<f64>,
        tmp_dir: Option<PathBuf>,
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
            max_mem_gb,
            tmp_dir,
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
        if self.max_mem_gb.is_some_and(|m| m.is_nan() || m <= 0.0) {
            return bad("max_mem_gb must be positive");
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

/// Tags of one read: per group, per chromosome, unsorted.
type BatchTags = Vec<FxHashMap<String, Vec<u32>>>;

/// First fragments of one read, as `(start, length)`, at most
/// [`TAG_SIZE_SAMPLE`] per group and chromosome.
type FirstFragments = Vec<FxHashMap<String, Vec<(u32, u32)>>>;

/// A segment's treatment pileup and the bits of its local lambda, MACS's
/// p-score key.
type ScoreKey = (u32, u32);

/// One segment: `(start, end, treatment count, lambda count)`.
type Segment = (i32, i32, u32, u32);

/// How the fragment file is read.
enum Reads {
    /// Chromosome batches through the tabix index, in file order.
    Batches(Vec<Vec<ChromSpan>>),
    /// The whole file at once.
    Whole,
}

/// Extension of the tags of both pileups.
#[derive(Clone, Copy)]
struct Extension {
    /// Distance from a tag to the start of its treatment extension (MACS
    /// `five_shift`).
    treat_five: i64,
    /// Distance from a tag to the end of its treatment extension (MACS
    /// `three_shift`).
    treat_three: i64,
    /// Same for the local lambda window.
    ctrl_five: i64,
    /// Same for the local lambda window.
    ctrl_three: i64,
}

/// Turns a lambda count into the local lambda.
#[derive(Clone, Copy)]
struct Lambda {
    /// `extsize / llocal`, f32 as in MACS.
    scale: f32,
    /// Genome background, the floor of the local lambda.
    background: f32,
}

impl Lambda {
    /// The local lambda of a count, as MACS's pileup computes it.
    ///
    /// ### Params
    ///
    /// * `count` - Tags in the window
    ///
    /// ### Returns
    ///
    /// `max(count * scale, background)` in f32.
    #[inline]
    fn value(&self, count: u32) -> f32 {
        (count as f32 * self.scale).max(self.background)
    }

    /// The score key of a segment.
    ///
    /// ### Params
    ///
    /// * `treat` - Treatment count
    /// * `ctrl` - Lambda count
    ///
    /// ### Returns
    ///
    /// `(treat, bits of the local lambda)`.
    #[inline]
    fn key(&self, treat: u32, ctrl: u32) -> ScoreKey {
        (treat, self.value(ctrl).to_bits())
    }
}

/// What scoring a group's segments needs.
struct GroupScores {
    /// The group's local lambda.
    lambda: Lambda,
    /// Treatment counts that pass, per lambda count, from
    /// [`pscore_thresholds`].
    pass_thresholds: Vec<u32>,
    /// `(p-score, q-score)` of every passing p-score, p-scores descending.
    passing: Vec<(f32, f32)>,
    /// From [`ln_factorials`], covering every passing treatment count.
    ln_fact: Vec<f64>,
}

impl GroupScores {
    /// Whether a segment passes the cutoff.
    ///
    /// ### Params
    ///
    /// * `t` - Treatment count
    /// * `c` - Lambda count, at most the group's largest
    ///
    /// ### Returns
    ///
    /// `true` when its q-score is above the cutoff.
    #[inline]
    fn passes(&self, t: u32, c: u32) -> bool {
        t >= self.pass_thresholds[c as usize]
    }

    /// p-score and q-score of a passing segment.
    ///
    /// ### Params
    ///
    /// * `t` - Treatment count
    /// * `c` - Lambda count
    ///
    /// ### Returns
    ///
    /// `(p-score, q-score)`.
    fn score(&self, t: u32, c: u32) -> (f32, f32) {
        let lam = self.lambda.value(c) as f64;
        let p = (-log10_poisson_upper(t, lam, &self.ln_fact)) as f32;
        let i = self
            .passing
            .binary_search_by(|probe| p.total_cmp(&probe.0))
            .expect("a passing segment has a passing p-score");
        (p, self.passing[i].1)
    }
}

/// A window of a chromosome's positions, `[lo, hi)`, and the slice
/// `tags[a..b]` of the tags whose extensions can reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Window {
    /// First position.
    lo: i64,
    /// One past the last position.
    hi: i64,
    /// First tag.
    a: usize,
    /// One past the last tag.
    b: usize,
}

////////////
// Pileup //
////////////

/// The breakpoints of one pileup, as MACS's `pileup_from_PN_shifted` emits
/// them for plus strand tags extended to `[tag - five, tag + three)`, ends
/// clipped at 0, without storing them.
///
/// Each item is `(segment end, count over the segment)`; a leading uncovered
/// segment `[0, first)` comes out with count 0.
struct PileupSweep<'a> {
    /// Sorted tags.
    tags: &'a [u32],
    /// Distance to the start of an extension.
    five: i64,
    /// Distance to the end of an extension.
    three: i64,
    /// Next start to visit.
    i_s: usize,
    /// Next end to visit.
    i_e: usize,
    /// Extensions covering the current position.
    count: u32,
    /// Last breakpoint emitted.
    pre_p: i32,
    /// The leading breakpoint, until emitted.
    lead: Option<i32>,
}

impl<'a> PileupSweep<'a> {
    /// Start a sweep.
    ///
    /// ### Params
    ///
    /// * `tags` - Sorted tag positions
    /// * `five` - Distance to the start of an extension
    /// * `three` - Distance to the end of an extension
    ///
    /// ### Returns
    ///
    /// The sweep.
    fn new(tags: &'a [u32], five: i64, three: i64) -> Self {
        let mut s = Self {
            tags,
            five,
            three,
            i_s: 0,
            i_e: 0,
            count: 0,
            pre_p: 0,
            lead: None,
        };
        if !tags.is_empty() {
            s.pre_p = s.start(0).min(s.end(0));
            if s.pre_p != 0 {
                s.lead = Some(s.pre_p);
            }
        }
        s
    }

    /// Start of extension `i`, clipped at 0.
    ///
    /// ### Params
    ///
    /// * `i` - Tag index
    ///
    /// ### Returns
    ///
    /// The position.
    #[inline]
    fn start(&self, i: usize) -> i32 {
        (self.tags[i] as i64 - self.five).max(0) as i32
    }

    /// End of extension `i`, clipped at 0.
    ///
    /// ### Params
    ///
    /// * `i` - Tag index
    ///
    /// ### Returns
    ///
    /// The position.
    #[inline]
    fn end(&self, i: usize) -> i32 {
        (self.tags[i] as i64 + self.three).max(0) as i32
    }
}

impl Iterator for PileupSweep<'_> {
    type Item = (i32, u32);

    /// The next breakpoint.
    ///
    /// ### Returns
    ///
    /// `(segment end, count)`, or `None` past the last extension.
    fn next(&mut self) -> Option<(i32, u32)> {
        if let Some(p) = self.lead.take() {
            return Some((p, 0));
        }
        let n = self.tags.len();
        while self.i_s < n && self.i_e < n {
            let (s, e) = (self.start(self.i_s), self.end(self.i_e));
            let c = self.count;
            if s < e {
                self.count += 1;
                self.i_s += 1;
                if s != self.pre_p {
                    self.pre_p = s;
                    return Some((s, c));
                }
            } else if s > e {
                self.count -= 1;
                self.i_e += 1;
                if e != self.pre_p {
                    self.pre_p = e;
                    return Some((e, c));
                }
            } else {
                self.i_s += 1;
                self.i_e += 1;
            }
        }
        while self.i_e < n {
            let e = self.end(self.i_e);
            let c = self.count;
            self.count -= 1;
            self.i_e += 1;
            if e != self.pre_p {
                self.pre_p = e;
                return Some((e, c));
            }
        }
        None
    }
}

/// Segments of the treatment and local lambda pileups paired on the union of
/// their breakpoints, up to the shorter one's end, as MACS pairs them.
struct Segments<'a> {
    /// Treatment breakpoints.
    treat: Peekable<PileupSweep<'a>>,
    /// Local lambda breakpoints.
    ctrl: Peekable<PileupSweep<'a>>,
    /// End of the previous segment.
    pre: i32,
}

impl<'a> Segments<'a> {
    /// Pair the two pileups of sorted tags.
    ///
    /// ### Params
    ///
    /// * `tags` - Sorted tags of one chromosome
    /// * `ext` - Extensions of both pileups
    ///
    /// ### Returns
    ///
    /// The segment iterator.
    fn new(tags: &'a [u32], ext: &Extension) -> Self {
        Self {
            treat: PileupSweep::new(tags, ext.treat_five, ext.treat_three).peekable(),
            ctrl: PileupSweep::new(tags, ext.ctrl_five, ext.ctrl_three).peekable(),
            pre: 0,
        }
    }
}

impl Iterator for Segments<'_> {
    type Item = Segment;

    /// The next segment.
    ///
    /// ### Returns
    ///
    /// `(start, end, treatment count, lambda count)`.
    fn next(&mut self) -> Option<Segment> {
        let &(tp, tv) = self.treat.peek()?;
        let &(cp, cv) = self.ctrl.peek()?;
        let p = tp.min(cp);
        if tp <= cp {
            self.treat.next();
        }
        if cp <= tp {
            self.ctrl.next();
        }
        let start = self.pre;
        self.pre = p;
        Some((start, p, tv, cv))
    }
}

/// Cut a chromosome into position windows that can be swept independently.
///
/// Windows start at tags, every `per_window` tags. A window's slice holds
/// every tag whose extension (in either pileup) can cover one of its
/// positions, so its sweep reproduces the full sweep's segments inside it,
/// and the tag it ends at keeps both pileups running past its end. Shifts
/// that break either property get one window.
///
/// ### Params
///
/// * `tags` - Sorted tags
/// * `ext` - Extensions of both pileups
/// * `per_window` - Tags per window
///
/// ### Returns
///
/// The windows, in order, covering every position.
fn windows(tags: &[u32], ext: &Extension, per_window: usize) -> Vec<Window> {
    let n = tags.len();
    let back = ext.treat_three.max(ext.ctrl_three);
    let fwd = ext.treat_five.max(ext.ctrl_five);
    let whole = vec![Window {
        lo: i64::MIN,
        hi: i64::MAX,
        a: 0,
        b: n,
    }];
    if n <= per_window || ext.treat_three <= 0 || fwd < 0 {
        return whole;
    }
    let mut cuts: Vec<i64> = (per_window..n)
        .step_by(per_window)
        .map(|i| tags[i] as i64)
        .collect();
    cuts.dedup();
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut lo = i64::MIN;
    for hi in cuts.into_iter().chain(std::iter::once(i64::MAX)) {
        let a = tags.partition_point(|&t| (t as i64) < lo.saturating_sub(back));
        let b = tags.partition_point(|&t| (t as i64) <= hi.saturating_add(fwd));
        out.push(Window { lo, hi, a, b });
        lo = hi;
    }
    out
}

/// The segments of one window, clipped to it.
///
/// ### Params
///
/// * `tags` - Sorted tags of the chromosome
/// * `ext` - Extensions of both pileups
/// * `w` - The window
///
/// ### Returns
///
/// `(segment, clipped)`: the clipped segment and whether it started before
/// the window, so a neighbour holds its other part.
fn window_segments<'a>(
    tags: &'a [u32],
    ext: &Extension,
    w: Window,
) -> impl Iterator<Item = (Segment, bool)> + 'a {
    Segments::new(&tags[w.a..w.b], ext)
        .take_while(move |seg| (seg.0 as i64) < w.hi)
        .filter_map(move |(s, e, t, c)| {
            let s2 = (s as i64).max(w.lo);
            let e2 = (e as i64).min(w.hi);
            (s2 < e2).then_some(((s2 as i32, e2 as i32, t, c), (s as i64) < w.lo))
        })
}

/// Segment lengths of a group: per score key for the candidates (segments
/// whose p-score can clear the cutoff), summed for the rest.
#[derive(Default)]
struct Histogram {
    /// Summed length per candidate score key.
    candidates: FxHashMap<ScoreKey, i64>,
    /// Summed length of every segment.
    n_bases: i64,
}

impl Histogram {
    /// Add another histogram.
    ///
    /// ### Params
    ///
    /// * `other` - Histogram
    ///
    /// ### Returns
    ///
    /// The sum.
    fn merge(mut self, mut other: Histogram) -> Histogram {
        if self.candidates.len() < other.candidates.len() {
            std::mem::swap(&mut self.candidates, &mut other.candidates);
        }
        for (k, l) in other.candidates {
            *self.candidates.entry(k).or_insert(0) += l;
        }
        self.n_bases += other.n_bases;
        self
    }
}

/// Whether a segment is a candidate: its treatment count reaches the
/// threshold of its lambda count.
///
/// ### Params
///
/// * `thresholds` - From [`pscore_thresholds`]
/// * `t` - Treatment count
/// * `c` - Lambda count
///
/// ### Returns
///
/// `true` for a candidate; a lambda count past the table counts as one, so
/// nothing is ever lost.
#[inline]
fn is_candidate(thresholds: &[u32], t: u32, c: u32) -> bool {
    thresholds.get(c as usize).is_none_or(|&th| t >= th)
}

/// Segment lengths of one chromosome.
///
/// ### Params
///
/// * `tags` - Sorted tags
/// * `ext` - Extensions of both pileups
/// * `lambda` - The group's local lambda
/// * `thresholds` - The group's candidate thresholds
///
/// ### Returns
///
/// The histogram.
fn length_histogram(
    tags: &[u32],
    ext: &Extension,
    lambda: Lambda,
    thresholds: &[u32],
) -> Histogram {
    windows(tags, ext, WINDOW_TAGS)
        .into_par_iter()
        .map(|w| {
            let mut h = Histogram::default();
            for ((s, e, t, c), _) in window_segments(tags, ext, w) {
                let len = (e - s) as i64;
                h.n_bases += len;
                if is_candidate(thresholds, t, c) {
                    *h.candidates.entry(lambda.key(t, c)).or_insert(0) += len;
                }
            }
            h
        })
        .reduce(Histogram::default, Histogram::merge)
}

/// The largest number of tags in any window of `width` positions, which
/// bounds the lambda count.
///
/// ### Params
///
/// * `tags` - Sorted tags
/// * `width` - Window width
///
/// ### Returns
///
/// The count.
fn max_window_count(tags: &[u32], width: u32) -> u32 {
    let mut best = 0;
    let mut j = 0;
    for (i, &t) in tags.iter().enumerate() {
        while j < tags.len() && (tags[j] as u64) < t as u64 + width as u64 {
            j += 1;
        }
        best = best.max(j - i);
    }
    best as u32
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

/// `ln(1) + ... + ln(m)` for `m` up to `n`, summed in ascending order from
/// 0.0, as MACS's cached table holds them.
///
/// ### Params
///
/// * `n` - Largest `m`
///
/// ### Returns
///
/// The table, `n + 1` entries.
fn ln_factorials(n: usize) -> Vec<f64> {
    let mut ln_fact = vec![0f64; n + 1];
    for m in 1..=n {
        ln_fact[m] = ln_fact[m - 1] + (m as f64).ln();
    }
    ln_fact
}

/// Per lambda count, the smallest treatment count whose p-score is above
/// `floor`.
///
/// The p-score grows with the treatment count at a fixed lambda and falls
/// with the lambda at a fixed count, so the thresholds never decrease with
/// the lambda count: one walk finds them all, about one p-score per count.
/// They split segments by p-score without a p-score per segment.
///
/// ### Params
///
/// * `lambda` - The group's local lambda
/// * `c_max` - Largest lambda count
/// * `floor` - p-score to exceed
///
/// ### Returns
///
/// The thresholds, indexed by lambda count.
fn pscore_thresholds(lambda: Lambda, c_max: u32, floor: f64) -> Vec<u32> {
    let mut out = Vec::with_capacity(c_max as usize + 1);
    // grown as the walk climbs; ascending sums, so the same values as a
    // table built in one go
    let mut ln_fact = vec![0f64];
    let mut t = 0u32;
    let mut last_bits = None;
    for c in 0..=c_max {
        let lam = lambda.value(c);
        // counts under the background share its lambda and its threshold
        if last_bits != Some(lam.to_bits()) {
            last_bits = Some(lam.to_bits());
            loop {
                while ln_fact.len() < t as usize + 2 {
                    let m = ln_fact.len();
                    ln_fact.push(ln_fact[m - 1] + (m as f64).ln());
                }
                if (-log10_poisson_upper(t, lam as f64, &ln_fact)) as f32 as f64 > floor {
                    break;
                }
                t += 1;
            }
        }
        out.push(t);
    }
    out
}

/// p-score levels for bounding which keys can still pass: just under the
/// cutoff, then geometric steps up.
///
/// ### Params
///
/// * `cutoff` - `-log10` q-value cutoff
///
/// ### Returns
///
/// The levels, increasing.
fn pscore_levels(cutoff: f64) -> Vec<f64> {
    let mut levels = vec![cutoff - CANDIDATE_MARGIN];
    let mut b = cutoff * LEVEL_RATIO;
    while b <= LEVEL_MAX {
        levels.push(b);
        b *= LEVEL_RATIO;
    }
    levels
}

/// Bases of one chromosome whose p-score is above each level, and all bases.
///
/// ### Params
///
/// * `tags` - Sorted tags
/// * `ext` - Extensions of both pileups
/// * `level_thresholds` - [`pscore_thresholds`] per level, levels increasing
///
/// ### Returns
///
/// `(bases above each level, all bases)`.
fn level_lengths(tags: &[u32], ext: &Extension, level_thresholds: &[Vec<u32>]) -> (Vec<i64>, i64) {
    let n_levels = level_thresholds.len();
    let (by_count, total) = windows(tags, ext, WINDOW_TAGS)
        .into_par_iter()
        .map(|w| {
            // bases by the number of levels their p-score clears
            let mut by_count = vec![0i64; n_levels + 1];
            let mut total = 0i64;
            for ((s, e, t, c), _) in window_segments(tags, ext, w) {
                let len = (e - s) as i64;
                total += len;
                let cleared = level_thresholds.partition_point(|th| is_candidate(th, t, c));
                by_count[cleared] += len;
            }
            (by_count, total)
        })
        .reduce(
            || (vec![0i64; n_levels + 1], 0),
            |(mut a, ta), (b, tb)| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                (a, ta + tb)
            },
        );
    // above level l: cleared more than l levels
    let mut above = vec![0i64; n_levels];
    let mut acc = 0;
    for l in (0..n_levels).rev() {
        acc += by_count[l + 1];
        above[l] = acc;
    }
    (above, total)
}

/// The highest level under which no key can pass.
///
/// A key with p-score `v <= B` has at least the bases above `B` ranked
/// before it, so its q-score is at most `B + log10((1 + above(B)) / N)`;
/// when that is under the cutoff, every key at or below `B` fails and only
/// keys above it need exact q-scores. Level 0 always qualifies, since a
/// q-score never exceeds its p-score.
///
/// ### Params
///
/// * `levels` - From [`pscore_levels`]
/// * `above` - Bases above each level
/// * `n_bases` - All bases
/// * `cutoff` - `-log10` q-value cutoff
///
/// ### Returns
///
/// The index of the level.
fn candidate_level(levels: &[f64], above: &[i64], n_bases: i64, cutoff: f64) -> usize {
    let n = (n_bases.max(1)) as f64;
    (1..levels.len())
        .rev()
        .find(|&l| levels[l] + ((1 + above[l]) as f64 / n).log10() <= cutoff - CANDIDATE_MARGIN)
        .unwrap_or(0)
}

/// Score a group from its candidate histogram.
///
/// q-scores are Benjamini-Hochberg over segments weighted by length: the
/// distinct p-scores in descending order, each `p + log10(rank) - log10(N)`
/// with `rank` the bases above it plus one and `N` all bases, kept monotone
/// and floored at 0. A q-score depends only on the p-scores above it, so the
/// candidates and `N` are all it needs. The arithmetic and its f32 roundings
/// are MACS's, which includes giving the lowest p-score of all a q-score of
/// 0 even when the loop runs to the end.
///
/// The q-scores fall with the p-score, so passing is a p-score threshold:
/// above the largest p-score that fails. The keys are dropped once their
/// p-scores are known; what remains is that threshold as treatment counts
/// per lambda count, and the q-scores of the passing p-scores for the
/// summits.
///
/// ### Params
///
/// * `hist` - The group's histogram, over all chromosomes
/// * `lambda` - The group's local lambda
/// * `c_max` - Largest lambda count
/// * `candidate_floor` - The p-score level the candidates are above
/// * `cutoff` - `-log10` q-value cutoff
///
/// ### Returns
///
/// The group's scores.
fn group_scores(
    hist: Histogram,
    lambda: Lambda,
    c_max: u32,
    candidate_floor: f64,
    cutoff: f64,
) -> GroupScores {
    let n_bases = hist.n_bases;
    let keys: Vec<(ScoreKey, i64)> = hist.candidates.into_iter().collect();
    let max_k = keys.iter().map(|&((k, _), _)| k).max().unwrap_or(0) as usize;
    let ln_fact = ln_factorials(max_k + 1);
    let mut pscores: Vec<(f32, i64)> = keys
        .into_par_iter()
        .map(|((t, bits), len)| {
            let p = -log10_poisson_upper(t, f32::from_bits(bits) as f64, &ln_fact);
            (p as f32, len)
        })
        .collect();

    // distinct p-scores, descending, with their summed lengths; equal
    // p-scores (0.0 and -0.0 included) merge as in MACS
    pscores.par_sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    let mut uniq: Vec<(f32, i64)> = Vec::new();
    for (p, len) in pscores {
        match uniq.last_mut() {
            Some(last) if qkey(last.0) == qkey(p) => last.1 += len,
            _ => uniq.push((p, len)),
        }
    }

    // q-scores down to the first that fails
    let mut passing: Vec<(f32, f32)> = Vec::new();
    let mut fail_floor = candidate_floor;
    if !uniq.is_empty() {
        let f = if n_bases > 0 {
            -(n_bases as f64).log10() as f32
        } else {
            0.0
        };
        let mut pre_q = if n_bases > 0 { Q_START } else { 0.0 };
        let mut k: i64 = 1;
        // MACS zeroes the lowest p-score of all; it is a candidate only when
        // every segment is one
        let complete = uniq.iter().map(|&(_, l)| l).sum::<i64>() == n_bases;
        let last = uniq.len() - 1;
        for (i, &(v, l)) in uniq.iter().enumerate() {
            let mut q = (v as f64 + ((k as f64).log10() + f as f64)) as f32;
            if q > pre_q {
                q = pre_q;
            }
            if q <= 0.0 || (complete && i == last) {
                q = 0.0;
            }
            if q as f64 <= cutoff {
                fail_floor = v as f64;
                break;
            }
            passing.push((v, q));
            pre_q = q;
            k += l;
        }
    }

    GroupScores {
        lambda,
        pass_thresholds: pscore_thresholds(lambda, c_max, fail_floor),
        passing,
        ln_fact,
    }
}

/////////////
// Reading //
/////////////

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

/// Plan the reads: chromosome batches under the memory budget when the file
/// has a tabix index, the whole file otherwise.
///
/// ### Params
///
/// * `path` - Fragment file
/// * `max_mem_gb` - Budget, `None` for one chromosome per batch
///
/// ### Returns
///
/// The reads and the chromosomes in file order (only the first one when
/// there is no index), or an I/O or index error.
fn plan_reads(path: &Path, max_mem_gb: Option<f64>) -> Result<(Reads, Vec<String>), AtacErrors> {
    let Some(spans) = chrom_spans(path)? else {
        let order = first_chromosome(path)?.into_iter().collect();
        return Ok((Reads::Whole, order));
    };
    let order = spans.iter().map(|s| s.name.clone()).collect();
    let cap = max_mem_gb.map_or(0.0, |gb| gb * 1e9 / MEM_PER_COMPRESSED_BYTE);
    let mut batches: Vec<Vec<ChromSpan>> = Vec::new();
    let mut size = 0.0;
    for span in spans {
        let bytes = span.compressed_bytes() as f64;
        match batches.last_mut() {
            Some(b) if size + bytes <= cap => b.push(span),
            _ => {
                size = 0.0;
                batches.push(vec![span]);
            }
        }
        size += bytes;
    }
    Ok((Reads::Batches(batches), order))
}

/// Read the tags of every group, plus their first fragments, from one batch
/// or the whole file.
///
/// ### Params
///
/// * `path` - Fragment file
/// * `batch` - Chromosomes to read, `None` for the whole file
/// * `index` - Cells
/// * `groups` - Group of each cell
/// * `n_groups` - Number of groups
/// * `source` - What becomes a tag
///
/// ### Returns
///
/// The tags and first fragments per group, or an I/O or parse error.
fn read_tags(
    path: &Path,
    batch: Option<&[ChromSpan]>,
    index: &BarcodeIndex,
    groups: &[u32],
    n_groups: usize,
    source: TagSource,
) -> Result<(BatchTags, FirstFragments), AtacErrors> {
    let init = || {
        (
            vec![FxHashMap::default(); n_groups],
            vec![FxHashMap::default(); n_groups],
        )
    };
    let fold = |acc: &mut (BatchTags, FirstFragments), chrom: &str, frags: &[Fragment]| {
        for f in frags {
            let g = groups[f.cell as usize] as usize;
            let tags = match acc.0[g].get_mut(chrom) {
                Some(t) => t,
                None => acc.0[g].entry(chrom.to_string()).or_default(),
            };
            tags.push(f.start);
            if source == TagSource::Insertion {
                tags.push(f.end);
            }
            let first = match acc.1[g].get_mut(chrom) {
                Some(fs) => fs,
                None => acc.1[g].entry(chrom.to_string()).or_default(),
            };
            // runs reach a worker in any order, so keep the smallest starts
            // seen, not the first ones
            first.push((f.start, f.len()));
            if first.len() > 2 * TAG_SIZE_SAMPLE {
                first.sort_unstable();
                first.truncate(TAG_SIZE_SAMPLE);
            }
        }
    };
    let reduce = |mut a: (BatchTags, FirstFragments), b: (BatchTags, FirstFragments)| {
        for (ta, tb) in a.0.iter_mut().zip(b.0) {
            for (chrom, mut t) in tb {
                let e = ta.entry(chrom).or_default();
                if e.len() < t.len() {
                    std::mem::swap(e, &mut t);
                }
                e.extend_from_slice(&t);
            }
        }
        for (fa, fb) in a.1.iter_mut().zip(b.1) {
            for (chrom, f) in fb {
                let e = fa.entry(chrom).or_default();
                e.extend(f);
                e.sort_unstable();
                e.truncate(TAG_SIZE_SAMPLE);
            }
        }
        a
    };
    match batch {
        Some(spans) => fold_fragment_spans(path, spans, index, init, fold, reduce),
        None => fold_fragments(path, index, init, fold, reduce),
    }
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

/// MACS's tag size of a fragment start group: the truncated mean length of
/// its first fragments.
///
/// MACS takes the first lines of the file it reads, which for a group is its
/// first fragments on the first chromosome, in file order, that it has any
/// on. Without an index only the file's first chromosome is known; past it
/// the fallback is name order.
///
/// ### Params
///
/// * `first` - The group's first fragments per chromosome
/// * `order` - Chromosomes in file order, as far as known
///
/// ### Returns
///
/// The tag size, 0 for an empty group.
fn fragment_tag_size(first: &FxHashMap<String, Vec<(u32, u32)>>, order: &[String]) -> u32 {
    let chrom = order
        .iter()
        .find(|c| first.contains_key(*c))
        .or_else(|| first.keys().min());
    let Some(chrom) = chrom else {
        return 0;
    };
    let mut frags = first[chrom].clone();
    frags.sort_unstable();
    frags.truncate(TAG_SIZE_SAMPLE);
    let sum: u64 = frags.iter().map(|&(_, l)| l as u64).sum();
    (sum as f64 / frags.len() as f64) as u32
}

///////////
// Spill //
///////////

/// Where one group's tags of one chromosome sit in the spill.
struct SpillEntry {
    /// Group.
    group: usize,
    /// Chromosome.
    chrom: String,
    /// Byte offset.
    offset: u64,
    /// Number of tags.
    len: usize,
}

/// Sorted, deduplicated tags kept on disk between passes, in native byte
/// order. The file is removed on drop.
struct TagSpill {
    /// The scratch file.
    path: PathBuf,
    /// Open for appending and mapping.
    file: File,
    /// Its contents, in write order.
    entries: Vec<SpillEntry>,
    /// Bytes written.
    size: u64,
}

impl TagSpill {
    /// Create an empty spill.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory of the scratch file
    ///
    /// ### Returns
    ///
    /// The spill, or an I/O error.
    fn create(dir: &Path) -> Result<Self, AtacErrors> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = dir.join(format!("bv_peak_tags_{}_{nanos}.bin", std::process::id()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            path,
            file,
            entries: Vec::new(),
            size: 0,
        })
    }

    /// Append one group's tags of one chromosome.
    ///
    /// ### Params
    ///
    /// * `group` - Group
    /// * `chrom` - Chromosome
    /// * `tags` - Sorted tags, not empty
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or an I/O error.
    fn push(&mut self, group: usize, chrom: String, tags: &[u32]) -> Result<(), AtacErrors> {
        // SAFETY: u32 has no padding, so its bytes are plain initialised
        // memory of the slice's length times 4.
        let bytes =
            unsafe { std::slice::from_raw_parts(tags.as_ptr().cast::<u8>(), tags.len() * 4) };
        self.file.write_all(bytes)?;
        self.entries.push(SpillEntry {
            group,
            chrom,
            offset: self.size,
            len: tags.len(),
        });
        self.size += bytes.len() as u64;
        Ok(())
    }

    /// Map one entry's tags.
    ///
    /// ### Params
    ///
    /// * `e` - The entry
    ///
    /// ### Returns
    ///
    /// The mapping, or an I/O error.
    fn map(&self, e: &SpillEntry) -> Result<Mmap, AtacErrors> {
        // SAFETY: the scratch file is private to this spill and not written
        // while mapped.
        let map = unsafe {
            MmapOptions::new()
                .offset(e.offset)
                .len(e.len * 4)
                .map(&self.file)?
        };
        Ok(map)
    }
}

impl Drop for TagSpill {
    /// Remove the scratch file.
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A mapped spill entry as tags.
///
/// ### Params
///
/// * `map` - From [`TagSpill::map`]
///
/// ### Returns
///
/// The tags.
fn mapped_tags(map: &Mmap) -> &[u32] {
    // SAFETY: every bit pattern is a valid u32, and the entries start at
    // multiples of 4 bytes in the file, so the mapping is 4-byte aligned;
    // align_to checks that.
    let (pre, tags, post) = unsafe { map.align_to::<u32>() };
    assert!(
        pre.is_empty() && post.is_empty(),
        "spill mapping is not aligned"
    );
    tags
}

/////////////
// Calling //
/////////////

/// Call the peaks of one chromosome of one group.
///
/// Windows find their segments above the cutoff in parallel; the regions are
/// then built in order over the concatenation, rejoining segments a window
/// cut in two.
///
/// ### Params
///
/// * `chrom` - Chromosome
/// * `tags` - Its sorted tags
/// * `ext` - Extensions of both pileups
/// * `scores` - The group's scores
/// * `min_length` - Shorter regions are dropped
/// * `max_gap` - Largest gap merged into one region
///
/// ### Returns
///
/// The peaks by position.
fn call_chromosome(
    chrom: &str,
    tags: &[u32],
    ext: &Extension,
    scores: &GroupScores,
    min_length: i64,
    max_gap: i64,
) -> Vec<NarrowPeak> {
    let above: Vec<Vec<(Segment, bool)>> = windows(tags, ext, WINDOW_TAGS)
        .into_par_iter()
        .map(|w| {
            window_segments(tags, ext, w)
                .filter(|&((_, _, t, c), _)| scores.passes(t, c))
                .collect()
        })
        .collect();

    let mut peaks = Vec::new();
    let mut region: Vec<Segment> = Vec::new();
    let mut close = |region: &[Segment]| {
        if let Some(p) = close_peak(chrom, region, min_length, scores) {
            peaks.push(p);
        }
    };
    for (seg, clipped) in above.into_iter().flatten() {
        if let Some(last) = region.last_mut() {
            if clipped && last.1 == seg.0 && (last.2, last.3) == (seg.2, seg.3) {
                last.1 = seg.1;
                continue;
            }
            if seg.0 as i64 - last.1 as i64 > max_gap {
                close(&region);
                region.clear();
            }
        }
        region.push(seg);
    }
    if !region.is_empty() {
        close(&region);
    }
    peaks
}

/// Close one region of segments above the cutoff into a peak.
///
/// The summit is the middle one, in order, of the segments holding the
/// highest treatment pileup.
///
/// ### Params
///
/// * `chrom` - Chromosome
/// * `region` - The region's segments, in order
/// * `min_length` - Shorter regions are dropped
/// * `scores` - The group's scores
///
/// ### Returns
///
/// The peak, or `None` for a region shorter than `min_length`.
fn close_peak(
    chrom: &str,
    region: &[Segment],
    min_length: i64,
    scores: &GroupScores,
) -> Option<NarrowPeak> {
    let start = region[0].0 as i64;
    let end = region[region.len() - 1].1 as i64;
    if end - start < min_length {
        return None;
    }

    let mut summit_value = 0f64;
    let mut first = 0;
    let mut n_tie: usize = 0;
    for (k, seg) in region.iter().enumerate() {
        let t = seg.2 as f64;
        if summit_value == 0.0 || summit_value < t {
            first = k;
            n_tie = 1;
            summit_value = t;
        } else if summit_value == t {
            n_tie += 1;
        }
    }
    let mid = n_tie.div_ceil(2) - 1;
    let &(s, e, t, c) = region[first..]
        .iter()
        .filter(|seg| seg.2 as f64 == summit_value)
        .nth(mid)
        .expect("the tie count covers the middle");

    let (pscore, qscore) = scores.score(t, c);
    let treat = t as f32;
    let ctrl = scores.lambda.value(c);
    Some(NarrowPeak {
        chrom: chrom.to_string(),
        start: start as u32,
        end: end as u32,
        summit: ((s as i64 + e as i64) / 2) as u32,
        pileup: treat,
        pscore,
        fold: ((treat as f64 + FOLD_PSEUDOCOUNT) / (ctrl as f64 + FOLD_PSEUDOCOUNT)) as f32,
        qscore,
    })
}

/// Call peaks per group of cells.
///
/// Pass 1 reads the tags batch by batch, sorts and deduplicates them, counts
/// them per group and spills them to disk. Pass 2 sums the segment lengths
/// per score from the spill, which fixes every group's q-scores. Pass 3
/// calls. Merge the groups' peaks with [`crate::regions::reduce_regions`]
/// for one peak set to count against.
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
/// The peaks of each group, indexed by group, chromosomes in name order and
/// peaks by position, or an I/O, parse or argument error.
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
    let index = BarcodeIndex::new(barcodes)?;
    let (reads, order) = plan_reads(path, params.max_mem_gb)?;
    let batches: Vec<Option<&[ChromSpan]>> = match &reads {
        Reads::Batches(b) => b.iter().map(|s| Some(s.as_slice())).collect(),
        Reads::Whole => vec![None],
    };
    let tmp_dir = params.tmp_dir.clone().unwrap_or_else(std::env::temp_dir);
    let mut spill = TagSpill::create(&tmp_dir)?;

    // pass 1: read, sort, deduplicate, count, spill
    let mut n_tags = vec![0usize; n_groups];
    let mut c_max = vec![0u32; n_groups];
    let mut firsts: FirstFragments = vec![FxHashMap::default(); n_groups];
    for &batch in &batches {
        let (tags, first) = read_tags(path, batch, &index, groups, n_groups, params.tag_source)?;
        for (g, f) in first.into_iter().enumerate() {
            firsts[g].extend(f);
        }
        let mut jobs: Vec<(usize, String, Vec<u32>)> = tags
            .into_iter()
            .enumerate()
            .flat_map(|(g, chroms)| chroms.into_iter().map(move |(c, t)| (g, c, t)))
            .collect();
        let window_max: Vec<u32> = jobs
            .par_iter_mut()
            .map(|(_, _, t)| {
                sort_and_dedup(t, params.keep_dup);
                max_window_count(t, params.llocal)
            })
            .collect();
        for ((g, chrom, t), w) in jobs.into_iter().zip(window_max) {
            if !t.is_empty() {
                n_tags[g] += t.len();
                c_max[g] = c_max[g].max(w);
                spill.push(g, chrom, &t)?;
            }
        }
    }
    spill.file.flush()?;
    // MACS takes Python's log(q, 10), not log10(q)
    let cutoff = -(params.q_value.ln() / 10f64.ln());

    let d = params.extsize as i64;
    let llocal = params.llocal as i64;
    let ext = Extension {
        treat_five: -(params.shift as i64),
        treat_three: params.shift as i64 + d,
        ctrl_five: llocal / 2,
        ctrl_three: llocal - llocal / 2,
    };
    let lambdas: Vec<Lambda> = n_tags
        .iter()
        .map(|&n| Lambda {
            scale: (params.extsize as f64 / params.llocal as f64) as f32,
            background: (params.extsize as f64 * n as f64 / params.genome_size) as f32,
        })
        .collect();
    // per group and p-score level, the treatment count thresholds
    let levels = pscore_levels(cutoff);
    let level_thresholds: Vec<Vec<Vec<u32>>> = lambdas
        .par_iter()
        .zip(&c_max)
        .map(|(&l, &c)| {
            levels
                .par_iter()
                .map(|&b| pscore_thresholds(l, c, b))
                .collect()
        })
        .collect();

    // pass 2a: bases above each level, to find the keys that can pass
    let mut above = vec![vec![0i64; levels.len()]; n_groups];
    let mut n_bases = vec![0i64; n_groups];
    for e in &spill.entries {
        let map = spill.map(e)?;
        let g = e.group;
        let (a, n) = level_lengths(mapped_tags(&map), &ext, &level_thresholds[g]);
        above[g].iter_mut().zip(a).for_each(|(x, y)| *x += y);
        n_bases[g] += n;
    }
    // per group, the level the candidates are above and its thresholds
    let (floors, thresholds): (Vec<f64>, Vec<Vec<u32>>) = level_thresholds
        .into_iter()
        .enumerate()
        .map(|(g, mut th)| {
            let l = candidate_level(&levels, &above[g], n_bases[g], cutoff);
            (levels[l], th.swap_remove(l))
        })
        .unzip();

    // pass 2b: length histograms of the candidates, then the scores
    let mut hists: Vec<Histogram> = (0..n_groups).map(|_| Histogram::default()).collect();
    for e in &spill.entries {
        let map = spill.map(e)?;
        let g = e.group;
        let h = length_histogram(mapped_tags(&map), &ext, lambdas[g], &thresholds[g]);
        hists[g] = std::mem::take(&mut hists[g]).merge(h);
    }
    drop(thresholds);
    // one group at a time, each scored in parallel, so one histogram is
    // turned into scores at once
    let scores: Vec<GroupScores> = hists
        .into_iter()
        .enumerate()
        .map(|(g, h)| group_scores(h, lambdas[g], c_max[g], floors[g], cutoff))
        .collect();

    let max_gaps: Vec<i64> = firsts
        .iter()
        .map(|f| match (params.max_gap, params.tag_source) {
            (Some(g), _) => g as i64,
            (None, TagSource::Insertion) => 1,
            (None, TagSource::FragmentStart) => fragment_tag_size(f, &order) as i64,
        })
        .collect();
    let min_length = params.min_length.unwrap_or(params.extsize) as i64;

    // pass 3: call
    let mut peaks: Vec<Vec<NarrowPeak>> = vec![Vec::new(); n_groups];
    for e in &spill.entries {
        let map = spill.map(e)?;
        let g = e.group;
        peaks[g].extend(call_chromosome(
            &e.chrom,
            mapped_tags(&map),
            &ext,
            &scores[g],
            min_length,
            max_gaps[g],
        ));
    }
    for p in &mut peaks {
        p.sort_unstable_by(|a, b| (&a.chrom, a.start).cmp(&(&b.chrom, b.start)));
    }
    Ok(peaks)
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
        let got: Vec<(i32, u32)> = PileupSweep::new(&[10, 15], 5, 5).collect();
        assert_eq!(got, vec![(5, 0), (10, 1), (15, 2), (20, 1)]);
    }

    #[test]
    fn test_pileup_clips_at_zero() {
        let got: Vec<(i32, u32)> = PileupSweep::new(&[2], 5, 5).collect();
        assert_eq!(got, vec![(7, 1)]);
    }

    #[test]
    fn test_segments_pair_both_pileups() {
        let ext = Extension {
            treat_five: 1,
            treat_three: 1,
            ctrl_five: 3,
            ctrl_three: 3,
        };
        // treatment [9, 11), lambda window [7, 13)
        let got: Vec<Segment> = Segments::new(&[10], &ext).collect();
        assert_eq!(got, vec![(0, 7, 0, 0), (7, 9, 0, 1), (9, 11, 1, 1)]);
    }

    /// Windowed sweeps, rejoined where a window cut a segment, give the full
    /// sweep's segments.
    #[test]
    fn test_windows_reproduce_the_full_sweep() {
        let ext = Extension {
            treat_five: 100,
            treat_three: 100,
            ctrl_five: 5000,
            ctrl_three: 5000,
        };
        // clustered tags with gaps wider than the lambda window, duplicates
        let mut x: u64 = 7;
        let mut tags: Vec<u32> = (0..30_000)
            .map(|i| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let cluster = (i / 500) as u32 * 40_000;
                cluster + ((x >> 33) % 3000) as u32
            })
            .collect();
        tags.sort_unstable();
        let ws = windows(&tags, &ext, 2000);
        assert!(ws.len() > 2);

        let full: Vec<Segment> = Segments::new(&tags, &ext).collect();
        let mut joined: Vec<Segment> = Vec::new();
        for w in ws {
            for (seg, clipped) in window_segments(&tags, &ext, w) {
                match joined.last_mut() {
                    Some(last)
                        if clipped && last.1 == seg.0 && (last.2, last.3) == (seg.2, seg.3) =>
                    {
                        last.1 = seg.1
                    }
                    _ => joined.push(seg),
                }
            }
        }
        assert_eq!(joined, full);
    }

    #[test]
    fn test_max_window_count() {
        assert_eq!(max_window_count(&[], 10), 0);
        // [5, 15) holds 5, 9, 10, 10
        assert_eq!(max_window_count(&[0, 5, 9, 10, 10, 30], 10), 4);
        assert_eq!(max_window_count(&[0, 5, 9, 10, 10, 30], 11), 5);
    }

    /// Each threshold is the exact boundary: one count under it the p-score
    /// is at or below the floor, at it above.
    #[test]
    fn test_pscore_thresholds_are_boundaries() {
        let lambda = Lambda {
            scale: 0.02,
            background: 0.7,
        };
        let ln_fact = ln_factorials(2000);
        let p = |t: u32, c: u32| {
            (-log10_poisson_upper(t, lambda.value(c) as f64, &ln_fact)) as f32 as f64
        };
        for floor in [1.3, 4.0, 50.0] {
            let th = pscore_thresholds(lambda, 3000, floor);
            for c in [0u32, 1, 34, 35, 36, 500, 3000] {
                let t = th[c as usize];
                assert!(p(t, c) > floor, "floor {floor}, c {c}");
                if t > 0 {
                    assert!(p(t - 1, c) <= floor, "floor {floor}, c {c}");
                }
            }
        }
    }

    #[test]
    fn test_candidate_level_bounds_the_q_score() {
        let levels = pscore_levels(1.30103);
        let n = 1_000_000_000i64;
        // nothing above any level: the bound is B - 9, so the highest level
        // within 9 of the cutoff qualifies
        let none = vec![0; levels.len()];
        let want = levels
            .iter()
            .rposition(|&b| b - 9.0 <= 1.30103 - CANDIDATE_MARGIN)
            .unwrap();
        assert!(want > 0);
        assert_eq!(candidate_level(&levels, &none, n, 1.30103), want);
        // everything above every level: only the floor qualifies
        let all = vec![n; levels.len()];
        assert_eq!(candidate_level(&levels, &all, n, 1.30103), 0);
    }

    #[test]
    fn test_lambda_floors_at_background() {
        let l = Lambda {
            scale: 0.02,
            background: 0.5,
        };
        assert_eq!(l.value(0), 0.5);
        assert_eq!(l.value(100), 2.0);
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
    fn test_spill_round_trips_tags() {
        let mut spill = TagSpill::create(&std::env::temp_dir()).unwrap();
        spill.push(0, "chr1".into(), &[1, 2, 3]).unwrap();
        spill.push(1, "chr2".into(), &[7, 9]).unwrap();
        spill.file.flush().unwrap();
        let path = spill.path.clone();
        let got: Vec<Vec<u32>> = spill
            .entries
            .iter()
            .map(|e| mapped_tags(&spill.map(e).unwrap()).to_vec())
            .collect();
        assert_eq!(got, vec![vec![1, 2, 3], vec![7, 9]]);
        drop(spill);
        assert!(!path.exists());
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
        let p = PeakCallParams {
            max_mem_gb: Some(0.0),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }
}
