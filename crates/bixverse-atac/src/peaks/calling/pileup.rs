//! Treatment and lambda pileups swept off sorted tags, in parallel windows.

use super::*;

////////////
// Pileup //
////////////

/// The breakpoints of one pileup, as MACS's `pileup_from_PN_shifted` emits
/// them for plus strand tags extended to `[tag - five, tag + three)`, ends
/// clipped at 0, without storing them.
///
/// Each item is `(segment end, count over the segment)`; a leading uncovered
/// segment `[0, first)` comes out with count 0.
pub(super) struct PileupSweep<'a> {
    /// Sorted tags.
    pub(super) tags: &'a [u32],
    /// Distance to the start of an extension.
    pub(super) five: i64,
    /// Distance to the end of an extension.
    pub(super) three: i64,
    /// Next start to visit.
    pub(super) i_s: usize,
    /// Next end to visit.
    pub(super) i_e: usize,
    /// Extensions covering the current position.
    pub(super) count: u32,
    /// Last breakpoint emitted.
    pub(super) pre_p: i32,
    /// The leading breakpoint, until emitted.
    pub(super) lead: Option<i32>,
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
    pub(super) fn new(tags: &'a [u32], five: i64, three: i64) -> Self {
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
    pub(super) fn start(&self, i: usize) -> i32 {
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
    pub(super) fn end(&self, i: usize) -> i32 {
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
pub(super) struct Segments<'a> {
    /// Treatment breakpoints.
    pub(super) treat: Peekable<PileupSweep<'a>>,
    /// Local lambda breakpoints.
    pub(super) ctrl: Peekable<PileupSweep<'a>>,
    /// End of the previous segment.
    pub(super) pre: i32,
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
    pub(super) fn new(tags: &'a [u32], ext: &Extension) -> Self {
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
pub(super) fn windows(tags: &[u32], ext: &Extension, per_window: usize) -> Vec<Window> {
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
pub(super) fn window_segments<'a>(
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
pub(super) struct Histogram {
    /// Summed length per candidate score key.
    pub(super) candidates: FxHashMap<ScoreKey, i64>,
    /// Summed length of every segment.
    pub(super) n_bases: i64,
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
    pub(super) fn merge(mut self, mut other: Histogram) -> Histogram {
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
pub(super) fn is_candidate(thresholds: &[u32], t: u32, c: u32) -> bool {
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
pub(super) fn length_histogram(
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
pub(super) fn max_window_count(tags: &[u32], width: u32) -> u32 {
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
