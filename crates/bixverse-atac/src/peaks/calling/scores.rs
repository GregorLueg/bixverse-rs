//! Poisson p-scores, q-scores and the candidate bounds that keep them exact.

use super::*;

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
pub(super) fn round5(x: f64) -> f64 {
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
pub(super) fn log10_poisson_upper(k: u32, lambda: f64, ln_fact: &[f64]) -> f64 {
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
pub(super) fn qkey(p: f32) -> u32 {
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
pub(super) fn ln_factorials(n: usize) -> Vec<f64> {
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
pub(super) fn pscore_thresholds(lambda: Lambda, c_max: u32, floor: f64) -> Vec<u32> {
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
pub(super) fn pscore_levels(cutoff: f64) -> Vec<f64> {
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
pub(super) fn level_lengths(
    tags: &[u32],
    ext: &Extension,
    level_thresholds: &[Vec<u32>],
) -> (Vec<i64>, i64) {
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
pub(super) fn candidate_level(levels: &[f64], above: &[i64], n_bases: i64, cutoff: f64) -> usize {
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
pub(super) fn group_scores(
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
