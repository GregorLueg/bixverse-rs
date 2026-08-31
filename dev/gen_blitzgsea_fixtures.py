"""Generate blitzGSEA parity fixtures by calling the reference implementation.

Everything the Rust side needs to rebuild an input is either an integer or a
value formed by exactly-representable f64 arithmetic, so no float data has to
cross as text. Only the reference's *outputs* are emitted.

Run from the repo root:
  uv run --python 3.11 --with 'numpy<2' --with pandas --with scipy \
      --with statsmodels --with mpmath --with matplotlib --with tqdm \
      python dev/gen_blitzgsea_fixtures.py > tests/blitzgsea_fixtures/mod.rs
  rustfmt --edition 2024 tests/blitzgsea_fixtures/mod.rs

The rustfmt pass only rewraps the index arrays; the values are identical
with or without it.
"""

import sys

sys.path.insert(0, "/Users/gregorlueg/repos/others/blitzgsea")

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
import blitzgsea as blitz

N_GENES = 2000
LCG_SEED = 12345

out = []


def emit(line=""):
    out.append(line)


def f(x):
    """Full-precision f64 literal."""
    return f"{float(x):.17e}"


# --------------------------------------------------------------- inputs
def lcg(n, seed=LCG_SEED):
    """Numerical Recipes LCG. Mirrored exactly on the Rust side."""
    x = seed
    for _ in range(n):
        x = (1664525 * x + 1013904223) % (2**32)
        yield x


def raw_signature(n):
    # x / 2**32 is exact (power of two), then one rounding each for * and -
    return [x / 4294967296.0 * 6.0 - 3.0 for x in lcg(n)]


def positive_sample(n):
    # integer / 100.0: a single rounding, identical in both languages
    return [(x % 10000 + 1) / 100.0 for x in lcg(n)]


# blitzgsea's own preprocessing, lifted from `gsea()`
sig = pd.DataFrame({"i": [f"g{j}" for j in range(N_GENES)], "v": raw_signature(N_GENES)})
sig = sig.sort_values("v", ascending=False).set_index("i")
sig = sig[~sig.index.duplicated(keep="first")]
sig.loc[:, "v"] -= np.mean(sig.loc[:, "v"])
abs_sig = np.array(np.abs(sig.loc[:, "v"]))
smap = {h: i for i, h in enumerate(sig.index)}

# gene sets as index positions into the sorted signature
GENE_SETS = {
    "top_50": list(range(50)),
    "bottom_50": list(range(N_GENES - 50, N_GENES)),
    "strided_50": [i * 40 for i in range(50)],
    "small_7": [3, 17, 40, 91, 220, 501, 1300],
    "head_heavy_30": [i * 2 for i in range(20)] + [1000 + i * 30 for i in range(10)],
}

emit("#![allow(clippy::excessive_precision)]")
emit("//")
emit("// The reference emits seventeen significant digits. Trimming them by hand would")
emit("// mean editing generated values, so the lint is turned off for this file only.")
emit()
emit("//! blitzGSEA parity fixtures, generated from the reference implementation.")
emit("//!")
emit("//! DO NOT EDIT. Regenerate with")
emit("//! `dev/gen_blitzgsea_fixtures.py` against")
emit("//! <https://github.com/MaayanLab/blitzgsea> and paste the output here.")
emit("//!")
emit(f"//! Signature: {N_GENES} genes from a Numerical Recipes LCG seeded at")
emit(f"//! {LCG_SEED}, mapped by `x / 2^32 * 6 - 3`, sorted descending, mean")
emit("//! centred. Every step is exactly representable, so the Rust side rebuilds")
emit("//! the identical input without any float data crossing as text.")
emit()
emit(f"/// Number of genes in the fixture signature.")
emit(f"pub const N_GENES: usize = {N_GENES};")
emit()
emit("/// Seed for the fixture LCG.")
emit(f"pub const LCG_SEED: u64 = {LCG_SEED};")
emit()

# ------------------------------------------------- enrichment scores
emit("/////////////////////////")
emit("// Enrichment scores //")
emit("/////////////////////////")
emit()
emit("/// One gene set and what the reference computes for it.")
emit("pub struct EsFixture {")
emit("    /// Name of the gene set, for assertion messages")
emit("    pub name: &'static str,")
emit("    /// Index positions into the sorted, centred signature")
emit("    pub indices: &'static [i32],")
emit("    /// `blitzgsea.enrichment_score` on those indices")
emit("    pub es: f64,")
emit("    /// Index positions of the leading edge genes, ascending")
emit("    pub leading_edge: &'static [i32],")
emit("}")
emit()
emit("/// Gene sets covering an enriched, a depleted and two mixed case.")
emit("pub const ES_FIXTURES: &[EsFixture] = &[")

es_by_name = {}
for name, idx in GENE_SETS.items():
    genes = [sig.index[i] for i in idx]
    running, es = blitz.enrichment_score(abs_sig, smap, genes)
    leading = blitz.get_leading_edge(running, sig, genes, smap)
    leading_idx = sorted(smap[g] for g in leading.split(",") if g)
    es_by_name[name] = float(es)

    emit("    EsFixture {")
    emit(f'        name: "{name}",')
    emit(f"        indices: &{idx},".replace("[", "[").replace("]", "]"))
    emit(f"        es: {f(es)},")
    emit(f"        leading_edge: &{leading_idx},")
    emit("    },")
emit("];")
emit()

# --------------------------------------------------- gamma MLE fit
emit("///////////////////")
emit("// Gamma fitting //")
emit("///////////////////")
emit()
emit("/// Sample size for the gamma fitting fixture.")
GAMMA_N = 500
emit(f"pub const GAMMA_FIT_N: usize = {GAMMA_N};")
emit()
sample = positive_sample(GAMMA_N)
a_fit, loc_fit, b_fit = gamma.fit(sample, floc=0)
emit("/// `scipy.stats.gamma.fit(x, floc=0)` shape on the fixture sample.")
emit(f"pub const GAMMA_FIT_SHAPE: f64 = {f(a_fit)};")
emit()
emit("/// `scipy.stats.gamma.fit(x, floc=0)` scale on the fixture sample.")
emit(f"pub const GAMMA_FIT_SCALE: f64 = {f(b_fit)};")
emit()

# --------------------------------------------- p-value and NES map
# The block below is lifted verbatim from `blitzgsea.gsea()` so the fixture
# records what the reference actually computes, quirks included.
def reference_pval_nes(es, pos_alpha, pos_beta, pos_ratio, neg_alpha, neg_beta):
    if es > 0:
        prob = gamma.cdf(es, float(pos_alpha), scale=float(pos_beta))
        prob_two_tailed = np.min([0.5, (1 - np.min([prob * pos_ratio + 1 - pos_ratio, 1]))])
        nes = blitz.invcdf(1 - np.min([1, prob_two_tailed]))
        pval = 2 * prob_two_tailed
    else:
        prob = gamma.cdf(-es, float(neg_alpha), scale=float(neg_beta))
        prob_two_tailed = np.min([0.5, (1 - np.min([(((prob) - (prob * pos_ratio)) + pos_ratio), 1]))])
        if prob_two_tailed == 0.5:
            prob_two_tailed = prob_two_tailed - prob
        nes = blitz.invcdf(np.min([1, prob_two_tailed]))
        pval = 2 * prob_two_tailed
    return float(pval), -float(nes)


TAIL = dict(pos_alpha=4.0, pos_beta=0.05, pos_ratio=0.5, neg_alpha=4.0, neg_beta=0.05)

emit("////////////////////////")
emit("// p-value and NES //")
emit("////////////////////////")
emit()
emit("/// Gamma tail parameters used by [`PVAL_FIXTURES`] and [`ES_FIXTURES`].")
emit("pub struct TailFixture {")
emit("    /// Positive-tail shape")
emit("    pub shape_pos: f64,")
emit("    /// Positive-tail scale")
emit("    pub scale_pos: f64,")
emit("    /// Negative-tail shape")
emit("    pub shape_neg: f64,")
emit("    /// Negative-tail scale")
emit("    pub scale_neg: f64,")
emit("    /// Fraction of the null mass above zero")
emit("    pub pos_ratio: f64,")
emit("}")
emit()
emit("/// The tail the p-value fixtures were generated against.")
emit("pub const FIXTURE_TAIL: TailFixture = TailFixture {")
emit(f"    shape_pos: {f(TAIL['pos_alpha'])},")
emit(f"    scale_pos: {f(TAIL['pos_beta'])},")
emit(f"    shape_neg: {f(TAIL['neg_alpha'])},")
emit(f"    scale_neg: {f(TAIL['neg_beta'])},")
emit(f"    pos_ratio: {f(TAIL['pos_ratio'])},")
emit("};")
emit()
emit("/// One enrichment score and the p-value and NES the reference maps it to.")
emit("pub struct PvalFixture {")
emit("    /// The enrichment score")
emit("    pub es: f64,")
emit("    /// Two-sided p-value from `blitzgsea.gsea`")
emit("    pub pval: f64,")
emit("    /// Normalised enrichment score as the reference reports it")
emit("    pub nes: f64,")
emit("}")
emit()
emit("/// A sweep across both signs and out into the tail scipy can still resolve.")
emit("pub const PVAL_FIXTURES: &[PvalFixture] = &[")
for es in [-0.9, -0.7, -0.5, -0.35, -0.2, -0.1, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]:
    pval, nes = reference_pval_nes(es, **TAIL)
    emit("    PvalFixture {")
    emit(f"        es: {f(es)},")
    emit(f"        pval: {f(pval)},")
    emit(f"        nes: {f(nes)},")
    emit("    },")
emit("];")
emit()
emit("/// The p-value and NES for each [`ES_FIXTURES`] entry under")
emit("/// [`FIXTURE_TAIL`], so the whole scoring path can be checked end to end.")
emit("pub const ES_FIXTURE_PVALS: &[PvalFixture] = &[")
for name in GENE_SETS:
    pval, nes = reference_pval_nes(es_by_name[name], **TAIL)
    emit("    PvalFixture {")
    emit(f"        es: {f(es_by_name[name])},")
    emit(f"        pval: {f(pval)},")
    emit(f"        nes: {f(nes)},")
    emit("    },")
emit("];")
emit()

# ------------------------------------------------ anchor calibration
# Statistical, not exact: the reference seeds Python's `random` but samples
# through `numpy.random`, so its per-anchor draws are not seed controlled.
emit("/////////////////////////")
emit("// Anchor calibration //")
emit("/////////////////////////")
emit()
emit("/// Gamma parameters the reference fits at one anchor size.")
emit("///")
emit("/// Statistical, not exact. The reference seeds Python's `random` but draws")
emit("/// through `numpy.random`, so its per-anchor samples are not seed")
emit("/// controlled and no bit-exact comparison is possible. These come from")
emit(f"/// {20000} permutations, which pins each parameter to well inside the")
emit("/// tolerance the Rust side asserts.")
emit("pub struct AnchorFixture {")
emit("    /// Gene set size the anchor was fitted at")
emit("    pub size: usize,")
emit("    /// Shape of the pooled gamma over the absolute null scores")
emit("    pub shape: f64,")
emit("    /// Scale of the pooled gamma over the absolute null scores")
emit("    pub scale: f64,")
emit("    /// Fraction of non-zero null scores that were positive")
emit("    pub pos_ratio: f64,")
emit("}")
emit()
emit("/// Anchors spanning the range where the gamma parameters move fastest.")
emit("pub const ANCHOR_FIXTURES: &[AnchorFixture] = &[")
ANCHOR_PERMS = 20000
for size in [5, 20, 100, 500]:
    np.random.seed(7)
    scores = np.array(blitz.get_peak_size_adv(abs_sig, size, ANCHOR_PERMS, 7))
    aes = np.abs(scores)[scores != 0]
    a, _, b = gamma.fit(aes, floc=0)
    n_pos = int((scores > 0).sum())
    n_neg = int((scores < 0).sum())
    emit("    AnchorFixture {")
    emit(f"        size: {size},")
    emit(f"        shape: {f(a)},")
    emit(f"        scale: {f(b)},")
    emit(f"        pos_ratio: {f(n_pos / (n_pos + n_neg))},")
    emit("    },")
emit("];")

print("\n".join(out))
