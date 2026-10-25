"""Generate MetaCells2 parity fixtures by calling the reference implementation.

Inputs are integer counts and integer-derived similarities rebuilt from a
seeded LCG on both sides, so no float input data crosses as text. Only the
reference's outputs are emitted.

Run from the repo root:
  uv run --no-project --python 3.12 --with metacells==0.9.5 \
      python -I dev/gen_metacells2_fixtures.py \
      > crates/bixverse-sc/src/mc_generation/metacells2/parity_fixtures.rs
  rustfmt --edition 2024 crates/bixverse-sc/src/mc_generation/metacells2/parity_fixtures.rs
"""

import contextlib
import io
import logging
import math

import anndata as ad
import igraph as ig
import numpy as np
import scipy.sparse as sp

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import metacells as mc
    from metacells import extensions as xt
    from metacells.tools.candidates import _seeds_count_for

logging.disable(logging.CRITICAL)

LCG_SEED = 12345
N_CLUSTERS = 4
CELLS_PER_CLUSTER = 60
N_CELLS = N_CLUSTERS * CELLS_PER_CLUSTER
N_GENES = 400
MARKERS = 25
HOUSEKEEPING = (100, 150)
DEVIANT_CELLS = [5, 70, 130, 200]
DEVIANT_GENE = 300
DEVIANT_SPIKE = 400

out = []


def emit(line=""):
    out.append(line)


def f64(x):
    return f"{float(x):.17e}"


def f32(x):
    return f"{float(np.float32(x)):.9e}"


class Lcg:
    """Numerical Recipes LCG. Mirrored exactly on the Rust side."""

    def __init__(self, seed):
        self.x = seed

    def next(self):
        self.x = (1664525 * self.x + 1013904223) % (2**32)
        return self.x


# --------------------------------------------------------------- inputs
def make_counts():
    rng = Lcg(LCG_SEED)
    counts = np.zeros((N_CELLS, N_GENES), dtype=np.int64)
    for i in range(N_CELLS):
        c = i // CELLS_PER_CLUSTER
        for g in range(N_GENES):
            x = rng.next()
            hi = x >> 16
            mid = (x >> 8) & 0xFF
            if c * MARKERS <= g < (c + 1) * MARKERS:
                v = hi % 30 + 5 if mid % 4 != 0 else 0
            elif HOUSEKEEPING[0] <= g < HOUSEKEEPING[1]:
                v = hi % 50 + 20
            else:
                v = hi % 4 if mid % 3 == 0 else 0
            counts[i, g] = v
    for i in DEVIANT_CELLS:
        counts[i, DEVIANT_GENE] += DEVIANT_SPIKE
    return counts


def make_similarity(n, n_hubs, seed):
    """Symmetric similarity with distinct off-diagonal values. Hub rows are
    lifted so every other node prefers them, which starves their out-degree."""
    rng = Lcg(seed)
    s = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i):
            v = (rng.next() >> 8) / 16777216.0
            if i < n_hubs or j < n_hubs:
                v += 1.0
            s[i, j] = v
            s[j, i] = v
    np.fill_diagonal(s, 2.0)
    return s


def make_cut_graph(n, seed, split, cross_mod):
    """Symmetric weights; two dense blocks joined by weak edges on about one
    pair in `cross_mod`."""
    rng = Lcg(seed)
    w = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i):
            x = rng.next()
            same = (i < split) == (j < split)
            if same:
                v = ((x >> 16) % 100 + 1) / 128.0
            elif not same and (x >> 24) % cross_mod == 0:
                v = ((x >> 16) % 10 + 1) / 1024.0
            else:
                v = 0.0
            w[i, j] = v
            w[j, i] = v
    return w


counts = make_counts()
umis = counts.sum(axis=1)


def rust_u64_list(xs):
    return "&[" + ", ".join(str(int(x)) for x in xs) + "]"


def rust_usize_list(xs):
    return rust_u64_list(xs)


def rust_i32_list(xs):
    return "&[" + ", ".join(str(int(x)) for x in xs) + "]"


def rust_f32_list(xs):
    return "&[" + ", ".join(f32(x) for x in xs) + "]"


def rust_f64_list(xs):
    return "&[" + ", ".join(f64(x) for x in xs) + "]"


def bool_indices(mask):
    return rust_usize_list(np.where(np.asarray(mask))[0])


emit("#![allow(clippy::excessive_precision, clippy::unreadable_literal)]")
emit("//")
emit("// The reference emits full-precision literals. Trimming them by hand would")
emit("// mean editing generated values, so the lints are turned off for this file only.")
emit()
emit("//! MetaCells2 parity fixtures, generated from metacells 0.9.5.")
emit("//!")
emit("//! DO NOT EDIT. Regenerate with `dev/gen_metacells2_fixtures.py`.")
emit("//!")
emit("//! Counts: Numerical Recipes LCG, see `make_counts` there and in `parity.rs`.")
emit()
emit(f"pub const LCG_SEED: u64 = {LCG_SEED};")
emit(f"pub const CELLS_PER_CLUSTER: usize = {CELLS_PER_CLUSTER};")
emit(f"pub const N_CELLS: usize = {N_CELLS};")
emit(f"pub const N_GENES: usize = {N_GENES};")
emit(f"pub const MARKERS: usize = {MARKERS};")
emit(f"pub const HOUSEKEEPING: (usize, usize) = ({HOUSEKEEPING[0]}, {HOUSEKEEPING[1]});")
emit(f"pub const DEVIANT_CELLS: &[usize] = {rust_usize_list(DEVIANT_CELLS)};")
emit(f"pub const DEVIANT_GENE: usize = {DEVIANT_GENE};")
emit(f"pub const DEVIANT_SPIKE: u32 = {DEVIANT_SPIKE};")
emit()

# --------------------------------------------------------------- downsample
adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
mc.tl.downsample_cells(adata, random_seed=1)
emit("/// `downsample_samples` at default quantiles (750, 0.05, 0.5).")
emit(f"pub const DOWNSAMPLE_SAMPLES: u64 = {int(adata.uns['downsample_samples'])};")
emit()

# --------------------------------------------------------------- select
csc = sp.csc_matrix(counts.astype(np.float32))
rel_var = mc.ut.relative_variance_per(csc, per="column", window_size=100)
top3 = mc.ut.rank_per(csc, rank=N_CELLS - 3 - 1, per="column")
emit("/// `relative_variance_per(raw, per=column, window_size=100)`.")
emit(f"pub const RELATIVE_VARIANCE: &[f64] = {rust_f64_list(rel_var)};")
# numpy's argsort of the f32 means, as `sliding_window_function` takes it. Ties
# between equal means fall wherever numpy's unstable sort puts them, so the
# order is emitted rather than rebuilt.
mean_order = np.argsort(mc.ut.mean_per(csc, per="column"))
emit("/// Upstream's gene order by mean, as its sliding window sees it.")
emit(f"pub const MEAN_ORDER: &[usize] = {rust_usize_list(mean_order)};")
emit("/// `rank_per(raw, rank=n-4, per=column)`, upstream's top-3 statistic.")
emit(f"pub const TOP3: &[f32] = {rust_f32_list(top3)};")


def selected_mask(min_rel_var, min_genes):
    a = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
    a.var["lateral_gene"] = False
    sdata = mc.pl.extract_selected_data(
        a,
        downsample_min_cell_quantile=1.0,
        downsample_max_cell_quantile=1.0,
        min_gene_relative_variance=min_rel_var,
        min_genes=min_genes,
        random_seed=1,
        top_level=False,
    )
    return a.var["selected_gene"].values, sdata


sel_default, _ = selected_mask(0.1, 100)
sel_relaxed, _ = selected_mask(1.5, 120)
emit("/// Selected genes at defaults (rel var 0.1, min genes 100), downsampling a no-op.")
emit(f"pub const SELECTED_DEFAULT: &[usize] = {bool_indices(sel_default)};")
emit("/// Selected genes at rel var 1.5, min genes 120: exercises relaxation and bisection.")
emit(f"pub const SELECTED_RELAXED: &[usize] = {bool_indices(sel_relaxed)};")
emit()

# --------------------------------------------------------------- similarity
dense = counts[:, np.asarray(sel_default)].astype(np.float32)
logged = np.ascontiguousarray(np.log2(dense + np.float32(1.0 / 7.0)).astype(np.float32))
sim = np.abs(mc.ut.corrcoef(logged, per="row", reproducible=True))
emit("/// `abs(corrcoef(log2(selected + 1/7)))`: row sums (diagonal included).")
emit(f"pub const SIMILARITY_ROW_SUMS: &[f64] = {rust_f64_list(sim.astype(np.float64).sum(axis=1))};")
emit("/// The same, full rows 0, 61, 239.")
for r in (0, 61, 239):
    emit(f"pub const SIMILARITY_ROW_{r}: &[f32] = {rust_f32_list(sim[r])};")
emit()


# --------------------------------------------------------------- knn
def knn_case(name, n, n_hubs, seed, k):
    s = make_similarity(n, n_hubs, seed)
    a = ad.AnnData(np.zeros((n, 1), dtype=np.float32))
    a.obsp["obs_similarity"] = s
    mc.tl.compute_obs_obs_knn_graph(a, k=k)
    g = sp.csr_matrix(a.obsp["obs_outgoing_weights"])
    g.sort_indices()
    emit(f"/// kNN case `{name}`: n = {n}, hubs = {n_hubs}, LCG seed {seed}, k = {k}.")
    emit(f"pub const KNN_{name}_N: usize = {n};")
    emit(f"pub const KNN_{name}_HUBS: usize = {n_hubs};")
    emit(f"pub const KNN_{name}_SEED: u64 = {seed};")
    emit(f"pub const KNN_{name}_K: usize = {k};")
    emit(f"pub const KNN_{name}_INDPTR: &[usize] = {rust_usize_list(g.indptr)};")
    emit(f"pub const KNN_{name}_INDICES: &[usize] = {rust_usize_list(g.indices)};")
    emit(f"pub const KNN_{name}_DATA: &[f32] = {rust_f32_list(g.data)};")
    emit()


knn_case("PLAIN", 150, 0, 777, 12)
knn_case("HUBS", 150, 12, 778, 12)

# --------------------------------------------------------------- min cut
# igraph silently ignores capacities given as np.float32, which is what
# `mc.ut.min_cut` passes, so upstream's cut is by edge count while its strength
# uses the weights. The Rust port reproduces that. A sparse `cross_mod` leaves
# a bridge with fewer edges than the minimum degree, so the cut is a real split;
# a dense one leaves the lowest-degree node as the cut.
assert ig.Graph(n=2, edges=[(0, 1)]).mincut(capacity=[np.float32(0.5)]).value == 1.0
emit("/// `mc.ut.min_cut` as upstream runs it, cut by edge count:")
emit("/// (n, split, seed, cross_mod, small side size, strength or NaN for None).")
emit("pub const MIN_CUT_CASES: &[(usize, usize, u64, u64, usize, f64)] = &[")
for n, split, seed, cross_mod in [
    (20, 8, 101, 7),
    (40, 20, 102, 61),
    (33, 5, 103, 7),
    (12, 1, 104, 7),
    (30, 15, 105, 61),
    (24, 12, 106, 97),
]:
    w = make_cut_graph(n, seed, split, cross_mod)
    cut, strength = mc.ut.min_cut(w)
    small = min(len(cut.partition[0]), len(cut.partition[1]))
    st = "f64::NAN" if strength is None else f64(strength)
    emit(f"    ({n}, {split}, {seed}, {cross_mod}, {small}, {st}),")
emit("];")
emit()

# --------------------------------------------------------------- seeds count
emit("/// `_seeds_count_for`: (size, umis, min_size, max_size, min_umis, max_umis, result).")
emit("pub const SEEDS_COUNT_CASES: &[(usize, f64, usize, usize, f64, f64, usize)] = &[")
for size, tot, mn, mx, mnu, mxu in [
    (1000, 3.2e6, 24, 95, 80000, 319999),
    (240, 7.3e5, 24, 95, 80000, 319999),
    (500, 1.0e7, 24, 95, 80000, 319999),
    (5000, 1.0e6, 24, 95, 80000, 319999),
    (97, 2.0e5, 12, 47, 40000, 159999),
    (130, 4.1e5, 24, 95, 80000, 319999),
]:
    r = _seeds_count_for(
        total_size=size,
        total_umis=tot,
        min_metacell_size=mn,
        max_metacell_size=mx,
        min_metacell_umis=mnu,
        max_metacell_umis=mxu,
    )
    emit(f"    ({size}, {f64(tot)}, {mn}, {mx}, {f64(mnu)}, {f64(mxu)}, {r}),")
emit("];")
emit()

# --------------------------------------------------------------- deviants / dissolve
# Candidates: each planted cluster split in two halves of 30, plus a 10-cell
# remnant at the end of the last cluster that dissolve should fold away.
candidates = np.array([i // 30 for i in range(N_CELLS)], dtype=np.int32)
candidates[230:] = 8
a = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
deviants = mc.tl.find_deviant_cells(a, candidates=candidates)
deviants = np.asarray(deviants, dtype=bool)
emit("/// Candidates for deviants and dissolve: `i / 30`, cells 230.. set to 8.")
emit(f"pub const DEVIANTS: &[usize] = {bool_indices(deviants)};")
mc.tl.dissolve_metacells(a, candidates=candidates, deviants=deviants)
emit(f"pub const DISSOLVE_METACELL: &[i32] = {rust_i32_list(a.obs['metacell'].values)};")
emit(f"pub const DISSOLVED: &[usize] = {bool_indices(a.obs['dissolved'].values)};")
# Same with a small target, so the robust-size rule rather than min size decides.
a2 = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
mc.tl.dissolve_metacells(
    a2,
    candidates=candidates,
    deviants=deviants,
    target_metacell_size=60,
    target_metacell_umis=200000,
)
emit("/// Dissolve with target size 60 and target UMIs 200000.")
emit(f"pub const DISSOLVE_METACELL_BIG_TARGET: &[i32] = {rust_i32_list(a2.obs['metacell'].values)};")
emit()

print("\n".join(out))
