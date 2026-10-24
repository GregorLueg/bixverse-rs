"""Generate divide-and-conquer MetaCells2 reference statistics.

Divide and conquer is stochastic, so the reference is a band: summary
statistics of upstream's `compute_divide_and_conquer_metacells` over several
seeds on planted-cluster counts rebuilt from a seeded LCG on both sides.

Run from the repo root:
  uv run --no-project --python 3.12 --with metacells==0.9.5 \
      --with "pandas<2.3" python -I dev/gen_metacells2_dac_fixtures.py \
      > crates/bixverse-rs/tests/meta_cells2_dac_fixtures/mod.rs
"""

import contextlib
import io
import logging

import anndata as ad
import numpy as np
import scipy.sparse as sp

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import metacells as mc

logging.disable(logging.CRITICAL)

LCG_SEED = 4242
N_CLUSTERS = 8
CELLS_PER_CLUSTER = 400
N_CELLS = N_CLUSTERS * CELLS_PER_CLUSTER
MARKERS = 30
NOISE_GENES = 400
N_GENES = N_CLUSTERS * MARKERS + NOISE_GENES
PILE_SIZE = 800
SEEDS = [1, 2, 3, 4, 5]
PARAMS = dict(
    target_metacell_size=25,
    target_metacell_umis=25_000,
    min_metacell_size=8,
    min_knn_k=10,
    target_metacells_in_pile=20,
    min_target_pile_size=PILE_SIZE,
    max_target_pile_size=PILE_SIZE,
)


def lcg(seed):
    x = seed
    while True:
        x = (1664525 * x + 1013904223) % (2**32)
        yield x


def make_counts():
    """Markers: uniform 0..40 in their own cluster; elsewhere 1 on about a
    quarter of the entries."""
    rng = lcg(LCG_SEED)
    counts = np.zeros((N_CELLS, N_GENES), dtype=np.int64)
    for i in range(N_CELLS):
        c = i // CELLS_PER_CLUSTER
        for g in range(N_GENES):
            hi = next(rng) >> 16
            if c * MARKERS <= g < (c + 1) * MARKERS:
                counts[i, g] = hi % 41
            else:
                counts[i, g] = 1 if hi % 4 == 0 else 0
    return counts


def stats(metacell, truth):
    """(n_metacells, outliers, mixed metacells, misplaced cells)."""
    n = int(metacell.max()) + 1
    mixed = 0
    misplaced = 0
    for m in range(n):
        members = truth[metacell == m]
        comp = np.bincount(members, minlength=N_CLUSTERS)
        if np.count_nonzero(comp) > 1:
            mixed += 1
        misplaced += int(members.size - comp.max())
    return n, int(np.sum(metacell < 0)), mixed, misplaced


counts = make_counts()
truth = np.arange(N_CELLS) // CELLS_PER_CLUSTER
rows = []
for seed in SEEDS:
    adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
    adata.var["lateral_gene"] = False
    mc.pl.compute_divide_and_conquer_metacells(adata, random_seed=seed, **PARAMS)
    rows.append(stats(np.asarray(adata.obs["metacell"].values), truth))
rows = np.array(rows)

out = [
    "#![cfg_attr(not(feature = \"large-test\"), allow(dead_code))]",
    "",
    "//! Divide-and-conquer reference statistics from metacells 0.9.5.",
    "//!",
    "//! DO NOT EDIT. Regenerate with `dev/gen_metacells2_dac_fixtures.py`.",
    "",
    f"pub const LCG_SEED: u64 = {LCG_SEED};",
    f"pub const N_CLUSTERS: usize = {N_CLUSTERS};",
    f"pub const CELLS_PER_CLUSTER: usize = {CELLS_PER_CLUSTER};",
    f"pub const MARKERS: usize = {MARKERS};",
    f"pub const NOISE_GENES: usize = {NOISE_GENES};",
    f"pub const PILE_SIZE: usize = {PILE_SIZE};",
    f"/// Upstream seeds run: {SEEDS}.",
    "/// Per seed: (n_metacells, outlier cells, mixed metacells, misplaced cells).",
    "pub const UPSTREAM_STATS: &[(usize, usize, usize, usize)] = &[",
]
out += [f"    ({a}, {b}, {c}, {d})," for a, b, c, d in rows]
out.append("];")
print("\n".join(out))
