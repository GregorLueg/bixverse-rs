"""Generate rare-population recovery statistics for divide-and-conquer MetaCells2.

Planted clusters plus two small populations that sit on a host cluster and
carry their own exclusive markers, rebuilt from a seeded LCG on both sides.
Upstream's `divide_and_conquer_pipeline` (which detects rare gene modules
first) is run over several seeds and the number of each population's cells
that land in metacells where that population is the majority is emitted.

Run from the repo root:
  uv run --no-project --python 3.12 --with metacells==0.9.5 \
      --with "pandas<2.3" python -I dev/gen_metacells2_rare_dac_fixtures.py \
      > crates/bixverse-rs/tests/meta_cells2_rare_dac_fixtures/mod.rs
"""

import contextlib
import io
import logging
import sys
import warnings

import anndata as ad
import numpy as np
import scipy.sparse as sp

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import metacells as mc

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

LCG_SEED = 9191
N_CLUSTERS = 8
CELLS_PER_CLUSTER = 400
MARKERS = 30
NOISE_GENES = 400
RARE_SIZES = [16, 48]
RARE_HOSTS = [0, 3]
RARE_MARKERS = 12
N_COMMON = N_CLUSTERS * CELLS_PER_CLUSTER
N_CELLS = N_COMMON + sum(RARE_SIZES)
N_GENES = N_CLUSTERS * MARKERS + NOISE_GENES + len(RARE_SIZES) * RARE_MARKERS
PILE_SIZE = 800
SEEDS = [1, 2, 3]
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


def population(i):
    """Planted cluster of a cell, then its rare population or -1."""
    if i < N_COMMON:
        return i // CELLS_PER_CLUSTER, -1
    r = 0
    start = N_COMMON
    while i >= start + RARE_SIZES[r]:
        start += RARE_SIZES[r]
        r += 1
    return RARE_HOSTS[r], r


def make_counts():
    """Host markers uniform 0..40; rare markers uniform 0..10 in their own
    population and zero elsewhere; everything else 1 on about a quarter."""
    rng = lcg(LCG_SEED)
    rare_start = N_CLUSTERS * MARKERS + NOISE_GENES
    counts = np.zeros((N_CELLS, N_GENES), dtype=np.int64)
    for i in range(N_CELLS):
        c, r = population(i)
        for g in range(N_GENES):
            hi = next(rng) >> 16
            if c * MARKERS <= g < (c + 1) * MARKERS:
                counts[i, g] = hi % 41
            elif g >= rare_start:
                own = rare_start + r * RARE_MARKERS <= g < rare_start + (r + 1) * RARE_MARKERS
                counts[i, g] = hi % 11 if r >= 0 and own else 0
            else:
                counts[i, g] = 1 if hi % 4 == 0 else 0
    return counts


def recovered(metacell):
    """Per rare population, its cells in metacells it is the majority of."""
    truth = np.array([population(i)[1] for i in range(N_CELLS)])
    out = []
    for r in range(len(RARE_SIZES)):
        n = 0
        for i in np.where(truth == r)[0]:
            m = metacell[i]
            if m < 0:
                continue
            members = truth[metacell == m]
            if np.sum(members == r) * 2 > members.size:
                n += 1
        out.append(n)
    return out


counts = make_counts()
rows = []
for seed in SEEDS:
    adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
    adata.var["lateral_gene"] = False
    mc.pl.divide_and_conquer_pipeline(
        adata, rare_max_gene_cell_fraction=0.02, random_seed=seed, **PARAMS
    )
    metacell = np.asarray(adata.obs["metacell"].values)
    modules = np.asarray(adata.obs["cells_rare_gene_module"].values)
    rows.append(recovered(metacell))
    print(f"seed {seed}: recovered {rows[-1]}, rare module cells {int(np.sum(modules >= 0))}", file=sys.stderr)

out = [
    "#![cfg_attr(not(feature = \"large-test\"), allow(dead_code))]",
    "",
    "//! Rare-population recovery of divide-and-conquer MetaCells2, metacells 0.9.5.",
    "//!",
    "//! DO NOT EDIT. Regenerate with `dev/gen_metacells2_rare_dac_fixtures.py`.",
    "",
    f"pub const LCG_SEED: u64 = {LCG_SEED};",
    f"pub const N_CLUSTERS: usize = {N_CLUSTERS};",
    f"pub const CELLS_PER_CLUSTER: usize = {CELLS_PER_CLUSTER};",
    f"pub const MARKERS: usize = {MARKERS};",
    f"pub const NOISE_GENES: usize = {NOISE_GENES};",
    f"pub const RARE_SIZES: [usize; {len(RARE_SIZES)}] = {RARE_SIZES};",
    f"pub const RARE_HOSTS: [usize; {len(RARE_HOSTS)}] = {RARE_HOSTS};",
    f"pub const RARE_MARKERS: usize = {RARE_MARKERS};",
    f"pub const PILE_SIZE: usize = {PILE_SIZE};",
    f"/// Upstream seeds run: {SEEDS}.",
    "/// Per seed: cells of each rare population in metacells it is the majority of.",
    f"pub const UPSTREAM_RECOVERED: &[[usize; {len(RARE_SIZES)}]] = &[",
]
out += [f"    [{', '.join(str(v) for v in r)}]," for r in rows]
out.append("];")
print("\n".join(out))
