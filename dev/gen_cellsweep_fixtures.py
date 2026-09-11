"""Generate CellSweep parity fixtures by calling the reference implementation.

The count matrix is rebuilt from a Numerical Recipes LCG on both sides, so only
integers describe the input and no float data has to cross as text. Only the
reference's outputs are emitted.

Two things constrain how this has to be run:

* `threads=1`. The reference reduces `p_numer_tls` over numba `prange` threads
  in f32, so a multi-threaded run is not reproducible against itself.
* `round_X=False`. The integerisation draws from numpy's generator, which
  cannot be mirrored in Rust, so parity is on the float denoised matrix and the
  rounding is tested separately for its expectation.

Emitted instead of the full nnz array: the fitted parameters in full, plus
denoised row and column sums. That pins the matrix without a 400 KB source file.

Run from the repo root:
  uv run --python 3.11 --with 'numpy<2' --with pandas --with scipy \
      --with anndata --with numba --with 'pydantic>=2.5,<3.0' \
      python dev/gen_cellsweep_fixtures.py > tests/cellsweep_fixtures/mod.rs
  rustfmt --edition 2024 tests/cellsweep_fixtures/mod.rs
"""

import sys

sys.path.insert(0, "/Users/gregor/repos/others/cellsweep")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellsweep import denoise_count_matrix

N_REAL = 200
N_EMPTY = 120
N_GENES = 60
N_CELLTYPES = 4
LCG_SEED = 20260907

# Weight of the first cell type's marker block in the ambient profile.
AMBIENT_SKEW = 8

# Contamination budget of a real barcode, drawn uniformly from
# [MIN_BUDGET, MIN_BUDGET + BUDGET_SPREAD).
MIN_BUDGET = 20
BUDGET_SPREAD = 300

# Total counts of an empty droplet, drawn the same way.
MIN_EMPTY_TOTAL = 200
EMPTY_SPREAD = 200

out = []


def emit(line=""):
    out.append(line)


def f(x):
    """Full-precision f64 literal."""
    return f"{float(x):.17e}"


# --------------------------------------------------------------- inputs
def lcg(seed=LCG_SEED):
    """Numerical Recipes LCG. Mirrored exactly on the Rust side."""
    x = seed
    while True:
        x = (1664525 * x + 1013904223) % (2**32)
        yield x


def ambient_shape():
    """Ambient profile, weighted towards the first cell type's marker block.

    A flat ambient profile is not identifiable: adding a constant to every
    `p_k` explains it exactly as well, and the reference duly collapses `alpha`
    to zero. Skewing the profile towards one block is what real emulsions look
    like, since the ambient soup is dominated by the most abundant cell type.

    ### Returns

    Per-gene ambient weights, summing to one.
    """
    block = N_GENES // N_CELLTYPES
    weights = np.array(
        [AMBIENT_SKEW if g < block else 1 for g in range(N_GENES)], dtype=float
    )
    return weights / weights.sum()


def build_counts():
    """Integer count matrix with cell-type marker blocks plus ambient.

    Every cell type owns a contiguous block of `N_GENES // N_CELLTYPES` genes
    and gets a high count there. On top of that each real barcode receives a
    contamination budget spread over the ambient profile.

    The budget varies widely from barcode to barcode, and that is the second
    thing the model needs to identify `alpha` at all: a shared `p_k` can absorb
    a constant amount of contamination but not a per-barcode varying one, so
    the per-cell `alpha` is the only parameter that can explain the spread.

    All values are small integers, so the Rust side rebuilds the matrix
    bit-for-bit from the same LCG.
    """
    rng = lcg()
    block = N_GENES // N_CELLTYPES
    ambient = ambient_shape()

    counts = np.zeros((N_REAL + N_EMPTY, N_GENES), dtype=np.int64)
    labels = []

    for i in range(N_REAL):
        k = i % N_CELLTYPES
        labels.append(k)
        budget = MIN_BUDGET + (next(rng) % BUDGET_SPREAD)
        for g in range(N_GENES):
            r = next(rng) % 8
            own = 30 + r if k * block <= g < (k + 1) * block else 0
            counts[i, g] = own + int(round(budget * ambient[g]))

    for i in range(N_REAL, N_REAL + N_EMPTY):
        total = MIN_EMPTY_TOTAL + (next(rng) % EMPTY_SPREAD)
        for g in range(N_GENES):
            counts[i, g] = int(round(total * ambient[g]))

    return counts, labels


counts, labels = build_counts()
celltype_names = [f"ct{k}" for k in range(N_CELLTYPES)]

obs = pd.DataFrame(
    {
        "celltype": [celltype_names[k] for k in labels] + ["ct0"] * N_EMPTY,
        "is_empty": [False] * N_REAL + [True] * N_EMPTY,
    },
    index=[f"cell{i}" for i in range(N_REAL + N_EMPTY)],
)
var = pd.DataFrame(index=[f"gene{g}" for g in range(N_GENES)])


# --------------------------------------------------------------- emit
def f64_array(name, values, doc):
    emit(f"    /// {doc}")
    emit(f"    pub const {name}: [f64; {len(values)}] = [")
    for value in values:
        emit(f"        {f(value)},")
    emit("    ];")
    emit()


def usize_array(name, values, doc):
    emit(f"    /// {doc}")
    emit(f"    pub const {name}: [usize; {len(values)}] = [")
    emit("        " + ", ".join(str(int(v)) for v in values) + ",")
    emit("    ];")
    emit()


emit("#![allow(clippy::excessive_precision)]")
emit("#![allow(dead_code)]")
emit("//")
emit("// The reference emits seventeen significant digits. Trimming them by hand")
emit("// would mean editing generated values, so the lint is turned off for this")
emit("// file only. `dead_code` likewise: not every test uses every array.")
emit()
emit("//! CellSweep parity fixtures. Generated by `dev/gen_cellsweep_fixtures.py`.")
emit("//!")
emit("//! Do not edit by hand. Regenerate against the reference implementation")
emit("//! instead; see the generator's module docstring for the exact command.")
emit("//!")
emit(f"//! Reference run: {N_REAL} real barcodes, {N_EMPTY} empty droplets,")
emit(f"//! {N_GENES} genes, {N_CELLTYPES} cell types, single-threaded,")
emit("//! `round_X = False`.")
emit("//!")
emit("//! Two configurations are emitted. At the reference's default stopping")
emit("//! rule the parameters are still drifting when it fires, so `beta` in")
emit("//! particular is nowhere near its fixed point and the two")
emit("//! implementations only sample the same trajectory at slightly")
emit("//! different places. The tight configuration drives both to the actual")
emit("//! fixed point and is the real parity gate.")
emit()
emit("/// Real barcodes in the fixture.")
emit(f"pub const N_REAL: usize = {N_REAL};")
emit()
emit("/// Empty droplets in the fixture.")
emit(f"pub const N_EMPTY: usize = {N_EMPTY};")
emit()
emit("/// Genes in the fixture.")
emit(f"pub const N_GENES: usize = {N_GENES};")
emit()
emit("/// Cell types in the fixture.")
emit(f"pub const N_CELLTYPES: usize = {N_CELLTYPES};")
emit()
emit("/// Seed of the LCG that rebuilds the count matrix.")
emit(f"pub const LCG_SEED: u32 = {LCG_SEED};")
emit()
emit("/// Weight of the first marker block in the fixture's ambient profile.")
emit(f"pub const AMBIENT_SKEW: f64 = {AMBIENT_SKEW}.0;")
emit()

def usize_array_top(name, values, doc):
    """Emit a `usize` array at module scope rather than inside a config module."""
    emit(f"/// {doc}")
    emit(f"pub const {name}: [usize; {len(values)}] = [")
    emit("    " + ", ".join(str(int(v)) for v in values) + ",")
    emit("];")
    emit()


raw_real = sp.csr_matrix(counts[:N_REAL].astype(np.float64))
usize_array_top(
    "RAW_ROW_SUMS",
    np.asarray(raw_real.sum(axis=1)).ravel(),
    "Raw library size per real barcode. Guards the rebuilt input matrix.",
)
usize_array_top(
    "RAW_COL_SUMS",
    np.asarray(raw_real.sum(axis=0)).ravel(),
    "Raw total per gene over the real barcodes. Guards the rebuilt input.",
)

CONFIGS = [
    (
        "default_tol",
        "The reference's default stopping rule.",
        dict(max_iter=500),
    ),
    (
        "tight_tol",
        "Driven to the fixed point, so both implementations can be held to a "
        "much tighter tolerance.",
        dict(
            max_iter=3000,
            del0_ll_tol=1e-8,
            min_ll_tol=1e-12,
            tol_p=1e-8,
            tol_f=1e-8,
        ),
    ),
]

for module, doc, kwargs in CONFIGS:
    adata = ad.AnnData(
        X=sp.csr_matrix(counts.astype(np.float32)), obs=obs.copy(), var=var.copy()
    )
    result = denoise_count_matrix(
        adata,
        round_X=False,
        threads=1,
        freeze_empties=True,
        freeze_ambient_profile=True,
        empty_droplet_method="threshold",
        umi_cutoff=1,
        inplace=False,
        quiet=True,
        **kwargs,
    )

    real = result.X.tocsr()[:N_REAL]
    alpha = np.asarray(result.obs["alpha_hat"])[:N_REAL]
    z_hat = np.asarray(result.obs["z_hat"])[:N_REAL] - 1

    emit(f"/// {doc}")
    emit(f"pub mod {module} {{")
    for key in sorted(kwargs):
        emit(f"    /// Reference parameter `{key} = {kwargs[key]!r}`.")
        if isinstance(kwargs[key], int):
            emit(f"    pub const {key.upper()}: usize = {kwargs[key]};")
        else:
            emit(f"    pub const {key.upper()}: f64 = {f(kwargs[key])};")
        emit()
    emit("    /// Fitted bulk contamination fraction.")
    emit(f"    pub const BETA: f64 = {f(result.uns['beta_hat'])};")
    emit()
    emit("    /// Mean per-barcode log-likelihood at the last iteration.")
    emit(f"    pub const LOG_LIKELIHOOD: f64 = {f(result.uns['loglike'])};")
    emit()
    f64_array("ALPHA", alpha, "Ambient fraction per real barcode.")
    f64_array(
        "AMBIENT", np.asarray(result.var["ambient_hat"]), "Fitted ambient profile."
    )
    f64_array(
        "PROFILES",
        np.asarray(result.uns["p_hat"]).ravel(),
        "Cell-type profiles, row-major `N_CELLTYPES x N_GENES`.",
    )
    f64_array(
        "DENOISED_ROW_SUMS",
        np.asarray(real.sum(axis=1)).ravel(),
        "Denoised library size per real barcode.",
    )
    f64_array(
        "DENOISED_COL_SUMS",
        np.asarray(real.sum(axis=0)).ravel(),
        "Denoised total per gene over the real barcodes.",
    )
    usize_array("Z_HAT", z_hat, "Final cell-type assignment per real barcode.")
    emit("}")
    emit()

print("\n".join(out))
