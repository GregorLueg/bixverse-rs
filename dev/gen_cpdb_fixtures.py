"""Generate CellPhoneDB parity fixtures by calling the reference implementation.

The expression matrix comes from a Numerical Recipes LCG, mirrored on the Rust
side. Every value is k / 8 with a small integer k, so it is exact in f16 and no
float data crosses as text. The interactions are real ones from the bundled
v5.0.0 database, restricted to proteins with a single gene name.

Run from the repo root:
  uv run --with /Users/gregorlueg/repos/others/CellphoneDB \
      python dev/gen_cpdb_fixtures.py > tests/cpdb_fixtures/mod.rs
  rustfmt --edition 2024 tests/cpdb_fixtures/mod.rs
"""

import os
import sys
import tempfile
import contextlib

import pandas as pd

from cellphonedb.utils import db_utils
from cellphonedb.src.core.core_logger import core_logger
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

# the CPDB logger is bound to stdout at import, which is where the fixture goes
for h in core_logger.handlers:
    h.setStream(sys.stderr)

DB = "/Users/gregorlueg/repos/others/CellphoneDB/NatureProtocols2024_case_studies/v5.0.0/cellphonedb.zip"
LCG_SEED = 12345
CLUSTER_SIZES = [40, 80, 60, 70, 50]
# per-mille, so the Rust side compares integers
DENSITY = [0, 50, 150, 300, 500, 800]
N_SIMPLE = 16
N_COMPLEX = 8
ITERATIONS = 1000


def lcg_state(seed):
    x = seed
    while True:
        x = (1664525 * x + 1013904223) % (2**32)
        yield x


def density(g, k):
    return DENSITY[(g * 7 + k * 5) % len(DENSITY)]


def build_matrix(n_genes):
    """Gene-major dense values. Two LCG draws per entry: presence, then value."""
    rng = lcg_state(LCG_SEED)
    labels = [k for k, n in enumerate(CLUSTER_SIZES) for _ in range(n)]
    rows = []
    for g in range(n_genes):
        row = []
        for k in labels:
            u = next(rng)
            v = next(rng)
            if (u >> 8) % 1000 < density(g, k):
                row.append(1.0 + ((v >> 16) % 24) / 8.0)
            else:
                row.append(0.0)
        rows.append(row)
    return rows


# ------------------------------------------------------------ interactions
interactions, genes, cc, _, _, _ = db_utils.get_interactions_genes_complex(DB)

name_counts = genes.groupby("gene_name")["id_multidata"].nunique()
gene_of_md = {}
for md, grp in genes.groupby("id_multidata"):
    names = grp["gene_name"].unique()
    if len(names) == 1 and name_counts[names[0]] == 1:
        gene_of_md[md] = names[0]

subunits = cc.groupby("complex_multidata_id")["protein_multidata_id"].apply(list).to_dict()


def resolve(md):
    if md in subunits:
        out = [gene_of_md.get(p) for p in subunits[md]]
        return None if None in out else out
    g = gene_of_md.get(md)
    return None if g is None else [g]


picked = []
n_simple = n_complex = 0
seen = set()
for _, row in interactions.sort_values("id_cp_interaction").iterrows():
    key = (row["multidata_1_id"], row["multidata_2_id"])
    if key in seen:
        continue
    a, b = resolve(row["multidata_1_id"]), resolve(row["multidata_2_id"])
    if a is None or b is None:
        continue
    is_complex = len(a) > 1 or len(b) > 1
    if is_complex and n_complex < N_COMPLEX:
        n_complex += 1
    elif not is_complex and n_simple < N_SIMPLE:
        n_simple += 1
    else:
        continue
    seen.add(key)
    picked.append((row["id_cp_interaction"], a, b))
    if n_simple == N_SIMPLE and n_complex == N_COMPLEX:
        break

gene_names = sorted({g for _, a, b in picked for g in a + b})
gene_idx = {g: i for i, g in enumerate(gene_names)}

# ---------------------------------------------------------------- run CPDB
matrix = build_matrix(len(gene_names))
n_cells = sum(CLUSTER_SIZES)
cells = [f"c{i:04d}" for i in range(n_cells)]
cluster_names = [f"k{k}" for k in range(len(CLUSTER_SIZES))]
labels = [cluster_names[k] for k, n in enumerate(CLUSTER_SIZES) for _ in range(n)]

with tempfile.TemporaryDirectory() as tmp:
    counts_fp = os.path.join(tmp, "counts.txt")
    meta_fp = os.path.join(tmp, "meta.txt")
    pd.DataFrame(matrix, index=gene_names, columns=cells).to_csv(counts_fp, sep="\t", index_label="Gene")
    pd.DataFrame({"Cell": cells, "cell_type": labels}).to_csv(meta_fp, sep="\t", index=False)
    with contextlib.redirect_stdout(sys.stderr):
        res = cpdb_statistical_analysis_method.call(
            cpdb_file_path=DB,
            meta_file_path=meta_fp,
            counts_file_path=counts_fp,
            counts_data="gene_name",
            output_path=tmp,
            iterations=ITERATIONS,
            threshold=0.1,
            threads=1,
            debug_seed=0,
            result_precision=15,
        )

means = res["means"].set_index("id_cp_interaction")
pvals = res["pvalues"].set_index("id_cp_interaction")
pair_cols = [c for c in means.columns if "|" in c]
pairs = [tuple(cluster_names.index(x) for x in c.split("|")) for c in pair_cols]

# ------------------------------------------------------------------- emit
out = []
emit = out.append


def f(x):
    return f"{float(x):.17e}"


def idx_list(xs):
    return "&[" + ", ".join(str(gene_idx[g]) for g in xs) + "]"


emit("#![allow(clippy::excessive_precision)]")
emit("")
emit("//! CellPhoneDB parity fixtures, generated from the reference implementation.")
emit("//!")
emit("//! DO NOT EDIT. Regenerate with `dev/gen_cpdb_fixtures.py`.")
emit("//!")
emit(f"//! Interactions from the CellPhoneDB v5.0.0 database. Statistical method with")
emit(f"//! {ITERATIONS} iterations, threshold 0.1. Values are `1 + k / 8` from the LCG,")
emit("//! exact in f16.")
emit("")
emit("/// Seed for the fixture LCG.")
emit(f"pub const LCG_SEED: u64 = {LCG_SEED};")
emit("")
emit("/// Cells per cluster, contiguous in cell order.")
emit(f"pub const CLUSTER_SIZES: &[usize] = &{CLUSTER_SIZES};")
emit("")
emit("/// Presence probability per mille, indexed by `(g * 7 + k * 5) % len`.")
emit(f"pub const DENSITY: &[u64] = &{DENSITY};")
emit("")
emit("/// Permutations used by the reference.")
emit(f"pub const ITERATIONS: usize = {ITERATIONS};")
emit("")
emit("/// Gene names, in matrix row order.")
emit("pub const GENES: &[&str] = &[" + ", ".join(f'"{g}"' for g in gene_names) + "];")
emit("")
emit("/// Cluster pairs `(A, B)` in the reference column order.")
emit("pub const PAIRS: &[(usize, usize)] = &[" + ", ".join(f"({a}, {b})" for a, b in pairs) + "];")
emit("")
emit("/// One interaction and the reference results across `PAIRS`.")
emit("pub struct CpdbFixture {")
emit("    /// CellPhoneDB interaction id")
emit("    pub id: &'static str,")
emit("    /// Partner a subunit rows")
emit("    pub partner_a: &'static [usize],")
emit("    /// Partner b subunit rows")
emit("    pub partner_b: &'static [usize],")
emit("    /// Interaction means")
emit("    pub means: &'static [f64],")
emit("    /// P-values")
emit("    pub pvals: &'static [f64],")
emit("}")
emit("")
emit("/// The picked interactions.")
emit("pub const FIXTURES: &[CpdbFixture] = &[")
for cpi, a, b in picked:
    m = means.loc[cpi, pair_cols].astype(float).tolist()
    p = pvals.loc[cpi, pair_cols].astype(float).tolist()
    emit("    CpdbFixture {")
    emit(f'        id: "{cpi}",')
    emit(f"        partner_a: {idx_list(a)},")
    emit(f"        partner_b: {idx_list(b)},")
    emit("        means: &[" + ", ".join(f(x) for x in m) + "],")
    emit("        pvals: &[" + ", ".join(f(x) for x in p) + "],")
    emit("    },")
emit("];")

print("\n".join(out))
