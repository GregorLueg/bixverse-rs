"""Write the pySCENIC AUCell fixture the recovery AUC test checks against.

Each cell's counts are a permutation of 1..n_genes, so there are no ties
within a cell: pySCENIC breaks ties at random and bixverse takes midranks, and
without ties both rank the same. Run from `python/`:

    uv run --no-project --python 3.10 --with pyscenic --with "numpy<2" \
        python dev/gen_aucell_fixture.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from ctxcore.genesig import GeneSignature
from pyscenic.aucell import aucell

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

rng = np.random.default_rng(3)
n_cells, n_genes = 60, 200
counts = np.vstack([rng.permutation(n_genes) + 1 for _ in range(n_cells)])
genes = [f"gene_{j}" for j in range(n_genes)]
sets = {
    f"set_{m}": [genes[j] for j in rng.choice(n_genes, m, replace=False)]
    for m in (5, 15, 30)
}

exp = pd.DataFrame(counts, index=[f"cell_{i}" for i in range(n_cells)], columns=genes)
sigs = [GeneSignature(name, {g: 1.0 for g in gs}) for name, gs in sets.items()]
auc = aucell(
    exp, sigs, auc_threshold=0.05, noweights=True, normalize=False, num_workers=1
)

members = {
    name: np.array([int(g.split("_")[1]) for g in gs]) for name, gs in sets.items()
}
OUT.mkdir(exist_ok=True)
np.savez(
    OUT / "aucell_pyscenic.npz",
    counts=counts.astype(np.int64),
    auc=auc.loc[exp.index, list(sets)].to_numpy(),
    set_5=members["set_5"],
    set_15=members["set_15"],
    set_30=members["set_30"],
)
print(f"wrote {OUT / 'aucell_pyscenic.npz'}")
