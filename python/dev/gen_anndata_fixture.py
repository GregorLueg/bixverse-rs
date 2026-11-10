"""Write the anndata-written h5ad fixtures the reader tests check against.

Every other h5ad in the tests comes from bixverse's own writer, so a convention
the reader and writer share would pass. These come from anndata itself. Run
from `python/`:

    uv run --no-project --with anndata --with scipy python dev/gen_anndata_fixture.py
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

rng = np.random.default_rng(1)
n_cells, n_genes = 30, 12
counts = rng.poisson(1.5, (n_cells, n_genes)).astype(np.float32)
counts[counts.sum(axis=1) == 0, 0] = 1  # no empty cells
obs = pd.DataFrame(
    {"cell_type": pd.Categorical(rng.choice(["b", "a", "c"], n_cells))},
    index=[f"cell_{i}" for i in range(n_cells)],
)
var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])

OUT.mkdir(exist_ok=True)
np.save(OUT / "anndata_counts.npy", counts.astype(np.int64))
for fmt, conv in (("csr", sp.csr_matrix), ("csc", sp.csc_matrix)):
    ad.AnnData(X=conv(counts), obs=obs, var=var).write_h5ad(OUT / f"anndata_{fmt}.h5ad")
print(f"wrote {OUT}")
