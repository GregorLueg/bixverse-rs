"""Write the anndata-written h5ad and Zarr fixtures the reader tests check against.

Every other h5ad in the tests comes from bixverse's own writer, so a convention
the reader and writer share would pass. These come from anndata itself. Run
from `python/`:

    uv run --no-project --with anndata --with scipy --with zarr \
        python dev/gen_anndata_fixture.py
"""

import shutil
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

# extra encodings for the Zarr parity tests, drawn after the counts and
# cell_type so the h5ad X and cell_type stay as they were
obs["n_umi_float"] = counts.sum(axis=1).astype(np.float64)
obs["is_doublet"] = rng.random(n_cells) < 0.2
obs["sample"] = [f"s{i % 3}" for i in range(n_cells)]
obs["barcode"] = [f"AC{i:03d}" for i in range(n_cells)]  # unique: stays a string
obs["batch_int"] = pd.array(
    [None if i % 7 == 0 else i % 4 for i in range(n_cells)], dtype="Int32"
)
obs["flag_na"] = pd.array(
    [None if i % 5 == 0 else bool(i % 2) for i in range(n_cells)], dtype="boolean"
)

OUT.mkdir(exist_ok=True)
np.save(OUT / "anndata_counts.npy", counts.astype(np.int64))
for fmt, conv in (("csr", sp.csr_matrix), ("csc", sp.csc_matrix)):
    ad.AnnData(X=conv(counts), obs=obs, var=var).write_h5ad(OUT / f"anndata_{fmt}.h5ad")
ad.AnnData(X=counts, obs=obs, var=var).write_h5ad(OUT / "anndata_dense.h5ad")

# Zarr twins: v2 (anndata's long-standing layout), v3, and v3 sharded
for zfmt in (2, 3):
    ad.settings.zarr_write_format = zfmt
    for fmt, x in (
        ("csr", sp.csr_matrix(counts)),
        ("csc", sp.csc_matrix(counts)),
        ("dense", counts),
    ):
        path = OUT / f"anndata_{fmt}_v{zfmt}.zarr"
        shutil.rmtree(path, ignore_errors=True)
        ad.AnnData(X=x, obs=obs, var=var).write_zarr(
            path, chunks=(8, 5) if fmt == "dense" else None
        )
ad.settings.zarr_write_format = 3
ad.settings.auto_shard_zarr_v3 = True
path = OUT / "anndata_csr_v3_sharded.zarr"
shutil.rmtree(path, ignore_errors=True)
ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var).write_zarr(path)
print(f"wrote {OUT}")
