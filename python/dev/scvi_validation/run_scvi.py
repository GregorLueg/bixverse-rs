"""Train scVI four ways on the same cells, genes and split.

Usage:
    uv run --group validation python -I dev/scvi_validation/run_scvi.py <out_dir>
        [--seeds 0 1 2] [--max-epochs N] [--arms scvi_torch bv_torch ...]

Arms:
    scvi_torch  scvi-tools `SCVI` on an AnnData built from the raw h5ads
    scvi_mlx    scvi-tools `mlxSCVI`, its own (different) network and defaults
    bv_torch    `bixverse.nn` on torch, reading the bixverse store
    bv_mlx      `bixverse.nn` on MLX, reading the bixverse store

Every arm trains on the same 90% of the kept cells and embeds all of them.
Held-out ELBO is scored on the other 10% by one implementation for every arm
with this architecture: scvi-tools weights are loaded into `bixverse.nn`'s
torch model, which the parity tests show reproduces scvi-tools' loss to
rounding. `mlxSCVI` has no ELBO method and a different network, so it gets
latents and wall time only.

Writes `<out_dir>/embeddings/<arm>_s<seed>.tsv` (cells.tsv order, for
metrics.R) and appends to `<out_dir>/runs.tsv`.
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import bixverse as bv
import bixverse.nn as bvnn
from bixverse.nn import _torch as bvt

SPLIT_SEED = 0
TRAIN_FRACTION = 0.9


def build_adata(out_dir: Path, cells: pd.DataFrame, genes: pd.DataFrame) -> ad.AnnData:
    """Kept cells x HVGs from the raw h5ads, rows in cells.tsv order."""
    blocks = np.empty(len(cells), dtype=object)
    for file in cells["h5ad_file"].unique():
        a = ad.read_h5ad(out_dir / "h5ad" / file)
        var_pos = {name: i for i, name in enumerate(a.var_names)}
        cols = np.array([var_pos[v] for v in genes["h5ad_var"]])
        pos = np.flatnonzero((cells["h5ad_file"] == file).to_numpy())
        x = sp.csr_matrix(a.X[cells["h5ad_row"].to_numpy()[pos]][:, cols])
        for i, p in enumerate(pos):
            blocks[p] = x[i]
    x = sp.vstack(list(blocks)).tocsr().astype(np.float32)
    obs = pd.DataFrame({"exp_id": cells["exp_id"].astype(str).to_numpy()})
    obs.index = obs.index.astype(str)
    return ad.AnnData(X=x, obs=obs)


def scvi_as_fit(
    module: torch.nn.Module, genes: np.ndarray, cats: np.ndarray
) -> bvnn.ScviFit:
    """Wrap scvi-tools' trained module weights in a `bixverse.nn` fit."""
    params = bvnn.ScviParams(gene_likelihood=module.gene_likelihood)
    vae = bvt.Vae(len(genes), len(cats), params)
    state = module.state_dict()
    vae.load_state_dict({k: state[k].cpu() for k in vae.state_dict()}, strict=True)
    vae.eval()
    return bvnn.ScviFit(
        model=vae,
        backend="torch",
        params=params,
        genes=genes,
        batch_categories=cats,
        device="cpu",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--likelihood", default="zinb", choices=["nb", "zinb"])
    p.add_argument(
        "--arms", nargs="+", default=["scvi_torch", "scvi_mlx", "bv_torch", "bv_mlx"]
    )
    args = p.parse_args()

    out = args.out_dir
    cells = pd.read_csv(out / "cells.tsv", sep="\t")
    genes = pd.read_csv(out / "genes.tsv", sep="\t")
    store = (out / "store_path.txt").read_text().strip()
    reader = bv.CellReader(store)
    store_cells = cells["store_idx"].to_numpy()
    store_genes = genes["store_idx"].to_numpy()
    batch = cells["exp_id"].astype(str).to_numpy()

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(cells))
    n_train = round(TRAIN_FRACTION * len(cells))
    train_pos, test_pos = np.sort(perm[:n_train]), np.sort(perm[n_train:])

    needs_adata = any(a.startswith("scvi") for a in args.arms)
    adata = build_adata(out, cells, genes) if needs_adata else None
    params = bvnn.ScviParams(
        gene_likelihood=args.likelihood, max_epochs=args.max_epochs
    )
    (out / "embeddings").mkdir(exist_ok=True)
    rows = []

    for seed in args.seeds:
        for arm in args.arms:
            t0 = time.perf_counter()
            elbo = None
            match arm:
                case "scvi_torch":
                    import scvi

                    scvi.settings.seed = seed
                    a_train = adata[train_pos].copy()
                    scvi.model.SCVI.setup_anndata(a_train, batch_key="exp_id")
                    m = scvi.model.SCVI(a_train, gene_likelihood=args.likelihood)
                    m.train(max_epochs=args.max_epochs, train_size=1.0, batch_size=128)
                    train_s = time.perf_counter() - t0
                    a_full = adata.copy()
                    scvi.model.SCVI.setup_anndata(a_full, batch_key="exp_id")
                    latent = m.get_latent_representation(a_full)
                    fit = scvi_as_fit(m.module.cpu(), store_genes, np.unique(batch))
                    elbo = bvnn.scvi_elbo(
                        fit, reader, store_cells[test_pos], batch[test_pos]
                    )
                case "scvi_mlx":
                    import scvi

                    scvi.settings.seed = seed
                    a_train = adata[train_pos].copy()
                    scvi.model.mlxSCVI.setup_anndata(a_train, batch_key="exp_id")
                    m = scvi.model.mlxSCVI(a_train)
                    m.train(max_epochs=args.max_epochs, train_size=1.0, batch_size=128)
                    train_s = time.perf_counter() - t0
                    a_full = adata.copy()
                    scvi.model.mlxSCVI.setup_anndata(a_full, batch_key="exp_id")
                    latent = np.asarray(m.get_latent_representation(a_full))
                case "bv_torch" | "bv_mlx":
                    fit = bvnn.train_scvi(
                        reader,
                        store_cells[train_pos],
                        store_genes,
                        batch[train_pos],
                        params=params,
                        backend="torch" if arm == "bv_torch" else "mlx",
                        seed=seed,
                    )
                    train_s = time.perf_counter() - t0
                    latent = bvnn.scvi_latent(fit, reader, store_cells, batch)
                    elbo = bvnn.scvi_elbo(
                        fit, reader, store_cells[test_pos], batch[test_pos]
                    )
                case _:
                    raise ValueError(f"unknown arm {arm}")

            name = f"{arm}_s{seed}"
            pd.DataFrame(
                latent, columns=[f"z_{i + 1}" for i in range(latent.shape[1])]
            ).to_csv(out / "embeddings" / f"{name}.tsv", sep="\t", index=False)
            row = {"arm": arm, "seed": seed, "train_s": round(train_s, 2)}
            if elbo is not None:
                row |= {f"heldout_{k}": v for k, v in elbo.items()}
            rows.append(row)
            print(row, file=sys.stderr, flush=True)

    runs = pd.DataFrame(rows)
    path = out / "runs.tsv"
    runs.to_csv(path, sep="\t", index=False, mode="a", header=not path.exists())


if __name__ == "__main__":
    main()
