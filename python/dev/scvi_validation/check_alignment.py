"""Check that the bixverse cell-major store lines up with the raw h5ads.

Usage: python -I check_alignment.py <out_dir>

Reads cells.tsv, genes.tsv and store_path.txt written by prepare_pbmc.R, then
compares the store counts (layer='raw', HVG columns) against the h5ad X at
(h5ad_row, h5ad_var) for a random sample of kept cells and then for all of
them. The raw h5ads are read from <out_dir>/h5ad (copied by prepare_pbmc.R).
"""

import sys
from pathlib import Path

import anndata as ad
import bixverse as bv
import numpy as np
import pandas as pd
import scipy.sparse as sp

N_SAMPLE = 20
SEED = 1


def load_store(store_path: str, store_cells: np.ndarray, store_genes: np.ndarray):
    reader = bv.CellReader(store_path)
    assert reader.is_cell_based
    loader = bv.CellLoader(
        reader,
        [int(i) for i in store_cells],
        layer="raw",
        shuffle=False,
        genes=[int(i) for i in store_genes],
        sparse=False,
        batch_size=1024,
    )
    mats, ids = [], []
    for batch in loader:
        mats.append(np.asarray(batch.raw))
        ids.append(np.asarray(batch.cell_ids))
    return np.vstack(mats), np.concatenate(ids), reader


def h5ad_block(h5ad_dir: Path, cells: pd.DataFrame, genes: pd.DataFrame):
    """Counts from the h5ads, rows in `cells` order, columns in `genes` order."""
    out = np.zeros((len(cells), len(genes)), dtype=np.float64)
    for file in cells["h5ad_file"].unique():
        adata = ad.read_h5ad(h5ad_dir / file, backed=None)
        var_pos = {name: i for i, name in enumerate(adata.var_names)}
        cols = np.array([var_pos[v] for v in genes["h5ad_var"]])
        pos = np.flatnonzero((cells["h5ad_file"] == file).to_numpy())
        rows = cells["h5ad_row"].to_numpy()[pos]
        # barcodes must agree too, not just the row index
        assert (
            adata.obs_names[rows].to_numpy() == cells["barcode"].to_numpy()[pos]
        ).all() or (
            adata.obs["Barcode"].to_numpy()[rows] == cells["barcode"].to_numpy()[pos]
        ).all(), f"barcode mismatch in {file}"
        x = adata.X
        x = x[rows][:, cols]
        out[pos] = x.toarray() if sp.issparse(x) else np.asarray(x)
    return out


def main() -> None:
    out_dir = Path(sys.argv[1])
    h5ad_dir = out_dir / "h5ad"
    cells = pd.read_csv(out_dir / "cells.tsv", sep="\t")
    genes = pd.read_csv(out_dir / "genes.tsv", sep="\t")
    store_path = (out_dir / "store_path.txt").read_text().strip()

    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(cells), N_SAMPLE, replace=False))

    for label, sel in (("random sample", sample), ("all kept cells", None)):
        sub = cells if sel is None else cells.iloc[sel]
        store_x, ids, reader = load_store(
            store_path, sub["store_idx"].to_numpy(), genes["store_idx"].to_numpy()
        )
        assert (ids == sub["store_idx"].to_numpy()).all(), "loader reordered cells"
        ref = h5ad_block(h5ad_dir, sub, genes)
        is_int = np.allclose(ref, np.round(ref))
        n_bad = int((store_x != ref).sum())
        print(
            f"{label}: {len(sub)} cells x {len(genes)} genes, "
            f"store {reader.total_cells}x{reader.total_genes}, "
            f"h5ad X integer-valued: {is_int}, "
            f"mismatching entries: {n_bad}, "
            f"counts sum store/h5ad: {store_x.sum():.0f}/{ref.sum():.0f}"
        )
        assert n_bad == 0, "store and h5ad disagree"
    print("ALIGNMENT OK")


if __name__ == "__main__":
    main()
