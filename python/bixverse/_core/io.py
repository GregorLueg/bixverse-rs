"""Persisting the in-memory state of a `SingleCells` directory.

R saves its map and cache as `memory.qs2` / `memory.rds`, which Python cannot
read; Python saves its own as `memory_py.npz` (the arrays) plus
`memory_py.json` (everything else), and neither side touches the other's
files. The ids are not saved: they always come from the DuckDB. Without a
Python memory file `load_existing` rebuilds the map from the DuckDB, as R
does without its own.

The DuckDB is the one state both front ends write. Whichever memory file is
loaded wins over it for the cell filter, as in R, so a filter set on the other
side after the save is reported rather than silently ignored.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
from beartype import beartype

from bixverse._core import db
from bixverse._core.single_cells import GENES_BIN, SingleCells
from bixverse._core.state import CsrMatrix, KnnGraph, MagicLayer, ScCache, ScMap, Stamp

###########
# Globals #
###########

MEMORY_NPZ: str = "memory_py.npz"
MEMORY_JSON: str = "memory_py.json"
MEMORY_FORMAT: int = 1

_PLAIN_SLOTS: tuple[str, ...] = ("pca_factors", "pca_loadings", "pca_singular_vals")


########
# Save #
########


@beartype
def save(sc: SingleCells) -> None:
    """Write the map's index sets and the cache next to the stores.

    obs / var changes are in the DuckDB already and need no saving. The
    residual model is opaque for now and is not written; a warning says so.

    Args:
        sc: The experiment.
    """
    arrays: dict[str, np.ndarray] = {"map/cells_to_keep": sc.map.cells_to_keep}
    if sc.map.hvg is not None:
        arrays["map/hvg"] = sc.map.hvg

    c = sc.cache
    c.artefacts()  # rejects reserved embedding names before anything is written
    for slot in _PLAIN_SLOTS:
        value = getattr(c, slot)
        if value is not None:
            arrays[f"cache/{slot}"] = value
    for i, arr in enumerate(c.embeddings.values()):
        arrays[f"cache/embedding/{i}"] = arr
    if c.knn is not None:
        arrays["cache/knn/indices"] = c.knn.indices
        arrays["cache/knn/distances"] = c.knn.distances
    if c.connectivities is not None:
        arrays["cache/connectivities/data"] = c.connectivities.data
        arrays["cache/connectivities/indices"] = c.connectivities.indices
        arrays["cache/connectivities/indptr"] = c.connectivities.indptr
    if c.magic is not None:
        arrays["cache/magic/data"] = c.magic.data
        arrays["cache/magic/features"] = np.asarray(c.magic.features, dtype=str)
    if c.residual_fit is not None:
        warnings.warn(
            "The residual model is not persisted yet; it will be absent on load.",
            stacklevel=2,
        )

    present = c.artefacts()
    meta = {
        "format": MEMORY_FORMAT,
        "embeddings": list(c.embeddings),
        "knn_dist_metric": None if c.knn is None else c.knn.dist_metric,
        "connectivities_shape": (
            None if c.connectivities is None else list(c.connectivities.shape)
        ),
        # only stamps of present artefacts; a dropped payload's stamp is dead
        "stamps": {
            label: {"cells": s.cells, "id": s.id, "parents": list(s.parents)}
            for label, s in present.items()
            if s is not None and label != "residuals"
        },
    }

    npz, js = sc.dir / MEMORY_NPZ, sc.dir / MEMORY_JSON
    tmp_npz, tmp_js = npz.with_suffix(".npz.tmp"), js.with_suffix(".json.tmp")
    with tmp_npz.open("wb") as fh:
        np.savez(fh, **arrays)  # ty: ignore[invalid-argument-type]
    tmp_js.write_text(json.dumps(meta, indent=1))
    os.replace(tmp_npz, npz)
    os.replace(tmp_js, js)


########
# Load #
########


@beartype
def _cache_from_disk(arrays: dict[str, np.ndarray], meta: dict) -> ScCache:
    """Rebuild a cache from the saved arrays and metadata."""
    cache = ScCache()
    for slot in _PLAIN_SLOTS:
        if f"cache/{slot}" in arrays:
            setattr(cache, slot, arrays[f"cache/{slot}"])
    for i, name in enumerate(meta["embeddings"]):
        cache.embeddings[name] = arrays[f"cache/embedding/{i}"]
    if "cache/knn/indices" in arrays:
        cache.knn = KnnGraph(
            arrays["cache/knn/indices"],
            arrays["cache/knn/distances"],
            meta["knn_dist_metric"],
        )
    if "cache/connectivities/data" in arrays:
        n_rows, n_cols = meta["connectivities_shape"]
        cache.connectivities = CsrMatrix(
            arrays["cache/connectivities/data"],
            arrays["cache/connectivities/indices"],
            arrays["cache/connectivities/indptr"],
            (n_rows, n_cols),
        )
    if "cache/magic/data" in arrays:
        cache.magic = MagicLayer(
            arrays["cache/magic/data"],
            arrays["cache/magic/features"].astype(object),
        )
    for label, s in meta["stamps"].items():
        cache.set_stamp(label, Stamp(s["cells"], s["id"], tuple(s["parents"])))
    return cache


@beartype
def load_existing(directory: Path | str) -> SingleCells:
    """Open an experiment directory and restore any state Python saved.

    Args:
        directory: Written by R or by Python.

    Returns:
        The experiment. With `memory_py.*` present, the cell filter, HVGs and
        cache come from there; otherwise the map is rebuilt from the DuckDB
        and the cache is empty.

    Raises:
        FileNotFoundError: If a store, the DuckDB, or one half of the memory
            file pair is missing.
        ValueError: If the memory file is of an unknown format or does not
            fit the store.
    """
    directory = Path(directory).expanduser()
    genes = directory / GENES_BIN
    if not genes.is_file():
        raise FileNotFoundError(f"{genes} does not exist")
    sc = SingleCells(directory)

    npz, js = directory / MEMORY_NPZ, directory / MEMORY_JSON
    if not npz.exists() and not js.exists():
        return sc
    if not (npz.exists() and js.exists()):
        raise FileNotFoundError(f"need both {MEMORY_NPZ} and {MEMORY_JSON}")

    meta = json.loads(js.read_text())
    if meta.get("format") != MEMORY_FORMAT:
        raise ValueError(f"unknown {MEMORY_JSON} format {meta.get('format')}")
    with np.load(npz, allow_pickle=False) as fh:
        arrays = {k: fh[k] for k in fh.files}

    from_db = sc.map.cells_to_keep
    sc.map = ScMap(
        sc.map.cell_ids,
        sc.map.gene_ids,
        arrays["map/cells_to_keep"],
        arrays.get("map/hvg"),
    )
    sc.cache = _cache_from_disk(arrays, meta)

    if not np.array_equal(np.sort(sc.map.cells_to_keep), from_db):
        warnings.warn(
            f"The saved cell filter ({len(sc.map.cells_to_keep)} cells) differs "
            f"from obs.to_keep in {db.DB_NAME} ({len(from_db)} cells). The saved "
            "one is used; call set_cells_to_keep() to adopt the other.",
            stacklevel=2,
        )
    return sc
