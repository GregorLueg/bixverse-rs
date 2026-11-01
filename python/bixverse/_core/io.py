"""Persisting the in-memory state of a `SingleCells` directory.

R saves its map and cache as `memory.qs2` / `memory.rds`, which Python cannot
read; Python saves its own as `memory_py.npz` (the arrays) plus
`memory_py.json` (everything else), and neither side touches the other's
files. The ids are not saved: they always come from the DuckDB. Without a
Python memory file `load_existing` rebuilds the map from the DuckDB, as R
does without its own.

The DuckDB is the one state both front ends write, and its `obs.to_keep` is
the cell filter: a memory file whose saved filter differs from it is
overruled with a warning, so a filter set on the other side after the save is
honoured and its stale cache reported.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
from beartype import beartype

from bixverse import _bixverse
from bixverse._core import db
from bixverse._core.multimodal import (
    SingleCellsMultiModal,
    mm_from_payload,
    mm_payload,
)
from bixverse._core.single_cells import CELLS_BIN, GENES_BIN, SingleCells
from bixverse._core.state import ScMap, cache_from_payload, cache_payload

###########
# Globals #
###########

MEMORY_NPZ: str = "memory_py.npz"
MEMORY_JSON: str = "memory_py.json"
MEMORY_FORMAT: int = 1
ARCHIVE_NAME: str = "counts.bxa"

logger = logging.getLogger(__name__)


########
# Save #
########


@beartype
def save(sc: SingleCells) -> None:
    """Write the map's index sets and the cache next to the stores.

    obs / var changes are in the DuckDB already and need no saving. The
    residual model is opaque for now and is not written; a warning says so.
    A `SingleCellsMultiModal` also writes its ADT counts, ADT cache and WNN
    slot, and `load_existing` hands it back as one.

    Args:
        sc: The experiment.

    Examples:
        >>> bv.io.save(sc)
    """
    arrays: dict[str, np.ndarray] = {"map/cells_to_keep": sc.map.cells_to_keep}
    if sc.map.hvg is not None:
        arrays["map/hvg"] = sc.map.hvg

    cache_arrays, cache_meta = cache_payload(sc.cache)
    arrays.update(cache_arrays)
    meta = {"format": MEMORY_FORMAT, **cache_meta}
    if isinstance(sc, SingleCellsMultiModal):
        mm_arrays, meta["multi_modal"] = mm_payload(sc)
        arrays.update(mm_arrays)

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
def load_existing(directory: Path | str, *, multi_modal: bool = False) -> SingleCells:
    """Open an experiment directory and restore any state Python saved.

    Args:
        directory: Written by R or by Python.
        multi_modal: Open it as a `SingleCellsMultiModal`. A memory file a
            multi-modal experiment saved does so regardless.

    Returns:
        The experiment. With `memory_py.*` present, the cell filter, HVGs and
        cache come from there (and for a multi-modal one the ADT counts, ADT
        cache and WNN slot); otherwise the map is rebuilt from the DuckDB and
        the cache is empty. R keeps its ADT counts in its own memory file, so
        they do not carry over from R.

    Restores the stores from `counts.bxa` when only the archive is there,
    and warns about a store in the v3 format, as R does.

    Raises:
        FileNotFoundError: If a store, the DuckDB, or one half of the memory
            file pair is missing.
        ValueError: If the memory file is of an unknown format or does not
            fit the store.

    Examples:
        >>> sc = bv.io.load_existing("experiments/demo")
        >>> mm = bv.io.load_existing("experiments/demo", multi_modal=True)
    """
    directory = Path(directory).expanduser()
    cells, genes = directory / CELLS_BIN, directory / GENES_BIN
    archive = directory / ARCHIVE_NAME
    if not cells.exists() and not genes.exists() and archive.is_file():
        logger.info("Found only %s. Restoring the binaries from it.", ARCHIVE_NAME)
        _bixverse.restore_cells_archive(archive, cells, genes, None, False)
    if not genes.is_file():
        raise FileNotFoundError(f"{genes} does not exist")
    if any(_bixverse.file_version(p) < 4 for p in (cells, genes)):
        warnings.warn(
            "This store uses the old v3 file format. It still works, but the v4 "
            "format is smaller and faster. Convert it with upgrade_sc_store().",
            stacklevel=2,
        )
    npz, js = directory / MEMORY_NPZ, directory / MEMORY_JSON
    if not npz.exists() and not js.exists():
        return (
            SingleCellsMultiModal(directory) if multi_modal else SingleCells(directory)
        )
    if not (npz.exists() and js.exists()):
        raise FileNotFoundError(f"need both {MEMORY_NPZ} and {MEMORY_JSON}")

    meta = json.loads(js.read_text())
    if meta.get("format") != MEMORY_FORMAT:
        raise ValueError(f"unknown {MEMORY_JSON} format {meta.get('format')}")
    mm_meta = meta.get("multi_modal")
    sc = (
        SingleCellsMultiModal(directory)
        if multi_modal or mm_meta is not None
        else SingleCells(directory)
    )
    with np.load(npz, allow_pickle=False) as fh:
        arrays = {k: fh[k] for k in fh.files}

    saved_keep = np.unique(arrays["map/cells_to_keep"])
    sc.map = ScMap(
        sc.map.cell_ids,
        sc.map.gene_ids,
        sc.map.cells_to_keep,
        arrays.get("map/hvg"),
    )
    sc.cache = cache_from_payload(arrays, meta)
    if isinstance(sc, SingleCellsMultiModal) and mm_meta is not None:
        mm_from_payload(sc, arrays, mm_meta)
        adt = sc.adt
        if adt is not None:
            n_missing = int((~np.isin(sc.cell_ids(filtered=True), adt.cell_ids)).sum())
            if n_missing:
                warnings.warn(
                    f"{n_missing} cells to keep are not present in the stored ADT "
                    "counts.",
                    stacklevel=2,
                )

    if not np.array_equal(saved_keep, sc.map.cells_to_keep):
        warnings.warn(
            f"The saved cell filter ({len(saved_keep)} cells) differs from "
            f"obs.to_keep in {db.DB_NAME} ({len(sc.map.cells_to_keep)} cells). "
            "obs.to_keep is used; cached artefacts from the saved filter are stale.",
            stacklevel=2,
        )
    return sc
