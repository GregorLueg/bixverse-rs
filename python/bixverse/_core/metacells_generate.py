"""Meta cell generators: bootstrapped (hdWGCNA), SEACells, SuperCells and
MetaCells2.

Each generator takes a `SingleCells` or `SingleCellsSubset`, leaves it
untouched, and returns a `MetaCells`. The search, the partition and the
aggregation run in Rust over the cell-major store; this layer resolves the
embedding, kNN and cell sets and assembles the result. `meta_cells_per_group` lives in
`bixverse._core.pipeline`.

    >>> params = BtMetacellsParams(target_no_metacells=200)
    >>> mc = generate_bt_meta_cells(sc, params=params)
    >>> mc = generate_seacells(sc, SeacellsParams(n_sea_cells=200))
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.metacells import MetaCells, build_metacells
from bixverse._core.subset import ScOrScSubset
from bixverse._validate import Verbose
from bixverse.params import (
    BtMetacellsParams,
    Metacells2DacParams,
    Metacells2Params,
    SeacellsParams,
    SupercellParams,
    to_rust,
)

###########
# Helpers #
###########


@beartype
def _require_fresh(sc: ScOrScSubset, artefacts: Sequence[str]) -> None:
    """Refuse stale inputs: cell indices go straight into Rust.

    Raises:
        ValueError: If an artefact is stale.
    """
    state = sc.check_state(list(artefacts))
    if state is not True:
        raise ValueError(state)


@beartype
def _embedding(
    sc: ScOrScSubset, name: str, no_embd_to_use: int | None
) -> np.ndarray | None:
    """A cached embedding by name (`"pca"` for the scores), leading columns
    only, as contiguous `float32`."""
    embd = sc.cache.pca_factors if name == "pca" else sc.cache.embeddings.get(name)
    if embd is None:
        return None
    if no_embd_to_use is not None:
        embd = embd[:, : min(no_embd_to_use, embd.shape[1])]
    return np.ascontiguousarray(embd, dtype=np.float32)


@beartype
def _no_embedding() -> None:
    """Warn that the embedding is missing."""
    warnings.warn(
        "The desired embedding was not found. Please check the parameters. "
        "Returning None.",
        stacklevel=3,
    )


@beartype
def _var(sc: ScOrScSubset) -> pl.DataFrame:
    """The source's `gene_idx` and `gene_id`."""
    return sc.var(cols=["gene_idx", "gene_id"])


##############
# Generators #
##############


@beartype
def generate_bt_meta_cells(
    sc: ScOrScSubset,
    *,
    params: BtMetacellsParams | None = None,
    regenerate_knn: bool = False,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    cells_to_use: Sequence[str] | None = None,
    target_size: float = 1e5,
    seed: int = 42,
    verbose: Verbose = True,
) -> MetaCells | None:
    """Bootstrapped meta cells after hdWGCNA.

    Each meta cell is a kNN neighbourhood (the centre and its neighbours), so
    meta cells can share cells. The cached kNN is used unless
    `regenerate_knn` asks for one built on the embedding; `cells_to_use`
    always rebuilds it on those cells.

    Args:
        sc: The experiment or a subset of it.
        params: The meta cell and kNN parameters. `None` uses the defaults.
        regenerate_knn: Build the kNN on the embedding instead of using the
            cached one.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        no_embd_to_use: Leading embedding dimensions to use. `None` for all.
        cells_to_use: Cell ids to restrict the meta cells to.
        target_size: Library size to normalise the meta cells to.
        seed: Seed.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The meta cells, or `None` (with a warning) if the kNN or embedding is
        missing.

    Raises:
        ValueError: If the kNN or embedding is stale.

    References:
        Morabito, et al., Cell Rep Methods, 2023.

    Examples:
        >>> params = bv.params.BtMetacellsParams(target_no_metacells=100, k=25)
        >>> mc = bv.metacells.generate_bt_meta_cells(sc, params=params)
        >>> mc = bv.metacells.generate_bt_meta_cells(sc, regenerate_knn=True)
    """
    p = params or BtMetacellsParams()
    _require_fresh(sc, [embd_to_use] if regenerate_knn else ["knn"])

    embd = knn = None
    if regenerate_knn:
        embd = _embedding(sc, embd_to_use, no_embd_to_use)
        if embd is None:
            _no_embedding()
            return None
    else:
        if sc.cache.knn is None:
            warnings.warn(
                "No kNN data could be found on the object. Set regenerate_knn to "
                "True or generate the kNN matrix via other means. Returning None.",
                stacklevel=2,
            )
            return None
        knn = np.ascontiguousarray(sc.cache.knn.indices)

    to_use = None
    if cells_to_use is not None:
        to_use = sc.cell_indices(list(cells_to_use))
        embd = _embedding(sc, embd_to_use, no_embd_to_use)
        if embd is None:
            _no_embedding()
            return None
        knn = None

    res = _bixverse.bootstrapped_metacells(
        str(sc.cells_path),
        knn,
        embd,
        sc.cells_to_keep,
        to_use,
        to_rust(p),
        target_size,
        seed,
        verbose,
    )
    return build_metacells(
        res, _var(sc), "meta_cells_hdwgcna", cells_to_keep=sc.cells_to_keep
    )


@beartype
def generate_seacells(
    sc: ScOrScSubset,
    seacell_params: SeacellsParams,
    *,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    cells_to_use: Sequence[str] | None = None,
    regenerate_knn: bool = False,
    target_size: float = 1e5,
    seed: int = 42,
    verbose: Verbose = True,
) -> MetaCells:
    """SEACells, kernel archetypal analysis.

    The kernel is built on the embedding. The cached kNN is used unless
    `regenerate_knn` is set, none is cached, or `cells_to_use` narrows the
    cells; then it is built on the embedding. Archetypes that end up empty
    are dropped. `other_data` holds `rss` per iteration and the store index
    of each meta cell's `archetypes`.

    Args:
        sc: The experiment or a subset of it.
        seacell_params: The SEACells and kNN parameters; `n_sea_cells` has no
            default.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        no_embd_to_use: Leading embedding dimensions to use. `None` for all.
        cells_to_use: Cell ids to restrict the meta cells to.
        regenerate_knn: Build the kNN on the embedding.
        target_size: Library size to normalise the meta cells to.
        seed: Seed.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Raises:
        ValueError: If the embedding is missing, or it or the kNN is stale.

    References:
        Persad, et al., Nat Biotechnol, 2023.

    Examples:
        >>> params = bv.params.SeacellsParams(n_sea_cells=100)
        >>> mc = bv.metacells.generate_seacells(sc, params)
    """
    return _generate_seacells(
        sc,
        seacell_params,
        embd_to_use=embd_to_use,
        no_embd_to_use=no_embd_to_use,
        cells_to_use=cells_to_use,
        regenerate_knn=regenerate_knn,
        target_size=target_size,
        seed=seed,
        verbose=verbose,
        run=_bixverse.seacells,
    )


@beartype
def _generate_seacells(
    sc: ScOrScSubset,
    seacell_params: SeacellsParams,
    *,
    embd_to_use: str,
    no_embd_to_use: int | None,
    cells_to_use: Sequence[str] | None,
    regenerate_knn: bool,
    target_size: float,
    seed: int,
    verbose: Verbose,
    run: Callable[..., dict[str, Any]],
) -> MetaCells:
    """The body of `generate_seacells`, with the extension call `run`
    supplied, so the GPU twin shares it."""
    _require_fresh(sc, [embd_to_use] if regenerate_knn else [embd_to_use, "knn"])
    embd = _embedding(sc, embd_to_use, no_embd_to_use)
    if embd is None:
        raise ValueError(f"Embedding '{embd_to_use}' was not found.")
    to_use = None if cells_to_use is None else sc.cell_indices(list(cells_to_use))
    knn = None if regenerate_knn else sc.cache.knn

    res = run(
        str(sc.cells_path),
        embd,
        None if knn is None else np.ascontiguousarray(knn.indices),
        None if knn is None else np.ascontiguousarray(knn.distances),
        sc.cells_to_keep,
        to_use,
        to_rust(seacell_params),
        target_size,
        seed,
        verbose,
    )
    other = {"rss": res["rss"], "archetypes": res["archetypes"]}
    return build_metacells(
        res, _var(sc), "seacell", cells_to_keep=sc.cells_to_keep, other_data=other
    )


@beartype
def generate_supercells(
    sc: ScOrScSubset,
    *,
    params: SupercellParams | None = None,
    regenerate_knn: bool = False,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    cells_to_use: Sequence[str] | None = None,
    target_size: float = 1e5,
    seed: int = 42,
    verbose: Verbose = True,
) -> MetaCells | None:
    """SuperCells, walktrap on the kNN graph.

    `ceil(n_cells / graining_factor)` meta cells. The cached kNN is used
    unless `regenerate_knn` asks for one built on the embedding;
    `cells_to_use` always rebuilds it on those cells.

    Args:
        sc: The experiment or a subset of it.
        params: The SuperCell and kNN parameters. `None` uses the defaults.
        regenerate_knn: Build the kNN on the embedding.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        no_embd_to_use: Leading embedding dimensions to use. `None` for all.
        cells_to_use: Cell ids to restrict the meta cells to.
        target_size: Library size to normalise the meta cells to.
        seed: Seed for a kNN build.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The meta cells, or `None` (with a warning) if the kNN or embedding is
        missing.

    Raises:
        ValueError: If the kNN or embedding is stale.

    References:
        Bilous, et al., BMC Bioinform, 2022.

    Examples:
        >>> params = bv.params.SupercellParams(graining_factor=20)
        >>> mc = bv.metacells.generate_supercells(sc, params=params)
    """
    p = params or SupercellParams()
    _require_fresh(sc, [embd_to_use] if regenerate_knn else ["knn"])

    embd = None
    knn = None
    if regenerate_knn:
        embd = _embedding(sc, embd_to_use, no_embd_to_use)
        if embd is None:
            _no_embedding()
            return None
    else:
        knn = sc.cache.knn
        if knn is None:
            warnings.warn(
                "No kNN data could be found on the object. Set regenerate_knn to "
                "True or generate the kNN matrix via other means. Returning None.",
                stacklevel=2,
            )
            return None

    to_use = None
    if cells_to_use is not None:
        to_use = sc.cell_indices(list(cells_to_use))
        embd = _embedding(sc, embd_to_use, no_embd_to_use)
        if embd is None:
            _no_embedding()
            return None
        knn = None

    res = _bixverse.supercells(
        str(sc.cells_path),
        None if knn is None else np.ascontiguousarray(knn.indices),
        None if knn is None else np.ascontiguousarray(knn.distances),
        embd,
        sc.cells_to_keep,
        to_use,
        to_rust(p),
        target_size,
        seed,
        verbose,
    )
    return build_metacells(res, _var(sc), "supercells", cells_to_keep=sc.cells_to_keep)


@beartype
def generate_metacells2(
    sc: ScOrScSubset,
    *,
    mc2_params: Metacells2Params | None = None,
    dac_params: Metacells2DacParams | None = None,
    cells_to_use: Sequence[str] | None = None,
    target_size: float = 1e5,
    seed: int = 42,
    verbose: Verbose = True,
) -> MetaCells:
    """MetaCells2 (divide and conquer) on the raw counts.

    No embedding or kNN needed. Aims at a target size and UMI count rather
    than a number of meta cells and leaves outliers unassigned. `other_data`
    holds `cell_info` (`original_cell_idx` 0-based, `dissolved`, `level`,
    `rare_module` 0-based or null per included cell), `rare_gene_modules`
    (`gene_id`, `rare_module`) and the `selected_genes`.

    Args:
        sc: The experiment or a subset of it.
        mc2_params: The per-pile parameters. `None` uses the defaults.
        dac_params: The piling and rare gene module parameters. `None` uses
            the defaults.
        cells_to_use: Cell ids to restrict to; ids not kept warn and drop.
        target_size: Library size to normalise the meta cells to.
        seed: Seed.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    References:
        Ben-Kiki, et al., Genome Biol, 2022.

    Examples:
        >>> mc = bv.metacells.generate_metacells2(sc)
        >>> ids = sc.obs()["cell_id"].head(1500).to_list()
        >>> mc = bv.metacells.generate_metacells2(sc, cells_to_use=ids)
    """
    cells = sc.cells_to_keep
    if cells_to_use is not None:
        to_use = sc.cell_indices(list(cells_to_use))
        n_out = int((~np.isin(to_use, cells)).sum())
        if n_out:
            warnings.warn(
                f"{n_out} of the cells_to_use are not kept by the object and are "
                "ignored.",
                stacklevel=2,
            )
        cells = cells[np.isin(cells, to_use)]

    res = _bixverse.metacells2(
        str(sc.cells_path),
        cells,
        to_rust(mc2_params or Metacells2Params()),
        to_rust(dac_params or Metacells2DacParams()),
        target_size,
        seed,
        verbose,
    )
    var = _var(sc)
    gene_ids = var["gene_id"].to_numpy()
    module_of_gene = res["rare_module_of_gene"]
    rare = np.flatnonzero(module_of_gene >= 0)
    module_of_cell = res["rare_module_of_cell"]
    other: dict[str, object] = {
        "cell_info": pl.DataFrame(
            {
                "original_cell_idx": pl.Series(cells, dtype=pl.UInt32),
                "dissolved": res["dissolved"],
                "level": res["level"],
                "rare_module": pl.Series(
                    np.where(module_of_cell < 0, None, module_of_cell).tolist(),
                    dtype=pl.Int32,
                ),
            }
        ),
        "rare_gene_modules": pl.DataFrame(
            {
                "gene_id": pl.Series(gene_ids[rare].tolist(), dtype=pl.String),
                "rare_module": pl.Series(module_of_gene[rare], dtype=pl.Int32),
            }
        ),
        "selected_genes": [str(g) for g in gene_ids[res["selected_genes"]]],
    }
    return build_metacells(
        res, var, "metacells2", cells_to_keep=sc.cells_to_keep, other_data=other
    )
