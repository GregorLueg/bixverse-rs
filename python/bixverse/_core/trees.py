"""Bonsai trees over cells and meta cells.

The port of R's `bonsai_sc`, `relayout_bonsai` and `set_bonsai_embedding`
(`R/methods_sc_trees.R`, the `BonsaiTree` class). R keeps the trees apart
from Palantir and PAGA because they infer a hierarchy over every cell, not a
process with a start; so does this module.

The raw counts go through Sanity for posterior log fold changes with error
bars, Bonsai builds a tree with every cell (or meta cell) a leaf, and the tree
is laid out in 2D. The work runs in Rust; the object is not touched.

    >>> tree = bonsai(sc)
    >>> tree = relayout_bonsai(tree, layout="dendrogram")
    >>> set_bonsai_embedding(sc, tree)

Nodes are 0-based, the root's parent is null, and `hvg` takes 0-based gene
indices; R numbers all three from 1. As in R, `bonsai` takes a `SingleCells`
or a `MetaCells`, not a `SingleCellsSubset`.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.metacells import MetaCells, csr_args
from bixverse._core.single_cells import SingleCells
from bixverse.params import BonsaiParams, to_rust

###########
# Globals #
###########

BonsaiLayout = Literal["equal_angle", "equal_daylight", "dendrogram"]

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(frozen=True, repr=False)
class BonsaiTree:
    """A Bonsai tree with its layout, R's `BonsaiTree`.

    Attributes:
        nodes: `node` (0-based), `parent` (0-based, null for the root),
            `branch`, `is_leaf`, `cell_id` (null for the inferred ancestors),
            `n_cells` (cells behind a leaf: 1 for a cell, the originating
            cells for a meta cell; null for the ancestors), `x` and `y`. The
            leaves come first, in the object's cell or meta cell order.
        loglik: Loglikelihood of the final tree.
        steps: `step`, `loglik` after it and its wall time in `seconds`.
        timings: Wall time in `seconds` per `stage`: `sanity`, `ingest`,
            `bonsai`, `layout` and `total`, the whole call.
        genes_used: Ids of the genes the tree was built on.
        genes_dropped: The candidate genes left out: no counts, ill
            conditioned Sanity posteriors, or a signal-to-noise below
            `min_signal_to_noise`.
        cell_idx: The 0-based cells (store indices) or meta cells the tree
            was built over, in leaf order.
        layout: The current layout.
        hyperbolic: Whether the layout is on the hyperbolic disk.
        params: The parameters of the run.
    """

    nodes: pl.DataFrame
    loglik: float
    steps: pl.DataFrame
    timings: pl.DataFrame
    genes_used: tuple[str, ...]
    genes_dropped: tuple[str, ...]
    cell_idx: np.ndarray
    layout: str
    hyperbolic: bool
    params: BonsaiParams

    def __repr__(self) -> str:
        """R's `print.BonsaiTree`."""
        leaves = self.nodes.filter(pl.col("is_leaf"))
        n_leaves = leaves.height
        n_cells = int(leaves["n_cells"].sum())
        cells = f" ({n_cells} cells)" if n_cells > n_leaves else ""
        seconds = " | ".join(
            f"{s} {t:.1f}"
            for s, t in zip(self.timings["stage"], self.timings["seconds"], strict=True)
        )
        return "\n".join(
            [
                f"BonsaiTree: {n_leaves} leaves{cells}, "
                f"{self.nodes.height - n_leaves} inferred ancestors",
                f"  Genes: {len(self.genes_used)} used, "
                f"{len(self.genes_dropped)} dropped",
                f"  Loglikelihood: {self.loglik:.1f}",
                f"  Layout: {self.layout}"
                + (" (hyperbolic)" if self.hyperbolic else ""),
                f"  Seconds: {seconds}",
            ]
        )


###########
# Helpers #
###########


@beartype
def _nodes(
    res: dict, cell_names: Sequence[str], leaf_sizes: np.ndarray
) -> pl.DataFrame:
    """The node table, R's `.bonsai_nodes`: leaves first, carrying the cell
    ids, the root's parent null."""
    parent = np.asarray(res["parent"], dtype=np.int64)
    n_nodes = len(parent)
    n_leaves = len(cell_names)
    n_anc = n_nodes - n_leaves
    return pl.DataFrame(
        {
            "node": pl.Series(np.arange(n_nodes), dtype=pl.UInt32),
            "parent": pl.Series(
                [None if p < 0 else int(p) for p in parent], dtype=pl.UInt32
            ),
            "branch": pl.Series(res["branch"], dtype=pl.Float64),
            "is_leaf": pl.Series(np.arange(n_nodes) < n_leaves),
            "cell_id": pl.Series([*cell_names, *([None] * n_anc)], dtype=pl.String),
            "n_cells": pl.Series(
                [*leaf_sizes.tolist(), *([None] * n_anc)], dtype=pl.UInt32
            ),
            "x": pl.Series(res["x"], dtype=pl.Float64),
            "y": pl.Series(res["y"], dtype=pl.Float64),
        }
    )


@beartype
def _finish(
    res: dict,
    started: float,
    cell_idx: np.ndarray,
    cell_names: Sequence[str],
    leaf_sizes: np.ndarray,
    genes_in: np.ndarray,
    gene_ids: np.ndarray,
    params: BonsaiParams,
) -> BonsaiTree:
    """The `BonsaiTree` with the `total` timing appended, R's
    `.bonsai_finish` and `new_bonsai_tree`.

    Raises:
        ValueError: If the leaves do not match the cells.
    """
    if res["n_leaves"] != len(cell_idx):
        raise ValueError(
            f"Bonsai returned {res['n_leaves']} leaves for {len(cell_idx)} cells"
        )
    # Rust hands back the 0-based genes themselves, not positions in genes_in
    pos = {int(g): i for i, g in enumerate(genes_in)}
    used = [str(gene_ids[pos[int(g)]]) for g in res["genes_used"]]
    used_set = set(used)
    timings = pl.DataFrame(
        {
            "stage": [*res["timings"]["stage"], "total"],
            "seconds": [*res["timings"]["seconds"], time.perf_counter() - started],
        },
        schema={"stage": pl.String, "seconds": pl.Float64},
    )
    return BonsaiTree(
        nodes=_nodes(res, cell_names, leaf_sizes),
        loglik=float(res["loglik"]),
        steps=pl.DataFrame(
            dict(res["steps"]),
            schema={"step": pl.String, "loglik": pl.Float64, "seconds": pl.Float64},
        ),
        timings=timings,
        genes_used=tuple(used),
        genes_dropped=tuple(str(g) for g in gene_ids if str(g) not in used_set),
        cell_idx=cell_idx,
        layout=params.layout,
        hyperbolic=params.hyperbolic,
        params=params,
    )


@beartype
def _genes_in(hvg: np.ndarray | Sequence[int] | None, n_genes: int) -> np.ndarray:
    """The candidate genes, ascending and unique so `genes_used` comes back
    in gene order.

    Raises:
        ValueError: On an index outside `[0, n_genes)`.
    """
    if hvg is None:
        return np.arange(n_genes, dtype=np.uint32)
    genes = np.unique(np.asarray(hvg, dtype=np.int64))
    if genes.size and (genes[0] < 0 or genes[-1] >= n_genes):
        raise ValueError(f"hvg holds a gene index outside [0, {n_genes})")
    return genes.astype(np.uint32)


##########
# Bonsai #
##########


@beartype
def bonsai(
    sc: SingleCells | MetaCells,
    *,
    hvg: np.ndarray | Sequence[int] | None = None,
    bonsai_params: BonsaiParams | None = None,
    verbose: bool = False,
) -> BonsaiTree:
    """A Bonsai tree over the cells, as R's `bonsai_sc`.

    Every cell is a leaf and the internal nodes are inferred ancestral
    states, with branch lengths that carry the amount of change between
    them. The raw counts go through Sanity first for posterior log fold
    changes with error bars, and only genes with enough signal over their own
    noise (`min_signal_to_noise`) are kept for the tree; no HVG selection
    needed.

    On `MetaCells` every meta cell is a leaf. Their summed raw counts are
    Poisson like a cell's, so Sanity treats them alike, and a bigger meta
    cell gets tighter error bars. That is the way to large data; what goes
    is any structure inside a meta cell.

    Args:
        sc: The experiment or meta cells.
        hvg: 0-based candidate genes, e.g. `sc.hvg`. `None` takes every gene.
        bonsai_params: The parameters. `None` uses the defaults.
        verbose: Controls verbosity.

    Returns:
        The `BonsaiTree`, leaves in the kept cell or meta cell order.

    Raises:
        ValueError: On a gene index out of range, or if no gene survives the
            signal-to-noise filter.

    References:
        de Groot, et al., Nat. Biotechnol., 2026; Breda, et al., Nat.
        Biotechnol., 2021.
    """
    return _bonsai(
        sc, hvg=hvg, bonsai_params=bonsai_params, verbose=verbose, ext=_bixverse
    )


@beartype
def _bonsai(
    sc: SingleCells | MetaCells,
    *,
    hvg: np.ndarray | Sequence[int] | None,
    bonsai_params: BonsaiParams | None,
    verbose: bool,
    ext: ModuleType,
) -> BonsaiTree:
    """The body of `bonsai`, with the extension module `ext` (its `sc_bonsai`
    and `mc_bonsai`) supplied, so the GPU twin shares it."""
    params = bonsai_params or BonsaiParams()
    genes_in = _genes_in(hvg, sc.n_genes)
    gene_ids = sc.gene_ids(genes_in)
    if isinstance(sc, MetaCells):
        obs = sc.obs(cols=["meta_cell_id", "no_originating_cells"])
        cell_idx = np.arange(sc.n_cells, dtype=np.uint32)
        names = obs["meta_cell_id"].to_list()
        sizes = obs["no_originating_cells"].to_numpy()
        what = "metacells"
    else:
        cell_idx = sc.cells_to_keep.copy()
        names = sc.cell_ids(filtered=True).tolist()
        sizes = np.ones(len(cell_idx), dtype=np.uint32)
        what = "cells"
    if verbose:
        logger.info(
            "Running Sanity and Bonsai over %i %s and %i candidate genes.",
            len(cell_idx),
            what,
            len(genes_in),
        )

    started = time.perf_counter()
    if isinstance(sc, MetaCells):
        data, indices, indptr, shape = csr_args(sc.raw)
        res = ext.mc_bonsai(
            data, indices, indptr, shape, genes_in, to_rust(params), verbose
        )
    else:
        res = ext.sc_bonsai(
            str(sc.genes_path),
            str(sc.cells_path),
            cell_idx,
            genes_in,
            to_rust(params),
            verbose,
        )
    return _finish(res, started, cell_idx, names, sizes, genes_in, gene_ids, params)


@beartype
def relayout_bonsai(
    x: BonsaiTree,
    *,
    layout: BonsaiLayout = "equal_angle",
    hyperbolic: bool = False,
) -> BonsaiTree:
    """The tree laid out again without searching, as R's `relayout_bonsai`.

    The tree is renumbered on the Rust side, so the inferred ancestors can
    come back with other node numbers; the leaves, and so the cells, keep
    theirs.

    Args:
        x: The tree.
        layout: `"equal_angle"`, `"equal_daylight"` or `"dendrogram"`.
        hyperbolic: Project the layout onto the hyperbolic disk.

    Returns:
        A new `BonsaiTree` with the new node table and layout.
    """
    nodes = x.nodes
    parent = nodes["parent"].fill_null(-1).cast(pl.Int64).to_numpy()
    leaves = nodes.filter(pl.col("is_leaf"))
    res = _bixverse.bonsai_layout(
        np.ascontiguousarray(parent),
        np.ascontiguousarray(nodes["branch"].to_numpy()),
        leaves.height,
        layout,
        hyperbolic,
    )
    return dataclasses.replace(
        x,
        nodes=_nodes(res, leaves["cell_id"].to_list(), leaves["n_cells"].to_numpy()),
        layout=layout,
        hyperbolic=hyperbolic,
    )


@beartype
def set_bonsai_embedding(
    sc: SingleCells | MetaCells, tree: BonsaiTree, *, name: str = "bonsai"
) -> None:
    """Store the leaf coordinates as an embedding, as R's
    `set_bonsai_embedding`.

    Writes the `(n_leaves, 2)` positions to `sc.cache.embeddings[name]` and
    stamps it, so the embedding helpers can use it.

    Args:
        sc: The experiment or meta cells the tree was built over.
        tree: The `bonsai` result.
        name: The embedding name.

    Raises:
        ValueError: If the tree was not built over the object's current cells.
    """
    expected = (
        np.arange(sc.n_cells, dtype=np.uint32)
        if isinstance(sc, MetaCells)
        else sc.cells_to_keep
    )
    if not np.array_equal(tree.cell_idx, expected):
        raise ValueError(
            "The tree was not built over the current cells of this object. Rerun "
            "bonsai() after changing the cells to keep."
        )
    leaves = tree.nodes.filter(pl.col("is_leaf"))
    sc.cache.embeddings[name] = np.ascontiguousarray(
        leaves.select("x", "y").to_numpy(), dtype=np.float32
    )
    sc.stamp(name)
