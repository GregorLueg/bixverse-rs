"""Backend dispatch for scVI: train on a store, then embed or score cells."""

from __future__ import annotations

from typing import Any

import numpy as np
from beartype import beartype

from .._bixverse import CellReader
from ._data import encode_batches, resolve_cells, resolve_genes
from ._params import Backend, ScviFit, ScviParams


@beartype
def _open(store: str | CellReader) -> CellReader:
    """Accept a path or an open reader.

    Args:
        store: Path to a cell-major `.bin` store, or a `CellReader` on one.

    Returns:
        An open reader.
    """
    return store if isinstance(store, CellReader) else CellReader(store)


@beartype
def _backend(name: Backend) -> Any:
    """Import a backend module, naming the extra when it is missing.

    Args:
        name: `"torch"`, `"mlx"` or `"jax"`.

    Returns:
        The backend module.

    Raises:
        ImportError: If the framework is not installed.
    """
    try:
        match name:
            case "torch":
                from . import _torch as mod
            case "mlx":
                from . import _mlx as mod
            case "jax":
                from . import _jax as mod
    except ImportError as e:
        raise ImportError(
            f"the {name} backend needs `pip install bixverse[{name}]`"
        ) from e
    return mod


@beartype
def train_scvi(
    store: str | CellReader,
    cells: Any = None,
    genes: Any = None,
    batch: Any = None,
    *,
    params: ScviParams | None = None,
    backend: Backend = "torch",
    device: str = "auto",
    seed: int = 42,
) -> ScviFit:
    """Train scVI on raw counts streamed from a bixverse store.

    Args:
        store: Path to a cell-major `.bin` store, or a `CellReader`.
        cells: Store cell indices to train on. `None` takes every cell.
        genes: Store gene indices to model, typically the HVGs. `None` takes
            every gene. Library size is the sum over these genes, as in
            scvi-tools on a gene-subset AnnData.
        batch: One batch label per entry of `cells`. `None` is one batch.
        params: Hyperparameters. `None` takes scvi-tools' defaults.
        backend: `"torch"`, `"mlx"` (Apple Silicon) or `"jax"`.
        device: Torch device, or `"auto"` for cuda, then mps, then cpu. On
            JAX a platform name (`"cpu"`, `"gpu"`), `"auto"` taking JAX's
            default. Ignored by MLX, which runs on the GPU.
        seed: Seeds initialisation, sampling and shuffling.

    Returns:
        The fitted model.

    Raises:
        ValueError: On bad indices, a batch vector of the wrong length, or a
            cell with no counts over `genes`.
    """
    reader = _open(store)
    cells_ = resolve_cells(reader, cells)
    genes_ = resolve_genes(reader, genes)
    codes, categories = encode_batches(batch, n_cells=len(cells_))
    params = params or ScviParams()
    mod = _backend(backend)
    match backend:
        case "torch" | "jax":
            return mod.train(
                reader,
                cells_,
                genes_,
                codes,
                categories,
                params=params,
                device=device,
                seed=seed,
            )
        case "mlx":
            return mod.train(
                reader, cells_, genes_, codes, categories, params=params, seed=seed
            )


@beartype
def _evaluate(
    fit: ScviFit, store: str | CellReader, cells: Any, batch: Any, seed: int
) -> dict[str, np.ndarray]:
    """Run a fitted model forward over some cells.

    Args:
        fit: A fitted model.
        store: The store, as for `train_scvi`.
        cells: Store cell indices, `None` for all.
        batch: Batch labels aligned with `cells`, from the training categories.
        seed: Seeds the draw of z behind the reconstruction term.

    Returns:
        `latent`, `reconstruction` and `kl`, in `cells` order.
    """
    reader = _open(store)
    cells_ = resolve_cells(reader, cells)
    codes, _ = encode_batches(
        batch, n_cells=len(cells_), categories=fit.batch_categories
    )
    return _backend(fit.backend).evaluate(fit, reader, cells_, codes, seed=seed)


@beartype
def scvi_latent(
    fit: ScviFit,
    store: str | CellReader,
    cells: Any = None,
    batch: Any = None,
) -> np.ndarray:
    """The scVI embedding: the posterior mean of z, as `get_latent_representation`.

    The encoder never sees the batch, so `batch` only has to be valid, not
    meaningful; it is accepted so a held-out set can be scored in one pass.

    Args:
        fit: A fitted model.
        store: The store.
        cells: Store cell indices, `None` for all. Rows come back in this order.
        batch: Batch labels aligned with `cells`, or `None` for a one-batch fit.

    Returns:
        `(n_cells, n_latent)` float32 array.
    """
    if batch is None and len(fit.batch_categories) > 1:
        batch = np.repeat(
            fit.batch_categories[0], len(resolve_cells(_open(store), cells))
        )
    return _evaluate(fit, store, cells, batch, seed=0)["latent"]


@beartype
def scvi_elbo(
    fit: ScviFit,
    store: str | CellReader,
    cells: Any = None,
    batch: Any = None,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Mean per-cell negative ELBO and its two terms, at KL weight 1.

    The reconstruction term decodes one draw of z per cell, as scvi-tools'
    `get_elbo` does, so it carries Monte Carlo noise.

    Args:
        fit: A fitted model.
        store: The store.
        cells: Store cell indices, `None` for all.
        batch: Batch labels aligned with `cells`.
        seed: Seeds the draw of z.

    Returns:
        `elbo`, `reconstruction` and `kl`, each a mean over cells.
    """
    out = _evaluate(fit, store, cells, batch, seed=seed)
    recon, kl = float(out["reconstruction"].mean()), float(out["kl"].mean())
    return {"elbo": recon + kl, "reconstruction": recon, "kl": kl}
