"""Neural network models trained straight off a bixverse store.

scVI (Lopez et al., 2018) for batch correction, on torch, MLX or JAX. Counts
stream from the store through the Rust loader; nothing goes through AnnData.
Importing this module needs no framework; training on one needs its extra,
`bixverse[torch]`, `bixverse[mlx]` or `bixverse[jax]`.

    >>> import bixverse.nn as bvnn
    >>> fit = bvnn.train_scvi("counts_cells.bin", cells, hvg, batch)
    >>> latent = bvnn.scvi_latent(fit, "counts_cells.bin", cells, batch)
"""

from ._api import scvi_elbo, scvi_latent, train_scvi
from ._params import ScviFit, ScviHistory, ScviParams

__all__ = [
    "ScviFit",
    "ScviHistory",
    "ScviParams",
    "scvi_elbo",
    "scvi_latent",
    "train_scvi",
]
