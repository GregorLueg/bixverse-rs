"""Neural network models trained straight off a bixverse store.

scVI (Lopez et al., 2018) for batch correction, on torch or on MLX. Counts
stream from the store through the Rust loader; nothing goes through AnnData.
Importing this module needs neither framework; training on one needs its
extra, `bixverse[torch]` or `bixverse[mlx]`.

    >>> import bixverse.nn as bvnn
    >>> fit = bvnn.train_scvi("counts_cells.bin", cells, hvg, batch)
    >>> latent = bvnn.scvi_latent(fit, "counts_cells.bin", cells, batch)
"""

from ._api import scvi_elbo, scvi_latent, train_scvi
from ._params import ScviFit, ScviParams

__all__ = ["ScviFit", "ScviParams", "scvi_elbo", "scvi_latent", "train_scvi"]
