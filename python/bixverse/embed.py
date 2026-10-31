"""2D embeddings, the port of the bixverse R UMAP, densMAP, t-SNE, den-SNE,
PHATE and ForceAtlas2 helpers.

Functions take a `SingleCells` first, write the embedding to
`sc.cache.embeddings` under R's name and stamp it. The numerics come from the
`manifolds-rs` wheel, an optional extra: `pip install bixverse[embed]`. See
`bixverse._core.embed` for the workflow. Every function also takes a
`SingleCellsSubset`, as in R.
"""

try:
    import manifolds_rs  # noqa: F401
except ImportError as e:
    raise ImportError(
        "bixverse.embed needs manifolds-rs: `pip install bixverse[embed]`"
    ) from e

from bixverse._core.embed import (
    densmap,
    densmap_defaults,
    densne,
    densne_defaults,
    forceatlas2,
    forceatlas2_defaults,
    phate,
    phate_defaults,
    tsne,
    tsne_defaults,
    umap,
    umap_defaults,
)

__all__ = [
    "densmap",
    "densmap_defaults",
    "densne",
    "densne_defaults",
    "forceatlas2",
    "forceatlas2_defaults",
    "phate",
    "phate_defaults",
    "tsne",
    "tsne_defaults",
    "umap",
    "umap_defaults",
]
