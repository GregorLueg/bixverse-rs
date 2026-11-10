"""2D embeddings: UMAP, densMAP, t-SNE, den-SNE, PHATE and ForceAtlas2.

Start with `umap` on the neighbours from `bv.graph.find_neighbours`. Every
function reads a cached embedding (the PCA by default), fits, and writes the
coordinates to `sc.cache.embeddings` under `slot_name` (`"umap"`, `"densmap"`,
`"tsne"`, `"densne"`, `"phate"`, `"fa2"`), stamped as derived from the input
embedding and, when the cached kNN fed it, from the kNN. Nothing is returned.
All of them take a `SingleCells`, `SingleCellsSubset` or `MetaCells`; on a
`SingleCellsMultiModal`, `modality="wnn"` takes the RNA embedding and the WNN kNN.

The numerics come from the `manifolds-rs` wheel, an optional extra:
`pip install bixverse[embed]`. Importing `bv.embed` without it raises.

Parameters are the `manifolds_rs` estimators themselves. The `*_defaults()`
builders return one set to the bixverse defaults, which differ from the
estimators' own in places (t-SNE perplexity is 10, not 30). Tweak one with
`set_params` and hand it over as `params`.

```python
bv.graph.find_neighbours(sc)
bv.embed.umap(sc)
bv.embed.tsne(sc, params=bv.embed.tsne_defaults().set_params(perplexity=20.0))
umap_xy = sc.cache.embeddings["umap"]
```

UMAP and densMAP take `optimiser="sgd_parallel"`, umap-learn's SGD run
lock-free over all threads (Hogwild!). It is fast and not reproducible: the
same seed gives a different embedding each run. `n_epochs` defaults to `None`,
which is 500, or 200 for the SGD optimisers above 10,000 cells.

**Gotchas.** With `use_knn=True` (the default for UMAP, densMAP and
ForceAtlas2) the cached kNN is reused and the neighbour search settings in
`params` don't apply. A graph wider than `n_neighbors` is cut to it, with a
warning on stderr; a narrower one is used as it is. With no kNN cached you get
a warning and a fresh search. t-SNE, den-SNE and PHATE search afresh unless you
ask for the cached kNN; if you do, keep its `k` near three times the perplexity
for t-SNE, and at `6k + 5` (35 by default) for PHATE. A stale input raises, and
so does a `slot_name` that is reserved (the PCA, kNN and so on). ForceAtlas2
lays out the fuzzy kNN graph; laying out a graph you supply is not possible.
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
