"""Highly variable genes, PCA, residual models and MAGIC.

Everything here reads the count stores in Rust and works on the cells you kept.
Start with `find_hvg`, then `calculate_pca`: the PCA uses the HVG set that
`find_hvg` stored. The kNN graph, clustering and embeddings live in `bv.graph`
and `bv.embed`.

**What lands where.** `find_hvg` writes the per-gene statistics to var and the
selected genes to `sc.map.hvg`. `calculate_pca` writes scores, loadings and
singular values to `sc.cache`. `fit_residuals` writes the fitted model to
`sc.cache.residual_fit`, which `find_hvg` (with `method="residual"`),
`calculate_pca(residuals=True)` and `sct_corrected_counts` then read.
`run_magic` writes the imputed genes to `sc.cache.magic`. Every cached result is
stamped, so `sc` prints what has gone stale after the cell filter moves.
`get_hvg_data`, `find_hvg_batch_aware` and `sct_corrected_counts` leave the
experiment alone and return their result.

**Object types.** `SingleCells` works everywhere. A `SingleCellsSubset` is
accepted by all functions except `get_hvg_data`; on a subset `find_hvg` stores
the selection but leaves var alone. `MetaCells` is accepted by `find_hvg`,
`get_hvg_data`, `calculate_pca` and `fit_residuals`, and computes on its
in-memory counts.

```python
bv.pp.find_hvg(sc, hvg_no=2000)
bv.pp.calculate_pca(sc, no_pcs=30)
```

The residual route fits a model once and reuses it for the HVGs and the PCA:

```python
bv.pp.fit_residuals(sc, method="analytic_pearson")
bv.pp.find_hvg(sc, hvg_params=bv.params.HvgParams(method="residual"))
bv.pp.calculate_pca(
    sc, 30, pca_params=bv.params.PcaParams(normalise_variance=False), residuals=True
)
```

**Gotchas.** `calculate_pca` without HVGs warns and returns, so run `find_hvg`
first or pass `hvg=`. A residual model is bound to the current cell set: change
the filter and `find_hvg` or `calculate_pca` refuse it until you refit. With
`method="residual"`, `hvg_no` counts per group and the union is kept. For
several batches use `find_hvg_batch_aware` and hand its `hvg_gene_idx` to
`calculate_pca`. `run_magic` needs a kNN graph over the kept cells from
`bv.graph.find_neighbours`; do not feed its output to correlation-based methods,
the smoothing inflates gene-gene correlation.
"""

from bixverse._core.pp import (
    BatchHvg,
    CorrectedCounts,
    ResidualFit,
    calculate_pca,
    find_hvg,
    find_hvg_batch_aware,
    fit_residuals,
    get_hvg_data,
    run_magic,
    sct_corrected_counts,
)

__all__ = [
    "BatchHvg",
    "CorrectedCounts",
    "ResidualFit",
    "calculate_pca",
    "find_hvg",
    "find_hvg_batch_aware",
    "fit_residuals",
    "get_hvg_data",
    "run_magic",
    "sct_corrected_counts",
]
