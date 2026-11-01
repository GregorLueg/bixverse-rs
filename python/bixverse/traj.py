"""Palantir pseudotime, PAGA graph abstraction and gene trends.

Palantir orders cells along differentiation and gives each cell a fate
probability per terminal state, PAGA abstracts the kNN graph into a cluster
graph, and `run_gene_trends` fits a curve per gene along each Palantir branch.
Start with `run_paga` to see the topology, then `run_palantir` with a start
cell (and terminal states, if you know them), then `run_gene_trends` on its
result. Trees over every cell live in `bv.trees`.

**What lands where.** Nothing. All three take a `SingleCells`,
`SingleCellsSubset` or `MetaCells`, read the cached kNN graph (and obs, for
the PAGA clusters) and return a frozen result keyed by cell id. To colour by
pseudotime, join `res.pseudotime` onto obs on `cell_id` with `sc.add_obs`.

```python
import polars as pl

bv.graph.find_clusters(sc, res=0.5, name="leiden_clusters")
paga = bv.traj.run_paga(sc, "leiden_clusters")

early_cell = sc.obs()["cell_id"][0]
res = bv.traj.run_palantir(sc, early_cell)
trends = bv.traj.run_gene_trends(sc, res, features=["gene_0001", "gene_0002"])

obs = sc.obs(cols=["cell_idx", "cell_id"], filtered=True)
sc.add_obs(
    obs.join(res.pseudotime, on="cell_id", how="left").select(
        "cell_idx", pl.col("pseudotime").alias("palantir_pseudotime")
    )
)
```

**Gotchas.** Run `bv.pp.find_hvg`, `bv.pp.calculate_pca` and
`bv.graph.find_neighbours` first. A stale kNN raises; a missing one warns and
`run_palantir` and `run_paga` return `None`. Palantir's start and terminal
cells must be ids the kNN was built over. Without terminal states it detects
them, and `res.terminal_states` names them by cell id. A disconnected kNN graph
is patched with repair edges, counted in `res.run_info`. The kNN metric should
be Euclidean or cosine, or Palantir warns. `run_gene_trends` needs `features`,
or `use_magic=True` after `bv.pp.run_magic` on a `SingleCells` or subset (not on
`MetaCells`). The Gaussian process defaults are prior-dominated, so shorten
`length_scale` before believing a bump. The PAGA graphs are dense arrays, one
row and column per cluster.
"""

from bixverse._core.traj import (
    GeneTrendsRes,
    PagaRes,
    PalantirRes,
    run_gene_trends,
    run_paga,
    run_palantir,
)

__all__ = [
    "GeneTrendsRes",
    "PagaRes",
    "PalantirRes",
    "run_gene_trends",
    "run_paga",
    "run_palantir",
]
