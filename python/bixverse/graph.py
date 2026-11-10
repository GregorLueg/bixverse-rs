"""kNN search, the neighbour graph and clustering.

Start with `find_neighbours` on a PCA (or any cached embedding), then
`find_clusters` for Leiden or Louvain on the resulting graph. The UMAP and the
other embeddings in `bv.embed`, and the integration metrics in `bv.integrate`,
read the same kNN.

**What lands where.** `find_neighbours` writes the kNN to `sc.cache.knn` and
its symmetrised graph to `sc.cache.connectivities`, both stamped, so `sc` prints
what has gone stale. `find_clusters` writes the labels to an obs column, 0-based
with 0 the largest community. `generate_knn`, `fast_cluster`, `calc_knn_metrics`
and `get_connectivities` return their result and leave the experiment alone.
`SingleCells` works everywhere. `SingleCellsSubset` is accepted by all but
`generate_knn`, and its labels go to the subset's obs. `find_neighbours`,
`find_clusters` and `get_connectivities` also take `MetaCells`, and on a
`SingleCellsMultiModal` they reach the ADT and WNN graphs through `modality`.

```python
bv.graph.find_neighbours(sc, neighbours_params=bv.params.NeighboursParams(k=15))
bv.graph.find_clusters(sc, res=1.0, name="leiden_clustering")
```

Big data, or only want rough labels? `fast_cluster` coarsens with k-means,
clusters the centroids and hands each cell the label of its centroid. With
several seeds it also reports how stable the labels are. To check an
approximate index, build an exact `generate_knn` as the reference and compare
with `calc_knn_metrics`.

**Gotchas.** The embedding must exist and be fresh: `find_neighbours` raises on
a stale PCA and warns and returns if the embedding is absent. `find_clusters`
raises if the graph is stale or does not cover the kept cells, so rebuild the
neighbours after moving the cell filter. Resolutions don't carry over between
`knn_sym` choices (`"snn"`, `"umap"`, `"knn"`): retune `res` when you switch.
`bv.integrate.bbknn` overwrites the kNN and the graph. The graph is the
symmetric CSR in `sc.cache.connectivities`; `get_connectivities(sc).scipy_csr()`
takes it to scipy, igraph or networkx.
"""

from bixverse._core.graph import (
    FastClusters,
    KnnMetrics,
    calc_knn_metrics,
    fast_cluster,
    find_clusters,
    find_neighbours,
    generate_knn,
    get_connectivities,
)

__all__ = [
    "FastClusters",
    "KnnMetrics",
    "calc_knn_metrics",
    "fast_cluster",
    "find_clusters",
    "find_neighbours",
    "generate_knn",
    "get_connectivities",
]
