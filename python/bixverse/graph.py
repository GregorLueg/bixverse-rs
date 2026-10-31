"""Neighbours and clustering, the port of the bixverse R kNN, neighbour graph
and clustering helpers.

Functions take a `SingleCells` first and write what R writes: the kNN and the
neighbour graph to `sc.cache`, stamped, cluster labels to obs. `generate_knn`,
`fast_cluster` and `calc_knn_metrics` return data and leave the experiment
alone. All but `generate_knn` also take a `SingleCellsSubset`, as in R;
its labels go to the subset's obs. `find_neighbours`, `find_clusters` and
`get_connectivities` also take `MetaCells`, as in R.
See `bixverse._core.graph` for the workflow.
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
