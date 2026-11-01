"""GPU kNN, neighbour graphs and clustering.

The port of R's `generate_gpu_knn_sc`, `find_neighbours_gpu_sc`,
`find_clusters_gpu_sc` and `fast_cluster_gpu_sc`. They write what their
`bixverse.graph` twins write: the kNN and the neighbour graph to the cache,
stamped, the cluster labels to obs. The kNN runs on the GPU; the graph built
from it is the CPU one, as in R.

    >>> find_neighbours(sc)
    >>> find_clusters(sc, res=1.0)
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.graph import (
    ClusterAlgorithm,
    FastClusters,
    LocalMoving,
    _embedding,
    _fast_cluster,
    _find_clusters,
    _first_dims,
)
from bixverse._core.multimodal import (
    CacheModality,
    Modality,
    modality_cache,
    require_fresh,
    stamp,
)
from bixverse._core.single_cells import SingleCells
from bixverse._core.state import CsrMatrix, KnnGraph
from bixverse._core.subset import ScOrMc, ScOrScSubset
from bixverse.params import KnnGraphDefaults, to_rust
from bixverse_gpu import _bixverse_gpu
from bixverse_gpu.params import FastClusterGpuParams, KnnGpuDefaults

__all__ = [
    "fast_cluster",
    "find_clusters",
    "find_neighbours",
    "generate_knn",
]

logger = logging.getLogger(__name__)


###########
# Helpers #
###########


@beartype
def _knn_params(params: KnnGpuDefaults) -> dict[str, object]:
    """The dict Rust reads the GPU kNN from: `extract_knn` on `"nndescent"`
    is the crate's `"nndescent_graph"` method, as on the CPU side."""
    d = to_rust(params)
    if d.pop("extract_knn") and d["knn_method"] == "nndescent":
        d["knn_method"] = "nndescent_graph"
    return d


@beartype
def _knn(
    embd: np.ndarray, params: KnnGpuDefaults, seed: int, verbose: bool
) -> KnnGraph:
    """`knn_with_dist` on the GPU over an embedding."""
    indices, distances = _bixverse_gpu.knn_with_dist(
        embd, _knn_params(params), seed, verbose
    )
    return KnnGraph(indices=indices, distances=distances, dist_metric=params.ann_dist)


#######
# kNN #
#######


@beartype
def generate_knn(
    sc: SingleCells,
    *,
    embd_to_use: str = "pca",
    cells_to_use: Sequence[str] | None = None,
    no_embd_to_use: int | None = None,
    knn_params: KnnGpuDefaults | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> KnnGraph | None:
    """A GPU kNN over an embedding, without touching the experiment, as R's
    `generate_gpu_knn_sc`.

    Methods: `"exhaustive"` (exact, the default), `"ivf"` and `"nndescent"`
    (a CAGRA graph, beam searched, or with `extract_knn=True` handed back as
    the descent left it). Euclidean distances are true L2.

    Args:
        sc: The experiment.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        cells_to_use: Cell ids to restrict the kNN to; they keep the
            embedding's row order. `None` for all kept cells.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        knn_params: The GPU kNN parameters, `k` included. `None` uses the
            defaults.
        seed: Seed for the index build.
        verbose: Controls verbosity.

    Returns:
        The kNN, indices into the rows used; `None` (with a warning) if the
        embedding is absent.

    Raises:
        BixverseError: Without a GPU adapter.
    """
    params = knn_params or KnnGpuDefaults()
    embd = _embedding(sc, embd_to_use)
    if embd is None:
        warnings.warn(
            "The desired embedding was not found. Returning None.", stacklevel=2
        )
        return None
    if cells_to_use is not None:
        embd = embd[np.isin(sc.cell_ids(filtered=True), list(cells_to_use))]
    return _knn(_first_dims(embd, no_embd_to_use), params, seed, verbose)


@beartype
def find_neighbours(
    sc: ScOrMc,
    *,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    modality: CacheModality = "rna",
    knn_params: KnnGpuDefaults | None = None,
    graph_params: KnnGraphDefaults | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """The GPU kNN and its neighbour graph over an embedding, as R's
    `find_neighbours_gpu_sc`.

    The kNN goes to the modality's cache, stamped as derived from the
    embedding, and its symmetrisation (`knn_sym`) to its `connectivities`,
    stamped as derived from `"knn"`, as `bixverse.graph.find_neighbours`.

    Args:
        sc: The experiment.
        embd_to_use: The embedding, `"pca"` or a name in the cache's
            `embeddings`.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        modality: `"rna"`, or `"adt"` on a `SingleCellsMultiModal`.
        knn_params: The GPU kNN parameters. `None` uses the defaults.
        graph_params: How the kNN becomes a graph. `None` uses the defaults.
        seed: Seed for the index build.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the embedding is stale.
        BixverseError: Without a GPU adapter.
    """
    params = knn_params or KnnGpuDefaults()
    graph = graph_params or KnnGraphDefaults()
    cache = modality_cache(sc, modality)
    embd = _embedding(sc, embd_to_use, modality)
    if embd is None:
        warnings.warn(
            "The desired embedding was not found. Returning class as is.",
            stacklevel=2,
        )
        return
    require_fresh(sc, [embd_to_use], modality)

    if verbose:
        logger.info("Generating GPU kNN data with %s method.", params.knn_method)
    knn = _knn(_first_dims(embd, no_embd_to_use), params, seed, verbose)
    cache.knn = knn
    stamp(sc, "knn", parents=(embd_to_use,), modality=modality)

    if verbose:
        logger.info("Generating the %s graph.", graph.knn_sym)
    data, indices, indptr = _bixverse.knn_graph(
        knn.indices,
        knn.distances if graph.knn_sym == "umap" else None,
        to_rust(graph),
        verbose,
    )
    n = knn.indices.shape[0]
    cache.connectivities = CsrMatrix(data, indices, indptr, (n, n))
    stamp(sc, "connectivities", parents=("knn",), modality=modality)


##############
# Clustering #
##############


@beartype
def find_clusters(
    sc: ScOrMc,
    *,
    cluster_algorithm: ClusterAlgorithm = "leiden",
    res: float = 1.0,
    name: str = "leiden_clustering",
    modality: Modality = "rna",
    seed: int = 42,
    beta: float = 0.01,
    n_iterations: int | None = None,
    min_gain: float = 0.0,
    local_moving: LocalMoving = "queue",
) -> None:
    """Leiden or Louvain on the neighbour graph, level-0 local moving on the
    GPU, as R's `find_clusters_gpu_sc`.

    Writes what `bixverse.graph.find_clusters` writes. Graphs below 50,000
    nodes, or with a node of very high degree, run fully on the CPU with
    coloured local moving, so the labels can differ from the CPU twin's
    queue-based ones. On one device the result depends on the seed only.

    Args:
        sc: The experiment, with a neighbour graph over the kept cells.
        cluster_algorithm: `"leiden"` or `"louvain"`.
        res: Modularity resolution; higher gives more communities.
        name: The obs column to write.
        modality: The graph: `"rna"`, or `"adt"` / `"wnn"` on a
            `SingleCellsMultiModal`.
        seed: Seed.
        beta: Randomness of the Leiden refinement. Ignored by Louvain.
        n_iterations: Full iterations; negative iterates until stable.
            `None` is 5 for Leiden and 2 for Louvain.
        min_gain: With a negative `n_iterations`, also stop once an
            iteration gains no more than this.
        local_moving: Local moving on the coarser levels, `"queue"` or
            `"coloured"`.

    Raises:
        ValueError: If the graph is stale or does not cover the kept cells.
        BixverseError: Without a GPU adapter that runs plane operations.

    References:
        Traag, Waltman & van Eck, Sci Rep, 2019.
    """
    _find_clusters(
        sc,
        cluster_algorithm=cluster_algorithm,
        res=res,
        name=name,
        modality=modality,
        seed=seed,
        beta=beta,
        n_iterations=n_iterations,
        min_gain=min_gain,
        local_moving=local_moving,
        run=_bixverse_gpu.leiden,
    )


@beartype
def fast_cluster(
    sc: ScOrScSubset,
    *,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    resolutions: Sequence[float] = (2.0, 1.0, 0.5),
    fc_params: FastClusterGpuParams | None = None,
    n_seeds: int = 1,
    return_kmeans: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> FastClusters:
    """k-means coarsening on the GPU, a centroid graph and Louvain / Leiden
    on it, as R's `fast_cluster_gpu_sc`.

    Returns what `bixverse.graph.fast_cluster` returns. Only the k-means (and
    the cell-mediated top-m search) run on the GPU; it is full-batch Lloyd's,
    so there are no mini-batch knobs.

    Args:
        sc: The experiment.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        resolutions: Modularity resolutions.
        fc_params: The parameters. `None` uses the defaults; a `None`
            `n_centroids` becomes `floor(4 * sqrt(n_cells))`.
        n_seeds: Seeds per resolution; 1 skips the stability statistics.
        return_kmeans: Return the k-means assignments and centroids.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the embedding is absent, or on an empty resolution
            list or `n_seeds < 1`.
        BixverseError: Without a GPU adapter.

    References:
        Wei, et al., PLoS Comput Biol, 2022.
    """
    return _fast_cluster(
        sc,
        embd_to_use=embd_to_use,
        no_embd_to_use=no_embd_to_use,
        resolutions=resolutions,
        params=to_rust(fc_params or FastClusterGpuParams()),
        n_seeds=n_seeds,
        return_kmeans=return_kmeans,
        seed=seed,
        verbose=verbose,
        run=_bixverse_gpu.fast_cluster,
    )
