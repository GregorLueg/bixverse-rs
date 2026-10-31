"""kNN searches, neighbour graphs and clustering.

The port of R's `generate_knn_sc`, `find_neighbours_sc`, `find_clusters_sc`,
`fast_cluster_sc` and `calc_knn_metrics`. The work runs in Rust on an
embedding from `sc.cache`; this layer resolves the inputs, writes what R
writes (the kNN and the neighbour graph to `sc.cache`, stamped as derived
from the embedding and the kNN, cluster labels to obs) and returns the rest.

    >>> find_neighbours(sc)
    >>> find_clusters(sc, res=1.0)

R holds the neighbour graph behind a Rust pointer. Here it is the symmetric
CSR in `sc.cache.connectivities`, handed back to Rust for clustering; use
`CsrMatrix.scipy_csr()` to take it elsewhere.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.single_cells import SingleCells
from bixverse._core.state import CsrMatrix, KnnGraph
from bixverse._core.subset import ScOrScSubset
from bixverse.params import FastClusterParams, KnnParams, NeighboursParams, to_rust

###########
# Globals #
###########

# R's `.default_community_iters`.
DEFAULT_COMMUNITY_ITERS: dict[str, int] = {"leiden": 5, "louvain": 2}

ClusterAlgorithm = Literal["leiden", "louvain"]
LocalMoving = Literal["queue", "coloured"]

logger = logging.getLogger(__name__)


###########
# Records #
###########


@beartype
@dataclass(frozen=True)
class KnnMetrics:
    """A kNN compared against a reference, R's `calc_knn_metrics` result.

    Attributes:
        all_matches: Neighbours shared with the reference, per cell.
        all_ratios: Summed query over summed reference distance, per cell.
            Cells whose reference distances sum to about zero are skipped.
        final_recall: Mean recall over all cells.
        final_ratio: Mean of `all_ratios`.
    """

    all_matches: np.ndarray
    all_ratios: np.ndarray
    final_recall: float
    final_ratio: float


@beartype
@dataclass(frozen=True)
class FastClusters:
    """Fast clustering result, R's `SingleCellFastClusters`.

    Attributes:
        memberships: `cell_idx` (0-based) and one community column per
            resolution, `res_<value>`, 0 the largest.
        stats: `resolution`, `mean_ari`, `median_ari`, `mean_conductance`,
            `median_conductance` and `mean_n_comms`; `None` for one seed.
        k_means_cluster: k-means centroid per cell, if asked for.
        centroids: `(n_centroids, n_dims)` centroids, if asked for.
        resolutions: The resolutions, in column order.
    """

    memberships: pl.DataFrame
    stats: pl.DataFrame | None
    k_means_cluster: np.ndarray | None
    centroids: np.ndarray | None
    resolutions: tuple[float, ...]


###########
# Helpers #
###########


@beartype
def _embedding(sc: ScOrScSubset, name: str) -> np.ndarray | None:
    """A cached embedding by name, `"pca"` for the PCA scores."""
    if name == "pca":
        return sc.cache.pca_factors
    return sc.cache.embeddings.get(name)


@beartype
def _first_dims(embd: np.ndarray, no_embd_to_use: int | None) -> np.ndarray:
    """The first `no_embd_to_use` columns as contiguous `float32`."""
    if no_embd_to_use is not None:
        embd = embd[:, : min(no_embd_to_use, embd.shape[1])]
    return np.ascontiguousarray(embd, dtype=np.float32)


@beartype
def _knn_params(params: NeighboursParams | KnnParams) -> dict[str, object]:
    """The dict Rust reads the kNN (and graph) settings from.

    `extract_knn` on `"nndescent"` is the crate's `"nndescent_graph"` method.
    """
    d = to_rust(params)
    if d.pop("extract_knn") and d["knn_method"] == "nndescent":
        d["knn_method"] = "nndescent_graph"
    return d


@beartype
def _knn(
    embd: np.ndarray,
    params: NeighboursParams | KnnParams,
    seed: int,
    validate_index: bool,
    verbose: bool,
) -> KnnGraph:
    """`rs_sc_knn_w_dist` over an embedding, as R's `generate_sc_knn`."""
    indices, distances = _bixverse.knn_with_dist(
        embd, _knn_params(params), validate_index, seed, verbose
    )
    return KnnGraph(indices=indices, distances=distances, dist_metric=params.ann_dist)


@beartype
def _leiden(
    graph: CsrMatrix,
    *,
    resolution: float,
    algorithm: ClusterAlgorithm,
    beta: float,
    n_iterations: int | None,
    min_gain: float,
    local_moving: LocalMoving,
    seed: int,
) -> np.ndarray:
    """Leiden or Louvain on a neighbour graph, as R's `leiden_clustering`.

    Returns:
        `uint32` community per node, 0 the largest.

    Raises:
        ValueError: On a non-positive `resolution` or `beta`, a zero
            `n_iterations`, a negative `min_gain` or `seed`, or a graph too
            large for `uint32` row pointers.
    """
    if resolution <= 0 or beta <= 0:
        raise ValueError("resolution and beta must be positive")
    if n_iterations == 0:
        raise ValueError(
            "n_iterations must be non-zero; negative iterates until stable"
        )
    if min_gain < 0 or seed < 0:
        raise ValueError("min_gain and seed must be non-negative")
    if int(graph.indptr[-1]) > np.iinfo(np.uint32).max:
        raise ValueError("the graph has more edges than uint32 row pointers can hold")
    params = {
        "resolution": resolution,
        "beta": beta,
        "n_iterations": DEFAULT_COMMUNITY_ITERS[algorithm]
        if n_iterations is None
        else n_iterations,
        "min_gain": min_gain,
        "refine": algorithm == "leiden",
        "local_moving": local_moving,
        "seed": seed,
    }
    return _bixverse.leiden(
        graph.data,
        graph.indices,
        np.ascontiguousarray(graph.indptr, dtype=np.uint32),
        params,
    )


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
    neighbours_params: NeighboursParams | KnnParams | None = None,
    seed: int = 42,
    validate_index: bool = True,
    verbose: bool = False,
) -> KnnGraph | None:
    """A kNN over an embedding, without touching the experiment, as R's
    `generate_knn_sc`.

    Exact methods: `"kmknn"` (the default) and `"exhaustive"`. Approximate:
    `"hnsw"`, `"nndescent"` (with `extract_knn=True` the descent graph is
    handed back without a query pass), `"ivf"` and `"annoy"`.

    Args:
        sc: The experiment.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        cells_to_use: Cell ids to restrict the kNN to; they keep the
            embedding's row order. `None` for all kept cells.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        neighbours_params: The kNN parameters. `None` uses the defaults.
        seed: Seed for the index build.
        validate_index: With `verbose`, report the recall of an approximate
            index against an exhaustive search on a random subset.
        verbose: Controls verbosity.

    Returns:
        The kNN, indices into the rows used; `None` (with a warning) if the
        embedding is absent.
    """
    params = neighbours_params or NeighboursParams()
    embd = _embedding(sc, embd_to_use)
    if embd is None:
        warnings.warn(
            "The desired embedding was not found. Please check the parameters. "
            "Returning None.",
            stacklevel=2,
        )
        return None
    if cells_to_use is not None:
        embd = embd[np.isin(sc.cell_ids(filtered=True), list(cells_to_use))]
    return _knn(
        _first_dims(embd, no_embd_to_use), params, seed, validate_index, verbose
    )


@beartype
def calc_knn_metrics(ref_knn: KnnGraph, query_knn: KnnGraph) -> KnnMetrics:
    """Recall at k and distance ratio of a kNN against a reference, as R's
    `calc_knn_metrics`.

    Args:
        ref_knn: The reference, e.g. an exhaustive search.
        query_knn: The kNN to evaluate.

    Raises:
        ValueError: If the two differ in cells or k.
    """
    if ref_knn.indices.shape[0] != query_knn.indices.shape[0]:
        raise ValueError("The two kNN data sets have different number of samples")
    matches, ratios, recall, ratio = _bixverse.knn_metrics(
        ref_knn.indices, ref_knn.distances, query_knn.indices, query_knn.distances
    )
    return KnnMetrics(
        all_matches=matches, all_ratios=ratios, final_recall=recall, final_ratio=ratio
    )


##############
# Neighbours #
##############


@beartype
def find_neighbours(
    sc: ScOrScSubset,
    *,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    neighbours_params: NeighboursParams | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """The kNN and the neighbour graph over an embedding, as R's
    `find_neighbours_sc`.

    The kNN goes to `sc.cache.knn`, stamped as derived from the embedding,
    and its symmetrisation (`knn_sym`: sNN, UMAP fuzzy simplicial set or kNN
    union) to `sc.cache.connectivities`, stamped as derived from `"knn"`.

    Args:
        sc: The experiment.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        neighbours_params: kNN and graph parameters. `None` uses the defaults.
        seed: Seed for the index build.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the embedding is stale.
    """
    params = neighbours_params or NeighboursParams()
    embd = _embedding(sc, embd_to_use)
    if embd is None:
        warnings.warn(
            "The desired embedding was not found. Returning class as is.",
            stacklevel=2,
        )
        return
    state = sc.check_state([embd_to_use])
    if state is not True:
        raise ValueError(state)

    if verbose:
        logger.info("Generating kNN data with %s method.", params.knn_method)
    knn = _knn(_first_dims(embd, no_embd_to_use), params, seed, False, verbose)
    sc.cache.knn = knn
    sc.stamp("knn", parents=(embd_to_use,))

    if verbose:
        logger.info("Generating the %s graph.", params.knn_sym)
    data, indices, indptr = _bixverse.knn_graph(
        knn.indices,
        knn.distances if params.knn_sym == "umap" else None,
        to_rust(params),
        verbose,
    )
    n = knn.indices.shape[0]
    sc.cache.connectivities = CsrMatrix(data, indices, indptr, (n, n))
    sc.stamp("connectivities", parents=("knn",))


@beartype
def get_connectivities(sc: ScOrScSubset) -> CsrMatrix | None:
    """The neighbour graph, R's `get_connectivities` / `as_igraph`.

    The undirected graph as its symmetric CSR, rows sorted, every edge
    stored in both directions. No graph library is pulled in: hand
    `.scipy_csr()` to scipy, or the three vectors to igraph / networkx.

    Returns:
        The graph, or `None` (with a warning) if none was built.
    """
    graph = sc.cache.connectivities
    if graph is None:
        warnings.warn(
            "No neighbour graph found. Did you run find_neighbours()?", stacklevel=2
        )
    return graph


##############
# Clustering #
##############


@beartype
def find_clusters(
    sc: ScOrScSubset,
    *,
    cluster_algorithm: ClusterAlgorithm = "leiden",
    res: float = 1.0,
    name: str = "leiden_clustering",
    seed: int = 42,
    beta: float = 0.01,
    n_iterations: int | None = None,
    min_gain: float = 0.0,
    local_moving: LocalMoving = "queue",
) -> None:
    """Leiden or Louvain on the neighbour graph, as R's `find_clusters_sc`.

    The communities go to obs column `name` as 0-based integers, 0 the
    largest. Depends on `seed` only, not on the thread count.

    Args:
        sc: The experiment, with a neighbour graph over the kept cells.
        cluster_algorithm: `"leiden"` or `"louvain"`.
        res: Modularity resolution; higher gives more communities.
        name: The obs column to write.
        seed: Seed.
        beta: Randomness of the Leiden refinement. Ignored by Louvain.
        n_iterations: Full iterations, each starting from the previous
            partition; negative iterates until stable. `None` is 5 for
            Leiden and 2 for Louvain.
        min_gain: With a negative `n_iterations`, also stop once an
            iteration gains no more than this.
        local_moving: `"queue"` (Traag's sequential queue) or `"coloured"`
            (parallel colour classes on large levels).

    Raises:
        ValueError: If the graph is stale or does not cover the kept cells.

    References:
        Traag, Waltman & van Eck, Sci Rep, 2019; Blondel, et al., J Stat
        Mech, 2008.
    """
    state = sc.check_state(["connectivities"])
    if state is not True:
        raise ValueError(state)
    graph = sc.cache.connectivities
    if graph is None:
        warnings.warn(
            "No neighbour graph found. Did you run find_neighbours(). Returning "
            "class as is.",
            stacklevel=2,
        )
        return
    if graph.shape[0] != sc.n_cells:
        raise ValueError(
            f"The neighbour graph covers {graph.shape[0]} cells but {sc.n_cells} are "
            "kept. Regenerate the neighbours so the two agree."
        )
    labels = _leiden(
        graph,
        resolution=res,
        algorithm=cluster_algorithm,
        beta=beta,
        n_iterations=n_iterations,
        min_gain=min_gain,
        local_moving=local_moving,
        seed=seed,
    )
    sc.add_obs(
        pl.DataFrame(
            {
                "cell_idx": sc.cells_to_keep,
                name: pl.Series(labels, dtype=pl.Int32),
            }
        )
    )


@beartype
def fast_cluster(
    sc: ScOrScSubset,
    *,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    resolutions: Sequence[float] = (2.0, 1.0, 0.5),
    fc_params: FastClusterParams | None = None,
    n_seeds: int = 1,
    return_kmeans: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> FastClusters:
    """k-means coarsening, a centroid graph and Louvain / Leiden on it, as R's
    `fast_cluster_sc`.

    Each cell inherits the community of its centroid. With `n_seeds > 1`
    every resolution runs once per seed; the labels come from the seed with
    the lowest mean conductance and the stability across seeds is returned.
    Does not touch the experiment.

    Args:
        sc: The experiment.
        embd_to_use: The embedding, `"pca"` or a name in `sc.cache.embeddings`.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        resolutions: Modularity resolutions.
        fc_params: The parameters. `None` uses the defaults; a `None`
            `n_centroids` becomes `floor(4 * sqrt(n_cells))` and a `None`
            `n_iterations` 2 for Louvain, 5 for Leiden.
        n_seeds: Seeds per resolution; 1 skips the stability statistics.
        return_kmeans: Return the k-means assignments and centroids.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the embedding is absent, or on an empty resolution
            list or `n_seeds < 1`.

    References:
        Wei, et al., PLoS Comput Biol, 2022.
    """
    params = fc_params or FastClusterParams()
    if not resolutions:
        raise ValueError("resolutions must be non-empty")
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    embd = _embedding(sc, embd_to_use)
    if embd is None:
        raise ValueError(f"Embedding '{embd_to_use}' was not found.")
    embd = _first_dims(embd, no_embd_to_use)

    d = to_rust(params)
    if params.n_centroids is None:
        # faiss' IVF default
        d["n_centroids"] = math.floor(4 * math.sqrt(embd.shape[0]))
    if params.n_iterations is None:
        d["n_iterations"] = DEFAULT_COMMUNITY_ITERS[params.algorithm]
    membership, assignments, centroids, seed_stats = _bixverse.fast_cluster(
        embd, [float(r) for r in resolutions], d, n_seeds, seed, verbose
    )

    cols = [f"res_{float(r):.15g}" for r in resolutions]
    memberships = pl.DataFrame(
        {
            "cell_idx": sc.cells_to_keep,
            **{
                c: pl.Series(membership[:, i], dtype=pl.Int32)
                for i, c in enumerate(cols)
            },
        }
    )
    stats = None
    if seed_stats is not None:
        stats = pl.DataFrame({"resolution": list(resolutions), **seed_stats})
    return FastClusters(
        memberships=memberships,
        stats=stats,
        k_means_cluster=assignments if return_kmeans else None,
        centroids=centroids if return_kmeans else None,
        resolutions=tuple(float(r) for r in resolutions),
    )
