"""kNN searches, neighbour graphs and clustering.

The work runs in Rust on an embedding from `sc.cache`; this layer resolves
the inputs, writes the kNN and the neighbour graph to `sc.cache` (stamped as
derived from the embedding and the kNN) and the cluster labels to obs, and
returns the rest.
`find_neighbours`, `find_clusters` and `get_connectivities` also take
`MetaCells`, whose labels go to its obs in row order.

    >>> find_neighbours(sc)
    >>> find_clusters(sc, res=1.0)

The neighbour graph is the symmetric CSR in `sc.cache.connectivities`,
handed back to Rust for clustering; use `CsrMatrix.scipy_csr()` to take it
elsewhere.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.metacells import MetaCells
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
from bixverse.params import (
    FastClusterParams,
    KnnParams,
    NeighboursParams,
    WnnParams,
    to_rust,
)

###########
# Globals #
###########

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
    """A kNN compared against a reference.

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
    """Fast clustering result.

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
def _embedding(sc: ScOrMc, name: str, modality: Modality = "rna") -> np.ndarray | None:
    """A cached embedding of a modality by name, `"pca"` for the PCA scores."""
    cache = modality_cache(sc, modality)
    if name == "pca":
        return cache.pca_factors
    return cache.embeddings.get(name)


@beartype
def _first_dims(embd: np.ndarray, no_embd_to_use: int | None) -> np.ndarray:
    """The first `no_embd_to_use` columns as contiguous `float32`."""
    if no_embd_to_use is not None:
        embd = embd[:, : min(no_embd_to_use, embd.shape[1])]
    return np.ascontiguousarray(embd, dtype=np.float32)


@beartype
def _knn_params(params: NeighboursParams | KnnParams | WnnParams) -> dict[str, object]:
    """The dict Rust reads the kNN (and graph, or WNN) settings from.

    `extract_knn` is dropped: `"nndescent"` is graph-only in the crate.
    """
    d = to_rust(params)
    d.pop("extract_knn", None)
    return d


@beartype
def _knn(
    embd: np.ndarray,
    params: NeighboursParams | KnnParams,
    seed: int,
    validate_index: bool,
    verbose: bool,
) -> KnnGraph:
    """A kNN with distances over an embedding."""
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
    run: Callable[..., np.ndarray] = _bixverse.leiden,
) -> np.ndarray:
    """Leiden or Louvain on a neighbour graph.

    `run` is the extension call, `leiden` of `bixverse._bixverse` or of its
    GPU twin.

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
    return run(
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
    """A kNN over an embedding, without touching the experiment.

    Exact methods: `"kmknn"` (the default) and `"exhaustive"`. Approximate:
    `"hnsw"`, `"nndescent"` (the descent graph, handed back without a query
    pass), `"ivf"` and `"annoy"`.

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

    Examples:
        >>> knn = bv.graph.generate_knn(sc)
        >>> knn_hnsw = bv.graph.generate_knn(
        ...     sc, neighbours_params=bv.params.NeighboursParams(knn_method="hnsw")
        ... )
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
    """Recall at k and distance ratio of a kNN against a reference.

    Args:
        ref_knn: The reference, e.g. an exhaustive search.
        query_knn: The kNN to evaluate.

    Raises:
        ValueError: If the two differ in cells or k.

    Examples:
        >>> ref_knn = bv.graph.generate_knn(sc, no_embd_to_use=20)
        >>> query_knn = bv.graph.generate_knn(
        ...     sc,
        ...     no_embd_to_use=20,
        ...     neighbours_params=bv.params.NeighboursParams(knn_method="annoy"),
        ... )
        >>> metrics = bv.graph.calc_knn_metrics(ref_knn, query_knn)
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
    sc: ScOrMc,
    *,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    modality: CacheModality = "rna",
    neighbours_params: NeighboursParams | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """The kNN and the neighbour graph over an embedding.

    The kNN goes to the modality's cache (`sc.cache` for RNA), stamped as
    derived from the embedding, and its symmetrisation (`knn_sym`: sNN, UMAP
    fuzzy simplicial set or kNN union) to its `connectivities`, stamped as
    derived from `"knn"`.

    Args:
        sc: The experiment.
        embd_to_use: The embedding, `"pca"` or a name in the cache's
            `embeddings`.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        modality: `"rna"`, or `"adt"` on a `SingleCellsMultiModal`.
        neighbours_params: kNN and graph parameters. `None` uses the defaults.
        seed: Seed for the index build.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the embedding is stale, or on `"adt"` without a
            multi-modal object.

    Examples:
        >>> bv.graph.find_neighbours(sc, no_embd_to_use=20)
        >>> bv.graph.find_neighbours(
        ...     sc, neighbours_params=bv.params.NeighboursParams(k=30, knn_sym="umap")
        ... )
    """
    params = neighbours_params or NeighboursParams()
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
        logger.info("Generating kNN data with %s method.", params.knn_method)
    knn = _knn(_first_dims(embd, no_embd_to_use), params, seed, False, verbose)
    cache.knn = knn
    stamp(sc, "knn", parents=(embd_to_use,), modality=modality)

    if verbose:
        logger.info("Generating the %s graph.", params.knn_sym)
    data, indices, indptr = _bixverse.knn_graph(
        knn.indices,
        knn.distances if params.knn_sym == "umap" else None,
        to_rust(params),
        verbose,
    )
    n = knn.indices.shape[0]
    cache.connectivities = CsrMatrix(data, indices, indptr, (n, n))
    stamp(sc, "connectivities", parents=("knn",), modality=modality)


@beartype
def get_connectivities(sc: ScOrMc, *, modality: Modality = "rna") -> CsrMatrix | None:
    """The neighbour graph.

    The undirected graph as its symmetric CSR, rows sorted, every edge
    stored in both directions. No graph library is pulled in: hand
    `.scipy_csr()` to scipy, or the three vectors to igraph / networkx.

    Args:
        sc: The experiment.
        modality: `"rna"`, or `"adt"` / `"wnn"` on a `SingleCellsMultiModal`.

    Returns:
        The graph, or `None` (with a warning) if none was built.
    """
    graph = modality_cache(sc, modality).connectivities
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
    """Leiden or Louvain on the neighbour graph.

    The communities go to obs column `name` as 0-based integers, 0 the
    largest. Depends on `seed` only, not on the thread count.

    Args:
        sc: The experiment, with a neighbour graph over the kept cells.
        cluster_algorithm: `"leiden"` or `"louvain"`.
        res: Modularity resolution; higher gives more communities.
        name: The obs column to write.
        modality: The graph: `"rna"`, or `"adt"` / `"wnn"` on a
            `SingleCellsMultiModal`.
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

    Examples:
        >>> bv.graph.find_clusters(sc, res=0.5, name="leiden_05")
        >>> bv.graph.find_clusters(sc, cluster_algorithm="louvain", name="louvain")
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
        run=_bixverse.leiden,
    )


@beartype
def _find_clusters(
    sc: ScOrMc,
    *,
    cluster_algorithm: ClusterAlgorithm,
    res: float,
    name: str,
    modality: Modality,
    seed: int,
    beta: float,
    n_iterations: int | None,
    min_gain: float,
    local_moving: LocalMoving,
    run: Callable[..., np.ndarray],
) -> None:
    """The body of `find_clusters`, with the extension call `run` supplied,
    so the GPU twin shares it."""
    require_fresh(sc, ["connectivities"], modality)
    graph = modality_cache(sc, modality).connectivities
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
        run=run,
    )
    labels = pl.Series(name, labels, dtype=pl.Int32)
    if isinstance(sc, MetaCells):
        sc.add_obs(labels.to_frame())
        return
    sc.add_obs(pl.DataFrame({"cell_idx": sc.cells_to_keep}).with_columns(labels))


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
    """k-means coarsening, a centroid graph and Louvain / Leiden on it.

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

    Examples:
        >>> fast = bv.graph.fast_cluster(sc, resolutions=(1.0, 0.5))
        >>> memberships = fast.memberships
        >>> fast = bv.graph.fast_cluster(sc, n_seeds=3, return_kmeans=True)
    """
    return _fast_cluster(
        sc,
        embd_to_use=embd_to_use,
        no_embd_to_use=no_embd_to_use,
        resolutions=resolutions,
        params=to_rust(fc_params or FastClusterParams()),
        n_seeds=n_seeds,
        return_kmeans=return_kmeans,
        seed=seed,
        verbose=verbose,
        run=_bixverse.fast_cluster,
    )


@beartype
def _fast_cluster(
    sc: ScOrScSubset,
    *,
    embd_to_use: str,
    no_embd_to_use: int | None,
    resolutions: Sequence[float],
    params: dict[str, object],
    n_seeds: int,
    return_kmeans: bool,
    seed: int,
    verbose: bool,
    run: Callable[..., tuple],
) -> FastClusters:
    """The body of `fast_cluster` over `to_rust(params)`, with the extension
    call `run` supplied, so the GPU twin shares it."""
    if not resolutions:
        raise ValueError("resolutions must be non-empty")
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    embd = _embedding(sc, embd_to_use)
    if embd is None:
        raise ValueError(f"Embedding '{embd_to_use}' was not found.")
    embd = _first_dims(embd, no_embd_to_use)

    d = dict(params)
    if d["n_centroids"] is None:
        # faiss' IVF default
        d["n_centroids"] = math.floor(4 * math.sqrt(embd.shape[0]))
    if d["n_iterations"] is None:
        d["n_iterations"] = DEFAULT_COMMUNITY_ITERS[str(d["algorithm"])]
    membership, assignments, centroids, seed_stats = run(
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
