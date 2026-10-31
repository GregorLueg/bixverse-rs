"""2D embeddings: UMAP, densMAP, t-SNE, den-SNE, PHATE and ForceAtlas2.

The port of R's `umap_sc`, `densmap_sc`, `tsne_sc`, `densne_sc`, `phate_sc`
and `forceatlas2_sc`. The work runs in the `manifolds-rs` wheel; this layer
resolves the inputs the way R's `.manifold_inputs` does (an embedding from
`sc.cache`, optionally the cached kNN), writes the result to
`sc.cache.embeddings` under R's slot name and stamps it with R's parents:
the input embedding, plus `"knn"` when the cached kNN fed it.

    >>> find_neighbours(sc)
    >>> umap(sc)
    >>> tsne(sc, params=tsne_defaults().set_params(perplexity=20.0))

Parameters are the `manifolds_rs` estimators themselves. The `*_defaults()`
builders return one set to R's defaults (`umap_sc` plus
`manifoldsR::params_umap()` and so on), which differ from the estimators' own
in places, e.g. t-SNE perplexity 10 rather than 30.
"""

from __future__ import annotations

import warnings

import manifolds_rs as mf
import numpy as np
from beartype import beartype

from bixverse._core.state import RESERVED_EMBEDDINGS
from bixverse._core.subset import ScOrScSubset

############
# Defaults #
############


@beartype
def _nn_defaults() -> mf.NeighbourParams:
    """R's `manifoldsR::params_nn()`."""
    return mf.NeighbourParams(
        n_tree=50,
        m=16,
        ef_construction=100,
        ef_search=100,
        diversify_prob=0.0,
        delta=0.001,
        extract_knn=True,
        bt_budget=0.1,
    )


@beartype
def umap_defaults() -> mf.UMAP:
    """UMAP set to R's `umap_sc` and `manifoldsR::params_umap()` defaults.

    R picks `n_epochs` from the optimiser and the cell count when it is
    `NULL`; with its default `"adam_parallel"` that is always 500.
    """
    return mf.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.5,
        spread=1.0,
        n_epochs=500,
        learning_rate=1.0,
        init="spectral",
        ann="kmknn",
        optimiser="adam_parallel",
        randomised=True,
        seed=42,
        nn_params=_nn_defaults(),
        graph_params=mf.UmapGraph(
            bandwidth=1e-5, local_connectivity=1.0, mix_weight=1.0
        ),
        optim_params=mf.UmapOptim(gamma=1.0, neg_sample_rate=5),
    )


@beartype
def densmap_defaults() -> mf.DensMAP:
    """densMAP set to R's `densmap_sc` and `manifoldsR::params_densmap()`
    defaults."""
    u = umap_defaults().get_params()
    return mf.DensMAP(
        **u, lambda_=2.0, dens_params=mf.DensParams(frac=0.3, var_shift=0.1)
    )


@beartype
def tsne_defaults() -> mf.TSNE:
    """t-SNE set to R's `tsne_sc` and `manifoldsR::params_tsne()` defaults."""
    return mf.TSNE(
        n_components=2,
        perplexity=10.0,
        n_epochs=1000,
        learning_rate=None,
        init="pca",
        ann="kmknn",
        approx="bh",
        randomised_init=True,
        seed=42,
        nn_params=_nn_defaults(),
        optim_params=mf.TsneOptim(
            early_exag_iter=250,
            early_exag_factor=12.0,
            theta=0.5,
            n_interp_points=3,
            max_depth=7,
        ),
    )


@beartype
def densne_defaults() -> mf.DensNE:
    """den-SNE set to R's `densne_sc` and `manifoldsR::params_densne()`
    defaults."""
    t = tsne_defaults().get_params()
    return mf.DensNE(
        **t, lambda_=0.5, dens_params=mf.DensParams(frac=0.3, var_shift=0.1)
    )


@beartype
def phate_defaults() -> mf.PHATE:
    """PHATE set to R's `phate_sc` and `manifoldsR::params_phate()` defaults.

    `randomised` and `mds_iter` are not in R's bundle; the estimator's own
    apply.
    """
    return mf.PHATE(
        n_components=2,
        k=5,
        decay=40.0,
        t=None,
        gamma=1.0,
        mds="sgd_dense",
        ann="kmknn",
        seed=42,
        nn_params=_nn_defaults(),
        diffusion_params=mf.PhateDiffusion(
            graph_symmetry="additive",
            n_landmarks=2048,
            landmark_method="spectral",
            t_max=100,
        ),
    )


@beartype
def forceatlas2_defaults() -> mf.ForceAtlas2:
    """ForceAtlas2 set to R's `forceatlas2_sc` and `manifoldsR::params_fa2()`
    defaults."""
    return mf.ForceAtlas2(
        n_neighbors=15,
        n_epochs=500,
        scaling_ratio=2.0,
        gravity=1.0,
        lin_log=False,
        init="spectral",
        ann="kmknn",
        randomised_init=False,
        seed=42,
        nn_params=_nn_defaults(),
        graph_params=mf.UmapGraph(bandwidth=1e-5, local_connectivity=1.0),
        optim_params=mf.Fa2Optim(
            strong_gravity=False,
            dissuade_hubs=False,
            edge_weight_influence=1.0,
            jitter_tolerance=1.0,
            theta=1.2,
        ),
    )


###########
# Helpers #
###########


@beartype
def _embed(
    sc: ScOrScSubset,
    est: mf.BaseEmbedding,
    *,
    use_knn: bool,
    embd_to_use: str,
    slot_name: str,
    no_embd_to_use: int | None,
) -> None:
    """Resolve the inputs as R's `.manifold_inputs`, fit, store and stamp.

    Raises:
        ValueError: On a reserved `slot_name`, a stale input, or an absent
            embedding.
    """
    if slot_name in RESERVED_EMBEDDINGS:
        raise ValueError(f"'{slot_name}' is a reserved artefact name")
    state = sc.check_state([embd_to_use, "knn"] if use_knn else [embd_to_use])
    if state is not True:
        raise ValueError(state)
    embd = (
        sc.cache.pca_factors
        if embd_to_use == "pca"
        else sc.cache.embeddings.get(embd_to_use)
    )
    if embd is None:
        raise ValueError(f"Embedding '{embd_to_use}' was not found.")
    if no_embd_to_use is not None:
        embd = embd[:, : min(no_embd_to_use, embd.shape[1])]
    embd = np.ascontiguousarray(embd, dtype=np.float32)

    knn = sc.cache.knn if use_knn else None
    if use_knn and knn is None:
        warnings.warn(
            "No kNN data found. Attempting to use embeddings for 2D embedding",
            stacklevel=3,
        )
    if knn is None:
        res = est.fit_transform(embd)
    else:
        res = est.fit_transform(
            embd, knn_indices=knn.indices, knn_distances=knn.distances
        )

    sc.cache.embeddings[slot_name] = res
    sc.stamp(
        slot_name, parents=(embd_to_use, "knn") if knn is not None else (embd_to_use,)
    )


##############
# Embeddings #
##############


@beartype
def umap(
    sc: ScOrScSubset,
    *,
    use_knn: bool = True,
    embd_to_use: str = "pca",
    slot_name: str = "umap",
    no_embd_to_use: int | None = None,
    params: mf.UMAP | None = None,
) -> None:
    """UMAP, as R's `umap_sc`.

    With `use_knn` the cached kNN (`find_neighbours`) is reused and the
    neighbour search knobs in `params` do not apply; without one cached, a
    warning and a search over the embedding.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding, `"pca"` or a name in
            `sc.cache.embeddings`.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        params: The estimator. `None` is `umap_defaults()`.

    Raises:
        ValueError: If the embedding or the kNN is stale, the embedding is
            absent, or `slot_name` is reserved.
    """
    _embed(
        sc,
        params or umap_defaults(),
        use_knn=use_knn,
        embd_to_use=embd_to_use,
        slot_name=slot_name,
        no_embd_to_use=no_embd_to_use,
    )


@beartype
def densmap(
    sc: ScOrScSubset,
    *,
    use_knn: bool = True,
    embd_to_use: str = "pca",
    slot_name: str = "densmap",
    no_embd_to_use: int | None = None,
    params: mf.DensMAP | None = None,
) -> None:
    """densMAP, UMAP with a density-preserving term, as R's `densmap_sc`.

    Inputs are handled as in `umap`; `lambda_=0` gives back plain UMAP.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        params: The estimator. `None` is `densmap_defaults()`.

    Raises:
        ValueError: As `umap`.

    References:
        Narayan, Berger & Cho, Nat Biotechnol, 2021.
    """
    _embed(
        sc,
        params or densmap_defaults(),
        use_knn=use_knn,
        embd_to_use=embd_to_use,
        slot_name=slot_name,
        no_embd_to_use=no_embd_to_use,
    )


@beartype
def tsne(
    sc: ScOrScSubset,
    *,
    use_knn: bool = False,
    embd_to_use: str = "pca",
    slot_name: str = "tsne",
    no_embd_to_use: int | None = None,
    params: mf.TSNE | None = None,
) -> None:
    """t-SNE, as R's `tsne_sc`.

    Off by default, `use_knn` reuses the cached kNN; perplexity then only
    shapes the affinities over it, so keep its k near `3 * perplexity`.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        params: The estimator. `None` is `tsne_defaults()`.

    Raises:
        ValueError: As `umap`.
    """
    _embed(
        sc,
        params or tsne_defaults(),
        use_knn=use_knn,
        embd_to_use=embd_to_use,
        slot_name=slot_name,
        no_embd_to_use=no_embd_to_use,
    )


@beartype
def densne(
    sc: ScOrScSubset,
    *,
    use_knn: bool = False,
    embd_to_use: str = "pca",
    slot_name: str = "densne",
    no_embd_to_use: int | None = None,
    params: mf.DensNE | None = None,
) -> None:
    """den-SNE, t-SNE with a density-preserving term, as R's `densne_sc`.

    Inputs are handled as in `tsne`; `lambda_=0` gives back plain t-SNE.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        params: The estimator. `None` is `densne_defaults()`.

    Raises:
        ValueError: As `umap`.

    References:
        Narayan, Berger & Cho, Nat Biotechnol, 2021.
    """
    _embed(
        sc,
        params or densne_defaults(),
        use_knn=use_knn,
        embd_to_use=embd_to_use,
        slot_name=slot_name,
        no_embd_to_use=no_embd_to_use,
    )


@beartype
def phate(
    sc: ScOrScSubset,
    *,
    use_knn: bool = True,
    embd_to_use: str = "pca",
    slot_name: str = "phate",
    no_embd_to_use: int | None = None,
    params: mf.PHATE | None = None,
) -> None:
    """PHATE, as R's `phate_sc`.

    Inputs are handled as in `umap`.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        params: The estimator. `None` is `phate_defaults()`.

    Raises:
        ValueError: As `umap`.

    References:
        Moon, et al., Nat Biotechnol, 2019.
    """
    _embed(
        sc,
        params or phate_defaults(),
        use_knn=use_knn,
        embd_to_use=embd_to_use,
        slot_name=slot_name,
        no_embd_to_use=no_embd_to_use,
    )


@beartype
def forceatlas2(
    sc: ScOrScSubset,
    *,
    use_knn: bool = True,
    embd_to_use: str = "pca",
    slot_name: str = "fa2",
    no_embd_to_use: int | None = None,
    params: mf.ForceAtlas2 | None = None,
) -> None:
    """ForceAtlas2 on the fuzzy kNN graph, as R's `forceatlas2_sc` with
    `graph = "knn"`.

    The kNN (cached with `use_knn`, else searched) becomes UMAP's fuzzy union
    graph first, as in scanpy's `draw_graph`. R's `graph = "connectivities"`
    is not here: the `manifolds-rs` wheel cannot lay out a given graph.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        params: The estimator. `None` is `forceatlas2_defaults()`.

    Raises:
        ValueError: As `umap`.

    References:
        Jacomy, et al., PLoS ONE, 2014.
    """
    _embed(
        sc,
        params or forceatlas2_defaults(),
        use_knn=use_knn,
        embd_to_use=embd_to_use,
        slot_name=slot_name,
        no_embd_to_use=no_embd_to_use,
    )
