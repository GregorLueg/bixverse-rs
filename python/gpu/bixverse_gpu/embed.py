"""GPU UMAP and t-SNE.

The work runs in the `manifolds_rs` GPU estimators (`UMAPGpu`, `TSNEGpu`);
inputs, slots and stamps are handled exactly as in `bixverse.embed`, which
runs the same code with the CPU estimators. The `*_defaults()` builders return
an estimator set to the defaults the `umap` and `tsne` functions here use.

    >>> bixverse_gpu.graph.find_neighbours(sc)
    >>> umap(sc)
"""

from __future__ import annotations

import manifolds_rs as mf
from beartype import beartype

from bixverse._core.embed import _embed
from bixverse._core.multimodal import Modality
from bixverse._core.subset import ScOrMc

__all__ = [
    "tsne",
    "tsne_defaults",
    "umap",
    "umap_defaults",
]

############
# Defaults #
############


@beartype
def _nn_defaults() -> mf.NeighbourParamsGpu:
    """GPU neighbour parameters: everything left to Rust but `delta`."""
    return mf.NeighbourParamsGpu(delta=0.001)


@beartype
def umap_defaults() -> mf.UMAPGpu:
    """GPU UMAP set to the defaults of `umap`.

    `n_epochs` follows the optimiser; with the default `"adam_gpu"` that is
    always 500.
    """
    return mf.UMAPGpu(
        n_components=2,
        n_neighbors=15,
        min_dist=0.5,
        spread=1.0,
        n_epochs=500,
        learning_rate=1.0,
        init="spectral",
        ann="nndescent_gpu",
        optimiser="adam_gpu",
        randomised=False,
        seed=42,
        nn_params=_nn_defaults(),
        graph_params=mf.UmapGraph(
            bandwidth=1e-5, local_connectivity=1.0, mix_weight=1.0
        ),
        optim_params=mf.UmapOptim(gamma=1.0, neg_sample_rate=5),
    )


@beartype
def tsne_defaults() -> mf.TSNEGpu:
    """GPU t-SNE set to the defaults of `tsne`: perplexity 20, the GPU
    three-kernel FFT optimiser and an IVF kNN."""
    return mf.TSNEGpu(
        n_components=2,
        perplexity=20.0,
        n_epochs=1000,
        learning_rate=None,
        init="pca",
        ann="ivf_gpu",
        approx="fft_3k_gpu",
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


##############
# Embeddings #
##############


@beartype
def umap(
    sc: ScOrMc,
    *,
    use_knn: bool = True,
    embd_to_use: str = "pca",
    slot_name: str = "umap",
    no_embd_to_use: int | None = None,
    modality: Modality = "rna",
    params: mf.UMAPGpu | None = None,
) -> None:
    """GPU UMAP.

    With `use_knn` the cached kNN is reused and only the optimisation runs
    on the GPU; without one cached, a warning and a GPU search.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding, `"pca"` or a name in
            `sc.cache.embeddings`.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        modality: The kNN and the slot: `"rna"`, or `"adt"` / `"wnn"` on a
            `SingleCellsMultiModal`.
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
        modality=modality,
    )


@beartype
def tsne(
    sc: ScOrMc,
    *,
    use_knn: bool = False,
    embd_to_use: str = "pca",
    slot_name: str = "tsne",
    no_embd_to_use: int | None = None,
    modality: Modality = "rna",
    params: mf.TSNEGpu | None = None,
) -> None:
    """GPU t-SNE.

    `approx="fft_3k_gpu"` (the default) runs the optimiser on the GPU in
    fp32; the CPU approximations keep it on the CPU with only the kNN on the
    GPU.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        modality: The kNN and the slot: `"rna"`, or `"adt"` / `"wnn"` on a
            `SingleCellsMultiModal`.
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
        modality=modality,
    )
