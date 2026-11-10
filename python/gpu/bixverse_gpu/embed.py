"""GPU UMAP, densMAP, t-SNE and PHATE.

The work runs in the `manifolds_rs` GPU estimators (`UMAPGpu`, `DensMAPGpu`,
`TSNEGpu`, `PHATEGpu`); inputs, slots and stamps are handled exactly as in
`bixverse.embed`, which runs the same code with the CPU estimators. The
`*_defaults()` builders return an estimator set to the defaults the functions
here use. There is no GPU den-SNE.

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
    "densmap",
    "densmap_defaults",
    "phate",
    "phate_defaults",
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

    `n_epochs=None` is 500 under the default `"adam_gpu"`.
    """
    return mf.UMAPGpu(
        n_components=2,
        n_neighbors=15,
        min_dist=0.5,
        spread=1.0,
        n_epochs=None,
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
def densmap_defaults() -> mf.DensMAPGpu:
    """GPU densMAP set to the defaults of `densmap`."""
    return mf.DensMAPGpu(
        **umap_defaults().get_params(),
        lambda_=2.0,
        dens_params=mf.DensParams(frac=0.3, var_shift=0.1),
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
        randomised=True,
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
def phate_defaults() -> mf.PHATEGpu:
    """GPU PHATE set to the defaults of `phate`: the CPU settings with an
    NN-Descent GPU kNN. The diffusion and the MDS stay on the CPU."""
    return mf.PHATEGpu(
        n_components=2,
        k=5,
        decay=40.0,
        t=None,
        gamma=1.0,
        n_landmarks=2048,
        landmark_method="spectral",
        mds="sgd_dense",
        ann="nndescent_gpu",
        seed=42,
        nn_params=_nn_defaults(),
        diffusion_params=mf.PhateDiffusion(graph_symmetry="additive", t_max=100),
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


@beartype
def densmap(
    sc: ScOrMc,
    *,
    use_knn: bool = True,
    embd_to_use: str = "pca",
    slot_name: str = "densmap",
    no_embd_to_use: int | None = None,
    modality: Modality = "rna",
    params: mf.DensMAPGpu | None = None,
) -> None:
    """GPU densMAP.

    Inputs are handled as in `umap`; with `"adam_gpu"` the density term runs
    on the device alongside the rest of the update.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        modality: The kNN and the slot: `"rna"`, or `"adt"` / `"wnn"` on a
            `SingleCellsMultiModal`.
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
        modality=modality,
    )


@beartype
def phate(
    sc: ScOrMc,
    *,
    use_knn: bool = False,
    embd_to_use: str = "pca",
    slot_name: str = "phate",
    no_embd_to_use: int | None = None,
    modality: Modality = "rna",
    params: mf.PHATEGpu | None = None,
) -> None:
    """GPU PHATE.

    The kNN search and, for spectral landmarks, the randomised SVD and
    k-means run on the device; the diffusion and the MDS on the CPU. Off by
    default, `use_knn` reuses the cached kNN, which should then be `6k + 5`
    wide, as in `bixverse.embed.phate`.

    Args:
        sc: The experiment.
        use_knn: Reuse the cached kNN.
        embd_to_use: The input embedding.
        slot_name: The name in `sc.cache.embeddings` to write.
        no_embd_to_use: Number of leading dimensions to use. `None` for all.
        modality: The kNN and the slot: `"rna"`, or `"adt"` / `"wnn"` on a
            `SingleCellsMultiModal`.
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
        modality=modality,
    )
