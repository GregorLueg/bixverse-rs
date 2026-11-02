"""Single cell pipelines: steps chained with `>>` and applied in order.

A `Step` is a function, the keyword
arguments it gets at apply time and the classes it accepts; a `Pipeline` is a
tuple of steps. `apply_pipeline` checks the chain against the class of the
object first (a meta cell step turns a `SingleCells` into a `MetaCells`),
then calls each function with the object first. Errors propagate.

    >>> p = step_hvg(hvg_no=2000) >> step_pca(30) >> step_neighbours()
    >>> p = p >> step_clusters(res=0.5)
    >>> apply_pipeline(p, sc)
    >>> per_group = apply_pipeline_per_group(p, sc, "sample")

Any function that takes the object first makes a step, the GPU twins
included:

    >>> from bixverse_gpu import pp as pp_gpu
    >>> s = Step("pca_gpu", pp_gpu.calculate_pca, {"no_pcs": 30})
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse._core.graph import (
    ClusterAlgorithm,
    LocalMoving,
    find_clusters,
    find_neighbours,
)
from bixverse._core.integrate import bbknn, fast_mnn, harmony, harmony_v2
from bixverse._core.metacells import MetaCells, merge_meta_cells
from bixverse._core.metacells_generate import (
    generate_bt_meta_cells,
    generate_seacells,
    generate_supercells,
)
from bixverse._core.multimodal import CacheModality, Modality
from bixverse._core.pp import calculate_pca, find_hvg
from bixverse._core.repr import compact_repr
from bixverse._core.single_cells import SingleCells
from bixverse._core.subset import SingleCellsSubset
from bixverse._validate import Verbose
from bixverse.params import (
    BbknnParams,
    FastmnnParams,
    HarmonyParams,
    HarmonyV2Params,
    HvgParams,
    NeighboursParams,
    PcaParams,
)

###########
# Globals #
###########

STEP_CLASSES: tuple[type, ...] = (SingleCells, SingleCellsSubset, MetaCells)

MetacellMethod = Literal["bootstrapped", "seacells", "supercells"]

# MetaCells2 is not one of the choices.
METACELL_GENERATORS: dict[str, Callable[..., MetaCells | None]] = {
    "bootstrapped": generate_bt_meta_cells,
    "seacells": generate_seacells,
    "supercells": generate_supercells,
}

logger = logging.getLogger(__name__)


###########
# Helpers #
###########


@beartype
def _short_value(v: object) -> object:
    """Scalars and short sequences of scalars verbatim, the rest as the type name."""
    scalar = (type(None), bool, int, float, str)
    if isinstance(v, scalar):
        return v
    if (
        isinstance(v, list | tuple)
        and len(v) <= 4
        and all(isinstance(x, scalar) for x in v)
    ):
        return v
    return type(v).__name__


@beartype
def _object_class(obj: SingleCells | SingleCellsSubset | MetaCells) -> type:
    """The step class of `obj`. A `SingleCellsMultiModal` counts as a
    `SingleCells`."""
    return next(c for c in STEP_CLASSES if isinstance(obj, c))


@beartype
def _groups(sc: SingleCells, group_col: str, groups: Sequence[str] | None) -> list[str]:
    """The requested levels, else every level among the kept cells in order
    of appearance, as strings.

    Raises:
        ValueError: If `group_col` is not in obs.
    """
    if group_col not in sc.obs_columns():
        raise ValueError(f"Column '{group_col}' not found in obs.")
    if groups is not None:
        return list(groups)
    values = sc.obs(cols=[group_col], filtered=True)[group_col]
    # whole doubles are "1", not "1.0"
    if values.dtype.is_float() and (values == values.floor()).all():
        values = values.cast(pl.Int64)
    return values.cast(pl.String).unique(maintain_order=True).to_list()


###########
# Classes #
###########


@beartype
@dataclass(frozen=True, repr=False)
class Step:
    """One pipeline step.

    The keyword arguments are bound against the function's signature, the
    object standing in first, so a misspelt or missing argument fails here
    rather than after the expensive steps before it ran.

    Attributes:
        name: Short identifier shown when printing the pipeline.
        fn: Takes the object first. Its return value is ignored unless
            `returns` is set.
        kwargs: Keyword arguments passed to `fn` at apply time.
        accepts: The classes `fn` can take.
        returns: The class `fn` returns, `None` if it modifies the object in
            place and hands nothing back.

    Raises:
        TypeError: If `kwargs` do not bind to `fn`.
        ValueError: If `accepts` or `returns` is not a step class.

    Examples:
        >>> fit = {"method": "analytic_pearson"}
        >>> step = bv.pipeline.Step("residuals", bv.pp.fit_residuals, fit)
    """

    name: str
    fn: Callable[..., object]
    kwargs: dict[str, object] = field(default_factory=dict)
    accepts: tuple[type, ...] = (SingleCells, SingleCellsSubset)
    returns: type | None = None

    def __post_init__(self) -> None:
        bad = [
            c for c in (*self.accepts, self.returns) if c not in (*STEP_CLASSES, None)
        ]
        if bad or not self.accepts:
            names = ", ".join(c.__name__ for c in STEP_CLASSES)
            raise ValueError(f"Step classes must be among {names}.")
        inspect.signature(self.fn).bind(None, **self.kwargs)

    def __rshift__(self, other: Step) -> Pipeline:
        return Pipeline((self, other))

    def __repr__(self) -> str:
        return compact_repr(
            "Step",
            name=self.name,
            kwargs={k: _short_value(v) for k, v in self.kwargs.items()},
        )


@beartype
@dataclass(frozen=True, repr=False)
class Pipeline:
    """A linear chain of steps. Inert until applied.

    Build it from a list of steps, `Pipeline([step_hvg(), step_pca(30)])`,
    or chain steps with `>>` as shorthand: `step_hvg() >> step_pca(30)`.

    Attributes:
        steps: The steps, in order. Any sequence; stored as a tuple.
    """

    steps: Sequence[Step] = ()

    def __post_init__(self) -> None:
        # beartype samples one element of a sequence, so check them all
        bad = [type(s).__name__ for s in self.steps if not isinstance(s, Step)]
        if bad:
            raise TypeError(f"Pipeline steps must be Step objects, got {bad}")
        object.__setattr__(self, "steps", tuple(self.steps))

    def __rshift__(self, other: Step) -> Pipeline:
        return Pipeline((*self.steps, other))

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return compact_repr(
            "Pipeline",
            n_steps=len(self.steps),
            steps=[s.name for s in self.steps],
        )


#########
# Apply #
#########


@beartype
def validate_pipeline(pipeline: Pipeline, cls: type) -> type:
    """Check that a pipeline can run on an object of class `cls`.

    Walks the chain tracking the class each step would receive: most steps
    hand back what they got, a meta cell step returns a `MetaCells`.

    Args:
        pipeline: The pipeline.
        cls: `SingleCells`, `SingleCellsSubset` or `MetaCells`.

    Returns:
        The class the pipeline would return.

    Raises:
        TypeError: If a step cannot take what the step before hands it.
        ValueError: If `cls` is not a step class.

    Examples:
        >>> end = bv.pipeline.validate_pipeline(mc_pipe, bv.SingleCellsSubset)
    """
    if cls not in STEP_CLASSES:
        names = ", ".join(c.__name__ for c in STEP_CLASSES)
        raise ValueError(f"cls must be one of {names}, got `{cls.__name__}`.")
    current = cls
    for i, s in enumerate(pipeline.steps, start=1):
        if current not in s.accepts:
            accepts = ", ".join(c.__name__ for c in s.accepts)
            raise TypeError(
                f"Step {i} ('{s.name}') cannot run on a `{current.__name__}`. "
                f"It takes: {accepts}.\n"
                f"The pipeline started on a `{cls.__name__}`."
            )
        if s.returns is not None:
            current = s.returns
    return current


@beartype
def apply_pipeline(
    pipeline: Pipeline, obj: SingleCells | SingleCellsSubset | MetaCells
) -> SingleCells | SingleCellsSubset | MetaCells:
    """Run each step in order.

    The chain is validated against the class of `obj` before anything runs.
    Steps modify the object in place; a meta cell step's result replaces it
    for the steps after.

    Args:
        pipeline: The pipeline.
        obj: The object the first step receives.

    Returns:
        `obj`, or the object the last class-changing step returned.

    Raises:
        TypeError: If the chain cannot run on `obj`, or a class-changing step
            returns something else, e.g. `None` from a generator without an
            embedding.

    Examples:
        >>> sc = bv.pipeline.apply_pipeline(pipe, sc)
        >>> mc = bv.pipeline.apply_pipeline(mc_pipe, sc)
    """
    if len(pipeline) == 0:
        return obj
    validate_pipeline(pipeline, _object_class(obj))
    for i, s in enumerate(pipeline.steps, start=1):
        res = s.fn(obj, **s.kwargs)
        if s.returns is not None:
            if not isinstance(res, SingleCells | SingleCellsSubset | MetaCells):
                raise TypeError(
                    f"Step {i} ('{s.name}') returned a `{type(res).__name__}`; "
                    "see the warning above."
                )
            obj = res
    return obj


@beartype
def apply_pipeline_per_group(
    pipeline: Pipeline,
    sc: SingleCells,
    group_col: str,
    *,
    groups: Sequence[str] | None = None,
    progress: bool = False,
) -> dict[str, SingleCells | SingleCellsSubset | MetaCells]:
    """Run a pipeline independently on each group.

    Each group is cut as a `SingleCellsSubset` over the kept cells, so the
    parent is left alone. A pipeline ending on `step_metacells` gives
    source-pure meta cells, which `merge_meta_cells` puts back together.

    Args:
        pipeline: The pipeline.
        sc: The experiment.
        group_col: obs column to split on.
        groups: The levels to use; `None` for every level among the kept
            cells, in order of appearance.
        progress: Log a line per group.

    Returns:
        The processed objects, keyed by group.

    Raises:
        TypeError: If the chain cannot run on a `SingleCellsSubset`.
        ValueError: If `group_col` is not in obs.

    Examples:
        >>> per_group = bv.pipeline.apply_pipeline_per_group(pipe, sc, "sample_id")
        >>> per_group = bv.pipeline.apply_pipeline_per_group(
        ...     pipe, sc, "sample_id", groups=["sample_1", "sample_2"]
        ... )
    """
    validate_pipeline(pipeline, SingleCellsSubset)
    levels = _groups(sc, group_col, groups)
    out: dict[str, SingleCells | SingleCellsSubset | MetaCells] = {}
    for i, g in enumerate(levels, start=1):
        if progress:
            logger.info("Applying pipeline to %s (%d out of %d).", g, i, len(levels))
        out[g] = apply_pipeline(pipeline, SingleCellsSubset(sc, group_col, g))
    return out


@beartype
def meta_cells_per_group(
    sc: SingleCells,
    group_col: str,
    *,
    method: MetacellMethod = "bootstrapped",
    mc_params: Mapping[str, object] | None = None,
    pipeline: Pipeline | None = None,
    groups: Sequence[str] | None = None,
    feature_space: str = "intersect",
    verbose: Verbose = True,
) -> MetaCells:
    """Source-pure meta cells: one generator run per group, merged.

    Meta cells never mix two groups (patients, samples), yet end up in one
    object. The generators need an embedding and usually a kNN, so `pipeline`
    will normally be `step_hvg() >> step_pca(30) >> step_neighbours()`.

    Args:
        sc: The experiment.
        group_col: obs column to split on.
        method: `"bootstrapped"`, `"seacells"` or `"supercells"`.
        mc_params: Keyword arguments for the generator; SEACells needs
            `seacell_params`.
        pipeline: Run on each subset before its meta cells are generated.
            Must hand back the subset.
        groups: The levels to use; `None` for every level among the kept
            cells, in order of appearance.
        feature_space: Passed to `merge_meta_cells`.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Returns:
        The merged meta cells, a `source_id` column in obs.

    Raises:
        TypeError: If `pipeline` cannot run on, or does not return, a
            `SingleCellsSubset`, or `mc_params` do not bind to the generator.
        ValueError: On a missing column, or a group without meta cells.

    Examples:
        >>> pipeline = bv.pipeline.step_hvg(hvg_no=500) >> bv.pipeline.step_pca(20)
        >>> pipeline = pipeline >> bv.pipeline.step_neighbours()
        >>> mc = bv.metacells.meta_cells_per_group(sc, "batch_index", pipeline=pipeline)
    """
    fn = METACELL_GENERATORS[method]
    kwargs = dict(mc_params or {})
    inspect.signature(fn).bind(None, **kwargs)
    if pipeline is not None:
        end = validate_pipeline(pipeline, SingleCellsSubset)
        if end is not SingleCellsSubset:
            raise TypeError(
                f"pipeline must return a `SingleCellsSubset`, not a `{end.__name__}`."
            )
    levels = _groups(sc, group_col, groups)
    out: dict[str, MetaCells] = {}
    for g in levels:
        if verbose:
            logger.info("Generating meta cells for group '%s'.", g)
        sub = SingleCellsSubset(sc, group_col, g)
        if pipeline is not None:
            apply_pipeline(pipeline, sub)
        mc = fn(sub, **kwargs)
        if mc is None:
            raise ValueError(f"No meta cells for group '{g}'; see the warning above.")
        out[g] = mc
    return merge_meta_cells(out, feature_space=feature_space, verbose=verbose)


#########
# Steps #
#########


@beartype
def step_hvg(
    *,
    hvg_no: int = 2000,
    hvg_params: HvgParams | None = None,
    streaming: bool | None = None,
    verbose: Verbose = True,
) -> Step:
    """Pipeline step for `bixverse.pp.find_hvg`.

    Args:
        hvg_no: Number of HVGs, forwarded.
        hvg_params: The HVG parameters, forwarded.
        streaming: Read the genes in batches, forwarded.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it, forwarded.

    Examples:
        >>> step = bv.pipeline.step_hvg(hvg_no=500)
    """
    return Step(
        "hvg",
        find_hvg,
        {
            "hvg_no": hvg_no,
            "hvg_params": hvg_params,
            "streaming": streaming,
            "verbose": verbose,
        },
        accepts=STEP_CLASSES,
    )


@beartype
def step_pca(
    no_pcs: int = 30,
    *,
    pca_params: PcaParams | None = None,
    sparse_svd: bool | None = None,
    hvg: np.ndarray | Sequence[int] | Sequence[str] | None = None,
    seed: int = 42,
    residuals: bool = False,
    verbose: Verbose = True,
) -> Step:
    """Pipeline step for `bixverse.pp.calculate_pca`.

    Args:
        no_pcs: Number of components, forwarded.
        pca_params: The PCA parameters, forwarded.
        sparse_svd: Solve on the sparse matrix, forwarded.
        hvg: Gene indices or ids to use instead of the stored HVGs, forwarded.
        seed: Seed, forwarded.
        residuals: PCA on the residuals of `fit_residuals`, forwarded.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it, forwarded.

    Examples:
        >>> step = bv.pipeline.step_pca(20, sparse_svd=False)
    """
    return Step(
        "pca",
        calculate_pca,
        {
            "no_pcs": no_pcs,
            "pca_params": pca_params,
            "sparse_svd": sparse_svd,
            "hvg": hvg,
            "seed": seed,
            "residuals": residuals,
            "verbose": verbose,
        },
        accepts=STEP_CLASSES,
    )


@beartype
def step_neighbours(
    *,
    embd_to_use: str = "pca",
    no_embd_to_use: int | None = None,
    modality: CacheModality = "rna",
    neighbours_params: NeighboursParams | None = None,
    seed: int = 42,
    verbose: Verbose = True,
) -> Step:
    """Pipeline step for `bixverse.graph.find_neighbours`.

    Args:
        embd_to_use: The embedding, forwarded.
        no_embd_to_use: Number of leading dimensions, forwarded.
        modality: `"rna"` or `"adt"`, forwarded.
        neighbours_params: kNN and graph parameters, forwarded.
        seed: Seed, forwarded.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it, forwarded.
    """
    return Step(
        "neighbours",
        find_neighbours,
        {
            "embd_to_use": embd_to_use,
            "no_embd_to_use": no_embd_to_use,
            "modality": modality,
            "neighbours_params": neighbours_params,
            "seed": seed,
            "verbose": verbose,
        },
        accepts=STEP_CLASSES,
    )


@beartype
def step_clusters(
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
) -> Step:
    """Pipeline step for `bixverse.graph.find_clusters`.

    Args:
        cluster_algorithm: `"leiden"` or `"louvain"`, forwarded.
        res: Modularity resolution, forwarded.
        name: The obs column to write, forwarded.
        modality: The graph to cluster, forwarded.
        seed: Seed, forwarded.
        beta: Randomness of the Leiden refinement, forwarded.
        n_iterations: Full iterations, forwarded.
        min_gain: Stopping gain with a negative `n_iterations`, forwarded.
        local_moving: `"queue"` or `"coloured"`, forwarded.
    """
    return Step(
        "clusters",
        find_clusters,
        {
            "cluster_algorithm": cluster_algorithm,
            "res": res,
            "name": name,
            "modality": modality,
            "seed": seed,
            "beta": beta,
            "n_iterations": n_iterations,
            "min_gain": min_gain,
            "local_moving": local_moving,
        },
        accepts=STEP_CLASSES,
    )


@beartype
def step_harmony(
    batch_column: str,
    *,
    additional_batch_columns: Sequence[str] | None = None,
    modality: CacheModality = "rna",
    harmony_params: HarmonyParams | None = None,
    seed: int = 42,
    verbose: Verbose = True,
) -> Step:
    """Pipeline step for `bixverse.integrate.harmony`.

    Args:
        batch_column: obs column with the primary batch, forwarded.
        additional_batch_columns: Further obs columns to regress out,
            forwarded.
        modality: The PCA to correct, forwarded.
        harmony_params: The parameters, forwarded.
        seed: Seed, forwarded.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it, forwarded.
    """
    return Step(
        "harmony",
        harmony,
        {
            "batch_column": batch_column,
            "additional_batch_columns": additional_batch_columns,
            "modality": modality,
            "harmony_params": harmony_params,
            "seed": seed,
            "verbose": verbose,
        },
    )


@beartype
def step_harmony_v2(
    batch_column: str,
    *,
    additional_batch_columns: Sequence[str] | None = None,
    modality: CacheModality = "rna",
    harmony_params: HarmonyV2Params | None = None,
    seed: int = 42,
    verbose: Verbose = True,
) -> Step:
    """Pipeline step for `bixverse.integrate.harmony_v2`.

    Args:
        batch_column: obs column with the primary batch, forwarded.
        additional_batch_columns: Further obs columns to regress out,
            forwarded.
        modality: The PCA to correct, forwarded.
        harmony_params: The parameters, forwarded.
        seed: Seed, forwarded.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it, forwarded.
    """
    return Step(
        "harmony_v2",
        harmony_v2,
        {
            "batch_column": batch_column,
            "additional_batch_columns": additional_batch_columns,
            "modality": modality,
            "harmony_params": harmony_params,
            "seed": seed,
            "verbose": verbose,
        },
    )


@beartype
def step_bbknn(
    batch_column: str,
    *,
    no_neighbours_to_keep: int = 5,
    embd_to_use: Literal["pca"] = "pca",
    no_embd_to_use: int | None = None,
    bbknn_params: BbknnParams | None = None,
    seed: int = 42,
    verbose: Verbose = True,
) -> Step:
    """Pipeline step for `bixverse.integrate.bbknn`.

    Args:
        batch_column: obs column with the batch, forwarded.
        no_neighbours_to_keep: Neighbours per cell in the stored kNN,
            forwarded.
        embd_to_use: The embedding, only `"pca"`, forwarded.
        no_embd_to_use: Number of leading dimensions, forwarded.
        bbknn_params: The parameters, forwarded.
        seed: Seed, forwarded.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it, forwarded.
    """
    return Step(
        "bbknn",
        bbknn,
        {
            "batch_column": batch_column,
            "no_neighbours_to_keep": no_neighbours_to_keep,
            "embd_to_use": embd_to_use,
            "no_embd_to_use": no_embd_to_use,
            "bbknn_params": bbknn_params,
            "seed": seed,
            "verbose": verbose,
        },
    )


@beartype
def step_fast_mnn(
    batch_column: str,
    batch_hvg_genes: np.ndarray | Sequence[int] | Sequence[str],
    *,
    fastmnn_params: FastmnnParams | None = None,
    use_precomputed_pca: bool = False,
    seed: int = 42,
    verbose: Verbose = True,
) -> Step:
    """Pipeline step for `bixverse.integrate.fast_mnn`.

    Args:
        batch_column: obs column with the batch, forwarded.
        batch_hvg_genes: Batch-aware HVG indices or ids, forwarded.
        fastmnn_params: The parameters, forwarded.
        use_precomputed_pca: Use the cached PCA, forwarded.
        seed: Seed, forwarded.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it, forwarded.
    """
    return Step(
        "fast_mnn",
        fast_mnn,
        {
            "batch_column": batch_column,
            "batch_hvg_genes": batch_hvg_genes,
            "fastmnn_params": fastmnn_params,
            "use_precomputed_pca": use_precomputed_pca,
            "seed": seed,
            "verbose": verbose,
        },
    )


@beartype
def step_metacells(method: MetacellMethod = "bootstrapped", **kwargs: object) -> Step:
    """Pipeline step for a meta cell generator.

    Unlike the other steps it changes the class of the object: it takes a
    `SingleCells` or `SingleCellsSubset` and returns a `MetaCells`, which the
    steps after it receive. `validate_pipeline` checks they take one.

    Args:
        method: `"bootstrapped"` (`bixverse.metacells.generate_bt_meta_cells`),
            `"seacells"` (`generate_seacells`) or `"supercells"`
            (`generate_supercells`).
        **kwargs: Keyword arguments for the generator, e.g. `params`,
            `target_size` or `verbose`; `seacell_params` for SEACells.

    Returns:
        The step.

    Raises:
        TypeError: If `kwargs` do not bind to the generator.

    Examples:
        >>> params = bv.params.BtMetacellsParams(target_no_metacells=50)
        >>> step = bv.pipeline.step_metacells("bootstrapped", params=params)
    """
    return Step(
        f"metacells ({method})",
        METACELL_GENERATORS[method],
        dict(kwargs),
        returns=MetaCells,
    )
