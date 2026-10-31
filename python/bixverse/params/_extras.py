"""Cross-field rules for the generated param bundles.

devforge carries these as R code in `extra_ctor`, which the generator cannot
translate, so they are ported here by hand and keyed by class name. Each takes
the bundle after its per-field checks have passed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from beartype import beartype


@beartype
def _clip_pair(p: object) -> None:
    """`clip_min` and `clip_max` come together, and in order."""
    lo, hi = getattr(p, "clip_min"), getattr(p, "clip_max")
    if (lo is None) != (hi is None):
        raise ValueError(
            "`clip_min` and `clip_max` must be given together. Supply both to "
            "set a clipping range, or neither to use the default of "
            "+/- sqrt(n_cells)."
        )
    if lo is not None and lo >= hi:
        raise ValueError("`clip_min` must be smaller than `clip_max`.")


@beartype
def _nebula(p: object) -> None:
    """The sigma and phi search intervals are not empty."""
    if getattr(p, "min_sigma") >= getattr(p, "max_sigma"):
        raise ValueError("`min_sigma` needs to be below `max_sigma`.")
    if getattr(p, "min_phi") >= getattr(p, "max_phi"):
        raise ValueError("`min_phi` needs to be below `max_phi`.")


@beartype
def _cellsweep(p: object) -> None:
    """Only the frozen-empties model is implemented."""
    if not getattr(p, "freeze_empties"):
        raise ValueError(
            "`freeze_empties = False` is not supported. The reference gives "
            "empty droplets a cell-type component they have no label for, "
            "which indexes past the end of the profile matrix."
        )


@beartype
def _dsb(p: object) -> None:
    """The quantile pair comes together, and in order."""
    lo, hi = getattr(p, "quantile_low"), getattr(p, "quantile_high")
    if (lo is None) != (hi is None):
        raise ValueError(
            "quantile_low and quantile_high must both be given or both None."
        )
    if lo is not None and lo >= hi:
        raise ValueError("quantile_low must be strictly less than quantile_high.")


@beartype
def _empty_droplets(p: object) -> None:
    """Each calling method has the one input it needs."""
    method = getattr(p, "method")
    match method:
        case "supplied":
            required = "is_empty_column"
        case "umi_cutoff":
            required = "umi_cutoff"
        case "expected_cells":
            required = "expected_cells"
        case "knee":
            return
        case _:
            raise ValueError(f"Unknown empty droplet method {method!r}.")
    if getattr(p, required) is None:
        raise ValueError(f"`method = '{method}'` needs `{required}`.")


@beartype
def _mtx_io(p: object) -> None:
    """The three input files exist."""
    for name in ("path_mtx", "path_obs", "path_var"):
        path = Path(getattr(p, name)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"`{name}` = {path} does not exist.")


@beartype
def _synthetic_cellsweep(p: object) -> None:
    """The marker blocks fit into the gene space."""
    if getattr(p, "n_markers") * getattr(p, "n_celltypes") > getattr(p, "n_genes"):
        raise ValueError("The marker gene blocks do not fit into n_genes.")


@beartype
def _synthetic_data(p: object) -> None:
    """Sample membership is all or nothing, with a known bias."""
    bias = getattr(p, "sample_bias")
    if bias is not None and bias not in ("even", "slightly_uneven", "very_uneven"):
        raise ValueError(f"Unknown sample_bias {bias!r}.")
    if (getattr(p, "n_samples") is None) != (bias is None):
        raise ValueError(
            "`n_samples` and `sample_bias` must be provided together. Supply "
            "both to add sample membership, or neither to omit it."
        )


@beartype
def _synthetic_dialogue(p: object) -> None:
    """The planted blocks fit into their feature and gene spaces."""
    if getattr(p, "n_sample_features") > getattr(p, "n_features"):
        raise ValueError("n_sample_features cannot exceed n_features.")
    if getattr(p, "n_planted") * getattr(p, "n_cell_types") > getattr(p, "n_genes"):
        raise ValueError("The planted gene blocks do not fit into n_genes.")


###########
# Globals #
###########

RULES: dict[str, Callable[[object], None]] = {
    "AprParams": _clip_pair,
    "CellsweepParams": _cellsweep,
    "DsbParams": _dsb,
    "EmptyDropletsParams": _empty_droplets,
    "MtxIoParams": _mtx_io,
    "NebulaParams": _nebula,
    "SctransformParams": _clip_pair,
    "SyntheticCellsweepParams": _synthetic_cellsweep,
    "SyntheticDataParams": _synthetic_data,
    "SyntheticDialogueParams": _synthetic_dialogue,
}


@beartype
def run(p: object) -> None:
    """Apply the cross-field rule for this bundle's class, if it has one.

    Args:
        p: A generated param bundle.

    Raises:
        ValueError: If a rule fails.
        FileNotFoundError: If an input file named by the bundle is missing.
    """
    rule = RULES.get(type(p).__name__)
    if rule is not None:
        rule(p)
