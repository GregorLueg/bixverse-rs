"""Cross-field rules for the generated GPU param bundles.

The bixverse.gpu specs carry these as R code (`extra_ctor`, `extra_check`),
which the generator cannot translate, so they are ported here by hand and keyed
by class name, as `bixverse.params._extras` does for the CPU bundles.
"""

from __future__ import annotations

from collections.abc import Callable

from beartype import beartype

from bixverse.params._extras import _nebula

###########
# Globals #
###########

KMEANS_INITS: frozenset[str] = frozenset({"random", "parallel", "plusplus"})


#########
# Rules #
#########


@beartype
def _kmeans_init(p: object) -> None:
    """`k_means_init` is one of the GPU k-means initialisations, or `None`."""
    init = getattr(p, "k_means_init")
    if init is not None and init not in KMEANS_INITS:
        raise ValueError(
            "`k_means_init` must be None or one of 'random', 'parallel' or "
            f"'plusplus', got '{init}'."
        )


@beartype
def _fastmnn(p: object) -> None:
    """fastMNN has no data-driven fallback for `k = 0`."""
    if getattr(p, "k") < 1:
        raise ValueError("`k` in GPU fastMNN params must be at least 1.")


RULES: dict[str, Callable[[object], None]] = {
    "FastClusterGpuParams": _kmeans_init,
    "FastmnnGpuParams": _fastmnn,
    "HarmonyV2GpuParams": _kmeans_init,
    "NebulaGpuParams": _nebula,
}


@beartype
def run(p: object) -> None:
    """Apply the cross-field rule for this bundle's class, if it has one.

    Args:
        p: A generated param bundle.

    Raises:
        ValueError: If a rule fails.
    """
    rule = RULES.get(type(p).__name__)
    if rule is not None:
        rule(p)
