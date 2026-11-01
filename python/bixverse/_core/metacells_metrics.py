"""Meta cell quality: label purity, diffusion density, compactness and
separation.

The port of R's `calc_meta_cell_purity`, `get_meta_cell_purity`,
`calc_diffusion_coordinates`, `calc_manifold_metrics`,
`calc_meta_cell_compactness` and `calc_meta_cell_separation`. The diffusion
map and the two manifold scores run in Rust; purity is a tabulation.

    >>> calc_meta_cell_purity(mc, sc.obs()["cell_type"].to_list())
    >>> calc_diffusion_coordinates(mc, sc.cache.knn)
    >>> calc_manifold_metrics(mc)
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from bixverse import _bixverse
from bixverse._core.metacells import MetaCells
from bixverse._core.pp import _factor_codes
from bixverse._core.state import KnnGraph
from bixverse.params import KnnDefaults, to_rust

###########
# Globals #
###########

PurityInfo = Literal["none", "top_label", "top_two_labels"]


##########
# Purity #
##########


@beartype
def _purity_cols(
    mc: MetaCells,
    original_cell_type: Sequence[str],
    add_additional_info: PurityInfo,
    add_entropy: bool,
) -> dict[str, pl.Series]:
    """The purity columns in obs row order, R's `.mc_purity_cols`.

    Raises:
        ValueError: If the labels match neither the source's obs rows nor its
            kept cells, or a merged object spans several obs spaces.
    """
    a = mc.original_assignment
    if (
        mc.is_merged
        and a.per_source is not None
        and len({v["n_cells"] for v in a.per_source.values()}) > 1
    ):
        raise ValueError(
            "The meta cells were merged from sources with different obs spaces, "
            "so `original_cell_idx` cannot be resolved against a single column. "
            "Calculate the purity per source before merging."
        )
    n_kept = 0 if a.cells_to_keep is None else len(a.cells_to_keep)
    if len(original_cell_type) == a.n_cells:
        rows = [
            np.asarray(m, dtype=np.int64)
            for m in mc.obs()["original_cell_idx"].to_list()
        ]
    elif n_kept > 0 and len(original_cell_type) == n_kept:
        rows = mc._artefact_rows(n_kept)
    else:
        kept = f" of which {n_kept} pass QC" if n_kept > 0 else ""
        raise ValueError(
            f"`original_cell_type` has {len(original_cell_type)} entries but the "
            f"meta cells were built over {a.n_cells} original cells{kept}. Pass the "
            "unfiltered obs column, or the QC-passing one."
        )

    codes, levels = _factor_codes(
        pl.Series(list(original_cell_type), dtype=pl.String), "labels"
    )
    n_labels = len(levels)
    counts = [np.bincount(codes[r], minlength=n_labels) for r in rows]
    sizes = np.array([len(r) for r in rows], dtype=np.float64)
    top = np.array([int(np.argmax(c)) for c in counts])
    cols = {
        "mc_purity": pl.Series(
            [c[t] / n for c, t, n in zip(counts, top, sizes, strict=True)],
            dtype=pl.Float64,
        )
    }
    if add_additional_info != "none":
        cols["mc_top_label"] = pl.Series([levels[t] for t in top], dtype=pl.String)
    if add_additional_info == "top_two_labels":
        second = []
        for c, t in zip(counts, top, strict=True):
            rest = c.copy()
            rest[t] = 0
            second.append(rest)
        cols["mc_second_label"] = pl.Series(
            [None if np.max(r) == 0 else levels[int(np.argmax(r))] for r in second],
            dtype=pl.String,
        )
        cols["mc_second_frac"] = pl.Series(
            [np.max(r) / n for r, n in zip(second, sizes, strict=True)],
            dtype=pl.Float64,
        )
    if add_entropy:
        if n_labels < 2:
            entropy = [0.0] * len(counts)
        else:
            entropy = []
            for c, n in zip(counts, sizes, strict=True):
                p = c[c > 0] / n
                entropy.append(float(-(p * np.log(p)).sum() / np.log(n_labels)))
        cols["mc_entropy"] = pl.Series(entropy, dtype=pl.Float64)
    return cols


@beartype
def calc_meta_cell_purity(
    mc: MetaCells,
    original_cell_type: Sequence[str],
    *,
    add_additional_info: PurityInfo = "none",
    add_entropy: bool = False,
) -> None:
    """Label purity of each meta cell, written to obs, as R's
    `calc_meta_cell_purity`.

    Adds `mc_purity` (fraction of the meta cell's cells carrying its most
    common label), and on request `mc_top_label`, `mc_second_label` (null for
    a pure meta cell), `mc_second_frac` and `mc_entropy` (Shannon entropy of
    the labels over `log(number of distinct labels)`, 0 with a single label).
    Ties resolve to the label that sorts first, by code point; R sorts in the
    locale's collation.

    Args:
        mc: The meta cells.
        original_cell_type: Labels of the source cells, either one per obs
            row of the source (`sc.obs()`) or one per kept cell in
            `cells_to_keep` order; the length decides which.
        add_additional_info: `"none"`, `"top_label"` or `"top_two_labels"`.
        add_entropy: Add `mc_entropy`.

    Raises:
        ValueError: If the labels fit neither cell space.
    """
    cols = _purity_cols(mc, original_cell_type, add_additional_info, add_entropy)
    mc.add_obs(pl.DataFrame(cols))


@beartype
def get_meta_cell_purity(
    mc: MetaCells,
    original_cell_type: Sequence[str],
    *,
    add_additional_info: PurityInfo = "none",
    add_entropy: bool = False,
) -> pl.DataFrame:
    """`calc_meta_cell_purity` without touching the object, as R's
    `get_meta_cell_purity`.

    Args:
        mc: The meta cells.
        original_cell_type: Labels of the source cells, see
            `calc_meta_cell_purity`.
        add_additional_info: `"none"`, `"top_label"` or `"top_two_labels"`.
        add_entropy: Add `mc_entropy`.

    Returns:
        `meta_cell_idx`, `meta_cell_id` and the purity columns, in obs order.
    """
    cols = _purity_cols(mc, original_cell_type, add_additional_info, add_entropy)
    return mc.obs(cols=["meta_cell_idx", "meta_cell_id"]).with_columns(**cols)


#############
# Diffusion #
#############


@beartype
def calc_diffusion_coordinates(
    mc: MetaCells,
    knn: KnnGraph,
    *,
    n_dcs: int = 10,
    k_density: int = 150,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Diffusion map of the source cells and a density region per meta cell,
    as R's `calc_diffusion_coordinates`.

    Builds multiscale diffusion components off the source kNN, takes the
    distance to the `k_density`-th neighbour in that space as a density
    proxy and tags its lower quartile `"high"`, upper quartile `"low"`, the
    rest `"mid"`. Each meta cell gets the most common region of its cells in
    obs `density_region` (ties to the name that sorts first, as R's
    `table()`); `other_data` gains `dcs`, `density_dist` and `regions`.

    A merged object warns and is left alone: its memberships live in several
    index spaces.

    Args:
        mc: The meta cells.
        knn: The kNN of the source cells, one row per kept cell in the
            source's `cells_to_keep` order, e.g. `sc.cache.knn`.
        n_dcs: Diffusion components.
        k_density: The neighbour giving the density.
        seed: Seed.
        verbose: Controls verbosity.

    Raises:
        ValueError: If the kNN does not cover the kept source cells.

    References:
        Persad, et al., Nat Biotechnol, 2023.
    """
    if mc.is_merged:
        warnings.warn(
            "The meta cells were merged, so `original_cell_idx` cannot be resolved "
            "against a single kNN graph. Run calc_diffusion_coordinates() per "
            "source before merging. Returning object as is.",
            stacklevel=2,
        )
        return
    rows = mc._artefact_rows(knn.indices.shape[0])
    dcs, density, regions = _bixverse.metacell_density(
        np.ascontiguousarray(knn.indices),
        np.ascontiguousarray(knn.distances),
        n_dcs,
        k_density,
        to_rust(KnnDefaults(k=1)),
        seed,
        verbose,
    )
    region_arr = np.asarray(regions)
    mc.add_obs(
        pl.DataFrame(
            {"density_region": [_most_common(region_arr[r]) for r in rows]},
            schema={"density_region": pl.String},
        )
    )
    mc.other_data["dcs"] = dcs
    mc.other_data["density_dist"] = density
    mc.other_data["regions"] = list(regions)


@beartype
def _most_common(values: np.ndarray) -> str:
    """The most frequent value, ties to the one that sorts first."""
    uniq, counts = np.unique(values, return_counts=True)
    return str(uniq[int(np.argmax(counts))])


@beartype
def calc_manifold_metrics(mc: MetaCells) -> None:
    """Compactness and separation of each meta cell in the diffusion map, as
    R's `calc_manifold_metrics`.

    Compactness is the mean variance over the diffusion components of a meta
    cell's cells, lower is better; separation the distance from its centroid
    to the nearest other centroid, higher is better. Both go to obs. Needs
    `calc_diffusion_coordinates` first; without it, or on a merged object, a
    warning and nothing changes.

    Args:
        mc: The meta cells.

    References:
        Persad, et al., Nat Biotechnol, 2023.
    """
    if mc.is_merged:
        warnings.warn(
            "The meta cells were merged, so there is no single diffusion map to "
            "resolve `original_cell_idx` against. Run calc_manifold_metrics() per "
            "source before merging. Returning object as is.",
            stacklevel=2,
        )
        return
    dcs = mc.other_data.get("dcs")
    if not isinstance(dcs, np.ndarray):
        warnings.warn(
            "No diffusion coordinates found. Please run calc_diffusion_coordinates(). "
            "Returning object as is.",
            stacklevel=2,
        )
        return
    rows = [r.astype(np.uint32) for r in mc._artefact_rows(dcs.shape[0])]
    dcs32 = np.ascontiguousarray(dcs, dtype=np.float32)
    mc.add_obs(
        pl.DataFrame(
            {
                "compactness": _bixverse.metacell_compactness(dcs32, rows),
                "separation": _bixverse.metacell_separation(dcs32, rows),
            }
        )
    )


@beartype
def _split_rows(
    assignment: Sequence[object], n_rows: int
) -> tuple[list[str], list[np.ndarray]]:
    """Rows per meta cell as R's `split(seq_along(x), x)`: levels sorted,
    missing values dropped.

    Raises:
        ValueError: If the assignment does not have one entry per row.
    """
    if len(assignment) != n_rows:
        raise ValueError(f"assignment has {len(assignment)} entries for {n_rows} rows")
    values = pl.Series(list(assignment))
    present = values.is_not_null().to_numpy()
    codes, levels = _factor_codes(values.filter(values.is_not_null()), "assignment")
    rows_all = np.flatnonzero(present)
    groups = [rows_all[codes == i].astype(np.uint32) for i in range(len(levels))]
    return levels, groups


@beartype
def calc_meta_cell_compactness(
    dcs: np.ndarray, assignment: Sequence[object]
) -> pl.DataFrame:
    """Compactness of any meta cell assignment in a diffusion map, as R's
    `calc_meta_cell_compactness`.

    Args:
        dcs: `(n_cells, n_dcs)` diffusion coordinates, e.g.
            `mc.other_data["dcs"]`.
        assignment: Meta cell of each row of `dcs`, `None` for none.
            Overlapping meta cells cannot be expressed this way; use
            `calc_manifold_metrics` for those.

    Returns:
        `meta_cell` and `compactness`, ordered as R's `split()` orders the
        levels.

    References:
        Persad, et al., Nat Biotechnol, 2023.
    """
    levels, rows = _split_rows(assignment, dcs.shape[0])
    values = _bixverse.metacell_compactness(
        np.ascontiguousarray(dcs, dtype=np.float32), rows
    )
    return pl.DataFrame({"meta_cell": levels, "compactness": values})


@beartype
def calc_meta_cell_separation(
    dcs: np.ndarray, assignment: Sequence[object]
) -> pl.DataFrame:
    """Separation of any meta cell assignment in a diffusion map, as R's
    `calc_meta_cell_separation`. A single meta cell yields `inf`.

    Args:
        dcs: `(n_cells, n_dcs)` diffusion coordinates.
        assignment: Meta cell of each row of `dcs`, `None` for none.

    Returns:
        `meta_cell` and `separation`, ordered as R's `split()` orders the
        levels.

    References:
        Persad, et al., Nat Biotechnol, 2023.
    """
    levels, rows = _split_rows(assignment, dcs.shape[0])
    values = _bixverse.metacell_separation(
        np.ascontiguousarray(dcs, dtype=np.float32), rows
    )
    return pl.DataFrame({"meta_cell": levels, "separation": values})
