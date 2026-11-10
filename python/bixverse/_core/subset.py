"""`SingleCellsSubset`: a view onto one level of an obs column.

The subset reads the parent's stores (no
data is copied) and holds its obs and var in memory, so nothing it does
touches the parent's DuckDB. The map stays in the parent's index space: cell
indices are 0-based store positions, `cells_to_keep` is the parent's filter
restricted to the subset cells, in the parent's order, and the HVG selection
starts empty. The cache starts empty too, so the subset's PCA, kNN, graph and
embeddings live in the subset alone.

    >>> import bixverse as bv
    >>> t_cells = bv.SingleCellsSubset(sc, "cell_type", "T cell")
    >>> bv.pp.find_hvg(t_cells)
    >>> bv.pp.calculate_pca(t_cells, 30)
    >>> bv.graph.find_neighbours(t_cells)
    >>> bv.graph.find_clusters(t_cells, name="t_subcluster")
    >>> bv.merge_subset_obs(sc, t_cells, cols=["t_subcluster"])

A subset is not persisted: `bixverse.io.save` takes a `SingleCells`
only. Results go back to the parent through `merge_subset_obs`.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np
import polars as pl
from beartype import beartype

from bixverse._core import db
from bixverse._core.metacells import MetaCells
from bixverse._core.single_cells import (
    ScBase,
    Selection,
    SingleCells,
    _check_range,
)
from bixverse._core.state import ScCache, ScMap, _lookup, stale_artefacts
from bixverse._validate import Verbose

logger = logging.getLogger(__name__)


#####################
# SingleCellsSubset #
#####################


class SingleCellsSubset(ScBase):
    """The cells of one level of an obs column.

    Integer cell selections are 0-based store indices, as on the parent, and
    must belong to the subset; cell ids are matched against the subset's
    cells only.

    Attributes:
        dir: The parent's directory; the stores are shared.
        map: Ids (the parent's) and index sets restricted to the subset.
        cache: The subset's own analysis results.
        grouping_column: The obs column the subset was cut on.
        group: The level of `grouping_column` it holds.
        subset_to_original: 0-based store index of each subset obs row.
    """

    @beartype
    def __init__(self, sc: SingleCells, grouping_column: str, group: str) -> None:
        """Cut the kept cells of `sc` whose `grouping_column` equals `group`.

        Args:
            sc: The parent experiment.
            grouping_column: obs column defining the groups.
            group: The level to keep, cast to the column's type.

        Raises:
            KeyError: If `grouping_column` is not in obs.
            ValueError: If no kept cell carries the level.
        """
        with db.connect(sc.db_path, read_only=True) as con:
            obs = db.get_table(
                con, "obs", filtered=True, where_equal={grouping_column: group}
            )
        if obs.height == 0:
            raise ValueError(
                f"No cells found for group '{group}' in column '{grouping_column}'."
            )
        self.dir = sc.dir
        self.reader = sc.reader
        self.grouping_column = grouping_column
        self.group = group
        self.subset_to_original = obs["cell_idx"].to_numpy().astype(np.uint32)
        self._obs = obs
        self._var = sc.var()
        parent_keep = sc.map.cells_to_keep
        self.map = ScMap(
            sc.map.cell_ids,
            sc.map.gene_ids,
            parent_keep[np.isin(parent_keep, self.subset_to_original)].copy(),
        )
        self.cache = ScCache()

    def _all_cells(self) -> np.ndarray:
        """The subset's cells: there is no wider cell set to fall back to."""
        return self.subset_to_original

    ##############
    # Id lookups #
    ##############

    @beartype
    def cell_ids(
        self,
        indices: np.ndarray | Sequence[int] | None = None,
        *,
        filtered: bool = False,
    ) -> np.ndarray:
        """Cell ids: the subset's in obs order, kept only in `cells_to_keep`
        order, or for 0-based store indices."""
        if indices is None and not filtered:
            return self._obs["cell_id"].to_numpy()
        return super().cell_ids(indices, filtered=filtered)

    @beartype
    def cell_indices(self, cell_ids: Sequence[str]) -> np.ndarray:
        """0-based store indices of subset cells by id. Ids outside the
        subset warn and drop."""
        pos = _lookup(self._obs["cell_id"].to_numpy(), cell_ids, "cell")
        return self.subset_to_original[pos]

    @beartype
    def _resolve_cells(self, cells: Selection) -> np.ndarray:
        """0-based cell indices from indices or ids, all inside the subset.

        Raises:
            ValueError: If an index is not a subset cell.
        """
        idx = super()._resolve_cells(cells)
        if not np.isin(idx, self.subset_to_original).all():
            raise ValueError("cells holds an index outside the subset")
        return idx

    #############
    # obs / var #
    #############

    @beartype
    def obs(
        self,
        *,
        indices: np.ndarray | Sequence[int] | None = None,
        cols: list[str] | None = None,
        filtered: bool = False,
    ) -> pl.DataFrame:
        """The subset's obs, `cell_idx` the 0-based store index.

        Args:
            indices: Optional 0-based row positions within the subset.
            cols: Optional columns.
            filtered: Accepted for parity with `SingleCells`; a subset is
                filtered by construction.

        Raises:
            KeyError: If a column is not in obs.
        """
        del filtered
        frame = self._obs
        if indices is not None:
            frame = frame[_check_range(indices, frame.height, "indices")]
        if cols is not None:
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise KeyError(f"column(s) not found in 'obs': {missing}")
            frame = frame.select(cols)
        return frame

    def obs_columns(self) -> list[str]:
        """The obs column names."""
        return self._obs.columns

    @beartype
    def var(
        self,
        *,
        indices: np.ndarray | Sequence[int] | None = None,
        cols: list[str] | None = None,
    ) -> pl.DataFrame:
        """The parent's var as it was when the subset was cut, `gene_idx`
        0-based.

        Args:
            indices: Optional 0-based gene indices.
            cols: Optional columns.

        Raises:
            KeyError: If a column is not in var.
        """
        frame = self._var
        if indices is not None:
            frame = frame[_check_range(indices, frame.height, "indices")]
        if cols is not None:
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise KeyError(f"column(s) not found in 'var': {missing}")
            frame = frame.select(cols)
        return frame

    @beartype
    def add_obs(self, frame: pl.DataFrame) -> None:
        """Add or replace columns of the in-memory obs.

        With a 0-based `cell_idx` column the frame joins on it. Without one
        it must hold one row per obs row, in obs (ascending cell) order.
        Subset cells the frame does not cover get NULL.

        Raises:
            ValueError: On a wrong row count, a `cell_idx` outside the subset
                or repeated, or a protected column.
        """
        protected = [c for c in frame.columns if c in ("cell_id", "to_keep")]
        if protected:
            raise ValueError(f"protected obs column(s): {protected}")
        if "cell_idx" not in frame.columns:
            if frame.height != self._obs.height:
                raise ValueError(
                    f"frame has {frame.height} rows; the subset obs has "
                    f"{self._obs.height}. Pass a cell_idx column to address cells "
                    "directly."
                )
            frame = frame.with_columns(self._obs["cell_idx"])
        idx = frame["cell_idx"].to_numpy()
        if not np.isin(idx, self.subset_to_original).all():
            raise ValueError("cell_idx holds a cell outside the subset")
        if len(np.unique(idx)) != len(idx):
            raise ValueError("cell_idx holds repeated values")

        frame = frame.with_columns(pl.col("cell_idx").cast(pl.UInt32))
        new = [c for c in frame.columns if c != "cell_idx"]
        self._obs = self._obs.drop([c for c in new if c in self._obs.columns]).join(
            frame, on="cell_idx", how="left", maintain_order="left"
        )

    def __repr__(self) -> str:
        c = self.cache
        stale = stale_artefacts(c, self.cells_hash())
        return (
            "SingleCellsSubset(\n"
            f"  dir={str(self.dir)!r},\n"
            f"  group={self.grouping_column} = {self.group},\n"
            f"  cells={self.n_cells}, genes={self.n_genes},\n"
            f"  hvg={self.map.hvg is not None}, pca={c.pca_factors is not None}, "
            f"embeddings={sorted(c.embeddings) or 'none'},\n"
            f"  knn={c.knn is not None}, graph={c.connectivities is not None}, "
            f"magic={c.magic is not None}, residuals={c.residual_fit is not None},\n"
            f"  stale={', '.join(stale) or 'none'}\n"
            ")"
        )


ScOrScSubset = SingleCells | SingleCellsSubset
ScOrMc = SingleCells | SingleCellsSubset | MetaCells


##############
# Write-back #
##############


@beartype
def merge_subset_obs(
    sc: SingleCells,
    subsets: SingleCellsSubset | Sequence[SingleCellsSubset],
    *,
    cols: Sequence[str] | None = None,
    new_names: Sequence[str] | None = None,
    prefix_values: bool = False,
    overwrite: bool = False,
    verbose: Verbose = True,
) -> None:
    """Write obs columns of one or more subsets into the parent's obs.

    One join on `cell_idx` for all subsets; parent cells in none of them get
    NULL.

    Args:
        sc: The parent the subsets were cut from.
        subsets: A subset or several, each from `sc`.
        cols: Subset obs columns to merge. `None` takes every column present
            in all subsets and absent from the parent.
        new_names: Names in the parent, one per `cols`, which must then be
            given.
        prefix_values: Prefix each value with its subset's group,
            `"<group>_<value>"`, which casts to string. Needed when several
            subsets share a column whose labels repeat across them.
        overwrite: Replace parent columns of the same name.
        verbose: `0` (`False`) silent, `1` (`True`) progress, `2` extra detail where the
            method has it.

    Raises:
        ValueError: If a subset is not from `sc`, a column is missing or
            clashes, or two subsets share a cell.
    """
    subs = [subsets] if isinstance(subsets, SingleCellsSubset) else list(subsets)
    if not subs:
        raise ValueError("subsets must be non-empty")
    parent_ids = sc.cell_ids()
    parent_cols = sc.obs_columns()
    for i, sub in enumerate(subs, start=1):
        idx = sub.subset_to_original
        if int(np.max(idx)) >= len(parent_ids):
            raise ValueError(f"Subset {i} has cell indices outside of the parent.")
        if not np.array_equal(sub.cell_ids(), parent_ids[idx]):
            raise ValueError(
                f"Subset {i} does not originate from the provided parent object."
            )

    shared = [
        c for c in subs[0].obs_columns() if all(c in s.obs_columns() for s in subs)
    ]
    if cols is None:
        if new_names is not None:
            raise ValueError(
                "`cols` needs to be given explicitly when using `new_names`."
            )
        cols = [c for c in shared if c not in parent_cols and c != "cell_idx"]
        if not cols:
            raise ValueError(
                "No new obs columns found in the subset(s). Nothing to merge."
            )
    else:
        if "cell_idx" in cols:
            raise ValueError("`cell_idx` is the join key and cannot be merged.")
        missing = [c for c in cols if c not in shared]
        if missing:
            raise ValueError(
                f"Column(s) {missing} missing from at least one subset obs table."
            )
    cols = list(cols)
    names = cols if new_names is None else list(new_names)
    if len(names) != len(cols):
        raise ValueError("new_names needs one name per column in cols")
    clashes = [n for n in names if n in parent_cols]
    if clashes and not overwrite:
        raise ValueError(
            f"Column(s) {clashes} already exist in the parent obs. Set overwrite=True."
        )

    frames = []
    for sub in subs:
        frame = sub.obs(cols=["cell_idx", *cols])
        if prefix_values:
            frame = frame.with_columns(
                pl.format("{}_{}", pl.lit(sub.group), pl.col(c)).alias(c) for c in cols
            )
        frames.append(frame.rename(dict(zip(cols, names, strict=True))))

    if len(frames) > 1 and not prefix_values:
        for name in names:
            seen = pl.concat(
                [f[name].cast(pl.String).drop_nulls().unique() for f in frames]
            )
            repeated = seen.filter(seen.is_duplicated()).unique().sort().to_list()
            if repeated:
                warnings.warn(
                    f"Value(s) {repeated[:5]} of column `{name}` appear in more than "
                    "one subset. Set prefix_values=True to disambiguate them.",
                    stacklevel=2,
                )

    merged = pl.concat(frames, how="vertical_relaxed")
    if merged["cell_idx"].is_duplicated().any():
        raise ValueError(
            "Duplicated cells across the subsets. Cells can be merged only once."
        )
    sc.add_obs(merged)
    if verbose:
        logger.info(
            "Merged %i column(s) for %i cells into obs (%i cells left as NULL).",
            len(names),
            merged.height,
            len(parent_ids) - merged.height,
        )
