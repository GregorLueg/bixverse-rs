"""In-memory state of a single cell experiment: the index map and the cache.

Every index in here is 0-based.

Provenance: each cached artefact can carry a stamp recording the cell set it
was computed on (a hash of `cells_to_keep`, order sensitive) and the stamp ids
of the artefacts it was derived from. The stamps live in `ScCache.stamps`, and
replacing a payload drops its stamp: assigning a slot
(`cache.pca_factors = ...`) drops its stamp, and an
embedding's stamp only counts while it still belongs to the very array it was
minted for.
"""

from __future__ import annotations

import hashlib
import uuid
import warnings
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, no_type_check

import numpy as np
import polars as pl
from beartype import beartype

from bixverse._core.repr import compact_repr

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scipy.sparse import csr_matrix

###########
# Globals #
###########

# Cache slot to artefact label. The embeddings are labelled by their name.
SLOT_LABELS: dict[str, str] = {
    "pca_factors": "pca",
    "knn": "knn",
    "connectivities": "connectivities",
    "magic": "magic",
    "residual_fit": "residuals",
}

# Embedding names that would collide with an artefact label.
RESERVED_EMBEDDINGS: frozenset[str] = frozenset(SLOT_LABELS.values())

STALE_CELLS = "the cell set changed since it was computed"
STALE_PARENT_GONE = "the artefact it was derived from was re-computed or removed"


###########
# Helpers #
###########


@beartype
def _check_array(
    arr: np.ndarray, name: str, *, dtypes: tuple[type, ...], ndim: int
) -> None:
    """Raise unless an array has one of the dtypes and the dimensionality.

    Raises:
        TypeError: On a wrong dtype.
        ValueError: On a wrong number of dimensions.
    """
    if arr.dtype.type not in dtypes:
        names = ", ".join(np.dtype(d).name for d in dtypes)
        raise TypeError(f"{name} must be {names}, got {arr.dtype}")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {arr.ndim}-D")


@beartype
def _as_index(arr: np.ndarray | Sequence[int], name: str) -> np.ndarray:
    """Coerce 0-based indices to a contiguous `uint32` vector.

    Raises:
        TypeError: If the input does not hold integers.
        ValueError: If it is not 1-D or holds a negative value.
    """
    out = np.asarray(arr)
    if out.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if out.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integers, got dtype {out.dtype}")
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {out.ndim}-D")
    if out.min() < 0:
        raise ValueError(f"{name} must be non-negative")
    return np.ascontiguousarray(out, dtype=np.uint32)


@beartype
def _lookup(ids: np.ndarray, wanted: Sequence[str], what: str) -> np.ndarray:
    """0-based positions of `wanted` in `ids`.

    Unmatched ids warn and are dropped.

    Raises:
        ValueError: If nothing matched.
    """
    pos = {v: i for i, v in enumerate(ids)}
    found = [pos.get(w) for w in wanted]
    n_missing = sum(f is None for f in found)
    if n_missing:
        warnings.warn(
            f"With the provided {what}_ids a total of {n_missing} could not be "
            "matched.",
            stacklevel=3,
        )
    hits = [f for f in found if f is not None]
    if not hits:
        raise ValueError(f"None of the provided {what}_ids matched.")
    return np.asarray(hits, dtype=np.uint32)


#########
# ScMap #
#########


@beartype
@dataclass(repr=False)
class ScMap:
    """Ids and index sets of an experiment.

    Attributes:
        cell_ids: Cell ids; position `i` is store cell `i`.
        gene_ids: Gene ids; position `i` is store gene `i`.
        cells_to_keep: 0-based store indices of the cells in use, in the order
            analysis results are aligned to.
        hvg: 0-based gene indices of the highly variable genes, if selected.
    """

    cell_ids: np.ndarray
    gene_ids: np.ndarray
    cells_to_keep: np.ndarray
    hvg: np.ndarray | None = None

    def __repr__(self) -> str:
        return compact_repr(
            "ScMap",
            n_cells=len(self.cell_ids),
            n_genes=len(self.gene_ids),
            n_kept=len(self.cells_to_keep),
            n_hvgs=None if self.hvg is None else len(self.hvg),
        )

    def __post_init__(self) -> None:
        """Coerce the index vectors to `uint32` and check their range.

        Raises:
            ValueError: On an index outside the store.
        """
        # a filter has no order: unique and ascending, as DuckDB `to_keep` holds it
        self.cells_to_keep = np.unique(_as_index(self.cells_to_keep, "cells_to_keep"))
        if self.cells_to_keep.size and self.cells_to_keep.max() >= len(self.cell_ids):
            raise ValueError("cells_to_keep holds an index past the last cell")
        if self.hvg is not None:
            self.hvg = _as_index(self.hvg, "hvg")
            if self.hvg.size and self.hvg.max() >= len(self.gene_ids):
                raise ValueError("hvg holds an index past the last gene")

    @beartype
    def cell_indices(self, cell_ids: Sequence[str]) -> np.ndarray:
        """0-based indices of cells by id. Unknown ids warn and are dropped.

        Raises:
            ValueError: If no id matched.
        """
        return _lookup(self.cell_ids, cell_ids, "cell")

    @beartype
    def gene_indices(self, gene_ids: Sequence[str]) -> np.ndarray:
        """0-based indices of genes by id. Unknown ids warn and are dropped.

        Raises:
            ValueError: If no id matched.
        """
        return _lookup(self.gene_ids, gene_ids, "gene")

    @beartype
    def cell_names(self, *, filtered: bool = False) -> np.ndarray:
        """Cell ids, all or only the kept ones in `cells_to_keep` order."""
        return self.cell_ids[self.cells_to_keep] if filtered else self.cell_ids


##########################
# Cached artefact shapes #
##########################


@beartype
@dataclass(frozen=True, repr=False)
class KnnGraph:
    """A kNN result. Both arrays are held in C order (copied if needed).

    Attributes:
        indices: `(n, k)` `uint32` neighbour indices into the cells the kNN
            was built on.
        distances: `(n, k)` `float32` distances.
        dist_metric: The metric, e.g. `"euclidean"` or `"cosine"`.
    """

    indices: np.ndarray
    distances: np.ndarray
    dist_metric: str

    def __repr__(self) -> str:
        return compact_repr(
            "KnnGraph",
            n_cells=self.indices.shape[0],
            k=self.indices.shape[1],
            metric=self.dist_metric,
        )

    def __post_init__(self) -> None:
        """Check dtypes and that both matrices have the same shape.

        Raises:
            TypeError: On a wrong dtype.
            ValueError: On a shape mismatch.
        """
        _check_array(self.indices, "indices", dtypes=(np.uint32,), ndim=2)
        _check_array(self.distances, "distances", dtypes=(np.float32,), ndim=2)
        if self.indices.shape != self.distances.shape:
            raise ValueError("indices and distances must have the same shape")
        # the bindings read rows from the flat buffer, so hold them in C order
        object.__setattr__(self, "indices", np.ascontiguousarray(self.indices))
        object.__setattr__(self, "distances", np.ascontiguousarray(self.distances))


@beartype
@dataclass(frozen=True, repr=False)
class CsrMatrix:
    """A CSR sparse matrix as its three component vectors.

    Attributes:
        data: `float32` values.
        indices: `uint32` column indices.
        indptr: `uint32` or `uint64` row pointers of length `shape[0] + 1`.
        shape: `(n_rows, n_cols)`.
    """

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]

    def __repr__(self) -> str:
        return compact_repr("CsrMatrix", shape=self.shape, nnz=len(self.data))

    def __post_init__(self) -> None:
        """Check dtypes and that the components line up.

        Raises:
            TypeError: On a wrong dtype.
            ValueError: On inconsistent lengths.
        """
        _check_array(self.data, "data", dtypes=(np.float32,), ndim=1)
        _check_array(self.indices, "indices", dtypes=(np.uint32,), ndim=1)
        _check_array(self.indptr, "indptr", dtypes=(np.uint32, np.uint64), ndim=1)
        if len(self.indptr) != self.shape[0] + 1:
            raise ValueError("indptr must have shape[0] + 1 entries")
        if not len(self.data) == len(self.indices) == int(self.indptr[-1]):
            raise ValueError("data, indices and indptr[-1] must agree")

    @no_type_check  # beartype cannot resolve the typing-only scipy import
    def scipy_csr(self) -> csr_matrix:
        """The matrix as a `scipy.sparse.csr_matrix`, sharing the buffers.

        Needs scipy (the `sparse` extra).
        """
        from scipy.sparse import csr_matrix

        return csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)


@beartype
@dataclass(frozen=True, repr=False)
class MagicLayer:
    """The MAGIC imputed layer.

    Attributes:
        data: `(n_cells, n_features)` `float32` imputed values.
        features: Gene ids of the columns.
    """

    data: np.ndarray
    features: np.ndarray

    def __repr__(self) -> str:
        return compact_repr("MagicLayer", shape=self.data.shape)

    def __post_init__(self) -> None:
        """Check the matrix and that one feature names each column.

        Raises:
            TypeError: On a wrong dtype.
            ValueError: On a column count mismatch.
        """
        _check_array(self.data, "data", dtypes=(np.float32,), ndim=2)
        if len(self.features) != self.data.shape[1]:
            raise ValueError("features must name every column of data")


@beartype
@dataclass(frozen=True)
class Stamp:
    """Provenance of one cached artefact.

    Attributes:
        cells: Hash of `cells_to_keep` when the artefact was computed.
        id: Unique id of this artefact.
        parents: Stamp ids of the artefacts it was derived from.
    """

    cells: str
    id: str
    parents: tuple[str, ...] = ()


@beartype
def cells_hash(cells_to_keep: np.ndarray) -> str:
    """Hash of a cell set, order sensitive on purpose.

    Analysis results come back in `cells_to_keep` order, so the order is what
    aligns a cached matrix to the cells.
    """
    data = np.ascontiguousarray(cells_to_keep, dtype=np.uint32).tobytes()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


###########
# ScCache #
###########


@beartype
@dataclass(repr=False)
class ScCache:
    """Analysis results held in memory.

    Assigning a slot drops its provenance stamp; stamp it again with
    `SingleCells.stamp()` once the new payload is in place.

    Attributes:
        pca_factors: `(n_kept, n_pcs)` `float32` cell scores.
        pca_loadings: `(n_genes_used, n_pcs)` `float32` loadings.
        pca_singular_vals: `float32` singular values.
        pca_genes: `uint32` 0-based gene index of each loadings row.
        embeddings: Other embeddings by name, e.g. `"umap"`.
        knn: The kNN graph.
        connectivities: The neighbour graph.
        magic: The MAGIC imputed layer.
        residual_fit: The fitted residual model. Opaque for now and not
            persisted by `save`.
        stamps: Provenance stamp per artefact label.
    """

    pca_factors: np.ndarray | None = None
    pca_loadings: np.ndarray | None = None
    pca_singular_vals: np.ndarray | None = None
    pca_genes: np.ndarray | None = None
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    knn: KnnGraph | None = None
    connectivities: CsrMatrix | None = None
    magic: MagicLayer | None = None
    residual_fit: object | None = None
    stamps: dict[str, Stamp] = field(default_factory=dict, repr=False)
    # embedding label -> the array its stamp was minted for
    _embedding_refs: dict[str, weakref.ref] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __repr__(self) -> str:
        pca = self.pca_factors
        return compact_repr(
            "ScCache",
            pca=None if pca is None else pca.shape,
            embeddings=list(self.embeddings),
            knn=self.knn,
            connectivities=self.connectivities,
            magic=self.magic,
            residual_fit=self.residual_fit is not None,
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Assign a slot, dropping the stamp of a replaced payload."""
        label = SLOT_LABELS.get(name)
        if label is not None and "stamps" in self.__dict__:
            self.stamps.pop(label, None)
        object.__setattr__(self, name, value)

    @beartype
    def payload(self, label: str) -> object | None:
        """The artefact behind a label, or `None` when absent."""
        for slot, slot_label in SLOT_LABELS.items():
            if label == slot_label:
                return getattr(self, slot)
        return self.embeddings.get(label)

    @beartype
    def artefacts(self) -> dict[str, Stamp | None]:
        """Every present artefact and its stamp, `None` when unstamped.

        In the order: the PCA, the embeddings, kNN, graph, MAGIC, residuals.

        Raises:
            ValueError: If an embedding uses a reserved name.
        """
        clash = sorted(RESERVED_EMBEDDINGS & set(self.embeddings))
        if clash:
            raise ValueError(f"reserved embedding name(s): {clash}")

        out: dict[str, Stamp | None] = {}
        if self.pca_factors is not None:
            out["pca"] = self.stamps.get("pca")
        for name, arr in self.embeddings.items():
            ref = self._embedding_refs.get(name)
            fresh = ref is not None and ref() is arr
            out[name] = self.stamps.get(name) if fresh else None
        for slot in ("knn", "connectivities", "magic", "residual_fit"):
            if getattr(self, slot) is not None:
                out[SLOT_LABELS[slot]] = self.stamps.get(SLOT_LABELS[slot])
        return out

    @beartype
    def set_stamp(self, label: str, stamp: Stamp) -> None:
        """Attach a stamp to a present artefact.

        Raises:
            KeyError: If no artefact sits under the label.
        """
        payload = self.payload(label)
        if payload is None:
            raise KeyError(f"no cached artefact '{label}'")
        if label in self.embeddings:
            self._embedding_refs[label] = weakref.ref(self.embeddings[label])
        self.stamps[label] = stamp

    @beartype
    def stamp(self, label: str, cells: str, parents: Sequence[str] = ()) -> Stamp:
        """Mint and attach a stamp.

        Args:
            label: The artefact, e.g. `"pca"` or an embedding name.
            cells: The current `cells_hash`.
            parents: Labels of the artefacts it was derived from. Labels that
                are absent or unstamped are dropped, so a chain built on an
                unstamped artefact does not claim a provenance it lacks.

        Returns:
            The new stamp.

        Raises:
            KeyError: If no artefact sits under the label.
            ValueError: If an embedding uses a reserved name.
        """
        present = self.artefacts()
        parent_ids = tuple(s.id for p in parents if (s := present.get(p)) is not None)
        new = Stamp(cells=cells, id=uuid.uuid4().hex[:16], parents=parent_ids)
        self.set_stamp(label, new)
        return new


###############
# Persistence #
###############

_PLAIN_SLOTS: tuple[str, ...] = (
    "pca_factors",
    "pca_loadings",
    "pca_singular_vals",
    "pca_genes",
)


@beartype
def cache_payload(
    cache: ScCache, prefix: str = "cache"
) -> tuple[dict[str, np.ndarray], dict]:
    """A cache as npz arrays (keys under `<prefix>/`) and JSON metadata.

    The residual model is opaque for now and is left out, with a warning.

    Raises:
        ValueError: If an embedding uses a reserved name.
    """
    present = cache.artefacts()
    arrays: dict[str, np.ndarray] = {}
    for slot in _PLAIN_SLOTS:
        value = getattr(cache, slot)
        if value is not None:
            arrays[f"{prefix}/{slot}"] = value
    for i, arr in enumerate(cache.embeddings.values()):
        arrays[f"{prefix}/embedding/{i}"] = arr
    if cache.knn is not None:
        arrays[f"{prefix}/knn/indices"] = cache.knn.indices
        arrays[f"{prefix}/knn/distances"] = cache.knn.distances
    if cache.connectivities is not None:
        arrays[f"{prefix}/connectivities/data"] = cache.connectivities.data
        arrays[f"{prefix}/connectivities/indices"] = cache.connectivities.indices
        arrays[f"{prefix}/connectivities/indptr"] = cache.connectivities.indptr
    if cache.magic is not None:
        arrays[f"{prefix}/magic/data"] = cache.magic.data
        arrays[f"{prefix}/magic/features"] = np.asarray(cache.magic.features, dtype=str)
    if cache.residual_fit is not None:
        warnings.warn(
            "The residual model is not persisted yet; it will be absent on load.",
            stacklevel=3,
        )
    # the residual model is not saved, but what was built from it stays valid:
    # cut it out of the parents, or every child loads as stale
    residual = present.get("residuals")
    dropped = {residual.id} if residual is not None else set()
    meta = {
        "embeddings": list(cache.embeddings),
        "knn_dist_metric": None if cache.knn is None else cache.knn.dist_metric,
        "connectivities_shape": (
            None if cache.connectivities is None else list(cache.connectivities.shape)
        ),
        # only stamps of present artefacts; a dropped payload's stamp is dead
        "stamps": {
            label: {
                "cells": s.cells,
                "id": s.id,
                "parents": [p for p in s.parents if p not in dropped],
            }
            for label, s in present.items()
            if s is not None and label != "residuals"
        },
    }
    return arrays, meta


@beartype
def cache_from_payload(
    arrays: dict[str, np.ndarray], meta: dict, prefix: str = "cache"
) -> ScCache:
    """Rebuild a cache from `cache_payload` output."""
    cache = ScCache()
    for slot in _PLAIN_SLOTS:
        if f"{prefix}/{slot}" in arrays:
            setattr(cache, slot, arrays[f"{prefix}/{slot}"])
    for i, name in enumerate(meta["embeddings"]):
        cache.embeddings[name] = arrays[f"{prefix}/embedding/{i}"]
    if f"{prefix}/knn/indices" in arrays:
        cache.knn = KnnGraph(
            arrays[f"{prefix}/knn/indices"],
            arrays[f"{prefix}/knn/distances"],
            meta["knn_dist_metric"],
        )
    if f"{prefix}/connectivities/data" in arrays:
        n_rows, n_cols = meta["connectivities_shape"]
        cache.connectivities = CsrMatrix(
            arrays[f"{prefix}/connectivities/data"],
            arrays[f"{prefix}/connectivities/indices"],
            arrays[f"{prefix}/connectivities/indptr"],
            (n_rows, n_cols),
        )
    if f"{prefix}/magic/data" in arrays:
        cache.magic = MagicLayer(
            arrays[f"{prefix}/magic/data"],
            arrays[f"{prefix}/magic/features"].astype(object),
        )
    for label, s in meta["stamps"].items():
        cache.set_stamp(label, Stamp(s["cells"], s["id"], tuple(s["parents"])))
    return cache


##############
# Staleness #
##############


@beartype
def stale_reason(
    label: str,
    artefacts: dict[str, Stamp | None],
    cells: str,
    seen: tuple[str, ...] = (),
) -> str | None:
    """Why an artefact is stale, if it is.

    Stale when the cell set moved under it, when a parent no longer exists
    (it was recomputed and minted a new id, or removed), or when a parent is
    itself stale. Unstamped artefacts pass: unknown provenance is not known
    bad.

    Args:
        label: The artefact to evaluate.
        artefacts: As returned by `ScCache.artefacts()`.
        cells: The current `cells_hash`.
        seen: Labels already on the recursion stack.

    Returns:
        `None` when fresh or unstamped, otherwise the reason.
    """
    stamp = artefacts.get(label)
    if stamp is None or label in seen:
        return None
    if stamp.cells != cells:
        return STALE_CELLS

    by_id = {s.id: lab for lab, s in artefacts.items() if s is not None}
    for parent_id in stamp.parents:
        parent = by_id.get(parent_id)
        if parent is None:
            return STALE_PARENT_GONE
        if stale_reason(parent, artefacts, cells, (*seen, label)) is not None:
            return f"its upstream `{parent}` is stale"
    return None


Artefacts = ScCache | dict[str, Stamp | None]


@beartype
def _present(src: Artefacts) -> dict[str, Stamp | None]:
    """The artefacts of a cache, or a ready artefact dict as it is.

    A multi-modal experiment hands in one dict over all its caches, labels
    qualified by modality (`"adt:pca"`), so parents resolve across them.
    """
    return src.artefacts() if isinstance(src, ScCache) else src


@beartype
def state_report(cache: Artefacts, cells: str) -> pl.DataFrame:
    """One row per cached artefact.

    Args:
        cache: The cache, or an artefact dict.
        cells: The current `cells_hash`.

    Returns:
        Columns `artefact`, `stamped`, `stale`, `reason`, `id` and `parents`.
    """
    artefacts = _present(cache)
    rows = []
    for label, stamp in artefacts.items():
        reason = stale_reason(label, artefacts, cells)
        rows.append(
            {
                "artefact": label,
                "stamped": stamp is not None,
                "stale": reason is not None,
                "reason": reason,
                "id": None if stamp is None else stamp.id,
                "parents": [] if stamp is None else list(stamp.parents),
            }
        )
    schema = {
        "artefact": pl.String,
        "stamped": pl.Boolean,
        "stale": pl.Boolean,
        "reason": pl.String,
        "id": pl.String,
        "parents": pl.List(pl.String),
    }
    return pl.DataFrame(rows, schema=schema)


@beartype
def stale_artefacts(cache: Artefacts, cells: str) -> list[str]:
    """Labels of every stale artefact."""
    artefacts = _present(cache)
    return [a for a in artefacts if stale_reason(a, artefacts, cells) is not None]


@beartype
def check_state(cache: Artefacts, cells: str, labels: Sequence[str]) -> bool | str:
    """Whether the given artefacts are fresh.

    Absent and unstamped artefacts pass.

    Returns:
        `True`, or a message naming the first stale artefact and why.
    """
    artefacts = _present(cache)
    for label in labels:
        reason = stale_reason(label, artefacts, cells)
        if reason is not None:
            return (
                f"Cached `{label}` is stale: {reason}. Re-run the step that "
                "produces it, or drop it, before using it."
            )
    return True
