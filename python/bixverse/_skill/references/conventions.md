# bixverse (Python) conventions

Learn these once and the module surface becomes predictable.

## The experiment is a directory

An experiment lives on disk; `bv.SingleCells` is a handle on it.

| File | Holds | Written by |
|---|---|---|
| `counts_cells.bin`, `counts_genes.bin` | raw counts, cell-major and gene-major, memory-mapped on read (store format v5) | the `bv.io.load_*` loaders |
| `sc_duckdb.db` | obs and var, plus the `obs.to_keep` cell filter | loaders, `add_obs`, `set_cells_to_keep`, at once |
| `memory_py.npz`, `memory_py.json` | the HVG set, the cell filter and the cache | `bv.io.save(sc)` only |
| `atac/` | the ATAC store of a `SingleCellsMultiModal`, one row per cell of the RNA store | `bv.atac.add_atac_tenx`, `add_atac_fragments` |

The handle carries `sc.dir`, `sc.map` (`cell_ids`, `gene_ids`,
`cells_to_keep`, `hvg`) and `sc.cache` (PCA, embeddings, kNN, neighbour graph,
MAGIC layer, residual model). obs and var edits reach the DuckDB immediately.
The map and the cache live in memory until `bv.io.save(sc)`.

## load_* versus load_existing

A loader creates the experiment; `load_existing` reopens it.

- `load_csr`, `load_h5ad`, `load_tenx_h5`, `load_mtx` and the `load_multi_*`
  twins take the directory to create and return the opened `SingleCells`.
- A load applies `MinQualityParams` (defaults: 100 genes and 250 counts per
  cell, 10 cells per gene, target size 1e4). Dropped cells and genes never
  enter the store. This is the only filter that deletes anything.
- A load adds `nnz` and `lib_size` to obs and `no_cells_exp` to var.
- Loading into an existing experiment overwrites its stores and drops any
  `memory_py.*`. Never call a loader to reopen.
- obs and var column names are snake_cased on load, and the first column
  becomes `cell_id` / `gene_id`. `add_obs` and `add_var` do not rename.

```python
sc = bv.io.load_h5ad("experiments/gbm", "raw/gbm.h5ad")
bv.io.save(sc)
sc = bv.io.load_existing("experiments/gbm")
```

`load_existing` restores the saved map and cache. When the DuckDB's
`obs.to_keep` differs from the saved filter, the DuckDB wins and a warning
flags the cache as stale. `bv.SingleCells(directory)` opens the directory with
an empty cache.

## Parameter bundles

Methods with many tunables take one bundle from `bixverse.params`: 64 classes,
named `<Method>Params` (`HvgParams`, `PcaParams`, `NeighboursParams`,
`ScrubletParams`, `HarmonyV2Params`, `MinQualityParams`).

- Each bundle is a frozen, keyword-only dataclass. Positional arguments raise
  `TypeError`; so does a misspelt field, with a "Did you mean" hint.
- Bundles are flat. `ScrubletParams` carries its HVG, PCA and kNN fields
  directly (`no_pcs`, `k`, `knn_method`); there is no nested block.
- Construction validates. A wrong type raises beartype's
  `BeartypeCallHintParamViolation`, an out-of-range value `ValueError`
  (`` `k` = -1 is outside [0, ). ``), a broken cross-field rule `ValueError`.
- Frozen means no assignment. Derive a variant with `dataclasses.replace`.
- A function takes the bundle by keyword, `<what>_params=`, and `None` means
  the defaults.

```python
import dataclasses

from bixverse.params import NeighboursParams

p = NeighboursParams(k=20, knn_method="hnsw")
bv.graph.find_neighbours(sc, neighbours_params=p)
p30 = dataclasses.replace(p, k=30)
```

Not uniform. `bv.embed.*` take `params=`, a `manifolds_rs` estimator from
`bv.embed.umap_defaults()` and friends, tuned with `.set_params(...)`. These
take plain arguments: `bv.graph.find_clusters`, `bv.de.find_markers`,
`bv.de.find_all_markers`, `bv.qc.run_cell_qc`, `bv.qc.gene_set_proportions`,
`sc.set_cells_to_keep`. When unsure, `inspect.signature(fn)` or `help(fn)`.

## Module functions, the experiment first

Every method is a module function that takes the experiment first:
`bv.<module>.<verb>(sc, ...)`. There are no analysis methods on the class.

Functions that build state write into `sc` and return `None`:

| Call | Writes |
|---|---|
| `bv.pp.find_hvg(sc, hvg_no=2000)` | `sc.map.hvg`, statistics to var |
| `bv.pp.calculate_pca(sc, no_pcs=30)` | `sc.cache.pca_*` |
| `bv.graph.find_neighbours(sc)` | `sc.cache.knn`, `sc.cache.connectivities` |
| `bv.graph.find_clusters(sc, name="leiden_clusters")` | obs column `name`, 0-based integers |
| `bv.embed.umap(sc)` | `sc.cache.embeddings["umap"]` |
| `bv.integrate.harmony(sc, "batch")` | `sc.cache.embeddings["harmony"]` |
| `bv.qc.gene_set_proportions(sc, gene_sets)` | one obs column per gene set |

Analyses that produce a result return an object and leave `sc` alone:
`run_cell_qc` returns `CellQc`, `scrublet` `ScrubletRes`, `fast_cluster`
`FastClusters`, `find_all_markers` a polars frame. The module's `get_data(res)`
turns a result into an obs frame with a `cell_idx` column, ready for
`sc.add_obs` (`bv.qc`, `bv.doublets`, `bv.pathways`, `bv.nmf`).

```python
res = bv.doublets.scrublet(sc, group_by="sample_id")
sc.add_obs(bv.doublets.get_data(res))
```

Tables come back as polars frames, never pandas. `bv.pipeline` chains steps
with `>>` (`step_hvg(...) >> step_pca(20) >> step_neighbours()`) and applies
them with `apply_pipeline` or `apply_pipeline_per_group`.

## Cells to keep

QC deletes nothing. The cell filter decides which cells every later step sees.

- `sc.set_cells_to_keep(cells)` takes 0-based store indices or cell ids, as a
  NumPy array or a list; a polars Series fails, so call `.to_numpy()`.
- The filter is stored unique and ascending. Results align to that order, not
  to the order given.
- It writes `obs.to_keep` in the DuckDB at once, and warns naming every cached
  artefact it made stale.
- `sc.cells_to_keep` returns the filter as 0-based `uint32`. `sc.n_cells` and
  `sc.shape` count kept cells; `sc.dims` counts the whole store.
- `sc.reset_cells_to_keep(force=False)` restores every cell and clears the HVG
  set and the cache. With anything to drop it raises `ValueError` unless
  `force=True`.

```python
keep = sc.obs(filtered=True).filter(pl.col("mt_frac") < 0.2)["cell_id"]
sc.set_cells_to_keep(keep.to_numpy())
```

## Indices and ids

Every index in the Python API is 0-based: `cells_to_keep`, obs `cell_idx`,
var `gene_idx`, `cells=` and `genes=` arguments, kNN neighbour indices. The
DuckDB stores the index columns 1-based; the API converts at the SQL
boundary. A raw `duckdb` query on `sc_duckdb.db` sees 1-based keys.

- Selections take integer indices or string ids. Unknown ids warn and drop;
  no match raises `ValueError`. An out-of-range index raises `ValueError`, a
  float index `TypeError`.
- `sc.cell_ids(filtered=False)`, `sc.gene_ids()`, `sc.cell_indices(ids)` and
  `sc.gene_indices(ids)` translate between the two.
- `bv.get_counts(sc, cells=None, genes=None, layer="raw")` returns `Counts`:
  the CSR matrix plus the 0-based store index of each row and column. Rows are
  the kept cells, or the given cells that are kept, in the order given.

## obs and var

- `sc.obs(indices=None, cols=None, filtered=False)` returns **every** cell in
  the store by default, kept or not, with a `to_keep` column. Pass
  `filtered=True` for the kept cells.
- `sc.var(indices=None, cols=None)` and `sc.obs_columns()` read without
  loading counts.
- `sc.add_obs(frame)` joins on a 0-based `cell_idx` column. Without one, the
  frame must hold one row per kept cell in ascending cell order. Uncovered
  cells get NULL, with a warning when that overwrites existing values.
- `sc.add_var(frame)` takes one row per gene, or a `gene_idx` column.
- `drop_obs_cols` and `drop_var_cols` skip protected columns: `cell_idx`,
  `cell_id`, `to_keep`; `gene_idx`, `gene_id`. `rename_obs_col` and
  `rename_var_col` warn on a missing column.

## Cache state: print before re-running

`print(sc)` shows store dimensions, kept cells, what the cache holds and what
is stale. Two finer probes:

```python
sc.cache_status()  # artefact, stamped, stale, reason, id, parents
sc.check_state(["knn"])  # True, or why it is stale
```

Each cached artefact carries a stamp: a hash of the cell set and the ids of
its parents. Re-running the PCA makes the kNN and the graph built on it stale,
even on the same cells. The response is graded:

- `set_cells_to_keep` and the cache getters warn.
- Functions that hand cached indices to Rust or write a derived result back
  (`find_neighbours`, `find_clusters`, the meta cell generators) raise
  `ValueError`.
- Nothing is recomputed or deleted silently.

Stamps track provenance, not parameters: a kNN on 5 of 30 PCs looks the same
as one on all 30. Assigning a cache slot by hand drops its stamp, and an
unstamped artefact passes every check. `bv.io.save` does not persist the
residual model and warns.

## Subsets

`bv.SingleCellsSubset(sc, grouping_column, group)` holds the kept cells of
`sc` whose obs column equals `group`.

- It reads the parent's stores; no data is copied.
- obs and var are in-memory snapshots. Nothing it does touches the parent's
  DuckDB.
- Cell indices stay in the parent's store index space. Exception:
  `subset.obs(indices=...)` takes row positions within the subset.
- Its HVG set and cache start empty.
- It cannot be saved. Results go back through `bv.merge_subset_obs`.

```python
t_cells = bv.SingleCellsSubset(sc, "cell_type", "T cell")
bv.pp.find_hvg(t_cells)
bv.pp.calculate_pca(t_cells, 30)
bv.graph.find_neighbours(t_cells)
bv.graph.find_clusters(t_cells, name="t_subcluster")
bv.merge_subset_obs(sc, t_cells, cols=["t_subcluster"])
```

`MetaCells` and `SingleCellsMultiModal` go through the same module functions;
the type hints say which functions accept which class.

## Verbosity and logging

- `verbose` takes `0` (`False`) silent, `1` (`True`) progress, `2` extra
  detail where the method has it. The default is `True`; other values fail
  validation.
- Python messages go through the `bixverse` logger (`bixverse_gpu` for the GPU
  wheel): INFO to stdout, WARNING and above to stderr. The handler stands down
  when the root logger has handlers.
- `logging.getLogger("bixverse").setLevel(logging.WARNING)` silences the
  Python lines, not what Rust prints. Only `verbose=0` silences both.
- Stale caches and unmatched ids raise Python `warnings`, filterable with the
  `warnings` module.

## Errors

| Raised | When |
|---|---|
| `BeartypeCallHintParamViolation` | wrong argument type, on any public function or bundle. Not a `TypeError` subclass |
| `ValueError` | out-of-range values, stale cached input, empty or unmatched selections, wrong row counts in `add_obs` |
| `TypeError` | non-integer index arrays, unknown bundle fields, positional bundle arguments |
| `KeyError` | a missing obs or var column |
| `FileNotFoundError`, `OSError` | a missing store or DuckDB, or an I/O failure in Rust |
| `RuntimeError` | `load_existing` on a v3 store; convert it with `bv.io.upgrade_sc_store("dir")` |
| `bv.BixverseError` | any failure in the Rust core, and every `bixverse_gpu` call without a GPU |
| `bv.SparseFileError` | a store that is truncated, corrupt or of the wrong format version; subclass of `BixverseError` |
| `bv.TableNotFoundError` | a DuckDB table the call needs is missing |

Catching `bv.BixverseError` does not catch the Python-side `ValueError`s.

## Streaming

`streaming: bool | None = None` on the count-reading methods (`find_hvg`,
`gene_set_proportions`, `scrublet`, `scdblfinder`, `aucell`, `module_scores`,
`vision`, `hotspot_*`, ...). `None` streams above
`bv.qc.N_CELLS_STREAMING_THRESHOLD` (100,000) kept cells; pass `True` to cap
memory on smaller data. The loaders take `csc_mem_gb` for the CSR to CSC
conversion buffers; `None` converts in one pass.

## Seeds

Every function with a `seed` argument defaults to `42`. The embeddings carry
theirs inside the estimator (`bv.embed.umap_defaults().set_params(seed=1)`).
Pass it explicitly when reproducibility matters. Two exceptions:

- UMAP and densMAP with `optimiser="sgd_parallel"` are not reproducible run to
  run, whatever the seed.
- `bv.CellLoader` mixes its seed with the epoch counter, so each pass
  reshuffles.

## NumPy input

Every binding takes C-ordered arrays only. A Fortran-ordered array raises
`ValueError`; wrap it in `np.ascontiguousarray`. Index arrays are coerced to
`uint32`.
