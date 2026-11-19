# Single cell: the core workflow

## The mental model

A `SingleCells` object holds no counts in Python memory. It is a handle over a
directory:

- `counts_cells.bin`, cell-major (CSR), for per-cell operations (store
  format v5)
- `counts_genes.bin`, gene-major (CSC), for per-gene operations
- `sc_duckdb.db`, the obs and var tables

Two in-memory side-cars live on the object: `sc.map` (`ScMap`: cell and gene
ids, `cells_to_keep`, HVG indices) and `sc.cache` (`ScCache`: PCA,
`embeddings`, `knn`, `connectivities`, `magic`, `residual_fit`, and a
provenance stamp per artefact).

`bv.io.save(sc)` writes the in-memory state as `memory_py.npz` plus
`memory_py.json`.

Consequences: the directory must outlive the object; counts are never fully
materialised (pull them with `bv.get_counts()`); the cache survives a restart
only after `bv.io.save(sc)`. `bv.SingleCells(directory)` opens an existing
store and does not create one; the loaders create it.

## The chain

Every step mutates `sc` in place and returns `None`.
Do not reassign: `sc = bv.pp.find_hvg(sc)` sets `sc` to `None`.

```python
import bixverse as bv
import bixverse.embed  # not loaded by `import bixverse`

# 1. ingest, exactly one loader; returns the opened SingleCells
sc = bv.io.load_mtx("exp/pbmc", bv.io.get_cell_ranger_params(mtx_dir))
# 2. QC metrics onto the obs table
bv.qc.gene_set_proportions(sc, {"MT": mt_gene_ids})
bv.qc.top_genes_perc(sc)
# 3. doublets, optional; returns a record, obs untouched
sc.add_obs(bv.doublets.get_data(bv.doublets.scrublet(sc, group_by="sample_id")))
# 4. call outliers and filter
qc = bv.qc.run_cell_qc(metrics, obs["cell_idx"].to_numpy(), directions=dirs)
sc.add_obs(bv.qc.get_data(qc))
sc.set_cells_to_keep(qc.cell_idx[~qc.combined])
# 5. feature selection and reduction
bv.pp.find_hvg(sc, hvg_no=2000)
bv.pp.calculate_pca(sc, no_pcs=30)
# 6. batch correction, optional, writes a NEW named embedding
bv.integrate.harmony(sc, "sample_id")
# 7. graph and clusters
bv.graph.find_neighbours(sc, embd_to_use="harmony")
bv.graph.find_clusters(sc, res=1.0, name="leiden_clusters")
# 8. embedding for plots
bv.embed.umap(sc)
# 9. markers
markers = bv.de.find_all_markers(sc, "leiden_clusters")
# 10. persist the in-memory state
bv.io.save(sc)
sc = bv.io.load_existing("exp/pbmc")
```

`print(sc)` shows kept cells, cached artefacts and which are stale;
`sc.cache_status()` gives one row per artefact with `stale` and `reason`.

## Loading

| Input | Function |
|---|---|
| Cell Ranger MTX directory | `bv.io.load_mtx()`, with `bv.io.get_cell_ranger_params(path)` |
| h5ad or anndata Zarr store | `bv.io.load_h5ad()`, `load_h5ad_norm()` for log-normalised only, `stream_h5ad()` |
| 10x HDF5 | `bv.io.load_tenx_h5()` (`feature_type="Gene Expression"`) |
| in-memory CSR + obs + var | `bv.io.load_csr(directory, counts, obs, var)` |
| a directory written earlier | `bv.io.load_existing(directory)` |

There is no Seurat or `SingleCellExperiment` reader on the Python side.
Every loader takes the target directory first and returns the `SingleCells`.
Shared keywords: `qc=MinQualityParams(...)`, `csc_mem_gb=None`,
`verbose=True`.

- `load_csr()` takes a scipy `csr_matrix` / `csr_array` or a `bv.CsrMatrix`,
  cells as rows, non-negative integer counts. obs / var names are
  snake_cased; the first column becomes `cell_id` / `gene_id`; repeated cell
  ids get `.1`, `.2` suffixes.
- `load_mtx()` takes genes as rows (10x convention) unless
  `get_cell_ranger_params(path, cells_as_rows=True)`. Set `has_hdr=True` if
  the barcode and feature files carry a header.

### Normalisation happens here

`bixverse.params.MinQualityParams` defaults: `min_unique_genes=100` and
`min_lib_size=250` per cell, `min_cells=10` per gene, `target_size=1e4`.
Genes are filtered first, then cells are checked against the reduced gene
set. The store holds the raw integer counts;
`log1p(count / lib_size * target_size)` is recomputed from them in `float32`
whenever the normalised layer is read. There is no `normalise()` function and
you should not write one. obs gains `lib_size` and
`nnz`; var gains `no_cells_exp`.

**The load-time cutoffs are irreversible.** Failing cells and genes are never
written to the store, and `qc=None` applies the defaults above. Set them
permissively, e.g. `MinQualityParams(min_unique_genes=0, min_lib_size=0,
min_cells=0)`, and do the real QC with `bv.qc.run_cell_qc()`, which is
reversible.

### h5ad specifics

Everything here takes an anndata Zarr store (v2 or v3, a directory) in place
of the h5ad file: the loaders, `detect_raw_count_slot`, `read_h5ad_metadata`
and `get_h5ad_dimensions`.

- `detect_raw_count_slot(path)` tests `"layers.counts"`, `"raw.X"`, `"X"` in
  that order and returns the first whose sampled non-zeros are at least 99 %
  integral (`threshold=0.99`, `n_sample=10_000`), else `None`.
  `load_h5ad(raw_count_slot="auto")` raises if it finds none.
- The detection samples 10,000 values, and READMEs rarely say which slot
  holds raw counts. Confirm the slot after loading: the store's obs
  `lib_size` should equal an author per-cell total (e.g. `total_counts`) for
  at least 99 % of cells.
- The `RawSlot` literal is not exported. Pass the string, e.g. `"raw.X"`.
- `load_h5ad_norm(directory, h5_path, obs_lib_size_col, target_size)` rebuilds
  counts assuming `X = log1p(x / lib_size * target_size)`. It cannot invert
  any other transform.
- `stream_h5ad()` is `load_h5ad(h5ad_streaming=True)`, already the default of
  `load_h5ad()`. It has no `cell_id_col` argument.
- `read_h5ad_metadata(path)` returns `obs`, `var` (polars, snake_cased, index
  as `cell_id` / `gene_id`), `dims` and `type` without reading `X`.
  `get_h5ad_dimensions(path).type` is `"CSR"`, `"CSC"`, `"DENSE_ROW"` or
  `"DENSE_COL"`.
- obs columns in an unsupported h5ad encoding are skipped with a warning.

### Several files

```python
prescan = bv.io.prescan_h5ad_files(
    {"d1": "d1.h5ad", "d2": "d2.h5ad"}, gene_universe="intersection"
)
sc = bv.io.load_multi_h5ad("exp/pooled", prescan)
```

- `prescan_h5ad_files()` and `prescan_tenx_h5_files()` take a list (ids are
  file stems) or a dict (ids are the keys); `gene_universe` is
  `"intersection"` or `"union"`. `prescan_mtx_dirs(dirs, exp_ids)` always
  intersects. All need two or more inputs with unique ids.
- obs gains `exp_id`, and `cell_id` is prefixed with it.
- `merge_sc_experiments(directory, inputs, exp_ids)` combines loaded
  experiments: shared genes, kept cells, no cache.

`csc_mem_gb` caps the buffers of the CSR-to-CSC conversion that builds the
gene-major store, and of the transpose of a CSC h5ad (`None` is one pass); pass e.g. `8.0` for million-cell
inputs. With `None` the buffers need about 8 bytes per non-zero (about 21 GB
for 2.6e9 non-zeros, estimate); above physical RAM the conversion swaps and
dominates the load time. Half of physical RAM is a safe cap: data that fits
still converts in one phase. Every load builds both stores; there is no
cell-major-only load. The QC passes, `find_hvg` and the doublet callers stream cells in
batches above `bv.qc.N_CELLS_STREAMING_THRESHOLD` (100,000 kept cells);
`streaming=` overrides it.

### Traps

- Loading into a directory that already holds an experiment overwrites its
  stores and deletes `memory_py.*`.
- `load_existing()` warns on a v4 store and refuses a v3 one.
  `bv.io.upgrade_sc_store(sc)` converts a v4 store in place, and
  `bv.io.upgrade_sc_store("dir")` a v3 one (`keep_backup=True` leaves
  `<file>.v<n>.bak`).
- The DuckDB `obs.to_keep` column is the source of truth for the cell filter.
  If it disagrees with `memory_py.npz`, `load_existing()` warns, uses
  `to_keep`, and the cached artefacts load as stale.

## Writing out

`bv.io.save(sc)` writes the cell filter, HVGs and cache to `memory_py.*`;
obs and var edits are already in the DuckDB. `write_h5ad_sc()`,
`write_tenx_h5_sc()` and `write_cellranger_output()` write in-memory CSR to
interchange formats. `archive_sc_exp(sc)` writes a zstd `counts.bxa` and
deletes both stores; `load_existing()` rebuilds them from it.

`bv.io.save_h5ad(sc, path, assay="raw")` writes the kept cells in ascending
store order, every gene of the store (no gene selection), obs / var without
the bookkeeping columns, and `X` as `float32` streamed in `chunk_size=10_000`
cell batches.
`assay="norm"` puts the normalised layer in `X`. From the cache it writes
`obsm/X_pca`, `obsm/X_<name>` per embedding, `varm/PCs` (padded to all genes
from the genes the PCA ran on, only alongside `X_pca`) and
`obsp/connectivities`. Stale artefacts are skipped with a warning.
`bv.io.save_zarr(sc, path)` writes the same to an AnnData Zarr store, with the
same arguments.

`write_h5ad_sc()` holds the whole matrix in memory and writes `X` as `float32`
CSR, the first obs / var column as index. It converts its input to contiguous
`float32` values and `uint32` indices, about 8 bytes per non-zero plus a copy
when the input dtypes differ. The file it writes has resizable, chunked,
gzip-compressed `X/data` and `X/indices` and an `int32` `X/indptr`.

### Writing a gene subset without materialising the matrix

`save_h5ad()` streams but writes every store gene and the store's obs / var;
`write_h5ad_sc()` takes a gene subset and new obs / var but needs the whole
matrix in memory. For a store too large for RAM (e.g. counts restricted to
genes mapped to a reference release, with harmonised obs), combine the two:

```python
import h5py
import numpy as np
import scipy.sparse as sp

partial = out.with_name(out.name + ".partial")
empty = sp.csr_matrix((obs.height, var.height), dtype=np.float32)
bv.io.write_h5ad_sc(partial, empty, obs, var)  # obs / var encoded by bixverse

indptr = np.zeros(cells.size + 1, dtype=np.int64)
nnz = 0
with h5py.File(partial, "r+") as f:
    data, indices = f["X"]["data"], f["X"]["indices"]
    for start in range(0, cells.size, 50_000):
        chunk = cells[start : start + 50_000]
        counts = bv.get_counts(sc, cells=chunk, genes=gene_idx)
        assert np.array_equal(counts.cells, chunk)
        m = counts.scipy_csr()
        m.sort_indices()  # see Reading counts
        n = m.data.size
        data.resize((nnz + n,))
        data[nnz:] = m.data
        indices.resize((nnz + n,))
        indices[nnz:] = m.indices.astype(np.int32)
        indptr[start + 1 : start + chunk.size + 1] = nnz + m.indptr[1:].astype(np.int64)
        nnz += n
    del f["X"]["indptr"]
    f["X"].create_dataset("indptr", data=indptr, compression="gzip")
partial.replace(out)  # only after validation
```

- Peak memory is obs plus one chunk, independent of the number of cells.
- Replace `X/indptr` with `int64`: the empty file's `int32` row pointers
  overflow above 2^31 - 1 non-zeros. Cast each chunk's `indptr` to `int64`
  before adding the offset; under NumPy 2, `int + int32 array` stays `int32`.
- Validate before the rename, e.g. that per-cell sums of the written genes
  plus those of the excluded genes equal `lib_size`. Writing under a
  temporary name means a crash never leaves a file that looks complete.
- Keep the gzip compression: the bixverse loaders do not read LZF or Blosc.

**Trap:** `save()` does not persist the residual model. It warns; refit after
`load_existing()`.

## Reading counts

`bv.get_counts(sc, *, cells=None, genes=None, layer="raw", assay="rna",
use_cells_to_keep=True)` takes 0-based indices or ids and returns `Counts`:
`matrix` (`CsrMatrix`, `data` float32, `indices` uint32), `cells` and `genes`
(0-based store indices of rows and columns), and `.scipy_csr()` (needs
scipy). `layer="norm"` reads the normalised layer.

- Raw counts come back as `float32`, not integers.
- With `use_cells_to_keep=True`, requested cells that are not kept are
  silently dropped.
- Unknown ids warn and are dropped; it raises only when none match. Check
  `counts.genes` against what you asked for.
- Columns follow the gene order passed. Rows follow the cell order passed, or
  `sc.cells_to_keep` when `cells=None`.
- A gene selection that is not in ascending store order returns unsorted
  column indices within each row (`scipy_csr().has_sorted_indices` is
  `False`). Values are correct, but call `sort_indices()` before writing the
  matrix anywhere a reader may assume sorted CSR.

## obs, var and indices

`sc.obs(filtered=, cols=)` and `sc.var(cols=)` return polars frames;
`add_obs`, `add_var`, `rename_obs_col`, `drop_obs_cols` and their var twins
edit the DuckDB in place.

| Call | Indexing |
|---|---|
| `sc.cells_to_keep`, `sc.map.hvg` | **0-based** store indices, `uint32` |
| `obs["cell_idx"]`, `var["gene_idx"]` | **0-based** |
| `sc.set_cells_to_keep()`, `calculate_pca(hvg=)`, `get_counts()` | 0-based indices or ids |

Python has no 1-based entry point.

### Traps

- `sc.obs()` defaults to `filtered=False`: every cell in the store, removed
  ones included.
- String obs columns come back as polars `Enum`. A join with a `String`
  column from another table fails (`datatypes of join keys don't match`);
  cast the key first, `pl.col("site_id").cast(pl.String)`.
- `sc.add_obs(frame)` joins on `cell_idx` when present. Without it, the frame
  must have one row per kept cell in ascending order and is matched by
  position; wrong order silently mislabels cells. Always include `cell_idx`.
- Cells the frame does not cover get NULL, with a warning when existing values
  are discarded. Re-running `gene_set_proportions()` after filtering nulls the
  metric for removed cells.
- `cells_to_keep` is a set: unique and ascending. Results follow that order,
  not the order you passed.
- `sc.reset_cells_to_keep()` restores all cells but clears the HVGs and the
  cache; with anything cached it raises unless `force=True`.

## QC

```python
import numpy as np
import polars as pl

var = sc.var()
mt = var.filter(pl.col("gene_symbol").str.starts_with("MT-"))["gene_id"].to_list()
bv.qc.gene_set_proportions(sc, {"MT": mt})

obs = sc.obs(filtered=True, cols=["cell_idx", "lib_size", "nnz", "MT"])
metrics = {
    "log10_lib_size": np.log10(obs["lib_size"].to_numpy()),
    "log10_nnz": np.log10(obs["nnz"].to_numpy()),
    "MT": obs["MT"].to_numpy(),
}
dirs = {"log10_lib_size": "twosided", "log10_nnz": "twosided", "MT": "above"}
qc = bv.qc.run_cell_qc(
    metrics,
    obs["cell_idx"].to_numpy(),
    directions=dirs,
    threshold=3.0,
    hard_thresholds={"MT": {"upper": 0.2}},
)
sc.add_obs(bv.qc.get_data(qc))
sc.set_cells_to_keep(qc.cell_idx[~qc.combined])
```

- `gene_set_proportions(sc, {name: gene_ids})` writes one obs column per set,
  a fraction in [0, 1]. `top_genes_perc(sc, top_n_vals=(25, 50, 100))`
  writes `top_<N>_genes_percentage`, also a fraction.
- `run_cell_qc()` returns a `CellQc`: `cell_idx`, `metrics`, `groups`,
  `thresholds` (`metric`, `group`, `median`, `upper_threshold`,
  `lower_threshold`), `outlier_mat`, `hard_mat`, `rescued_mat`, `combined`.
- The MAD is scaled by 1.4826 (`r_standard=True`), so `threshold=3` is three
  standard deviations under normality.
- With `groups=` (sample, donor) MAD runs per group; hard thresholds ignore
  groups. Bounds are inclusive.
- `mad=False` with `hard_thresholds`, or `run_cell_qc_fixed()`, gives fixed
  thresholds only.
- `get_data(qc)` returns the metrics, `cell_idx`, `grp`, `global_outlier`,
  and per metric `<m>_is_outlier`, `<m>_mad_outlier`, `<m>_hard_outlier`,
  `<m>_rescued`.
- `bv.pl.violin_plot(qc)` and `bv.pl.joint_plot(qc)` plot it;
  `bv.pl.density_plot(qc)` needs a run with `groups`.

### Traps

- `rescue_cells()` and `flag_cells()` return a new `CellQc`; reassign
  (`qc = bv.qc.rescue_cells(qc, {"MT": {"upper": 0.1}})`). This is the
  opposite of the `SingleCells` functions.
- `rescue_cells()` clears MAD flags only, never hard flags, and replaces any
  earlier rescue of that metric. `flag_cells()` unions with existing flags;
  `reset=True` clears them first.
- `directions`, if given, must name every metric; omitted, all are
  `"twosided"`. A `lower` bound on an `"above"` metric raises.
- `get_data()` writes metrics back under their own names. Name a transformed
  metric distinctly (`log10_lib_size`) or it overwrites the raw column.
- Take the metrics and `cell_idx` from the same `sc.obs(filtered=True)` frame
  so they align.

## Doublets

```python
from bixverse.params import ScdblfinderParams

params = ScdblfinderParams(expected_doublet_rate=0.06)
res = bv.doublets.scdblfinder(sc, scdblfinder_params=params, group_by="sample_id")
sc.add_obs(bv.doublets.get_data(res))
```

| Caller | Params | `get_data()` columns (plus `cell_idx`) |
|---|---|---|
| `scrublet()` | `ScrubletParams` | `doublet`, `doublet_score` |
| `doublet_detection_boost()` | `BoostParams` | `doublet`, `doublet_score`, `voting_avg` |
| `scdblfinder()` | `ScdblfinderParams` | `predicted_doublets`, `doublet_score`, `cxds_scores`, `weighted`, `cluster_labels` |

- All run on the kept cells unless `cells_to_use=` is given, and return a
  frozen record. Obs is untouched until you `add_obs`.
- `group_by` runs one fit per group; per-group scalars such as `threshold`
  become `{group: value}` dicts. Group by sample or lane, never across
  emulsions.
- `ScrubletParams.expected_doublet_rate` defaults to 0.1; the threshold comes
  from the score histogram unless `manual_threshold` is set.
  `call_doublets_manual(res, threshold)` re-calls a Scrublet result (one
  group on a grouped result: `for_sample`, else the first).
- `ScdblfinderParams.expected_doublet_rate` is a fraction (0.06 = 6 %), which
  Rust clamps to [0.001, 0.5]. Left `None`, it is `0.008 * n_cells / 1000`
  per run, which understates the rate for author-filtered files.

### Traps

- A group below 50 cells raises; below 500 warns. Missing labels raise.
- Scrublet and Boost both write `doublet` and `doublet_score`; adding the
  second replaces the first. Rename before `add_obs` when comparing callers.
- scDblFinder writes `predicted_doublets`, not `doublet`. A filter written for
  Scrublet columns finds nothing.

## HVG and PCA

`bv.pp.find_hvg(sc, hvg_no=2000, hvg_params=HvgParams(method="vst"))`, then
`bv.pp.calculate_pca(sc, no_pcs=30, pca_params=PcaParams())`.

- `find_hvg()` writes per-gene statistics to var (`mean`, `var`, `var_exp`,
  `var_std` for `"vst"`; `dispersion*` and `bin` for `"meanvarbin"` /
  `"dispersion"`; `scran_*` for `"scran"`) and the selection to
  `sc.map.hvg`. `get_hvg_data(sc)` returns the statistics without touching
  the experiment.
- `calculate_pca()` writes `sc.cache.pca_factors`, `pca_loadings`,
  `pca_singular_vals` and `pca_genes`.
- `sparse_svd=None` means sparse on the normalised path. Above 500,000 kept
  cells a dense request is switched to sparse, with a log message.
- `PcaParams.svd_solver="auto"` (default) uses Lanczos on the sparse path and
  `"covariance"` on the dense one. `"covariance"` memory grows with the square
  of the gene count; use `"randomised"` for many features.

### Missing prerequisites warn, they do not error

`calculate_pca()` without HVGs warns "Returning object as is" and does
nothing. So do `find_neighbours()` without the named embedding,
`find_clusters()` without a graph, and every batch correction on a single
batch except `fast_mnn`, which raises. A chain that looks like it ran can have
done nothing. Print `sc` after any step you are unsure of.

| Step | Needs |
|---|---|
| `calculate_pca` | `find_hvg`, or explicit `hvg=` |
| batch correction | `calculate_pca` |
| `find_neighbours` | an embedding matching `embd_to_use` |
| `find_clusters` | `find_neighbours` |
| `fast_mnn`, `seurat_cca`, `seurat_rpca` | `find_hvg_batch_aware(...).hvg_gene_idx` |
| `umap`, `densmap`, `forceatlas2` (`use_knn=True`) | a kNN, else a fresh search with a warning |
| `find_hvg(method="residual")`, `calculate_pca(residuals=True)` | `fit_residuals` |
| `cellsweep` | cell type labels on the raw barcodes |

### Filter first, then HVG and PCA

Every cached artefact carries a stamp with the cell-set hash and its parents.
`set_cells_to_keep()` after a PCA does not delete the PCA; it warns and marks
it stale, transitively taking the kNN, graph and embeddings with it.
`find_neighbours()` then raises on the stale PCA, and `find_clusters()`
raises on a stale graph or one covering a different number of cells.
Recompute from the step that went stale.

### Traps

- The HVG selection is not stamped. It survives a filter change silently and
  still reflects the old cells. Rerun `find_hvg()` after filtering.
- `calculate_pca(hvg=...)` replaces `sc.map.hvg` with the genes you pass.
- Reserved embedding names: `pca`, `knn`, `connectivities`, `magic`,
  `residuals`. Do not use them as a `slot_name`.

## Pearson residuals and scTransform

Opt-in. `fit_residuals()` caches a model in `sc.cache.residual_fit`; the
counts on disk are untouched. The Python default is `method="sctransform"`
(v2 NB GLM, `SctransformParams`, the only one that takes covariates or gives
corrected counts). `method="analytic_pearson"` is closed form and cheaper
(`AprParams`).

```python
from bixverse.params import HvgParams, PcaParams

bv.pp.fit_residuals(sc, method="analytic_pearson")
bv.pp.find_hvg(sc, hvg_no=2000, hvg_params=HvgParams(method="residual"))
# either the normal PCA on the log-normalised layer ...
bv.pp.calculate_pca(sc, no_pcs=30)
# ... or PCA on the residuals themselves
bv.pp.calculate_pca(
    sc, 30, pca_params=PcaParams(normalise_variance=False), residuals=True
)
```

- The fit is bound to the cells it saw. Move `cells_to_keep` and every
  consumer raises until you refit.
- `residuals=True` raises unless `normalise_variance=False` and `clr=False`.
  It is dense and warns when cells x HVGs as `float64` exceeds 8 GB.
- `group_column` fits one model per group; `hvg_no` is then per group and
  unioned, so you get more than `hvg_no` genes.
- `covariate_columns` (sctransform only) must be numeric, without missing or
  non-finite values.
- `find_hvg_batch_aware()` refuses the residual method.
- `sct_corrected_counts(sc, dir_out=...)` writes a new `SingleCells` on the
  model's gene axis, narrower than the source.

## Batch correction

All run after `calculate_pca()` and write a new named embedding for
`find_neighbours(embd_to_use=...)`. Harmony also corrects another embedding:
`harmony(mm, "batch", embd_to_use="lsi", modality="atac")` writes `"harmony"`
to the ATAC cache.

| Function | Embedding | Params |
|---|---|---|
| `bv.integrate.harmony()` (v2) | `"harmony"` | `HarmonyV2Params` |
| `bv.integrate.harmony_v1()` | `"harmony_v1"` | `HarmonyParams` |
| `bv.integrate.fast_mnn()` | `"mnn"` | `FastmnnParams` |
| `bv.integrate.seurat_cca()` | `"cca"` | `SeuratCcaParams` |
| `bv.integrate.seurat_rpca()` | `"rpca"` | `SeuratRpcaParams` |
| `bv.integrate.bbknn()` | overwrites `knn` and `connectivities` | `BbknnParams` |

```python
batch_hvg = bv.pp.find_hvg_batch_aware(sc, "exp_id", hvg_no=2000)
bv.integrate.fast_mnn(sc, "exp_id", batch_hvg.hvg_gene_idx)
bv.graph.find_neighbours(sc, embd_to_use="mnn")
metrics = bv.integrate.calculate_integration_metrics(
    sc, "exp_id", cell_type_column="cell_type", embd_to_use="mnn"
)
```

- Harmony's initial clustering is e-means; `HarmonyV2Params(
  k_means_algorithm="lloyd")` gives Lloyd's.
- `find_hvg_batch_aware()` returns a `BatchHvg` (`hvg_genes`,
  `hvg_gene_idx`, `hvg_data`) and leaves `sc.map.hvg` alone.
  `gene_comb_method` is `"union"` (default), `"average"` or
  `"intersection"`.
- `seurat_cca()` and `seurat_rpca()` take a `SingleCells` only.
- Batch levels are sorted; a polars `Enum` keeps its own order. Missing labels
  raise.

`calculate_integration_metrics()` returns one polars row: `embedding`,
`kbet_accept`, `batch_asw`, `ilisi`, `pcr_comparison`, `clisi`,
`cell_type_asw`, `graph_connectivity`. All are higher-is-better on [0, 1]
except `pcr_comparison`, which can be negative. A metric that cannot run is
null. The parts are `calculate_kbet`, `calculate_lisi`,
`calculate_batch_asw`, `calculate_cell_type_asw`, `calculate_pcr` and
`calculate_graph_connectivity`, each returning a record or `None`.

### Traps

- kBET, LISI and graph connectivity read the cached kNN. Rebuild neighbours on
  the corrected embedding first; a stale kNN only warns, so you can silently
  score the uncorrected graph.
- For BBKNN pass `embd_to_use=None` to `calculate_integration_metrics()`.

## Neighbours and clustering

```python
from bixverse.params import NeighboursParams

nn = NeighboursParams(k=15, knn_sym="snn", knn_method="kmknn")
bv.graph.find_neighbours(sc, embd_to_use="pca", neighbours_params=nn)
bv.graph.find_clusters(sc, res=1.0, name="leiden_clusters")
```

- `find_neighbours()` writes `sc.cache.knn` (`KnnGraph`: `indices` uint32,
  `distances` float32) and `sc.cache.connectivities` (`CsrMatrix`).
  `no_embd_to_use` limits the leading dimensions.
- `knn_sym`: `"snn"` (default, Seurat/bluster sNN), `"umap"` (fuzzy
  simplicial set) or `"knn"`. Resolutions do not carry over between them.
- `knn_method`: `"kmknn"` (default), `"hnsw"`, `"annoy"`, `"nndescent"`,
  `"ivf"`, `"exhaustive"`.
- `find_clusters()` writes 0-based `Int32` labels, 0 the largest community,
  to obs column `name` (default `"leiden_clustering"`, also for Louvain).
  Both algorithms run 2 iterations by default (`n_iterations`). Leiden visits
  nodes in BFS order (`node_order="bfs"`), Louvain moves them in parallel
  colour classes on large levels (`local_moving="coloured"`).
  `local_moving="queue"`, `node_order="shuffle"` is igraph's order.
- `get_connectivities(sc).scipy_csr()` gives scipy; hand the three vectors to
  igraph or networkx.
- `fast_cluster(sc, resolutions=(2.0, 1.0, 0.5), n_seeds=1)` coarsens with
  k-means (e-means by default; `FastClusterParams(km_type="standard")` for
  Lloyd's, `"minibatch"` for mini-batch) and clusters the centroids. Its
  `memberships` (`cell_idx`, `res_<value>`) go straight into `sc.add_obs()`.

### Traps

- Cluster labels are integers. `bv.pl.embedding_plot()` treats them as
  continuous unless `discrete=True`, and `label_by` refuses them.
- `SingleCellsSubset(sc, "leiden_clusters", "3")` takes the group as a string
  and casts it to the column type.

## Embeddings

`bv.embed` needs the `embed` extra (`manifolds-rs`) and an explicit
`import bixverse.embed`.

```python
import bixverse.embed

bv.embed.umap(sc)
bv.embed.tsne(sc, params=bv.embed.tsne_defaults().set_params(perplexity=20.0))
bv.embed.phate(sc)
xy = sc.cache.embeddings["umap"]
```

Slots: `umap`, `densmap`, `fa2` (`forceatlas2`), `tsne`, `densne`, `phate`.
`use_knn` defaults to `True` for UMAP, densMAP and ForceAtlas2, `False` for
t-SNE, den-SNE and PHATE.

- Params are `manifolds_rs` estimators. The `*_defaults()` builders differ
  from the estimator defaults in places: t-SNE perplexity is 10, not 30.
- With `use_knn=True` the cached kNN is reused and the neighbour settings in
  `params` do not apply. A wider graph is cut to `n_neighbors` with a warning
  on stderr; a narrower one is used as it is.
- PHATE wants a `6k + 5` wide graph (35 by default), so it searches afresh.
- UMAP `optimiser="sgd_parallel"` is not reproducible. `n_epochs=None` is
  500, or 200 for the SGD optimisers above 10,000 cells.
- A stale input embedding raises.

## Markers

The full DE suite lives in `bv.de`; the core chain needs one call:

`markers = bv.de.find_all_markers(sc, "leiden_clusters", min_prop=0.05)`.

- `method=None` runs a Wilcoxon on RNA. `method="lr"` runs Seurat's logistic
  regression test with the library size as covariate (`z_scores` are then
  signed roots of the likelihood ratio). `modality="atac"` tests the peaks of
  a `SingleCellsMultiModal` and defaults to `"lr"`.

- Columns: `grp`, `gene_id`, `lfc`, `prop1`, `prop2`, `z_scores`,
  `p_values`, `fdr`. `alternative="greater"` (default) tests genes up in the
  group.
- Cells with a null label are left out of every group and every rest.
- `find_markers(sc, cells_1, cells_2)` is an explicit contrast, two-sided by
  default. `find_specific_markers()` finds genes specific to one group.
- There is no rest-group downsampling.

## Ambient RNA: CellSweep

`bv.doublets.cellsweep()` splits every barcode's counts into ambient soup, a
bulk profile and its cell type profile, and writes the denoised counts to a
new directory. It runs after annotation and needs the empty droplets, so
there are two objects: a permissive raw one and a filtered one you cluster and
annotate.

```python
from bixverse.params import EmptyDropletsParams, MinQualityParams

raw = bv.io.load_tenx_h5(
    "exp/raw",
    h5_path,
    qc=MinQualityParams(min_unique_genes=0, min_lib_size=1, min_cells=0),
)
# join `cell_type` from the annotated object onto raw by cell_id (left join,
# unmatched barcodes stay null), add a `sample_id` column, then:
clean = bv.doublets.cellsweep(
    "exp/clean",
    raw,
    "cell_type",
    "sample_id",
    EmptyDropletsParams(method="umi_cutoff", umi_cutoff=257),
)
```

- `EmptyDropletsParams.method`: `"supplied"` (default; a boolean obs column
  named by `is_empty_column`, default `"is_empty"`), `"umi_cutoff"`,
  `"expected_cells"` or `"knee"`. `infer_empty_droplets(lib_size, params)`
  compares the methods before committing.
- Outputs: obs `cellsweep_alpha`, `cellsweep_z`, `cellsweep_beta`,
  `cellsweep_ll`, `cellsweep_converged`, with `nnz` and `lib_size`
  recomputed; var `cellsweep_ambient` and a fresh `no_cells_exp`. The raw
  layer holds stochastically rounded counts.

### Traps

- With the default `freeze_ambient_profile=True` it raises if the smallest
  library size is above 100: the empties were filtered at ingest.
- Format labels before the join. Formatting after it turns nulls into
  `"cluster_null"`, which CellSweep fits as one more cell type.
- `sample_column` is required even for one emulsion. Never pool emulsions.
- Barcodes neither empty nor annotated-and-kept are dropped from the fit and
  the output.
- HVG, PCA and everything after ran on contaminated counts. Redo them on
  `clean`.

## Subsets

For sub-clustering one cell type without touching the parent:

```python
t_cells = bv.SingleCellsSubset(sc, "cell_type", "T cells")
bv.pp.find_hvg(t_cells)
bv.pp.calculate_pca(t_cells, 30)
bv.graph.find_neighbours(t_cells)
bv.graph.find_clusters(t_cells, name="t_subcluster")
bv.merge_subset_obs(sc, t_cells, cols=["t_subcluster"])
```

- The subset shares the parent's stores (no copy) and keeps indices in the
  parent's index space.
- It holds obs and var in memory; nothing it does touches the parent's
  DuckDB. It is not persisted.
- It starts with an empty cache and no HVGs, so the chain restarts at
  `find_hvg()`, which stores the selection but writes no var statistics.
- `merge_subset_obs(sc, subsets, cols=None, new_names=None,
  prefix_values=False, overwrite=False)` joins on `cell_idx`; parent cells in
  no subset get NULL. `prefix_values=True` is needed when several subsets
  share a column whose labels repeat. A parent column clash raises unless
  `overwrite=True`.
- `get_hvg_data()`, `generate_knn()`, `seurat_cca()` and `seurat_rpca()`
  refuse a subset.

## Plotting

`bv.pl` has two layers. The `extract_*` functions return polars frames and
need only the core package. The plots need the `plots` extra (matplotlib):
`embedding_plot(sc, "umap", "leiden_clusters", discrete=True)`,
`feature_plot(sc, gene_ids, "umap")`, `dot_plot(sc, gene_ids,
"leiden_clusters")`, `stacked_violin_plot`, `heatmap_plot`.

- Plots read the kept cells and never write to the experiment.
- Single-panel plots take `ax=` and return the `Axes`; `violin_plot`,
  `joint_plot` and `density_plot` return a `Figure`.
- Gene arguments are var ids; `feature_labels` maps them to symbols. `"pca"`
  works as an embedding after `calculate_pca()`.

## GPU twins

`bixverse_gpu` (`uv pip install "bixverse[gpu]"`, plus `embed` for the
embedding twins) mirrors the CPU modules on wgpu with the same objects and
`bv.params` bundles. Twins: `pp.calculate_pca`; `doublets.scrublet`
(`ScrubletGpuParams`); `integrate.harmony`, `fast_mnn`, `bbknn`;
`graph.generate_knn`, `find_neighbours`, `find_clusters`, `fast_cluster`;
`embed.umap`, `densmap`, `tsne`, `phate`.

- GPU Harmony and fastMNN write `"harmony_gpu"` and `"mnn_gpu"`, not
  `"harmony"` / `"mnn"`. Update `embd_to_use` when you swap the import.
- Without a device the twins raise `bixverse.BixverseError`; probe with
  `bixverse_gpu.gpu_available()`.

## Minimal working example

No downloads, runs in seconds.

```python
import tempfile

import bixverse as bv
from bixverse.params import MinQualityParams, NeighboursParams, SyntheticDataParams

tmp = tempfile.TemporaryDirectory()
sim = bv.io.generate_single_cell_test_data(
    SyntheticDataParams(n_cells=1000, n_genes=100), seed=42
)
sc = bv.io.load_csr(
    f"{tmp.name}/mwe",
    sim.counts,
    sim.obs,
    sim.var,
    qc=MinQualityParams(min_unique_genes=45, min_lib_size=300, min_cells=500),
)
bv.pp.find_hvg(sc, hvg_no=30)
bv.pp.calculate_pca(sc, no_pcs=10)
bv.graph.find_neighbours(
    sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
)
bv.graph.find_clusters(sc, res=1.0, name="clusters")
markers = bv.de.find_all_markers(sc, "clusters")
tmp.cleanup()
```

For real data: `bv.io.download_pbmc3k()`, `download_pbmc8k()`,
`download_pbmc_batches()`, `download_pbmc_1k_5p()`, `download_kang_pbmc()`,
`download_cd34_data()`, `download_demuxlet_pbmc()`, and
`download_pbmc_multiome()` for RNA plus ATAC (h5, fragments with index,
Cell Ranger peaks, GTF). Each caches to `~/.cache/bixverse` and returns the
path.
