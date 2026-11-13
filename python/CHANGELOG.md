# Changelog

Versions here track the Python package, not the `bixverse-rs` crate. A wheel
reports the crate it vendored as `bixverse.__core_version__`.

## 0.4.0

Builds against the `bixverse-rs` workspace at 0.11.0.

**Breaking**

- k-means defaults to e-means in Harmony, fast_cluster and SCENIC, in both
  wheels. Results differ at the same seed. The new `k_means_algorithm` and
  `e_means_epochs` keys are not in the generated parameters yet; they follow
  once the R specs land and `dev/gen_params.py` is rerun.

**Changes**

- Harmony is faster on CPU and GPU. Its block order is now drawn once per run,
  which changes results at the same seed.

## 0.3.2

Builds against `bixverse-rs` workspace at 0.10.3.

**Fixes**

- Takes the NEBULA-LN stage fix forward from the meta crate.

## 0.3.1

Builds against the `bixverse-rs` workspace at 0.10.1. R parity is tested against
bixverse 0.9.0 with the matching BBKNN changes.

**Fixes**

- `integrate.bbknn`: the default `trim` is `10 * neighbours_within_batch *
  n_batches`, as in the reference, and the stored kNN keeps the nearest
  `ceil(k / n_batches)` neighbours per batch instead of the `k` nearest overall.
  Both the kNN and the connectivities change on the defaults. Same for the GPU
  wheel.
- Batch correction vignette: the BBKNN section is rewritten. kBET and iLISI no
  longer count for BBKNN, and it is judged on graph weights and cluster mixing.

## 0.3.0

Builds against the `bixverse-rs` workspace at 0.10.0: `bixverse-core`,
`bixverse-methods`, `bixverse-sc` and `bixverse-sc-data`, plus `bixverse-gpu` and
`bixverse-sc-gpu` for the GPU wheel. R parity is tested against bixverse 0.9.0.

**Features**

- The single cell suite of the bixverse R package, ported, from loading and QC
  through integration, DE, meta cells, trajectories and pathways. `SingleCells`
  opens the same experiment directories as R, and parity is tested function by
  function. The README lists what each module holds.
- `bixverse-gpu`, a separate wheel (`gpu` extra) with GPU twins under the same
  module names.
- `bixverse.nn` on JAX (`bixverse[jax]`, `backend="jax"`): the same scVI as the
  torch and MLX backends. On 5963 PBMC cells x 2000 HVGs, 400 epochs on an M1
  Max: JAX on CPU 106 s against 202 s for torch on CPU, with the same held-out
  ELBO.
- `train_scvi` takes `verbose`: a tqdm bar over the epochs with the current loss,
  and at `2` one logged line per epoch.
- Log messages show without any logging setup, unless the root logger has
  handlers. `logging.getLogger("bixverse").setLevel(logging.WARNING)` silences
  them.
- 22 vignettes as executed notebooks, API reference pages and a rewritten README.

**Notes**

- `duckdb`, `polars` and `tqdm` are now required dependencies.
- The 2D embeddings need `manifolds-rs` 0.6.0. UMAP and densMAP take its
  `optimiser="sgd_parallel"` (lock-free SGD, not reproducible run to run), and
  `n_epochs` defaults to `None`: 500, or 200 for the SGD optimisers above
  10,000 cells. A cached kNN wider than `n_neighbors` is cut to it. PHATE runs
  its own `6k + 5` wide search by default (`use_knn=False`), so it no longer
  matches R's PHATE, which reuses the cached kNN.
- `bixverse_gpu.embed` has `umap`, `densmap`, `tsne` and `phate`.
- Harmony v2 is the default: `bv.integrate.harmony`, `step_harmony` and
  `build_symphony_ref` without `harmony_params` run it, and it lands under
  `"harmony"`. The original is `harmony_v1` (`step_harmony_v1`, under
  `"harmony_v1"`). The GPU twin is `bixverse_gpu.integrate.harmony`. In
  bixverse 0.9.0 `harmony_sc` is still v1, so R's `"harmony"` is Python's
  `"harmony_v1"`. Both v2s estimate lambda per cluster by default
  (`use_dynamic_lambda=True`); bixverse.gpu's R spec still says `FALSE`.
- Every binding takes NumPy input in C order only and raises `ValueError` on a
  Fortran-ordered array, which was read silently transposed. `KnnGraph` copies
  its arrays to C order, so a hand-built graph needs no care.
- `save_h5ad` writes `varm/PCs` on the genes the PCA ran on (the cache keeps
  them as `pca_genes`), and only with a fresh `X_pca`; it used to place the
  loadings on the current HVGs.
- `add_obs` / `add_var` warn when replacing a column would set values of rows
  the new frame does not cover to NULL, e.g. writing QC metrics of the kept
  cells back over a column every cell had.
- `io.save` drops the residual model (it is not persisted yet) but keeps what
  was built from it valid on load: a residual PCA and its kNN no longer come
  back stale.

## 0.2.1

Requires `bixverse-sc-data` 0.9.1.

**Features**

- Smaller wheel: depends on `bixverse-sc-data` without its `analysis` feature,
  which drops `faer`, `ann-search-rs`, `statrs` and `wide` from the build.

## 0.2.0

Requires `bixverse-sc-data` 0.9.0.

**Features**

- Reads stores of file version 4, the new format of the crate: delta
  bit-packed chunks that are about five times smaller on disk. Stores of
  version 3 keep opening, so no existing file needs converting.
- Cold reads prefetch their byte range into the page cache with large
  parallel reads (unix), which roughly doubled cold read throughput on the
  stores measured.

**Notes**

- Version 4 files are not readable by wheels built against `bixverse-sc-data`
  0.8.x.

## 0.1.1

Requires `bixverse-sc-data` 0.8.0.

**Features**

- Leverages the redesign of the crate into smaller subcrates.

## 0.1.0

First release.

- `CellReader` opens a bixverse binary sparse store and reports its shape,
  layout and normalisation target size.
- `CellLoader` iterates a cell-major store in minibatches. One feeder thread
  decompresses ahead of the consumer into a bounded queue, with the GIL
  released while it works. Each pass reshuffles and spawns its own feeder, so
  concurrent passes do not share state.
- Batches are sparse CSR by default. On a 20k x 20k store at 10% density the
  dense layout measured 53% slower per epoch, which is what settled the
  default.
- `Batch.scipy_csr()` rebuilds a batch as a `scipy.sparse.csr_matrix` over the
  same buffers.
- Errors from the crate surface as `BixverseError` and `SparseFileError`, with
  missing and unreadable files mapping to the built-in `OSError` hierarchy.
- `CellLoader(genes=...)` loads a gene subset in the order given, on both the
  sparse and the dense path, and every `Batch` carries `cell_ids`, the store
  index of each row, so per-cell covariates can be joined back on.
- `bixverse.nn`: a barebone scVI trained straight off a store, on torch
  (`bixverse[torch]`) or MLX (`bixverse[mlx]`, Apple Silicon). `train_scvi`,
  `scvi_latent` and `scvi_elbo`, with scvi-tools 1.5.1's defaults (ZINB or NB,
  observed library size, gene-wise dispersion). A trained scvi-tools module
  loads into it and reproduces its loss to `rtol = 1e-5`. On 5257 PBMC cells
  x 2000 HVGs, 400 epochs on an M1 Max: scvi-tools on MPS 170 s, ours on MPS
  74 s, ours on MLX 31 s, with held-out ELBO and integration metrics inside
  scvi-tools' seed-to-seed spread.
