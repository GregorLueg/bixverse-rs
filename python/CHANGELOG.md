# Changelog

Versions here track the Python package, not the `bixverse-rs` crate. A wheel
reports the crate it vendored as `bixverse.__core_version__`.

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
