# Changelog

Versions here track the Python package, not the `bixverse-rs` crate. A wheel
reports the crate it vendored as `bixverse.__core_version__`.

## 0.3.0

Requires `bixverse-sc-data` 0.9.5.

**Features**

- `bixverse.nn` on JAX (`bixverse[jax]`, `backend="jax"`): the same scVI as
  the torch and MLX backends, as pure functions over parameter pytrees with an
  optax Adam and one `jax.jit` per training step. Torch weights convert with
  `_jax.torch_state_to_jax` and reproduce the loss to `rtol = 1e-5`. On 5963
  PBMC cells x 2000 HVGs, 400 epochs on an M1 Max: JAX on CPU 108 s against
  200 s for torch on CPU, with the same held-out ELBO. On a GPU MLX (33 s) and
  torch on MPS (82 s) stay ahead.

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
