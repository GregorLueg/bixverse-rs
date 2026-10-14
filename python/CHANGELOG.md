# Changelog

Versions here track the Python package, not the `bixverse-rs` crate. A wheel
reports the crate it vendored as `bixverse.__core_version__`.

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
