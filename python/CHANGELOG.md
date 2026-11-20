# Changelog

Versions here track the Python package, not the `bixverse-rs` crate. A wheel
reports the crate it vendored as `bixverse.__core_version__`.

## 0.4.0

Builds against the `bixverse-rs` workspace at 0.11.0.

**Breaking**

- Loaders write store format v5, which 0.3.x and R builds on crates 0.10.x
  cannot read. A v4 store opens with a warning; a v3 store no longer opens.
  `bv.io.upgrade_sc_store` converts either and takes a directory for v3.
- `FastClusterDefault` and `KmeansDefaults` are gone from `bv.params`, and so
  is `NmfConsensusParams.kmeans_iters`.
- The parameter bundles are written in Python now, no longer generated from
  the R specs. `bixverse.params._generated` is `bixverse.params._specs`.
- `bv.io.load_tenx_h5(feature_type="Peaks")` raises; peaks load through
  `bv.atac.add_atac_tenx` or `bv.io.load_atac_tenx`.
- Opening an experiment checks that its cell and gene files belong together,
  and raises when they do not.

**Features**

- `bv.atac`, single cell ATAC on `SingleCellsMultiModal`: 10x multiome peaks
  or fragment files counted into peaks or tiles, fragment QC, AMULET
  multiplets, peak calling per group (a MACS3 `callpeak` port), TF-IDF, top
  features, LSI, LSI projection, rLSI integration and gene activity; TSS
  profiles and fragment length histograms, blacklist removal, closest-gene
  annotation, fragment file merging, `barcode_map` for barcodes that differ
  from the cell ids, and peak to gene links (`peak_to_gene`). Fragment
  passes raise when no cell id is a barcode of the file or the fragments
  share no chromosome name with the features, and warn on partial mismatches.
- `bv.SingleCellsAtac`, ATAC without RNA: `bv.io.load_atac_fragments` (cells
  from a barcode list, Cell Ranger's calls or a fragment count cut-off) and
  `bv.io.load_atac_tenx`. Every `bv.atac` function, the graph, embedding,
  marker and plot functions take it; `bv.io.load_existing` opens it.
- The ATAC state is rebuilt from the store on open; `bv.io.save` keeps only
  the LSI. `bv.atac.get_gene_activity` reopens the gene activity with the
  experiment's cell filter, which `set_cells_to_keep` now carries over.
- Plots: `bv.pl.tss_profile_plot`, `fragment_length_plot`,
  `lsi_depth_plot`; peaks on a plot with `expr_modality="atac"`;
  `feature_plot(expr_sc=...)` colours one object's embedding with another's
  expression; `joint_plot` takes axis labels.
- `transfer_labels_seurat(weight_owner=...)` checks the weight embedding keeps
  the query's cells.
- Methods that assume RNA counts refuse a gene activity store (residuals,
  doublets, Bonsai, gene set proportions) or warn on it, in both wheels.
- `modality="atac"` on `bv.graph`, `bv.embed`, `bv.integrate.harmony`, the
  WNN (`modality_2="atac"`, `embd_to_use_2="lsi"`), the marker functions and
  pseudobulk DE. Same for the GPU wheel's graph, embeddings and Harmony.
- The marker functions take `method="lr"`, Seurat's logistic regression test
  with the library size as covariate. `None`, the new default, picks `"lr"` on
  ATAC and `"wilcox"` on RNA.
- Harmony (v1, v2, the GPU twin and the pipeline steps) takes `embd_to_use`,
  so it corrects the ATAC LSI as well as a PCA.
- `bv.annotate.transfer_labels_seurat`: Seurat's CCA anchors and
  `TransferData`, RNA to RNA or RNA to gene activity.
- The h5ad loaders and readers take anndata Zarr stores (v2 and v3, Blosc
  included); `bv.io.save_zarr` writes one, `genes=` included.
- `bv.io.save_h5ad` takes `genes=` (indices or ids, in column order) and
  streams only those genes out of the store (#129).
- `bv.io.download_pbmc_multiome` (now with the ENCODE hg38 blacklist), a
  joint RNA and ATAC vignette and an ATAC-only vignette on it.
- `node_order` on `find_clusters` and the fast clustering bundles; the
  Harmony bundles take `k_means_algorithm` and `e_means_epochs`, fast
  clustering `km_type="emeans"` and `e_means_epochs`.

**Changed results**

- e-means is the k-means default of Harmony v1 and v2, fast clustering (also
  inside Boost and scDblFinder), SCENIC, DSB and NMF consensus, on CPU and
  GPU. Results differ at the same seed; `km_type="standard"` or
  `k_means_algorithm="lloyd"` gives Lloyd's back.
- Leiden runs 2 iterations by default (was 5) and Louvain coloured local
  moving, fast clustering included. Results differ at the same seed.
- Harmony v2 follows harmony2: `HarmonyV2Params.block_size` is 0.05 (was
  0.2), and the GPU twin estimates lambda per cluster and runs a fixed
  iteration count under Lloyd's. Results differ at the same seed.
- Harmony draws its block order once per run. Results differ at the same
  seed.
- rPCA and CCA integration search within-batch neighbours at `k + 1` and cap
  each cell at `k_weight` anchor pairs, as Seurat does. Results differ at the
  same seed.
- The normalised layer is read at f32, also from v4 stores, which gave f16
  before. Values differ slightly.

**Fixes**

- `csc_mem_gb` also bounds the transpose of a streamed CSC h5ad, which held it
  whole before.
- `scrublet`, `boost`, `scdblfinder` and the GPU `scrublet` read a small cell
  selection, such as one `group_by` group, from the cell-based store instead
  of sweeping the gene-based one (#134). Same results at the same seed.

**Notes**

- The parity tests against R are gone; `tests/test_r_interop.py` only checks
  that a store written on one side opens on the other.

## 0.3.3

Builds against `bixverse-rs` workspace at 0.10.3.

**Features**

- Agent skill: `bixverse/_skill/` teaches coding agents the package API. It
  ships in the wheel. `bixverse skill install [--agent codex] [--project]`
  (or `bv.install_agent_skill()`) installs it; `bixverse skill status` lists
  installs.
- Installs are version-stamped; `import bixverse` warns when one is outdated.
  A test fails when the generated API index is stale.
- `bixverse-gpu`: an experimental MLX backend (mlx-c) on Apple Silicon. PCA,
  the kNN, fast clustering, Harmony and SEACells take `backend="mlx"`. Source
  build only (`--features mlx`, see the README); published wheels are
  unchanged. `bixverse_gpu.mlx_available()` probes it. Results on the default
  `backend="wgpu"` are unchanged.

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
