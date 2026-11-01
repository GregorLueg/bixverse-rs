# %% [markdown]
# # Single-cell data I/O in bixverse
#
# *Python port of the R vignette [Single-cell data I/O in bixverse](https://gregorlueg.github.io/bixverse/articles/single_cell_io.html). Original authors: Andrew Skelton and Gregor Lueg.*
#
# ## Intro
#
# `bixverse` keeps single-cell counts on disk, in compact Rust binaries for the
# count matrix and a DuckDB for the `obs` / `var` metadata, so that millions of
# cells can be analysed on a laptop. Ingestion happens once, up front, through
# format-specific readers that stream the data into that on-disk layout and apply
# the basic cell/gene QC and normalisations along the way.
#
# This vignette walks through every reader on a small synthetic data set, so it
# runs end-to-end without any downloads. It covers:
#
# - in-memory CSR matrices plus polars tables (`load_csr`)
# - 10x CellRanger MTX, single and multi-sample (`load_mtx`, `load_multi_mtx`)
# - AnnData `h5ad`, single and multi-sample (`load_h5ad`, `load_multi_h5ad`)
# - 10x CellRanger HDF5, single and multi-sample (`load_tenx_h5`,
#   `load_multi_tenx_h5`)
# - checkpointing to and from disk (`save`, `load_existing`)
# - archiving the counts for cold storage (`archive_sc_exp`)
#
# If you have not seen the `SingleCells` class before, read the
# [design choices](design_single_cell.ipynb) vignette first.

# %%
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

import bixverse as bv
from bixverse.params import MinQualityParams, MtxIoParams

# %% [markdown]
# ## A small synthetic data set
#
# We generate a synthetic data set and treat its two halves as two "samples", so
# that the multi-sample readers can be demonstrated too. The counts are a
# cells x genes CSR matrix; `obs` and `var` are polars DataFrames.

# %%
sim = bv.io.generate_single_cell_test_data()

counts = sim.counts.scipy_csr()  # cells x genes
obs = sim.obs
var = sim.var
print(counts.shape)

# two "samples" for the multi-sample examples
counts_a = counts[:500]
counts_b = counts[500:1000]
obs_a = obs.slice(0, 500)
obs_b = obs.slice(500, 500)

# scratch directory; every loader creates its own store directory in here
tempdir_io = tempfile.TemporaryDirectory()
base = Path(tempdir_io.name)

# light QC thresholds, suitable for this small toy data set
qc = MinQualityParams(min_unique_genes=5, min_lib_size=10, min_cells=3)

# %% [markdown]
# ## In-memory matrices
#
# The simplest entry point: hand the reader a cells x genes CSR matrix (a scipy
# `csr_matrix`, or anything else that carries `data`, `indices`, `indptr` and a
# `shape`) plus the `obs` and `var` tables directly.

# %%
sc_mem = bv.io.load_csr(base / "store_mem", counts, obs, var, qc=qc)
sc_mem

# %% [markdown]
# ## 10x CellRanger MTX
#
# ### Single sample
#
# We write one sample out in CellRanger flat-file form and point an explicit
# `MtxIoParams` at the trio (for real CellRanger output, the convenience
# helper `bv.io.get_cell_ranger_params()` auto-locates `matrix.mtx` /
# `barcodes.tsv` / `genes.tsv`).
#
# Two `MtxIoParams` fields handle the common deviations from the strict
# CellRanger convention:
#
# - `cells_as_rows`: whether the .mtx has cells on the rows. The CellRanger
#   default is `False` (genes are rows). Set to `True` for externally produced
#   matrices that are transposed.
# - `has_hdr`: whether the barcodes/features flat files carry a header row.
#   CellRanger itself writes headerless TSVs (`has_hdr=False`), but CSV/TSV
#   exports from downstream pipelines often carry one. The example below uses
#   `has_hdr=True` because `write_cellranger_output()` emits headers.

# %%
dir_a = base / "mtx_a"
# real CellRanger MTX is genes x cells, matching cells_as_rows=False below
bv.io.write_cellranger_output(dir_a, counts_a, obs_a, var, rows="genes")

mtx_params = MtxIoParams(
    path_mtx=str(dir_a / "matrix.mtx"),
    path_obs=str(dir_a / "barcodes.csv"),
    path_var=str(dir_a / "features.csv"),
    cells_as_rows=False,
    has_hdr=True,
)

sc_mtx = bv.io.load_mtx(base / "store_mtx", mtx_params, qc=qc)
sc_mtx

# %% [markdown]
# ### Multiple samples
#
# `prescan_mtx_dirs()` takes one directory per sample, builds the shared gene
# universe across them, and `load_multi_mtx()` then stacks all cells into a
# single object with a global gene QC. The originating sample is recorded in the
# `exp_id` `obs` column.

# %%
dir_b = base / "mtx_b"
bv.io.write_cellranger_output(dir_b, counts_b, obs_b, var, rows="genes")

ps_mtx = bv.io.prescan_mtx_dirs(
    [dir_a, dir_b],
    ["sample_a", "sample_b"],
    has_hdr=True,
)

sc_multi_mtx = bv.io.load_multi_mtx(base / "store_multi_mtx", ps_mtx, qc=qc)
sc_multi_mtx

# %%
# cells per sample
sc_multi_mtx.obs().group_by("exp_id", maintain_order=True).len()

# %% [markdown]
# ### File layout variations
#
# To make the `cells_as_rows` and `has_hdr` knobs concrete, here is the same
# sample written with cells on the rows of the .mtx (i.e. transposed relative to
# CellRanger). The only thing that changes at load time is `cells_as_rows=True`:

# %%
dir_swapped = base / "mtx_swapped"
bv.io.write_cellranger_output(dir_swapped, counts_a, obs_a, var, rows="cells")

mtx_params_swapped = MtxIoParams(
    path_mtx=str(dir_swapped / "matrix.mtx"),
    path_obs=str(dir_swapped / "barcodes.csv"),
    path_var=str(dir_swapped / "features.csv"),
    cells_as_rows=True,
    has_hdr=True,
)

sc_mtx_swapped = bv.io.load_mtx(base / "store_mtx_swapped", mtx_params_swapped, qc=qc)
sc_mtx_swapped

# %% [markdown]
# The resulting on-disk object is identical to the standard-layout single-sample
# load above; only the path through the reader differs.

# %%
raw_std = bv.get_counts(sc_mtx).matrix
raw_swapped = bv.get_counts(sc_mtx_swapped).matrix

assert sc_mtx.obs().equals(sc_mtx_swapped.obs())
assert np.array_equal(raw_std.indptr, raw_swapped.indptr)
assert np.array_equal(raw_std.indices, raw_swapped.indices)
assert np.array_equal(raw_std.data, raw_swapped.data)

# %% [markdown]
# For headerless barcodes/features files (the strict CellRanger convention), set
# `has_hdr=False`. The same flag applies uniformly across all input directories
# in `prescan_mtx_dirs()` for the multi-sample case.
#
# ## AnnData h5ad
#
# ### Single
#
# `read_h5ad_metadata()` lets you inspect the `obs` / `var` and dimensions
# without loading the counts; `load_h5ad()` ingests the matrix.

# %%
h5_a = base / "sample_a.h5ad"
bv.io.write_h5ad_sc(h5_a, counts_a, obs_a, var)

meta = bv.io.read_h5ad_metadata(h5_a)
print(meta["dims"])

sc_h5 = bv.io.load_h5ad(base / "store_h5", h5_a, qc=qc)
sc_h5

# %% [markdown]
# Two more h5ad entry points: `stream_h5ad()` is `load_h5ad(h5ad_streaming=True)`
# under its own name, and `load_h5ad_norm()` loads files that only carry
# log-normalised counts, reconstructing the raw counts from a library size
# column in obs and the original target size.
#
# ### Multiple
#
# `prescan_h5ad_files()` accepts a dict of paths (the keys become the sample
# identifiers); `load_multi_h5ad()` mirrors the multi-MTX workflow.

# %%
h5_b = base / "sample_b.h5ad"
bv.io.write_h5ad_sc(h5_b, counts_b, obs_b, var)

ps_h5 = bv.io.prescan_h5ad_files({"sample_a": h5_a, "sample_b": h5_b})

sc_multi_h5 = bv.io.load_multi_h5ad(base / "store_multi_h5", ps_h5, qc=qc)
sc_multi_h5

# %% [markdown]
# ### Export
#
# `save_h5ad()` goes in the other direction: it writes a loaded object to
# `.h5ad` for reading with ScanPy. Only kept cells are written; `assay` picks raw
# or normalised counts for `X`, and any cached PCA, embeddings or sNN graph go to
# `obsm`/`varm`/`obsp`. The counts are streamed in cell batches, so the full
# matrix never sits in memory.

# %%
h5_out = base / "export.h5ad"
bv.io.save_h5ad(sc_h5, h5_out, assay="raw")

print(bv.io.read_h5ad_metadata(h5_out)["dims"])

# %% [markdown]
# ## 10x CellRanger HDF5
#
# ### Single sample
#
# `write_tenx_h5_sc()` expects the counts plus a barcode list and a `features`
# table (with at least `id` and `name`). `load_tenx_h5()` reads the
# gene-expression modality (V3 files may also carry e.g. Antibody Capture, which
# is filtered out via `feature_type`).

# %%
features = pl.DataFrame({"id": var["gene_id"], "name": var["ensembl_id"]})
h5_tenx = base / "sample_a_10x.h5"
bv.io.write_tenx_h5_sc(
    h5_tenx,
    counts_a,
    obs_a["cell_id"].to_list(),
    features,
    version="v3",
)

sc_10x = bv.io.load_tenx_h5(
    base / "store_10x", h5_tenx, qc=qc, feature_type="Gene Expression"
)
sc_10x

# %% [markdown]
# ### Multiple samples
#
# `prescan_tenx_h5_files()` takes a dict of `.h5` paths (the keys become the
# sample identifiers), inspects each file for its CellRanger version (v2 vs
# v3), filters to the requested `feature_type`, and builds the global gene
# space. `load_multi_tenx_h5()` then mirrors the multi-MTX / multi-h5ad
# workflow.
#
# `gene_universe` controls how the feature space is reconciled across files:
#
# - `"intersection"` keeps genes present in **every** input.
# - `"union"` keeps genes present in **any** input; cells from a file that lacks
#   a given gene get zero counts for it.
#
# Mixed v2 / v3 inputs in the same call are handled transparently, and non-gene
# modalities (e.g. Antibody Capture) are silently dropped when
# `feature_type="Gene Expression"`.

# %%
features_b = features.with_columns(feature_type=pl.lit("Gene Expression"))
h5_tenx_b = base / "sample_b_10x.h5"
bv.io.write_tenx_h5_sc(
    h5_tenx_b,
    counts_b,
    obs_b["cell_id"].to_list(),
    features_b,
    version="v3",
)

ps_10x = bv.io.prescan_tenx_h5_files(
    {"sample_a": h5_tenx, "sample_b": h5_tenx_b},
    feature_type="Gene Expression",
    gene_universe="intersection",
)

sc_multi_10x = bv.io.load_multi_tenx_h5(base / "store_multi_10x", ps_10x, qc=qc)
sc_multi_10x

# %%
# cells per sample
sc_multi_10x.obs().group_by("exp_id", maintain_order=True).len()

# %% [markdown]
# For multi-modal `.h5` files that carry both Gene Expression and Antibody
# Capture features, load the gene side as above. There is no dedicated ADT
# reader for 10x h5 in Python yet (`read_tenx_h5_adt`); load the file a second
# time with `feature_type="Antibody Capture"` into its own store, read the
# counts back and attach them to a `SingleCellsMultiModal` via
# `bv.mm.add_adt_counts()`. The
# [multi-modal vignette](multi_modal_single_cells.ipynb) does exactly that.
#
# ## Checkpointing to disk
#
# Once an object is loaded, the in-memory mappings can be persisted so a fresh
# session can be restored without re-ingesting the counts. The on-disk binary and
# DuckDB already live in the store directory; `save()` adds the lightweight
# mapping cache.

# %%
bv.io.save(sc_h5)

sc_restored = bv.io.load_existing(base / "store_h5")
sc_restored

# %% [markdown]
# ### Archiving for cold storage
#
# Project done, but you want to keep the data around? The two count binaries
# eat disk. `archive_sc_exp()` compresses `counts_cells.bin` into a single zstd
# `counts.bxa` and, by default, deletes both binaries. The gene-based file is
# not archived at all, it's just a transpose of the cell-based one. Normalised
# values only go into the archive where they can't be recomputed from the raw
# counts. `level` (1 to 22) trades write time for size; decompression speed
# barely changes. Run `save()` first, the archive only covers the counts.

# %%
dir_h5 = base / "store_h5"
bin_bytes = sum(
    (dir_h5 / f).stat().st_size for f in ("counts_cells.bin", "counts_genes.bin")
)

archive_stats = bv.io.archive_sc_exp(sc_restored, verbose=True)

print(f"binaries: {bin_bytes / 1e6:.3f} MB")
print(f"archive:  {archive_stats['archive_bytes'] / 1e6:.3f} MB")
sorted(p.name for p in dir_h5.iterdir())

# %% [markdown]
# Nothing changes on the way back in. `load_existing()` spots that only the
# archive is there and rebuilds both binaries before loading.

# %%
sc_archived = bv.io.load_existing(dir_h5)

raw_archived = bv.get_counts(sc_archived).matrix
raw_h5 = bv.get_counts(sc_h5).matrix

assert np.array_equal(raw_archived.indptr, raw_h5.indptr)
assert np.array_equal(raw_archived.indices, raw_h5.indices)
assert np.array_equal(raw_archived.data, raw_h5.data)

# %% [markdown]
# ## Summary
#
# | Format | Single sample | Multiple samples |
# | --- | --- | --- |
# | In-memory CSR | `bv.io.load_csr()` | |
# | 10x MTX | `bv.io.load_mtx()` | `bv.io.prescan_mtx_dirs()` + `bv.io.load_multi_mtx()` |
# | AnnData h5ad | `bv.io.load_h5ad()` | `bv.io.prescan_h5ad_files()` + `bv.io.load_multi_h5ad()` |
# | 10x HDF5 | `bv.io.load_tenx_h5()` | `bv.io.prescan_tenx_h5_files()` + `bv.io.load_multi_tenx_h5()` |
#
# Checkpoint with `bv.io.save()` and `bv.io.load_existing()`; archive with
# `bv.io.archive_sc_exp()`. Already loaded experiments can be stacked into a new
# one with `bv.io.merge_sc_experiments()`, and stores in the old v3 binary
# format are converted in place with `bv.io.upgrade_sc_store()`. There is no
# Seurat reader on the Python side.
#
# All readers share the same `MinQualityParams` QC interface and write into the
# same on-disk `SingleCells` layout, so everything downstream (normalisation,
# HVGs, PCA, neighbours, clustering, ...) is identical regardless of how the data
# came in.
#
# ## Clean up

# %%
tempdir_io.cleanup()
