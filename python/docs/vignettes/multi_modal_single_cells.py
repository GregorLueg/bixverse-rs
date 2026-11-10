# %% [markdown]
# # Multi-modal single cell analysis with bixverse
#
# *Python port of the R vignette [Multi-modal single cell analysis with bixverse](https://gregorlueg.github.io/bixverse/articles/multi_modal_single_cells.html). Original author: Gregor Lueg.*
#
# ## Intro
#
# This vignette walks through a multi-modal single cell workflow on a CITE-seq
# data set (10x's 10k Human PBMC TotalSeq-B). The general `bixverse` philosophy
# still applies: do not keep things in memory if you do not have to, keep heavy
# lifting in Rust, and only round-trip to Python when the user actually needs to
# look at something. The bit that changes is that there is now a second modality
# hanging off the main class: Antibody-Derived Tags (ADT). ATAC is on the roadmap
# but not yet wired up.
#
# Before going further, please read the
# [design choices](design_single_cell.ipynb) and the
# [introductory vignette](thinking_single_cell.ipynb) if you have not already.
# This vignette assumes you are comfortable with the `SingleCells` class, the
# on-disk count storage, and the cells-to-keep logic.

# %%
import re
import tempfile
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import bixverse as bv
import bixverse.embed
from bixverse.params import DsbParams, MinQualityParams, NeighboursParams, WnnParams

# %% [markdown]
# ## Loading the data
#
# A 10x Cell Ranger H5 with multiple feature types contains both the RNA and
# the ADT counts in one file, separated by `feature_type`. We peek at the
# metadata first to confirm what is in there before committing to anything.

# %%
h5_10x_path = bv.io.download_pbmc_totalseq_data()

tempdir_10x_total_seq = tempfile.TemporaryDirectory()

# %% [markdown]
# ### EDA

# %%
h5_metadata = bv.io.read_tenx_h5_metadata(h5_10x_path)

h5_metadata["feature_types"]

# %% [markdown]
# Two feature types: gene expression and antibody capture. Good, we have both
# modalities to play with.
#
# ### Loading in the RNA
#
# The RNA goes through `SingleCellsMultiModal`, which is the same class as
# `SingleCells` with extra slots for the ADT counts, the ADT cache and the WNN
# integration (and, later, an ATAC connection). RNA loading and storage are
# identical to the single-modal case: counts get pre-filtered, normalised,
# transposed, and written to the two binary files on disk. The loader writes the
# directory; we then open it as a `SingleCellsMultiModal`.

# %%
exp_dir = Path(tempdir_10x_total_seq.name) / "pbmc_totalseq"

bv.io.load_tenx_h5(exp_dir, h5_10x_path)

sc = bv.SingleCellsMultiModal(exp_dir)

sc.obs().head()

# %% [markdown]
# ### Loading in the ADT counts
#
# The ADT design philosophy differs from RNA. Protein panels are tiny: dozens of
# antibodies, sometimes a couple of hundred for the larger commercial panels.
# The cost-benefit calculus that justifies the on-disk binary format for RNA
# just does not hold here. Even a million cells times 200 proteins fits in
# memory without breaking a sweat. So the whole `AdtCounts` object (raw counts,
# normalised counts, a bit of metadata) lives in memory as a plain matrix. The
# saving is in code complexity rather than RAM.
#
# For normalisation, you currently have two options:
#
# - **CLR** (centred log-ratio): the Seurat default. Comes in two flavours via
#   `seurat_clr`: the original CLR (allows negatives) and the modified
#   Seurat-style version (does not).
# - **DSB** (denoised and scaled by background): from
#   [Mulè et al., 2022](https://www.nature.com/articles/s41467-022-29356-8).
#   More principled if you have empty droplets to estimate the protein
#   background; falls back to a 2-component k-means on log counts if you do
#   not (the "ModelNegativeADTnorm" path).
#
# Isotype controls help DSB tease apart per-cell technical noise from true
# biological signal. This panel names them `Isotype_*`, so a case-insensitive
# regex on "isotype" finds them; for anything unusual just pass the feature ids
# directly via `isotype_names`.
#
# There is no ADT reader for 10x h5 files in Python yet (`read_tenx_h5_adt`), so
# we load the antibody capture modality into its own small store with the QC
# filters off and pull the dense raw counts out of it. Same barcodes as the RNA,
# 140 proteins.

# %%
adt_sc = bv.io.load_tenx_h5(
    Path(tempdir_10x_total_seq.name) / "adt",
    h5_10x_path,
    feature_type="Antibody Capture",
    qc=MinQualityParams(min_unique_genes=0, min_lib_size=0, min_cells=0),
)

adt_counts = bv.get_counts(adt_sc).scipy_csr().toarray()
adt_barcodes = adt_sc.cell_ids()
adt_features = adt_sc.gene_ids().tolist()

iso = [f for f in adt_features if re.search("isotype", f, re.IGNORECASE)]

print(f"ADT counts: {adt_counts.shape[0]} cells x {adt_counts.shape[1]} proteins")
print(f"isotypes: {iso}")

# %%
bv.mm.add_adt_counts(
    sc,
    adt_counts,
    adt_barcodes,
    adt_features,
    method="dsb",
    dsb_params=DsbParams(
        use_isotype_controls=True,
        quantile_low=0.01,
        quantile_high=1.0,
    ),
    isotype_names=iso,
)

sc

# %% [markdown]
# ## QC
#
# ### Low quality cells
#
# QC runs jointly across modalities. The RNA loader already populated
# `lib_size` and `nnz` on the obs table; `add_adt_counts` added `adt_lib_size`
# and `adt_nnz` on top. We feed all four into MAD-based outlier detection.
# Two-sided is sensible across the board: extremely high ADT library size can
# also flag doublets or aggregates, not just dead cells.

# %%
qc_df = sc.obs(
    cols=["cell_id", "lib_size", "nnz", "adt_lib_size", "adt_nnz"], filtered=True
)

metrics = {
    "log10_rna_lib_size": np.log10(qc_df["lib_size"].to_numpy()),
    "log10_rna_nnz": np.log10(qc_df["nnz"].to_numpy()),
    "log10_adt_lib_size": np.log10(qc_df["adt_lib_size"].to_numpy()),
    "log10_adt_nnz": np.log10(qc_df["adt_nnz"].to_numpy()),
}

directions: dict[str, Literal["twosided", "below", "above"]] = dict.fromkeys(
    metrics, "twosided"
)

qc = bv.qc.run_cell_qc(
    metrics,
    sc.cells_to_keep,
    directions=directions,
    threshold=3.0,
)

print(f"{qc.combined.sum()} of {qc.combined.size} cells flagged")

bv.pl.violin_plot(qc)
plt.show()

# %%
sc.add_obs(pl.DataFrame({"cell_idx": qc.cell_idx, "outlier": qc.combined}))

sc.set_cells_to_keep(qc.cell_idx[~qc.combined])

# %% [markdown]
# ## Standard workflows
#
# For the initial steps, the two modalities are treated independently. Each
# gets its own dimensionality reduction, kNN graph, and embeddings. The
# `modality` argument routes to the relevant Rust kernel and stores results in
# the right cache: `sc.cache` for RNA, `sc.adt_cache` for ADT. Cross-modal
# integration comes later, in the WNN section.
#
# ### RNA
#
# Standard recipe: HVGs, PCA, neighbours, Leiden, t-SNE.

# %%
bv.pp.find_hvg(sc, hvg_no=2000)

bv.pp.calculate_pca(sc, no_pcs=30)

# the data is so tiny that exhaustive kNN search is faster than building
# an approximate nearest neighbour index
bv.graph.find_neighbours(
    sc,
    no_embd_to_use=16,
    neighbours_params=NeighboursParams(knn_method="exhaustive"),
)

# %%
bv.graph.find_clusters(sc)

sc.obs(filtered=True).head()

# %%
bv.embed.tsne(sc, modality="rna")

# %%
bv.pl.embedding_plot(
    sc,
    "tsne",
    "leiden_clustering",
    discrete=True,
    point_size=0.5,
)
plt.show()

# %% [markdown]
# ### ADT
#
# For ADT we skip HVG selection, there is nothing to select from in a panel of
# 140 antibodies, and feed the panel directly into PCA, minus the isotype
# controls (they only add background noise). Otherwise the recipe is the same as
# for RNA, with `modality="adt"` doing the routing.

# %%
adt_names = bv.mm.get_adt_names(sc)
assert adt_names is not None

bv.mm.calculate_pca_adt(
    sc,
    no_pcs=15,
    features=[f for f in adt_names if f not in iso],
)

# the data is so tiny that exhaustive kNN search is faster than building
# an approximate nearest neighbour index
bv.graph.find_neighbours(
    sc,
    neighbours_params=NeighboursParams(knn_method="exhaustive"),
    modality="adt",
)

# %%
bv.embed.tsne(sc, modality="adt")

bv.pl.feature_plot(
    sc,
    ["CD3D"],
    "tsne",
    expr_modality="adt",
    embd_modality="adt",
    point_size=1.0,
)
plt.show()

# %% [markdown]
# ## Plotting
#
# `bv.pl` has a few helpers tuned to multi-modal data. The main mechanism is
# that the plotting functions take an `expr_modality` (where the values come
# from) and an `embd_modality` (which embedding to overlay them on) separately.
# That split is what lets you colour an ADT-derived t-SNE by RNA expression, or
# any other combination.
#
# ### Features against each other
#
# The classic CITE-seq sanity check: does the protein signal match the mRNA
# signal where you would expect it to? `feature_scatter_plot` takes two features
# with optional `_rna` / `_adt` suffixes, so you can mix modalities freely. mRNA
# tends to be sparse and noisy; ADT tends to be smoother but with more
# background. A scatter against each other surfaces both effects in one picture.

# %%
ax = bv.pl.feature_scatter_plot(sc, "ENSG00000167286_rna", "CD3D_adt")
ax.set_xlabel("CD3D mRNA")
ax.set_ylabel("CD3D ADT")
plt.show()

# %%
ax = bv.pl.feature_scatter_plot(sc, "ENSG00000177455_rna", "CD19_adt")
ax.set_xlabel("CD19 mRNA")
ax.set_ylabel("CD19 ADT")
plt.show()

# %% [markdown]
# ### Embeddings
#
# Same idea on the embedding side. The cell below paints RNA expression of CD3D
# onto an ADT-derived t-SNE. `feature_labels` is the cleanest way to render a
# human-readable label without renaming columns upstream.

# %%
bv.pl.feature_plot(
    sc,
    ["ENSG00000167286"],
    "tsne",
    feature_labels={"ENSG00000167286": "CD3D_rna"},
    expr_modality="rna",
    embd_modality="adt",
    point_size=1.0,
)
plt.show()

# %% [markdown]
# For the reverse, ADT expression on the same ADT embedding, features just go by
# their panel name:

# %%
bv.pl.feature_plot(
    sc,
    ["CD19"],
    "tsne",
    expr_modality="adt",
    embd_modality="adt",
    point_size=1.0,
)
plt.show()

# %% [markdown]
# `embedding_plot` accepts the same `embd_modality` argument when you want to
# overlay obs columns (clusters, donor IDs, cell type calls) on a particular
# modality's embedding:

# %%
bv.pl.embedding_plot(
    sc,
    "tsne",
    "leiden_clustering",
    embd_modality="adt",
    discrete=True,
    point_size=1.0,
)
plt.show()

# %% [markdown]
# ## WNN
#
# So far we have looked at each modality in isolation. For a single cell type
# assignment that uses both, you want a joint representation. The current
# implementation in `bixverse` is Weighted Nearest Neighbours (WNN) from
# [Hao et al., 2021](https://www.cell.com/cell/fulltext/S0092-8674(21)00583-3),
# the same method Seurat v4 introduced.
#
# The intuition is that different modalities carry different information for
# different cells. For something like monocyte versus B cell, the RNA signal
# is very strong. For CD4 versus CD8 T cell subsets, the protein signal is
# much cleaner: the mRNAs are barely differential at that resolution. WNN
# formalises this: for each cell, it asks how well that cell's neighbours in
# one modality predict its profile in the *other* modality, and uses the
# answer to weight the two modalities per cell. The weighted neighbours are
# then merged into a single joint graph.
#
# The whole thing runs in Rust (per-cell weights, neighbour selection, graph
# merging), so it scales to hundreds of thousands of cells comfortably. The
# resulting graph lands in `sc.wnn` and can be embedded just like any other
# modality.

# %%
bv.mm.generate_wnn_graph(sc, wnn_params=WnnParams(sd_scale=1.0))

bv.mm.get_wnn_weights(sc).head()

# %%
bv.embed.tsne(sc, modality="wnn")

# %% [markdown]
# A nice sanity check is to look at the same marker from both modalities,
# projected onto the WNN embedding. If WNN is doing its job, the two pictures
# should largely agree, even though they come from completely independent
# measurements.

# %%
bv.pl.feature_plot(
    sc,
    ["ENSG00000167286"],
    "tsne",
    feature_labels={"ENSG00000167286": "CD3D_rna"},
    expr_modality="rna",
    embd_modality="wnn",
    point_size=1.0,
)
plt.show()

# %%
bv.pl.feature_plot(
    sc,
    ["CD3D"],
    "tsne",
    feature_labels={"CD3D": "CD3D_adt"},
    expr_modality="adt",
    embd_modality="wnn",
    point_size=1.0,
)
plt.show()

# %% [markdown]
# From here, the rest of the pipeline (clustering, marker detection, cell type
# annotation) works exactly as in the [standard vignette](pbmc_single_cell.ipynb):
# just point the relevant functions at the WNN graph with `modality="wnn"`
# instead of the RNA one.
#
# ## Clean up

# %%
tempdir_10x_total_seq.cleanup()
