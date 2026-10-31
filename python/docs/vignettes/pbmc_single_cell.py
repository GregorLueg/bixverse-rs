# %% [markdown]
# # Analysing PBMCs with bixverse
#
# *Grant Neilson, Gregor Lueg and Liesbeth François*
#
# A standard single cell analysis on the PBMC3k data set with `bixverse`. If you
# have not read the [design choices](design_single_cell.ipynb) and the
# [introductory vignette](thinking_single_cell.ipynb), do so first; this one
# assumes you know how the `SingleCells` class, the on-disk storage and the
# cells-to-keep logic work.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import bixverse as bv
import bixverse.embed
from bixverse.params import NeighboursParams

# %% [markdown]
# ## Loading the data
#
# We download the PBMC3k data and load it through the Cell Ranger style MTX
# reader. The experiment lives in a directory: the cell- and gene-major binary
# stores plus a DuckDB with obs and var.

# %%
pbmc3k_path = bv.io.download_pbmc3k()

tempdir_pbmc = tempfile.TemporaryDirectory()

sc = bv.io.load_mtx(
    Path(tempdir_pbmc.name) / "pbmc3k",
    bv.io.get_cell_ranger_params(pbmc3k_path),
    verbose=True,
)

sc

# %% [markdown]
# A quick look at the var table. The column names from the MTX files are not
# always informative, so we rename the gene symbol column to something sensible
# and set up mappings between Ensembl IDs and symbols.

# %%
sc.var().head()

# %%
sc.rename_var_col("column1", "gene_symbol")

var = sc.var()

ensembl_to_symbol = dict(zip(var["gene_id"], var["gene_symbol"], strict=True))
symbol_to_ensembl = dict(zip(var["gene_symbol"], var["gene_id"], strict=True))

# %% [markdown]
# ## Quality control
#
# ### Gene set proportions
#
# First up, the proportion of counts that map to mitochondrial and ribosomal
# genes per cell. They land directly in the obs table in the DuckDB.

# %%
gs_of_interest = {
    "MT": var.filter(pl.col("gene_symbol").str.contains("^MT-"))["gene_id"].to_list(),
    "Ribo": var.filter(pl.col("gene_symbol").str.contains("^RP[SL]"))[
        "gene_id"
    ].to_list(),
}

bv.qc.gene_set_proportions(sc, gs_of_interest, streaming=False, verbose=True)

sc.obs().head()

# %% [markdown]
# MT and Ribo are now columns in obs.
#
# ### MAD outlier detection
#
# Per-cell QC metrics and MAD-based outlier detection flag the problematic
# cells. `run_cell_qc` returns a `CellQc` that carries the metrics, the
# per-metric outlier calls and a combined outlier vector.

# %%
qc_df = sc.obs(cols=["cell_id", "lib_size", "nnz", "MT"])

metrics = {
    "log10_lib_size": np.log10(qc_df["lib_size"].to_numpy()),
    "log10_nnz": np.log10(qc_df["nnz"].to_numpy()),
    "MT": qc_df["MT"].to_numpy(),
}

directions = {
    "log10_lib_size": "twosided",
    "log10_nnz": "twosided",
    "MT": "above",
}

qc = bv.qc.run_cell_qc(
    metrics,
    sc.cells_to_keep,
    directions=directions,
    threshold=3.0,
)

print(f"{qc.combined.sum()} of {qc.combined.size} cells flagged")

qc.thresholds

# %% [markdown]
# `CellQc` comes with two plots. `violin_plot` draws one violin per metric with
# the outliers highlighted:

# %%
bv.pl.violin_plot(qc)
plt.show()
# %% [markdown]
# and `joint_plot` compares the genes against the UMIs per cell.

# %%
bv.pl.joint_plot(qc)
plt.show()
# %% [markdown]
# ### Filtering cells
#
# We store the outlier flag in obs and then set the cells to keep. From here
# on every downstream method (HVG selection, PCA, ...) only sees the retained
# cells.

# %%
sc.add_obs(pl.DataFrame({"cell_idx": qc.cell_idx, "outlier": qc.combined}))

sc.set_cells_to_keep(qc.cell_idx[~qc.combined])

sc

# %% [markdown]
# ## Feature selection, PCA and neighbours
#
# With QC done, the standard pipeline: highly variable genes, PCA and the
# nearest neighbours.

# %%
bv.pp.find_hvg(sc, hvg_no=2000, verbose=True)

bv.pp.calculate_pca(sc, no_pcs=30, sparse_svd=True)

# the data is so tiny that an exhaustive kNN search is faster than building an
# approximate nearest neighbour index
bv.graph.find_neighbours(
    sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
)

# %% [markdown]
# ## Clustering and marker detection
#
# ### Leiden clustering
#
# Leiden clustering, then differential expression across all clusters. Louvain
# is there too if you want it, but Leiden has the better properties, see
# [Traag et al.](https://www.nature.com/articles/s41598-019-41695-z).

# %%
bv.graph.find_clusters(sc, res=1.0, name="leiden_clusters")

all_markers = bv.de.find_all_markers(sc, "leiden_clusters").with_columns(
    pl.col("gene_id").replace_strict(ensembl_to_symbol).alias("gene_symbol")
)

all_markers.filter(pl.col("fdr") <= 0.05).sort(
    pl.col("lfc").abs(), descending=True
).head()

# %% [markdown]
# ### Fast clustering
#
# For large data sets there is an accelerated clustering: k-means first
# (`sqrt(N)` centroids by default), then Louvain or Leiden on a graph over the
# centroids, across a set of resolutions. The centroid graph options, the seed
# stability statistics and how the result compares against the full graph
# clustering are in the
# [clustering and annotation vignette](cell_type_annotation.ipynb).

# %%
fast_cluster_res = bv.graph.fast_cluster(
    sc,
    resolutions=(2.0, 1.0, 0.5),
    # also return the k-means clustering
    return_kmeans=True,
)

fast_cluster_res.memberships.head()

# %% [markdown]
# The k-means memberships and centroids sit on the result.

# %%
centroids = fast_cluster_res.centroids
kmeans_membership = fast_cluster_res.k_means_cluster

centroids.shape

# %% [markdown]
# The memberships carry `cell_idx`, so they go straight into obs.

# %%
sc.add_obs(fast_cluster_res.memberships)

sc.obs(filtered=True).head()

# %% [markdown]
# ### Dimensionality reduction
#
# UMAP and t-SNE, both from
# [manifolds-rs](https://github.com/GregorLueg/manifolds-rs) (the `embed`
# extra).

# %%
bv.embed.umap(sc)

# perplexity 10 is the default; `bv.embed.tsne_defaults()` has the rest
bv.embed.tsne(sc)

# %% [markdown]
# `bv.pl` (the `plots` extra) has the basic plots. `embedding_plot` shows an
# embedding coloured by an obs column:

# %%
bv.pl.embedding_plot(
    sc,
    "umap",
    "leiden_clusters",
    label_by="leiden_clusters",
    discrete=True,
)
plt.show()
# %% [markdown]
# Same again for the t-SNE:

# %%
bv.pl.embedding_plot(
    sc,
    "tsne",
    "leiden_clusters",
    label_by="leiden_clusters",
    discrete=True,
)
plt.show()
# %% [markdown]
# And `feature_plot` puts the expression of a gene on the embedding:

# %%
bv.pl.feature_plot(
    sc,
    ["ENSG00000156738"],
    "umap",
    feature_labels={"ENSG00000156738": "MS4A1"},
)
plt.show()
# %% [markdown]
# ### Cell type annotation
#
# A set of marker genes for the annotation:

# %%
cell_markers = {
    "CD3D": "T cells",
    "CD3E": "T cells",
    "CD3G": "T cells",
    "IL7R": "CD4+ T",
    "CD4": "CD4+ T",
    "CD8A": "CD8+ T",
    "CD8B": "CD8+ T",
    "MS4A1": "B cells",
    "CD79A": "B cells",
    "CD19": "B cells",
    "CD14": "CD14+ Mono",
    "LYZ": "CD14+ Mono",
    "S100A8": "CD14+ Mono",
    "FCGR3A": "CD16+ Mono",
    "CDKN1C": "CD16+ Mono",
    "GNLY": "NK",
    "NKG7": "NK",
    "NCAM1": "NK",
    "FCER1A": "mDC",
    "CD1C": "mDC",
    "LILRA4": "pDC",
    "CLEC4C": "pDC",
    "PPBP": "Platelet",
    "PF4": "Platelet",
}

# %% [markdown]
# We annotate the PBMCs with scType from
# [Ianevski et al. (2022)](https://doi.org/10.1038/s41467-022-28803-w).
# `prepare_cell_markers` gets the markers into shape, `calc_sc_type_scores`
# scores every cell and `score_clusters` summarises the scores per cluster into
# a cell type call.

# %%
cell_markers_df = pl.DataFrame(
    {"gene_symbol": list(cell_markers), "cell_type": list(cell_markers.values())}
).join(var.select("gene_symbol", "gene_id"), on="gene_symbol")

marker_list = bv.annotate.prepare_cell_markers(sc, cell_markers_df)

sctype_scores = bv.annotate.calc_sc_type_scores(sc, marker_list)

leiden = sc.obs(cols=["leiden_clusters"], filtered=True)["leiden_clusters"]

cell_type_anno = bv.annotate.score_clusters(sctype_scores, leiden.to_numpy())

cell_type_anno

# %% [markdown]
# Back into obs it goes.

# %%
sc.add_obs(
    sc.obs(cols=["cell_idx", "leiden_clusters"], filtered=True)
    .join(
        cell_type_anno.select(
            pl.col("cluster_id").alias("leiden_clusters"),
            pl.col("cell_type").alias("sc_type"),
        ),
        on="leiden_clusters",
        how="left",
    )
    .select("cell_idx", "sc_type")
)

bv.pl.embedding_plot(
    sc,
    "umap",
    "sc_type",
    label_by="sc_type",
    discrete=True,
)
plt.show()

# %% [markdown]
# The marker expression per cluster as a dot plot with `dot_plot`:

# %%
features_vec = [symbol_to_ensembl[s] for s in cell_markers if s in symbol_to_ensembl]

bv.pl.dot_plot(
    sc,
    features_vec,
    "leiden_clusters",
    feature_labels={f: ensembl_to_symbol[f] for f in features_vec},
    # keyed by the display label, here the symbol
    feature_grouping=cell_markers,
    scale_exp=True,
    cluster_groups=True,
)
plt.show()
# %% [markdown]
# ### Per-cell gene expression
#
# Next to the feature plot, a stacked violin plot shows the expression per
# cluster.

# %%
features = [symbol_to_ensembl[s] for s in ("CD3D", "IL7R", "CD8A", "MS4A1")]

bv.pl.stacked_violin_plot(
    sc,
    features,
    "leiden_clusters",
    feature_labels={f: ensembl_to_symbol[f] for f in features},
)
plt.show()
# %% [markdown]
# ## Clean up

# %%
tempdir_pbmc.cleanup()
