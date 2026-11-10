# %% [markdown]
# # Cell trees with Bonsai
#
# *Python port of the R vignette [Cell trees with Bonsai](https://gregorlueg.github.io/bixverse/articles/bonsai.html). Original author: Gregor Lueg.*
#
# ## Intro
#
# UMAP and t-SNE squash a kNN graph into two dimensions and lose most of the
# global structure on the way.
# [Bonsai (de Groot, et al., Nat. Biotechnol., 2026)](https://www.nature.com/articles/s41587-026-03220-2)
# takes a different route: it reconstructs a tree over the cells, every cell a
# leaf and every internal node an inferred ancestral state, with branch lengths
# that carry how much changed between them. The tree is the representation; a
# 2D layout is just a way to draw it.
#
# What makes it work is that Bonsai uses the error bar on every measurement, not
# only the measurement. Those come from
# [Sanity (Breda, et al., Nat. Biotechnol., 2021)](https://www.nature.com/articles/s41587-021-00875-x),
# which turns raw UMI counts into posterior log fold changes with error bars,
# gene by gene. `bv.trees.bonsai()` runs both: counts from disk, Sanity, Bonsai,
# layout. Both are Rust ports under the hood,
# [`sanity-sc-rs`](https://github.com/GregorLueg/sanity-sc-rs) and
# [`bonsai-rs`](https://github.com/GregorLueg/bonsai-rs).
#
# We use the PBMC3k data here. If you have not worked through the
# [PBMC vignette](pbmc_single_cell.ipynb), do that first; the preprocessing
# below is the same, just condensed.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection

import bixverse as bv
import bixverse.embed
from bixverse.params import NeighboursParams

# %% [markdown]
# ## Preparing the data
#
# Loading, QC and the usual HVG, PCA, neighbours and clustering chain. Bonsai
# itself needs none of the last four, but the clusters and cell types make the
# tree readable, and the UMAP gives us something to compare against.

# %%
pbmc3k_path = bv.io.download_pbmc3k()
tempdir_pbmc = tempfile.TemporaryDirectory()

# load
sc = bv.io.load_mtx(
    Path(tempdir_pbmc.name) / "pbmc3k",
    bv.io.get_cell_ranger_params(pbmc3k_path),
)

sc.rename_var_col("column1", "gene_symbol")
var = sc.var()

# doublets
scrublet_res = bv.doublets.scrublet(sc)
sc.add_obs(bv.doublets.get_data(scrublet_res))

sc.set_cells_to_keep(
    sc.obs(cols=["cell_id", "doublet"]).filter(~pl.col("doublet"))["cell_id"].to_list()
)

# qc
bv.qc.gene_set_proportions(
    sc,
    {"MT": var.filter(pl.col("gene_symbol").str.contains("^MT-"))["gene_id"].to_list()},
    streaming=False,
)

qc_df = sc.obs(cols=["cell_id", "lib_size", "nnz", "MT"], filtered=True)
qc = bv.qc.run_cell_qc(
    {
        "log10_lib_size": np.log10(qc_df["lib_size"].to_numpy()),
        "log10_nnz": np.log10(qc_df["nnz"].to_numpy()),
        "MT": qc_df["MT"].to_numpy(),
    },
    sc.cells_to_keep,
    directions={
        "log10_lib_size": "twosided",
        "log10_nnz": "twosided",
        "MT": "above",
    },
    threshold=3.0,
)
sc.set_cells_to_keep(qc.cell_idx[~qc.combined])

# hvg, pca, knn, clustering, umap
bv.pp.find_hvg(sc, hvg_no=2000)
bv.pp.calculate_pca(sc, no_pcs=30, verbose=True)
bv.graph.find_neighbours(
    sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
)
bv.graph.find_clusters(sc, res=1.0, name="leiden_clusters")
bv.embed.umap(sc)

sc

# %% [markdown]
# Cell types via scType, with the same markers as the PBMC vignette.

# %%
cell_markers = {
    "CD3D": "T cells",
    "CD3E": "T cells",
    "IL7R": "CD4+ T",
    "CD4": "CD4+ T",
    "CD8A": "CD8+ T",
    "CD8B": "CD8+ T",
    "MS4A1": "B cells",
    "CD79A": "B cells",
    "CD14": "CD14+ Mono",
    "LYZ": "CD14+ Mono",
    "FCGR3A": "CD16+ Mono",
    "CDKN1C": "CD16+ Mono",
    "GNLY": "NK",
    "NKG7": "NK",
    "FCER1A": "mDC",
    "CD1C": "mDC",
    "LILRA4": "pDC",
    "PPBP": "Platelet",
    "PF4": "Platelet",
}

cell_markers_df = pl.DataFrame(
    {"gene_symbol": list(cell_markers), "cell_type": list(cell_markers.values())}
).join(var.select("gene_symbol", "gene_id"), on="gene_symbol")

sctype_scores = bv.annotate.calc_sc_type_scores(
    sc, bv.annotate.prepare_cell_markers(sc, cell_markers_df)
)
obs = sc.obs(cols=["cell_idx", "leiden_clusters"], filtered=True)
cell_type_anno = bv.annotate.score_clusters(
    sctype_scores, obs["leiden_clusters"].to_numpy()
)

sc.add_obs(
    obs.join(
        cell_type_anno.select(
            pl.col("cluster_id").alias("leiden_clusters"), "cell_type"
        ),
        on="leiden_clusters",
        how="left",
    ).select("cell_idx", "cell_type")
)

# %% [markdown]
# ## Running Bonsai
#
# One call. No HVG selection needed: by default every gene is a candidate, and
# Sanity streams them through in chunks of 1,024, keeping only the genes with
# enough signal over their own noise (`min_signal_to_noise` in `BonsaiParams`,
# 1 by default). That is the gene selection of the Bonsai paper. Memory stays at
# one chunk of counts plus the survivors, whatever the number of candidates.

# %%
tree = bv.trees.bonsai(sc, verbose=True)

tree

# %% [markdown]
# Where did the time go? Every stage is timed, and `total` is the whole call as
# Python saw it, so a gap between `total` and the stages would be time spent
# somewhere nobody measured. Sanity fits every one of the 11k candidate genes,
# the search only sees the few hundred that survive but has to place 2,400
# leaves, and on PBMC3k the search ends up taking the bigger share.

# %%
tree.timings

# %% [markdown]
# The search reports its steps too: the loglikelihood after each and what it
# took. Steps 4 and 7 re-fit the branch lengths, 5 and 6 rearrange the tree
# (subtree pruning and regrafting, then nearest-neighbour interchanges), and 8
# collapses the zero-length edges the rest leave behind.

# %%
tree.steps

# %% [markdown]
# ### Which genes made it?
#
# Most genes carry no usable signal over their Poisson noise and are dropped.
# The ones that stay overlap with, but are not, the HVGs.

# %%
hvg_ids = set(sc.gene_ids(sc.map.hvg))

pl.DataFrame(
    {
        "genes_used": [len(tree.genes_used)],
        "genes_dropped": [len(tree.genes_dropped)],
        "also_hvg": [sum(g in hvg_ids for g in tree.genes_used)],
    }
)

# %% [markdown]
# Want the tree on your own gene set instead? Pass `hvg` (0-based), e.g.
# `bv.trees.bonsai(sc, hvg=sc.map.hvg)`. The signal-to-noise filter still
# applies on top.
#
# ## Plotting
#
# There is no tree plot in `bv.pl` yet, so here is a short matplotlib one: every
# branch as a segment from a node to its parent and a point per leaf. Radial
# layouts get straight branches at a fixed aspect ratio, the dendrogram right
# angles. `colour_by` takes one value per leaf, in the order of the object's
# kept cells.


# %%
def plot_tree(tree, colour_by=None, point_size=2.0, size_by_cells=False, ax=None):
    ax = ax or plt.subplots(figsize=(7, 7))[1]
    nodes = tree.nodes
    xy = nodes.select("x", "y").to_numpy()
    edges = nodes.filter(pl.col("parent").is_not_null())
    child = xy[edges["node"].to_numpy()]
    parent = xy[edges["parent"].to_numpy()]
    if tree.layout == "dendrogram":
        # depth along x, leaves along y: across to the parent's depth, then up
        elbow = np.column_stack([parent[:, 0], child[:, 1]])
        segs = np.concatenate(
            [np.stack([child, elbow], 1), np.stack([elbow, parent], 1)]
        )
    else:
        segs = np.stack([child, parent], 1)
        ax.set_aspect("equal")
    ax.add_collection(LineCollection(list(segs), colors="grey", linewidths=0.2))

    leaves = nodes.filter(pl.col("is_leaf"))
    sizes: float | np.ndarray = point_size
    if size_by_cells:
        n = leaves["n_cells"].to_numpy()
        sizes = point_size * (1 + 4 * (n - n.min()) / max(n.max() - n.min(), 1))
    if colour_by is None:
        ax.scatter(leaves["x"], leaves["y"], s=sizes, linewidths=0)
    else:
        colour_by = np.asarray(colour_by)
        groups = np.unique(colour_by)
        palette = bv.pl.bx_colors("main", n=len(groups))
        for i, grp in enumerate(groups):
            m = colour_by == grp
            s = sizes[m] if isinstance(sizes, np.ndarray) else sizes
            ax.scatter(
                leaves["x"].to_numpy()[m],
                leaves["y"].to_numpy()[m],
                s=s,
                color=palette[i],
                label=grp,
                linewidths=0,
            )
        legend = ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
        for handle in legend.legend_handles:
            handle.set_sizes([20])  # ty: ignore[missing-union-attribute] # matplotlib stubs type legend handles as Artist
    ax.set_axis_off()
    return ax


cell_type = sc.obs(cols=["cell_type"], filtered=True)["cell_type"].to_numpy()

plot_tree(tree, colour_by=cell_type)
plt.show()

# %% [markdown]
# Every cell type gets its own subtree, and none of that was told to Bonsai: the
# annotation only colours the leaves. The CD16+ monocytes form their own
# clump next to the CD14+ monocytes, and the NK cells branch off close to
# the T cells, which is what you would expect from their biology. Branch length
# is change, so the long isolated branches are cells that look like nothing
# else in the data.
#
# The default is the equal-angle layout: every subtree gets a wedge
# proportional to its size. Two alternatives, recomputed from the stored tree
# with `relayout_bonsai()` without searching again. The dendrogram draws the
# branch lengths along x:

# %%
plot_tree(
    bv.trees.relayout_bonsai(tree, layout="dendrogram"),
    colour_by=cell_type,
    point_size=1.0,
)
plt.show()

# %% [markdown]
# The hyperbolic projection maps the same layout onto the unit disk. How much
# it changes the picture depends on how far the tips sit from the root; on
# PBMC3k the branches are short and it barely does, on deeper trees it gives
# the crowded centre more room:

# %%
plot_tree(bv.trees.relayout_bonsai(tree, hyperbolic=True), colour_by=cell_type)
plt.show()

# %% [markdown]
# `relayout_bonsai()` returns a new tree and leaves the original alone, so keep
# the result if you want that layout for the next step.
#
# ## Bonsai next to UMAP
#
# The leaf coordinates can go into the object as an embedding, so everything
# that plots embeddings works on them.

# %%
bv.trees.set_bonsai_embedding(sc, tree)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
bv.pl.embedding_plot(sc, "umap", "cell_type", discrete=True, ax=axes[0])
axes[0].set_title("UMAP")
bv.pl.embedding_plot(sc, "bonsai", "cell_type", discrete=True, ax=axes[1])
axes[1].set_title("Bonsai, equal angle")
plt.show()

# %% [markdown]
# ## Bonsai on metacells
#
# At 100k cells a tree over every cell is slow and hard to look at. Metacells
# compress first. Their raw counts are sums over their cells, and a sum of
# Poisson counts is Poisson, so Sanity treats a metacell exactly as it treats a
# cell, with its total counts as the library size. A metacell of 20 cells has
# about 20 times the counts, so its error bars are much tighter, and Bonsai
# weights by them. The same `bonsai()` call works on `MetaCells`; every
# metacell becomes a leaf.
#
# Here SuperCells compresses the PBMCs into metacells, and each metacell gets
# the cell type most of its cells carry.

# %%
supercells = bv.metacells.generate_supercells(sc)
assert supercells is not None

bv.metacells.calc_meta_cell_purity(
    supercells, cell_type.tolist(), add_additional_info="top_label"
)

tree_mc = bv.trees.bonsai(supercells, verbose=True)

tree_mc

# %% [markdown]
# The same PBMCs as 120 metacells, in a fraction of the time the cell-level
# tree took. And the summed counts are deep enough that thousands of
# genes clear the signal-to-noise bar, against a few hundred for single cells,
# so the tree sees far more of the transcriptome.
#
# `size_by_cells=True` scales every leaf by the cells behind it.

# %%
plot_tree(
    tree_mc,
    colour_by=supercells.obs(cols=["mc_top_label"])["mc_top_label"].to_numpy(),
    size_by_cells=True,
    point_size=10.0,
)
plt.show()

# %% [markdown]
# The subtrees of the cell-level tree survive the compression, and the
# monocytes come out sharper: CD14+ and CD16+ split into two branches of one
# monocyte subtree. A few T-cell metacells sit on the NK branch; T and NK cells
# share much of their cytotoxic programme, so that is where the tree is least
# sure of the labels. What the metacell tree cannot show is anything inside a
# metacell, so pick the compression to match the resolution you care about.
#
# ## Scaling
#
# The search grows a little faster than linearly with the cell number. In
# `bonsai-rs`'s own benchmarks on ten cores it took about 70 seconds at 10,000
# cells and 210 seconds at 25,000. Sanity comes on top and grows with the
# number of candidate genes; CPU has been made fast, GPU is even faster.
# `bixverse_gpu.trees.bonsai()` takes the same arguments, returns the same
# `BonsaiTree`, and runs Sanity on the device.
#
# On the 2,399 PBMCs of this vignette with every gene as a candidate (M1 Max,
# ten cores, mean of two runs):
#
# | | Sanity | Bonsai search | total |
# |---|---|---|---|
# | CPU | 9.6 s | 14.6 s | 24.2 s |
# | GPU | 1.0 s | 14.6 s | 15.6 s |
#
# Both kept the same 348 genes. With Sanity on the GPU the tree search is the
# clear cost, and that is where 100k cells will hurt. Metacells are the way
# around it: compress first and the tree has a few thousand leaves instead of
# 100k.
#
# ## Clean up

# %%
tempdir_pbmc.cleanup()
