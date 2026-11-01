# %% [markdown]
# # KNN searches for single cell
#
# *Python port of the R vignette [KNN searches for single cell](https://gregorlueg.github.io/bixverse/articles/knn_searches.html). Original author: Gregor Lueg.*
#
# ## Intro
#
# kNN searches are the bedrock of a lot of methods in the single cell space. This
# vignette is basically a primer to understand the large number of (approximate)
# nearest neighbour search methods available in `bixverse`, when to use what,
# when to be careful with them and hopefully gives you a general idea on
# what they can (and cannot do) well.

# %%
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

import bixverse as bv
from bixverse.params import KnnParams, NeighboursParams

# %% [markdown]
# ## Loading the data and basic workflow
#
# We start by downloading the PBMC3k data set and loading it via Cell
# Ranger-style MTX I/O, run a quick and dirty QC, HVG and PCA to then move onto
# the kNN searches...

# %%
pbmc3k_path = bv.io.download_pbmc3k()

tempdir_pbmc = tempfile.TemporaryDirectory()

sc = bv.io.load_mtx(
    Path(tempdir_pbmc.name) / "pbmc3k",
    bv.io.get_cell_ranger_params(pbmc3k_path),
    verbose=True,
)

# ensembl to gene symbols
sc.rename_var_col("column1", "gene_symbol")

var = sc.var()

ensembl_to_symbol = dict(zip(var["gene_id"], var["gene_symbol"], strict=True))
symbol_to_ensembl = dict(zip(var["gene_symbol"], var["gene_id"], strict=True))

gs_of_interest = {
    "MT": var.filter(pl.col("gene_symbol").str.contains("^MT-"))["gene_id"].to_list(),
    "Ribo": var.filter(pl.col("gene_symbol").str.contains("^RP[SL]"))[
        "gene_id"
    ].to_list(),
}

# remove high mitochondrial genes and generally low quality cells
bv.qc.gene_set_proportions(sc, gs_of_interest, streaming=False, verbose=True)

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

sc.add_obs(pl.DataFrame({"cell_idx": qc.cell_idx, "outlier": qc.combined}))

sc.set_cells_to_keep(qc.cell_idx[~qc.combined])

# HVG
bv.pp.find_hvg(sc, hvg_no=2000, verbose=True)

# PCA
bv.pp.calculate_pca(sc, no_pcs=30, sparse_svd=True)

# %% [markdown]
# ## kNN searches
#
# Okay, with all of this out of the way, let's look at the different methods
# available in bixverse to generate kNN graphs for single cell. You will realise
# that there is a LARGE number of methods that take kNN parameters, either via
# `KnnParams` or via the kNN fields of a bigger params class such as
# `NeighboursParams`. There is a lot of different parameters in there. Let's
# first check out what we have there...

# %%
KnnParams()

# %% [markdown]
# ### The methods
#
# There is a lot of different parameters in here and it is worthwhile actually
# understanding them and where they are being called used. Let's first give an
# overview of the different kNN searches implemented/exposed in this package
# (all happens via [ann-search-rs](https://crates.io/crates/ann-search-rs)).
#
# | Method | Features | Class | Use case | Ideal k range |
# |--------|----------|-------|----------|---------------|
# | Exhaustive | What the name says | Exact | Works well on smaller data sets where the CPU can just crunch through the numbers | Does not matter |
# | KmKnn | An exact method that takes a shortcut via k-means cluster | (Fast) exact | On well structured (low dimensional) data, it gives you an exact kNN search that is substantially faster than Exhaustive on large data sets | Does not matter |
# | Annoy | The classic approximate nearest neighbour search you know from Seurat | Tree-based | A memory-based version of Annoy that avoids the round trip to the disk-based index of the original. Good default for data sets with up to half a million cells with fast index building (but slower querying) | Ideally ≤ 100 as otherwise the query phase can become quite long... |
# | HNSW | A powerful graph-based index for large data sets and vector search index powering a large number of data bases. | Graph-based | Need to tackle large data sets ... ? HNSW does that for you. An important note is that due to the benign race condition during graph construction this index is not deterministic. On large data sets this makes basically no difference, but something to keep in mind for smaller ones! | ≤50 or you need to bump the `ef_search` substantially |
# | NNDescent | Another graph-based index that is ideal for tackling large data sets | Graph-based | Another graph-based index that is very good at dealing with large data sets. Uses tricks from PyNNDescent and is designed to be very memory-efficient. there is no query phase: it hands you the descent graph | ≤30. The initial kNN is designed with a node degree of 30, so you might see degradation in the performance when you ask for higher k. If you try this, set `ef_budget` high! |
# | IVF | A cluster-based index that partitions the data via k-means into Voronoi cells | Cluster-based | If you read a bit about vector searches, this is a very common algorithm which works well with quantisations in particular (and GPU-acceleration...). For the usual ephemeral kNN graph generation in single cell maybe not the right call, but if you need A LOT of neighbours on large data sets, this is a very good choice! | Does not matter (outside of absurd high numbers) |
#
# There are additional GPU-accelerated kNN searches in the `bixverse_gpu`
# package.
#
# ### The parameters
#
# Now let's see which parameter belongs to which method and does what.
#
# | Parameter | Default | Meaning |
# |-----------|---------|---------|
# | k | `15` | Kinda obvious... Number of neighbours |
# | knn_method | `"kmknn"` | The approximate nearest neighbour method. One of `"kmknn"`, `"exhaustive"`, `"annoy"`, `"ivf"`, `"hnsw"`, `"nndescent"` |
# | ann_dist | `"euclidean"` | The distance metric to use. Two options are available... Euclidean distance or Cosine distance. These are true distances: the underlying library works in squared Euclidean space for speed, but the square root is taken at the boundary, so what you get back is the Euclidean distance. |
# | n_trees | `50` | **Annoy:** The number of trees to use for Annoy. Generally speaking the generation is very fast in terms of indexing and in different (empirical) experiments after 50 trees one gets diminishing returns |
# | search_budget | `None` | **Annoy:** The search budget during querying. If `None` defaults to `k * n_trees * 20`. Higher values allow for better Recall. |
# | delta | `0.001` | **NNDescent:** The termination criterion for the NNDescent iterations. When less than 0.001 (0.1%) of the edges are updated, the iterations stop and the algorithm is considered converged. |
# | diversify_prob | `0.0` | **NNDescent:** If you want to do additional diversification after the NNDescent algorithm converged. This is based on the original paper, but in reality usually degrades graph quality while not providing too many benefits. |
# | ef_budget | `None` | **NNDescent:** The beam width for the query phase for NNDescent. If `None`, it will use the library default, i.e., `(k * 2).clamp(50, 200).max(k)`. Generally not really needed to touch, but limits the number of k neighbours you can confidently return. |
# | m | `16` | **HNSW:** The node degree during the index creation. Creates 16 edges per given node in the graph-based index. |
# | ef_construction | `200` | **HNSW:** The budget to generate good connections during construction of the index. Higher values yields a better initial graph-index. |
# | ef_search | `100` | **HNSW:** The beam width during querying of the index. If a high quality graph was generated, usually does not need to be too high. If you set `k` high, you should adapt this parameter. |
# | n_list | `None` | **IVF and KmKnn:** This parameter is shared between the IVF and KmKnn index. Defines the number of Voronoi cells/clusters to generate. If not provided defaults to `sqrt(N)`. Be careful on smaller data sets and possibly set it to more clusters here. |
# | n_probe | `None` | **IVF:** The number of clusters to probe per query. Defaults to `sqrt(n_list)`. On smaller data sets you need to set this higher. |
#
# ### Exploration of the kNN search methods
#
# Okay, still do not get it. When do I use what and how ... ? The standard
# approach would be to do something like:

# %%
bv.graph.find_neighbours(
    sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
)

# %% [markdown]
# You can however also generate the kNN indices like this:

# %%
exhaustive_knn = bv.graph.generate_knn(
    sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
)

print(
    f"KnnGraph: {exhaustive_knn.indices.shape[0]} cells, "
    f"k = {exhaustive_knn.indices.shape[1]}, {exhaustive_knn.dist_metric} distance"
)

# %% [markdown]
# This is what is actually stored in the object after running
# `find_neighbours()`, under `sc.cache.knn`.

# %%
exhaustive_knn_obj = sc.cache.knn

assert np.array_equal(exhaustive_knn.indices, exhaustive_knn_obj.indices)
assert np.array_equal(exhaustive_knn.distances, exhaustive_knn_obj.distances)

# %% [markdown]
# Potentially more interesting is what you can do with the approximate nearest
# neighbour searches.

# %%
annoy_knn = bv.graph.generate_knn(
    sc, neighbours_params=NeighboursParams(knn_method="annoy"), verbose=True
)

# %% [markdown]
# The approximate versions have the option to test the index against a subset of
# the data (1000 randomly selected cells) and compare against the exhaustive
# search for these 1000 cells (`validate_index=True`, the default, reported with
# `verbose=True`). `calc_knn_metrics` does the same over all cells against a
# reference kNN:

# %%
annoy_vs_exhaustive = bv.graph.calc_knn_metrics(exhaustive_knn, annoy_knn)

print(
    "Annoy vs exhaustive.\n"
    f"  Recall@15: {annoy_vs_exhaustive.final_recall:.3f}.\n"
    f"  Distance ratio: {annoy_vs_exhaustive.final_ratio:.3f}."
)

# %% [markdown]
# This allows us to actually explore some parameters more in depth:

# %%
annoy_knn_less_trees = bv.graph.generate_knn(
    sc,
    neighbours_params=NeighboursParams(knn_method="annoy", n_trees=1),
    verbose=True,
)

# %% [markdown]
# With one tree, we can appreciate that the Recall is dropping. Let's get some
# more detailed info here:

# %%
annoy_lt_vs_exhaustive = bv.graph.calc_knn_metrics(exhaustive_knn, annoy_knn_less_trees)

print(
    "Annoy (one tree) vs exhaustive.\n"
    f"  Recall@15: {annoy_lt_vs_exhaustive.final_recall:.3f}.\n"
    f"  Distance ratio: {annoy_lt_vs_exhaustive.final_ratio:.3f}."
)

# %% [markdown]
# We can appreciate that with one tree we do not get a Recall of 1 and the
# distance ratio is worse than the exhaustive one. Let's test another index

# %%
nndescent_knn = bv.graph.generate_knn(
    sc, neighbours_params=NeighboursParams(knn_method="nndescent"), verbose=True
)

# %% [markdown]
# And benchmark it:

# %%
nndescent_vs_exhaustive = bv.graph.calc_knn_metrics(exhaustive_knn, nndescent_knn)

print(
    "NNDescent vs exhaustive.\n"
    f"  Recall@15: {nndescent_vs_exhaustive.final_recall:.3f}.\n"
    f"  Distance ratio: {nndescent_vs_exhaustive.final_ratio:.3f}."
)

# %% [markdown]
# As we can see, this approximate index also reaches perfect recall and distance
# ratio.
#
# NNDescent has no query phase: the descent already builds a kNN graph on its
# way to convergence, and `"nndescent"` hands you that graph instead of running
# a beam search over an index. You trade a bit of recall for speed, which is the
# difference worth having on million-cell data.
#
# ## Conclusions
#
# We could iterate through the others, but it does not have much of a point. The
# goal of this vignette was to check the type of hyperparameters and type of
# indices provided and give an idea when to use which and how to work with them
# and validate parameters if needed.
#
# ## Clean up

# %%
tempdir_pbmc.cleanup()
