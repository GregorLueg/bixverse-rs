# %% [markdown]
# # Differential abundance with Milo and MELD
#
# *Gregor Lueg*
#
# ## Intro
#
# Differential expression asks whether cells changed what they express.
# Differential abundance asks whether the cells that are there changed at all.
# Both matter, and the second is the one people usually answer badly.
#
# The obvious approach is to cluster, count cells per cluster per sample, and
# test the proportions. It works, and it inherits every problem the clustering
# had. If a condition shifts a subpopulation that your resolution parameter
# merged into a bigger cluster, you see nothing. If it shifts half a cluster, the
# proportion barely moves. The answer depends on a choice you made before you
# asked the question.
#
# Two methods here avoid that, in different ways.
#
# [Milo](https://doi.org/10.1038/s41587-021-01033-z) tests overlapping
# neighbourhoods of the kNN graph instead of clusters. Each neighbourhood is a
# cell and its neighbours, so the resolution is the graph's rather than the
# clustering's, and a shift that spans part of a cluster shows up. Counts per
# neighbourhood per sample go through the same negative binomial machinery bulk
# RNA-seq uses.
#
# [MELD](https://doi.org/10.1038/s41587-020-00803-5) does not test anything. It
# smooths the condition labels over the graph and returns, per cell, how likely
# that cell is under each condition. It is a continuous score rather than a set
# of calls, which suits a graded response better than a discrete one.
#
# They answer the same question and they should agree. At the end of this
# vignette they do, which is a better check on both than either provides alone.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

import bixverse as bv
import bixverse.embed
from bixverse.params import MinQualityParams, NeighboursParams

# %% [markdown]
# ## The data
#
# [Baran-Gale, et al.](https://doi.org/10.1242/dev.183996) profiled mouse thymic
# epithelial cells at one, four and sixteen weeks. The thymus involutes with age,
# so the cell type proportions genuinely move, which is what makes this a
# sensible differential abundance example.
#
# > **A stimulation experiment is a bad Milo example**
# >
# > It is tempting to reuse a treatment dataset here. Do not. In vitro
# > stimulation moves cells a long way in embedding space without moving the
# > cell type proportions much, so stimulated and control cells separate into
# > different neighbourhoods and almost every neighbourhood comes out
# > significant. The result looks spectacular and says nothing beyond "the
# > treatment did something", which you knew.
# >
# > Differential abundance needs a design where the *composition* plausibly
# > changed.
#
# The download is an h5ad with the raw counts in `X`, the cell annotations in
# obs and the Ensembl identifiers as var index. 336 genes have no annotation
# whatsoever and carry a generated `unannotated_<row>` identifier.

# %%
# the per cell type tables further down have 23 rows, show them in full
pl.Config.set_tbl_rows(25)

thymus_path = bv.io.download_thymus_ageing()

tempdir_thymus = tempfile.TemporaryDirectory()

sc = bv.io.load_h5ad(
    Path(tempdir_thymus.name) / "thymus",
    thymus_path,
    qc=MinQualityParams(
        min_unique_genes=200,
        min_lib_size=500,
        min_cells=20,
        target_size=1e4,
    ),
    verbose=True,
)

sc

# %% [markdown]
# Now the design, and there is a trap in it. `samp_id` looks like the sample. It
# is not: it is the sequencing run, and every run carries all three ages, because
# the ages were hashed together and demultiplexed.

# %%
obs = sc.obs(filtered=True)

obs.group_by("samp_id", "age").len().sort("age").pivot(
    on="age", index="samp_id", values="len"
).sort("samp_id")

# %% [markdown]
# The sample is the run and the age together, which gives eighteen. Using
# `samp_id` as the sample would ask Milo whether abundance differs between
# sequencing runs, and every run contains every age, so the answer would be a
# flat no.

# %%
sc.add_obs(
    obs.select(
        "cell_idx",
        pl.concat_str(["samp_id", "age"], separator="_").alias("sample"),
    )
)
obs = sc.obs(filtered=True)

obs.group_by("sample", "age").len().sort("age").pivot(
    on="age", index="sample", values="len"
).fill_null(0).sort("sample").head(6)

# %% [markdown]
# ## The embedding
#
# Milo and MELD both run on the kNN graph, so the usual pipeline comes first.

# %%
bv.pp.find_hvg(sc, hvg_no=2000)

bv.pp.calculate_pca(sc, no_pcs=30)

# %% [markdown]
# The default `k` of 15 is fine for clustering and too small for Milo. A
# neighbourhood is the index cell, its `k` neighbours and every cell that lists it
# as a neighbour, so it holds at least `k + 1` cells, and those get spread over
# the samples. With eighteen samples and `k = 15` the smallest neighbourhoods have
# under one cell per sample, and a negative binomial fitted to counts that are
# mostly zero or one has very little to work with.
#
# Rough rule: pick `k` so the average neighbourhood holds at least a few cells
# per sample. Eighteen samples and `k = 60` gives around eight here, the
# neighbourhoods averaging just under 150 cells.

# %%
bv.graph.find_neighbours(sc, neighbours_params=NeighboursParams(k=60), verbose=True)

# %% [markdown]
# ## Milo
#
# Sampling the neighbourhoods and counting the cells per sample is one call. The
# counts come back as neighbourhoods by samples, with the graph-overlap weighting
# and the k-th neighbour distances taken at the same time, since all three are
# functions of the neighbourhood matrix alone.

# %%
milo = bv.de.get_milor_abundances(
    sc,
    "sample",
    milor_params=bv.de.MiloRParams(prop=0.1),
    verbose=True,
)

print(milo.sample_counts.shape)
print(milo.sample_counts.sum(axis=1).mean())

# %% [markdown]
# The test needs a design table with one row per sample and a column holding the
# sample names that match the columns of the counts. Blocking on the run is
# worth it: every run contributes all three ages, so run-to-run variation is a
# nuisance term the model can remove.

# %%
design_df = pl.DataFrame({"sample": milo.samples}).join(
    obs.select("sample", pl.col("samp_id").alias("run"), "age").unique(),
    on="sample",
    how="left",
    maintain_order="left",
)

design_df.group_by("age", "run").len().sort("run").pivot(
    on="run", index="age", values="len"
).sort("age")

# %% [markdown]
# > **Check which coefficient you are testing**
# >
# > `test_nhoods()` defaults to the last column of the design, the way edgeR and
# > limma do. Which level ends up last depends on the level order of `age`. An
# > Enum keeps its own order, and `age` comes out of the h5ad as one ordered
# > `Wk1`, `Wk4`, `Wk16`, so here the default happens to be right. Cast it to a
# > plain string and the levels get sorted: `Wk1`, `Wk16`, `Wk4`. The last
# > coefficient is then **Wk4 against Wk1**, not the sixteen week contrast you
# > probably wanted.
# >
# > Name the coefficient rather than trusting the default.

# %%
milo = bv.de.test_nhoods(
    milo,
    "~ run + age",
    design_df,
    sample_col="sample",
    coef="ageWk16",
)

da_res = bv.de.get_differential_abundance_res(milo)

da_res.filter(pl.col("SpatialFDR") <= 0.1).height

# %%
da_res.sort("PValue").select("Nhood", "logFC", "F", "PValue", "FDR", "SpatialFDR").head(
    6
)

# %% [markdown]
# A neighbourhood is not a cell type, so on its own a list of neighbourhood
# indices is not much use. `add_nhoods_info()` tags each with the cell type its
# cells mostly belong to.

# %%
milo = bv.de.add_nhoods_info(milo, cell_info=obs["cluster_annot"])
da_res = bv.de.get_differential_abundance_res(milo)

sig = da_res.filter(pl.col("SpatialFDR") <= 0.1).with_columns(
    direction=pl.when(pl.col("logFC") > 0)
    .then(pl.lit("up at Wk16"))
    .otherwise(pl.lit("down at Wk16"))
)

(
    sig.group_by("majority_celltype", "direction")
    .len()
    .pivot(on="direction", index="majority_celltype", values="len")
    .fill_null(0)
    .sort("up at Wk16", descending=True)
)

# %% [markdown]
# The two proliferating populations are the most strongly depleted of anything
# here, which is the clearest ageing signal the tissue has: an involuting thymus
# stops making new cells. The intertypical TECs dominate the enriched side.
#
# The medullary subsets split both ways, so "mTECs go up" would be the wrong
# summary to take away. That is the point of testing neighbourhoods rather than
# clusters: a cluster level test would have averaged those subsets into one
# number and reported whichever direction happened to win.
#
# ### On the embedding
#
# The figure everyone expects from Milo is the neighbourhood graph on a UMAP.
# Each neighbourhood sits at its index cell, sized by how many cells it holds and
# connected to the neighbourhoods it shares cells with. Significant ones are
# coloured by logFC, the rest stay white. The UMAP is only for the picture, the
# test above never touched it.

# %%
bv.embed.umap(sc)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
bv.pl.milo_nhood_plot(sc, milo, embedding="umap", alpha=0.1, ax=ax)
plt.show()

# %% [markdown]
# `colour_by="majority_celltype"` swaps the logFC for the annotation from
# `add_nhoods_info()`, which helps to read the logFC plot next to it.

# %%
fig, ax = plt.subplots(figsize=(8, 6))
bv.pl.milo_nhood_plot(sc, milo, embedding="umap", colour_by="majority_celltype", ax=ax)
plt.show()

# %% [markdown]
# The nodes and edges behind the plot come from `extract_milo_plot_data`, if
# you want to draw them yourself.

# %%
milo_plot_data = bv.pl.extract_milo_plot_data(sc, milo, embedding="umap")

milo_plot_data.nodes.select("Nhood", "dim_1", "dim_2", "size", "is_sig").head()

# %% [markdown]
# ### On the spatial FDR
#
# Neighbourhoods overlap, so their tests are not independent and a plain
# Benjamini-Hochberg adjustment is anti-conservative. Milo's spatial FDR weights
# each p-value by the reciprocal of its connectivity and runs the step-up on
# those weights.

# %%
pl.DataFrame(
    {
        "plain_fdr": [da_res.filter(pl.col("FDR") <= 0.1).height],
        "spatial_fdr": [da_res.filter(pl.col("SpatialFDR") <= 0.1).height],
    }
)

# %% [markdown]
# On this data the two barely differ. That is worth knowing rather than
# disappointing: the correction bites when neighbourhoods overlap heavily and the
# connectivity varies a lot between them, and here it does not. Do not assume it
# is always doing work.
#
# The weighting scheme is a choice. `"k-distance"` weights by the distance to the
# k-th neighbour, `"graph-overlap"` by how many cells a neighbourhood shares with
# the others. Both are precomputed, so switching is cheap.
#
# ## MELD
#
# MELD needs no design and no neighbourhood sampling. Hand it the condition
# column and it smooths the indicator over the graph.

# %%
meld_res = bv.de.meld(sc, "age")

print(meld_res.norm_scores.shape)
meld_res.norm_scores.head(4).with_columns(pl.selectors.numeric().round(3))

# %% [markdown]
# The normalised scores are clamped at zero and L1 normalised per cell, so each
# row is a likelihood over the three ages. A cell sitting at 0.6 for Wk16 lives
# in a part of the manifold that is enriched for sixteen week old thymus.
#
# Averaging per cell type turns that into something readable.

# %%
per_celltype = (
    meld_res.norm_scores.drop("cell_id")
    .with_columns(cluster=obs["cluster_annot"].cast(pl.String))
    .group_by("cluster")
    .mean()
)

per_celltype.sort("Wk16", descending=True).head(8)

# %% [markdown]
# ## Do they agree
#
# They are different methods on the same graph, so this is a real check rather
# than a formality. Milo's mean log fold change per cell type against MELD's mean
# Wk16 likelihood per cell type.

# %%
milo_per_ct = sig.group_by(pl.col("majority_celltype").alias("cluster")).agg(
    milo_logfc=pl.col("logFC").mean()
)
comparison = per_celltype.join(milo_per_ct, on="cluster")

comparison.sort("milo_logfc", descending=True).select(
    "cluster", "Wk1", "Wk16", "milo_logfc"
)

# %%
comparison.select(pl.corr("Wk16", "milo_logfc", method="spearman")).item()

# %% [markdown]
# Strong agreement, and the disagreements are informative rather than
# embarrassing. Milo tests neighbourhoods and reports only the ones that clear a
# threshold, so a cell type that shifted a little everywhere can score low on the
# Milo axis while MELD still registers the shift. The two are measuring related
# but not identical things.
#
# ## Which one to reach for
#
# Milo when you want calls with error control: a list of regions that changed,
# with p-values you can defend and a multiple testing correction that accounts
# for the overlap. It needs a real design, enough samples to fit a negative
# binomial, and a `k` large enough that the counts are not mostly zero.
#
# MELD when the response is graded, when you want a per-cell score to carry into
# downstream analysis, or when the design is too thin for a test. It gives you no
# p-values, which is a feature when you have four samples and would only have
# been fooling yourself.
#
# Running both costs almost nothing once the graph is built, and their agreement
# is the most useful diagnostic either one has.
#
# ## Where next
#
# The other half of this is differential *expression* with the same donor
# structure, which has its own [vignette](differential_expression.ipynb).
# Neighbourhood level expression testing, rather than abundance, is not wired up
# yet.
#
# ## Clean up

# %%
tempdir_thymus.cleanup()
