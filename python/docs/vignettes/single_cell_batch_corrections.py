# %% [markdown]
# # Batch correction with bixverse
#
# *Python port of the R vignette [Batch correction with bixverse](https://gregorlueg.github.io/bixverse/articles/single_cell_batch_corrections.html). Original authors: Grant Neilson, Gregor Lueg and Liesbeth François.*
#
# ## Intro
#
# Batch effects are systematic technical differences between experimental runs
# that can obscure genuine biological signal in single cell data. When datasets
# from multiple experiments, technologies, or laboratories are combined, cells
# tend to group by their source rather than by cell type. Batch correction
# methods aim to remove these technical artefacts whilst preserving biological
# variation.
#
# `bixverse` provides four batch correction approaches, each with different
# trade-offs:
#
# - **fastMNN** ([Haghverdi, et al., 2018](https://doi.org/10.1038/nbt.4091))
#   identifies mutual nearest neighbours across batches: pairs of cells from
#   different batches that are each other's closest match. Correction vectors
#   are computed from these pairs and applied to align the embeddings. It is
#   particularly effective when cell type composition differs across batches.
# - **Harmony** ([Korsunsky, et al., 2019](https://doi.org/10.1038/s41592-019-0619-0))
#   operates on PCA embeddings using iterative soft clustering with diversity
#   penalties. It assigns cells to clusters, estimates batch effects per cluster
#   via ridge regression, and corrects the embedding. It tends to be fast and
#   works well across a range of scenarios. A version 2 has been released too,
#   see [Patikas, et al., 2026](https://www.biorxiv.org/content/10.64898/2026.03.16.711825v1),
#   with improvements that increase scalability to larger data sets and
#   modifications to the objective functions to not overcorrect across batches
#   when different cell types are there.
# - **BBKNN** ([Polanski, et al., 2020](https://doi.org/10.1093/bioinformatics/btz625))
#   takes a fundamentally different approach. Rather than correcting an
#   embedding, it constructs a batch-balanced k-nearest neighbour graph by
#   building separate neighbour indices per batch. The corrected graph can then
#   be used directly for clustering and visualisation.
# - **Seurat anchors** ([Stuart, et al., 2019](https://doi.org/10.1016/j.cell.2019.05.031))
#   find pairs of cells that are mutual nearest neighbours in a space shared by
#   two batches, score them by how much of their neighbourhood they share, and
#   use them to pull one batch onto the other. Two flavours are available. CCA
#   builds the shared space from canonical correlations and corrects hardest,
#   which suits batches that don't share their cell types. rPCA projects each
#   batch into the other's PCA basis instead: cheaper, gentler, and the better
#   default when the batches broadly match. Both need batch-aware HVGs.
#
# A correction has two jobs: mix the batches and keep the biology intact. Going
# too hard on the first ruins the second, so `bixverse` scores both, following
# the scIB benchmark ([Luecken, et al., 2022](https://doi.org/10.1038/s41592-021-01336-8)).
#
# Batch mixing:
#
# - **kBET**: tests whether neighbourhood batch proportions match the global
#   ones.
# - **Batch ASW**: silhouette width on batch labels in the embedding.
# - **iLISI**: effective number of batches per neighbourhood. Works on any kNN
#   graph.
# - **PCR**: how much of the embedding variance batch explains, before versus
#   after correction.
#
# Bio conservation, which needs cell type labels:
#
# - **cLISI**: effective number of cell types per neighbourhood. You want one.
# - **Cell type ASW**: silhouette width on cell type labels.
# - **Graph connectivity**: whether each cell type stays one connected piece of
#   the kNN graph.
#
# `bv.integrate.calculate_integration_metrics()` runs the lot and returns one
# row, every column rescaled so that higher is better.

# %%
import tempfile
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import bixverse as bv
import bixverse.embed
import bixverse.params

# %% [markdown]
# ## Preparing the data
#
# ### Loading the batches
#
# We use two PBMC datasets (pbmc3k and pbmc4k) as a standard benchmark for
# batch correction. These come from the same tissue but differ in sequencing
# depth and cell counts, introducing a clear batch effect.

# %%
dir_data = bv.io.download_pbmc_batches()

tempdir_batch_cor = tempfile.TemporaryDirectory()

h5ad_paths = {
    "pbmc3k": dir_data / "pbmc3k.h5ad",
    "pbmc4k": dir_data / "pbmc4k.h5ad",
}

h5_tasks = bv.io.prescan_h5ad_files(h5ad_paths)

sc = bv.io.load_multi_h5ad(
    Path(tempdir_batch_cor.name) / "batch_cor",
    h5_tasks,
    verbose=True,
)

sc

# %% [markdown]
# ### Quality control
#
# Before batch correction, we filter out low-quality cells using mitochondrial
# gene proportions and library complexity metrics. In a multi file import the
# additional var columns are dropped, so the gene symbols come from one of the
# h5ad files.

# %%
h5_metadata = bv.io.read_h5ad_metadata(h5ad_paths["pbmc3k"])
h5_var = h5_metadata["var"]
assert isinstance(h5_var, pl.DataFrame)

var = sc.var().join(
    h5_var.select(
        pl.col("ensembl_id").alias("gene_id"),
        pl.col("symbol_tenx").alias("gene_symbol"),
    ),
    on="gene_id",
)

gs_of_interest = {
    "MT": var.filter(pl.col("gene_symbol").str.contains("^MT-"))["gene_id"].to_list(),
    "Ribo": var.filter(pl.col("gene_symbol").str.contains("^RP[SL]"))[
        "gene_id"
    ].to_list(),
}

bv.qc.gene_set_proportions(sc, gs_of_interest, streaming=False, verbose=True)

qc_df = sc.obs(cols=["cell_id", "lib_size", "nnz", "MT"])

metrics = {
    "log10_lib_size": np.log10(qc_df["lib_size"].to_numpy()),
    "log10_nnz": np.log10(qc_df["nnz"].to_numpy()),
    "MT": qc_df["MT"].to_numpy(),
}

directions: dict[str, Literal["twosided", "below", "above"]] = {
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

# set the cells to keep
sc.add_obs(pl.DataFrame({"cell_idx": qc.cell_idx, "outlier": qc.combined}))
sc.set_cells_to_keep(qc.cell_idx[~qc.combined])

sc

# %% [markdown]
# ### Pre-processing
#
# Standard pre-processing: HVG selection, PCA, and neighbour computation.

# %%
bv.pp.find_hvg(sc, hvg_no=2000, verbose=True)

bv.pp.calculate_pca(sc, no_pcs=30)

bv.graph.find_neighbours(sc)

# %% [markdown]
# ### Cell type labels
#
# The bio conservation metrics need cell types. We cluster the uncorrected data
# and annotate the clusters with scType
# ([Ianevski, et al., 2022](https://doi.org/10.1038/s41467-022-28803-w)) on a
# handful of canonical PBMC markers, same as in the
# [PBMC vignette](pbmc_single_cell.ipynb). The labels are deliberately coarse:
# T cells stay lumped, since splitting CD4 from CD8 on uncorrected data is
# asking for trouble.
#
# One caveat: the labels come from uncorrected data, so they carry whatever
# the batch effect did to the clusters. Good enough to spot a method that
# smears cell types together, not a ground truth.

# %%
bv.graph.find_clusters(sc, res=1.0, name="leiden_uncorrected")

cell_markers = {
    "CD3D": "T cells",
    "CD3E": "T cells",
    "IL7R": "T cells",
    "CD8A": "T cells",
    "MS4A1": "B cells",
    "CD79A": "B cells",
    "CD14": "Monocytes",
    "LYZ": "Monocytes",
    "FCGR3A": "Monocytes",
    "GNLY": "NK",
    "NKG7": "NK",
    "FCER1A": "DC",
    "CD1C": "DC",
    "PPBP": "Platelets",
}

cell_markers_df = pl.DataFrame(
    {"gene_symbol": list(cell_markers), "cell_type": list(cell_markers.values())}
).join(var.select("gene_symbol", "gene_id"), on="gene_symbol")

marker_list = bv.annotate.prepare_cell_markers(sc, cell_markers_df)

sctype_scores = bv.annotate.calc_sc_type_scores(sc, marker_list)

obs = sc.obs(cols=["cell_idx", "leiden_uncorrected"], filtered=True)

cell_type_anno = bv.annotate.score_clusters(
    sctype_scores, obs["leiden_uncorrected"].to_numpy()
)

obs = obs.join(
    cell_type_anno.select(
        pl.col("cluster_id").alias("leiden_uncorrected"), "cell_type"
    ),
    on="leiden_uncorrected",
    how="left",
)

sc.add_obs(obs.select("cell_idx", "cell_type"))

obs["cell_type"].value_counts(sort=True)

# %% [markdown]
# ## Comparing the different methods
#
# Let's check out the different methods and how they behave.
#
# ### Uncorrected data
#
# Before applying any correction, we can visualise the batch effect and compute
# baseline metrics. All t-SNEs here use the quick-and-dirty Barnes-Hut
# optimiser.

# %%
bv.embed.tsne(sc, params=bv.embed.tsne_defaults().set_params(approx="qd"))  # ty: ignore[invalid-argument-type] # manifolds-rs types set_params as BaseEmbedding

# %% [markdown]
# tSNE plot (tSNEs usually look prettier... It's anyway not a good way to
# assess batch effect correction, so at least let's make it pleasant on the
# eye).

# %%
ax = bv.pl.embedding_plot(sc, "tsne", "exp_id", label_by="exp_id", discrete=True)
ax.set_title("Before batch correction")
plt.show()

# %% [markdown]
# Each metric has its own function and its own result record. Here's kBET on
# its own:

# %%
kbet = bv.integrate.calculate_kbet(sc, "exp_id")
assert kbet is not None

print(
    f"kBET rejection rate: {kbet.kbet_score:.3f} over {kbet.n_batches} batches "
    f"(median chi-square {kbet.median_chi_square:.2f})"
)

# %% [markdown]
# In practice, you'd want all of them at once.
# `calculate_integration_metrics()` gives one row per call, and we'll stack one
# row per method into `metrics_df`:

# %%
metrics_df = bv.integrate.calculate_integration_metrics(
    sc, "exp_id", cell_type_column="cell_type"
).with_columns(method=pl.lit("Uncorrected"))

metrics_df

# %% [markdown]
# The high kBET rejection rate (so a low `kbet_accept`) and a low `ilisi`
# confirm a substantial batch effect, and the tSNE shows clear separation by
# batch. `pcr_comparison` is null here: it compares a corrected embedding
# against this PCA, so the PCA has nothing to compare against.
#
# ### fastMNN
#
# fastMNN works on batch-aware HVGs and produces a corrected embedding. It
# identifies mutual nearest neighbours across batches to compute correction
# vectors. For it to work best, we need the batch-aware HVGs. We calculate
# these, provide them to fastMNN (which regenerates the PCA embedding based on
# the batch-aware HVGs) and then run the actual algorithm.

# %%
batch_aware_hvg = bv.pp.find_hvg_batch_aware(sc, "exp_id")

bv.integrate.fast_mnn(sc, "exp_id", batch_aware_hvg.hvg_gene_idx)

bv.graph.find_neighbours(sc, embd_to_use="mnn")

# %%
metrics_df = pl.concat(
    [
        metrics_df,
        bv.integrate.calculate_integration_metrics(
            sc, "exp_id", cell_type_column="cell_type", embd_to_use="mnn"
        ).with_columns(method=pl.lit("fastMNN")),
    ]
)

metrics_df.filter(pl.col("method") == "fastMNN")

# %% [markdown]
# We can see clear improvements on the batch side: kBET acceptance, batch ASW
# and iLISI all go up.

# %%
bv.embed.tsne(
    sc,
    slot_name="tsne_mnn",
    use_knn=True,
    params=bv.embed.tsne_defaults().set_params(approx="qd"),  # ty: ignore[invalid-argument-type] # manifolds-rs types set_params as BaseEmbedding
)

# %%
ax = bv.pl.embedding_plot(sc, "tsne_mnn", "exp_id", label_by="exp_id", discrete=True)
ax.set_title("fastMNN batch correction")
plt.show()

# %% [markdown]
# Also, visually the batches get mixed now.
#
# ### Harmony
#
# Harmony operates on the PCA embedding directly. Leave `k` as `None` and the
# number of clusters is set to one per 30 cells, capped at 100. Both versions
# are supported and share one numerical core in Rust. They differ in the
# diversity penalty, the ridge regression and how the ridge penalty gets set.
# The intercept is never penalised in either. `bv.integrate.harmony()` runs v2;
# the original is `bv.integrate.harmony_v1()`. Let's start with the original:
#
# #### Version 1

# %%
bv.integrate.harmony_v1(sc, "exp_id")

bv.graph.find_neighbours(sc, embd_to_use="harmony_v1")

# %% [markdown]
# Let's calculate the batch correction-related scores.

# %%
metrics_df = pl.concat(
    [
        metrics_df,
        bv.integrate.calculate_integration_metrics(
            sc, "exp_id", cell_type_column="cell_type", embd_to_use="harmony_v1"
        ).with_columns(method=pl.lit("Harmony")),
    ]
)

metrics_df.filter(pl.col("method") == "Harmony")

# %% [markdown]
# Also here, we observe improvements across the board.

# %%
bv.embed.tsne(
    sc,
    slot_name="tsne_harm",
    use_knn=True,
    params=bv.embed.tsne_defaults().set_params(approx="qd"),  # ty: ignore[invalid-argument-type] # manifolds-rs types set_params as BaseEmbedding
)

# %%
ax = bv.pl.embedding_plot(sc, "tsne_harm", "exp_id", label_by="exp_id", discrete=True)
ax.set_title("Harmony batch correction")
plt.show()

# %% [markdown]
# #### Version 2
#
# [Patikas, et al., 2026](https://www.biorxiv.org/content/10.64898/2026.03.16.711825v1)
# changed four things compared to v1. The last three have knobs in
# `bv.params.HarmonyV2Params`:
#
# - Stabilised diversity penalty, which makes the objective scale-invariant.
# - Batch pruning in the ridge regression. A batch that puts less than
#   `batch_proportion_cutoff` of its cells into a cluster is dropped from that
#   cluster's regression. A cluster left with a single batch isn't corrected,
#   so a cell type unique to one batch doesn't get pulled towards the others.
# - Dynamic lambda. With `use_dynamic_lambda=True` (the default) the ridge
#   penalty is set per cluster as `alpha` times the expected counts and the
#   fixed `lambda_` is ignored.
# - Theta scaling by batch size via `tau`. Off by default (`tau=0`).
#
# The paper's arrowhead matrix inversion for a single batch covariate is used
# by both versions here, so v1 gets that speed-up too. v2 also runs only 4
# k-means iterations per Harmony round versus 20 in v1. Unless you need to
# reproduce an old v1 result, go with v2, which is why it is the default.

# %%
bv.integrate.harmony(sc, "exp_id")

bv.graph.find_neighbours(sc, embd_to_use="harmony")

# for the cluster comparison in the BBKNN section
bv.graph.find_clusters(sc, res=1.0, name="leiden_harmony")

# %% [markdown]
# Let's calculate the batch correction-related scores.

# %%
metrics_df = pl.concat(
    [
        metrics_df,
        bv.integrate.calculate_integration_metrics(
            sc, "exp_id", cell_type_column="cell_type", embd_to_use="harmony"
        ).with_columns(method=pl.lit("Harmony v2")),
    ]
)

metrics_df.filter(pl.col("method") == "Harmony v2")

# %%
bv.embed.tsne(
    sc,
    slot_name="tsne_harm_v2",
    use_knn=True,
    params=bv.embed.tsne_defaults().set_params(approx="qd"),  # ty: ignore[invalid-argument-type] # manifolds-rs types set_params as BaseEmbedding
)

# %%
ax = bv.pl.embedding_plot(
    sc, "tsne_harm_v2", "exp_id", label_by="exp_id", discrete=True
)
ax.set_title("Harmony (version 2) batch correction")
plt.show()

# %% [markdown]
# Got a GPU? `bixverse_gpu.integrate.harmony()` runs the same v2 algorithm
# and stores the result as `"harmony_gpu"`. It takes a single batch column
# only.
#
# ### Seurat CCA
#
# Seurat's anchor methods take a different route. For each pair of batches they
# find cells that are mutual nearest neighbours in a shared space, treat those
# pairs as anchors, score them by how much of their neighbourhood they share,
# and then apply a kernel-weighted correction that pulls the query batch onto
# the reference. Batches get merged in the order of their pairwise anchor
# counts.
#
# For CCA the shared space is a canonical correlation embedding computed per
# batch pair. Anchors also get filtered in gene space before scoring, which is
# CCA-only in Seurat.
#
# A few things differ from Seurat's reference implementation:
#
# - The per-gene `ScaleData` step is skipped. We work directly from per-cell
#   standardised log-normalised HVG expression.
# - `M = X1^T @ X2` is never materialised. The canonical correlations come from
#   a matrix-free randomised SVD, so memory stays at `O((n1 + n2) * num_cc)`
#   rather than `n1 x n2`. On big batch pairs that is the difference between
#   running and not running.
# - The correction is applied to the embedding (dims x cells), not to full
#   log-expression.
# - The effective CC rank is `max(num_cc, dims)`, not `num_cc`.
#
# Anchor structure comes out close to identical. Like fastMNN, CCA wants
# batch-aware HVGs, so we reuse the ones computed above.

# %%
bv.integrate.seurat_cca(sc, "exp_id", batch_aware_hvg.hvg_gene_idx)

bv.graph.find_neighbours(sc, embd_to_use="cca")

# %%
metrics_df = pl.concat(
    [
        metrics_df,
        bv.integrate.calculate_integration_metrics(
            sc, "exp_id", cell_type_column="cell_type", embd_to_use="cca"
        ).with_columns(method=pl.lit("Seurat CCA")),
    ]
)

metrics_df.filter(pl.col("method") == "Seurat CCA")

# %%
bv.embed.tsne(
    sc,
    slot_name="tsne_cca",
    use_knn=True,
    params=bv.embed.tsne_defaults().set_params(approx="qd"),  # ty: ignore[invalid-argument-type] # manifolds-rs types set_params as BaseEmbedding
)

# %%
ax = bv.pl.embedding_plot(sc, "tsne_cca", "exp_id", label_by="exp_id", discrete=True)
ax.set_title("Seurat CCA batch correction")
plt.show()

# %% [markdown]
# ### Seurat rPCA
#
# rPCA runs the same anchor pipeline on a cheaper shared space. Instead of
# computing canonical correlations, each batch keeps its own PCA basis and the
# other batch's expression is projected into it. Cross-batch neighbour queries
# then happen in these projected bases.
#
# The trade-off is the usual one: rPCA is faster and corrects less
# aggressively, which makes it the safer choice when your batches share most of
# their cell types. CCA is the one to reach for when they don't. As in Seurat,
# no gene-space anchor filter runs here, that step is CCA-only.

# %%
bv.integrate.seurat_rpca(sc, "exp_id", batch_aware_hvg.hvg_gene_idx)

bv.graph.find_neighbours(sc, embd_to_use="rpca")

# %%
metrics_df = pl.concat(
    [
        metrics_df,
        bv.integrate.calculate_integration_metrics(
            sc, "exp_id", cell_type_column="cell_type", embd_to_use="rpca"
        ).with_columns(method=pl.lit("Seurat rPCA")),
    ]
)

metrics_df.filter(pl.col("method") == "Seurat rPCA")

# %%
bv.embed.tsne(
    sc,
    slot_name="tsne_rpca",
    use_knn=True,
    params=bv.embed.tsne_defaults().set_params(approx="qd"),  # ty: ignore[invalid-argument-type] # manifolds-rs types set_params as BaseEmbedding
)

# %%
ax = bv.pl.embedding_plot(sc, "tsne_rpca", "exp_id", label_by="exp_id", discrete=True)
ax.set_title("Seurat rPCA batch correction")
plt.show()

# %% [markdown]
# One caveat worth knowing about both anchor methods: on data that already
# mixes well they have nothing to gain, and the correction can nudge the batch
# metrics slightly in the wrong direction while still sharpening the cell type
# structure. Check your uncorrected metrics before reaching for them.
#
# ### BBKNN
#
# BBKNN doesn't touch the embedding. It builds a batch-balanced neighbour graph
# instead: `neighbours_within_batch` nearest neighbours from every batch, turned
# into UMAP-style connectivities. Here that's 10 per batch, so 20 neighbours per
# cell feed the connectivities. `no_neighbours_to_keep` only cuts the stored
# kNN, and the cut stays balanced: each batch keeps its nearest
# `ceil(15 / 2) = 8`, so a cell ends up with 8 from one batch and 7 from the
# other.
#
# That makes BBKNN awkward to score. kBET and iLISI read the stored kNN, which
# is balanced by construction, so they come out near perfect whatever the data
# looks like. That's the method grading its own homework, not a sign of mixing.
# ASW and PCR need an embedding, and BBKNN doesn't have one, so we pass
# `embd_to_use=None` and those come back null. cLISI and graph connectivity are
# still fair, since they only look at cell types.

# %%
bv.integrate.bbknn(
    sc,
    "exp_id",
    no_neighbours_to_keep=15,
    bbknn_params=bv.params.BbknnParams(neighbours_within_batch=10),
)

# %%
metrics_df = pl.concat(
    [
        metrics_df,
        bv.integrate.calculate_integration_metrics(
            sc, "exp_id", cell_type_column="cell_type", embd_to_use=None
        ).with_columns(method=pl.lit("BBKNN")),
    ]
)

metrics_df.filter(pl.col("method") == "BBKNN")

# %% [markdown]
# So what does the graph actually look like? Counting edges is not enough: the
# connectivities are weighted, and everything downstream (clustering, t-SNE,
# UMAP) follows the weights. Below: the cross-batch share of the stored kNN, of
# the graph's edges and of its total weight, plus the median distance from each
# cell to its nearest neighbour in its own batch and in the other one.

# %%
batch = sc.obs(cols=["exp_id"], filtered=True)["exp_id"].to_numpy()

knn_idx = sc.cache.knn.indices
knn_dist = sc.cache.knn.distances
other_batch = batch[knn_idx] != batch[:, None]

conn = sc.cache.connectivities
edge_rows = np.repeat(np.arange(conn.shape[0]), np.diff(conn.indptr))
cross_edge = batch[edge_rows] != batch[conn.indices]

pl.DataFrame(
    {
        "cross_batch_share_knn": [other_batch.mean()],
        "cross_batch_share_edges": [cross_edge.mean()],
        "cross_batch_share_weight": [conn.data[cross_edge].sum() / conn.data.sum()],
        "median_nearest_same_batch": [
            np.median(np.where(other_batch, np.inf, knn_dist).min(axis=1))
        ],
        "median_nearest_other_batch": [
            np.median(np.where(other_batch, knn_dist, np.inf).min(axis=1))
        ],
    }
)

# %% [markdown]
# Half the edges cross batches, but they carry only a few percent of the weight.
# The nearest cell in the other batch sits about twice as far away as the
# nearest one in the cell's own batch, and the UMAP connectivities decay with
# distance, so the cross-batch edges end up with next to no weight. This isn't
# an implementation detail: the reference `bbknn` package behaves the same on
# this data. pbmc3k and pbmc4k were run on different 10x chemistries (GemCode
# v1 against v2), and that gap is real.
#
# The fair test of a graph method is clustering, so let's run Leiden on the
# BBKNN graph and compare its clusters against the uncorrected ones and
# Harmony v2. A cluster where one batch holds more than 90% of the cells counts
# as unmixed.

# %%
bv.graph.find_clusters(sc, res=1.0, name="leiden_bbknn")

# share of a cluster's cells from one batch above which it counts as unmixed
SINGLE_BATCH_CUTOFF = 0.9


def batch_composition(cluster_col: str) -> pl.DataFrame:
    obs = sc.obs(cols=[cluster_col, "exp_id"], filtered=True)
    per_cluster = (
        obs.group_by(cluster_col, "exp_id")
        .len()
        .with_columns(share=pl.col("len") / pl.col("len").sum().over(cluster_col))
        .group_by(cluster_col)
        .agg(n_cells=pl.col("len").sum(), top_batch_share=pl.col("share").max())
    )
    unmixed = per_cluster.filter(pl.col("top_batch_share") > SINGLE_BATCH_CUTOFF)
    return pl.DataFrame(
        {
            "clustering": [cluster_col],
            "n_clusters": [per_cluster.height],
            "n_unmixed_clusters": [unmixed.height],
            "share_cells_in_unmixed": [unmixed["n_cells"].sum() / obs.height],
        }
    )


pl.concat(
    [
        batch_composition(col)
        for col in ["leiden_uncorrected", "leiden_harmony", "leiden_bbknn"]
    ]
)

# %%
bv.embed.tsne(sc, use_knn=True, params=bv.embed.tsne_defaults().set_params(approx="qd"))  # ty: ignore[invalid-argument-type] # manifolds-rs types set_params as BaseEmbedding

# %%
ax = bv.pl.embedding_plot(sc, "tsne", "exp_id", label_by="exp_id", discrete=True)
ax.set_title("BBKNN batch correction")
plt.show()

# %% [markdown]
# The clusters and the t-SNE tell the same story. Leiden on the BBKNN graph
# still puts most cells into single-batch clusters, barely better than doing
# nothing, while Harmony mixes nearly all of them. The t-SNE follows the same
# weights and stays split by batch. On a batch effect this strong, BBKNN on its
# own doesn't correct much. It's built for milder effects, where the
# cross-batch neighbours sit close enough to keep real weight.
#
# ## Choosing a method
#
# Here's everything side by side:

# %%
batch_metrics = ["kbet_accept", "batch_asw", "ilisi", "pcr_comparison"]
bio_metrics = ["clisi", "cell_type_asw", "graph_connectivity"]

metrics_df.select("method", *batch_metrics, *bio_metrics)

# %%
fig, axes = plt.subplots(
    1,
    2,
    figsize=(9, 5),
    sharey=True,
    layout="constrained",
    gridspec_kw={"width_ratios": [len(batch_metrics), len(bio_metrics)]},
)

methods = metrics_df["method"].to_list()
cmap = plt.get_cmap("Blues").with_extremes(bad="0.9")

# BBKNN's kBET and iLISI read its own balanced kNN, so they are greyed out
plot_df = metrics_df.with_columns(
    pl.when(pl.col("method") == "BBKNN").then(None).otherwise(pl.col(col)).alias(col)
    for col in ["kbet_accept", "ilisi"]
)

for ax, cols, title in zip(
    axes,
    [batch_metrics, bio_metrics],
    ["Batch mixing", "Bio conservation"],
    strict=True,
):
    scores = plot_df.select(cols).to_numpy().astype(np.float64)
    im = ax.imshow(np.ma.masked_invalid(scores), cmap=cmap, vmin=0, vmax=1)
    for i, j in np.ndindex(scores.shape):
        if not np.isnan(scores[i, j]):
            ax.text(
                j,
                i,
                f"{scores[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if scores[i, j] > 0.6 else "black",
            )
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    ax.set_title(title)

fig.colorbar(im, ax=axes, label="Sc metrics", shrink=0.6)
plt.show()

# %% [markdown]
# Read the two panels together. A method that tops batch mixing but drops on
# bio conservation has merged cell types along with the batches, which is worse
# than doing nothing.
#
# Two of these tell you little on this data set. cLISI sits at 1 for every
# method: six coarse, well separated cell types keep their neighbourhoods pure
# no matter what. Batch ASW barely moves either, since two PBMC batches of the
# same tissue never separate much on silhouette to begin with. Both earn their
# keep on finer labels and nastier batch effects. On two PBMC runs, kBET, iLISI
# and PCR do the talking. BBKNN's row is greyed out on the batch side: its kBET
# and iLISI in the table above are the artefact from its section, not a win.
#
# There is no universally best batch correction method. Some practical
# guidance:
#
# - **Harmony** is a good default. It is fast, operates on the PCA embedding
#   directly, and handles multiple batch variables simultaneously. It tends to
#   work well when batch effects are moderate and cell type composition is
#   broadly similar across batches. Version 2 prunes batches per cluster and sets
#   the ridge penalty dynamically. On this data set it is not an improvement:
#   v1 is ahead on kBET acceptance and PCR, the rest is tied, and v2 is only
#   modestly faster (46 ms against 51 ms).
# - **fastMNN** is worth considering when cell type composition differs
#   substantially across batches. The mutual nearest neighbour approach is less
#   sensitive to this because it only aligns cells that have a plausible
#   biological match in another batch.
# - **Seurat CCA and rPCA** are the anchor-based options. Like fastMNN they
#   only align cells with a plausible match in another batch, so they hold up
#   when cell type composition differs. They are the heaviest of the five
#   though, and on data that already mixes well they can make the batch metrics
#   slightly worse. rPCA is the cheaper and gentler of the two and a sensible
#   default; go to CCA when the batches genuinely don't share their cell types.
# - **BBKNN** leaves the embedding alone and hands a batch-balanced graph
#   straight to clustering. It's the lightest touch of the lot, and that cuts
#   both ways: when the batches sit far apart, as here, the cross-batch edges
#   carry next to no weight and the clusters stay split by batch. Judge it on
#   clustering, not on kBET or iLISI.
#
# In practice, running multiple methods and looking at quantitative metrics is
# the most reliable approach. No single metric captures everything: kBET and
# PCR only see batch, cLISI and cell type ASW only see biology, and the kNN
# metrics can't score a method that builds the kNN to look balanced. UMAPs and
# tSNE assessments should be taken with a big pinch of
# salt, see
# [Chari, et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288)
# Admittedly, they do look pretty however.
#
# ## Clean up

# %%
tempdir_batch_cor.cleanup()
