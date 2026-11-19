# %% [markdown]
# # Joint RNA and ATAC analysis with bixverse
#
# *Follows Signac's
# [10x multiome vignette](https://stuartlab.org/signac/articles/pbmc_multiomic)
# on the same data.*
#
# ## Intro
#
# 10x multiome measures gene expression and chromatin accessibility in the
# same nuclei. This vignette takes the 10k PBMC granulocyte-sorted data set
# (Cell Ranger ARC 2.0.0) through the steps of the Signac vignette: fragment
# QC, multiplet calls, peak calling per cluster, TF-IDF and LSI, the weighted
# nearest neighbour graph over both modalities, gene activity and label
# transfer from RNA to ATAC.
#
# ATAC lives next to the RNA in a `SingleCellsMultiModal`: a second v5 store
# under `atac/` with peaks as features, TF-IDF as the norm and the genomic
# coordinates in an interval index. Its cells are the RNA cells, in the same
# order, so the RNA cell filter applies to both. Read the
# [multi-modal vignette](multi_modal_single_cells.ipynb) first if you have not.

# %%
import tempfile
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import bixverse as bv
import bixverse.embed

# %% [markdown]
# ## The data
#
# Four files from the 10x data set page: the filtered feature barcode matrix
# (RNA and Cell Ranger's peak counts in one h5), the fragment file with its
# tabix index, and the GENCODE v32 annotation Cell Ranger ARC's GRCh38
# reference was built from.

# %%
data_dir = Path.home() / "data" / "bixverse" / "atac"
h5_path = data_dir / "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
fragments = data_dir / "pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz"
gtf = data_dir / "gencode.v32.primary_assembly.annotation.gtf.gz"

tempdir = tempfile.TemporaryDirectory()
exp_dir = Path(tempdir.name) / "pbmc_multiome"

# %% [markdown]
# ## Loading
#
# The RNA comes from the `Gene Expression` rows of the h5, the ATAC from its
# `Peaks` rows. `add_atac_tenx` writes the peak store for every cell of the
# RNA store, in its order.

# %%
bv.io.load_tenx_h5(exp_dir, h5_path)
mm = bv.SingleCellsMultiModal(exp_dir)
bv.atac.add_atac_tenx(mm, h5_path)
mm

# %%
var = mm.var()
var.head()

# %%
mm.var(modality="atac").head()

# %% [markdown]
# ## Quality control
#
# Per cell from the fragment file: the number of fragments, the nucleosome
# signal (mono- over nucleosome-free fragments), the TSS enrichment (Signac's
# fast definition), the fraction on chrM and the fraction of fragments in
# peaks. Mitochondrial RNA counts come from the RNA side as usual. Cell
# Ranger ARC leaves chrM out of the fragment file, so the ATAC mito fraction
# is zero throughout here.

# %%
atac_qc = bv.atac.atac_fragment_qc(mm, fragments, gtf)
atac_qc.head()

# %%
symbol_col = "gene_name"
mito = var.filter(pl.col(symbol_col).str.contains("^MT-"))["gene_id"].to_list()
bv.qc.gene_set_proportions(mm, {"MT": mito}, streaming=False, verbose=False)

# %% [markdown]
# ### Multiplets
#
# Multiome nuclei carry two copies of each chromosome, so a locus covered by
# more than two of a cell's fragments points at a second nucleus. AMULET counts
# those loci per cell and tests them against a Poisson background. It needs no
# clustering and no RNA. With a repeat list (`restrict_beds`) the calls are
# cleaner; without one, loci in repeats inflate the counts of every cell alike.

# %%
amulet = bv.atac.amulet(mm, fragments)
print(f"{amulet['amulet_multiplet'].sum()} of {amulet.height} cells called multiplets")

# %% [markdown]
# ### Thresholds
#
# Signac's thresholds for this data set on the ATAC side: `1800 < nCount <
# 1e5` peak counts, nucleosome signal under 2, TSS enrichment over 1. On the
# RNA side we flag MAD outliers on library size and genes, and cells above
# 3 MADs of mitochondrial share.

# %%
obs = mm.obs(
    cols=[
        "cell_idx",
        "lib_size",
        "nnz",
        "MT",
        "atac_lib_size",
        "atac_nucleosome_signal",
        "atac_tss_enrichment",
        "atac_frip",
        "amulet_multiplet",
    ]
)
obs.describe()

# %%
atac_metrics = {
    "atac_lib_size": obs["atac_lib_size"].to_numpy(),
    "atac_nucleosome_signal": obs["atac_nucleosome_signal"].to_numpy(),
    "atac_tss_enrichment": obs["atac_tss_enrichment"].to_numpy(),
}
atac_calls = bv.qc.run_cell_qc_fixed(
    atac_metrics,
    obs["cell_idx"].to_numpy(),
    {
        "atac_lib_size": {"lower": 1800, "upper": 100_000},
        "atac_nucleosome_signal": {"upper": 2},
        "atac_tss_enrichment": {"lower": 1},
    },
)
bv.pl.violin_plot(atac_calls)
plt.show()

# %%
rna_metrics = {
    "log10_lib_size": np.log10(obs["lib_size"].to_numpy()),
    "log10_nnz": np.log10(obs["nnz"].to_numpy()),
    "MT": obs["MT"].to_numpy(),
}
directions: dict[str, Literal["twosided", "below", "above"]] = {
    "log10_lib_size": "twosided",
    "log10_nnz": "twosided",
    "MT": "above",
}
rna_calls = bv.qc.run_cell_qc(
    rna_metrics, obs["cell_idx"].to_numpy(), directions=directions, threshold=3.0
)
bv.pl.violin_plot(rna_calls)
plt.show()

# %%
multiplet = obs["amulet_multiplet"].to_numpy()
drop = atac_calls.combined | rna_calls.combined | multiplet
print(
    f"ATAC fails {atac_calls.combined.sum()}, RNA fails {rna_calls.combined.sum()}, "
    f"multiplets {multiplet.sum()}, dropped {drop.sum()} of {drop.size}"
)
mm.set_cells_to_keep(obs["cell_idx"].to_numpy()[~drop])

# %% [markdown]
# ## RNA
#
# The usual: highly variable genes, PCA, kNN, Leiden. These clusters are what
# the peaks get called on below.

# %%
bv.pp.find_hvg(mm, hvg_no=2000)
bv.pp.calculate_pca(mm, no_pcs=50)
bv.graph.find_neighbours(mm, no_embd_to_use=30)
bv.graph.find_clusters(mm, res=0.8)
bv.embed.umap(mm, no_embd_to_use=30)
bv.pl.embedding_plot(mm, "umap", "leiden_clustering", discrete=True, point_size=0.5)
plt.show()

# %% [markdown]
# ## Peaks
#
# Cell Ranger calls peaks on all cells pooled, which under-calls rare
# populations. As in the Signac vignette we call them per cluster with a native
# port of MACS3 `callpeak` (Signac's settings: `--nomodel --extsize 200
# --shift -100`), merge the per cluster peaks and count the fragments against
# the merged set. The new store replaces the Cell Ranger one.

# %%
cr_peaks = mm.var(modality="atac").height
peaks = bv.atac.call_peaks(mm, fragments, group_by="leiden_clustering")
print(f"Cell Ranger peaks: {cr_peaks}, merged per-cluster peaks: {peaks.height}")
peaks.head()

# %%
peaks = peaks.filter(pl.col("chrom").str.contains(r"^chr[0-9XY]+$"))
bv.atac.add_atac_fragments(mm, fragments, peaks=peaks)
mm.var(modality="atac").head()

# %% [markdown]
# ## TF-IDF and LSI
#
# TF-IDF over the kept cells (Signac method 1), the most accessible peaks, then
# a truncated SVD. The first component tracks sequencing depth; we look before
# dropping it.

# %%
bv.atac.run_tfidf(mm)
top = bv.atac.find_top_features(mm, min_cutoff="q5")
print(f"{top.size} top peaks")
bv.atac.calculate_lsi(mm, n_comp=50)
bv.atac.get_lsi_depth_cor(mm).head(5)

# %%
bv.graph.find_neighbours(mm, embd_to_use="lsi", modality="atac")
bv.embed.umap(mm, embd_to_use="lsi", modality="atac")
bv.pl.embedding_plot(
    mm, "umap", "leiden_clustering", discrete=True, embd_modality="atac", point_size=0.5
)
plt.show()

# %% [markdown]
# The RNA clusters should stay together on the ATAC UMAP: the two modalities
# come from the same nuclei.
#
# ## Weighted nearest neighbours
#
# WNN fuses the RNA PCA and the ATAC LSI per cell, weighting each by how well
# it predicts the cell's neighbourhood.

# %%
bv.mm.generate_wnn_graph(
    mm,
    modality_2="atac",
    embd_to_use_2="lsi",
    no_embd_to_use_1=50,
)
bv.graph.find_clusters(mm, modality="wnn", name="wnn_clusters", res=0.8)
bv.embed.umap(mm, modality="wnn")
bv.pl.embedding_plot(
    mm,
    "umap",
    "wnn_clusters",
    discrete=True,
    embd_modality="wnn",
    point_size=0.5,
)
plt.show()

# %%
weights = bv.mm.get_wnn_weights(mm)
weights.head()

# %%
symbol_to_id = dict(zip(var[symbol_col], var["gene_id"], strict=False))
markers = ["CD3D", "CD8A", "MS4A1", "CD14", "FCGR3A", "NKG7", "LYZ", "IL7R"]
bv.pl.feature_plot(
    mm,
    [symbol_to_id[g] for g in markers],
    "umap",
    feature_labels={symbol_to_id[g]: g for g in markers},
    embd_modality="wnn",
    point_size=0.5,
)
plt.show()

# %% [markdown]
# ## Gene activity and label transfer
#
# Gene activity counts fragments over gene body plus 2 kb upstream (Signac's
# `GeneActivity`) into a gene by cell store of its own. It is the shared
# feature space for moving labels from an RNA reference onto ATAC cells. With
# paired data we know the answer: the RNA clusters of the same nuclei. Transfer
# them through gene activity and the ATAC LSI and count how many come back.

# %%
ga = bv.atac.gene_activity(mm, fragments, gtf)
ga.var().head()

# %%
transfer = bv.annotate.transfer_labels_seurat(
    mm,
    ga,
    "leiden_clustering",
    mm.atac_cache.embeddings["lsi"],
)
truth = mm.obs(cols=["leiden_clustering"], filtered=True)["leiden_clustering"]
predicted = transfer["predicted_leiden_clustering"].cast(pl.String)
agree = (predicted == truth.cast(pl.String)).mean()
print(f"transferred label equals the RNA cluster for {agree:.1%} of cells")

# %% [markdown]
# ## Differential accessibility
#
# The marker functions take `modality="atac"` and test peaks instead of
# genes: here every WNN cluster against the rest. On ATAC the default is
# Signac's recommended test, a logistic regression per peak with the
# library size as covariate (`group ~ peak + nCount` against
# `group ~ nCount`), so a cluster with deeper cells does not light up every
# peak. `method="wilcox"` gets you the Wilcoxon on the TF-IDF layer instead.
#
# Don't read these p-values as inference. Every cell counts as a replicate,
# so with a few hundred cells per cluster they go astronomically small with
# either test; use them to rank peaks. With replicates,
# `bv.de.pseudobulk_dge(..., modality="atac")` runs edgeR on summed peak
# counts per sample, which is the test to trust.

# %%
da = bv.de.find_all_markers(mm, "wnn_clusters", modality="atac", verbose=0)
da.filter(pl.col("fdr") < 0.05).group_by("grp").len().sort("grp")

# %%
da.filter(pl.col("fdr") < 0.05).sort("lfc", descending=True).head(10)
