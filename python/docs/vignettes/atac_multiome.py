# %% [markdown]
# # Joint RNA and ATAC analysis with bixverse
#
# ## Intro
#
# 10x multiome measures gene expression and chromatin accessibility in the
# same nuclei. This vignette takes the unsorted 3k PBMC data set (Cell Ranger
# ARC 2.0.0) from the raw files to linked peaks and genes: fragment QC and
# multiplet calls, peaks called per cluster, TF-IDF and LSI, the weighted
# nearest neighbour graph over both modalities, gene activity and label
# transfer, differential accessibility and peak to gene links.
#
# ATAC lives next to the RNA in a `SingleCellsMultiModal`: a second v5 store
# under `atac/` with peaks as features, log-CPM as the norm until `run_tfidf`
# switches it to TF-IDF, and the genomic coordinates in an interval index. Its
# cells are the RNA cells, in the same order, so the RNA cell filter applies to
# both. Read the [multi-modal vignette](multi_modal_single_cells.ipynb) first
# if you have not. ATAC without RNA has [its own vignette](atac_only.ipynb).
#
# Motif analysis (chromVAR) and coverage tracks are not in bixverse yet.

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
# `download_pbmc_multiome` fetches the filtered feature barcode matrix (RNA
# and Cell Ranger's peak counts in one h5), the fragment file with its tabix
# index, Cell Ranger's peaks, the GENCODE v32 annotation Cell Ranger ARC's
# GRCh38 reference was built from, and the ENCODE hg38 blacklist.

# %%
data_dir = bv.io.download_pbmc_multiome()
h5_path = data_dir / "pbmc_unsorted_3k_filtered_feature_bc_matrix.h5"
fragments = data_dir / "pbmc_unsorted_3k_atac_fragments.tsv.gz"
gtf = data_dir / "gencode.v32.primary_assembly.annotation.gtf.gz"
blacklist = data_dir / "hg38-blacklist.v2.bed.gz"

tempdir = tempfile.TemporaryDirectory()
exp_dir = Path(tempdir.name) / "pbmc_multiome"

# %% [markdown]
# ## Loading
#
# The RNA comes from the `Gene Expression` rows of the h5, the ATAC from its
# `Peaks` rows. `add_atac_tenx` writes the peak store for every cell of the
# RNA store, in its order; it counts Tn5 insertions, as Cell Ranger does.

# %%
bv.io.load_tenx_h5(exp_dir, h5_path)
mm = bv.SingleCellsMultiModal(exp_dir)
bv.atac.add_atac_tenx(mm, h5_path)
mm

# %%
var = mm.var()
mm.var(modality="atac").head()

# %% [markdown]
# ## Quality control
#
# Per cell from the fragment file: fragments, nucleosome signal (mono- over
# nucleosome-free fragments), TSS enrichment, the fraction on chrM and the
# fraction of insertions in peaks. Cell Ranger ARC leaves chrM out of the
# fragment file, so the ATAC mito fraction is zero throughout here; the RNA
# side has its own.

# %%
atac_qc = bv.atac.fragment_qc(mm, fragments, gtf)
atac_qc.head()

# %%
symbol_col = "gene_name"
mito = var.filter(pl.col(symbol_col).str.contains("^MT-"))["gene_id"].to_list()
bv.qc.gene_set_proportions(mm, {"MT": mito}, streaming=False, verbose=False)

# %% [markdown]
# The two library-level checks first: insertions pile up at TSSs, and fragment
# lengths come in nucleosome steps. A flat TSS profile or a smooth length
# distribution means a failed library, whatever the per-cell numbers say.

# %%
tss_profile, frag_lengths = bv.atac.fragment_profiles(mm, fragments, gtf)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
bv.pl.tss_profile_plot(tss_profile, ax=axes[0])
bv.pl.fragment_length_plot(frag_lengths, ax=axes[1])
plt.show()

# %%
obs = mm.obs(
    cols=[
        "cell_idx",
        "lib_size",
        "nnz",
        "MT",
        "atac_n_fragments",
        "atac_lib_size",
        "atac_nucleosome_signal",
        "atac_tss_enrichment",
        "atac_frip",
    ]
)
bv.pl.joint_plot(
    obs,
    "atac_tss_enrichment",
    "atac_n_fragments",
    log_scale=True,
    x_label="Fragments per cell",
    y_label="TSS enrichment",
)
plt.show()

# %% [markdown]
# ### Multiplets
#
# Nuclei carry two copies of each chromosome, so a locus covered by more than
# two of a cell's fragments points at a second nucleus. AMULET counts those loci
# per cell and tests them against a Poisson background, without clustering or
# RNA. Regions that pile up fragments in every cell, the blacklist, are left
# out; a repeat list would clean the calls further.

# %%
amulet = bv.atac.amulet(mm, fragments, restrict_beds=[blacklist])
print(f"{amulet['amulet_multiplet'].sum()} of {amulet.height} cells called multiplets")

# %% [markdown]
# ### Thresholds
#
# On the ATAC side we keep cells with 1,000 to 100,000 insertions in peaks, a
# nucleosome signal under 2 and a TSS enrichment over 2, cut-offs that sit in
# the tails of the distributions above. On the RNA side we flag MAD outliers on
# library size and genes, and cells above 3 MADs of mitochondrial share.

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
        "atac_lib_size": {"lower": 1000, "upper": 100_000},
        "atac_nucleosome_signal": {"upper": 2},
        "atac_tss_enrichment": {"lower": 2},
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
multiplet = amulet["amulet_multiplet"].to_numpy()
drop = atac_calls.combined | rna_calls.combined | multiplet
print(
    f"ATAC fails {atac_calls.combined.sum()}, RNA fails {rna_calls.combined.sum()}, "
    f"multiplets {multiplet.sum()}, dropped {drop.sum()} of {drop.size}"
)
mm.set_cells_to_keep(obs["cell_idx"].to_numpy()[~drop])

# %% [markdown]
# ## RNA
#
# Highly variable genes, PCA, kNN, Leiden. These clusters are what the peaks
# get called on below.

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
# populations. We call them per RNA cluster with bixverse's port of MACS3
# `callpeak` (`--nomodel --extsize 200 --shift -100` on the Tn5 insertions),
# merge the clusters' peaks, drop those on the standard chromosomes' blacklist
# regions and on contigs, and recount the fragments against the merged set. The
# new store replaces the Cell Ranger one.

# %%
cr_peaks = mm.var(modality="atac").height
peaks = bv.atac.call_peaks(mm, fragments, group_by="leiden_clustering")
peaks = peaks.filter(pl.col("chrom").str.contains(r"^chr[0-9XY]+$"))
peaks = bv.atac.remove_blacklisted(peaks, blacklist)
print(f"Cell Ranger peaks: {cr_peaks}, merged per-cluster peaks: {peaks.height}")
bv.atac.add_atac_fragments(mm, fragments, peaks=peaks, overwrite=True)
bv.atac.annotate_peaks(mm, gtf)
mm.var(modality="atac").head()

# %% [markdown]
# ## TF-IDF and LSI
#
# TF-IDF over the kept cells, then a truncated SVD of the TF-IDF. We keep peaks
# above the 25th percentile of total counts: the bottom quarter is mostly
# noise at this depth. The first LSI component of ATAC data usually tracks
# sequencing depth rather than cell identity; `calculate_lsi` leaves it out of
# the embedding by default, and the depth plot shows whether that is right
# here (hollow points are the components left out).

# %%
bv.atac.run_tfidf(mm)
top = bv.atac.find_top_features(mm, min_cutoff="q25")
print(f"{top.size} of {mm.var(modality='atac').height} peaks kept for the LSI")
bv.atac.calculate_lsi(mm, n_comp=50)
bv.pl.lsi_depth_plot(bv.atac.get_lsi_depth_cor(mm))
plt.show()

# %%
bv.graph.find_neighbours(mm, embd_to_use="lsi", modality="atac", no_embd_to_use=30)
bv.embed.umap(mm, embd_to_use="lsi", modality="atac")
bv.pl.embedding_plot(
    mm, "umap", "leiden_clustering", discrete=True, embd_modality="atac", point_size=0.5
)
plt.show()

# %% [markdown]
# The RNA clusters stay together on the ATAC UMAP: the two modalities come
# from the same nuclei.
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
    no_embd_to_use_1=30,
    no_embd_to_use_2=30,
)
bv.graph.find_clusters(mm, modality="wnn", name="wnn_clusters", res=0.8)
bv.embed.umap(mm, modality="wnn")
bv.pl.embedding_plot(
    mm, "umap", "wnn_clusters", discrete=True, embd_modality="wnn", point_size=0.5
)
plt.show()

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
# Gene activity counts fragments over gene body plus 2 kb upstream into a gene
# by cell store of its own, the shared feature space for moving labels from an
# RNA reference onto ATAC cells. With paired data we know the answer: the RNA
# clusters of the same nuclei. Transfer them through gene activity and the
# ATAC LSI and count how many come back; `weight_owner` checks the LSI and the
# gene activity keep the same cells.

# %%
ga = bv.atac.gene_activity(mm, fragments, gtf)
transfer = bv.annotate.transfer_labels_seurat(
    mm,
    ga,
    "leiden_clustering",
    mm.atac_cache.embeddings["lsi"],
    weight_owner=mm,
)
truth = mm.obs(cols=["leiden_clustering"], filtered=True)["leiden_clustering"]
predicted = transfer["predicted_leiden_clustering"].cast(pl.String)
agree = (predicted == truth.cast(pl.String)).mean()
print(f"transferred label equals the RNA cluster for {agree:.1%} of cells")

# %% [markdown]
# Gene activity is coarse, but the markers show through on the WNN UMAP:

# %%
bv.pl.feature_plot(
    mm,
    [symbol_to_id[g] for g in ["MS4A1", "CD14", "NKG7"]],
    "umap",
    feature_labels={symbol_to_id[g]: g for g in ["MS4A1", "CD14", "NKG7"]},
    embd_modality="wnn",
    expr_sc=ga,
    point_size=0.5,
)
plt.show()

# %% [markdown]
# ## Differential accessibility
#
# The marker functions take `modality="atac"` and test peaks instead of
# genes, here every WNN cluster against the rest. On ATAC the default test is
# a logistic regression per peak with the library size as covariate (`group ~
# peak + depth` against `group ~ depth`), so a cluster with deeper cells does
# not light up every peak. `method="wilcox"` runs the Wilcoxon on the TF-IDF
# instead.
#
# Don't read these p-values as inference. Every cell counts as a replicate, so
# with a few hundred cells per cluster they go astronomically small with either
# test; use them to rank peaks. With replicates,
# `bv.de.pseudobulk_dge(..., modality="atac")` runs edgeR on summed peak counts
# per sample, which is the test to trust. The closest gene from
# `annotate_peaks` names the peaks.

# %%
da = bv.de.find_all_markers(mm, "wnn_clusters", modality="atac", verbose=0)
annotation = mm.var(
    modality="atac", cols=["feature_id", "closest_gene", "distance_to_tss"]
)
da = da.join(annotation, on="feature_id", how="left")
da.filter(pl.col("fdr") < 0.05).group_by("grp").len().sort("grp")

# %%
da.filter(pl.col("fdr") < 0.05).sort("lfc", descending=True).head(10)

# %% [markdown]
# A top peak's accessibility on the WNN UMAP:

# %%
top_peak = da.filter(pl.col("fdr") < 0.05).sort("lfc", descending=True)["feature_id"][0]
bv.pl.feature_plot(
    mm,
    [top_peak],
    "umap",
    expr_modality="atac",
    embd_modality="wnn",
    point_size=0.5,
)
plt.show()

# %% [markdown]
# ## Peak to gene links
#
# A peak whose accessibility rises and falls with a nearby gene's expression
# across cell states is a candidate regulatory element for that gene.
# `peak_to_gene` sums both modalities over groups of cells, here fine WNN
# clusters, correlates every peak within 250 kb of a TSS with the gene, and
# tests the correlation. Fine clusters rather than the cells themselves: single
# cells are too sparse for a correlation to mean anything.

# %%
bv.graph.find_clusters(mm, modality="wnn", name="wnn_fine", res=5.0)
links = bv.atac.peak_to_gene(mm, gtf, "wnn_fine", min_cells=10)
print(f"{links.height} links at FDR < 0.05 over {links['gene_id'].n_unique()} genes")
links.sort("r", descending=True).head(10)
