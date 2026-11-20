# %% [markdown]
# # Single cell ATAC without RNA
#
# ## Intro
#
# Most single cell ATAC data has no RNA next to it: Cell Ranger ATAC runs, or
# public fragment files. This vignette starts from a fragment file alone:
# cells called from the fragment counts, QC and multiplets, peaks per cluster,
# TF-IDF and LSI, clusters and differential accessibility, then gene activity
# and labels transferred from an RNA reference.
#
# The fragments are the ATAC half of the 10x PBMC 3k multiome used in the
# [multiome vignette](atac_multiome.ipynb). That has a use here: its RNA half is
# a reference built from the very same nuclei, so we can check the transferred
# labels against the RNA clusters of each cell, which a real ATAC-only data
# set cannot offer.
#
# An ATAC-only experiment is a `SingleCellsAtac`: its own store holds the
# peaks or tiles, and the ATAC functions take it exactly as they take the ATAC
# side of a `SingleCellsMultiModal`. `modality="atac"` and the default both
# name its store.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import bixverse as bv
import bixverse.embed

# %%
data_dir = bv.io.download_pbmc_multiome()
fragments = data_dir / "pbmc_unsorted_3k_atac_fragments.tsv.gz"
gtf = data_dir / "gencode.v32.primary_assembly.annotation.gtf.gz"
blacklist = data_dir / "hg38-blacklist.v2.bed.gz"
h5_path = data_dir / "pbmc_unsorted_3k_filtered_feature_bc_matrix.h5"

tempdir = tempfile.TemporaryDirectory()
base = Path(tempdir.name)

# %% [markdown]
# ## Calling cells
#
# Without a barcode list the cells come from the fragment counts: real cells
# carry thousands of fragments, empty droplets a handful. The knee plot of
# fragments per barcode, sorted, shows where the two meet. With Cell Ranger
# ATAC output, `cellranger="singlecell.csv"` takes its calls instead.

# %%
counts = bv.atac.barcode_fragment_counts(fragments)
rank = np.arange(1, counts.height + 1)
fig, ax = plt.subplots(figsize=(5, 3.6), layout="constrained")
ax.loglog(rank, counts["n_fragments"].to_numpy(), lw=1)
ax.axhline(3000, color="grey", ls="--", lw=0.8)
ax.set_xlabel("Barcode rank")
ax.set_ylabel("Fragments")
plt.show()

# %% [markdown]
# The knee sits near 3,000 fragments. We call cells there and count them into
# the peaks Cell Ranger called on all cells pooled, which every Cell Ranger ATAC
# run ships. They are good enough to cluster on; better peaks come from the
# clusters below.

# %%
first = bv.io.load_atac_fragments(
    base / "first",
    fragments,
    peaks=data_dir / "pbmc_unsorted_3k_atac_peaks.bed",
    min_fragments=3000,
)
first

# %% [markdown]
# ## Quality control
#
# The same metrics as with RNA next to it: fragments, nucleosome signal, TSS
# enrichment, and AMULET's multiplet calls, which need nothing but the
# fragments.

# %%
qc = bv.atac.fragment_qc(first, fragments, gtf)
amulet = bv.atac.amulet(first, fragments, restrict_beds=[blacklist])
tss_profile, frag_lengths = bv.atac.fragment_profiles(first, fragments, gtf)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
bv.pl.tss_profile_plot(tss_profile, ax=axes[0])
bv.pl.fragment_length_plot(frag_lengths, ax=axes[1])
plt.show()

# %%
obs = first.obs(
    cols=[
        "cell_idx",
        "atac_n_fragments",
        "atac_nucleosome_signal",
        "atac_tss_enrichment",
        "amulet_multiplet",
    ]
)
keep = (
    (obs["atac_nucleosome_signal"] < 2)
    & (obs["atac_tss_enrichment"] > 2)
    & ~obs["amulet_multiplet"]
)
print(f"keeping {keep.sum()} of {obs.height} cells")
first.set_cells_to_keep(obs.filter(keep)["cell_idx"].to_numpy())

# %% [markdown]
# ## First clusters
#
# TF-IDF, LSI and Leiden on Cell Ranger's peaks give the clusters to call
# peaks on.

# %%
bv.atac.run_tfidf(first)
bv.atac.find_top_features(first, min_cutoff="q25")
bv.atac.calculate_lsi(first, n_comp=30)
bv.graph.find_neighbours(first, embd_to_use="lsi")
bv.graph.find_clusters(first, res=0.8)

# %% [markdown]
# ## Peaks
#
# Peaks per cluster, merged, blacklist and contigs dropped, then the fragments
# counted against them into the experiment we keep, for the cells that passed
# QC.

# %%
peaks = bv.atac.call_peaks(first, fragments, group_by="leiden_clustering")
peaks = peaks.filter(pl.col("chrom").str.contains(r"^chr[0-9XY]+$"))
peaks = bv.atac.remove_blacklisted(peaks, blacklist)
x = bv.io.load_atac_fragments(
    base / "peaks",
    fragments,
    peaks=peaks,
    barcodes=first.cell_ids(filtered=True),
)
bv.atac.fragment_qc(x, fragments, gtf)
bv.atac.annotate_peaks(x, gtf)
x

# %% [markdown]
# ## TF-IDF, LSI, clusters

# %%
bv.atac.run_tfidf(x)
bv.atac.find_top_features(x, min_cutoff="q25")
bv.atac.calculate_lsi(x, n_comp=50)
bv.pl.lsi_depth_plot(bv.atac.get_lsi_depth_cor(x))
plt.show()

# %%
bv.graph.find_neighbours(x, embd_to_use="lsi", no_embd_to_use=30)
bv.graph.find_clusters(x, res=0.8)
bv.embed.umap(x, embd_to_use="lsi")
bv.pl.embedding_plot(x, "umap", "leiden_clustering", discrete=True, point_size=0.5)
plt.show()

# %% [markdown]
# ## Differential accessibility
#
# On an ATAC experiment the marker functions test peaks by default, with the
# logistic regression that takes the library size as covariate. As in the
# multiome vignette, treat the p-values as a ranking.

# %%
da = bv.de.find_all_markers(x, "leiden_clustering", verbose=0)
annotation = x.var(modality="atac", cols=["feature_id", "closest_gene"])
da = da.join(annotation, on="feature_id", how="left")
da.filter(pl.col("fdr") < 0.05).sort("lfc", descending=True).head(10)

# %% [markdown]
# ## Labels from an RNA reference
#
# Gene activity puts the ATAC cells in gene space, where an RNA reference can
# meet them. The reference here is the RNA of the same 3k nuclei, clustered
# the usual way; in practice it would be any annotated RNA data set of the
# tissue.

# %%
rna = bv.io.load_tenx_h5(base / "rna", h5_path)
bv.pp.find_hvg(rna, hvg_no=2000)
bv.pp.calculate_pca(rna, no_pcs=30)
bv.graph.find_neighbours(rna)
bv.graph.find_clusters(rna, res=0.8, name="rna_cluster")

# %%
ga = bv.atac.gene_activity(x, fragments, gtf)
transfer = bv.annotate.transfer_labels_seurat(
    rna, ga, "rna_cluster", x.atac_cache.embeddings["lsi"], weight_owner=x
)

# %% [markdown]
# Each ATAC cell's predicted cluster against the RNA cluster of the same
# barcode:

# %%
truth = rna.obs(cols=["cell_id", "rna_cluster"])
pred = pl.DataFrame(
    {
        "cell_id": x.cell_ids(filtered=True),
        "predicted": transfer["predicted_rna_cluster"].cast(pl.String),
    }
).join(truth, on="cell_id", how="inner")
agree = (pred["predicted"] == pred["rna_cluster"].cast(pl.String)).mean()
print(f"{pred.height} shared cells; the transferred label is right for {agree:.1%}")

# %%
x.add_obs(
    pl.DataFrame(
        {
            "predicted": transfer["predicted_rna_cluster"].cast(pl.String),
        }
    )
)
bv.pl.embedding_plot(x, "umap", "predicted", discrete=True, point_size=0.5)
plt.show()
