# %% [markdown]
# # Scaling to millions of cells with bixverse
#
# *Python port of the R vignette [Scaling to millions of cells with bixverse](https://gregorlueg.github.io/bixverse/articles/single_cell_big_data.html). Original author: Gregor Lueg.*
#
# ## Intro
#
# This vignette demonstrates how to run a single cell analysis on a data set
# containing nearly 1 million cells on a local machine... In this case, a
# MacBook Air with 24 GBs (if you think *"IMPOSSIBLE!"*, you would not be wrong
# coming from other languages/frameworks). If you have not read the
# [design choices](design_single_cell.ipynb) and the
# [introductory vignette](thinking_single_cell.ipynb), please do so first. The
# [PBMC3k walkthrough](pbmc_single_cell.ipynb) covers every step in detail on a
# small data set; here we focus on what changes when the cell count moves from
# thousands to a million.
#
# The data used here is the 1 million cell PBMC data set from Parse Biosciences,
# a commonly used reference for showing large single cell analyses, see the
# original article
# [here](https://support.parsebiosciences.com/hc/en-us/articles/7704577188500-How-to-analyze-a-1-million-cell-data-set-using-Scanpy-and-Harmony).
# **The important bit here is that this was run successfully on a MacBook Air
# with 24 GB (M3)**. And please contrast that with the following line for the
# analysis of this data:
#
# <i>To avoid computer crashes and program failures, we recommend you execute
# the Python code on a cloud computing platform. The instance you select on the
# cloud platform to run this analysis should consist of at least 8 threads and
# 160 GB of RAM.</i> (Parse Biosciences)
#
# > **Note:** None of the code blocks below are executed. This vignette is a
# > reference for how one *would* run such an analysis, not a live rendering:
# > the data set is far too large for CI/CD runners and GitHub would not be
# > happy. The timings quoted were measured through the R package on the
# > MacBook Air. R and Python call the same Rust core for every step here, but
# > the Python timings have not been measured.
#
# If you want to reproduce this, download the data from Parse Biosciences (or
# any other fat data set you have) and go have fun.
#
# ```bash
# wget "https://cdn.parsebiosciences.com/1M_PBMC_T1D_Parse.zip"
# ```
#
# ```python
# import tempfile
# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import numpy as np
# import polars as pl
#
# import bixverse as bv
# import bixverse.embed
# from bixverse.params import MtxIoParams, NeighboursParams
#
# # GPU acceleration (experimental), the `gpu` extra
# import bixverse_gpu as bvg
# from bixverse_gpu.params import KnnGpuDefaults, ScrubletGpuParams
# ```

# %% [markdown]
# ## Loading the data
#
# Once you have unzipped the data, we can start streaming it into memory and
# writing the files to disk.
#
# ```python
# data_path = Path("~/Desktop/parse_pbmc_1m/").expanduser()
#
# load_params = MtxIoParams(
#     path_mtx=str(data_path / "DGE_1M_PBMC.mtx"),
#     path_obs=str(data_path / "cell_metadata_1M_PBMC.csv"),
#     path_var=str(data_path / "all_genes_1M_PBMC.csv"),
#     cells_as_rows=True,
#     has_hdr=True,
# )
#
# # We will write everything into a temporary directory
# tempdir_big = tempfile.TemporaryDirectory()
# exp_dir = Path(tempdir_big.name) / "parse_pbmc_1m"
#
# # We cap the cell-to-gene conversion at 8 GB here. It will run for a bit
# # longer, but ensures your memory never blows up
# sc = bv.io.load_mtx(exp_dir, load_params, csc_mem_gb=8.0, verbose=True)
# ```
#
# This step is one of the slowest... A lot of things are happening here.
#
# - The MTX (plain or gzipped) is parsed once. The entries spill into per-cell
#   buckets on disk while the per-gene cell counts accumulate.
# - Genes expressed in fewer than `min_cells` cells are dropped, and the cell
#   QC (minimum library size, minimum features) is corrected for the dropped
#   genes afterwards.
# - The kept cells are written to the CSR-type file with the genes already
#   re-indexed; the normalisation is done on the fly. The spill files are
#   cleaned up at the end.
# - Lastly, the cell data are read back in chunks and transposed to the CSC
#   style file for quick gene retrieval, with the buffers capped at
#   `csc_mem_gb`.
#
# This approach avoids loading the whole data into memory at any point. The
# DuckDB also gets populated with the obs (based on what has been filtered) and
# the var. Reading from an h5ad instead? `bv.io.stream_h5ad()` is the streaming
# twin of `bv.io.load_h5ad()` and takes the same `csc_mem_gb`.
#
# Streaming the data in and processing it is what takes the most time. If you
# have already loaded the data and it exists on disk from a previous session,
# skip the streaming step entirely and reconnect. On a MacBook Air with 24 GB
# and an M3 chip, the load takes 90 seconds.
#
# ```python
# # the folder is hopefully not temp...
# sc = bv.io.load_existing(exp_dir)
# ```

# %% [markdown]
# ## Doublet detection
#
# Before any QC on library sizes and gene set proportions, we run Scrublet to
# flag doublets. Doublets are called per sample via the `group_by` argument,
# which is what you want when sample-level differences in capture rates would
# otherwise distort a single global score. Results are merged back onto obs and
# the doublets are dropped from the set of cells to keep. The runs over the
# individual samples happen sequentially to not overwhelm memory, and each run
# is massively parallelised internally in Rust.
#
# ```python
# scrublet_res = bv.doublets.scrublet(sc, group_by="sample", verbose=True)
#
# sc.add_obs(bv.doublets.get_data(scrublet_res))
#
# cells_without_doublets = (
#     sc.obs(cols=["cell_idx", "doublet"], filtered=True)
#     .filter(~pl.col("doublet"))["cell_idx"]
#     .to_numpy()
# )
#
# sc.set_cells_to_keep(cells_without_doublets)
# ```
#
# Due to having run over 24 samples in this case, this will take ~6 minutes.
# Ideal moment for a podcast, video of your favourite YouTuber and/or coffee,
# whichever floats your boat. Once you have passed this, things get
# substantially faster.
#
# ### Scrublet on the GPU
#
# Got a GPU? `bixverse_gpu.doublets.scrublet()` is a drop-in for
# `bv.doublets.scrublet()`. Three stages move to the device: the randomised
# sparse SVD, the projection of the simulated doublets into that PC space, and
# the kNN over the combined embedding. That embedding is
# `(1 + sim_doublet_ratio) * n_cells` rows tall, so the kNN is the stage that
# gets painful on the CPU as samples grow. HVG selection, the doublet
# simulation and the Otsu threshold stay on the CPU. `group_by` works the same
# way and the result is the same `ScrubletRes`, so everything downstream is
# unchanged.
#
# ```python
# scrublet_res = bvg.doublets.scrublet(
#     sc,
#     group_by="sample",
#     # ivf squeezes more speed out on the GPU compared to exhaustive
#     scrublet_params=ScrubletGpuParams(knn=KnnGpuDefaults(knn_method="ivf")),
#     verbose=True,
# )
#
# sc.add_obs(bv.doublets.get_data(scrublet_res))
# ```
#
# Pick one of the two. Scores correlate (~0.99 on the demuxlet PBMCs) but do
# not match bit for bit: the randomised SVD draws a different sketch and the
# GPU index breaks neighbour ties differently, so a handful of borderline calls
# can flip. On the MacBook Air this does not accelerate the method massively.
# More towards 5 minutes.

# %% [markdown]
# ## Quality control
#
# ### Gene set proportions
#
# The same gene set proportion logic from the PBMC3k vignette applies. The
# functions automatically recognise if you have more than 100k cells and use
# the streaming engine by default in this case. You have manual control via
# `streaming=None`. If you have a fat memory machine, you can set this to
# `False` despite a large data set.
#
# ```python
# var = sc.var()
#
# gs_of_interest = {
#     "MT": var.filter(pl.col("gene_name").str.contains("^MT-"))["gene_id"].to_list(),
#     "Ribo": var.filter(pl.col("gene_name").str.contains("^RP[SL]"))[
#         "gene_id"
#     ].to_list(),
# }
#
# bv.qc.gene_set_proportions(sc, gs_of_interest, verbose=True)
# ```
#
# This is where the streaming engine really comes into play. Done in under 10
# seconds.
#
# ### MAD outlier detection
#
# Outlier detection on library size, the number of detected genes and the
# mitochondrial proportion. We compute outliers per sample (`groups=`), which
# matters once batches have non-trivial differences in sequencing depth: a
# global threshold would otherwise penalise whole batches rather than
# individual cells within them.
#
# ```python
# qc_df = sc.obs(cols=["cell_idx", "sample", "lib_size", "nnz", "MT"], filtered=True)
#
# metrics = {
#     "log10_lib_size": np.log10(qc_df["lib_size"].to_numpy()),
#     "log10_nnz": np.log10(qc_df["nnz"].to_numpy()),
#     "MT": qc_df["MT"].to_numpy(),
# }
#
# directions = {
#     "log10_lib_size": "below",
#     "log10_nnz": "below",
#     "MT": "above",
# }
#
# qc = bv.qc.run_cell_qc(
#     metrics,
#     qc_df["cell_idx"].to_numpy(),
#     directions=directions,
#     threshold=3.0,
#     groups=qc_df["sample"],
# )
#
# sc.add_obs(bv.qc.get_data(qc))
# ```
#
# I cannot recommend trying to plot the
# metric distributions naively... matplotlib will not be happy at this cell
# count either.
#
# ### Filtering cells
#
# ```python
# sc.set_cells_to_keep(qc.cell_idx[~qc.combined])
# ```

# %% [markdown]
# ## Feature selection and PCA
#
# HVG selection at this scale again streams (left as is, the auto-detection
# takes care of this and streams when more than `1e5` cells are detected;
# `streaming=True` forces it). There are two data passes to avoid materialising
# large data in memory: the first calculates means and standard deviations,
# and the second does the variance stabilisation. Again, the streaming keeps
# memory pressure low and the function runs in ~10 seconds on the mentioned
# MacBook Air.
#
# ```python
# bv.pp.find_hvg(sc, hvg_no=2000, verbose=True)
# ```
#
# PCA is one of the points where GPU acceleration starts to noticeably matter
# (if you have some VRAM or unified memory to spare, it will make a
# difference). Both implementations share the same parameter interface, so
# swapping between them is a one-line change. The leading components should
# agree up to sign between the two, which is worth a quick visual check.
#
# ```python
# bv.pp.calculate_pca(sc, no_pcs=32)
#
# cpu_res = sc.cache.pca_factors.copy()
# cpu_s = sc.cache.pca_singular_vals.copy()
# ```
#
# This runs in ~8 seconds. Let's compare this against the GPU-accelerated
# version. Here the GEMMs of the randomised SVD, the ones before the thin SVD
# on the massively reduced matrix, run on the GPU. You pay for moving the data,
# but you should still see an acceleration compared to the CPU form.
#
# ```python
# bvg.pp.calculate_pca(sc, no_pcs=32, verbose=True)
#
# gpu_res = sc.cache.pca_factors.copy()
# gpu_s = sc.cache.pca_singular_vals.copy()
#
# # sanity check on the leading component
# fig, ax = plt.subplots(figsize=(4, 4))
# ax.scatter(
#     cpu_res[:, 0],
#     gpu_res[:, 0],
#     s=0.5,
#     color=bv.pl.bx_colors("main", n=1)[0],
#     rasterized=True,
# )
# ax.set_xlabel("PC1 (CPU)")
# ax.set_ylabel("PC1 (GPU)")
# plt.show()
# ```
#
# On the test MacBook Air, this runs in 5 seconds. In practice you would pick
# one. The GPU path becomes the more attractive option as the cell count grows
# and the more powerful your GPU is; on this data set you can feel the
# difference (if VRAM permits). You might see some slight differences due to
# floating point differences between CPU and GPU.

# %% [markdown]
# ## Nearest neighbours (GPU-accelerated)
#
# The GPU wheel exposes GPU-accelerated nearest neighbour search via
# `bixverse_gpu.graph.find_neighbours()`, thanks to CubeCL and WGPU, as long
# as you can fit the data into VRAM. `knn_method="nndescent"` builds a
# CAGRA-style graph; `"ivf"` and `"exhaustive"` are also on offer. The GPU
# version takes ~10 seconds, the CPU version with NNDescent ~15 seconds.
#
# ```python
# bv.graph.find_neighbours(
#     sc,
#     embd_to_use="pca",
#     neighbours_params=NeighboursParams(knn_method="nndescent"),
# )
# ```
#
# ```python
# bvg.graph.find_neighbours(
#     sc,
#     embd_to_use="pca",
#     knn_params=KnnGpuDefaults(knn_method="nndescent"),
#     verbose=True,
# )
# ```

# %% [markdown]
# ## Embeddings before batch correction
#
# ### Batch effect quantification
#
# A kBET score on the uncorrected neighbour graph gives a quantitative baseline
# to compare against once batch correction is applied...
#
# ```python
# kbet_prior = bv.integrate.calculate_kbet(sc, "sample")
#
# print(
#     f"kBET rejection rate: {kbet_prior.kbet_score:.3f} over "
#     f"{kbet_prior.n_batches} batches"
# )
# ```
#
# ### Plots
#
# Or if you are a visual person, let's check UMAP and tSNE.
#
# ```python
# bv.embed.umap(sc, slot_name="umap_prior")
# ```
#
# ```python
# bv.embed.tsne(
#     sc,
#     slot_name="tsne_prior",
#     # use FFT here! It will make a massive difference
#     params=bv.embed.tsne_defaults().set_params(approx="fft", ann="nndescent"),
# )
# ```
#
# You might wonder why in the case of tSNE we regenerate the kNN graph. You do
# not have to (`use_knn=True` reuses the cached one), but the rule of thumb is
# `k_neighbours = perplexity * 3.0`. With the default perplexity of 10 we would
# need 30 neighbours, but we previously only returned 15. So, we just rerun
# this. Skip tSNE if you don't need it, but the (GPU-accelerated)
# FFT-accelerated version is substantially faster than you would expect from
# tSNE. Same for the GPU-accelerated Adam optimiser for UMAP. You can run them
# via:
#
# ```python
# bvg.embed.umap(sc, slot_name="umap_prior_gpu")
# ```
#
# ```python
# bvg.embed.tsne(sc, slot_name="tsne_prior_gpu")
# ```
#
# ```python
# ax = bv.pl.embedding_plot(sc, "umap_prior", "sample", discrete=True)
# ax.set_title("No batch correction")
# plt.show()
# ```
#
# ```python
# ax = bv.pl.embedding_plot(sc, "tsne_prior", "sample", discrete=True)
# ax.set_title("No batch correction")
# plt.show()
# ```

# %% [markdown]
# ## Batch correction
#
# Okay, there is clearly a big sample batch effect. Let's remove that one.
# Again you have two options here... Harmony
# ([version 2](https://www.biorxiv.org/content/10.64898/2026.03.16.711825v2),
# recommended) on the CPU or a GPU-accelerated version. The CPU version first
# (~4 seconds):
#
# ```python
# bv.integrate.harmony_v2(sc, "sample")
# ```
#
# If you have a GPU at hand, you can also just use the GPU version (also 4
# seconds on the Air: the CPU path is already very optimised and the GPU is not
# particularly powerful). Again, you will observe some slight numerical
# differences... This is due to the GPU implementation, but the overall
# structure of the data is recovered in the same way. The result lands in the
# `"harmony_gpu"` embedding.
#
# ```python
# bvg.integrate.harmony_v2(sc, "sample")
# ```
#
# We overwrite the kNN graph with the now batch-corrected embedding:
#
# ```python
# bvg.graph.find_neighbours(
#     sc,
#     embd_to_use="harmony_gpu",  # or harmony_v2 if you just did the CPU version
#     knn_params=KnnGpuDefaults(knn_method="nndescent"),
#     verbose=True,
# )
# ```
#
# ### Batch correction quantification
#
# Requantify the kBET scores and see if we removed some of the batch
# effects...
#
# ```python
# kbet_post_gpu = bv.integrate.calculate_kbet(sc, "sample")
#
# print(
#     f"kBET rejection rate: {kbet_post_gpu.kbet_score:.3f} over "
#     f"{kbet_post_gpu.n_batches} batches"
# )
# ```

# %% [markdown]
# ## Cell type annotations
#
# This is a nice data set where the fast clustering makes a lot of sense... We
# use k-means to quantise the data to representative centroids, run the
# Louvain clustering on the centroids only and project the community
# membership back.
#
# ```python
# fast_cluster_res = bv.graph.fast_cluster(
#     sc,
#     embd_to_use="harmony_gpu",  # or harmony_v2 if you just did the CPU version
#     return_kmeans=False,
#     n_seeds=10,
#     verbose=True,
# )
#
# sc.add_obs(fast_cluster_res.memberships)
#
# sc.obs(filtered=True).head()
# ```

# %% [markdown]
# ## Differential gene expression
#
# ```python
# meta_data = sc.obs(cols=["cell_idx", "res_2"], filtered=True)
#
# cells_grp_a = meta_data.filter(pl.col("res_2") == 1)["cell_idx"].to_numpy()
# cells_grp_b = meta_data.filter(pl.col("res_2") == 2)["cell_idx"].to_numpy()
#
# dge_results = bv.de.find_markers(sc, cells_grp_a, cells_grp_b)
#
# dge_results.sort("fdr").head(25)
# ```
#
# The data is again streamed from disk, which makes this very fast...
#
# Training a model on the counts instead? `bv.CellLoader` streams shuffled
# minibatches straight out of the cell-major store, e.g. into torch, without
# ever holding the matrix in memory.

# %% [markdown]
# ## Key differences from the small-data workflow
#
# The API surface is intentionally identical to the PBMC3k walkthrough. The
# meaningful differences when working at the million-cell scale are:
#
# 1. **Streaming I/O.** Use `bv.io.load_mtx(..., csc_mem_gb=8.0)` (or
#    `bv.io.stream_h5ad()`, or the equivalent on whatever loader fits your
#    input), and the streaming behaviour propagates through
#    `bv.qc.gene_set_proportions()` and `bv.pp.find_hvg()`. This is what makes
#    working with this data set on a laptop possible at all.
# 2. **On-disk persistence.** `bv.io.load_existing()` lets you reconnect to a
#    previously streamed data set without re-reading the source. (Works for
#    any data.)
# 3. **Per-sample QC.** With batches comes the need to compute outliers per
#    sample rather than globally.
# 4. **Doublet detection per sample.** Scrublet (or any other doublet
#    detection) can be run with `group_by`, which matters once capture rates
#    differ between batches.
# 5. **GPU-accelerated Scrublet, PCA, kNN and Harmony.** `bixverse_gpu`
#    provides GPU paths for Scrublet, PCA, nearest neighbour search and
#    Harmony. The CPU versions remain highly optimised with SIMD and
#    cache-aware memory layouts, but the GPU paths get faster as the data
#    scales.
#
# For context... The whole workflow here was run on a MacBook Air and did not
# cause any substantial memory pressure.
#
# ## Clean up
#
# ```python
# tempdir_big.cleanup()
# ```
