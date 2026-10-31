# %% [markdown]
# # Technical design choices for the single cell implementation
#
# *Gregor Lueg*
#
# *This is an opinionated piece, so take what is written here with a pinch of
# salt. The statements here reflect my personal opinions, not those of any of my
# past, current, or future employers.*
#
# ## Why do we need yet another single cell framework ... ?
#
# Why the hell would you write a single cell framework from scratch and not use
# one of the established libraries? In Python we have
# [scanpy](https://scanpy.readthedocs.io/en/stable/) on top of
# [AnnData](https://anndata.readthedocs.io/en/stable/), and even
# GPU-accelerated [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/)
# using CUDA and Nvidia GPUs. In R there is
# [Seurat](https://satijalab.org/seurat/) and the BioConductor
# [single cell experiments](https://bioconductor.org/books/release/OSCA/). In
# Rust, we have [SingleRust](https://singlerust.com) which is porting scanpy
# over... What the hell is the point, you might ask? Well, the design
# philosophy in this package is quite different from all of the above. Whether
# better or worse you shall discover, but it *is* different. Let's start with
# the problem though...
#
# ### Chapter 1: Single cell is just not (very) performant (on CPU)...
#
# Some quotes from recent papers:
#
# > We benchmarked rapids-singlecell on 1 million cells from the 10x Genomics
# > mouse brain dataset, running a standard workflow: preprocessing, normalization,
# > HVG selection, dimensionality reduction (PCA, UMAP, t-SNE), neighborhood graph
# > construction, and Leiden clustering (Methods). **On a 32-core workstation,
# > the pipeline took over 52 minutes (Methods)** [...]
# >
# > - [Dicks, et al., arXiv, 2026](https://arxiv.org/abs/2603.02402)
#
# > For instance, a typical single-cell data analysis pipeline, encompassing
# > data integration, clustering, visualisation, and differential expression
# > analysis, requires about 16 h to process half a million cells on a standard
# > desktop. When the cell number slightly increases to 600,000, **the above
# > pipeline can crash due to memory exceeding, even on a professional computing
# > platform with 512 GB RAM**
# >
# > - [Li, et al., Nat Comm, 2025](https://www.nature.com/articles/s41467-025-56424-6)
#
# > <u>Recommendations for executing the Scanpy analysis</u>
# >
# > To avoid computer crashes and program failures, we recommend you execute the
# > Python code on a cloud computing platform. The instance you select on the
# > cloud platform to run this analysis should consist of at **least 8 threads and
# > 160 GB of RAM.**
# >
# > - [Parse biosciences](https://support.parsebiosciences.com/hc/en-us/articles/7704577188500-How-to-analyze-a-1-million-cell-data-set-using-Scanpy-and-Harmony)
#
# Anyone who has worked with more than 500k cells can attest to it being painful.
# Actually VERY painful. The question is... why? Why can analytical engines like
# [DuckDB](https://duckdb.org) and [polars](https://pola.rs) handle massive data
# sets without needing fat memory machines? You can easily analyse hundreds of
# millions of rows on a local laptop with those two. Why can we not do the same
# with single cell? Is the answer really "Bro, just throw more cores, memory,
# and GPUs at the problem"? Or is it time to rethink **HOW** we are doing this.
#
# ### Chapter 2: Do not keep data in memory...
#
# Let's do some maths. Say we have a single cell experiment with 1M cells, with
# on average 1000 genes detected per cell, held the AnnData way: a scipy CSR
# matrix in `X`, `float32` values, `int32` indices.
#
# ```
# # raw counts (X, or layers["counts"])
# data    (values, float32):   1B x 4 bytes = 4 GB
#                  (float64):  1B x 8 bytes = 8 GB
# indices (column ids, int32): 1B x 4 bytes = 4 GB
# indptr  (row pointers):      (1M+1) x 4   ~ 4 MB
#
# Total: ~8 GB (float32) / ~12 GB (float64)
#
# # normalised counts (X after normalise_total + log1p, raw kept in a layer)
# Same sparsity, so another ~8 GB
#
# # scaled counts with 2k HVG (dense; if 3k, just increase this by 50%)
# 2k HVGs:  1M x 2,000 x 4 bytes = 8 GB (float32), 16 GB (float64)
#
# # embeddings (obsm)
# PCA:              1M x 50 x 4 = 200 MB
# Batch correction: 1M x 50 x 4 = 200 MB
#
# Total: ~400 MB
#
# # kNN graph (obsp, k = 20)
# indices (int32):    20M x 4 = 80 MB
# data (float32):     20M x 4 = 80 MB
# indptr (int32):      1M x 4 =  4 MB
#
# Total: ~164 MB, and you usually hold distances and connectivities
# ```
#
# You are looking at 25 GB of occupied memory before anything clever has
# happened, and the moment something upcasts to `float64` that climbs fast. Then
# you have obs as a pandas DataFrame with 1M rows; throw in a bunch of string
# columns and you are adding a few hundred MB more. Subset the AnnData, call
# `.copy()` on the view, and you hold all of it twice. You end up with bloated
# objects storing data you rarely actually need in memory. The question is:
# should we be doing any of this? Why do we keep raw and normalised counts in
# memory at all? AnnData's backed mode, [BPCells](https://github.com/bnprks/BPCells)
# or the [Dask-based approaches](https://scanpy.readthedocs.io/en/1.11.x/tutorials/experimental/dask.html)
# are going in the right direction here, but can we do even better?
#
# ### Chapter 3: Python and R are slow...
#
# There. It has been said. They are slow and let us stop pretending otherwise.
# There is a reason why "Rust rewrite by LLM" is becoming a meme. Yes, you can
# drop into NumPy, numba or Cython for the heavy lifting, but that argument
# rather proves the point! The real problem is not any single kernel; it is what
# happens between them.
#
# A lot of algorithms in practice end up looking like this:
#
# ```
# Python → low-level kernel → Python → low-level kernel → Python → ...
# ```
#
# Each arrow is a context switch. Data gets passed across a language boundary,
# some "simple" bookkeeping happens in the interpreter (a filter here, a reshape
# there, maybe a quick normalisation) and then you hand it back down again. None
# of those individual steps looks expensive. Collectively, they are death by a
# thousand cuts: repeated allocations, temporary arrays you did not ask for, and
# the garbage collector doing God knows what in between. You also lose any hope
# of the compiler reasoning across those boundaries and optimising the full
# pipeline. Writing kernel fusions is maybe possible, but admittedly just easier
# in the low level part directly.
#
# The solution is not to make the interpreted bits faster. It is to keep the data
# in low-level code for as long as possible and only surface results to Python
# when the user actually needs to look at something. Python then becomes a thin
# orchestration layer, not an active participant in the computation.
#
# ## What is your proposal then ... ?
#
# Okay, everything so far reads like a rant. What is the solution, mate?
#
# ### Let's not keep data in memory to start with ... ?
#
# If we think this through, we can arrive at a solution. Ideally, we want an
# interface from an interpreted, dynamically typed language: going all-in on Rust
# is just too heavy. The average single cell enthusiast does not want to deal with
# the borrow checker, does not want to compile code, and wants to quickly analyse
# and iterate over data. The average bioinformatician lives in blissful ignorance
# and will not be able to comment on `f32` vs `f64` float representation in
# memory, when to use which and frankly should not have to. That means the
# interface should be a dynamically typed, interpreted language. This opens the
# package up to more people and makes plotting and the deep learning stack a
# lot easier to reach. From there we can make some design decisions.
#
# 1. **_We do not keep data in memory if not needed!_** Most analyses
# using counts only need subsets of cells and genes. We can also avoid
# holding metadata in memory via DuckDB. Data will be streamed from disk when
# needed and Rust frees memory once it is no longer needed, sometimes enforced
# explicitly with `drop()`. Raw counts? Nope. Normalised counts? Nope. Scaled
# counts? Nope. We just removed the bulk of those 25 GB from memory (in our 1M
# cell example). Data can be loaded into memory in milliseconds if need be.
#
# 2. **_We do not even load all data into memory to begin with._** In
# 99.99% of cases, we know we do not want low quality cells with tiny library
# sizes and few unique features. We also almost always do the usual
# library-size-to-target-size normalisation with log transformation. (Anyone who
# has tried a fancy modelling approach with Pearson residual modelling on large
# data sets knows what I am talking about. But `bixverse` does also offer
# memory-frugal versions of [scTransform v2](https://pmc.ncbi.nlm.nih.gov/articles/PMC8764781/)
# if you want them, see the [residuals vignette](single_cell_residuals.ipynb).)
# That means when reading h5ad or mtx files, we can pre-scan them, take only what
# we want, and normalise in a single pass, avoiding the load-everything,
# then-filter, then-normalise pattern. The data also gets streamed into the
# binary files, which avoids sudden memory spikes and OOM errors.
#
# 3. **_We accept that single cell is noisy_** and do not bother with
# `f32` or even `f64` precision for storage. Raw counts are integers and most of
# them are tiny, so on disk they get bit-packed to whatever width the block
# actually needs. A block of ones and twos costs a single bit per value, the rare
# gene with a count in the thousands only widens its own block, and nothing gets
# clipped. Normalised counts live at `f16` precision: the ML field has shown you
# can get away with it, and they are pushing quantisation even further than
# that. We accept the precision loss and upcast to `f64` where it matters, to
# avoid catastrophic cancellation issues at large N.
#
# 4. **_We are not doing any heavy calculations in Python._** Rust is superior
# for speed and optimisation: aggressive multi-threading, no interpreter in the
# hot loop, tight memory control, and SIMD acceleration for specific bottlenecks
# (distance calculations, averaging, summing, normalising). Python here is just
# an interface, telling the underlying Rust code what to do, and the GIL is
# released while Rust works.
#
# But if the data is not in memory, will this not be slow? Well, no. We store data
# in structures like this. One for cells:
#
# ```rust
# /// CsrCellChunk
# ///
# /// This structure is designed to store the data of a single cell in a
# /// CSR-like format optimised for rapid access on disk.
# #[derive(Debug)]
# pub struct CsrCellChunk {
#     pub data_raw: RawCounts,
#     pub data_norm: Vec<F16>,
#     pub library_size: usize,
#     pub indices: Vec<u32>,
#     pub original_index: usize,
#     pub to_keep: bool,
# }
# ```
#
# One for genes (if you are even vaguely familiar with compressed sparse formats,
# you have probably already clocked that this is essentially CSR and CSC storage
# with two data layers):
#
# ```rust
# /// CscGeneChunk
# ///
# /// This structure is designed to store the data of a single gene in a
# /// CSC-like format optimised for rapid access on disk.
# #[derive(Encode, Decode, Serialize, Deserialize, Debug)]
# pub struct CscGeneChunk {
#     pub data_raw: RawCounts,
#     pub data_norm: Vec<F16>,
#     pub avg_exp: F16,
#     pub nnz: usize,
#     pub indices: Vec<u32>,
#     pub original_index: usize,
#     pub to_keep: bool,
# }
# ```
#
# But you are duplicating the data! Yep, indeed. But this allows extremely
# efficient retrieval. Want to run DGE between two groups? Only those cells are
# loaded via `CsrCellChunk`s. Want to iterate over genes to identify HVGs? That
# goes through `CscGeneChunk`. A further advantage is that for quite a few
# analyses, we can load subsets and stream over the data rather than reading
# everything at once, which massively reduces the memory footprint. The core
# philosophy is:
#
# > **Disk space is cheap; memory ain't**
#
# The two-layer approach (heavily inspired by TileDB) means raw and normalised
# counts share one set of indices instead of each dragging their own. Every cell
# and every gene is its own independently compressed chunk, the file tail holds
# the offsets, and reads go through `mmap`. This assumes decent SSD speeds, but
# most modern hardware has that covered.
#
# Here is what that looks like on disk for PBMC3k. One directory, two binary
# stores (cell-major and gene-major) and a DuckDB with obs and var:

# %%
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

import bixverse as bv
from bixverse.params import NeighboursParams

pbmc3k_path = bv.io.download_pbmc3k()

tempdir_design = tempfile.TemporaryDirectory()

sc = bv.io.load_mtx(
    Path(tempdir_design.name) / "pbmc3k",
    bv.io.get_cell_ranger_params(pbmc3k_path),
)

files = [*sorted(sc.dir.iterdir()), Path(pbmc3k_path) / "matrix.mtx"]

pl.DataFrame(
    {
        "file": [f.name for f in files],
        "size_mb": [round(f.stat().st_size / 1e6, 2) for f in files],
    }
)

# %% [markdown]
# Both stores together still come in well under the plain text `matrix.mtx`
# they were built from (which, to be fair, also holds the genes and cells that
# the loader's QC dropped). `bv.CellReader` is the raw handle on the cell-major
# store: it reads any set of cells into CSR in one call, decoded in Rust with
# the GIL released, either layer.

# %%
reader = bv.CellReader(str(sc.cells_path))

print(
    f"cell based: {reader.is_cell_based}, cells: {reader.total_cells}, "
    f"genes: {reader.total_genes}, target size: {reader.target_size}"
)

cells = np.array([0, 1, 2], dtype=np.uint32)

raw, indices, indptr, shape = reader.read_csr(cells, "raw")
norm, _, _, _ = reader.read_csr(cells, "norm")

shape

# %% [markdown]
# ### How does this compare to AnnData ... ?
#
# If you come from scanpy, the mental model shifts quite a bit. An `AnnData`
# holds `X` (dense or scipy sparse), any `layers`, and obs and var as pandas
# DataFrames in memory; everything you compute lands in `obsm`, `obsp`, `uns`
# and friends on the same object. Backed mode (`backed="r"`) leaves `X` in the
# h5ad file and reads it on access, but obs and var still sit in memory, and
# `X` stays in the one orientation it was written in.
#
# `bixverse` flips that around. The counts stay on disk, twice, once per
# orientation, and only ever get streamed through Rust. obs and var live in the
# DuckDB; ask for them and you get a polars frame back, only the columns you
# asked for if you like. What sits in Python memory is the light stuff: the
# kept-cell indices, the HVG selection and the cache of results (PCA, kNN,
# embeddings). Counts only reach Python when you ask for them.

# %%
print(type(sc.obs()))

sc.obs(cols=["cell_id", "lib_size", "nnz"]).head(3)

# %% [markdown]
# #### The v4 file format
#
# Cheap disk space is no excuse to waste it though. The first versions of the
# format were lazy: absolute `u32` indices, `u16` raw counts, the full `f16`
# normalised layer, all thrown at [lz4](https://en.wikipedia.org/wiki/LZ4_(compression_algorithm)).
# lz4 is fast, but it cannot do much with a sorted list of absolute indices.
# Version 4 (the default for everything the package writes now) looks at what
# a chunk actually holds:
#
# - **Indices are delta bit-packed** in blocks of 128. Sorted gene (or cell)
# indices turn into small gaps, and each block is packed to the width of its
# largest gap.
# - **Raw counts are bit-packed the same way**, stored as `count - 1` since a
# stored entry is never zero. That saves another bit per value, and in UMI data
# that bit matters.
# - **The normalised layer is dropped.** Library-size normalisation plus `log1p`
# is a pure function of the raw count, the library size and the target size, so
# we recompute it on read, bit for bit identical to what used to be stored. A
# chunk where the recompute does not reproduce the stored values keeps them
# verbatim, so nothing changes under your feet. Gene chunks need the library size
# of each cell for this, so the gene file carries a per-cell library-size table
# in its tail, plus a small lookup table for counts 1 to 4, which covers the
# common low counts without a single `ln_1p` call.
# - **A fixed 32 byte header sits uncompressed** in front of every chunk: nnz,
# library size (or average expression for genes), original index, flags and the
# codec. Things like library sizes can be read without decompressing anything.
# - **The codec is per chunk:** lz4 by default, zstd if you want smaller files
# and can live with slower decoding, or none. A chunk that does not shrink is
# stored raw whatever you asked for.
#
# So in practice we store raw counts and only raw counts, wherever possible, and
# the normalised layer is computed on the fly. Reads also prefetch their byte
# range into the page cache, so cold reads do not crawl through one page fault at
# a time.
#
# You can check the recompute yourself: `log1p(count / library size x target
# size)`, rounded to `f16`, is exactly the norm layer the reader hands back.

# %%
lib_size = sc.obs(cols=["lib_size"])["lib_size"].to_numpy()[cells]
row = np.repeat(np.arange(len(cells)), np.diff(indptr.astype(np.int64)))

recomputed = np.log1p(raw / lib_size[row] * reader.target_size).astype(np.float16)

assert np.array_equal(recomputed.astype(np.float32), norm)

# %% [markdown]
# Got stores from an older version of the package? They still open and work, you
# just get a warning from `bv.io.load_existing()` nudging you to convert them.
# `bv.io.upgrade_sc_store()` does that in place: it writes the v4 files next to
# the old ones, checks them, and only then swaps them in, keeping `.v3.bak`
# copies unless you tell it otherwise.
#
# ```python
# sc = bv.io.load_existing("path/to/old_store")
# sc = bv.io.upgrade_sc_store(sc, codec="zstd", keep_backup=False)
# ```
#
# Once you are done with the analysis, `bv.io.archive_sc_exp()` archives the
# binary files, squeezing the disk footprint further, and `load_existing()`
# restores them from the archive when it finds only that. The idea is to use
# your disk space temporarily, stream data into the analysis methods in the
# right shape and when done, compress everything down. The store format is the
# same one the R package writes, so a directory written by one opens in the
# other.
#
# ### Let's not round trip into interpreted languages
#
# Continuing from step 4, we avoid round trips to Python entirely. The core
# philosophy is to take a hard look at the algorithms and methods used and,
# where needed, implement a highly specialised version for single cell. Some
# examples:
#
# - Custom implementation of various kNN searches, all running in Rust:
# [ann-search-rs](https://crates.io/crates/ann-search-rs). No on-disk index round
# trips, SIMD-accelerated distance calculations, optimised memory layouts,
# aggressive optimisations with single cell in mind. Since kNN graphs underlie a
# lot of downstream methods, speeding this up has a wide impact. The
# [kNN vignette](knn_searches.ipynb) has the options.
#
# - PCA underlies a lot of algorithms, is a key dimensionality reduction step,
# and, perhaps embarrassingly, [still beats fancy foundation models](https://arxiv.org/abs/2410.13956).
# A few tricks make it fast:
#
#   * [Lanczos](https://en.wikipedia.org/wiki/Lanczos_algorithm) on the sparse
#   matrix. We only want the top 30 to 50 components, so there is no point in
#   decomposing the whole thing. Lanczos runs on the `n HVG x n HVG` Gram matrix
#   without ever building it: all it needs are sparse matrix-vector products,
#   which is exactly what CSR and CSC storage is good at. It is thick-restarted,
#   so the trailing components converge too, even on the flat spectrum single
#   cell data likes to produce. And it is exact, not an approximation. This is
#   the default now. [Randomised SVD](https://arxiv.org/abs/0909.4061) is still
#   there if you want it, but it is approximate and the trailing components
#   suffer the most. On the dense path, the default eigendecomposes the
#   `n HVG x n HVG` cross-product instead.
#   * Smart scaling without ever densifying the matrix. As shown in the numbers
#   above, densifying with 2k HVGs means holding an 8 GB (`float32`) matrix in
#   memory for a million cells, 40 GB for five million. Centring and scaling get
#   folded into the matrix-vector products, so you never have to densify at all.
#
# - Of course there is also a library for everyone's favourite 2D embedding
#   generation: [manifolds-rs](https://github.com/GregorLueg/manifolds-rs), behind
#   the `embed` extra. Same philosophy: parallelise what can be parallelised, use
#   CPU-friendly memory layouts, avoid round trips to the interpreter.
#   Million-cell UMAP? Done in minutes on a laptop. Even good old tSNE, infamous
#   for being slow, has two different FFT-accelerated versions (see
#   [Linderman, et al.](https://www.nature.com/articles/s41592-018-0308-4) for the
#   OG variant). The "quick and dirty" Barnes-Hut approximation is provided too.
#
# - [SCENIC](https://pubmed.ncbi.nlm.nih.gov/32561888/) infers gene regulatory
#   networks by asking, for each target gene: which transcription factors best
#   predict its expression? It does this by training a tree ensemble per gene,
#   using the importance scores from those trees as a proxy for regulatory
#   influence. The trees themselves are then thrown away. Only the importance
#   scores matter. That means we can optimise purely for speed, without caring
#   about prediction quality at all. A few tricks compound here:
#
#   * Rather than training one ensemble per gene independently, we share the same
#     tree structure across a batch of genes simultaneously: the splits are chosen
#     to reduce prediction error across all targets in the batch at once, and each
#     gene contributes its own importance scores as a side effect. This amortises
#     most of the cost.
#   * Transcription factor expression data is quantised down to 256 bins (one byte
#     per cell), which makes histogram construction and split evaluation very fast
#     and cache-friendly.
#   * Single cell data is sparse, most genes are zero in most cells, so rather
#     than densifying the target matrix (which would be enormous), we exploit that
#     sparsity directly during tree building.
#
#   The net result is GRN inference that runs in Rust end-to-end, with no data
#   leaving low-level code until results are ready.
#
# These are just some examples where purpose-built code can deliver massive gains
# in both memory usage and speed. This allows the package to run multi-million
# cell analyses (the GRNs will still be slow and take hours despite my best
# efforts, sorry) on a decent local laptop. I have churned through 3 million
# cell data sets on my loyal M1 Max MacBook Pro with 64 GB, doing the reading
# from h5ad, mt percentage detection, some cell filtering, HVG detection, PCA,
# kNN/sNN graph generation + clustering in <30 minutes. Something that would be
# completely impossible with several of the other data-held-in-memory libraries
# and approaches.
#
# ## There is no free lunch
#
# While this all sounds great, one significant caveat is that using this package
# forces you to think differently about your data. Because data is NOT in memory,
# we need to carefully synchronise state between:
#
# - The binary files Rust writes to disk with specific indexing for cells and
# genes. At least Python and Rust agree on 0-indexing, which is one headache the
# R side has and you do not.
#
# - What is stored in DuckDB as metadata. This can get tricky once you start
# selecting subgroups of cells. In a standard workflow you might:
#
#   1. Filter a first wave of cells/spots based on library size and number of
#      features.
#   2. Run doublet detection to remove doublets.
#   3. Recheck library sizes, feature counts, transcriptional complexity,
#   mitochondrial counts, and only then finalise your cell selection with
#   `sc.set_cells_to_keep()`.
#
#   Any subsequent analysis (HVG detection, PCA, kNN construction, etc.) will
#   operate on whatever is set here. This used to be THE footgun of the package.
#   Move the cell set, forget to re-run the kNN, and you had a PCA over one set
#   of cells sitting next to a kNN over another. Nothing complained. It either
#   crashed deep in Rust or, much worse, quietly handed you mis-aligned biology.
#
# - The "ecosystem" (quotation marks because at the moment it is just the R
# package, this Python package and several Rust crates) is in its
# ~~infancy~~puberty. Stuff will bug, break, breaking changes have been and will
# likely continue to be introduced. If you want to contribute at the methods
# level, you will need to go low-level and learn Rust (or ask your favourite LLM.
# But please no obvious AI slop.) The Rust code powering all of this is
# [here](https://crates.io/crates/bixverse-rs).
#
# This is a side/hobby project, mostly a labour of love. I do hope other people
# find it useful and at least rethink some of their assumptions.
#
# ### The cache checks
#
# So the cell set footgun now complains. Every artefact that lands in the
# cache, the PCA, the embeddings, the kNN, the neighbour graph, the MAGIC layer,
# carries a provenance stamp: a hash of the cell set it was computed on, plus
# the ids of whatever it was derived from. The kNN records the PCA it was built
# on and the graph records the kNN, which is the bit that matters. Re-run the
# PCA on the **same** cells with fewer components and the kNN hanging off it is
# now stale, even though its own cell count never moved. A row-count check
# waves that straight through.
#
# Here it is on PBMC3k. HVGs, a 30 component PCA and the neighbours, then
# `cache_status()`:

# %%
bv.pp.find_hvg(sc, hvg_no=2000)

bv.pp.calculate_pca(sc, no_pcs=30, sparse_svd=True)

bv.graph.find_neighbours(
    sc, neighbours_params=NeighboursParams(knn_method="exhaustive")
)

sc.cache_status().select("artefact", "stale", "id", "parents")

# %% [markdown]
# Now the PCA again, same cells, 15 components. The kNN still has the right
# number of rows, but its parent is gone:

# %%
bv.pp.calculate_pca(sc, no_pcs=15, sparse_svd=True)

pl.Config.set_fmt_str_lengths(80)

sc.cache_status().select("artefact", "stale", "reason")

# %% [markdown]
# Two things worth saying about how it is built. The stamps sit on the cache
# next to the payloads, and assigning a new payload to a slot drops its stamp,
# so a stamp cannot outlive the result it describes. And it tracks provenance,
# not parameters: a kNN over the first 5 PCs and one over all 15 of the same
# PCA look identical to it. Neither of those can produce an index mismatch,
# which is the only thing this is here to stop.
#
# The response is graded rather than one blanket error. Moving the cell set
# with `set_cells_to_keep()` warns and names everything that just went stale,
# and so does reading a stale artefact through the getters, because plenty of
# call sites use those as presence probes on something they are about to
# overwrite anyway. Anything that hands cached indices to Rust or writes a
# derived result back raises outright. Clustering on the stale graph from
# above:

# %%
try:
    bv.graph.find_clusters(sc, res=1.0, name="leiden_clusters")
except ValueError as e:
    print(e)

# %% [markdown]
# Nothing is ever silently recomputed or binned: a forty minute Harmony run is
# not the package's to throw away. Compare that with AnnData, where `obsm` and
# `obsp` are aligned to the object's rows and subsetting slices them along, so
# the rows always line up (though a PCA sliced down to a subset is still the
# PCA of the full set). Here the results and the cell set live apart, and the
# stamps are the price of that.
#
# None of which means you get to stop thinking about on-disk state. It only
# means you find out before the biology is wrong rather than after. The worked
# example lives in the [thinking about single cells](thinking_single_cell.ipynb)
# vignette.
#
# ## Future directions
#
# A fair amount of the original roadmap has landed since this was first written.
# Trajectory inference is in ([Palantir](https://github.com/dpeerlab/Palantir)
# from [Setty, et al.](https://www.nature.com/articles/s41587-019-0068-4), PAGA
# and gene trends along pseudotime), so is a decent spread of batch
# correction (Harmony, Harmony v2, fastMNN, BBKNN, Seurat CCA and rPCA, with
# kBET, ASW and LISI to score whether any of it worked), differential expression
# and abundance (Wilcoxon markers, pseudobulk into limma-voom and edgeR
# quasi-likelihood, NEBULA, Milo, MELD), the full SCENIC stack with cisTarget,
# meta cells via SEACells and SuperCell, DIALOGUE, Hotspot, VISION, NMF,
# multi-modal CITE-seq with WNN graphs, and Symphony reference mapping. The
# plots live in `bv.pl`, behind the `plots` extra.
#
# What is still ahead:
#
# - **More methods:** the approach has always been to implement what I needed or
# found interesting, and that will not change. [Slingshot](https://pubmed.ncbi.nlm.nih.gov/29914354/)
# never got done and is still on the list.
#
# - **Spatial transcriptomics:** Spatial data does not (yet) reach the scale of
# the largest single cell data sets most of the time. However, if you are
# analysing many Visium or similar data sets, it can become quite heavy, and a
# lot of current frameworks are built on top of single cell ones anyway. There
# are also some interesting methods that use the spatial grid information
# directly, which I find genuinely exciting. None of it exists yet though, and
# to be clear about what does: the graph autocorrelation in VISION and Hotspot
# runs over the kNN graph, not over tissue coordinates. There are some prototype
# branches the keen observer might have seen already. Watch the space.
#
# - **GPU acceleration:** GPUs are becoming increasingly accessible, and if you
# have the budget for data-centre-scale GPUs, please do use them. Nvidia knows
# what they are doing and cuBLAS is insanely fast, in ways that simply cannot be
# matched on CPUs. However, most people do not have several B200s lying around at
# home. So the GPU work goes through
# [cubecl](https://github.com/tracel-ai/cubecl) on the wgpu backend, which means
# Metal on macOS, Vulkan on Linux and DX12 on Windows off one set of kernels
# rather than a CUDA-only path. (A native CUDA backend is aspirational. I do not
# own an Nvidia GPU to test it on.) In Python it lives in the `bixverse_gpu`
# companion package, and there is a fair amount of it now: block Lanczos (or
# randomised SVD) for the PCA, Harmony v2 with the arrowhead inversion, kNN
# across exhaustive, IVF and NNDescent-into-CAGRA, k-means coarsened fast
# clustering, the UMAP Adam optimiser, SEACells (both Frank-Wolfe solves),
# Scrublet, the NMF HALS family, BBKNN and fastMNN with GPU-accelerated kNN
# searches, SCENIC's ExtraTrees and RandomForest learners, and Pearson/Spearman
# correlation.
#
# - **Deep learning integration:** this is the reason the Python package exists
# at all. It opens the same binary files and streams shuffled minibatches out of
# them through `bv.CellLoader`, raw, normalised or both, sparse by default.
# Decoding never leaves Rust and the GIL is released while it happens, so the
# loader keeps a GPU training loop fed without ever materialising the
# experiment in memory. The first consumer is a barebone scVI in `bixverse.nn`
# on torch, MLX or JAX, with the scvi-tools 1.5.1 defaults: load a trained
# scvi-tools module into it and you get the same loss back. On two PBMC batches
# (5k cells, M1 Max), the MLX version trains in about 30 seconds against just
# under three minutes for scvi-tools on MPS, at the same held-out ELBO.
# Toy-sized, sure, and the million-cell runs are still to come. Watch this
# space.
#
# ## Clean up

# %%
tempdir_design.cleanup()
