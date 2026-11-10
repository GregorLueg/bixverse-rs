# %% [markdown]
# # Thinking about single cells the bixverse way
#
# *Python port of the R vignette [Thinking about single cells the bixverse way](https://gregorlueg.github.io/bixverse/articles/thinking_single_cell.html). Original authors: Gregor Lueg and Grant Neilson.*
#
# ## Intro
#
# Before diving into this vignette, it would be really recommended to read the
# [design choices](design_single_cell.ipynb) to understand **why** single cell
# has been implemented the way it is in this package. This will make
# understanding the downstream text much easier. The aim of this vignette is to
# take some small synthetic data and point out some quirks due to design choices
# made.

# %%
import tempfile
import timeit
from pathlib import Path

import polars as pl

import bixverse as bv
from bixverse.params import MinQualityParams

# %% [markdown]
# ### Synthetic data
#
# Let's start with some synthetic data that is powering all the tests in the
# package. It's very simple data with 1,000 cells, 100 genes and three very
# clearly defined cell groups. Genes 1 to 10 define cell_type 1, genes 11 to 20
# the next cell type and 21 to 30 the last cell type. The remaining genes are
# noise genes.

# %%
single_cell_test_data = bv.io.generate_single_cell_test_data()

print(f"counts: {single_cell_test_data.counts.shape} CSR matrix")

single_cell_test_data.obs.head()

# %%
single_cell_test_data.var.head()

# %% [markdown]
# We have a count matrix with pseudo raw counts (cells x genes, CSR), an obs
# table and a var table. Let us load this into a `bixverse` `SingleCells`
# experiment and explore what's actually happening.
#
# ### Loading in data
#
# An experiment is a directory. If you read the design docs, you know why you
# have to provide a path here. If you do not know why, go back to the docs. We
# point it at a temporary directory that is still empty:

# %%
tempdir_sc = tempfile.TemporaryDirectory()
test_dir = Path(tempdir_sc.name) / "single_cell_test"

test_dir.exists()

# %% [markdown]
# Nope, nothing there. Let's now load in the data and see what happens:

# %%
sc = bv.io.load_csr(
    test_dir,
    single_cell_test_data.counts,
    single_cell_test_data.obs,
    single_cell_test_data.var,
    # we let all cells pass for now
    qc=MinQualityParams(
        min_unique_genes=0,
        min_lib_size=0,
        min_cells=0,
        target_size=1e3,  # target size of 1000
    ),
    verbose=True,
)

sorted(p.name for p in test_dir.iterdir())

# %% [markdown]
# So, what has just happened ... ? Actually quite a lot of things:
#
# 1. The matrix is being pulled into Rust and checked against the number of
#    `min_unique_genes`, the `min_lib_size` and in how many cells a given gene
#    needs to be expressed to be included. Once the genes and cells have been
#    identified that pass the minimum threshold (in that order: first gene
#    filtering, then check the cells against that reduced feature space), the
#    cells are written to the first binary file: `counts_cells.bin`. At the same
#    time the counts are normalised: division by library size, projected against
#    desired target size and log transformed.
# 2. Rust now transposes under the hood the data that is at the moment very nice
#    to read in from a cell-centric level, less so from a gene-centric level.
#    That is why we also have `counts_genes.bin`.
# 3. Lastly, the observation and variable data is being reduced based on step 1
#    and written to the `sc_duckdb.db` and stored there.
#
# `load_csr` hands back the opened experiment. If we print it:

# %%
sc

# %% [markdown]
# We can now see the number of cells and genes stored in the experiment. Nothing
# but the indices and ids lives in memory; the counts stay on disk. A directory
# written earlier (by Python or by the R package) opens again with
# `bv.io.load_existing(test_dir)`.
#
# ### Interacting with the data
#
# Let's now look into how we interact with the data... We will actually apply
# some filters here (the synthetic data is weird, but does the job of assumption
# testing).

# %%
sc = bv.io.load_csr(
    test_dir,
    single_cell_test_data.counts,
    single_cell_test_data.obs,
    single_cell_test_data.var,
    qc=MinQualityParams(
        min_unique_genes=45,
        min_lib_size=300,
        min_cells=500,  # the data is weird and a lot cells express genes
        target_size=1e3,  # target size of 1000
    ),
    verbose=True,
)

sc

# %% [markdown]
# We now have overwritten the files on disk with the filtered versions.

# %%
print(f"{sc.dims[0]} cells and {sc.dims[1]} genes pass the threshold")

# %% [markdown]
# But how do I interact with the data?
#
# #### Understanding mapping and filtering
#
# As data is stored on disk, we need to carefully synchronise state here between
# the DuckDB, the binary files and tell the Rust kernel which cells to use and
# what to do... There is a bunch of helpers for this. You might have spotted the
# `kept` count in the print. This is very important! Let's check this out:

# %%
sc.cells_to_keep[:10]

# %% [markdown]
# What is cells to keep? Well, these are the (0-based) indices of cells that
# shall be included in the analysis, think HVG selection, PCA, kNN generation,
# etc. This used to be **THE** foot gun of this package, and it is still the
# thing to keep in your head. The `cells_to_keep` indices will be the ones
# loaded in by Rust. The other cells are not lost, they are still on disk (a
# massive difference between filtering in memory and on-disk work!). Also the
# DuckDB state will get updated, but let's get to this later.
#
# Every index you hand to or get back from the package is a 0-based position in
# the binary store: `cells_to_keep`, the `cell_idx` column of obs, the
# `gene_idx` column of var and the `cells` / `genes` arguments of the count
# getter. There are also getters for the identifiers.

# %%
# the cell identifiers, only the kept cells; without filtered=True you get all
print(sc.cell_ids(filtered=True)[:10])

print(sc.gene_ids()[:10])

# %% [markdown]
# Let's quickly check out how you access the obs table and then let's
# exemplify some quirks with data being held on disk. `sc.obs()` reads the
# table out of the DuckDB into a polars frame:

# %%
sc.obs()

# %%
# only specific rows, by 0-based cell index
sc.obs(indices=[0, 1, 2])

# %%
# for a quick glimpse in the obs metadata
sc.obs().head()

# %%
# specific columns
sc.obs(cols=["cell_idx", "cell_id", "cell_grp"])

# %% [markdown]
# Let's pretend we want to do mitochondrial (or other gene set-based) removal.
# `gene_set_proportions` is like every function in the package that adds
# something to the experiment: it returns `None` and writes into `sc`, here a
# new obs column per gene set straight into the DuckDB.

# %%
# pseudo gene sets
gs_of_interest = {
    "gs_1": ["gene_001", "gene_002", "gene_003", "gene_004"],
    "gs_2": ["gene_096", "gene_097", "gene_100"],
}

# calculate the gene set proportions
bv.qc.gene_set_proportions(sc, gs_of_interest, verbose=True)

# %% [markdown]
# If we now look at the obs again:

# %%
sc.obs().head()

# %% [markdown]
# We now have two new columns here! That worked well... Your own columns go in
# the same way with `sc.add_obs()`, joined on `cell_idx`. Let's now do filtering
# to really understand what's happening under the hood:

# %%
threshold = 0.05

cells_to_keep = (
    sc.obs(filtered=True).filter(pl.col("gs_2") < threshold)["cell_id"].to_numpy()
)

sc.set_cells_to_keep(cells_to_keep)

sc

# %% [markdown]
# The print changed now! We have now told the experiment that only a subset of
# 682 cells shall be used in any downstream analyses. Understanding this is
# **very important**. Most functions will only return and work on the filtered
# cells, and you have to carefully think and remember what you did when.
# `set_cells_to_keep` takes 0-based indices or cell ids, as a numpy array or a
# list; hand it a polars Series and it complains, hence the `.to_numpy()`.
#
# Here is the Python-specific foot gun: `sc.obs()` returns **all** cells in the
# store, kept or not. The `to_keep` column tells you which is which, and
# `filtered=True` gives you only the kept ones.

# %%
obs_unfiltered = sc.obs()

obs_filtered = sc.obs(filtered=True)

print(f"No of rows unfiltered: {obs_unfiltered.height}")
print(f"No of rows filtered: {obs_filtered.height}")
print(f"No of rows with to_keep set: {obs_unfiltered['to_keep'].sum()}")

# %% [markdown]
# #### The cache knows when you've moved the goalposts
#
# Here is where the foot gun used to go off. Anything cached on the experiment,
# the PCA, the embeddings, the kNN, the neighbour graph, was computed against a
# particular set of cells. Change the cell set and those artefacts describe a
# world that no longer exists. Re-run the PCA, forget the kNN, and you'd end up
# with a PCA over 682 cells sitting next to a kNN over 929. Nothing complained.
# It either crashed somewhere deep in Rust or, worse, quietly gave you
# mis-aligned biology.
#
# The cache lives in memory on `sc.cache`; `bv.io.save(sc)` writes it next to the
# stores so `load_existing` brings it back. Every cached artefact carries a
# provenance stamp: a hash of the cell set it was computed on, plus pointers to
# whatever it was derived from. Let's build a small cache and look at it. Note
# that none of these return anything; they write into `sc`.

# %%
bv.pp.find_hvg(sc, hvg_no=30)
bv.pp.calculate_pca(sc, 10)
bv.graph.find_neighbours(sc)

sc.cache_status()

# %% [markdown]
# The `parents` column is what makes this more than a cell counter. The kNN
# records the id of the PCA it was built on, and the neighbour graph
# (`connectivities`, the sNN by default) records the kNN. So watch what happens
# if we re-run the PCA on the **same** cells, just with fewer components, and
# don't touch the neighbours:

# %%
bv.pp.calculate_pca(sc, 5)

sc.cache_status().select("artefact", "stale", "reason")

# %% [markdown]
# The PCA is fine. The kNN is stale because the PCA it points at no longer
# exists, and the neighbour graph is stale through the kNN, even though its own
# cell set never moved. A row-count check would have waved both of those
# through. The print says so too:

# %%
sc

# %% [markdown]
# `sc.check_state()` asks the same question for the artefacts you name, and
# returns `True` or the reason:

# %%
print(sc.check_state(["pca"]))
print(sc.check_state(["connectivities"]))

# %% [markdown]
# Functions that only read a cached artefact for a metric warn on a stale one.
# Anything that hands cached indices to Rust or writes a result back onto the
# experiment, `find_neighbours`, `find_clusters`, the meta cell generators,
# errors outright rather than computing nonsense:

# %%
try:
    bv.graph.find_clusters(sc)
except ValueError as e:
    print(f"ValueError: {e}")

# %% [markdown]
# Regenerating the neighbours puts the chain back together:

# %%
bv.graph.find_neighbours(sc)

sc.cache_status().select("artefact", "stale", "reason")

# %% [markdown]
# `set_cells_to_keep()` warns about exactly the same thing, listing what it just
# invalidated. There is no global switch to turn the checks off or into errors.
#
# Two things it deliberately does not do. Artefacts that carry no stamp (say,
# you assigned `sc.cache.pca_factors` yourself, which drops the stamp) report
# as unstamped and pass, because nothing is known about them either way. And the
# stamp tracks provenance, not parameters: a kNN built on the first 5 PCs and
# one built on all 10 of the same PCA look identical to it. Neither of those can
# produce an index mismatch, which is what this is here to stop.
#
# If you want to go back to a clean slate, `reset_cells_to_keep()` restores
# every cell in the binary file and wipes the cache and the HVG selection. A
# PCA that took forty minutes is not the package's to bin, so with anything in
# the cache it raises unless you pass `force=True`:
#
# ```python
# sc.reset_cells_to_keep(force=True)
# ```
#
# Filtering never deleted anything, it only ever flipped a `to_keep` flag in the
# DuckDB, so nothing is lost by resetting. The cache has to go though: a PCA
# computed on 682 cells does not describe 929 of them.
#
# If you understand the concept shown above, you are good to go to analyse
# millions of cells on small compute without having 100's of GB of memory
# available. Let's just talk also about the 2nd big foot gun: **the counts**.
#
# #### Understanding count retrieval
#
# You have actually two files on disk... If you read the design doc you know
# why (if you don't, please read it!). The gene-major file serves the
# gene-centric work under the hood (HVG selection, for example). When you pull
# counts yourself, `bv.get_counts()` reads the cell-major file and hands back a
# `Counts`: the CSR matrix plus the 0-based store indices of its rows and
# columns. Let's say you are interested in the first 50 cells.

# %%
raw_counts = bv.get_counts(sc, cells=range(50))

print(raw_counts.matrix.shape)

csr = raw_counts.scipy_csr()

print(type(csr))

# %% [markdown]
# Fewer than 50 rows? `get_counts` respects the cells to keep by default, so of
# the first 50 store cells you only get the kept ones back (more on that below).
# This is the smart way: rows are cells, so Rust ONLY loads the cells that are
# needed. The stupid way is to pull everything and slice afterwards. Timing
# both on this toy data set:

# %%
n_rep = 10

the_correct_way = timeit.timeit(
    lambda: bv.get_counts(sc, cells=range(50)).scipy_csr(), number=n_rep
)
the_incorrect_way = timeit.timeit(
    lambda: bv.get_counts(sc).scipy_csr()[:50], number=n_rep
)

print(f"the_correct_way: {the_correct_way / n_rep * 1e3:.2f} ms")
print(f"the_incorrect_way: {the_incorrect_way / n_rep * 1e3:.2f} ms")

# %% [markdown]
# The difference seems marginal here, but it WILL bite you if you do this on
# large data sets. Also, the memory advantages this approach gives you will
# disappear if you load everything to throw most of it away. Also, another thing
# to REALLY consider... The indices you supply should match what the binary
# files have (0-based!) and again you need to THINK about the cells to keep. A
# nasty example here:

# %%
counts_filtered = bv.get_counts(
    sc,
    # cells 0 to 9 as they were written to disk
    cells=range(10),
    # genes 0 to 9 as they were written to disk
    genes=range(10),
    use_cells_to_keep=True,
)

counts_unfiltered = bv.get_counts(
    sc,
    cells=range(10),
    genes=range(10),
    use_cells_to_keep=False,
)

for label, counts in (("filtered", counts_filtered), ("unfiltered", counts_unfiltered)):
    n_cells, n_genes = counts.matrix.shape
    print(f"Number of cells: {n_cells} | Number of genes: {n_genes} ({label} version)")

print(f"cells returned with use_cells_to_keep=True: {counts_filtered.cells}")

# %% [markdown]
# Wait, what happened here ... ? Because we set the cells to keep in the step
# with the pseudo mitochondrial counts, some of these indices are now "bad
# cells". **They are not filtered, they are just not returned.** The `cells`
# attribute of the result tells you which store cells you actually got.
#
# If you want to avoid thinking about indices altogether (which is
# recommended), just use the identifiers:

# %%
counts_by_id = bv.get_counts(
    sc,
    cells=["cell_0001", "cell_0003", "cell_0100", "cell_0117"],
    genes=["gene_001", "gene_002", "gene_003", "gene_081"],
)

print(f"cells: {sc.cell_ids(counts_by_id.cells)}")
print(f"genes: {sc.gene_ids(counts_by_id.genes)}")

counts_by_id.scipy_csr().toarray()

# %% [markdown]
# And that's it. If you understand this vignette and the design choices, we have
# answered the *why*, the *how* and can move on to the *what* in the other
# vignettes with some example analyses and how to actually use the package to
# analyse single cell experiments.
#
# ## Clean up

# %%
tempdir_sc.cleanup()
