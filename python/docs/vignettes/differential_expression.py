# %% [markdown]
# # Differential expression with donor structure
#
# *Python port of the R vignette [Differential expression with donor structure](https://gregorlueg.github.io/bixverse/articles/differential_expression.html). Original author: Gregor Lueg.*
#
# ## Intro
#
# You have 3,000 monocytes from eight donors and you want to know which genes
# respond to a treatment. The obvious move is a Wilcoxon test over cells. It is
# also the wrong one, and the reason is worth being precise about.
#
# Those 3,000 cells are not 3,000 independent observations. They came from eight
# people. Cells from the same donor are correlated, because donors differ in
# genotype, in batch, in how the sample was handled, in everything. A test that
# treats each cell as independent has an effective sample size of eight and
# thinks it has 3,000, so its p-values are far too small. This is
# pseudoreplication, and in single cell data it is the default failure mode
# rather than an exotic one.
#
# There are two honest ways out.
#
# **Pseudobulk.** Sum the counts per donor per condition and run a bulk method.
# The unit of replication becomes the donor, which is correct. You throw away
# everything the cell-level data told you about variability within a donor, and
# with few donors you end up with very little power.
#
# **A mixed model.** Keep the cells, but model the donor as a random effect.
# [NEBULA](https://doi.org/10.1038/s42003-021-02146-6) fits a negative binomial
# gamma mixed model per gene, splitting the variance into a subject-level term
# and a cell-level term. You keep the cell-level information and still pay the
# right price for the donor structure.
#
# `bixverse` gives you all three, and this vignette runs them side by side, first
# on a real effect and then on no effect at all. The second one is the
# interesting bit.

# %%
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

import bixverse as bv
from bixverse.params import EdgerQlParams, MinQualityParams, NebulaParams

# %% [markdown]
# ## The data
#
# [Kang, et al.](https://doi.org/10.1038/nbt.4042) pooled PBMCs from eight lupus
# donors, split each sample in two, and stimulated one half with IFN-β. Every
# donor contributes to both arms, so the design is paired and the donor is a
# genuine random effect. Roughly 29k cells before quality control.
#
# The data comes as an h5ad: raw counts in `X`, the cell annotations in obs and
# the gene annotations in var. `load_h5ad` takes it directly and the counts go
# through the usual Rust quality control and normalisation.

# %%
kang_path = bv.io.download_kang_pbmc()

tempdir_kang = tempfile.TemporaryDirectory()

sc = bv.io.load_h5ad(
    Path(tempdir_kang.name) / "kang",
    kang_path,
    qc=MinQualityParams(
        min_unique_genes=200,
        min_lib_size=500,
        min_cells=20,
        target_size=1e4,
    ),
    verbose=True,
)

# %% [markdown]
# Drop the doublets next. The `multiplets` column is the demultiplexing call,
# and a doublet is two cells' counts added together, which is not something any
# of these models should see. A handful of cells also have no cell type call;
# they go too.

# %%
obs = sc.obs(cols=["cell_idx", "multiplets", "cell"], filtered=True)

keep = obs.filter((pl.col("multiplets") == "singlet") & pl.col("cell").is_not_null())[
    "cell_idx"
]

sc.set_cells_to_keep(keep.to_numpy())

sc

# %% [markdown]
# Look at the design before anything else.

# %%
obs = sc.obs(filtered=True)

obs.pivot(on="stim", index="ind", values="cell_id", aggregate_function="len").sort(
    "ind"
)

# %% [markdown]
# Eight donors, both arms, hundreds to thousands of cells each. That is what
# NEBULA needs. It also needs enough cells per donor: below thirty it quietly
# downgrades from its LN method to HL, as the original package does.
#
# ## One cell type at a time
#
# Differential expression across a mixed population mostly measures composition.
# If the treatment shifts the monocyte fraction, every monocyte gene comes out
# "differential" whether or not any monocyte changed. Work within a cell type.

# %%
mono = bv.SingleCellsSubset(sc, "cell", "CD14+ Monocytes")

bv.pp.find_hvg(mono, hvg_no=500)
hvgs = mono.gene_ids(mono.map.hvg)

mono

# %% [markdown]
# Restricting to the highly variable genes is not just for speed. NEBULA fits an
# optimiser per gene, and running it over 12,000 genes to find the 300 that
# matter is a poor trade.
#
# ## The real contrast
#
# IFN-β stimulation is about as strong a perturbation as single cell data offers.
# All three approaches should find it. Same gene set for all three, so the counts
# are comparable.

# %%
mono_obs = mono.obs()

wilcox_res = bv.de.find_markers(
    mono,
    cells_1=mono_obs.filter(pl.col("stim") == "stim")["cell_id"].to_list(),
    cells_2=mono_obs.filter(pl.col("stim") == "ctrl")["cell_id"].to_list(),
    verbose=True,
)

wilcox_res.filter(pl.col("gene_id").is_in(hvgs) & (pl.col("fdr") <= 0.05)).height

# %% [markdown]
# For pseudobulk, the samples are donor by condition, so sixteen of them. The
# design blocks on the donor, which is what makes it a paired test. The design
# is a plain matrix with one row per pseudobulk sample in `cell_list` order;
# a polars frame keeps the column names, so the coefficient can be picked by
# name.

# %%
groups = (
    mono_obs.with_columns(pl.col("ind").cast(pl.String), pl.col("stim").cast(pl.String))
    .group_by("ind", "stim", maintain_order=True)
    .agg("cell_id")
    .sort("ind", "stim")
)

cell_list = {
    f"{ind}_{stim}": cells
    for ind, stim, cells in groups.select("ind", "stim", "cell_id").iter_rows()
}

donor_levels = sorted(groups["ind"].unique())
design_pb = pl.DataFrame(
    {
        "(Intercept)": np.ones(groups.height),
        **{f"ind{d}": (groups["ind"] == d).cast(pl.Float64) for d in donor_levels[1:]},
        "stimstim": (groups["stim"] == "stim").cast(pl.Float64),
    }
)

pb_res = bv.de.pseudobulk_dge(
    mono,
    cell_list,
    design_pb,
    coef="stimstim",
    edger_params=EdgerQlParams(filter=False),
    verbose=True,
)

pb_res.filter(pl.col("feature_id").is_in(hvgs) & (pl.col("fdr") <= 0.05)).height

# %% [markdown]
# NEBULA takes the design as a formula against the obs table and the donor column
# separately. The donor is not a term in the design; it is the random effect.

# %%
nebula_res = bv.de.nebula(
    mono,
    subject_col="ind",
    design="~ stim",
    genes_to_use=hvgs,
    verbose=True,
)

nebula_res

# %%
nebula_res.results.sort("p_value", maintain_order=True).select(
    "gene_id", "log_fc", "z", "p_value", "fdr"
).head(8)

# %% [markdown]
# CXCL10, CCL8 and APOBEC3B at the top, IL8 down. That is the interferon
# response, and it is reassuring rather than surprising.
#
# Note that the Wilcoxon test finds *fewer* genes here than the other two, not
# more. Pseudoreplication inflates significance, but the Wilcoxon test is also
# a rank test with an expression-proportion filter, and it is simply less
# powerful for a large shift in mean. On a real, strong effect all three agree
# the effect is there. The difference shows up somewhere else.
#
# ### Selecting cells without a subset
#
# You do not need a `SingleCellsSubset` to restrict the fit. `cells_to_use` takes
# cell names and runs NEBULA on those cells of the full object. Handy when the
# selection is a one-off, say one cell type in one condition, and building a
# subset for it is overkill. Same cells, same fit:

# %%
nebula_res_cells = bv.de.nebula(
    sc,
    subject_col="ind",
    design="~ stim",
    genes_to_use=hvgs,
    cells_to_use=mono_obs["cell_id"].to_list(),
)

assert nebula_res.results.equals(nebula_res_cells.results)

# %% [markdown]
# ### Genes carried by one donor
#
# NEBULA's expression filter, `cpc` and `mincp`, pools every cell, so a single
# donor expressing a gene is enough to get it fitted. `min_subjects` adds a
# per-donor check: the gene has to clear `cpc` within at least that many donors.
# It defaults to `0`, which is off and matches the `nebula` package.

# %%
nebula_res_min = bv.de.nebula(
    mono,
    subject_col="ind",
    design="~ stim",
    genes_to_use=hvgs,
    nebula_params=NebulaParams(min_subjects=6),
)

{
    "default": nebula_res.results.height,
    "min_subjects": nebula_res_min.results.height,
}

# %% [markdown]
# ## No contrast at all
#
# Here is the test that matters. Take the control cells only, so there is no
# treatment anywhere in the data, and split the eight donors into two arbitrary
# groups of four. Any gene called differential is a false positive by
# construction.

# %%
ctrl = mono_obs.filter(pl.col("stim") == "ctrl")
donors = sorted(ctrl["ind"].unique())
group_a = donors[0::2]

ctrl = ctrl.with_columns(
    fake=pl.when(pl.col("ind").is_in(group_a)).then(pl.lit("a")).otherwise(pl.lit("b"))
)

ctrl.pivot(on="fake", index="ind", values="cell_id", aggregate_function="len").select(
    "ind", "a", "b"
).sort("ind")

# %%
wilcox_null = bv.de.find_markers(
    mono,
    cells_1=ctrl.filter(pl.col("fake") == "a")["cell_id"].to_list(),
    cells_2=ctrl.filter(pl.col("fake") == "b")["cell_id"].to_list(),
)

wilcox_null.filter(pl.col("gene_id").is_in(hvgs) & (pl.col("fdr") <= 0.05)).height

# %%
groups_null = ctrl.group_by("ind").agg("cell_id", pl.col("fake").first()).sort("ind")

cell_list_null = {
    str(ind): cells for ind, cells in groups_null.select("ind", "cell_id").iter_rows()
}

design_null = pl.DataFrame(
    {
        "(Intercept)": np.ones(groups_null.height),
        "fakeb": (groups_null["fake"] == "b").cast(pl.Float64),
    }
)

pb_null = bv.de.pseudobulk_dge(
    mono,
    cell_list_null,
    design_null,
    edger_params=EdgerQlParams(filter=False),
)

pb_null.filter(pl.col("feature_id").is_in(hvgs) & (pl.col("fdr") <= 0.05)).height

# %% [markdown]
# A `SingleCellsSubset` cuts a `SingleCells`, not another subset, so fold both
# restrictions into one grouping column on the parent.

# %%
parent_obs = sc.obs(cols=["cell_idx", "cell", "stim", "ind"], filtered=True)

sc.add_obs(
    parent_obs.select(
        "cell_idx",
        cell_stim=pl.concat_str(
            pl.col("cell").cast(pl.String),
            pl.col("stim").cast(pl.String),
            separator="__",
        ),
        fake=pl.when(pl.col("ind").is_in(group_a))
        .then(pl.lit("a"))
        .otherwise(pl.lit("b")),
    )
)

mono_ctrl = bv.SingleCellsSubset(sc, "cell_stim", "CD14+ Monocytes__ctrl")

nebula_null = bv.de.nebula(
    mono_ctrl,
    subject_col="ind",
    design="~ fake",
    genes_to_use=hvgs,
    verbose=True,
)

nebula_null.results.filter(pl.col("fdr") <= 0.05).height

# %% [markdown]
# The cell-level test calls dozens of genes differential between two groups of
# donors that differ in nothing. The donor-aware tests call almost none. That gap
# is the entire argument, and it does not go away with more cells: sequencing
# more cells per donor makes the Wilcoxon test *more* confident about a
# difference that is not there.
#
# NEBULA is not exactly zero, and it should not be advertised as such. Eight
# donors split four against four is a small experiment, real donors genuinely
# differ, and an arbitrary split can land partway along some real axis of donor
# variation. A handful of calls out of several hundred genes is the honest
# expectation.
#
# ## Reading the rest of the output
#
# The Wald test is the headline, but the fit carries the variance decomposition
# that motivated the model.

# %%
nebula_res.results.sort("subject_overdispersion", descending=True).select(
    "gene_id",
    "subject_overdispersion",
    "cell_overdispersion",
    "sigma_at_bound",
    "convergence",
).head(6)

# %% [markdown]
# `subject_overdispersion` is how much the gene's expression moves between
# donors, and it is exactly the term a cell-level test pretends is zero. Genes
# with a large value are the ones a Wilcoxon test would most confidently get
# wrong.
#
# Two columns are diagnostics rather than results. `convergence` at or below
# `-20` marks a fit that likely failed, and `sigma_at_bound` marks a gene where
# the subject-level variance finished pinned on its lower bound, meaning the
# mixed model collapsed to a plain negative binomial. Neither is fatal, but a
# sweep where most genes show either is telling you something about the design.

# %%
nebula_res.results.select(
    n=pl.len(),
    failed=(pl.col("convergence") <= -20).sum(),
    collapsed=pl.col("sigma_at_bound").sum(),
)

# %% [markdown]
# The fixed effects and their standard errors sit on the result as frames of
# genes by coefficients (`coefficients` and `se`), so the full model is
# available without refitting.

# %%
nebula_res.coefficients.head(4)

# %% [markdown]
# ## Which one to reach for
#
# Use pseudobulk when the number of donors is decent and you want something
# conservative, well understood and fast. It is a bulk method with decades of
# scrutiny behind it, and `bv.de.run_edger_ql()` gives you the edgeR
# quasi-likelihood chain in Rust.
#
# Use NEBULA when the cell-level information is worth keeping: few donors, uneven
# cell counts, or when you care about the variance decomposition rather than only
# the fold change.
#
# Use the Wilcoxon test for marker detection, which is what it is good at. Asking
# "which genes distinguish this cluster" is a description of the data in front of
# you, not an inference about a population of donors, so pseudoreplication does
# not apply. Asking "does this treatment change expression" is an inference, and
# then it does.
#
# ## Cost
#
# NEBULA fits an optimiser per gene and the fits are independent, so it
# parallelises well and scales linearly in genes. On this data 500 genes over
# 5,000 cells take a fraction of a second (NEBULA prints its timing above with
# `verbose=True`). Ten times the cells and ten times the genes is a hundred
# times the work, which is fine for a considered analysis and painful in a loop
# over twenty cell types. Restrict the genes.
#
# `gene_batch_size` in `NebulaParams` bounds how much of the store is resident
# at once. It changes nothing about the answer, since NEBULA is
# gene-independent, so leave it alone unless memory is tight.
#
# ## Where next
#
# The differential *abundance* question is the other half of this, and it has its
# own [vignette](differential_abundance.ipynb): whether the treatment changed
# which cells are there, rather than what those cells are expressing. The same
# donor structure argument applies, and Milo handles it the same way.
#
# ## Clean up

# %%
tempdir_kang.cleanup()
