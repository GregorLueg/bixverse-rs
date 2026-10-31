# %% [markdown]
# # Ligand-receptor analysis with bixverse
#
# *Gregor Lueg*
#
# ## Intro
#
# This vignette runs two flavours of ligand-receptor analysis on the PBMC3k data
# set. It assumes you know your way around the `SingleCells` class and the basic
# processing pipeline. If not, have a look at the
# [PBMC processing vignette](pbmc_single_cell.ipynb) first.
#
# The two approaches answer different questions:
#
# 1. [CellPhoneDB](https://www.nature.com/articles/s41596-024-01137-1) asks which
#    ligand-receptor pairs are expressed between two cell types, more so than
#    you'd expect by chance. It works off expression alone, handles multi-subunit
#    receptor complexes and comes with a curated database of ~2,900
#    interactions.
# 2. [NicheNet](https://pubmed.ncbi.nlm.nih.gov/31819264/) goes one step further
#    and asks which ligands best explain a downstream transcriptional response in
#    the receiver. That needs a signalling and a gene regulatory network on top.
#
# Both share the processing and the marker genes, so we do that once.

# %%
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import bixverse as bv
from bixverse.params import CellphonedbParams, LigandTargetParams

# %% [markdown]
# ## Loading and processing PBMC3k
#
# We reuse the standard processing pipeline. See the PBMC processing vignette
# for commentary on each step.

# %%
pbmc3k_path = bv.io.download_pbmc3k()
tempdir_pbmc = tempfile.TemporaryDirectory()

sc = bv.io.load_mtx(
    Path(tempdir_pbmc.name) / "pbmc3k",
    bv.io.get_cell_ranger_params(pbmc3k_path),
)

sc.rename_var_col("column1", "gene_symbol")
var = sc.var()
symbol_to_ensembl = dict(zip(var["gene_symbol"], var["gene_id"], strict=True))
ensembl_to_symbol = dict(zip(var["gene_id"], var["gene_symbol"], strict=True))

# QC and processing
gs_of_interest = {
    "MT": var.filter(pl.col("gene_symbol").str.contains("^MT-"))["gene_id"].to_list(),
    "Ribo": var.filter(pl.col("gene_symbol").str.contains("^RPS|^RPL"))[
        "gene_id"
    ].to_list(),
}
bv.qc.gene_set_proportions(sc, gs_of_interest)

# minimal cell filtering
cells_to_keep = sc.obs().filter(pl.col("MT") < 0.2)["cell_id"].to_list()
sc.set_cells_to_keep(cells_to_keep)

bv.pp.find_hvg(sc, hvg_no=2000)
bv.pp.calculate_pca(sc, no_pcs=30)
bv.graph.find_neighbours(sc)
bv.graph.find_clusters(sc, res=0.5)

# %% [markdown]
# For demonstration purposes we use the Leiden cluster IDs directly as cell
# types. In a real analysis you'd annotate the clusters first (marker genes, a
# canonical panel, `bv.annotate.calc_sc_type_scores()`, take your pick). Senders
# and receivers are biological concepts, and "cluster_3 talks to cluster_0"
# doesn't tell anyone much.

# %%
sc.add_obs(
    sc.obs(cols=["cell_idx", "leiden_clustering"], filtered=True).select(
        "cell_idx",
        pl.format("cluster_{}", pl.col("leiden_clustering")).alias("celltype"),
    )
)

celltype_counts = (
    sc.obs(cols=["celltype"])
    .drop_nulls("celltype")
    .group_by("celltype")
    .len()
    .sort("len", descending=True)
)

celltype_counts

# %% [markdown]
# Both methods want per-cluster marker genes, so let's get them out of the way.
# `bv.de.find_all_markers()` runs a one-vs-rest Wilcoxon test per cluster.

# %%
celltype_de = bv.de.find_all_markers(sc, "celltype").rename(
    # rename to the schema the downstream functions expect
    {"gene_id": "gene", "grp": "cluster_id", "p_values": "pval"}
)

celltype_de.head()

# %% [markdown]
# ## CellPhoneDB
#
# ### The database
#
# `bv.lr.get_cellphonedb_db()` pulls the CellPhoneDB v5 database and flattens it
# to one row per interaction. Each partner comes with the gene symbols of its
# subunits: one gene for a plain protein, several for a complex.

# %%
cpdb_db = bv.lr.get_cellphonedb_db(verbose=False)

cpdb_db.group_by("directionality").len()

# %% [markdown]
# A few receptors are complexes, e.g. the TGF-beta receptors:

# %%
cpdb_db.filter(pl.col("partner_a") == "TGFB1").select(
    "partner_a", "partner_b", "genes_b"
)

# %% [markdown]
# Got your own curated list? Any polars DataFrame with `interaction_id`,
# `partner_a`, `partner_b` and the list columns `genes_a` and `genes_b` works.
#
# ### Statistical analysis
#
# For every interaction and every ordered (sender, receiver) cluster pair,
# partner a is measured in the sender and partner b in the receiver. The
# interaction mean is the average of the two partner means, and a complex takes
# the minimum over its subunits. Both partners need to be expressed in more than
# 10% of their cluster's cells (the gate). The p-value comes from shuffling the
# cluster labels 1,000 times. The database speaks gene symbols, so we point
# `gene_id_col` at the symbol column.

# %%
cpdb_res = bv.lr.cellphonedb(
    sc,
    "celltype",
    cpdb_db,
    gene_id_col="gene_symbol",
    params=CellphonedbParams(n_perm=1000),
)

cpdb_sig = cpdb_res.filter(pl.col("gate") & (pl.col("pval") <= 0.05))

cpdb_sig.height

# %% [markdown]
# That's a lot of dropped interactions. `load_mtx()` throws out genes seen in
# fewer than 10 cells, and much of CellPhoneDB (neurotransmitter transporters,
# neuronal receptors, ...) is simply not expressed in blood:

# %%
cpdb_genes = pl.concat(
    [cpdb_db["genes_a"].explode(), cpdb_db["genes_b"].explode()]
).unique()

{
    "in_database": cpdb_genes.len(),
    "in_object": int(cpdb_genes.is_in(var["gene_symbol"].implode()).sum()),
}

# %% [markdown]
# An interaction needs every subunit of both partners, which leaves this many
# interactions to test:

# %%
cpdb_res["interaction_id"].n_unique()

# %% [markdown]
# Which cluster pairs talk the most? Counting the significant interactions per
# pair gives the classic CellPhoneDB heatmap.

# %%
pair_counts = cpdb_sig.group_by("sender", "receiver").len()

clusters = sorted(
    set(pair_counts["sender"]) | set(pair_counts["receiver"]),
    key=lambda x: int(x.removeprefix("cluster_")),
)
heat = np.full((len(clusters), len(clusters)), np.nan)
pos = {c: i for i, c in enumerate(clusters)}
for s, r, n in pair_counts.iter_rows():
    heat[pos[s], pos[r]] = n

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(heat, cmap="viridis", origin="lower")
ax.set_xticks(range(len(clusters)), clusters, rotation=45, ha="right")
ax.set_yticks(range(len(clusters)), clusters)
ax.set_xlabel("Receiver")
ax.set_ylabel("Sender")
fig.colorbar(im, ax=ax, label="Significant\ninteractions")
fig.tight_layout()
plt.show()

# %% [markdown]
# And the strongest interactions by mean:

# %%
cpdb_sig.sort("mean", descending=True).select(
    "sender", "receiver", "partner_a", "partner_b", "mean", "pval"
).head(10)

# %% [markdown]
# Two things to keep in mind here. First, the permutation test asks for
# specificity, not just expression. Take PPIA-BSG: how many cluster pairs pass
# the expression gate, and in how many is it significant?

# %%
ppia_bsg = (pl.col("partner_a") == "PPIA") & (pl.col("partner_b") == "BSG")
n_pairs = cpdb_res.select("sender", "receiver").unique().height

print(
    f"PPIA-BSG passes the gate in {cpdb_res.filter(ppia_bsg)['gate'].sum()} of "
    f"{n_pairs} cluster pairs, significant in {cpdb_sig.filter(ppia_bsg).height}"
)

# %% [markdown]
# Second, the top of that list is dominated by clusters 5 and 6, the two
# smallest ones:

# %%
celltype_counts.tail(2)

# %% [markdown]
# Means over a handful of cells are noisy, so treat hits from tiny clusters
# with care.
#
# ### DEG-based analysis
#
# The alternative is CellPhoneDB's DEG-based method. No permutations here.
# Instead, an interaction is kept for a pair if the expression gate passes and
# partner a is a marker of the sender or partner b a marker of the receiver.
# You decide what counts as a marker. Here we take genes with a log fold change
# above 0.5 and FDR <= 0.05, and convert them to symbols to match the database.

# %%
deg_table = celltype_de.filter((pl.col("lfc") > 0.5) & (pl.col("fdr") <= 0.05)).select(
    "cluster_id", pl.col("gene").replace_strict(ensembl_to_symbol)
)

# same dropped interactions as above, no need to hear about them twice
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cpdb_deg_res = bv.lr.cellphonedb(
        sc,
        "celltype",
        cpdb_db,
        method="degs",
        deg_table=deg_table,
        gene_id_col="gene_symbol",
    )

cpdb_relevant = cpdb_deg_res.filter(pl.col("gate"))

cpdb_relevant.sort("mean", descending=True).select(
    "sender", "receiver", "partner_a", "partner_b", "mean"
).head(10)

# %% [markdown]
# How many interaction and cluster pair combinations pass the DEG gate, and how
# many of those are also significant in the permutation run?

# %%
overlap = cpdb_relevant.join(
    cpdb_sig, on=["interaction_id", "sender", "receiver"], how="inner"
)

{"deg_gate": cpdb_relevant.height, "also_significant": overlap.height}

# %% [markdown]
# This is the route to take if you've got a proper DE setup, e.g. pseudo-bulk
# across samples (see the notes at the end). The cell type specificity then
# comes from the DE test instead of from shuffling labels.
#
# ## NicheNet
#
# NicheNet ranks upstream ligands by how well their predicted target genes line
# up with a gene set of interest in a receiver cell type. The pipeline:
#
# 1. Build a ligand-target regulatory potential matrix from a signalling
#    network (PPI) and a gene regulatory network (GRN).
# 2. Define a gene set of interest in the receiver (here, its marker genes).
# 3. Score each ligand by how well its targets overlap with that gene set
#    (AUROC, AUPR, correlations).
# 4. Combine ligand activity with sender/receiver expression and DE into a
#    single prioritisation score.
#
# ### Building the networks
#
# A production analysis would load the curated NicheNet networks (signalling
# and GRN) from [Zenodo](https://zenodo.org/records/7074291). To keep the
# vignette self-contained, we construct a small network covering a handful of
# canonical immune signalling pathways. The shape of the inputs is what matters
# here... and it means you can feed in your own networks.

# %%
# Ligand-receptor pairs (canonical immune signalling)
lr_network = pl.DataFrame(
    {
        # ligands
        "from": ["CCL5", "CXCL10", "IL15", "TNF", "TGFB1", "IL10", "ICAM1"],
        # receptors
        "to": ["CCR5", "CXCR3", "IL2RG", "TNFRSF1A", "TGFBR2", "IL10RA", "ITGAL"],
        "weight": 1.0,
    }
)

# Signalling (PPI) layer: ligand -> receptor -> intracellular signalling -> TFs
ppi_network = pl.DataFrame(
    {
        "from": [
            "CCL5",
            "CXCL10",
            "IL15",
            "TNF",
            "TGFB1",
            "IL10",
            "ICAM1",
            "CCR5",
            "CXCR3",
            "IL2RG",
            "TNFRSF1A",
            "TGFBR2",
            "IL10RA",
            "ITGAL",
            "JAK1",
            "JAK2",
            "JAK3",
            "STAT1",
            "STAT3",
            "STAT5A",
            "NFKB1",
            "SMAD2",
        ],
        "to": [
            "CCR5",
            "CXCR3",
            "IL2RG",
            "TNFRSF1A",
            "TGFBR2",
            "IL10RA",
            "ITGAL",
            "JAK1",
            "JAK1",
            "JAK3",
            "NFKB1",
            "SMAD2",
            "JAK1",
            "NFKB1",
            "STAT1",
            "STAT3",
            "STAT5A",
            "STAT3",
            "STAT1",
            "STAT3",
            "STAT3",
            "STAT3",
        ],
        "weight": 1.0,
    }
)

# GRN layer: TF -> targets
grn_network = pl.DataFrame(
    {
        "from": ["STAT1"] * 6
        + ["STAT3"] * 6
        + ["STAT5A"] * 4
        + ["NFKB1"] * 6
        + ["SMAD2"] * 4,
        "to": [
            "IRF1",
            "MX1",
            "ISG15",
            "OAS1",
            "GBP1",
            "CXCL10",
            "SOCS3",
            "IL6",
            "BCL2",
            "VEGFA",
            "MYC",
            "FOS",
            "CCND1",
            "BCL2L1",
            "IL2RA",
            "FOXP3",
            "TNF",
            "IL1B",
            "IL6",
            "NFKBIA",
            "CCL5",
            "ICAM1",
            "SERPINE1",
            "CDKN1A",
            "JUNB",
            "ID1",
        ],
        "weight": 1.0,
    }
)

# %% [markdown]
# Restrict the networks to genes present in PBMC3k; the mapping to the Ensembl
# IDs used internally comes later.

# %%
in_data = list(symbol_to_ensembl)


def keep_present(network: pl.DataFrame) -> pl.DataFrame:
    return network.filter(pl.col("from").is_in(in_data) & pl.col("to").is_in(in_data))


ppi_network = keep_present(ppi_network)
grn_network = keep_present(grn_network)
lr_network = keep_present(lr_network)

# %% [markdown]
# ### Ligand-target regulatory potential
#
# Construct the ligand-target influence matrix. Each row is one ligand and
# each column is a gene from the union of both networks; entries are
# NicheNet-style regulatory potential scores.

# %%
ligand_seeds = {lig: [lig] for lig in lr_network["from"]}

ligand_influence = bv.lr.generate_ligand_target_influence(
    ligand_seeds,
    pl.concat([lr_network, ppi_network]),
    grn_network,
    # because of the tiny network, the original ltf_cutoff of 0.99 would be way
    # too aggressive
    params=LigandTargetParams(ltf_cutoff=0.0),
)

ligand_influence

# %% [markdown]
# This returns a `LigandTargetInfluence`. The actual scores sit in a ligands x
# genes array, with the names on `ligand_names` and `gene_ids`:

# %%
influence = bv.lr.get_influence(ligand_influence)

pl.DataFrame(influence, schema=list(ligand_influence.gene_ids)).insert_column(
    0, pl.Series("ligand", ligand_influence.ligand_names)
)

# %% [markdown]
# ### Defining the gene set of interest
#
# In a case/control experiment, the gene set of interest would be the DEGs
# between conditions in the receiver cell type. PBMC3k is a single condition,
# so we use the receiver's markers as a stand-in. We pick the largest cluster as
# receiver and a few others as senders.

# %%
receiver_id = celltype_counts["celltype"][0]
senders_oi = celltype_counts["celltype"][1:5].to_list()

# upregulated genes with FDR <= 0.05 form the gene set
geneset_oi = celltype_de.filter(
    (pl.col("cluster_id") == receiver_id)
    & (pl.col("lfc") > 0)
    & (pl.col("fdr") <= 0.05)
)["gene"].to_list()

len(geneset_oi)

# %% [markdown]
# ### Ligand activity scoring
#
# Score each ligand by how well its target potential vector aligns with
# `geneset_oi`. The background is the full set of genes in the influence
# matrix.

# %%
activity = bv.lr.ligand_activity_scores(
    ligand_influence,
    # need to transform to gene symbols
    {"receiver_degs": [ensembl_to_symbol[g] for g in geneset_oi]},
)

activity.sort("aupr_corrected", descending=True)

# %% [markdown]
# ### Sender and receiver tables
#
# Prioritisation needs the DE of ligands across senders and of receptors in the
# receiver (we've got that from the markers), plus the average expression of
# ligands and receptors per cluster. The latter calls into the streaming engine
# via `bv.lr.compute_expression_info()`.

# %%
genes_of_interest = list(
    dict.fromkeys(lr_network["to"].to_list() + lr_network["from"].to_list())
)
genes_of_interest_ids = [symbol_to_ensembl[g] for g in genes_of_interest]

expression_info = bv.lr.compute_expression_info(sc, "celltype", genes_of_interest_ids)

expression_info.head()

# %% [markdown]
# ### Prioritisation
#
# Combine all of the above into a single weighted score. PBMC3k has no
# condition contrast, so we use the `"one_condition"` scenario, which sets
# the condition-specificity weights to zero.

# %%
# Reduce activity table to one row per ligand (we ran one gene set)
la_input = activity.select(
    pl.col("ligand").replace_strict(symbol_to_ensembl), "aupr_corrected"
)

lr_network_ensembl = lr_network.select(
    pl.col("from").replace_strict(symbol_to_ensembl).alias("ligand"),
    pl.col("to").replace_strict(symbol_to_ensembl).alias("receptor"),
)

result = bv.lr.prioritise_interactions(
    celltype_de.select("cluster_id", "gene", "lfc", "pval"),
    expression_info,
    la_input,
    lr_network_ensembl,
    senders_oi,
    [receiver_id],
    scenario="one_condition",
)

# %% [markdown]
# Add symbol columns for readability.

# %%
result = result.with_columns(
    pl.col("ligand").replace_strict(ensembl_to_symbol).alias("ligand_symbol"),
    pl.col("receptor").replace_strict(ensembl_to_symbol).alias("receptor_symbol"),
)

result.select(
    "sender",
    "receiver",
    "ligand_symbol",
    "receptor_symbol",
    "prioritisation_score",
    "prioritisation_rank",
).head(10)

# %% [markdown]
# ### Top interactions
#
# A quick dotplot of the top prioritised interactions.

# %%
top_n = 15

plot_df = result.head(top_n).with_columns(
    pl.format("{} -> {}", "ligand_symbol", "receptor_symbol").alias("interaction")
)

senders = sorted(plot_df["sender"].unique())
interactions = sorted(plot_df["interaction"].unique())
x = [senders.index(s) for s in plot_df["sender"]]
y = [interactions.index(i) for i in plot_df["interaction"]]
score = plot_df["prioritisation_score"].to_numpy()

fig, ax = plt.subplots(figsize=(7, 4))
pts = ax.scatter(x, y, s=300 * score / score.max(), c=score, cmap="viridis")
ax.set_xticks(range(len(senders)), senders)
ax.set_yticks(range(len(interactions)), interactions)
ax.set_xlim(-0.5, len(senders) - 0.5)
ax.set_ylim(-0.5, len(interactions) - 0.5)
ax.set_xlabel("Sender")
ax.set_ylabel("Ligand -> Receptor")
fig.colorbar(pts, ax=ax, label="Score")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Notes on DE methods
#
# `bv.de.find_all_markers()` runs a Wilcoxon test, which is convenient but
# inflates p-values in single cell data: each cell is treated as an independent
# observation. For inference that matters, go pseudo-bulk with DESeq2 or edgeR
# style methods:
#
# 1. Aggregate counts per (sample, cluster) into pseudo-bulk.
# 2. Run a standard bulk DE pipeline.
# 3. Feed the result into `bv.lr.cellphonedb(method="degs")` as a `cluster_id`,
#    `gene` table, or into `bv.lr.prioritise_interactions()` after renaming the
#    columns to `cluster_id`, `gene`, `lfc`, `pval`.
#
# Neither function cares which method produced the DE table.
#
# ## Clean up

# %%
tempdir_pbmc.cleanup()
