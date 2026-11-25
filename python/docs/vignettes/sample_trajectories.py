# %% [markdown]
# # Sample-level trajectories with scSLIDE
#
# ## Intro
#
# Most single cell questions are about cells. Plenty of clinical ones are about
# people: which patients look alike, is there a severity axis, does time since
# onset show up anywhere. Pseudobulk and Milo answer "did this condition change
# something", but they need you to name the condition up front and they test one
# gene or one neighbourhood at a time. They do not give you a map of the donors.
#
# [scSLIDE](https://doi.org/10.1016/j.cell.2026.10.018) (Jiang, et al., Cell,
# 2026, from the Satija lab, now part of Seurat v6) builds one. Every donor
# becomes a vector of counts over a few thousand landmark cells:
#
# 1. A uniform sketch of each donor trains a PLS model against a sample-level
#    phenotype (here: COVID-19 severity), and every cell is projected onto it.
# 2. Landmarks are drawn from the sketch by leverage score, so rare states get a
#    landmark too.
# 3. Each cell is matched to its nearest landmarks with a fixed-weight WNN over
#    the PCA (cell identity) and the PLS (disease-related variation).
# 4. Counting those matches per donor gives a landmarks x donors table. Its
#    chi-squared residuals go into a diffusion map over the donors.
#
# Step 4 is correspondence analysis on a donor x micro-state contingency table,
# with a diffusion map instead of the SVD. The PLS tilts which micro-states get
# told apart.
#
# > **The PLS has seen the labels**
# >
# > The phenotype goes into the PLS, and the PLS shapes the landmark
# > neighbourhoods of the very donors it was fitted on. A component that
# > separates the phenotype is therefore partly separating it by construction.
# > `bixverse` can cross-fit the PLS by donor (`n_folds > 1`): every donor is
# > embedded by a model that never saw it. More on that below.

# %%
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import bixverse as bv
from bixverse.params import EdgerQlParams, MinQualityParams

# %% [markdown]
# ## The data
#
# The 104k cell subset of [Ahern, et al.](https://pubmed.ncbi.nlm.nih.gov/35216673/)
# the scSLIDE vignette uses: PBMCs of 40 donors, ten healthy and ten each with
# mild, severe and critical COVID-19, with 1,500 to 3,000 cells per donor.
# Raw counts plus donor-level metadata.

# %%
ahern_h5ad = bv.io.download_ahern_covid()

tempdir_ahern = tempfile.TemporaryDirectory()

sc = bv.io.load_h5ad(
    Path(tempdir_ahern.name) / "ahern",
    ahern_h5ad,
    qc=MinQualityParams(min_unique_genes=200, min_lib_size=500, min_cells=10),
)

sc

# %%
donors = (
    sc.obs(filtered=True)
    .with_columns(pl.col("donor", "disease").cast(pl.String))
    .group_by("donor", "disease", "time_since_onset")
    .agg(pl.len().alias("n_cells"))
    .sort("disease", "donor")
)

donors.group_by("disease").agg(
    pl.len().alias("n_donors"), pl.col("n_cells").sum()
).sort("disease")

# %% [markdown]
# ## The usual preprocessing
#
# scSLIDE needs HVGs and a PCA. The reference vignette uses Azimuth's
# reference-mapped embedding as the cell identity modality; a PCA over the HVGs
# does the same job here. Leiden clusters stand in for the cell types: they
# pick the phenotype-correlated genes per cell type and label the landmarks.

# %%
bv.pp.find_hvg(sc, hvg_no=2000)
bv.pp.calculate_pca(sc, no_pcs=50)
bv.graph.find_neighbours(sc)
bv.graph.find_clusters(sc, res=0.5, name="leiden")

# %% [markdown]
# ## Landmark densities
#
# `run_scslide` does steps 1 to 3 and the counting. The defaults follow Seurat's
# `PrepareSampleObject` with the vignette's 20 PLS components: 1,000 cells per
# donor, 2,000 landmarks, PCA and PLS weighted 0.5 each, every cell counted
# against its five nearest landmarks. First the reference's way: one PLS,
# fitted on every donor (`n_folds=1`).

# %%
t0 = time.perf_counter()
res = bv.traj.run_scslide(
    sc,
    "donor",
    "disease",
    cell_type_col="leiden",
    scslide_params=bv.traj.ScSlideParams(n_folds=1),
)
print(f"scSLIDE: {time.perf_counter() - t0:.1f} s")

res

# %% [markdown]
# The output is a landmarks x donors count table. Each landmark carries the
# cluster of the cell it was drawn from, so a row of `counts` is "how many of
# each donor's cells sit next to this little piece of cluster 3".

# %%
pl.DataFrame({"cluster": res.landmark_labels}).group_by("cluster").len().sort(
    "len", descending=True
).head(10)

# %% [markdown]
# ## Donor coordinates
#
# `sample_diffusion_map` takes the chi-squared residuals of that table, centres
# each landmark across donors and runs a diffusion map over the donors with
# cosine distance. Seurat does the same through destiny with `1 - Pearson` and a
# complete graph, so the components agree with it up to sign and scale, not to
# the digit.

# %%
dm = bv.traj.sample_diffusion_map(res)
dcs = dm.dcs.join(
    donors.select("donor", "time_since_onset"), left_on="sample", right_on="donor"
)

dcs.head()

# %%
order = ["HV", "COVID_MILD", "COVID_SEV", "COVID_CRIT"]
colours = dict(zip(order, ["#4c72b0", "#dd8452", "#c44e52", "#8172b3"], strict=True))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (x, y) in zip(axes, [("DC_1", "DC_2"), ("DC_1", "DC_3")], strict=True):
    for d in order:
        sub = dcs.filter(pl.col("disease") == d)
        ax.scatter(sub[x], sub[y], label=d, color=colours[d], s=40)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
axes[0].legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
covid = dcs.filter(pl.col("disease") != "HV")
severity = pl.col("disease").replace_strict(
    {"COVID_MILD": 1, "COVID_SEV": 2, "COVID_CRIT": 3}, return_dtype=pl.Float64
)
covid.select(
    *[
        pl.corr(f"DC_{i}", severity, method="spearman").alias(f"DC_{i} ~ severity")
        for i in (1, 2, 3)
    ],
    *[
        pl.corr(f"DC_{i}", "time_since_onset", method="spearman").alias(
            f"DC_{i} ~ onset"
        )
        for i in (1, 2, 3)
    ],
)

# %% [markdown]
# DC_1 is the healthy versus COVID axis, which is the least interesting thing
# here: the PLS was told about it. The paper's claims are the ones about things
# the model was not told. Within the COVID donors, look for the component that
# orders mild, severe and critical, and for one that tracks days since symptom
# onset. Severity is nested inside the disease label, so it is not entirely
# independent of what the PLS saw; time since onset is.
#
# ## Cross-fitting the PLS
#
# Back to the warning at the top. With `n_folds = 5` the donors are split into
# five folds, balanced over the phenotype classes, and each fold gets its own
# PLS, phenotype-correlated genes included, fitted on the other four. A held-out
# donor's cells, the landmarks and the WNN bandwidths of that fold all live in
# that model's space. The PCA half of the WNN is shared.

# %%
t0 = time.perf_counter()
res_cf = bv.traj.run_scslide(
    sc,
    "donor",
    "disease",
    cell_type_col="leiden",
    scslide_params=bv.traj.ScSlideParams(n_folds=5),
)
print(f"scSLIDE, 5 folds: {time.perf_counter() - t0:.1f} s")
dm_cf = bv.traj.sample_diffusion_map(res_cf)

pl.DataFrame(
    {
        "component": [f"DC_{i}" for i in (1, 2, 3)],
        "abs_r_in_sample_vs_cross_fitted": [
            abs(np.corrcoef(dm.dcs[f"DC_{i}"], dm_cf.dcs[f"DC_{i}"])[0, 1])
            for i in (1, 2, 3)
        ],
    }
)

# %% [markdown]
# DC_1 and DC_2 barely move. The later components are where the two disagree,
# which is where the in-sample run had the most room to fit the labels.
#
# How much of the in-sample separation is the PLS fitting the labels? Shuffle the
# phenotype across donors, so it means nothing, and see how well the components
# separate the shuffled groups. The yardstick is the best between-group R^2 over
# DC_1 to DC_3: four groups of ten donors with labels that have nothing to do
# with the coordinates give about 0.13 on average. A model that only reports
# biology should land there on a shuffled label; one that overfits will find
# whatever label you hand it.


# %%
def best_r2(dcs: pl.DataFrame, labels: dict[str, str]) -> float:
    """Largest between-group share of variance over DC_1..DC_3."""
    groups = np.array([labels[s] for s in dcs["sample"]])
    out = []
    for i in (1, 2, 3):
        y = dcs[f"DC_{i}"].to_numpy()
        between = sum(
            (groups == g).sum() * (y[groups == g].mean() - y.mean()) ** 2
            for g in np.unique(groups)
        )
        out.append(between / ((y - y.mean()) ** 2).sum())
    return float(max(out))


disease = dict(donors.select("donor", "disease").iter_rows())
donor_ids = sorted(disease)
obs = sc.obs(cols=["cell_idx", "donor"], filtered=True)
rng = np.random.default_rng(1)
rows = []
for p in range(5):
    shuffled = dict(
        zip(donor_ids, rng.permutation([disease[d] for d in donor_ids]), strict=True)
    )
    sc.add_obs(
        obs.select(
            "cell_idx",
            pl.col("donor").cast(pl.String).replace_strict(shuffled).alias("shuffled"),
        )
    )
    for n_folds in (1, 5):
        r = bv.traj.run_scslide(
            sc,
            "donor",
            "shuffled",
            cell_type_col="leiden",
            scslide_params=bv.traj.ScSlideParams(n_folds=n_folds),
            seed=100 + p,
            verbose=False,
        )
        d = bv.traj.sample_diffusion_map(r)
        rows.append({"shuffle": p, "n_folds": n_folds, "r2": best_r2(d.dcs, shuffled)})

shuffles = pl.DataFrame(rows)
shuffles.group_by("n_folds").agg(
    pl.col("r2").mean().alias("mean_r2"), pl.col("r2").max().alias("max_r2")
).sort("n_folds")

# %% [markdown]
# In-sample, the shuffled labels come out above chance: the PLS bends the
# landmark neighbourhoods towards whatever it was trained on. Cross-fitted, they
# sit at chance. Five shuffles is a small sample, so take the size of the gap
# with a pinch of salt; twenty permutations of the same experiment gave 0.22
# in-sample against 0.14 cross-fitted. The real labels survive cross-fitting
# comfortably:

# %%
pl.DataFrame(
    {
        "labels": ["real, in-sample", "real, cross-fitted"],
        "r2": [best_r2(dm.dcs, disease), best_r2(dm_cf.dcs, disease)],
    }
)

# %% [markdown]
# ## Trajectory DE
#
# A donor coordinate is a covariate like any other. Pseudobulk one cell type per
# donor and test a component with edgeR QL. Here: the cluster with the highest
# CD14 expression, against the cross-fitted DC_1.

# %%
cd14 = bv.get_counts(sc, genes=sc.gene_indices(["CD14"]), layer="norm")
leiden = sc.obs(cols=["cell_idx", "leiden"], filtered=True)
mono = (
    leiden.with_columns(pl.Series("cd14", cd14.scipy_csr().toarray()[:, 0]))
    .group_by("leiden")
    .agg(pl.col("cd14").mean())
    .sort("cd14", descending=True)["leiden"][0]
)

mono_cells = (
    sc.obs(cols=["cell_id", "donor", "leiden"], filtered=True)
    .with_columns(pl.col("donor").cast(pl.String))
    .filter(pl.col("leiden") == mono)
    .group_by("donor")
    .agg("cell_id")
    .sort("donor")
)
cell_list = dict(mono_cells.iter_rows())

design = (
    pl.DataFrame({"sample": list(cell_list)})
    .join(dm_cf.dcs.select("sample", "DC_1"), on="sample", maintain_order="left")
    .select(pl.lit(1.0).alias("(Intercept)"), "DC_1")
)

traj_de = bv.de.pseudobulk_dge(
    sc, cell_list, design, coef="DC_1", edger_params=EdgerQlParams()
)
traj_de.sort("p_value").head(15)

# %% [markdown]
# Interferon-stimulated genes (`IFI27`, `IFITM3`) and the S100 alarmins along
# the axis that separates COVID from healthy donors in classical monocytes. No
# surprise, which is the point: a donor coordinate drops straight into the
# pseudobulk machinery as a numeric covariate, so the same call works for the
# severity or the onset component, with the healthy donors filtered out first.
#
# ## Where next
#
# - Leiden clusters stood in for Azimuth labels. Better cell type labels give
#   better correlated genes and more readable landmarks.
# - Not here yet: ComBat on the landmark table for multi-batch donors, and
#   Seurat's CPPLS variant of the PLS.
# - Cross-fitting costs about two and a half times the single fit and is cheap
#   next to the preprocessing. With a phenotype in the model, it is the version
#   to report.
#
# ## Clean up

# %%
tempdir_ahern.cleanup()
