"""Ligand-receptor analysis, two ways: CellPhoneDB on the expression in your
experiment, and NicheNet-style ligand activity and interaction prioritisation.

**CellPhoneDB.** `get_cellphonedb_db` fetches the v5 database (once, it is
cached) and `cellphonedb` scores every interaction for every ordered pair of
clusters in an obs column. `method` picks permutation p-values, a DEG-gated
test or the plain means. The database speaks gene symbols: if your var ids are
Ensembl, point `gene_id_col` at the symbol column.

**NicheNet.** Build a ligand to target influence matrix from a signalling and a
gene regulatory network with `generate_ligand_target_influence`, or wrap one you
already have with `as_ligand_target_influence`. `ligand_activity_scores` ranks
the ligands against a gene set (the receiver's DEGs, say) within a background,
`compute_expression_info` gives per-cluster expression of the ligands and
receptors, and `prioritise_interactions` combines it all with the DE tables
into one ranked table of sender, ligand, receiver, receptor.

**What lands where.** Nothing here writes to the experiment. Every function
returns a polars frame or a `LigandTargetInfluence`. Only `cellphonedb` and
`compute_expression_info` read the counts, and they take a `SingleCells` (kept
cells only); the NicheNet steps work on the networks and tables you hand them.
The DE tables come from you: `bv.de.find_all_markers` renamed to `cluster_id`,
`gene`, `lfc`, `pval` does the job.

```python
import numpy as np
import polars as pl

genes = sc.gene_ids().tolist()
ligands, receptors, tfs, targets = genes[:5], genes[5:10], genes[10:15], genes[20:120]
ppi = pl.DataFrame({"from": ligands + receptors, "to": receptors + tfs, "weight": 1.0})
grn = pl.DataFrame(
    {"from": [t for t in tfs for _ in range(20)], "to": targets, "weight": 1.0}
)

influence = bv.lr.generate_ligand_target_influence(ligands, ppi, grn)
activity = bv.lr.ligand_activity_scores(
    influence, {"degs": targets[:20]}, background=genes[:300]
).filter(pl.col("gene_set") == "degs")

# Per cluster DE and expression: yours would come from bv.de and the experiment
cell_types = sorted(sc.obs()["cell_grp"].unique().to_list())
rng = np.random.default_rng(1)
n = len(cell_types) * len(genes)
celltype_de = pl.DataFrame(
    {
        "cluster_id": np.repeat(cell_types, len(genes)),
        "gene": genes * len(cell_types),
        "lfc": rng.normal(size=n),
        "pval": rng.uniform(1e-6, 1, size=n),
    }
)
expr = bv.lr.compute_expression_info(sc, "cell_grp", ligands + receptors)
prio = bv.lr.prioritise_interactions(
    celltype_de,
    expr,
    activity,
    pl.DataFrame({"ligand": ligands, "receptor": receptors}),
    senders_oi=cell_types[:1],
    receivers_oi=cell_types[1:],
    scenario="one_condition",
)

# CellPhoneDB on the experiment, with an L/R table of your own
lr_db = pl.DataFrame(
    {
        "interaction_id": [f"CPI{i}" for i in range(5)],
        "partner_a": ligands,
        "partner_b": receptors,
        "genes_a": [[g] for g in ligands],
        "genes_b": [[g] for g in receptors],
    }
)
cpdb = bv.lr.cellphonedb(
    sc, "cell_grp", lr_db, params=bv.params.CellphonedbParams(n_perm=100)
)
```

**Gotchas.** `prioritise_interactions` defaults to `scenario="case_control"`,
which weighs the condition terms and then refuses to run without a
`condition_de`; pass `scenario="one_condition"` or your own `weights` when there
is no contrast. It wants one row per ligand in `ligand_activities`: filter
`ligand_activity_scores` down to a single gene set first. Ligand and receptor
ids must be in the same namespace in the networks, the DE table and
`compute_expression_info`, which reads var ids. `cellphonedb` drops interactions
with a subunit missing from the experiment, with a warning, and errors if none
are left. `get_cellphonedb_db` needs a download on first use.
"""

from bixverse._core.lr import (
    LigandTargetInfluence,
    as_ligand_target_influence,
    cellphonedb,
    compute_expression_info,
    generate_ligand_target_influence,
    get_cellphonedb_db,
    get_influence,
    ligand_activity_scores,
    prioritise_interactions,
    resolve_weights,
)

__all__ = [
    "LigandTargetInfluence",
    "as_ligand_target_influence",
    "cellphonedb",
    "compute_expression_info",
    "generate_ligand_target_influence",
    "get_cellphonedb_db",
    "get_influence",
    "ligand_activity_scores",
    "prioritise_interactions",
    "resolve_weights",
]
