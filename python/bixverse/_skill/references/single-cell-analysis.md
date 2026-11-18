# Single cell: downstream analysis

Everything here assumes the core chain has run. Read `single-cell.md` first.
Most methods read the kNN, so `bv.pp.find_hvg()`, then `bv.pp.calculate_pca()`,
then `bv.graph.find_neighbours()` is the entry price.

```python
import numpy as np
import polars as pl
import bixverse as bv
import bixverse.embed  # `import bixverse` does not load bv.embed
```

Rules that hold across this file:

- **Every index is 0-based**: cells, genes, HVGs, clusters, Milo `Nhood`,
  Bonsai nodes, DIALOGUE programmes (`mcp_00` is programme 0).
- **Cell indices are store indices** (obs `cell_idx`), unless a docstring says
  "positions among the kept cells" (Milo index cells do).
- **`sc.obs()` returns every cell, filtered ones included.** Build labels and
  groups from `sc.obs(filtered=True)`.
- **Most functions return a result and leave obs alone.** Write scores back
  with `sc.add_obs(frame)`: a `cell_idx` column joins on it; without one the
  frame needs one row per kept cell in ascending cell order.
- **Gene sets, markers, TFs and networks use var gene ids**, not symbols.
  Unknown ids drop with a warning.
- **A stale embedding or kNN raises; a missing kNN warns and returns `None`.**
  Rerun PCA and neighbours after `set_cells_to_keep()`.
- Parameter bundles are frozen dataclasses in `bv.params`, except
  `bv.de.MiloRParams`.

## Differential expression and markers

Markers describe the cells in front of you, so a Wilcoxon over cells is fine.
A condition effect is an inference about donors. Cells within a donor are not
independent, so use pseudobulk edgeR or NEBULA for it. `bv.de` takes
`SingleCells` or `SingleCellsSubset` and writes nothing to the object.

**Markers (Wilcoxon / Mann-Whitney).**

```python
markers = bv.de.find_all_markers(sc, "leiden_clustering")    # one-vs-rest
obs = sc.obs(filtered=True)
t = obs.filter(pl.col("cell_type") == "T")["cell_id"].to_list()
b = obs.filter(pl.col("cell_type") == "B")["cell_id"].to_list()
pair = bv.de.find_markers(sc, t, b)                           # two groups
spec = bv.de.find_specific_markers(sc, "cell_type")           # one-vs-many
bv.de.get_marker_summary(spec).sort("min_auroc", descending=True)
```

- Columns: `grp` (all-markers only), `gene_id`, `lfc`, `prop1`, `prop2`,
  `z_scores`, `p_values`, `fdr`.
- `alternative` defaults to `"greater"` for `find_all_markers` and
  `find_specific_markers`, `"twosided"` for `find_markers`.
- `min_prop=0.05`: share of either group that must express a gene to test it.
- `find_specific_markers` tests each group against every rival separately.
  Its summary has `min_auroc`, `worst_rival`, `simes_fdr`, `max_p_fdr`; the
  per-rival table comes from `get_marker_comparisons()`. It returns `None` if
  no gene passes `min_prop`.

**Pseudobulk, then edgeR quasi-likelihood.** The design is a matrix or polars
frame, not a formula; its rows follow the order of `cell_list`.

```python
obs = sc.obs(filtered=True).with_columns(pl.col("donor").cast(pl.String))
groups = obs.group_by("donor", "condition").agg("cell_id").sort("donor", "condition")
cell_list = {f"{d}_{c}": cells for d, c, cells in groups.iter_rows()}
design = pl.DataFrame({
    "(Intercept)": np.ones(groups.height),
    "conditiontreated": (groups["condition"] == "treated").cast(pl.Float64),
})
res = bv.de.pseudobulk_dge(sc, cell_list, design, coef="conditiontreated")
```

- `cell_list` maps sample name to cell ids or 0-based indices; an empty match
  raises.
- `pseudobulk_dge` sums **raw** counts. Output: `feature_id`, `log_fc`
  (**log2**), `log_cpm`, `f_stat`, `p_value`, `fdr`.
- `coef` is a 0-based design column or its name. With neither `coef` nor
  `contrast`, the **last** column is tested. Name it.
- The design needs an intercept plus at least one column.
- `EdgerQlParams` defaults: `filter=True` (`filterByExpr`), `"TMM"`,
  `robust=False`, `legacy=False`.
- For the matrix: `pb = bv.de.get_pseudobulked(sc, cell_list)` (samples x
  genes), then `bv.de.run_edger_ql(pb.counts.T, design, pb.gene_ids, coef=...)`
  (features x samples). `assay="norm"` averages normalised counts and is not
  edgeR input.

**NEBULA.** Negative binomial gamma mixed model per gene. The design is a
formula over obs; the subject is the random effect, not a design term.

```python
mono = bv.SingleCellsSubset(sc, "cell_type", "CD14+ Monocytes")
bv.pp.find_hvg(mono, hvg_no=500)
res = bv.de.nebula(
    mono, subject_col="donor", design="~ condition + age",
    coef="conditiontreated",                     # or `contrast`, not both
    genes_to_use=mono.gene_ids(mono.map.hvg),
    nebula_params=bv.params.NebulaParams(nebula_method="ln"),
)
res                      # repr reports n_failed and n_collapsed
res.results.sort("p_value"); res.coefficients; res.se
```

- `results`: `gene_id`, `log_fc` (**natural log**), `effect_se`, `z`,
  `p_value`, `fdr`, `subject_overdispersion`, `cell_overdispersion`,
  `convergence`, `sigma_at_bound` (+ `cell_overdispersion_shrunk`, on by
  default).
- `convergence <= -20` marks a likely failed fit. `sigma_at_bound` marks a gene
  collapsed to a plain negative binomial. Check both before trusting `fdr`.
- A fit costs milliseconds to seconds per gene; `genes_to_use=None` fits all
  genes. `sc.map.hvg` holds indices, `sc.gene_ids(sc.map.hvg)` the ids.
- `cells_to_use` (cell ids) restricts the fit without building a subset.
- `NebulaParams(min_subjects=n)` requires a gene to pass `cpc` in `n` donors
  (default 0, off). Below about 30 cells per subject LN downgrades to HL.
- No NEBULA on `MetaCells`.

**Formula rules** (`nebula`, `test_nhoods`): additive main effects only.
Numeric columns enter as they are; others are treatment coded against the
first level and named `<column><level>`. Strings sort their levels; a polars
`Enum` keeps its own order, so a cast can change the reference level.

## Differential abundance

**Milo.** Neighbourhood counts per sample, edgeR QL, spatial FDR.
`SingleCells` only.

```python
bv.graph.find_neighbours(sc, neighbours_params=bv.params.NeighboursParams(k=60))
milo = bv.de.get_milor_abundances(
    sc, "sample", milor_params=bv.de.MiloRParams(prop=0.1))
design_df = pl.DataFrame({"sample": milo.samples}).join(
    sc.obs(filtered=True).select("sample", "run", "age").unique(),
    on="sample", how="left", maintain_order="left",
)
milo = bv.de.test_nhoods(milo, "~ run + age", design_df,
                         sample_col="sample", coef="ageWk16")
milo = bv.de.add_nhoods_info(milo, sc.obs(filtered=True)["cell_type"])
da = bv.de.get_differential_abundance_res(milo)
da.filter(pl.col("SpatialFDR") <= 0.1)
```

- `test_nhoods` and `add_nhoods_info` return a **new** `MiloR`; reassign.
- Columns are edgeR-style: `Nhood`, `logFC`, `logCPM`, `F`, `PValue`, `FDR`,
  `SpatialFDR`, then `majority_celltype`, `majority_prop`.
- `design_df` has one row per sample; `sample_col` defaults to `"sample_id"`.
- The default kNN `k=15` is small for Milo. Choose `k` so a neighbourhood
  holds a few cells per sample (vignette: `k=60` for 18 samples).
- Check the sample x condition table first. A run that carries every
  condition is not the sample.
- `test_nhoods` defaults: `"TMM"`, `robust=True`, `legacy=True`,
  `fdr_weighting="k-distance"`, no `filterByExpr`; `min_mean` drops sparse
  neighbourhoods.
- `get_index_cells(milo)` returns positions among kept cells, not `cell_idx`.
- Plot: `bv.pl.milo_nhood_plot(sc, milo, embedding="umap")`.

**MELD.** Per-cell likelihood per level of an obs column, smoothed over the
kNN. No design, no p-values.

```python
meld_res = bv.de.meld(sc, "age")
meld_res.norm_scores     # cell_id + one column per level, rows sum to 1
```

The argument is named `sample_id_col`, but the vignette passes the condition.
Use Milo for calls with error control, MELD for a graded per-cell score.

## Cell type annotation

`bv.annotate` takes `SingleCells`.

**scType.** Marker scoring, no reference.

```python
marker_df = pl.DataFrame({"cell_type": [...], "gene_id": [...]})
markers = bv.annotate.prepare_cell_markers(sc, marker_df)
scores = bv.annotate.calc_sc_type_scores(sc, markers)
labels = sc.obs(filtered=True)["leiden_clustering"].to_numpy()
per_cluster = bv.annotate.score_clusters(scores, labels)
calls = bv.annotate.assign_sc_type(sc, scores, cluster_col="leiden_clustering")
sc.add_obs(pl.DataFrame({"cell_idx": sc.cells_to_keep,
                         "sc_type": calls.assignments,
                         "sc_type_hybrid": calls.hybrid}))
```

- `marker_df` gene ids must be in the experiment's id type.
- `score_clusters` returns `cluster_id`, `cell_type`, `scores`, `n_cells`;
  a best score below a quarter of the cluster size gives `"Unknown"`.
- `assign_sc_type` smooths scores over the neighbour graph and warns without
  one. `cluster_col` must be a non-negative integer column covering every
  scored cell, as `find_clusters` writes.
- `calls.assignments` is null for Unknown. `calls.margins` (best minus second)
  flags ambiguous cells. `calls.hybrid` is `None` without `cluster_col`.
  `calls.composition` has per-cluster `purity` and `cluster_mixed`.

**Symphony.** Compressed reference, query mapping, kNN label vote.

```python
bv.pp.find_hvg(ref_sc, hvg_no=2000)
ref = bv.annotate.build_symphony_ref(ref_sc, "sample_id", ref_sc.map.hvg,
                                     no_pcs=30, label_columns=["cell_type"])
bv.annotate.save_symphony_ref(ref, "ref.npz")
bv.annotate.map_symphony_query(ref, query_sc, batch_column="sample_id")
pred = bv.annotate.transfer_labels_symphony(ref, query_sc, "cell_type")
```

- `build_symphony_ref` reruns PCA and Harmony (v2 by default; pass
  `HarmonyParams` for v1) and touches neither experiment.
- `map_symphony_query` returns `None` and **writes**
  `query.cache.embeddings["symphony"]` (corrected), `["symphony_pca"]`,
  `["symphony_r"]`. `batch_column=None` skips the correction.
- Genes match by id; a reference HVG missing from the query becomes a zero
  column.
- `transfer_labels_symphony` **writes** `predicted_<col>` and
  `confidence_<col>` to query obs and returns them. It raises without a
  current `"symphony"` embedding.
- A reference is immutable. `add_symphony_labels()` returns a new one and
  needs the cell set it was built on.

## Ligand-receptor

`bv.lr` writes nothing to the experiment.

**CellPhoneDB.**

```python
lr_db = bv.lr.get_cellphonedb_db()        # v5, downloaded once then cached
cpdb = bv.lr.cellphonedb(sc, "cell_type", lr_db, method="statistical",
                         gene_id_col="gene_symbol",   # a var column of symbols
                         params=bv.params.CellphonedbParams(n_perm=1000))
```

- The database uses gene symbols; with Ensembl var ids set `gene_id_col`.
- Output: `interaction_id`, `partner_a`, `partner_b`, `sender`, `receiver`,
  `mean`, `pval` (null unless `"statistical"`), `gate`.
- `method` is `"statistical"`, `"degs"` (needs `deg_table`) or `"simple"`.
  Interactions with a missing subunit drop with a warning.

**NicheNet-style** activity and prioritisation:

```python
influence = bv.lr.generate_ligand_target_influence(ligands, ppi, grn)
activity = bv.lr.ligand_activity_scores(
    influence, {"degs": receiver_degs}, background=receiver_expressed
).filter(pl.col("gene_set") == "degs")
expr = bv.lr.compute_expression_info(sc, "cell_type", ligands + receptors)
prio = bv.lr.prioritise_interactions(
    celltype_de, expr, activity,      # celltype_de: cluster_id, gene, lfc, pval
    pl.DataFrame({"ligand": ligands, "receptor": receptors}),
    senders_oi=["Myeloid"], receivers_oi=["Malignant"],
    scenario="one_condition",
)
```

- `ppi` and `grn` are frames with `from`, `to`, `weight`, from outside
  bixverse.
- `prioritise_interactions` defaults to `scenario="case_control"`, which
  raises without `condition_de`.
- `ligand_activities` must have one row per ligand; filter to one gene set.

## Gene set scoring, HotSpot and SCENIC

`bv.pathways` takes `SingleCells`, `SingleCellsSubset` or `MetaCells`
(`module_scores`: no `MetaCells`) and writes nothing.

| Method | Function | Input |
|---|---|---|
| Seurat module scores | `module_scores(sc, gs)` | name -> gene ids |
| AUCell | `aucell(sc, gs, aucell_params=...)` | name -> gene ids |
| VISION | `vision(sc, sigs)`, `vision_w_autocor(sc, sigs, "pca")` | name -> `{"pos": [...], "neg": [...]}` |
| ULM / MLM | `lm_activity(sc, net, lm_params=...)` | `source`, `target`, `weight` |

```python
res = bv.pathways.aucell(sc, {"hypoxia": [...], "mes_like": [...]})
sc.add_obs(bv.pathways.get_data(res))       # cell_idx + one column per set
```

- `lm_activity` returns `ScMatrixRes`, or `LmActivityRes` (`es`, `pvals`,
  `fdr`) with `p_values=True`, or `None` when no source keeps `tmin` targets.

**HotSpot.**

```python
autocor = bv.pathways.hotspot_autocor(sc)   # gene_id, gearys_c, z_score, pval, fdr
top = autocor.sort("z_score", descending=True)["gene_id"].head(500).to_list()
hs = bv.pathways.hotspot_gene_cor(sc, genes_to_take=top)
hs = bv.pathways.generate_hotspot_membership(hs, min_size=10)
modules = bv.pathways.get_hotspot_membership(hs)
```

`hotspot_gene_cor` output scales with the square of `genes_to_take`; filter by
autocorrelation first.

**SCENIC.** Each step returns an updated copy of the `ScenicGrn`.

```python
grn = bv.pathways.scenic_grn(sc, tf_ids)
grn = bv.pathways.identify_tf_to_genes(grn, n_sd=2.0)
grn = bv.pathways.tf_to_genes_correlations(grn, sc)
grn = bv.pathways.tf_to_genes_motif_enrichment(          # optional
    grn, rankings, rank_genes, rank_motifs, annot_data)
regulons = bv.pathways.build_regulons(grn)
auc = bv.pathways.aucell(sc, regulons)
on_off = bv.pathways.binarise_regulon_activity(auc)      # .binary, .thresholds
```

- Python ships no TF list and no motif download. `rankings` is a genes x
  motifs integer matrix; `annot_data` has `motif`, `TF`, `annotationSource`.
- `build_regulons` defaults to `use_leading_edge=True` (motif-supported links
  only). Skipping the motif step needs `use_leading_edge=False`, or it
  returns `None`.
- Regressors: `ScenicParams(learner_type=..., learner_params=...)` with
  `ScenicRandomForestDefaults`, `ScenicGradientBoostingDefaults` or
  `ScenicExtraTreesDefaults`.

## NMF

HALS NMF of genes x cells into `w` (genes x k) and `h` (k x cells), on
`SingleCells`, `SingleCellsSubset` or `MetaCells`.

```python
sweep = bv.nmf.nmf_k_sweep(sc, k_range=range(5, 21), n_runs=20)
sweep.frame                                   # stability and error per k
res = bv.nmf.consensus_nmf(sc, k=10, n_runs=30)
sc.add_obs(bv.nmf.get_data(res))              # comp_01, ... + cell_idx
bv.nmf.get_w(res)
```

- `consensus_nmf` is cNMF; `stabilised_nmf` + `get_best_run` keeps the lowest
  loss restart; `nmf` is one run.
- Genes default to the HVGs (run `find_hvg` or pass `gene_ids`).
- Input is the **normalised** layer; `use_second_layer=False` uses raw counts.
- Consensus raises when restarts disagree: raise `density_threshold`
  (`NmfConsensusParams`), add runs or change `k`. The sweep nulls that `k`.
- `consensus_target="w"` clusters in cell space; expensive past ~20,000 cells.
- No LDA / topic model.

## DIALOGUE

Multicellular programmes: cell-type-specific genes whose activity covaries
across shared samples. Takes `SingleCells`, `SingleCellsSubset` or
`MetaCells`; reads the normalised layer.

```python
features = {}
for ct in ["Malignant", "Myeloid", "T"]:
    sub = bv.SingleCellsSubset(sc, "cell_type", ct)
    bv.pp.find_hvg(sub, hvg_no=2000)
    bv.pp.calculate_pca(sub, no_pcs=10)
    pcs = sub.cache.pca_factors                  # rows = subset kept cells
    features[ct] = pl.DataFrame({"cell_id": sub.cell_ids(filtered=True),
                                 **{f"PC_{i+1}": pcs[:, i] for i in range(10)}})
bv.pp.find_hvg(sc, hvg_no=2000)
res = bv.dialogue.dialogue(sc, "cell_type", "sample_id", features,
                           pmd_params=bv.params.DialoguePmdParams(k=3))
bv.dialogue.get_results(res)
res.signatures.filter(pl.col("list") == "strict")
```

- Compute features **within** each cell type. A slice of a global PCA carries
  cell type identity, which is near-constant within one type.
- Each frame needs `cell_id` plus at least two numeric columns and must cover
  every cell of the type.
- Hard errors: fewer than two cell types, fewer than five samples present in
  every type, or too few cells per sample per type for `abn_c` (15).
- `k` is the number of programmes; no sweep exists. `n_permutations` (100)
  sets the floor of the empirical p-value.
- Without `quality_col` the quality covariate needs obs `lib_size`.
- On `MetaCells`, build meta cells within samples and key features by
  `meta_cell_id`.

## Trajectories

`bv.traj` takes `SingleCells`, `SingleCellsSubset` or `MetaCells`, reads the
cached kNN and returns frozen results keyed by `cell_id`.

```python
paga = bv.traj.run_paga(sc, "leiden_clusters")
pal = bv.traj.run_palantir(sc, early_cell="AAACCTG-1",
                           terminal_states={"mes": "TTTGG-1", "npc": "CCGTA-1"})
bv.pp.run_magic(sc, features=genes)          # writes sc.cache.magic
trends = bv.traj.run_gene_trends(sc, pal, features=genes, use_magic=True)
obs = sc.obs(cols=["cell_idx", "cell_id"], filtered=True)
sc.add_obs(obs.join(pal.pseudotime, on="cell_id").select(
    "cell_idx", pl.col("pseudotime").alias("palantir_pt")))
```

- Start and terminal cells are cell ids the kNN was built over.
  `terminal_states=None` detects them.
- `pal.pseudotime`: `cell_id`, `pseudotime` (scaled to [0, 1]), `entropy`.
  `branch_probs` rows need not sum to 1.
- Check `pal.run_info`: `eigen_converged=False` means the eigensolve ran out
  of restarts; `repair_edges` counts edges added to a disconnected graph.
- `run_gene_trends` needs `features`, or `use_magic=True` after `run_magic`
  (no MAGIC on `MetaCells`).
- Gene trend defaults are prior-dominated (`length_scale=1.0` over [0, 1]).
  Shorten `GeneTrendsParams(length_scale=...)` before believing a bump.
- `run_magic` requires `features`. Do not feed the MAGIC layer to correlation
  methods; smoothing inflates gene-gene correlation.

## Bonsai trees

A tree with every cell as a leaf, on Sanity posterior log fold changes of the
**raw** counts. No HVGs, PCA or kNN needed.

```python
tree = bv.trees.bonsai(sc)
tree = bv.trees.relayout_bonsai(tree, layout="equal_daylight")
bv.trees.set_bonsai_embedding(sc, tree)      # sc.cache.embeddings["bonsai"]
```

- Takes `SingleCells` or `MetaCells`, **not** `SingleCellsSubset`.
- `hvg=` takes 0-based gene indices to restrict candidates; otherwise
  `min_signal_to_noise` in `BonsaiParams` selects genes.
- `set_bonsai_embedding` refuses a tree built over another cell set.
- `tree.nodes` is 0-based, leaves first; `relayout_bonsai` renumbers internal
  nodes only.
- For large data run it on `MetaCells`; each meta cell is a leaf.

## Meta cells

Four generators return a new in-memory `MetaCells` and leave the source alone.

```python
mc = bv.metacells.generate_bt_meta_cells(                 # hdWGCNA-style
    sc, params=bv.params.BtMetacellsParams(target_no_metacells=250))
mc = bv.metacells.generate_seacells(sc, bv.params.SeacellsParams(n_sea_cells=200))
mc = bv.metacells.generate_supercells(
    sc, params=bv.params.SupercellParams(graining_factor=20))
mc = bv.metacells.generate_metacells2(sc)                 # raw counts, no kNN

bv.metacells.calc_meta_cell_purity(mc, sc.obs()["cell_type"].to_list())
bv.metacells.calc_diffusion_coordinates(mc, sc.cache.knn)
bv.metacells.calc_manifold_metrics(mc)                    # compactness, separation
```

- The first three read the source's PCA and kNN. Without a kNN, bootstrapped
  and SuperCells return `None`; SEACells raises. `n_sea_cells` has no default.
- Bootstrapped meta cells overlap; the others partition. MetaCells2 leaves
  outliers unassigned (`mc.other_data["cell_info"]`).
- Source cells per meta cell are in obs `original_cell_idx`.
- Purity labels: one per source obs row or one per kept cell.
  `calc_manifold_metrics` needs `calc_diffusion_coordinates` first.
- `MetaCells` runs through `bv.pp`, `bv.graph`, `bv.pathways`, `bv.nmf`,
  `bv.dialogue`, `bv.traj` and `bv.trees`, but not `bv.de` or `bv.integrate`.

**Sample-pure meta cells.**

```python
pipe = (bv.pipeline.step_hvg(hvg_no=2000) >> bv.pipeline.step_pca(30)
        >> bv.pipeline.step_neighbours())
mc = bv.metacells.meta_cells_per_group(
    sc, "sample_id", method="bootstrapped", pipeline=pipe,
    mc_params={"params": bv.params.BtMetacellsParams(target_no_metacells=50)})
```

- Merging (`merge_meta_cells`, `meta_cells_per_group`) sets `is_merged`, adds
  obs `source_id` and drops caches and HVGs; redo the chain afterwards.
- Diffusion and manifold metrics do nothing on a merged object; run them per
  source first.
- `save_metacells` / `load_metacells` persist a set; the residual model is not
  saved.

## Multi-modal (CITE-seq / WNN)

`SingleCellsMultiModal` is a `SingleCells` with an ADT arm. Load RNA, settle
`cells_to_keep`, then add ADT.

```python
bv.io.load_tenx_h5(rna_dir, h5_path)
mm = bv.SingleCellsMultiModal(rna_dir)
# ... RNA QC, mm.set_cells_to_keep(...) ...
adt_sc = bv.io.load_tenx_h5(adt_dir, h5_path, feature_type="Antibody Capture",
    qc=bv.params.MinQualityParams(min_unique_genes=0, min_lib_size=0, min_cells=0))
adt = bv.get_counts(adt_sc).scipy_csr().toarray()        # dense cells x proteins
ids = adt_sc.gene_ids().tolist()
iso = [f for f in ids if "isotype" in f.lower()]

bv.mm.add_adt_counts(mm, adt, adt_sc.cell_ids(), ids, method="dsb", isotype_names=iso)
bv.pp.find_hvg(mm, hvg_no=2000); bv.pp.calculate_pca(mm, no_pcs=30)
bv.mm.calculate_pca_adt(mm, no_pcs=10, features=[f for f in ids if f not in iso])
bv.mm.generate_wnn_graph(mm)
bv.graph.find_clusters(mm, modality="wnn")
```

- Python has no 10x ADT reader; load the antibody capture as its own store
  with QC off, as above.
- `adt_counts` is a **dense** numpy array of raw counts, cells x proteins.
- Every kept cell needs an ADT barcode. ADT is cut to the kept cells when
  added: dropping cells later works, restoring dropped cells does not.
- CLR (default) or DSB normalisation runs inside `add_adt_counts`. DSB defaults
  to `use_isotype_controls=True`, so pass `isotype_names`.
- `generate_wnn_graph` needs both PCAs, rebuilds per-modality kNNs at
  `knn_range=200` and replaces any earlier WNN result.
- Getters: `get_adt_counts(mm, layer=...)`, `get_adt_names`, `get_wnn_weights`.

## Pipelines

Chain preprocessing once, apply to many subsets.

```python
pipe = (bv.pipeline.step_hvg(hvg_no=2000) >> bv.pipeline.step_pca(30)
        >> bv.pipeline.step_neighbours()
        >> bv.pipeline.step_clusters(res=0.5, name="sub_leiden"))
per_sample = bv.pipeline.apply_pipeline_per_group(pipe, sc, "sample_id")
```

- `apply_pipeline(pipe, sc)` writes into `sc`. `apply_pipeline_per_group`
  returns a dict of subsets by group and leaves the parent alone.
- Arguments bind at step construction and the class check runs before the
  first step, so errors surface early. Step order is not checked.
- `step_metacells` turns the object into `MetaCells`; integration steps
  (`step_harmony`, `step_bbknn`, `step_fast_mnn`, ...) must come before it.
- After `step_harmony`, use `step_neighbours(embd_to_use="harmony")`.

## Minibatch loader

```python
reader = bv.CellReader(str(sc.cells_path))
loader = bv.CellLoader(reader, sc.cells_to_keep, batch_size=256,
                       layer="norm", seed=42, genes=sc.map.hvg)
for batch in loader:
    x = batch.scipy_csr()          # CSR, rows = cells
    idx = batch.cell_ids           # store index per row
```

- `indices` and `genes` are 0-based store indices. `indices=None` iterates
  every cell in the store, filtered ones included.
- `layer` is `"raw"`, `"norm"` or `"both"`; the other layer is `None`.
- `norm` is `log1p(count / library_size * target_size)` stored at float16,
  accurate to about three decimals.
- Sparse is the default; `sparse=False` costs a zero-fill over
  `batch_size * n_genes`. Cell-major stores only.

## scVI (`bixverse.nn`)

scVI with scvi-tools 1.5.1 defaults, trained off the store on torch, MLX or
JAX. Training needs the extra (`bixverse[torch]`, `[mlx]`, `[jax]`); a missing
framework raises `ImportError` naming it.

```python
import bixverse.nn as bvnn

obs = sc.obs(cols=["cell_idx", "sample_id"], filtered=True)
cells, batch = obs["cell_idx"].to_numpy(), obs["sample_id"].to_numpy()
store = str(sc.cells_path)
fit = bvnn.train_scvi(store, cells, sc.map.hvg, batch,
                      params=bvnn.ScviParams(n_latent=10), backend="torch")
latent = bvnn.scvi_latent(fit, store, cells, batch)       # (n_cells, 10)
bvnn.scvi_elbo(fit, store, held_out_cells, held_out_batch)
```

- Trains on **raw** counts. Library size is the sum over the genes passed; a
  cell with zero counts over them raises.
- `batch` has one label per entry of `cells`; scoring labels must come from
  training.
- `latent` follows `cells` order and is not stored on the object.
- `max_epochs=None` uses `min(round(20000 / n_cells * 400), 400)`.
- Not supported: learned library size, covariates, labels (no scANVI).
- Torch, MLX and JAX were not installed when this file was checked; the `nn`
  API above is read from source (`bixverse/nn/_api.py`), not executed.
