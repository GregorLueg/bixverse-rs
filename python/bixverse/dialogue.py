"""DIALOGUE: multicellular programmes, genes that are cell type specific but whose
activity covaries across the samples the cell types share.

One function does the work. `dialogue` takes an experiment, the obs columns that
hold the cell types and the samples, and a per-cell-type feature frame, and
returns a `DialogueResult`. Read it with `get_results` (the programme table) or
straight off its fields: `signatures` for the permissive and strict gene lists,
`scores` for the per-cell programme scores, `refit_fidelity` for how well each
cell type's score survived the final refit. Programmes are 0-based: `mcp_00` is
programme 0.

**What lands where.** Nothing is written to the experiment. The result is
returned, and the run reads the normalised layer of the kept cells only.
`SingleCells`, `SingleCellsSubset` and `MetaCells` are all accepted; for meta
cells the ids are `meta_cell_id` and you build them within samples, or a meta
cell has no well-defined sample.

**Features are yours to make.** Per cell type, a frame of `cell_id` plus at
least two numeric columns, rows matched to cells by id. Compute them within the
cell type (subset, `find_hvg`, `calculate_pca`): slicing a global PCA hands every
cell type the same axis. Genes for the signatures default to the HVGs, so run
`bv.pp.find_hvg` first or pass `gene_ids`.

```python
sim = bv.io.generate_dialogue_test_data()
sc = bv.io.load_csr("dialogue_demo", sim.counts.scipy_csr(), sim.obs, sim.var)
bv.pp.find_hvg(sc, hvg_no=200)

res = bv.dialogue.dialogue(
    sc,
    "cell_grp",
    "sample_id",
    sim.features,
    pmd_params=bv.params.DialoguePmdParams(k=2, n_permutations=100),
)
bv.dialogue.get_results(res)
res.signatures.filter(res.signatures["list"] == "strict")
```

**Gotchas.** The design decides whether it runs at all: at least two cell types
and five samples present in every one of them, with enough cells per sample per
cell type to clear `abn_c`. It errors below that. `k` is the number of
programmes you ask for, with no sweep to pick it, and `n_permutations` sets the
floor of the empirical p-value. Without `quality_col` the cell quality covariate
is the z-scored log library size, which needs `lib_size` in obs.

References: Jerby-Arnon & Regev, Nat. Biotechnol., 2022.
"""

from bixverse._core.dialogue import (
    DialogueResult,
    dialogue,
    get_params,
    get_results,
)

__all__ = ["DialogueResult", "dialogue", "get_params", "get_results"]
