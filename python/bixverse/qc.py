"""Cell QC: per-cell metrics, then MAD or fixed-threshold outlier calls.

Two steps. `gene_set_proportions` and `top_genes_perc` read the cell store in
Rust and write their metrics to obs. The outlier calls (`run_cell_qc`,
`run_cell_qc_fixed`) work on plain vectors you pull out of obs and return a
`CellQc` record. `get_data` turns that into an obs frame, and you decide which
cells to keep with `sc.set_cells_to_keep`. Nothing is deleted. `rescue_cells`
and `flag_cells` adjust a `CellQc` afterwards. The metric passes take a
`SingleCells`; the outlier calls know nothing about experiments.

**What lands where.** `gene_set_proportions` writes one obs column per gene set,
named by its key, as a fraction in `[0, 1]`. `top_genes_perc` writes
`top_<N>_genes_percentage`. `lib_size` and `nnz` are already in obs from the
loaders. The outlier calls write nothing: `sc.add_obs(get_data(qc))` stores the
calls, and `sc.set_cells_to_keep` applies the filter.

```python
mito_genes = sc.gene_ids()[:13].tolist()  # on real data: the MT- genes
bv.qc.gene_set_proportions(sc, {"mito": mito_genes})
bv.qc.top_genes_perc(sc)

obs = sc.obs(filtered=True)
metrics = {"lib_size": obs["lib_size"], "nnz": obs["nnz"], "mito": obs["mito"]}
directions = {"lib_size": "below", "nnz": "below", "mito": "above"}
qc = bv.qc.run_cell_qc(metrics, obs["cell_idx"].to_numpy(), directions=directions)

sc.add_obs(bv.qc.get_data(qc))
sc.set_cells_to_keep(qc.cell_idx[~qc.combined])
```

**Gotchas.** Pass the metrics for the kept cells (`sc.obs(filtered=True)`) with
their `cell_idx`, so the two line up. `get_data` writes the metrics back to obs
under their own names, so a transformed metric such as `log10_lib_size` needs
its own key or it overwrites the original column. `directions`, if given, must
name every metric. With `groups` (donor, sample) MAD runs per group, hard
thresholds do not. `rescue_cells` only clears MAD flags, never hard ones. The
metric passes stream the cells in batches above `N_CELLS_STREAMING_THRESHOLD`
kept cells, and `streaming=` overrides that.
"""

from bixverse._core.qc import (
    N_CELLS_STREAMING_THRESHOLD,
    CellQc,
    MadOutlier,
    flag_cells,
    gene_set_proportions,
    get_available_features,
    get_data,
    per_cell_qc_outlier,
    rescue_cells,
    run_cell_qc,
    run_cell_qc_fixed,
    top_genes_perc,
)

__all__ = [
    "N_CELLS_STREAMING_THRESHOLD",
    "CellQc",
    "MadOutlier",
    "flag_cells",
    "gene_set_proportions",
    "get_available_features",
    "get_data",
    "per_cell_qc_outlier",
    "rescue_cells",
    "run_cell_qc",
    "run_cell_qc_fixed",
    "top_genes_perc",
]
