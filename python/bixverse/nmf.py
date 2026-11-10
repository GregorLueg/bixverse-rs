"""Non-negative matrix factorisation of the counts into gene programmes.

HALS NMF, factorising genes by cells into gene loadings (`w`, genes by k) and
cell usages (`h`, k by cells). One run, random restarts, consensus NMF in the
cNMF style, and a sweep over k.

Start with `nmf_k_sweep` to pick k: it runs the consensus step for every k in
the range and returns stability and reconstruction error per k. High stability
while the error is still falling is the sweet spot. Then fit `consensus_nmf`
at that k. `nmf` is one run, `stabilised_nmf` keeps all the random restarts
and `get_best_run` pulls out the lowest-loss one.

**What lands where.** Nothing is written to the experiment. Every function
returns a result object, and `get_w` gives the gene loadings as a frame.
`get_data` turns the cell usages of an `NmfResult` or `ConsensusNmfResult`
into an obs frame, one `comp_XX` column per component, for `sc.add_obs`. All
functions take a `SingleCells`, a `SingleCellsSubset` or a `MetaCells`. Genes
default to the HVGs, so run `bv.pp.find_hvg` first or pass `gene_ids`; cells
default to the kept ones.

```python
bv.pp.find_hvg(sc, hvg_no=500)
sweep = bv.nmf.nmf_k_sweep(sc, k_range=range(3, 8), n_runs=10)
sweep.frame
res = bv.nmf.consensus_nmf(sc, k=5, n_runs=10)
sc.add_obs(bv.nmf.get_data(res))
loadings = bv.nmf.get_w(res)
```

**Gotchas.** The factorisation uses the normalised counts; pass
`use_second_layer=False` for the raw ones. The consensus step needs `k` and
`n_runs` of at least 2 and raises when the restarts do not agree, with a
message on what to change (a higher `density_threshold` in
`NmfConsensusParams`, more runs, another k). In `nmf_k_sweep` a failing k gets
a null stability and a warning instead. `consensus_target="w"` clusters in
cell space and gets expensive past 20,000 cells. Indices are 0-based, the
`comp_XX` and `component_id` names count from 1.
"""

from bixverse._core.nmf import (
    ConsensusNmfResult,
    NmfKSweepResult,
    NmfResult,
    StabilisedNmfResult,
    consensus_nmf,
    get_best_run,
    get_data,
    get_h,
    get_params,
    get_stability,
    get_w,
    nmf,
    nmf_k_sweep,
    stabilised_nmf,
)

__all__ = [
    "ConsensusNmfResult",
    "NmfKSweepResult",
    "NmfResult",
    "StabilisedNmfResult",
    "consensus_nmf",
    "get_best_run",
    "get_data",
    "get_h",
    "get_params",
    "get_stability",
    "get_w",
    "nmf",
    "nmf_k_sweep",
    "stabilised_nmf",
]
