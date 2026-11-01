"""Getting counts into an experiment directory, and back out.

Every loader takes the directory to create and returns the opened
`SingleCells`. Pick the one for your format: `load_csr` for counts already in
memory, `load_h5ad`, `load_tenx_h5` or `load_mtx` for files, and the
`load_multi_*` twins for several files in one go. After the first load you
only need `load_existing`.

**What a load does.** It applies the cell and gene QC of `MinQualityParams`
(by default at least 100 genes, 250 counts per cell and 10 cells per gene), so
dropped cells and genes never enter the store. It then writes the cell-major
and gene-major stores and loads obs and var into the DuckDB, adding `nnz` and
`lib_size` to obs and `no_cells_exp` to var. Loading into a directory that
already holds an experiment overwrites its stores and drops any saved
`memory_py.*`. A directory written by the R package opens with `load_existing`
too, with an empty cache.

**Several files.** `prescan_h5ad_files`, `prescan_tenx_h5_files` and
`prescan_mtx_dirs` read the gene sets first and fix the gene universe
(intersection by default, or union), then `load_multi_*` loads against that
prescan. obs gains `exp_id` and `cell_id` is prefixed with it. To combine
experiments you already loaded, `merge_sc_experiments` keeps the shared genes
and each input's kept cells, and carries no cache over.

**Saving.** obs and var edits go to the DuckDB at once. `save` writes what
lives in memory: the cell filter, the HVG set and the cache, as `memory_py.npz`
and `memory_py.json`. It does not write a fitted residual model, and warns.
`load_existing` brings it all back, and restores the stores from `counts.bxa`
if only the archive is left (`archive_sc_exp`). `save_h5ad` exports kept cells,
obs, var and the embeddings and graph that still match to an h5ad file;
`write_h5ad_sc`, `write_tenx_h5_sc` and `write_cellranger_output` write
in-memory counts as files other tools read.

**Test data.** `generate_single_cell_test_data` plants cell types in synthetic
counts. The `generate_*` twins serve the ADT, CellSweep and DIALOGUE examples.
The `download_*` functions fetch the vignette data sets into
`~/.cache/bixverse` and return the path, or the cached one.

```python
params = bv.params.SyntheticDataParams(n_cells=2000, n_genes=500)
sim = bv.io.generate_single_cell_test_data(params, seed=1)
sc = bv.io.load_csr("experiments/demo", sim.counts, sim.obs, sim.var)

bv.pp.find_hvg(sc, hvg_no=200)
bv.io.save(sc)
sc = bv.io.load_existing("experiments/demo")
```

**Gotchas.** The loaders want integral raw counts. `load_h5ad` finds them with
`detect_raw_count_slot`, which prefers `layers.counts`, then `raw.X`, then `X`;
set `raw_count_slot` if it picks wrong, and use `load_h5ad_norm` for files that
only hold log-normalised values. `load_mtx` takes genes as rows, the 10x
convention, unless `cells_as_rows` says otherwise; `get_cell_ranger_params`
finds the three files for you. Nothing here needs h5py. A store in the old v3
format still opens with a warning, and `upgrade_sc_store` converts it.
"""

from bixverse._core.download import (
    download_cd34_data,
    download_demuxlet_pbmc,
    download_dialogue_uc,
    download_kang_pbmc,
    download_marrow_cd34,
    download_pbmc3k,
    download_pbmc8k,
    download_pbmc_1k_5p,
    download_pbmc_batches,
    download_pbmc_totalseq_data,
    download_thymus_ageing,
)
from bixverse._core.ingest import (
    archive_sc_exp,
    get_cell_ranger_params,
    load_csr,
    load_h5ad,
    load_h5ad_norm,
    load_mtx,
    load_multi_h5ad,
    load_multi_mtx,
    load_multi_tenx_h5,
    load_tenx_h5,
    merge_sc_experiments,
    stream_h5ad,
    upgrade_sc_store,
)
from bixverse._core.io import load_existing, save
from bixverse._core.scan import (
    H5adDims,
    Prescan,
    TenxH5Meta,
    detect_raw_count_slot,
    get_h5ad_dimensions,
    get_tenx_h5_metadata,
    prescan_h5ad_files,
    prescan_mtx_dirs,
    prescan_tenx_h5_files,
    read_h5ad_metadata,
    read_tenx_h5_metadata,
)
from bixverse._core.synthetic import (
    SyntheticAdtData,
    SyntheticCellsweepData,
    SyntheticData,
    SyntheticDialogueData,
    generate_cellsweep_test_data,
    generate_dialogue_test_data,
    generate_single_cell_test_data,
    generate_single_cell_test_data_adt,
)
from bixverse._core.writers import (
    save_h5ad,
    write_cellranger_output,
    write_h5ad_sc,
    write_tenx_h5_sc,
)

__all__ = [
    "H5adDims",
    "Prescan",
    "SyntheticAdtData",
    "SyntheticCellsweepData",
    "SyntheticData",
    "SyntheticDialogueData",
    "TenxH5Meta",
    "archive_sc_exp",
    "detect_raw_count_slot",
    "download_cd34_data",
    "download_demuxlet_pbmc",
    "download_dialogue_uc",
    "download_kang_pbmc",
    "download_marrow_cd34",
    "download_pbmc3k",
    "download_pbmc8k",
    "download_pbmc_1k_5p",
    "download_pbmc_batches",
    "download_pbmc_totalseq_data",
    "download_thymus_ageing",
    "generate_cellsweep_test_data",
    "generate_dialogue_test_data",
    "generate_single_cell_test_data",
    "generate_single_cell_test_data_adt",
    "get_cell_ranger_params",
    "get_h5ad_dimensions",
    "get_tenx_h5_metadata",
    "load_csr",
    "load_existing",
    "load_h5ad",
    "load_h5ad_norm",
    "load_mtx",
    "load_multi_h5ad",
    "load_multi_mtx",
    "load_multi_tenx_h5",
    "load_tenx_h5",
    "merge_sc_experiments",
    "prescan_h5ad_files",
    "prescan_mtx_dirs",
    "prescan_tenx_h5_files",
    "read_h5ad_metadata",
    "read_tenx_h5_metadata",
    "save",
    "save_h5ad",
    "stream_h5ad",
    "upgrade_sc_store",
    "write_cellranger_output",
    "write_h5ad_sc",
    "write_tenx_h5_sc",
]
