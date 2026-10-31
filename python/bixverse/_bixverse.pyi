"""Type stubs for the compiled extension module."""

from pathlib import Path
from typing import Any

import numpy as np

__version__: str
__core_version__: str

class BixverseError(Exception): ...
class SparseFileError(BixverseError): ...

class CellReader:
    def __init__(self, path: str) -> None: ...
    @property
    def total_cells(self) -> int: ...
    @property
    def total_genes(self) -> int: ...
    @property
    def is_cell_based(self) -> bool: ...
    @property
    def target_size(self) -> float | None: ...
    def read_csr(
        self,
        cells: np.ndarray,
        layer: str = ...,
        genes: np.ndarray | None = ...,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]: ...

class CellLoaderIter:
    def __iter__(self) -> CellLoaderIter: ...
    def __next__(
        self,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        tuple[int, int],
        np.ndarray,
    ]: ...

class CellLoader:
    def __init__(
        self,
        reader: CellReader,
        indices: Any,
        batch_size: int = ...,
        shuffle: bool = ...,
        seed: int | None = ...,
        layer: str = ...,
        sparse: bool = ...,
        prefetch: int = ...,
        drop_last: bool = ...,
        genes: np.ndarray | None = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> CellLoaderIter: ...

def _write_synthetic_store(
    path: str,
    n_cells: int,
    n_genes: int,
    density: float = ...,
    seed: int = ...,
) -> np.ndarray: ...

# Ingestion. The dicts are heterogeneous FFI results, hence `Any`.

def csr_to_file(
    f_path_cells: Path,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    qc_params: dict[str, object],
    verbose: bool = ...,
) -> dict[str, Any]: ...
def h5ad_to_file(
    f_path_cells: Path,
    h5_path: Path,
    cs_type: str,
    no_cells: int,
    no_genes: int,
    qc_params: dict[str, object],
    slot: str = ...,
    streaming: bool = ...,
    verbose: bool = ...,
) -> dict[str, Any]: ...
def norm_h5ad_to_file(
    f_path_cells: Path,
    h5_path: Path,
    cs_type: str,
    no_cells: int,
    no_genes: int,
    obs_lib_size_col: str,
    target_size: float,
    qc_params: dict[str, object],
    verbose: bool = ...,
) -> dict[str, Any]: ...
def mtx_to_file(
    f_path_cells: Path,
    mtx_path: Path,
    qc_params: dict[str, object],
    cells_as_rows: bool,
    streaming: bool = ...,
    verbose: bool = ...,
) -> dict[str, Any]: ...
def tenx_h5_to_file(
    f_path_cells: Path,
    h5_path: Path,
    version: str | None,
    no_cells: int,
    no_genes: int,
    qc_params: dict[str, object],
    feature_type: str | None = ...,
    verbose: bool = ...,
) -> dict[str, Any]: ...
def multi_h5ad_to_file(
    f_path_cells: Path,
    file_tasks: list[dict[str, object]],
    universe_size: int,
    qc_params: dict[str, object],
    verbose: bool = ...,
) -> dict[str, Any]: ...
def multi_mtx_to_file(
    f_path_cells: Path,
    file_tasks: list[dict[str, object]],
    universe_size: int,
    qc_params: dict[str, object],
    verbose: bool = ...,
) -> dict[str, Any]: ...
def multi_tenx_h5_to_file(
    f_path_cells: Path,
    file_tasks: list[dict[str, object]],
    universe_size: int,
    qc_params: dict[str, object],
    verbose: bool = ...,
) -> dict[str, Any]: ...
def merge_sc_files(
    f_path_cells: Path,
    merge_tasks: list[dict[str, object]],
    universe_size: int,
    renormalise: bool,
    target_size: float,
    verbose: bool = ...,
) -> dict[str, Any]: ...

# Store maintenance

def generate_gene_based_data(
    f_path_cells: Path,
    f_path_genes: Path,
    max_mem_gb: float | None = ...,
    verbose: bool = ...,
) -> None: ...
def gene_nnz(f_path_genes: Path) -> np.ndarray: ...
def archive_cells(
    f_path_cells: Path, f_path_archive: Path, level: int = ..., verbose: bool = ...
) -> dict[str, int]: ...
def restore_cells_archive(
    f_path_archive: Path,
    f_path_cells: Path,
    f_path_genes: Path,
    max_mem_gb: float | None = ...,
    verbose: bool = ...,
) -> None: ...
def file_version(path: Path) -> int: ...
def upgrade_to_v4(
    f_path_cells: Path,
    f_path_genes: Path,
    codec: str = ...,
    keep_backup: bool = ...,
    verbose: bool = ...,
) -> None: ...

# obs / var and scans

def h5ad_obs_var_to_parquet(
    h5_path: Path,
    frame: str,
    keep: np.ndarray | None,
    id_col: str | None,
    out_path: Path,
) -> dict[str, Any]: ...
def tenx_obs_var_to_parquet(
    h5_path: Path,
    version: str,
    frame: str,
    keep: np.ndarray | None,
    out_path: Path,
) -> dict[str, Any]: ...
def snake_case(names: list[str]) -> list[str]: ...
def h5ad_dimensions(h5_path: Path) -> tuple[int, int, str]: ...
def raw_count_slot(
    h5_path: Path, candidates: list[str], n_sample: int, threshold: float
) -> str | None: ...
def h5_strings(h5_path: Path, dataset: str) -> list[str]: ...
def tenx_h5_metadata(h5_path: Path) -> tuple[str, int, int]: ...

# Writers

def write_tenx(
    h5_path: Path,
    version: str,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    barcodes: list[str],
    ids: list[str],
    names: list[str],
    feature_types: list[str] | None = ...,
) -> None: ...
def save_h5ad(
    f_path_cells: Path,
    h5_path: Path,
    cell_indices: np.ndarray,
    norm: bool,
    obs_index: list[str],
    obs: list[tuple[str, str, object]],
    var_index: list[str],
    var: list[tuple[str, str, object]],
    obsm: dict[str, np.ndarray],
    varm: dict[str, np.ndarray],
    obsp: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    uns_json: str,
    chunk_size: int = ...,
) -> None: ...
def write_h5ad(
    h5_path: Path,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    obs_index: list[str],
    obs: list[tuple[str, str, object]],
    var_index: list[str],
    var: list[tuple[str, str, object]],
) -> None: ...

# Synthetic data

def synthetic_sc_data(
    n_cells: int,
    n_genes: int,
    n_batches: int,
    n_samples: int | None,
    cell_configs: list[dict[str, Any]],
    batch_effect_strength: str,
    sample_bias: str | None,
    seed: int,
) -> dict[str, Any]: ...
def synthetic_adt_data(
    n_cells: int,
    n_proteins: int,
    n_batches: int,
    isotype_controls: list[int],
    cell_configs: list[dict[str, Any]],
    batch_effect_strength: str,
    seed: int,
) -> dict[str, Any]: ...
def synthetic_dialogue_data(params: dict[str, object], seed: int) -> dict[str, Any]: ...
def synthetic_cellsweep_data(
    params: dict[str, object], seed: int
) -> dict[str, Any]: ...
