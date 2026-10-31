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

# QC

def mad_outlier(
    x: np.ndarray, threshold: float, direction: str, r_standard: bool
) -> tuple[np.ndarray, float]: ...
def top_genes_perc(
    f_path_cell: str,
    top_n_vals: list[int],
    cell_indices: np.ndarray,
    streaming: bool,
    verbose: bool = False,
) -> list[np.ndarray]: ...
def gene_set_perc(
    f_path_cell: str,
    gene_sets: list[np.ndarray],
    cell_indices: np.ndarray,
    streaming: bool,
    verbose: bool = False,
) -> list[np.ndarray]: ...
def hvg(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    params: dict[str, object],
    streaming: bool,
    verbose: bool = False,
) -> dict[str, np.ndarray]: ...
def hvg_batch_aware(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    batch_labels: np.ndarray,
    params: dict[str, object],
    streaming: bool,
    verbose: bool = False,
) -> dict[str, np.ndarray]: ...
def pca(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    no_pcs: int,
    params: dict[str, object],
    sparse: bool,
    seed: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

class ResidualFit:
    @property
    def method(self) -> str: ...
    @property
    def n_groups(self) -> int: ...
    @property
    def cell_indices(self) -> np.ndarray: ...
    @property
    def genes(self) -> np.ndarray: ...
    @property
    def group_of_cell(self) -> np.ndarray: ...

def fit_residuals(
    f_path_gene: str,
    f_path_cell: str,
    method: str,
    cell_indices: np.ndarray,
    group_of_cell: np.ndarray | None,
    covariates: list[tuple[str, np.ndarray]],
    params: dict[str, object],
    gene_batch_size: int | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> ResidualFit: ...
def residual_variance(
    f_path_gene: str,
    f_path_cell: str,
    fit: ResidualFit,
    cell_indices: np.ndarray,
    n_hvg: int,
    gene_batch_size: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def pca_residuals(
    f_path_gene: str,
    f_path_cell: str,
    fit: ResidualFit,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    no_pcs: int,
    params: dict[str, object],
    seed: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def sct_corrected_counts(
    f_path_gene: str,
    fit: ResidualFit,
    cell_indices: np.ndarray,
    f_path_out: str,
    gene_batch_size: int | None = None,
    verbose: bool = False,
) -> np.ndarray: ...
def gene_store_to_cell_store(
    f_path_in: str,
    f_path_out: str,
    cells_per_phase: int,
    gene_batch_size: int,
    verbose: bool = False,
) -> None: ...
def magic_impute(
    f_path_gene: str,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    params: dict[str, object],
    verbose: bool = False,
) -> np.ndarray: ...
def knn_with_dist(
    embd: np.ndarray,
    params: dict[str, object],
    validate_index: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]: ...
def knn_metrics(
    idx_ref: np.ndarray,
    dist_ref: np.ndarray,
    idx_query: np.ndarray,
    dist_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]: ...
def knn_graph(
    indices: np.ndarray,
    distances: np.ndarray | None,
    params: dict[str, object],
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def leiden(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    params: dict[str, object],
) -> np.ndarray: ...
def fast_cluster(
    embd: np.ndarray,
    resolutions: list[float],
    params: dict[str, object],
    n_seeds: int = 1,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]: ...
def harmony(
    pca: np.ndarray,
    params: dict[str, object],
    batch_labels: list[np.ndarray],
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray: ...
def harmony_v2(
    pca: np.ndarray,
    params: dict[str, object],
    batch_labels: list[np.ndarray],
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray: ...
def bbknn_graph(
    embd: np.ndarray,
    batch_labels: np.ndarray,
    params: dict[str, object],
    k: int,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def fast_mnn(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    batch_indices: np.ndarray,
    precomputed_pca: np.ndarray | None,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray: ...
def seurat_cca(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    batch_indices: np.ndarray,
    precomputed_pca: np.ndarray | None,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray: ...
def seurat_rpca(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    batch_indices: np.ndarray,
    precomputed_pca: np.ndarray | None,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray: ...
def kbet(
    knn: np.ndarray, batches: np.ndarray, verbose: bool = False
) -> dict[str, Any]: ...
def lisi(
    knn: np.ndarray,
    knn_dist: np.ndarray | None,
    labels: np.ndarray,
    perplexity: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]: ...
def batch_asw(
    embd: np.ndarray,
    batches: np.ndarray,
    max_cells: int | None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def cell_type_asw(
    embd: np.ndarray,
    labels: np.ndarray,
    max_cells: int | None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def pcr(embd: np.ndarray, batches: np.ndarray) -> dict[str, Any]: ...
def graph_connectivity(knn: np.ndarray, labels: np.ndarray) -> dict[str, Any]: ...
def scrublet(
    f_path_gene: str,
    f_path_cell: str,
    cells: np.ndarray,
    params: dict[str, object],
    seed: int = 42,
    streaming: bool = False,
    return_combined_pca: bool = False,
    return_pairs: bool = False,
    verbose: bool = False,
) -> dict[str, Any]: ...
def boost(
    f_path_gene: str,
    f_path_cell: str,
    cells: np.ndarray,
    params: dict[str, object],
    seed: int = 42,
    streaming: bool = False,
    verbose: bool = False,
) -> dict[str, Any]: ...
def scdblfinder(
    f_path_gene: str,
    f_path_cell: str,
    cells: np.ndarray,
    params: dict[str, object],
    return_features: bool = False,
    seed: int = 42,
    streaming: bool = False,
    verbose: bool = False,
) -> dict[str, Any]: ...
def empty_droplets(lib_size: np.ndarray, params: dict[str, object]) -> np.ndarray: ...
def cellsweep(
    f_path_source: str,
    f_path_target: str,
    samples: list[dict[str, Any]],
    params: dict[str, object],
    target_size: float,
    verbose: bool = False,
) -> dict[str, Any]: ...
