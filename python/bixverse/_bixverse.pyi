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
def dge_mann_whitney(
    f_path_cell: str,
    cells_1: np.ndarray,
    cells_2: np.ndarray,
    min_prop: float,
    alternative: str,
    verbose: bool = False,
) -> dict[str, Any]: ...
def dge_one_vs_rest(
    f_path_gene: str,
    cell_groups: list[np.ndarray],
    min_prop: float,
    alternative: str,
    verbose: bool = False,
) -> list[dict[str, Any]]: ...
def dge_one_vs_many(
    f_path_gene: str,
    cell_groups: list[np.ndarray],
    references: np.ndarray,
    min_prop: float,
    alternative: str,
    verbose: bool = False,
) -> list[dict[str, Any]]: ...
def pseudobulk(
    f_path_cell: str,
    cell_groups: list[np.ndarray],
    assay: str,
    sparse: bool = False,
    verbose: bool = False,
) -> Any: ...
def edger_ql(
    counts: np.ndarray,
    design: np.ndarray,
    params: dict[str, object],
    coef: list[int] | None = None,
    contrast: list[float] | None = None,
    n_contrasts: int = 1,
) -> dict[str, Any]: ...
def nebula(
    f_path_gene: str,
    f_path_cell: str,
    cells_to_keep: np.ndarray,
    gene_indices: np.ndarray,
    subject_ids: np.ndarray,
    design: np.ndarray,
    params: dict[str, object],
    offset: np.ndarray | None = None,
    verbose: bool = False,
) -> dict[str, Any]: ...
def milor_nhoods(
    embd: np.ndarray,
    knn_indices: np.ndarray,
    sample_ids: np.ndarray,
    n_samples: int,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def spatial_fdr(p_values: np.ndarray, connectivity: np.ndarray) -> np.ndarray: ...
def meld(
    knn_indices: np.ndarray,
    knn_dist: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def sc_type(
    f_path_gene: str,
    cell_indices: np.ndarray,
    markers: list[dict[str, Any]],
    sensitivity: bool,
    weight_floor: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]: ...
def sc_type_cluster_assignment(
    cell_types: list[str], scores: np.ndarray, cluster_labels: np.ndarray
) -> dict[str, Any]: ...
def sc_type_assign_cells(
    cell_types: list[str],
    scores: np.ndarray,
    params: dict[str, object],
    graph: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    cluster_labels: np.ndarray | None = None,
) -> dict[str, Any]: ...
def build_symphony_ref(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    hvg_indices: np.ndarray,
    batch_labels: list[np.ndarray],
    pca_params: dict[str, object],
    no_pcs: int,
    harmony_params: dict[str, object],
    harmony_version: str,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def symphony_map_query(
    f_path_query: str,
    cell_indices: np.ndarray,
    gene_means: np.ndarray,
    gene_sds: np.ndarray,
    loadings: np.ndarray,
    centroids: np.ndarray,
    nr: np.ndarray,
    c_cache: np.ndarray,
    gene_map: np.ndarray,
    batch_labels: list[np.ndarray],
    params: dict[str, object],
    verbose: bool = False,
) -> dict[str, Any]: ...
def transfer_labels_symphony(
    reference_z_corr: np.ndarray,
    query_z_corr: np.ndarray,
    reference_labels: np.ndarray,
    n_labels: int,
    knn_params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]: ...
def bootstrapped_metacells(
    f_path_cell: str,
    knn: np.ndarray | None,
    embd: np.ndarray | None,
    cells_to_keep: np.ndarray,
    cells_to_use: np.ndarray | None,
    params: dict[str, object],
    target_size: float,
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]: ...
def seacells(
    f_path_cell: str,
    embd: np.ndarray,
    knn_indices: np.ndarray | None,
    knn_distances: np.ndarray | None,
    cells_to_keep: np.ndarray,
    cells_to_use: np.ndarray | None,
    params: dict[str, object],
    target_size: float,
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]: ...
def supercells(
    f_path_cell: str,
    knn_indices: np.ndarray | None,
    knn_distances: np.ndarray | None,
    embd: np.ndarray | None,
    cells_to_keep: np.ndarray,
    cells_to_use: np.ndarray | None,
    params: dict[str, object],
    target_size: float,
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]: ...
def metacells2(
    f_path_cell: str,
    cell_indices: np.ndarray,
    mc2_params: dict[str, object],
    dac_params: dict[str, object],
    target_size: float,
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]: ...
def metacell_density(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_dcs: int,
    k_density: int,
    knn_params: dict[str, object],
    seed: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]: ...
def metacell_compactness(
    dcs: np.ndarray, metacells: list[np.ndarray]
) -> np.ndarray: ...
def metacell_separation(dcs: np.ndarray, metacells: list[np.ndarray]) -> np.ndarray: ...
def mc_hvg(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    params: dict[str, object],
) -> dict[str, np.ndarray]: ...
def mc_pca(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    no_pcs: int,
    params: dict[str, object],
    clr_offsets: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def mc_fit_residuals(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    method: str,
    group_of_cell: np.ndarray | None,
    covariates: list[tuple[str, np.ndarray]],
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> ResidualFit: ...
def mc_residual_variance(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    fit: ResidualFit,
    n_hvg: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def mc_pca_residuals(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    fit: ResidualFit,
    gene_indices: np.ndarray,
    no_pcs: int,
    params: dict[str, object],
    seed: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def palantir(
    knn_indices: np.ndarray,
    knn_dist: np.ndarray,
    early_cell: int,
    terminal_states: np.ndarray | None,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def paga(
    knn_indices: np.ndarray,
    partitions: np.ndarray,
    n_partitions: int,
) -> dict[str, Any]: ...
def gene_trends(
    expression: np.ndarray,
    pseudotime: np.ndarray,
    branch_probs: np.ndarray,
    branch_params: dict[str, object],
    gene_trend_params: dict[str, object],
) -> dict[str, Any]: ...
def sc_bonsai(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    params: dict[str, object],
    verbose: bool = False,
) -> dict[str, Any]: ...
def mc_bonsai(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    gene_indices: np.ndarray,
    params: dict[str, object],
    verbose: bool = False,
) -> dict[str, Any]: ...
def bonsai_layout(
    parent: np.ndarray,
    branch: np.ndarray,
    n_leaves: int,
    layout: str,
    hyperbolic: bool,
) -> dict[str, Any]: ...
def module_scores(
    f_path_cells: str,
    f_path_genes: str,
    gene_sets: list[np.ndarray],
    cells_to_keep: np.ndarray,
    nbin: int,
    ctrl: int,
    seed: int,
    streaming: bool,
    verbose: bool = False,
) -> np.ndarray: ...
def aucell(
    f_path: str,
    gene_sets: list[np.ndarray],
    cells_to_keep: np.ndarray,
    params: dict[str, object],
    streaming: bool,
    verbose: bool = False,
) -> np.ndarray: ...
def mc_aucell(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    gene_sets: list[np.ndarray],
    params: dict[str, object],
    verbose: bool = False,
) -> np.ndarray: ...
def regulon_thresholds(
    auc: np.ndarray, params: dict[str, object]
) -> dict[str, Any]: ...
def vision(
    f_path: str,
    sigs: list[tuple[np.ndarray, np.ndarray]],
    cells_to_keep: np.ndarray,
    streaming: bool,
    verbose: bool = False,
) -> np.ndarray: ...
def mc_vision(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    sigs: list[tuple[np.ndarray, np.ndarray]],
    verbose: bool = False,
) -> np.ndarray: ...
def vision_autocor(
    f_path: str,
    sigs: list[tuple[np.ndarray, np.ndarray]],
    random_sigs: list[list[tuple[np.ndarray, np.ndarray]]],
    cluster_membership: np.ndarray,
    cells_to_keep: np.ndarray,
    embd: np.ndarray,
    knn_indices: np.ndarray | None,
    knn_dist: np.ndarray | None,
    params: dict[str, object],
    streaming: bool,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def mc_vision_autocor(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    sigs: list[tuple[np.ndarray, np.ndarray]],
    random_sigs: list[list[tuple[np.ndarray, np.ndarray]]],
    cluster_membership: np.ndarray,
    embd: np.ndarray,
    knn_indices: np.ndarray | None,
    knn_dist: np.ndarray | None,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def lm_activity(
    f_path: str,
    net: list[tuple[np.ndarray, np.ndarray]],
    cells_to_keep: np.ndarray,
    params: dict[str, object],
    verbose: bool = False,
) -> dict[str, Any]: ...
def mc_lm_activity(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    net: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, object],
    verbose: bool = False,
) -> dict[str, Any]: ...
def hotspot_autocor(
    f_path_genes: str,
    f_path_cells: str,
    embd: np.ndarray,
    knn_indices: np.ndarray | None,
    knn_dist: np.ndarray | None,
    params: dict[str, object],
    cells_to_keep: np.ndarray,
    genes_to_use: np.ndarray,
    streaming: bool,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, np.ndarray]: ...
def hotspot_gene_cor(
    f_path_genes: str,
    f_path_cells: str,
    embd: np.ndarray,
    knn_indices: np.ndarray | None,
    knn_dist: np.ndarray | None,
    params: dict[str, object],
    cells_to_keep: np.ndarray,
    genes_to_use: np.ndarray,
    working_mem_gb: float,
    streaming: bool,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, np.ndarray]: ...
def mc_hotspot_autocor(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    embd: np.ndarray,
    knn_indices: np.ndarray | None,
    knn_dist: np.ndarray | None,
    params: dict[str, object],
    cells_to_keep: np.ndarray,
    genes_to_use: np.ndarray,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, np.ndarray]: ...
def mc_hotspot_gene_cor(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    embd: np.ndarray,
    knn_indices: np.ndarray | None,
    knn_dist: np.ndarray | None,
    params: dict[str, object],
    cells_to_keep: np.ndarray,
    genes_to_use: np.ndarray,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, np.ndarray]: ...
def hotspot_cluster_genes(
    z: np.ndarray, fdr_threshold: float, min_size: int
) -> np.ndarray: ...
def scenic_filter(
    f_path_genes: str,
    cell_indices: np.ndarray,
    params: dict[str, object],
    verbose: bool = False,
) -> np.ndarray: ...
def scenic_grn(
    f_path_genes: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    tf_indices: np.ndarray,
    params: dict[str, object],
    streaming: bool,
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray: ...
def mc_scenic_grn(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    tf_indices: np.ndarray,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray: ...
def top_k_targets(
    importance: np.ndarray, k: int, by_gene: bool, min_value: float | None = None
) -> dict[str, np.ndarray]: ...
def importance_threshold(
    importance: np.ndarray, n_sd: float, min_value: float | None = None
) -> dict[str, np.ndarray]: ...
def pairwise_gene_cors(
    f_path_genes: str,
    genes_1: np.ndarray,
    genes_2: np.ndarray,
    cells_to_keep: np.ndarray,
    spearman: bool,
    verbose: bool = False,
) -> np.ndarray: ...
def mc_pairwise_gene_cors(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    genes_1: np.ndarray,
    genes_2: np.ndarray,
    spearman: bool,
    verbose: bool = False,
) -> np.ndarray: ...
def cistarget(
    rankings: np.ndarray,
    gene_sets: list[np.ndarray],
    auc_threshold: int,
    nes_threshold: float,
    max_rank: int,
    method: str,
    n_mean: int,
    verbose: bool = False,
) -> list[dict[str, Any]]: ...
def nmf_single_sc(
    f_path_gene: str,
    gene_indices: np.ndarray,
    cell_indices: np.ndarray,
    k: int,
    preprocessing: str,
    use_second_layer: bool,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def nmf_multi_sc(
    f_path_gene: str,
    gene_indices: np.ndarray,
    cell_indices: np.ndarray,
    k: int,
    preprocessing: str,
    use_second_layer: bool,
    params: dict[str, object],
    n_runs: int,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def nmf_consensus_sc(
    f_path_gene: str,
    gene_indices: np.ndarray,
    cell_indices: np.ndarray,
    k: int,
    preprocessing: str,
    use_second_layer: bool,
    params: dict[str, object],
    consensus: dict[str, object],
    n_runs: int,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def nmf_k_sweep_sc(
    f_path_gene: str,
    gene_indices: np.ndarray,
    cell_indices: np.ndarray,
    k_range: list[int],
    preprocessing: str,
    use_second_layer: bool,
    params: dict[str, object],
    consensus: dict[str, object],
    n_runs: int,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def nmf_single_mc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    k: int,
    preprocessing: str,
    use_second_layer: bool,
    params: dict[str, object],
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def nmf_multi_mc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    k: int,
    preprocessing: str,
    use_second_layer: bool,
    params: dict[str, object],
    n_runs: int,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def nmf_consensus_mc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    k: int,
    preprocessing: str,
    use_second_layer: bool,
    params: dict[str, object],
    consensus: dict[str, object],
    n_runs: int,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def nmf_k_sweep_mc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    k_range: list[int],
    preprocessing: str,
    use_second_layer: bool,
    params: dict[str, object],
    consensus: dict[str, object],
    n_runs: int,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]: ...
def adt_clr(counts: np.ndarray, seurat_clr: bool) -> np.ndarray: ...
def dsb(
    raw: np.ndarray,
    background: np.ndarray | None,
    isotype_indices: list[int],
    params: dict[str, object],
    scale_factor: str,
    seed: int,
    verbose: int,
) -> dict[str, Any]: ...
def adt_pca(
    x: np.ndarray, no_pcs: int, randomised: bool, seed: int
) -> tuple[np.ndarray, np.ndarray, list[float]]: ...
def wnn(
    emb_one: np.ndarray,
    emb_two: np.ndarray,
    params: dict[str, object],
    seed: int,
    verbose: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def ligand_target_influence(
    ligand_seeds: list[np.ndarray],
    ppi_from: np.ndarray,
    ppi_to: np.ndarray,
    ppi_weight: np.ndarray,
    grn_from: np.ndarray,
    grn_to: np.ndarray,
    grn_weight: np.ndarray,
    n_nodes: int,
    params: dict[str, object],
) -> np.ndarray: ...
def ligand_activity_scores(
    influence: np.ndarray,
    in_gene_sets: list[list[bool]],
) -> list[dict[str, np.ndarray]]: ...
def cluster_expr_stats(
    f_path_gene: str,
    gene_indices: np.ndarray,
    clusters: list[np.ndarray],
) -> dict[str, np.ndarray]: ...
def cellphonedb(
    f_path_gene: str,
    partner_a: list[np.ndarray],
    partner_b: list[np.ndarray],
    clusters: list[np.ndarray],
    pairs: list[tuple[int, int]] | None,
    deg_genes: list[np.ndarray] | None,
    statistical: bool,
    params: dict[str, object],
    verbose: int,
) -> dict[str, Any]: ...
def dialogue_sc(
    f_path_gene: str,
    cell_type_indices: list[np.ndarray],
    features: list[np.ndarray],
    sample_ids: np.ndarray,
    cell_quality: np.ndarray,
    gene_indices: np.ndarray,
    pmd: dict[str, object],
    hlm: dict[str, object],
    refine: dict[str, object],
    verbose: int,
) -> dict[str, Any]: ...
def mc_dialogue(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    cell_type_indices: list[np.ndarray],
    features: list[np.ndarray],
    sample_ids: np.ndarray,
    cell_quality: np.ndarray,
    gene_indices: np.ndarray,
    pmd: dict[str, object],
    hlm: dict[str, object],
    refine: dict[str, object],
    verbose: int,
) -> dict[str, Any]: ...
def extract_genes_dense(
    f_path_gene: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    scale: bool = False,
    clip: float | None = None,
) -> np.ndarray: ...
def grouped_gene_stats(
    f_path_gene: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]: ...
