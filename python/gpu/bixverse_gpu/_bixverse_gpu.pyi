"""Type stubs for the compiled GPU extension module."""

from typing import Any

import numpy as np

__version__: str

def gpu_available() -> bool: ...
def gpu_plane_ops() -> bool: ...
def pca(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    no_pcs: int,
    params: dict[str, object],
    sparse: bool = True,
    seed: int = 42,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def knn_with_dist(
    embd: np.ndarray,
    params: dict[str, object],
    seed: int = 42,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray]: ...
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
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]: ...
def harmony_v2(
    pca: np.ndarray,
    params: dict[str, object],
    batch_labels: list[np.ndarray],
    seed: int = 42,
    verbose: int = 1,
) -> np.ndarray: ...
def bbknn_graph(
    embd: np.ndarray,
    batch_labels: np.ndarray,
    params: dict[str, object],
    k: int,
    seed: int = 42,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def fast_mnn(
    embd: np.ndarray,
    batch_labels: np.ndarray,
    params: dict[str, object],
    seed: int = 42,
    verbose: int = 1,
) -> np.ndarray: ...
def scrublet(
    f_path_gene: str,
    f_path_cell: str,
    cells: np.ndarray,
    params: dict[str, object],
    seed: int = 42,
    streaming: bool = False,
    return_combined_pca: bool = False,
    return_pairs: bool = False,
    verbose: int = 1,
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
    verbose: int = 1,
) -> dict[str, Any]: ...
def sc_bonsai(
    f_path_gene: str,
    f_path_cell: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    params: dict[str, object],
    verbose: int = 1,
) -> dict[str, Any]: ...
def mc_bonsai(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    gene_indices: np.ndarray,
    params: dict[str, object],
    verbose: int = 1,
) -> dict[str, Any]: ...
def nmf_single_sc(
    f_path_gene: str,
    gene_indices: np.ndarray,
    cell_indices: np.ndarray,
    k: int,
    preprocessing: str,
    use_second_layer: bool,
    params: dict[str, object],
    seed: int = 42,
    verbose: int = 1,
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
    verbose: int = 1,
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
    verbose: int = 1,
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
    verbose: int = 1,
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
    verbose: int = 1,
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
    verbose: int = 1,
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
    verbose: int = 1,
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
    verbose: int = 1,
) -> dict[str, Any]: ...
def scenic_grn(
    f_path_genes: str,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    tf_indices: np.ndarray,
    params: dict[str, object],
    streaming: bool,
    seed: int = 42,
    verbose: int = 1,
    wave_byte_budget: int = ...,
) -> np.ndarray: ...
def mc_scenic_grn(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    tf_indices: np.ndarray,
    params: dict[str, object],
    seed: int = 42,
    verbose: int = 1,
    wave_byte_budget: int = ...,
) -> np.ndarray: ...
def nebula(
    f_path_gene: str,
    f_path_cell: str,
    cells_to_keep: np.ndarray,
    gene_indices: np.ndarray,
    subject_ids: np.ndarray,
    design: np.ndarray,
    params: dict[str, object],
    offset: np.ndarray | None = None,
    verbose: int = 1,
) -> dict[str, Any]: ...
