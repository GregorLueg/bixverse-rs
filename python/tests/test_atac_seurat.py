"""LSI projection, rLSI integration and label transfer against Seurat 6.0.0.

The fixtures under `crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/seurat/`
(`dev/atac/gen_atac_seurat.R`) split the 500 cells into two batches; the
Rust tests cover the numerics, these the wiring. Where an SVD sign would
differ, Seurat's own LSI goes in, as in the Rust tests.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csr_matrix

import bixverse as bv
from bixverse import atac
from bixverse.params import KnnParams, MinQualityParams, SyntheticDataParams

FIX = Path(__file__).parents[2] / "crates/bixverse-atac/tests/fixtures/pbmc3k_chr22"
QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
EXACT = KnnParams(knn_method="exhaustive")


def _lines(rel: str) -> list[str]:
    return (FIX / rel).read_text().split()


def _tsv(rel: str) -> pl.DataFrame:
    return pl.read_csv(FIX / rel, separator="\t", null_values=["NaN"])


def _dense(rel: str) -> np.ndarray:
    """A fixture matrix whose first column is a row label."""
    return _tsv(rel).drop(pl.first()).to_numpy().astype(np.float32)


@pytest.fixture(scope="module")
def barcodes() -> list[str]:
    return _lines("cells.txt")


@pytest.fixture(scope="module")
def batch() -> np.ndarray:
    return np.array([int(x) for x in _lines("seurat/batch.txt")])


@pytest.fixture(scope="module")
def lsi_features() -> np.ndarray:
    return np.array([int(x) for x in _lines("seurat/features.txt")])


def _experiment(path: Path, barcodes: list[str]) -> bv.SingleCellsMultiModal:
    """Synthetic RNA keyed by the barcodes, the fixture peaks as ATAC."""
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=len(barcodes), n_genes=60, n_batches=1), seed=3
    )
    obs = data.obs.with_columns(pl.Series("cell_id", barcodes))
    bv.io.load_csr(path, data.counts, obs, data.var, qc=QC)
    exp = bv.SingleCellsMultiModal(path)
    assert exp.n_cells == len(barcodes)
    atac.add_atac_fragments(
        exp, FIX / "fragments.tsv.gz", peaks=FIX / "peaks.bed", mode="fragment"
    )
    atac.run_tfidf(exp)
    return exp


def test_project_lsi_matches_seurat(tmp_path, barcodes, batch, lsi_features):
    ref = _experiment(
        tmp_path / "ref", [b for b, x in zip(barcodes, batch, strict=True) if x == 0]
    )
    query = _experiment(
        tmp_path / "query", [b for b, x in zip(barcodes, batch, strict=True) if x == 1]
    )
    atac.calculate_lsi(
        ref, n_comp=30, features=lsi_features, drop_first=False, solver="exact"
    )
    assert ref.atac is not None
    ref.atac.lsi_loadings = _dense("seurat/batch0_lsi_loadings.tsv.gz")
    ref.atac.lsi_singular_values = (
        _tsv("seurat/batch0_lsi_d.tsv")["d"].to_numpy().astype(np.float32)
    )

    got = atac.project_lsi(ref, query)
    want = _dense("seurat/project_batch1_on_batch0.tsv.gz")
    np.testing.assert_allclose(got, want, atol=1e-6)
    assert query.atac_cache.embeddings["lsi_projected"] is got


def test_project_lsi_needs_the_reference_features(tmp_path, barcodes, batch):
    ref = _experiment(tmp_path / "ref", barcodes[:250])
    atac.calculate_lsi(ref, n_comp=5, solver="exact")
    data = bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=250, n_genes=60, n_batches=1), seed=3
    )
    obs = data.obs.with_columns(pl.Series("cell_id", barcodes[250:]))
    bv.io.load_csr(tmp_path / "query", data.counts, obs, data.var, qc=QC)
    query = bv.SingleCellsMultiModal(tmp_path / "query")
    atac.add_atac_fragments(
        query,
        FIX / "fragments.tsv.gz",
        tile_width=5000,
        chrom_sizes={"chr22": 50_818_468},
        mode="fragment",
    )
    atac.run_tfidf(query)
    with pytest.raises(ValueError, match="not in the query"):
        atac.project_lsi(ref, query)


def test_rlsi_matches_seurat(tmp_path, barcodes, batch, lsi_features):
    exp = _experiment(tmp_path / "exp", barcodes)
    exp.add_obs(
        pl.DataFrame({"cell_idx": exp.cells_to_keep, "seurat_batch": batch.tolist()})
    )
    atac.calculate_lsi(
        exp, n_comp=30, features=lsi_features, drop_first=False, solver="exact"
    )
    # Seurat's combined LSI, so the comparison does not hinge on SVD signs
    exp.atac_cache.embeddings["lsi"] = _dense("seurat/combined_lsi_embeddings.tsv.gz")

    got = atac.run_rlsi(
        exp, "seurat_batch", k_weight=20, solver="exact", knn_params=EXACT
    )
    np.testing.assert_allclose(got, _dense("seurat/rlsi_integrated.tsv.gz"), atol=1e-5)


def test_transfer_labels_seurat_matches_seurat(tmp_path, barcodes, batch):
    genes = _tsv("signac/gene_activity/genes.tsv")["gene_name"].to_list()
    rna = _tsv("seurat/rna/matrix.tsv.gz")
    counts = csr_matrix(
        (
            rna["value"].to_numpy(),
            (rna["cell_idx"].to_numpy(), rna["feature_idx"].to_numpy()),
        ),
        shape=(len(barcodes), len(genes)),
    )
    ref_cells = np.flatnonzero(batch == 0)
    query_cells = np.flatnonzero(batch == 1)
    labels: list[str | None] = [None] * len(barcodes)
    for i, lab in zip(ref_cells, _lines("seurat/transfer_labels.txt"), strict=True):
        labels[i] = lab
    obs = pl.DataFrame({"cell_id": barcodes, "cell_type": labels})
    var = pl.DataFrame({"gene_id": genes})
    keep_all = MinQualityParams(
        min_unique_genes=0, min_lib_size=0, min_cells=0, target_size=1e4
    )
    bv.io.load_csr(tmp_path / "rna", counts, obs, var, qc=keep_all)
    ref = bv.SingleCells(tmp_path / "rna")
    assert ref.dims == (len(barcodes), len(genes))
    ref.set_cells_to_keep(ref_cells)

    exp = _experiment(tmp_path / "exp", barcodes)
    target = float(_lines("seurat/transfer_ga_target.txt")[0])
    query = atac.gene_activity(
        exp, FIX / "fragments.tsv.gz", FIX / "genes.gtf.gz", target_size=target
    )
    query.set_cells_to_keep(query_cells)

    features = [genes[int(i)] for i in _lines("seurat/transfer_features.txt")]
    lsi = _dense("seurat/batch1_lsi_embeddings.tsv.gz")
    out = bv.annotate.transfer_labels_seurat(
        ref,
        query,
        "cell_type",
        np.ascontiguousarray(lsi[:, 1:30]),
        features=features,
        k_weight=int(_lines("seurat/transfer_k_weight.txt")[0]),
        knn_params=EXACT,
    )
    want = _tsv("seurat/transfer_predictions_exact_knn.tsv")
    names = [c.removeprefix("prediction.score.") for c in want.columns[1:]]
    ours = np.column_stack([out[f"score_{n}"].to_numpy() for n in names])
    theirs = np.column_stack([want[f"prediction.score.{n}"].to_numpy() for n in names])
    diff = np.abs(ours - theirs).max(axis=1)
    # 971 of 973 anchors agree; the other two swap between tied zero-count
    # reference cells and move the scores of the query cells they weight
    assert diff.max() < 1e-2
    assert (diff > 1e-5).sum() <= 20
    predicted = [names[i] for i in theirs.argmax(axis=1)]
    assert out["predicted_cell_type"].to_list() == predicted
