"""Training and scoring through the public `bixverse.nn` surface."""

import platform

import numpy as np
import pytest

import bixverse.nn as bvnn

from .conftest import N_CELLS

BACKENDS = ["torch", "jax"]
if platform.system() == "Darwin" and platform.machine() == "arm64":
    BACKENDS.append("mlx")

SHORT = bvnn.ScviParams(max_epochs=5, batch_size=64)


@pytest.fixture(scope="module", params=BACKENDS)
def backend(request):
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture(scope="module")
def batch():
    return np.where(np.arange(N_CELLS) % 2 == 0, "a", "b")


def test_train_then_embed_every_cell(store, backend, batch):
    path, _ = store
    fit = bvnn.train_scvi(
        path, batch=batch, params=SHORT, backend=backend, device="cpu"
    )
    latent = bvnn.scvi_latent(fit, path, batch=batch)
    assert latent.shape == (N_CELLS, SHORT.n_latent)
    assert np.isfinite(latent).all()
    assert list(fit.batch_categories) == ["a", "b"]
    assert len(fit.history.loss) == 5
    assert fit.history.kl_weight[0] == 0.0


def test_latent_rows_follow_the_requested_cells(store, backend, batch):
    path, _ = store
    cells = np.array([10, 3, 150, 42])
    fit = bvnn.train_scvi(
        path, batch=batch, params=SHORT, backend=backend, device="cpu"
    )
    full = bvnn.scvi_latent(fit, path, batch=batch)
    sub = bvnn.scvi_latent(fit, path, cells, batch[cells])
    np.testing.assert_allclose(sub, full[cells], rtol=1e-5, atol=1e-6)


def test_gene_subset_sets_the_model_width(store, backend):
    path, _ = store
    genes = np.arange(0, 40, 2)
    fit = bvnn.train_scvi(
        path, genes=genes, params=SHORT, backend=backend, device="cpu"
    )
    np.testing.assert_array_equal(fit.genes, genes)
    assert bvnn.scvi_latent(fit, path).shape == (N_CELLS, SHORT.n_latent)


def test_elbo_terms_are_finite_and_add_up(store, backend, batch):
    path, _ = store
    fit = bvnn.train_scvi(
        path, batch=batch, params=SHORT, backend=backend, device="cpu"
    )
    elbo = bvnn.scvi_elbo(fit, path, batch=batch)
    assert np.isfinite(list(elbo.values())).all()
    assert elbo["elbo"] == pytest.approx(elbo["reconstruction"] + elbo["kl"])


def test_torch_is_reproducible_under_a_seed(store, batch):
    pytest.importorskip("torch")
    path, _ = store
    fits = [
        bvnn.train_scvi(path, batch=batch, params=SHORT, device="cpu", seed=3)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(
        bvnn.scvi_latent(fits[0], path, batch=batch),
        bvnn.scvi_latent(fits[1], path, batch=batch),
    )


def test_rejects_a_batch_vector_of_the_wrong_length(store):
    path, _ = store
    with pytest.raises(ValueError, match="one label per cell"):
        bvnn.train_scvi(path, batch=np.zeros(3), params=SHORT)


def test_rejects_unknown_batch_labels_when_scoring(store, batch):
    pytest.importorskip("torch")
    path, _ = store
    fit = bvnn.train_scvi(path, batch=batch, params=SHORT, device="cpu")
    with pytest.raises(ValueError, match="not trained on"):
        bvnn.scvi_latent(fit, path, batch=np.repeat("zzz", N_CELLS))


def test_rejects_cells_without_counts_over_the_genes(store, truth):
    pytest.importorskip("torch")
    path, _ = store
    empty = np.flatnonzero(truth[:, 0] == 0)[0]
    with pytest.raises(ValueError, match="no counts"):
        bvnn.train_scvi(path, cells=[empty], genes=[0], params=SHORT)
