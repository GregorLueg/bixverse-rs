"""Weight transfer: scvi-tools' SCVI into ours, our torch model into MLX and JAX.

Same weights, same batch, eval mode, z at the posterior mean: the encoder
output, the reconstruction term and the KL must agree to float rounding. A
mismatch localises to a layer or a likelihood term. Skips without the
`validation` group (scvi-tools), off Apple Silicon (MLX) or without JAX.
"""

import platform

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bixverse.nn import ScviParams  # noqa: E402
from bixverse.nn import _torch as bvt  # noqa: E402

N_BATCH_CELLS = 64


@pytest.fixture(scope="module")
def counts(truth):
    """Raw counts and two batch codes, as float32 tensors."""
    x = torch.from_numpy(np.asarray(truth[:N_BATCH_CELLS], dtype=np.float32))
    b = torch.from_numpy(np.arange(N_BATCH_CELLS) % 2).long()
    return x, b


def _scvi_reference(truth, lik):
    """A scvi-tools SCVI on the synthetic counts, trained briefly so BatchNorm
    running statistics are not at their initial values."""
    scvi = pytest.importorskip("scvi")
    anndata = pytest.importorskip("anndata")
    import pandas as pd  # anndata's own dependency; only used to build obs

    n = truth.shape[0]
    adata = anndata.AnnData(
        X=np.asarray(truth, dtype=np.float32),
        obs=pd.DataFrame({"batch": np.where(np.arange(n) % 2 == 0, "a", "b")}),
    )
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    model = scvi.model.SCVI(adata, gene_likelihood=lik)
    model.train(max_epochs=2, accelerator="cpu", batch_size=64)
    model.module.eval()
    return model.module.cpu()


@pytest.mark.parametrize("lik", ["nb", "zinb"])
def test_scvi_tools_weights_reproduce_its_loss(truth, counts, lik):
    from torch.distributions import Normal, kl_divergence

    ref = _scvi_reference(truth, lik)
    x, b = counts

    ours = bvt.Vae(x.shape[1], 2, ScviParams(gene_likelihood=lik))
    wanted = set(ours.state_dict())
    ref_state = ref.state_dict()
    assert wanted <= set(ref_state), wanted - set(ref_state)
    ours.load_state_dict({k: ref_state[k] for k in wanted}, strict=True)
    ours.eval()

    with torch.no_grad():
        inf = ref._regular_inference(x, b[:, None])
        qz = inf["qz"]
        gen = ref.generative(z=qz.loc, library=inf["library"], batch_index=b[:, None])
        ref_recon = -gen["px"].log_prob(x).sum(-1)
        ref_kl = kl_divergence(qz, Normal(0.0, 1.0)).sum(-1)
        out = ours(x, b, 1.0, sample=False)

    np.testing.assert_allclose(out["qz_m"], qz.loc, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out["reconstruction"], ref_recon, rtol=1e-5)
    np.testing.assert_allclose(out["kl"], ref_kl, rtol=1e-5, atol=1e-6)


def test_lanczos_lgamma_matches_torch():
    mx = pytest.importorskip("mlx.core")
    from bixverse.nn import _mlx as bvm

    x = np.concatenate([np.geomspace(1e-3, 1.0, 200), np.geomspace(1.0, 1e5, 400)])
    got = np.array(bvm.lgamma(mx.array(x.astype(np.float32))))
    want = torch.lgamma(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="MLX needs Apple Silicon",
)
@pytest.mark.parametrize("lik", ["nb", "zinb"])
def test_torch_weights_reproduce_in_mlx(counts, lik):
    mx = pytest.importorskip("mlx.core")
    from bixverse.nn import _mlx as bvm

    x, b = counts
    params = ScviParams(gene_likelihood=lik)
    torch.manual_seed(0)
    src = bvt.Vae(x.shape[1], 2, params)
    # Move BatchNorm off its initial statistics so they are checked too.
    src.train()
    with torch.no_grad():
        src(x, b, 1.0)
    src.eval()

    dst = bvm.Vae(x.shape[1], 2, params)
    dst.load_weights(bvm.torch_state_to_mlx(src.state_dict()), strict=True)
    dst.eval()
    n_torch = sum(p.numel() for p in src.parameters())
    assert bvm.n_parameters(dst) == n_torch

    with torch.no_grad():
        want = src(x, b, 1.0, sample=False)
    got = dst(mx.array(x.numpy()), mx.array(b.numpy().astype(np.int32)), 1.0, False)

    np.testing.assert_allclose(
        np.array(got["qz_m"]), want["qz_m"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        np.array(got["reconstruction"]), want["reconstruction"], rtol=1e-4
    )
    np.testing.assert_allclose(np.array(got["kl"]), want["kl"], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("lik", ["nb", "zinb"])
def test_torch_weights_reproduce_in_jax(counts, lik):
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from bixverse.nn import _jax as bvj

    x, b = counts
    params = ScviParams(gene_likelihood=lik)
    torch.manual_seed(0)
    src = bvt.Vae(x.shape[1], 2, params)
    src.train()
    with torch.no_grad():
        src(x, b, 1.0)
    src.eval()

    dst = bvj.torch_state_to_jax(src.state_dict(), n_batch=2)
    assert bvj.n_parameters(dst) == sum(p.numel() for p in src.parameters())

    with torch.no_grad():
        want = src(x, b, 1.0, sample=False)
    got, _ = bvj.forward(
        dst.params,
        dst.state,
        jnp.asarray(x.numpy()),
        jnp.asarray(b.numpy().astype(np.int32)),
        1.0,
        jax.random.key(0),
        n_batch=2,
        dropout=params.dropout_rate,
        train=False,
        sample=False,
    )

    np.testing.assert_allclose(
        np.asarray(got["qz_m"]), want["qz_m"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(got["reconstruction"]), want["reconstruction"], rtol=1e-5
    )
    np.testing.assert_allclose(np.asarray(got["kl"]), want["kl"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("lik", ["nb", "zinb"])
def test_jax_batchnorm_training_update_matches_torch(counts, lik):
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from bixverse.nn import _jax as bvj

    x, b = counts
    params = ScviParams(gene_likelihood=lik, dropout_rate=0.0)
    torch.manual_seed(0)
    src = bvt.Vae(x.shape[1], 2, params)
    dst = bvj.torch_state_to_jax(src.state_dict(), n_batch=2)
    src.train()
    with torch.no_grad():
        want = src(x, b, 1.0, sample=False)
    got, state = bvj.forward(
        dst.params,
        dst.state,
        jnp.asarray(x.numpy()),
        jnp.asarray(b.numpy().astype(np.int32)),
        1.0,
        jax.random.key(0),
        n_batch=2,
        dropout=0.0,
        train=True,
        sample=False,
    )
    np.testing.assert_allclose(
        np.asarray(got["reconstruction"]), want["reconstruction"], rtol=1e-5
    )
    after = bvj.torch_state_to_jax(src.state_dict(), n_batch=2).state
    for g, w in zip(jax.tree.leaves(state), jax.tree.leaves(after), strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-5, atol=1e-6)
