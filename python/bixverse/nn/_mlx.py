"""scVI on MLX, for Apple Silicon.

The same model as `bixverse.nn._torch`, layer for layer, so a torch
`state_dict` converts with `torch_state_to_mlx` and the two backends can be
checked against each other. This is deliberately not scvi-tools' `mlxSCVI`,
which is a different network (two hidden layers, BatchNorm momentum 0.9, no
ZINB).

References:
    Lopez et al., Nature Methods, 2018.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from beartype import beartype
from mlx.utils import tree_flatten, tree_map

from .._bixverse import CellReader
from ._data import iterate_counts
from ._params import (
    MIN_TRAIN_BATCH,
    ScviFit,
    ScviParams,
    kl_weight,
    resolve_max_epochs,
)

logger = logging.getLogger(__name__)

###########
# Globals #
###########

#: Same as the torch backend; MLX's BatchNorm uses the torch momentum convention.
BN_MOMENTUM: float = 0.01
BN_EPS: float = 1e-3
VAR_EPS: float = 1e-4
LIK_EPS: float = 1e-8
EVAL_BATCH_SIZE: int = 1024

#: Lanczos coefficients (Numerical Recipes `gammln`), as in scvi-tools'
#: `_mlxvae._lgamma`. MLX has no `lgamma` of its own.
LANCZOS_COEF: tuple[float, ...] = (
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
)
LANCZOS_SERIES0: float = 1.000000000190015
SQRT_2PI: float = 2.5066282746310005

###################
# Likelihoods     #
###################


def lgamma(x: mx.array) -> mx.array:
    """Log-gamma for `x > 0` via the Lanczos approximation, differentiable.

    Args:
        x: Positive input.

    Returns:
        `lgamma(x)`, elementwise.
    """
    tmp = x + 5.5
    tmp = tmp - (x + 0.5) * mx.log(tmp)
    ser = mx.full(x.shape, LANCZOS_SERIES0, dtype=x.dtype)
    y = x
    for c in LANCZOS_COEF:
        y = y + 1.0
        ser = ser + c / y
    return -tmp + mx.log(SQRT_2PI * ser / x)


def log_nb_positive(
    x: mx.array, mu: mx.array, theta: mx.array, lgamma_x1: mx.array
) -> mx.array:
    """NB log-likelihood per entry; see the torch backend.

    Args:
        x: Counts, `(n, g)`.
        mu: Means, `(n, g)`.
        theta: Inverse dispersion, `(g,)`.
        lgamma_x1: `lgamma(x + 1)`.

    Returns:
        Log-likelihoods, `(n, g)`.
    """
    log_theta_mu_eps = mx.log(theta + mu + LIK_EPS)
    return (
        theta * (mx.log(theta + LIK_EPS) - log_theta_mu_eps)
        + x * (mx.log(mu + LIK_EPS) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma_x1
    )


def log_zinb_positive(
    x: mx.array, mu: mx.array, theta: mx.array, pi: mx.array, lgamma_x1: mx.array
) -> mx.array:
    """ZINB log-likelihood per entry; see the torch backend.

    Args:
        x: Counts, `(n, g)`.
        mu: Means, `(n, g)`.
        theta: Inverse dispersion, `(g,)`.
        pi: Dropout logits, `(n, g)`.
        lgamma_x1: `lgamma(x + 1)`.

    Returns:
        Log-likelihoods, `(n, g)`.
    """
    softplus_pi = nn.softplus(-pi)
    log_theta_mu_eps = mx.log(theta + mu + LIK_EPS)
    pi_theta_log = -pi + theta * (mx.log(theta + LIK_EPS) - log_theta_mu_eps)
    case_zero = nn.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (mx.log(mu + LIK_EPS) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma_x1
    )
    return mx.where(x < LIK_EPS, case_zero, 0.0) + mx.where(
        x > LIK_EPS, case_non_zero, 0.0
    )


#############
# Modules   #
#############


class _Block(nn.Module):
    """Linear, BatchNorm, ReLU, optional dropout."""

    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        """Build one block.

        Args:
            d_in: Input width, including any injected one-hot.
            d_out: Output width.
            dropout: Dropout rate; 0 for none.
        """
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm(d_out, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the block.

        Args:
            x: Input.

        Returns:
            Activations.
        """
        x = nn.relu(self.bn(self.linear(x)))
        return self.dropout(x) if self.dropout is not None else x


class _FcLayers(nn.Module):
    """Stack of blocks with the one-hot batch injected into each, as the torch one."""

    def __init__(
        self, n_in: int, n_out: int, n_cov: int, params: ScviParams, dropout: float
    ) -> None:
        """Build the stack.

        Args:
            n_in: Input width.
            n_out: Output width.
            n_cov: One-hot width injected per layer, 0 for none.
            params: Hyperparameters.
            dropout: Dropout rate after each block.
        """
        super().__init__()
        self.n_cov = n_cov
        dims = [n_in] + [params.n_hidden] * (params.n_layers - 1) + [n_out]
        self.layers = [
            _Block(d_in + n_cov, d_out, dropout)
            for d_in, d_out in itertools.pairwise(dims)
        ]

    def __call__(self, x: mx.array, cov: mx.array | None) -> mx.array:
        """Run the stack.

        Args:
            x: Input.
            cov: One-hot batch, or `None`.

        Returns:
            Activations.
        """
        for block in self.layers:
            if cov is not None:
                x = mx.concatenate([x, cov], axis=-1)
            x = block(x)
        return x


class _Encoder(nn.Module):
    """q(z | x), blind to batch."""

    def __init__(self, n_input: int, params: ScviParams) -> None:
        """Build the encoder.

        Args:
            n_input: Genes.
            params: Hyperparameters.
        """
        super().__init__()
        self.encoder = _FcLayers(
            n_input, params.n_hidden, 0, params, params.dropout_rate
        )
        self.mean_encoder = nn.Linear(params.n_hidden, params.n_latent)
        self.var_encoder = nn.Linear(params.n_hidden, params.n_latent)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Posterior parameters.

        Args:
            x: `log1p` counts.

        Returns:
            `(mean, variance)`.
        """
        q = self.encoder(x, None)
        return self.mean_encoder(q), mx.exp(self.var_encoder(q)) + VAR_EPS


class _Decoder(nn.Module):
    """p(x | z, batch)."""

    def __init__(self, n_input: int, n_batch: int, params: ScviParams) -> None:
        """Build the decoder.

        Args:
            n_input: Genes.
            n_batch: Batches.
            params: Hyperparameters.
        """
        super().__init__()
        n_cov = n_batch if n_batch > 1 else 0
        self.px_decoder = _FcLayers(
            params.n_latent, params.n_hidden, n_cov, params, 0.0
        )
        self.px_scale_decoder = nn.Linear(params.n_hidden, n_input)
        self.px_dropout_decoder = (
            nn.Linear(params.n_hidden, n_input)
            if params.gene_likelihood == "zinb"
            else None
        )

    def __call__(
        self, z: mx.array, cov: mx.array | None
    ) -> tuple[mx.array, mx.array | None]:
        """Decode.

        Args:
            z: Latent.
            cov: One-hot batch or `None`.

        Returns:
            `(px_scale, px_dropout)`; the second is `None` under `nb`.
        """
        px = self.px_decoder(z, cov)
        dropout = (
            self.px_dropout_decoder(px) if self.px_dropout_decoder is not None else None
        )
        return mx.softmax(self.px_scale_decoder(px), axis=-1), dropout


class Vae(nn.Module):
    """scVI on MLX: the torch `Vae`, ported."""

    def __init__(self, n_input: int, n_batch: int, params: ScviParams) -> None:
        """Build the model.

        Args:
            n_input: Genes.
            n_batch: Batches.
            params: Hyperparameters.
        """
        super().__init__()
        self.n_batch = n_batch
        self.px_r = mx.random.normal((n_input,))
        self.z_encoder = _Encoder(n_input, params)
        self.decoder = _Decoder(n_input, n_batch, params)

    def __call__(
        self, x: mx.array, batch: mx.array, kl_w: float, sample: bool = True
    ) -> dict[str, mx.array]:
        """Loss for one minibatch; see the torch `Vae.forward`.

        Args:
            x: Raw counts, `(n, g)`.
            batch: Batch codes, `(n,)`.
            kl_w: KL weight.
            sample: Decode a draw of z rather than the posterior mean.

        Returns:
            `loss`, per-cell `reconstruction` and `kl`, and `qz_m`.
        """
        library = mx.log(x.sum(axis=1, keepdims=True))
        qz_m, qz_v = self.z_encoder(mx.log1p(x))
        z = qz_m + mx.sqrt(qz_v) * mx.random.normal(qz_m.shape) if sample else qz_m
        cov = (
            (batch[:, None] == mx.arange(self.n_batch)[None, :]).astype(x.dtype)
            if self.n_batch > 1
            else None
        )
        px_scale, px_dropout = self.decoder(z, cov)
        px_rate = mx.exp(library) * px_scale
        theta = mx.exp(self.px_r)
        lgamma_x1 = mx.stop_gradient(lgamma(x + 1))
        if px_dropout is None:
            log_px = log_nb_positive(x, px_rate, theta, lgamma_x1)
        else:
            log_px = log_zinb_positive(x, px_rate, theta, px_dropout, lgamma_x1)
        reconstruction = -log_px.sum(axis=-1)
        kl = (0.5 * (qz_v + mx.square(qz_m) - 1 - mx.log(qz_v))).sum(axis=-1)
        return {
            "loss": mx.mean(reconstruction + kl_w * kl),
            "reconstruction": reconstruction,
            "kl": kl,
            "qz_m": qz_m,
        }


################
# Conversion   #
################

#: torch `state_dict` key fragment -> MLX parameter path fragment.
_TORCH_BLOCK = re.compile(r"fc_layers\.Layer (\d+)\.([01])\.")


@beartype
def torch_state_to_mlx(state: dict) -> list[tuple[str, mx.array]]:
    """Convert a torch `Vae` (or matching scvi-tools) `state_dict` for MLX.

    Args:
        state: Torch tensors keyed as in `bixverse.nn._torch.Vae`.

    Returns:
        `(path, array)` pairs for `mlx.nn.Module.load_weights`.
    """
    out = []
    for key, value in state.items():
        if key.endswith("num_batches_tracked"):
            continue
        new = _TORCH_BLOCK.sub(
            lambda m: f"layers.{m.group(1)}.{'linear' if m.group(2) == '0' else 'bn'}.",
            key,
        )
        new = new.replace("px_scale_decoder.0.", "px_scale_decoder.")
        out.append((new, mx.array(value.detach().cpu().numpy())))
    return out


#############
# Training  #
#############


@beartype
def _batches(
    reader: CellReader,
    cells: np.ndarray,
    genes: np.ndarray,
    codes: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
) -> Iterator[tuple[mx.array, mx.array, np.ndarray]]:
    """Minibatches as MLX arrays.

    Args:
        reader: The store.
        cells: Store cell indices.
        genes: Store gene indices.
        codes: Batch code per cell.
        batch_size: Cells per batch.
        shuffle: Shuffle the order.
        seed: Shuffle seed.

    Yields:
        `(x, batch, cell_ids)`.
    """
    for x, b, ids in iterate_counts(
        reader, cells, genes, codes, batch_size=batch_size, shuffle=shuffle, seed=seed
    ):
        yield mx.array(x), mx.array(b.astype(np.int32)), ids


@beartype
def train(
    reader: CellReader,
    cells: np.ndarray,
    genes: np.ndarray,
    codes: np.ndarray,
    categories: np.ndarray,
    *,
    params: ScviParams,
    seed: int,
) -> ScviFit:
    """Fit scVI on MLX with the same Adam loop as the torch backend.

    MLX's `Adam` has no weight decay, so the coupled L2 term torch's `Adam`
    applies is added to the gradients before the update; bias correction is
    switched on to match torch.

    Args:
        reader: The store.
        cells: Store cell indices to train on.
        genes: Store gene indices.
        codes: Batch code per cell.
        categories: Batch labels.
        params: Hyperparameters.
        seed: Seeds the initialisation, the sampling and the shuffle.

    Returns:
        The fitted model, in eval mode.
    """
    mx.random.seed(seed)
    model = Vae(len(genes), len(categories), params)
    mx.eval(model.parameters())
    optimiser = optim.Adam(
        learning_rate=params.lr, eps=params.eps, bias_correction=True
    )
    wd = params.weight_decay

    def loss_fn(x: mx.array, b: mx.array, kl_w: float) -> tuple[mx.array, mx.array]:
        """Scalar loss plus the per-cell terms, for `value_and_grad`."""
        out = model(x, b, kl_w)
        return out["loss"], mx.stack([out["reconstruction"].sum(), out["kl"].sum()])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    n_epochs = resolve_max_epochs(params, len(cells))
    history: dict[str, list[float]] = {
        "loss": [],
        "reconstruction": [],
        "kl": [],
        "kl_weight": [],
    }

    model.train()
    for epoch in range(n_epochs):
        kl_w = kl_weight(epoch, params.n_epochs_kl_warmup)
        totals = mx.zeros(3)
        n_seen = 0
        for x, b, _ in _batches(
            reader,
            cells,
            genes,
            codes,
            batch_size=params.batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            n = x.shape[0]
            if n < MIN_TRAIN_BATCH:
                continue
            (loss, terms), grads = loss_and_grad(x, b, kl_w)
            if wd > 0:
                grads = tree_map(
                    lambda g, p: g + wd * p, grads, model.trainable_parameters()
                )
            optimiser.update(model, grads)
            totals = totals + mx.concatenate([loss[None] * n, terms])
            # One eval per step materialises the step's whole lazy graph,
            # including the BatchNorm running statistics.
            mx.eval(model.parameters(), optimiser.state, totals)
            n_seen += n
        loss, recon, kl = (np.array(totals) / max(n_seen, 1)).tolist()
        history["loss"].append(loss)
        history["reconstruction"].append(recon)
        history["kl"].append(kl)
        history["kl_weight"].append(kl_w)
        logger.info("epoch %d/%d: loss %.2f", epoch + 1, n_epochs, loss)

    model.eval()
    return ScviFit(
        model=model,
        backend="mlx",
        params=params,
        genes=genes,
        batch_categories=categories,
        device="gpu",
        history=history,
    )


@beartype
def evaluate(
    fit: ScviFit,
    reader: CellReader,
    cells: np.ndarray,
    codes: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run the trained model forward, in cell order; see the torch `evaluate`.

    Args:
        fit: An MLX `ScviFit`.
        reader: The store.
        cells: Store cell indices.
        codes: Batch code per cell.
        seed: Seeds the draw of z.

    Returns:
        `latent`, and per-cell `reconstruction` and `kl`.
    """
    mx.random.seed(seed)
    model = fit.model
    model.eval()
    latent, recon, kl = [], [], []
    for x, b, _ in _batches(
        reader,
        cells,
        fit.genes,
        codes,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        seed=None,
    ):
        out = model(x, b, 1.0)
        mx.eval(out)
        latent.append(np.array(out["qz_m"]))
        recon.append(np.array(out["reconstruction"]))
        kl.append(np.array(out["kl"]))
    return {
        "latent": np.concatenate(latent),
        "reconstruction": np.concatenate(recon),
        "kl": np.concatenate(kl),
    }


@beartype
def n_parameters(model: Vae) -> int:
    """Trainable parameter count, for checking a conversion is complete.

    Args:
        model: An MLX `Vae`.

    Returns:
        Number of scalars across trainable parameters.
    """
    leaves = tree_flatten(model.trainable_parameters())
    return sum(v.size for _, v in leaves if isinstance(v, mx.array))
