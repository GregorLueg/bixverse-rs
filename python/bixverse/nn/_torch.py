"""scVI on torch.

Layer layout and parameter names follow scvi-tools' `VAE` (module
`scvi.module._vae`) for the configuration kept here, so a scvi-tools
`state_dict` restricted to these keys loads one to one. The parameters
scvi-tools allocates but never uses in this configuration (the library-size
encoder, the gene-cell dispersion head, and the dropout head under `nb`) are
not created.

References:
    Lopez et al., Nature Methods, 2018.
"""

from __future__ import annotations

import collections
import itertools
import logging
from collections.abc import Iterator

import numpy as np
import torch
from beartype import beartype
from torch import nn
from torch.nn import functional as F  # noqa: N812 - torch convention

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

#: BatchNorm settings scvi-tools inherited from the original TensorFlow code.
BN_MOMENTUM: float = 0.01
BN_EPS: float = 1e-3

#: Floor on the encoder variance.
VAR_EPS: float = 1e-4

#: Stability constant inside the NB and ZINB log-likelihoods.
LIK_EPS: float = 1e-8

#: Cells per batch when only running the model forward.
EVAL_BATCH_SIZE: int = 1024

###################
# Likelihoods     #
###################


@beartype
def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    lgamma_x1: torch.Tensor,
    eps: float = LIK_EPS,
) -> torch.Tensor:
    """NB log-likelihood per entry, as scvi-tools' `log_nb_positive`.

    Args:
        x: Counts, `(n, g)`.
        mu: Means, `(n, g)`.
        theta: Inverse dispersion, `(g,)`.
        lgamma_x1: `lgamma(x + 1)`, which has no gradient and is computed once.
        eps: Stability constant.

    Returns:
        Log-likelihoods, `(n, g)`.
    """
    log_theta_mu_eps = torch.log(theta + mu + eps)
    return (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - lgamma_x1
    )


@beartype
def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    lgamma_x1: torch.Tensor,
    eps: float = LIK_EPS,
) -> torch.Tensor:
    """ZINB log-likelihood per entry, as scvi-tools' `log_zinb_positive`.

    Args:
        x: Counts, `(n, g)`.
        mu: Means, `(n, g)`.
        theta: Inverse dispersion, `(g,)`.
        pi: Dropout logits, `(n, g)`.
        lgamma_x1: `lgamma(x + 1)`.
        eps: Stability constant.

    Returns:
        Log-likelihoods, `(n, g)`.
    """
    softplus_pi = F.softplus(-pi)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (torch.log(theta + eps) - log_theta_mu_eps)
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - lgamma_x1
    )
    return (x < eps).float() * case_zero + (x > eps).float() * case_non_zero


@beartype
def kl_normal_std(qz_m: torch.Tensor, qz_v: torch.Tensor) -> torch.Tensor:
    """KL(N(m, v) || N(0, 1)) summed over the latent axis.

    Written the way `torch.distributions.kl_divergence` evaluates it for two
    `Normal`s with the scale as `sqrt(v)`, so the two agree to rounding.

    Args:
        qz_m: Posterior means, `(n, k)`.
        qz_v: Posterior variances, `(n, k)`.

    Returns:
        KL per cell, `(n,)`.
    """
    var_ratio = qz_v.sqrt().pow(2)
    return (0.5 * (var_ratio + qz_m.pow(2) - 1 - var_ratio.log())).sum(-1)


#############
# Modules   #
#############


class _FcLayers(nn.Module):
    """Linear, BatchNorm, ReLU, optional dropout, with one-hot batch injected.

    Mirrors `scvi.nn.FCLayers` with `inject_covariates=True`: the one-hot batch
    is concatenated onto the input of every linear layer. A single batch
    injects nothing, as scvi-tools ignores a one-category covariate.
    """

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
        blocks = []
        for i, (d_in, d_out) in enumerate(itertools.pairwise(dims)):
            parts: list[nn.Module] = [
                nn.Linear(d_in + n_cov, d_out),
                nn.BatchNorm1d(d_out, momentum=BN_MOMENTUM, eps=BN_EPS),
                nn.ReLU(),
            ]
            if dropout > 0:
                parts.append(nn.Dropout(p=dropout))
            blocks.append((f"Layer {i}", nn.Sequential(*parts)))
        self.fc_layers = nn.Sequential(collections.OrderedDict(blocks))

    def forward(self, x: torch.Tensor, cov: torch.Tensor | None) -> torch.Tensor:
        """Run the stack.

        Args:
            x: Input, `(n, n_in)`.
            cov: One-hot batch, `(n, n_cov)`, or `None` when `n_cov == 0`.

        Returns:
            Activations, `(n, n_out)`.
        """
        for block in self.fc_layers:
            if cov is not None:
                x = torch.cat((x, cov), dim=-1)
            x = block(x)
        return x


class _Encoder(nn.Module):
    """q(z | x): blind to batch, as scvi-tools' default `encode_covariates=False`."""

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior parameters.

        Args:
            x: `log1p` counts, `(n, g)`.

        Returns:
            `(mean, variance)`, each `(n, n_latent)`.
        """
        q = self.encoder(x, None)
        return self.mean_encoder(q), torch.exp(self.var_encoder(q)) + VAR_EPS


class _Decoder(nn.Module):
    """p(x | z, batch): softmax scale per gene, plus ZINB dropout logits."""

    def __init__(self, n_input: int, n_batch: int, params: ScviParams) -> None:
        """Build the decoder.

        Args:
            n_input: Genes.
            n_batch: Batches; one-hot injected when more than one.
            params: Hyperparameters.
        """
        super().__init__()
        n_cov = n_batch if n_batch > 1 else 0
        self.px_decoder = _FcLayers(
            params.n_latent, params.n_hidden, n_cov, params, 0.0
        )
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(params.n_hidden, n_input), nn.Softmax(dim=-1)
        )
        if params.gene_likelihood == "zinb":
            self.px_dropout_decoder = nn.Linear(params.n_hidden, n_input)

    def forward(
        self, z: torch.Tensor, cov: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode.

        Args:
            z: Latent, `(n, n_latent)`.
            cov: One-hot batch or `None`.

        Returns:
            `(px_scale, px_dropout)`; the second is `None` under `nb`.
        """
        px = self.px_decoder(z, cov)
        dropout = (
            self.px_dropout_decoder(px) if hasattr(self, "px_dropout_decoder") else None
        )
        return self.px_scale_decoder(px), dropout


class Vae(nn.Module):
    """scVI: NB or ZINB VAE with observed library size and gene-wise dispersion."""

    def __init__(self, n_input: int, n_batch: int, params: ScviParams) -> None:
        """Build the model.

        Args:
            n_input: Genes.
            n_batch: Batches.
            params: Hyperparameters.
        """
        super().__init__()
        self.n_batch = n_batch
        self.gene_likelihood = params.gene_likelihood
        self.px_r = nn.Parameter(torch.randn(n_input))
        self.z_encoder = _Encoder(n_input, params)
        self.decoder = _Decoder(n_input, n_batch, params)

    def _one_hot(self, batch: torch.Tensor) -> torch.Tensor | None:
        """One-hot batch codes, or `None` for a single batch.

        Args:
            batch: `int64` codes, `(n,)`.

        Returns:
            `(n, n_batch)` float one-hot, or `None`.
        """
        if self.n_batch <= 1:
            return None
        return F.one_hot(batch, self.n_batch).float()

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor, kl_w: float, sample: bool = True
    ) -> dict[str, torch.Tensor]:
        """Loss for one minibatch, as scvi-tools' `VAE.loss`.

        Args:
            x: Raw counts, `(n, g)`.
            batch: Batch codes, `(n,)`.
            kl_w: KL weight.
            sample: Decode a reparameterised draw of z, as scvi-tools always
                does. `False` decodes the posterior mean, for exact comparisons.

        Returns:
            `loss` (scalar, the mean over cells of reconstruction plus weighted
            KL), and per-cell `reconstruction`, `kl` and the posterior mean `qz_m`.
        """
        library = torch.log(x.sum(1, keepdim=True))
        qz_m, qz_v = self.z_encoder(torch.log1p(x))
        z = qz_m + qz_v.sqrt() * torch.randn_like(qz_m) if sample else qz_m
        px_scale, px_dropout = self.decoder(z, self._one_hot(batch))
        px_rate = torch.exp(library) * px_scale
        theta = torch.exp(self.px_r)
        lgamma_x1 = torch.lgamma(x + 1)
        if px_dropout is None:
            log_px = log_nb_positive(x, px_rate, theta, lgamma_x1)
        else:
            log_px = log_zinb_positive(x, px_rate, theta, px_dropout, lgamma_x1)
        reconstruction = -log_px.sum(-1)
        kl = kl_normal_std(qz_m, qz_v)
        return {
            "loss": torch.mean(reconstruction + kl_w * kl),
            "reconstruction": reconstruction,
            "kl": kl,
            "qz_m": qz_m,
        }


#############
# Training  #
#############


@beartype
def resolve_device(device: str) -> torch.device:
    """Pick the torch device.

    Args:
        device: `"auto"` (cuda, then mps, then cpu) or any torch device string.

    Returns:
        The device.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@beartype
def _batches(
    reader: CellReader,
    cells: np.ndarray,
    genes: np.ndarray,
    codes: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    """Minibatches moved onto the device.

    Args:
        reader: The store.
        cells: Store cell indices.
        genes: Store gene indices.
        codes: Batch code per cell.
        device: Target device.
        batch_size: Cells per batch.
        shuffle: Shuffle the order.
        seed: Shuffle seed.

    Yields:
        `(x, batch, cell_ids)`.
    """
    for x, b, ids in iterate_counts(
        reader, cells, genes, codes, batch_size=batch_size, shuffle=shuffle, seed=seed
    ):
        yield (
            torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(b).to(device, non_blocking=True),
            ids,
        )


@beartype
def train(
    reader: CellReader,
    cells: np.ndarray,
    genes: np.ndarray,
    codes: np.ndarray,
    categories: np.ndarray,
    *,
    params: ScviParams,
    device: str,
    seed: int,
) -> ScviFit:
    """Fit scVI with a plain Adam loop.

    Args:
        reader: The store.
        cells: Store cell indices to train on.
        genes: Store gene indices.
        codes: Batch code per cell.
        categories: Batch labels.
        params: Hyperparameters.
        device: Device string, see `resolve_device`.
        seed: Seeds the initialisation, the sampling and the shuffle.

    Returns:
        The fitted model, in eval mode.
    """
    dev = resolve_device(device)
    torch.manual_seed(seed)
    model = Vae(len(genes), len(categories), params).to(dev)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=params.lr,
        eps=params.eps,
        weight_decay=params.weight_decay,
    )
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
        totals = torch.zeros(3, device=dev)
        n_seen = 0
        for x, b, _ in _batches(
            reader,
            cells,
            genes,
            codes,
            dev,
            batch_size=params.batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            if x.shape[0] < MIN_TRAIN_BATCH:
                continue
            out = model(x, b, kl_w)
            optimiser.zero_grad(set_to_none=True)
            out["loss"].backward()
            optimiser.step()
            n = x.shape[0]
            # Accumulate on device: a `.item()` per step would sync every batch.
            totals += torch.stack(
                (
                    out["loss"].detach() * n,
                    out["reconstruction"].detach().sum(),
                    out["kl"].detach().sum(),
                )
            )
            n_seen += n
        loss, recon, kl = (totals / max(n_seen, 1)).tolist()
        history["loss"].append(loss)
        history["reconstruction"].append(recon)
        history["kl"].append(kl)
        history["kl_weight"].append(kl_w)
        logger.info("epoch %d/%d: loss %.2f", epoch + 1, n_epochs, loss)

    model.eval()
    return ScviFit(
        model=model,
        backend="torch",
        params=params,
        genes=genes,
        batch_categories=categories,
        device=str(dev),
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
    """Run the trained model forward, in cell order.

    Args:
        fit: A torch `ScviFit`.
        reader: The store.
        cells: Store cell indices.
        codes: Batch code per cell.
        seed: Seeds the draw of z.

    Returns:
        `latent` (`(n, n_latent)`, the posterior mean), and per-cell
        `reconstruction` (on one draw of z, as scvi-tools' `get_elbo`) and
        `kl` at KL weight 1.
    """
    torch.manual_seed(seed)
    model = fit.model
    model.eval()
    dev = torch.device(fit.device)
    latent, recon, kl = [], [], []
    with torch.inference_mode():
        for x, b, _ in _batches(
            reader,
            cells,
            fit.genes,
            codes,
            dev,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            seed=None,
        ):
            out = model(x, b, 1.0)
            latent.append(out["qz_m"].cpu().numpy())
            recon.append(out["reconstruction"].cpu().numpy())
            kl.append(out["kl"].cpu().numpy())
    return {
        "latent": np.concatenate(latent),
        "reconstruction": np.concatenate(recon),
        "kl": np.concatenate(kl),
    }
