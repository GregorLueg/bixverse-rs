"""scVI hyperparameters and the fitted-model record, free of any backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from beartype import beartype

###########
# Globals #
###########

Likelihood = Literal["nb", "zinb"]
Backend = Literal["torch", "mlx", "jax"]

#: scvi-tools' epoch heuristic: the cap, and the cell count it starts decaying at.
EPOCHS_CAP: int = 400
EPOCHS_DECAY_AT_N: int = 20_000

#: Training batches smaller than this are skipped. BatchNorm in training mode
#: is undefined on one cell and unstable on two; scvi-tools warns at the same
#: threshold.
MIN_TRAIN_BATCH: int = 3


@dataclass(frozen=True)
class ScviParams:
    """scVI hyperparameters.

    Defaults are scvi-tools 1.5.1's for `SCVI` plus its `TrainingPlan`, so a
    run with defaults on both sides is the same model and the same optimiser.

    Attributes:
        n_hidden: Nodes per hidden layer.
        n_latent: Latent dimensions.
        n_layers: Hidden layers in the encoder and in the decoder.
        dropout_rate: Encoder dropout. The decoder has none, as in scvi-tools.
        gene_likelihood: `"zinb"` (scvi-tools' default) or `"nb"`.
        lr: Adam learning rate.
        weight_decay: Adam L2 penalty (coupled, `torch.optim.Adam` style).
        eps: Adam epsilon. scvi-tools uses 0.01, far above the usual 1e-8.
        batch_size: Cells per training step.
        n_epochs_kl_warmup: Epochs over which the KL weight ramps linearly
            from 0 to 1.
        max_epochs: Training epochs. `None` takes scvi-tools' heuristic,
            `min(round(20000 / n_cells * 400), 400)`.
    """

    n_hidden: int = 128
    n_latent: int = 10
    n_layers: int = 1
    dropout_rate: float = 0.1
    gene_likelihood: Likelihood = "zinb"
    lr: float = 1e-3
    weight_decay: float = 1e-6
    eps: float = 0.01
    batch_size: int = 128
    n_epochs_kl_warmup: int = 400
    max_epochs: int | None = None


@dataclass
class ScviHistory:
    """Per-epoch training means, one entry per epoch.

    Attributes:
        loss: Minibatch loss, reconstruction plus weighted KL.
        reconstruction: Negative log likelihood per cell.
        kl: KL divergence of z per cell.
        kl_weight: The KL weight used that epoch.
    """

    loss: list[float] = field(default_factory=list)
    reconstruction: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    kl_weight: list[float] = field(default_factory=list)


@dataclass
class ScviFit:
    """A trained scVI model plus what is needed to use it again.

    Attributes:
        model: The backend model (`torch.nn.Module`, `mlx.nn.Module` or the
            JAX `Vae` record).
        backend: Which backend `model` belongs to.
        params: The hyperparameters it was trained with.
        genes: Store gene indices the model was trained on, in column order.
        batch_categories: Batch labels, in the order of the one-hot columns.
        device: Torch device string, `"gpu"` for MLX, or the JAX platform.
        history: Per-epoch training means. Empty for a fit wrapped around
            weights trained elsewhere.
    """

    model: Any  # torch, MLX or JAX model; none is importable here
    backend: Backend
    params: ScviParams
    genes: np.ndarray
    batch_categories: np.ndarray
    device: str
    history: ScviHistory = field(default_factory=ScviHistory)


@beartype
def resolve_max_epochs(params: ScviParams, n_cells: int) -> int:
    """Training epochs, applying scvi-tools' heuristic when unset.

    Args:
        params: Hyperparameters.
        n_cells: Cells the model is trained on.

    Returns:
        The number of epochs to run.
    """
    if params.max_epochs is not None:
        return params.max_epochs
    epochs = min(round(EPOCHS_DECAY_AT_N / n_cells * EPOCHS_CAP), EPOCHS_CAP)
    return max(epochs, 1)


@beartype
def kl_weight(epoch: int, n_epochs_kl_warmup: int) -> float:
    """Linear KL warmup by epoch, as scvi-tools' `_compute_kl_weight`.

    Args:
        epoch: Zero-based epoch.
        n_epochs_kl_warmup: Ramp length. Zero disables the warmup.

    Returns:
        The KL weight in `[0, 1]`.
    """
    if n_epochs_kl_warmup > 0 and epoch < n_epochs_kl_warmup:
        return epoch / n_epochs_kl_warmup
    return 1.0
