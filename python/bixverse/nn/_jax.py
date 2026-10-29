"""scVI on JAX, with optax for the optimiser.

The same model as `bixverse.nn._torch`, layer for layer, written as pure
functions over parameter pytrees: no flax. A torch `state_dict` converts with
`torch_state_to_jax`, so the backends can be checked against each other. The
whole training step (forward, gradient, Adam) is one `jax.jit` call.

References:
    Lopez et al., Nature Methods, 2018.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from beartype import beartype
from jax.scipy.special import gammaln

from .._bixverse import CellReader
from ._data import iterate_counts
from ._params import (
    MIN_TRAIN_BATCH,
    ScviFit,
    ScviHistory,
    ScviParams,
    kl_weight,
    resolve_max_epochs,
)

logger = logging.getLogger(__name__)

###########
# Globals #
###########

#: Same as the torch backend, torch momentum convention: the weight on the
#: new batch statistic.
BN_MOMENTUM: float = 0.01
BN_EPS: float = 1e-3
VAR_EPS: float = 1e-4
LIK_EPS: float = 1e-8
EVAL_BATCH_SIZE: int = 1024

###################
# Likelihoods     #
###################


@beartype
def log_nb_positive(
    x: jax.Array, mu: jax.Array, theta: jax.Array, lgamma_x1: jax.Array
) -> jax.Array:
    """NB log-likelihood per entry; see the torch backend.

    Args:
        x: Counts, `(n, g)`.
        mu: Means, `(n, g)`.
        theta: Inverse dispersion, `(g,)`.
        lgamma_x1: `lgamma(x + 1)`.

    Returns:
        Log-likelihoods, `(n, g)`.
    """
    log_theta_mu_eps = jnp.log(theta + mu + LIK_EPS)
    return (
        theta * (jnp.log(theta + LIK_EPS) - log_theta_mu_eps)
        + x * (jnp.log(mu + LIK_EPS) - log_theta_mu_eps)
        + gammaln(x + theta)
        - gammaln(theta)
        - lgamma_x1
    )


@beartype
def log_zinb_positive(
    x: jax.Array, mu: jax.Array, theta: jax.Array, pi: jax.Array, lgamma_x1: jax.Array
) -> jax.Array:
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
    softplus_pi = jax.nn.softplus(-pi)
    log_theta_mu_eps = jnp.log(theta + mu + LIK_EPS)
    pi_theta_log = -pi + theta * (jnp.log(theta + LIK_EPS) - log_theta_mu_eps)
    case_zero = jax.nn.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (jnp.log(mu + LIK_EPS) - log_theta_mu_eps)
        + gammaln(x + theta)
        - gammaln(theta)
        - lgamma_x1
    )
    return jnp.where(x < LIK_EPS, case_zero, 0.0) + jnp.where(
        x > LIK_EPS, case_non_zero, 0.0
    )


###########
# Model   #
###########


@dataclass
class Vae:
    """scVI on JAX: parameters, BatchNorm statistics and the batch count.

    Attributes:
        params: Trainable pytree. `px_r`, then `z_encoder` and `decoder`, each
            a nest of `{"w", "b"}` linears (`w` as `(d_in, d_out)`) and, per
            hidden block, `{"linear", "bn": {"scale", "bias"}}`.
        state: BatchNorm running `{"mean", "var"}` per hidden block, under
            `z_encoder` and `decoder`.
        n_batch: Batches; one-hot injected into the decoder when more than one.
    """

    params: dict
    state: dict
    n_batch: int


@beartype
def _init_linear(key: jax.Array, d_in: int, d_out: int) -> dict:
    """A linear layer initialised as torch's `nn.Linear`: U(+-1/sqrt(d_in)).

    Args:
        key: PRNG key.
        d_in: Input width.
        d_out: Output width.

    Returns:
        `{"w": (d_in, d_out), "b": (d_out,)}`.
    """
    bound = 1.0 / np.sqrt(d_in)
    kw, kb = jax.random.split(key)
    return {
        "w": jax.random.uniform(kw, (d_in, d_out), minval=-bound, maxval=bound),
        "b": jax.random.uniform(kb, (d_out,), minval=-bound, maxval=bound),
    }


@beartype
def _init_stack(
    key: jax.Array, n_in: int, n_out: int, n_cov: int, params: ScviParams
) -> tuple[list[dict], list[dict]]:
    """Linear plus BatchNorm blocks with `n_cov` one-hot inputs per layer.

    Args:
        key: PRNG key.
        n_in: Input width.
        n_out: Output width.
        n_cov: One-hot width injected per layer, 0 for none.
        params: Hyperparameters.

    Returns:
        `(blocks, bn_state)`, one entry per block.
    """
    dims = [n_in] + [params.n_hidden] * (params.n_layers - 1) + [n_out]
    keys = jax.random.split(key, len(dims) - 1)
    blocks, state = [], []
    for k, (d_in, d_out) in zip(keys, itertools.pairwise(dims), strict=True):
        blocks.append(
            {
                "linear": _init_linear(k, d_in + n_cov, d_out),
                "bn": {"scale": jnp.ones(d_out), "bias": jnp.zeros(d_out)},
            }
        )
        state.append({"mean": jnp.zeros(d_out), "var": jnp.ones(d_out)})
    return blocks, state


@beartype
def init_vae(key: jax.Array, n_input: int, n_batch: int, params: ScviParams) -> Vae:
    """A freshly initialised model, initialised as the torch one.

    Args:
        key: PRNG key.
        n_input: Genes.
        n_batch: Batches.
        params: Hyperparameters.

    Returns:
        The model.
    """
    k = jax.random.split(key, 7)
    n_cov = n_batch if n_batch > 1 else 0
    enc, enc_state = _init_stack(k[0], n_input, params.n_hidden, 0, params)
    dec, dec_state = _init_stack(k[1], params.n_latent, params.n_hidden, n_cov, params)
    decoder = {
        "px_decoder": dec,
        "px_scale_decoder": _init_linear(k[2], params.n_hidden, n_input),
    }
    if params.gene_likelihood == "zinb":
        decoder["px_dropout_decoder"] = _init_linear(k[3], params.n_hidden, n_input)
    return Vae(
        params={
            "px_r": jax.random.normal(k[4], (n_input,)),
            "z_encoder": {
                "encoder": enc,
                "mean_encoder": _init_linear(k[5], params.n_hidden, params.n_latent),
                "var_encoder": _init_linear(k[6], params.n_hidden, params.n_latent),
            },
            "decoder": decoder,
        },
        state={"z_encoder": enc_state, "decoder": dec_state},
        n_batch=n_batch,
    )


@beartype
def _linear(p: dict, x: jax.Array) -> jax.Array:
    """Apply a linear layer.

    Args:
        p: `{"w", "b"}`.
        x: Input, `(n, d_in)`.

    Returns:
        `(n, d_out)`.
    """
    return x @ p["w"] + p["b"]


@beartype
def _stack(
    blocks: list[dict],
    state: list[dict],
    x: jax.Array,
    cov: jax.Array | None,
    key: jax.Array,
    *,
    dropout: float,
    train: bool,
) -> tuple[jax.Array, list[dict]]:
    """Linear, BatchNorm, ReLU, optional dropout, per block; as `_FcLayers`.

    In training BatchNorm normalises by the biased batch variance and folds
    the unbiased one into the running statistics, as torch's `BatchNorm1d`.

    Args:
        blocks: Block parameters.
        state: Running statistics per block.
        x: Input.
        cov: One-hot batch, or `None`.
        key: PRNG key for dropout.
        dropout: Dropout rate; 0 for none.
        train: Batch statistics and dropout on, rather than running statistics.

    Returns:
        `(activations, new_state)`.
    """
    new_state = []
    keys = jax.random.split(key, len(blocks))
    for blk, st, k in zip(blocks, state, keys, strict=True):
        if cov is not None:
            x = jnp.concatenate([x, cov], axis=-1)
        h = _linear(blk["linear"], x)
        if train:
            mean = h.mean(axis=0)
            var = h.var(axis=0)
            n = h.shape[0]
            st = {
                "mean": (1 - BN_MOMENTUM) * st["mean"] + BN_MOMENTUM * mean,
                "var": (1 - BN_MOMENTUM) * st["var"]
                + BN_MOMENTUM * var * (n / (n - 1)),
            }
        else:
            mean, var = st["mean"], st["var"]
        h = (h - mean) / jnp.sqrt(var + BN_EPS) * blk["bn"]["scale"] + blk["bn"]["bias"]
        x = jax.nn.relu(h)
        if train and dropout > 0:
            keep = jax.random.bernoulli(k, 1.0 - dropout, x.shape)
            x = jnp.where(keep, x / (1.0 - dropout), 0.0)
        new_state.append(st)
    return x, new_state


@beartype
def forward(
    params: dict,
    state: dict,
    x: jax.Array,
    batch: jax.Array,
    kl_w: float | jax.Array,
    key: jax.Array,
    *,
    n_batch: int,
    dropout: float,
    train: bool,
    sample: bool = True,
) -> tuple[dict[str, jax.Array], dict]:
    """Loss for one minibatch; see the torch `Vae.forward`.

    Args:
        params: Trainable pytree, as `Vae.params`.
        state: BatchNorm statistics, as `Vae.state`.
        x: Raw counts, `(n, g)`.
        batch: Batch codes, `(n,)`.
        kl_w: KL weight.
        key: PRNG key for dropout and the draw of z.
        n_batch: Batches.
        dropout: Encoder dropout rate.
        train: Training mode for BatchNorm and dropout.
        sample: Decode a draw of z rather than the posterior mean.

    Returns:
        `(out, new_state)`: `out` holds `loss`, per-cell `reconstruction` and
        `kl`, and `qz_m`; `new_state` the updated BatchNorm statistics.
    """
    k_enc, k_z, k_dec = jax.random.split(key, 3)
    enc, dec = params["z_encoder"], params["decoder"]
    library = jnp.log(x.sum(axis=1, keepdims=True))
    q, enc_state = _stack(
        enc["encoder"],
        state["z_encoder"],
        jnp.log1p(x),
        None,
        k_enc,
        dropout=dropout,
        train=train,
    )
    qz_m = _linear(enc["mean_encoder"], q)
    qz_v = jnp.exp(_linear(enc["var_encoder"], q)) + VAR_EPS
    z = qz_m + jnp.sqrt(qz_v) * jax.random.normal(k_z, qz_m.shape) if sample else qz_m
    cov = jax.nn.one_hot(batch, n_batch, dtype=x.dtype) if n_batch > 1 else None
    px, dec_state = _stack(
        dec["px_decoder"], state["decoder"], z, cov, k_dec, dropout=0.0, train=train
    )
    px_rate = jnp.exp(library) * jax.nn.softmax(
        _linear(dec["px_scale_decoder"], px), axis=-1
    )
    theta = jnp.exp(params["px_r"])
    lgamma_x1 = gammaln(x + 1)
    if "px_dropout_decoder" in dec:
        pi = _linear(dec["px_dropout_decoder"], px)
        log_px = log_zinb_positive(x, px_rate, theta, pi, lgamma_x1)
    else:
        log_px = log_nb_positive(x, px_rate, theta, lgamma_x1)
    reconstruction = -log_px.sum(axis=-1)
    # sqrt then square, as `torch.distributions.kl_divergence` evaluates it.
    var_ratio = jnp.square(jnp.sqrt(qz_v))
    kl = (0.5 * (var_ratio + jnp.square(qz_m) - 1 - jnp.log(var_ratio))).sum(axis=-1)
    out = {
        "loss": jnp.mean(reconstruction + kl_w * kl),
        "reconstruction": reconstruction,
        "kl": kl,
        "qz_m": qz_m,
    }
    return out, {"z_encoder": enc_state, "decoder": dec_state}


################
# Conversion   #
################

#: `<prefix>.fc_layers.Layer <i>.<0 linear | 1 BatchNorm>.<field>`.
_TORCH_BLOCK = re.compile(r"^(.*)\.fc_layers\.Layer (\d+)\.([01])\.(\w+)$")


@beartype
def torch_state_to_jax(state: dict, n_batch: int) -> Vae:
    """Convert a torch `Vae` (or matching scvi-tools) `state_dict` for JAX.

    Args:
        state: Torch tensors keyed as in `bixverse.nn._torch.Vae`.
        n_batch: Batches the model was built with.

    Returns:
        The JAX model, with the same weights and BatchNorm statistics.
    """
    arr = {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in state.items()}

    def linear(prefix: str) -> dict:
        """A torch `nn.Linear`, transposed to `(d_in, d_out)`."""
        return {"w": arr[f"{prefix}.weight"].T, "b": arr[f"{prefix}.bias"]}

    def stack(prefix: str) -> tuple[list[dict], list[dict]]:
        """The blocks and running statistics of one `_FcLayers`."""
        idx = sorted(
            {int(m[2]) for k in arr if (m := _TORCH_BLOCK.match(k)) and m[1] == prefix}
        )
        blocks, stats = [], []
        for i in idx:
            base = f"{prefix}.fc_layers.Layer {i}"
            blocks.append(
                {
                    "linear": linear(f"{base}.0"),
                    "bn": {
                        "scale": arr[f"{base}.1.weight"],
                        "bias": arr[f"{base}.1.bias"],
                    },
                }
            )
            stats.append(
                {
                    "mean": arr[f"{base}.1.running_mean"],
                    "var": arr[f"{base}.1.running_var"],
                }
            )
        return blocks, stats

    enc, enc_state = stack("z_encoder.encoder")
    dec, dec_state = stack("decoder.px_decoder")
    decoder = {
        "px_decoder": dec,
        "px_scale_decoder": linear("decoder.px_scale_decoder.0"),
    }
    if "decoder.px_dropout_decoder.weight" in arr:
        decoder["px_dropout_decoder"] = linear("decoder.px_dropout_decoder")
    return Vae(
        params={
            "px_r": arr["px_r"],
            "z_encoder": {
                "encoder": enc,
                "mean_encoder": linear("z_encoder.mean_encoder"),
                "var_encoder": linear("z_encoder.var_encoder"),
            },
            "decoder": decoder,
        },
        state={"z_encoder": enc_state, "decoder": dec_state},
        n_batch=n_batch,
    )


@beartype
def n_parameters(model: Vae) -> int:
    """Trainable parameter count, for checking a conversion is complete.

    Args:
        model: A JAX `Vae`.

    Returns:
        Number of scalars across trainable parameters.
    """
    return sum(leaf.size for leaf in jax.tree.leaves(model.params))


#############
# Training  #
#############


@beartype
def resolve_device(device: str) -> jax.Device:
    """Pick the JAX device.

    Args:
        device: `"auto"` for JAX's default backend, or a platform name such
            as `"cpu"`, `"gpu"` or `"METAL"`.

    Returns:
        The first device of that platform.
    """
    return jax.devices()[0] if device == "auto" else jax.devices(device)[0]


@beartype
def _batches(
    reader: CellReader,
    cells: np.ndarray,
    genes: np.ndarray,
    codes: np.ndarray,
    device: jax.Device,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
) -> Iterator[tuple[jax.Array, jax.Array, np.ndarray]]:
    """Minibatches placed on the device.

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
            jax.device_put(x, device),
            jax.device_put(b.astype(np.int32), device),
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
    """Fit scVI on JAX with the same Adam loop as the torch backend.

    Coupled L2 as torch's `Adam(weight_decay=...)`: the decay goes onto the
    gradient before the moments, hence `add_decayed_weights` ahead of `adam`.
    The final short batch has its own shape, so the step compiles twice.

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
        The fitted model.
    """
    dev = resolve_device(device)
    k_init, key = jax.random.split(jax.random.key(seed))
    model = init_vae(k_init, len(genes), len(categories), params)
    tx = optax.chain(
        optax.add_decayed_weights(params.weight_decay),
        optax.adam(params.lr, eps=params.eps),
    )
    weights, bn_state, key = jax.device_put((model.params, model.state, key), dev)
    opt_state = tx.init(weights)
    fwd = partial(
        forward, n_batch=model.n_batch, dropout=params.dropout_rate, train=True
    )

    @partial(jax.jit, donate_argnums=(0, 1, 2, 3))
    def step(
        weights: dict,
        bn_state: dict,
        opt_state: optax.OptState,
        key: jax.Array,
        x: jax.Array,
        b: jax.Array,
        kl_w: jax.Array,
    ) -> tuple[dict, dict, optax.OptState, jax.Array, jax.Array]:
        """One Adam step; returns the new state, key and `[loss * n, recon, kl]`."""
        key, sub = jax.random.split(key)

        def loss_fn(w: dict) -> tuple[jax.Array, tuple[dict, jax.Array]]:
            """Scalar loss, with the new BatchNorm state and summed terms."""
            out, new_state = fwd(w, bn_state, x, b, kl_w, sub)
            terms = jnp.stack(
                [
                    out["loss"] * x.shape[0],
                    out["reconstruction"].sum(),
                    out["kl"].sum(),
                ]
            )
            return out["loss"], (new_state, terms)

        (_, (new_state, terms)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            weights
        )
        updates, opt_state = tx.update(grads, opt_state, weights)
        new_weights = cast(dict, optax.apply_updates(weights, updates))
        return new_weights, new_state, opt_state, key, terms

    n_epochs = resolve_max_epochs(params, len(cells))
    history = ScviHistory()
    for epoch in range(n_epochs):
        kl_w = kl_weight(epoch, params.n_epochs_kl_warmup)
        kl_w_arr = jax.device_put(jnp.float32(kl_w), dev)
        totals = jax.device_put(jnp.zeros(3), dev)
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
            n = x.shape[0]
            if n < MIN_TRAIN_BATCH:
                continue
            weights, bn_state, opt_state, key, terms = step(
                weights, bn_state, opt_state, key, x, b, kl_w_arr
            )
            # Accumulate on device; dispatch stays asynchronous until the epoch end.
            totals = totals + terms
            n_seen += n
        loss, recon, kl = (np.asarray(totals) / max(n_seen, 1)).tolist()
        history.loss.append(loss)
        history.reconstruction.append(recon)
        history.kl.append(kl)
        history.kl_weight.append(kl_w)
        logger.info("epoch %d/%d: loss %.2f", epoch + 1, n_epochs, loss)

    return ScviFit(
        model=Vae(params=weights, state=bn_state, n_batch=model.n_batch),
        backend="jax",
        params=params,
        genes=genes,
        batch_categories=categories,
        device=dev.platform,
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
        fit: A JAX `ScviFit`.
        reader: The store.
        cells: Store cell indices.
        codes: Batch code per cell.
        seed: Seeds the draw of z.

    Returns:
        `latent`, and per-cell `reconstruction` and `kl`.
    """
    dev = resolve_device(fit.device)
    model = fit.model
    fwd = jax.jit(
        partial(
            forward,
            n_batch=model.n_batch,
            dropout=fit.params.dropout_rate,
            train=False,
        )
    )
    weights, bn_state = jax.device_put((model.params, model.state), dev)
    key = jax.random.key(seed)
    latent, recon, kl = [], [], []
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
        key, sub = jax.random.split(key)
        out, _ = fwd(weights, bn_state, x, b, 1.0, sub)
        latent.append(np.asarray(out["qz_m"]))
        recon.append(np.asarray(out["reconstruction"]))
        kl.append(np.asarray(out["kl"]))
    return {
        "latent": np.concatenate(latent),
        "reconstruction": np.concatenate(recon),
        "kl": np.concatenate(kl),
    }
