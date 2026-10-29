# scVI

`bixverse.nn` carries a barebone scVI (Lopez et al., Nature Methods, 2018) for
batch correction. It trains straight off a bixverse store: counts stream through
the Rust loader, so there's no AnnData, no Lightning and no registry in the way.
Three backends with the same network on all of them: torch (cuda, mps or cpu),
MLX on Apple Silicon and JAX (cpu or cuda).

```bash
pip install "bixverse[torch]"   # or bixverse[mlx] on Apple Silicon, bixverse[jax]
```

## Train and embed

You hand it the store, the cells to use, the genes to model and one batch label
per cell. Indices are the store's own, 0-based.

```python
import numpy as np
import bixverse.nn as bvnn

store = "counts_cells.bin"
cells = np.array([...])  # store cell indices, e.g. the QC survivors
hvg = np.array([...])  # store gene indices, e.g. 2000 HVGs
batch = np.array([...])  # one label per entry of `cells`

fit = bvnn.train_scvi(store, cells, hvg, batch, backend="torch", seed=42)
latent = bvnn.scvi_latent(fit, store, cells, batch)  # (n_cells, 10)
```

`latent` comes back in the order of `cells`, and is the posterior mean of z,
same as scvi-tools' `get_latent_representation()`. Feed it to whatever builds
your kNN graph.

Want a held-out score? Train on a subset of the cells and score the rest:

```python
bvnn.scvi_elbo(fit, store, test_cells, test_batch)
# {'elbo': ..., 'reconstruction': ..., 'kl': ...}, means per cell
```

## What's in, what's out

The defaults are scvi-tools 1.5.1's `SCVI` plus its `TrainingPlan`: ZINB
likelihood (NB via `ScviParams(gene_likelihood="nb")`), observed library size,
gene-wise dispersion, one hidden layer of 128, a 10-d latent, batch one-hot
into every decoder layer with the encoder blind to batch, Adam with lr 1e-3,
eps 0.01 and weight decay 1e-6, batch size 128, KL warmup over 400 epochs and
scvi-tools' epoch heuristic (`min(round(20000 / n_cells * 400), 400)`).

Not there, and not coming until something asks for it: Poisson and Gaussian
likelihoods, learned library size, batch-, label- or cell-wise dispersion,
extra covariates, labels (so no scANVI), minified data and scArches surgery.

Library size is the sum over the genes you pass, like scvi-tools on an HVG
AnnData. A cell with no counts over those genes raises rather than training on
`log(0)`.

## How close is it to scvi-tools?

Two levels of check.

**Same weights, same answer.** The tests load a trained scvi-tools 1.5.1 module
into ours, run both on one batch with z at the posterior mean, and compare. The
encoder mean, reconstruction term and KL agree to `rtol = 1e-5`, for NB and
ZINB. Our torch weights then go into the MLX model and agree to `1e-4` (MLX has
no `lgamma`, so it uses a Lanczos approximation), and into the JAX model, which
agrees to `1e-5`, BatchNorm running statistics included.

**Same training, same kind of result.** Every arm trains on the same cells of
the two PBMC batches from the bixverse batch correction vignette (6626 cells
after QC, 2000 HVGs, 90/10 split, so 5963 train and 663 held out), 400 epochs,
three seeds each, all in one sitting on an M1 Max:

| arm | device | train, s | held-out ELBO | kBET accept | iLISI | Procrustes | kNN Jaccard | Leiden ARI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scvi-tools `SCVI` | cpu | 260 | 769.2 | 0.684 | 1.480 | 0.489 | 0.084 | 0.741 |
| scvi-tools `SCVI` | mps | 194 | 768.4 | 0.687 | 1.482 | 0.446 | 0.094 | 0.727 |
| scvi-tools `mlxSCVI` | gpu | 124 | | 0.846 | 1.605 | 0.526 | 0.082 | 0.724 |
| bixverse torch | cpu | 202 | 768.8 | 0.689 | 1.486 | 0.469 | 0.088 | 0.718 |
| bixverse torch | mps | 82 | 769.2 | 0.686 | 1.484 | 0.491 | 0.086 | 0.714 |
| bixverse JAX | cpu | 106 | 768.3 | 0.694 | 1.491 | 0.475 | 0.087 | 0.721 |
| bixverse MLX | gpu | 35 | 768.3 | 0.667 | 1.473 | 0.458 | 0.091 | 0.711 |
| uncorrected PCA | | | | 0.013 | 1.014 | | | |
| Harmony | | | | 0.883 | 1.622 | | | |

Means over three seeds. Held-out ELBO is the per-cell negative ELBO on the
held-out cells, every arm scored by the same code (scvi-tools weights loaded
into ours); its seed-to-seed range is up to about 2. kBET and iLISI come from
bixverse's `calculate_integration_metrics_sc` and `calculate_lisi_sc`.

The last three columns compare each arm against scvi-tools on cpu: Procrustes
disparity (0 is identical), the mean Jaccard of each cell's 15 nearest
neighbours, and the ARI between Leiden clusterings, averaged over all pairs of
seeds. VAE latents aren't identifiable, so two seeds of scvi-tools already
disagree with each other, and the scvi-tools cpu row is exactly that: its own
seed-to-seed spread. Ours sits inside it on every measure. A Procrustes
disparity near 0.5 between two runs of the *same* code is also the reason a
small Procrustes delta is the wrong thing to ask for.

On speed: MLX is the crown on Apple Silicon. torch on MPS beats scvi-tools on
MPS by more than 2x; the network is the same, so the gap sits outside the
maths, mostly `torch.distributions` argument checks that stall the GPU queue
every step, plus Lightning's per-step bookkeeping. Data loading costs the same
on both sides. No GPU? Take JAX: it's almost twice as fast as torch on the same
CPU, for the same ELBO. That was scvi-tools' original pitch for `JaxSCVI` back
in 0.15, and it still holds for this network.

scvi-tools' own `mlxSCVI` is a different network (two hidden layers, BatchNorm
momentum 0.9, NB or Poisson only), so it's not a port of `SCVI` and has no
held-out ELBO here. It integrates harder on these two batches, closer to
Harmony than to `SCVI`.

## Caveats

- scvi-tools' `accelerator="auto"` ignores MPS. Ask for `"mps"` explicitly if
  you benchmark against it.
- The MLX backend compiles its training step with `mx.compile`. On this data
  that took 400 epochs from 82 s to 50 s.
- `torch.compile` isn't used. On MPS it cut compute from 0.175 s to 0.136 s
  per epoch but costs about 5 s up front, and the inductor backend there is
  still experimental.
- JAX on Apple's GPU goes through the `jax-metal` plugin, which only runs
  pinned to jax 0.5.0. It trains, but took about 180 s on this data, slower
  than JAX on CPU, so it isn't supported. Use MLX there.
- The JAX training step compiles twice per fit, once for the full batch size
  and once for the short final batch.
- There's no torch `DataLoader` in the loop: the Rust feeder thread already
  decodes ahead of the training step.
