# scVI

`bixverse.nn` carries a barebone scVI (Lopez et al., Nature Methods, 2018) for
batch correction. It trains straight off a bixverse store: counts stream through
the Rust loader, so there's no AnnData, no Lightning and no registry in the way.
Two backends, torch (cuda, mps or cpu) and MLX on Apple Silicon, with the same
network on both.

```bash
pip install "bixverse[torch]"   # or bixverse[mlx] on Apple Silicon
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
no `lgamma`, so it uses a Lanczos approximation).

**Same training, same kind of result.** On the two PBMC batches from the
bixverse batch correction vignette (5841 cells after QC, 2000 HVGs, 90/10
split, three seeds each, M1 Max), each arm trains on the same 5257 cells:

| arm | device | train, s | held-out ELBO | kBET accept | iLISI |
| --- | --- | --- | --- | --- | --- |
| scvi-tools `SCVI` | cpu | 243 | 754.8 | 0.645 | 1.475 |
| scvi-tools `SCVI` | mps | 170 | 754.5 | 0.653 | 1.484 |
| bixverse torch | cpu | 220 | 754.7 | 0.636 | 1.476 |
| bixverse torch | mps | 74 | 753.6 | 0.650 | 1.484 |
| bixverse MLX | gpu | 31 | 753.9 | 0.623 | 1.463 |
| uncorrected PCA | | | | 0.013 | 1.013 |
| Harmony | | | | 0.857 | 1.621 |

Means over three seeds. Held-out ELBO is the per-cell negative ELBO on the 584
held-out cells, every arm scored by the same code (scvi-tools weights loaded
into ours); its seed-to-seed range is about 2. kBET and iLISI come from
bixverse's `calculate_integration_metrics_sc` and `calculate_lisi_sc`. The
network is the same on both sides, so the MPS gap sits outside the maths;
where scvi-tools spends it (Lightning, the AnnData loader) hasn't been
profiled.

VAE latents aren't identifiable, so two seeds of scvi-tools already disagree
with each other. The fair comparison is ours against scvi-tools, read against
scvi-tools against itself:

| pair (scvi-tools on cpu) | Procrustes disparity | kNN Jaccard (k = 15) | Leiden ARI |
| --- | --- | --- | --- |
| scvi-tools vs scvi-tools | 0.470 | 0.085 | 0.689 |
| bixverse torch (mps) vs scvi-tools | 0.463 | 0.085 | 0.683 |
| bixverse torch (cpu) vs scvi-tools | 0.482 | 0.082 | 0.690 |
| bixverse MLX vs scvi-tools | 0.472 | 0.085 | 0.683 |

Means over all pairs of runs, three seeds a side.

Ours sits inside scvi-tools' own seed-to-seed spread on every measure. A
Procrustes disparity near 0.47 between two runs of the *same* code is also the
reason a small Procrustes delta is the wrong thing to ask for.

scvi-tools' own `mlxSCVI` is a different network (two hidden layers, BatchNorm
momentum 0.9, NB or Poisson only), so it's not a port of `SCVI` and doesn't
land on the same embedding. It integrates harder on these two batches.

## Caveats

- scvi-tools' `accelerator="auto"` ignores MPS. Ask for `"mps"` explicitly if
  you benchmark against it.
- The MLX backend compiles its training step with `mx.compile`. On this data
  that took 400 epochs from 82 s to 50 s.
- `torch.compile` isn't used. On MPS it cut compute from 0.175 s to 0.136 s
  per epoch but costs about 5 s up front, and the inductor backend there is
  still experimental.
- There's no torch `DataLoader` in the loop: the Rust feeder thread already
  decodes ahead of the training step.
