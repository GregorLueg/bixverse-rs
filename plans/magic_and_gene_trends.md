# To-do: MAGIC imputation and gene trends

Two work pieces, deliberately decoupled. MAGIC is a general denoiser that happens to be
useful to Palantir; gene trends are a Palantir output that should take whatever expression
matrix the caller hands over, imputed or not. Neither should import the other.

Everything below was read out of the installed reference
(`palantir/utils.py`, `palantir/presults.py`, `mellon/`), not recalled. Line numbers refer
to those files.

## Piece 1: MAGIC imputation

### What the reference does

`run_magic_imputation` (`utils.py:731`) is three lines of maths wrapped in a lot of AnnData
plumbing:

1. `T = diag(1 / rowsum(K)) @ K`, the row-stochastic diffusion operator, built in
   `run_diffusion_maps` (`utils.py:482-484`) from the same adaptive-bandwidth kernel
   Palantir already uses.
2. `T_steps = T ** n_steps`, a genuine sparse matrix power, `n_steps = 3` by default.
3. `imputed = T_steps @ X`, run over blocks of 100 gene columns, then values below
   `clip_threshold = 1e-2` are zeroed and the result optionally kept sparse.

### The one thing to do differently

**Never form `T^3`.** With `knn = 30` the operator has ~30 non-zeros per row, `T²` has
~900 and `T³` ~27,000, so at 100k cells the explicit power is on the order of 2.7e9
non-zeros. Apply the operator three times instead: `T @ (T @ (T @ X))`. Identical result,
and it is exactly the pattern the crate's own conventions already mandate.

The memory wall then moves to the intermediate, which densifies fast even when `X` is
sparse counts. At 100k cells by 20k genes an `f32` intermediate is 8 GB, so the column
blocking in the reference is not an optimisation, it is what makes it run. Keep it: process
genes in column blocks, three passes per block, peak memory `n_cells × block_width`.

### Shape of the work

- Generic primitive: apply a CSR operator `t` times to a dense column block. Belongs next
  to the existing sparse kernels in `src/core/math/sparse.rs`, which already has
  `csr_sparse_matmul_dense` and `csr_matmul_csr` to build on.
- Single-cell entry point: `src/single_cell/sc_processing/magic.rs`, holding the kernel to
  operator step, the blocking, the clip, and a `MagicParams` with `n_steps` and
  `clip_threshold`. Same split as PAGA: generic core in a generic module, thin
  single-cell wrapper.
- Reuse `compute_diffusion_kernel` (`mc_generation/seacells.rs`) for `K`. Note it returns
  the **symmetric** kernel and `multiscale_components` then normalises it in place to
  `D^-1/2 K D^-1/2` for the eigensolve. MAGIC needs `D^-1 K` instead, so take the row sums
  before any normalisation, which `diffusion.rs::kernel_row_sums` already does.
- Should stream from the disk-backed store rather than materialising all genes. The block
  structure lines up with `CELL_BATCH_SIZE` and the existing chunked readers.

### Watch out for

- Row-stochastic means the operator preserves total mass per cell, so imputed values are
  on the same scale as the input. Feeding it raw counts and feeding it log-normalised data
  give very different things; the reference quietly uses whatever is in `.X`. Make the
  expected input explicit in the docs.
- The clip at `1e-2` is applied once at the end, so it is not a per-pass truncation and the
  three-pass form does not change it.
- Zero-degree rows: the reference guards with `D[D != 0] = 1 / D[D != 0]`, leaving a zero
  row rather than a NaN. Match that, or error, but do it deliberately.

## Piece 2: Gene trends

### What the reference does

`compute_gene_trends` (`presults.py:210`), per branch:

1. Take the branch mask, `pt = pseudotime[mask]`.
2. `grid = linspace(pt.min(), pt.max(), 500)` (`PSEUDOTIME_RES = 500`).
3. `mellon.FunctionEstimator(sigma=1, ls=1, landmarks=grid)`, then
   `fit_predict(pt, expr[mask], grid).T`.

So: a 1-D Gaussian process posterior mean, multi-output over genes, with the prediction
grid doubling as the landmark set.

### The maths, pinned down

Mellon's default covariance is Matérn-5/2 (`mellon/base_model.py:49`), prior mean `mu = 0`.
The landmark conditional mean (`mellon/conditional.py:455-546`, `_sparse_solve` at `:57`)
is the standard subset-of-regressors posterior:

```
Lp      = chol(k(u, u) + jitter·I)          # u = landmarks, 500 x 500
A       = Lp⁻¹ k(u, x)                      # 500 x n_cells, triangular solve
r       = y - mu                            # n_cells x n_genes
r_l     = r / sigma²,  A_l = A / sigma²
L_B     = chol(A_l Aᵀ + I)                  # 500 x 500
c       = L_B⁻¹ (A r_l)                     # 500 x n_genes
weights = Lp⁻ᵀ L_B⁻ᵀ c                      # 500 x n_genes
trend   = mu + k(grid, u) @ weights         # 500 x n_genes
```

This is a good fit for the crate. Two Cholesky factorisations of a fixed 500 x 500, and
everything else is GEMM, so `faer` does the work. Genes enter only through `A @ r_l` and
the final product, both of which parallelise over gene blocks. Since the landmarks *are*
the prediction grid, `k(grid, u)` is the same matrix as `k(u, u)`: compute it once.

### Honest caveat, worth putting in the module docs

The defaults are why the plots look good, and they are also why they are a bit dodgy. With
pseudotime min-max scaled to `[0, 1]`, a Matérn-5/2 length scale of `ls = 1` spans the whole
domain, and `sigma = 1` sets the noise standard deviation at roughly the signal scale of
log-normalised expression. The posterior is therefore dominated by the prior: it will
flatten genuine transient structure and make almost any gene resolve to a smooth monotone
or single-peaked curve. That is a presentation choice, not an inference.

Expose `ls` and `sigma` as first-class parameters carrying the reference defaults, and say
plainly in the docs what they do. A user who wants to see whether a bump is real needs to
be able to shorten the length scale.

### Placement

- The landmark GP regressor is not single-cell specific. It belongs somewhere generic
  (`src/ml/` or `src/core/math/`) so it can be reused, with Matérn-5/2 as one kernel among
  whatever else gets added.
- `src/single_cell/sc_trajectory/gene_trends.rs` holds the per-branch loop, the grid, and
  the result type.
- Takes an expression matrix as an argument. It must not call MAGIC itself. Whether the
  caller passes raw, log-normalised or imputed data is their business.

## Prerequisite: branch masks

Gene trends need per-branch cell masks, which the crate does not have. The reference is
`select_branch_cells` (`presults.py:554`), and it derives them from the fate probabilities
Palantir already returns:

- NaN fate probabilities are replaced by `1 / n_fates`.
- Cells are walked in pseudotime order; a rolling `1 - q` quantile of the fate
  probabilities seen so far gives a per-fate threshold, which is then made monotone with a
  cumulative maximum.
- `mask = threshold - eps < fate_prob`, with `q = 1e-2` and `eps = 1e-2`.

I read the shape of this loop but not closely enough to transcribe the rolling-quantile
schedule exactly. **Read `presults.py:558-580` line by line before implementing it**; the
window growth is the part that will be got wrong from a summary.

Note the legacy path (`compute_gene_trends_legacy`, `presults.py:80`) fits a GAM using the
fate probabilities as *weights* rather than thresholding them into hard masks. That is
arguably the more honest treatment, since a cell with 0.6 probability on a branch is not a
member of it. Worth keeping in mind as an alternative rather than assuming the newer path
is the better one.

## Open questions

1. **"Whatever the user wants to use".** I read this as the gene-trends function accepting
   any expression matrix, which is what the decoupling above assumes. If you meant the
   *estimator* should be pluggable (GP now, GAM or LOESS later), that is a different and
   larger design: it needs a trait over "fit a smooth 1-D trend to a multi-output response"
   and changes the module layout. Say which.
2. **Does MAGIC need to reuse the Palantir kernel, or its own?** Sharing means a caller who
   wants imputation alone still pays for a kNN search and a kernel build. That is fine, but
   it argues for MAGIC taking a prebuilt operator as its primary entry point with a
   convenience constructor from kNN output.
3. **Sparse or dense output for MAGIC.** The reference offers both via a `sparse` flag. The
   disk-backed store has opinions here; worth deciding before writing the entry point.
4. **Where does the GP live.** `src/ml/` has clustering only today, so a `gp` submodule is
   new ground. `src/core/math/` is the alternative and is where the linear algebra already
   lives.

## Not in scope

- Mellon's density estimator (`run_density`, `utils.py:218`) and the dimensionality
  estimator. Different algorithms that happen to ship in the same package.
- `cluster_gene_trends` (`presults.py:473`). Clustering the trends is a downstream
  convenience and the crate already has the clustering to do it in R.
- Anything JAX-shaped. Mellon optimises hyperparameters by gradient descent when `ls` is
  left as `None`; Palantir pins both, so the port only needs the closed-form posterior
  mean above, not the optimiser.
