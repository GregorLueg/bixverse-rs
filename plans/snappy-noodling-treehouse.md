# MAGIC imputation and Palantir gene trends

## Context

Palantir landed in the crate but its gene-trends output is missing, and gene trends are the
main reason anyone runs Palantir. That needs three things the crate does not have: a smooth
1-D multi-output regressor, per-branch cell masks, and an expression matrix to fit against.
MAGIC is the usual choice for the last one.

MAGIC is also the piece worth being careful about. It is a graph smoother: three steps of a
row-stochastic operator over a kNN graph, so every cell becomes a weighted average of its
neighbourhood, and neighbourhoods overlap. That inflates gene-gene correlation hard (Andrews
and Hemberg, F1000Res, 2018). This crate is full of correlation-based methods (Hotspot,
SCENIC, differential correlation, CoReMo) and feeding them imputed counts would be a
straightforward way to manufacture results. Second problem: the output is dense, which is
diametrically opposed to the streaming store in `src/single_cell/sc_data/data_io.rs` that
makes big analyses feasible from R at all.

Both problems dissolve if MAGIC is **gene-subset-first**. The reference imputes everything
because AnnData already holds everything in memory. We do not have to. Imputing 200 genes
across 100k cells is 80 MB, which is fine, and the `.bin` store is never touched, written or
versioned. Genome-wide imputation is simply not offered.

Outcome: MAGIC as a standalone, bounded, dense-subset tool for visualisation and trends;
gene trends that accept whatever matrix the caller hands over and never impute on their own;
and a generic landmark GP in `src/ml/` that is not single-cell specific.

Supersedes the higher-level notes in `plans/magic_and_gene_trends.md`, which stay as
background. Reference behaviour was read out of `~/repos/others/MAGIC`,
`~/repos/others/Palantir` and the installed `mellon`.

## Fixes to land first

Two things found while reading, both small and both on the critical path:

1. `normalise_csr_rows_l1` (`src/core/math/sparse.rs:1552`) **panics** on a zero-sum row
   despite returning `Result`. Replace the panic with a new
   `BixverseErrors::SparseMatrixIsolatedRow { row }`. No signature change, strictly better,
   and MAGIC's row-stochastic step goes straight through it. The reference guards zero rows
   by leaving them zero; erroring is the better call here because
   `compute_diffusion_kernel` builds `W + W^T` on a kNN graph where every cell has `k`
   out-edges, so a zero row means the weights underflowed and the result is garbage anyway.
   Document the divergence.
2. `BixverseSimd` (`src/utils/simd.rs:342`) only has `bxv_dot_simd`. MAGIC's inner loop is
   pure axpy. Add `fn bxv_axpy_simd(y: &mut [Self], a: Self, x: &[Self])` to the trait with
   `f32`/`f64` impls, next to the existing dot ones. Generic, reusable, and per the style
   rules SIMD belongs in the trait rather than inline in the algorithm.

## Piece 1: sparse operator applied to a dense block

New in `src/core/math/sparse.rs`, beside `csr_sparse_matmul_dense`:

```rust
/// Apply a CSR operator to a dense row-major block: `out = a @ block`.
///
/// Row-major flat buffers rather than `Mat` so each output row is a contiguous
/// `width`-length axpy target. Rayon over output rows, `bxv_axpy_simd` per
/// non-zero.
pub fn csr_matmul_dense_block<T>(
    a: &CompressedSparseData2<T>,
    block: &[T],
    width: usize,
    out: &mut [T],
) -> Result<(), BixverseErrors>
where
    T: BixverseFloat + BixverseSimd;
```

Row-major flat `Vec<T>`, not `faer::Mat`, on purpose: the operation is
`out_row_i = sum_j w_ij * block_row_j`, so contiguous rows make every inner step a SIMD
axpy over `width` floats. `Mat` is column-major and would strided-scatter.

Existing precedent to replace, not extend: `chebyshev_apply_columns`
(`src/single_cell/sc_analysis/meld.rs:455`) does the same job column-by-column via the
serial `csr_matvec`, restreaming the whole CSR once per column. Leave MELD alone in this
change, but note it in the doc comment as a candidate follow-up.

## Piece 2: MAGIC

New file `src/single_cell/sc_processing/magic.rs`, registered in `sc_processing/mod.rs`.

### The operator

```rust
/// Row-stochastic diffusion operator over a cell subset.
pub struct MagicOperator {
    /// `T = D^-1 K`, CSR, `n_sub x n_sub`, rows in `cell_indices` order.
    t: CompressedSparseData2<f32>,
    /// Global cell ids, one per row of `t`.
    cell_indices: Vec<usize>,
    /// Global id -> local row, `u32::MAX` for excluded cells. Length `total_cells`.
    lookup: Vec<u32>,
}

impl MagicOperator {
    pub fn from_knn(
        knn_indices: &[Vec<usize>],
        knn_distances: &[Vec<f32>],
        squared_dist: bool,
        cell_indices: &[usize],
        total_cells: usize,
    ) -> Result<Self, BixverseErrors>;

    pub fn n_cells(&self) -> usize;
    pub fn cell_indices(&self) -> &[usize];
}
```

Built from two existing functions and nothing else: `compute_diffusion_kernel`
(`src/single_cell/mc_generation/seacells.rs:313`) for the adaptive-bandwidth kernel `W + W^T`,
then the fixed `normalise_csr_rows_l1` for `D^-1 K`. That is Palantir's operator exactly
(`utils.py:432-448, 481-484`), which is what we want since this feeds Palantir.

MAGIC builds its own operator rather than reusing Palantir's. `multiscale_components`
currently consumes and drops the kernel, and threading it out would widen `PalantirResult`
(already 13 fields) and couple the two. The duplicate kernel build is cheap next to the kNN
search, and MAGIC stays usable with no trajectory analysis at all.

The `lookup` vector mirrors `CellBatchIndex::lookup` (`src/single_cell/sc_processing/hvg.rs:127`)
and removes any ordering constraint on `cell_indices`: gene chunks scatter straight into
local rows. 400 KB at 100k cells.

Note the bandwidth divergence to document: `compute_diffusion_kernel` uses
`BANDWIDTH_RANK_DIVISOR = 3`, i.e. sigma is the `knn/3`-th neighbour distance, matching
Palantir. MAGIC's own graphtools kernel uses the `knn`-th neighbour with a tunable decay
exponent and a `1e-4` affinity threshold. We follow Palantir, not graphtools, and say so.

### Params and the size guard

```rust
#[derive(Clone, Copy, Debug)]
pub struct MagicParams {
    /// Diffusion steps. Reference default: 3.
    pub n_steps: usize,
    /// Values below this are zeroed after the last step. Reference: 1e-2.
    pub clip_threshold: f32,
    /// Genes per streaming block. Bounds the ping-pong scratch.
    pub gene_batch_size: usize,
    /// Which stored layer to impute. Log-normalised is the sane default.
    pub layer: MagicLayer,
    /// Skip the output-size check. Off by default.
    pub allow_large: bool,
}
```

Defaults: `n_steps = 3`, `clip_threshold = 1e-2`, `gene_batch_size = 1000` (matching
`GENE_BATCH_SIZE` in `hvg.rs:33`), `layer = Norm`, `allow_large = false`.

Module-level const with the reasoning in its doc comment:

```rust
/// Element budget for the dense output. 250e6 f32 elements is 1 GB, which is
/// the point at which an R session on a laptop starts swapping. Exceeding it
/// is an error rather than a warning because R users do not reliably see
/// stderr, and the alternative failure mode is an OOM kill.
const MAGIC_MAX_ELEMENTS: usize = 250_000_000;
```

Over budget gives `BixverseErrors::MagicOutputTooLarge { n_cells, n_genes, max_elements }`,
with the message naming the implied MB and telling the caller to subset genes or set
`allow_large`.

`n_steps = 0` is legal and returns the un-imputed dense block. That is deliberate: it makes
the same function the "give me a dense expression matrix for these genes and cells" path
that gene trends need when the user does *not* want imputation, so there is one code path,
not two.

### Entry points

```rust
/// Imputed expression, cells by genes.
pub struct MagicImputed {
    /// `n_cells x n_genes`, row-major, rows in operator order.
    pub data: Vec<f32>,
    pub n_cells: usize,
    /// Gene ids, in the caller's order.
    pub gene_indices: Vec<usize>,
    /// Global cell ids, one per row.
    pub cell_indices: Vec<usize>,
}

impl MagicImputed {
    /// Column-major view for the faer-based consumers (gene trends).
    pub fn to_mat(&self) -> Mat<f32>;
}

/// Stream gene blocks off the store and diffuse each.
pub fn magic_impute_genes<S: SingleCellReading>(
    reader: &S,
    operator: &MagicOperator,
    gene_indices: &[usize],
    params: Option<MagicParams>,
    verbose: usize,
) -> Result<MagicImputed, BixverseErrors>;

/// In-memory variant for callers that already hold the matrix, and for tests.
pub fn magic_impute_dense(
    operator: &MagicOperator,
    block: &[f32],
    width: usize,
    params: Option<MagicParams>,
) -> Result<Vec<f32>, BixverseErrors>;
```

`reader` must be gene-based (`is_gene_based()`), else `ReaderModeMismatch`. Per block:

1. `reader.read_gene_parallel(&block_genes)?` gives `CscGeneChunk`s carrying cell ids.
2. Scatter into a dense `n_sub x block_width` row-major buffer via `lookup`, picking
   `data_raw` or `data_norm` per `params.layer`. Cells outside the subset are skipped.
3. `n_steps` ping-pong passes of `csr_matmul_dense_block` between two scratch buffers.
4. Clip below `clip_threshold` (par over the buffer), then copy into the output at the
   block's column offset.

Peak memory is `n_cells * n_genes` for the output plus `2 * n_cells * gene_batch_size` for
the scratch. At 100k cells, 200 genes and a 1000-gene batch the scratch dominates at 800 MB,
so `gene_batch_size` should be clamped to `n_genes` at entry. Do that.

Follows the `sweep_gene_block` pattern in `hvg.rs`: pre-allocate run-wide, no per-block
allocation, rayon inside the block sweep.

### Documentation requirements

Non-negotiable, in the module `//!` doc:

- MAGIC inflates gene-gene correlation. Its output must not be fed into Hotspot, SCENIC,
  differential correlation or CoReMo. State the mechanism, not just the warning.
- The operator preserves per-cell mass, so imputed values sit on the input's scale.
  Imputing raw counts and imputing log-normalised counts give different objects. Say which
  layer `MagicLayer` picks and why `Norm` is the default.
- Gene trends already smooth over pseudotime with a GP. Running MAGIC first smooths twice.
  It is defensible for visualisation and often unnecessary for trends. Say it plainly.
- We never form `T^3`. At `knn = 30` that is ~27k non-zeros per row, so 2.7e9 at 100k cells.
  Three applications of `T` give the identical result at ~30 per row.

## Piece 3: landmark Gaussian process

New `src/ml/gp/` with `mod.rs`, `kernels.rs`, `landmark.rs`; `pub mod gp;` in `src/ml/mod.rs`.
Not feature-gated, so it must compile under `--no-default-features`.

Placement: not `src/core/math/rbf.rs`. That module is `f(distance) -> affinity` over an
already-materialised distance buffer and is parameterised by `epsilon`; routing the GP
through it would force materialising a 500 x 100k distance matrix just to run a second
elementwise pass over it, and would sit a `ls`-parameterised Matern next to
`epsilon`-parameterised Gaussians. For 1-D inputs the distance is `|x - x'|`, so fusing it
into the kernel fill is free. `src/ml/clustering/k_means.rs` is the precedent for a model
with params, fitted state and its own errors.

### Kernel: two free functions, no trait

```rust
/// Matern-5/2 at a scalar distance: `(1 + r + r²/3) exp(-r)`, `r = sqrt(5) d / ls`.
#[inline]
pub fn matern52(dist: f64, length_scale: f64) -> f64;

/// Fill `dst` (`m x n`) with `k(a_i, b_j)`, fusing the 1-D distance in.
pub fn fill_matern52_cross_1d(dst: MatMut<'_, f64>, a: &[f64], b: &[f64], length_scale: f64);
```

A `dyn Fn` costs a virtual call per element across 2.5e7 elements per chunk; a `K: Kernel`
bound propagates through every signature to support one kernel. When a second one arrives,
promote to a plain enum matched once per chunk inside the fill. Do not reach for a trait now.

### API

```rust
#[derive(Clone, Debug)]
pub struct LandmarkGpParams {
    /// Matern-5/2 length scale. Reference default: 1.0.
    pub length_scale: f64,
    /// Noise standard deviation. Reference default: 1.0. Must be > 0.
    pub sigma: f64,
    /// Added to the diagonal of `k(u, u)` before its Cholesky. mellon: 1e-6.
    pub jitter: f64,
    /// Cholesky retries, each multiplying the jitter by 10.
    pub max_jitter_retries: usize,
    /// Training rows per accumulation chunk. Sets the per-thread footprint.
    pub chunk_size: usize,
}

pub struct LandmarkGpFit {
    pub landmarks: Vec<f64>,
    /// Posterior weights, `m x n_outputs`.
    pub weights: Mat<f64>,
    pub length_scale: f64,
    /// Jitter the landmark Cholesky actually needed. Above the requested value
    /// means the Gram was singular there.
    pub jitter_used: f64,
    /// `k(u, u)` without jitter. Prediction on the landmarks is then one GEMM.
    k_uu: Mat<f64>,
}

pub fn fit_landmark_gp<T: BixverseFloat>(
    x: &[T],
    y: MatRef<'_, T>,
    landmarks: &[f64],
    sample_weights: Option<&[f64]>,
    params: &LandmarkGpParams,
) -> Result<LandmarkGpFit, BixverseErrors>;

impl LandmarkGpFit {
    pub fn predict<T: BixverseFloat>(&self, x_new: &[T]) -> Mat<T>;
    /// Posterior mean on the landmarks. Reuses the cached `k(u, u)`.
    pub fn predict_on_landmarks<T: BixverseFloat>(&self) -> Mat<T>;
}
```

Prior mean is fixed at zero, matching mellon (`function_estimator.py:139`) and therefore
Palantir. No `GpPriorMean` enum: YAGNI, and a non-zero prior is a silent divergence a user
could pick up without realising it changes the shrinkage.

`sample_weights` is in the signature from the start even though gene trends default to hard
masks. Retrofitting it means rewriting the chunk body, and it is the mechanism behind the
weighted alternative below. Cost is ~20 lines: scale each column of `A` and row of `r` by
`sqrt(w_i)` inside the chunk loop.

`k(grid, u) == k(u, u)` when the landmarks are the grid, which is the gene-trends case.
Caching `k_uu` on the fit makes "compute once" structural rather than a comment.

### The maths

Subset-of-regressors posterior mean (`mellon/conditional.py:57-63, 455-546`):

```
Lp      = chol(k(u,u) + jitter I)     # m x m
A       = Lp^-1 k(u,x)                # m x n, triangular solve
r       = y                           # n x p, prior mean is 0
G       = A A^T,   P = A r            # accumulated in chunks
B       = G/sigma^2 + I               # m x m
L_B     = chol(B)
weights = Lp^-T L_B^-T L_B^-1 (P/sigma^2)
trend   = k(u,u) @ weights            # m x p
```

faer 0.23.2 (locked), imports `faer::linalg::matmul::{matmul, triangular::matmul as
triangular_matmul}`, `faer::linalg::triangular_solve::{solve_lower_triangular_in_place,
solve_upper_triangular_in_place}`, `faer::{Accum, Mat, MatMut, MatRef, Par, Side}`.
`mat.llt(Side::Lower)?` already converts via
`BixverseErrors::FaerCholeskyError(#[from] LltError)` (`src/errors.rs:47`). `Llt::L()` has
its strict upper triangle zeroed by `new_imp`, so no masking. The triangular solves take
`impl Stride` on both operands, so `lb.transpose()` and `lp.transpose()` are legal arguments
with no copy. `triangular_matmul` argument order follows
`src/core/base/cors_similarity.rs:42-53`.

`Llt::new` uses faer's global parallelism internally, not our `Par` argument. Irrelevant at
500 x 500, but keep both factorisations outside the rayon region, which the design does.

### Parallelism, memory, numerics

Rayon goes in exactly one place: the chunk accumulation of `G` and `P`, which is the only
part touching `n_cells`. All faer calls inside that region take `Par::Seq`; the crate has no
`Par::Seq` today (everything routes through `crate::utils::faer_parallelism()`), so this is
new ground and worth calling out in review. The three post-reduction weight solves and the
final GEMM use `faer_parallelism()`.

Partition into exactly `available_parallelism()` contiguous slabs, `map` to per-slab
partials, `collect` (index order preserved), sum serially. A bare `par_chunks().reduce()`
gives a split-dependent summation order and non-reproducible low bits; the crate cares
about this elsewhere (see the divergence notes in `markov.rs`).

`A` is never materialised at `m x n_cells` (400 MB in f64 at 100k cells). Per-thread
footprint is `8 * (m*chunk + chunk*p + m*m + m*p)`, about 16 MB at `m = 500`, `p = 300`,
`chunk = 2048`. Document that formula on `chunk_size`.

**f64 internally regardless of `T`.** With pseudotime on `[0, 1]` and `ls = 1`, every
pairwise distance is <= 1, so every entry of `k(u,u)` lies in `[0.524, 1]`: a near-constant
500 x 500 matrix with one eigenvalue around 400 and the rest collapsing toward zero. At
jitter 1e-6 the condition number is ~4e8. In f64 that leaves ~8 digits. In f32 it is
numerically indefinite and `llt` fails or returns nonsense; forcing it through needs jitter
around 1e-2, which is a materially different prior. Separately, `A = Lp^-1 k(u,x)` has
mixed signs scaled by ~1e3 and `A A^T` sums 100k of those per entry, so the accumulation
wants the headroom too. Accept `T`, widen at the chunk boundary, cast the trend back.
`LandmarkGpParams` fields are f64 for the same reason.

Do **not** "optimise" by accumulating `K K^T` and applying `Lp^-1` once at 500 scale. It
saves ~30% of the flops and squares the condition number of an already brutal Gram. Leave a
comment saying so.

Jitter ladder: 1e-6, retry x10 up to `max_jitter_retries` (default 3), record `jitter_used`.
mellon just errors; the ladder is a deliberate, reportable divergence. The second Cholesky
needs none (`min eig >= 1`) so a failure there means a NaN got in: give it a distinct error
variant.

Validate up front: length mismatch, empty inputs, `landmarks.len() < 2`, non-finite `x` or
`y`, `sigma <= 0`, `length_scale <= 0`, all-equal landmarks. Silent NaN propagation surfaces
as an unexplained Cholesky failure 30 seconds later.

## Piece 4: branch masks and gene trends

New file `src/single_cell/sc_trajectory/gene_trends.rs`, registered in `sc_trajectory/mod.rs`.
`select_branch_cells` reads only `pseudotime` and `branch_probs`, and gene trends is its only
consumer, so both live here. Split later if a second consumer appears.

### `select_branch_cells`

The rolling quantile is an **expanding prefix**, not a sliding window. Transcribed from
`presults.py:597-638` (note: the docstring runs to 595, the loop is at 613-632):

1. `fate_probs[isnan] = 1 / n_fates`.
2. `idx = argsort(pseudotime)`; work in sorted order throughout.
3. `resolution = min(500, n)`; `step = n / resolution`; `nsteps = n / step`, integer
   division throughout.
4. For `i in 0..nsteps` with `l = i*step`, `r = (i+1)*step`: rows `l..r` get
   `quantile(sorted_probs[..r], 1 - q)` per fate. The threshold slice always starts at 0 and
   **includes** the block being assigned.
5. Rows `r..` (the `n % step` tail, using the loop's final `r`) get the quantile over all `n`.
6. Cumulative max down the rows, per fate, so a fate's bar can only rise with pseudotime.
7. `mask[idx] = threshold - eps < sorted_prob`, strict. `q = eps = 1e-2`.

```rust
#[derive(Clone, Copy, Debug)]
pub struct BranchSelectionParams {
    /// Upper-tail quantile of the fate probability. Reference: 1e-2.
    pub q: f64,
    /// Slack subtracted from the threshold. Reference: 1e-2.
    pub eps: f64,
    /// Bucket count, capped at the cell count. Reference: 500.
    pub resolution: usize,
}

/// Per-fate cell selections, ascending cell index. Column order matches
/// `branch_probs` and therefore `PalantirResult::terminal_states`.
pub fn select_branch_cells(
    pseudotime: &[f32],
    branch_probs: MatRef<'_, f32>,
    params: &BranchSelectionParams,
) -> Result<Vec<Vec<usize>>, BixverseErrors>;
```

Reuse `quantile_sorted` (`src/core/math/vector_helpers.rs:171`), which already implements
numpy's linear interpolation. Do not re-partition the prefix 500 times: per fate keep a
sorted `Vec<f64>` and two-pointer merge each block's `step` new values in, `O(n)` per block.
Rayon over fates.

`Vec<Vec<usize>>` rather than a boolean matrix: faer has no `Mat<bool>` (`ComplexField`
bound), and both gene trends and the R side want indices to gather with.

Divergences to document: `sort_by` with `total_cmp` is stable so pseudotime ties break on
ascending cell index, where `np.argsort` defaults to unstable quicksort (only shifts tied
cells between adjacent prefix buckets). We also error on non-finite pseudotime and on
`pseudotime.len() != branch_probs.nrows()`, neither of which the reference checks.
`PalantirResult::branch_probs` is thresholded without renormalisation, so rows need not sum
to one; that is fine for a per-fate quantile, but say so.

### `compute_gene_trends`

```rust
/// How a branch's cells enter its fit.
#[derive(Clone, Copy, Debug, Default)]
pub enum BranchWeighting {
    /// Hard mask from `select_branch_cells`, every member weighted equally.
    /// What Palantir's current path does.
    #[default]
    HardMask,
    /// Every cell enters every branch, weighted by its fate probability.
    /// Closer to the legacy GAM path and more defensible: a cell at 0.6 is
    /// not a member.
    FateProbability,
}

#[derive(Clone, Debug)]
pub struct GeneTrendsParams {
    /// Grid points per branch. Reference: 500.
    pub resolution: usize,
    pub weighting: BranchWeighting,
    /// Carries the reference `ls = 1`, `sigma = 1`.
    pub gp: LandmarkGpParams,
}

pub struct GeneTrendsResult<T> {
    /// One per branch, `resolution x n_genes`.
    pub trends: Vec<Mat<T>>,
    /// Per-branch pseudotime grid, branch min to branch max.
    pub grids: Vec<Vec<T>>,
    pub n_cells: Vec<usize>,
    pub jitter_used: Vec<f64>,
}

/// Fit a landmark GP per branch.
///
/// Takes whatever expression matrix the caller hands over. Raw, log-normalised
/// or MAGIC-imputed is their decision; this never imputes.
pub fn compute_gene_trends<T: BixverseFloat>(
    expression: MatRef<'_, T>,
    pseudotime: &[f32],
    branch_cells: &[Vec<usize>],
    branch_probs: Option<MatRef<'_, f32>>,
    params: &GeneTrendsParams,
) -> Result<GeneTrendsResult<T>, BixverseErrors>;

/// Convenience over a finished Palantir run.
pub fn compute_gene_trends_palantir<T: BixverseFloat>(
    expression: MatRef<'_, T>,
    palantir: &PalantirResult,
    selection: &BranchSelectionParams,
    params: &GeneTrendsParams,
) -> Result<GeneTrendsResult<T>, BixverseErrors>;
```

`branch_probs` is required only for `BranchWeighting::FateProbability`; error if the
weighting asks for it and it is absent.

Per branch, sequentially (2 to 5 branches, each fit already saturates the machine, and
sequential keeps peak memory at one branch):

1. `pt = pseudotime[cells]` widened to f64.
2. `grid = linspace(pt.min(), pt.max(), resolution)`. No `linspace` in the crate; write a
   private one pinning both endpoints exactly, matching `np.linspace`.
3. Gather the branch submatrix via `subset_rows` (`src/core/math/matrix_helpers.rs:257`).
4. `fit_landmark_gp` then `predict_on_landmarks`.

Keep the grid at 500 even when a branch has fewer cells. That is what the reference does
(`presults.py:301`), and it is structurally fine: `Lp` depends only on the grid, `A A^T` is
just rank-deficient, and `+ I` keeps `L_B` conditioned. The result is a prior-dominated
curve. Note the reference's own asymmetry, that `select_branch_cells` *does* clamp its
bucket count via `min(500, n)`. Add one guard the reference lacks: error below 3 cells, and
error on zero pseudotime range within a branch.

Output orientation is `resolution x n_genes`, which is what the maths produces. Palantir
transposes for AnnData `varm`; do that in the R wrapper, not here.

### The honest caveat, in the module docs

With pseudotime min-max scaled to `[0, 1]`, a Matern-5/2 length scale of `ls = 1` spans the
whole domain and `sigma = 1` puts the noise at roughly the signal scale of log-normalised
expression. The posterior is prior-dominated: it flattens genuine transient structure and
resolves almost any gene to a smooth monotone or single-peaked curve. That is a presentation
choice, not inference. Say it, and say that a user checking whether a bump is real needs to
shorten `length_scale`.

## Errors

New section in `src/errors.rs`. Ungated (`src/ml/` is not feature-gated):
`GpEmptyInput`, `GpDimensionMismatch { n_x, n_y }`, `GpInvalidHyperparameter { name, value }`,
`GpNonFiniteInput { source }`, `GpLandmarkCholeskyFailed { jitter, n_landmarks }`,
`GpPosteriorCholeskyFailed`, `GpDegenerateLandmarks`. Plus `SparseMatrixIsolatedRow { row }`
in the sparse section.

Gated on `single-cell`, in the existing Palantir section at `src/errors.rs:655`:
`MagicOutputTooLarge { n_cells, n_genes, max_elements }`,
`GeneTrendsShapeMismatch { n_cells, n_pseudotime }`, `GeneTrendsBranchEmpty { branch }`,
`GeneTrendsBranchTooFewCells { branch, n_cells }`,
`GeneTrendsDegeneratePseudotime { branch }`, `GeneTrendsMissingFateProbabilities`.

## R surface

`from_r_list` impls in `src/single_cell/sc_r_wrappers.rs` for `MagicParams`,
`BranchSelectionParams`, `GeneTrendsParams` and `LandmarkGpParams`, following
`PalantirParams::from_r_list` (`sc_r_wrappers.rs:3125`): `r_list_to_map`, per-field
`.get().and_then(...).unwrap_or(defaults.field)`, defaults sourced from `Default::default()`
so the two sides cannot drift. The `#[extendr]` shims live in the `bixverse` R package, not
here. Note that `run_palantir` itself still has no R entry point, so gene trends will need
one alongside.

## Files

| File | Change |
|---|---|
| `src/utils/simd.rs` | add `bxv_axpy_simd` to `BixverseSimd` + f32/f64 impls |
| `src/core/math/sparse.rs` | fix the `normalise_csr_rows_l1` panic; add `csr_matmul_dense_block` |
| `src/errors.rs` | new GP section, plus MAGIC and gene-trends variants |
| `src/ml/mod.rs` | `pub mod gp;` |
| `src/ml/gp/{mod,kernels,landmark}.rs` | new |
| `src/single_cell/sc_processing/mod.rs` | `pub mod magic;` |
| `src/single_cell/sc_processing/magic.rs` | new |
| `src/single_cell/sc_trajectory/mod.rs` | `pub mod gene_trends;` |
| `src/single_cell/sc_trajectory/gene_trends.rs` | new |
| `src/single_cell/sc_r_wrappers.rs` | four `from_r_list` impls |

## Order of work

1. SIMD axpy and the `normalise_csr_rows_l1` fix, with unit tests.
2. `csr_matmul_dense_block` against a hand-computed 4x4 case and a dense reference.
3. Error variants.
4. `src/ml/gp/kernels.rs`, then `landmark.rs`.
5. `magic.rs`.
6. `select_branch_cells`, then `compute_gene_trends`.
7. R params plumbing.

## Verification

Unit tests, all toy-sized so CI runs them:

- `bxv_axpy_simd` against a scalar loop on lengths straddling the lane width.
- `csr_matmul_dense_block` against a dense reference on a random sparse operator; and
  `n_steps` applications against `csr_matmul_csr` powering the operator explicitly, on a
  matrix small enough that the explicit power is legal. That directly checks the
  "never form `T^3`" claim.
- `normalise_csr_rows_l1` returns `SparseMatrixIsolatedRow` instead of panicking.
- `matern52(0, ls) == 1`, monotone decreasing, one hardcoded value against the formula.
- Landmark GP: (a) tiny `sigma` with landmarks equal to the training points interpolates
  `y`; (b) a known smooth function recovered on a 200-point grid to a few percent;
  (c) `chunk_size` above and below `n` agree; (d) `n_train < n_landmarks` returns finite
  values; (e) `sigma = 0`, mismatched dims and NaN inputs give the right errors;
  (f) uniform `sample_weights` matches the unweighted fit.
- `select_branch_cells` against a hand-computed 12-cell, 2-fate example exercising the
  expanding prefix, the cumulative max and the tail block. This is the piece most likely to
  be got subtly wrong.
- MAGIC: `n_steps = 0` returns the raw block unchanged; a round trip through
  `magic_impute_dense` on a hand-built operator; `MagicOutputTooLarge` fires at the boundary
  and `allow_large` suppresses it; row-stochastic mass preservation, i.e. a constant input
  column comes back constant.
- `compute_gene_trends` on the existing Palantir test fixture plus a synthetic gene with a
  known peak, checking the peak lands in the right pseudotime interval.

End to end, an integration test in `tests/` gated on `single-cell`: synthesise a branching
trajectory, run `run_palantir`, `select_branch_cells`, `magic_impute_dense` and
`compute_gene_trends`, and assert a planted branch-specific gene rises on its own branch and
not the other. Anything past a second or two goes behind `large_scale_diagnostics` per
CLAUDE.md, keeping at least one cheap structural test outside the gate.

Commands:

```bash
cargo test --no-default-features                    # src/ml/gp must build here
cargo test --features single-cell,multi-modal
cargo clippy --features single-cell,multi-modal --all-targets
cargo clippy --features gpu,single-cell,large_scale_diagnostics --all-targets
cargo fmt
```

Numerical parity against the references is worth doing by hand once, outside CI: dump a
small kNN graph and expression block, run Palantir's `run_magic_imputation` and
`compute_gene_trends` on it in Python, and compare. Not automatable without a Python
dependency, but the divergences listed above are all deliberate and should be the only ones.
