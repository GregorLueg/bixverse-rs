# GPU-accelerated Frank-Wolfe updates for SEACells

## Context

`src/single_cell/mc_generation/seacells.rs` runs two Frank-Wolfe loops per outer
iteration: `update_a_mat` (`:1610`) and `update_b_mat` (`:1700`). Goal is a
`gpu`-gated fit whose FW updates run device-resident, without undoing the
dense-to-sparse work that fixed the memory explosion at scale.

**Hard constraint: nothing densifies.** `K`, `K²B`, `K²Aᵀ`, `A` and `B` keep the
nnz they have today. The only dense objects are shared-memory accumulators of
length `k` per workgroup, which are *smaller* than the CPU's current per-thread
`vec![0.0f32; n]` in `fw_argmins_b` (`:708`).

## Phase 0 results (done)

Measured with temporary phase-timing instrumentation and a CPU bench, both since
removed: they existed to make the decisions below and the findings are recorded
here instead. Three shapes, three pruning settings, `max_iter = 3`,
`max_fw_iters = 50`.

Share of attributed time, pruning at 1e-7:

| phase | 20k / k=266 | 50k / k=666 | 50k / k=200 |
|---|---|---|---|
| **B argmin** | 13.6% | **64.8%** | **48.0%** |
| A argmin | 5.1% | 12.3% | 11.9% |
| A assemble (E, sort, add) | 6.3% | 7.2% | 15.0% |
| A prune | 2.4% | 3.8% | 8.4% |
| B `K@(K@B)` per iter | 3.0% | 5.9% | 5.9% |
| B setup | 1.3% | 1.9% | 4.0% |
| B transposes per iter | 0.6% | 1.4% | 1.1% |
| RSS | **67.3%** | 2.1% | 4.8% |
| total | 10.1s | 19.3s | 11.5s |

Five findings, three of which contradict the previous version of this plan.

**1. The two argmin scans are the whole problem: 61-77%.** Everything else is
noise by comparison.

**2. The per-iteration `K@(K@B)` rebuild is 3-6%, not the bottleneck.** The
previous plan was built around removing it. `nnz(K²B)` is only 3-4% dense, so the
SpGEMM is already cheap. Both proposed restructurings are dead:

- *Incremental `K²B`* would save 3-6% for a growing sparse support. Not worth it.
- *Reassociating to `K²(B·t1)`* costs `2·k·nnz(K)` per iteration against the
  current `k·nnz(K²B)`, and measured `nnz(K²B) ≈ nnz(K)`, so it is roughly **2×
  worse**. Dropped.

There is no restructuring to do on the B side. `G = K²B·t1 - K²Aᵀ` reduced to a
column argmin costs `nnz(K²B)·k` FMAs and that is irreducible in this
formulation. It is not doing too much work, it is running that work at 3.7
GFLOP/s of scattered sparse accumulation. That is a kernel problem, which is
exactly what a GPU fixes.

**3. The cost model is confirmed, so the design can be priced.**
`fw_argmins_b` = `2nk + Σ_c Σ_{m ∈ nz(t1[:,c])} nnz(K²B[:,m])`. Across the two
k=266 / k=666 points at fixed density that is 13.0× the work for 12.2× the time.
`fw_argmins_a` = `n · T · (d · nnz(t1_a)/k + k)`: 3.4× the work for 3.3× the time.

The k=200 point runs at half the GFLOP/s of k=666, which is `CHUNK = 64` in
`fw_argmins_b` (`:701`) giving only `⌈200/64⌉ = 4` rayon tasks. Worth a one-line
fix on the CPU path; irrelevant on the GPU.

**4. Pruning on is 2.3× faster end to end** (19.3s vs 44.2s at 50k/666), and my
earlier claim that it "provably cannot fire" was wrong for a reason I had not
considered. Only 7 of 350 calls remove anything, but `γ_0 = 1` zeroes the previous
`A`'s values while leaving them in the sparsity pattern, and `sparse_add_csr` keeps
those zeros, so with pruning off the pattern carries them:

| | nnz(A) atoms/cell | nnz(t1_b) density |
|---|---|---|
| pruning off | 64.5 | 1.000 |
| pruning 1e-7 | 22.6 | 0.352 |

`B argmin` is linear in both, hence the 2.3×. At `max_iter = 100` rather than the
3 used here the gap widens further. Pruning is load-bearing, and the reformulation
must support it exactly rather than gate it out.

**5. The `<= 20000` RSS boundary is backwards.** At n = 20 000 exactly,
`compute_rss_simple` materialises the n×n reconstruction and costs 10.7s / 67% of
runtime; at n = 50 000 the trace path costs 1.0s / 2%. The threshold makes the
smaller problem 10× slower than the bigger one. Independent bug, cheap fix.

## Phase 1: CPU reformulation (done)

- `FwAtoms` / `FwPruneOutcome` - per-column atom bookkeeping. The convex step,
  prune and renormalise each report what the caller must correct in its gradient
  state, so an incrementally maintained `t1 · A[:,j]` stays exact under pruning.
- `fw_columns_a` - cell-major pass replacing `fw_argmins_a` plus the whole
  `E` / sort / `sparse_add_csr` / `prune_and_renormalise` chain.
- `fw_atoms_to_csr`; `update_a_mat` reduced to a thin driver.
  `update_a_mat_iteration_major` and `fw_argmins_a` kept `#[cfg(test)]`.
- RSS: `compute_rss_simple` retired to `#[cfg(test)]`, the trace path is now
  unconditional and `k_frobenius_norm_sq` always cached. Measured 1.9-3.3x faster
  at every size. The three terms cancel down to the residual, so the traces, the
  cached `||K||_F^2` and the combination all run in `f64`; doing that in `f32` was
  the dominant error and cost an order of magnitude. Agreement with the
  materialising path is 4e-6 to 8e-6 at `k = n/75`, against a 1e-3 convergence
  threshold, and 3e-3 to 1e-2 at `k = n` where the residual is 0.2% of `||K||_F`.
  The result is clamped at zero before the root so that regime cannot return NaN.
- The structural zeros go with it: `nnz(A)` with pruning off drops from 64.5 to
  22.6 atoms per cell, matching what pruning-on gives, so pruning is no longer
  load-bearing for memory. The mechanism is `fw_atoms_to_csr` going through
  `coo_to_csr`, which drops exact zeros, rather than anything to do with merging
  repeated argmins: `sparse_add_csr` merges those too. At `n = 2000`, `k = 200`
  the two paths sit at 25.6 against 34.6 atoms per cell and both settle after the
  first outer iteration, so this is a constant factor rather than unbounded growth.
- Phase-timing scaffolding (`SEACellsDiagnostics`, `PruneStats`,
  `benches/seacells_bench.rs`) removed once it had served its purpose.

## Phase 2: GPU path (done)

- `fw_argmin_b` + `reduce_argmin_blocks` in
  `src/gpu/sc_gpu/kernels/seacells_kernels.rs`. Fused `K²B · t1 - K²Aᵀ` with a
  column-wise argmin and the duality-gap term; the gradient is never
  materialised. Each thread owns a strided slice of the `k` columns in registers,
  so there is no `k`-dependent shared-memory budget and every register-array
  index stays comptime.
- `GpuFwArgminB` + `seacells_fit_gpu` in `src/gpu/sc_gpu/seacells_gpu.rs`, a
  separate public GPU entry point matching the `harmony_v2_gpu` /
  `pca_on_sc_sparse_gpu` convention.
- `FwArgminB` seam in `seacells.rs` with `begin` (per B update) and `argmins`
  (per FW iteration), so the GPU path reuses the fit loop instead of forking it.
  `SEACells::fit` delegates to `fit_with`; the public CPU API is unchanged.

## Results

Kernel, against the CPU scan it replaces: 3.4x at 20k/266, 10.7x at 50k/666,
13.9x at 50k/200.

End to end, 3 outer iterations, pruning 1e-7:

| shape | original CPU | CPU now | GPU now | total |
|---|---|---|---|---|
| 20k / 266 | 10.05s | 2.48s | 2.11s | 4.8x |
| 50k / 666 | 19.30s | 16.29s | 6.09s | 3.2x |

Hard assignments agree 100% on both shapes; RSS identical at 50k/666.

## Verification

`cargo test --features single-cell,multi-modal` 576 pass;
`cargo test --features gpu,single-cell` 595 pass plus `tests/seacells_gpu.rs`.
Clippy and `cargo fmt` clean on both feature sets.

Covering: the `FwAtoms` gradient invariant with and without pruning, a dropped
atom returning, a fully pruned column, the closed-form weights, cell-major vs
iteration-major parity at three pruning settings, RSS path agreement, kernel vs
CPU reference including empty rows and a deliberate argmin tie, and end-to-end
CPU vs GPU parity with pruning off and on.

## Scaling

Measured, real pipeline, GPU path, 3 outer iterations:

| n | k | nnz(K2B) | density | fit time | B argmin share |
|---|---|---|---|---|---|
| 50 000 | 666 | 1.07M | 3.08% | 5.9s | 65% |
| 200 000 | 2 666 | 12.2M | 2.28% | 78.2s | 56% |

Synthetic kernel probe at 500k: no dispatch or binding limit is hit and VRAM peaks
at 476 MB, but throughput slides from 167 to 92 GFLOP/s as `t1` (dense `k x k`)
grows past cache at k > ~3500.

Nothing explodes. The cost is inherent to the algorithm: at the SEACells
convention `k = n/75`, work scales as `nnz(K2B) * k ~ n^2`. Holding `k`
sub-linear in `n` keeps 500k comfortable (500k with k = 1000 measures 100 ms per
Frank-Wolfe iteration).

## Open items

- The kernel runs at ~1% of device peak with one `t1` load per FMA, so it is
  issue/bandwidth bound. Row-blocking to reuse each `t1` load across several rows
  is the obvious next lever, but B argmin is now ~20% of runtime, so `A argmin`
  is the larger target first.
- `CHUNK = 64` in `fw_argmins_b` starves rayon below k ~ 640. Only affects the
  CPU path.
- No R wrapper yet for `seacells_fit_gpu`.
- `A argmin` is 24.8% at 200k/2666 and is the next GPU target, followed by the
  per-iteration `K@(K@B)` at 10.9%.
- `t1` leaves cache above k ~ 3500. Blocking the `t1` column range so each loaded
  row is reused across several rows of `K2B` would address both that and the
  one-load-per-FMA issue rate.
