# Make the x86 SIMD real, and unpick the `wide` version skew

## Context

Two problems surfaced from the SEACells round, and they turn out to share a root.

**1. `use wide::CmpLt` needs an `#[allow]` to compile.** `Cargo.toml:74` pins
`wide = "1.4.0"`, a caret range. In wide 1.4.0 `CmpLt` is a normal trait and
`simd_lt` exists only as its method, so the import is required. In wide 1.5.0
`macros.rs` gained `simd_comparison_fns!`, which adds an inherent `simd_lt` to
every vector type, and `CmpLt` became `#[deprecated(since = "1.5.0")]`. The
worktree lock resolved 1.5.0, rextendr resolved 1.4.x. One source file cannot
satisfy both: on 1.5.0 the import is unused *and* deprecated (`cargo check`
emits `use of deprecated trait wide::CmpLt` today), on 1.4.x removing it is a
hard error.

**2. The AVX-512 arms never execute, and neither do the AVX-2 arms.** Every
`_mm512` body in the crate sits behind
`#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]`. In a `cfg`,
`target_feature` is a compile-time query, true only when the whole crate is
built with `-C target-feature=+avx512f` or `-C target-cpu=native`. There is no
`.cargo/config.toml`, no `build.rs`, and rextendr sets neither, so those
functions are always the `not(...)` fallbacks that call the AVX-2 versions,
while `detect_simd_level()` returns `SimdLevel::Avx512` at runtime and
dispatches straight into them.

The AVX-2 arms are no better. `wide::f32x8` is `{ a: f32x4, b: f32x4 }` unless
`target_feature="avx"` is set at compile time (`wide-1.5.0/src/f32x8_.rs:3-13`),
and baseline `x86_64-unknown-linux-gnu` is SSE2. So on every x86 build the
entire `SimdLevel` ladder collapses to SSE2. aarch64 is the one architecture
where the module does what it claims, because NEON is baseline there — which is
why this went unnoticed on a Mac while the R package shipped SSE2 to x86 Linux
and Windows.

The fused subtract-and-argmin added last round is the only function that was
honest about this ("AVX-512 dispatches to the 256-bit path, matching the rest of
this module"). It behaves exactly like its neighbours; it just does not carry
dead code to disguise it.

Outcome wanted: runtime-dispatched AVX-2 and AVX-512 that actually run on x86
from a stock build, a genuine AVX-512 fused argmin, and the `CmpLt` hack gone.

## Approach

The correct pattern is `#[target_feature(enable = "...")] unsafe fn`, which
compiles the body unconditionally on x86_64 and is safe to call behind
`is_x86_feature_detected!`. This idiom is already in the sister crate at
`ann-search-rs/src/binary/dist_binary.rs:36` (AVX-512) and
`src/binary/turboquant/search.rs:472` (`avx2` + `fma`), with the dispatch shape
at `dist_binary.rs:230-253`. Follow it exactly.

Scalar and SSE arms stay on `wide` and are untouched: SSE2 is baseline on
x86_64 and NEON is baseline on aarch64, so both are real today.

### Per-family shape

```rust
// real body, x86_64 only
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2_f32(a: &[f32], b: &[f32]) -> f32 { /* _mm256_fmadd_ps ... */ }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512_f32(a: &[f32], b: &[f32]) -> f32 { /* _mm512_fmadd_ps ... */ }

#[inline]
pub fn dot_simd_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: each arm is reached only once `detect_simd_level` has confirmed
    // the corresponding feature is present on this CPU.
    unsafe {
        match detect_simd_level() {
            SimdLevel::Avx512 => dot_avx512_f32(a, b),
            SimdLevel::Avx2 => dot_avx2_f32(a, b),
            SimdLevel::Sse => dot_sse_f32(a, b),
            SimdLevel::Scalar => dot_scalar_f32(a, b),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    match detect_simd_level() {
        SimdLevel::Sse => dot_sse_f32(a, b),
        _ => dot_scalar_f32(a, b),
    }
}
```

The `#[cfg(not(all(...)))]` fallback shims are deleted rather than re-cfg'd —
the dispatcher's two blocks replace them, and `detect_simd_level` already
returns only `Sse`/`Scalar` off x86_64.

`detect_simd_level` (`src/utils/simd.rs:24`) tightens its AVX-2 probe to
`avx2 && fma`, so it matches `#[target_feature(enable = "avx2", enable = "fma")]`.
No shipping x86 part has AVX-2 without FMA, and `SimdLevel` /
`detect_simd_level` are used only by the two SIMD files (nothing in `prelude.rs`
re-exports them), so this is contained.

## Work

### 1. `wide` version skew

- `Cargo.toml:74`: `wide = "1.4.0"` -> `wide = "1.5"`.
- `src/utils/simd.rs:4-5`: drop `#[allow(unused_imports)]` and `CmpLt` from the
  import, keeping `f32x4, f32x8, f64x2, f64x4`. The inherent `simd_lt` takes over.
- The R package needs `cargo update -p wide` so rextendr stops resolving 1.4.x.

### 2. `src/utils/simd.rs` — 8 families

`sum_squares_f32`, `dot_f32`, `dot_f64`, `sum_f32`, `sum_f64`,
`sum_squared_dev_f32`, `sum_squared_dev_f64`, plus `argmin_diff_f32` which today
has no AVX-512 arm at all. 8 AVX-2 and 8 AVX-512 bodies.

Reductions use `_mm256_fmadd_ps` / `_mm512_fmadd_ps` and the existing tail
handling. `_mm512_reduce_add_ps` / `_pd` are available inside an `avx512f`
target-feature function; the AVX-2 horizontal reduction needs writing out
(`_mm256_extractf128_ps` + `_mm_hadd_ps`, or extract to `[f32; 8]` and sum —
it runs once per call, so clarity wins).

### 3. Fused argmin: AVX-512 arm and the unroll

The loop carries a serial dependency — each iteration's `best`/`best_idx` blend
depends on the previous — so it is latency-bound, not width-bound. Widening
alone will under-deliver. Break the chain with independent accumulator pairs:

```rust
/// Independent (best, best_idx) accumulator pairs in the argmin scan. The
/// blend chain is loop-carried, so at ~4 cycles of latency a single
/// accumulator leaves most of the issue width idle regardless of vector
/// width. Four covers that latency; more only lengthens the final reduction.
const ARGMIN_UNROLL: usize = 4;
```

Apply at every width, SSE included, so the unroll is measurable on aarch64
where the x86 arms cannot run.

Tie semantics are preserved and must stay that way: per-lane strict `<` keeps
the earliest iteration within a lane, accumulator `p` seeds its lane vector at
`p * W + [0..W)` and steps by `ARGMIN_UNROLL * W`, and the existing
`reduce_lane_argmin_f32` (`:467`) already breaks cross-lane ties to the lowest
index. Fold all `ARGMIN_UNROLL` pairs through that same reduction.

Comparison intrinsics: AVX-512 uses mask registers,
`_mm512_cmp_ps_mask(diff, best, _CMP_LT_OQ)` into `_mm512_mask_blend_ps`;
AVX-2 uses `_mm256_cmp_ps(_CMP_LT_OQ)` into `_mm256_blendv_ps`.

The `f32` lane-index and NaN caveats in the dispatch doc comment (`:576-603`)
still hold and stay. Drop the "AVX-512 dispatches to the 256-bit path" line.

### 4. `src/single_cell/sc_utils/simd.rs` — 6 families

`fused_mul_square_sum`, `center_values`, `elementwise_mul`, `fused_mul_add`,
`accumulate_f32`, `evaluate_split_score_f32`. Same treatment, 6 AVX-2 and 6
AVX-512 bodies.

Note `evaluate_split_score_f32_avx512` (`:966`) currently has *only* the
`not(...)` fallback and no AVX-512 body, so building with
`-C target-cpu=native` on an AVX-512 host is a missing-function compile error
today. It gets a real body here.

### 5. Bench

Temporary `benches/simd_bench.rs` covering the argmin at `k` in
{200, 666, 2666} (the range `fw_columns_a` actually passes at
`src/single_cell/mc_generation/seacells.rs:1156`) plus one reduction family, to
justify `ARGMIN_UNROLL` against the current single-accumulator code. Remove it
once it has served its purpose, as `benches/seacells_bench.rs` was last round.

## Outcome

Done. Three deviations from the plan above, all forced by what the code turned
out to be doing.

**1. `wide` was not bumped; `Cargo.toml` and `Cargo.lock` are untouched.**
Checking 1.6.0 turned up two things: it deprecates `blend` the same way 1.5
deprecated `CmpLt`, and *both* 1.5.0 and 1.6.0 fail to compile under
`-C target-feature=+avx512f` because of an upstream bug in `wide`'s own
`f64x8` AVX-512 branch (`no field 'a' on type f64x8`, ~20 errors). Two API
breaks in two minor releases meant pinning a floor was treating the symptom, so
the fused argmin came off `wide`'s comparison surface entirely: all four arms
(SSE4.1, NEON, AVX-2, AVX-512) are now per-architecture intrinsics. Verified
clean on 1.4.0, 1.5.0 and 1.6.0, so rextendr resolving 1.4.x compiles with no
`cargo update` needed downstream. The reductions keep `wide` for their 128-bit
arm, which only needs arithmetic and `reduce_add` — stable across the range.

That upstream bug also kills the "ship RUSTFLAGS from Makevars" option that was
on the table: `-C target-cpu=native` on an AVX-512 host cannot build this
dependency tree at all. The `#[target_feature]` route was the only one that
worked.

**2. `UNROLL` covers the reductions too, not just the argmin.** Same
loop-carried-latency argument, and with the arms being rewritten anyway the
marginal cost was a few lines each.

**3. The argmin needed a whole-vector remainder loop, not just the unroll.**
First A/B showed `UNROLL = 4` *losing* 1.4x at k = 200 while winning 2-3x
higher up. Cause: the block loop left up to `BLOCK - 1` elements (15 on NEON,
63 on AVX-512) to the scalar tail, which dominates at small k. Each arm now
folds its accumulators, runs the leftover whole vectors through the merged
accumulator, then goes scalar for the last `< W`. Tie semantics hold because
the leftover indices are strictly higher than anything already accumulated.

### Measured

`argmin_diff_simd_f32`, aarch64/NEON, A/B on the shipped code by flipping
`UNROLL`. This is the only arm that exercises `UNROLL` on this machine, so the
x86 arms remain unmeasured — compile-verified only.

| k | UNROLL = 1 | UNROLL = 4 + tail | speedup |
|---|---|---|---|
| 200 | 40.9 ns | 29.3 ns | 1.40x |
| 666 | 177.1 ns | 84.3 ns | 2.10x |
| 2666 | 860.5 ns | 305.9 ns | 2.81x |
| 16384 | 5358.8 ns | 1818.1 ns | 2.95x |

The bench was removed once it had served its purpose, as
`benches/seacells_bench.rs` was last round.

### Verified

`cargo test`: 225 no-default-features, 586 + 8 single-cell/multi-modal, 261
gpu, 608 + 8 + 13 + 1 gpu,single-cell including `tests/seacells_gpu.rs` CPU/GPU
parity. `cargo fmt` and `cargo clippy --all-targets` clean on both feature
sets. `cargo check --target x86_64-apple-darwin --all-targets` clean, which is
what actually compiles the 28 x86 intrinsic bodies — under the old
`cfg(target_feature)` gating none of them were ever built. Under
`-C target-feature=+avx512f` the error count in `src/` is zero; the remaining
failures are all inside `wide` itself.

### Left undone

- The same dead `cfg(all(target_arch, target_feature))` pattern is in
  `ann-search-rs/src/utils/dist.rs` in 10 places, which is where this module
  was derived from. Out of scope for this round; belongs upstream.
- Runtime behaviour of the AVX-2 and AVX-512 arms has never executed anywhere.
  The width-agreement tests are runtime-guarded on `is_x86_feature_detected!`,
  so they are no-ops here and only do real work in x86 CI.
- Worth reporting the `f64x8` AVX-512 bug upstream to `wide`.

## Risks

- The x86 arms cannot be *run* on this machine. Mitigated by compile coverage
  (below) plus runtime-guarded tests that x86 CI will exercise.
- AVX-512 downclocking on Skylake-X era parts can make the 512-bit arm a net
  loss on those CPUs. Not worth gating further, but worth knowing if a Linux
  benchmark comes back flat.
- Raising the `wide` floor breaks any consumer locked to 1.4.x until it runs
  `cargo update -p wide`.

## Verification

```bash
# every intrinsic compiles, without leaving the Mac
rustup target add x86_64-apple-darwin
cargo check --target x86_64-apple-darwin --features single-cell,multi-modal

# the old latent break: must now succeed
RUSTFLAGS="-C target-feature=+avx512f" cargo check --target x86_64-apple-darwin \
  --features single-cell,multi-modal

# CI parity
cargo test --no-default-features
cargo test --features single-cell,multi-modal
cargo test --features gpu

cargo fmt
cargo clippy --features single-cell,multi-modal --all-targets
cargo clippy --no-default-features --all-targets
```

- `cargo check` must be warning-free: the `use of deprecated trait wide::CmpLt`
  warning is the signal that step 1 landed.
- Extend `test_argmin_diff_widths_agree` (`:1469`) to call the AVX-512 arm,
  guarded by `is_x86_feature_detected!("avx512f")`, and to sweep lengths across
  the unrolled block sizes (4x8 = 32 and 4x16 = 64) as well as the existing
  vector boundaries. The tied-input case in
  `test_argmin_diff_simd_matches_scalar` (`:1427`) is what guards the unroll's
  tie semantics — keep it and add the new lengths there too.
- Add the same runtime-guarded width-agreement pattern for the reduction
  families in both files, comparing each explicit arm against its scalar
  reference. These are no-ops on aarch64 and do the real work in x86 CI.
- End-to-end: SEACells CPU-path hard assignments must be unchanged. Run the
  existing `cargo test --features single-cell,multi-modal` SEACells tests and
  confirm CPU/GPU parity still holds in `tests/seacells_gpu.rs`.
