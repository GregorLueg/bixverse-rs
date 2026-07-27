# Documentation cleanup: GPU correlation and Gram

## Context

`d2c51ed` replaced the cubek product in the GPU correlation path with a dedicated
symmetric Gram kernel. `gram.rs` landed with thorough documentation; `corr.rs`
was carried along and its docs never caught up. The gap shows in four ways:

- `corr.rs` and `gram.rs` describe the *same buffer* with opposite shape
  metadata, and `corr.rs`'s own doc disagrees with its own allocation calls.
- `corr.rs` carries a bare `#![allow(missing_docs)]` with no explanation, unlike
  every other file in `src/gpu/` that has one.
- The bench module doc still says the product goes through cubek. It doesn't.
  That is the one piece of prose in the set that states the opposite of what the
  code does.
- Several `### Params` / `### Returns` entries restate the parameter name instead
  of saying anything (`verbose` - "Controls verbosity of the function").

Goal: `corr.rs` reads at the standard `gram.rs` already sets, the two files agree
on layout notation, and the bench stops lying. Plus one non-behavioural code fix
the user approved (shape vectors).

Not in scope: `linalg/mod.rs`, `gpu/mod.rs`'s `WORKGROUP_64` typo,
`sparse_rand_svd_gpu.rs`'s wrong filename reference. Flagged, left alone.

## Files

- `src/gpu/linalg/corr.rs` (the bulk of the work)
- `src/gpu/linalg/gram.rs` (small)
- `benches/gpu_corr_bench.rs` (module doc only, plus one shape vector)

## 1. Shape metadata fix (code, no behaviour change)

`GpuTensor::{empty, from_slice}` take shape and derive row-major strides from it.
Every kernel on this path flat-indexes (`data[idx as usize]`), so the strides are
never read and allocation size is `shape.iter().product()`, identical either way.
The swap is metadata only.

- `src/gpu/linalg/corr.rs:295` — `vec![n_rows, n_cols]` -> `vec![n_cols, n_rows]`
- `src/gpu/linalg/corr.rs:450` — same
- `benches/gpu_corr_bench.rs:272` — `vec![n, d]` -> `vec![d, n]`

`gram.rs:481` is already `vec![d, n]` and needs nothing. After this, all four
upload sites agree with the docs and with `gram_aat`'s declared `[d, n]`.

## 2. Settle one layout notation, use it everywhere

`corr.rs` currently mixes three spellings for one thing: `[n_cols, n_rows]` in the
kernel docs, `(N x d)` in `column_pairwise_cor_gpu`, and `[n_rows, n_cols]` in the
code. `gram.rs` says `[d, n]`.

Pick **feature-major `[d, n]`**, matching `gram.rs`, and state it once per module
doc: feature `i`'s samples live contiguously at `A[i * n ..]`. Kernel-level docs
then use the local parameter names (`[n_cols, n_rows]` in `corr.rs`) but only
after the module doc has established that this *is* `[d, n]`. One sentence in each
module doc cross-referencing the other file is enough to join them.

## 3. `corr.rs` specifics

- **`:1-3` module doc.** Rewrite. Drop "the faer stuff" and the unquantified
  "should be faster" claim, or point it at `benches/gpu_corr_bench.rs` instead of
  asserting a crossover nobody measured. State the pipeline: optional host-side
  rank transform, centre and scale on device, one `gram_aat`. State the layout
  convention (section 2). Add a `###` subsection if it earns one; neighbours use
  `### Algorithm` / `### Pipeline` / `### Structure`.
- **`:5`.** Add the house comment above the attribute, verbatim from `gram.rs:30`:
  `// The `#[cube]` macro generates undocumented launcher structs and functions.`
- **`:22-32` `GpuCorCov`.** "Enum for the dispatch" is not a sentence. Say it
  selects the variant host-side. Move `#[default]` below `Covariance`'s doc
  comment (attributes go after doc comments; it currently sits between the enum
  doc and the variant doc). Fix mid-sentence capitals in the variant docs.
- **`:34-50` `parse_gpu_cor`.** `### Returns` describes the input, not the
  return. Say what it returns and that an unrecognised string gives `None`. List
  the accepted aliases; they are invisible from the signature.
- **`:56-78` `column_stats`.** Good, one gap: the unrolled reduction ladder
  (`if tx < 64 / 32 / 16 / ...`) is hardcoded to a 128-thread workgroup. Launching
  it at any other width silently gives wrong sums. That invariant belongs in the
  doc, tied to the `WORKGROUP_128` at `:302`.
- **`:258-281` `scale_matrix_col_gpu`.** Say why the two dispatches use different
  widths: `column_stats` is one workgroup per column and pinned to 128 by the
  reduction ladder, `apply_centre_scale` is a flat elementwise pass at 256.
  Currently undocumented at both ends.
- **`:372-400` `column_pairwise_cor_gpu`.** `mat` "(N x d)" -> file convention.
  `### Returns` "(d x d)" likewise. `verbose` - "Controls verbosity of the
  function" -> say it prints upload time and total elapsed. Keep the cubek
  paragraph at `:379-382`, it is the most useful prose in the file.
- **Test module.** Add one-line docs to the seven helpers (`try_device`,
  `assert_mat_close`, `cpu_col_means`, `cpu_pearson`, `cpu_covariance`,
  `cpu_rank_cols`, `cpu_spearman`), matching `gram.rs`'s treatment of `build_a` /
  `gram_host` / `run`. Leave the `//` comments above individual `#[test]` fns as
  they are; that is house style.
- **`:600` divider.** `// Actual tests //` -> `// Tests //`. That label appears
  exactly twice in the repo, in these two files; everywhere else uses `// Tests //`.

## 4. `gram.rs` specifics

Already the best-documented file in the module. Three small things:

- Docs for `try_device` and `assert_close` in the test module (3/5 -> 5/5).
- `:507` `// Actual tests //` -> `// Tests //`.
- One line in the module doc tying `[d, n]` to `corr.rs`'s `[n_cols, n_rows]`,
  per section 2.

Leave the measured M1 Max numbers and the const rationale blocks alone. They are
the reason the file is worth reading.

## 5. `benches/gpu_corr_bench.rs:3-8`

Currently: "That product currently goes through cubek with `Strategy::DoubleUnit`,
pinned there because `Strategy::Auto` blows up on Apple devices."

The body (`:288-300`) runs `gram_aat` as the timed product and cubek `DoubleUnit`
as a comparison arm scored into `Stages::product_cubek`, which `Stages::total`
deliberately excludes. Rewrite the paragraph to say that: `gram_aat` is the
product, cubek is the baseline it replaced, both run so the replacement is checked
against the thing it replaced in one pass. The `Stages` field docs at `:212-229`
already describe this correctly and can be borrowed from.

Also `:3` `G = S^T S` -> `A A^T`. The buffer uploaded at `:272` is feature-major,
so `S^T S` implies a layout the bench does not use.

## Flagged, not touched

- `GpuCorCov` derives `CubeType` (`corr.rs:23`) but never enters a `#[cube]` body.
  Every use is host-side `matches!`. The derive is dead.
- `parse_gpu_cor` (`corr.rs:43`) has zero callers in `src/`, `tests/` or
  `benches/`, and is not re-exported through `gpu_r_wrappers.rs`.

## Verification

```bash
cargo fmt
cargo clippy --features gpu --all-targets
cargo test --features gpu
cargo doc --features single-cell,multi-modal   # doc links resolve
```

`cargo test --features gpu` is the real check on the shape swap: `gram.rs`'s
`test_gram_aat_*` and `corr.rs`'s `test_pearson_matches_cpu` /
`test_covariance_matches_cpu` / `test_spearman_matches_cpu` all compare against a
CPU reference, so a stride regression would show as a numerical failure rather
than silently. `tests/gpu_corr.rs` covers the same path end to end.

`cargo doc` catches broken intra-doc links, which matters because the pass adds
cross-file references between `corr.rs` and `gram.rs`.

A GPU-less machine skips rather than fails every one of these tests (`try_device`
returns `None`), so a green run on such a box proves nothing. Confirm the tests
actually ran.
