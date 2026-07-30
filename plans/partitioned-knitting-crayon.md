# Abstract the single-cell streaming reader behind a trait

## Context

`ParallelSparseReader` in `src/single_cell/sc_data/data_io.rs` is welded to
local files: it mmaps a `.bin` and every downstream function reaches for it by
constructing one from a `&str` path. That works well today, but it binds the
whole single-cell stack to on-disk binary files. Atlas-scale data living
somewhere else has no way in without touching every consumer.

This refactor extracts the reading surface into a `SingleCellReading` trait and
makes downstream functions generic over it. `ParallelSparseReader` stays the
only implementor. The file format, the writers, and the R-facing API do not
change.

Explicitly **not** in scope: Zarr, S3, object-store dependencies, async, and any
speculative error variants for network failure. The goal is to make a second
backend *possible* later, not to build one now.

## Design decisions (confirmed with user)

- **Static generics**, not `dyn`: `fn foo<S: SingleCellReading>(reader: &S)`.
  With one implementor, monomorphisation is free.
- **Readers only.** `CellGeneSparseWriter` and the h5ad / mtx / 10x ingestion
  converters keep taking paths. They are bound to local-file semantics (seek,
  `BufWriter`, in-place header rewrite) and are a separate concern.

## The three repos

```
bixverse-rs  (crate)
    |  path dep, features: single-cell, multi-modal
    v
bixverse  (R package, Rust inside)
    |  R-level: Imports bixverse (>= 0.4.0), via Remotes: GregorLueg/bixverse
    v
bixverse.gpu  (R package, Rust inside)
       crates.io dep on bixverse-rs 0.3.9, features: gpu, single-cell
```

`bixverse` and `bixverse.gpu` are **siblings** as far as the crate goes: both
depend on bixverse-rs directly, and each has its own Rust crate and extendr
wrapper layer. They are only stacked at the R level, where `bixverse.gpu`
imports `bixverse`. Both need the same wrapper-layer treatment.

They differ in how they pin the crate, and that difference matters:

```toml
# bixverse/src/rust/Cargo.toml  -- local path, already coupled
# TODO: flip back to `version = "0.3.10"` once bixverse-rs is on crates.io
bixverse-rs = { path = "/Users/gregorlueg/repos/shared/bixverse-rs", features = [
  "single-cell", "multi-modal" ] }

# bixverse.gpu/src/rust/Cargo.toml  -- published version, NOT a path
bixverse-rs = { version = "0.3.9", features = [ "gpu", "single-cell" ] }
```

`bixverse` has no version-pin escape hatch: a breaking change lands on its next
`cargo build`. Its path also points at the **main checkout, not this worktree**,
so testing without repointing silently builds the old crate and looks like a
clean pass.

`bixverse.gpu` has the opposite hazard. It resolves 0.3.9 from crates.io, so
until it is given a path override or a `[patch.crates-io]` entry, it will keep
building against the *published* crate and its tests will pass while proving
nothing. There is currently no `[patch.crates-io]` section. Its
`feat-rust-refactor` branch exists locally with no commits ahead of `main`.

Feature coverage across the two consumers is complementary: `bixverse` enables
`single-cell,multi-modal`; `bixverse.gpu` enables `gpu,single-cell`. Between
them every feature combination the crate ships is exercised, and `bixverse.gpu`
is the **only** consumer that exercises the GPU code at all.

**The R boundary already sits above the reader.** The extendr layer constructs
`ParallelSparseReader` itself in 21 places. The wrappers keep their `f_path`
strings, build a reader, and pass `&reader` down. So the change is fully
containable in the wrapper bodies: no `.R` file, no roxygen, no `man/` page
changes.

**Nothing is dead.** An initial sweep flagged `read_gene`, `get_all_cells`,
`read_cells_range`, `get_clr_offsets`, `read_gene_nnz` and `get_all_gene_nnz` as
having zero in-crate callers and suggested trimming them. They are all called
from the R package (`get_clr_offsets` at 6 sites, `read_cells_range` inside the
memory-bounded transpose). Zero in-repo callers on a `pub` item is not dead
code. The trait keeps the full method set.

## Why this is tractable

- The reader already returns **owned** chunks (decompress into `Vec<u8>`, then
  parse). Nothing borrows from the mmap across the API boundary, so no lifetime
  gymnastics.
- The API is already **batch-oriented** (`&[usize]` of indices).
- `memmap2` appears in exactly two files: `data_io.rs` and the already-deprecated
  `depracated_conversion.rs`. The boundary is clean.

The dominant call-site pattern is a one-line construction at the top of a
function whose only reason to take `f_path` is to build the reader:

```rust
pub fn get_hvg_vst(f_path: &str, ...) -> Result<HvgRes, BixverseErrors> {
    let reader = ParallelSparseReader::new(f_path)?;
    ...
}
```

becomes

```rust
pub fn get_hvg_vst<S: SingleCellReading>(reader: &S, ...) -> Result<HvgRes, BixverseErrors> {
    ...
}
```

## The trait

Lives in `src/single_cell/sc_data/data_io.rs`, re-exported from `prelude.rs`
alongside the existing `ParallelSparseReader` export.

`Send + Sync` is mandatory, not optional: readers are used inside `rayon`
closures throughout (e.g. `cell_aggregation_utils.rs:52`, the scenic batch
loops) and the R package puts one in an `Arc`.

```rust
pub trait SingleCellReading: Send + Sync {
    // cell-based (CSR)
    fn read_cells_parallel(&self, indices: &[usize]) -> Result<Vec<CsrCellChunk>, BixverseErrors>;
    fn read_cell(&self, index: usize) -> Result<CsrCellChunk, BixverseErrors>;
    fn read_cells_range(&self, start: usize, end: usize) -> Result<Vec<CsrCellChunk>, BixverseErrors>;
    fn get_all_cells(&self) -> Result<Vec<CsrCellChunk>, BixverseErrors>;
    fn read_cell_library_sizes(&self, indices: &[usize]) -> Result<Vec<usize>, BixverseErrors>;
    fn get_clr_offsets(&self, indices: &[usize], batch_size: Option<usize>) -> Result<Vec<f64>, BixverseErrors>;

    // gene-based (CSC)
    fn read_gene_parallel(&self, indices: &[usize]) -> Result<Vec<CscGeneChunk>, BixverseErrors>;
    fn read_gene(&self, index: usize) -> Result<CscGeneChunk, BixverseErrors>;
    fn read_gene_parallel_filtered(&self, indices: &[usize], cells_to_keep: &IndexSet<u32>) -> Result<Vec<CscGeneChunk>, BixverseErrors>;
    fn get_all_genes(&self) -> Result<Vec<CscGeneChunk>, BixverseErrors>;
    fn read_gene_nnz(&self, indices: &[usize]) -> Result<Vec<usize>, BixverseErrors>;
    fn get_all_gene_nnz(&self) -> Result<Vec<usize>, BixverseErrors>;

    // metadata
    fn get_header(&self) -> SparseDataHeader;
    fn is_cell_based(&self) -> bool;
    fn is_gene_based(&self) -> bool { !self.is_cell_based() }
}
```

Give default bodies where one method is expressible via another (`read_cell` via
`read_cells_parallel`, `get_all_cells` / `read_cells_range` via the header plus
`read_cells_parallel`, `get_all_gene_nnz` via `read_gene_nnz`, `is_gene_based`).
A future backend then only implements the genuinely primitive operations.

`ParallelSparseReader`'s inherent methods move into
`impl SingleCellReading for ParallelSparseReader` essentially unchanged. Keep
`ParallelSparseReader::new(path)` as an inherent constructor: it is
backend-specific, and a `-> Result<Self, _>` return would break object safety if
`dyn` is ever wanted.

**Name the generic parameter `S`, not `R`.** `gpu/sc_gpu/pca_gpu.rs:46` and
`scenic_gpu.rs` already carry `R: Runtime`.

## Structs that must become generic

Five, more than a first pass suggests:

- `ScenicSetup` (`sc_analysis/scenic.rs:3758`) holds `pub reader: ParallelSparseReader`.
  `pub(crate)`, so generifying is contained. Consumed at scenic.rs:3936, 3950,
  4013, 4027 and scenic_gpu.rs:6133, 6321.
- `Hotspot` (`sc_analysis/hotspot.rs:804`) holds `f_path_gene: String` and rebuilds
  a reader four times (`:953`, `:1034`, `:1310`, `:1465`). Hold the reader
  instead: also a performance win, since it stops re-mmapping the same file.
- `Scrublet` (`sc_processing/scrublet.rs:227/229`), `BoostClassifier`
  (`doublet_detection.rs:244/246`) and `ScDblFinder` (`scdblfinder.rs:1189/1191`)
  each store **both** `f_path_gene` and `f_path_cell`.

## Gene-based and cell-based in the same function (confirmed with user)

Plenty of methods need both a CSC (gene-major) and a CSR (cell-major) file:
`module_scoring::calculate_module_scores_main`, `Hotspot::new`,
`ScDblFinder::new`, `Scrublet::new`, `BoostClassifier::new`, and on the R side
`rs_sc_pca`, `rs_mnn`, `rs_seurat_cca`, `rs_seurat_rpca`, `rs_build_symphony_ref`.

**Decision: one trait, one type parameter, two arguments.**

```rust
pub fn calculate_module_scores_main<S: SingleCellReading>(
    gene_reader: &S,
    cell_reader: &S,
    ...
)

pub struct ScDblFinder<S: SingleCellReading> {
    gene_reader: S,
    cell_reader: S,
    ...
}
```

Cell-based vs gene-based stays a **runtime** property of the file
(`header.cell_based`), exactly as today. Methods keep their self-guards and keep
returning `ReaderModeMismatch`. This is deliberate: splitting into separate
`CellReading` / `GeneReading` traits would read nicer but buys no compile-time
safety, because one concrete `ParallelSparseReader` would implement both (the
mode is a property of the file it opened, not of its type). Real compile-time
mode safety needs distinct types with marker parameters, which is a much larger
change and squarely in the deferred bucket.

Two arguments of the same `S` rather than two type parameters: in practice both
`.bin` files always come from the same backend, so an independent `G` and `C`
would add a parameter to every two-file signature for flexibility that may never
be used. If a future backend ever needs to mix sources, widening `S` to `<G, C>`
is a mechanical follow-up.

Where a reader is only used transiently in a constructor (Hotspot opens the cell
file in `new` to read library sizes, then discards it), keep it as a generic on
`new` only rather than a second field on the struct.

`symphony_map_query` is a variant worth noting: it reads a *different dataset*
(`f_path_query`), not a second view of the same one. That is another argument for
readers being passed in rather than derived from one shared context object.

## Files to change

26 files reference `ParallelSparseReader` (72 references, 55 construction sites).
Three buckets:

**Already take a reader, bound swap only** (`&ParallelSparseReader` to `&S`):
`seurat_cca.rs:110,158`, `cell_aggregation_utils.rs:33`, `metacells2/pile.rs:59`,
`scenic.rs:3041,3233,3366,3579`, `nichenet/prioritisation.rs:53` (already generic
over `T`, so it gains a second param).

**Take `f_path` only to build a reader, drop the path, add the generic**:
`sc_processing/{hvg,pca,qc,metrics,utils_doublets,scdblfinder,scrublet,doublet_detection}.rs`,
`sc_analysis/{hotspot,dge_pathway_scores,module_scoring,vision,nmf_sc,scenic}.rs`,
`sc_annotation/{sc_type,symphony}.rs`,
`sc_batch_correction/{seurat_cca,seurat_rpca,fast_mnn}.rs`,
`mc_generation/{cell_aggregation_utils,metacells2/mod}.rs`,
`sc_utils/cxds.rs`, `sc_data/plotting.rs`.

`hvg.rs` is the heaviest (8 construction sites plus 4 pure forwarders) and
`scenic.rs` the most tangled. Remember the intermediate forwarders (e.g.
`pca.rs:583,634,1012,1059`) become generic pass-throughs too.

**GPU** (needs `gpu` + `single-cell`): `gpu/sc_gpu/pca_gpu.rs:81`, and
`scenic_gpu.rs:6074,6240` via `scenic_common_setup`.

**Unchanged**: `bin_merge_io.rs` (needs real paths for the output writer), all
ingestion converters, `CellGeneSparseWriter`, `depracated_conversion.rs`,
`r_obj_io.rs`.

## Cleanups worth folding in

Small, in-scope, and this refactor touches the lines anyway:

- `cell_aggregation_utils.rs:225,294` use `ParallelSparseReader::new(f_path).unwrap()`.
  These become `?` once the reader is passed in.
- `scenic.rs:3041` and `:3366` take `f_path: &str` **and** `reader: &ParallelSparseReader`
  for the same file, and the path forwards down a chain that opens a *second*
  reader on it (`batch_genes` → `batch_genes_correlated` → `pca_on_sc_streaming`).
  Once generic, the path parameter disappears and the chain shares one reader.
- The metacell trio in the R package opens each file twice, once via
  `count_file_n_cells` for the header and once for data. Passing one reader down
  collapses that.

Do **not** fold in: renaming `f_path: &str` to `P: AsRef<Path>` for consistency
with the writers, and centralising the hardcoded `counts_cells.bin` /
`counts_genes.bin` filename literals on the R side. Both are real, neither is
this change.

## bixverse (R package) side

Change is confined to wrapper bodies in `src/rust/src/single_cell/`:
`r_count_obj.rs` (an extendr struct with 22 methods), `r_sc_analysis.rs`,
`r_sc_processing.rs`, `r_sc_metacells.rs`, `r_sc_batch_corr.rs`,
`r_sc_annotation.rs`, `r_sc_plot_extraction.rs`. Each becomes
`let reader = ParallelSparseReader::new(f_path)?;` then passes `&reader` down.

Three things to hold the line on:

- **Keep every R-facing signature byte-identical**, including the `streaming: bool`
  argument that most wrappers carry. Even if the trait makes streaming vestigial,
  it is part of the documented R API. Dropping it is the most likely accidental
  R break.
- **Never let the trait into a wrapper signature.** `#[extendr]` cannot expose a
  trait object or a generic. Paths in, reader constructed inside.
- **Convert the `.unwrap()`s** at `r_sc_metacells.rs:380,628,893` to `?` while
  hoisting construction. They currently panic-abort R rather than raising a
  condition; `?` is strictly better, though the error text R users see changes.

`r_sc_data.rs` (the v2→v3 migration) is inherently about paths and file formats.
Leave it path-based. Separately worth flagging: it is `pub use`d in `lib.rs` but
its `extendr_module!` is not listed in the top-level block, so
`rs_data_v2_3_conversion` may not be reachable from R at all. Existing issue, not
this refactor's problem.

## bixverse.gpu side

Much smaller than `bixverse`: **three extendr functions, four call sites, and no
struct anywhere holds a path or a reader.** Paths arrive fresh from R on every
call and the one reader is a local that dies at the end of an `if` block.

- `single_cell/pca_gpu.rs:61` `rs_sc_pca_sparse_gpu(f_path_gene: &str, f_path_cell: &str, ...)`.
  Constructs the repo's only `ParallelSparseReader` at `:83` (solely for
  `get_clr_offsets` at `:85`) and forwards `f_path_gene` into
  `pca_on_sc_sparse_gpu` at `:96`.
- `single_cell/scenic_gpu.rs:89` `rs_scenic_grn_gpu(f_path_genes: String, ...)`,
  forwarding at `:111`.
- `single_cell/scenic_gpu.rs:155` `rs_scenic_grn_streaming_gpu(f_path_genes: String, ...)`,
  forwarding at `:177`.

Same rule as `bixverse`: keep the extendr signatures taking strings, build the
reader inside, pass `&reader` down. Then `R/extendr-wrappers.R`, the three
`man/*.Rd` files, and `vignettes/gpu_single_cell.qmd` / `gpu_scenic.qmd` all stay
untouched.

Note `pca_gpu.rs:83` is the same conditional-open shape seen in `rs_mnn`: the
reader is built only to compute CLR offsets. Since `pca_on_sc_sparse_gpu` opens
the gene file regardless, hoisting to one reader per file is fine, but check the
cell-file open is still only paid when actually needed.

`run_scenic_grn_in_memory_gpu` (`scenic_gpu.rs:240`) and `harmony_v2_gpu`,
`knn_gpu`, the k-means and correlation entry points all take matrices or
in-memory sparse lists. Unaffected.

Two cross-repo couplings worth knowing. The R layer fetches both paths through
`bixverse:::get_rust_count_gene_f_path()` / `get_rust_count_cell_f_path()`,
unexported internals of the *other* R package (`R/sc_gpu.R:663-664`,
`R/scenic_gpu.R:186`). Keeping paths at the R boundary means these are
unaffected, but they are the coupling point if the R object ever hands over a
reader handle instead of a path. Also, `bixverse.gpu/CLAUDE.md` claims
bixverse-rs is pinned to branch `fix-scenic-gpu`; that is stale, it is a
crates.io pin. Worth correcting while in there.

## Verification

```bash
# crate, matching CI (.github/workflows/test.yml)
cargo test --no-default-features
cargo test --features single-cell,multi-modal
cargo test --features gpu,single-cell   # NOT --features gpu alone: sc_gpu is
                                        # gated on single-cell too. CLAUDE.md
                                        # documents the wrong command here.
cargo clippy --features single-cell,multi-modal --all-targets
cargo fmt
```

Both R packages: repoint `src/rust/Cargo.toml` at the local crate, then
`rextendr::document()` and `tinytest::test_package(...)`. For `bixverse` that
means correcting the existing path to this worktree; for `bixverse.gpu` it means
*replacing the crates.io pin* with a path dep or a `[patch.crates-io]` entry.
Skipping that step in `bixverse.gpu` means testing the published 0.3.9 and
getting a green run that proves nothing.

**The acceptance criterion for "R users notice nothing" is an empty diff on
`R/extendr-wrappers.R` in both packages.** That file is generated from the
wrapper signatures. If it regenerates byte-identical, the R surface is provably
untouched. Any diff means the abstraction leaked and belongs back in the wrapper
layer.

The R suites are the real regression gate: there are **no Rust unit tests in
either R package**. `bixverse` has 46 tinytest files, 27 single-cell, ~1,150
expectations, 25 of which write real `counts_cells.bin` / `counts_genes.bin` into
`tempdir()` through the full ingestion path. `bixverse.gpu` has 8 files, of which
`test_sc_gpu.R` (739 lines) and `test_scenic_gpu.R` (304 lines) build real
`SingleCells` objects on disk via `bixverse::load_r_data()` and exercise the
disk-backed reader end to end. Those two are the only GPU coverage that exists.

Since `ParallelSparseReader` stays the only implementor and method bodies are
moved rather than rewritten, numerical output must be **bit-identical**. Capture
HVG selection, a PCA embedding and scenic scores on the pre-refactor commit and
diff after.

One behavioural check to make deliberately: several wrappers currently open a
reader only conditionally (e.g. `rs_mnn` builds one solely to fetch CLR offsets
when `pca_params.clr` is set). Hoisting to a single eager open is usually fine
because the downstream function opens the same file anyway, but confirm per site
that you are not introducing an error on a file that was previously never
touched.

## Sequencing

Crate first, entirely locally, with the R package pointing at the local checkout
throughout. Nothing is published or pushed at any point.

1. Define the trait, impl it for `ParallelSparseReader`, re-export in prelude.
   Crate compiles, nothing else touched.
2. Convert the "already take a reader" bucket. Mechanical, low risk.
3. Convert the `f_path` bucket module by module. Start with `sc_data/plotting.rs`
   (smallest, 4 sites) as the pilot, then `hvg.rs` as the heaviest to surface
   ergonomic problems early.
4. Generify the five structs. `ScenicSetup` first (contained), then the four
   two-file doublet/hotspot structs.
5. GPU module. `cargo test --features gpu,single-cell` plus, later,
   `bixverse.gpu`'s tinytest suite, which is the only real GPU coverage.
6. Crate green on all feature combinations before touching either R package.

## Parallelisation

A caveat first: a crate is one compile unit, so parallel agents editing
different files inside `bixverse-rs` still break each other. A signature change
in `pca.rs` forces edits in eleven other modules, and until those land the tree
does not compile, which means no agent can verify its own work. Fanning out
blindly across the crate produces a pile of conflicting half-edits.

So the split is by **dependency position**, measured by in-crate importers:

| Module | Importers | Treatment |
|---|---|---|
| `sc_processing/pca.rs` | 11 | hub, sequential, first |
| `sc_processing/utils_doublets.rs` | 4 | hub, sequential |
| `sc_processing/hvg.rs` | 2 | hub, sequential |
| `sc_analysis/scenic.rs` | 2 | hub, sequential |
| `qc.rs`, `sc_type.rs`, `symphony.rs`, `vision.rs`, `module_scoring.rs`, `plotting.rs`, `cell_aggregation_utils.rs` | 0 | leaf, parallel |
| `metrics.rs`, `cxds.rs`, `nmf_sc.rs`, `dge_pathway_scores.rs` | 1 | near-leaf, parallel |

**Sequential, single agent:** the trait definition (sequencing step 1), then the
four hubs in importer order. These are the changes that ripple, and they want one
coherent head.

**Parallel, worth fanning out:**

- The eleven leaf and near-leaf modules, once the trait and hubs are stable.
  Each touches files nobody else imports, so a `cargo check` per agent is
  meaningful. Give each agent one module and the trait signature.
- The two R packages, once the crate is green. `bixverse` and `bixverse.gpu` are
  separate repos with separate crates and no shared files, so these are
  genuinely independent. `bixverse.gpu` is the smaller job by far (3 functions).
- The five struct generifications, if `ScenicSetup` is done first. The four
  doublet/hotspot structs live in different files and do not import each other.

**Never in parallel:** anything touching `data_io.rs`, and the hub modules.

Rule of thumb for the fan-out stages: one agent per file, each responsible for
leaving the crate compiling on its own file's terms, with a single `cargo check
--features single-cell,multi-modal` at the join point before moving on.

## Local validation workflow

Everything stays local. **No `cargo publish`, no `git push`, no version bumps,
no release-facing changes in any of the three repos.** The work ends when all
three are green on the local machine. Gregor does the sniff tests and drives the
existing CI/CD for anything that leaves the laptop.

1. **Point both R packages at the local crate.**
   - `bixverse`: already a path dep, but it targets the *main checkout*. Land the
     crate work there or repoint at this worktree.
   - `bixverse.gpu`: currently a crates.io pin on 0.3.9. Swap for a path dep or
     add `[patch.crates-io]`.

   Getting either wrong means silently testing the old crate and getting a clean
   pass that means nothing. Confirm with `cargo tree -p bixverse-rs` that the
   resolved source is the local one before trusting any result.

   These overrides are **local scaffolding, not deliverables**. Flag them clearly
   so they are not mistaken for intended state; reverting them is Gregor's call
   at release time.
2. **Update the crate** (sequencing steps 1-6). Green on
   `--no-default-features`, `single-cell,multi-modal`, and `gpu,single-cell`.
3. **Update `bixverse`**: wrapper bodies, `rextendr::document()`, confirm the
   empty `R/extendr-wrappers.R` diff, run `tinytest::test_package("bixverse")`.
   All 46 files must pass with zero changes to any `.R` file or test. A test that
   needs editing is a bug in the refactor, not in the test.
4. **Update `bixverse.gpu`** on its `feat-rust-refactor` branch (currently empty):
   the three extendr functions, `rextendr::document()`, empty wrapper diff, then
   `tinytest::test_package("bixverse.gpu")`. Needs a working GPU.
5. **Stop.** Report the state of all three repos: what changed, what passed, and
   that the local dependency overrides are still in place and deliberately not
   reverted.

For reference only, not to be acted on: the eventual release order is crate,
then `bixverse`, then `bixverse.gpu`, since the last has
`Imports: bixverse (>= 0.4.0)` and reaches into its internals via `bixverse:::`.
