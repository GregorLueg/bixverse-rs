# Fix-up pass on `single_cell/sc_data/`

## Context

A review of the single-cell I/O layer turned up ten issues, none of which have bitten yet in
practice but several of which can. All ten were independently verified against the tree. Two are
genuinely dangerous:

- `CscGeneChunk::read_from_buffer` guards on 32 bytes and then slices 36, so a 32-35 byte chunk
  panics instead of returning `ChunkBufferTooSmall`.
- `migrate_v2_to_v3` compares the file version against `SC_FILE_VERSION` (3) rather than 2. It
  rejects the v2 files it exists to migrate, and lets v3 files through into the v2 chunk parsers,
  which read u32 gene indices as u16. That is silent corruption of a migrated dataset.

The root cause behind both is issue 9: `sc_data/` is 10,387 lines across 13 files with zero
`#[cfg(test)]` blocks. The only coverage is indirect, via `tests/scenic_gpu.rs`, which is gated
behind `feature = "gpu"`, so on a CPU-only build the binary format is never exercised at all. A
six-line round-trip test would have caught both bugs.

Two corrections to the original list: `GENE_BATCH_SIZE` is re-declared in `hotspot.rs:1027`, not
`scenic.rs` (which has no such const), and `CELL_BATCH_SIZE` has eight declarations rather than
three, with function-local consts silently shadowing the `pub` global brought in by the prelude glob.

Decisions taken: breaking changes to the public surface are in scope (the downstream R package gets
updated to match), the format fix is limited to `target_size` with obs/var left to the atlas
manifest, the v2 migration gets fixed rather than deleted, and test coverage targets the binary
format core rather than the h5ad/mtx readers.

## Key finding that shapes the work

`FileHeader` (`data_io.rs:962-978`) is bincode-encoded into a **zero-padded fixed 64-byte slot**
(`data_io.rs:1056-1061`, mirrored in `finalise` at `1184-1190`) and decoded from `&mmap[0..64]`
(`data_io.rs:1522`). It carries 35 unused reserved bytes. Swapping four of them for a
`target_size: f32` is byte-compatible in both directions:

- Existing files have zeros there, so the new decoder yields `0.0`, a usable "unknown" sentinel
  (a real `target_size` is never 0).
- Total encoded length is unchanged, so the 64-byte slot still fits.

No `SC_FILE_VERSION` bump, no invalidated files on disk. `finalise` rebuilds the header from
scratch via `FileHeader::new(self.cell_based)`, so `target_size` must be stored on the writer
struct to survive into finalisation.

Second shaping finding: `num-traits` is already a non-optional dependency (`Cargo.toml:63`) and
`RawCounts` already exposes `RawCountsIter` yielding `u32` (`data_io.rs:184-195`). The u16 clamp is
forced purely by the `T: From<u16>` bound, because `f32` implements `From<u16>` but not
`From<u32>`. Switching to `FromPrimitive` fixes it with no new machinery and lets the U16/U32 match
arms collapse into one loop.

## Work

Ordered so signature changes cascade forwards rather than needing revisits.

### 1. `src/errors.rs`

Add to the `// -- Binary file I/O --` section (line 184), matching the existing style: `///` doc
comment, then `#[cfg(feature = "single-cell")]`, then `#[error(...)]`, named struct fields each with
their own doc comment.

- `RawCountOverflow { value: u32, target_type: &'static str }` for a raw count that does not fit `T`.
- `RawElemSizeInvalid(u8)` for an unrecognised element-size discriminant.
- `ChunkPayloadTruncated { expected: usize, found: usize }` for a chunk whose declared payload
  lengths exceed the buffer.
- `TargetSizeMismatch { header: f32, requested: f32 }` for issue 10.

Delete `SerialisationFailed` (275-278) and `DeserialisationFailed` (280-284). Both have zero
construction sites and no path needs them; `HeaderEncodeFailed` / `HeaderDecodeFailed` already
cover header serialisation. Note `DeserialisationFailed`'s `#[error]` string is a copy-paste of
`SerialisationFailed`'s, which is further evidence it was never used.

`ObsColumnMissing`, `LibSizeLengthMismatch`, `MtxHeaderInvalid` and `MtxParseError` get wired in
below rather than deleted.

### 2. Chunk parsing: issues 1 and 3

`src/single_cell/sc_data/data_io.rs`, both `read_from_buffer` impls (`CsrCellChunk` at 457,
`CscGeneChunk` at 796).

Replace the two magic header lengths with module-top consts carrying the layout reasoning, per the
`GEMM_TILE_SIZE` convention:

```rust
/// Header length of a CSR cell chunk: 4+4+4+8+8+4 bytes. See
/// `CsrCellChunk::write_to_bytes` for the field layout.
const CSR_CHUNK_HEADER_LEN: usize = 32;
/// Header length of a CSC gene chunk: 4+4+4+2+2+8+8+4 bytes.
const CSC_CHUNK_HEADER_LEN: usize = 36;
```

Then in both impls:

- Guard on the correct constant. This alone fixes issue 1: the `CscGeneChunk` guard currently says
  32 while `data_start = 36` (line 819) and `write_to_bytes` (759-770) writes 36.
- After parsing the three length fields, guard the **total** required size
  (`norm_end + col_indices_len * 4`) and return `ChunkPayloadTruncated`. This is the missing check
  that matters more than the off-by-four: all three parsers currently compute payload offsets from
  untrusted length fields and never validate them. `&buffer[data_start..data_end]` panics on a bad
  length, and the two `unsafe` reads that follow read out of bounds silently.
- Replace the raw-pointer reads (`496-507`, `836-847`) with
  `chunks_exact(N).map(|c| uX::from_le_bytes(c.try_into().unwrap()))`. This matches the safe
  precedent already in the same file at `data_io.rs:1536-1540` and `1573-1577`, and it also removes
  a real alignment bug: `norm_end` is `header_len + data_raw_len * elem_size + data_norm_len * 2`,
  so for `elem_size == 2` and odd `data_raw_len + data_norm_len` the `*const u32` read lands on a
  2-byte-aligned address. The buffer is an lz4-decompressed `Vec<u8>`
  (`decompress_chunk`, `data_io.rs:1565-1583`), so `align_of::<u8>() == 1` is all that is
  guaranteed. If a bench shows this on a hot path, keep an unsafe fast path behind an explicit
  alignment check and give it a `// SAFETY:` comment; `data_io.rs` currently has nine `unsafe`
  blocks with none.

`RawCounts::read_from_buffer` (142-161) becomes
`Result<Self, BixverseErrors>`: check `buffer.len() >= count * elem_size as usize`, match
`RAW_ELEM_U32 => u32`, `RAW_ELEM_U16 | 0 => u16` (zero covers legacy files where the discriminant
byte was padding, per the existing comment), and return `RawElemSizeInvalid` for anything else. The
current `_ =>` catch-all silently reinterprets a corrupt discriminant as u16 and misparses the whole
payload.

`RawCounts::write_bytes` (115-128) casts `*const u16`/`*const u32` to `*const u8` and writes
**native-endian**, while every header field in the format uses `to_le_bytes`. Switch it to
`to_le_bytes` so the format is consistently little-endian and matches the new reader.

### 3. Raw count widening: issue 2

`from_gene_chunks` (`data_io.rs:1813`) and `from_cell_chunks` (`1887`):

- Bound becomes `T: BixverseNumeric + FromPrimitive` (`num_traits::FromPrimitive`, already imported
  in `src/utils/traits.rs:7`).
- Return `Result<CompressedSparseData2<T, f32>, BixverseErrors>`.
- Drop the `RawCounts::U16` / `U32` match entirely and iterate `chunk.data_raw.iter()`, which
  already yields `u32`. Convert with
  `T::from_u32(val).ok_or(BixverseErrors::RawCountOverflow { value: val, target_type: "..." })?`.
- Update the doc comments, which currently advertise the saturation as a caveat (lines 1801 and
  1876). Note in the docs that `T = f32` rounds above 2^24; irrelevant for counts, but honest.

Six call sites need `?`: `gpu/sc_gpu/pca_gpu.rs:111`, `sc_processing/utils_doublets.rs:382`,
`sc_processing/pca.rs:910`, `sc_analysis/nmf_sc.rs:197` and `:277`, `sc_analysis/scenic.rs:3799`.
All six pass `DataLayerReturn::Norm` today, which is why the clamp is currently dead code and the
bug has not surfaced.

Then delete `assemble_pile_csr` (`mc_generation/metacells2/pile.rs:90-103`) and call
`from_cell_chunks::<u32>(chunks, &DataLayerReturn::Raw, n_genes)?` instead. That private helper
exists only to dodge the saturation, and its doc comment says so; once the saturation is gone it is
a duplicate. This also gives `from_cell_chunks` its first caller (it currently has zero).

### 4. Writer: issues 4a, 4b, 6

`CellGeneSparseWriter` in `data_io.rs`:

- `write_cell_chunk` (1095) and `write_gene_chunk` (1131): replace the `assert!` on `self.cell_based`
  with `ReaderModeMismatch { actual, requested }`, which already exists (`errors.rs:261-273`) and is
  already used on the reader side with the literals `"cell-based"` / `"gene-based"`. Factor the
  check into one private `fn check_mode(&self, want_cell_based: bool) -> Result<(), BixverseErrors>`.
  Return type goes from `std::io::Result<()>` to `Result<(), BixverseErrors>`; every existing `?`
  call site keeps compiling because `BixverseErrors: From<io::Error>`. Remove the "will panic" lines
  from both doc comments.
- The tails of the two writers are otherwise identical (write length prefix, write payload, bump
  `no_chunks`, flush check). Factor into one private helper. This fixes issue 6 by construction:
  `chunks_since_flush` is currently incremented at exactly one line (1157), inside
  `write_gene_chunk` only, so the `cell_based` arm of `flush_frequency` (the `100000` at line 1066)
  is unreachable dead configuration and its doc comment at 1022-1024 describes behaviour that does
  not exist.
- Promote the two flush cadences to documented module-top consts. Flag for review: 100,000 cell
  chunks against a 128 MiB `BufWriter` (line 1051) means the flush effectively never fires even
  once it is wired up. 10,000 is a more meaningful cell-side default, but that is a tuning change
  on top of a correctness fix, so keep the existing values unless you want otherwise.
- `finalise` (1168): replace the two `.unwrap()`s on `encode_to_vec` (1172, 1181) with
  `.map_err(|_| BixverseErrors::HeaderEncodeFailed)?`. `new` already handles the identical call
  correctly fourteen lines up (1054-1055). Return type becomes `Result<(), BixverseErrors>`; all
  17 call sites keep working.

### 5. Remaining panics and dead variants: issues 4c, 4d, 5

`src/single_cell/sc_data/r_obj_io.rs`: `write_r_counts` (29) and `write_r_counts_csr` (80) return a
bare `(usize, usize, CellQuality)` and unwrap three fallible writer calls (116, 130, 150). Change
both to `Result<(usize, usize, CellQuality), BixverseErrors>` and use `?`. This is the only writer
path in `sc_data/` that is not `Result`-based; every h5ad/h5/mtx equivalent already returns
`Result<(usize, usize, CellQuality), BixverseErrors>`. As it stands a bad `bin_path` from R aborts
the R session.

`src/single_cell/sc_data/h5ad_io.rs`, all inside functions that already return
`Result<_, BixverseErrors>`:

- `:2735` `unwrap_or_else(|_| panic!("obs column '{}' not found in h5ad file", ...))` becomes
  `ObsColumnMissing`. The panic string is a verbatim duplicate of that variant's `#[error]` string
  (`errors.rs:322`), and the variant's doc comment already names this function as its raising site.
- `:2739` `assert_eq!` on library-size length becomes `LibSizeLengthMismatch`. Same story.
- `:2802` and `:2812`: `reconstruct_and_write_csc(...).unwrap()` / `_csr(...).unwrap()`. Both
  helpers return `Result<CellQuality, BixverseErrors>`; plain `?` works.
- `:394` `writer.finalise().unwrap()` inside `write_h5_counts`: `?`.
- `:1959-1960` and `:2080-2081` `*cell_chunk.iter().min().unwrap()` / `.max().unwrap()`: confirm the
  batch slice cannot be empty, then switch to `expect("batch non-empty by construction")`, which the
  style rules allow for documented by-construction invariants.

`src/single_cell/sc_data/mtx_io.rs`: inspect the header and body parsing. If malformed input
currently panics or unwraps, wire in `MtxHeaderInvalid` and `MtxParseError`. If it already errors
through a different variant, delete those two variants instead. This is the one sub-step whose
outcome is not yet determined.

### 6. `target_size` in the header: issue 10

`data_io.rs`:

- `FileHeader`: `_reserved_1: [u8; 32]` becomes `target_size: f32, _reserved_1: [u8; 28]`. Field
  order matters, so put `target_size` first. `FileHeader::new` gains a `target_size: f32` parameter.
- `CellGeneSparseWriter`: new documented `target_size: f32` field, new fifth parameter on `new`, and
  `finalise` passes `self.target_size` into `FileHeader::new`. Seventeen construction sites to
  update; pass `cell_quality.target_size` where a `MinCellQuality` is in scope (`data_io.rs:236-247`
  already carries it) and `0.0` on raw-only paths such as `bin_merge_io.rs:179` and
  `depracated_conversion.rs:275`.
- `ParallelSparseReader`: store the decoded value and expose
  `pub fn target_size(&self) -> Option<f32>`, returning `None` for `0.0`. The reader currently
  decodes `FileHeader` and throws everything away except `main_header_offset`.

Then close the two holes the missing field causes:

- `CscGeneChunk::transform_to_clr` (`sc_processing/pca.rs:145-164`) has to undo the
  `ln1p(count / lib_size * target_size)` normalisation, so it needs the exact factor the writer
  used, but can only take it as an argument. `params_pca.size_factor` defaults to `1e4`
  (`pca.rs:136`), so a file written with `target_size = 1e5` and an un-overridden default computes
  garbage with no warning. At the five call sites (`pca.rs:453`, `:749`, `:897`,
  `gpu/sc_gpu/pca_gpu.rs:88`, `sc_batch_correction/seurat_cca.rs:125` and `:173`) prefer the
  reader's header value when present, and return `TargetSizeMismatch` when the header and the
  requested factor disagree. A mismatch is always a bug, since undoing the file's own normalisation
  is the entire point of the call.
- `merge_sc_bin_files` (`bin_merge_io.rs`) documents the invariant instead of checking it: lines
  146-151 say "the caller is responsible for ensuring all inputs were normalised against the same
  `target_size`". With the field in the header, compare the inputs when `renormalise == false` and
  return `TargetSizeMismatch`. Update the doc comment to describe the check rather than the
  obligation.

Barcodes, gene names and feature types stay out of the format, per the atlas manifest design.

### 7. Batch-size constants: issue 7

The values are **not** interchangeable, so this is a renaming and hoisting job, not a unification
one. Collapsing everything onto the 50,000 global would change HDF5 slice heights by 50x.

- `single_cell/mod.rs:21` keeps `pub const CELL_BATCH_SIZE: usize = 50_000;` as the reader default
  (consumed at `data_io.rs:1457`).
- `sc_processing/qc.rs:113` and `:271`: hoist the two function-local `100000`s to one documented
  module-level `QC_CELL_BATCH_SIZE`.
- `h5ad_io.rs:1547`, `:2047`, `:2429` and `h5_10x_io.rs:702`: these four `1000`s are HDF5 slice
  heights, not cell batches. Replace with one shared documented
  `pub(crate) const H5_CELL_SLICE_SIZE: usize = 1_000;` in `sc_data/mod.rs` (15 lines today).
- `h5_10x_multifile_io.rs:86` is already documented and module-private; rename for consistency.
- `GENE_BATCH_SIZE`: hoist the four identical copies in `hvg.rs` (580, 849, 1200, 1499) to one
  module-level const, same for `utils_doublets.rs:1030` and `hotspot.rs:1027`. Leave
  `sc_annotation/sc_type.rs:24` alone; its `100` is documented and used as a genuine default
  (`sc_type.rs:332`).

The point is to kill the silent shadowing, not to retune anything. Behaviour should be
byte-identical afterwards.

### 8. v2 migration guard: issue 8

`src/prelude.rs`: add `pub const SC_FILE_VERSION_V2: u32 = 2;` next to `SC_FILE_VERSION` (line 36).
No v2 constant exists anywhere today.

`src/single_cell/sc_data/depracated_conversion.rs:239`: compare against `SC_FILE_VERSION_V2` and
report it as `expected`. Fix the misleading struct comment at line 126 ("v2 file header (same layout
as v3, just a different version number)"), which is where the confusion came from. The module doc
at 1-11 is unambiguous about the direction the migration runs.

### 9. Tests: issue 9

Inline `#[cfg(test)] mod tests` at the bottom of `data_io.rs`, under a `///////////\n// Tests //\n///////////`
banner, `use super::*;` first, matching `sc_analysis/fast_ranking.rs:303-333`. Already behind
`single-cell`, so `cargo test --features single-cell,multi-modal` picks it up. Must not require
`gpu`.

Temp files follow the `tests/scenic_gpu.rs:1364` precedent
(`std::env::temp_dir().join("bixverse_<name>.bin")`, unique per test) rather than adding a
dev-dependency: there is no `[dev-dependencies]` section in `Cargo.toml` at all, `approx` is a
normal dependency, and `tempfile` is absent from `Cargo.lock` even transitively. Add a small RAII
guard struct in the test module so cleanup survives a failing assert, which the existing integration
tests do not manage.

Coverage, each test targeting a specific bug on this list:

1. `write_to_bytes` then `read_from_buffer` field-equality round-trip, both chunk types.
2. A 32-to-35 byte buffer handed to `CscGeneChunk::read_from_buffer` returns
   `ChunkBufferTooSmall { expected: 36, .. }` rather than panicking. Direct regression for issue 1.
3. Valid header, payload lengths claiming more bytes than present, expect `ChunkPayloadTruncated`.
   Regression for the missing payload bounds check.
4. Odd `data_raw_len` with `elem_size == 2`, so the indices read starts at a 2-byte-aligned offset.
   Regression for the alignment bug in issue 3.
5. A raw count of 70,000 survives write, read and `from_cell_chunks::<u32>`. Regression for issue 2.
6. `from_gene_chunks::<u16>` on that same value returns `RawCountOverflow` instead of silently
   yielding 65,535.
7. An invalid element-size discriminant returns `RawElemSizeInvalid`; a zero discriminant still
   parses as u16 for legacy files.
8. Full writer/reader round-trip through a temp file, cell-based and gene-based, comparing
   reconstructed data against the input.
9. A cell-based writer handed a gene chunk returns `ReaderModeMismatch`. Regression for issue 4a.
10. `target_size` survives a header round-trip, and a header with zeroed reserved bytes reads back
    as `None`. Regression for issue 10 and proof the format stayed backwards-compatible.
11. `migrate_v2_to_v3` accepts a synthetic v2 fixture and rejects a v3 file. Regression for issue 8.
    The fixture has to be hand-assembled: a 64-byte `FileHeaderV2` with `version = 2`, one
    lz4-compressed chunk in the v2 layout (u16 counts, u16 indices, no discriminant byte, 32-byte
    header), and the tail `SparseDataHeader`. There is no v2 writer to borrow, so this is the most
    involved test of the set.

## Files touched

Core: `src/errors.rs`, `src/prelude.rs`, `src/single_cell/sc_data/data_io.rs` (the bulk of it),
`src/single_cell/sc_data/r_obj_io.rs`, `src/single_cell/sc_data/h5ad_io.rs`,
`src/single_cell/sc_data/depracated_conversion.rs`, `src/single_cell/sc_data/bin_merge_io.rs`,
`src/single_cell/sc_data/mod.rs`, `src/single_cell/sc_data/mtx_io.rs`.

Mechanical follow-on from the signature changes: `src/single_cell/mc_generation/metacells2/pile.rs`,
`src/single_cell/sc_processing/pca.rs`, `src/single_cell/sc_processing/qc.rs`,
`src/single_cell/sc_processing/hvg.rs`, `src/single_cell/sc_processing/utils_doublets.rs`,
`src/single_cell/sc_analysis/nmf_sc.rs`, `src/single_cell/sc_analysis/scenic.rs`,
`src/single_cell/sc_analysis/hotspot.rs`, `src/single_cell/sc_batch_correction/seurat_cca.rs`,
`src/gpu/sc_gpu/pca_gpu.rs`, plus the remaining writer-construction sites in
`h5_10x_io.rs`, `h5_10x_multifile_io.rs`, `h5ad_multifile_io.rs`, `mtx_multifile_io.rs`, and
`tests/scenic_gpu.rs:1275`.

## Verification

```bash
cargo fmt
cargo clippy --features single-cell,multi-modal --all-targets
cargo test --no-default-features
cargo test --features single-cell,multi-modal
cargo test --features gpu,single-cell          # tests/scenic_gpu.rs uses the writer directly
cargo doc --features single-cell,multi-modal
```

The `gpu` pass matters more than usual here: `tests/scenic_gpu.rs` is the only existing exercise of
the binary format, and it constructs a `CellGeneSparseWriter` at line 1275 and calls `finalise` at
1307, both of which change signature.

Backwards compatibility is the one thing the test suite cannot prove on its own, since no `.bin`
fixture is checked in. Confirm by hand against a real pre-change file: open an existing `.bin` with
the new `ParallelSparseReader`, check it opens without a version error and that `target_size()`
returns `None`, and diff a reconstructed `CompressedSparseData2` against the same file read by the
current `main` build. Test 10 covers the synthetic half of this; the on-disk half needs a real file.

Also worth a real-data run of one full ingest path end to end (h5ad in, `.bin` out, PCA on top),
since `write_h5_normalised_counts` is where four of the panic sites live and where `target_size`
first enters the format.

## Downstream

The R package needs updating for these signature changes:

- `write_r_counts` / `write_r_counts_csr` now return `Result<(usize, usize, CellQuality), BixverseErrors>`
  instead of a bare tuple. `IntoExtendrErr::to_extendr` (`src/utils/traits.rs:274-285`) exists for
  exactly this and is currently called nowhere in this crate.
- `CellGeneSparseWriter::new` takes a fifth `target_size: f32` argument.
- `write_cell_chunk`, `write_gene_chunk` and `finalise` return `Result<_, BixverseErrors>` rather
  than `std::io::Result<_>`.
- `from_gene_chunks` and `from_cell_chunks` are now fallible.
