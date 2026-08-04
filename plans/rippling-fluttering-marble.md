# Fix-up pass on `single_cell/sc_data/`

## Context

A review of the single-cell I/O layer turned up ten issues in `sc_data/`. None have bitten in
practice, but several can. Two are dangerous:

- `CscGeneChunk::read_from_buffer` (`data_io.rs:797`) guards on 32 bytes then slices `buffer[0..36]`,
  so a 32-35 byte chunk panics instead of returning `ChunkBufferTooSmall`.
- `migrate_v2_to_v3` (`depracated_conversion.rs:239`) compares the file version against
  `SC_FILE_VERSION` (3) rather than 2. It rejects the v2 files it exists to migrate and lets v3 files
  through into the v2 chunk parsers, which read u32 gene indices as u16. Silent corruption.

Behind both sits the real problem: `sc_data/` is 10,387 lines across 13 files with zero
`#[cfg(test)]` blocks. The only coverage is indirect via `tests/scenic_gpu.rs`, gated behind
`feature = "gpu"`, so a CPU-only build never exercises the binary format at all.

Separately, raw counts are saturated to `u16` at four points on the write and read paths. The
machinery has been tested extensively downstream and no single gene has been seen to exceed
`u16::MAX`, but silent saturation is the wrong failure mode and it is not free to fix later once
corrupt `.bin` files exist.

## Backwards compatibility

**Existing `.bin` files keep opening. No `SC_FILE_VERSION` bump.**

- `FileHeader` (`data_io.rs:962-978`) is bincode-encoded into a zero-padded fixed 64-byte slot
  (`data_io.rs:1056-1061`, mirrored in `finalise` at `1184-1190`) and decoded from `&mmap[0..64]`
  (`data_io.rs:1522`). bincode 2 `config::standard()` encodes `[u8; 32]` as 32 raw bytes and `f32` as
  4 fixed bytes, so replacing `_reserved_1: [u8; 32]` with `target_size: f32, _reserved_1: [u8; 28]`
  occupies the same 32 bytes at the same offset. Old files hold zeros there, decoding to `0.0`, a
  usable "unknown" sentinel because a real `target_size` is never 0.
- It is forward-compatible too: a file written by the new code reads on an old build, which folds
  those 4 bytes back into `_reserved_1` and ignores them.
- `RawCounts::write_bytes` switching from native-endian pointer casts to `to_le_bytes` is a
  byte-level no-op on every platform in the CI matrix (x86_64, aarch64). Same for the read side.
- The chunk-header const fix only tightens a guard. `data_start` was already 36 and
  `write_to_bytes` already wrote 36 bytes.
- The mtx bucket spill record grows 10 to 12 bytes, but those temp files are created and deleted
  inside a single `process_mtx_and_write_bin_streaming` call under `TempFileGuard`. Nothing on disk
  survives the call.
- v2 files are currently *unopenable* (the reader rejects `version != 3` and the migration is
  broken). Step 8 restores that path rather than breaking anything.

## Work

Ordered so signature changes cascade forwards.

### 1. `src/errors.rs`

Add to the `// -- Binary file I/O --` section (line 184), matching the existing style: `///` doc
comment, `#[cfg(feature = "single-cell")]`, `#[error(...)]`, named struct fields each documented.

- `RawCountOverflow { value: u32, target_type: &'static str }`
- `RawElemSizeInvalid(u8)`
- `ChunkPayloadTruncated { expected: usize, found: usize }`
- `TargetSizeMismatch { header: f32, requested: f32 }`

Delete `SerialisationFailed` (275-278) and `DeserialisationFailed` (280-284): zero construction
sites, and `HeaderEncodeFailed` / `HeaderDecodeFailed` already cover header serialisation.
`DeserialisationFailed`'s `#[error]` string is a copy-paste of `SerialisationFailed`'s.

`ObsColumnMissing`, `LibSizeLengthMismatch`, `MtxHeaderInvalid` and `MtxParseError` are also
unconstructed today but get wired in below.

### 2. Chunk parsing: issues 1 and 3

`data_io.rs`, both `read_from_buffer` impls (`CsrCellChunk` at 457, `CscGeneChunk` at 796).

Replace the magic header lengths with module-top consts carrying the layout reasoning, per the
`GEMM_TILE_SIZE` convention:

```rust
/// Header length of a CSR cell chunk: 4+4+4+8+8+4 bytes. See
/// `CsrCellChunk::write_to_bytes` for the field layout.
const CSR_CHUNK_HEADER_LEN: usize = 32;
/// Header length of a CSC gene chunk: 4+4+4+2+2+8+8+4 bytes.
const CSC_CHUNK_HEADER_LEN: usize = 36;
```

In both impls:

- Guard on the correct constant. Fixes issue 1.
- After parsing the three length fields, guard the **total** required size
  (`norm_end + indices_len * 4`) and return `ChunkPayloadTruncated`. This matters more than the
  off-by-four: both parsers compute payload offsets from untrusted length fields and never validate
  them, so `&buffer[data_start..data_end]` panics on a bad length and the two `unsafe` reads that
  follow read out of bounds silently.
- Replace the raw-pointer reads (`496-507`, `836-847`) with
  `chunks_exact(N).map(|c| uX::from_le_bytes(c.try_into().expect("...")))`. This matches the safe
  precedent in the same file at `1536-1540` and `1573-1577`, and removes a real alignment bug:
  `norm_end = header_len + data_raw_len * elem_size + data_norm_len * 2`, so for `elem_size == 2`
  and odd `data_raw_len + data_norm_len` the `*const u32` read lands on a 2-byte-aligned address.
  The buffer is an lz4-decompressed `Vec<u8>` (`decompress_chunk`, `1565-1583`), so
  `align_of::<u8>() == 1` is all that is guaranteed. `data_io.rs` currently has nine `unsafe` blocks
  with zero `// SAFETY:` comments.

`RawCounts::read_from_buffer` (142-161) becomes `Result<Self, BixverseErrors>`: check
`buffer.len() >= count * elem_size as usize`, match `RAW_ELEM_U32 => u32`, `RAW_ELEM_U16 | 0 => u16`
(zero covers legacy files where the discriminant byte was padding, per the existing comment), and
return `RawElemSizeInvalid` otherwise. The current `_ =>` catch-all silently reinterprets a corrupt
discriminant as u16 and misparses the whole payload.

`RawCounts::write_bytes` (115-128) casts to `*const u8` and writes native-endian while every header
field uses `to_le_bytes`. Switch to `to_le_bytes` for consistency and to match the new reader.

### 3. Raw count widening: issue 2

`from_gene_chunks` (`data_io.rs:1813`) and `from_cell_chunks` (`1887`):

- Bound becomes `T: BixverseNumeric + FromPrimitive` (`num_traits::FromPrimitive`, already imported
  in `src/utils/traits.rs:7`; `num-traits` is a non-optional dependency at `Cargo.toml:63`).
- Return `Result<CompressedSparseData2<T, f32>, BixverseErrors>`.
- Drop the `RawCounts::U16` / `U32` match entirely and iterate `chunk.data_raw.iter()`, which
  already yields `u32` via `RawCountsIter` (`data_io.rs:184-195`). Convert with
  `T::from_u32(val).ok_or(BixverseErrors::RawCountOverflow { value: val, target_type: "..." })?`.
- Update the doc comments at 1801 and 1876, which currently advertise the saturation as a caveat.
  Note that `T = f32` rounds above 2^24; irrelevant for counts, but honest.

Six call sites need `?`: `gpu/sc_gpu/pca_gpu.rs:111`, `sc_processing/utils_doublets.rs:382`,
`sc_processing/pca.rs:910`, `sc_analysis/nmf_sc.rs:197` and `:277`, `sc_analysis/scenic.rs:3799`.
All six pass `DataLayerReturn::Norm`, which is why the clamp is dead code today.

Then delete `assemble_pile_csr` (`mc_generation/metacells2/pile.rs:90-103`) and call
`from_cell_chunks::<u32>(chunks, &DataLayerReturn::Raw, n_genes)?`. That private helper exists only
to dodge the saturation and its doc comment says so. It also gives `from_cell_chunks` its first
caller.

### 3b. Remaining write-path saturation

- `sc_processing/scrublet.rs:187`: the doublet simulation sums two cells then does
  `count.min(u16::MAX as u32) as u16` and stores `RawCounts::U16(data_raw)` at `:201`. Keep the
  accumulator as `Vec<u32>` and use `RawCounts::from_u32_auto(&data_raw)`, which already picks the
  narrow variant when it fits (`data_io.rs:167`).
- `sc_data/mtx_io.rs`: widen `parse_mtx_line` (863) to `Option<(u32, u32, u32)>` and drop the
  `.min(u16::MAX as u32)`. Cascades to `cell_data: Vec<Vec<(u32, u32)>>` (425), `gene_counts:
  Vec<u32>` (497, 722), the bucket record (691-722 and the writer at ~660: 10 bytes to 12, value as
  `u32`), and `n_entries = bucket_bytes.len() / 12`. `CsrCellChunk::from_data` is generic over
  `T: FloatAndUInt` and `u32` already implements it (`sc_traits.rs:270-279`), and internally goes
  through `to_u32()` then `RawCounts::from_u32_auto`, so no signature change is needed there.
  Update the `total_umi` accumulation to stay `u64` given wider counts.
- `sc_data/mtx_multifile_io.rs`: same widening for `parse_mtx_coord` (293-330),
  `cell_data: Vec<Vec<(u32, u32)>>` (500) and `gene_counts` (535). No spill format here, it is
  purely in-memory. Fix the two doc comments at 243 and 283 that promise saturation.

`sc_annotation`/`mc_analysis` `as u16` casts are in synthetic test data generators and stay as they
are.

### 4. Writer: issues 4a, 4b, 6

`CellGeneSparseWriter` in `data_io.rs`:

- `write_cell_chunk` (1095) and `write_gene_chunk` (1131): replace the `assert!` on
  `self.cell_based` with `ReaderModeMismatch { actual, requested }`, which exists
  (`errors.rs:261-273`) and is already used reader-side with the literals `"cell-based"` /
  `"gene-based"`. Factor into one private
  `fn check_mode(&self, want_cell_based: bool) -> Result<(), BixverseErrors>`. Return type goes from
  `std::io::Result<()>` to `Result<(), BixverseErrors>`; existing `?` call sites keep compiling
  because `BixverseErrors: From<io::Error>`. Drop the "will panic" lines from both doc comments.
- The tails of the two writers are otherwise identical (length prefix, payload, bump `no_chunks`,
  flush check). Factor into one private helper. This fixes issue 6 by construction:
  `chunks_since_flush` is incremented at exactly one line (1157) inside `write_gene_chunk` only, so
  the `cell_based` arm of `flush_frequency` (the `100000` at 1066) is unreachable dead config and
  the doc comment at 1022-1024 describes behaviour that does not exist.
- Promote the two flush cadences to documented module-top consts. **Keep the existing values**
  (100,000 cell-side, 1,000 gene-side). Correctness fix only, no retuning.
- `finalise` (1168): replace the two `.unwrap()`s on `encode_to_vec` (1172, 1181) with
  `.map_err(|_| BixverseErrors::HeaderEncodeFailed)?`, which is what `new` already does fourteen
  lines up (1054-1055). Return type becomes `Result<(), BixverseErrors>`.

### 5. Remaining panics and dead variants: issues 4c, 4d, 5

`sc_data/r_obj_io.rs`: `write_r_counts` (29) and `write_r_counts_csr` (80) return a bare
`(usize, usize, CellQuality)` and unwrap three fallible writer calls (116, 130, 150). Change both to
`Result<(usize, usize, CellQuality), BixverseErrors>` and use `?`. This is the only writer path in
`sc_data/` that is not `Result`-based; every h5ad/h5/mtx equivalent already returns that exact type.
As it stands a bad `bin_path` from R aborts the R session.

`sc_data/h5ad_io.rs`, all inside functions already returning `Result<_, BixverseErrors>`:

- `:2735` `unwrap_or_else(|_| panic!("obs column '{}' not found in h5ad file", ...))` becomes
  `ObsColumnMissing`. The panic string is a verbatim duplicate of that variant's `#[error]` string
  (`errors.rs:323`) and the variant's doc comment already names this function.
- `:2739` `assert_eq!` on library-size length becomes `LibSizeLengthMismatch`.
- `:2802`, `:2812`: `reconstruct_and_write_csc(...).unwrap()` / `_csr(...).unwrap()`. Both return
  `Result<CellQuality, BixverseErrors>`; plain `?`.
- `:394` `writer.finalise().unwrap()` in `write_h5_counts`: `?`.
- `:1959-1960`, `:2080-2081` `*cell_chunk.iter().min().unwrap()` / `.max().unwrap()`: confirm the
  batch slice cannot be empty, then switch to `expect("batch non-empty by construction")`.

`sc_data/mtx_io.rs`: resolved during exploration. It has zero `unwrap`/`panic!`/`assert` and already
errors through `std::io::Error::new(ErrorKind::InvalidData, ...)` in `parse_header` (129-145).
Wire `MtxHeaderInvalid` / `MtxParseError` in to replace those ad-hoc io errors so the failure mode
is typed like the rest of the crate, and give `parse_header` a `Result<MtxHeader, BixverseErrors>`
return.

### 6. `target_size` in the header: issue 10

`data_io.rs`:

- `FileHeader`: `_reserved_1: [u8; 32]` becomes `target_size: f32, _reserved_1: [u8; 28]`. Field
  order matters for the encoding; `target_size` goes first. `FileHeader::new` gains a
  `target_size: f32` parameter.
- `CellGeneSparseWriter`: new documented `target_size: f32` field, new fifth parameter on `new`, and
  `finalise` passes `self.target_size` into `FileHeader::new` (it rebuilds the header from scratch,
  so the value has to live on the writer struct to survive finalisation).
- 17 construction sites. Pass `cell_quality.target_size` where a `MinCellQuality` is in scope
  (`data_io.rs:236-247` carries it), `target_size` directly in `bin_merge_io.rs:179` where it is
  already a function parameter, and `0.0` on genuinely raw-only paths such as
  `depracated_conversion.rs:275`.
- `ParallelSparseReader`: store the decoded value and expose `pub fn target_size(&self) -> Option<f32>`
  returning `None` for `0.0`. The reader currently decodes `FileHeader` and discards everything
  except `main_header_offset`.

Then close the two holes:

- `CscGeneChunk::transform_to_clr` (`sc_processing/pca.rs:145-164`) undoes the
  `ln1p(count / lib_size * target_size)` normalisation, so it needs the exact factor the writer
  used, but can only take it as an argument. `SingleCellPcaParams::size_factor` defaults to `1e4`
  (`pca.rs:136`), so a file written with `target_size = 1e5` and an un-overridden default computes
  garbage silently. At the six call sites (`pca.rs:453`, `:749`, `:897`, `gpu/sc_gpu/pca_gpu.rs:88`,
  `sc_batch_correction/seurat_cca.rs:125` and `:173`) prefer the reader's header value when present
  and return `TargetSizeMismatch` when header and request disagree. A mismatch is always a bug,
  since undoing the file's own normalisation is the point of the call. Files written before this
  change report `None` and fall back to the requested value, so nothing regresses.
- `merge_sc_bin_files` (`bin_merge_io.rs`) documents the invariant instead of checking it: lines
  146-151 say "the caller is responsible for ensuring all inputs were normalised against the same
  `target_size`". With the field in the header, compare the inputs when `renormalise == false` and
  return `TargetSizeMismatch`. Skip the check when any input reports `None`. Update the doc comment
  to describe the check rather than the obligation.

Barcodes, gene names and feature types stay out of the format, per the atlas manifest design.

### 7. Batch-size constants: issue 7

The values are **not** interchangeable, so this is renaming and hoisting, not unification.
`CELL_BATCH_SIZE` has eight declarations and the function-local ones silently shadow the `pub`
global brought in by the prelude glob (`prelude.rs:19`).

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

Behaviour must be byte-identical afterwards.

### 8. v2 migration guard: issue 8

`src/prelude.rs`: add `pub const SC_FILE_VERSION_V2: u32 = 2;` next to `SC_FILE_VERSION` (line 36).

`sc_data/depracated_conversion.rs:239`: compare against `SC_FILE_VERSION_V2` and report it as
`expected`. Fix the misleading struct comment at line 126 ("v2 file header (same layout as v3, just
a different version number)"), which is where the confusion came from. `FileHeaderV2` keeps its
`[u8; 32]` reserved block; only v3's `FileHeader` gains `target_size`. The migration writer passes
`0.0`.

### 9. Tests: issue 9

Inline `#[cfg(test)] mod tests` at the bottom of `data_io.rs`, under a
`///////////\n// Tests //\n///////////` banner, `use super::*;` first, matching
`sc_analysis/fast_ranking.rs:303-333`. Already behind `single-cell`, so
`cargo test --features single-cell,multi-modal` picks it up. Must not require `gpu`.

Temp files follow the `tests/scenic_gpu.rs:1364` precedent
(`std::env::temp_dir().join("bixverse_<name>.bin")`, unique per test) rather than adding a
dev-dependency: there is no `[dev-dependencies]` section in `Cargo.toml`, `approx` is a normal
dependency, and `tempfile` is absent from `Cargo.lock` even transitively. Add a small RAII guard
struct so cleanup survives a failing assert (`mtx_io.rs:59` has a `TempFileGuard` Drop impl to copy).

Coverage, each test targeting one bug:

1. `write_to_bytes` then `read_from_buffer` field-equality round-trip, both chunk types.
2. A 32-to-35 byte buffer to `CscGeneChunk::read_from_buffer` returns
   `ChunkBufferTooSmall { expected: 36, .. }` rather than panicking. Issue 1.
3. Valid header, payload lengths claiming more bytes than present, expect `ChunkPayloadTruncated`.
4. Odd `data_raw_len` with `elem_size == 2`, so the indices read starts 2-byte aligned. Issue 3.
5. A raw count of 70,000 survives write, read and `from_cell_chunks::<u32>`. Issue 2.
6. `from_gene_chunks::<u16>` on that value returns `RawCountOverflow` rather than 65,535.
7. An invalid element-size discriminant returns `RawElemSizeInvalid`; a zero discriminant still
   parses as u16 for legacy files.
8. Full writer/reader round-trip through a temp file, cell-based and gene-based.
9. A cell-based writer handed a gene chunk returns `ReaderModeMismatch`. Issue 4a.
10. `target_size` survives a header round-trip, and a header with zeroed reserved bytes reads back
    as `None`. Issue 10, and proof the format stayed backwards-compatible.
11. `migrate_v2_to_v3` accepts a synthetic v2 fixture and rejects a v3 file. Issue 8. The fixture is
    hand-assembled: a 64-byte `FileHeaderV2` with `version = 2`, one lz4-compressed chunk in the v2
    layout (u16 counts, u16 indices, no discriminant byte, 36-byte header per the doc block at
    `depracated_conversion.rs:117-135`), and the tail `SparseDataHeader`. There is no v2 writer to
    borrow, so this is the most involved test of the set.
12. An mtx line with a count above `u16::MAX` parses to the full `u32` value. Step 3b.

## Files touched

Core: `src/errors.rs`, `src/prelude.rs`, `src/single_cell/sc_data/data_io.rs` (the bulk),
`r_obj_io.rs`, `h5ad_io.rs`, `depracated_conversion.rs`, `bin_merge_io.rs`, `mod.rs`, `mtx_io.rs`,
`mtx_multifile_io.rs`.

Mechanical follow-on from the signature changes: `mc_generation/metacells2/pile.rs`,
`sc_processing/{pca,qc,hvg,utils_doublets,scrublet}.rs`,
`sc_analysis/{nmf_sc,scenic,hotspot}.rs`, `sc_batch_correction/seurat_cca.rs`,
`gpu/sc_gpu/pca_gpu.rs`, the remaining writer-construction sites in `h5_10x_io.rs`,
`h5_10x_multifile_io.rs`, `h5ad_multifile_io.rs`, and `tests/scenic_gpu.rs:1275`.

## Downstream

Smaller than it looks. `IntoExtendrErr` (`utils/traits.rs:274-285`) is blanket over `E: Display`, so
every existing `.to_extendr()?` keeps compiling when a return type moves from `std::io::Result<_>`
to `Result<_, BixverseErrors>`. **bixverse.gpu needs no source changes at all**: it only calls
`aggregate_meta_cells`, which is untouched.

`~/repos/shared/bixverse` (branch `feat-bixverse-update`), all in
`src/rust/src/single_cell/r_count_obj.rs`:

- `:331` `write_r_counts(...)` now returns a `Result`. Add `.to_extendr()?`.
- `:1128`, `:1195`, `:1326` `CellGeneSparseWriter::new(&self.f_path_genes, false, ...)` needs the
  fifth `target_size` argument. All three sites have a `ParallelSparseReader` open on
  `self.f_path_cells` in scope (confirmed at `:1180`, `:1310`; verify the `:1128` site during
  implementation), so `reader.target_size().unwrap_or(0.0)` works with no R-side API change.

For local testing, add a temporary `[patch.crates-io]` entry to `bixverse/src/rust/Cargo.toml`
pointing `bixverse-rs` at this worktree, so `R CMD INSTALL` builds against the change. Strip it
before release. Both packages currently pin `bixverse-rs = "0.3.12"` while the crate is at `0.3.13`,
so the pin needs bumping at release time regardless.

## Verification

```bash
cargo fmt
cargo clippy --features single-cell,multi-modal --all-targets
cargo test --no-default-features
cargo test --features single-cell,multi-modal
cargo test --features gpu,single-cell          # tests/scenic_gpu.rs uses the writer directly
cargo doc --features single-cell,multi-modal
```

The `gpu` pass matters more than usual: `tests/scenic_gpu.rs` is the only existing exercise of the
binary format and it constructs a `CellGeneSparseWriter` at 1275 and calls `finalise` at 1307, both
of which change signature.

Backwards compatibility is the one thing the suite cannot prove on its own, since no `.bin` fixture
is checked in. Confirm by hand against a real pre-change file: open an existing `.bin` with the new
`ParallelSparseReader`, check it opens without a version error, that `target_size()` returns `None`,
and diff a reconstructed `CompressedSparseData2` against the same file read by the current `main`
build. Test 10 covers the synthetic half; the on-disk half needs a real file.

Also run one full ingest path end to end (h5ad in, `.bin` out, PCA on top), since
`write_h5_normalised_counts` is where four of the panic sites live and where `target_size` first
enters the format. Do one mtx ingest too, both the in-memory and the bucketed streaming path, since
the bucket record width changes.
