# Handover: v5 store, Zarr ingest, scx export

Plan: `~/.claude/plans/wondrous-foraging-kernighan.md`. Branch
`worktree-wondrous-foraging-kernighan`, on top of `feat-new-annsearch-rs`
(`e85a8b7`; not `main`, which lacks the e-means work).

## Done

**Commit 1 (`4f1c3c9`): phases 1 to 3.** All CI test sets green (sc-data 146,
sc 723, facade, sc-gpu 46, core 323), all-features clippy clean, Python
workspace clippy clean. `bixverse-r` clippy clean. Python tests (pytest) not
run.

- `SingleCellReading::get_header()` replaced by `dims() -> StoreDims`
  (`total_cells`, `total_genes`, `cell_based`). Breaks scx's bixverse reader
  and the R package's Rust code wherever they call `get_header()`.
- Norm is f32 end to end: `norm_value -> f32`, `data_norm: Vec<f32>`, gene LUT
  f32, ranking (`fast_ranking`) on f32 bits. `avg_exp` stays `F16` (v3/v4 header
  field). Downstream numbers shift slightly; R/Python parity fixtures not
  regenerated yet.
- v5 container (`v5_io.rs`): 128 byte header (feature kind, norm kind, n_layers,
  target size, cell-ID hash, incompat bits), section directory (offsets read via
  mmap, library index / cell table, original-index pairs only when out of order,
  IDF, interval index). Writers default to v5; `new_v4` / `new_v3` for tests.
  Verbatim norm stored f32 in v5 (fixes gene/cell view disagreement).
- Readers: v5 and v4 open; v3 refused (`V3NeedsConversion`), `open_any` for
  converters. Converters `convert_*_to_v5`, `upgrade_store_to_v5` (v3 and v4 in).
- Raw layers: `write_cell_layers` / `write_gene_layers`, `with_layer`,
  `write_gene_file` handles layers.
- TF-IDF: `tf_idf_weights`, `set_tf_idf` (metadata only, no chunk rewrite).
  Tested against the formula, not yet against Signac/muon (Signac not installed;
  use `uv run --with muon`).
- Cell IDs: `set_cell_ids`, `check_same_cells`, writer `set_cell_ids`. Ingest
  paths do not yet record barcodes.
- Interval index (`interval_index.rs`): `GenomeIntervals`, overlap / window /
  contiguous range, `attach_interval_index`, `reader.genome_intervals()`.
- Archive bumped to version 2 (f32 verbatim norms; v1 still restores).
- Python: `upgrade_to_v5` binding, `upgrade_sc_store` takes a directory too,
  `load_existing` raises on v3, warns on v4.
- CLAUDE.md format section rewritten for v5.

**Commit 2 (WIP): phase 4 started.** Compiles with and without `zarr`, sc-data
tests pass. No Zarr test yet.

- Features `zarr` (implies `h5`) and `zarr-blosc` on sc-data, forwarded by the
  facade and `bixverse-r`. zarrs' blosc builds snappy (C++), so it is split out
  for the R link.
- `ann_store.rs`: `AnnFile` / `AnnArray` over HDF5 and Zarr (open by path,
  node, attrs, members, shape, ranged reads, indptr, 2D blocks, strings).
- `h5ad_io.rs` and `h5ad_multifile_io.rs` switched to `AnnFile`; `read_indptr`
  removed.

## Next

1. Phase 4 rest: `h5_scan` (`get_h5ad_dimensions`, `detect_raw_count_slot`)
   and `h5ad_frame_io::read_frame` / `obs_var_io` on `AnnFile` (categoricals,
   nullable, string arrays in Zarr); bindings accept `.zarr`. Fixtures written by
   anndata (`python/dev/gen_anndata_fixture.py`, v2 and v3, sharded, CSR / CSC /
   dense); check the Zarr-built store equals the h5ad-built one.
2. Record cell IDs during ingest (obs index of kept cells).
3. Phase 5: `crates/bixverse-sc-export` on scx-core with `save_h5ad` and
   `save_zarr_ad`; move `rs_save_h5ad` / `h5ad_write.rs` glue there.
4. Regenerate parity fixtures for the f32 norm; vignettes still say v4
   (`single_cell_io.py`, `design_single_cell.py`).
5. Measure v5 vs v4 file size and read speed (small case first).
