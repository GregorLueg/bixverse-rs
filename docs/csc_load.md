# Load path: where the memory and the time go

Observations from `~/repos/shared/single-cell-benchmark` (branch `dry-run`),
bixverse 0.5.3 built from `main` (888cc02), Docker capped at 24 GB and 8
cores, cgroup `anon` memory sampled once a second. Parse 1M PBMC: 987,585
cells, 60,664 genes, 1.18B non-zeros; 984,669 cells and 39,137 genes pass the
min-quality filter. One repeat per run.

## Numbers

Full data set, `streaming = 2L` for both loaders:

| loader | load | peak anon during load | share of the run without tSNE |
|---|---|---|---|
| `load_mtx` (`mtx_streaming = TRUE`) | 231.6 s | 10.8 GB | 232 of 392 s |
| `load_h5ad` | 157.7 s | 17.2 GB | 158 of 327 s |

Everything after load stays at or below 10 GB. For scale: Scanpy Dask and
Seurat + BPCells peak at 9.5 and 9.4 GB for the whole run.

## h5ad: the peak is the cell-side write, not the transpose

Timeline of the full-data `load_h5ad`, from its log and the memory samples:

| phase | when | memory |
|---|---|---|
| Step 1/4, two stats passes over the CSR (4.8 s + 9.1 s) | t+0 to ~15 s | ~2 GB |
| Step 3/4, "Loading filtered data from h5", 984,669 cells (16.5 s) | ~15 to ~31 s | climbing |
| Step 4/4, "Writing to binary format" | from ~32 s | **17.2 GB at t+31.9 s** |
| CSR to CSC in `.dispatch_gene_based_data` (what `streaming` controls) | rest of the 158 s | 8 to 10 GB |

`streaming = 2L` only reaches the last phase. The peak comes before it.

Code path:

- `bixverse/R/methods_sc_io.R:1018`: `load_h5ad` always calls
  `rust_con$h5ad_to_file()`, whatever `streaming` is.
- `bixverse-rs/src/single_cell/sc_data/h5ad_io.rs:279`, `write_h5_counts`:
  Step 3 (line 337) materialises the whole filtered matrix via
  `read_h5ad_x_data_csr` into a `CompressedSparseData2<u32>`, and only then
  does Step 4 (line 353) walk it cell by cell into `CellGeneSparseWriter`.
- `read_h5ad_x_data_csr` (line 1392) builds it in `Vec<u32>` data,
  `Vec<usize>` indices and `Vec<usize>` indptr, starting from `Vec::new()`
  without a capacity. That's 12 bytes per non-zero, about 14 GB for ~1.17B
  non-zeros, and every reallocation doubles transiently. That fits the 17.2
  GB. It then converts to `u32` indices, another copy.

A streaming path already exists and is not wired up:
`SingleCellCountData$h5ad_to_file_streaming` (`bixverse/src/rust/src/single_cell/r_count_obj.rs:522`)
calls `stream_h5_counts` (`h5ad_io.rs:439`), and there are
`write_h5_csr_streaming` (1531), `write_h5_csc_to_csr_streaming` (957) and
`write_h5_dense_row_streaming` (2041). `load_mtx` switches to
`mtx_to_file_streaming` when `mtx_streaming = TRUE`
(`methods_sc_io.R:1618`), which probably explains its lower 10.8 GB peak.
`load_h5ad` has no equivalent switch.

## Suggested order

1. Route `load_h5ad` (and `stream_h5ad`) through `h5ad_to_file_streaming` for
   `streaming >= 1L`, like `load_mtx`. Measure the peak. The expectation is
   that h5ad drops to the 8 to 10 GB of the conversion phase or below, but
   that's unmeasured.
2. If the non-streaming path stays: pre-size the vectors from `indptr` and
   the filter masks, keep indices as `u32` from the start, and drop the
   conversion copy.
3. Then the transpose. For h5ad, everything after Step 4 starts (Step 4's
   write plus the CSR to CSC conversion) is ~125 s of the 158 s load at full
   size. The mtx log has no timestamps, so its split is not measured. Either
   way it is the biggest single item in a cold run.

## Watch out

- `CompressedSparseData2` keeps `indptr` as `u32`. That caps a matrix at
  ~4.29B non-zeros, and a ~5M cell Tahoe-100M plate may be near or past that.
  Check before the scale sweep.
- These are single runs. Re-run the full data set after each change:
  `INPUT=h5ad TAG=h5ad BIX_KNN=nndescent scripts/run.sh bixverse parse_full`
  (and without `INPUT`/`TAG` for mtx) from the benchmark repo, after
  `scripts/vendor.sh` and an image rebuild.
