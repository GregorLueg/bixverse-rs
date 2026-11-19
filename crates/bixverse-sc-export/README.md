# bixverse-sc-export

Writes a [`bixverse-rs`](https://github.com/GregorLueg/bixverse-rs) single cell
store to h5ad or AnnData Zarr, through
[`scx-core`](https://crates.io/crates/scx-core).

The counts stream out of the store in cell batches, so `X` never sits in memory
whole: peak memory follows the batch, not the matrix. Pick any cells and any
genes, in any order. Genes you don't want are dropped while each cell is
decoded, so a batch over every gene is never built. obs, var, obsm, varm, obsp
and uns are checked against the output shape before the file is created, so bad
input fails up front and not halfway through the stream.

`X` is CSR, `float32`, raw or normalised counts. `indptr` switches to `int64`
once the non-zeros pass 2^31. anndata reads the output as is.

## Features

| Feature | What it adds |
| --- | --- |
| (none) | h5ad, gzip level of your choice |
| `zarr` | AnnData Zarr v3. Turns on Blosc in `zarrs`, which builds C++ |
| `hdf5-static` | Builds the bundled HDF5 instead of linking an external `libhdf5` |

## Use

```rust
use bixverse_sc_data::store::reader::ParallelSparseReader;
use bixverse_sc_export::ad_export::{AdFormat, AdParts, ObsTable, VarTable, save_store};

let reader = ParallelSparseReader::new("experiment/counts_cells.bin")?;
let cells: Vec<usize> = (0..1_000).collect();
let genes = [12, 3, 40];

let parts = AdParts {
    obs: ObsTable { index: cell_ids, columns: vec![] },
    var: VarTable { index: gene_ids, columns: vec![] }, // one per entry of `genes`
    ..Default::default()
};

save_store(
    &reader,
    "out.h5ad".as_ref(),
    AdFormat::H5ad { compression: Some(6) },
    &cells,
    Some(&genes), // None writes every gene
    false,        // raw counts, `true` for the normalised layer
    parts,
    10_000,       // cells per batch
)?;
```

`write_csr` does the same for a CSR matrix you already hold in memory.

From Python this is `bv.io.save_h5ad` and `bv.io.save_zarr`, both with
`genes=`.
