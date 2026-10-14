# Quickstart

## Open a store

```python
import bixverse as bv

reader = bv.CellReader("counts.bin")
reader.total_cells
reader.total_genes
```

## Iterate batches

```python
loader = bv.CellLoader(reader, batch_size=256, layer="norm", seed=42)

for batch in loader:
    x = batch.scipy_csr()
```

Each pass over a loader reshuffles and spawns its own feeder thread, so two
concurrent passes don't share state. The seed is mixed with the epoch counter:
same seed means the same sequence of epochs, not the same epoch twice.

## Train and validation splits

Hand the loader an index array to iterate a subset:

```python
import numpy as np

rng = np.random.default_rng(0)
perm = rng.permutation(reader.total_cells)
train = bv.CellLoader(reader, perm[:180_000], batch_size=256)
val = bv.CellLoader(reader, perm[180_000:], batch_size=256, shuffle=False)
```

## Subset or reorder genes

`genes=` takes store gene indices, in the column order you want. Repeats raise.

```python
loader = bv.CellLoader(reader, batch_size=256, genes=hvg_indices)
```

## Batches

A `Batch` is a `NamedTuple` of `(raw, norm, indices, indptr, shape, cell_ids)`.
The layer you didn't ask for is `None`. `cell_ids` is the store index of each
row, which is how per-cell covariates get joined back on.

| Field | Sparse | Dense |
| --- | --- | --- |
| `raw` / `norm` | CSR-ordered `float32` | `(n_cells, n_genes)` `float32` |
| `indices` / `indptr` | `uint32` CSR components | `None` |
| `shape` | `(n_cells, n_genes)` | same |
| `cell_ids` | store index per row | same |

`batch.values()` returns the single requested layer. `batch.scipy_csr()`
rebuilds a `scipy.sparse.csr_matrix` sharing the same buffers.

## Layers

`"raw"` gives the counts. `"norm"` gives `log1p(count / library_size *
target_size)`, computed at write time and stored as `float16`, so expect
agreement with a recomputation to about three decimal places. `"both"` returns
the two together over one sparsity pattern.

## Sparse by default

`sparse=True` is the default and you want it. A dense batch pays a zero-fill
and a scatter over `batch_size * n_genes` floats however few counts it holds.
At droplet sparsity that's roughly twenty times the memory traffic, on the
feeder thread, every batch. Build the dense tensor on the GPU if you need one.
Pass `sparse=False` only when you have to.

## Caveats

- `Ctrl-C` can't interrupt a batch mid-decode: the GIL is released while the
  feeder works, and Python signal handlers only run while it's held.
- The store is memory-mapped and the reader hints random access. Right for a
  shuffled loader, wrong for a sequential full scan.
