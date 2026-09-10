# bixverse

Python bindings for [`bixverse-rs`](https://github.com/GregorLueg/bixverse-rs).
The crate does the work: this is a thin layer that opens a bixverse binary
sparse store and streams minibatches out of it fast enough to keep a GPU
training loop fed.

The intended consumer is scVI-shaped: an `IterableDataset` that pulls cells
straight off disk, shuffled, without ever materialising the experiment in
memory. Decoding never leaves Rust, and the GIL is released while it happens.

## Install

```bash
uv pip install bixverse            # numpy only
uv pip install "bixverse[sparse]"  # adds scipy, for Batch.scipy_csr
```

## Use

```python
import bixverse as bv

reader = bv.CellReader("counts.bin")
loader = bv.CellLoader(reader, batch_size=256, layer="norm", seed=42)

for batch in loader:
    x = batch.scipy_csr()  # (256, n_genes) CSR
```

Pass an index array to iterate a subset, which is how a train/validation split
gets expressed:

```python
import numpy as np

rng = np.random.default_rng(0)
perm = rng.permutation(reader.total_cells)
train = bv.CellLoader(reader, perm[:180_000], batch_size=256)
val = bv.CellLoader(reader, perm[180_000:], batch_size=256, shuffle=False)
```

Each pass over a loader reshuffles and spawns its own feeder thread, so two
concurrent passes don't share state. The seed is mixed with the epoch counter:
same seed means the same sequence of epochs, not the same epoch twice.

## Batches

A `Batch` is a `NamedTuple` of `(raw, norm, indices, indptr, shape)`. The layer
you didn't ask for is `None`.

| Field | Sparse | Dense |
| --- | --- | --- |
| `raw` / `norm` | CSR-ordered `float32` | `(n_cells, n_genes)` `float32` |
| `indices` / `indptr` | `uint32` CSR components | `None` |
| `shape` | `(n_cells, n_genes)` | same |

`batch.values()` returns the single requested layer, `batch.scipy_csr()`
rebuilds a `scipy.sparse.csr_matrix` sharing the same buffers.

## Sparse by default

`sparse=True` is the default and you want it. A dense batch pays a zero-fill
and a scatter over `batch_size * n_genes` floats regardless of how few counts
it actually holds. At droplet sparsity that's roughly twenty times the memory
traffic for the same information, on the feeder thread, every batch. Build the
dense tensor on the GPU instead if you need one.

## Layers

`"raw"` gives the counts. `"norm"` gives `log1p(count / library_size *
target_size)`, computed at write time and stored as `float16`, so expect
agreement with a recomputation to about three decimal places rather than to
machine precision. `"both"` returns the two together over one sparsity pattern.

## Caveats

- Cell-major stores only. A gene-major file raises on construction.
- `Ctrl-C` can't interrupt a batch mid-decode: the GIL is released while the
  feeder works, and Python signal handlers only run while it's held.
- The store is memory-mapped and the reader hints random access, which is right
  for a shuffled loader and wrong for a sequential full scan.

## Development

```bash
uv venv
uv pip install "maturin>=1.15,<2" numpy scipy pytest beartype
maturin develop --release
pytest tests -q
```
