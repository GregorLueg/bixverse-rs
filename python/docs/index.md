# bixverse

Streams a bixverse binary cell-major store into minibatches for neural network
training. The [Rust crate](https://github.com/GregorLueg/bixverse-rs) does the
work: a feeder thread decompresses and materialises batches ahead of your
training step, and the GIL is released while it does. No materialising the
experiment in memory, no Python-side decode.

## Install

```bash
uv pip install bixverse            # numpy only
uv pip install "bixverse[sparse]"  # adds scipy, for Batch.scipy_csr
uv pip install "bixverse[torch]"   # torch, for bixverse.nn
uv pip install "bixverse[mlx]"     # MLX, Apple Silicon only
uv pip install "bixverse[jax]"     # JAX, for bixverse.nn on CPU or CUDA
```

## Thirty seconds

```python
import bixverse as bv

reader = bv.CellReader("counts.bin")
loader = bv.CellLoader(reader, batch_size=256, layer="norm", seed=42)

for batch in loader:
    x = batch.scipy_csr()  # (256, n_genes) CSR
```

Cell-major stores only. A gene-major file raises on construction.

Next: the [quickstart](quickstart.md), then the [API reference](api/toplevel.md).
