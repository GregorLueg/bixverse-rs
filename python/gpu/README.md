# bixverse-gpu

GPU twins of the [bixverse](https://github.com/GregorLueg/bixverse-rs) single
cell methods, on wgpu, in Rust. Import it as `bixverse_gpu`.

## Install

```bash
uv pip install "bixverse[gpu]"
```

This pulls `bixverse` and `manifolds-rs` (GPU UMAP and t-SNE run there). For
the UMAP and t-SNE twins you'll want `bixverse[gpu,embed]`.

## Layout

Each module mirrors its `bixverse` namesake: `pp`, `graph`, `embed`,
`integrate`, `de`, `doublets`, `metacells`, `nmf`, `pathways` and `trees`. The
functions take the same `SingleCells` objects, write the same cache slots and
obs columns and return the same result types, so a pipeline swaps one import
for the other. GPU-specific knobs live in `bixverse_gpu.params`; everything
else takes the `bixverse.params` bundles.

```python
import bixverse as bv
import bixverse_gpu as bvg

bv.pp.find_hvg(sc, hvg_no=2000)
bvg.pp.calculate_pca(sc, 30)
bvg.graph.find_neighbours(sc)
bvg.graph.find_clusters(sc, res=1.0, name="leiden_clusters")
```

`bvg.gpu_available()` probes for an adapter. Without one the twins raise
`bixverse.BixverseError`. A few results get their own name in the cache
(`harmony_gpu`, `mnn_gpu`).
