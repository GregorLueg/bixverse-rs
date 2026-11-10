"""Bonsai trees over cells or meta cells.

Bonsai builds a tree with every cell a leaf and inferred ancestors inside, from
the raw counts after Sanity, and lays it out in 2D. Use it where a hierarchy
over all cells says more than a process with a start (that is `bv.traj`). Call
`bonsai` first, change the drawing with `relayout_bonsai`, and put the leaf
coordinates in `sc.cache.embeddings` with `set_bonsai_embedding` to plot them
like a UMAP.

**What lands where.** `bonsai` and `relayout_bonsai` return a `BonsaiTree` and
leave the experiment alone. `set_bonsai_embedding` is the only writer: it
stores the `(n_leaves, 2)` leaf positions in `sc.cache.embeddings[name]`
(default `"bonsai"`) and stamps it. `bonsai` takes a `SingleCells` or a
`MetaCells`, not a `SingleCellsSubset`.

```python
tree = bv.trees.bonsai(sc)
tree = bv.trees.relayout_bonsai(tree, layout="dendrogram")
bv.trees.set_bonsai_embedding(sc, tree)
```

On meta cells every meta cell is a leaf, with `n_cells` counting the cells
behind it. That is the way to large data. The generators want the kNN from
`bv.graph.find_neighbours` on `sc` first:

```python
mc = bv.metacells.generate_supercells(sc)
tree_mc = bv.trees.bonsai(mc)
```

**Gotchas.** No HVGs, PCA or kNN are needed; the signal-to-noise filter picks
the genes, or pass `hvg=` to restrict the candidates. `tree.nodes` is 0-based with a
null parent for the root, leaves first in the object's cell order. `relayout_bonsai`
renumbers the inferred ancestors, the leaves keep their numbers.
`set_bonsai_embedding` refuses a tree built over other cells than the object
now keeps, so rerun `bonsai` after changing the cell filter.
"""

from bixverse._core.trees import (
    BonsaiTree,
    bonsai,
    relayout_bonsai,
    set_bonsai_embedding,
)

__all__ = ["BonsaiTree", "bonsai", "relayout_bonsai", "set_bonsai_embedding"]
