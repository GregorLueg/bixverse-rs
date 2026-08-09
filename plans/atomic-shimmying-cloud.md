# Port PAGA (partition-based graph abstraction)

## Context

`sc_trajectory/` currently holds one trajectory method, Palantir. PAGA (Wolf et al.,
Genome Biol., 2019) answers a different question: instead of a per-cell pseudotime it
coarse-grains the kNN graph over a clustering and scores each pair of clusters by how
many edges run between them versus how many a random null would predict. That gives an
abstracted graph of a few dozen nodes plus a spanning tree of it, which is what people
actually plot as the "topology" of a dataset.

The original PAGA repo has been dead for seven years; scanpy's `tl.paga` is the living
implementation, so `~/repos/others/scanpy/src/scanpy/tools/_paga.py` is the reference.

Scope agreed: **model v1.2 only** (the scanpy default), no v1.0, no RNA-velocity
transitions, no satellite functions (`paga_degrees`, `paga_compare_paths`, layout
positions). Placement agreed: generic core in `src/graph/`, thin single-cell entry point
in `src/single_cell/sc_trajectory/`.

The good news is that PAGA is small. The connectivity step is one `O(nnz)` counting pass
over the kNN adjacency plus arithmetic on a `k × k` matrix where `k` is the cluster
count. The only genuinely missing primitive is a spanning tree.

## The algorithm (v1.2), precisely

Input is the **directed, binarised** kNN adjacency (scanpy uses `neighbors.distances`
with `.data` set to ones and `directed=True`, i.e. edge `i → j` iff `j` is in `i`'s kNN
list, self excluded) plus a partition label per cell.

1. `ns[c]` = cells in partition `c`; `n = sum(ns)`.
2. Count directed edges into a dense `k × k` `u64` matrix: `counts[g[i]][g[j]] += 1` for
   every stored `(i, j)`. The **diagonal holds the within-partition edge count**, so
   `es[c]`, scanpy's `es_inner_cluster[c] + inter_es.sum(axis=1)[c]`, is just row sum
   `c` of `counts`. No second accumulator needed.
3. Zero the diagonal (igraph's `simplify` drops the self-loops the cluster graph would
   otherwise carry), then symmetrise: `sym[a][b] = counts[a][b] + counts[b][a]`.
4. For every `a < b` with `sym[a][b] > 0`:
   `expected = (es[a] * ns[b] + es[b] * ns[a]) / (n - 1)`,
   `conn = min(sym[a][b] / expected, 1)`, or `1` when `expected == 0`.
5. Tree: maximum spanning forest of `conn`. scanpy inverts the data and runs
   `scipy.sparse.csgraph.minimum_spanning_tree`; `1/x` is monotone decreasing so a
   maximum spanning forest on the values directly is the same thing without the
   inversion.

## Files

### New: `src/graph/spanning_tree.rs`

The missing primitive. Kruskal over a CSR treated as undirected.

```rust
pub fn minimum_spanning_forest<T>(graph: &CompressedSparseData2<T>)
    -> Result<CompressedSparseData2<T>, BixverseErrors>
where T: BixverseFloat + BixverseNumeric;

pub fn maximum_spanning_forest<T>(...) -> ... // same shape
```

Both are thin wrappers over one private Kruskal core. Behaviour to document in the doc
comments:

- Disconnected input gives a **forest**, matching `scipy.sparse.csgraph`.
- A pair stored in both directions with differing weights: take the weight that favours
  the objective (smaller for minimum, larger for maximum). scipy takes the smaller for
  its MST, so this matches on the minimum side.
- Non-finite weights are skipped, same convention as
  `dijkstra_from_source` in `src/graph/shortest_paths.rs:47`.
- Ties broken by `(weight, lo, hi)` so the output is deterministic.
- Output is a **symmetric** CSR carrying the retained weights, shape unchanged.

Reuses `UnionFind`. That struct is currently private in
`src/graph/graph_components.rs:20`. **Move it to `src/graph/graph_structures.rs` as
`pub(crate)`** (that file is named for graph data structures) and have
`graph_components.rs` import it. Mechanical ~70-line move, no behaviour change.

### New: `src/graph/graph_abstraction.rs`

The PAGA connectivity statistic, generic and free of single-cell content.

```rust
pub struct PartitionConnectivity<T> {
    /// Connectivities between partitions, `n_partitions` square, symmetric CSR
    /// with a zero diagonal. Values in `(0, 1]`.
    pub connectivities: CompressedSparseData2<T>,
    /// Nodes per partition, indexed by partition id. Empty partitions are kept.
    pub sizes: Vec<usize>,
}

pub fn partition_connectivities<T, U>(
    graph: &CompressedSparseData2<U>,
    partitions: &[usize],
    n_partitions: Option<usize>,
) -> Result<PartitionConnectivity<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseNumeric,
    U: Clone;
```

Two generics because the input's stored values are **ignored entirely** (only the
sparsity pattern matters), so callers can hand in a `CompressedSparseData2<u8>` and pay
one byte per edge instead of four. `u8` satisfies `BixverseNumeric`, so `new_csr` accepts
it. State the "values ignored" contract in the doc comment.

`n_partitions = None` derives `max(label) + 1`. Passing it explicitly keeps empty
partitions (an R factor level with no cells), which is what scanpy's categorical does.

Accumulation is in `f64` throughout, then cast to `T` at the end: edge counts reach `10^8`
on a million cells and the ratio is a difference of large products. Document that.

Validation, reusing existing variants: `SparseMatrixMustBeCsr`, `ShapeMismatch` (not
square), `CommunityAssignmentMismatch` (label count vs node count). One new variant is
needed, in the `// -- graph based errors ---` section of `src/errors.rs` (around line 88,
**not** feature-gated):

```rust
/// Error if a partition label sits outside the declared partition count
#[error("Partition label {label} is out of range for {n_partitions} partitions")]
PartitionLabelOutOfRange {
    /// The offending label
    label: usize,
    /// Number of declared partitions
    n_partitions: usize,
},
```

Parallelism: rayon `fold`/`reduce` over rows with one dense `k²` `u64` accumulator per
thread, merged in `reduce`. Gate it on a const, sequential single accumulator above it:

```rust
/// Partition count above which edge counting drops to a single accumulator.
///
/// The parallel path keeps one dense `n_partitions²` `u64` accumulator per rayon
/// thread. At 512 partitions that is 2 MB each, which is the most worth paying for
/// a pass that is only `O(nnz)` to begin with. Single-cell clusterings sit two
/// orders of magnitude below this, so the sequential arm is a guard, not a path.
const PARALLEL_PARTITION_LIMIT: usize = 512;
```

`n < 2` returns empty connectivities rather than dividing by `n - 1`.

### New: `src/single_cell/sc_trajectory/paga.rs`

The kNN-facing entry point, matching how `run_palantir` takes raw kNN output.

```rust
pub struct PagaResult<T> {
    pub connectivities: CompressedSparseData2<T>,
    pub connectivities_tree: CompressedSparseData2<T>,
    pub sizes: Vec<usize>,
}

pub fn run_paga<T>(
    knn_indices: &[Vec<usize>],
    partitions: &[usize],
    n_partitions: Option<usize>,
) -> Result<PagaResult<T>, BixverseErrors>
where T: BixverseFloat + BixverseNumeric;
```

Body is short: build the binarised directed CSR (private helper, per-row sort of the
neighbour indices, `data` filled with `1u8`, self hits dropped), call
`partition_connectivities`, call `maximum_spanning_forest` on the result, assemble.

Note the CSR builder is deliberately **not** `coo_to_csr` (it would sort the whole edge
list when the rows are already grouped) and **not** `knn_to_sparse_dist`
(`src/single_cell/sc_processing/knn.rs:240`, which drops zero-distance edges and carries
`f32` values we would ignore).

**No `PagaParams` struct.** With v1.2 fixed and no velocity, PAGA has zero tuning knobs;
an empty params struct is scaffolding. Consequently nothing is needed in
`src/single_cell/sc_r_wrappers.rs` either. That file only holds `from_r_list` param
deserialisers, and there is no result-side helper for `PalantirResult` to be consistent
with.

Module doc follows the `palantir.rs:1-47` template: what the file does, then a
**deliberate divergences** list:

1. The maximum spanning forest is computed on the connectivities directly rather than the
   minimum spanning tree of their reciprocals. Same edges, one fewer pass, and it does
   not need `1/x` on values that can be arbitrarily small.
2. `connectivities_tree` is stored symmetrically. scipy emits one direction per MST edge
   and scanpy keeps that asymmetry; a symmetric tree matches `connectivities` and does
   not surprise callers that iterate stored entries.
3. Empty partitions are retained with size zero rather than being dropped.
4. Ties in the spanning forest resolve by `(weight, lo, hi)` rather than by whatever order
   scipy's Cython happens to visit.
5. Degenerate cases return typed errors instead of scipy exceptions.

### Registrations

- `src/graph/mod.rs`: add `pub mod graph_abstraction;` and `pub mod spanning_tree;`.
- `src/single_cell/sc_trajectory/mod.rs`: add `pub mod paga;` and extend the module doc,
  which currently only describes Palantir.
- `src/prelude.rs`: no change. It re-exports nothing from `sc_trajectory` today and only
  `SparseGraph`/`NodeData`/`EdgeData` from `graph`.

## Tests

Inline `#[cfg(test)] mod tests`, no fixtures (the crate has none anywhere), assertions
against hand-computed values on toy graphs.

`spanning_tree.rs`:
- Known 4-node weighted graph, minimum and maximum forests hit different edge sets.
- Disconnected input gives `n_nodes - n_components` edges.
- Asymmetric weights on one pair pick the objective-favouring side.
- Ties are deterministic across runs.
- CSC input and non-square input are rejected.

`graph_abstraction.rs`:
- Two cliques bridged by one edge: the single off-diagonal entry equals the hand-computed
  `sym / expected`, and the value is below 1.
- Three-partition chain: `conn[0][2] == 0` (absent from the CSR), `conn[0][1] > 0`.
- A partition pair with more edges than the null predicts clamps to exactly `1`.
- Single partition gives empty connectivities.
- Empty partition declared via `n_partitions` gets size 0 and no edges.
- Both counting arms agree: same graph run under and over `PARALLEL_PARTITION_LIMIT`
  produces identical output. Forces the path dispatch.
- Rejects CSC input, label/node length mismatch, out-of-range label.

`paga.rs`:
- Y-shaped kNN graph over three partitions: the tree keeps two edges and drops the weak
  third, connectivities are symmetric with a zero diagonal.
- Self hits in `knn_indices` are dropped rather than counted as within-partition edges.

Everything is toy-sized, so nothing goes behind `large_scale_diagnostics`.

## Verification

```bash
cargo fmt
cargo clippy --features single-cell,multi-modal --all-targets
cargo test --no-default-features -- graph_abstraction spanning_tree
cargo test --features single-cell,multi-modal -- paga
```

The core lives under default features, so the first `cargo test` covers it without
`single-cell`.

Numerical cross-check against the reference, since there are no checked-in fixtures. Build
a toy `AnnData` with a hand-written kNN distance graph and labels, and compare:

```python
import numpy as np, scanpy as sc, anndata as ad
from scipy.sparse import csr_matrix
# 30 cells, 3 labelled blocks, explicit k=4 neighbour lists
adata = ad.AnnData(np.zeros((30, 1)))
adata.obs["grp"] = pd.Categorical([0]*10 + [1]*10 + [2]*10)
adata.obsp["distances"] = csr_matrix(D)          # the same directed kNN we feed Rust
adata.uns["neighbors"] = {"params": {"n_neighbors": 4}, "connectivities_key": ...}
sc.tl.paga(adata, groups="grp")
print(adata.uns["paga"]["connectivities"].toarray())
```

Feed the identical neighbour lists to `run_paga` and check the `k × k` matrices agree to
`1e-6`. Requires `igraph` in the scanpy environment. Do this once by hand during
implementation; the resulting numbers then get frozen into the inline tests as
`assert_relative_eq!` targets, which is how the rest of the crate keeps reference parity
without shipping fixtures.

## Out of scope (recorded so it is a decision, not an omission)

- **v1.0 model**: needs the UMAP fuzzy simplicial set. The only implementation in the
  crate is buried inside `src/single_cell/sc_batch_correction/bbknn.rs` with
  `apply_set_operations` and `trim_graph` private. Adding v1.0 means either lifting those
  out or accepting an arbitrary undirected weighted graph.
- **RNA-velocity transitions**: no velocity infrastructure exists; the caller would have
  to supply the velocity graph. Same counting machinery, ~40 lines, addable later.
- **Layout positions**: the abstracted graph has tens of nodes, so igraph on the R side
  lays it out in microseconds. Only worth Rust if scanpy's `init_pos='paga'` UMAP seeding
  is wanted.
- **`paga_degrees` / `paga_expression_entropies` / `paga_compare_paths`**: one-liners in R,
  or plotting territory.
- **DPT**: the crate has geodesic (Palantir) pseudotime but no diffusion pseudotime.
  scanpy pairs PAGA with `tl.dpt`; Palantir covers the same need here.
- **`CLAUDE.md`**: the `graph/` and `single_cell/` module descriptions do not mention
  spanning trees, graph abstraction, or trajectory at all. Worth a line each, but that is
  your file to change.
