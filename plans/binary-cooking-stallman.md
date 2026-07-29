# Per-cell ScType annotation with neighbourhood smoothing

## Context

`run_sctype` already produces a full per-cell score matrix (`SctypeRes.scores`, row-major
`n_cells x n_cell_types`, `sc_type.rs:258`). `assign_clusters` sums it per cluster, takes the
argmax and stamps one label on every cell in the cluster. When a Leiden cluster on the sNN graph
is impure, that whole minority population gets the majority's label and there is nothing in the
output that hints at it.

Per-cell scoring is not the expensive part. It is already computed. What is missing is a way to
(a) denoise the per-cell scores so a per-cell argmax is trustworthy, (b) see which clusters are
mixed, and (c) only fall back to per-cell calls where the cluster call is actually unsafe.

The plan adds three things to `sc_type.rs`, keeps `run_sctype` and `assign_clusters` untouched,
and reuses the existing graph and label-propagation primitives rather than adding new ones.

## Design decisions (agreed)

- **Hybrid output.** Pure clusters keep the cluster-level call. Mixed clusters fall through to
  smoothed per-cell calls. Purity threshold is a parameter.
- **Caller supplies the graph.** The smoothing entry point takes a prebuilt
  `KnnLabPropGraph<f32>`, so the smoothing runs on the exact sNN graph Leiden ran on. No
  embedding or `KnnParams` reaches `sc_type.rs`.
- **Calibration is opt-in, default off.** Existing behaviour is unchanged unless asked for.
- **Purity threshold 0.9.** A cluster has to be near-clean to keep its cluster-level call.
  Anything below falls through to per-cell.

## Files

- `src/single_cell/sc_annotation/sc_type.rs` — all new code
- `src/errors.rs` — one new variant in the `// -- sctype --` block (line 515-526), ungated to
  match the existing variant there
- `src/single_cell/sc_r_wrappers.rs` — serialisers, optional final step (see below)

## Reused, do not reimplement

- `KnnLabPropGraph<f32>` (`src/graph/graph_label_propagations.rs:51`) — CSR with public
  `offsets` / `neighbours` / `weights`. Builders `from_edge_list` (:80) and
  `from_weighted_edge_list` (:209) already row-normalise the weights.
- `SymmetryWeightStrategy` (:17) for symmetrising the sNN edges.
- `generate_snn_full` (`src/single_cell/sc_processing/snn.rs:93`) returns
  `(edges, weights)` in exactly the flat alternating-pair layout `from_weighted_edge_list`
  expects. Caller-side recipe, documented in the doc comment, not implemented here.
- `assign_clusters` (`sc_type.rs:304`) — called internally by the hybrid path for pure clusters.
- `Verbosity` / `parse_verbosity_level` (`src/prelude.rs:48`, `:80`).

`KnnLabPropGraph::label_spreading` (:293) is the right *algorithm* (`y = α·y₀ + (1−α)·W·y`) but
the wrong *layout*: it takes `&[Vec<T>]`, which for 5M cells x 20 types costs ~120 MB of `Vec`
headers on top of the data, doubled by its scratch buffer. Write the smoothing kernel against
the flat row-major score matrix instead, iterating the graph's public CSR fields directly.

## New surface in `sc_type.rs`

### Consts (top of file, with doc comments explaining the reasoning)

```rust
/// Default self-retention in the smoothing step. 0.5 keeps half the cell's own
/// signal per iteration; lower values smooth harder and risk erasing small
/// populations sitting inside a large cluster.
const SMOOTH_ALPHA: f32 = 0.5;

/// Default smoothing iterations. Two hops is enough to average out dropout on a
/// k=15 graph without bleeding across cell type boundaries.
const SMOOTH_ITERATIONS: usize = 2;

/// Convergence tolerance for the smoothing iteration.
const SMOOTH_TOLERANCE: f32 = 1e-4;

/// Minimum score for a per-cell call. Mirrors the cluster-level `n_cells / 4`
/// heuristic from Ianevski et al., expressed per cell.
const CELL_SCORE_FLOOR: f32 = 0.25;

/// Cluster purity above which the cluster-level call is kept as-is. Set high on
/// purpose: a cluster has to be near-clean to earn a blanket label, anything
/// else is resolved per cell.
const PURITY_THRESHOLD: f32 = 0.9;
```

### Types

```rust
/// How the raw ScType scores are rescaled before a per-cell argmax.
pub enum ScoreCalibration {
    /// Raw scores as produced by `run_sctype`.
    None,
    /// Standardise each cell type's score column across cells. Removes the bias
    /// towards cell types whose marker sets give a larger score magnitude.
    ColumnZ,
}

/// Parameters for the per-cell assignment path.
#[derive(Clone, Copy, Debug)]
pub struct ScTypeCellParams { alpha, iterations, tolerance, calibration, score_floor, purity_threshold }
// + documented `new()` and `impl Default`
```

Results are struct-of-arrays, not `Vec<Struct>` with a `String` per cell. At 5M cells a per-cell
`String` is not acceptable.

```rust
/// Per-cell ScType assignment.
pub struct ScTypeCellRes {
    /// Winning cell type index per cell, `None` == Unknown (below `score_floor`)
    pub assignments: Vec<Option<usize>>,
    /// Score of the winning cell type per cell (post calibration and smoothing)
    pub scores: Vec<f32>,
    /// Gap between the best and second-best score per cell. Low margin == ambiguous call
    pub margins: Vec<f32>,
    /// Fraction of graph neighbours sharing this cell's call. `None` when no graph was given
    pub agreement: Option<Vec<f32>>,
    /// Cell type names, indexed by `assignments`
    pub cell_types: Vec<String>,
}

/// Cell type composition of a single cluster, derived from the per-cell calls.
pub struct ScTypeClusterComposition {
    pub cluster: usize,
    pub n_cells: usize,
    /// Per-cell-type counts, indexed as `ScTypeCellRes::cell_types`
    pub counts: Vec<usize>,
    pub n_unknown: usize,
    /// Most frequent cell type, `None` if the cluster is all Unknown
    pub dominant: Option<usize>,
    /// Fraction of cells carrying the dominant call
    pub purity: f32,
    /// Shannon entropy of the composition (nats). 0 == pure
    pub entropy: f32,
    /// Runner-up cell type and its fraction. This is what surfaces a mixture
    pub second: Option<usize>,
    pub second_fraction: f32,
}

/// Final hybrid annotation.
pub struct ScTypeHybridRes {
    /// Final per-cell cell type index, `None` == Unknown
    pub assignments: Vec<Option<usize>>,
    /// Per-cluster composition, in cluster order
    pub composition: Vec<ScTypeClusterComposition>,
    /// `true` where the cluster fell below `purity_threshold` and was resolved per cell
    pub cluster_mixed: Vec<bool>,
    pub cell_types: Vec<String>,
}
```

### Functions

1. `fn calibrate_scores(scores: &[f32], n_cells: usize, n_ct: usize, calibration: ScoreCalibration) -> Vec<f32>`
   (private). `ColumnZ` accumulates mean and variance in `f64` before downcasting, matching
   `scale_csc_chunk` (`sc_processing/pca.rs:183`); a column with near-zero sd is left at zero.
   `None` returns a clone.

2. `pub fn smooth_scores(scores: &[f32], n_ct: usize, graph: &KnnLabPropGraph<f32>, alpha: f32, iterations: usize, tolerance: f32) -> Result<Vec<f32>, BixverseErrors>`

   Flat sparse matvec on the row-major score matrix, ping-pong buffers, `par_chunks_mut(n_ct)`
   over rows, per-iteration max-change reduction for early exit. Weights are already
   row-normalised by the builder, so no renormalisation here.

   **Invariant to implement explicitly:** a node with no neighbours (isolated after sNN pruning)
   must keep its original score, not be shrunk to `alpha * y0`. Guard on
   `offsets[i] == offsets[i + 1]`.

   Errors with the new variant if `graph.offsets.len() - 1 != scores.len() / n_ct`.

3. `pub fn assign_cells(res: &SctypeRes, graph: Option<&KnnLabPropGraph<f32>>, params: Option<ScTypeCellParams>) -> Result<ScTypeCellRes, BixverseErrors>`

   Order is calibrate, then smooth, then argmax. Calibrating first matters: smoothing must act
   on comparable scales. Argmax and the top-two margin are computed in a single fused scan over
   each row (`par_chunks(n_ct)`), not two passes. `agreement` is a second pass over the CSR,
   only when a graph was supplied. A cell whose best score is below `score_floor` becomes `None`.

4. `pub fn cluster_composition(cell_res: &ScTypeCellRes, cluster_labels: &[usize]) -> Result<Vec<ScTypeClusterComposition>, BixverseErrors>`

   Single sequential pass over `n_cells` scattering into a `n_clusters * n_ct` histogram, then a
   parallel pass over clusters for the derived stats. The scatter is one increment per cell with
   no arithmetic; going parallel would need per-thread histograms for no measurable gain. Note
   this in a comment so it does not read as an oversight. Reuses the existing length check and
   `ScTypeClusterAssignmentNotEqualNCells`.

5. `pub fn assign_hybrid(res: &SctypeRes, cell_res: &ScTypeCellRes, cluster_labels: &[usize], purity_threshold: f32) -> Result<ScTypeHybridRes, BixverseErrors>`

   Calls `assign_clusters` and `cluster_composition`, then per cluster: if
   `purity >= purity_threshold` stamp the cluster-level label on every member cell, otherwise
   copy the per-cell calls through. Records which clusters took which path in `cluster_mixed`.

### Error variant

Insert after `errors.rs:525`, before the `// -- wnn --` comment, ungated to match the sibling
variant:

```rust
/// Error when the smoothing graph node count != the number of cells
#[error("SCType: The graph has {n_nodes} nodes but there are {n_cells} cells.")]
ScTypeGraphNodesNotEqualNCells { n_cells: usize, n_nodes: usize },
```

## Downstream wiring (`~/repos/shared/bixverse`)

How it hangs together today:

- `calc_sc_type_scores()` (`R/methods_sc_annotations_reference.R:39`) calls `rs_sc_type`
  (`src/rust/src/single_cell/r_sc_annotation.rs:63`) and stamps class `ScTypeResults` on the
  returned list.
- `score_clusters.ScTypeResults()` (`R/classes_single_cell_others.R:2562`) calls
  `rs_sc_type_cluster_assignment` (`r_sc_annotation.rs:116`), which parses the list back into
  `SctypeRes` via `SctypeRes::from_r_list` and returns a `data.table`.
- The sNN graph lives on the object as an **igraph**, built in `methods_sc_processing.R:1437-1458`
  from `rs_sc_snn()` (which hands back `$edges` as a flat 1-indexed pair vector and `$weights`),
  stored via `set_snn_graph`.
- Leiden runs in R: `igraph::cluster_leiden(graph = snn_graph, resolution = res)`
  (`methods_sc_processing.R:1512`), writing membership into the obs table.

So the graph the smoothing needs is already sitting on the object. It comes out as an edge list
with `igraph::as_edgelist()` plus `igraph::E(g)$weight`, which maps straight onto
`KnnLabPropGraph::from_weighted_node_pairs(from, to, n_nodes, symmetrise)`
(`graph_label_propagations.rs:148`). `rs_knn_label_propagation(from, to, one_hot_encoding,
label_mask, weights, label_prop_params)` is the existing precedent for that handoff, so the new
entry point should follow the same shape:

```
rs_sc_type_hybrid(sc_type_res, cluster_labels, from, to, weights, sc_type_cell_params)
```

Follow-up work in the R package, once the crate side lands:

1. `r_sc_annotation.rs` — new `#[extendr] fn rs_sc_type_hybrid`, added to the `extendr_module!`
   block (`:17`). Rebuilds `SctypeRes::from_r_list`, builds the `KnnLabPropGraph`, calls
   `assign_cells` then `assign_hybrid`, serialises out. **Watch the index base:** igraph edge
   lists are 1-based, the Rust side is 0-based; `rs_knn_mat_to_edge_list` handles this with an
   explicit `one_index` flag, do the same rather than assuming.
2. `params_sc_type_cells()` constructor in `R/param_constructors.R` for the alpha / iterations /
   calibration / floors / purity knobs, matching the existing `params_*` pattern.
3. `annotate_cells()` S7 method next to `score_clusters()` in `classes_single_cell_others.R`,
   pulling the graph via `get_snn_graph(object, modality)` and the Leiden labels from obs, so the
   user does not assemble the edge list by hand.
4. `devtools::document()` for the `.Rd` files and `R/extendr-wrappers.R`.

Note: `cluster_leiden` membership is 1-based and `assign_clusters` assumes 0-based
(`sc_type.rs:316-325`). Today that silently produces an empty cluster 0 which is skipped at
`:341`, so results are correct but a slot is wasted. Keep the same tolerance in the new
functions rather than changing the contract.

## Out of scope, flagged

- **R serialisation helpers in this crate.** `sc_r_wrappers.rs` only holds `from_r_list` parsers
  for ScType (`:2875`, `:2914`); serialisation currently happens inline in the R package's
  `r_sc_annotation.rs`. Keep that split. If a `*_to_r_list` helper is wanted here,
  `assignments_to_r_list` (`sc_r_wrappers.rs:65`) already takes `&[Option<usize>]` and is the
  pattern to follow.
- **Prelude.** `sc_annotation` is not re-exported today. Leaving it that way for consistency.
- **Subclustering mixed clusters.** The composition output tells you which clusters to re-run
  Leiden on at higher resolution. Doing that inside the crate would need the embedding here and
  `fast_louvain_clusters` operates on k-means centroids, not cells. Left to the caller.

## Tests

Inline `#[cfg(test)] mod tests` at the bottom of `sc_type.rs`, `approx::assert_relative_eq!` for
floats:

- `test_sctype_smoothing_two_cliques` — two disconnected 5-node cliques, one cell in clique A
  flipped to B's score profile; after smoothing its argmax returns to A
- `test_sctype_smoothing_isolated_node` — a node with no edges keeps its original score
- `test_sctype_column_z_calibration` — each column comes out zero-mean, unit-sd
- `test_sctype_composition_pure_cluster` — entropy 0, purity 1, `second` is `None`
- `test_sctype_composition_mixed_cluster` — 50/50 split gives entropy `ln(2)`, purity 0.5, and
  `second_fraction` 0.5
- `test_sctype_hybrid_keeps_pure_cluster` — pure cluster gets the cluster-level label everywhere
- `test_sctype_hybrid_splits_mixed_cluster` — 70/30 cluster keeps both populations' calls
- `test_sctype_graph_size_mismatch_errors` — the new error variant fires

## Verification

```bash
cargo fmt
cargo clippy --features single-cell,multi-modal --all-targets
cargo test --features single-cell,multi-modal -- sctype
cargo test --no-default-features            # sc_type is feature-gated, confirm this still builds
cargo test --features single-cell,multi-modal   # full pass, nothing regressed
```

End-to-end sanity on real data (manual, outside the test suite, once the R wrapper exists): take a
Leiden run that shows the problem, pull the sNN graph off the object with `get_snn_graph()`, hand
its edge list and weights through, then run `assign_cells` and `cluster_composition`. The clusters
you already suspect should come back with purity well below 0.9 and a `second_fraction` matching
the minority population you can see in the marker expression. Cross-check a handful of those cells
against their marker genes directly before trusting the new labels.
