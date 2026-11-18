# scATAC port: handover (2026-11-18)

Branch `worktree-merry-tickling-snail`, on top of `main`, fast-forwards.
Read `dev/atac/plan.md` (decisions, phases) and
`crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/README.md` (fixture layout,
Signac and Seurat quirks) first. Workspace 0.11.0, Python 0.4.0.

## State

| Phase | Rust | Python | R |
|---|---|---|---|
| 0 to 4: crate, 10x peaks, fragments, QC, counting, gene activity | done | done | handover written |
| 5: modality and bindings | | `bv.atac` on `SingleCellsMultiModal` | handover written |
| 6: LSI projection, rLSI, label transfer | done | done (`project_lsi`, `run_rlsi`, `transfer_labels_seurat`) | handover written |
| 7: peak caller, AMULET, LinkPeaks, motifs, footprinting | not started | | |

R handover: `~/Desktop/handovers/handover_atac.md`.

Python parity against the Seurat fixtures (`python/tests/test_atac_seurat.py`):

| Check | Result |
|---|---|
| `project_lsi` vs `ProjectSVD(mode = "lsi")` | < 1e-6 |
| `run_rlsi` vs `IntegrateEmbeddings` | < 1e-5 |
| `transfer_labels_seurat` end to end | same label on all 250 query cells; scores < 1e-5 except 18 cells (max 5e-3), from the 2 anchor pairs that swap between tied reference cells |
| `gene_activity` vs Signac `GeneActivity` | counts identical |

Gene activity is its own `SingleCells` with `FeatureKind::GeneActivity` in
the v5 header. Python methods that assume RNA counts refuse it or warn
(`check_rna_counts` in `python/bixverse/_core/single_cells.py`); the tiers
are in the R handover.

## Next steps

1. Phase 7, in the plan's order, scoped one item at a time with the user.
2. ATAC-only experiments (no RNA store), where gene activity transfer
   becomes the main route.

## Gotchas

- The worktree shell guard refuses heredocs and `cd` into computed paths:
  put scripts in the scratchpad and run them with `python3 -I`.
- New Seurat fixtures print 9 significant digits (read as f32); the older ones
  17.
- Seurat's `FindIntegrationAnchors` scales and L2-normalises the rLSI anchor
  spaces over all components before the `dims` subset.
- `lsiproject` uses the query's own TF-IDF and no z-scoring (checked in
  Seurat's source); the plan's "reference IDF, then z-scoring" was wrong.
- `lsi_on_matrix` takes the full library sizes explicitly.
- `test_r_interop.py` norm check fails until R builds against 0.11: 0.10.1
  read v4 norms at f16, 0.11 at f32.
- `MetaCells` carry no feature kind, so the RNA-count guards do not see
  metacells built from gene activity.
