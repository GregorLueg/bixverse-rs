# scATAC port: handover (2026-11-17)

Branch `worktree-linear-kindling-bee`, 25 commits on `main`, fast-forwards.
Read `dev/atac/plan.md` (decisions, phases) and
`crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/README.md` (fixture layout,
Signac and Seurat quirks) first.

## State

| Phase | Rust | Python | R |
|---|---|---|---|
| 0 to 4: crate, 10x peaks, fragments, QC, counting, gene activity | done | done, except the gene activity store | not started |
| 5: modality and bindings | | `bv.atac` on `SingleCellsMultiModal` through LSI | handover not written |
| 6: LSI projection, rLSI, label transfer | done | not bound | not started |
| 7: peak caller, AMULET, LinkPeaks, motifs, footprinting | not started | | |

Phase 6 parity against Seurat 6.0.0 (`dev/atac/gen_atac_seurat.R`, fixtures
in `pbmc3k_chr22/seurat/`, tests in `crates/bixverse-atac/tests/pbmc3k_chr22.rs`):

| Check | Result |
|---|---|
| `ProjectSVD(mode = "lsi")` | < 1e-6 |
| rLSI anchors | 42 of 42 pairs, scores 1e-6 |
| rLSI `IntegrateEmbeddings` | 2e-6 on Seurat's LSIs, < 1e-5 on ours |
| `FindTransferAnchors(cca)` | 971 of 973; the rest swap between 3 tied zero-count reference cells |
| `TransferData` | 1e-5 against an exact-kNN reference (Seurat's own uses annoy, off by 2.4e-3) |

Where the phase 6 code lives:

- `bixverse-atac::lsi`: `read_tf_idf` (store IDF or any supplied IDF),
  `project_lsi`, `lsi_on_matrix`; `lsi_on_store` is built on them.
- `bixverse-atac::rlsi`: `RlsiParams`, `rlsi_anchors`, `rlsi_integration`,
  `rlsi_on_store` (per-batch IDF and LSI on one shared feature set).
- `bixverse-sc::sc_annotation::seurat_transfer`: `find_transfer_anchors`,
  `transfer_labels`, exact CCA through two thin QRs. Modality agnostic, so it
  also serves RNA to RNA transfer.
- `bixverse-sc::sc_batch_correction::seurat_anchors::find_weights` now caps a
  cell at `k_weight` anchor pairs, as Seurat's `FindWeightsC`. This changes
  rPCA and CCA integration output in Python and R.

## Next steps

1. Python bindings for phase 6: `atac_project_lsi`, `atac_rlsi`,
   `transfer_labels` (the last belongs in `annotate.py`, not `atac.py`), with
   tests against the same Seurat fixtures.
2. Python gene activity store: regions exist (`atac_gene_activity_regions`),
   the counting into a `FeatureKind::Gene` log-CPM store at the median RNA
   depth is not bound.
3. R handover: write `~/Desktop/handovers/handover_atac.md` (does not exist
   yet; the R repo belongs to another agent). Shape is in the plan's Bindings
   section.
4. Phase 7, in the plan's order.

## Open decisions (the user's)

1. Versions and CHANGELOGs: workspace and `python/` bumps, and a line for the
   `find_weights` change (rPCA and CCA results differ at the same seed).
2. `python/tests/test_r_interop.py` norm failure on the v4 R store: not
   investigated.
3. Fixture folder is 5.6 MB against the 5 MB the README aims for: accept, or
   trim further.

## Gotchas

- The worktree shell guard refuses complex heredocs and `git -C` on the main
  checkout: put scripts in the scratchpad and run them with `python3 -I`.
- New Seurat fixtures print 9 significant digits (read as f32); the older ones
  17.
- Seurat's `FindIntegrationAnchors` scales and L2-normalises the rLSI anchor
  spaces over all components before the `dims` subset; normalising after the
  subset gives 140 anchors instead of 42.
- `lsi_on_matrix` takes the full library sizes explicitly; depth over the
  selected features is wrong.
- Not run this session: the all-features clippy line from `CLAUDE.md`.
