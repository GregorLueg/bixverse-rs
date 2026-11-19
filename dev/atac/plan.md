# scATAC in bixverse: summary

What the first ATAC release contains, the decisions behind it and what is
left for later. Numerics reference for the Signac parts:
`dev/atac/signac_reference.md`. Fixture layout and the Signac / Seurat quirks
they pin: `crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/README.md`.

## What is in

`bixverse-atac` (Rust) and `bv.atac` (Python), paired RNA plus ATAC on a
`SingleCellsMultiModal`. Peaks and tiles are ordinary v5 stores
(`FeatureKind::Peak` / `Tile`, TF-IDF as the norm, interval index), so every
store-level tool reads them. Blacklists enter as AMULET's restriction
regions; there is no separate blacklist filter. Every public function that
prints takes `verbose: usize` (`0` silent, `1` normal, `2` detailed).

| Module | Contents | Checked against |
|---|---|---|
| `io::fragments` | BGZF fragment reader, parallel block inflate, tabix chromosome spans | |
| `io::annotation` | GTF genes, TSS, gene activity regions | Signac `GetTSSPositions`, `GeneActivity` |
| `io::tenx_peaks` | 10x multiome `Peaks` into a peak store | Cell Ranger ARC h5 |
| `peaks::regions` | BED, genome tiles, region union | Signac, SnapATAC2 tiles |
| `peaks::calling` | MACS3 `callpeak` port, per cell group | MACS3 3.1.0, exact incl. 10k PBMC |
| `processing::qc` | fragments, nucleosome signal, TSS enrichment | Signac 1.17.1 (fast TSSe) |
| `processing::amulet` | AMULET multiplets, clean room from the paper | reference as a black box: exact on chr22, 14 of 11,898 calls differ on 10k |
| `processing::counting` | peak / tile / gene activity counts into a store | Signac, SnapATAC2 (all three modes), Cell Ranger `Peaks` |
| `processing::features` | top features (`FindTopFeatures`), selected in Rust for Python too | Signac |
| `processing::lsi` | TF-IDF, LSI, depth correlation, LSI projection | Signac, muon, Seurat `ProjectSVD` |
| `integration::rlsi` | rLSI anchors and integration, through the shared Seurat anchor search of `bixverse-sc` | Seurat `IntegrateEmbeddings` |
| `bv.atac` (Python only) | FRiP from the peak counts and the fragment totals | Signac `FRiP` |

Downstream of the LSI the regular single cell suite takes over: kNN,
clustering, UMAP, Harmony, WNN. Label transfer from RNA uses gene activity
(its own `SingleCells`, `FeatureKind::GeneActivity`) through the Seurat CCA
anchors.

## Decisions

1. One crate above `bixverse-sc`; fragment I/O could move down later if a
   consumer wants it without sc.
2. Count default: insertions, both ends (ArchR, Cell Ranger). `Fragment`
   (Signac) and `PairedInsertion` (SnapATAC2) as options. Insertions are the
   raw start and end of a fragment line, which reproduces Cell Ranger's
   `Peaks` matrix exactly.
3. QC and pileups use insertions, raw BED coordinates, no Tn5 shift, as
   Signac. TSS enrichment is Signac's fast mode, shifted window included.
4. TF-IDF method 1 only (`ln1p(x / lib * scale * idf)`); methods 2 to 4 do
   not fit the store's norm.
5. `lsiproject` uses the query's own TF-IDF and no z-scoring (checked in
   Seurat's source).
6. Gene activity is a standalone `SingleCells`, not a modality. Python methods
   that assume an mRNA count model refuse it or warn.
7. AMULET is GPL-3.0: built from the paper, the reference run only as a black
   box. The 10k residual (dense repeat clusters) is accepted.
8. Python first, R through a handover; no R parity tests in `python/`.
9. The anchor search (`FindNN` with the within-batch search at `k + 1`,
   `FindAnchorPairs`, `ScoreAnchors`) and the pairwise `IntegrateEmbeddings`
   live once in `bixverse-sc`'s `seurat_anchors`, shared by rLSI, rPCA, CCA
   and label transfer.
10. `RegionIndex` stays next to the store's `GenomeIntervals`: the fragment
    sweeps resolve the chromosome once per run and visit overlaps without
    allocating.

## Not in, for a later release

1. Motifs and chromVAR: genome FASTA, PWM source, motif scanner, GC-matched
   background peaks.
2. LinkPeaks: needs the same GC background.
3. Footprinting.
4. Coverage plots through tabix region queries.
5. ATAC-only experiments (no RNA store), where gene activity transfer becomes
   the main route.
