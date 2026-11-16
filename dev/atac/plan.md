# scATAC in bixverse: plan

Status: proposal, nothing built. Numerics reference for the Signac port:
`dev/atac/signac_reference.md`.

## Where we start

Already there, no work needed:

- v5 store: `FeatureKind::Peak` / `Tile`, `NormKind::TfIdf` (Signac `RunTFIDF`
  method 1, `ln1p(x / lib * scale * idf)`, IDF = `n_cells / rowSums`, counts not
  presence), `tf_idf_weights`, `set_tf_idf` (metadata only), raw layers,
  `set_cell_ids` / `check_same_cells`.
- `interval_index.rs`: `GenomeIntervals`, `overlapping`, `window`,
  `overlapping_range`, identity-order flag for stores written in genomic order,
  `attach_interval_index`.
- Embedding consumers that take any `MatRef<f32>`: Harmony (v1, v2), WNN
  (`run_wnn`, two modalities, modality-agnostic), the Seurat anchor primitives
  (`seurat_anchors.rs`), kNN, clustering, UMAP.
- `pca_on_sc_sparse` with `mean_center = false, normalise_variance = false`
  reads the norm layer, so on a TF-IDF store it is an uncentred truncated SVD,
  which is LSI up to the embedding scaling.
- Pseudobulk DE through edge-rs (`pseudobulk_dge`): differential accessibility
  for free once peaks are a store.
- R: `SingleCellsMultiModal` reserves `peak_connection` and `atac_cache`;
  `.sc_modalities()` and the cache routing already accept `"atac"`. Python
  mirrors the class, with `Modality = Literal["rna", "adt", "wnn"]`.

Missing: everything that touches fragments, counting against intervals,
annotation (GTF), top features, the LSI wrapper and its projection, gene
activity, the ATAC modality in both bindings.

## Architecture

### Crate: `bixverse-atac`

New workspace member, depends on `bixverse-sc` (and through it sc-data, methods,
core). Non-optional, so the sibling rule holds. Facade feature `atac` (implies
`single-cell`), forwarded by `bixverse-r`; the Python wheel builds against it.

Why not sc-data: fragment ingest needs no new external dependency, but QC, LSI
and gene activity need `bixverse-sc`, and splitting the ATAC code over two
crates buys nothing. Fragment I/O can move down later if a consumer wants it
without sc.

Error enum `AtacErrors` wrapping `ScErrors`. Modules:

| Module | Contents |
|---|---|
| `fragments` | BGZF reader (parallel block inflate), record parser, barcode dictionary |
| `annotation` | minimal GTF reader (gene rows: id, name, biotype, strand, coords), TSS, gene body plus promoter, blacklist BED |
| `qc` | pass 1: per-barcode metrics |
| `counting` | pass 2: sweep line into a store (peaks, tiles, gene activity) |
| `regions` | peak union / reduce, tile generation, tile aggregation |
| `features` | top features (`FindTopFeatures` q cutoff), optionally mean-var |
| `lsi` | TF-IDF set-up, LSI on the store, depth correlation, projection |

### Fragment pipeline: two passes, one sweep each

The fragment file is coordinate sorted and so are peaks, tiles and gene
regions. Every quantity Signac computes with repeated tabix queries falls out of
a single linear sweep with a handful of active-interval cursors.

**Pass 1 (all barcodes):** per barcode, fragments, reads (col 5),
nucleosome-free / mono counts (Signac bins `< 147`, `(147, 294)`), TSS centre
and flank cut sites (Signac fast definition), chrM fragments, blacklist
fragments, fragments in peaks if a peak set is given. Output is a per-barcode
table, which becomes obs. Barcodes are interned to `u32` through an
`FxHashMap<Box<[u8]>, u32>`.

**Cell selection** between the passes: either the RNA store's barcodes
(multiome: cells are defined by the paired RNA, by far the common case), a
whitelist (`singlecell.csv`, `per_barcode_metrics.csv`), or thresholds on the
pass 1 table.

**Pass 2 (kept cells only):** counts into one or more feature sets at once:
peaks, tiles, gene activity regions. Features come in genomic order, so each
feature's column is complete once the sweep passes its end: the gene-major
(CSC) file is written streaming, in identity order, and the cell-major file is
derived from it. Cell order follows the RNA store when paired, so `cell_idx` is
shared between modalities and the cell-ID hash checks it.

Open check before building: sc-data has CSR to CSC (`write_gene_file`), not the
reverse. Either add the gene file to cell file conversion, or bucket per cell as
`MtxReader::parse_mtx_quality` does. Measure both on a real fragment file.

BGZF: blocks are independent deflate streams with their size in the gzip
extra field, so the h5 chunk trick applies: index block offsets, inflate in
parallel on flate2 (already a dependency), parse in parallel, merge per
chromosome. No tabix needed for full passes. Tabix (or our own index) only
matters later for region queries (coverage plots, footprinting on a subset).

### Count semantics

What the main tools count into a feature (checked in source or docs):

| Tool | Unit | Fragment with both ends in one feature |
|---|---|---|
| Signac `FeatureMatrix` | whole fragment overlapping | 1 |
| ArchR `addPeakMatrix` / `addTileMatrix` | insertions (both ends) | 2 (peaks capped at 4; tiles binarised by default) |
| Cell Ranger ATAC `generate_peak_matrix` | insertions (both ends) | 2 |
| SnapATAC2 (`counting_strategy` default `paired-insertion`) | insertions | 1 |

Majority: insertions, both ends counted (ArchR, Cell Ranger). Default
`CountMode::Insertion`; `Fragment` (Signac parity) and `PairedInsertion`
(SnapATAC2) as options, each a one-line branch in the sweep. Side benefit: our
insertion peak matrix on Cell Ranger's peaks must equal the `Peaks` matrix in
the 10x h5, a free end-to-end check.

Insertion positions: Cell Ranger takes the raw start and raw end of the
fragment line (a peak `[ps, pe)` contains `c` iff `ps <= c < pe`), which
reproduces its `Peaks` matrix exactly on the fixtures; SnapATAC2 and ArchR use
`end - 1` (62 entries differ on 500 cells, chr22). We follow Cell Ranger.

QC and footprint pileups always use insertions (both ends, raw BED coordinates,
no Tn5 shift, as Signac).

### Peaks

Do we need peak calling? Not for v1:

1. 10x multiome h5 already carries `Peaks` with `matrix/features/interval`. The
   sparse 10x reader with `feature_type = "Peaks"` plus parsing the intervals
   gives a peak store with no fragments at all. Cheapest first deliverable.
2. A BED of peaks (ENCODE cCREs, a previous run, Cell Ranger's
   `atac_peaks.bed`), unified with `reduce` across samples.
3. Tiles (default 5 kb as `GenomeBinMatrix`, or 500 bp ArchR style), with
   `AggregateTiles` merging.

Cell Ranger peaks under-call rare populations; the Signac recommendation is
MACS per cluster after a first clustering. That needs pseudobulk insertion
tracks per group (one more sweep, cheap) and a caller. Native MACS-style caller
(extended cut pileup, local lambda over 1 kb / 10 kb windows, Poisson p, merge)
validated against MACS3 through `uv run --with macs3` is a later phase.
Shelling out to `macs2` from a library is not an option.

### Normalisation, features, LSI

- TF-IDF: method 1 exists. Recompute the IDF on kept cells (`tf_idf_weights`
  with `cells`). Methods 2 to 4 do not fit the `ln1p(x / lib * t * w)` form; skip
  until someone asks (YAGNI).
- Top features: `percentile = ecdf(rowSums)(rowSums) > q` on the gene file,
  default q5. Trivial. Signac's `FitMeanVar` only if asked.
- LSI: `pca_on_sc_sparse` uncentred on the top features, then embeddings
  z-scored per column (n - 1). Keep `idf`, loadings `V` and `d` so new cells
  project (`X_tfidf(ref idf) V diag(1/d)`, then the reference z-scoring): that is
  Seurat's `lsiproject` and the core of rLSI.
- Depth correlation: Pearson of each component with the raw peak library
  size (Signac `DepthCor`), return
  it; the caller drops component 1 when it correlates (the vignette's `2:30`).
  Do not drop silently.
- Memory: `sparse_pca` loads cells x top peaks. Whether `pca_on_sc_streaming`
  runs uncentred is unverified; check before claiming LSI is out of core.

### Batch correction and cross-modality

- Harmony on LSI `2:30`: works today, nothing to build.
- rLSI (ATAC-ATAC anchors): rPCA path in `seurat_rpca.rs` with LSI projection
  instead of PCA projection. The anchor primitives take embeddings, so this is
  a new projection plus wiring, not new anchor code.
- Paired multiome: WNN on RNA PCA plus ATAC LSI, exists.
- Unpaired RNA to ATAC label transfer: gene activity as the shared feature
  space, CCA anchors, weights from ATAC LSI (Seurat `TransferData`). Check how
  much of the CCA path exists in `sc_batch_correction` before scoping.
- scVI family (PeakVI, MultiVI) and GLUE: need a VAE trainer and its validation
  burden. Out of scope here; revisit if burn becomes part of the stack anyway.
  Linear alternatives that fit the house style: MOFA+ (group factor analysis,
  handles unpaired views through groups), SnapATAC2's spectral embedding
  (cosine kernel, Nyström) as an LSI alternative. Neither is needed for v1.

### Gene activity

Pass 2 feature set: protein-coding genes, `CollapseToLongestTranscript`, width
under 500 kb, extended 2 kb upstream, strand aware, fragment overlap counting.
Written as its own gene store (FeatureKind::Gene, log-CPM with target = median
RNA library size as the vignette), so every RNA tool reads it. Needs the GTF
reader. Note Signac's `GetTSSPositions` ignores its `biotypes` argument; do not
copy that.

### Bindings

Store layout: `<dir_data>/atac/{counts_cells.bin, counts_genes.bin}` (peaks or
tiles), optional `<dir_data>/gene_activity/`. Per-barcode QC goes into obs as
`atac_*` columns, features into a `var_atac` table (chrom, start, end, plus
`peak_called_in` when merged), both in the existing DuckDB.

R: `peak_connection` becomes the ATAC store handle, `"atac"` added to every
`modality = c(...)` match, new generics `load_atac_fragments_sc` /
`add_atac_peaks_sc`, `calculate_lsi_atac_sc`, `gene_activity_sc`, params specs
in `inst/params`. Python: same shape in `multimodal.py`, `Modality` gains
`"atac"`. The R repo is owned by another agent: hand over through
`~/Desktop/handovers/handover_atac.md`, do not edit it from here.

ATAC-only datasets (no RNA) are a second step: same store as the primary of a
`SingleCells`, with the RNA-shaped generics refusing a peak store.

## Phases

| # | Deliverable | Reuses | New |
|---|---|---|---|
| 0 | Crate skeleton, errors, prelude, facade feature | | crate |
| 1 | 10x multiome `Peaks` into a peak store with interval index, TF-IDF, top features, LSI, depth correlation | 10x sparse reader, v5 kinds, `pca_on_sc_sparse` | interval parse, `lsi`, `features` |
| 2 | Fragment reader, pass 1 QC (fragments, NS, TSSe, chrM, blacklist, FRiP), cell selection | | `fragments`, `annotation`, `qc` |
| 3 | Pass 2 counting: peaks / tiles from BED or tiling, multi-sample union | interval index, writers | `counting`, `regions` |
| 4 | Gene activity store | pass 2, GTF reader | |
| 5 | Python modality and bindings; R handover | WNN, Harmony, kNN, clustering | wrappers |
| 6 | LSI projection, rLSI integration, gene-activity label transfer | anchor primitives | projection |
| 7 | Later: native peak caller per cluster, AMULET doublets, LinkPeaks (needs genome FASTA for GC), motifs / chromVAR, footprinting, tabix region queries | | |

The single cell suite takes over after phase 1 (paired) or phase 3 (fragments):
kNN, clustering, UMAP, WNN and Harmony all consume the LSI embedding.

## Validation

No R parity: the release chain is Rust, then Python, then R. Signac and the
scverse tools are references for the numerics, not for a wrapper.

- Every numeric from a generated fixture, never recalled. Signac 1.17.1 is
  installed (R 4.6.1). It delegates `ATACqc()` to the external `fragtk` binary,
  so fixtures use the deprecated in-R `NucleosomeSignal` / `TSSEnrichment`,
  and the definition we implement is pinned to that code.
- scverse references through uv: SnapATAC2 (fragment import, tile and peak
  matrices in all three counting modes, TSSe, also the speed baseline since it
  is Rust underneath) and muon (TF-IDF, LSI). Where a definition differs from
  Signac (TSSe does), compare against the tool whose definition we implement
  and report the other side by side, not as a test.
- MACS3 through uv for the caller in phase 7.
- Data: 10x PBMC multiome (Cell Ranger ARC 2.0.0). `pbmc_unsorted_3k`
  (fragments 537 MB, h5 32 MB) for fixtures, cut to one chromosome under
  `python/tests/fixtures/` by a generator in `python/dev/`;
  `pbmc_granulocyte_sorted_10k` (fragments 2.4 GB) for benchmarks and the
  vignette, the Signac multiome vignette's dataset.

## Decisions (2026-11-16)

1. One `bixverse-atac` crate above `bixverse-sc`.
2. Count default: insertions, both ends (majority); fragment and paired
   insertion as options.
3. Peak source for v1: 10x peaks, BED, tiles; native caller in phase 7.
4. Python first, R handover after. No R parity tests (removed from `python/`).
5. References: Signac installed for fixtures, SnapATAC2 and muon through uv.
6. Paired RNA plus ATAC for v1, on 10x PBMC multiome; ATAC-only later.
