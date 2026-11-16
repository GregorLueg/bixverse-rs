# Signac 1.17.1 numerics, as read from source

Source: `~/repos/others/signac` (MIT). Read from code, not run: every number
here still needs a generated fixture before it becomes a test.

Version notes: `RunSVD` uses `RSpectra::svds` (not irlba); chromVAR interface
removed in 1.17.0; `NucleosomeSignal` / `TSSEnrichment` deprecated in favour of
`ATACqc()`, which shells out to the Rust `fragtk qc` (R/preprocessing.R:9).

## Fragments and counting

- Fragment file: 5 or 6 tab columns (chr, start, end, barcode, reads[, strand]),
  `#` lines skipped. `cells` maps object name to file barcode
  (R/fragments.R:307).
- `FeatureMatrix` (R/quantification.R:221, R/utilities.R:768, 2302): per
  feature, tabix overlap of the whole fragment, `sparseMatrix(x = 1)` so
  duplicates sum. Column 5 ignored. Multiple files: padded and summed.
  Features on seqnames absent from the index are dropped with a warning.
- `GenomeBinMatrix` (R/quantification.R:158): `tileGenome(binsize = 5000,
  cut.last.tile.in.chrom = TRUE)` then `FeatureMatrix`.
- `AggregateTiles` (:86): keep bins with `rowSums > 5`, merge adjacent with
  `reduce(with.revmap)`, sum rows.
- `CountFragments` (src/group.cpp:5): per barcode `frequency_count` (+1 per
  line), `reads_count` (sum col 5), `nucleosome_free` if `len < 147`,
  `mononucleosomal` if `147 < len < 294` (147 itself in neither).
- Cut sites (R/utilities.R:1666): `c(start, end) - (start(region) + 1)`, kept if
  `0 < pos <= width`. Both ends, raw coordinates, no Tn5 shift. Minus-strand
  regions reversed and added (R/utilities.R:1867).

## QC

- Nucleosome signal (R/preprocessing.R:1114): `mono / nucleosome_free` on the
  first `5000 * n_cells` lines. Zero denominator unhandled (Inf / NaN).
- TSS enrichment (R/region-enrichment.R:52), TSS from the longest transcript's
  5' end, protein coding only (the `biotypes` argument is ignored, a bug), chrM
  dropped.
  - Fast (default): centre `+-500` (1001 bp), flanks `[-1000, -901]` and
    `[901, 1000]`. Per cell summed over TSSs: `flank_mean = flank / 200`, zero
    flank replaced by the mean of `flank_mean`,
    `tsse = (centre / flank_mean) / 1001`, NA to 0.
  - Full: cell x 2001 pileup, `flank_mean = rowMeans(cols 1:100, 1902:2001)`,
    zero replaced by the mean, `tsse = rowMeans(norm[, 500:1500])`.
  - Percentiles: `round(ecdf(x)(x), 2)`.
- FRiP (R/preprocessing.R:1061): `colSums(peaks) / total_fragments`.
- Blacklist ratio: `FractionCountsInRegion` on the peak matrix
  (R/utilities.R:865), counts in peaks overlapping blacklist / all counts.
- `mito.R` is mgatk lineage tracing, not chrM QC.
- Vignette thresholds: PBMC `9000 < nCount < 1e5`, FRiP pct `> 40`, blacklist
  `< 0.01`, NS `< 4`, TSSe `> 4`; multiome `1800 < nCount < 1e5`, NS `< 2`,
  TSSe `> 1`.

## Peaks

- `CallPeaks` (R/peaks.R:271): `macs2 callpeak -t ... -g 2.7e9 -f BED --nomodel
  --extsize 200 --shift -100`; per group via `SplitFragments`, then
  `reduce(with.revmap)`, `peak_called_in` lists groups.
- `UnifyPeaks` (R/utilities.R:1134): `reduce` or `disjoin` of the union.

## Normalisation and DR

- `RunTFIDF` (R/preprocessing.R:1312), `scale.factor = 1e4`:
  `tf = X / colSums` (method 4: `tf = X`), `idf = ncol / rowSums` (counts).
  1: `log1p(tf * idf * sf)`; 2: `tf * log(1 + idf)`;
  3: `log1p(tf * sf) * log(1 + idf)`; 4: `idf * X`. NA set to 0.
- `FindTopFeatures` (:426), `min.cutoff = "q5"`: keep `ecdf(rowSums) > 0.05`;
  numeric cutoff keeps `rowSums > cutoff`.
- `FitMeanVar` (:701): loess of `log1p(var)` on `log1p(mean)`, binned sampling,
  rank blend. `PearsonResidualVar` (:834): theta 10, clip `+-sqrt(N)`,
  population variance.
- `RunSVD` (R/dimension_reduction.R:69), n 50, tol 1e-5:
  `svds(t(X), k = min(n, ncol - 1))`, loadings `v`,
  `sdev = d / sqrt(max(1, n_features - 1))`, LSI embeddings `u` z-scored per
  column (n - 1), optional clip at `scale.max`. No component dropped in code.
- `DepthCor` (R/visualization.R:203): `cor(emb[, 1:10], nCount)`; the vignette
  drops LSI 1.

## Gene activity (R/utilities.R:410)

Empty gene name to gene id; `CollapseToLongestTranscript` (min start, max end
per gene id, `*` strand to `+`, `make.unique` names); protein coding,
`width < 5e5`; extend 2000 upstream, 0 downstream, strand aware, trimmed;
`FeatureMatrix`. Vignette normalises with `LogNormalize`, scale factor
`median(nCount_RNA)`.

## Peak to gene links (R/links.R:239)

Distance 5e5 around the TSS, peaks and genes in more than 10 cells, Pearson
`corSparse`, keep `|r| > 0.05`, 200 background peaks per peak from other
chromosomes matched on GC, count and length (`MatchRegionStats`, Gaussian KDE
ratio, bw 1, R/utilities.R:1025), `z = (r - mean_bg) / sd_bg`,
`p = pnorm(-|z|)`, keep `p < 0.05`.

## Motifs, footprinting, DA

- `FindMotifs` (R/motifs.R:203): hypergeometric against 40000 GC-matched
  background peaks, BH.
- Footprinting (R/footprinting.R): hexamer insertion bias on chr1, expected =
  hexamer counts per position times bias, flank-normalised (first 50 and last
  51 positions). Possible index misalignment in `FindExpectedInsertions` (:377).
- DA: Signac owns only `FoldChange` (log2 of `rowMeans(data) + 1e-4`); testing
  is Seurat `FindMarkers`, vignettes use LR with `nCount_peaks` as latent.

## Integration (Seurat owned)

- Label transfer: gene activity, `FindTransferAnchors(reduction = "cca")`,
  `TransferData(weight.reduction = lsi, dims = 2:30)`.
- ATAC-ATAC: `FindIntegrationAnchors(reduction = "rlsi", dims = 2:30)`,
  `IntegrateEmbeddings` on LSI.
- Reference mapping: `reduction = "lsiproject"`, needs reference IDF, loadings
  and `d`.
