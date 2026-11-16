# pbmc3k_chr22 fixtures

10x PBMC unsorted 3k multiome (Cell Ranger ARC 2.0.0), chr22 only, first 500
cells of the filtered h5. Generated, never edited by hand. From the repository
root, in order:

```
Rscript dev/atac/gen_atac_inputs.R [data_dir]      # default ~/data/bixverse/atac
Rscript dev/atac/gen_atac_signac.R
RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1 uv run --python 3.12 \
  --with snapatac2==2.10.0 --with muon==0.2.1 --with beartype \
  python dev/atac/gen_atac_scverse.py
```

Tool versions are in each folder's `versions.txt` (R 4.6.1, Signac 1.17.1,
SeuratObject 5.4.0, RSpectra 0.16.2; SnapATAC2 2.10.0; muon 0.2.1, scipy
1.18.1).

## Conventions

- Every coordinate in every file is 0-based half-open (BED), including the
  Signac outputs (converted from 1-based closed GRanges).
- Sparse matrices: `matrix.tsv.gz`, header `feature_idx cell_idx value`,
  0-based, feature-major order, cells in `cells.txt` order, zeros omitted.
  Integer counts print as integers, other values with 17 significant digits.
- `features.tsv.gz`: `chrom start end name`. Peak matrices follow `peaks.bed`
  line order. Tile matrices have no features file: tile `i` is
  `[i * bin, min((i + 1) * bin, 50818468))` on chr22 (10164 tiles at 5 kb,
  101637 at 500 bp), as both Signac and SnapATAC2 tile.
- Dense tables: TSV with header, floats at 17 significant digits; `.tsv.gz`
  for the large ones.

## Inputs

| File | Content |
|---|---|
| `cells.txt` | 500 barcodes, h5 order |
| `fragments.tsv.gz` (`.tbi`) | chr22 fragments of those cells (130540), 10x header kept, BGZF + tabix (Rsamtools) |
| `peaks.bed` | chr22 rows of `atac_peaks.bed` (1883) |
| `genes.gtf.gz` | GENCODE v32 chr22, `gene` and `transcript` rows only |
| `blacklist.bed` | Signac `blacklist_hg38_unified`, chr22 |
| `chrom_sizes.tsv` | chr22 50818468 (UCSC hg38 via GenomeInfoDb) |
| `tenx/` | h5 `Peaks` counts for the 500 cells and the chr22 peaks |

## signac/

- `peaks/`: `FeatureMatrix` on `peaks.bed`. Equals a plain half-open overlap
  count (fragment `[s, e)` hits peak `[ps, pe)` iff `s < pe && e > ps`), one
  per fragment line, column 5 ignored (checked in the generator).
- `bins_5000/`: `GenomeBinMatrix(binsize = 5000)`.
- `cell_metrics.tsv`: `nCount_peaks`, `nFeature_peaks`; `CountFragments`
  (`frequency_count`, `reads_count`, `nucleosome_free` `< 147`,
  `mononucleosomal` `147 < len < 294`); `NucleosomeSignal`; `TSSEnrichment`
  fast and full with percentiles; `FRiP` (`nCount_peaks / frequency_count`,
  both chr22 only); blacklist ratio (`FractionCountsInRegion` on the peak
  matrix with `blacklist_hg38_unified`).
- `tss.bed`: the TSS set `GetTSSPositions` produced (443 genes).
  `tss_windows.tsv`: the centre, flank and full windows Signac derived from it.
  `tss_pileup.tsv`: full-mode cut matrix summed over cells, minus strand
  reversed, Signac's position labels (`-999..1001`).
- `top_features.tsv`: `FindTopFeatures` count, ecdf percentile, `kept` for
  `min.cutoff = "q5"` (1791 of 1883).
- `lsi_*`: `RunSVD(n = 10)` on the TF-IDF of the kept features. Embeddings are
  z-scored `u` (sd with n - 1); loadings `v` (feature_idx into `peaks.bed`);
  `lsi_components.tsv` has `d`, `sdev = d / sqrt(n_kept - 1)` and
  `DepthCor` (Pearson against raw `nCount_peaks`, not log). Signs are
  arbitrary per component.
- `gene_activity/`: `GeneActivity` defaults, raw counts; `features.tsv.gz` is
  the extended region counted, `genes.tsv` id and strand. 440 genes.

### Signac caveats

- Annotation: the GTF `transcript` rows through rtracklayer, `gene_type`
  renamed `gene_biotype`, `transcript_id` as `tx_id`. `CollapseToLongestTranscript`
  does not pick a transcript: it takes min start and max end over all rows of a
  gene id, so the TSS is the gene's 5' end. First row's strand and biotype.
- `GetTSSPositions` filters on `gene_biotype == "protein_coding"` whatever
  `biotypes` says.
- Fast TSSe windows (0-based, `t` the TSS): centre `[t - 500, t + 501)`, flanks
  `[t - 1000, t - 900)` and `[t + 901, t + 1001)`, mirrored on minus strand.
  Cut sites are not counted where they fall: for a fragment overlapping window
  `[ws, we)`, `start` and `end` (raw file values) count iff
  `ws + 3 <= c <= we + 2`. Reimplemented this way, the per-cell fast TSSe
  matches Signac to 4e-16.
- Full mode averages columns `500:1500` (labels `-500..500`) of a 2001 bp
  matrix whose label 0 is column 1000.
- 2 cells have zero peak counts (one has a single fragment, one none). Their
  TF-IDF column is empty, blacklist ratio NaN (0 / 0); the fragment-free cell
  also has NaN FRiP and nucleosome signal. NaN prints as `NaN`.

## snapatac2/

`import_fragments(sorted_by_barcode = False, whitelist = cells,
min_num_fragments = 0)`, chrom sizes chr22 only. One cell has no chr22
fragments and is not imported: `imported = 0`, zero matrix column, metrics printed as `nan`.

- Counting (snapatac2-core `feature_count`): insertions at `s` and `e - 1`.
  `fragment` counts every feature the fragment overlaps once; `insertion`
  counts each insertion; `paired-insertion` counts a feature once if both
  insertions land in it.
- Not written because they are identical to Signac (asserted): `fragment`
  peaks (= `signac/peaks`), `fragment` and `paired-insertion` 5 kb tiles
  (= `signac/bins_5000`; no fragment here spans more than two 5 kb bins).
- `peaks_insertion`, `peaks_paired-insertion`, `tiles_5000_insertion`,
  `tiles_500_{fragment,insertion}`.
- `cell_metrics.tsv`: `n_fragment` (unique fragments, equals Signac
  `frequency_count`), `frac_dup` (`1 - lines / sum(col 5)`), `frac_mito`,
  `tsse`.
- TSSe: TSS from every GTF `transcript` row (all biotypes), `start - 1` on
  plus, `end - 1` on minus, deduplicated on (chrom, pos, strand): `tss.bed.gz`,
  4493 sites. Window `[t - 2000, t + 2001)`; every insertion (`s`, `e - 1`)
  in a window adds to that position (strand-flipped). Per cell
  `tsse = mean(profile[1995..=2005]) / ((sum(profile[0..100]) + sum(profile[3901..4001])) / 200 + 0.1)`.
  `tss_profile.tsv` is the library profile (offset `-2000..2000`),
  `library_tsse.tsv` the library score and fraction of insertions in a window.
  Not comparable to Signac's TSSe.
- `frag_size_distr.tsv`: `frag_size_distr(max_recorded_size = 1000)`, count
  per `e - s`; row 0 holds fragments longer than 1000.

## muon/

On `signac/peaks` counts. `tfidf` defaults (`log_tf = True`, `log_idf = True`,
scale 1e4): `log1p(x / n_cell * 1e4) * log1p(n_cells / n_feature)`, NaN to 0.
The matrix is not stored; `idf.tsv` has the column sums and `log1p` IDF (inf
for zero-count peaks, which never multiply a stored value).
`lsi` defaults except `n_comps = 10` (default 50): scipy `svds` (ARPACK,
`np.random.seed(0)`) on all 1883 peaks, no top-feature selection, components
reversed to descending, embeddings z-scored with population sd (ddof 0),
`stdev = d / sqrt(n_cells - 1)` (Signac divides by `n_features - 1`).

## tenx/

Cell Ranger ARC `Peaks` counts. Insertions at `s` and `e` (the raw file
values), counted in a peak iff `ps <= c < pe`: this reproduces the matrix
exactly. SnapATAC2 `insertion` (which uses `e - 1`) differs in 62 entries.

Trimmed after generation to stay under 5 MB: `signac/tfidf/` (`RunTFIDF`
method 1 on `peaks/`) and `snapatac2/tiles_500_paired-insertion/` are
produced by the generators but not committed.
