# Signac 1.17.1 reference values on the chr22 fixture inputs.
#
# Run from the repository root, after gen_atac_inputs.R:
#   Rscript dev/atac/gen_atac_signac.R
#
# Single-threaded (no future plan). Writes
# crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/signac/. Every coordinate
# written is 0-based half-open; Signac works in 1-based closed GRanges, so
# starts are shifted by one on the way out.

suppressPackageStartupMessages({
  library(Signac)
  library(SeuratObject)
  library(GenomicRanges)
  library(rtracklayer)
  library(Matrix)
})
set.seed(42)

fix_dir <- "crates/bixverse-atac/tests/fixtures/pbmc3k_chr22"
out_dir <- file.path(fix_dir, "signac")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fmt <- function(x) sprintf("%.17g", x)

write_tsv <- function(df, path) {
  num <- vapply(df, is.double, logical(1))
  df[num] <- lapply(df[num], fmt)
  con <- if (endsWith(path, ".gz")) gzfile(path, "w") else file(path, "w")
  write.table(df, con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(con)
}

# `ranges` in the row order of `mat`; writes matrix.tsv.gz and, unless the
# features are tiles (implied by bin size and chromosome length), features.tsv.gz
write_sparse <- function(mat, ranges, name, dir, features = TRUE) {
  dir.create(dir, showWarnings = FALSE)
  if (features) {
    con <- gzfile(file.path(dir, "features.tsv.gz"), "w")
    write.table(
      data.frame(
        chrom = as.character(seqnames(ranges)),
        start = start(ranges) - 1L,
        end = end(ranges),
        name = name
      ),
      con,
      sep = "\t", quote = FALSE, row.names = FALSE
    )
    close(con)
  }
  trip <- summary(as(mat, "CsparseMatrix"))
  trip <- data.frame(feature_idx = trip$i - 1L, cell_idx = trip$j - 1L, value = trip$x)
  trip <- trip[order(trip$feature_idx, trip$cell_idx), ]
  is_int <- all(trip$value == round(trip$value))
  if (!is_int) trip$value <- fmt(trip$value)
  con <- gzfile(file.path(dir, "matrix.tsv.gz"), "w")
  write.table(trip, con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(con)
}

cells <- readLines(file.path(fix_dir, "cells.txt"))
frag_path <- file.path(fix_dir, "fragments.tsv.gz")
chr_len <- read.table(file.path(fix_dir, "chrom_sizes.tsv"), sep = "\t")
genome <- setNames(chr_len$V2, chr_len$V1)

frags <- CreateFragmentObject(
  path = frag_path,
  cells = setNames(cells, cells),
  validate.fragments = FALSE,
  verbose = FALSE
)

# peaks.bed is 0-based; GRanges start = BED start + 1
bed <- read.table(file.path(fix_dir, "peaks.bed"), sep = "\t")
peaks <- GRanges(bed$V1, IRanges(bed$V2 + 1L, bed$V3))
peak_names <- GRangesToString(peaks)

## counting

fm <- FeatureMatrix(frags, features = peaks, cells = cells, verbose = FALSE)
stopifnot(setequal(rownames(fm), peak_names))
fm <- fm[peak_names, cells]
write_sparse(fm, peaks, sprintf("chr22:%d-%d", bed$V2, bed$V3), file.path(out_dir, "peaks"))

tiles <- tileGenome(seqlengths = genome, tilewidth = 5000, cut.last.tile.in.chrom = TRUE)
bm <- GenomeBinMatrix(frags, genome = genome, cells = cells, binsize = 5000, verbose = FALSE)
tile_names <- GRangesToString(tiles)
stopifnot(all(start(tiles) - 1L == (seq_along(tiles) - 1L) * 5000L))
bm_full <- Matrix(0, nrow = length(tiles), ncol = length(cells), sparse = TRUE,
                  dimnames = list(tile_names, cells))
bm_full[rownames(bm), ] <- bm[, cells]
write_sparse(bm_full, tiles, tile_names, file.path(out_dir, "bins_5000"), features = FALSE)

## annotation from the GENCODE GTF

gtf <- import(file.path(fix_dir, "genes.gtf.gz"))
annot <- gtf[gtf$type == "transcript"]
mcols(annot) <- data.frame(
  tx_id = annot$transcript_id,
  gene_name = annot$gene_name,
  gene_id = annot$gene_id,
  gene_biotype = annot$gene_type,
  type = as.character(annot$type)
)
genome(annot) <- "hg38"

## Seurat object

assay <- CreateChromatinAssay(
  counts = fm,
  fragments = frags,
  annotation = annot,
  validate.fragments = FALSE,
  verbose = FALSE
)
obj <- CreateSeuratObject(counts = assay, assay = "peaks")

## QC

cf <- CountFragments(frag_path, cells = cells, verbose = FALSE)
cf <- cf[match(cells, cf$CB), ]
stopifnot(all(as.character(cf$CB) == cells))
obj$frequency_count <- cf$frequency_count

obj <- suppressWarnings(NucleosomeSignal(obj, verbose = FALSE))

tss <- GetTSSPositions(annot)
tss <- tss[!as.character(seqnames(tss)) %in% c("chrM", "Mt", "MT")]

fast <- suppressWarnings(TSSEnrichment(obj, fast = TRUE, verbose = FALSE))
full <- suppressWarnings(TSSEnrichment(obj, fast = FALSE, verbose = FALSE))

obj <- FRiP(obj, assay = "peaks", total.fragments = "frequency_count", verbose = FALSE)
bl <- suppressWarnings(Signac::blacklist_hg38_unified)
obj$blacklist_ratio <- suppressWarnings(
  FractionCountsInRegion(obj, assay = "peaks", regions = bl)
)

# TSS set and the windows Signac derives from it (TSSFast and the full pileup)
write.table(
  data.frame(
    chrom = as.character(seqnames(tss)), start = start(tss) - 1L, end = end(tss),
    gene_name = tss$gene_name, gene_id = tss$gene_id, strand = as.character(strand(tss))
  ),
  file.path(out_dir, "tss.bed"),
  sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE
)
win <- list(
  centre = Signac:::Extend(tss, upstream = 500, downstream = 500, from.midpoint = TRUE),
  up_flank = Signac:::Extend(tss, upstream = 1000, downstream = -901, from.midpoint = TRUE),
  down_flank = Signac:::Extend(tss, upstream = -901, downstream = 1000, from.midpoint = TRUE),
  full = Signac:::Extend(tss, upstream = 1000, downstream = 1000, from.midpoint = TRUE)
)
write.table(
  do.call(cbind, lapply(names(win), function(k) {
    setNames(data.frame(start(win[[k]]) - 1L, end(win[[k]])), paste0(k, c("_start", "_end")))
  })),
  file.path(out_dir, "tss_windows.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

# full-mode cut pileup summed over cells, minus strand already reversed
pile <- Signac:::CreateRegionPileupMatrix(obj, regions = win$full, assay = "peaks", verbose = FALSE)
write_tsv(
  data.frame(position = as.integer(colnames(pile)), insertions = as.double(colSums(pile))),
  file.path(out_dir, "tss_pileup.tsv")
)

## TF-IDF, top features, LSI

obj <- RunTFIDF(obj, method = 1, scale.factor = 1e4, verbose = FALSE)
tfidf <- GetAssayData(obj, assay = "peaks", layer = "data")
stopifnot(identical(rownames(tfidf), peak_names))
write_sparse(tfidf, peaks, sprintf("chr22:%d-%d", bed$V2, bed$V3), file.path(out_dir, "tfidf"))

obj <- FindTopFeatures(obj, min.cutoff = "q5", verbose = FALSE)
top <- FindTopFeatures(fm, verbose = FALSE)
top <- top[peak_names, ]
var_feat <- VariableFeatures(obj)
write_tsv(
  data.frame(
    feature_idx = seq_along(peak_names) - 1L,
    count = as.double(top$count),
    percentile = top$percentile,
    kept = as.integer(peak_names %in% var_feat)
  ),
  file.path(out_dir, "top_features.tsv")
)

obj <- RunSVD(obj, n = 10, verbose = FALSE)
lsi <- obj[["lsi"]]
emb <- Embeddings(lsi)[cells, ]
load <- Loadings(lsi)
write_tsv(data.frame(barcode = cells, emb, check.names = FALSE), file.path(out_dir, "lsi_embeddings.tsv.gz"))
write_tsv(
  data.frame(feature_idx = match(rownames(load), peak_names) - 1L, unclass(load), check.names = FALSE),
  file.path(out_dir, "lsi_loadings.tsv.gz")
)
dc <- DepthCor(obj, n = 10)$data
svd_raw <- Misc(lsi)
write_tsv(
  data.frame(
    component = seq_len(10),
    d = svd_raw$d,
    sdev = Stdev(lsi),
    depth_cor = dc$counts
  ),
  file.path(out_dir, "lsi_components.tsv")
)

## gene activity

ga <- suppressWarnings(GeneActivity(obj, verbose = FALSE))
ga_annot <- annot
ga_annot$gene_name <- ifelse(is.na(ga_annot$gene_name) | ga_annot$gene_name == "",
                             ga_annot$gene_id, ga_annot$gene_name)
tx <- Signac:::CollapseToLongestTranscript(ga_annot)
tx <- tx[tx$gene_biotype %in% "protein_coding"]
tx <- tx[width(tx) < 5e5]
tx <- Signac:::Extend(tx, upstream = 2000, downstream = 0)
stopifnot(all(rownames(ga) %in% tx$gene_name))
ga_full <- Matrix(0, nrow = length(tx), ncol = length(cells), sparse = TRUE,
                  dimnames = list(tx$gene_name, cells))
ga_full[rownames(ga), ] <- ga[, cells]
write_sparse(ga_full, tx, tx$gene_name, file.path(out_dir, "gene_activity"))
write.table(
  data.frame(gene_name = tx$gene_name, gene_id = tx$gene_id, strand = as.character(strand(tx))),
  file.path(out_dir, "gene_activity", "genes.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

## per-cell table

md <- obj[[]][cells, ]
write_tsv(
  data.frame(
    barcode = cells,
    nCount_peaks = as.double(md$nCount_peaks),
    nFeature_peaks = as.double(md$nFeature_peaks),
    frequency_count = as.double(cf$frequency_count),
    reads_count = as.double(cf$reads_count),
    nucleosome_free = as.double(cf$nucleosome_free),
    mononucleosomal = as.double(cf$mononucleosomal),
    nucleosome_signal = md$nucleosome_signal,
    nucleosome_percentile = md$nucleosome_percentile,
    tss_enrichment_fast = fast$TSS.enrichment[cells],
    tss_percentile_fast = fast$TSS.percentile[cells],
    tss_enrichment_full = full$TSS.enrichment[cells],
    tss_percentile_full = full$TSS.percentile[cells],
    frip = md$FRiP,
    blacklist_ratio = md$blacklist_ratio
  ),
  file.path(out_dir, "cell_metrics.tsv")
)

writeLines(
  c(
    paste("R", getRversion()),
    paste("Signac", packageVersion("Signac")),
    paste("SeuratObject", packageVersion("SeuratObject")),
    paste("GenomicRanges", packageVersion("GenomicRanges")),
    paste("Rsamtools", packageVersion("Rsamtools")),
    paste("rtracklayer", packageVersion("rtracklayer")),
    paste("Matrix", packageVersion("Matrix")),
    paste("RSpectra", packageVersion("RSpectra"))
  ),
  file.path(out_dir, "versions.txt")
)
