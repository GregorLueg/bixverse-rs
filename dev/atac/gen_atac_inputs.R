# Builds the chr22 input fixtures for bixverse-atac from the 10x PBMC unsorted
# 3k multiome (Cell Ranger ARC 2.0.0).
#
# Run from the repository root:
#   Rscript dev/atac/gen_atac_inputs.R [data_dir]
#
# data_dir defaults to ~/data/bixverse/atac. The fragment file is only read
# through its tabix index (chr22), never decompressed in full. The chr22 length
# comes from UCSC through GenomeInfoDb (needs network).
#
# Writes crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/:
#   cells.txt, fragments.tsv.gz(.tbi), peaks.bed, genes.gtf.gz, blacklist.bed,
#   chrom_sizes.tsv, tenx/{features.tsv.gz, matrix.tsv.gz}

suppressPackageStartupMessages({
  library(Rsamtools)
  library(rhdf5)
  library(GenomicRanges)
  library(Signac)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[1] else path.expand("~/data/bixverse/atac")
out_dir <- "crates/bixverse-atac/tests/fixtures/pbmc3k_chr22"
dir.create(file.path(out_dir, "tenx"), recursive = TRUE, showWarnings = FALSE)

chrom <- "chr22"
n_cells <- 500L
prefix <- file.path(data_dir, "pbmc_unsorted_3k")
h5 <- paste0(prefix, "_filtered_feature_bc_matrix.h5")
frag_in <- paste0(prefix, "_atac_fragments.tsv.gz")

# cells: first 500 barcodes in h5 order
barcodes <- as.vector(h5read(h5, "matrix/barcodes"))
cells <- barcodes[seq_len(n_cells)]
writeLines(cells, file.path(out_dir, "cells.txt"))

# fragments: tabix query on chr22, filter to the cells, keep the 10x header
hdr <- headerTabix(frag_in)$header
chr_len <- GenomeInfoDb::seqlengths(GenomeInfoDb::Seqinfo(genome = "hg38"))[[chrom]]
lines <- scanTabix(frag_in, param = GRanges(chrom, IRanges(1, chr_len)))[[1]]
bc <- vapply(strsplit(lines, "\t", fixed = TRUE), `[[`, "", 4L)
lines <- lines[bc %in% cells]
plain <- tempfile(fileext = ".tsv")
writeLines(c(hdr, lines), plain)
frag_out <- file.path(out_dir, "fragments.tsv.gz")
unlink(c(frag_out, paste0(frag_out, ".tbi")))
bgzip(plain, dest = frag_out)
indexTabix(frag_out, format = "bed", comment = "#")
unlink(plain)
message("fragments kept: ", length(lines))

# chrom sizes
write.table(
  data.frame(chrom = chrom, length = chr_len),
  file.path(out_dir, "chrom_sizes.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE
)

# peaks: chr22 rows of the Cell Ranger BED, comment lines dropped
bed <- readLines(paste0(prefix, "_atac_peaks.bed"))
bed <- bed[startsWith(bed, paste0(chrom, "\t"))]
writeLines(bed, file.path(out_dir, "peaks.bed"))
peaks <- read.table(text = bed, sep = "\t", col.names = c("chrom", "start", "end"))

# GENCODE v32: chr22 gene and transcript rows, streamed through the shell.
# Exons are left out: Signac's collapse takes min start / max end per gene,
# which the transcript rows already give
gtf_out <- file.path(out_dir, "genes.gtf.gz")
cmd <- sprintf(
  "gzip -dc %s | awk -F'\\t' '$1 == \"%s\" && ($3 == \"gene\" || $3 == \"transcript\")' | gzip -n -9 > %s",
  shQuote(file.path(data_dir, "gencode.v32.primary_assembly.annotation.gtf.gz")),
  chrom, shQuote(gtf_out)
)
stopifnot(system(cmd) == 0L)

# blacklist: Signac's bundled hg38 unified list, chr22, written as 0-based BED
bl <- suppressWarnings(Signac::blacklist_hg38_unified)
bl <- sort(bl[seqnames(bl) == chrom])
write.table(
  data.frame(seqnames(bl), start(bl) - 1L, end(bl)),
  file.path(out_dir, "blacklist.bed"),
  sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE
)

# 10x Peaks counts for the 500 cells and the chr22 peaks, peaks.bed order
ids <- as.vector(h5read(h5, "matrix/features/id"))
ftype <- as.vector(h5read(h5, "matrix/features/feature_type"))
indptr <- as.vector(h5read(h5, "matrix/indptr"))
nnz_end <- indptr[n_cells + 1L]
indices <- as.vector(h5read(h5, "matrix/indices", index = list(seq_len(nnz_end))))
values <- as.vector(h5read(h5, "matrix/data", index = list(seq_len(nnz_end))))
cell_idx <- rep(seq_len(n_cells) - 1L, diff(indptr[seq_len(n_cells + 1L)]))
peak_ids <- sprintf("%s:%d-%d", peaks$chrom, peaks$start, peaks$end)
h5_to_peak <- match(ids, peak_ids)
stopifnot(all(ftype[!is.na(h5_to_peak)] == "Peaks"))
stopifnot(sum(!is.na(h5_to_peak)) == nrow(peaks))
feat <- h5_to_peak[indices + 1L]
keep <- !is.na(feat)
trip <- data.frame(feature_idx = feat[keep] - 1L, cell_idx = cell_idx[keep], value = values[keep])
trip <- trip[order(trip$feature_idx, trip$cell_idx), ]
con <- gzfile(file.path(out_dir, "tenx", "features.tsv.gz"), "w")
write.table(data.frame(peaks, name = peak_ids), con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)
con <- gzfile(file.path(out_dir, "tenx", "matrix.tsv.gz"), "w")
write.table(trip, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)
message("tenx nnz: ", nrow(trip))
