# Seurat 6 / Signac reference values for LSI projection and rLSI integration
# on the chr22 fixture.
#
# Run from the repository root, after gen_atac_signac.R:
#   Rscript dev/atac/gen_atac_seurat.R
#
# Writes crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/seurat/. The 500
# cells split into two batches by position (first 250, last 250). Every LSI
# runs on one shared feature set (the q5 top features of all cells) so the
# per-batch loadings are aligned; Seurat itself intersects per-object feature
# sets. Anchors use exact kNN (nn.method = "rann"). IntegrateEmbeddings ignores
# nn.method and weights through annoy, so the integrated embedding is only
# comparable within a tolerance.

suppressPackageStartupMessages({
  library(Signac)
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
})
set.seed(42)

fix_dir <- "crates/bixverse-atac/tests/fixtures/pbmc3k_chr22"
out_dir <- file.path(fix_dir, "seurat")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 9 significant digits round-trip the f32 the tests read
fmt <- function(x) sprintf("%.9g", x)

write_tsv <- function(df, path) {
  num <- vapply(df, is.double, logical(1))
  df[num] <- lapply(df[num], fmt)
  con <- if (endsWith(path, ".gz")) gzfile(path, "w") else file(path, "w")
  write.table(df, con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(con)
}

write_lsi <- function(red, cells, peak_names, prefix, loadings = TRUE) {
  emb <- Embeddings(red)[cells, ]
  load <- Loadings(red)
  write_tsv(data.frame(barcode = cells, emb, check.names = FALSE),
            file.path(out_dir, paste0(prefix, "_embeddings.tsv.gz")))
  if (!loadings) return(invisible())
  write_tsv(
    data.frame(feature_idx = match(rownames(load), peak_names) - 1L, unclass(load),
               check.names = FALSE),
    file.path(out_dir, paste0(prefix, "_loadings.tsv.gz"))
  )
  write_tsv(data.frame(component = seq_along(Misc(red)$d), d = Misc(red)$d),
            file.path(out_dir, paste0(prefix, "_d.tsv")))
}

## inputs: the Signac peak matrix of gen_atac_signac.R

cells <- readLines(file.path(fix_dir, "cells.txt"))
trip <- read.table(file.path(fix_dir, "signac", "peaks", "matrix.tsv.gz"), header = TRUE)
feat <- read.table(file.path(fix_dir, "signac", "peaks", "features.tsv.gz"), header = TRUE)
peak_names <- sprintf("%s-%d-%d", feat$chrom, feat$start + 1L, feat$end)
fm <- sparseMatrix(
  i = trip$feature_idx + 1L, j = trip$cell_idx + 1L, x = trip$value,
  dims = c(nrow(feat), length(cells)), dimnames = list(peak_names, cells)
)

n_comp <- 30
dims_anchor <- 2:30

make_obj <- function(m) {
  obj <- CreateSeuratObject(counts = CreateAssayObject(counts = m), assay = "peaks")
  RunTFIDF(obj, method = 1, scale.factor = 1e4, verbose = FALSE)
}

top <- FindTopFeatures(fm, verbose = FALSE)
features <- rownames(top)[top$percentile > 0.05]
features <- peak_names[peak_names %in% features]
writeLines(as.character(match(features, peak_names) - 1L), file.path(out_dir, "features.txt"))

batch <- rep(c(0L, 1L), each = length(cells) / 2)
writeLines(as.character(batch), file.path(out_dir, "batch.txt"))

objs <- lapply(0:1, function(b) {
  obj <- make_obj(fm[, cells[batch == b]])
  RunSVD(obj, features = features, n = n_comp, verbose = FALSE)
})
for (b in 0:1) {
  write_lsi(objs[[b + 1]][["lsi"]], cells[batch == b], peak_names, paste0("batch", b, "_lsi"))
}

## LSI projection: batch 1 onto batch 0's LSI, the raw ProjectSVD of
## FindTransferAnchors(reduction = "lsiproject")

proj <- Seurat:::ProjectSVD(
  reduction = objs[[1]][["lsi"]],
  data = GetAssayData(objs[[2]], assay = "peaks", layer = "data"),
  mode = "lsi", do.center = FALSE, do.scale = FALSE, use.original.stats = FALSE,
  verbose = FALSE
)
write_tsv(data.frame(barcode = rownames(proj), proj, check.names = FALSE),
          file.path(out_dir, "project_batch1_on_batch0.tsv.gz"))

## rLSI anchors

anchors <- FindIntegrationAnchors(
  object.list = objs, anchor.features = peak_names, reduction = "rlsi",
  dims = dims_anchor, nn.method = "rann", verbose = FALSE
)
a <- slot(anchors, "anchors")
a <- a[a[, "dataset1"] == 1 & a[, "dataset2"] == 2, ]
write_tsv(
  data.frame(cell_a = as.integer(a[, "cell1"]) - 1L, cell_b = as.integer(a[, "cell2"]) - 1L,
             score = as.double(a[, "score"])),
  file.path(out_dir, "rlsi_anchors.tsv")
)

## IntegrateEmbeddings on the LSI of all cells

combined <- make_obj(fm)
combined <- RunSVD(combined, features = features, n = n_comp, verbose = FALSE)
write_lsi(combined[["lsi"]], cells, peak_names, "combined_lsi", loadings = FALSE)

integrated <- IntegrateEmbeddings(
  anchorset = anchors, reductions = combined[["lsi"]],
  new.reduction.name = "integrated_lsi", dims.to.integrate = 1:n_comp, k.weight = 20,
  verbose = FALSE
)
int_emb <- Embeddings(integrated[["integrated_lsi"]])[cells, ]
write_tsv(data.frame(barcode = cells, int_emb, check.names = FALSE),
          file.path(out_dir, "rlsi_integrated.tsv.gz"))

writeLines(
  c(
    paste("R", getRversion()),
    paste("Seurat", packageVersion("Seurat")),
    paste("SeuratObject", packageVersion("SeuratObject")),
    paste("Signac", packageVersion("Signac")),
    paste("RANN", packageVersion("RANN")),
    paste("RcppAnnoy", packageVersion("RcppAnnoy")),
    paste("Matrix", packageVersion("Matrix"))
  ),
  file.path(out_dir, "versions.txt")
)
