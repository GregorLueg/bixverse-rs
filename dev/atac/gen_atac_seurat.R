# Seurat 6 / Signac reference values for LSI projection, rLSI integration and
# gene activity label transfer on the chr22 fixture.
#
# Run from the repository root, after gen_atac_signac.R:
#   Rscript dev/atac/gen_atac_seurat.R [data_dir]      # default ~/data/bixverse/atac
#
# data_dir holds the 10x multiome h5 the RNA counts come from.
#
# Writes crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/seurat/. The 500
# cells split into two batches by position (first 250, last 250). Every LSI
# runs on one shared feature set (the q5 top features of all cells) so the
# per-batch loadings are aligned; Seurat itself intersects per-object feature
# sets. Anchors use exact kNN (nn.method = "rann"). IntegrateEmbeddings ignores
# nn.method and weights through annoy, so the integrated embedding is only
# comparable within a tolerance.

suppressPackageStartupMessages({
  library(rhdf5)
  library(Signac)
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
})
set.seed(42)

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[[1]] else path.expand("~/data/bixverse/atac")
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

## label transfer: batch 0 RNA (reference) onto batch 1 gene activity (query)

# RNA counts of the 500 cells for the gene activity genes, in their order
ga_genes <- read.table(file.path(fix_dir, "signac", "gene_activity", "genes.tsv"), header = TRUE)
ga_ids <- sub("[.].*$", "", ga_genes$gene_id)
h5 <- file.path(data_dir, "pbmc_unsorted_3k_filtered_feature_bc_matrix.h5")
h5_ids <- as.vector(h5read(h5, "matrix/features/id"))
h5_bc <- as.vector(h5read(h5, "matrix/barcodes"))
stopifnot(identical(h5_bc[seq_along(cells)], cells))
# genes outside the 10x reference stay all zero
message("gene activity genes not in the h5: ", sum(!ga_ids %in% h5_ids))
indptr <- as.vector(h5read(h5, "matrix/indptr"))
nnz_end <- indptr[length(cells) + 1L]
indices <- as.vector(h5read(h5, "matrix/indices", index = list(seq_len(nnz_end))))
values <- as.vector(h5read(h5, "matrix/data", index = list(seq_len(nnz_end))))
row <- match(h5_ids[indices + 1L], ga_ids)
col <- rep(seq_along(cells), diff(indptr[seq_len(length(cells) + 1L)]))
keep <- !is.na(row)
rna <- sparseMatrix(
  i = row[keep], j = col[keep], x = as.double(values[keep]),
  dims = c(length(ga_ids), length(cells)), dimnames = list(ga_genes$gene_name, cells)
)
dir.create(file.path(out_dir, "rna"), showWarnings = FALSE)
trip <- summary(rna)
trip <- data.frame(feature_idx = trip$i - 1L, cell_idx = trip$j - 1L, value = as.integer(trip$x))
trip <- trip[order(trip$feature_idx, trip$cell_idx), ]
con <- gzfile(file.path(out_dir, "rna", "matrix.tsv.gz"), "w")
write.table(trip, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)

ga_trip <- read.table(file.path(fix_dir, "signac", "gene_activity", "matrix.tsv.gz"), header = TRUE)
ga <- sparseMatrix(
  i = ga_trip$feature_idx + 1L, j = ga_trip$cell_idx + 1L, x = as.double(ga_trip$value),
  dims = dim(rna), dimnames = dimnames(rna)
)

ref_cells <- cells[batch == 0]
query_cells <- cells[batch == 1]
ref <- CreateSeuratObject(counts = rna[, ref_cells], assay = "RNA")
ref <- NormalizeData(ref, verbose = FALSE)
ref <- FindVariableFeatures(ref, nfeatures = 200, verbose = FALSE)
ref <- ScaleData(ref, verbose = FALSE)
ref <- RunPCA(ref, npcs = 10, verbose = FALSE)
ref <- FindNeighbors(ref, dims = 1:10, verbose = FALSE)
ref <- FindClusters(ref, resolution = 0.8, random.seed = 0, verbose = FALSE)
labels <- paste0("c", as.character(ref$seurat_clusters[ref_cells]))
writeLines(labels, file.path(out_dir, "transfer_labels.txt"))

# Signac vignette: gene activity log-normalised to the median RNA depth
target <- median(colSums(rna[, query_cells]))
writeLines(fmt(target), file.path(out_dir, "transfer_ga_target.txt"))
query <- CreateSeuratObject(counts = ga[, query_cells], assay = "ACTIVITY")
query <- NormalizeData(query, scale.factor = target, verbose = FALSE)

features <- VariableFeatures(ref)
writeLines(as.character(match(features, rownames(rna)) - 1L), file.path(out_dir, "transfer_features.txt"))

transfer <- FindTransferAnchors(
  reference = ref, query = query, features = features, reference.assay = "RNA",
  query.assay = "ACTIVITY", reduction = "cca", nn.method = "rann", verbose = FALSE
)
ta <- slot(transfer, "anchors")
write_tsv(
  data.frame(ref_cell = as.integer(ta[, "cell1"]) - 1L, query_cell = as.integer(ta[, "cell2"]) - 1L,
             score = as.double(ta[, "score"])),
  file.path(out_dir, "transfer_anchors.tsv")
)

n_anchor_cells <- length(unique(ta[, "cell2"]))
k_weight <- min(50L, n_anchor_cells)
writeLines(as.character(k_weight), file.path(out_dir, "transfer_k_weight.txt"))
lsi_query <- objs[[2]][["lsi"]]
pred <- TransferData(
  anchorset = transfer, refdata = setNames(labels, ref_cells), weight.reduction = lsi_query,
  dims = 2:30, k.weight = k_weight, verbose = FALSE
)
pred <- pred[query_cells, ]
write_tsv(
  data.frame(barcode = query_cells, predicted_id = pred$predicted.id,
             pred[, grep("^prediction.score", colnames(pred))], check.names = FALSE),
  file.path(out_dir, "transfer_predictions.tsv")
)

# TransferData's FindWeights runs annoy whatever nn.method says; the same
# weights through Seurat's own FindWeightsC with exact kNN, on these anchors
emb <- Embeddings(lsi_query)[query_cells, 2:30]
anchor_q <- as.integer(ta[, "cell2"])
anchor_cells <- unique(anchor_q)
nn <- Seurat:::NNHelper(data = emb[anchor_cells, ], query = emb, k = k_weight, method = "rann")
dist <- Distances(nn)
dist <- 1 - dist / dist[, ncol(dist)]
w <- Seurat:::FindWeightsC(
  cells2 = 0:(length(query_cells) - 1), distances = as.matrix(dist),
  anchor_cells2 = as.character(anchor_cells), integration_matrix_rownames = as.character(anchor_q),
  cell_index = Indices(nn), anchor_score = ta[, "score"], min_dist = 0, sd = 1,
  display_progress = FALSE
)
ids <- unique(labels)
one_hot <- sapply(ids, function(l) as.numeric(labels[ta[, "cell1"]] == l))
exact <- as.matrix(t(w) %*% one_hot)
colnames(exact) <- paste0("prediction.score.", ids)
write_tsv(data.frame(barcode = query_cells, exact, check.names = FALSE),
          file.path(out_dir, "transfer_predictions_exact_knn.tsv"))

writeLines(
  c(
    paste("R", getRversion()),
    paste("Seurat", packageVersion("Seurat")),
    paste("SeuratObject", packageVersion("SeuratObject")),
    paste("Signac", packageVersion("Signac")),
    paste("RANN", packageVersion("RANN")),
    paste("RcppAnnoy", packageVersion("RcppAnnoy")),
    paste("Matrix", packageVersion("Matrix")),
    paste("irlba", packageVersion("irlba")),
    paste("rhdf5", packageVersion("rhdf5"))
  ),
  file.path(out_dir, "versions.txt")
)
