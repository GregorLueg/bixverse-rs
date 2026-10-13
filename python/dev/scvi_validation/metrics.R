# Batch-mixing metrics and Leiden clusters for a directory of embeddings.
#
# Usage: Rscript metrics.R <out_dir> <embeddings_dir>
#
# Every <name>.tsv in embeddings_dir has one row per kept cell in cells.tsv
# order. Writes metrics.tsv and clusters/<name>.tsv into embeddings_dir.

suppressPackageStartupMessages({
  library(bixverse)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
checkmate::assertCharacter(args, len = 2L)
out_dir <- normalizePath(args[[1L]], mustWork = TRUE)
embd_dir <- normalizePath(args[[2L]], mustWork = TRUE)

cells <- fread(file.path(out_dir, "cells.tsv"))
sc_object <- load_existing(
  SingleCells(dir_data = file.path(out_dir, "sc")),
  .verbose = FALSE
)

# the saved object carries the QC-kept cell set
stopifnot(identical(
  sort(as.integer(get_cells_to_keep(sc_object))),
  as.integer(cells$store_idx)
))
stopifnot(identical(get_sc_obs(sc_object, filtered = TRUE)$exp_id, cells$exp_id))

files <- list.files(embd_dir, pattern = "\\.tsv$", full.names = TRUE)
files <- files[basename(files) != "metrics.tsv"]
dir.create(file.path(embd_dir, "clusters"), showWarnings = FALSE)

res <- purrr::map(files, \(f) {
  nm <- sub("\\.tsv$", "", basename(f))
  embd <- as.matrix(fread(f))
  stopifnot(nrow(embd) == nrow(cells), !anyNA(embd))
  rownames(embd) <- NULL

  # "pca", "knn" and "snn" are reserved embedding names
  slot <- paste0("emb_", nm)
  obj <- set_embedding(sc_object, embd = embd, name = slot)
  obj <- find_neighbours_sc(
    obj,
    embd_to_use = slot,
    neighbours_params = params_sc_neighbours(),
    .verbose = FALSE
  )

  kbet <- calculate_kbet_sc(obj, batch_column = "exp_id", .verbose = FALSE)
  asw <- calculate_batch_asw_sc(
    obj,
    batch_column = "exp_id",
    embd_to_use = slot,
    .verbose = FALSE
  )
  lisi <- calculate_lisi_sc(
    obj,
    label_column = "exp_id",
    type = "batch",
    .verbose = FALSE
  )
  obj <- find_clusters_sc(obj, res = 1, name = "leiden")
  clusters <- obj[[]][["leiden"]]
  fwrite(
    data.table(cluster = clusters),
    file.path(embd_dir, "clusters", paste0(nm, ".tsv")),
    sep = "\t"
  )

  # rescaled, higher is better (as in the vignette), plus the raw values
  scaled <- calculate_integration_metrics_sc(
    obj,
    batch_column = "exp_id",
    embd_to_use = slot,
    .verbose = FALSE
  )
  data.table(
    embedding = nm,
    kbet_accept = scaled$kbet_accept,
    batch_asw = scaled$batch_asw,
    ilisi = scaled$ilisi,
    kbet_reject_rate = kbet$kbet_score,
    asw_mean = asw$mean_asw,
    lisi_mean = lisi$mean_lisi,
    n_clusters = data.table::uniqueN(clusters)
  )
})

metrics <- rbindlist(res)
fwrite(metrics, file.path(embd_dir, "metrics.tsv"), sep = "\t")
print(metrics)
