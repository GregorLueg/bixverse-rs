# Prepare the pbmc3k/pbmc4k benchmark for the scVI validation harness.
#
# Usage: Rscript prepare_pbmc.R <out_dir>
#
# Follows vignettes/single_cell_batch_corrections.qmd: default
# params_sc_min_quality(), MT/ribo-based 3 MAD cell QC, non-batch-aware
# find_hvg_sc(hvg_no = 2000L) with default params_sc_hvg() (vst), PCA with
# 30 PCs and Harmony v1 with defaults on exp_id.

suppressPackageStartupMessages({
  library(bixverse)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
checkmate::assertCharacter(args, len = 1L)
out_dir <- normalizePath(args[[1L]], mustWork = FALSE)
dir.create(file.path(out_dir, "embeddings"), showWarnings = FALSE, recursive = TRUE)
dir_sc <- file.path(out_dir, "sc")
unlink(dir_sc, recursive = TRUE, force = TRUE)
dir.create(dir_sc, showWarnings = FALSE, recursive = TRUE)

# load -------------------------------------------------------------------------

dir_data <- download_pbmc_batches()
h5ad_paths <- list.files(dir_data, pattern = "\\.h5ad$", full.names = TRUE)
names(h5ad_paths) <- c("pbmc3k", "pbmc4k")
stopifnot(identical(basename(h5ad_paths), c("pbmc3k.h5ad", "pbmc4k.h5ad")))

# the download lives in tempdir(), keep a copy so Python can read the raw h5ads
dir.create(file.path(out_dir, "h5ad"), showWarnings = FALSE)
file.copy(h5ad_paths, file.path(out_dir, "h5ad"), overwrite = TRUE)

h5_tasks <- prescan_h5ad_files(h5_paths = h5ad_paths)
sc_object <- SingleCells(dir_data = dir_sc)
sc_object <- load_multi_h5ad(
  object = sc_object,
  prescan_result = h5_tasks,
  .verbose = TRUE
)

# qc ---------------------------------------------------------------------------

var <- get_sc_var(sc_object)
h5_metadata <- read_h5ad_metadata(h5ad_paths[[1L]])
var <- merge(
  var,
  h5_metadata$var[, c("ENSEMBL_ID", "Symbol_TENx")],
  by.x = "gene_id",
  by.y = "ENSEMBL_ID"
)
setnames(var, old = "Symbol_TENx", new = "gene_symbol", skip_absent = TRUE)

gs_of_interest <- list(
  MT = var[grepl("^MT-", gene_symbol), gene_id],
  Ribo = var[grepl("^RPS|^RPL", gene_symbol), gene_id]
)
sc_object <- gene_set_proportions_sc(
  sc_object,
  gs_of_interest,
  streaming = FALSE,
  .verbose = TRUE
)

qc_df <- sc_object[[c("cell_id", "lib_size", "nnz", "MT")]]
qc <- run_cell_qc(
  metrics = list(
    log10_lib_size = log10(qc_df$lib_size),
    log10_nnz = log10(qc_df$nnz),
    MT = qc_df$MT
  ),
  cells_to_keep = get_cells_to_keep(sc_object),
  directions = c(
    log10_lib_size = "twosided",
    log10_nnz = "twosided",
    MT = "above"
  ),
  threshold = 3
)
sc_object[["outlier"]] <- qc$combined
sc_object <- set_cells_to_keep(sc_object, qc_df[!qc$combined, cell_id])

# hvg, pca, harmony ------------------------------------------------------------

sc_object <- find_hvg_sc(object = sc_object, hvg_no = 2000L, .verbose = TRUE)
sc_object <- calculate_pca_sc(object = sc_object, no_pcs = 30L)
sc_object <- harmony_sc(
  object = sc_object,
  batch_column = "exp_id",
  harmony_params = params_sc_harmony()
)

save_sc_exp_to_disk(sc_object, type = "rds")

# index tables -----------------------------------------------------------------

# counts_cells.bin holds ALL cells in original load order (kept or not), so the
# store row of a cell is cell_idx - 1, independent of QC.
obs_all <- get_sc_obs(sc_object, filtered = FALSE)
kept_0 <- get_cells_to_keep(sc_object) # 0-based store rows, by contract
kept_0 <- sort(kept_0)
obs <- obs_all[kept_0 + 1L]
stopifnot(identical(obs$cell_idx - 1L, as.integer(kept_0)))
stopifnot(identical(
  obs$cell_idx,
  get_sc_obs(sc_object, filtered = TRUE)$cell_idx
))

# h5ad row: match barcode within the batch's own obs table
h5_obs <- purrr::map(h5ad_paths, \(p) read_h5ad_metadata(p)$obs[["Barcode"]])
cells <- obs[, .(
  store_idx = as.integer(cell_idx - 1L),
  barcode,
  exp_id,
  cell_id
)]
cells[,
  h5ad_row := match(barcode, h5_obs[[exp_id[1L]]]) - 1L,
  by = exp_id
]
stopifnot(!anyNA(cells$h5ad_row))
# cell ids are <exp_id>_<obs row>; cross-check against the barcode match
stopifnot(identical(
  as.integer(sub("^.*_", "", cells$cell_id)),
  as.integer(cells$h5ad_row)
))
cells[, h5ad_file := paste0(exp_id, ".h5ad")]
cells[, cell_id := NULL]
setcolorder(cells, c("store_idx", "barcode", "exp_id", "h5ad_file", "h5ad_row"))
fwrite(cells, file.path(out_dir, "cells.tsv"), sep = "\t")

hvg_0 <- get_hvg(sc_object) # 0-based store gene indices
var_all <- get_sc_var(sc_object)
genes <- var_all[hvg_0 + 1L, .(store_idx = as.integer(gene_idx - 1L), gene_id)]
stopifnot(identical(genes$store_idx, as.integer(hvg_0)))
# the h5ad var index is the Ensembl id; check both files carry every HVG
h5_var <- purrr::map(h5ad_paths, \(p) read_h5ad_metadata(p)$var[[".id"]])
stopifnot(all(genes$gene_id %in% h5_var[[1L]]), all(genes$gene_id %in% h5_var[[2L]]))
genes[, h5ad_var := gene_id]
fwrite(genes, file.path(out_dir, "genes.tsv"), sep = "\t")

writeLines(
  bixverse:::get_rust_count_cell_f_path(sc_object),
  file.path(out_dir, "store_path.txt")
)

# embeddings (rows are in kept-cell order, same as cells.tsv) -------------------

for (nm in c("pca", "harmony")) {
  embd <- if (nm == "pca") {
    get_pca_factors(sc_object)
  } else {
    get_embedding(sc_object, nm)
  }
  stopifnot(nrow(embd) == nrow(cells))
  colnames(embd) <- paste0("pc_", seq_len(ncol(embd)))
  fwrite(
    as.data.table(embd),
    file.path(out_dir, "embeddings", paste0(nm, ".tsv")),
    sep = "\t"
  )
}

message(sprintf(
  "kept %d / %d cells, %d genes in store, %d HVGs",
  nrow(cells), nrow(obs_all), nrow(var_all), nrow(genes)
))
