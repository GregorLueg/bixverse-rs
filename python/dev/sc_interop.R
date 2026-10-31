# R side of the SingleCells interop test (tests/test_r_interop.py).
#
#   Rscript dev/sc_interop.R write <dir> <reference.json>
#     Builds a small SingleCells directory through the regular R loading path,
#     sets a cell filter, adds an obs column, saves it, and writes what Python
#     has to find (ids, filter, obs column, counts of a few cells) as JSON.
#
#   Rscript dev/sc_interop.R read <dir> <out.json>
#     Opens a directory with load_existing() and writes back what R sees.
#
# Everything the test compares against is produced here, never typed in.

suppressPackageStartupMessages({
  library(bixverse)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
mode <- args[[1]]
dir_data <- args[[2]]
out_json <- args[[3]]

write_json <- function(x, path) {
  jsonlite::write_json(
    x,
    path,
    auto_unbox = TRUE,
    digits = NA,
    na = "null",
    null = "null"
  )
}

if (mode == "write") {
  dir.create(dir_data, recursive = TRUE, showWarnings = FALSE)

  data <- generate_single_cell_test_data(
    syn_data_params = params_sc_synthetic_data(n_cells = 300L, n_genes = 60L),
    seed = 123L
  )

  sc <- load_r_data(
    object = SingleCells(dir_data = dir_data),
    counts = data$counts,
    obs = data$obs,
    var = data$var,
    sc_qc_param = params_sc_min_quality(
      min_unique_genes = 5L,
      min_lib_size = 25L,
      min_cells = 5L
    ),
    .verbose = FALSE
  )

  # the filter: every cell outside cell_type_3, in descending order on
  # purpose, since the cell set is order sensitive
  obs <- get_sc_obs(sc)
  keep <- rev(obs$cell_idx[obs$cell_grp != "cell_type_3"])
  sc <- set_cells_to_keep(sc, as.integer(keep))

  # one value per kept cell, in ascending cell order
  kept_sorted <- sort(get_cells_to_keep(sc) + 1L)
  arm <- ifelse(kept_sorted %% 2L == 0L, "even", "odd")
  sc <- set_sc_new_obs_col(sc, col_name = "arm", new_data = arm)

  save_sc_exp_to_disk(sc)

  # counts of a handful of cells, dense over all genes
  probe <- as.integer(kept_sorted[c(1L, 2L, 10L, 50L)])
  mat <- as.matrix(sc[probe, , assay = "raw"])
  norm <- as.matrix(sc[probe, , assay = "norm"])

  obs <- get_sc_obs(sc)
  write_json(
    list(
      dims = as.integer(dim(sc)),
      n_cells_total = nrow(obs),
      cell_ids = get_cell_names(sc),
      gene_ids = get_gene_names(sc),
      cells_to_keep_r = as.integer(get_cells_to_keep(sc)),
      obs_cell_idx = obs$cell_idx,
      obs_to_keep = obs$to_keep,
      obs_arm = obs$arm,
      obs_cell_grp = as.character(obs$cell_grp),
      probe_cell_idx = probe,
      probe_raw = unname(split(mat, row(mat))),
      probe_norm = unname(split(norm, row(norm)))
    ),
    out_json
  )
} else if (mode == "read") {
  sc <- suppressMessages(load_existing(SingleCells(dir_data = dir_data)))
  obs <- get_sc_obs(sc)
  write_json(
    list(
      dims = as.integer(dim(sc)),
      cells_to_keep_r = as.integer(get_cells_to_keep(sc)),
      obs_cols = names(obs),
      obs_cell_idx = obs$cell_idx,
      obs_to_keep = obs$to_keep,
      obs_py_score = if ("py_score" %in% names(obs)) obs$py_score else NULL,
      obs_py_label = if ("py_label" %in% names(obs)) {
        as.character(obs$py_label)
      } else {
        NULL
      },
      py_label_is_factor = is.factor(obs$py_label),
      py_label_levels = levels(obs$py_label)
    ),
    out_json
  )
} else {
  stop(sprintf("unknown mode '%s'", mode))
}
