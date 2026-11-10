# R side of the QC parity tests (tests/test_qc_r.py).
#
#   Rscript dev/sc_qc_fixtures.R write <root> <reference.json>
#     Loads seeded synthetic data twice, into <root>/base (left untouched for
#     Python to copy) and <root>/r_qc, runs the QC helpers on r_qc and writes
#     what Python has to reproduce as JSON: the final obs table and its types,
#     the CellQc internals, the bounds used, and per_cell_qc_outlier on a
#     fixed vector.
#
#   Rscript dev/sc_qc_fixtures.R read <dir> <out.json>
#     Opens a directory with load_existing() and reports its obs table.
#
# Everything the tests compare against is produced here, never typed in.

suppressPackageStartupMessages({
  library(bixverse)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
mode <- args[[1]]

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

column_values <- function(dt) {
  lapply(as.list(dt), function(x) if (is.factor(x)) as.character(x) else x)
}

obs_report <- function(sc) {
  con <- DBI::dbConnect(
    duckdb::duckdb(shared_home = FALSE),
    file.path(sc@dir_data, "sc_duckdb.db"),
    read_only = TRUE
  )
  types <- DBI::dbGetQuery(con, "DESCRIBE obs")
  DBI::dbDisconnect(con, shutdown = TRUE)
  list(
    obs_types = setNames(as.list(types$column_type), types$column_name),
    obs = column_values(get_sc_obs(sc))
  )
}

new_dir <- function(root, name) {
  d <- file.path(root, name)
  unlink(d, recursive = TRUE)
  dir.create(d, recursive = TRUE)
  d
}

thresholds_table <- function(qc) {
  rbindlist(lapply(names(qc$per_metric), function(m) {
    th <- qc$per_metric[[m]]$metrics
    rbindlist(lapply(names(th), function(g) {
      data.table(
        metric = m,
        group = g,
        median = th[[g]][["median"]],
        upper_threshold = th[[g]][["upper_threshold"]],
        lower_threshold = th[[g]][["lower_threshold"]]
      )
    }))
  }))
}

qc_report <- function(qc) {
  list(
    combined = qc$combined,
    data = column_values(get_data(qc)),
    thresholds = if (length(qc$per_metric)) column_values(thresholds_table(qc)),
    per_group_stats = if (!is.null(qc$per_group_stats)) {
      column_values(qc$per_group_stats)
    }
  )
}

if (mode == "write") {
  root <- args[[2]]
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  ref <- list()

  data <- generate_single_cell_test_data(
    syn_data_params = params_sc_synthetic_data(
      n_cells = 300L,
      n_genes = 60L,
      n_samples = 3L,
      sample_bias = "slightly_uneven"
    ),
    seed = 123L
  )
  min_qc <- params_sc_min_quality(
    min_unique_genes = 5L,
    min_lib_size = 25L,
    min_cells = 5L
  )
  load <- function(name) {
    suppressWarnings(load_r_data(
      SingleCells(dir_data = new_dir(root, name)),
      data$counts,
      data$obs,
      data$var,
      sc_qc_param = min_qc,
      .verbose = FALSE
    ))
  }
  invisible(load("base"))
  sc <- load("r_qc")

  genes <- get_gene_names(sc)
  ref$gene_sets <- list(
    set_a = genes[1:5],
    set_b = genes[10:20],
    set_c = genes[30:40]
  )

  sc <- gene_set_proportions_sc(
    sc,
    gene_set_list = ref$gene_sets[c("set_a", "set_b")],
    .verbose = FALSE
  )
  ref$top_n_vals <- c(5L, 10L, 20L)
  sc <- top_genes_perc_sc(sc, top_n_vals = ref$top_n_vals, .verbose = FALSE)

  obs <- get_sc_obs(sc, filtered = TRUE)
  metrics <- list(
    lib_size = obs$lib_size,
    nnz = obs$nnz,
    set_a = obs$set_a,
    top_10 = obs$top_10_genes_percentage
  )
  directions <- c(
    lib_size = "below",
    nnz = "twosided",
    set_a = "above",
    top_10 = "above"
  )
  ref$directions <- as.list(directions)
  ref$hard <- list(set_a = list(upper = unname(quantile(obs$set_a, 0.9))))
  ref$group_col <- "sample_id"

  qc <- run_cell_qc(
    metrics = metrics,
    cells_to_keep = obs$cell_idx - 1L,
    directions = directions,
    threshold = 2,
    groups = obs[[ref$group_col]]
  )
  ref$qc_mad <- qc_report(qc)
  # rescue the upper half of the library size MAD calls
  flagged <- obs$lib_size[qc$outlier_mat[, "lib_size"]]
  ref$rescue <- list(lib_size = list(lower = median(flagged)))
  qc <- flag_cells(qc, lapply(ref$hard, unlist))
  ref$qc_flagged <- qc_report(qc)
  qc <- rescue_cells(qc, lapply(ref$rescue, unlist))
  ref$qc_rescued <- qc_report(qc)

  ref$qc_ungrouped <- qc_report(run_cell_qc(
    metrics = metrics,
    cells_to_keep = obs$cell_idx - 1L,
    threshold = 2
  ))
  ref$qc_fixed <- qc_report(run_cell_qc_fixed(
    metrics = metrics,
    cells_to_keep = obs$cell_idx - 1L,
    hard_thresholds = lapply(ref$hard, unlist)
  ))

  ref$outlier_input <- as.numeric(obs$lib_size)
  ref$outlier <- list()
  for (dir in c("twosided", "below", "above")) {
    for (std in c(TRUE, FALSE)) {
      res <- per_cell_qc_outlier(
        ref$outlier_input,
        threshold = 2,
        direction = dir,
        r_standard = std
      )
      ref$outlier[[sprintf("%s_%s", dir, std)]] <- list(
        outlier = res$outlier,
        metrics = as.list(res$metrics)
      )
    }
  }

  sc <- add_sc_new_obs(sc, get_data(qc))
  keep <- obs$cell_id[!qc$combined]
  sc <- set_cells_to_keep(sc, keep)
  sc <- gene_set_proportions_sc(
    sc,
    gene_set_list = ref$gene_sets["set_c"],
    .verbose = FALSE
  )
  sc <- setnames_sc(sc, table = "obs", old = "set_b", new = "set_b_renamed")
  sc <- drop_cols_sc(sc, table = "obs", cols = "top_20_genes_percentage")

  ref$keep <- keep
  ref$final <- obs_report(sc)
  ref$features <- column_values(get_sc_available_features(sc))
  ref$base_dir <- file.path(root, "base")
  ref$r_dir <- sc@dir_data
  write_json(ref, args[[3]])
} else if (mode == "read") {
  sc <- suppressMessages(load_existing(SingleCells(dir_data = args[[2]])))
  write_json(obs_report(sc), args[[3]])
} else {
  stop(sprintf("unknown mode '%s'", mode))
}
