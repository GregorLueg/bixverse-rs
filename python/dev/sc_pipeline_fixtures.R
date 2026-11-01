# R side of the pipeline parity tests (tests/test_pipeline_r.py).
#
#   Rscript dev/sc_pipeline_fixtures.R <root> <reference.json>
#
# Loads seeded synthetic data twice, into <root>/base (left untouched for
# Python to copy) and <root>/r_pipe, drops the first cells on r_pipe and
# applies four pipelines: HVGs, PCA, neighbours and Leiden on the experiment;
# the same chain per cell type through apply_pipeline_per_group; meta cells
# followed by the MetaCells chain on the experiment; and per cell type
# pre-processing ending on meta cells, merged. Writes what Python has to
# reproduce as JSON.
#
# Everything the tests compare against is produced here, never typed in.

suppressPackageStartupMessages({
  library(bixverse)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
root <- args[[1]]
dir.create(root, recursive = TRUE, showWarnings = FALSE)

new_dir <- function(name) {
  d <- file.path(root, name)
  unlink(d, recursive = TRUE)
  dir.create(d, recursive = TRUE)
  d
}

csr <- function(m) list(p = m@p, j = m@j, x = m@x, dim = dim(m))

pca_report <- function(obj) {
  list(
    factors = unname(get_pca_factors(obj)),
    loadings = unname(get_pca_loadings(obj)),
    singular_values = get_pca_singular_val(obj)
  )
}

knn_report <- function(obj) {
  knn <- get_knn_obj(obj)
  list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)
}

# what a MetaCells looks like from the outside, 0-based where Python is
mc_report <- function(mc) {
  obs <- get_sc_obs(mc)
  a <- S7::prop(mc, "original_assignment")
  list(
    ids = obs$meta_cell_id,
    members = lapply(obs$original_cell_idx, function(x) x - 1L),
    n_cells = a$n_cells,
    n_metacells = a$n_metacells,
    n_unassigned = a$n_unassigned,
    raw = csr(S7::prop(mc, "data")$raw),
    norm = csr(S7::prop(mc, "data")$norm)
  )
}

data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 900L,
    n_genes = 300L,
    n_batches = 2L,
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
    SingleCells(dir_data = new_dir(name)),
    data$counts,
    data$obs,
    data$var,
    sc_qc_param = min_qc,
    .verbose = FALSE
  ))
}
invisible(load("base"))
sc <- load("r_pipe")

ref <- list(base_dir = file.path(root, "base"), n_drop = 40L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)

## experiment ------------------------------------------------------------------

p_sc <- step_hvg_sc(hvg_no = 100L, .verbose = FALSE) %>>%
  step_pca_sc(no_pcs = 15L, .verbose = FALSE) %>>%
  step_neighbours_sc(.verbose = FALSE) %>>%
  step_clusters_sc(res = 1.0, name = "pipe_leiden")
sc <- apply_pipeline(p_sc, sc)

ref$sc <- list(
  hvg = get_hvg(sc),
  pca = pca_report(sc),
  knn = knn_report(sc),
  leiden = get_sc_obs(sc, filtered = TRUE)$pipe_leiden
)

## per group -------------------------------------------------------------------

p_grp <- step_hvg_sc(hvg_no = 80L, .verbose = FALSE) %>>%
  step_pca_sc(no_pcs = 10L, .verbose = FALSE) %>>%
  step_neighbours_sc(.verbose = FALSE) %>>%
  step_clusters_sc(res = 1.0, name = "grp_leiden")
per_group <- apply_pipeline_per_group(p_grp, sc, group_col = "cell_grp")

ref$groups <- names(per_group)
ref$per_group <- lapply(per_group, function(sub) {
  list(
    cell_ids = get_cell_names(sub),
    hvg = get_hvg(sub),
    pca = pca_report(sub),
    knn = knn_report(sub),
    leiden = get_sc_obs(sub)$grp_leiden
  )
})

# a numeric group column
per_batch <- apply_pipeline_per_group(p_grp, sc, group_col = "batch_index")
ref$batches <- names(per_batch)
ref$per_batch_hvg <- lapply(per_batch, get_hvg)

## meta cells, then the MetaCells chain ----------------------------------------

p_mc <- step_metacells_sc(
  "bootstrapped",
  sc_meta_cell_params = params_sc_bt_metacells(
    target_no_metacells = 40L,
    knn = list(k = 10L)
  ),
  .verbose = FALSE
) %>>%
  step_hvg_sc(hvg_no = 50L, .verbose = FALSE) %>>%
  step_pca_sc(no_pcs = 10L, .verbose = FALSE) %>>%
  step_neighbours_sc(
    neighbours_params = params_sc_neighbours(knn = list(k = 8L)),
    .verbose = FALSE
  ) %>>%
  step_clusters_sc(res = 1.0, name = "mc_leiden")
ref$mc_end_class <- validate_pipeline(p_mc, "SingleCells")
mc <- apply_pipeline(p_mc, sc)

ref$mc <- c(
  mc_report(mc),
  list(
    hvg = get_hvg(mc) - 1L,
    pca = pca_report(mc),
    leiden = get_sc_obs(mc)$mc_leiden
  )
)

## per group meta cells, merged ------------------------------------------------

p_grp_mc <- step_hvg_sc(hvg_no = 100L, .verbose = FALSE) %>>%
  step_pca_sc(no_pcs = 10L, .verbose = FALSE) %>>%
  step_neighbours_sc(.verbose = FALSE) %>>%
  step_metacells_sc(
    "bootstrapped",
    sc_meta_cell_params = params_sc_bt_metacells(
      target_no_metacells = 10L,
      knn = list(k = 10L)
    ),
    .verbose = FALSE
  )
grp_mc <- apply_pipeline_per_group(p_grp_mc, sc, group_col = "cell_grp")
merged <- merge_meta_cells(grp_mc, .verbose = FALSE)
ref$grp_mc <- c(
  mc_report(merged),
  list(source_id = get_sc_obs(merged)$source_id)
)

jsonlite::write_json(
  ref,
  args[[2]],
  auto_unbox = TRUE,
  digits = NA,
  na = "null",
  null = "null",
  matrix = "rowmajor"
)
