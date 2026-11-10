# R side of the NMF parity tests (tests/test_nmf_r.py).
#
#   Rscript dev/sc_nmf_fixtures.R <root> <reference.json>
#
# Loads seeded synthetic data twice, into <root>/base (left untouched for
# Python to copy) and <root>/r_nmf, drops the first cells on r_nmf and picks
# HVGs there, which Python takes over. Then: one NMF run (NNDSVD and random
# initialisation, raw and normalised counts, sd scaling), stabilised restarts,
# consensus NMF and the k sweep on the cells, and the same on bootstrapped
# meta cells over the HVGs R picks for them. Writes what Python has to
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

## reports ---------------------------------------------------------------------

single_report <- function(res) {
  list(
    w = unname(res$w),
    h = unname(res$h),
    genes = res$gene_ids,
    cells = res$cell_ids,
    final_loss = res$final_loss,
    n_iter = res$n_iter,
    converged = res$converged
  )
}

multi_report <- function(res) {
  list(
    w_all = unname(res$w_all),
    w_all_columns = colnames(res$w_all),
    h_per_run = lapply(res$h_per_run, unname),
    losses = res$losses,
    converged = res$converged,
    best_idx = res$best_idx - 1L
  )
}

consensus_report <- function(res) {
  cl <- res$clusters
  list(
    w = unname(res$w),
    h = unname(res$h),
    stability = res$stability,
    rel_error = res$rel_error,
    rel_run_errors = res$rel_run_errors,
    component_id = cl$component_id,
    cluster = cl$cluster - 1L,
    local_density = cl$local_density,
    silhouette = cl$silhouette,
    kept = cl$kept,
    cluster_sizes = res$cluster_sizes$n,
    n_dropped = res$n_dropped,
    n_empty_clusters = res$n_empty_clusters
  )
}

sweep_report <- function(res) as.list(as.data.frame(res))

## data ------------------------------------------------------------------------

data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(n_cells = 600L, n_genes = 120L),
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
sc <- load("r_nmf")

ref <- list(base_dir = file.path(root, "base"), n_drop = 20L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)
sc <- find_hvg_sc(sc, hvg_no = 40L, .verbose = FALSE)
ref$hvg <- get_hvg(sc)

## cells -----------------------------------------------------------------------

ref$single <- single_report(nmf_sc(sc, k = 3L, .verbose = FALSE))
ref$single_random_raw <- single_report(nmf_sc(
  sc,
  k = 3L,
  use_second_layer = FALSE,
  nmf_hals_params = params_nmf_hals(nmf_init = "random", max_iter = 100L),
  seed = 7L,
  .verbose = FALSE
))
ref$cell_sel <- ref$keep[seq(1L, 400L, by = 2L)]
ref$gene_sel <- get_gene_names(sc)[1:30]
ref$single_sel <- single_report(nmf_sc(
  sc,
  k = 3L,
  cell_ids = ref$cell_sel,
  gene_ids = ref$gene_sel,
  preprocessing = "sd",
  .verbose = FALSE
))
ref$multi <- multi_report(stabilised_nmf_sc(
  sc,
  k = 3L,
  n_runs = 4L,
  nmf_hals_params = params_nmf_hals(max_iter = 100L),
  .verbose = FALSE
))
ref$consensus <- consensus_report(consensus_nmf_sc(
  sc,
  k = 3L,
  n_runs = 6L,
  nmf_hals_params = params_nmf_hals(max_iter = 100L),
  .verbose = FALSE
))
ref$consensus_w <- consensus_report(consensus_nmf_sc(
  sc,
  k = 3L,
  n_runs = 6L,
  preprocessing = "sqrt_sd",
  nmf_hals_params = params_nmf_hals(max_iter = 100L),
  nmf_consensus_params = params_nmf_consensus(
    consensus_target = "w",
    density_threshold = 2
  ),
  seed = 3L,
  .verbose = FALSE
))
ref$sweep <- sweep_report(suppressWarnings(nmf_k_sweep_sc(
  sc,
  k_range = 2:4,
  n_runs = 4L,
  nmf_hals_params = params_nmf_hals(max_iter = 100L),
  .verbose = FALSE
)))

## meta cells ------------------------------------------------------------------

sc <- calculate_pca_sc(sc, no_pcs = 10L, .verbose = FALSE)
sc <- find_neighbours_sc(
  sc,
  neighbours_params = params_sc_neighbours(
    knn = list(k = 15L, knn_method = "exhaustive")
  ),
  .verbose = FALSE
)
ref$pca <- unname(get_embedding(sc, "pca"))
knn <- get_knn_obj(sc)
ref$knn <- list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)

mc <- generate_bt_meta_cells_sc(
  sc,
  params_sc_bt_metacells(target_no_metacells = 60L, knn = list(k = 10L)),
  .verbose = FALSE
)
ref$mc_ids <- get_sc_obs(mc)$meta_cell_id
mc <- find_hvg_sc(mc, hvg_no = 40L, .verbose = FALSE)
ref$mc_hvg <- get_hvg(mc)

ref$mc_single <- single_report(nmf_sc(mc, k = 3L, .verbose = FALSE))
ref$mc_single_raw <- single_report(nmf_sc(
  mc,
  k = 3L,
  use_second_layer = FALSE,
  preprocessing = "sd",
  .verbose = FALSE
))
ref$mc_multi <- multi_report(stabilised_nmf_sc(
  mc,
  k = 3L,
  n_runs = 4L,
  nmf_hals_params = params_nmf_hals(max_iter = 100L),
  .verbose = FALSE
))
ref$mc_consensus <- consensus_report(consensus_nmf_sc(
  mc,
  k = 3L,
  n_runs = 6L,
  nmf_hals_params = params_nmf_hals(max_iter = 100L),
  .verbose = FALSE
))
ref$mc_sweep <- sweep_report(suppressWarnings(nmf_k_sweep_sc(
  mc,
  k_range = 2:4,
  n_runs = 4L,
  nmf_hals_params = params_nmf_hals(max_iter = 100L),
  .verbose = FALSE
)))

jsonlite::write_json(
  ref,
  args[[2]],
  auto_unbox = TRUE,
  digits = NA,
  na = "null",
  null = "null",
  matrix = "rowmajor"
)
