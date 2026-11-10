# R side of the SingleCellsSubset parity tests (tests/test_subset_r.py).
#
#   Rscript dev/sc_subset_fixtures.R <root> <reference.json>
#
# Loads seeded synthetic data twice, into <root>/base (left untouched for
# Python to copy) and <root>/r_subset, drops the first cells on r_subset, cuts
# a subset on one cell type and runs the subset chain on it: HVGs, PCA,
# neighbours, Leiden, fast clustering, markers on the sub-clusters and, when
# manifoldsR is installed, PHATE. Then records what the parent looks like
# afterwards. Writes what Python has to reproduce as JSON.
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

column_values <- function(dt) {
  lapply(as.list(dt), function(x) if (is.factor(x)) as.character(x) else x)
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
sc <- load("r_subset")

ref <- list(
  base_dir = file.path(root, "base"),
  n_drop = 40L,
  grouping_column = "cell_grp",
  group = "cell_type_1",
  has_phate = requireNamespace("manifoldsR", quietly = TRUE)
)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)
parent_obs_cols <- names(get_sc_obs(sc))
parent_var_cols <- names(get_sc_var(sc))
parent_keep <- get_cells_to_keep(sc)

sub <- SingleCellsSubset(
  sc_object = sc,
  grouping_column = ref$grouping_column,
  group = ref$group
)
ref$subset_to_original <- S7::prop(sub, "subset_to_original") - 1L
ref$cells_to_keep <- get_cells_to_keep(sub)
ref$cell_ids <- get_cell_names(sub)

## subset chain ----------------------------------------------------------------

sub <- find_hvg_sc(sub, hvg_no = 100L, .verbose = FALSE)
ref$hvg <- get_hvg(sub)
sub <- calculate_pca_sc(sub, no_pcs = 15L, .verbose = FALSE)
ref$pca <- list(
  factors = unname(get_pca_factors(sub)),
  loadings = unname(get_pca_loadings(sub)),
  singular_values = get_pca_singular_val(sub)
)

sub <- find_neighbours_sc(sub, .verbose = FALSE)
knn <- get_knn_obj(sub)
ref$knn <- list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)
conn <- get_connectivities(sub)
ref$graph <- c(
  bixverse:::rs_graph_to_csr(bixverse:::.connectivity_ptr(conn)),
  list(n = conn$n, n_edges = conn$n_edges)
)

sub <- find_clusters_sc(sub, res = 1.0, name = "sub_leiden", seed = 42L)
ref$leiden <- get_sc_obs(sub)[["sub_leiden"]]

ref$fast <- column_values(get_data(fast_cluster_sc(
  sub,
  resolutions = c(1.0, 0.5),
  fc_params = params_sc_fast_cluster(n_centroids = 20L),
  .verbose = FALSE
)))

ref$markers <- column_values(find_all_markers_sc(
  sub,
  column_of_interest = "sub_leiden",
  .verbose = FALSE
))

if (ref$has_phate) {
  sub <- phate_sc(sub, .verbose = FALSE)
  ref$phate <- unname(get_embedding(sub, "phate"))
}

## parent afterwards -----------------------------------------------------------

ref$parent <- list(
  hvg = length(suppressWarnings(get_hvg(sc))),
  pca = !is.null(get_pca_factors(sc)),
  knn = !is.null(get_knn_obj(sc)),
  obs_cols_unchanged = identical(names(get_sc_obs(sc)), parent_obs_cols),
  var_cols_unchanged = identical(names(get_sc_var(sc)), parent_var_cols),
  keep_unchanged = identical(get_cells_to_keep(sc), parent_keep)
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
