# R side of the processing parity tests (tests/test_pp_r.py).
#
#   Rscript dev/sc_pp_fixtures.R write <root> <reference.json>
#     Loads seeded synthetic data twice, into <root>/base (left untouched for
#     Python to copy) and <root>/r_pp, drops the first cells on r_pp, and runs
#     the HVG, PCA, residual and MAGIC helpers on it. Writes what Python has to
#     reproduce as JSON. The scTransform corrected counts land in
#     <root>/r_corrected for Python to open.
#
#   Rscript dev/sc_pp_fixtures.R read <dir> <out.json>
#     Opens a directory with load_existing() and reports its var table.
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
    null = "null",
    matrix = "rowmajor"
  )
}

column_values <- function(dt) {
  lapply(as.list(dt), function(x) if (is.factor(x)) as.character(x) else x)
}

if (mode == "read") {
  sc <- suppressMessages(load_existing(SingleCells(dir_data = args[[2]])))
  write_json(list(var = column_values(get_sc_var(sc))), args[[3]])
  quit(save = "no")
}
stopifnot(mode == "write")
root <- args[[2]]
dir.create(root, recursive = TRUE, showWarnings = FALSE)

new_dir <- function(name) {
  d <- file.path(root, name)
  unlink(d, recursive = TRUE)
  dir.create(d, recursive = TRUE)
  d
}

pca_report <- function(sc) {
  list(
    factors = unname(get_pca_factors(sc)),
    loadings = unname(get_pca_loadings(sc)),
    singular_values = get_pca_singular_val(sc)
  )
}

hvg_var <- function(sc, cols) {
  column_values(get_sc_var(sc)[, ..cols])
}

data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 600L,
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
sc <- load("r_pp")

ref <- list(base_dir = file.path(root, "base"), n_drop = 40L)
cell_names <- get_cell_names(sc)
ref$keep <- cell_names[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)

## hvg ------------------------------------------------------------------------

stat_cols <- list(
  vst = c("mean", "var", "var_exp", "var_std"),
  meanvarbin = c("mean", "dispersion", "dispersion_scaled", "bin"),
  dispersion = c("mean", "dispersion", "dispersion_scaled", "bin"),
  scran = c("scran_mean", "scran_var", "scran_fitted", "scran_residual")
)
ref$hvg_no <- 60L
ref$hvg <- list()
for (m in names(stat_cols)) {
  sc <- find_hvg_sc(
    sc,
    hvg_no = ref$hvg_no,
    hvg_params = params_sc_hvg(method = m),
    .verbose = FALSE
  )
  ref$hvg[[m]] <- list(
    var = hvg_var(sc, stat_cols[[m]]),
    hvg = get_hvg(sc)
  )
}
ref$hvg_equal_freq <- local({
  p <- params_sc_hvg(method = "meanvarbin", bin_method = "equal_freq")
  hvg_data <- get_hvg_data_sc(sc, hvg_no = 40L, hvg_params = p, .verbose = FALSE)
  column_values(hvg_data)
})

# 30 cells is below the 10 % gate, so this takes the in-memory subset path
ref$subset_cells <- ref$keep[seq(1L, length(ref$keep), by = 19L)]
ref$hvg_data_subset <- column_values(get_hvg_data_sc(
  sc,
  cell_ids = ref$subset_cells,
  hvg_no = 25L,
  .verbose = FALSE
))

ref$batch_aware <- list()
for (comb in c("union", "average", "intersection")) {
  res <- find_hvg_batch_aware_sc(
    sc,
    batch_column = "batch_index",
    hvg_no = 40L,
    gene_comb_method = comb,
    .verbose = FALSE
  )
  ref$batch_aware[[comb]] <- list(
    hvg_genes = unname(res$hvg_genes),
    hvg_gene_idx = res$hvg_gene_idx,
    hvg_data = column_values(res$hvg_data)
  )
}
ref$batch_aware_scran <- local({
  res <- find_hvg_batch_aware_sc(
    sc,
    batch_column = "sample_id",
    hvg_no = 40L,
    hvg_params = params_sc_hvg(method = "scran"),
    .verbose = FALSE
  )
  list(hvg_gene_idx = res$hvg_gene_idx, hvg_data = column_values(res$hvg_data))
})

## pca ------------------------------------------------------------------------

sc <- find_hvg_sc(sc, hvg_no = 100L, .verbose = FALSE)
ref$pca_hvg <- get_hvg(sc)
ref$no_pcs <- 10L
ref$pca <- list()
pca_runs <- list(
  dense_covariance = list(solver = "covariance", sparse = FALSE, clr = FALSE),
  dense_exact = list(solver = "exact", sparse = FALSE, clr = FALSE),
  dense_randomised = list(solver = "randomised", sparse = FALSE, clr = FALSE),
  sparse_exact = list(solver = "exact", sparse = TRUE, clr = FALSE),
  sparse_randomised = list(solver = "randomised", sparse = TRUE, clr = FALSE),
  dense_clr = list(solver = "covariance", sparse = FALSE, clr = TRUE)
)
for (run in names(pca_runs)) {
  r <- pca_runs[[run]]
  sc <- calculate_pca_sc(
    sc,
    no_pcs = ref$no_pcs,
    pca_params = params_sc_pca(svd_solver = r$solver, clr = r$clr),
    sparse_svd = r$sparse,
    seed = 7L,
    .verbose = FALSE
  )
  ref$pca[[run]] <- pca_report(sc)
}
# the default path: sparse, `auto` (Lanczos)
sc <- calculate_pca_sc(sc, no_pcs = ref$no_pcs, .verbose = FALSE)
ref$pca$default <- pca_report(sc)

## knn and magic --------------------------------------------------------------

sc <- find_neighbours_sc(
  sc,
  neighbours_params = params_sc_neighbours(knn = list(knn_method = "exhaustive")),
  .verbose = FALSE
)
knn <- get_knn_obj(sc)
ref$knn <- list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)
ref$magic_features <- get_gene_names(sc)[c(3:12, 150:155)]
ref$magic <- list()
for (layer in c("norm", "raw")) {
  sc <- run_magic_sc(
    sc,
    features = ref$magic_features,
    magic_params = params_sc_magic(layer = layer),
    .verbose = FALSE
  )
  mg <- get_magic(sc)
  ref$magic[[layer]] <- list(data = unname(mg$data), features = mg$features)
}

## residuals ------------------------------------------------------------------

residual_report <- function(sc, n_hvg) {
  sc <- find_hvg_sc(
    sc,
    hvg_no = n_hvg,
    hvg_params = params_sc_hvg(method = "residual"),
    .verbose = FALSE
  )
  fit <- get_residual_fit(sc)
  sc <- calculate_pca_sc(
    sc,
    no_pcs = ref$no_pcs,
    pca_params = params_sc_pca(
      normalise_variance = FALSE,
      svd_solver = "covariance"
    ),
    sparse_svd = FALSE,
    residuals = TRUE,
    .verbose = FALSE
  )
  list(
    genes = fit$genes,
    n_groups = fit$n_groups,
    residual_variance = get_sc_var(sc)$residual_variance,
    hvg = get_hvg(sc),
    pca = pca_report(sc)
  )
}

ref$residual_hvg_no <- 50L
sc <- fit_residuals_sc(sc, method = "sctransform", .verbose = FALSE)
ref$sct <- residual_report(sc, ref$residual_hvg_no)

ref$corrected_dir <- file.path(root, "r_corrected")
unlink(ref$corrected_dir, recursive = TRUE)
corrected <- sct_corrected_counts_sc(
  sc,
  dir_out = ref$corrected_dir,
  .verbose = FALSE
)

sc <- fit_residuals_sc(
  sc,
  method = "sctransform",
  covariate_columns = "nnz",
  .verbose = FALSE
)
ref$sct_covariate <- residual_report(sc, ref$residual_hvg_no)

sc <- fit_residuals_sc(
  sc,
  method = "analytic_pearson",
  group_column = "sample_id",
  .verbose = FALSE
)
ref$apr_grouped <- residual_report(sc, ref$residual_hvg_no)

write_json(ref, args[[3]])
