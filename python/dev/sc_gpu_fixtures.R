# R side of the GPU parity test (tests/test_gpu_r.py).
#
#   Rscript dev/sc_gpu_fixtures.R <root> <reference.json>
#     Loads seeded synthetic data twice, into <root>/base (left untouched for
#     Python to copy) and <root>/r_gpu, drops the first cells on r_gpu, finds
#     the HVGs and runs bixverse.gpu's sparse PCA with the block Lanczos and
#     the randomised solver. Writes what Python has to reproduce as JSON.
#
# Everything the test compares against is produced here, never typed in.

suppressPackageStartupMessages({
  library(bixverse)
  library(bixverse.gpu)
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
sc <- load("r_gpu")

ref <- list(base_dir = file.path(root, "base"), n_drop = 40L, hvg_no = 60L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)
sc <- find_hvg_sc(sc, hvg_no = ref$hvg_no, .verbose = FALSE)
ref$hvg <- get_hvg(sc)

ref$pca <- list()
for (solver in c("exact", "randomised")) {
  sc <- calculate_pca_gpu_sc(
    sc,
    no_pcs = 10L,
    pca_params = params_sc_pca(svd_solver = solver),
    seed = 42L,
    .verbose = FALSE
  )
  ref$pca[[solver]] <- list(
    factors = unname(get_pca_factors(sc)),
    loadings = unname(get_pca_loadings(sc)),
    singular_values = get_pca_singular_val(sc)
  )
}

jsonlite::write_json(
  ref,
  args[[2]],
  auto_unbox = TRUE,
  digits = NA,
  na = "null",
  null = "null",
  matrix = "rowmajor"
)
