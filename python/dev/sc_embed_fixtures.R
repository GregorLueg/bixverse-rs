# R side of the 2D embedding parity tests (tests/test_embed_r.py).
#
#   Rscript dev/sc_embed_fixtures.R <root> <reference.json>
#     Loads seeded synthetic data twice, into <root>/base (left untouched for
#     Python to copy) and <root>/r_embed, drops the first cells on r_embed,
#     runs the HVGs, the PCA and the neighbours on it, and from those every
#     2D embedding at its defaults. Writes what Python has to reproduce as
#     JSON, together with the default parameter bundles and the versions.
#
# Everything the tests compare against is produced here, never typed in.

suppressPackageStartupMessages({
  library(bixverse)
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
sc <- load("r_embed")

ref <- list(
  base_dir = file.path(root, "base"),
  n_drop = 40L,
  versions = list(
    bixverse = as.character(utils::packageVersion("bixverse")),
    manifoldsR = as.character(utils::packageVersion("manifoldsR"))
  )
)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)

sc <- find_hvg_sc(sc, hvg_no = 100L, .verbose = FALSE)
sc <- calculate_pca_sc(sc, no_pcs = 15L, .verbose = FALSE)
sc <- find_neighbours_sc(sc, .verbose = FALSE)
ref$pca <- unname(get_pca_factors(sc))
knn <- get_knn_obj(sc)
ref$knn <- list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)

## defaults --------------------------------------------------------------------

top_level <- function(generic, names) {
  f <- formals(generic)
  lapply(stats::setNames(names, names), function(n) eval(f[[n]])[[1]])
}
ref$defaults <- list(
  nn = manifoldsR::params_nn(),
  umap = manifoldsR::params_umap(),
  densmap = manifoldsR::params_densmap(),
  tsne = manifoldsR::params_tsne(),
  densne = manifoldsR::params_densne(),
  phate = manifoldsR::params_phate(),
  fa2 = manifoldsR::params_fa2(),
  umap_sc = top_level(
    umap_sc,
    c("n_dim", "k", "min_dist", "spread", "knn_method", "seed")
  ),
  tsne_sc = top_level(
    tsne_sc,
    c("n_dim", "perplexity", "approx_type", "knn_method", "seed")
  ),
  phate_sc = top_level(phate_sc, c("n_dim", "k", "knn_method", "seed")),
  forceatlas2_sc = top_level(forceatlas2_sc, c("k", "knn_method", "seed"))
)

## embeddings ------------------------------------------------------------------

ref$embeddings <- list()
for (fn in c("umap", "densmap", "tsne", "densne", "phate", "forceatlas2")) {
  sc <- get(paste0(fn, "_sc"))(sc, .verbose = FALSE)
  slot <- if (fn == "forceatlas2") "fa2" else fn
  ref$embeddings[[slot]] <- unname(get_embedding(sc, slot))
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
