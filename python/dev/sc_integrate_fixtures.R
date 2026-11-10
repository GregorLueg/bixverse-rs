# R side of the batch correction and integration metric parity tests
# (tests/test_integrate_r.py).
#
#   Rscript dev/sc_integrate_fixtures.R write <root> <reference.json>
#     Loads seeded synthetic data with three batches twice, into <root>/base
#     (left untouched for Python to copy) and <root>/r_int, drops the first
#     cells on r_int, runs the HVGs and the PCA on it, and from there every
#     batch correction and integration metric. Writes what Python has to
#     reproduce as JSON.
#
# Everything the tests compare against is produced here, never typed in.

suppressPackageStartupMessages({
  library(bixverse)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L, args[[1]] == "write")
root <- args[[2]]
dir.create(root, recursive = TRUE, showWarnings = FALSE)

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

new_dir <- function(name) {
  d <- file.path(root, name)
  unlink(d, recursive = TRUE)
  dir.create(d, recursive = TRUE)
  d
}

data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 900L,
    n_genes = 300L,
    n_batches = 3L,
    n_samples = 4L,
    sample_bias = "even"
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
sc <- load("r_int")

ref <- list(base_dir = file.path(root, "base"), n_drop = 30L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)

sc <- find_hvg_sc(sc, hvg_no = 100L, .verbose = FALSE)
sc <- calculate_pca_sc(sc, no_pcs = 15L, .verbose = FALSE)
ref$pca <- unname(get_pca_factors(sc))

embd <- function(sc, name) unname(get_embedding(sc, name))

## harmony ---------------------------------------------------------------------

sc <- harmony_sc(sc, batch_column = "batch_index", seed = 42L, .verbose = FALSE)
ref$harmony <- embd(sc, "harmony")

sc_multi <- harmony_sc(
  sc,
  batch_column = "batch_index",
  additional_batch_columns = "sample_id",
  harmony_params = params_sc_harmony(
    k = 12L,
    theta = c(2, 1),
    kmeans = list(k_means_init = "random")
  ),
  seed = 7L,
  .verbose = FALSE
)
ref$harmony_multi <- embd(sc_multi, "harmony")

sc <- harmony_v2_sc(
  sc,
  batch_column = "batch_index",
  seed = 42L,
  .verbose = FALSE
)
ref$harmony_v2 <- embd(sc, "harmony_v2")

sc_v2 <- harmony_v2_sc(
  sc,
  batch_column = "batch_index",
  harmony_params = params_sc_harmony_v2(
    use_dynamic_lambda = FALSE,
    tau = 5,
    lambda = 0.5
  ),
  seed = 3L,
  .verbose = FALSE
)
ref$harmony_v2_fixed <- embd(sc_v2, "harmony_v2")

## anchors ---------------------------------------------------------------------

hvg <- find_hvg_batch_aware_sc(
  sc,
  batch_column = "batch_index",
  hvg_no = 80L,
  .verbose = FALSE
)
ref$batch_hvg <- hvg$hvg_gene_idx

sc <- fast_mnn_sc(
  sc,
  batch_column = "batch_index",
  batch_hvg_genes = hvg$hvg_gene_idx,
  fastmnn_params = params_sc_fastmnn(no_pcs = 15L, knn = list(k = 10L)),
  seed = 42L,
  .verbose = FALSE
)
ref$mnn <- embd(sc, "mnn")

sc_pre <- fast_mnn_sc(
  sc,
  batch_column = "batch_index",
  batch_hvg_genes = hvg$hvg_gene_idx,
  fastmnn_params = params_sc_fastmnn(
    no_pcs = 15L,
    cos_norm = FALSE,
    knn = list(k = 10L, ann_dist = "euclidean")
  ),
  use_precomputed_pca = TRUE,
  seed = 42L,
  .verbose = FALSE
)
ref$mnn_pre <- embd(sc_pre, "mnn")

sc <- seurat_cca_sc(
  sc,
  batch_column = "batch_index",
  batch_hvg_genes = hvg$hvg_gene_idx,
  cca_params = params_sc_seurat_cca(
    num_cc = 10L,
    dims = 10L,
    k_filter = 50L,
    k_weight = 50L,
    n_top_features = 50L
  ),
  seed = 42L,
  .verbose = FALSE
)
ref$cca <- embd(sc, "cca")

sc <- seurat_rpca_sc(
  sc,
  batch_column = "batch_index",
  batch_hvg_genes = hvg$hvg_gene_idx,
  rpca_params = params_sc_seurat_rpca(dims = 10L, k_weight = 50L),
  seed = 42L,
  .verbose = FALSE
)
ref$rpca <- embd(sc, "rpca")

sc_rpca_pre <- seurat_rpca_sc(
  sc,
  batch_column = "batch_index",
  batch_hvg_genes = hvg$hvg_gene_idx,
  rpca_params = params_sc_seurat_rpca(dims = 10L, k_weight = 50L),
  use_precomputed_pca = TRUE,
  seed = 42L,
  .verbose = FALSE
)
ref$rpca_pre <- embd(sc_rpca_pre, "rpca")

## bbknn -----------------------------------------------------------------------

graph_csr <- function(sc) {
  conn <- get_connectivities(sc)
  csr <- bixverse:::rs_graph_to_csr(bixverse:::.connectivity_ptr(conn))
  c(csr, list(n = conn$n, n_edges = conn$n_edges, method = conn$method))
}
knn_report <- function(knn) {
  list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)
}

sc_bb <- suppressWarnings(bbknn_sc(
  sc,
  batch_column = "batch_index",
  no_neighbours_to_keep = 5L,
  seed = 42L,
  .verbose = FALSE
))
ref$bbknn <- list(
  knn = knn_report(get_knn_obj(sc_bb)),
  graph = graph_csr(sc_bb)
)

sc_bb <- suppressWarnings(bbknn_sc(
  sc,
  batch_column = "batch_index",
  no_neighbours_to_keep = 8L,
  no_embd_to_use = 10L,
  bbknn_params = params_sc_bbknn(
    neighbours_within_batch = 4L,
    set_op_mix_ratio = 0.5,
    trim = 15L,
    knn = list(knn_method = "exhaustive", ann_dist = "cosine")
  ),
  seed = 42L,
  .verbose = FALSE
))
ref$bbknn_custom <- list(
  knn = knn_report(get_knn_obj(sc_bb)),
  graph = graph_csr(sc_bb)
)

## metrics ---------------------------------------------------------------------

sc <- find_neighbours_sc(sc, .verbose = FALSE)
ref$knn <- knn_report(get_knn_obj(sc))

strip <- function(x) unclass(x)

ref$kbet <- strip(calculate_kbet_sc(
  sc,
  batch_column = "batch_index",
  .verbose = FALSE
))
ref$kbet_strict <- strip(calculate_kbet_sc(
  sc,
  batch_column = "sample_id",
  threshold = 0.2,
  .verbose = FALSE
))
ref$lisi_batch <- strip(calculate_lisi_sc(
  sc,
  label_column = "batch_index",
  .verbose = FALSE
))
ref$lisi_weighted <- strip(calculate_lisi_sc(
  sc,
  label_column = "batch_index",
  weighted = TRUE,
  perplexity = 10,
  .verbose = FALSE
))
ref$lisi_cell_type <- strip(calculate_lisi_sc(
  sc,
  label_column = "cell_grp",
  type = "cell_type",
  .verbose = FALSE
))
ref$batch_asw <- strip(calculate_batch_asw_sc(
  sc,
  batch_column = "batch_index",
  .verbose = FALSE
))
ref$batch_asw_sub <- strip(calculate_batch_asw_sc(
  sc,
  batch_column = "batch_index",
  embd_to_use = "harmony",
  max_cells = 300L,
  seed = 11L,
  .verbose = FALSE
))
ref$pcr_pca <- strip(calculate_pcr_sc(
  sc,
  batch_column = "batch_index",
  .verbose = FALSE
))
ref$pcr_harmony <- strip(calculate_pcr_sc(
  sc,
  batch_column = "batch_index",
  embd_to_use = "harmony",
  .verbose = FALSE
))
ref$cell_type_asw <- strip(calculate_cell_type_asw_sc(
  sc,
  cell_type_column = "cell_grp",
  embd_to_use = "mnn",
  max_cells = NULL,
  .verbose = FALSE
))
conn <- calculate_graph_connectivity_sc(
  sc,
  cell_type_column = "cell_grp",
  .verbose = FALSE
)
ref$connectivity <- list(
  names = names(conn$per_cell_type),
  values = unname(conn$per_cell_type),
  mean = conn$mean_connectivity,
  median = conn$median_connectivity
)
ref$summary <- as.list(calculate_integration_metrics_sc(
  sc,
  batch_column = "batch_index",
  cell_type_column = "cell_grp",
  embd_to_use = "harmony",
  .verbose = FALSE
))
ref$summary_no_embedding <- as.list(calculate_integration_metrics_sc(
  sc,
  batch_column = "batch_index",
  embd_to_use = NULL,
  .verbose = FALSE
))

write_json(ref, args[[3]])
