# R side of the kNN, graph and clustering parity tests (tests/test_graph_r.py).
#
#   Rscript dev/sc_graph_fixtures.R write <root> <reference.json>
#     Loads seeded synthetic data twice, into <root>/base (left untouched for
#     Python to copy) and <root>/r_graph, drops the first cells on r_graph,
#     runs the HVGs and the PCA on it, and from that PCA the kNN searches,
#     neighbour graphs, Leiden / Louvain, fast clustering and kNN metrics.
#     Writes what Python has to reproduce as JSON.
#
#   Rscript dev/sc_graph_fixtures.R read <dir> <out.json>
#     Opens a directory with load_existing() and reports its obs table with
#     the R class of every column.
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
  obs <- get_sc_obs(sc)
  write_json(
    list(
      obs = column_values(obs),
      class = lapply(as.list(obs), function(x) class(x)[[1]])
    ),
    args[[3]]
  )
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
sc <- load("r_graph")

ref <- list(base_dir = file.path(root, "base"), n_drop = 40L)
cell_names <- get_cell_names(sc)
ref$keep <- cell_names[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)

sc <- find_hvg_sc(sc, hvg_no = 100L, .verbose = FALSE)
sc <- calculate_pca_sc(sc, no_pcs = 15L, .verbose = FALSE)
ref$pca <- unname(get_pca_factors(sc))

knn_report <- function(knn) {
  list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)
}

## knn ------------------------------------------------------------------------

ref$k <- 15L
ref$knn <- list()
for (dist in c("euclidean", "cosine")) {
  for (m in c("exhaustive", "kmknn", "annoy", "hnsw", "nndescent", "ivf")) {
    knn <- generate_knn_sc(
      sc,
      neighbours_params = params_sc_neighbours(
        knn = list(knn_method = m, ann_dist = dist, k = ref$k)
      ),
      seed = 42L,
      .validate_index = FALSE,
      .verbose = FALSE
    )
    ref$knn[[paste(m, dist, sep = "_")]] <- knn_report(knn)
  }
}

# fewer PCs and a cell subset
ref$subset_cells <- ref$keep[seq(1L, length(ref$keep), by = 3L)]
ref$knn_subset <- knn_report(generate_knn_sc(
  sc,
  cells_to_use = ref$subset_cells,
  neighbours_params = params_sc_neighbours(knn = list(knn_method = "exhaustive")),
  .validate_index = FALSE,
  .verbose = FALSE
))

ref$metrics <- local({
  exact <- generate_knn_sc(
    sc,
    neighbours_params = params_sc_neighbours(knn = list(knn_method = "exhaustive")),
    .validate_index = FALSE,
    .verbose = FALSE
  )
  approx <- generate_knn_sc(
    sc,
    neighbours_params = params_sc_neighbours(
      knn = list(knn_method = "ivf", n_list = 20L, n_probe = 1L)
    ),
    .validate_index = FALSE,
    .verbose = FALSE
  )
  c(calc_knn_metrics(exact, approx), list(query = knn_report(approx)))
})

## graphs and clusters --------------------------------------------------------

graph_csr <- function(sc) {
  conn <- get_connectivities(sc)
  csr <- bixverse:::rs_graph_to_csr(bixverse:::.connectivity_ptr(conn))
  c(csr, list(n = conn$n, n_edges = conn$n_edges, method = conn$method))
}

graph_runs <- list(
  snn_jaccard = list(knn_sym = "snn", snn_similarity = "jaccard"),
  snn_rank = list(knn_sym = "snn", snn_similarity = "rank", pruning = 0.1),
  snn_no_rescue = list(knn_sym = "snn", rescue = FALSE, pruning = 0.2),
  umap = list(knn_sym = "umap"),
  umap_mixed = list(knn_sym = "umap", local_connectivity = 1.5, mix_ratio = 0.5),
  knn = list(knn_sym = "knn"),
  knn_binary = list(knn_sym = "knn", binary = TRUE)
)
ref$graphs <- list()
for (run in names(graph_runs)) {
  sc <- find_neighbours_sc(
    sc,
    neighbours_params = params_sc_neighbours(graph = graph_runs[[run]]),
    .verbose = FALSE
  )
  ref$graphs[[run]] <- graph_csr(sc)
}

# the default graph, then the clusterings on it
sc <- find_neighbours_sc(sc, no_embd_to_use = 10L, .verbose = FALSE)
ref$graph_default <- graph_csr(sc)
ref$knn_default <- knn_report(get_knn_obj(sc))

cluster_runs <- list(
  leiden = list(cluster_algorithm = "leiden", res = 1.0, seed = 42L),
  leiden_hi = list(cluster_algorithm = "leiden", res = 2.0, seed = 7L),
  louvain = list(cluster_algorithm = "louvain", res = 1.0, seed = 42L),
  leiden_stable = list(
    cluster_algorithm = "leiden",
    res = 0.5,
    seed = 3L,
    n_iterations = -1L,
    beta = 0.05
  )
)
ref$clusters <- list()
for (run in names(cluster_runs)) {
  sc <- do.call(
    find_clusters_sc,
    c(list(object = sc, name = run), cluster_runs[[run]])
  )
  ref$clusters[[run]] <- get_sc_obs(sc, filtered = TRUE)[[run]]
}
ref$clusters_class <- class(get_sc_obs(sc)[["leiden"]])

## fast clustering ------------------------------------------------------------

fc_report <- function(res) {
  list(
    memberships = column_values(get_data(res)),
    stats = if (is.null(res$stats)) NULL else column_values(res$stats),
    k_means_cluster = res$k_means_cluster,
    centroids = res$centroids
  )
}

ref$fast <- list(
  single = fc_report(fast_cluster_sc(
    sc,
    resolutions = c(1.0, 0.5),
    fc_params = params_sc_fast_cluster(n_centroids = 30L),
    return_kmeans = TRUE,
    .verbose = FALSE
  )),
  seeds = fc_report(fast_cluster_sc(
    sc,
    resolutions = c(2.0, 1.0),
    fc_params = params_sc_fast_cluster(
      n_centroids = 40L,
      algorithm = "leiden",
      km_type = "minibatch",
      batch_size = 128L
    ),
    n_seeds = 3L,
    seed = 11L,
    .verbose = FALSE
  )),
  cell_mediated = fc_report(fast_cluster_sc(
    sc,
    no_embd_to_use = 8L,
    resolutions = 1.0,
    fc_params = params_sc_fast_cluster(
      cell_mediated = TRUE,
      dist = "cosine",
      centroid_graph = list(knn_sym = "umap")
    ),
    .verbose = FALSE
  )),
  umap = fc_report(fast_cluster_sc(
    sc,
    resolutions = 1.5,
    fc_params = params_sc_fast_cluster(
      n_centroids = 50L,
      centroid_k = 10L,
      centroid_graph = list(knn_sym = "umap")
    ),
    .verbose = FALSE
  ))
)

write_json(ref, args[[3]])
