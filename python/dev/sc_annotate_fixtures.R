# R side of the ScType and Symphony parity tests (tests/test_annotate_r.py).
#
#   Rscript dev/sc_annotate_fixtures.R write <root> <reference.json>
#     Generates two seeded synthetic data sets with planted markers (genes 0
#     to 9, 10 to 19 and 20 to 29 for cell types 1 to 3) and two batches
#     each: the reference and a query. The query loses a handful of genes,
#     two of them HVGs of the reference, so the gene matching has to cope.
#     Each is loaded twice, into <root>/<name>_base (left untouched for
#     Python to copy) and <root>/r_<name>. On r_ref it drops the first cells,
#     scores ScType against the planted markers (plus a decoy type sharing
#     markers with types 1 and 2), clusters and assigns, then builds Symphony
#     references (Harmony v1 and v2), maps the query with and without its
#     batch and transfers the cell type labels. Writes what Python has to
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
    matrix = "rowmajor",
    dataframe = "columns"
  )
}

new_dir <- function(name) {
  d <- file.path(root, name)
  unlink(d, recursive = TRUE)
  dir.create(d, recursive = TRUE)
  d
}

min_qc <- params_sc_min_quality(
  min_unique_genes = 5L,
  min_lib_size = 25L,
  min_cells = 5L
)
load <- function(data, name) {
  suppressWarnings(load_r_data(
    SingleCells(dir_data = new_dir(name)),
    data$counts,
    data$obs,
    data$var,
    sc_qc_param = min_qc,
    .verbose = FALSE
  ))
}

## reference ---------------------------------------------------------------------

ref_data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 1200L,
    n_genes = 200L,
    n_batches = 2L,
    batch_effect_strength = "medium"
  ),
  seed = 123L
)
invisible(load(ref_data, "ref_base"))
sc <- load(ref_data, "r_ref")

out <- list(ref_base = file.path(root, "ref_base"), n_drop = 20L)
out$keep <- get_cell_names(sc)[-seq_len(out$n_drop)]
sc <- set_cells_to_keep(sc, out$keep)

sc <- find_hvg_sc(sc, hvg_no = 80L, .verbose = FALSE)
sc <- calculate_pca_sc(sc, no_pcs = 15L, .verbose = FALSE)
out$pca <- unname(get_pca_factors(sc))
sc <- find_neighbours_sc(sc, .verbose = FALSE)
sc <- find_clusters_sc(sc, name = "leiden")
out$leiden <- get_sc_obs(sc, filtered = TRUE)$leiden

## ScType ------------------------------------------------------------------------

gene_ids <- get_gene_names(sc)
marker_df <- data.table(
  cell_type = c(
    rep(sprintf("cell_type_%i", 1:3), each = 10L),
    rep("decoy", 6L),
    "missing_only"
  ),
  gene_id = c(gene_ids[1:30], gene_ids[c(1:3, 11:13)], "not_a_gene")
)
out$marker_df <- copy(marker_df)
markers <- prepare_cell_markers(obj = sc, marker_df = copy(marker_df))
out$marker_names <- names(markers)
out$marker_indices <- lapply(markers, function(m) m$positive_indices)

sc_type_out <- function(res) {
  list(cell_types = res$cell_types, scores = unname(get_scores(res)))
}
scores <- calc_sc_type_scores(sc, cell_marker_list = markers, .verbose = FALSE)
out$sctype <- sc_type_out(scores)
out$sctype_flat <- sc_type_out(calc_sc_type_scores(
  sc,
  cell_marker_list = markers,
  sensitivity = FALSE,
  .verbose = FALSE
))
out$sctype_floor <- sc_type_out(calc_sc_type_scores(
  sc,
  cell_marker_list = markers,
  weight_floor = 0.4,
  .verbose = FALSE
))

out$score_clusters <- score_clusters(scores, cluster_labels = out$leiden)

cell_out <- function(res) {
  list(
    cell_types = res$cell_types,
    assignments = res$assignments,
    scores = res$scores,
    margins = res$margins,
    agreement = res$agreement,
    hybrid = res$hybrid,
    composition = res$composition,
    counts = if (is.null(res$counts)) NULL else unname(res$counts)
  )
}
out$assign <- cell_out(assign_sc_type(sc, scores, .verbose = FALSE))
out$assign_hybrid <- cell_out(assign_sc_type(
  sc,
  scores,
  cluster_col = "leiden",
  .verbose = FALSE
))
out$assign_z <- cell_out(assign_sc_type(
  sc,
  scores,
  cluster_col = "leiden",
  sctype_cell_params = params_sctype_cells(
    calibration = "column_z",
    alpha = 0.3,
    iterations = 5L,
    score_floor = 0.5,
    purity_threshold = 0.7
  ),
  .verbose = FALSE
))

## Symphony reference ------------------------------------------------------------

out$hvg <- get_hvg(sc)
ref_out <- function(r) {
  list(
    hvg_gene_names = get_symphony_hvg_names(r),
    gene_means = S7::prop(r, "gene_means"),
    gene_sds = S7::prop(r, "gene_sds"),
    loadings = get_symphony_loadings(r),
    z_orig = S7::prop(r, "z_orig"),
    z_corr = get_symphony_z_corr(r),
    r = S7::prop(r, "r"),
    centroids = S7::prop(r, "centroids"),
    nr = S7::prop(r, "nr"),
    c_cache = S7::prop(r, "c_cache"),
    labels = get_symphony_labels(r)
  )
}
symphony_ref <- build_symphony_ref(
  sc,
  batch_column = "batch_index",
  hvg = get_hvg(sc) + 1L,
  harmony_params = params_sc_harmony(),
  no_pcs = 10L,
  label_columns = "cell_grp",
  .verbose = FALSE
)
out$symphony <- ref_out(symphony_ref)
out$symphony_v2 <- ref_out(build_symphony_ref(
  sc,
  batch_column = "batch_index",
  hvg = get_hvg(sc) + 1L,
  harmony_params = params_sc_harmony_v2(k = 12L),
  no_pcs = 8L,
  slim = TRUE,
  seed = 7L,
  .verbose = FALSE
))

## query -------------------------------------------------------------------------

query_data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 700L,
    n_genes = 200L,
    n_batches = 2L,
    batch_effect_strength = "strong"
  ),
  seed = 456L
)
# two reference HVGs plus three other genes the query does not measure
hvg_drop <- out$hvg[1:2] + 1L
other_drop <- setdiff(seq_len(200L), out$hvg + 1L)[1:3]
drop <- c(hvg_drop, other_drop)
out$query_dropped_genes <- gene_ids[drop]
query_data$counts <- query_data$counts[, -drop, drop = FALSE]
query_data$var <- query_data$var[-drop]
invisible(load(query_data, "query_base"))
query <- load(query_data, "r_query")
out$query_base <- file.path(root, "query_base")

query <- map_symphony_query(
  symphony_ref,
  query = query,
  batch_column = "batch_index",
  .verbose = FALSE
)
out$map <- list(
  z_corr = unname(get_embedding(query, "symphony")),
  z_pca = unname(get_embedding(query, "symphony_pca")),
  r = unname(get_embedding(query, "symphony_r"))
)
out$transfer <- transfer_labels_symphony(
  symphony_ref,
  query = query,
  label_column = "cell_grp",
  knn_params = params_sc_knn(ann_dist = "euclidean"),
  .verbose = FALSE
)
out$query_truth <- get_sc_obs(query, filtered = TRUE)$cell_grp

query_nb <- map_symphony_query(
  symphony_ref,
  query = query,
  params = params_symphony_map(sigma = 0.2),
  .verbose = FALSE
)
out$map_no_batch <- list(
  z_corr = unname(get_embedding(query_nb, "symphony")),
  z_pca = unname(get_embedding(query_nb, "symphony_pca")),
  r = unname(get_embedding(query_nb, "symphony_r"))
)

write_json(out, args[[3]])
