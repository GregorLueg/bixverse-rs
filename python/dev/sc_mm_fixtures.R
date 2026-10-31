# R side of the multi-modal parity tests (tests/test_mm_r.py).
#
#   Rscript dev/sc_mm_fixtures.R write <root> <reference.json>
#     Loads seeded synthetic RNA twice, into <root>/base (left untouched for
#     Python to copy) and <root>/r_mm as a SingleCellsMultiModal, and drops
#     the first cells on r_mm. Adds the seeded synthetic ADT counts with their
#     rows reversed (so the alignment to the kept cells is exercised), then:
#     CLR (both variants, with and without the percentile clean), DSB (k-means
#     fallback with and without isotypes, empty droplets with both scale
#     factors and clipping), the ADT PCA (full and randomised), the RNA PCA,
#     WNN (default and a non-default parameter set with a UMAP graph), the ADT
#     kNN and graph, Leiden on the ADT and WNN graphs, PAGA and MAGIC over the
#     WNN and ADT kNN. Writes what Python has to reproduce as JSON.
#
#   Rscript dev/sc_mm_fixtures.R read <dir> <out.json>
#     Opens a directory as a SingleCellsMultiModal and reports its var_adt
#     table and its obs, with the R class of every column.
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

columns <- function(d) {
  lapply(as.list(d), function(x) if (is.factor(x)) as.character(x) else x)
}
classes <- function(d) lapply(as.list(d), function(x) class(x)[[1]])

if (mode == "read") {
  mm <- suppressMessages(load_existing(SingleCellsMultiModal(dir_data = args[[2]])))
  var_adt <- get_sc_var(mm, modality = "adt")
  obs <- get_sc_obs(mm)
  write_json(
    list(
      var_adt = columns(var_adt),
      var_adt_class = classes(var_adt),
      obs = columns(obs),
      obs_class = classes(obs)
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

mat <- function(m) unname(as.matrix(unclass(m)))
dt <- function(d) as.list(as.data.frame(d))
knn_list <- function(k) {
  list(indices = k$indices, dist = k$dist, metric = k$dist_metric)
}
graph_csr <- function(conn) {
  csr <- bixverse:::rs_graph_to_csr(bixverse:::.connectivity_ptr(conn))
  c(csr, list(n = conn$n, n_edges = conn$n_edges, method = conn$method))
}

## data ------------------------------------------------------------------------

n_cells <- 600L
data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(n_cells = n_cells, n_genes = 120L),
  seed = 123L
)
adt <- generate_single_cell_test_data_adt(
  syn_data_params = params_sc_synthetic_data_adt(n_cells = n_cells),
  seed = 123L
)
empty <- floor(
  generate_single_cell_test_data_adt(
    syn_data_params = params_sc_synthetic_data_adt(n_cells = 300L),
    seed = 7L
  )$counts /
    10
)
adt_rev <- adt$counts[rev(seq_len(nrow(adt$counts))), ]
iso <- adt$var$protein_id[adt$var$is_isotype]

min_qc <- params_sc_min_quality(
  min_unique_genes = 5L,
  min_lib_size = 25L,
  min_cells = 5L
)
load <- function(cls, name) {
  suppressWarnings(load_r_data(
    cls(dir_data = new_dir(name)),
    data$counts,
    data$obs,
    data$var,
    sc_qc_param = min_qc,
    .verbose = FALSE
  ))
}
invisible(load(SingleCells, "base"))
mm <- load(SingleCellsMultiModal, "r_mm")

ref <- list(base_dir = file.path(root, "base"), n_drop = 20L, isotypes = iso)
ref$keep <- get_cell_names(mm)[-seq_len(ref$n_drop)]
mm <- set_cells_to_keep(mm, ref$keep)
cell_info <- get_cell_info(mm, filtered = TRUE)

## ADT -------------------------------------------------------------------------

mm <- add_adt_counts_sc(mm, adt_counts = adt_rev, method = "clr")
adt_obj <- S7::prop(mm, "adt_counts")
ref$adt_rows <- rownames(adt_obj$raw_counts)
ref$clr <- mat(adt_obj$norm_counts)
ref$var_adt <- dt(get_sc_var(mm, modality = "adt"))
ref$var_adt_class <- classes(get_sc_var(mm, modality = "adt"))
obs <- get_sc_obs(mm)
ref$obs_adt <- dt(obs[, .(cell_idx, adt_nnz, adt_lib_size)])
ref$obs_adt_class <- classes(obs[, .(adt_nnz, adt_lib_size)])

ref$clr_seurat <- mat(
  new_adt_counts_clr(
    adt_rev,
    cell_info,
    seurat_clr = TRUE,
    clean_clr_counts = FALSE
  )$norm_counts
)
ref$clr_seurat_clean <- mat(
  new_adt_counts_clr(
    adt_rev,
    cell_info,
    seurat_clr = TRUE,
    percentile = 0.05
  )$norm_counts
)

dsb_report <- function(x) {
  od <- x$other_data
  list(
    norm = mat(x$norm_counts),
    protein_background_mean = od$protein_background_mean,
    protein_background_sd = od$protein_background_sd,
    technical_component = od$technical_component,
    cellwise_background_mean = od$cellwise_background_mean
  )
}
ref$dsb <- list(
  kmeans_iso = dsb_report(new_adt_counts_dsb(
    adt_rev,
    cell_info,
    isotype_names = iso,
    .verbose = FALSE
  )),
  kmeans_no_iso = dsb_report(new_adt_counts_dsb(
    adt_rev,
    cell_info,
    dsb_params = params_sc_dsb(use_isotype_controls = FALSE, pseudocount = 1),
    .verbose = FALSE
  )),
  empty_std = dsb_report(new_adt_counts_dsb(
    adt_rev,
    cell_info,
    empty_drops = empty,
    isotype_names = iso,
    seed = 3L,
    .verbose = FALSE
  )),
  empty_ms_clip = dsb_report(new_adt_counts_dsb(
    adt_rev,
    cell_info,
    empty_drops = empty,
    dsb_params = params_sc_dsb(
      denoise_counts = FALSE,
      use_isotype_controls = FALSE,
      quantile_low = 0.01,
      quantile_high = 0.99
    ),
    scale_factor = "mean_subtract",
    .verbose = FALSE
  ))
)

## PCA -------------------------------------------------------------------------

mm <- calculate_pca_adt_sc(mm, no_pcs = 10L)
ref$adt_pca <- list(
  factors = mat(get_pca_factors(mm, modality = "adt")),
  loadings = mat(get_pca_loadings(mm, modality = "adt")),
  sv = get_pca_singular_val(mm, modality = "adt")
)
mm_rand <- calculate_pca_adt_sc(mm, no_pcs = 5L, randomised_svd = TRUE, seed = 9L)
ref$adt_pca_rand <- list(
  factors = mat(get_pca_factors(mm_rand, modality = "adt")),
  loadings = mat(get_pca_loadings(mm_rand, modality = "adt")),
  sv = get_pca_singular_val(mm_rand, modality = "adt")
)

mm <- find_hvg_sc(mm, hvg_no = 40L, .verbose = FALSE)
mm <- calculate_pca_sc(mm, no_pcs = 10L, .verbose = FALSE)
ref$rna_pca <- mat(get_pca_factors(mm))

## WNN -------------------------------------------------------------------------

wnn_report <- function(x) {
  slot <- S7::prop(x, "other_data")[["wnn"]]
  list(
    knn = knn_list(get_knn_obj(x, modality = "wnn")),
    weights = dt(slot$weights),
    graph = graph_csr(get_connectivities(x, modality = "wnn"))
  )
}
mm <- generate_wnn_graph_sc(
  mm,
  wnn_params = params_sc_wnn(knn_range = 100L),
  .verbose = FALSE
)
ref$wnn <- wnn_report(mm)

mm_alt <- generate_wnn_graph_sc(
  mm,
  no_embd_to_use_1 = 8L,
  wnn_params = params_sc_wnn(
    k_nn = 15L,
    knn_range = 60L,
    s_nn = 12L,
    sd_scale = 1.5,
    kernel_power = 2,
    cross_const = 1e-3
  ),
  graph_params = modifyList(params_knn_graph_defaults(), list(knn_sym = "umap")),
  seed = 5L,
  .verbose = FALSE
)
ref$wnn_alt <- wnn_report(mm_alt)

## modality routing ------------------------------------------------------------

mm <- find_neighbours_sc(
  mm,
  modality = "adt",
  neighbours_params = params_sc_neighbours(
    knn = list(k = 15L, knn_method = "exhaustive")
  ),
  .verbose = FALSE
)
ref$adt_knn <- knn_list(get_knn_obj(mm, modality = "adt"))
ref$adt_graph <- graph_csr(get_connectivities(mm, modality = "adt"))

mm <- find_clusters_sc(mm, modality = "adt", name = "adt_leiden")
mm <- find_clusters_sc(mm, modality = "wnn", name = "wnn_leiden", res = 0.5)
ref$clusters <- dt(get_sc_obs(mm, filtered = TRUE)[, .(adt_leiden, wnn_leiden)])

paga <- run_paga_sc(mm, cluster_col = "cell_grp", modality = "wnn", .verbose = FALSE)
ref$paga_wnn <- list(
  connectivities = unname(as.matrix(paga$connectivities)),
  tree = unname(as.matrix(paga$connectivities_tree))
)

genes <- get_gene_names(mm)[1:6]
ref$magic_genes <- genes
mm_magic <- run_magic_sc(mm, features = genes, modality = "adt", .verbose = FALSE)
ref$magic_adt <- mat(get_magic(mm_magic)$data)

write_json(ref, args[[3]])
