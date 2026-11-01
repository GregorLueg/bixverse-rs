# R side of the plot data extractor parity tests (tests/test_pl_r.py).
#
#   Rscript dev/sc_pl_fixtures.R <root> <reference.json>
#
# Loads seeded synthetic RNA three times: <root>/base (left untouched for
# Python to copy), <root>/r_sc as a SingleCells and <root>/r_mm as a
# SingleCellsMultiModal, and drops the first cells on both. On r_sc: HVGs,
# PCA, neighbours, Leiden, MAGIC, PAGA, Milo with its test, a subset and
# bootstrapped meta cells. On r_mm: the same PCA plus the seeded synthetic ADT
# counts (rows reversed) with their PCA. Then every plot data extractor over
# all of them. Python takes R's PCA, kNN, clusters, MAGIC layer, PAGA and Milo
# results, so the tests check the extraction, not the upstream steps. Writes
# what Python has to reproduce as JSON.
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

mat <- function(m) unname(as.matrix(unclass(m)))
cols <- function(d) {
  lapply(as.list(d), function(x) if (is.factor(x)) as.character(x) else x)
}
levels_of <- function(d) {
  Filter(Negate(is.null), lapply(as.list(d), levels))
}
tab <- function(d) list(data = cols(d), levels = levels_of(d))
plot_list <- function(x) {
  list(nodes = tab(x$nodes), edges = tab(x$edges), embedding = attr(x, "embedding"))
}

## data ------------------------------------------------------------------------

data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 600L,
    n_genes = 120L,
    n_samples = 4L,
    sample_bias = "even"
  ),
  seed = 123L
)
adt <- generate_single_cell_test_data_adt(
  syn_data_params = params_sc_synthetic_data_adt(n_cells = 600L),
  seed = 123L
)
adt_rev <- adt$counts[rev(seq_len(nrow(adt$counts))), ]

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
sc <- load(SingleCells, "r_sc")
mm <- load(SingleCellsMultiModal, "r_mm")

ref <- list(base_dir = file.path(root, "base"), n_drop = 20L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)
mm <- set_cells_to_keep(mm, ref$keep)

## upstream steps --------------------------------------------------------------

sc <- find_hvg_sc(sc, hvg_no = 40L, .verbose = FALSE)
sc <- calculate_pca_sc(sc, no_pcs = 10L, .verbose = FALSE)
sc <- find_neighbours_sc(sc, .verbose = FALSE)
sc <- find_clusters_sc(sc, res = 1.0)
ref$pca <- mat(get_pca_factors(sc))
knn <- get_knn_obj(sc)
ref$knn <- list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)
ref$leiden <- as.character(get_sc_obs(sc, filtered = TRUE)$leiden_clustering)

genes <- get_gene_names(sc)
ref$genes <- genes[c(3L, 1L, 7L, 12L)]
ref$magic_genes <- genes[1:6]
sc <- run_magic_sc(sc, features = ref$magic_genes, .verbose = FALSE)
ref$magic <- mat(get_magic(sc)$data)

mm <- find_hvg_sc(mm, hvg_no = 40L, .verbose = FALSE)
mm <- calculate_pca_sc(mm, no_pcs = 10L, .verbose = FALSE)
mm <- add_adt_counts_sc(mm, adt_counts = adt_rev, method = "clr")
mm <- calculate_pca_adt_sc(mm, no_pcs = 5L)
ref$mm_pca <- mat(get_pca_factors(mm))
ref$adt_pca <- mat(get_pca_factors(mm, modality = "adt"))
ref$proteins <- adt$var$protein_id[c(2L, 5L, 1L)]

## embeddings ------------------------------------------------------------------

ref$embedding <- tab(extract_embedding_data(
  sc,
  "pca",
  obs_cols = c("cell_grp", "lib_size")
))
ref$embedding_adt <- tab(extract_embedding_data(mm, "pca", modality = "adt"))

## expression ------------------------------------------------------------------

# one unknown gene, which warns and drops
wanted <- c(ref$genes, "not_a_gene")
ref$expr <- tab(suppressWarnings(extract_gene_expression(
  sc,
  wanted,
  obs_cols = "cell_grp"
)))
ref$expr_scaled <- tab(extract_gene_expression(
  sc,
  ref$genes,
  scale = TRUE,
  clip = 1.5
))
ref$expr_magic <- tab(extract_gene_expression(
  sc,
  ref$magic_genes[c(4L, 2L)],
  layer = "magic",
  scale = TRUE
))
ref$expr_adt <- tab(extract_gene_expression(
  mm,
  ref$proteins,
  obs_cols = "cell_grp",
  modality = "adt"
))
ref$expr_adt_scaled <- tab(extract_gene_expression(
  mm,
  ref$proteins,
  scale = TRUE,
  clip = 2,
  modality = "adt"
))

## dot plots -------------------------------------------------------------------

ref$dot <- tab(extract_dot_plot_data(sc, ref$genes, "cell_grp"))
ref$dot_raw <- tab(extract_dot_plot_data(
  sc,
  ref$genes,
  "sample_id",
  scale_exp = FALSE
))
ref$dot_adt <- tab(extract_dot_plot_data(
  mm,
  ref$proteins,
  "cell_grp",
  modality = "adt"
))

## violins, feature plots, pairs -----------------------------------------------

ref$violin <- tab(extract_gene_violin_data(
  sc,
  ref$genes,
  "cell_grp",
  scale = TRUE
))
ref$feature <- tab(extract_feature_plot_data(sc, ref$genes[1:2], "pca"))
ref$feature_adt <- tab(extract_feature_plot_data(
  mm,
  ref$proteins[1:2],
  "pca",
  expr_modality = "adt",
  embd_modality = "adt"
))
ref$feature_cross <- tab(extract_feature_plot_data(
  mm,
  ref$genes[1],
  "pca",
  expr_modality = "rna",
  embd_modality = "adt"
))
ref$pair <- tab(extract_feature_pair(
  sc,
  ref$genes[1],
  ref$genes[2],
  obs_cols = "cell_grp"
))
ref$pair_cross <- tab(extract_feature_pair(
  mm,
  paste0(ref$genes[1], "_rna"),
  paste0(ref$proteins[1], "_adt"),
  scale = TRUE
))

## subset and meta cells -------------------------------------------------------

ref$subset_group <- "cell_type_2"
sub <- SingleCellsSubset(
  sc_object = sc,
  grouping_column = "cell_grp",
  group = ref$subset_group
)
ref$subset_expr <- tab(extract_gene_expression(
  sub,
  ref$genes,
  obs_cols = "sample_id"
))
ref$subset_dot <- tab(extract_dot_plot_data(sub, ref$genes, "sample_id"))

mc <- generate_bt_meta_cells_sc(
  sc,
  params_sc_bt_metacells(target_no_metacells = 40L, knn = list(k = 10L)),
  .verbose = FALSE
)
n_mc <- nrow(get_sc_obs(mc))
mc[["mc_grp"]] <- rep(c("b", "a", "c"), length.out = n_mc)
ref$mc_grp <- rep(c("b", "a", "c"), length.out = n_mc)
ref$mc_expr <- tab(extract_gene_expression(
  mc,
  ref$genes,
  obs_cols = "mc_grp",
  scale = TRUE
))
ref$mc_dot <- tab(extract_dot_plot_data(mc, ref$genes, "mc_grp"))
ref$single_group_dot <- tab(extract_dot_plot_data(sub, ref$genes, "cell_grp"))
ref$mc_levels <- c("a", "b", "c", "unused")
mc[["mc_fac"]] <- factor(ref$mc_grp, levels = ref$mc_levels)
ref$mc_dot_unused <- tab(extract_dot_plot_data(mc, ref$genes, "mc_fac"))
ref$mc_expr_unused <- tab(extract_gene_expression(
  mc,
  ref$genes,
  obs_cols = "mc_fac",
  scale = TRUE
))

## PAGA ------------------------------------------------------------------------

paga <- run_paga_sc(sc, cluster_col = "leiden_clustering", .verbose = FALSE)
ref$paga_res <- list(
  connectivities = unname(as.matrix(paga$connectivities)),
  tree = unname(as.matrix(paga$connectivities_tree)),
  clusters = rownames(paga$connectivities),
  sizes = cols(paga$sizes)
)
ref$paga <- plot_list(extract_paga_plot_data(sc, paga, embedding = "pca"))
ref$paga_tree <- plot_list(extract_paga_plot_data(
  sc,
  paga,
  embedding = "pca",
  tree_only = TRUE,
  centroid = "mean",
  threshold = 0.05,
  node_stat_col = "lib_size"
))

## Milo ------------------------------------------------------------------------

milo <- get_miloR_abundances_sc(
  sc,
  sample_id_col = "sample_id",
  miloR_params = params_sc_miloR(k_refine = 10L, index_type = "exhaustive"),
  seed = 42L,
  .verbose = FALSE
)
samples <- colnames(milo$sample_counts)
design_df <- data.frame(
  condition = ifelse(samples %in% c("sample_1", "sample_2"), "ctr", "trt"),
  row.names = samples
)
milo <- test_nhoods(milo, design = ~condition, design_df = design_df)
nh <- Matrix::summary(milo$nhoods)
ref$milo_res <- list(
  nhoods_i = nh$i - 1L,
  nhoods_j = nh$j - 1L,
  n_cells = nrow(milo$nhoods),
  n_nhoods = ncol(milo$nhoods),
  index_cell = milo$params$index_cell,
  nhoods_info = cols(get_differential_abundance_res(milo))
)
ref$milo <- plot_list(extract_milo_plot_data(sc, milo, embedding = "pca"))
ref$milo_alt <- plot_list(extract_milo_plot_data(
  sc,
  milo,
  embedding = "pca",
  alpha = 0.9,
  overlap = 3L
))

## filter change after the ADT counts went in --------------------------------

ref$keep_late <- ref$keep[-seq_len(30L)]
mm <- set_cells_to_keep(mm, ref$keep_late)
ref$late_expr_adt <- tab(extract_gene_expression(
  mm,
  ref$proteins,
  obs_cols = "cell_grp",
  modality = "adt"
))
ref$late_dot_adt <- tab(extract_dot_plot_data(
  mm,
  ref$proteins,
  "cell_grp",
  modality = "adt"
))

write_json(ref, args[[2]])
