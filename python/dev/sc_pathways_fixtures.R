# R side of the pathway, HotSpot and SCENIC parity tests
# (tests/test_pathways_r.py).
#
#   Rscript dev/sc_pathways_fixtures.R <root> <reference.json>
#
# Loads seeded synthetic data twice, into <root>/base (left untouched for
# Python to copy) and <root>/r_pw, and drops the first cells on r_pw. The
# three planted marker blocks (genes 1-10, 11-20, 21-30) are the gene sets.
# PCA and the kNN are computed here and handed over, so every method sees the
# same embedding and graph. Then: module scores, AUCell (all three
# statistics), VISION (with a negative arm), VISION with autocorrelation (the
# null gene sets are drawn here and handed over, as Python draws its own),
# ULM / MLM with p-values, HotSpot autocorrelation (cached kNN and a searched
# one over a cell subset) and gene-gene correlations with modules, the SCENIC
# gene filter, GRN inference with all three learners, the TF to gene links,
# their correlations, CisTarget on a synthetic ranking, regulons and their
# binarisation; and the meta cell twins on bootstrapped meta cells. R has no
# `SingleCellsSubset` methods for any of these. Writes what Python has to reproduce as JSON.
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

mat <- function(m) unname(as.matrix(unclass(m)))
dt <- function(d) as.list(as.data.frame(d))

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
sc <- load("r_pw")

ref <- list(base_dir = file.path(root, "base"), n_drop = 20L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)
genes <- get_gene_names(sc)

sc <- find_hvg_sc(sc, hvg_no = 40L, .verbose = FALSE)
sc <- calculate_pca_sc(sc, no_pcs = 10L, .verbose = FALSE)
sc <- find_neighbours_sc(
  sc,
  neighbours_params = params_sc_neighbours(
    knn = list(k = 15L, knn_method = "exhaustive")
  ),
  .verbose = FALSE
)
ref$pca <- mat(get_embedding(sc, "pca"))
knn <- get_knn_obj(sc)
ref$knn <- list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)

gs <- list(
  type_1 = genes[1:10],
  type_2 = genes[11:20],
  type_3 = genes[21:30]
)
ref$gs <- gs

## module scores and AUCell ----------------------------------------------------

ref$module_scores <- mat(module_scores_sc(
  sc,
  gs_list = gs,
  n_ctrl = 5L,
  .verbose = FALSE
))

ref$aucell <- list()
for (m in c("recovery", "wilcox", "ap")) {
  ref$aucell[[m]] <- mat(aucell_sc(
    sc,
    gs_list = gs,
    aucell_params = params_sc_aucell(auc_type = m, standardise = m == "ap"),
    .verbose = FALSE
  ))
}
ref$aucell_max_rank <- mat(aucell_sc(
  sc,
  gs_list = gs,
  aucell_params = params_sc_aucell(max_rank = 15),
  .verbose = FALSE
))

## VISION ----------------------------------------------------------------------

vgs <- list(
  type_1 = list(pos = genes[1:10]),
  type_2_vs_3 = list(pos = genes[11:20], neg = genes[21:30]),
  type_3 = list(pos = genes[21:30])
)
ref$vgs <- vgs
ref$vision <- mat(vision_sc(sc, gs_list = vgs, .verbose = FALSE))

vision_params <- params_sc_vision(
  n_perm = 50L,
  knn = list(k = 15L, knn_method = "exhaustive")
)
null_gs <- bixverse:::.generate_null_perm_gs(
  gs_list = vgs,
  expr_genes = genes,
  n_perm = vision_params$n_perm,
  n_comp = vision_params$n_cluster,
  random_seed = 42L
)
ref$vision_null <- list(
  random = null_gs$random_signatures,
  clusters = as.integer(null_gs$clusters) - 1L
)
autocor_report <- function(res) {
  list(
    vision = mat(res$vision_matrix),
    auto_cor = res$auto_cor_dt$auto_cor,
    p_val = res$auto_cor_dt$p_val,
    fdr = res$auto_cor_dt$fdr
  )
}
ref$vision_autocor <- autocor_report(vision_w_autocor_sc(
  sc,
  gs_list = vgs,
  embd_to_use = "pca",
  vision_params = vision_params,
  .verbose = FALSE
))
ref$vision_autocor_search <- autocor_report(vision_w_autocor_sc(
  sc,
  gs_list = vgs,
  embd_to_use = "pca",
  no_embd_to_use = 5L,
  use_knn = FALSE,
  vision_params = vision_params,
  .verbose = FALSE
))

## ULM / MLM -------------------------------------------------------------------

set.seed(5L)
net <- data.table(
  source = rep(c("tf_a", "tf_b", "tf_c"), each = 12L),
  target = c(genes[1:12], genes[9:20], genes[c(21:30, 40, 41)]),
  weight = round(runif(36L, -1, 1), 3)
)
ref$net <- dt(net)
lm_report <- function(res) {
  list(es = mat(res$es), pvals = mat(res$pvals), fdr = mat(res$fdr))
}
ref$ulm <- lm_report(lm_activity_sc(
  sc,
  net = net,
  lm_params = params_lm_activity(p_values = TRUE),
  .verbose = FALSE
))
ref$mlm <- lm_report(lm_activity_sc(
  sc,
  net = net,
  lm_params = params_lm_activity(method = "mlm", p_values = TRUE),
  .verbose = FALSE
))
ref$ulm_coef <- mat(lm_activity_sc(
  sc,
  net = net,
  lm_params = params_lm_activity(stat = "coef"),
  .verbose = FALSE
))

## HotSpot ---------------------------------------------------------------------

hs_params <- params_sc_hotspot(knn = list(k = 15L, knn_method = "exhaustive"))
ref$hs_genes <- genes[1:40]
ref$hs_autocor <- dt(hotspot_autocor_sc(
  sc,
  hotspot_params = hs_params,
  genes_to_take = ref$hs_genes,
  .verbose = FALSE
))
ref$hs_cells <- get_cell_names(sc, filtered = TRUE)[seq(1L, 580L, by = 2L)]
ref$hs_autocor_sub <- dt(hotspot_autocor_sc(
  sc,
  hotspot_params = params_sc_hotspot(
    model = "normal",
    knn = list(k = 10L, knn_method = "exhaustive")
  ),
  no_embd_to_use = 5L,
  cells_to_take = ref$hs_cells,
  genes_to_take = ref$hs_genes,
  .verbose = FALSE
))
hs <- hotspot_gene_cor_sc(
  sc,
  hotspot_params = hs_params,
  genes_to_take = genes[1:30],
  .verbose = FALSE
)
hs <- generate_hotspot_membership(hs, fdr_threshold = 0.05, min_size = 3L)
ref$hs_pair <- list(
  z = mat(hs$z),
  cor = mat(hs$cor),
  member = hs$module_memership$cluster_member
)

## SCENIC ----------------------------------------------------------------------

tfs <- genes[c(1, 2, 11, 12, 21, 22, 50, 60, 70, 80)]
ref$tfs <- tfs
scenic_p <- params_scenic(min_counts = 700L, min_cells = 0.47)
ref$scenic_filter <- scenic_gene_filter_sc(
  sc,
  scenic_params = scenic_p,
  .verbose = FALSE
)
learners <- list(
  rf = params_scenic(
    min_counts = 700L,
    min_cells = 0.47,
    gene_batch_strategy = "random",
    learner_params = list(n_trees = 20L, min_samples_leaf = 10L)
  ),
  et = params_scenic(
    min_counts = 700L,
    min_cells = 0.47,
    learner_type = "extratrees",
    gene_batch_strategy = "random",
    learner_params = list(n_trees = 20L, min_samples_leaf = 10L)
  ),
  gbm = params_scenic(
    min_counts = 700L,
    min_cells = 0.47,
    learner_type = "grnboost2",
    learner_params = list(n_trees_max = 50L, min_samples_leaf = 10L)
  )
)
grn_report <- function(g) {
  list(imp = mat(g$importance_matrix), genes = g$gene_ids, tfs = g$tf_ids)
}
grns <- lapply(learners, \(p) {
  scenic_grn_sc(sc, tf_ids = tfs, scenic_params = p, .verbose = FALSE)
})
ref$grn <- lapply(grns, grn_report)
ref$grn_streaming <- grn_report(scenic_grn_sc(
  sc,
  tf_ids = tfs,
  scenic_params = learners$rf,
  streaming = TRUE,
  .verbose = FALSE
))

grn <- grns$rf
links_thr <- get_tf_to_gene(identify_tf_to_genes(grn, n_sd = 1, .verbose = FALSE))
ref$links_threshold <- dt(links_thr)
ref$links_top_k <- dt(get_tf_to_gene(identify_tf_to_genes(
  grn,
  method = "top_k",
  k_tfs = 2L,
  k_genes = 5L,
  min_importance = 0.01,
  .verbose = FALSE
)))
grn <- identify_tf_to_genes(grn, n_sd = 1, .verbose = FALSE)
grn <- tf_to_genes_correlations(grn, sc, mode = "both", .verbose = FALSE)
ref$links_cor <- dt(get_tf_to_gene(grn))

# synthetic motif ranking, as inst/tinytest/test_sc_gs_activity.R: each motif
# ranks its five target genes on top
set.seed(42L)
n_motifs <- 6L
rankings <- matrix(0L, nrow = length(genes), ncol = n_motifs)
rownames(rankings) <- genes
colnames(rankings) <- sprintf("motif_%i", seq_len(n_motifs))
for (j in seq_len(n_motifs)) {
  targets <- ((j - 1L) * 5L + 1L):(j * 5L)
  rest <- setdiff(seq_along(genes), targets)
  rankings[targets, j] <- sample(seq_along(targets))
  rankings[rest, j] <- sample((length(targets) + 1L):length(genes), length(rest))
}
annot <- data.table(
  motif = sprintf("motif_%i", c(1:6, 1L, 3L)),
  TF = tfs[c(1:6, 7L, 8L)],
  annotationSource = c(
    rep("directAnnotation", 6L),
    "inferredBy_Orthology",
    "inferredBy_MotifSimilarity"
  )
)
ref$rankings <- unname(rankings)
ref$motifs <- colnames(rankings)
ref$annot <- dt(annot)
cis_params <- params_cistarget(nes_threshold = 1, auc_threshold = 0.2)
ref$cistarget <- dt(run_cistarget(
  gs_list = list(set_a = genes[1:10], set_b = genes[11:20], set_c = genes[1:30]),
  rankings = rankings,
  annot_data = annot,
  cis_target_params = cis_params,
  .verbose = FALSE
))
grn <- tf_to_genes_motif_enrichment(
  grn,
  motif_rankings = rankings,
  annot_data = annot,
  cis_target_params = cis_params,
  only_high_conf_tf = FALSE,
  .verbose = FALSE
)
ref$motif_links <- dt(get_tf_to_gene(grn))
ref$motif_cis <- dt(get_cistarget_res(grn))
regulons <- build_regulons(grn, min_genes = 2L, mode = "both", .verbose = FALSE)
ref$regulons <- regulons
ref$regulons_no_le <- build_regulons(
  grn,
  use_leading_edge = FALSE,
  min_genes = 3L,
  .verbose = FALSE
)

auc_reg <- aucell_sc(sc, gs_list = gs, .verbose = FALSE)
bin <- binarise_regulon_activity(auc_reg, .verbose = FALSE)
ref$binarise <- list(
  binary = unname(bin$binary * 1L),
  thresholds = bin$thresholds$threshold,
  bimodal = bin$thresholds$bimodal,
  n_cells_on = bin$thresholds$n_cells_on
)

## meta cells ------------------------------------------------------------------

mc <- generate_bt_meta_cells_sc(
  sc,
  params_sc_bt_metacells(target_no_metacells = 60L, knn = list(k = 10L)),
  .verbose = FALSE
)
ref$mc_ids <- get_sc_obs(mc)$meta_cell_id
mc <- find_hvg_sc(mc, hvg_no = 40L, .verbose = FALSE)
mc <- calculate_pca_sc(mc, no_pcs = 8L, .verbose = FALSE)
mc <- find_neighbours_sc(
  mc,
  neighbours_params = params_sc_neighbours(
    knn = list(k = 8L, knn_method = "exhaustive")
  ),
  .verbose = FALSE
)
ref$mc_pca <- mat(get_embedding(mc, "pca"))
mknn <- get_knn_obj(mc)
ref$mc_knn <- list(indices = mknn$indices, dist = mknn$dist, metric = mknn$dist_metric)

ref$mc_aucell <- mat(aucell_sc(mc, gs_list = gs, .verbose = FALSE))
ref$mc_vision <- mat(vision_sc(mc, gs_list = vgs, .verbose = FALSE))
ref$mc_vision_autocor <- autocor_report(vision_w_autocor_sc(
  mc,
  gs_list = vgs,
  embd_to_use = "pca",
  vision_params = params_sc_vision(
    n_perm = 50L,
    knn = list(k = 8L, knn_method = "exhaustive")
  ),
  .verbose = FALSE
))
ref$mc_ulm <- lm_report(lm_activity_sc(
  mc,
  net = net,
  lm_params = params_lm_activity(p_values = TRUE),
  .verbose = FALSE
))
mc_hs <- params_sc_hotspot(knn = list(k = 8L, knn_method = "exhaustive"))
ref$mc_hs_autocor <- dt(hotspot_autocor_sc(
  mc,
  hotspot_params = mc_hs,
  genes_to_take = ref$hs_genes,
  .verbose = FALSE
))
mhs <- hotspot_gene_cor_sc(
  mc,
  hotspot_params = mc_hs,
  genes_to_take = genes[1:30],
  .verbose = FALSE
)
ref$mc_hs_pair <- list(z = mat(mhs$z), cor = mat(mhs$cor))
mc_scenic_p <- params_scenic(
  min_counts = 1100L,
  min_cells = 0.1,
  gene_batch_strategy = "random",
  learner_params = list(n_trees = 20L)
)
ref$mc_scenic_filter <- scenic_gene_filter_sc(
  mc,
  scenic_params = mc_scenic_p,
  .verbose = FALSE
)
mgrn <- scenic_grn_sc(mc, tf_ids = tfs, scenic_params = mc_scenic_p, .verbose = FALSE)
ref$mc_grn <- grn_report(mgrn)
mgrn <- identify_tf_to_genes(mgrn, n_sd = 1, .verbose = FALSE)
mgrn <- tf_to_genes_correlations(mgrn, mc, mode = "both", .verbose = FALSE)
ref$mc_links_cor <- dt(get_tf_to_gene(mgrn))

jsonlite::write_json(
  ref,
  args[[2]],
  auto_unbox = TRUE,
  digits = NA,
  na = "null",
  null = "null",
  matrix = "rowmajor"
)
