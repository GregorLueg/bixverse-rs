# R side of the doublet detection, empty droplet and CellSweep parity tests
# (tests/test_doublets_r.py).
#
#   Rscript dev/sc_doublets_fixtures.R write <root> <reference.json>
#     Loads seeded synthetic data with two samples twice, into <root>/base
#     (left untouched for Python to copy) and <root>/r_dbl, drops the first
#     cells on r_dbl and runs Scrublet (plain and per sample), the manual
#     re-calls, Boost and scDblFinder on it. Loads seeded CellSweep data with
#     permissive QC into <root>/cs_base (untouched) and <root>/cs_r, infers
#     empty droplets from its library sizes and runs CellSweep twice, into
#     <root>/cs_out_supplied and <root>/cs_out_cutoff. Writes what Python has
#     to reproduce as JSON.
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

## doublets ---------------------------------------------------------------------

data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 1200L,
    n_genes = 300L,
    n_samples = 2L,
    sample_bias = "even"
  ),
  seed = 123L
)
# plant heterotypic doublets: the summed counts of a type 1 and a type 2 cell
set.seed(1L)
n_dbl <- 60L
grp <- data$obs$cell_grp
pa <- sample(which(grp == "cell_type_1"), n_dbl)
pb <- sample(which(grp == "cell_type_2"), n_dbl)
dbl_ids <- sprintf("doublet_%02d", seq_len(n_dbl))
dbl <- data$counts[pa, , drop = FALSE] + data$counts[pb, , drop = FALSE]
rownames(dbl) <- dbl_ids
data$counts <- as(rbind(data$counts, dbl), "RsparseMatrix")
data$obs <- rbind(
  data$obs,
  data.table(
    cell_id = dbl_ids,
    cell_grp = "doublet",
    batch_index = data$obs$batch_index[pa],
    sample_id = data$obs$sample_id[pa]
  )
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
sc <- load("r_dbl")

ref <- list(base_dir = file.path(root, "base"), n_drop = 40L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)

scrublet_out <- function(res) {
  list(
    predicted_doublets = res$predicted_doublets,
    doublet_scores_obs = res$doublet_scores_obs,
    doublet_errors_obs = res$doublet_errors_obs,
    z_scores = res$z_scores,
    threshold = as.list(res$threshold),
    detected_doublet_rate = as.list(res$detected_doublet_rate),
    detectable_doublet_fraction = as.list(res$detectable_doublet_fraction),
    overall_doublet_rate = as.list(res$overall_doublet_rate),
    doublet_scores_sim = res$doublet_scores_sim,
    cell_indices = attr(res, "cell_indices"),
    cell_groups = res$cell_groups,
    print = capture.output(print(res)),
    get_data_cols = names(get_data(res))
  )
}

scrublet_p <- params_scrublet(
  pca = list(no_pcs = 10L),
  hvg = list(min_gene_var_pctl = 0.0),
  n_bins_histogram = 20L
)
scr <- scrublet_sc(
  sc,
  scrublet_params = scrublet_p,
  return_combined_pca = TRUE,
  return_pairs = TRUE,
  .verbose = FALSE
)
ref$scrublet <- scrublet_out(scr)
ref$scrublet$pca <- unname(scr$pca)
ref$scrublet$pair_1 <- scr$pair_1
ref$scrublet$pair_2 <- scr$pair_2
ref$scrublet_manual <- scrublet_out(
  call_doublets_manual(scr, threshold = 0.3, .verbose = FALSE)
)

scr_grp <- suppressWarnings(scrublet_sc(
  sc,
  scrublet_params = scrublet_p,
  group_by = "sample_id",
  seed = 7L,
  .verbose = FALSE
))
ref$scrublet_grouped <- scrublet_out(scr_grp)
ref$scrublet_grouped_manual <- scrublet_out(
  call_doublets_manual(
    scr_grp,
    threshold = 0.25,
    for_sample = "sample_2",
    .verbose = FALSE
  )
)

boost_out <- function(res) {
  list(
    doublet = res$doublet,
    doublet_score = res$doublet_score,
    voting_avg = res$voting_avg,
    cell_indices = attr(res, "cell_indices"),
    cell_groups = res$cell_groups,
    print = capture.output(print(res)),
    get_data_cols = names(get_data(res))
  )
}
boost_p <- params_boost(
  hvg = list(min_gene_var_pctl = 0.0),
  pca = list(no_pcs = 10L),
  n_iters = 5L
)
ref$boost <- boost_out(
  doublet_detection_boost_sc(sc, boost_params = boost_p, .verbose = FALSE)
)
ref$boost_grouped <- boost_out(suppressWarnings(
  doublet_detection_boost_sc(
    sc,
    boost_params = boost_p,
    group_by = "sample_id",
    .verbose = FALSE
  )
))

scdbl_out <- function(res) {
  list(
    predicted_doublets = res$predicted_doublets,
    doublet_score = res$doublet_score,
    cxds_scores = res$cxds_scores,
    weighted = res$weighted,
    threshold = as.list(res$threshold),
    cluster_labels = res$cluster_labels,
    detected_doublet_rate = as.list(res$detected_doublet_rate),
    cell_indices = attr(res, "cell_indices"),
    cell_groups = res$cell_groups,
    features = if (is.null(res$features)) NULL else unname(res$features),
    feature_names = colnames(res$features),
    print = capture.output(print(res)),
    get_data_cols = names(get_data(res))
  )
}
scdbl_p <- params_scdblfinder(
  pca = list(no_pcs = 10L),
  n_genes = 50L,
  cxds_genes = 50L
)
ref$scdblfinder <- scdbl_out(scdblfinder_sc(
  sc,
  scdblfinder_params = scdbl_p,
  return_features = TRUE,
  .verbose = FALSE
))
ref$scdblfinder_short <- scdbl_out(scdblfinder_sc(
  sc,
  scdblfinder_params = params_scdblfinder(
    pca = list(no_pcs = 10L),
    n_genes = 50L,
    cxds_genes = 50L,
    gbm_n_trees = 3L
  ),
  .verbose = FALSE
))
ref$scdblfinder_grouped <- scdbl_out(suppressWarnings(scdblfinder_sc(
  sc,
  scdblfinder_params = scdbl_p,
  group_by = "sample_id",
  .verbose = FALSE
)))

## cellsweep --------------------------------------------------------------------

cs <- generate_cellsweep_test_data(
  syn_data_params = params_sc_synthetic_cellsweep(
    n_real = 400L,
    n_empty = 1500L,
    n_genes = 120L
  ),
  seed = 11L
)
# a second emulsion split, so the per-sample fits and their order are tested
cs$obs[, sample_two := ifelse(seq_len(.N) %% 2L == 0L, "s_b", "s_a")]
open_qc <- params_sc_min_quality(
  min_unique_genes = 0L,
  min_lib_size = 0L,
  min_cells = 0L
)
load_cs <- function(name) {
  suppressWarnings(load_r_data(
    SingleCells(dir_data = new_dir(name)),
    cs$counts,
    cs$obs,
    cs$var,
    sc_qc_param = open_qc,
    .verbose = FALSE
  ))
}
invisible(load_cs("cs_base"))
cs_sc <- load_cs("cs_r")
ref$cs_base_dir <- file.path(root, "cs_base")

# leave some annotated barcodes out of the kept cells: they must drop out
ref$cs_n_drop <- 10L
cs_keep <- get_cell_names(cs_sc)[-seq_len(ref$cs_n_drop)]
cs_sc <- set_cells_to_keep(cs_sc, cs_keep)
ref$cs_keep <- cs_keep

lib <- as.integer(get_sc_obs(cs_sc, filtered = FALSE)$lib_size)
ref$empty_knee <- rs_sc_infer_empty_droplets(
  lib,
  unclass(params_sc_empty_droplets(method = "knee"))
)
ref$empty_cutoff <- rs_sc_infer_empty_droplets(
  lib,
  unclass(params_sc_empty_droplets(method = "umi_cutoff", umi_cutoff = 400L))
)
ref$empty_expected <- rs_sc_infer_empty_droplets(
  lib,
  unclass(params_sc_empty_droplets(
    method = "expected_cells",
    expected_cells = 380L
  ))
)

out_supplied <- cellsweep_sc(
  target = SingleCells(dir_data = new_dir("cs_out_supplied")),
  input = cs_sc,
  celltype_column = "cell_grp",
  sample_column = "sample_id",
  empty_params = params_sc_empty_droplets(
    method = "supplied",
    is_empty_column = "is_empty"
  ),
  .verbose = FALSE
)
out_cutoff <- cellsweep_sc(
  target = SingleCells(dir_data = new_dir("cs_out_cutoff")),
  input = cs_sc,
  celltype_column = "cell_grp",
  sample_column = "sample_two",
  empty_params = params_sc_empty_droplets(
    method = "umi_cutoff",
    umi_cutoff = 400L
  ),
  cellsweep_params = params_sc_cellsweep(max_iter = 300L, seed = 3L),
  sc_qc_param = params_sc_min_quality(target_size = 5000),
  .verbose = FALSE
)
ref$cs_out_supplied <- file.path(root, "cs_out_supplied")
ref$cs_out_cutoff <- file.path(root, "cs_out_cutoff")

write_json(ref, args[[3]])
