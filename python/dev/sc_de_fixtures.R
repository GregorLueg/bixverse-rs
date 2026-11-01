# R side of the DGE, marker, pseudobulk, NEBULA, Milo and MELD parity tests
# (tests/test_de_r.py).
#
#   Rscript dev/sc_de_fixtures.R write <root> <reference.json>
#     Generates seeded synthetic data with six samples, three control and
#     three treated. Plants a sample-level condition effect (genes 40 to 44
#     tripled in the treated samples) and an abundance shift (60% of the
#     treated cell type 1 cells dropped), then loads it twice, into
#     <root>/base (left untouched for Python to copy) and <root>/r_de. On
#     r_de it drops the first cells, runs the marker tests, the pseudobulk
#     aggregation and edgeR test, NEBULA, then HVG, PCA and the neighbours
#     for Milo and MELD. Writes what Python has to reproduce as JSON.
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

## data -------------------------------------------------------------------------

data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 1500L,
    n_genes = 200L,
    n_samples = 6L,
    sample_bias = "even"
  ),
  seed = 123L
)
trt_samples <- sprintf("sample_%i", 4:6)
data$obs[, condition := ifelse(sample_id %in% trt_samples, "trt", "ctr")]
data$obs[, age := as.numeric(sub("sample_", "", sample_id)) * 7 + 20]

# the condition effect: genes 40 to 44 (0-based) tripled in the treated cells
trt <- which(data$obs$condition == "trt")
counts <- as(data$counts, "CsparseMatrix")
counts[trt, 41:45] <- counts[trt, 41:45] * 3

# the abundance shift: most treated cell type 1 cells go
set.seed(5L)
type1_trt <- which(data$obs$condition == "trt" & data$obs$cell_grp == "cell_type_1")
drop <- sample(type1_trt, as.integer(0.6 * length(type1_trt)))
counts <- counts[-drop, , drop = FALSE]
data$counts <- as(counts, "RsparseMatrix")
data$obs <- data$obs[-drop]

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
sc <- load("r_de")

ref <- list(base_dir = file.path(root, "base"), n_drop = 30L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)
obs <- get_sc_obs(sc, filtered = TRUE)

## markers ------------------------------------------------------------------------

ref$cells_1 <- obs$cell_id[obs$cell_grp == "cell_type_1"]
ref$cells_2 <- obs$cell_id[obs$cell_grp == "cell_type_2"]
ref$markers <- find_markers_sc(
  sc,
  cells_1 = ref$cells_1,
  cells_2 = ref$cells_2,
  .verbose = FALSE
)
ref$markers_greater <- find_markers_sc(
  sc,
  cells_1 = ref$cells_1,
  cells_2 = ref$cells_2,
  alternative = "greater",
  min_prop = 0.2,
  .verbose = FALSE
)
ref$all_markers <- find_all_markers_sc(
  sc,
  column_of_interest = "cell_grp",
  .verbose = FALSE
)
ref$all_markers_twosided <- find_all_markers_sc(
  sc,
  column_of_interest = "sample_id",
  alternative = "twosided",
  min_prop = 0.1,
  .verbose = FALSE
)

specific_out <- function(res) {
  list(
    summary = res$summary,
    per_comparison = res$per_comparison
  )
}
ref$specific <- specific_out(find_specific_markers_sc(
  sc,
  column_of_interest = "cell_grp",
  .verbose = FALSE
))
ref$specific_ref <- specific_out(find_specific_markers_sc(
  sc,
  column_of_interest = "cell_grp",
  reference_group = "cell_type_2",
  alternative = "twosided",
  .verbose = FALSE
))

## pseudobulk ---------------------------------------------------------------------

cell_list <- split(obs$cell_id, obs$sample_id)
ref$pb_samples <- names(cell_list)
ref$pb_raw <- unname(get_pseudobulked_sc(sc, cell_list, .verbose = FALSE))
ref$pb_norm <- unname(get_pseudobulked_sc(
  sc,
  cell_list,
  assay = "norm",
  .verbose = FALSE
))
pb_sparse <- get_pseudobulked_sc(
  sc,
  cell_list,
  return_format = "sparse",
  .verbose = FALSE
)
ref$pb_sparse <- list(p = pb_sparse@p, j = pb_sparse@j, x = pb_sparse@x)

pb_condition <- ifelse(names(cell_list) %in% trt_samples, "trt", "ctr")
pb_design <- stats::model.matrix(~pb_condition)
ref$pb_design <- unname(pb_design)
ref$pb_dge <- pseudobulk_dge_sc(sc, cell_list, pb_design, .verbose = FALSE)
ref$pb_dge_robust <- pseudobulk_dge_sc(
  sc,
  cell_list,
  pb_design,
  edger_params = params_edger_ql(
    norm_method = "RLE",
    robust = TRUE,
    legacy = TRUE
  ),
  .verbose = FALSE
)

## NEBULA -------------------------------------------------------------------------

nebula_out <- function(res) {
  list(
    results = res$results,
    coefficients = unname(res$coefficients),
    se = unname(res$se),
    coef_names = colnames(res$coefficients),
    tested = res$params$tested
  )
}
ref$nebula_genes <- get_gene_names(sc)[1:60]
ref$nebula <- nebula_out(nebula_sc(
  sc,
  subject_col = "sample_id",
  design = ~condition,
  genes_to_use = ref$nebula_genes,
  .verbose = FALSE
))
ref$nebula_age <- nebula_out(nebula_sc(
  sc,
  subject_col = "sample_id",
  design = ~ condition + age,
  coef = "conditiontrt",
  genes_to_use = ref$nebula_genes,
  nebula_params = params_nebula(nebula_method = "hl", shrink_dispersion = FALSE),
  .verbose = FALSE
))
ref$nebula_cells <- obs$cell_id[obs$cell_grp != "cell_type_3"]
ref$nebula_contrast <- nebula_out(nebula_sc(
  sc,
  subject_col = "sample_id",
  design = ~condition,
  contrast = c(0, 1),
  genes_to_use = ref$nebula_genes,
  cells_to_use = ref$nebula_cells,
  .verbose = FALSE
))

## Milo and MELD ------------------------------------------------------------------

sc <- find_hvg_sc(sc, hvg_no = 100L, .verbose = FALSE)
sc <- calculate_pca_sc(sc, no_pcs = 15L, .verbose = FALSE)
ref$pca <- unname(get_pca_factors(sc))
sc <- find_neighbours_sc(sc, .verbose = FALSE)
ref$knn <- get_knn_mat(sc)

milo_out <- function(milo) {
  nh <- Matrix::summary(milo$nhoods)
  list(
    index_cell = milo$params$index_cell,
    nhoods_i = nh$i - 1L,
    nhoods_j = nh$j - 1L,
    sample_counts = unname(milo$sample_counts),
    samples = colnames(milo$sample_counts),
    spatial_dist = milo$spatial_dist,
    nhood_overlap = milo$nhood_overlap
  )
}
ref$milo <- list()
for (strategy in c("approximate", "bruteforce", "index")) {
  milo <- get_miloR_abundances_sc(
    sc,
    sample_id_col = "sample_id",
    miloR_params = params_sc_miloR(
      k_refine = 10L,
      refinement_strategy = strategy,
      index_type = if (strategy == "index") "exhaustive" else "nndescent"
    ),
    seed = 42L,
    .verbose = FALSE
  )
  ref$milo[[strategy]] <- milo_out(milo)
}

design_df <- data.frame(
  condition = ifelse(colnames(milo$sample_counts) %in% trt_samples, "trt", "ctr"),
  row.names = colnames(milo$sample_counts)
)
ref$design_condition <- design_df$condition
tested <- test_nhoods(milo, design = ~condition, design_df = design_df)
# a copy: add_nhoods_info() below adds its columns to this table by reference
ref$nhoods_info <- data.table::copy(get_differential_abundance_res(tested))
ref$nhoods_info_overlap <- get_differential_abundance_res(test_nhoods(
  milo,
  design = ~condition,
  design_df = design_df,
  norm_method = "logMS",
  legacy = FALSE,
  robust = FALSE,
  fdr_weighting = "graph-overlap"
))
ref$nhoods_info_none <- get_differential_abundance_res(test_nhoods(
  milo,
  design = ~condition,
  design_df = design_df,
  min_mean = 5,
  fdr_weighting = "none"
))
annotated <- add_nhoods_info(tested, cell_info = obs$cell_grp)
ref$nhoods_annotated <- get_differential_abundance_res(annotated)

meld_out <- function(res) {
  list(
    raw = unname(res$raw_scores),
    norm = unname(res$norm_scores),
    samples = colnames(res$raw_scores),
    cells = rownames(res$raw_scores)
  )
}
ref$meld <- meld_out(meld_sc(sc, sample_id_col = "sample_id", .verbose = FALSE))
ref$meld_lap <- meld_out(meld_sc(
  sc,
  sample_id_col = "condition",
  meld_params = params_meld(
    filter = "laplacian",
    lap_type = "normalised",
    beta = 20,
    normalise_indicators = FALSE
  ),
  seed = 3L,
  .verbose = FALSE
))

write_json(ref, args[[3]])
