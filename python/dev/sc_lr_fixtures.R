# R side of the ligand-receptor and DIALOGUE parity tests (tests/test_lr_r.py).
#
#   Rscript dev/sc_lr_fixtures.R <root> <reference.json>
#
# Ligand-receptor: loads seeded synthetic data into <root>/base (left
# untouched for Python to copy) and <root>/r_lr and drops the first cells on
# r_lr. Builds seeded signalling and gene regulatory networks over the store's
# genes (weights rounded to three decimals, so they cross JSON exactly), runs
# the influence in two parameter sets, the ligand activities, the per-cluster
# expression (with and without a condition), the prioritisation in four
# weightings, and CellPhoneDB on a synthetic database written as the
# `cellphonedb.zip` the downloader would fetch, in all three methods and with
# restricted pairs.
#
# DIALOGUE: loads the seeded DIALOGUE fixture into <root>/dlg_base (untouched)
# and <root>/r_dlg and runs `dialogue_sc` with the quality column, with the
# library size and the HVGs, on a `SingleCellsSubset`, and on bootstrapped meta
# cells. The meta cell PCA and kNN, labels and the averaged features are handed
# over; the doubles go as little-endian binaries (`<root>/*.f64`), not JSON.
# Meta cells built across samples blur the per-sample signal, so their ANOVA
# filter is switched off (`p_anova = 1`).
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
dt <- function(d) {
  lapply(as.list(as.data.frame(d)), function(x) {
    if (is.factor(x)) as.character(x) else x
  })
}
write_f64 <- function(x, name) {
  writeBin(as.double(x), file.path(root, name), size = 8L, endian = "little")
}

## ligand-receptor data ---------------------------------------------------------

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
sc <- load("r_lr")

ref <- list(base_dir = file.path(root, "base"), n_drop = 20L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)
genes <- get_gene_names(sc)

## networks --------------------------------------------------------------------

set.seed(42L)
ligands <- genes[1:10]
receptors <- genes[11:20]
tfs <- genes[21:30]
targets <- genes[31:120]
sig <- sprintf("SIG_%02d", 1:15)
w <- function(n) round(runif(n, 0.1, 1), 3)

lr_network <- data.table(
  ligand = c(ligands, ligands[1:3]),
  receptor = c(receptors, receptors[c(4, 7, 9)])
)
r_sig <- data.table(from = sample(receptors, 30, TRUE), to = sample(sig, 30, TRUE))
sig_tf <- data.table(from = sample(sig, 30, TRUE), to = sample(tfs, 30, TRUE))
ppi <- unique(rbind(
  lr_network[, .(from = ligand, to = receptor)],
  r_sig,
  sig_tf
))
ppi[, weight := w(.N)]
grn <- unique(data.table(
  from = sample(tfs, 250, TRUE),
  to = sample(c(targets, tfs), 250, TRUE)
))
grn[, weight := w(.N)]
ref$ppi <- dt(ppi)
ref$grn <- dt(grn)

seeds <- c(
  stats::setNames(as.list(ligands[1:8]), ligands[1:8]),
  list(complex = ligands[9:10])
)
ref$seeds <- seeds

ref$influence <- list()
lt_params <- list(
  default = params_ligand_target(),
  tuned = params_ligand_target(
    ltf_cutoff = 0,
    lr_sig_hub = 0.5,
    gr_hub = 0.5,
    secondary_targets = TRUE,
    topology_correction = TRUE
  )
)
infs <- list()
for (nm in names(lt_params)) {
  infs[[nm]] <- generate_ligand_target_influence(
    ligand_seeds = seeds,
    ppi_network = ppi,
    grn_network = grn,
    params = lt_params[[nm]]
  )
  ref$influence[[nm]] <- list(
    m = mat(get_influence(infs[[nm]])),
    rows = rownames(get_influence(infs[[nm]])),
    cols = colnames(get_influence(infs[[nm]]))
  )
}
unnamed <- generate_ligand_target_influence(
  ligand_seeds = unname(seeds[c(1, 9)]),
  ppi_network = ppi,
  grn_network = grn
)
ref$unnamed_rows <- rownames(get_influence(unnamed))

## ligand activities -----------------------------------------------------------

gene_sets <- list(degs = targets[1:20], other = c(targets[41:60], "NOTAGENE"))
background <- c(targets, tfs[1:5], "NOTAGENE", targets[1])
ref$gene_sets <- gene_sets
ref$background <- background
activities <- list()
for (nm in names(infs)) {
  activities[[nm]] <- ligand_activity_scores(
    infs[[nm]],
    gene_sets = gene_sets,
    background = background
  )
}
ref$activity <- lapply(activities, dt)
ref$activity_single <- dt(ligand_activity_scores(infs$default, targets[1:20]))

## expression info -------------------------------------------------------------

ref$expr_info <- dt(suppressWarnings(compute_expression_info_sc(
  sc,
  celltype_colname = "cell_grp",
  genes = c(ligands, receptors, "NOTAGENE")
)))
ref$expr_info_cond <- dt(compute_expression_info_sc(
  sc,
  celltype_colname = "cell_grp",
  genes = c(ligands, receptors),
  condition_colname = "batch_index",
  condition_oi = "1"
))

## prioritisation --------------------------------------------------------------

cell_types <- sort(unique(data$obs$cell_grp))
celltype_de <- CJ(cluster_id = cell_types, gene = c(ligands, receptors))
celltype_de[, lfc := round(rnorm(.N), 3)]
celltype_de[, pval := round(runif(.N), 4)]
celltype_de[cluster_id == "cell_type_1" & gene == ligands[1], pval := 0]
celltype_de[cluster_id == "cell_type_2" & gene == receptors[2], lfc := 0]
condition_de <- data.table(gene = c(ligands, receptors))
condition_de[, lfc := round(rnorm(.N), 3)]
condition_de[, pval := round(runif(.N), 4)]
ref$celltype_de <- dt(celltype_de)
ref$condition_de <- dt(condition_de)
ref$lr_network <- dt(lr_network)

expression_info <- compute_expression_info_sc(
  sc,
  celltype_colname = "cell_grp",
  genes = c(ligands, receptors)
)
la <- activities$default[gene_set == "degs", .(ligand, aupr_corrected)]
senders <- c("cell_type_1", "cell_type_2")
receivers <- c("cell_type_3", "cell_type_1")

prio <- function(...) {
  dt(prioritise_interactions(
    celltype_de = celltype_de,
    expression_info = expression_info,
    ligand_activities = la,
    lr_network = lr_network,
    senders_oi = senders,
    receivers_oi = receivers,
    ...
  ))
}
weights_partial <- c(
  de_ligand = 0,
  de_receptor = 1,
  activity_scaled = 2,
  exprs_ligand = 1,
  exprs_receptor = 0,
  ligand_condition_specificity = 0.5,
  receptor_condition_specificity = 0
)
weights_ties <- c(
  de_ligand = 0,
  de_receptor = 0,
  activity_scaled = 1,
  exprs_ligand = 0,
  exprs_receptor = 0,
  ligand_condition_specificity = 0,
  receptor_condition_specificity = 0
)
ref$prio <- list(
  one_condition = prio(scenario = "one_condition"),
  case_control = prio(condition_de = condition_de, scenario = "case_control"),
  partial = prio(condition_de = condition_de, weights = weights_partial),
  ties = prio(weights = weights_ties)
)
ref$weights_partial <- as.list(weights_partial)
ref$weights_ties <- as.list(weights_ties)

## cellphonedb -----------------------------------------------------------------

cpdb_genes <- c(genes[1:40], "NOTAGENE")
gene_table <- data.table(
  id_gene = seq_along(cpdb_genes),
  protein_id = seq_along(cpdb_genes),
  gene_name = cpdb_genes
)
# an alternative haplotype repeating a symbol, and a second protein per gene
gene_table <- rbind(
  gene_table,
  data.table(id_gene = 100L, protein_id = 5L, gene_name = cpdb_genes[5])
)
protein_table <- data.table(
  id_protein = seq_along(cpdb_genes),
  protein_multidata_id = seq_along(cpdb_genes)
)
complexes <- list(
  cpx_a = c(12L, 14L),
  cpx_b = c(3L, 30L, 22L),
  cpx_c = c(25L, 41L)
)
complex_ids <- 200L + seq_along(complexes)
complex_composition_table <- rbindlist(lapply(seq_along(complexes), \(i) {
  data.table(
    complex_multidata_id = complex_ids[i],
    protein_multidata_id = complexes[[i]]
  )
}))
multidata_table <- data.table(
  id_multidata = c(seq_along(cpdb_genes), complex_ids),
  name = c(sprintf("UNIPROT_%02d", seq_along(cpdb_genes)), names(complexes)),
  is_complex = c(rep("False", length(cpdb_genes)), rep("True", length(complexes)))
)
n_inter <- 30L
interaction_table <- data.table(
  id_cp_interaction = sprintf("CPI-%03d", sample(n_inter)),
  multidata_1_id = c(sample(1:20, n_inter - 4L, TRUE), 201L, 3L, 41L, 202L),
  multidata_2_id = c(sample(c(11:40, 201:203), n_inter - 4L, TRUE), 5L, 203L, 7L, 201L),
  directionality = sample(c("Ligand-Receptor", "Adhesion-Adhesion"), n_inter, TRUE),
  classification = sample(c("Signaling by X", "Signaling by Y", ""), n_inter, TRUE)
)
cpdb_dir <- new_dir("cpdb")
tables <- list(
  interaction_table = interaction_table,
  multidata_table = multidata_table,
  protein_table = protein_table,
  gene_table = gene_table,
  complex_composition_table = complex_composition_table
)
csvs <- character()
for (nm in names(tables)) {
  f <- file.path(cpdb_dir, paste0(nm, ".csv"))
  fwrite(tables[[nm]], f)
  csvs <- c(csvs, f)
}
utils::zip(file.path(cpdb_dir, "cellphonedb.zip"), csvs, flags = "-jq")
unlink(csvs)
ref$cpdb_dir <- cpdb_dir

lr_db <- get_cellphonedb_db(dir = cpdb_dir, .verbose = FALSE)
ref$lr_db <- dt(lr_db)

deg_table <- data.table(
  cluster_id = c("cell_type_1", "cell_type_1", "cell_type_2", "cell_type_3"),
  gene = c(genes[1], genes[12], genes[30], "NOTAGENE")
)
ref$deg_table <- dt(deg_table)
cpdb_params <- params_sc_cellphonedb(n_perm = 200L, seed = 7L)
cpdb <- function(...) {
  dt(suppressWarnings(cellphonedb_sc(
    sc,
    celltype_colname = "cell_grp",
    lr_db = lr_db,
    params = cpdb_params,
    .verbose = FALSE,
    ...
  )))
}
ref$cpdb <- list(
  simple = cpdb(method = "simple"),
  statistical = cpdb(method = "statistical"),
  degs = cpdb(method = "degs", deg_table = deg_table),
  restricted = cpdb(
    method = "statistical",
    senders = c("cell_type_3", "cell_type_1"),
    receivers = "cell_type_2"
  )
)

## dialogue --------------------------------------------------------------------

dlg <- generate_dialogue_test_data(seed = 11L)
dlg_qc <- params_sc_min_quality(
  min_unique_genes = 10L,
  min_lib_size = 50L,
  min_cells = 10L
)
load_dlg <- function(name) {
  load_r_data(
    SingleCells(dir_data = new_dir(name)),
    counts = dlg$counts,
    obs = dlg$obs,
    var = dlg$var,
    sc_qc_param = dlg_qc,
    .verbose = FALSE
  )
}
invisible(load_dlg("dlg_base"))
dsc <- load_dlg("r_dlg")
ref$dlg_base_dir <- file.path(root, "dlg_base")

# no `seed`: bixverse-r 0.9.x reads it with `as_real()`, which drops the
# integer `params_dialogue_pmd()` hands over, so R always runs on 1234
pmd <- params_dialogue_pmd(
  k = 2L,
  n_permutations = 20L,
  abn_c = 10L,
  n_genes = 12L
)
dlg_list <- function(res) {
  sig <- res$signatures
  list(
    programmes = dt(res$programmes),
    mcp_cell_types = lapply(res$mcp_cell_types, as.list),
    scores = lapply(res$scores, \(m) list(m = mat(m), rows = rownames(m))),
    cca_scores = lapply(res$cca_scores, \(m) list(m = mat(m), rows = rownames(m))),
    refit_fidelity = mat(res$refit_fidelity),
    ws = lapply(res$ws, \(m) list(m = mat(m), rows = rownames(m))),
    kept_features = lapply(res$kept_features, as.list),
    verdicts = dt(res$verdicts),
    signatures = dt(sig),
    shared_samples = as.list(res$shared_samples),
    cell_types = as.list(res$cell_types),
    source_class = res$source_class
  )
}

ref$dialogue <- list()
ref$dialogue$quality <- dlg_list(dialogue_sc(
  dsc,
  cell_type_col = "cell_grp",
  sample_col = "sample_id",
  features = dlg$features,
  quality_col = "cell_quality",
  gene_ids = dlg$var$gene_id,
  pmd_params = pmd,
  .verbose = FALSE
))

dsc <- find_hvg_sc(dsc, hvg_no = 150L, .verbose = FALSE)
ref$dlg_hvg <- as.integer(get_hvg(dsc))
ref$dialogue$lib_size <- dlg_list(dialogue_sc(
  dsc,
  cell_type_col = "cell_grp",
  sample_col = "sample_id",
  features = dlg$features,
  pmd_params = pmd,
  hlm_params = params_dialogue_hlm(satterthwaite = FALSE),
  .verbose = FALSE
))

### subset ----------------------------------------------------------------------

split_samples <- sort(unique(dlg$obs$sample_id))[1:11]
ref$dlg_split_samples <- split_samples
dsc[["dlg_split"]] <- ifelse(
  unlist(dsc[["sample_id"]]) %in% split_samples,
  "in",
  "out"
)
sub <- SingleCellsSubset(dsc, grouping_column = "dlg_split", group = "in")
ref$dialogue$subset <- dlg_list(dialogue_sc(
  sub,
  cell_type_col = "cell_grp",
  sample_col = "sample_id",
  features = dlg$features,
  quality_col = "cell_quality",
  gene_ids = dlg$var$gene_id,
  pmd_params = pmd,
  refine_params = params_dialogue_refine(support_p = 0.2),
  .verbose = FALSE
))

### meta cells ------------------------------------------------------------------

dsc <- calculate_pca_sc(dsc, no_pcs = 10L, .verbose = FALSE)
dsc <- find_neighbours_sc(
  dsc,
  neighbours_params = params_sc_neighbours(
    knn = list(k = 10L, knn_method = "exhaustive")
  ),
  .verbose = FALSE
)
ref$dlg_pca <- mat(get_embedding(dsc, "pca"))
knn <- get_knn_obj(dsc)
ref$dlg_knn <- list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)

mc <- generate_bt_meta_cells_sc(
  dsc,
  sc_meta_cell_params = params_sc_bt_metacells(
    target_no_metacells = 150L,
    knn = list(k = 10L)
  ),
  .verbose = FALSE
)
mc_obs <- S7::prop(mc, "obs_table")
ref$mc_ids <- mc_obs$meta_cell_id
membership <- mc_obs$original_cell_idx
dlg_obs <- dlg$obs
mc_cell_grp <- purrr::map_chr(membership, \(i) {
  names(which.max(table(dlg_obs$cell_grp[i])))
})
mc_sample <- purrr::map_chr(membership, \(i) {
  names(which.max(table(dlg_obs$sample_id[i])))
})
mc_quality <- purrr::map_dbl(membership, \(i) mean(dlg_obs$cell_quality[i]))
mc[["cell_grp"]] <- mc_cell_grp
mc[["sample_id"]] <- mc_sample
mc[["cell_quality"]] <- mc_quality
ref$mc_cell_grp <- mc_cell_grp
ref$mc_sample <- mc_sample
write_f64(mc_quality, "mc_quality.f64")

feature_all <- do.call(rbind, dlg$features)[dlg_obs$cell_id, ]
mc_feature_all <- t(vapply(
  membership,
  \(i) colMeans(feature_all[i, , drop = FALSE]),
  numeric(ncol(feature_all))
))
rownames(mc_feature_all) <- mc_obs$meta_cell_id
colnames(mc_feature_all) <- colnames(feature_all)
ref$mc_feature_dim <- dim(mc_feature_all)
ref$mc_feature_names <- colnames(mc_feature_all)
write_f64(t(mc_feature_all), "mc_features.f64")
mc_types <- sort(unique(mc_cell_grp))
mc_features <- purrr::map(mc_types, \(ct) {
  mc_feature_all[mc_obs$meta_cell_id[mc_cell_grp == ct], , drop = FALSE]
})
names(mc_features) <- mc_types

mc_pmd <- params_dialogue_pmd(
  k = 2L,
  n_permutations = 20L,
  abn_c = 2L,
  p_anova = 1,
  n_genes = 12L
)
ref$dialogue$mc_quality <- dlg_list(dialogue_sc(
  mc,
  cell_type_col = "cell_grp",
  sample_col = "sample_id",
  features = mc_features,
  quality_col = "cell_quality",
  gene_ids = dlg$var$gene_id,
  pmd_params = mc_pmd,
  .verbose = FALSE
))
ref$dialogue$mc_raw <- dlg_list(dialogue_sc(
  mc,
  cell_type_col = "cell_grp",
  sample_col = "sample_id",
  features = mc_features,
  gene_ids = dlg$var$gene_id,
  pmd_params = mc_pmd,
  .verbose = FALSE
))

write_json(ref, args[[2]])
