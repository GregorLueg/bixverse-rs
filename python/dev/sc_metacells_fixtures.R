# R side of the MetaCells parity tests (tests/test_metacells_r.py).
#
#   Rscript dev/sc_metacells_fixtures.R <root> <reference.json>
#
# Loads seeded synthetic data twice, into <root>/base (left untouched for
# Python to copy) and <root>/r_mc, drops the first cells on r_mc and runs HVGs,
# PCA and neighbours. Python takes R's PCA and kNN, so every generator sees
# the same inputs. Then: bootstrapped meta cells (cached kNN, a rebuilt kNN
# and a cell type restriction), SEACells, SuperCells and MetaCells2, the
# MetaCells chain on the bootstrapped ones (HVGs, PCA with and without CLR,
# a residual fit with its HVGs and PCA, neighbours, Leiden), purity, the
# diffusion map and manifold metrics, the free compactness / separation
# scores, a merge, bootstrapped meta cells on a subset and per group. Writes
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

new_dir <- function(name) {
  d <- file.path(root, name)
  unlink(d, recursive = TRUE)
  dir.create(d, recursive = TRUE)
  d
}

column_values <- function(dt) {
  lapply(as.list(dt), function(x) if (is.factor(x)) as.character(x) else x)
}

csr <- function(m) list(p = m@p, j = m@j, x = m@x, dim = dim(m))

# what a MetaCells looks like from the outside, 0-based where Python is
mc_report <- function(mc) {
  obs <- get_sc_obs(mc)
  a <- S7::prop(mc, "original_assignment")
  list(
    ids = obs$meta_cell_id,
    members = lapply(obs$original_cell_idx, function(x) x - 1L),
    n_cells = a$n_cells,
    n_metacells = a$n_metacells,
    n_unassigned = a$n_unassigned,
    raw = csr(S7::prop(mc, "data")$raw),
    norm = csr(S7::prop(mc, "data")$norm)
  )
}

data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 900L,
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
sc <- load("r_mc")

ref <- list(base_dir = file.path(root, "base"), n_drop = 40L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)

sc <- find_hvg_sc(sc, hvg_no = 100L, .verbose = FALSE)
sc <- calculate_pca_sc(sc, no_pcs = 15L, .verbose = FALSE)
sc <- find_neighbours_sc(sc, .verbose = FALSE)
ref$pca <- unname(get_pca_factors(sc))
knn <- get_knn_obj(sc)
ref$knn <- list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)

## generators ------------------------------------------------------------------

bt_params <- params_sc_bt_metacells(target_no_metacells = 40L, knn = list(k = 10L))
mc_bt <- generate_bt_meta_cells_sc(sc, bt_params, .verbose = FALSE)
ref$bt <- mc_report(mc_bt)

ref$bt_regen <- mc_report(generate_bt_meta_cells_sc(
  sc,
  bt_params,
  regenerate_knn = TRUE,
  no_embd_to_use = 10L,
  .verbose = FALSE
))

obs_full <- get_sc_obs(sc)
ref$type_cells <- obs_full[cell_grp == "cell_type_2"]$cell_id
ref$bt_restricted <- mc_report(generate_bt_meta_cells_sc(
  sc,
  bt_params,
  cells_to_use = ref$type_cells,
  .verbose = FALSE
))

mc_sea <- generate_seacells_sc(
  sc,
  seacell_params = params_sc_seacells(
    n_sea_cells = 30L,
    min_iter = 5L,
    max_iter = 20L,
    knn = list(k = 10L)
  ),
  .verbose = FALSE
)
ref$sea <- c(
  mc_report(mc_sea),
  list(
    rss = S7::prop(mc_sea, "other_data")$rss,
    archetypes = S7::prop(mc_sea, "other_data")$archetypes
  )
)

ref$super <- mc_report(generate_supercells_sc(
  sc,
  sc_supercell_params = params_sc_supercell(graining_factor = 10, knn = list(k = 10L)),
  .verbose = FALSE
))

mc2 <- generate_metacells2_sc(
  sc,
  mc2_params = params_sc_metacells2(
    target_metacell_size = 20L,
    target_metacell_umis = 20000
  ),
  dac_params = params_sc_metacells2_dac(rare_gene_modules = FALSE),
  .verbose = FALSE
)
mc2_other <- S7::prop(mc2, "other_data")
ref$mc2 <- c(
  mc_report(mc2),
  list(
    cell_info = column_values(mc2_other$cell_info),
    selected_genes = mc2_other$selected_genes
  )
)

## MetaCells chain -------------------------------------------------------------

mc <- find_hvg_sc(mc_bt, hvg_no = 50L, .verbose = FALSE)
ref$mc_hvg <- get_hvg(mc)
ref$mc_var <- column_values(get_sc_var(mc))
ref$mc_hvg_scran <- get_hvg(find_hvg_sc(
  mc_bt,
  hvg_no = 50L,
  hvg_params = params_sc_hvg(method = "scran"),
  .verbose = FALSE
))
ref$mc_hvg_data <- column_values(get_hvg_data_sc(
  mc_bt,
  cell_ids = get_sc_obs(mc_bt)$meta_cell_id[1:30],
  hvg_no = 20L,
  hvg_params = params_sc_hvg(method = "dispersion"),
  .verbose = FALSE
))

pca_report <- function(obj) {
  list(
    factors = unname(get_pca_factors(obj)),
    loadings = unname(get_pca_loadings(obj)),
    singular_values = get_pca_singular_val(obj)
  )
}
mc <- calculate_pca_sc(mc, no_pcs = 10L, .verbose = FALSE)
ref$mc_pca <- pca_report(mc)
ref$mc_pca_clr <- pca_report(calculate_pca_sc(
  mc,
  no_pcs = 10L,
  pca_params = params_sc_pca(clr = TRUE),
  .verbose = FALSE
))

mc_res <- fit_residuals_sc(mc_bt, method = "analytic_pearson", .verbose = FALSE)
mc_res <- find_hvg_sc(
  mc_res,
  hvg_no = 50L,
  hvg_params = params_sc_hvg(method = "residual"),
  .verbose = FALSE
)
ref$mc_res_hvg <- get_hvg(mc_res)
mc_res <- calculate_pca_sc(
  mc_res,
  no_pcs = 10L,
  pca_params = params_sc_pca(normalise_variance = FALSE),
  residuals = TRUE,
  .verbose = FALSE
)
ref$mc_res_pca <- pca_report(mc_res)

mc <- find_neighbours_sc(
  mc,
  neighbours_params = params_sc_neighbours(knn = list(k = 8L)),
  .verbose = FALSE
)
mc_knn <- get_knn_obj(mc)
ref$mc_knn <- list(indices = mc_knn$indices, dist = mc_knn$dist)
mc <- find_clusters_sc(mc, res = 1.0, name = "mc_leiden")
ref$mc_leiden <- get_sc_obs(mc)$mc_leiden

## metrics ---------------------------------------------------------------------

mc <- calc_meta_cell_purity(
  mc,
  original_cell_type = sc[["cell_grp"]]$cell_grp,
  add_additional_info = "top_two_labels",
  add_entropy = TRUE
)
ref$purity <- column_values(get_sc_obs(
  mc,
  cols = c("mc_purity", "mc_top_label", "mc_second_label", "mc_second_frac", "mc_entropy")
))
ref$purity_filtered <- column_values(get_meta_cell_purity(
  mc,
  original_cell_type = get_sc_obs(sc, filtered = TRUE)$cell_grp,
  add_additional_info = "top_label"
))

mc <- calc_diffusion_coordinates(mc, knn_data = knn, n_dcs = 6L, k_density = 30L, .verbose = FALSE)
mc <- calc_manifold_metrics(mc)
mc_other <- S7::prop(mc, "other_data")
ref$diffusion <- list(
  dcs = unname(mc_other$dcs),
  density_dist = mc_other$density_dist,
  regions = mc_other$regions,
  density_region = get_sc_obs(mc)$density_region,
  compactness = get_sc_obs(mc)$compactness,
  separation = get_sc_obs(mc)$separation
)

assignment <- sprintf("mc_%02i", seq_len(nrow(mc_other$dcs)) %% 13L)
assignment[seq(5L, length(assignment), by = 50L)] <- NA
ref$free_assignment <- assignment
comp <- calc_meta_cell_compactness(mc_other$dcs, assignment)
sep <- calc_meta_cell_separation(mc_other$dcs, assignment)
ref$free <- list(names = names(comp), compactness = unname(comp), separation = unname(sep))

## merge -----------------------------------------------------------------------

mc_regen <- generate_bt_meta_cells_sc(
  sc,
  bt_params,
  regenerate_knn = TRUE,
  no_embd_to_use = 10L,
  .verbose = FALSE
)
merged <- suppressWarnings(merge_meta_cells(
  list(donor_a = mc, donor_b = mc_regen),
  .verbose = FALSE
))
merged_obs <- get_sc_obs(merged)
ref$merged <- c(
  mc_report(merged),
  list(
    source_id = merged_obs$source_id,
    obs_cols = names(merged_obs),
    sources = S7::prop(merged, "other_data")$sources
  )
)

## subset and per group --------------------------------------------------------

sub <- SingleCellsSubset(sc_object = sc, grouping_column = "cell_grp", group = "cell_type_1")
sub <- find_hvg_sc(sub, hvg_no = 100L, .verbose = FALSE)
sub <- calculate_pca_sc(sub, no_pcs = 10L, .verbose = FALSE)
sub <- find_neighbours_sc(sub, .verbose = FALSE)
ref$sub_bt <- mc_report(generate_bt_meta_cells_sc(
  sub,
  params_sc_bt_metacells(target_no_metacells = 15L, knn = list(k = 10L)),
  .verbose = FALSE
))

prep <- step_hvg_sc(hvg_no = 100L, .verbose = FALSE) %>>%
  step_pca_sc(no_pcs = 10L, .verbose = FALSE) %>>%
  step_neighbours_sc(.verbose = FALSE)
per_group <- meta_cells_per_group(
  sc,
  group_col = "cell_grp",
  method = "bootstrapped",
  mc_params = list(
    sc_meta_cell_params = params_sc_bt_metacells(target_no_metacells = 10L, knn = list(k = 10L)),
    .verbose = FALSE
  ),
  pipeline = prep,
  .verbose = FALSE
)
ref$per_group <- c(
  mc_report(per_group),
  list(source_id = get_sc_obs(per_group)$source_id)
)

jsonlite::write_json(
  ref,
  args[[2]],
  auto_unbox = TRUE,
  digits = NA,
  na = "null",
  null = "null",
  matrix = "rowmajor"
)
