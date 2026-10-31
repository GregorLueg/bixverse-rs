# R side of the trajectory and Bonsai parity tests (tests/test_traj_r.py).
#
#   Rscript dev/sc_traj_fixtures.R <root> <reference.json>
#
# Loads seeded synthetic data twice, into <root>/base (left untouched for
# Python to copy) and <root>/r_traj, and drops the first cells on r_traj. The
# synthetic generator makes cell types, not a continuous process, so the kNN
# graph is planted: a noisy Y over the kept cells (trunk and two arms), as
# R's own trajectory tests build it, and a noisy line over the subset and the
# meta cells, whose single terminal state is named. Python takes the planted kNN graphs and the bin labels, so every
# method sees the same inputs. Then: Palantir (detected and labelled terminal
# states), PAGA over the bins, gene trends on the normalised counts, on the
# MAGIC layer and fate weighted, the same chain on a subset and on
# bootstrapped meta cells, Bonsai over the cells (all genes and a candidate
# set) and the meta cells, relayouts and the Bonsai embedding. Writes what
# Python has to reproduce as JSON.
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

## planted manifolds -----------------------------------------------------------

# as inst/tinytest/test_sc_trajectory.R: genuinely noisy, so the adaptive
# bandwidths have something to adapt to
y_manifold <- function(trunk, arm, divergence = 1.4) {
  t_trunk <- (seq_len(trunk) - 1) / trunk * 5
  t_arm <- seq_len(arm) / arm * 5
  rbind(
    cbind(t_trunk, runif(trunk, -0.15, 0.15)),
    cbind(5 + t_arm, t_arm * divergence + runif(arm, -0.15, 0.15)),
    cbind(5 + t_arm, -t_arm * divergence + runif(arm, -0.15, 0.15))
  )
}

linear_manifold <- function(n) {
  t <- (seq_len(n) - 1) / (n - 1)
  cbind(t * 10, runif(n, -0.3, 0.3), runif(n, -0.3, 0.3))
}

planted_knn <- function(m, cells, k) {
  rownames(m) <- cells
  knn <- generate_sc_knn(
    data = m,
    neighbours_params = params_sc_neighbours(
      knn = list(k = k, ann_dist = "euclidean", knn_method = "exhaustive")
    ),
    seed = 42L,
    .verbose = FALSE
  )
  new_sc_knn(
    knn_data = list(
      indices = knn$indices,
      dist = knn$dist,
      dist_metric = knn$dist_metric
    ),
    used_cells = cells
  )
}

knn_report <- function(knn) {
  list(indices = knn$indices, dist = knn$dist, metric = knn$dist_metric)
}

## reports ---------------------------------------------------------------------

palantir_report <- function(res) {
  list(
    cell_id = res$pseudotime$cell_id,
    pseudotime = res$pseudotime$pseudotime,
    entropy = res$pseudotime$entropy,
    branch_probs = unname(res$branch_probs),
    branches = colnames(res$branch_probs),
    terminal_states = unname(res$terminal_states),
    waypoints = res$waypoints,
    start_cell = res$start_cell,
    multiscale = unname(res$multiscale),
    run_info = res$run_info
  )
}

paga_report <- function(res) {
  list(
    connectivities = unname(as.matrix(res$connectivities)),
    connectivities_tree = unname(as.matrix(res$connectivities_tree)),
    clusters = rownames(res$connectivities),
    sizes = res$sizes$n_cells
  )
}

trends_report <- function(res) {
  list(
    branch = as.character(res$trends$branch),
    gene = as.character(res$trends$gene),
    pseudotime = res$trends$pseudotime,
    expression = res$trends$expression,
    branches = res$branches,
    branch_cells = unname(res$branch_cells),
    n_cells = unname(res$run_info$n_cells),
    jitter_used = unname(res$run_info$jitter_used),
    resolution = res$run_info$resolution
  )
}

nodes_report <- function(nodes) {
  list(
    parent = nodes$parent - 1L,
    branch = nodes$branch,
    is_leaf = nodes$is_leaf,
    cell_id = nodes$cell_id,
    n_cells = nodes$n_cells,
    x = nodes$x,
    y = nodes$y
  )
}

bonsai_report <- function(tree) {
  list(
    nodes = nodes_report(tree$nodes),
    loglik = tree$loglik,
    steps = tree$steps$step,
    steps_loglik = tree$steps$loglik,
    stages = tree$timings$stage,
    genes_used = tree$genes_used,
    genes_dropped = tree$genes_dropped,
    cell_idx = tree$cell_idx
  )
}

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
sc <- load("r_traj")

ref <- list(base_dir = file.path(root, "base"), n_drop = 20L)
ref$keep <- get_cell_names(sc)[-seq_len(ref$n_drop)]
sc <- set_cells_to_keep(sc, ref$keep)
cells <- get_cell_names(sc, filtered = TRUE)
genes <- get_gene_names(sc)

set.seed(7L)
arm <- floor(length(cells) / 4)
trunk <- length(cells) - 2L * arm
sc <- set_knn(sc, planted_knn(y_manifold(trunk, arm), cells, 15L))
ref$knn <- knn_report(get_knn_obj(sc))

# trunk thirds and arm halves
pos_trunk <- cut(seq_len(trunk), 3L, labels = FALSE)
pos_arm <- cut(seq_len(arm), 2L, labels = FALSE)
ref$bins <- sprintf("bin_%i", c(pos_trunk, 3L + pos_arm, 5L + pos_arm))
sc[["traj_bin"]] <- ref$bins

palantir_params <- params_sc_palantir(
  knn = 24L,
  num_waypoints = 200L,
  n_eigs = 3L,
  use_early_cell_as_start = TRUE,
  knn_params = list(knn_method = "exhaustive")
)
ref$early_cell <- cells[1]

## palantir --------------------------------------------------------------------

pal <- run_palantir_sc(
  sc,
  early_cell = ref$early_cell,
  palantir_params = palantir_params,
  .verbose = FALSE
)
ref$palantir <- palantir_report(pal)

ref$labels <- sprintf("fate_%i", seq_along(pal$terminal_states))
labelled_states <- pal$terminal_states
names(labelled_states) <- ref$labels
pal_lab <- run_palantir_sc(
  sc,
  early_cell = ref$early_cell,
  terminal_states = labelled_states,
  palantir_params = palantir_params,
  .verbose = FALSE
)
ref$palantir_labelled <- palantir_report(pal_lab)
ref$palantir_labelled$state_names <- names(pal_lab$terminal_states)

## paga ------------------------------------------------------------------------

ref$paga <- paga_report(run_paga_sc(sc, cluster_col = "traj_bin", .verbose = FALSE))

## gene trends -----------------------------------------------------------------

ref$features <- genes[1:10]
trend_params <- params_sc_gene_trends(resolution = 50L)
ref$trends <- trends_report(run_gene_trends_sc(
  sc,
  palantir_res = pal_lab,
  features = ref$features,
  gene_trend_params = trend_params,
  .verbose = FALSE
))

ref$trends_weighted <- trends_report(run_gene_trends_sc(
  sc,
  palantir_res = pal_lab,
  features = ref$features,
  gene_trend_params = params_sc_gene_trends(
    resolution = 30L,
    weighting = "fate_probability",
    length_scale = 0.3
  ),
  .verbose = FALSE
))

sc <- run_magic_sc(sc, features = ref$features, .verbose = FALSE)
ref$magic <- unname(get_magic(sc)$data)
ref$trends_magic <- trends_report(run_gene_trends_sc(
  sc,
  palantir_res = pal_lab,
  use_magic = TRUE,
  gene_trend_params = trend_params,
  .verbose = FALSE
))

## subset ----------------------------------------------------------------------

ref$grouping_column <- "cell_grp"
ref$group <- "cell_type_1"
sub <- SingleCellsSubset(
  sc_object = sc,
  grouping_column = ref$grouping_column,
  group = ref$group
)
sub_cells <- get_cell_names(sub, filtered = TRUE)
set.seed(11L)
sub <- set_knn(sub, planted_knn(linear_manifold(length(sub_cells)), sub_cells, 12L))
ref$sub_knn <- knn_report(get_knn_obj(sub))
ref$sub_bins <- sprintf(
  "seg_%i",
  cut(seq_along(sub_cells), 4L, labels = FALSE)
)
sub[["seg"]] <- ref$sub_bins

sub_params <- params_sc_palantir(
  knn = 15L,
  num_waypoints = 80L,
  n_eigs = 3L,
  use_early_cell_as_start = TRUE,
  knn_params = list(knn_method = "exhaustive")
)
sub_pal <- run_palantir_sc(
  sub,
  early_cell = sub_cells[1],
  terminal_states = sub_cells[length(sub_cells)],
  palantir_params = sub_params,
  .verbose = FALSE
)
ref$sub_palantir <- palantir_report(sub_pal)
ref$sub_paga <- paga_report(run_paga_sc(sub, cluster_col = "seg", .verbose = FALSE))
ref$sub_trends <- trends_report(run_gene_trends_sc(
  sub,
  palantir_res = sub_pal,
  features = ref$features,
  gene_trend_params = params_sc_gene_trends(resolution = 25L),
  .verbose = FALSE
))

## meta cells ------------------------------------------------------------------

mc <- generate_bt_meta_cells_sc(
  sc,
  params_sc_bt_metacells(target_no_metacells = 60L, knn = list(k = 10L)),
  .verbose = FALSE
)
mc_obs <- get_sc_obs(mc)
ref$mc_ids <- mc_obs$meta_cell_id
mc_cells <- mc_obs$meta_cell_id
set.seed(13L)
mc <- set_knn(mc, planted_knn(linear_manifold(length(mc_cells)), mc_cells, 8L))
ref$mc_knn <- knn_report(get_knn_obj(mc))
ref$mc_bins <- sprintf("seg_%i", cut(seq_along(mc_cells), 3L, labels = FALSE))
mc[["seg"]] <- ref$mc_bins

mc_params <- params_sc_palantir(
  knn = 10L,
  num_waypoints = 30L,
  n_eigs = 3L,
  use_early_cell_as_start = TRUE,
  knn_params = list(knn_method = "exhaustive")
)
mc_pal <- run_palantir_sc(
  mc,
  early_cell = mc_cells[1],
  terminal_states = mc_cells[length(mc_cells)],
  palantir_params = mc_params,
  .verbose = FALSE
)
ref$mc_palantir <- palantir_report(mc_pal)
ref$mc_paga <- paga_report(run_paga_sc(mc, cluster_col = "seg", .verbose = FALSE))
# no gene trends: run_gene_trends_sc() on MetaCells calls get_cell_names(),
# which has no MetaCells method, so R errors before reaching Rust

## bonsai ----------------------------------------------------------------------

tree <- bonsai_sc(sc, .verbose = FALSE)
ref$bonsai <- bonsai_report(tree)

ref$hvg <- c(2L, 5L, 9L, seq(20L, 80L, by = 3L))
ref$bonsai_hvg <- bonsai_report(bonsai_sc(
  sc,
  hvg = ref$hvg,
  bonsai_params = params_sc_bonsai(
    start = "greedy_merge",
    nni_search = "exact",
    layout = "dendrogram",
    seed = 3L
  ),
  .verbose = FALSE
))

ref$relayout_dendrogram <- nodes_report(
  relayout_bonsai(tree, layout = "dendrogram")$nodes
)
ref$relayout_hyperbolic <- nodes_report(
  relayout_bonsai(tree, layout = "equal_daylight", hyperbolic = TRUE)$nodes
)

sc <- set_bonsai_embedding(sc, tree)
ref$bonsai_embedding <- unname(get_embedding(sc, "bonsai"))

mc_tree <- bonsai_sc(mc, .verbose = FALSE)
ref$mc_bonsai <- bonsai_report(mc_tree)

jsonlite::write_json(
  ref,
  args[[2]],
  auto_unbox = TRUE,
  digits = NA,
  na = "null",
  null = "null",
  matrix = "rowmajor"
)
