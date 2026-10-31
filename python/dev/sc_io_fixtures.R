# R side of the ingestion parity tests (tests/test_io_r.py).
#
#   Rscript dev/sc_io_fixtures.R write <root> <reference.json>
#     Writes small h5ad, mtx and 10x h5 inputs with the R writers, ingests
#     each with the R loaders into <root>/r_<case>, and writes what Python has
#     to reproduce (dims, ids, obs / var tables and types, a few cells of
#     counts) as JSON, together with the synthetic generators' output.
#
#   Rscript dev/sc_io_fixtures.R read <tasks.json> <out.json>
#     For each task either opens an experiment directory with
#     load_existing() or ingests a file Python wrote, and reports it the same
#     way.
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
    null = "null"
  )
}

qc <- params_sc_min_quality(
  min_unique_genes = 5L,
  min_lib_size = 25L,
  min_cells = 5L
)

# the cells of a report: a spread over the kept cells
probe_cells <- function(n) unique(pmin(c(1L, 2L, 10L, 50L, n), n))

column_values <- function(dt) {
  lapply(as.list(dt), function(x) if (is.factor(x)) as.character(x) else x)
}

report <- function(sc) {
  con <- DBI::dbConnect(
    duckdb::duckdb(shared_home = FALSE),
    file.path(sc@dir_data, "sc_duckdb.db"),
    read_only = TRUE
  )
  obs_types <- DBI::dbGetQuery(con, "DESCRIBE obs")
  var_types <- DBI::dbGetQuery(con, "DESCRIBE var")
  DBI::dbDisconnect(con, shutdown = TRUE)

  obs <- get_sc_obs(sc)
  var <- get_sc_var(sc)
  probe <- probe_cells(nrow(obs))
  raw <- as.matrix(sc[probe, , assay = "raw"])
  norm <- as.matrix(sc[probe, , assay = "norm"])

  list(
    dims = as.integer(dim(sc)),
    obs_types = setNames(as.list(obs_types$column_type), obs_types$column_name),
    var_types = setNames(as.list(var_types$column_type), var_types$column_name),
    obs = column_values(obs),
    var = column_values(var),
    probe = probe,
    raw = unname(split(raw, row(raw))),
    norm = unname(split(norm, row(norm)))
  )
}

new_dir <- function(root, name) {
  d <- file.path(root, name)
  unlink(d, recursive = TRUE)
  dir.create(d, recursive = TRUE)
  d
}

csr_list <- function(m) {
  list(
    indptr = as.integer(m@p),
    indices = as.integer(m@j),
    data = as.numeric(m@x),
    shape = as.integer(dim(m))
  )
}

if (mode == "write") {
  root <- args[[2]]
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  ref <- list()
  files <- list()

  data <- generate_single_cell_test_data(
    syn_data_params = params_sc_synthetic_data(
      n_cells = 300L,
      n_genes = 60L,
      n_samples = 3L,
      sample_bias = "slightly_uneven"
    ),
    seed = 123L
  )
  data_b <- generate_single_cell_test_data(
    syn_data_params = params_sc_synthetic_data(n_cells = 200L, n_genes = 60L),
    seed = 7L
  )

  # synthetic generators, for parity of the Python wrappers
  adt <- generate_single_cell_test_data_adt(
    params_sc_synthetic_data_adt(n_cells = 300L),
    seed = 123L
  )
  sweep <- generate_cellsweep_test_data(
    params_sc_synthetic_cellsweep(n_real = 60L, n_empty = 100L, n_genes = 60L),
    seed = 3L
  )
  dlg <- generate_dialogue_test_data(
    params_sc_synthetic_dialogue(
      n_samples = 6L,
      cells_per_sample = 5L,
      n_genes = 40L
    ),
    seed = 5L
  )
  ref$synthetic <- list(
    rna = c(
      csr_list(data$counts),
      list(obs = column_values(data$obs), var = column_values(data$var))
    ),
    adt = list(
      counts = unname(split(adt$counts, row(adt$counts))),
      obs = column_values(adt$obs),
      var = column_values(adt$var)
    ),
    cellsweep = c(
      csr_list(sweep$counts),
      list(
        obs = column_values(sweep$obs),
        ambient_true = unname(sweep$ambient_true),
        celltype_profiles_true = unname(split(
          sweep$celltype_profiles_true,
          row(sweep$celltype_profiles_true)
        ))
      )
    ),
    dialogue = c(
      csr_list(dlg$counts),
      list(
        obs = column_values(dlg$obs),
        latent = as.list(dlg$latent),
        planted = dlg$planted,
        features_ct1 = unname(split(dlg$features[[1]], row(dlg$features[[1]]))),
        features_ct1_cells = rownames(dlg$features[[1]])
      )
    )
  )

  # h5ad, CSR and CSC
  files$h5ad_csr <- file.path(root, "csr.h5ad")
  write_h5ad_sc(files$h5ad_csr, data$counts, data$obs, data$var, .verbose = FALSE)
  files$h5ad_csc <- file.path(root, "csc.h5ad")
  write_h5ad_sc(
    files$h5ad_csc,
    as(data$counts, "CsparseMatrix"),
    data$obs,
    data$var,
    .verbose = FALSE
  )
  files$h5ad_dense <- file.path(root, "dense.h5ad")
  write_h5ad_sc_dense(
    files$h5ad_dense,
    data$counts,
    data$obs,
    data$var,
    .verbose = FALSE
  )
  files$h5ad_b <- file.path(root, "b.h5ad")
  write_h5ad_sc(files$h5ad_b, data_b$counts, data_b$obs, data_b$var, .verbose = FALSE)

  # normalised only
  lib_size <- Matrix::rowSums(data$counts)
  norm_counts <- data$counts
  norm_counts@x <- log1p(norm_counts@x / rep(lib_size, diff(norm_counts@p)) * 1e4)
  files$h5ad_norm <- file.path(root, "norm.h5ad")
  write_h5ad_sc(
    files$h5ad_norm,
    norm_counts,
    data.table::copy(data$obs)[, total_counts := lib_size],
    data$var,
    .verbose = FALSE
  )

  # mtx trios, cell- and gene-major
  files$mtx_a <- file.path(root, "mtx_a")
  dir.create(files$mtx_a)
  write_cellranger_output(
    files$mtx_a,
    data$counts,
    data$obs,
    data$var,
    rows = "cells",
    format_type = "csv",
    .verbose = FALSE
  )
  files$mtx_b <- file.path(root, "mtx_b")
  dir.create(files$mtx_b)
  write_cellranger_output(
    files$mtx_b,
    data_b$counts,
    data_b$obs,
    data_b$var,
    rows = "cells",
    format_type = "tsv",
    .verbose = FALSE
  )

  # 10x h5: v3 with an ADT block, and v2
  features <- data.table(
    id = c(data$var$gene_id, colnames(adt$counts)),
    name = c(data$var$ensembl_id, colnames(adt$counts)),
    feature_type = rep(
      c("Gene Expression", "Antibody Capture"),
      c(ncol(data$counts), ncol(adt$counts))
    )
  )
  files$tenx_v3 <- file.path(root, "v3.h5")
  write_tenx_h5_sc(
    files$tenx_v3,
    cbind(data$counts, as(adt$counts, "RsparseMatrix")),
    data$obs$cell_id,
    features
  )
  files$tenx_v2 <- file.path(root, "v2.h5")
  write_tenx_h5_sc(
    files$tenx_v2,
    data_b$counts,
    data_b$obs$cell_id,
    data.table(id = data_b$var$gene_id, name = data_b$var$ensembl_id),
    version = "v2"
  )

  ref$h5ad_dims <- list(
    csr = get_h5ad_dimensions(files$h5ad_csr),
    csc = get_h5ad_dimensions(files$h5ad_csc),
    dense = get_h5ad_dimensions(files$h5ad_dense)
  )
  ref$raw_slot <- detect_raw_count_slot(files$h5ad_csr)
  ref["raw_slot_norm"] <- list(detect_raw_count_slot(files$h5ad_norm))
  ref$tenx_meta <- bixverse:::get_tenx_h5_metadata(files$tenx_v3)

  load_r <- function(name, loader) {
    sc <- suppressWarnings(loader(SingleCells(dir_data = new_dir(root, name))))
    report(sc)
  }

  ref$r_data <- load_r("r_data", function(o) {
    load_r_data(o, data$counts, data$obs, data$var, sc_qc_param = qc, .verbose = FALSE)
  })
  ref$h5ad_csr <- load_r("h5ad_csr", function(o) {
    load_h5ad(o, files$h5ad_csr, sc_qc_param = qc, .verbose = FALSE)
  })
  ref$h5ad_csc <- load_r("h5ad_csc", function(o) {
    load_h5ad(o, files$h5ad_csc, sc_qc_param = qc, h5ad_streaming = FALSE, .verbose = FALSE)
  })
  ref$h5ad_dense <- load_r("h5ad_dense", function(o) {
    load_h5ad(o, files$h5ad_dense, sc_qc_param = qc, .verbose = FALSE)
  })
  ref$h5ad_norm <- load_r("h5ad_norm", function(o) {
    load_h5ad_norm(
      o,
      files$h5ad_norm,
      obs_lib_size_col = "total_counts",
      target_size = 1e4,
      sc_qc_param = qc,
      .verbose = FALSE
    )
  })
  ref$mtx <- load_r("mtx", function(o) {
    load_mtx(
      o,
      sc_mtx_io_param = params_sc_mtx_io(
        path_mtx = file.path(files$mtx_a, "matrix.mtx"),
        path_obs = file.path(files$mtx_a, "barcodes.csv"),
        path_var = file.path(files$mtx_a, "features.csv"),
        cells_as_rows = TRUE,
        has_hdr = TRUE
      ),
      sc_qc_param = qc,
      .verbose = FALSE
    )
  })
  ref$tenx_v3 <- load_r("tenx_v3", function(o) {
    load_tenx_h5(o, files$tenx_v3, sc_qc_param = qc, .verbose = FALSE)
  })
  ref$tenx_v2 <- load_r("tenx_v2", function(o) {
    load_tenx_h5(o, files$tenx_v2, sc_qc_param = qc, .verbose = FALSE)
  })

  scan_h5ad <- prescan_h5ad_files(
    c(a = files$h5ad_csr, b = files$h5ad_b),
    .verbose = FALSE
  )
  ref$prescan_h5ad <- list(
    universe = scan_h5ad$universe,
    mapping_b = scan_h5ad$file_tasks$b$gene_local_to_universe
  )
  ref$multi_h5ad <- load_r("multi_h5ad", function(o) {
    load_multi_h5ad(o, scan_h5ad, sc_qc_param = qc, .verbose = FALSE)
  })

  scan_mtx <- prescan_mtx_dirs(
    c(files$mtx_a, files$mtx_b),
    exp_ids = c("a", "b"),
    cells_as_rows = TRUE,
    has_hdr = TRUE,
    .verbose = FALSE
  )
  ref$multi_mtx <- load_r("multi_mtx", function(o) {
    load_multi_mtx(o, scan_mtx, sc_qc_param = qc, .verbose = FALSE)
  })

  scan_tenx <- prescan_tenx_h5_files(
    c(a = files$tenx_v3, b = files$tenx_v2),
    .verbose = FALSE
  )
  ref$prescan_tenx <- list(
    universe = scan_tenx$universe,
    mapping_a = scan_tenx$file_tasks$a$gene_local_to_universe
  )
  ref$multi_tenx <- load_r("multi_tenx", function(o) {
    load_multi_tenx_h5(o, scan_tenx, sc_qc_param = qc, .verbose = FALSE)
  })

  # merge two R experiments, one of them filtered
  sc_a <- load_existing(SingleCells(dir_data = file.path(root, "h5ad_csr")), .verbose = FALSE)
  obs_a <- get_sc_obs(sc_a)
  sc_a <- set_cells_to_keep(sc_a, as.integer(obs_a$cell_idx[obs_a$cell_grp != "cell_type_2"]))
  sc_b <- load_existing(SingleCells(dir_data = file.path(root, "tenx_v2")), .verbose = FALSE)
  ref$merge <- load_r("merge", function(o) {
    merge_sc_experiments(o, list(sc_a, sc_b), exp_ids = c("a", "b"), .verbose = FALSE)
  })

  ref$files <- files
  write_json(ref, args[[3]])
} else if (mode == "read") {
  tasks <- jsonlite::read_json(args[[2]])
  out <- list()
  for (name in names(tasks)) {
    t <- tasks[[name]]
    sc <- switch(
      t$kind,
      existing = suppressMessages(load_existing(SingleCells(dir_data = t$dir))),
      h5ad = load_h5ad(
        SingleCells(dir_data = new_dir(t$root, name)),
        t$path,
        sc_qc_param = qc,
        .verbose = FALSE
      ),
      tenx = load_tenx_h5(
        SingleCells(dir_data = new_dir(t$root, name)),
        t$path,
        sc_qc_param = qc,
        .verbose = FALSE
      ),
      stop(sprintf("unknown task kind '%s'", t$kind))
    )
    out[[name]] <- report(sc)
  }
  write_json(out, args[[3]])
} else {
  stop(sprintf("unknown mode '%s'", mode))
}
