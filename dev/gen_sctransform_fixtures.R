# Generates tests/sctransform_fixtures/mod.rs from sctransform 0.4.3 with
# glmGamPoi 1.20.0.
#
# The count matrix is rebuilt from a seeded Numerical Recipes LCG on both
# sides, so no float data crosses as text. Only integers and the answers do.
# The draw is u^3 scaled by a per-gene rate and a per-cell library size, which
# is heavy-tailed enough to be genuinely overdispersed while using nothing but
# multiplication and a floor, so R and Rust agree bit for bit.
#
# The fixture is deliberately small enough that both of sctransform's step-1
# subsamples are skipped: with fewer cells than `n_cells` and fewer genes than
# `n_genes`, `cells_step1` is every cell and `genes_step1` is every eligible
# gene. That removes R's RNG from the comparison entirely.
#
#   Rscript dev/gen_sctransform_fixtures.R

suppressPackageStartupMessages({
  library(Matrix)
  library(sctransform)
  library(glmGamPoi)
})

stopifnot(packageVersion("sctransform") >= "0.4.3")

LCG_SEED   <- 20260101
N_GENES    <- 300
N_CELLS    <- 400
GMEAN_EPS  <- 1
MIN_CELLS  <- 5

lcg_state <- LCG_SEED
lcg_next <- function() {
  lcg_state <<- (1664525 * lcg_state + 1013904223) %% 4294967296
  lcg_state
}
lcg_unif <- function() lcg_next() / 4294967296

# Per-cell library scale and per-gene rate, drawn first so the count loop is a
# single ordered sweep the Rust side can mirror exactly.
lib_scale  <- vapply(seq_len(N_CELLS), function(i) 2000 + 6000 * lcg_unif(), numeric(1))
gene_rate  <- vapply(seq_len(N_GENES), function(i) 10^(-5 + 3.2 * lcg_unif()), numeric(1))

counts <- matrix(0L, N_GENES, N_CELLS)
for (g in seq_len(N_GENES)) {
  for (c in seq_len(N_CELLS)) {
    u <- lcg_unif()
    counts[g, c] <- as.integer(floor(u * u * u * gene_rate[g] * lib_scale[c] * 12))
  }
}
rownames(counts) <- sprintf("g%04d", seq_len(N_GENES))
colnames(counts) <- sprintf("c%04d", seq_len(N_CELLS))
counts <- as(counts, "dgCMatrix")

vst_out <- vst(
  counts,
  vst.flavor = "v2",
  min_cells = MIN_CELLS,
  gmean_eps = GMEAN_EPS,
  return_cell_attr = TRUE,
  return_gene_attr = TRUE,
  verbosity = 0
)

# The modelled gene set, in the same order the Rust side will produce it.
genes_cell_count <- rowSums(counts >= 0.01)
modelled <- which(genes_cell_count >= MIN_CELLS)
umi <- counts[modelled, ]
genes <- rownames(umi)

log_gmean <- log10(sctransform:::row_gmean(umi, eps = GMEAN_EPS))
amean     <- rowMeans(umi)
gvar      <- sctransform:::row_var(umi)

genes_step1 <- rownames(vst_out$model_pars)
step1_pos   <- match(genes_step1, genes) - 1L   # 0-based into `modelled`

mp     <- vst_out$model_pars
mp_fit <- vst_out$model_pars_fit[genes, , drop = FALSE]

resid_var <- vst_out$gene_attr[genes, "residual_variance"]

# Corrected UMI counts. This is `correct_counts`, which recomputes the Pearson
# residual from the raw counts with no clipping and no variance floor, rather
# than `correct`, which reverses the already-clipped residual matrix. Seurat's
# SCTransform() puts `correct_counts` in the SCT `counts` slot, so that is the
# one worth matching.
corrected <- sctransform::correct_counts(vst_out, counts, verbosity = 0)[genes, ]
corrected_sum <- as.numeric(Matrix::rowSums(corrected))
corrected_nnz <- as.numeric(Matrix::rowSums(corrected > 0))

# Three full rows, spanning the abundance range, so a per-gene total that
# happens to agree cannot hide a redistribution across cells.
probe_pos <- c(1L, which.min(abs(log_gmean - median(log_gmean))), length(genes))
probe_rows <- as.matrix(corrected[probe_pos, ])

## ---- a second run with a covariate -------------------------------------------
#
# sctransform's `latent_var` beyond `log_umi`: the extra variable becomes a
# fitted coefficient in the NB GLM while the library size stays a fixed offset.
# Drawn from the same LCG after the counts, so the count matrix is unchanged and
# both sides rebuild the covariate identically.

# A percent-mitochondrial analogue: the share of each cell's counts falling in a
# fixed block of genes. Real structure, unlike a noise covariate whose
# coefficients would all be ~0 and would let a dropped term pass unnoticed.
COV_BLOCK <- 30L
cov_x <- as.numeric(100 * Matrix::colSums(counts[seq_len(COV_BLOCK), ]) /
                      Matrix::colSums(counts))
cell_attr_cov <- data.frame(
  log_umi = log10(Matrix::colSums(counts)),
  cov_x = cov_x,
  row.names = colnames(counts)
)

vst_cov <- vst(
  counts,
  vst.flavor = "v2",
  cell_attr = cell_attr_cov,
  latent_var = c("log_umi", "cov_x"),
  min_cells = MIN_CELLS,
  gmean_eps = GMEAN_EPS,
  return_cell_attr = TRUE,
  return_gene_attr = TRUE,
  verbosity = 0
)

stopifnot(identical(colnames(vst_cov$model_pars),
                    c("theta", "(Intercept)", "log_umi", "cov_x")))

cov_step1_pos <- match(rownames(vst_cov$model_pars), genes) - 1L
cov_mp        <- vst_cov$model_pars
cov_fit       <- vst_cov$model_pars_fit[genes, , drop = FALSE]
cov_resid_var <- vst_cov$gene_attr[genes, "residual_variance"]

cat(sprintf("covariate run: %d step-1 genes, %d poisson in fit\n",
            length(cov_step1_pos), sum(!is.finite(cov_fit[, "theta"]))))
cat(sprintf("cov_x range %.2f..%.2f, coefficient range: %.6f .. %.6f\n",
            min(cov_x), max(cov_x),
            min(cov_fit[, "cov_x"]), max(cov_fit[, "cov_x"])))

## ---- emit -------------------------------------------------------------------

# R prints a non-finite as "Inf"/"NaN", which is not Rust syntax.
num_vec <- function(x) {
  out <- sprintf("%.17e", x)
  out[is.infinite(x) & x > 0] <- "f64::INFINITY"
  out[is.infinite(x) & x < 0] <- "f64::NEG_INFINITY"
  out[is.nan(x)] <- "f64::NAN"
  paste(out, collapse = ",\n    ")
}
int_vec <- function(x) paste(x, collapse = ", ")

dir.create("tests/sctransform_fixtures", showWarnings = FALSE, recursive = TRUE)
con <- file("tests/sctransform_fixtures/mod.rs", "w")

writeLines(c(
  "//! Generated by `dev/gen_sctransform_fixtures.R`. Do not edit by hand.",
  "//!",
  sprintf("//! sctransform %s, glmGamPoi %s, %s.",
          packageVersion("sctransform"), packageVersion("glmGamPoi"), R.version.string),
  "//!",
  "//! The counts are rebuilt from the LCG below rather than stored, so nothing",
  "//! but integers and reference answers crosses as text.",
  "",
  "#![allow(clippy::excessive_precision)]",
  "#![allow(clippy::approx_constant)]",
  "#![allow(dead_code)]",
  "//",
  "// R emits seventeen significant digits and a generated value will now and",
  "// then land near a mathematical constant. Both lints are off for this file",
  "// only, rather than hand-editing generated numbers.",
  "",
  sprintf("/// LCG seed shared with the generator."),
  sprintf("pub const LCG_SEED: u64 = %d;", LCG_SEED),
  sprintf("/// Genes in the generated matrix."),
  sprintf("pub const N_GENES: usize = %d;", N_GENES),
  sprintf("/// Cells in the generated matrix."),
  sprintf("pub const N_CELLS: usize = %d;", N_CELLS),
  sprintf("/// `gmean_eps` the reference ran with."),
  sprintf("pub const GMEAN_EPS: f64 = %.1f;", GMEAN_EPS),
  sprintf("/// `min_cells` the reference ran with."),
  sprintf("pub const MIN_CELLS: usize = %d;", MIN_CELLS),
  sprintf("/// Median of every non-zero count across the modelled genes."),
  sprintf("pub const MEDIAN_NONZERO: f64 = %.17e;", sctransform:::get_nz_median2(umi)),
  sprintf("/// `mean(colSums(umi))` over the modelled genes."),
  sprintf("pub const MEAN_CELL_SUM: f64 = %.17e;", mean(colSums(umi))),
  "",
  "/// Store indices of the genes passing the `min_cells` filter, 0-based.",
  sprintf("pub const MODELLED: [usize; %d] = [%s];", length(modelled), int_vec(modelled - 1L)),
  "",
  "/// Positions within `MODELLED` of the step-1 genes, 0-based.",
  sprintf("pub const STEP1_POS: [usize; %d] = [%s];", length(step1_pos), int_vec(step1_pos)),
  ""
), con)

emit <- function(name, doc, x) {
  writeLines(c(
    sprintf("/// %s", doc),
    sprintf("pub const %s: [f64; %d] = [", name, length(x)),
    sprintf("    %s,", num_vec(x)),
    "];",
    ""
  ), con)
}

emit("LOG_GMEAN",  "`log10(row_gmean(umi, eps))` per modelled gene.", log_gmean)
emit("AMEAN",      "`rowMeans(umi)` per modelled gene.", amean)
emit("GENE_VAR",   "`row_var(umi)` per modelled gene, sample variance.", gvar)
emit("STEP1_THETA",     "Unregularised step-1 theta, from glmGamPoi.", mp[, "theta"])
emit("STEP1_INTERCEPT", "Unregularised step-1 intercept, from glmGamPoi.", mp[, "(Intercept)"])
emit("FIT_THETA",       "Regularised theta per modelled gene.", mp_fit[, "theta"])
emit("FIT_INTERCEPT",   "Regularised intercept per modelled gene.", mp_fit[, "(Intercept)"])
emit("RESIDUAL_VARIANCE", "`gene_attr$residual_variance` per modelled gene.", resid_var)
emit("CORRECTED_SUM", "`rowSums(correct_counts(...))` per modelled gene.", corrected_sum)
emit("CORRECTED_NNZ", "Non-zero corrected counts per modelled gene.", corrected_nnz)

writeLines(c(
  "/// Positions within `MODELLED` of the genes `CORRECTED_ROWS` holds in full.",
  sprintf("pub const PROBE_POS: [usize; %d] = [%s];", length(probe_pos), int_vec(probe_pos - 1L)),
  "",
  "/// Full corrected count rows for `PROBE_POS`, row-major.",
  sprintf("pub const CORRECTED_ROWS: [[f64; %d]; %d] = [", ncol(probe_rows), nrow(probe_rows))
), con)
for (i in seq_len(nrow(probe_rows))) {
  writeLines(c("    [", sprintf("        %s,", num_vec(probe_rows[i, ])), "    ],"), con)
}
writeLines(c("];", ""), con)

writeLines(c(
  "//////////////////////////////",
  "// Covariate parity fixture //",
  "//////////////////////////////",
  "",
  "/// Positions within `MODELLED` of the step-1 genes in the covariate run.",
  sprintf("pub const COV_STEP1_POS: [usize; %d] = [%s];",
          length(cov_step1_pos), int_vec(cov_step1_pos)),
  ""
), con)

emit("COV_X", "Per-cell covariate: percent of counts in the first 30 genes.", cov_x)
emit("COV_STEP1_THETA", "Unregularised step-1 theta, covariate run.", cov_mp[, "theta"])
emit("COV_STEP1_INTERCEPT", "Unregularised step-1 intercept, covariate run.", cov_mp[, "(Intercept)"])
emit("COV_STEP1_COEF", "Unregularised step-1 `cov_x` coefficient.", cov_mp[, "cov_x"])
emit("COV_FIT_THETA", "Regularised theta per modelled gene, covariate run.", cov_fit[, "theta"])
emit("COV_FIT_INTERCEPT", "Regularised intercept per modelled gene, covariate run.", cov_fit[, "(Intercept)"])
emit("COV_FIT_COEF", "Regularised `cov_x` coefficient per modelled gene.", cov_fit[, "cov_x"])
emit("COV_RESIDUAL_VARIANCE", "Residual variance per modelled gene, covariate run.", cov_resid_var)

close(con)

cat(sprintf("wrote tests/sctransform_fixtures/mod.rs: %d modelled genes, %d step-1 genes\n",
            length(modelled), length(step1_pos)))
cat(sprintf("theta range: %.4f .. %.4f (%d infinite)\n",
            min(mp[is.finite(mp[, "theta"]), "theta"]),
            max(mp[is.finite(mp[, "theta"]), "theta"]),
            sum(!is.finite(mp[, "theta"]))))
cat(sprintf("poisson genes in fit: %d / %d\n",
            sum(!is.finite(mp_fit[, "theta"])), nrow(mp_fit)))
