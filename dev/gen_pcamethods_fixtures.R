# Parity fixtures for src/core/math/pca_missing.rs: pcaMethods' ppca() and
# bpca() on data with missing values.
#
# Regenerates tests/pcamethods_fixtures/mod.rs. Run with
#   Rscript dev/gen_pcamethods_fixtures.R > tests/pcamethods_fixtures/mod.rs
#
# The data come from a Numerical Recipes LCG: an integer rank-k part plus
# (lcg %% 2001 - 1000) / 250 noise. Every value is one correctly rounded
# division away from an integer, so Rust rebuilds the identical doubles. The
# mask is lcg %% 100 < missing percentage, redrawn if a row or column ends up
# empty. The PPCA start is R's own random draw and crosses as text.

suppressMessages(library(pcaMethods))

lcg_state <- 0
lcg_next <- function() {
  lcg_state <<- (1664525 * lcg_state + 1013904223) %% 4294967296
  lcg_state
}

build_case <- function(n, d, k, miss_pct, seed) {
  lcg_state <<- seed
  z <- matrix(0, n, k)
  w <- matrix(0, d, k)
  for (i in seq_len(n)) for (l in seq_len(k)) z[i, l] <- lcg_next() %% 11 - 5
  for (j in seq_len(d)) for (l in seq_len(k)) w[j, l] <- lcg_next() %% 7 - 3
  y <- matrix(0, n, d)
  for (i in seq_len(n)) {
    for (j in seq_len(d)) {
      y[i, j] <- sum(z[i, ] * w[j, ]) + (lcg_next() %% 2001 - 1000) / 250
    }
  }
  repeat {
    mask <- matrix(FALSE, n, d)
    for (i in seq_len(n)) for (j in seq_len(d)) mask[i, j] <- lcg_next() %% 100 < miss_pct
    if (all(rowSums(!mask) > 0) && all(colSums(!mask) > 0)) break
  }
  y[mask] <- NA
  y
}

fmt <- function(x) paste(sprintf("%.17e", as.vector(x)), collapse = ",\n    ")

emit <- function(name, doc, x) {
  cat(sprintf("/// %s\npub const %s: &[f64] = &[\n    %s,\n];\n\n", doc, name, fmt(x)))
}

cases <- list(
  list(tag = "TALL", n = 60L, d = 12L, k = 3L, miss = 20L, seed = 20260926),
  list(tag = "WIDE", n = 16L, d = 80L, k = 3L, miss = 25L, seed = 20260927)
)
PPCA_SEED <- 7L

cat("#![allow(clippy::excessive_precision)]\n")
cat("//! pcaMethods parity fixtures, generated against pcaMethods ",
    as.character(packageVersion("pcaMethods")), " and R ",
    R.version$major, ".", R.version$minor, ".\n", sep = "")
cat("//!\n")
cat("//! DO NOT EDIT. Regenerate with\n")
cat("//! `Rscript dev/gen_pcamethods_fixtures.R > tests/pcamethods_fixtures/mod.rs`.\n")
cat("//!\n")
cat("//! Matrices are flattened column-major. See the script for how the data\n")
cat("//! and the mask are built from the LCG. `ppca()` ran with seed ", PPCA_SEED,
    "; its start is\n//! recorded as `*_PPCA_C0`.\n\n", sep = "")

for (cs in cases) {
  y <- build_case(cs$n, cs$d, cs$k, cs$miss, cs$seed)
  t <- cs$tag
  cat(sprintf("/// Rows of the %s case.\npub const %s_N: usize = %d;\n\n", t, t, cs$n))
  cat(sprintf("/// Columns of the %s case.\npub const %s_D: usize = %d;\n\n", t, t, cs$d))
  cat(sprintf("/// Components of the %s case.\npub const %s_K: usize = %d;\n\n", t, t, cs$k))
  cat(sprintf("/// Missing percentage of the %s case.\npub const %s_MISS_PCT: u64 = %d;\n\n",
              t, t, cs$miss))
  cat(sprintf("/// LCG seed of the %s case.\npub const %s_SEED: u64 = %d;\n\n", t, t, cs$seed))

  # ppca() draws sample(N) and then its start from the same stream
  set.seed(PPCA_SEED)
  invisible(sample(cs$n))
  c0 <- matrix(rnorm(cs$d * cs$k), cs$d, cs$k)
  emit(paste0(t, "_PPCA_C0"), "R's initial PPCA loadings, D x k.", c0)

  pp <- pca(y, method = "ppca", nPcs = cs$k, center = TRUE, seed = PPCA_SEED)
  emit(paste0(t, "_PPCA_SCORES"), "PPCA scores, N x k.", scores(pp))
  emit(paste0(t, "_PPCA_LOADINGS"), "PPCA loadings, D x k.", loadings(pp))
  emit(paste0(t, "_PPCA_R2CUM"), "PPCA cumulative R^2.", pp@R2cum)
  emit(paste0(t, "_PPCA_COMPLETED"), "PPCA completeObs, N x D.", completeObs(pp))

  bp <- pca(y, method = "bpca", nPcs = cs$k, center = TRUE, verbose = FALSE)
  emit(paste0(t, "_BPCA_SCORES"), "BPCA scores, N x k.", scores(bp))
  emit(paste0(t, "_BPCA_LOADINGS"), "BPCA loadings, D x k.", loadings(bp))
  emit(paste0(t, "_BPCA_R2CUM"), "BPCA cumulative R^2.", bp@R2cum)
  emit(paste0(t, "_BPCA_COMPLETED"), "BPCA completeObs, N x D.", completeObs(bp))
}
