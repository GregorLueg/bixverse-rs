# Parity fixtures for src/core/math/pca_missing.rs: pcaMethods' ppca() and
# bpca() on data with missing values.
#
# Regenerates tests/pcamethods_fixtures/{tall,wide,large}.txt. Run from the
# crate root with
#   Rscript dev/gen_pcamethods_fixtures.R
#
# One line per quantity: a key, then its values separated by spaces, matrices
# flattened column-major. `%.17e` round-trips a double exactly. Lines starting
# with `#` are comments. large.txt is the 100 x 1000 case behind `large-test`
# and leaves out the completed matrix (100k entries); scores and loadings
# determine it.
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

emit <- function(key, x) {
  cat(key, sprintf("%.17e", as.vector(x)), "\n")
}

PPCA_SEED <- 7L

write_case <- function(cs, completed) {
  sink(sprintf("tests/pcamethods_fixtures/%s.txt", cs$tag))
  on.exit(sink())
  cat("# pcaMethods ", as.character(packageVersion("pcaMethods")), ", R ",
      R.version$major, ".", R.version$minor, ". DO NOT EDIT; regenerate with\n",
      "# `Rscript dev/gen_pcamethods_fixtures.R`. ppca() ran with seed ", PPCA_SEED,
      "; PPCA_C0 is its start.\n", sep = "")
  cat("N", cs$n, "\nD", cs$d, "\nK", cs$k, "\nMISS_PCT", cs$miss, "\nSEED", cs$seed, "\n")

  y <- build_case(cs$n, cs$d, cs$k, cs$miss, cs$seed)

  # ppca() draws sample(N) and then its start from the same stream
  set.seed(PPCA_SEED)
  invisible(sample(cs$n))
  emit("PPCA_C0", matrix(rnorm(cs$d * cs$k), cs$d, cs$k))

  pp <- pca(y, method = "ppca", nPcs = cs$k, center = TRUE, seed = PPCA_SEED)
  emit("PPCA_SCORES", scores(pp))
  emit("PPCA_LOADINGS", loadings(pp))
  emit("PPCA_R2CUM", pp@R2cum)
  if (completed) emit("PPCA_COMPLETED", completeObs(pp))

  bp <- pca(y, method = "bpca", nPcs = cs$k, center = TRUE, verbose = FALSE)
  emit("BPCA_SCORES", scores(bp))
  emit("BPCA_LOADINGS", loadings(bp))
  emit("BPCA_R2CUM", bp@R2cum)
  if (completed) emit("BPCA_COMPLETED", completeObs(bp))
}

dir.create("tests/pcamethods_fixtures", showWarnings = FALSE)
write_case(list(tag = "tall", n = 60L, d = 12L, k = 3L, miss = 20L, seed = 20260926), TRUE)
write_case(list(tag = "wide", n = 16L, d = 80L, k = 3L, miss = 25L, seed = 20260927), TRUE)
write_case(list(tag = "large", n = 100L, d = 1000L, k = 5L, miss = 40L, seed = 11), FALSE)
