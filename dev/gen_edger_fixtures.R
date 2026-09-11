# Parity fixtures for src/methods/dge_bulk.rs: the edgeR quasi-likelihood
# chain, limma-voom and limma-trend, all on the same counts and design.
#
# Regenerates the constants in tests/edger_fixtures/mod.rs. Run with
#   Rscript dev/gen_edger_fixtures.R
# and paste the output into that file.
#
# The counts are produced by a Numerical Recipes LCG and a modulo, so both
# languages build the identical integer matrix without any data crossing as
# text. Everything here is exact in a double: 1664525 * (2^32 - 1) is below
# 2^53.

N_GENES <- 200L
N_SAMPLES <- 8L
N_COEF <- 2L
LCG_SEED <- 20260101

lcg_state <- LCG_SEED
lcg_next <- function() {
  lcg_state <<- (1664525 * lcg_state + 1013904223) %% 4294967296
  lcg_state
}

# Gene modulus per group. Every fortieth gene sits at three counts or fewer, so
# `filterByExpr` has something to remove; every seventh is three times higher in
# the second group, so the test has something to find.
gene_modulus <- function(gene, group) {
  base <- 3 + (gene %% 40) * 5
  if (group == 1 && gene %% 7 == 0) base * 3 else base
}

group <- rep(c(0, 1), each = N_SAMPLES / 2)

counts <- matrix(0, nrow = N_GENES, ncol = N_SAMPLES)
for (g in seq_len(N_GENES)) {
  for (s in seq_len(N_SAMPLES)) {
    counts[g, s] <- lcg_next() %% gene_modulus(g - 1L, group[s])
  }
}

mm <- cbind(`(Intercept)` = 1, group = group)

dge <- edgeR::DGEList(counts)
keep <- edgeR::filterByExpr(dge, design = mm)
dge <- dge[keep, ]
dge <- edgeR::calcNormFactors(dge, method = "TMM")
fit <- edgeR::glmQLFit(dge, mm, robust = FALSE, legacy = FALSE)
res <- edgeR::glmQLFTest(fit, coef = 2)
tt <- as.data.frame(edgeR::topTags(res, sort.by = "none", n = Inf))

fmt <- function(x) paste(sprintf("%.17e", x), collapse = ",\n    ")

cat("/// Genes kept by `filterByExpr`, zero based.\n")
cat("pub const KEPT: &[usize] = &[\n    ")
cat(paste(which(keep) - 1L, collapse = ", "))
cat(",\n];\n\n")

cat("/// edgeR's `logFC` for the group coefficient.\n")
cat("pub const LOG_FC: &[f64] = &[\n    ")
cat(fmt(tt$logFC))
cat(",\n];\n\n")

cat("/// edgeR's `logCPM`.\n")
cat("pub const LOG_CPM: &[f64] = &[\n    ")
cat(fmt(tt$logCPM))
cat(",\n];\n\n")

cat("/// edgeR's quasi-likelihood `F`.\n")
cat("pub const F_STAT: &[f64] = &[\n    ")
cat(fmt(tt$F))
cat(",\n];\n\n")

cat("/// edgeR's `PValue`.\n")
cat("pub const P_VALUE: &[f64] = &[\n    ")
cat(fmt(tt$PValue))
cat(",\n];\n\n")

cat("/// edgeR's `FDR`.\n")
cat("pub const FDR: &[f64] = &[\n    ")
cat(fmt(tt$FDR))
cat(",\n];\n")

# limma-voom and limma-trend on the same filtered, TMM-normalised DGEList and
# the same design. voomLmFit's defaults (span 0.5, adaptive.span TRUE, prior
# count 0.5) already match what the Rust side asks for, and normalize.method
# stays at "none" because the library sizes carry the TMM factors already.
vfit <- edgeR::voomLmFit(dge, mm, block = NULL, sample.weights = FALSE)
veb <- limma::eBayes(vfit)
vtt <- limma::topTable(veb, coef = 2, number = Inf, sort.by = "none")

# limma-trend: log-CPM against the same effective library sizes, then lmFit and
# a prior trended on the average log-expression.
tfit <- limma::lmFit(edgeR::cpm(dge, log = TRUE, prior.count = 2), mm)
teb <- limma::eBayes(tfit, trend = TRUE)
ttt <- limma::topTable(teb, coef = 2, number = Inf, sort.by = "none")

emit <- function(doc, name, x) {
  cat(sprintf("/// %s\n", doc))
  cat(sprintf("pub const %s: &[f64] = &[\n    ", name))
  cat(fmt(x))
  cat(",\n];\n\n")
}

cat("\n")
emit("limma-voom `logFC` for the group coefficient.", "VOOM_LOG_FC", vtt$logFC)
emit("limma-voom `AveExpr`.", "VOOM_AVE_EXPR", vtt$AveExpr)
emit("limma-voom moderated `t`.", "VOOM_T", vtt$t)
emit("limma-voom `P.Value`.", "VOOM_P", vtt$P.Value)
emit("limma-voom `adj.P.Val`.", "VOOM_ADJ_P", vtt$adj.P.Val)
emit("limma-voom `B`.", "VOOM_B", vtt$B)

emit("limma-trend `logFC` for the group coefficient.", "TREND_LOG_FC", ttt$logFC)
emit("limma-trend `AveExpr`.", "TREND_AVE_EXPR", ttt$AveExpr)
emit("limma-trend moderated `t`.", "TREND_T", ttt$t)
emit("limma-trend `P.Value`.", "TREND_P", ttt$P.Value)
emit("limma-trend `adj.P.Val`.", "TREND_ADJ_P", ttt$adj.P.Val)
emit("limma-trend `B`.", "TREND_B", ttt$B)
