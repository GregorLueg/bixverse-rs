# Export an igraph graph to the flat binary edge list that the gated Leiden
# tests in crates/bixverse-gpu/src/graph/leiden_gpu.rs read through
# BIXVERSE_LEIDEN_GRAPH.
#
#   Rscript dev/export_leiden_graph.R <graph.qs2> <out.bin>
#
# The input is a qs2 file holding an undirected igraph object. Layout of the
# output, all little-endian 4-byte values: n and m as i32, then m sources and
# m targets (0-based i32), then m weights as f32. Unweighted graphs get weight 1.

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)

g <- qs2::qs_read(path.expand(args[1]))
el <- igraph::as_edgelist(g, names = FALSE)
w <- igraph::E(g)$weight
if (is.null(w)) {
  w <- rep(1, nrow(el))
}

con <- file(args[2], "wb")
writeBin(as.integer(c(igraph::vcount(g), nrow(el))), con, size = 4)
writeBin(as.integer(el[, 1] - 1L), con, size = 4)
writeBin(as.integer(el[, 2] - 1L), con, size = 4)
writeBin(as.numeric(w), con, size = 4)
close(con)
