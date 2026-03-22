//! Boost doublet detection for single cell data.
//!
//! Implements an iterative community-based doublet detection method inspired
//! by the Python doubletdetection package. Each iteration simulates doublets,
//! embeds observed and simulated cells together via PCA, clusters the combined
//! graph with Louvain, and scores communities by synthetic doublet enrichment
//! (hypergeometric test). Across iterations, doublets are called by majority
//! voting on per-iteration significance.

use rand::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::core::math::stats::*;
use crate::graph::community_detections::*;
use crate::graph::graph_structures::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::utils_doublets::*;

////////////////////////
// Params and results //
////////////////////////

/// Parameters for the Boost doublet detection algorithm.
///
/// Controls preprocessing, simulation, clustering, iteration count and
/// the voting-based doublet calling.
#[derive(Clone, Debug)]
pub struct BoostParams {
    /// Whether to log-transform counts after normalisation.
    pub log_transform: bool,
    /// Whether to mean-centre genes before PCA.
    pub mean_center: bool,
    /// Whether to scale genes to unit variance before PCA.
    pub normalise_variance: bool,
    /// Optional target library size. Defaults to the mean HVG library size.
    pub target_size: Option<f32>,
    /// Percentile threshold for HVG selection.
    pub min_gene_var_pctl: f32,
    /// HVG method: `"vst"`, `"mvb"`, or `"dispersion"`.
    pub hvg_method: String,
    /// Loess span for VST fitting.
    pub loess_span: f64,
    /// Optional clip max for variance stabilisation.
    pub clip_max: Option<f32>,
    /// Ratio of simulated doublets to observed cells.
    pub boost_rate: f32,
    /// Whether to sample cell pairs with replacement.
    pub replace: bool,
    /// Number of principal components.
    pub no_pcs: usize,
    /// Whether to use randomised SVD.
    pub random_svd: bool,
    /// Resolution parameter for Louvain clustering.
    pub resolution: f32,
    /// Number of Louvain iterations per clustering step.
    pub louvain_iters: usize,
    /// Number of boost iterations (each generates fresh doublets).
    pub n_iters: usize,
    /// P-value threshold for per-iteration significance.
    pub p_thresh: f32,
    /// Fraction threshold for majority voting (0-1).
    pub voter_thresh: f32,
    /// Parameters for kNN construction.
    pub knn_params: KnnParams,
}

/// Results from the Boost doublet detection algorithm.
#[derive(Clone, Debug)]
pub struct BoostResult {
    /// Per-cell doublet predictions (`true` = doublet).
    pub predicted_doublets: Vec<bool>,
    /// Doublet score per cell, averaged across iterations.
    pub doublet_scores: Vec<f32>,
    /// Average voting fraction per cell across iterations.
    pub voting_average: Vec<f32>,
}

//////////////////////
// Boost-specific   //
//////////////////////

/// Score communities by synthetic doublet enrichment.
///
/// For each community, computes the fraction of members that are synthetic
/// and a hypergeometric p-value testing whether synthetic cells are
/// over-represented.
///
/// ### Params
///
/// * `orig_communities` - Cluster assignments for observed cells.
/// * `synth_communities` - Cluster assignments for simulated doublets.
///
/// ### Returns
///
/// Tuple of (enrichment_scores, log_p_values) per observed cell.
fn score_communities(
    orig_communities: &[usize],
    synth_communities: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut synth_counts: FxHashMap<usize, usize> = FxHashMap::default();
    let mut orig_counts: FxHashMap<usize, usize> = FxHashMap::default();

    for &c in synth_communities {
        *synth_counts.entry(c).or_insert(0) += 1;
    }
    for &c in orig_communities {
        *orig_counts.entry(c).or_insert(0) += 1;
    }

    let unique_comms: FxHashSet<usize> = orig_communities.iter().copied().collect();
    let mut comm_scores: FxHashMap<usize, f32> = FxHashMap::default();
    let mut comm_log_p: FxHashMap<usize, f32> = FxHashMap::default();

    for &c in &unique_comms {
        let n_synth = *synth_counts.get(&c).unwrap_or(&0);
        let n_orig = *orig_counts.get(&c).unwrap_or(&0);

        comm_scores.insert(c, n_synth as f32 / (n_synth + n_orig) as f32);

        let k = n_synth + n_orig;
        let log_p =
            hypergeom_pval::<f32>(n_synth, synth_communities.len(), orig_communities.len(), k).ln();
        comm_log_p.insert(c, log_p);
    }

    let scores: Vec<f32> = orig_communities.iter().map(|&c| comm_scores[&c]).collect();
    let log_p_values: Vec<f32> = orig_communities.iter().map(|&c| comm_log_p[&c]).collect();

    (scores, log_p_values)
}

/// Call doublets via majority voting across iterations.
///
/// A cell is called a doublet if the fraction of iterations in which its
/// log p-value fell below `p_thresh` exceeds `voter_thresh`.
///
/// ### Params
///
/// * `all_log_p_values` - Log p-values per cell, one vector per iteration.
/// * `p_thresh` - P-value threshold for per-iteration significance.
/// * `voter_thresh` - Fraction of iterations required for a doublet call.
///
/// ### Returns
///
/// Tuple of (predictions, voting_averages) per cell.
fn predict_voting(
    all_log_p_values: &[Vec<f32>],
    p_thresh: f32,
    voter_thresh: f32,
) -> (Vec<bool>, Vec<f32>) {
    let n_cells = all_log_p_values[0].len();
    let n_iters = all_log_p_values.len();
    let log_p_thresh = p_thresh.ln();

    let voting_avg: Vec<f32> = (0..n_cells)
        .map(|cell_idx| {
            let votes: usize = all_log_p_values
                .iter()
                .filter(|iter_vals| iter_vals[cell_idx] <= log_p_thresh)
                .count();
            votes as f32 / n_iters as f32
        })
        .collect();

    let labels: Vec<bool> = voting_avg.iter().map(|&avg| avg >= voter_thresh).collect();

    (labels, voting_avg)
}

/// Find a score cutoff using the largest-gap heuristic.
///
/// Sorts scores and picks the value at the largest gap, separating the
/// low-score (singlet) and high-score (doublet) groups. Used as a fallback
/// for single-iteration mode where voting is not meaningful.
///
/// ### Params
///
/// * `scores` - Community enrichment scores per cell.
///
/// ### Returns
///
/// The threshold score value.
fn find_score_cutoff(scores: &[f32]) -> f32 {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if sorted.len() < 2 {
        return sorted[0];
    }

    let mut max_gap = 0.0f32;
    let mut gap_idx = 0;

    for i in 1..sorted.len() {
        let gap = sorted[i] - sorted[i - 1];
        if gap > max_gap {
            max_gap = gap;
            gap_idx = i;
        }
    }

    sorted[gap_idx]
}

////////////////////
// Main structure //
////////////////////

/// Boost doublet detection.
///
/// Iterative method that simulates doublets, clusters observed and simulated
/// cells together via Louvain, and scores communities by synthetic enrichment.
/// Doublets are called by majority voting across iterations.
#[derive(Clone, Debug)]
pub struct BoostClassifier {
    /// Path to the gene-based binary file (CSC format).
    f_path_gene: String,
    /// Path to the cell-based binary file (CSR format).
    f_path_cell: String,
    /// Algorithm parameters.
    params: BoostParams,
    /// Number of observed cells.
    n_cells: usize,
    /// Number of simulated doublets per iteration.
    n_cells_sim: usize,
    /// Cell indices included in this analysis.
    cells_to_keep: Vec<usize>,
    /// Per-cell library sizes computed over HVG genes only.
    hvg_library_sizes: Vec<usize>,
}

impl BoostClassifier {
    /// Create a new BoostClassifier instance.
    ///
    /// ### Params
    ///
    /// * `f_path_gene` - Path to the gene-based binary file (CSC).
    /// * `f_path_cell` - Path to the cell-based binary file (CSR).
    /// * `params` - Boost parameters.
    /// * `cell_indices` - Cell indices to include in the analysis.
    ///
    /// ### Returns
    ///
    /// Initialised `BoostClassifier`.
    pub fn new(
        f_path_gene: &str,
        f_path_cell: &str,
        params: BoostParams,
        cell_indices: &[usize],
    ) -> Self {
        BoostClassifier {
            f_path_gene: f_path_gene.to_string(),
            f_path_cell: f_path_cell.to_string(),
            params,
            n_cells: cell_indices.len(),
            n_cells_sim: 0,
            cells_to_keep: cell_indices.to_vec(),
            hvg_library_sizes: Vec::new(),
        }
    }

    /// Run the full Boost pipeline.
    ///
    /// Selects HVGs, computes library sizes, then runs `n_iters` iterations
    /// of simulate-embed-cluster-score. Final doublet calls are made by
    /// majority voting (multi-iteration) or largest-gap cutoff (single
    /// iteration).
    ///
    /// ### Params
    ///
    /// * `streaming` - Stream HVG computation to reduce memory pressure.
    /// * `seed` - Seed for reproducibility.
    /// * `verbose` - Controls verbosity.
    ///
    /// ### Returns
    ///
    /// `BoostResult` with predictions, scores and voting averages.
    pub fn run_boost(&mut self, streaming: bool, seed: usize, verbose: bool) -> BoostResult {
        let start_all = Instant::now();

        let hvg_opts = HvgOpts {
            method: self.params.hvg_method.clone(),
            loess_span: self.params.loess_span as f32,
            clip_max: self.params.clip_max,
            min_gene_var_pctl: self.params.min_gene_var_pctl,
        };

        if verbose {
            println!("Identifying highly variable genes...");
        }
        let start_hvg = Instant::now();

        let hvg_genes = select_hvg(
            &self.f_path_gene,
            &self.cells_to_keep,
            &hvg_opts,
            streaming,
            verbose,
        );

        if verbose {
            println!(
                "Using {} highly variable genes. Done in {:.2?}",
                hvg_genes.len(),
                start_hvg.elapsed()
            );
        }

        self.hvg_library_sizes =
            compute_hvg_library_sizes(&self.f_path_cell, &self.cells_to_keep, &hvg_genes);
        let target_size = resolve_target_size(self.params.target_size, &self.hvg_library_sizes);
        self.n_cells_sim = (self.n_cells as f32 * self.params.boost_rate) as usize;

        // Run iterations
        let start_iters = Instant::now();

        let iter_results: Vec<(Vec<f32>, Vec<f32>)> = (0..self.params.n_iters)
            .map(|iter| {
                if verbose {
                    println!(
                        " == Running iteration {} of {} ==",
                        iter + 1,
                        self.params.n_iters
                    );
                }
                self.one_iteration(target_size, &hvg_genes, seed + iter, verbose)
            })
            .collect();

        if verbose {
            println!(
                "Completed {} iterations in {:.2?}\n",
                self.params.n_iters,
                start_iters.elapsed()
            );
        }

        // Aggregate
        let all_scores: Vec<Vec<f32>> = iter_results.iter().map(|(s, _)| s.clone()).collect();
        let all_log_p_values: Vec<Vec<f32>> = iter_results.iter().map(|(_, p)| p.clone()).collect();

        let result = if self.params.n_iters > 1 {
            let (labels, voting_avg) = predict_voting(
                &all_log_p_values,
                self.params.p_thresh,
                self.params.voter_thresh,
            );

            let avg_scores: Vec<f32> = (0..self.n_cells)
                .map(|i| {
                    all_scores.iter().map(|iter| iter[i]).sum::<f32>() / self.params.n_iters as f32
                })
                .collect();

            BoostResult {
                predicted_doublets: labels,
                doublet_scores: avg_scores,
                voting_average: voting_avg,
            }
        } else {
            let scores = &all_scores[0];
            let cutoff = find_score_cutoff(scores);

            if verbose {
                println!("Score cutoff: {:.4}", cutoff);
            }

            BoostResult {
                predicted_doublets: scores.iter().map(|&s| s >= cutoff).collect(),
                doublet_scores: scores.clone(),
                voting_average: vec![0.0; self.n_cells],
            }
        };

        if verbose {
            let n_doublets = result.predicted_doublets.iter().filter(|&&d| d).count();
            println!(
                "Detected {} doublets ({:.1}%)",
                n_doublets,
                100.0 * n_doublets as f32 / self.n_cells as f32
            );
            println!("Total runtime: {:.2?}", start_all.elapsed());
        }

        result
    }

    /// Execute a single Boost iteration.
    ///
    /// Simulates doublets, runs PCA, builds a kNN graph, clusters via
    /// Louvain, and scores communities by synthetic doublet enrichment.
    ///
    /// ### Params
    ///
    /// * `target_size` - Normalisation target library size.
    /// * `hvg_genes` - Indices of highly variable genes.
    /// * `seed` - Seed for this iteration.
    /// * `verbose` - Controls verbosity.
    ///
    /// ### Returns
    ///
    /// Tuple of (community_scores, log_p_values) per observed cell.
    fn one_iteration(
        &self,
        target_size: f32,
        hvg_genes: &[usize],
        seed: usize,
        verbose: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let pca_opts = PcaOpts {
            log_transform: self.params.log_transform,
            mean_center: self.params.mean_center,
            normalise_variance: self.params.normalise_variance,
            no_pcs: self.params.no_pcs,
            random_svd: self.params.random_svd,
        };

        // Generate pairs (Boost supports with/without replacement)
        let pairs = self.generate_pairs(seed);

        // Simulate
        let sim_chunks = simulate_from_pairs(
            &pairs,
            &self.cells_to_keep,
            &self.hvg_library_sizes,
            hvg_genes,
            &self.f_path_cell,
            target_size,
            self.params.log_transform,
        );

        // PCA + projection
        let (combined_pca, _) = pca_and_project(
            &self.f_path_gene,
            &self.cells_to_keep,
            hvg_genes,
            &self.hvg_library_sizes,
            target_size,
            &sim_chunks,
            &pca_opts,
            seed,
            verbose,
        );

        // kNN
        let k_adj = adjusted_k(self.params.knn_params.k, self.n_cells, self.n_cells_sim);
        if verbose {
            println!("Using {} neighbours in the kNN generation.", k_adj);
        }
        let knn = dispatch_knn(
            combined_pca.as_ref(),
            k_adj,
            &self.params.knn_params,
            seed,
            verbose,
        );

        // Cluster
        let start_graph = Instant::now();
        let graph = knn_to_sparse_graph(&knn);
        if verbose {
            println!(
                "Transformed kNN graph. Done in {:.2?}",
                start_graph.elapsed()
            );
        }

        let start_cluster = Instant::now();
        let communities = louvain_sparse_graph(
            &graph,
            self.params.resolution,
            self.params.louvain_iters,
            seed,
        );
        if verbose {
            println!(
                "Generated communities via Louvain clustering. Done in {:.2?}",
                start_cluster.elapsed()
            );
        }

        // Score
        let orig_communities = communities[..self.n_cells].to_vec();
        let synth_communities = communities[self.n_cells..].to_vec();

        score_communities(&orig_communities, &synth_communities)
    }

    /// Generate cell pairs for doublet simulation.
    ///
    /// Supports both with-replacement (random pairs) and without-replacement
    /// (shuffled pairing) modes.
    ///
    /// ### Params
    ///
    /// * `seed` - Seed for reproducibility.
    ///
    /// ### Returns
    ///
    /// Vector of (position_a, position_b) pairs into `cells_to_keep`.
    fn generate_pairs(&self, seed: usize) -> Vec<(usize, usize)> {
        let n_sim = (self.n_cells as f32 * self.params.boost_rate) as usize;

        if self.params.replace {
            random_pairs(self.cells_to_keep.len(), n_sim, seed)
        } else {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let mut available: Vec<usize> = (0..self.cells_to_keep.len()).collect();
            available.shuffle(&mut rng);
            available
                .chunks(2)
                .take(n_sim)
                .filter_map(|chunk| {
                    if chunk.len() == 2 {
                        Some((chunk[0], chunk[1]))
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
}
