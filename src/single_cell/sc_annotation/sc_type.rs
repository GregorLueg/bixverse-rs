//! ScType cell type annotation, streaming over gene chunks. Based on Ianevski
//! et al. (2022).
//!
//! Beyond the original cluster-level assignment this module also offers a
//! per-cell path: the raw scores are smoothed over a caller-supplied cell-cell
//! graph, assigned per cell, and only clusters that turn out to be impure are
//! resolved at cell resolution. Cluster-level calls are robust but silently
//! swallow minority populations sitting inside a mixed cluster.

use indexmap::IndexSet;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::graph::graph_label_propagations::KnnLabPropGraph;
use crate::prelude::*;
use crate::single_cell::sc_processing::pca::scale_csc_chunk;

////////////
// Consts //
////////////

/// The batch size for the genes
const GENE_BATCH_SIZE: usize = 100;

/// Minimum weight a given marker can have
const WEIGHT_FLOOR: f32 = 0.1;

/// Default self-retention in the smoothing step. 0.5 keeps half of a cell's own
/// signal per iteration; lower values smooth harder and start erasing small
/// populations sitting inside a large cluster.
const SMOOTH_ALPHA: f32 = 0.5;

/// Default number of smoothing iterations. Two hops averages out most of the
/// dropout noise on a k = 15 graph without bleeding across cell type boundaries.
const SMOOTH_ITERATIONS: usize = 2;

/// Convergence tolerance for the smoothing iteration. Stops early once the
/// largest score change across all cells and cell types drops below this.
const SMOOTH_TOLERANCE: f32 = 1e-4;

/// Minimum score for a per-cell call. Mirrors the cluster-level `n_cells / 4`
/// heuristic of Ianevski et al., expressed per cell.
const CELL_SCORE_FLOOR: f32 = 0.25;

/// Cluster purity above which the cluster-level call is kept as-is. Set high on
/// purpose: a cluster has to be near-clean to earn a blanket label, everything
/// else gets resolved per cell.
const PURITY_THRESHOLD: f32 = 0.9;

/// Standard deviations below this are treated as degenerate during column
/// standardisation and the column is zeroed instead of divided through.
const MIN_SD: f64 = 1e-8;

////////////////
// Structures //
////////////////

/// Struture to hold the information on the cell type markers
pub struct CellTypeMarkers {
    /// The represented cell type
    pub cell_type: String,
    /// The positive marker indices
    pub positive_indices: Vec<usize>,
    /// The negative marker indices
    pub negative_indices: Vec<usize>,
}

/// The ScType results structure
pub struct SctypeRes {
    /// The represented cell types
    pub cell_types: Vec<String>,
    /// Row-major scores (cells x cell_types).
    pub scores: Vec<f32>,
    /// Number of cells
    pub n_cells: usize,
    /// Number of represented cell types
    pub n_cell_types: usize,
}

/// Structure for the cluster assignment
pub struct ScTypeClusterAssignment {
    /// Cluster index
    pub cluster: usize,
    /// Final cell type annotation
    pub cell_type: String,
    /// Final score of the cluster
    pub score: f32,
    /// Number of cells in the cluster
    pub n_cells: usize,
}

/// How the raw ScType scores get rescaled before a per-cell argmax
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScoreCalibration {
    /// Raw scores as produced by [run_sctype]
    #[default]
    None,
    /// Standardise each cell type's score column across cells. Removes the bias
    /// towards cell types whose marker sets happen to produce a larger score
    /// magnitude, which matters far more for per-cell calls than for the
    /// cluster-level aggregate.
    ColumnZ,
}

/// Parse the score calibration string
///
/// ### Params
///
/// * `s` - String to parse.
///
/// ### Returns
///
/// The Option of the [ScoreCalibration] to apply.
pub fn parse_score_calibration(s: &str) -> Option<ScoreCalibration> {
    match s.to_lowercase().as_str() {
        "none" | "raw" => Some(ScoreCalibration::None),
        "column_z" | "columnz" | "z" => Some(ScoreCalibration::ColumnZ),
        _ => None,
    }
}

/// Parameters for the per-cell ScType assignment path
#[derive(Clone, Copy, Debug)]
pub struct ScTypeCellParams {
    /// Self-retention during smoothing. Each iteration computes
    /// `alpha * original + (1 - alpha) * neighbour_average`.
    pub alpha: f32,
    /// Number of smoothing iterations
    pub iterations: usize,
    /// Convergence tolerance for the smoothing iteration
    pub tolerance: f32,
    /// How to rescale the scores before the argmax
    pub calibration: ScoreCalibration,
    /// Minimum score for a cell to get a call instead of `Unknown`
    pub score_floor: f32,
    /// Cluster purity above which the cluster-level call is kept
    pub purity_threshold: f32,
}

impl Default for ScTypeCellParams {
    fn default() -> Self {
        Self {
            alpha: SMOOTH_ALPHA,
            iterations: SMOOTH_ITERATIONS,
            tolerance: SMOOTH_TOLERANCE,
            calibration: ScoreCalibration::None,
            score_floor: CELL_SCORE_FLOOR,
            purity_threshold: PURITY_THRESHOLD,
        }
    }
}

impl ScTypeCellParams {
    /// Generate the per-cell ScType parameters
    ///
    /// ### Params
    ///
    /// * `alpha` - Self-retention during smoothing, between 0 and 1. If not
    ///   provided, defaults to `SMOOTH_ALPHA` (0.5).
    /// * `iterations` - Number of smoothing iterations. If not provided,
    ///   defaults to `SMOOTH_ITERATIONS` (2).
    /// * `tolerance` - Convergence tolerance. If not provided, defaults to
    ///   `SMOOTH_TOLERANCE` (1e-4).
    /// * `calibration` - The [ScoreCalibration] to apply. If not provided,
    ///   defaults to [ScoreCalibration::None].
    /// * `score_floor` - Minimum score for a per-cell call. If not provided,
    ///   defaults to `CELL_SCORE_FLOOR` (0.25).
    /// * `purity_threshold` - Cluster purity above which the cluster-level call
    ///   is kept. If not provided, defaults to `PURITY_THRESHOLD` (0.9).
    ///
    /// ### Returns
    ///
    /// The initialised [ScTypeCellParams].
    pub fn new(
        alpha: Option<f32>,
        iterations: Option<usize>,
        tolerance: Option<f32>,
        calibration: Option<ScoreCalibration>,
        score_floor: Option<f32>,
        purity_threshold: Option<f32>,
    ) -> Self {
        let defaults = Self::default();
        Self {
            alpha: alpha.unwrap_or(defaults.alpha),
            iterations: iterations.unwrap_or(defaults.iterations),
            tolerance: tolerance.unwrap_or(defaults.tolerance),
            calibration: calibration.unwrap_or(defaults.calibration),
            score_floor: score_floor.unwrap_or(defaults.score_floor),
            purity_threshold: purity_threshold.unwrap_or(defaults.purity_threshold),
        }
    }
}

/// Per-cell ScType assignments
///
/// Stored as a struct of arrays rather than one structure per cell: at millions
/// of cells a per-cell `String` is not affordable, so the cell type names live
/// once in `cell_types` and everything else indexes into them.
#[derive(Clone, Debug)]
pub struct ScTypeCellRes {
    /// Winning cell type index per cell. `None` means the best score fell below
    /// the score floor, i.e. Unknown.
    pub assignments: Vec<Option<usize>>,
    /// Score of the winning cell type per cell, after calibration and smoothing
    pub scores: Vec<f32>,
    /// Gap between the best and second-best score per cell. A small margin flags
    /// an ambiguous call.
    pub margins: Vec<f32>,
    /// Fraction of graph neighbours sharing this cell's call. `None` when no
    /// graph was supplied.
    pub agreement: Option<Vec<f32>>,
    /// The represented cell types, indexed by `assignments`
    pub cell_types: Vec<String>,
}

/// Cell type composition of a single cluster, derived from the per-cell calls
#[derive(Clone, Debug)]
pub struct ScTypeClusterComposition {
    /// Cluster index
    pub cluster: usize,
    /// Number of cells in the cluster
    pub n_cells: usize,
    /// Per-cell-type counts, indexed as [ScTypeCellRes::cell_types]
    pub counts: Vec<usize>,
    /// Number of cells without a call
    pub n_unknown: usize,
    /// Most frequent cell type. `None` if every cell is Unknown.
    pub dominant: Option<usize>,
    /// Fraction of cells carrying the dominant call, Unknowns included in the
    /// denominator
    pub purity: f32,
    /// Shannon entropy of the composition in nats. Zero for a pure cluster.
    pub entropy: f32,
    /// Runner-up cell type. This is what surfaces a mixture.
    pub second: Option<usize>,
    /// Fraction of cells carrying the runner-up call
    pub second_fraction: f32,
}

/// Final hybrid annotation
#[derive(Clone, Debug)]
pub struct ScTypeHybridRes {
    /// Final cell type index per cell. `None` means Unknown.
    pub assignments: Vec<Option<usize>>,
    /// Per-cluster composition, in ascending cluster order
    pub composition: Vec<ScTypeClusterComposition>,
    /// `true` where the cluster fell below the purity threshold and was
    /// resolved per cell. Indexed by cluster.
    pub cluster_mixed: Vec<bool>,
    /// The represented cell types, indexed by `assignments`
    pub cell_types: Vec<String>,
}

/////////////
// Helpers //
/////////////

/// Per-gene marker sensitivity from the positive marker sets.
///
/// Genes occurring in many cell types get a low score (down to `weight_floor`),
/// genes occurring in only one cell type get 1. Genes absent from every
/// positive set are not in the map; callers should default to 1.0.
///
/// ### Params
///
/// * `markers` - Slice of [CellTypeMarkers]
/// * `weight_floor` - The lowest weight that can be assigned to a given marker.
///
/// ### Returns
///
/// A FxHashMap with gene index and sensitivity
fn compute_marker_sensitivity(
    markers: &[CellTypeMarkers],
    weight_floor: f32,
) -> FxHashMap<usize, f32> {
    let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
    for m in markers {
        for &g in &m.positive_indices {
            *counts.entry(g).or_insert(0) += 1;
        }
    }
    let n_ct = markers.len() as f32;
    if n_ct <= 1.0 {
        return counts.into_keys().map(|g| (g, 1.0)).collect();
    }
    counts
        .into_iter()
        .map(|(g, c)| {
            let w = (n_ct - c as f32) / (n_ct - 1.0);
            (g, w.max(weight_floor))
        })
        .collect()
}

//////////
// Main //
//////////

/// Run the ScType algorithm
///
/// ### Params
///
/// * `f_path` - Path to the gene-based binary file
/// * `cell_indices` - HashSet with the cell indices to keep.
/// * `markers` - A slice of [CellTypeMarkers].
/// * `use_sensitivity` - Boolean. If set up, common cell type markers are down
///   weighted.
/// * `gene_batch_size` - Optional gene batch size. If not provided, defaults
///   to `GENE_BATCH_SIZE` (100).
/// * `weight_floor` - Optional weight floor. If not provided, defaults to
///   `WEIGHT_FLOOR` (0.1)
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The [SctypeRes].
pub fn run_sctype(
    f_path: &str,
    cell_indices: &[usize],
    markers: &[CellTypeMarkers],
    use_sensitivity: bool,
    gene_batch_size: Option<usize>,
    weight_floor: Option<f32>,
    verbose: usize,
) -> Result<SctypeRes, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start_total = Instant::now();

    let weight_floor = weight_floor.unwrap_or(WEIGHT_FLOOR);
    let gene_batch_size = gene_batch_size.unwrap_or(GENE_BATCH_SIZE);

    let reader = ParallelSparseReader::new(f_path)?;
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let no_cells = cell_set.len();
    let n_cell_types = markers.len();

    let sensitivity = if use_sensitivity {
        compute_marker_sensitivity(markers, weight_floor)
    } else {
        FxHashMap::default()
    };

    let mut gene_to_ct_pos: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut gene_to_ct_neg: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (ct, m) in markers.iter().enumerate() {
        for &g in &m.positive_indices {
            gene_to_ct_pos.entry(g).or_default().push(ct);
        }
        for &g in &m.negative_indices {
            gene_to_ct_neg.entry(g).or_default().push(ct);
        }
    }

    let mut marker_set: FxHashSet<usize> = FxHashSet::default();
    for m in markers {
        marker_set.extend(&m.positive_indices);
        marker_set.extend(&m.negative_indices);
    }
    let marker_genes: Vec<usize> = marker_set.into_iter().collect();

    if verbosity.normal_verbosity() {
        println!(
            "ScType: {} cell types, {} marker genes, {} cells",
            n_cell_types,
            marker_genes.len(),
            no_cells
        );
    }

    let mut sum_t1: Vec<Vec<f32>> = markers
        .iter()
        .map(|m| {
            if m.positive_indices.is_empty() {
                Vec::new()
            } else {
                vec![0.0_f32; no_cells]
            }
        })
        .collect();
    let mut sum_t2: Vec<Vec<f32>> = markers
        .iter()
        .map(|m| {
            if m.negative_indices.is_empty() {
                Vec::new()
            } else {
                vec![0.0_f32; no_cells]
            }
        })
        .collect();

    let num_batches = marker_genes.len().div_ceil(gene_batch_size);
    let start_stream = Instant::now();

    for batch_idx in 0..num_batches {
        let start_gene = batch_idx * gene_batch_size;
        let end_gene = ((batch_idx + 1) * gene_batch_size).min(marker_genes.len());
        let batch = &marker_genes[start_gene..end_gene];

        if verbosity.detailed_verbosity() {
            println!(
                "  Batch {}/{}: {} genes",
                batch_idx + 1,
                num_batches,
                batch.len()
            );
        }

        let mut chunks = reader.read_gene_parallel(batch)?;
        chunks
            .par_iter_mut()
            .for_each(|c| c.filter_selected_cells(&cell_set));

        let z_per_gene: Vec<(usize, Vec<f32>)> = chunks
            .par_iter()
            .map(|chunk| {
                let (z, _, _) = scale_csc_chunk(chunk, no_cells, true, true, None);
                (chunk.original_index, z)
            })
            .collect();

        for (gene_idx, z) in &z_per_gene {
            let sens = sensitivity.get(gene_idx).copied().unwrap_or(1.0);

            if let Some(cts) = gene_to_ct_pos.get(gene_idx) {
                for &ct in cts {
                    let row = &mut sum_t1[ct];
                    for (c, &v) in z.iter().enumerate() {
                        row[c] += v * sens;
                    }
                }
            }

            if let Some(cts) = gene_to_ct_neg.get(gene_idx) {
                for &ct in cts {
                    let row = &mut sum_t2[ct];
                    for (c, &v) in z.iter().enumerate() {
                        row[c] -= v * sens;
                    }
                }
            }
        }
    }

    if verbosity.normal_verbosity() {
        println!("ScType: streamed scoring in {:.2?}", start_stream.elapsed());
    }

    let n_pos: Vec<usize> = markers.iter().map(|m| m.positive_indices.len()).collect();
    let n_neg: Vec<usize> = markers.iter().map(|m| m.negative_indices.len()).collect();

    let mut scores = vec![0.0_f32; no_cells * n_cell_types];
    for ct in 0..n_cell_types {
        let denom_pos = (n_pos[ct] as f32).sqrt();
        let denom_neg = (n_neg[ct] as f32).sqrt();
        let has_pos = !sum_t1[ct].is_empty();
        let has_neg = !sum_t2[ct].is_empty();
        for c in 0..no_cells {
            let t1 = if has_pos {
                sum_t1[ct][c] / denom_pos
            } else {
                0.0
            };
            let t2 = if has_neg {
                sum_t2[ct][c] / denom_neg
            } else {
                0.0
            };
            scores[c * n_cell_types + ct] = t1 + t2;
        }
    }

    let cell_types: Vec<String> = markers.iter().map(|m| m.cell_type.clone()).collect();

    if verbosity.normal_verbosity() {
        println!("ScType: total run time -> {:.2?}", start_total.elapsed());
    }

    Ok(SctypeRes {
        cell_types,
        scores,
        n_cells: no_cells,
        n_cell_types,
    })
}

/// Assign the cluster cell types based on ScType
///
/// ### Params
///
/// * `res` - Reference to the [SctypeRes].
/// * `cluster_labels` - Reference of usizes of length n_cells that does
///   cluster membership.
///
/// ### Returns
///
/// A Vec of [ScTypeClusterAssignment].
pub fn assign_clusters(
    res: &SctypeRes,
    cluster_labels: &[usize],
) -> Result<Vec<ScTypeClusterAssignment>, BixverseErrors> {
    if cluster_labels.len() != res.n_cells {
        return Err(BixverseErrors::ScTypeClusterAssignmentNotEqualNCells {
            n_cells: res.n_cells,
            n_cluster_assignment: cluster_labels.len(),
        });
    }

    let n_ct = res.n_cell_types;
    let n_clusters = cluster_labels
        .iter()
        .max()
        .copied()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut sums = vec![0.0_f32; n_clusters * n_ct];
    let mut counts = vec![0_usize; n_clusters];

    for (c, &cl) in cluster_labels.iter().enumerate() {
        counts[cl] += 1;
        let row = &res.scores[c * n_ct..(c + 1) * n_ct];
        let dst = &mut sums[cl * n_ct..(cl + 1) * n_ct];
        for (d, &v) in dst.iter_mut().zip(row.iter()) {
            *d += v;
        }
    }

    let mut out = Vec::with_capacity(n_clusters);
    for cl in 0..n_clusters {
        if counts[cl] == 0 {
            continue;
        }
        let row = &sums[cl * n_ct..(cl + 1) * n_ct];
        let (best_idx, &best_score) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let threshold = counts[cl] as f32 / 4.0;
        let cell_type = if best_score < threshold {
            "Unknown".to_string()
        } else {
            res.cell_types[best_idx].clone()
        };

        out.push(ScTypeClusterAssignment {
            cluster: cl,
            cell_type,
            score: best_score,
            n_cells: counts[cl],
        });
    }

    Ok(out)
}

//////////////////////
// Per-cell scoring //
//////////////////////

/// Rescale the ScType score matrix ahead of a per-cell argmax.
///
/// [ScoreCalibration::ColumnZ] standardises each cell type's score column across
/// cells. The raw scores are `sum(z) / sqrt(n_markers)`, so cell types with
/// differently sized or differently correlated marker sets end up on different
/// scales and the argmax drifts towards whichever type has the larger spread.
/// Columns with a degenerate standard deviation are zeroed rather than divided
/// through.
///
/// Accumulation runs in `f64`: at millions of cells an `f32` sum of squares
/// loses too much to cancellation.
///
/// ### Params
///
/// * `scores` - Row-major scores (cells x cell_types).
/// * `n_cells` - Number of cells.
/// * `n_ct` - Number of cell types.
/// * `calibration` - The [ScoreCalibration] to apply.
///
/// ### Returns
///
/// A new row-major score matrix of the same shape.
fn calibrate_scores(
    scores: &[f32],
    n_cells: usize,
    n_ct: usize,
    calibration: ScoreCalibration,
) -> Vec<f32> {
    if calibration == ScoreCalibration::None || n_cells < 2 || n_ct == 0 {
        return scores.to_vec();
    }

    // Fold over rows rather than columns: the matrix is row-major, so a
    // per-column pass would stride through memory with n_ct spacing.
    let (sums, sum_sqs) = scores
        .par_chunks(n_ct)
        .fold(
            || (vec![0.0_f64; n_ct], vec![0.0_f64; n_ct]),
            |(mut sums, mut sum_sqs), row| {
                for (ct, &v) in row.iter().enumerate() {
                    let v = v as f64;
                    sums[ct] += v;
                    sum_sqs[ct] += v * v;
                }
                (sums, sum_sqs)
            },
        )
        .reduce(
            || (vec![0.0_f64; n_ct], vec![0.0_f64; n_ct]),
            |(mut sums_a, mut sq_a), (sums_b, sq_b)| {
                for ct in 0..n_ct {
                    sums_a[ct] += sums_b[ct];
                    sq_a[ct] += sq_b[ct];
                }
                (sums_a, sq_a)
            },
        );

    let n = n_cells as f64;
    let stats: Vec<(f32, f32)> = (0..n_ct)
        .map(|ct| {
            let mean = sums[ct] / n;
            let var = (sum_sqs[ct] - sums[ct] * mean) / (n - 1.0);
            let sd = var.max(0.0).sqrt();
            if sd < MIN_SD {
                (mean as f32, 0.0)
            } else {
                (mean as f32, (1.0 / sd) as f32)
            }
        })
        .collect();

    let mut out = scores.to_vec();
    out.par_chunks_mut(n_ct).for_each(|row| {
        for (ct, v) in row.iter_mut().enumerate() {
            let (mean, inv_sd) = stats[ct];
            *v = (*v - mean) * inv_sd;
        }
    });

    out
}

/// Smooth per-cell scores over a cell-cell graph.
///
/// Iterates `y = alpha * y0 + (1 - alpha) * W y`, the label-spreading update of
/// Zhou et al. (2004), with `W` the row-normalised adjacency the
/// [KnnLabPropGraph] builders already produce. Anchoring on the original scores
/// `y0` rather than the previous iterate keeps a cell's own evidence in play, so
/// a small population inside a large cluster is not averaged away.
///
/// This deliberately does not call [KnnLabPropGraph::label_spreading]: that one
/// takes `&[Vec<T>]`, which at millions of cells costs a `Vec` header per cell on
/// top of the data and doubles again for its scratch buffer. Here the kernel runs
/// straight over the flat row-major matrix and the graph's CSR arrays.
///
/// Cells with no neighbours (isolated after graph pruning) keep their original
/// scores instead of being shrunk towards zero.
///
/// ### Params
///
/// * `scores` - Row-major scores (cells x cell_types).
/// * `n_ct` - Number of cell types.
/// * `graph` - The cell-cell graph. Node count must equal the cell count.
/// * `alpha` - Self-retention, between 0 and 1. Higher keeps more of the cell's
///   own signal.
/// * `iterations` - Maximum number of smoothing iterations.
/// * `tolerance` - Stops early once the largest change across all cells and cell
///   types drops below this.
///
/// ### Returns
///
/// A new row-major score matrix of the same shape.
///
/// ### References
///
/// Zhou et al., Advances in Neural Information Processing Systems, 2004
pub fn smooth_scores(
    scores: &[f32],
    n_ct: usize,
    graph: &KnnLabPropGraph<f32>,
    alpha: f32,
    iterations: usize,
    tolerance: f32,
) -> Result<Vec<f32>, BixverseErrors> {
    let n_cells = scores.len().checked_div(n_ct).unwrap_or(0);
    let n_nodes = graph.offsets.len().saturating_sub(1);

    if n_nodes != n_cells {
        return Err(BixverseErrors::ScTypeGraphNodesNotEqualNCells { n_cells, n_nodes });
    }

    if iterations == 0 || n_cells == 0 {
        return Ok(scores.to_vec());
    }

    let mut current = scores.to_vec();
    let mut next = vec![0.0_f32; scores.len()];
    let beta = 1.0 - alpha;

    for _ in 0..iterations {
        let max_change = next
            .par_chunks_mut(n_ct)
            .enumerate()
            .map(|(cell, row)| {
                let start = graph.offsets[cell];
                let end = graph.offsets[cell + 1];
                let origin = &scores[cell * n_ct..(cell + 1) * n_ct];
                let previous = &current[cell * n_ct..(cell + 1) * n_ct];

                if start == end {
                    row.copy_from_slice(origin);
                    return 0.0_f32;
                }

                row.fill(0.0);
                for edge in start..end {
                    let w = graph.weights[edge];
                    let nb = graph.neighbours[edge];
                    let nb_row = &current[nb * n_ct..(nb + 1) * n_ct];
                    for (acc, &v) in row.iter_mut().zip(nb_row.iter()) {
                        *acc += w * v;
                    }
                }

                let mut max_diff = 0.0_f32;
                for ct in 0..n_ct {
                    row[ct] = alpha * origin[ct] + beta * row[ct];
                    max_diff = max_diff.max((row[ct] - previous[ct]).abs());
                }
                max_diff
            })
            .reduce(|| 0.0_f32, f32::max);

        std::mem::swap(&mut current, &mut next);

        if max_change < tolerance {
            break;
        }
    }

    Ok(current)
}

/// Assign a cell type to every cell.
///
/// Calibrates, then smooths, then takes the argmax. The order matters: smoothing
/// averages scores across neighbouring cells, so the columns have to be on a
/// comparable scale first.
///
/// Without a graph this is a raw per-cell argmax, which on typical scRNA-seq
/// depth is noisy enough that the cluster-level call usually wins. Pass the same
/// sNN graph the clustering ran on and the noise largely goes away. Build it with
/// [KnnLabPropGraph::from_weighted_edge_list] from the `(edges, weights)` pair
/// that `generate_snn_full` returns, or
/// [KnnLabPropGraph::from_weighted_node_pairs] when the edge list arrives as
/// separate from/to vectors.
///
/// ### Params
///
/// * `res` - Reference to the [SctypeRes].
/// * `graph` - Optional cell-cell graph used for smoothing and for the neighbour
///   agreement statistic.
/// * `params` - Optional [ScTypeCellParams]. Defaults via `unwrap_or_default`.
///
/// ### Returns
///
/// The [ScTypeCellRes].
pub fn assign_cells(
    res: &SctypeRes,
    graph: Option<&KnnLabPropGraph<f32>>,
    params: Option<ScTypeCellParams>,
) -> Result<ScTypeCellRes, BixverseErrors> {
    let params = params.unwrap_or_default();
    let n_cells = res.n_cells;
    let n_ct = res.n_cell_types;

    let calibrated = calibrate_scores(&res.scores, n_cells, n_ct, params.calibration);

    let final_scores = match graph {
        Some(g) => smooth_scores(
            &calibrated,
            n_ct,
            g,
            params.alpha,
            params.iterations,
            params.tolerance,
        )?,
        None => calibrated,
    };

    let mut assignments: Vec<Option<usize>> = vec![None; n_cells];
    let mut scores = vec![0.0_f32; n_cells];
    let mut margins = vec![0.0_f32; n_cells];

    // Fused scan: best, second best and the winning index in one pass per row.
    assignments
        .par_iter_mut()
        .zip(scores.par_iter_mut())
        .zip(margins.par_iter_mut())
        .zip(final_scores.par_chunks(n_ct))
        .for_each(|(((assignment, score), margin), row)| {
            let mut best = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            let mut best_idx = 0_usize;

            for (ct, &v) in row.iter().enumerate() {
                if v > best {
                    second = best;
                    best = v;
                    best_idx = ct;
                } else if v > second {
                    second = v;
                }
            }

            *score = best;
            // A single cell type has no competitor, so the margin is unbounded.
            *margin = if second.is_finite() {
                best - second
            } else {
                f32::INFINITY
            };
            *assignment = (best >= params.score_floor).then_some(best_idx);
        });

    let agreement = graph.map(|g| {
        (0..n_cells)
            .into_par_iter()
            .map(|cell| {
                let start = g.offsets[cell];
                let end = g.offsets[cell + 1];
                if start == end {
                    // Isolated cell, trivially agrees with itself.
                    return 1.0;
                }
                let own = assignments[cell];
                let matching = (start..end)
                    .filter(|&edge| assignments[g.neighbours[edge]] == own)
                    .count();
                matching as f32 / (end - start) as f32
            })
            .collect()
    });

    Ok(ScTypeCellRes {
        assignments,
        scores,
        margins,
        agreement,
        cell_types: res.cell_types.clone(),
    })
}

/// Break down each cluster into its per-cell cell type calls.
///
/// This is the diagnostic for mixed clusters: a cluster that looks clean at the
/// aggregate level but comes back with a purity of 0.6 and a `second_fraction`
/// of 0.35 is two populations sharing one Leiden community.
///
/// `dominant` and `second` are picked among the cell types only, so a cluster
/// that is mostly Unknown still reports the best-supported real label. The
/// fractions and the entropy however use the full cluster size and treat Unknown
/// as its own category: a cluster half of which cannot be annotated is not a pure
/// cluster and should not be handed a blanket label.
///
/// Empty clusters are skipped, matching [assign_clusters].
///
/// ### Params
///
/// * `cell_res` - Reference to the [ScTypeCellRes].
/// * `cluster_labels` - Cluster membership, of length n_cells.
///
/// ### Returns
///
/// A Vec of [ScTypeClusterComposition], in ascending cluster order.
pub fn cluster_composition(
    cell_res: &ScTypeCellRes,
    cluster_labels: &[usize],
) -> Result<Vec<ScTypeClusterComposition>, BixverseErrors> {
    let n_cells = cell_res.assignments.len();
    if cluster_labels.len() != n_cells {
        return Err(BixverseErrors::ScTypeClusterAssignmentNotEqualNCells {
            n_cells,
            n_cluster_assignment: cluster_labels.len(),
        });
    }

    let n_ct = cell_res.cell_types.len();
    let n_clusters = cluster_labels
        .iter()
        .max()
        .copied()
        .map(|m| m + 1)
        .unwrap_or(0);

    // Sequential scatter: one increment per cell and no arithmetic, so the
    // per-thread histograms a parallel version needs buy nothing here.
    let mut hist = vec![0_usize; n_clusters * n_ct];
    let mut counts = vec![0_usize; n_clusters];
    let mut unknowns = vec![0_usize; n_clusters];

    for (cell, &cl) in cluster_labels.iter().enumerate() {
        counts[cl] += 1;
        match cell_res.assignments[cell] {
            Some(ct) => hist[cl * n_ct + ct] += 1,
            None => unknowns[cl] += 1,
        }
    }

    let out = (0..n_clusters)
        .into_par_iter()
        .filter(|&cl| counts[cl] > 0)
        .map(|cl| {
            let row = &hist[cl * n_ct..(cl + 1) * n_ct];
            let total = counts[cl] as f32;

            let mut best = (0_usize, 0_usize);
            let mut second = (0_usize, 0_usize);
            for (ct, &c) in row.iter().enumerate() {
                if c > best.1 {
                    second = best;
                    best = (ct, c);
                } else if c > second.1 {
                    second = (ct, c);
                }
            }

            let entropy: f64 = row
                .iter()
                .chain(std::iter::once(&unknowns[cl]))
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / counts[cl] as f64;
                    -p * p.ln()
                })
                .sum();

            ScTypeClusterComposition {
                cluster: cl,
                n_cells: counts[cl],
                counts: row.to_vec(),
                n_unknown: unknowns[cl],
                dominant: (best.1 > 0).then_some(best.0),
                purity: best.1 as f32 / total,
                entropy: entropy as f32,
                second: (second.1 > 0).then_some(second.0),
                second_fraction: second.1 as f32 / total,
            }
        })
        .collect();

    Ok(out)
}

/// Hybrid cluster-level and per-cell ScType annotation.
///
/// Clusters at or above `purity_threshold` keep their cluster-level call, which
/// is the robust option when the cluster really is one population. Anything below
/// falls through to the per-cell calls, which is what rescues a minority
/// population sharing a Leiden community with a larger one.
///
/// ### Params
///
/// * `res` - Reference to the [SctypeRes].
/// * `cell_res` - Reference to the [ScTypeCellRes] from [assign_cells].
/// * `cluster_labels` - Cluster membership, of length n_cells.
/// * `purity_threshold` - Optional purity cut-off. If not provided, defaults to
///   `PURITY_THRESHOLD` (0.9).
///
/// ### Returns
///
/// The [ScTypeHybridRes].
pub fn assign_hybrid(
    res: &SctypeRes,
    cell_res: &ScTypeCellRes,
    cluster_labels: &[usize],
    purity_threshold: Option<f32>,
) -> Result<ScTypeHybridRes, BixverseErrors> {
    if cell_res.assignments.len() != res.n_cells {
        return Err(BixverseErrors::ScTypeClusterAssignmentNotEqualNCells {
            n_cells: res.n_cells,
            n_cluster_assignment: cell_res.assignments.len(),
        });
    }

    let purity_threshold = purity_threshold.unwrap_or(PURITY_THRESHOLD);

    let cluster_assignments = assign_clusters(res, cluster_labels)?;
    let composition = cluster_composition(cell_res, cluster_labels)?;

    let n_clusters = cluster_labels
        .iter()
        .max()
        .copied()
        .map(|m| m + 1)
        .unwrap_or(0);

    let ct_index: FxHashMap<&str, usize> = res
        .cell_types
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // Empty clusters never appear in either vector, so they stay pure and
    // unlabelled. No cell references them anyway.
    let mut cluster_mixed = vec![false; n_clusters];
    for comp in &composition {
        cluster_mixed[comp.cluster] = comp.purity < purity_threshold;
    }

    let mut cluster_call: Vec<Option<usize>> = vec![None; n_clusters];
    for assignment in &cluster_assignments {
        cluster_call[assignment.cluster] = ct_index.get(assignment.cell_type.as_str()).copied();
    }

    let assignments = cluster_labels
        .par_iter()
        .enumerate()
        .map(|(cell, &cl)| {
            if cluster_mixed[cl] {
                cell_res.assignments[cell]
            } else {
                cluster_call[cl]
            }
        })
        .collect();

    Ok(ScTypeHybridRes {
        assignments,
        composition,
        cluster_mixed,
        cell_types: res.cell_types.clone(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a [SctypeRes] where every cell scores `2.0` for its own cell type
    /// and `0.0` for all others.
    ///
    /// ### Params
    ///
    /// * `cell_type_per_cell` - The true cell type index of each cell.
    /// * `n_ct` - Number of cell types.
    ///
    /// ### Returns
    ///
    /// The synthetic [SctypeRes].
    fn one_hot_res(cell_type_per_cell: &[usize], n_ct: usize) -> SctypeRes {
        let n_cells = cell_type_per_cell.len();
        let mut scores = vec![0.0_f32; n_cells * n_ct];
        for (cell, &ct) in cell_type_per_cell.iter().enumerate() {
            scores[cell * n_ct + ct] = 2.0;
        }
        SctypeRes {
            cell_types: (0..n_ct).map(|i| format!("type_{i}")).collect(),
            scores,
            n_cells,
            n_cell_types: n_ct,
        }
    }

    /// Fully connected edge list over `[start, start + size)`.
    ///
    /// ### Params
    ///
    /// * `start` - First node index.
    /// * `size` - Number of nodes in the clique.
    ///
    /// ### Returns
    ///
    /// A flat alternating pair vector of edges.
    fn clique_edges(start: usize, size: usize) -> Vec<usize> {
        let mut edges = Vec::with_capacity(size * (size - 1));
        for i in 0..size {
            for j in (i + 1)..size {
                edges.push(start + i);
                edges.push(start + j);
            }
        }
        edges
    }

    #[test]
    fn test_sctype_smoothing_two_cliques() {
        // Two 5-cliques. Cell 4 sits in clique A but carries clique B's profile.
        let res = one_hot_res(&[0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 2);

        let mut edges = clique_edges(0, 5);
        edges.extend(clique_edges(5, 5));
        let graph = KnnLabPropGraph::<f32>::from_edge_list(&edges, 10, true);

        let params = ScTypeCellParams::new(Some(0.3), Some(3), None, None, None, None);
        let cells = assign_cells(&res, Some(&graph), Some(params)).unwrap();

        // Four unambiguous neighbours drag the odd cell back to its clique.
        for cell in 0..5 {
            assert_eq!(cells.assignments[cell], Some(0));
        }
        for cell in 5..10 {
            assert_eq!(cells.assignments[cell], Some(1));
        }

        // The rescued cell is the least confident of its clique.
        let margins = &cells.margins;
        assert!(margins[4] < margins[0]);

        // Its neighbours all disagree with the raw call, which is the tell.
        let raw = assign_cells(&res, None, None).unwrap();
        assert_eq!(raw.assignments[4], Some(1));
        assert!(raw.agreement.is_none());

        let raw_with_graph = assign_cells(
            &res,
            Some(&graph),
            Some(ScTypeCellParams::new(None, Some(0), None, None, None, None)),
        )
        .unwrap();
        assert_relative_eq!(raw_with_graph.agreement.unwrap()[4], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sctype_smoothing_isolated_node() {
        let res = one_hot_res(&[0, 0, 1], 2);
        // Only cells 0 and 1 are connected, cell 2 is on its own.
        let graph = KnnLabPropGraph::<f32>::from_edge_list(&[0, 1], 3, true);

        let smoothed = smooth_scores(&res.scores, 2, &graph, 0.5, 5, SMOOTH_TOLERANCE).unwrap();

        assert_relative_eq!(smoothed[4], 0.0, epsilon = 1e-6);
        assert_relative_eq!(smoothed[5], 2.0, epsilon = 1e-6);

        // An isolated cell still counts as agreeing with itself.
        let cells = assign_cells(&res, Some(&graph), None).unwrap();
        assert_relative_eq!(cells.agreement.unwrap()[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sctype_column_z_calibration() {
        // Column 0 varies, column 1 is constant and must not blow up.
        let scores = vec![1.0, 10.0, 2.0, 10.0, 3.0, 10.0, 4.0, 10.0];
        let out = calibrate_scores(&scores, 4, 2, ScoreCalibration::ColumnZ);

        let col: Vec<f64> = out.iter().step_by(2).map(|&v| v as f64).collect();
        let mean = col.iter().sum::<f64>() / 4.0;
        let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3.0).sqrt();

        assert_relative_eq!(mean, 0.0, epsilon = 1e-6);
        assert_relative_eq!(sd, 1.0, epsilon = 1e-6);

        for v in out.iter().skip(1).step_by(2) {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }

        // None is a pass-through.
        let raw = calibrate_scores(&scores, 4, 2, ScoreCalibration::None);
        assert_eq!(raw, scores);
    }

    #[test]
    fn test_sctype_composition_pure_cluster() {
        let res = one_hot_res(&[0, 0, 0, 0], 2);
        let cells = assign_cells(&res, None, None).unwrap();
        let comp = cluster_composition(&cells, &[0, 0, 0, 0]).unwrap();

        assert_eq!(comp.len(), 1);
        assert_eq!(comp[0].n_cells, 4);
        assert_eq!(comp[0].dominant, Some(0));
        assert_eq!(comp[0].second, None);
        assert_eq!(comp[0].n_unknown, 0);
        assert_relative_eq!(comp[0].purity, 1.0, epsilon = 1e-6);
        assert_relative_eq!(comp[0].entropy, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sctype_composition_mixed_cluster() {
        let res = one_hot_res(&[0, 0, 1, 1], 2);
        let cells = assign_cells(&res, None, None).unwrap();
        let comp = cluster_composition(&cells, &[0, 0, 0, 0]).unwrap();

        assert_eq!(comp[0].dominant, Some(0));
        assert_eq!(comp[0].second, Some(1));
        assert_relative_eq!(comp[0].purity, 0.5, epsilon = 1e-6);
        assert_relative_eq!(comp[0].second_fraction, 0.5, epsilon = 1e-6);
        assert_relative_eq!(comp[0].entropy, std::f32::consts::LN_2, epsilon = 1e-6);
    }

    #[test]
    fn test_sctype_composition_counts_unknown() {
        // Scores below the floor never get a call.
        let res = SctypeRes {
            cell_types: vec!["a".to_string(), "b".to_string()],
            scores: vec![2.0, 0.0, 0.01, 0.0],
            n_cells: 2,
            n_cell_types: 2,
        };
        let cells = assign_cells(&res, None, None).unwrap();
        assert_eq!(cells.assignments, vec![Some(0), None]);

        let comp = cluster_composition(&cells, &[0, 0]).unwrap();
        assert_eq!(comp[0].n_unknown, 1);
        assert_relative_eq!(comp[0].purity, 0.5, epsilon = 1e-6);
        // Unknown counts as its own category in the entropy.
        assert_relative_eq!(comp[0].entropy, std::f32::consts::LN_2, epsilon = 1e-6);
    }

    #[test]
    fn test_sctype_hybrid_keeps_pure_cluster() {
        // 19 of 20 cells agree, purity 0.95, so the cluster call wins outright.
        let mut labels = vec![0_usize; 20];
        labels[19] = 1;
        let res = one_hot_res(&labels, 2);
        let cells = assign_cells(&res, None, None).unwrap();
        assert_eq!(cells.assignments[19], Some(1));

        let clusters = vec![0_usize; 20];
        let hybrid = assign_hybrid(&res, &cells, &clusters, None).unwrap();

        assert_eq!(hybrid.cluster_mixed, vec![false]);
        assert!(hybrid.assignments.iter().all(|&a| a == Some(0)));
    }

    #[test]
    fn test_sctype_hybrid_splits_mixed_cluster() {
        // 7/3 split, purity 0.7, below the threshold, so the per-cell calls stand.
        let labels = vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1];
        let res = one_hot_res(&labels, 2);
        let cells = assign_cells(&res, None, None).unwrap();

        let clusters = vec![0_usize; 10];
        let hybrid = assign_hybrid(&res, &cells, &clusters, None).unwrap();

        assert_eq!(hybrid.cluster_mixed, vec![true]);
        assert_eq!(hybrid.assignments, cells.assignments);
        assert_relative_eq!(hybrid.composition[0].purity, 0.7, epsilon = 1e-6);

        // Raising the threshold above the purity is what flips the behaviour.
        let strict = assign_hybrid(&res, &cells, &clusters, Some(0.5)).unwrap();
        assert_eq!(strict.cluster_mixed, vec![false]);
        assert!(strict.assignments.iter().all(|&a| a == Some(0)));
    }

    #[test]
    fn test_sctype_graph_size_mismatch_errors() {
        let res = one_hot_res(&[0, 0, 1], 2);
        let graph = KnnLabPropGraph::<f32>::from_edge_list(&[0, 1], 2, true);

        let err = assign_cells(&res, Some(&graph), None).unwrap_err();
        assert!(matches!(
            err,
            BixverseErrors::ScTypeGraphNodesNotEqualNCells {
                n_cells: 3,
                n_nodes: 2
            }
        ));
    }

    #[test]
    fn test_sctype_cluster_labels_length_mismatch_errors() {
        let res = one_hot_res(&[0, 0, 1], 2);
        let cells = assign_cells(&res, None, None).unwrap();

        assert!(cluster_composition(&cells, &[0, 0]).is_err());
    }
}
