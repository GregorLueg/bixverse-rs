//! ScType cell type annotation, streaming over gene chunks. Based on Ianevski
//! et al. (2022).

use indexmap::IndexSet;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_processing::pca::scale_csc_chunk;

////////////
// Consts //
////////////

/// The batch size for the genes
const GENE_BATCH_SIZE: usize = 100;

/// Minimum weight a given marker can have
const WEIGHT_FLOOR: f32 = 0.1;

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
