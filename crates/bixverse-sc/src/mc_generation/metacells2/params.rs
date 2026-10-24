//! Parameter structs for the MetaCells2 pipeline.
//!
//! The hierarchy mirrors the algorithm stages: each stage's parameters live in
//! their own struct, composed into the top-level [`MetacellsParams`]. Defaults
//! are upstream's (`metacells/parameters.py`).

/////////////////
// Main params //
/////////////////

/// Top-level parameters for the MC2 algorithm.
///
/// Composed of one struct per pipeline stage plus a few cross-cutting knobs
/// (target metacell size/UMI count, must-complete-cover mode). The RNG seed is
/// passed to the entry points separately.
#[derive(Clone, Debug)]
pub struct MetacellsParams {
    /// The [SelectParams] for feature selections.
    pub select: SelectParams,
    /// The [SimilarityParams] for the MetaCell similarity
    pub similarity: SimilarityParams,
    /// The [MC2KnnParams] - these are MetaCell2-specific parameters that are
    /// different from [`crate::prelude::KnnParams`].
    pub knn: MC2KnnParams,
    /// The [PartitionParams] that defines how to partition the cells into the
    /// meta cells.
    pub partition: PartitionParams,
    /// The [DeviantsParams] that mark genes/cells as deviant
    pub deviants: DeviantsParams,
    /// The [DissolveParams] that dissolve small, unconvincing candidates.
    pub dissolve: DissolveParams,
    /// Target cell count per metacell.
    pub target_metacell_size: usize,
    /// Below this, a candidate metacell is dissolved to outliers regardless of
    /// UMI counts or convincing genes.
    pub min_metacell_size: usize,
    /// Target total UMIs per metacell. Drives seed count and band penalties.
    pub target_metacell_umis: u64,
    /// When true, deviant detection and dissolution are bypassed and every cell
    /// ends up in some metacell. Required by the divide-and-conquer preliminary
    /// phase, where piles are random and "outliers" would just be rare types
    /// stuck in the wrong pile.
    pub must_complete_cover: bool,
}

/// Default for [MetacellsParams]
impl Default for MetacellsParams {
    fn default() -> Self {
        Self {
            select: SelectParams::default(),
            similarity: SimilarityParams::default(),
            knn: MC2KnnParams::default(),
            partition: PartitionParams::default(),
            deviants: DeviantsParams::default(),
            dissolve: DissolveParams::default(),
            target_metacell_size: 48,
            min_metacell_size: 12,
            target_metacell_umis: 160_000,
            must_complete_cover: false,
        }
    }
}

//////////////////////
// Selection params //
//////////////////////

/// Feature gene selection and per-pile downsampling parameters.
#[derive(Clone, Debug)]
pub struct SelectParams {
    /// Floor for the downsampling target. Even very-low-UMI piles get at least
    /// this many samples per cell (subject to caps from `max_quantile`).
    pub downsample_min_samples: u32,
    /// Lower quantile of cell library sizes used to clamp the downsample
    /// target from below.
    pub downsample_min_cell_quantile: f32,
    /// Upper quantile of cell library sizes used to clamp the downsample target
    /// from above.
    pub downsample_max_cell_quantile: f32,
    /// Minimum total UMI count for a gene to pass the "high total" filter.
    /// `None` skips the filter.
    pub min_gene_total: Option<u32>,
    /// Minimum value for a gene's 3rd-highest cell count to pass the
    /// "high top-3" filter. `None` skips.
    pub min_gene_top3: Option<u32>,
    /// Minimum windowed relative variance (variance/mean normalised against
    /// genes with similar mean) for a gene to pass the "high relative
    /// variance" filter. `None` skips.
    pub min_gene_relative_variance: Option<f32>,
    /// Floor on the total number of selected genes. If filters yield fewer,
    /// the relative-variance threshold is binary-searched downward; if still
    /// short, the variance filter is dropped entirely.
    pub min_genes: usize,
    /// Window size (in rank order of gene mean) for the relative-variance
    /// median normalisation. Each gene's relative variance is its normalised
    /// variance minus the median normalised variance over its window.
    pub relative_variance_window_size: usize,
    /// Optional mask of genes excluded from selection (e.g. lateral genes
    /// like cell-cycle markers). v1 leaves this unwired but the field is
    /// here so the API doesn't churn when it's added.
    pub lateral_gene_mask: Option<Vec<bool>>,
}

/// Default implementation for [SelectParams]
impl Default for SelectParams {
    fn default() -> Self {
        Self {
            downsample_min_samples: 750,
            downsample_min_cell_quantile: 0.05,
            downsample_max_cell_quantile: 0.5,
            min_gene_total: Some(50),
            min_gene_top3: Some(4),
            min_gene_relative_variance: Some(0.1),
            min_genes: 100,
            relative_variance_window_size: 100, // TODO: confirm against metacells.parameters
            lateral_gene_mask: None,
        }
    }
}

////////////////////////
// Similiarity params //
////////////////////////

/// Cell-cell similarity computation method.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SimilarityMethod {
    /// Absolute Pearson correlation. Upstream's default (`abs_pearson`).
    #[default]
    AbsPearson,
    /// Signed Pearson correlation (`pearson`).
    Pearson,
}

/// Cell-cell similarity computation parameters.
#[derive(Clone, Debug)]
pub struct SimilarityParams {
    /// Enum defining the chosen similarity method, see [SimilarityMethod].
    pub method: SimilarityMethod,
    /// If true, correlate `log2(x + value_regularisation)` rather than
    /// `x + value_regularisation`.
    pub log_data: bool,
    /// Constant added to every value before the optional log transform.
    pub value_regularisation: f32,
}

/// Default implementation for [SimilarityParams]
impl Default for SimilarityParams {
    fn default() -> Self {
        Self {
            method: SimilarityMethod::AbsPearson,
            log_data: true,
            value_regularisation: 1.0 / 7.0,
        }
    }
}

/////////////////////////////
// MC2-specific kNN params //
/////////////////////////////

/// Balanced KNN graph construction parameters.
#[derive(Clone, Debug)]
pub struct MC2KnnParams {
    /// Edges with balanced rank above `k * balanced_ranks_factor` are dropped.
    pub balanced_ranks_factor: f64,
    /// Multiplier on `k` for the per-cell incoming-edge cap.
    pub incoming_degree_factor: f64,
    /// Multiplier on `k` for the per-cell outgoing-edge cap.
    pub outgoing_degree_factor: f64,
    /// Minimum outgoing degree of every cell after pruning. If any cell falls
    /// below it, `k` is raised by 10% and the graph rebuilt, as upstream.
    pub min_outgoing_degree: usize,
    /// Multiplier in the `k` heuristic:
    /// `target_metacell_umis / median_umis * k_size_factor`.
    pub k_size_factor: f64,
    /// Quantile of cell library sizes used in the `k` heuristic via
    /// `target_metacell_umis / quantile_umis`.
    pub k_umis_quantile: f64,
    /// Optional floor on `k`. `None` defers entirely to the heuristic.
    pub min_knn_k: Option<usize>,
    /// If you wish to override the kNN parameter
    pub knn_k_override: Option<usize>,
}

/// Default implementation for [MC2KnnParams]
impl Default for MC2KnnParams {
    fn default() -> Self {
        Self {
            balanced_ranks_factor: 10_f64.sqrt(),
            incoming_degree_factor: 3.0,
            outgoing_degree_factor: 1.0,
            min_outgoing_degree: 2,
            k_size_factor: 2.0,
            k_umis_quantile: 0.1,
            min_knn_k: Some(30),
            knn_k_override: None,
        }
    }
}

//////////////////////
// Partition params //
//////////////////////

/// Simulated-annealing partition optimiser parameters.
#[derive(Clone, Debug)]
pub struct PartitionParams {
    /// Per-pass temperature decay rate.
    pub cooldown_pass: f64,
    /// Per-node temperature decay applied when a node fails to improve.
    pub cooldown_node: f64,
    /// Between-phase cooldown multiplier on `cooldown_pass`.
    pub cooldown_phase: f64,
    /// Multiplier on `target_metacell_size` above which a partition is
    /// considered too large and a min-cut split is attempted.
    pub min_split_size_factor: f64,
    /// Multiplier on `target_metacell_size` below which a partition is
    /// considered too small and is dissolved/reseeded.
    pub max_merge_size_factor: f64,
    /// Maximum allowed min-cut strength for a split to be accepted.
    pub max_split_min_cut_strength: f64,
    /// Minimum cells per side of a min-cut split.
    pub min_cut_seed_cells: usize,
    /// Lower quantile of seed-region sizes considered "large enough" during
    /// initial seed placement.
    pub min_seed_size_quantile: f32,
    /// Upper quantile of seed-region sizes considered "large enough" during
    /// initial seed placement.
    pub max_seed_size_quantile: f32,
}

/// Default implementation for [PartitionParams]
impl Default for PartitionParams {
    fn default() -> Self {
        Self {
            cooldown_pass: 0.02,
            cooldown_node: 0.25,
            cooldown_phase: 0.75,
            min_split_size_factor: 2.0,
            max_merge_size_factor: 0.5,
            max_split_min_cut_strength: 0.1,
            min_cut_seed_cells: 7,
            min_seed_size_quantile: 0.85,
            max_seed_size_quantile: 0.95,
        }
    }
}

////////////////////
// Deviant params //
////////////////////

/// Adaptive deviant (outlier) detection parameters, upstream's `gaps` policy.
#[derive(Clone, Debug)]
pub struct DeviantsParams {
    /// Minimum log2 fold-factor gap (between cells sorted by log-fraction)
    /// within a candidate metacell for a gene to mark cells as deviant. Raised
    /// by 1/8 while more than `max_cell_fraction` of the cells are deviant.
    pub min_gene_fold_factor: f32,
    /// Added to `min_gene_fold_factor` for genes in `noisy_gene_mask`.
    pub min_noisy_gene_fold_factor: f32,
    /// Optional per-gene mask of noisy genes, which need a larger gap.
    pub noisy_gene_mask: Option<Vec<bool>>,
    /// Cap on the fraction of cells marked as deviants, in `(0, 1)`.
    pub max_cell_fraction: f32,
    /// Cells skipped over when measuring a gap, in `0..=2` as upstream's
    /// user-facing value: the gap is between sorted positions `i` and
    /// `i + gap_skip_cells + 1`.
    pub gap_skip_cells: usize,
    /// Maximum number of cells beyond a gap that are marked as deviants for
    /// a single gene.
    pub max_gap_cells_count: usize,
    /// Maximum fraction of cells beyond a gap. `0.0` means the absolute count
    /// (`max_gap_cells_count`) is used.
    pub max_gap_cells_fraction: f32,
    /// Minimum total UMIs (summed across the two cells being compared) for a
    /// gap to be considered. Filters out gaps caused by genes that are too
    /// lowly expressed to be reliably compared.
    pub min_compare_umis: f32,
    /// Quantile of the candidate's library size distribution used as a floor
    /// for the per-cell regularisation in
    /// `log2(fraction + 1/max(library_size, floor))`. A small positive value
    /// (e.g. 0.25) prevents extreme log values for the lowest-coverage cells.
    pub cells_regularisation_quantile: f32,
}

/// Default implementation for [DeviantsParams]
impl Default for DeviantsParams {
    fn default() -> Self {
        Self {
            min_gene_fold_factor: 3.0,
            min_noisy_gene_fold_factor: 2.0,
            noisy_gene_mask: None,
            max_cell_fraction: 0.25,
            gap_skip_cells: 1,
            max_gap_cells_count: 3,
            max_gap_cells_fraction: 0.1,
            min_compare_umis: 8.0,
            cells_regularisation_quantile: 0.25,
        }
    }
}

/////////////////////
// Dissolve params //
/////////////////////

/// Small/unconvincing metacell dissolution parameters.
#[derive(Clone, Debug)]
pub struct DissolveParams {
    /// Multiplier on `target_metacell_size` (and `target_metacell_umis`)
    /// defining the "robust" threshold. Metacells at or above this size or UMI
    /// count are kept unconditionally.
    pub min_robust_size_factor: f32,
    /// Minimum log2 fold-factor of any single gene above its expected
    /// expression for a sub-robust metacell to be kept anyway. `None` disables
    /// the convincing-gene rule, in which case all sub-robust metacells are
    /// kept.
    pub min_convincing_gene_fold_factor: Option<f32>,
}

/// Default implementation for [DissolveParams]
impl Default for DissolveParams {
    fn default() -> Self {
        Self {
            min_robust_size_factor: 0.5,
            min_convincing_gene_fold_factor: Some(3.0),
        }
    }
}

///////////////////////////////
// Divide and conquer params //
///////////////////////////////

/// Divide-and-conquer parameters, upstream's `DacParameters`.
///
/// The per-pile runs use [`MetacellsParams`]; these only control how cells
/// are piled and how the groups-of-metacells phase is configured.
#[derive(Clone, Debug)]
pub struct DacParams {
    /// Stop after the final phase's outliers instead of regrouping the
    /// outliers-of-outliers.
    pub quick_and_dirty: bool,
    /// Lower clamp on the computed target pile size.
    pub min_target_pile_size: usize,
    /// Upper clamp on the computed target pile size.
    pub max_target_pile_size: usize,
    /// Metacells wanted per pile; also the target group size of the groups
    /// phase.
    pub target_metacells_in_pile: usize,
    /// Fixed target pile size. `None` derives it from the data (upstream's
    /// `compute_target_pile_size`).
    pub target_pile_size: Option<usize>,
    /// `k_size_factor` of the groups phase.
    pub piles_knn_k_size_factor: f64,
    /// `min_split_size_factor` of the groups phase.
    pub piles_min_split_size_factor: f64,
    /// `min_robust_size_factor` of the groups phase.
    pub piles_min_robust_size_factor: f32,
    /// `max_merge_size_factor` of the groups phase.
    pub piles_max_merge_size_factor: f64,
    /// Similarity of the groups phase.
    pub groups_similarity: SimilarityParams,
    /// Piles computed concurrently. Each pile holds two dense
    /// `pile_size × pile_size` matrices while it runs, so this trades memory
    /// for throughput. Not tuned.
    pub max_parallel_piles: usize,
}

/// Default implementation for [DacParams]
impl Default for DacParams {
    fn default() -> Self {
        Self {
            quick_and_dirty: false,
            min_target_pile_size: 8000,
            max_target_pile_size: 24000,
            target_metacells_in_pile: 100,
            target_pile_size: None,
            piles_knn_k_size_factor: 3.0,
            piles_min_split_size_factor: 1.5,
            piles_min_robust_size_factor: 0.3,
            piles_max_merge_size_factor: 0.15,
            groups_similarity: SimilarityParams::default(),
            max_parallel_piles: 1,
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults other modules assume must not drift from the reference.
    #[test]
    fn metacells_defaults() {
        // Only the constants other modules assume. These are the published
        // metacells2 defaults (Ben-Kiki et al., Genome Biology 2022), so a
        // change here is a change in behaviour against the reference.
        let p = MetacellsParams::default();
        assert_eq!(p.target_metacell_size, 48);
        assert_eq!(p.min_metacell_size, 12);
        assert_eq!(p.target_metacell_umis, 160_000);
        assert_eq!(p.knn.balanced_ranks_factor, 10_f64.sqrt());
    }
}
