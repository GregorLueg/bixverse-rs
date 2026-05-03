//! This module contains all of the parameter structures needed for MetaCells2.
//! Parameter structs for the MetaCells2 pipeline.
//!
//! The hierarchy mirrors the algorithm stages: each stage's parameters live in
//! their own struct, composed into the top-level [`MetacellsParams`].

/////////////////
// Main params //
/////////////////

/// Top-level parameters for the MC2 algorithm.
///
/// Composed of one struct per pipeline stage plus a few cross-cutting knobs
/// (target metacell size/UMI count, must-complete-cover mode, seed).
#[derive(Clone, Debug)]
pub struct MetacellsParams {
    /// The [SelectParams] for feature selections.
    pub select: SelectParams,
    /// The [SimilarityParams] for the MetaCell similarity
    pub similarity: SimilarityParams,
    /// The [MC2KnnParams] - these are MetaCell2-specific parameters that are
    /// different from [`KnnParams`].
    pub knn: MC2KnnParams,
    /// The [PartitionParams] that defines how to partition the cells into the
    /// meta cells.
    pub partition: PartitionParams,
    /// The [DeviantsParams] that mark genes/cells as deviant
    pub deviants: DeviantsParams,
    /// The [DissolveParams] that TODO
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
    /// Master seed. Per-row, per-cell, per-pile seeds are derived from this.
    pub random_seed: u64,
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
            target_metacell_size: 96,
            min_metacell_size: 12,
            target_metacell_umis: 160_000,
            must_complete_cover: false,
            random_seed: 0,
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
            min_genes: 30,
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
    /// ln1p transform, then Pearson correlation. MC2 default. (They use log2
    /// under the hood).
    #[default]
    LogPearson,
    /// Plain Pearson correlation on raw values. Will behave badly without a log
    /// transform — exposed mainly for parity testing.
    Pearson,
    /// Spearman rank correlation. Non-parametric; pays a per-row sort but
    /// avoids the log/Pearson assumption mismatch.
    Spearman,
}

/// Cell-cell similarity computation parameters.
#[derive(Clone, Debug)]
pub struct SimilarityParams {
    /// Enum defining the chosen similarity method, see [SimilarityMethod].
    pub method: SimilarityMethod,
    /// Constant added to every value before the (optional) log transform.
    /// Mainly relevant if you ever want to use plain Pearson without log.
    pub value_regularization: f32,
}

/// Default implementation for [SimilarityParams]
impl Default for SimilarityParams {
    fn default() -> Self {
        Self {
            method: SimilarityMethod::LogPearson,
            value_regularization: 0.0,
        }
    }
}

/////////////////////////////
// MC2-specific kNN params //
/////////////////////////////

/// Balanced KNN graph construction parameters.
#[derive(Clone, Debug)]
pub struct MC2KnnParams {
    /// Multiplier on `k` for the per-row top-K of outgoing ranks before
    /// symmetrisation by geometric mean. Edges with balanced rank above
    /// `k * balanced_ranks_factor` are pruned.
    pub balanced_ranks_factor: f32,
    /// Multiplier on `k` for the per-cell incoming-edge cap.
    pub incoming_degree_factor: f32,
    /// Multiplier on `k` for the per-cell outgoing-edge cap.
    pub outgoing_degree_factor: f32,
    /// Floor on per-cell outgoing edges; the heaviest outgoing edge per cell
    /// is preserved even if pruning would drop it.
    pub min_outgoing_degree: usize,
    /// Multiplier in the `k` heuristic:
    /// `target_metacell_umis / median_umis * k_size_factor`.
    pub k_size_factor: f32,
    /// Quantile of cell library sizes used in the `k` heuristic via
    /// `target_metacell_umis / quantile_umis`.
    pub k_umis_quantile: f32,
    /// Optional floor on `k`. `None` defers entirely to the heuristic.
    pub min_knn_k: Option<usize>,
}

/// Default implementation for [MC2KnnParams]
impl Default for MC2KnnParams {
    fn default() -> Self {
        Self {
            balanced_ranks_factor: 4.0,
            incoming_degree_factor: 3.0,
            outgoing_degree_factor: 1.0,
            min_outgoing_degree: 1,
            k_size_factor: 1.0,   // TODO: confirm against metacells.parameters
            k_umis_quantile: 0.1, // TODO: confirm against metacells.parameters
            min_knn_k: None,      // TODO: confirm against metacells.parameters
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
            cooldown_pass: 0.03,
            cooldown_node: 0.25,
            cooldown_phase: 0.5,
            min_split_size_factor: 2.0,
            max_merge_size_factor: 0.25,
            max_split_min_cut_strength: 0.1,
            min_cut_seed_cells: 7,
            min_seed_size_quantile: 0.05,
            max_seed_size_quantile: 0.95,
        }
    }
}

////////////////////
// Deviant params //
////////////////////

/// Adaptive deviant (outlier) detection parameters.
#[derive(Clone, Debug)]
pub struct DeviantsParams {
    /// Minimum log2 fold-factor gap (between consecutive cells, sorted by
    /// log-fraction) within a candidate metacell for a gene to mark cells as
    /// deviant. Auto-tuned upward if too many genes show gaps.
    pub min_gene_fold_factor: f32,
    /// Cap on the fraction of genes allowed to show any gap. The fold-factor
    /// threshold is raised until this is met.
    pub max_gene_fraction: f32,
    /// Cap on the fraction of cells marked as deviants. The per-gene cap is
    /// raised until this is met.
    pub max_cell_fraction: f32,
    /// Number of cells skipped when computing the "gap" — i.e. the gap is
    /// `log_fraction[i + skip] - log_fraction[i]` to absorb local noise.
    pub gap_skip_cells: usize,
    /// Maximum number of cells beyond a gap that are marked as deviants for
    /// a single gene.
    pub max_gap_cells_count: usize,
    /// Maximum fraction of cells beyond a gap. `0.0` means the absolute count
    /// (`max_gap_cells_count`) is used.
    pub max_gap_cells_fraction: f32,
}

/// Default implementation for [DeviantsParams]
impl Default for DeviantsParams {
    fn default() -> Self {
        Self {
            min_gene_fold_factor: 3.0, // log2(8)
            max_gene_fraction: 0.03,
            max_cell_fraction: 0.25,
            gap_skip_cells: 3,
            max_gap_cells_count: 1,
            max_gap_cells_fraction: 0.0,
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metacells_defaults_match_plan() {
        let p = MetacellsParams::default();
        assert_eq!(p.target_metacell_size, 96);
        assert_eq!(p.min_metacell_size, 12);
        assert_eq!(p.target_metacell_umis, 160_000);
        assert!(!p.must_complete_cover);
        assert_eq!(p.random_seed, 0);
    }

    #[test]
    fn similarity_method_default_is_log_pearson() {
        assert_eq!(SimilarityMethod::default(), SimilarityMethod::LogPearson);
    }

    #[test]
    fn select_defaults() {
        let s = SelectParams::default();
        assert_eq!(s.downsample_min_samples, 750);
        assert_eq!(s.downsample_min_cell_quantile, 0.05);
        assert_eq!(s.downsample_max_cell_quantile, 0.5);
        assert_eq!(s.min_gene_total, Some(50));
        assert_eq!(s.min_gene_top3, Some(4));
        assert_eq!(s.min_gene_relative_variance, Some(0.1));
        assert_eq!(s.min_genes, 30);
        assert!(s.lateral_gene_mask.is_none());
    }

    #[test]
    fn knn_defaults() {
        let k = MC2KnnParams::default();
        assert_eq!(k.balanced_ranks_factor, 4.0);
        assert_eq!(k.incoming_degree_factor, 3.0);
        assert_eq!(k.outgoing_degree_factor, 1.0);
        assert_eq!(k.min_outgoing_degree, 1);
    }

    #[test]
    fn deviants_and_dissolve_defaults() {
        let d = DeviantsParams::default();
        assert_eq!(d.min_gene_fold_factor, 3.0);
        assert_eq!(d.max_gene_fraction, 0.03);
        assert_eq!(d.max_cell_fraction, 0.25);

        let ds = DissolveParams::default();
        assert_eq!(ds.min_robust_size_factor, 0.5);
        assert_eq!(ds.min_convincing_gene_fold_factor, Some(3.0));
    }

    #[test]
    fn composition_preserves_unrelated_subparams() {
        // Touch one knob deep in the tree, verify others stay default.
        let mut p = MetacellsParams {
            target_metacell_size: 50,
            ..Default::default()
        };
        p.knn.balanced_ranks_factor = 2.0;
        p.must_complete_cover = true;

        assert_eq!(p.target_metacell_size, 50);
        assert_eq!(p.knn.balanced_ranks_factor, 2.0);
        assert!(p.must_complete_cover);
        // Untouched fields hold defaults.
        assert_eq!(p.min_metacell_size, 12);
        assert_eq!(p.knn.incoming_degree_factor, 3.0);
        assert_eq!(p.select.min_genes, 30);
    }
}
