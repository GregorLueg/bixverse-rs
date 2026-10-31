//! Parameter structs for the MetaCells2 pipeline.
//!
//! The hierarchy mirrors the algorithm stages: each stage's parameters live in
//! their own struct, composed into the top-level [`MetacellsParams`]. Defaults
//! are upstream's (`metacells/parameters.py`).
//!
//! Every struct deserialises from the one flat R `params_sc_metacells2()` or
//! `params_sc_metacells2_dac()` list, as the R list conversion reads them;
//! missing keys take the defaults.

use bixverse_core::ml::clustering::hierarchical::{LinkageMethod, parse_linkage_method};
use serde::{Deserialize, Deserializer, de::Error as _};

/////////////
// Parsers //
/////////////

/// Parse a MetaCells2 similarity method.
///
/// ### Params
///
/// * `s` - `"abs_pearson"` or `"pearson"`, case-insensitive.
///
/// ### Returns
///
/// The method, or `None` if unrecognised.
pub fn parse_similarity_method(s: &str) -> Option<SimilarityMethod> {
    match s.to_lowercase().as_str() {
        "abs_pearson" => Some(SimilarityMethod::AbsPearson),
        "pearson" => Some(SimilarityMethod::Pearson),
        _ => None,
    }
}

/// Parse a MetaCells2 rare gene similarity method.
///
/// ### Params
///
/// * `s` - `"pearson"`, `"abs_pearson"`, `"repeated_pearson"` or
///   `"repeated_abs_pearson"`, case-insensitive.
///
/// ### Returns
///
/// The method, or `None` if unrecognised.
pub fn parse_rare_similarity_method(s: &str) -> Option<RareSimilarityMethod> {
    match s.to_lowercase().as_str() {
        "pearson" => Some(RareSimilarityMethod::Pearson),
        "abs_pearson" => Some(RareSimilarityMethod::AbsPearson),
        "repeated_pearson" => Some(RareSimilarityMethod::RepeatedPearson),
        "repeated_abs_pearson" => Some(RareSimilarityMethod::RepeatedAbsPearson),
        _ => None,
    }
}

/// Deserialise a [SimilarityMethod] through [parse_similarity_method]
///
/// ### Params
///
/// * `d` - The deserialiser, holding a string
///
/// ### Returns
///
/// The method, or an error naming the unrecognised value.
pub fn de_similarity_method<'de, D: Deserializer<'de>>(d: D) -> Result<SimilarityMethod, D::Error> {
    let s = String::deserialize(d)?;
    parse_similarity_method(&s).ok_or_else(|| {
        D::Error::custom(format!(
            "similarity method must be 'abs_pearson' or 'pearson', got '{s}'"
        ))
    })
}

/// Deserialise a [RareSimilarityMethod] through [parse_rare_similarity_method]
///
/// ### Params
///
/// * `d` - The deserialiser, holding a string
///
/// ### Returns
///
/// The method, or an error naming the unrecognised value.
pub fn de_rare_similarity_method<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<RareSimilarityMethod, D::Error> {
    let s = String::deserialize(d)?;
    parse_rare_similarity_method(&s).ok_or_else(|| {
        D::Error::custom(format!(
            "rare similarity method must be one of 'pearson', 'abs_pearson', \
             'repeated_pearson' or 'repeated_abs_pearson', got '{s}'"
        ))
    })
}

/// Deserialise a [LinkageMethod] through [parse_linkage_method]
///
/// ### Params
///
/// * `d` - The deserialiser, holding a string
///
/// ### Returns
///
/// The linkage, or an error naming the unrecognised value.
pub fn de_linkage_method<'de, D: Deserializer<'de>>(d: D) -> Result<LinkageMethod, D::Error> {
    let s = String::deserialize(d)?;
    parse_linkage_method(&s).ok_or_else(|| {
        D::Error::custom(format!(
            "cluster method must be one of 'ward', 'complete', 'average' or \
             'weighted', got '{s}'"
        ))
    })
}

/// Deserialise a `u64` that may arrive as a real, as R hands over counts past
/// `i32` range
///
/// ### Params
///
/// * `d` - The deserialiser, holding a non-negative number
///
/// ### Returns
///
/// The value truncated towards zero, as the R list conversion casts it, or
/// an error for a negative or non-finite one.
pub fn de_u64_from_real<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
    let v = f64::deserialize(d)?;
    if !v.is_finite() || v < 0.0 {
        return Err(D::Error::custom(format!(
            "expected a non-negative finite number, got {v}"
        )));
    }
    Ok(v as u64)
}

/////////////////
// Main params //
/////////////////

/// Top-level parameters for the MC2 algorithm.
///
/// Composed of one struct per pipeline stage plus a few cross-cutting knobs
/// (target metacell size/UMI count, must-complete-cover mode). The RNG seed is
/// passed to the entry points separately.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct MetacellsParams {
    /// The [SelectParams] for feature selections.
    #[serde(flatten)]
    pub select: SelectParams,
    /// The [SimilarityParams] for the MetaCell similarity
    #[serde(flatten)]
    pub similarity: SimilarityParams,
    /// The [MC2KnnParams] - these are MetaCell2-specific parameters that are
    /// different from [`crate::prelude::KnnParams`].
    #[serde(flatten)]
    pub knn: MC2KnnParams,
    /// The [PartitionParams] that defines how to partition the cells into the
    /// meta cells.
    #[serde(flatten)]
    pub partition: PartitionParams,
    /// The [DeviantsParams] that mark genes/cells as deviant
    #[serde(flatten)]
    pub deviants: DeviantsParams,
    /// The [DissolveParams] that dissolve small, unconvincing candidates.
    #[serde(flatten)]
    pub dissolve: DissolveParams,
    /// Target cell count per metacell.
    pub target_metacell_size: usize,
    /// Below this, a candidate metacell is dissolved to outliers regardless of
    /// UMI counts or convincing genes.
    pub min_metacell_size: usize,
    /// Target total UMIs per metacell. Drives seed count and band penalties.
    #[serde(deserialize_with = "de_u64_from_real")]
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
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
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
    #[serde(skip)]
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
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SimilarityParams {
    /// Enum defining the chosen similarity method, see [SimilarityMethod].
    #[serde(
        rename = "similarity_method",
        deserialize_with = "de_similarity_method"
    )]
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
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
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
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
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
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct DeviantsParams {
    /// Minimum log2 fold-factor gap (between cells sorted by log-fraction)
    /// within a candidate metacell for a gene to mark cells as deviant. Raised
    /// by 1/8 while more than `max_cell_fraction` of the cells are deviant.
    pub min_gene_fold_factor: f32,
    /// Added to `min_gene_fold_factor` for genes in `noisy_gene_mask`.
    pub min_noisy_gene_fold_factor: f32,
    /// Optional per-gene mask of noisy genes, which need a larger gap.
    #[serde(skip)]
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
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
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

/////////////////////////////
// Rare gene module params //
/////////////////////////////

/// Gene-gene similarity used to cluster rare gene candidates, upstream's
/// `genes_similarity_method` minus the logistics variants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RareSimilarityMethod {
    /// Pearson correlation between genes over cells (`pearson`).
    Pearson,
    /// Absolute Pearson correlation (`abs_pearson`).
    AbsPearson,
    /// Correlation of the rows of the Pearson matrix (`repeated_pearson`).
    /// Upstream's default.
    #[default]
    RepeatedPearson,
    /// Absolute correlation of the rows of the absolute Pearson matrix
    /// (`repeated_abs_pearson`).
    RepeatedAbsPearson,
}

/// Rare gene module detection, upstream's `find_rare_gene_modules` plus the
/// `rare_*` overrides `divide_and_conquer_pipeline` applies to the rare piles.
///
/// Its keys carry upstream's `rare_` prefix, so they share the [DacParams]
/// list.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct RareParams {
    /// Most candidate genes kept, those expressed in the fewest cells.
    #[serde(rename = "rare_max_genes")]
    pub max_genes: usize,
    /// Largest fraction of cells a candidate may be expressed in.
    #[serde(rename = "rare_max_gene_cell_fraction")]
    pub max_gene_cell_fraction: f64,
    /// Smallest maximum UMI count over cells for a candidate, and for a
    /// related gene over the module's cells.
    #[serde(rename = "rare_min_gene_maximum")]
    pub min_gene_maximum: u32,
    /// Similarity between candidate genes.
    #[serde(
        rename = "rare_genes_similarity_method",
        deserialize_with = "de_rare_similarity_method"
    )]
    pub similarity: RareSimilarityMethod,
    /// Linkage for the hierarchical clustering of candidates.
    #[serde(
        rename = "rare_genes_cluster_method",
        deserialize_with = "de_linkage_method"
    )]
    pub cluster_method: LinkageMethod,
    /// Fewest genes in a module.
    #[serde(rename = "rare_min_genes_of_modules")]
    pub min_genes_of_modules: usize,
    /// Fewest cells in a module. Upstream's pipeline uses its own default
    /// here, not the run's `min_metacell_size`.
    #[serde(rename = "rare_min_cells_of_modules")]
    pub min_cells_of_modules: usize,
    /// Largest mean number of module cells in a random pile, as a fraction of
    /// the target metacell size, for the module to count as rare.
    #[serde(rename = "rare_max_cells_factor_of_random_pile")]
    pub max_cells_factor_of_random_pile: f64,
    /// Smallest mean off-diagonal similarity within a module.
    #[serde(rename = "rare_min_module_correlation")]
    pub min_module_correlation: f64,
    /// A gene is related to a module if its fraction in the module's cells is
    /// at least `2^min_related_gene_fold_factor` times its background fraction.
    #[serde(rename = "rare_min_related_gene_fold_factor")]
    pub min_related_gene_fold_factor: f64,
    /// A related gene is dropped if adding it raises the module's strong cells
    /// beyond this factor of the original count.
    #[serde(rename = "rare_max_related_gene_increase_factor")]
    pub max_related_gene_increase_factor: f64,
    /// Smallest module UMI total for a cell to express the module.
    #[serde(rename = "rare_min_cell_module_total")]
    pub min_cell_module_total: u32,
    /// `deviants.max_cell_fraction` of the rare piles.
    #[serde(rename = "rare_deviants_max_cell_fraction")]
    pub deviants_max_cell_fraction: f32,
    /// `dissolve.min_robust_size_factor` of the rare piles.
    #[serde(rename = "rare_dissolve_min_robust_size_factor")]
    pub dissolve_min_robust_size_factor: f32,
    /// `dissolve.min_convincing_gene_fold_factor` of the rare piles.
    #[serde(rename = "rare_dissolve_min_convincing_gene_fold_factor")]
    pub dissolve_min_convincing_gene_fold_factor: Option<f32>,
}

/// Default implementation for [RareParams], upstream's `rare_*` defaults.
impl Default for RareParams {
    fn default() -> Self {
        Self {
            max_genes: 500,
            max_gene_cell_fraction: 1e-3,
            min_gene_maximum: 7,
            similarity: RareSimilarityMethod::RepeatedPearson,
            cluster_method: LinkageMethod::Ward,
            min_genes_of_modules: 4,
            min_cells_of_modules: 12,
            max_cells_factor_of_random_pile: 0.5,
            min_module_correlation: 0.1,
            min_related_gene_fold_factor: 7.0,
            max_related_gene_increase_factor: 4.0,
            min_cell_module_total: 4,
            deviants_max_cell_fraction: 0.25,
            dissolve_min_robust_size_factor: 0.5,
            dissolve_min_convincing_gene_fold_factor: Some(3.0),
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
///
/// Deserialises from the flat R `params_sc_metacells2_dac()` list, see
/// [`DacParamsSpec`].
#[derive(Clone, Debug, Deserialize)]
#[serde(from = "DacParamsSpec")]
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
    /// Rare gene module detection before the main run, as upstream's
    /// `divide_and_conquer_pipeline`: cells of each module become their own
    /// pile and the rest are divided and conquered. `None` skips it, as
    /// upstream's `compute_divide_and_conquer_metacells`.
    pub rare: Option<RareParams>,
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
            rare: Some(RareParams::default()),
        }
    }
}

/// The keys a [`DacParams`] deserialises from, as its R list conversion reads
/// them; missing keys take the [`DacParams`] defaults.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct DacParamsSpec {
    /// See [`DacParams::quick_and_dirty`].
    pub quick_and_dirty: bool,
    /// See [`DacParams::min_target_pile_size`].
    pub min_target_pile_size: usize,
    /// See [`DacParams::max_target_pile_size`].
    pub max_target_pile_size: usize,
    /// See [`DacParams::target_metacells_in_pile`].
    pub target_metacells_in_pile: usize,
    /// See [`DacParams::target_pile_size`].
    pub target_pile_size: Option<usize>,
    /// See [`DacParams::piles_knn_k_size_factor`].
    pub piles_knn_k_size_factor: f64,
    /// See [`DacParams::piles_min_split_size_factor`].
    pub piles_min_split_size_factor: f64,
    /// See [`DacParams::piles_min_robust_size_factor`].
    pub piles_min_robust_size_factor: f32,
    /// See [`DacParams::piles_max_merge_size_factor`].
    pub piles_max_merge_size_factor: f64,
    /// Similarity method of the groups phase.
    #[serde(deserialize_with = "de_similarity_method")]
    pub groups_similarity_method: SimilarityMethod,
    /// Log transform in the groups phase similarity.
    pub groups_log_data: bool,
    /// Regularisation of the groups phase similarity.
    pub groups_value_regularisation: f32,
    /// See [`DacParams::max_parallel_piles`].
    pub max_parallel_piles: usize,
    /// Detect rare gene modules first.
    pub rare_gene_modules: bool,
    /// The `rare_*` keys, used when `rare_gene_modules` is set.
    #[serde(flatten)]
    pub rare: RareParams,
}

impl Default for DacParamsSpec {
    fn default() -> Self {
        let d = DacParams::default();
        Self {
            quick_and_dirty: d.quick_and_dirty,
            min_target_pile_size: d.min_target_pile_size,
            max_target_pile_size: d.max_target_pile_size,
            target_metacells_in_pile: d.target_metacells_in_pile,
            target_pile_size: d.target_pile_size,
            piles_knn_k_size_factor: d.piles_knn_k_size_factor,
            piles_min_split_size_factor: d.piles_min_split_size_factor,
            piles_min_robust_size_factor: d.piles_min_robust_size_factor,
            piles_max_merge_size_factor: d.piles_max_merge_size_factor,
            groups_similarity_method: d.groups_similarity.method,
            groups_log_data: d.groups_similarity.log_data,
            groups_value_regularisation: d.groups_similarity.value_regularisation,
            max_parallel_piles: d.max_parallel_piles,
            rare_gene_modules: d.rare.is_some(),
            rare: d.rare.unwrap_or_default(),
        }
    }
}

impl From<DacParamsSpec> for DacParams {
    /// Assemble the groups phase similarity and drop the rare parameters
    /// when the detection is off.
    ///
    /// ### Params
    ///
    /// * `s` - The keys
    ///
    /// ### Returns
    ///
    /// The [`DacParams`].
    fn from(s: DacParamsSpec) -> Self {
        Self {
            quick_and_dirty: s.quick_and_dirty,
            min_target_pile_size: s.min_target_pile_size,
            max_target_pile_size: s.max_target_pile_size,
            target_metacells_in_pile: s.target_metacells_in_pile,
            target_pile_size: s.target_pile_size,
            piles_knn_k_size_factor: s.piles_knn_k_size_factor,
            piles_min_split_size_factor: s.piles_min_split_size_factor,
            piles_min_robust_size_factor: s.piles_min_robust_size_factor,
            piles_max_merge_size_factor: s.piles_max_merge_size_factor,
            groups_similarity: SimilarityParams {
                method: s.groups_similarity_method,
                log_data: s.groups_log_data,
                value_regularisation: s.groups_value_regularisation,
            },
            max_parallel_piles: s.max_parallel_piles,
            rare: s.rare_gene_modules.then_some(s.rare),
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
