//! Parameters for DIALOGUE.
//!
//! Three blocks, one per stage, composed into [DialogueParams]. Defaults match
//! `DLG.get.param` except where that function has a bug; those cases are called
//! out on the field.

use crate::prelude::*;

////////////
// Consts //
////////////

/// Minimum samples shared across every cell type before the CCA is worth
/// running. Upstream stops here too.
pub const MIN_SHARED_SAMPLES: usize = 5;

/// Minimum features surviving the ANOVA filter in any one cell type.
pub const MIN_ANOVA_FEATURES: usize = 5;

///////////////
// Averaging //
///////////////

/// How cell-level features are collapsed to one value per sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Averaging {
    /// Column medians. What every published DIALOGUE run actually used.
    #[default]
    Median,
    /// Column means.
    Mean,
}

/// Parses an averaging function from a string.
///
/// ### Params
///
/// * `s` - `"median"` or `"mean"`, case insensitive
///
/// ### Returns
///
/// The [Averaging], or `None` for anything else.
pub fn parse_averaging(s: &str) -> Option<Averaging> {
    match s.to_lowercase().as_str() {
        "median" => Some(Averaging::Median),
        "mean" => Some(Averaging::Mean),
        _ => None,
    }
}

///////////////
// PmdParams //
///////////////

/// Stage one: penalised matrix decomposition and the programme scores.
#[derive(Clone, Debug)]
pub struct PmdParams {
    /// Number of multicellular programmes to extract.
    pub k: usize,
    /// Permutations backing the empirical p-value per programme.
    ///
    /// Upstream uses 100, of which one is the real fit and 99 are permuted.
    pub n_permutations: usize,
    /// Tune the L1 bound by permutation instead of fixing it at
    /// `sqrt(p_1) / 2`. Costs ten more fits per permutation and is off by
    /// default, as upstream's `extra.sparse` is.
    pub extra_sparse: bool,
    /// Minimum cells a sample must contribute, within a cell type, before it
    /// counts towards the feature-level ANOVA.
    pub abn_c: usize,
    /// BH-adjusted ANOVA cutoff for keeping a feature.
    pub p_anova: f64,
    /// Centre and scale the sample-level feature matrix, then winsorise it.
    pub centre: bool,
    /// Winsorising tail fraction applied to each column.
    pub cap: f64,
    /// Spatial data: skip the ANOVA feature filter entirely. Niches are small,
    /// so a feature need not vary across them to be real.
    pub spatial: bool,
    /// Genes taken per programme per direction when building a signature.
    pub n_genes: usize,
    /// Minimum absolute correlation for a gene to enter a signature.
    pub min_ci: f64,
    /// How cell-level features are collapsed per sample.
    ///
    /// Upstream exposes this argument and then ignores it, hard-coding column
    /// medians. The knob works here; the default is what upstream actually did.
    pub averaging: Averaging,
    /// Empirical p below which a cell-type pair counts as connected when
    /// deciding which cell types a programme spans.
    pub mcp_assignment_p: f64,
    /// Seed for the permutation null.
    pub seed: u64,
}

impl PmdParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `k` - Number of programmes
    /// * `n_permutations` - Permutations for the empirical p-value
    /// * `extra_sparse` - Tune the L1 bound by permutation
    /// * `abn_c` - Minimum cells per sample for the ANOVA filter
    /// * `p_anova` - BH-adjusted ANOVA cutoff
    /// * `centre` - Centre, scale and winsorise the sample matrix
    /// * `cap` - Winsorising tail fraction
    /// * `spatial` - Skip the ANOVA filter
    /// * `n_genes` - Genes per programme per direction
    /// * `min_ci` - Minimum absolute correlation for a signature gene
    /// * `averaging` - How to collapse features per sample
    /// * `mcp_assignment_p` - Empirical p cutoff for the cell-type assignment
    /// * `seed` - Seed for the permutation null
    ///
    /// ### Returns
    ///
    /// The parameter set. Nothing is validated here.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: usize,
        n_permutations: usize,
        extra_sparse: bool,
        abn_c: usize,
        p_anova: f64,
        centre: bool,
        cap: f64,
        spatial: bool,
        n_genes: usize,
        min_ci: f64,
        averaging: Averaging,
        mcp_assignment_p: f64,
        seed: u64,
    ) -> Self {
        Self {
            k,
            n_permutations,
            extra_sparse,
            abn_c,
            p_anova,
            centre,
            cap,
            spatial,
            n_genes,
            min_ci,
            averaging,
            mcp_assignment_p,
            seed,
        }
    }
}

impl Default for PmdParams {
    fn default() -> Self {
        Self {
            k: 2,
            n_permutations: 100,
            extra_sparse: false,
            abn_c: 15,
            p_anova: 0.05,
            centre: true,
            cap: 0.01,
            spatial: false,
            n_genes: 200,
            min_ci: 0.05,
            averaging: Averaging::Median,
            mcp_assignment_p: 0.1,
            seed: 1234,
        }
    }
}

///////////////
// HlmParams //
///////////////

/// Stage two: the mixed models that prune the signatures.
///
/// Upstream also filters each pair's results to the top hundred genes with
/// `|Z| >= 1` and stores them as `sig1f` / `sig2f`. Those lists are assigned
/// from the wrong object and come back empty, and nothing downstream reads
/// them: stage three works from the full association table instead. The filter
/// is not reproduced here, because reproducing it would mean reproducing
/// something that never had an effect.
#[derive(Clone, Copy, Debug)]
pub struct HlmParams {
    /// Minimum cells a sample must contribute, in *both* cell types of a pair,
    /// before it takes part in that pair's models.
    pub min_cells_per_sample: usize,
    /// Include the partner cell type's mean quality in that sample as a fixed
    /// effect, upstream's `tme.qc`.
    pub use_tme_qc: bool,
    /// Include the responding cell's own quality as a fixed effect, upstream's
    /// `cellQ`.
    ///
    /// Note this covariate has already been regressed out of the scores by
    /// ordinary least squares in stage one. Upstream conditions on it twice;
    /// the default here reproduces that.
    pub use_cell_quality: bool,
    /// Compute Satterthwaite denominator degrees of freedom, as `lmerTest`
    /// does. Turning it off falls back to the residual count, which is far
    /// cheaper and barely differs once a cell type has thousands of cells.
    pub satterthwaite: bool,
}

impl HlmParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `min_cells_per_sample` - Minimum cells per sample in both cell types
    /// * `use_tme_qc` - Condition on the partner's mean quality
    /// * `use_cell_quality` - Condition on the cell's own quality
    /// * `satterthwaite` - Compute Satterthwaite degrees of freedom
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(
        min_cells_per_sample: usize,
        use_tme_qc: bool,
        use_cell_quality: bool,
        satterthwaite: bool,
    ) -> Self {
        Self {
            min_cells_per_sample,
            use_tme_qc,
            use_cell_quality,
            satterthwaite,
        }
    }
}

impl Default for HlmParams {
    fn default() -> Self {
        Self {
            min_cells_per_sample: 2,
            use_tme_qc: true,
            use_cell_quality: true,
            satterthwaite: true,
        }
    }
}

//////////////////
// RefineParams //
//////////////////

/// Stage three: cross-partner meta-analysis and the non-negative refit.
#[derive(Clone, Copy, Debug)]
pub struct RefineParams {
    /// Adjusted p below which one partner counts as supporting a gene.
    pub support_p: f64,
    /// Minimum supporting fraction for a stratum to enter the staged fit.
    pub min_support_fraction: f64,
    /// Minimum genes in a stratum before it is worth fitting.
    pub min_stratum: usize,
    /// Correlation between the original score and the running fit at which the
    /// staged fit stops early.
    pub early_stop_cor: f64,
    /// Fisher-combined p for the permissive gene list, when a gene is carried
    /// by partner support rather than by a positive coefficient.
    pub permissive_p: f64,
    /// Fisher-combined p for the strict gene list, which also demands that
    /// every partner supports the gene.
    pub strict_p: f64,
}

impl RefineParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `support_p` - Adjusted p for one partner to count as supporting
    /// * `min_support_fraction` - Minimum supporting fraction for a stratum
    /// * `min_stratum` - Minimum genes in a stratum
    /// * `early_stop_cor` - Correlation at which the staged fit stops
    /// * `permissive_p` - Fisher p for the permissive list
    /// * `strict_p` - Fisher p for the strict list
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(
        support_p: f64,
        min_support_fraction: f64,
        min_stratum: usize,
        early_stop_cor: f64,
        permissive_p: f64,
        strict_p: f64,
    ) -> Self {
        Self {
            support_p,
            min_support_fraction,
            min_stratum,
            early_stop_cor,
            permissive_p,
            strict_p,
        }
    }
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            support_p: 0.1,
            min_support_fraction: 1.0 / 3.0,
            min_stratum: 5,
            early_stop_cor: 0.95,
            permissive_p: 1e-3,
            strict_p: 0.05,
        }
    }
}

////////////////////
// DialogueParams //
////////////////////

/// Everything DIALOGUE takes, one block per stage.
#[derive(Clone, Debug, Default)]
pub struct DialogueParams {
    /// Stage one.
    pub pmd: PmdParams,
    /// Stage two.
    pub hlm: HlmParams,
    /// Stage three.
    pub refine: RefineParams,
}

impl DialogueParams {
    /// Builds a parameter set from the three stage blocks.
    ///
    /// ### Params
    ///
    /// * `pmd` - Stage one parameters
    /// * `hlm` - Stage two parameters
    /// * `refine` - Stage three parameters
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(pmd: PmdParams, hlm: HlmParams, refine: RefineParams) -> Self {
        Self { pmd, hlm, refine }
    }

    /// Checks the parameters against a run of the given shape.
    ///
    /// ### Params
    ///
    /// * `n_cell_types` - Number of cell types supplied
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [BixverseErrors::InvalidArgument] describing the first
    /// problem found.
    pub fn validate(&self, n_cell_types: usize) -> Result<(), BixverseErrors> {
        if n_cell_types < 2 {
            return Err(BixverseErrors::DialogueTooFewCellTypes { n_cell_types });
        }
        if self.pmd.k == 0 {
            return Err(BixverseErrors::InvalidArgument(
                "DIALOGUE needs at least one programme.".to_string(),
            ));
        }
        if self.pmd.n_permutations < 2 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "the empirical p-value needs at least two runs; got {}.",
                self.pmd.n_permutations
            )));
        }
        if !(0.0..0.5).contains(&self.pmd.cap) {
            return Err(BixverseErrors::InvalidArgument(format!(
                "the winsorising fraction must lie in [0, 0.5); got {}.",
                self.pmd.cap
            )));
        }
        Ok(())
    }
}
