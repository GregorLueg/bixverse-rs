//! Parameters, the closed-form model and the residual arithmetic.
//!
//! ### References
//!
//! Lause, Berens & Kobak, Genome Biology, 2021, 22:258

use crate::errors::BixverseErrors;
use crate::single_cell::sc_processing::residuals::{
    ResidualSource, intersect_gene_sets, validate_groups,
};

////////////
// Consts //
////////////

/// Technical overdispersion shared across genes.
///
/// Lause et al. estimate this from negative control datasets, where there is no
/// biological variability to confound it, and find UMI counts consistent with
/// `theta >= 100` across droplet- and plate-based protocols. The per-gene
/// estimates that scTransform fits are, on their analysis, dominated by
/// estimation bias rather than a real mean-overdispersion relationship.
const DEFAULT_THETA: f64 = 100.0;

////////////
// Params //
////////////

/// Tuning knobs for analytic Pearson residuals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AprParams {
    /// Shared inverse overdispersion. Use `f64::INFINITY` for the Poisson
    /// limit.
    pub theta: f64,
    /// Minimum number of cells a gene must be detected in to be retained.
    ///
    /// Matches scTransform's own filter. The retained set defines the count
    /// block the three sums are taken over, so this changes `mu`, not just
    /// which genes come back.
    pub min_cells: usize,
    /// Residual clipping range. `None` resolves to `+/- sqrt(n_cells)`.
    pub clip_range: Option<(f64, f64)>,
}

impl Default for AprParams {
    fn default() -> Self {
        Self {
            theta: DEFAULT_THETA,
            min_cells: 5,
            clip_range: None,
        }
    }
}

impl AprParams {
    /// Resolves the residual clipping range against the number of cells.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells the residuals are computed over.
    ///
    /// ### Returns
    ///
    /// The clipping range, either as supplied or `+/- sqrt(n_cells)`.
    pub fn resolve_clip_range(&self, n_cells: usize) -> (f64, f64) {
        self.clip_range.unwrap_or_else(|| {
            let c = (n_cells as f64).sqrt();
            (-c, c)
        })
    }

    /// Rejects a non-positive overdispersion.
    ///
    /// ### Returns
    ///
    /// `()`, or [`BixverseErrors::AnalyticPearsonInvalidTheta`].
    pub fn validate(&self) -> Result<(), BixverseErrors> {
        if self.theta <= 0.0 || self.theta.is_nan() {
            return Err(BixverseErrors::AnalyticPearsonInvalidTheta { theta: self.theta });
        }
        Ok(())
    }
}

///////////
// Model //
///////////

/// The closed-form model over one group of cells.
///
/// Three numbers per gene set describe it entirely: the gene totals, the grand
/// total and the shared overdispersion. Combined with a cell's own total they
/// give `mu_cg` for any entry, which is why nothing has to be fitted and why
/// the residual stage streams.
#[derive(Clone, Debug, PartialEq)]
pub struct AprModel {
    /// Store gene indices this model covers, ascending.
    pub genes: Vec<usize>,
    /// Column sums `sum_c X_cg` over the retained genes and selected cells,
    /// parallel to `genes`.
    pub gene_sums: Vec<f64>,
    /// Grand total `sum_cg X_cg` over the same block.
    pub total: f64,
    /// Shared inverse overdispersion.
    pub theta: f64,
    /// Residual clipping range.
    pub clip_range: (f64, f64),
}

impl AprModel {
    /// Number of genes covered.
    ///
    /// ### Returns
    ///
    /// The gene count.
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    /// Whether any gene is covered.
    ///
    /// ### Returns
    ///
    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    /// Position of a store gene index within this model.
    ///
    /// ### Params
    ///
    /// * `gene` - Store gene index.
    ///
    /// ### Returns
    ///
    /// The position, or `None` when the gene is not covered.
    pub fn position(&self, gene: usize) -> Option<usize> {
        self.genes.binary_search(&gene).ok()
    }

    /// The gene's share of the total, `p_g`.
    ///
    /// ### Params
    ///
    /// * `pos` - Position within the model.
    ///
    /// ### Returns
    ///
    /// The expression fraction.
    pub fn p_gene(&self, pos: usize) -> f64 {
        self.gene_sums[pos] / self.total
    }
}

/// One gene's residual parameters, resolved once per gene.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AprGeneParams {
    /// The gene's share of the total, `p_g`.
    p_gene: f64,
    /// Shared inverse overdispersion.
    theta: f64,
    /// Residual clipping range.
    clip: (f64, f64),
}

impl AprGeneParams {
    /// Pulls one gene's parameters out of a model.
    ///
    /// ### Params
    ///
    /// * `model` - The model.
    /// * `pos` - Position of the gene within it.
    ///
    /// ### Returns
    ///
    /// The parameters.
    pub(crate) fn new(model: &AprModel, pos: usize) -> Self {
        Self {
            p_gene: model.p_gene(pos),
            theta: model.theta,
            clip: model.clip_range,
        }
    }

    /// One cell's residual.
    ///
    /// ### Params
    ///
    /// * `cell_total` - The cell's total over the retained genes, `n_c`.
    /// * `y` - The observed count.
    ///
    /// ### Returns
    ///
    /// The clipped Pearson residual.
    #[inline(always)]
    pub(crate) fn residual(&self, cell_total: f64, y: f64) -> f32 {
        let mu = self.p_gene * cell_total;
        let var = mu + mu * mu / self.theta;
        let (lo, hi) = self.clip;
        // A cell with no counts at all over the retained genes has `mu = 0`,
        // which would divide by zero rather than produce the residual of zero
        // the model implies.
        if var <= 0.0 {
            return 0.0;
        }
        ((y - mu) / var.sqrt()).clamp(lo, hi) as f32
    }
}

//////////////////
// AprResiduals //
//////////////////

/// Analytic Pearson residuals from one or more fitted models.
///
/// Grouped the same way as
/// [`SctResiduals`](crate::single_cell::sc_processing::sctransform::residuals::SctResiduals):
/// one model per sample, every cell scored under its own sample's model, gene
/// axis narrowed to the intersection.
#[derive(Clone, Debug)]
pub struct AprResiduals<'a> {
    /// One model per group, indexed by group id.
    models: &'a [AprModel],
    /// Per-cell totals over each group's retained genes, in the selected cell
    /// order.
    cell_totals: &'a [f64],
    /// Store gene indices covered in every group, ascending.
    genes: Vec<usize>,
    /// `positions[group][gene_pos]` is that group's model row for the shared
    /// gene.
    positions: Vec<Vec<u32>>,
    /// Group id per selected cell.
    group_of_cell: Vec<u32>,
}

impl<'a> AprResiduals<'a> {
    /// Builds a grouped residual source.
    ///
    /// ### Params
    ///
    /// * `models` - One model per group, in group id order.
    /// * `cell_totals` - Per-cell totals over the retained genes, one entry per
    ///   selected cell, each computed under that cell's own group's gene set.
    /// * `group_of_cell` - Group id per selected cell. Must densely cover
    ///   `0..models.len()`.
    ///
    /// ### Returns
    ///
    /// The source, or a [`BixverseErrors`] when the grouping is malformed or no
    /// gene is covered in every group.
    pub fn new(
        models: &'a [AprModel],
        cell_totals: &'a [f64],
        group_of_cell: Vec<u32>,
    ) -> Result<Self, BixverseErrors> {
        let n_groups = validate_groups(&group_of_cell, cell_totals.len())?;
        if n_groups != models.len() {
            return Err(BixverseErrors::ResidualEmptyGroup {
                group: models.len().min(n_groups),
                n_groups: models.len(),
            });
        }

        let sets: Vec<&[usize]> = models.iter().map(|m| m.genes.as_slice()).collect();
        let genes = intersect_gene_sets(&sets)?;

        let positions = models
            .iter()
            .map(|model| {
                genes
                    .iter()
                    .map(|&g| {
                        model
                            .position(g)
                            .map(|p| p as u32)
                            .ok_or(BixverseErrors::SctGeneNotModelled { gene: g })
                    })
                    .collect::<Result<Vec<u32>, BixverseErrors>>()
            })
            .collect::<Result<Vec<_>, BixverseErrors>>()?;

        Ok(Self {
            models,
            cell_totals,
            genes,
            positions,
            group_of_cell,
        })
    }

    /// Builds a source over a single model, with every cell in one group.
    ///
    /// ### Params
    ///
    /// * `model` - The model.
    /// * `cell_totals` - Per-cell totals over the retained genes.
    ///
    /// ### Returns
    ///
    /// The source, or a [`BixverseErrors`] when the inputs disagree.
    pub fn single(
        model: &'a AprModel,
        cell_totals: &'a [f64],
    ) -> Result<AprResiduals<'a>, BixverseErrors> {
        let group_of_cell = vec![0_u32; cell_totals.len()];
        Self::new(std::slice::from_ref(model), cell_totals, group_of_cell)
    }

    /// The models, in group id order.
    ///
    /// ### Returns
    ///
    /// The models.
    pub fn models(&self) -> &[AprModel] {
        self.models
    }
}

impl ResidualSource for AprResiduals<'_> {
    fn genes(&self) -> &[usize] {
        &self.genes
    }

    fn n_cells(&self) -> usize {
        self.cell_totals.len()
    }

    fn group_of_cell(&self) -> &[u32] {
        &self.group_of_cell
    }

    fn n_groups(&self) -> usize {
        self.models.len()
    }

    fn residual_row(
        &self,
        counts: &[f64],
        indices: &[u32],
        gene_pos: usize,
        out: &mut [f32],
    ) -> Result<(), BixverseErrors> {
        if gene_pos >= self.genes.len() {
            return Err(BixverseErrors::SctGeneIndexOutOfRange {
                index: gene_pos,
                n_genes: self.genes.len(),
            });
        }
        if out.len() != self.cell_totals.len() {
            return Err(BixverseErrors::LengthMismatch {
                name: "out",
                expected: self.cell_totals.len(),
                found: out.len(),
            });
        }

        let per_group: Vec<AprGeneParams> = self
            .models
            .iter()
            .zip(&self.positions)
            .map(|(model, pos)| AprGeneParams::new(model, pos[gene_pos] as usize))
            .collect();

        // Zero counts first, then the stored non-zeros over the top: one pass
        // over the cells plus one over the non-zeros, no densified counts.
        for (c, slot) in out.iter_mut().enumerate() {
            let gene = &per_group[self.group_of_cell[c] as usize];
            *slot = gene.residual(self.cell_totals[c], 0.0);
        }
        for (&i, &y) in indices.iter().zip(counts.iter()) {
            let c = i as usize;
            let gene = &per_group[self.group_of_cell[c] as usize];
            out[c] = gene.residual(self.cell_totals[c], y);
        }

        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A 2 genes by 3 cells block with gene sums 6 and 9, total 15.
    fn toy_model() -> AprModel {
        AprModel {
            genes: vec![0, 1],
            gene_sums: vec![6.0, 9.0],
            total: 15.0,
            theta: 100.0,
            clip_range: (-10.0, 10.0),
        }
    }

    #[test]
    fn test_apr_residual_matches_closed_form() {
        let model = toy_model();
        let gene = AprGeneParams::new(&model, 0);
        // p_g = 6/15 = 0.4, n_c = 5 -> mu = 2.0, var = 2 + 4/100 = 2.04.
        let expected = (3.0 - 2.0) / 2.04_f64.sqrt();
        assert_relative_eq!(gene.residual(5.0, 3.0) as f64, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_apr_residual_poisson_limit() {
        let mut model = toy_model();
        model.theta = f64::INFINITY;
        let gene = AprGeneParams::new(&model, 0);
        // var collapses to mu = 2.0.
        let expected = (3.0 - 2.0) / 2.0_f64.sqrt();
        assert_relative_eq!(gene.residual(5.0, 3.0) as f64, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_apr_residual_clips() {
        let mut model = toy_model();
        model.clip_range = (-0.5, 0.5);
        let gene = AprGeneParams::new(&model, 0);
        assert_relative_eq!(gene.residual(5.0, 100.0) as f64, 0.5, epsilon = 1e-9);
        assert_relative_eq!(gene.residual(5.0, 0.0) as f64, -0.5, epsilon = 1e-9);
    }

    #[test]
    fn test_apr_residual_empty_cell_is_zero() {
        let model = toy_model();
        let gene = AprGeneParams::new(&model, 0);
        assert_eq!(gene.residual(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_apr_residual_row_zeros_and_nonzeros() {
        let model = toy_model();
        let totals = [5.0, 5.0, 5.0];
        let source = AprResiduals::single(&model, &totals).unwrap();
        let mut row = vec![0.0_f32; 3];
        source.residual_row(&[3.0], &[1], 0, &mut row).unwrap();

        let gene = AprGeneParams::new(&model, 0);
        assert_relative_eq!(row[0], gene.residual(5.0, 0.0), epsilon = 1e-9);
        assert_relative_eq!(row[1], gene.residual(5.0, 3.0), epsilon = 1e-9);
        assert_relative_eq!(row[2], gene.residual(5.0, 0.0), epsilon = 1e-9);
    }

    #[test]
    fn test_apr_residual_row_routes_cells_to_their_group() {
        // Two groups with deliberately different gene shares, so a cell scored
        // under the wrong model gives a visibly different residual.
        let a = toy_model();
        let b = AprModel {
            genes: vec![0, 1],
            gene_sums: vec![1.0, 9.0],
            total: 10.0,
            theta: 100.0,
            clip_range: (-10.0, 10.0),
        };
        let models = [a.clone(), b.clone()];
        let totals = [5.0, 5.0];
        let source = AprResiduals::new(&models, &totals, vec![0, 1]).unwrap();

        let mut row = vec![0.0_f32; 2];
        source.residual_row(&[], &[], 0, &mut row).unwrap();

        assert_relative_eq!(
            row[0],
            AprGeneParams::new(&a, 0).residual(5.0, 0.0),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            row[1],
            AprGeneParams::new(&b, 0).residual(5.0, 0.0),
            epsilon = 1e-9
        );
        assert_ne!(row[0], row[1]);
    }

    #[test]
    fn test_apr_residuals_intersects_gene_sets() {
        let a = AprModel {
            genes: vec![0, 1, 2],
            gene_sums: vec![1.0, 2.0, 3.0],
            total: 6.0,
            theta: 100.0,
            clip_range: (-10.0, 10.0),
        };
        let b = AprModel {
            genes: vec![1, 2, 5],
            gene_sums: vec![2.0, 3.0, 4.0],
            total: 9.0,
            theta: 100.0,
            clip_range: (-10.0, 10.0),
        };
        let models = [a, b];
        let totals = [5.0, 5.0];
        let source = AprResiduals::new(&models, &totals, vec![0, 1]).unwrap();
        assert_eq!(source.genes(), &[1, 2]);
    }

    #[test]
    fn test_apr_params_rejects_non_positive_theta() {
        let params = AprParams {
            theta: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            params.validate(),
            Err(BixverseErrors::AnalyticPearsonInvalidTheta { .. })
        ));
    }

    #[test]
    fn test_apr_params_default_clip_is_sqrt_n() {
        let params = AprParams::default();
        let (lo, hi) = params.resolve_clip_range(100);
        assert_relative_eq!(lo, -10.0, epsilon = 1e-12);
        assert_relative_eq!(hi, 10.0, epsilon = 1e-12);
    }
}
