//! scTransform as a [`ResidualSource`].
//!
//! Wraps one fitted model per group and scores every cell under its own
//! group's model. A single-sample run is one group, so there is exactly one
//! code path from here down through residual variance, HVG selection and PCA.

use crate::errors::BixverseErrors;
use crate::single_cell::sc_processing::residuals::{
    ResidualSource, intersect_gene_sets, validate_clip_range, validate_groups,
    validate_residual_row_inputs,
};

use super::model::{SctCellContext, SctGeneParams, SctModel, fill_residual_row};

//////////////////
// SctResiduals //
//////////////////

/// Pearson residuals from one or more fitted scTransform models.
///
/// The gene axis is the intersection of the per-group modelled sets: a gene
/// that one sample's `min_cells` filter dropped has no model there, so it
/// cannot carry a residual for that sample's cells and does not belong on a
/// shared axis. This is Seurat's "present in all layers" rule.
#[derive(Clone, Debug)]
pub struct SctResiduals<'a> {
    /// One fitted model per group, indexed by group id.
    models: &'a [SctModel],
    /// Per-cell library sizes and covariates, in the selected cell order.
    cells: SctCellContext<'a>,
    /// Store gene indices modelled in every group, ascending.
    genes: Vec<usize>,
    /// `positions[group][gene_pos]` is that group's model row for the shared
    /// gene. Resolved once here rather than binary-searched per gene per pass.
    positions: Vec<Vec<u32>>,
    /// Group id per selected cell, in the selected cell order.
    group_of_cell: Vec<u32>,
}

impl<'a> SctResiduals<'a> {
    /// Builds a grouped residual source.
    ///
    /// ### Params
    ///
    /// * `models` - One fitted model per group, in group id order.
    /// * `cells` - Per-cell library sizes and covariates, over the selected
    ///   cells of every group together.
    /// * `group_of_cell` - Group id per selected cell. Must densely cover
    ///   `0..models.len()`.
    ///
    /// ### Returns
    ///
    /// The source, or a [`BixverseErrors`] when the grouping is malformed, a
    /// model disagrees with the supplied covariates, or no gene is modelled in
    /// every group.
    pub fn new(
        models: &'a [SctModel],
        cells: SctCellContext<'a>,
        group_of_cell: Vec<u32>,
    ) -> Result<Self, BixverseErrors> {
        let n_groups = validate_groups(&group_of_cell, cells.n_cells())?;
        if n_groups != models.len() {
            return Err(BixverseErrors::ResidualGroupModelCountMismatch {
                implied: n_groups,
                models: models.len(),
            });
        }

        for model in models {
            if model.n_coef == 0 {
                return Err(BixverseErrors::SctModelWithoutCoefficients);
            }
            if cells.covariates.n_covariates() + 1 != model.n_coef {
                return Err(BixverseErrors::SctCovariateCountMismatch {
                    model: model.n_coef - 1,
                    supplied: cells.covariates.n_covariates(),
                });
            }
            // The count matching is exactly the case where a reordered data
            // frame slips through: the coefficients would then be applied to
            // the wrong covariate, giving plausible residuals and a wrong
            // embedding. Names are the only thing that catches it.
            for (position, (model_name, supplied)) in model
                .covariate_names
                .iter()
                .zip(&cells.covariates.names)
                .enumerate()
            {
                if model_name != supplied {
                    return Err(BixverseErrors::SctCovariateNameMismatch {
                        position,
                        model: model_name.clone(),
                        supplied: supplied.clone(),
                    });
                }
            }
        }

        for model in models {
            validate_clip_range(model.clip_range)?;
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
            cells,
            genes,
            positions,
            group_of_cell,
        })
    }

    /// Builds a source over a single model, with every cell in one group.
    ///
    /// ### Params
    ///
    /// * `model` - The fitted model.
    /// * `cells` - Per-cell library sizes and covariates.
    ///
    /// ### Returns
    ///
    /// The source, or a [`BixverseErrors`] when the model and covariates
    /// disagree.
    pub fn single(
        model: &'a SctModel,
        cells: SctCellContext<'a>,
    ) -> Result<SctResiduals<'a>, BixverseErrors> {
        let group_of_cell = vec![0_u32; cells.n_cells()];
        Self::new(std::slice::from_ref(model), cells, group_of_cell)
    }

    /// The fitted models, in group id order.
    ///
    /// ### Returns
    ///
    /// The models.
    pub fn models(&self) -> &[SctModel] {
        self.models
    }

    /// The per-cell context the residuals are scored against.
    ///
    /// ### Returns
    ///
    /// The context.
    pub fn cells(&self) -> &SctCellContext<'a> {
        &self.cells
    }

    /// Group id per selected cell.
    ///
    /// Same as the trait method, available without importing
    /// [`ResidualSource`].
    ///
    /// ### Returns
    ///
    /// The group map.
    pub fn group_of_cell_slice(&self) -> &[u32] {
        &self.group_of_cell
    }

    /// Where a shared-axis gene sits in one group's own model.
    ///
    /// ### Params
    ///
    /// * `group` - Group id.
    /// * `gene_pos` - Position on the shared gene axis.
    ///
    /// ### Returns
    ///
    /// The position within `models[group]`.
    pub fn model_gene_position(&self, group: usize, gene_pos: usize) -> usize {
        self.positions[group][gene_pos] as usize
    }
}

impl ResidualSource for SctResiduals<'_> {
    fn genes(&self) -> &[usize] {
        &self.genes
    }

    fn n_cells(&self) -> usize {
        self.cells.n_cells()
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
        if out.len() != self.cells.n_cells() {
            return Err(BixverseErrors::LengthMismatch {
                name: "out",
                expected: self.cells.n_cells(),
                found: out.len(),
            });
        }
        validate_residual_row_inputs(counts, indices, out.len())?;

        let per_group: Vec<SctGeneParams<'_>> = self
            .models
            .iter()
            .zip(&self.positions)
            .map(|(model, pos)| SctGeneParams::new(model, pos[gene_pos] as usize))
            .collect();

        fill_residual_row(counts, indices, &self.cells, out, |c| {
            &per_group[self.group_of_cell[c] as usize]
        });

        Ok(())
    }
}
