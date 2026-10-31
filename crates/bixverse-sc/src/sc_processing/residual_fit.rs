//! A fitted residual model over a cell selection of an on-disk experiment, and
//! everything computed from it: residual variance HVGs, residual PCA and
//! scTransform corrected counts.
//!
//! scTransform and analytic Pearson differ only in the fit. [`ResidualFit`]
//! carries either, together with the cells it was fitted on, and hands out the
//! matching [`ResidualSource`] for the downstream steps. Small selections are
//! read into memory first ([`CellSubset`]), as the HVG entry points do.

use std::path::Path;

use crate::prelude::*;
use crate::sc_processing::analytic_pearson::model::{AprParams, AprResiduals};
use crate::sc_processing::analytic_pearson::stream::{
    AprGroupedFit, apr_gene_pass, build_apr_model, cell_totals_over_genes,
    fit_analytic_pearson_grouped,
};
use crate::sc_processing::hvg::select_residual_hvg;
use crate::sc_processing::pca::{SingleCellPcaParams, SingleCellPcaRes, pca_on_sc_residuals};
use crate::sc_processing::residuals::{intersect_gene_sets, residual_variance};
use crate::sc_processing::sctransform::model::{SctCellContext, SctCovariates, SctParams};
use crate::sc_processing::sctransform::residuals::SctResiduals;
use crate::sc_processing::sctransform::stream::{
    SctGroupedFit, SctStreamOpts, fit_sctransform, fit_sctransform_grouped, sct_corrected_counts,
};
use bixverse_sc_data::in_memory_io::CellSubset;

///////////
// Types //
///////////

/// The parameters of a residual fit, which also pick the method.
#[derive(Clone, Copy, Debug)]
pub enum ResidualParams {
    /// scTransform v2
    Sct(SctParams),
    /// Analytic Pearson residuals
    Apr(AprParams),
}

/// The fitted model of either method.
#[derive(Clone, Debug)]
pub enum ResidualModel {
    /// scTransform, one model per group.
    Sct {
        /// The per-group models plus the shared gene axis.
        fit: SctGroupedFit,
        /// `log10(library size)` per selected cell, the model's offset.
        log10_umi: Vec<f64>,
        /// Cell-level covariates, in the order the models were fitted with.
        covariates: SctCovariates,
    },
    /// Analytic Pearson, one model per group.
    Apr {
        /// The per-group models plus the per-cell totals.
        fit: AprGroupedFit,
    },
}

/// A fitted residual model and the cells it was fitted on.
#[derive(Clone, Debug)]
pub struct ResidualFit {
    /// The model.
    pub model: ResidualModel,
    /// The fitted cells, 0-based store indices in fitting order. Every later
    /// use must hand over exactly these.
    pub cell_indices: Vec<usize>,
}

impl ResidualFit {
    /// The method name, as the R side records it.
    ///
    /// ### Returns
    ///
    /// `"sctransform"` or `"analytic_pearson"`.
    pub fn method(&self) -> &'static str {
        match self.model {
            ResidualModel::Sct { .. } => "sctransform",
            ResidualModel::Apr { .. } => "analytic_pearson",
        }
    }

    /// Number of groups, one model each.
    ///
    /// ### Returns
    ///
    /// The group count.
    pub fn n_groups(&self) -> usize {
        match &self.model {
            ResidualModel::Sct { fit, .. } => fit.models.len(),
            ResidualModel::Apr { fit } => fit.models.len(),
        }
    }

    /// Group id per fitted cell.
    ///
    /// ### Returns
    ///
    /// The group map, in fitting order.
    pub fn group_of_cell(&self) -> &[u32] {
        match &self.model {
            ResidualModel::Sct { fit, .. } => &fit.group_of_cell,
            ResidualModel::Apr { fit } => &fit.group_of_cell,
        }
    }

    /// Store genes modelled in every group, ascending.
    ///
    /// ### Returns
    ///
    /// The shared gene axis; empty if no gene is modelled everywhere.
    pub fn genes(&self) -> Vec<usize> {
        match &self.model {
            ResidualModel::Sct { fit, .. } => fit.genes.clone(),
            ResidualModel::Apr { fit } => {
                let sets: Vec<&[usize]> = fit.models.iter().map(|m| m.genes.as_slice()).collect();
                intersect_gene_sets(&sets).unwrap_or_default()
            }
        }
    }

    /// Refuse any cell selection but the fitted one.
    ///
    /// ### Params
    ///
    /// * `cell_indices` - The selection about to be used
    ///
    /// ### Returns
    ///
    /// `()`, or [`ScErrors::ResidualFitCellMismatch`].
    pub fn check_cells(&self, cell_indices: &[usize]) -> Result<(), ScErrors> {
        if self.cell_indices != cell_indices {
            return Err(ScErrors::ResidualFitCellMismatch {
                fitted: self.cell_indices.len(),
                given: cell_indices.len(),
            });
        }
        Ok(())
    }

    /// Run a closure against the model's [`ResidualSource`].
    ///
    /// ### Params
    ///
    /// * `f` - Receives the source
    ///
    /// ### Returns
    ///
    /// Whatever `f` returns, or the error building the source raised.
    pub fn with_source<R>(
        &self,
        f: impl FnOnce(&dyn ResidualSource) -> Result<R, ScErrors>,
    ) -> Result<R, ScErrors> {
        match &self.model {
            ResidualModel::Sct {
                fit,
                log10_umi,
                covariates,
            } => {
                let cells = SctCellContext::new(log10_umi, covariates)?;
                let source = SctResiduals::new(&fit.models, cells, fit.group_of_cell.clone())?;
                f(&source)
            }
            ResidualModel::Apr { fit } => {
                let source =
                    AprResiduals::new(&fit.models, &fit.cell_totals, fit.group_of_cell.clone())?;
                f(&source)
            }
        }
    }
}

/////////////
// Fitting //
/////////////

/// Fit scTransform over one reader.
///
/// An ungrouped fit is not the one-group case of the grouped fit: it
/// subsamples for step 1 over the whole selection.
///
/// ### Params
///
/// * `reader` - Gene-major store, on disk or an in-memory cell subset
/// * `cell_set` - The selected cells, as indices into `reader`
/// * `groups` - Optional group id per selected cell
/// * `library_sizes` - Library size per selected cell
/// * `covariates` - Cell-level covariates over the selected cells
/// * `params` - Tuning knobs
/// * `opts` - Disk batching, seed and reporting
///
/// ### Returns
///
/// The grouped fit, one model when `groups` is `None`.
fn fit_sct_on_reader<S: SingleCellReading>(
    reader: &S,
    cell_set: &[usize],
    groups: Option<&[u32]>,
    library_sizes: &[f64],
    covariates: &SctCovariates,
    params: &SctParams,
    opts: SctStreamOpts,
) -> Result<SctGroupedFit, ScErrors> {
    match groups {
        None => {
            let (model, _pass) = fit_sctransform(
                reader,
                cell_set,
                library_sizes,
                covariates,
                params,
                None,
                None,
                opts,
            )?;
            Ok(SctGroupedFit {
                genes: model.genes.clone(),
                models: vec![model],
                passes: Vec::new(),
                group_of_cell: vec![0_u32; cell_set.len()],
            })
        }
        Some(groups) => fit_sctransform_grouped(
            reader,
            cell_set,
            groups,
            library_sizes,
            covariates,
            params,
            opts,
        ),
    }
}

/// Fit analytic Pearson residuals over a gene- and a cell-major reader.
///
/// ### Params
///
/// * `gene_reader` - Gene-major store
/// * `cell_reader` - Cell-major store over the same cells
/// * `cell_set` - The selected cells, as indices into both readers
/// * `groups` - Optional group id per selected cell
/// * `params` - Tuning knobs
/// * `opts` - Disk batching and reporting
///
/// ### Returns
///
/// The grouped fit, one model when `groups` is `None`.
fn fit_apr_on_readers<G: SingleCellReading, C: SingleCellReading>(
    gene_reader: &G,
    cell_reader: &C,
    cell_set: &[usize],
    groups: Option<&[u32]>,
    params: &AprParams,
    opts: SctStreamOpts,
) -> Result<AprGroupedFit, ScErrors> {
    match groups {
        None => {
            let pass = apr_gene_pass(gene_reader, cell_set, params, opts)?;
            let totals = cell_totals_over_genes(cell_reader, cell_set, &pass.retained)?;
            let model = build_apr_model(&pass, &totals, params)?;
            Ok(AprGroupedFit {
                models: vec![model],
                cell_totals: totals,
                group_of_cell: vec![0_u32; cell_set.len()],
            })
        }
        Some(groups) => {
            fit_analytic_pearson_grouped(gene_reader, cell_reader, cell_set, groups, params, opts)
        }
    }
}

/// Fit a residual model over a cell selection of an on-disk experiment.
///
/// The library sizes for scTransform are read from the store rather than
/// asked of the caller, where a stale vector could silently shift every
/// offset. Covariates (scTransform only) must be aligned to `cell_set`.
///
/// ### Params
///
/// * `f_path_gene` - Path to the `counts_genes.bin` file
/// * `f_path_cell` - Path to the `counts_cells.bin` file
/// * `cell_set` - The selected cells, 0-based, unique, in fitting order
/// * `groups` - Optional group id per selected cell, densely covering
///   `0..n_groups`; `None` fits one model
/// * `covariates` - Cell-level covariates, empty for none
/// * `params` - The method and its parameters
/// * `opts` - Disk batching, seed and reporting
///
/// ### Returns
///
/// The [`ResidualFit`] over `cell_set`.
///
/// ### References
///
/// Choudhary & Satija, Genome Biology, 2022; Lause, Berens & Kobak, Genome
/// Biology, 2021
pub fn fit_residuals_on_store(
    f_path_gene: &str,
    f_path_cell: &str,
    cell_set: &[usize],
    groups: Option<&[u32]>,
    covariates: SctCovariates,
    params: ResidualParams,
    opts: SctStreamOpts,
) -> Result<ResidualFit, ScErrors> {
    let gene_reader = ParallelSparseReader::new(f_path_gene)?;
    let cell_reader = ParallelSparseReader::new(f_path_cell)?;
    let n_total = gene_reader.get_header().total_cells;
    let subset = CellSubset::load(f_path_cell, cell_set, n_total, opts.verbose > 0)?;

    // the drivers see the renumbered subset; the fit keeps the original
    // selection as its cells
    let model = match params {
        ResidualParams::Sct(sct_params) => {
            covariates.validate(cell_set.len())?;
            let library_sizes: Vec<f64> = cell_reader
                .read_cell_library_sizes(cell_set)?
                .into_iter()
                .map(|s| s as f64)
                .collect();
            let log10_umi: Vec<f64> = library_sizes.iter().map(|&s| s.log10()).collect();

            let fit = match &subset {
                Some(sub) => fit_sct_on_reader(
                    &sub.reader()?,
                    &sub.local,
                    groups,
                    &library_sizes,
                    &covariates,
                    &sct_params,
                    opts,
                )?,
                None => fit_sct_on_reader(
                    &gene_reader,
                    cell_set,
                    groups,
                    &library_sizes,
                    &covariates,
                    &sct_params,
                    opts,
                )?,
            };
            ResidualModel::Sct {
                fit,
                log10_umi,
                covariates,
            }
        }
        ResidualParams::Apr(apr_params) => {
            let fit = match &subset {
                Some(sub) => fit_apr_on_readers(
                    &sub.reader()?,
                    &sub.cell_reader()?,
                    &sub.local,
                    groups,
                    &apr_params,
                    opts,
                )?,
                None => fit_apr_on_readers(
                    &gene_reader,
                    &cell_reader,
                    cell_set,
                    groups,
                    &apr_params,
                    opts,
                )?,
            };
            ResidualModel::Apr { fit }
        }
    };

    Ok(ResidualFit {
        model,
        cell_indices: cell_set.to_vec(),
    })
}

/////////////////////
// Downstream uses //
/////////////////////

/// Per-group residual variance and the HVGs selected from it.
#[derive(Clone, Debug)]
pub struct ResidualVarianceRes {
    /// The model's gene axis, ascending.
    pub genes: Vec<usize>,
    /// One variance per gene of `genes`, per group.
    pub variance: Vec<Vec<f64>>,
    /// The union of the top `n_hvg` genes of each group, ascending.
    pub hvg: Vec<usize>,
}

/// Residual variance per gene and group, and the residual HVGs.
///
/// ### Params
///
/// * `f_path_gene` - Path to the `counts_genes.bin` file
/// * `f_path_cell` - Path to the `counts_cells.bin` file
/// * `fit` - The fitted model
/// * `cell_set` - The selected cells; must be the fitted ones
/// * `n_hvg` - Variable features per group
/// * `opts` - Disk batching and reporting
///
/// ### Returns
///
/// The [`ResidualVarianceRes`].
pub fn residual_variance_on_store(
    f_path_gene: &str,
    f_path_cell: &str,
    fit: &ResidualFit,
    cell_set: &[usize],
    n_hvg: usize,
    opts: SctStreamOpts,
) -> Result<ResidualVarianceRes, ScErrors> {
    fit.check_cells(cell_set)?;
    let reader = ParallelSparseReader::new(f_path_gene)?;
    let n_total = reader.get_header().total_cells;
    let subset = CellSubset::load(f_path_cell, cell_set, n_total, opts.verbose > 0)?;
    let in_memory = subset.as_ref().map(CellSubset::reader).transpose()?;

    fit.with_source(|source| {
        let variance = match (&subset, &in_memory) {
            (Some(sub), Some(mem)) => residual_variance(mem, source, &sub.local, opts)?,
            _ => residual_variance(&reader, source, cell_set, opts)?,
        };
        let hvg = select_residual_hvg(&variance, source.genes(), n_hvg)?;
        Ok(ResidualVarianceRes {
            genes: source.genes().to_vec(),
            variance,
            hvg,
        })
    })
}

/// PCA on the residuals of a fitted model.
///
/// ### Params
///
/// * `f_path_gene` - Path to the `counts_genes.bin` file
/// * `f_path_cell` - Path to the `counts_cells.bin` file
/// * `fit` - The fitted model
/// * `cell_set` - The selected cells; must be the fitted ones
/// * `gene_indices` - The genes, all covered by the model
/// * `no_pcs` - Number of principal components
/// * `params_pca` - Parameters for this PCA run; `clr` and
///   `normalise_variance` must be off
/// * `seed` - Seed for the randomised SVD
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [`SingleCellPcaRes`].
#[allow(clippy::too_many_arguments)]
pub fn pca_residuals_on_store(
    f_path_gene: &str,
    f_path_cell: &str,
    fit: &ResidualFit,
    cell_set: &[usize],
    gene_indices: &[usize],
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    seed: usize,
    verbose: usize,
) -> SingleCellPcaRes {
    fit.check_cells(cell_set)?;
    let reader = ParallelSparseReader::new(f_path_gene)?;
    let n_total = reader.get_header().total_cells;
    let subset = CellSubset::load(f_path_cell, cell_set, n_total, verbose > 0)?;
    let in_memory = subset.as_ref().map(CellSubset::reader).transpose()?;

    let (scores, loadings, s, _) = fit.with_source(|source| match (&subset, &in_memory) {
        (Some(sub), Some(mem)) => pca_on_sc_residuals(
            mem,
            &sub.local,
            gene_indices,
            no_pcs,
            params_pca,
            source,
            seed,
            false,
            verbose,
        ),
        _ => pca_on_sc_residuals(
            &reader,
            cell_set,
            gene_indices,
            no_pcs,
            params_pca,
            source,
            seed,
            false,
            verbose,
        ),
    })?;
    Ok((scores, loadings, s))
}

/// Write scTransform corrected counts as a gene-major store.
///
/// ### Params
///
/// * `f_path_gene` - Path to the source `counts_genes.bin` file
/// * `fit` - The fitted model; must be scTransform
/// * `cell_set` - The selected cells; must be the fitted ones
/// * `f_path_out` - The gene-major store to write
/// * `opts` - Disk batching and reporting
///
/// ### Returns
///
/// The store genes written, ascending, which are the new store's gene axis.
pub fn sct_corrected_counts_on_store<P: AsRef<Path>>(
    f_path_gene: &str,
    fit: &ResidualFit,
    cell_set: &[usize],
    f_path_out: P,
    opts: SctStreamOpts,
) -> Result<Vec<usize>, ScErrors> {
    fit.check_cells(cell_set)?;
    let ResidualModel::Sct {
        fit,
        log10_umi,
        covariates,
    } = &fit.model
    else {
        return Err(ScErrors::ResidualCorrectedCountsNeedSct);
    };
    let reader = ParallelSparseReader::new(f_path_gene)?;
    let cells = SctCellContext::new(log10_umi, covariates)?;
    let source = SctResiduals::new(&fit.models, cells, fit.group_of_cell.clone())?;
    sct_corrected_counts(&reader, &source, cell_set, f_path_out, opts)?;
    Ok(source.genes().to_vec())
}
