//! NEBULA over the streamed single-cell store, see He et al., Commun Biol,
//! 2021.
//!
//! The model itself lives in [`edge_rs`], ported from the `nebula` package's
//! own C++ and gated against it. Everything here is the adapter: pick the
//! cells, order them by subject, stream the genes in batches and hand each
//! batch to [`edge_rs::sc::nebula::nebula_sparse`].
//!
//! Two things make the batching free. NEBULA is gene-independent, so splitting
//! the genes changes nothing about the answer, and a gene chunk is already a
//! sparse row over cells, so the CSR the kernels want is a concatenation rather
//! than a conversion.
//!
//! Cells have to arrive grouped by subject, because `edge-rs` reads the subject
//! labels as a run encoding. The caller does not: this module sorts
//! `cells_to_keep` and permutes the design and offsets to match.

use edge_rs::prelude::{CompressedSparse, SparseFormat};
use edge_rs::sc::nebula::{NebulaMethod, NebulaParams, nebula_sparse};
use edge_rs::sc::shrink::{sc_residual_df, shrink_sc_dispersion};
use edge_rs::sc::test::{ScTested, glm_sc_test};
use indexmap::IndexSet;
use std::time::Instant;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Genes read and fitted per batch.
///
/// A NEBULA fit is milliseconds to seconds per gene, so the batch is never the
/// bottleneck. It exists to bound how much of the store is resident at once,
/// and 1000 matches the other streaming methods in this crate.
const GENE_BATCH_SIZE: usize = 1000;

/// Added to the mean count per cell before the log, for the shrinkage
/// covariate. edgePython's `log2(mean + 0.5)`.
const SHRINK_COVARIATE_OFFSET: f64 = 0.5;

///////////
// Enums //
///////////

/// Parses the NEBULA variant.
///
/// ### Params
///
/// * `s` - `"ln"` or `"hl"`, any case
///
/// ### Returns
///
/// The [NebulaMethod], or `None` if the string is not one of the two.
pub fn parse_nebula_method(s: &str) -> Option<NebulaMethod> {
    match s.to_lowercase().as_str() {
        "ln" => Some(NebulaMethod::Ln),
        "hl" => Some(NebulaMethod::Hl),
        _ => None,
    }
}

////////////
// Params //
////////////

/// Parameters for [run_nebula].
#[derive(Clone, Debug)]
pub struct NebulaScParams {
    /// The upstream knobs, defaulting to the R package's own.
    pub nebula: NebulaParams,
    /// Genes read and fitted per batch.
    pub gene_batch_size: usize,
    /// Shrink the cell-level overdispersions towards an empirical Bayes prior
    /// once the sweep is done.
    pub shrink_dispersion: bool,
    /// Which coefficient or contrast the Wald test reports.
    pub tested: ScTested,
}

impl Default for NebulaScParams {
    /// Upstream defaults, with the intercept tested.
    ///
    /// `tested` cannot be resolved without knowing the design width, so it
    /// starts on the intercept and the caller is expected to set it.
    fn default() -> Self {
        Self {
            nebula: NebulaParams::default(),
            gene_batch_size: GENE_BATCH_SIZE,
            shrink_dispersion: true,
            tested: ScTested::Coef(0),
        }
    }
}

/////////////
// Results //
/////////////

/// Per-gene NEBULA fits and the Wald test on them.
///
/// Every vector is one entry per gene that survived the expression filter, in
/// the order the genes were requested. `gene_idx` maps back onto the original
/// gene indices.
#[derive(Clone, Debug)]
pub struct NebulaScRes {
    /// Original indices of the genes that survived the filter.
    pub gene_idx: Vec<usize>,
    /// Fixed effects on the user's design scale, row-major `n_kept * n_coef`.
    pub coefficients: Vec<f64>,
    /// Standard errors, row-major `n_kept * n_coef`.
    pub se: Vec<f64>,
    /// Subject-level overdispersion, nebula's `sigma^2`.
    pub subject_overdispersion: Vec<f64>,
    /// Cell-level overdispersion, nebula's `phi^-1`.
    pub cell_overdispersion: Vec<f64>,
    /// Cell-level overdispersion after empirical Bayes shrinkage, when
    /// `shrink_dispersion` was on.
    pub cell_overdispersion_shrunk: Option<Vec<f64>>,
    /// nebula's convergence code. Anything at or below `-20` is a likely
    /// failure.
    pub convergence: Vec<i32>,
    /// Whether the subject-level variance finished pinned on its lower bound,
    /// in which case the mixed model collapsed to a plain negative binomial.
    pub sigma_at_bound: Vec<bool>,
    /// Effect of the tested coefficient or contrast, natural log scale.
    pub log_fc: Vec<f64>,
    /// Standard error of that effect.
    pub effect_se: Vec<f64>,
    /// Wald statistic, `log_fc / effect_se`.
    pub z: Vec<f64>,
    /// Two-sided p-value.
    pub p_val: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value.
    pub fdr: Vec<f64>,
    /// Number of coefficients, the stride of `coefficients` and `se`.
    pub n_coef: usize,
}

/////////////
// Helpers //
/////////////

/// Orders the selected cells so each subject's cells form one run.
///
/// Sorts on `(subject, global cell index)` rather than on the subject alone.
/// The second key is what makes the fit a function of the cell *set* instead of
/// the order the caller happened to list it in: NEBULA treats cells within a
/// subject as exchangeable, but its likelihood is a sum over them, and floating
/// point addition is not associative, so a different within-subject order moves
/// the optimiser's answer around at roughly `1e-8` relative. Small, and not
/// something a caller should have to think about.
///
/// ### Params
///
/// * `cells_to_keep` - Global indices of the cells to analyse
/// * `subject_ids` - Subject label per global cell
///
/// ### Returns
///
/// The positions into `cells_to_keep` in subject order.
fn subject_order(cells_to_keep: &[usize], subject_ids: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..cells_to_keep.len()).collect();
    order.sort_by_key(|&i| (subject_ids[cells_to_keep[i]], cells_to_keep[i]));
    order
}

/// Concatenates gene chunks into the gene-major matrix `edge-rs` reads.
///
/// The chunks have already been filtered to the cell selection, which remaps
/// their indices into `0..n_cells` in selection order and leaves them
/// ascending. So each chunk is a finished CSR row and this is a copy, not a
/// conversion.
///
/// ### Params
///
/// * `chunks` - One filtered chunk per gene, in the requested order
/// * `n_cells` - Number of cells in the selection
///
/// ### Returns
///
/// The counts as CSR over `(chunks.len(), n_cells)`.
fn chunks_to_csr(
    chunks: &[CscGeneChunk],
    n_cells: usize,
) -> Result<CompressedSparse<f64>, BixverseErrors> {
    let nnz: usize = chunks.iter().map(|c| c.data_raw.len()).sum();
    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);
    let mut indptr = Vec::with_capacity(chunks.len() + 1);
    indptr.push(0u32);

    for chunk in chunks {
        data.extend(chunk.data_raw.iter().map(|v| v as f64));
        indices.extend_from_slice(&chunk.indices);
        indptr.push(data.len() as u32);
    }

    Ok(CompressedSparse::from_parts(
        data,
        indices,
        indptr,
        SparseFormat::Csr,
        (chunks.len(), n_cells),
    )?)
}

//////////
// Main //
//////////

/// Fits NEBULA's negative binomial gamma mixed model to every requested gene.
///
/// Genes are streamed in batches and fitted independently, which is exact:
/// NEBULA's only cross-gene step is its own expression filter and that is
/// evaluated per gene. A batch that loses every gene to the filter is skipped;
/// only an empty sweep is an error.
///
/// ### Params
///
/// * `gene_reader` - Gene-major store the counts come from
/// * `cell_reader` - Cell-major store the library sizes come from. Pass
///   `gene_reader` again for an in-memory store, which serves both
/// * `cells_to_keep` - Global indices of the cells to analyse, in any order
/// * `gene_indices` - Indices of the genes to fit
/// * `subject_ids` - Subject label per global cell, as DIALOGUE takes them
/// * `design` - Predictors, row-major `cells_to_keep.len() * n_coef`, rows
///   aligned to `cells_to_keep` and including an intercept
/// * `n_coef` - Number of design columns
/// * `offset` - Strictly positive scaling factor per selected cell, aligned to
///   `cells_to_keep`, or `None` to use the library sizes
/// * `params` - See [NebulaScParams]
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [NebulaScRes], or the first shape problem found,
/// [`BixverseErrors::NebulaNoGenesKept`] if nothing survived the filter, or an
/// [`edge_rs`] error from the fit itself.
///
/// ### References
///
/// He et al., Communications Biology 4, 629, 2021
#[allow(clippy::too_many_arguments)]
pub fn run_nebula<S: SingleCellReading>(
    gene_reader: &S,
    cell_reader: &S,
    cells_to_keep: &[usize],
    gene_indices: &[usize],
    subject_ids: &[usize],
    design: &[f64],
    n_coef: usize,
    offset: Option<&[f64]>,
    params: &NebulaScParams,
    verbose: usize,
) -> Result<NebulaScRes, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start_all = Instant::now();

    let n_cells = cells_to_keep.len();
    let header = gene_reader.get_header();

    if n_coef == 0 {
        return Err(BixverseErrors::MustBePositive("n_coef".to_string()));
    }
    if params.gene_batch_size == 0 {
        return Err(BixverseErrors::MustBePositive(
            "gene_batch_size".to_string(),
        ));
    }
    if design.len() != n_cells * n_coef {
        return Err(BixverseErrors::DgeShapeMismatch {
            name: "design",
            expected: n_cells * n_coef,
            got: design.len(),
        });
    }
    if subject_ids.len() != header.total_cells {
        return Err(BixverseErrors::DgeShapeMismatch {
            name: "subject_ids",
            expected: header.total_cells,
            got: subject_ids.len(),
        });
    }
    if let Some(o) = offset
        && o.len() != n_cells
    {
        return Err(BixverseErrors::DgeShapeMismatch {
            name: "offset",
            expected: n_cells,
            got: o.len(),
        });
    }
    if let Some(&cell) = cells_to_keep.iter().find(|&&c| c >= header.total_cells) {
        return Err(BixverseErrors::ChunkIndexNotFound(cell));
    }
    if let Some(&gene) = gene_indices.iter().find(|&&g| g >= header.total_genes) {
        return Err(BixverseErrors::ChunkIndexNotFound(gene));
    }

    // Subject-contiguous cells, and the design and offsets moved with them.
    let order = subject_order(cells_to_keep, subject_ids);
    let ordered_cells: Vec<usize> = order.iter().map(|&i| cells_to_keep[i]).collect();
    let subject_run: Vec<usize> = ordered_cells.iter().map(|&c| subject_ids[c]).collect();
    let n_subjects = subject_run.windows(2).filter(|w| w[0] != w[1]).count() + 1;

    let mut design_ordered = Vec::with_capacity(design.len());
    for &i in &order {
        design_ordered.extend_from_slice(&design[i * n_coef..(i + 1) * n_coef]);
    }

    let offsets: Vec<f64> = match offset {
        Some(o) => order.iter().map(|&i| o[i]).collect(),
        None => cell_reader
            .read_cell_library_sizes(&ordered_cells)?
            .iter()
            .map(|&v| v as f64)
            .collect(),
    };

    // `filter_selected_cells` walks this in order, so the local cell indices it
    // writes are already subject-contiguous. It also deduplicates, and a
    // silently shorter cell axis than the design assumes is the kind of thing
    // that fits a model rather than failing, so check for it.
    let cell_set: IndexSet<u32> = ordered_cells.iter().map(|&c| c as u32).collect();
    if cell_set.len() != n_cells {
        return Err(BixverseErrors::DgeShapeMismatch {
            name: "cells_to_keep (holds duplicates)",
            expected: n_cells,
            got: cell_set.len(),
        });
    }

    let n_batches = gene_indices.len().div_ceil(params.gene_batch_size);
    let mut gene_idx: Vec<usize> = Vec::new();
    let mut coefficients: Vec<f64> = Vec::new();
    let mut covariance: Vec<f64> = Vec::new();
    let mut se: Vec<f64> = Vec::new();
    let mut subject_overdispersion: Vec<f64> = Vec::new();
    let mut cell_overdispersion: Vec<f64> = Vec::new();
    let mut convergence: Vec<i32> = Vec::new();
    let mut sigma_at_bound: Vec<bool> = Vec::new();
    let mut mean_count: Vec<f64> = Vec::new();

    for batch in 0..n_batches {
        let start = batch * params.gene_batch_size;
        let end = ((batch + 1) * params.gene_batch_size).min(gene_indices.len());
        let batch_genes = &gene_indices[start..end];

        let start_loading = Instant::now();
        let chunks = gene_reader.read_gene_parallel_filtered(batch_genes, &cell_set)?;
        let sparse = chunks_to_csr(&chunks, n_cells)?;
        if verbosity.detailed_verbosity() {
            println!("   Loaded batch in: {:.2?}.", start_loading.elapsed());
        }

        let start_fit = Instant::now();
        let fit = match nebula_sparse(
            &sparse,
            &subject_run,
            &design_ordered,
            n_coef,
            Some(&offsets),
            Some(params.nebula),
        ) {
            Ok(fit) => Some(fit),
            // A batch can legitimately lose every gene to the expression
            // filter. Only an empty sweep is an error.
            Err(edge_rs::errors::EdgeErrors::NoGenesAfterFiltering { .. }) => None,
            Err(e) => return Err(e.into()),
        };
        if verbosity.detailed_verbosity() {
            println!("   Fitted batch in: {:.2?}.", start_fit.elapsed());
        }

        if let Some(fit) = fit {
            for &row in &fit.gene_index {
                gene_idx.push(chunks[row].original_index);
                let (_, values) = sparse.outer(row);
                mean_count.push(values.iter().sum::<f64>() / n_cells as f64);
            }
            coefficients.extend_from_slice(&fit.coefficients);
            covariance.extend_from_slice(&fit.covariance);
            se.extend_from_slice(&fit.se);
            subject_overdispersion.extend_from_slice(&fit.subject_overdispersion);
            cell_overdispersion.extend_from_slice(&fit.cell_overdispersion);
            convergence.extend_from_slice(&fit.convergence);
            sigma_at_bound.extend_from_slice(&fit.sigma_at_bound);
        }

        // Genes read, not genes kept: a batch the filter empties costs the same
        // disk pass as any other.
        if verbosity.normal_verbosity() {
            report_decile_progress(end, start, gene_indices.len(), "genes", start_all.elapsed());
        }
    }

    let n_kept = gene_idx.len();
    if n_kept == 0 {
        return Err(BixverseErrors::NebulaNoGenesKept {
            n_genes: gene_indices.len(),
        });
    }

    let test = glm_sc_test(&coefficients, &covariance, n_kept, n_coef, &params.tested)?;

    // The shrinkage is a joint fit over every usable gene, so it can only run
    // once the whole sweep is in.
    let cell_overdispersion_shrunk = if params.shrink_dispersion {
        let covariate: Vec<f64> = mean_count
            .iter()
            .map(|m| (m + SHRINK_COVARIATE_OFFSET).log2())
            .collect();
        let df_residual = sc_residual_df(n_cells, n_coef, n_subjects);
        let shrunk = shrink_sc_dispersion(
            &cell_overdispersion,
            &convergence,
            Some(&covariate),
            df_residual,
            None,
        )?;
        Some(shrunk.dispersion_shrunk)
    } else {
        None
    };

    if verbosity.normal_verbosity() {
        println!(
            "NEBULA: fitted {n_kept} of {} genes in {:.2?}.",
            gene_indices.len(),
            start_all.elapsed()
        );
    }

    Ok(NebulaScRes {
        gene_idx,
        coefficients,
        se,
        subject_overdispersion,
        cell_overdispersion,
        cell_overdispersion_shrunk,
        convergence,
        sigma_at_bound,
        log_fc: test.log_fc,
        effect_se: test.se,
        z: test.z,
        p_val: test.p_value,
        fdr: test.fdr,
        n_coef,
    })
}
