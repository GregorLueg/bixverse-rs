//! MAGIC imputation (Markov affinity-based graph imputation of cells).
//!
//! Three steps of a row-stochastic diffusion operator over the kNN graph, so
//! every cell becomes a weighted average of its neighbourhood.
//!
//! ### What this will do to your results
//!
//! Neighbourhoods overlap, so smoothing over them makes nearby cells resemble
//! each other in *every* gene at once. Gene-gene correlation after imputation is
//! therefore inflated, badly, and the inflation is a property of the graph
//! rather than of the biology. Do not feed imputed values into Hotspot, SCENIC,
//! differential correlation or CoReMo: those methods measure exactly the
//! quantity MAGIC manufactures, and they will happily report the graph back to
//! you as a finding. Visualisation and trend fitting are what it is for.
//!
//! The gene trends in [crate::single_cell::sc_trajectory::gene_trends] already
//! fit a smooth posterior over pseudotime. Running MAGIC first smooths twice.
//! That is a defensible presentation choice and not usually a necessary one.
//!
//! ### Scope
//!
//! Gene-subset only. The output is unavoidably dense, and the reference imputes
//! everything because its data already sits in memory; the disk-backed store
//! here exists precisely so that it does not have to. Pick the genes you want
//! to plot or fit and impute those. Genome-wide imputation is not offered, the
//! binary store is never written to, and nothing here bumps a file version.
//!
//! ### Implementation
//!
//! `T^3` is never formed. At `knn = 30` the operator carries ~30 non-zeros per
//! row, `T^2` ~900 and `T^3` ~27,000, so the explicit cube is on the order of
//! 2.7e9 non-zeros at 100k cells. Three applications give the identical answer
//! at ~30 per row, which is what [csr_matmul_dense_block] does.
//!
//! The operator is Palantir's, not graphtools': adaptive bandwidth taken at the
//! `knn/3`-th neighbour distance, `exp(-d / sigma_i)`, symmetrised by plain
//! addition, then row-normalised. MAGIC's own Python path uses the `knn`-th
//! neighbour with a tunable decay exponent and a `1e-4` affinity threshold. The
//! constants differ; the shape does not.
//!
//! ### References
//!
//! van Dijk, et al., Cell, 2018.
//! Setty, et al., Nat. Biotechnol., 2019 (the operator this uses).
//! Andrews and Hemberg, F1000Research, 2018 (why the warning above is there).

use faer::Mat;
use rayon::prelude::*;
use std::time::Instant;

use crate::core::math::sparse::{csr_matmul_dense_block, normalise_csr_rows_l1};
use crate::prelude::*;
use crate::single_cell::mc_generation::seacells::compute_diffusion_kernel;
use crate::single_cell::sc_data::data_io::SingleCellReading;

////////////
// Consts //
////////////

/// Element budget for the dense output.
///
/// 1e9 `f32` elements is four gigabytes, which is roughly where an R session on
/// a laptop starts swapping. Exceeding it raises
/// [BixverseErrors::MagicOutputTooLarge] rather than printing a warning: this
/// crate is driven from R, where stderr is easily missed, and the alternative
/// failure mode is the session being killed by the OOM reaper with no message
/// at all. `allow_large` on [MagicParams] is the deliberate override.
const MAGIC_MAX_ELEMENTS: usize = 1_000_000_000;

/// Genes read per disk batch.
///
/// Matches the streaming HVG path. Peak scratch is
/// `2 * n_cells * gene_batch_size` floats for the ping-pong buffers, so this is
/// the knob to turn down on a memory-tight machine. It is clamped to the gene
/// count at entry, so a small selection never over-allocates.
const MAGIC_GENE_BATCH_SIZE: usize = 1000;

/////////////
// Helpers //
/////////////

/// Guard the dense output against the memory budget.
///
/// ### Params
///
/// * `n_cells` - Rows of the output.
/// * `n_genes` - Columns of the output.
/// * `params` - Parameters, supplying the override.
///
/// ### Returns
///
/// `Ok(())`, or [BixverseErrors::MagicOutputTooLarge].
fn check_output_size(
    n_cells: usize,
    n_genes: usize,
    params: &MagicParams,
) -> Result<(), BixverseErrors> {
    if params.allow_large {
        return Ok(());
    }
    // Saturating, so an absurd request reports rather than wrapping to a small
    // number and sailing through.
    if n_cells.saturating_mul(n_genes) > MAGIC_MAX_ELEMENTS {
        return Err(BixverseErrors::MagicOutputTooLarge {
            n_cells,
            n_genes,
            max_elements: MAGIC_MAX_ELEMENTS,
        });
    }
    Ok(())
}

////////////
// Params //
////////////

/// Which stored layer to impute.
///
/// The operator is row-stochastic, so it preserves per-cell mass and the
/// imputed values come back on the scale of whatever went in. Imputing raw
/// counts and imputing log-normalised counts are therefore different
/// operations, not the same one at a different scale, and the reference quietly
/// uses whatever happens to be in `.X`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MagicLayer {
    /// Log-normalised counts, `ln1p(count / library_size * target_size)`.
    ///
    /// The default, and what downstream trend fitting expects.
    #[default]
    Norm,
    /// Raw counts.
    Raw,
}

/// Parameters for the MAGIC entry points.
#[derive(Clone, Copy, Debug)]
pub struct MagicParams {
    /// Diffusion steps. Reference default: `3`.
    ///
    /// `0` is legal and returns the un-imputed dense block, which is how the
    /// same call site serves callers who want the raw matrix.
    pub n_steps: usize,
    /// Values strictly below this are zeroed after the last step. Reference:
    /// `1e-2`. Applied once at the end, not per pass.
    pub clip_threshold: f32,
    /// Genes per disk batch. Clamped to the gene count at entry.
    pub gene_batch_size: usize,
    /// Which stored layer to impute.
    pub layer: MagicLayer,
    /// Skip the `MAGIC_MAX_ELEMENTS` output-size check.
    pub allow_large: bool,
}

impl MagicParams {
    /// Construct the parameter set.
    ///
    /// ### Params
    ///
    /// * `n_steps` - Diffusion steps.
    /// * `clip_threshold` - Post-diffusion floor.
    /// * `gene_batch_size` - Genes per disk batch.
    /// * `layer` - Which stored layer to impute.
    /// * `allow_large` - Skip the output-size check.
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(
        n_steps: usize,
        clip_threshold: f32,
        gene_batch_size: usize,
        layer: MagicLayer,
        allow_large: bool,
    ) -> Self {
        Self {
            n_steps,
            clip_threshold,
            gene_batch_size,
            layer,
            allow_large,
        }
    }
}

impl Default for MagicParams {
    fn default() -> Self {
        Self {
            n_steps: 3,
            clip_threshold: 1e-2,
            gene_batch_size: MAGIC_GENE_BATCH_SIZE,
            layer: MagicLayer::Norm,
            allow_large: false,
        }
    }
}

//////////////
// Operator //
//////////////

/// Row-stochastic diffusion operator over a cell subset.
///
/// `T = D^-1 K` with `K` the symmetrised adaptive-bandwidth kernel. Built once
/// and reused across as many gene selections as the caller wants, which matters
/// because the kernel build is the expensive part.
///
/// This is deliberately not shared with a Palantir run. `multiscale_components`
/// consumes and drops its kernel, threading it back out would widen an already
/// wide result type, and the duplicate build is cheap next to the kNN search
/// that both of them depend on.
#[derive(Clone, Debug)]
pub struct MagicOperator {
    /// `T`, CSR, `n_sub` by `n_sub`, rows in `cell_indices` order.
    t: CompressedSparseData2<f32>,
    /// Global cell ids, one per row of `t`.
    cell_indices: Vec<usize>,
    /// Global cell id to local row, `u32::MAX` for cells outside the selection.
    /// Length is the store's `total_cells`, so gene chunks scatter straight in
    /// and `cell_indices` need not be sorted.
    lookup: Vec<u32>,
}

impl MagicOperator {
    /// Build the operator from a kNN graph over the selected cells.
    ///
    /// The kNN rows must be in `cell_indices` order and their entries must be
    /// *local* positions into that selection, which is the same convention
    /// [crate::single_cell::sc_trajectory::palantir::run_palantir] uses.
    ///
    /// ### Params
    ///
    /// * `knn_indices` - kNN indices per cell, local positions.
    /// * `knn_distances` - Matching kNN distances.
    /// * `squared_dist` - Whether the distances are squared.
    /// * `cell_indices` - Global cell ids, aligned with the kNN rows.
    /// * `total_cells` - `SparseDataHeader::total_cells` of the store the
    ///   selection indexes into.
    ///
    /// ### Returns
    ///
    /// The operator, or the matching [BixverseErrors] variant if the selection
    /// is misaligned with the kNN graph, holds a duplicate, or points outside
    /// the store.
    pub fn from_knn(
        knn_indices: &[Vec<usize>],
        knn_distances: &[Vec<f32>],
        squared_dist: bool,
        cell_indices: &[usize],
        total_cells: usize,
    ) -> Result<Self, BixverseErrors> {
        if knn_indices.len() != cell_indices.len() {
            return Err(BixverseErrors::MagicKnnSelectionMismatch {
                n_knn: knn_indices.len(),
                n_selected: cell_indices.len(),
            });
        }

        let mut lookup = vec![u32::MAX; total_cells];
        for (local, &global) in cell_indices.iter().enumerate() {
            if global >= total_cells {
                return Err(BixverseErrors::MagicCellOutOfRange {
                    cell: global,
                    total_cells,
                });
            }
            if lookup[global] != u32::MAX {
                return Err(BixverseErrors::MagicDuplicateCell { cell: global });
            }
            lookup[global] = local as u32;
        }

        let mut t = compute_diffusion_kernel(knn_indices, knn_distances, squared_dist)?;
        normalise_csr_rows_l1(&mut t)?;

        Ok(Self {
            t,
            cell_indices: cell_indices.to_vec(),
            lookup,
        })
    }

    /// Cells the operator spans.
    ///
    /// ### Returns
    ///
    /// The row count of `T`.
    pub fn n_cells(&self) -> usize {
        self.cell_indices.len()
    }

    /// Global cell ids, one per operator row.
    ///
    /// ### Returns
    ///
    /// The selection, in row order.
    pub fn cell_indices(&self) -> &[usize] {
        &self.cell_indices
    }

    /// The operator itself.
    ///
    /// ### Returns
    ///
    /// `T = D^-1 K` in CSR.
    pub fn transition_matrix(&self) -> &CompressedSparseData2<f32> {
        &self.t
    }
}

/////////////
// Results //
/////////////

/// Imputed expression, cells by genes.
#[derive(Clone, Debug)]
pub struct MagicImputed {
    /// `n_cells * n_genes`, row-major, rows in operator order.
    pub data: Vec<f32>,
    /// Rows, i.e. `operator.n_cells()`.
    pub n_cells: usize,
    /// Gene ids, in the order the caller asked for them.
    pub gene_indices: Vec<usize>,
    /// Global cell ids, one per row.
    pub cell_indices: Vec<usize>,
}

impl MagicImputed {
    /// Genes the result spans.
    ///
    /// ### Returns
    ///
    /// The column count.
    pub fn n_genes(&self) -> usize {
        self.gene_indices.len()
    }

    /// Copy into a column-major [faer::Mat].
    ///
    /// The faer-based consumers, gene trends among them, want this orientation.
    /// It is a genuine copy of `n_cells * n_genes` floats, so hold on to only
    /// one of the two.
    ///
    /// ### Returns
    ///
    /// `n_cells` by `n_genes`.
    pub fn to_mat(&self) -> Mat<f32> {
        let n_genes = self.n_genes();
        Mat::from_fn(self.n_cells, n_genes, |i, j| self.data[i * n_genes + j])
    }
}

//////////////////
// Entry points //
//////////////////

/// Impute a gene selection, streaming the blocks off a gene-based store.
///
/// Peak memory is `n_cells * n_genes` for the output plus
/// `2 * n_cells * gene_batch_size` for the ping-pong scratch.
///
/// ### Params
///
/// * `reader` - Reader over a **gene-based** store.
/// * `operator` - The diffusion operator, defining the cell selection.
/// * `gene_indices` - Genes to impute, in the order they should come back.
/// * `params` - Parameters, `None` for [MagicParams::default].
/// * `verbose` - Verbosity level, see [parse_verbosity_level].
///
/// ### Returns
///
/// The dense imputed block, or the matching [BixverseErrors] variant if the
/// reader is cell-based or the selection is too large.
pub fn magic_impute_genes<S: SingleCellReading>(
    reader: &S,
    operator: &MagicOperator,
    gene_indices: &[usize],
    params: Option<MagicParams>,
    verbose: usize,
) -> Result<MagicImputed, BixverseErrors> {
    let params = params.unwrap_or_default();
    let verbosity = parse_verbosity_level(verbose);
    let start_time = Instant::now();

    if !reader.is_gene_based() {
        return Err(BixverseErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }

    // The operator's lookup is indexed by global cell id, so a store with more
    // cells than the operator was built for would index past its end. Catching
    // it here turns a panic deep inside the scatter into a named error.
    let store_cells = reader.get_header().total_cells;
    if operator.lookup.len() != store_cells {
        return Err(BixverseErrors::MagicStoreSizeMismatch {
            operator_cells: operator.lookup.len(),
            store_cells,
        });
    }

    let n_cells = operator.n_cells();
    let n_genes = gene_indices.len();
    check_output_size(n_cells, n_genes, &params)?;

    if n_genes == 0 {
        return Ok(MagicImputed {
            data: Vec::new(),
            n_cells,
            gene_indices: Vec::new(),
            cell_indices: operator.cell_indices.to_vec(),
        });
    }

    // Clamped to the block ceiling as well as to the selection: the scratch is
    // `2 * n_cells * batch` floats, so an unclamped `gene_batch_size` could put
    // peak memory at several times the output budget the size guard checks.
    let batch = params
        .gene_batch_size
        .clamp(1, n_genes.min(MAGIC_GENE_BATCH_SIZE));
    let mut out = vec![0.0_f32; n_cells * n_genes];

    // Ping-pong scratch, allocated once for the whole run.
    let mut buf_a = vec![0.0_f32; n_cells * batch];
    let mut buf_b = vec![0.0_f32; n_cells * batch];

    for block_start in (0..n_genes).step_by(batch) {
        let block_end = (block_start + batch).min(n_genes);
        let width = block_end - block_start;

        let chunks = reader.read_gene_parallel(&gene_indices[block_start..block_end])?;

        // Scatter the block in, cell-major. Cells outside the selection drop
        // out here, which is why the lookup exists.
        let block = &mut buf_a[..n_cells * width];
        block.fill(0.0);
        for (col, chunk) in chunks.iter().enumerate() {
            for (k, &cell) in chunk.indices.iter().enumerate() {
                let local = operator.lookup[cell as usize];
                if local == u32::MAX {
                    continue;
                }
                block[local as usize * width + col] = match params.layer {
                    MagicLayer::Norm => chunk.data_norm[k].to_f32(),
                    MagicLayer::Raw => chunk.data_raw.get(k) as f32,
                };
            }
        }

        for _ in 0..params.n_steps {
            let (src, dst) = (&buf_a[..n_cells * width], &mut buf_b[..n_cells * width]);
            csr_matmul_dense_block(&operator.t, src, width, dst)?;
            std::mem::swap(&mut buf_a, &mut buf_b);
        }

        // Clip once at the end, matching the reference. Then place the block.
        // Zero steps means nothing diffused, so there is no diffusion noise to
        // floor and the block passes through untouched.
        let threshold = if params.n_steps == 0 {
            f32::NEG_INFINITY
        } else {
            params.clip_threshold
        };
        out.par_chunks_mut(n_genes)
            .zip(buf_a[..n_cells * width].par_chunks(width))
            .for_each(|(out_row, src_row)| {
                for (dst, &v) in out_row[block_start..block_end].iter_mut().zip(src_row) {
                    *dst = if v < threshold { 0.0 } else { v };
                }
            });

        if verbosity.detailed_verbosity() {
            println!(
                "MAGIC: imputed genes {}..{} of {} ({:.2?})",
                block_start,
                block_end,
                n_genes,
                start_time.elapsed()
            );
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "MAGIC: imputed {} genes across {} cells in {:.2?}",
            n_genes,
            n_cells,
            start_time.elapsed()
        );
    }

    Ok(MagicImputed {
        data: out,
        n_cells,
        gene_indices: gene_indices.to_vec(),
        cell_indices: operator.cell_indices.to_vec(),
    })
}

/// Impute a dense block that is already in memory.
///
/// The same maths as [magic_impute_genes] without the disk streaming, for
/// callers holding a meta-cell matrix or driving the tests.
///
/// ### Params
///
/// * `operator` - The diffusion operator.
/// * `block` - Dense input, row-major, `operator.n_cells() * width` elements,
///   rows in operator order.
/// * `width` - Columns in the block.
/// * `params` - Parameters, `None` for [MagicParams::default].
///
/// ### Returns
///
/// The imputed block in the same layout, or the matching [BixverseErrors]
/// variant on a shape mismatch or an oversized request.
pub fn magic_impute_dense(
    operator: &MagicOperator,
    block: &[f32],
    width: usize,
    params: Option<MagicParams>,
) -> Result<Vec<f32>, BixverseErrors> {
    let params = params.unwrap_or_default();
    let n_cells = operator.n_cells();

    check_output_size(n_cells, width, &params)?;

    if block.len() != n_cells * width {
        return Err(BixverseErrors::SparseMatrixMultiplication {
            n_col_a: n_cells,
            n_row_b: block.len().checked_div(width).unwrap_or(0),
        });
    }

    let mut buf_a = block.to_vec();
    let mut buf_b = vec![0.0_f32; n_cells * width];

    for _ in 0..params.n_steps {
        csr_matmul_dense_block(&operator.t, &buf_a, width, &mut buf_b)?;
        std::mem::swap(&mut buf_a, &mut buf_b);
    }

    // Nothing diffused at zero steps, so there is no diffusion noise to floor.
    if params.n_steps == 0 {
        return Ok(buf_a);
    }

    let threshold = params.clip_threshold;
    buf_a.par_iter_mut().for_each(|v| {
        if *v < threshold {
            *v = 0.0;
        }
    });

    Ok(buf_a)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A path graph over `n` cells, each cell holding its two nearest
    /// neighbours. Distances grow with the hop, so the adaptive bandwidth has
    /// something to bite on.
    fn path_knn(n: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let mut indices = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let mut idx = Vec::new();
            let mut dst = Vec::new();
            for hop in 1..=2usize {
                if i >= hop {
                    idx.push(i - hop);
                    dst.push(hop as f32);
                }
                if i + hop < n {
                    idx.push(i + hop);
                    dst.push(hop as f32);
                }
            }
            indices.push(idx);
            distances.push(dst);
        }
        (indices, distances)
    }

    fn path_operator(n: usize) -> MagicOperator {
        let (idx, dist) = path_knn(n);
        let cells: Vec<usize> = (0..n).collect();
        MagicOperator::from_knn(&idx, &dist, false, &cells, n).expect("operator builds")
    }

    /// The operator must be row-stochastic by construction.
    #[test]
    fn test_operator_rows_are_stochastic() {
        let op = path_operator(20);
        let t = op.transition_matrix();

        assert_eq!(t.shape(), (20, 20));
        for i in 0..20 {
            let lo = t.indptr[i] as usize;
            let hi = t.indptr[i + 1] as usize;
            let sum: f32 = t.data[lo..hi].iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    /// Zero steps is the identity, which is what makes the same entry point
    /// serve callers who do not want imputation.
    #[test]
    fn test_zero_steps_returns_input_unchanged() {
        let op = path_operator(12);
        let block: Vec<f32> = (0..12 * 3).map(|i| i as f32 * 0.5 + 1.0).collect();

        let params = MagicParams {
            n_steps: 0,
            clip_threshold: 0.0,
            ..Default::default()
        };
        let out = magic_impute_dense(&op, &block, 3, Some(params)).unwrap();

        assert_eq!(out, block);
    }

    /// Row-stochastic means total mass per cell is preserved, so a constant
    /// column has to survive any number of steps untouched. This is the
    /// invariant that keeps imputed values on the input's scale.
    #[test]
    fn test_constant_column_is_preserved() {
        let op = path_operator(30);
        let block = vec![4.25_f32; 30 * 4];

        for n_steps in [1usize, 3, 7] {
            let params = MagicParams {
                n_steps,
                clip_threshold: 0.0,
                ..Default::default()
            };
            let out = magic_impute_dense(&op, &block, 4, Some(params)).unwrap();
            for v in out {
                assert_relative_eq!(v, 4.25, epsilon = 1e-4);
            }
        }
    }

    /// Diffusion has to actually smooth: a spike on one cell must spread to its
    /// neighbours and shrink at the source.
    #[test]
    fn test_diffusion_smooths_a_spike() {
        let op = path_operator(21);
        let mut block = vec![0.0_f32; 21];
        block[10] = 100.0;

        let params = MagicParams {
            n_steps: 3,
            clip_threshold: 0.0,
            ..Default::default()
        };
        let out = magic_impute_dense(&op, &block, 1, Some(params)).unwrap();

        assert!(out[10] < 100.0, "the spike did not shrink");
        assert!(out[9] > 0.0 && out[11] > 0.0, "the spike did not spread");
        // Mass is conserved across the whole path.
        let before: f32 = block.iter().sum();
        let after: f32 = out.iter().sum();
        assert_relative_eq!(after, before, max_relative = 1e-4);
    }

    /// `n_steps` must be exactly the number of operator applications.
    ///
    /// The invariance and smoothing tests are all blind to the step count: a
    /// constant column survives any power, "shrank and spread" is true at two
    /// steps as at three, and comparing the streaming path against the dense one
    /// compares two things that would be wrong together. Composition pins the
    /// count itself, so `for _ in 1..n_steps` fails here and nowhere else.
    #[test]
    fn test_step_count_composes() {
        let op = path_operator(24);
        let block: Vec<f32> = (0..24 * 3).map(|i| ((i * 37) % 11) as f32).collect();
        let no_clip = |n_steps| MagicParams {
            n_steps,
            clip_threshold: 0.0,
            ..Default::default()
        };

        // T^(a+b) X == T^a (T^b X), for a couple of splits.
        for (a, b) in [(1usize, 1usize), (2, 1), (1, 3), (2, 3)] {
            let direct = magic_impute_dense(&op, &block, 3, Some(no_clip(a + b))).unwrap();
            let staged = magic_impute_dense(&op, &block, 3, Some(no_clip(b))).unwrap();
            let staged = magic_impute_dense(&op, &staged, 3, Some(no_clip(a))).unwrap();

            for (d, s) in direct.iter().zip(staged.iter()) {
                assert_relative_eq!(d, s, epsilon = 1e-4, max_relative = 1e-4);
            }
        }

        // One step must be exactly one application of the operator, checked
        // against the kernel directly rather than against another MAGIC call.
        let mut want = vec![0.0_f32; 24 * 3];
        csr_matmul_dense_block(op.transition_matrix(), &block, 3, &mut want).unwrap();
        let got = magic_impute_dense(&op, &block, 3, Some(no_clip(1))).unwrap();
        assert_eq!(got, want);
    }

    /// The clip is a floor applied once at the end, not per pass.
    #[test]
    fn test_clip_threshold_zeroes_small_values() {
        let op = path_operator(15);
        let block = vec![1.0_f32; 15];

        let params = MagicParams {
            n_steps: 1,
            clip_threshold: 2.0,
            ..Default::default()
        };
        let out = magic_impute_dense(&op, &block, 1, Some(params)).unwrap();

        // Every diffused value is 1.0, which is below the threshold.
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// The size guard has to fire at the boundary and be suppressible.
    #[test]
    fn test_output_size_guard() {
        let params = MagicParams::default();

        assert!(check_output_size(1000, 1000, &params).is_ok());
        assert!(check_output_size(MAGIC_MAX_ELEMENTS, 1, &params).is_ok());

        match check_output_size(MAGIC_MAX_ELEMENTS, 2, &params) {
            Err(BixverseErrors::MagicOutputTooLarge { n_genes, .. }) => assert_eq!(n_genes, 2),
            other => panic!("expected MagicOutputTooLarge, got {:?}", other),
        }

        let allowed = MagicParams {
            allow_large: true,
            ..params
        };
        assert!(check_output_size(MAGIC_MAX_ELEMENTS, 100, &allowed).is_ok());
    }

    #[test]
    fn test_operator_rejects_bad_selections() {
        let (idx, dist) = path_knn(5);

        // kNN rows against a shorter selection.
        assert!(matches!(
            MagicOperator::from_knn(&idx, &dist, false, &[0, 1, 2], 5),
            Err(BixverseErrors::MagicKnnSelectionMismatch {
                n_knn: 5,
                n_selected: 3
            })
        ));

        // A cell beyond the store.
        assert!(matches!(
            MagicOperator::from_knn(&idx, &dist, false, &[0, 1, 2, 3, 99], 5),
            Err(BixverseErrors::MagicCellOutOfRange { cell: 99, .. })
        ));

        // A duplicate.
        assert!(matches!(
            MagicOperator::from_knn(&idx, &dist, false, &[0, 1, 2, 3, 3], 5),
            Err(BixverseErrors::MagicDuplicateCell { cell: 3 })
        ));
    }

    /// The selection need not be contiguous or sorted; the lookup is what makes
    /// gene chunks scatter into the right local rows.
    #[test]
    fn test_operator_accepts_a_scattered_selection() {
        let (idx, dist) = path_knn(4);
        let op = MagicOperator::from_knn(&idx, &dist, false, &[7, 2, 9, 0], 10).unwrap();

        assert_eq!(op.n_cells(), 4);
        assert_eq!(op.cell_indices(), &[7, 2, 9, 0]);
        assert_eq!(op.lookup[7], 0);
        assert_eq!(op.lookup[2], 1);
        assert_eq!(op.lookup[9], 2);
        assert_eq!(op.lookup[0], 3);
        assert_eq!(op.lookup[1], u32::MAX);
    }

    /// The row-major result and the faer view must agree.
    #[test]
    fn test_imputed_to_mat_round_trips() {
        let imputed = MagicImputed {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            n_cells: 2,
            gene_indices: vec![10, 20, 30],
            cell_indices: vec![0, 1],
        };

        let mat = imputed.to_mat();
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3);
        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(mat[(i, j)], imputed.data[i * 3 + j]);
            }
        }
    }

    #[test]
    fn test_dense_rejects_shape_mismatch() {
        let op = path_operator(6);
        let block = vec![1.0_f32; 6 * 2];

        assert!(magic_impute_dense(&op, &block, 5, None).is_err());
    }

    ////////////////////
    // Streaming path //
    ////////////////////

    use crate::single_cell::sc_data::data_io::{
        CellGeneSparseWriter, CscGeneChunk, ParallelSparseReader, RawCounts,
    };
    use crate::single_cell::sc_traits::F16;

    /// Scratch store that removes itself on drop.
    struct TempStore(std::path::PathBuf);

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempStore {
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_magic_{name}.bin")))
        }

        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// Write `dense[gene][cell]` raw counts out as a gene-based store.
    fn write_store(path: &str, dense: &[Vec<u32>], n_cells: usize) {
        let mut writer = CellGeneSparseWriter::new(path, false, n_cells, dense.len(), 1e4)
            .expect("writer opens");

        for (gene_idx, gene) in dense.iter().enumerate() {
            let mut raw = Vec::new();
            let mut indices = Vec::new();
            for (cell, &value) in gene.iter().enumerate() {
                if value > 0 {
                    raw.push(value);
                    indices.push(cell);
                }
            }
            let norms: Vec<F16> = raw
                .iter()
                .map(|&v| F16::from_f32((v as f32).ln_1p()))
                .collect();

            writer
                .write_gene_chunk(CscGeneChunk::from_conversion(
                    RawCounts::from_u32_auto(&raw),
                    &norms,
                    &indices,
                    gene_idx,
                    true,
                ))
                .expect("write gene chunk");
        }

        writer.finalise().expect("finalise");
    }

    /// Deterministic counts, no RNG dependency.
    fn synthetic_counts(n_genes: usize, n_cells: usize) -> Vec<Vec<u32>> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut dense = vec![vec![0u32; n_cells]; n_genes];
        for gene in dense.iter_mut() {
            for value in gene.iter_mut() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let draw = (state >> 33) as u32;
                *value = if draw % 5 < 2 { draw % 40 + 1 } else { 0 };
            }
        }
        dense
    }

    /// Streaming off the store must agree with running the same maths on a
    /// dense block assembled by hand, and gene blocking must not change the
    /// answer.
    #[test]
    fn test_streaming_matches_dense_and_is_block_invariant() {
        let (n_genes, n_cells) = (17usize, 25usize);
        let store = TempStore::new("stream");
        let dense = synthetic_counts(n_genes, n_cells);
        write_store(store.path(), &dense, n_cells);
        let reader = ParallelSparseReader::new(store.path()).expect("reader opens");

        let op = path_operator(n_cells);
        let genes: Vec<usize> = (0..n_genes).collect();
        let params = MagicParams {
            n_steps: 3,
            clip_threshold: 0.0,
            gene_batch_size: 4,
            ..Default::default()
        };

        let streamed = magic_impute_genes(&reader, &op, &genes, Some(params), 0).unwrap();

        // The same input, assembled directly from the dense counts.
        let mut block = vec![0.0_f32; n_cells * n_genes];
        for (g, gene) in dense.iter().enumerate() {
            for (c, &v) in gene.iter().enumerate() {
                if v > 0 {
                    block[c * n_genes + g] = F16::from_f32((v as f32).ln_1p()).to_f32();
                }
            }
        }
        let want = magic_impute_dense(&op, &block, n_genes, Some(params)).unwrap();

        for (got, expected) in streamed.data.iter().zip(want.iter()) {
            assert_relative_eq!(got, expected, epsilon = 1e-6, max_relative = 1e-5);
        }

        // A single block must give the same thing as four-gene blocks, up to
        // rounding: `axpy_simd_f32` fuses the multiply-add on full SIMD lanes
        // but not on the scalar tail, so the block width decides which lanes
        // get an FMA. Bit-exactness across widths is not on offer.
        let single = magic_impute_genes(
            &reader,
            &op,
            &genes,
            Some(MagicParams {
                gene_batch_size: n_genes,
                ..params
            }),
            0,
        )
        .unwrap();
        assert_eq!(single.data.len(), streamed.data.len());
        for (one, four) in single.data.iter().zip(streamed.data.iter()) {
            assert_relative_eq!(one, four, epsilon = 1e-6, max_relative = 1e-5);
        }
    }

    /// A cell subset must impute only over that subset, and the gene order the
    /// caller asked for must be the order that comes back.
    #[test]
    fn test_streaming_honours_cell_subset_and_gene_order() {
        let (n_genes, n_cells) = (8usize, 20usize);
        let store = TempStore::new("subset");
        let dense = synthetic_counts(n_genes, n_cells);
        write_store(store.path(), &dense, n_cells);
        let reader = ParallelSparseReader::new(store.path()).expect("reader opens");

        // Ten of the twenty cells, deliberately not contiguous.
        let selection: Vec<usize> = (0..n_cells).step_by(2).collect();
        let (idx, dist) = path_knn(selection.len());
        let op = MagicOperator::from_knn(&idx, &dist, false, &selection, n_cells).unwrap();

        let genes = vec![5usize, 0, 3];
        let out = magic_impute_genes(
            &reader,
            &op,
            &genes,
            Some(MagicParams {
                n_steps: 0,
                clip_threshold: 0.0,
                ..Default::default()
            }),
            0,
        )
        .unwrap();

        assert_eq!(out.n_cells, selection.len());
        assert_eq!(out.gene_indices, genes);
        assert_eq!(out.cell_indices, selection);
        assert_eq!(out.data.len(), selection.len() * genes.len());

        // With zero steps this is a pure gather, so every value is checkable.
        for (row, &cell) in selection.iter().enumerate() {
            for (col, &gene) in genes.iter().enumerate() {
                let raw = dense[gene][cell];
                let want = if raw > 0 {
                    F16::from_f32((raw as f32).ln_1p()).to_f32()
                } else {
                    0.0
                };
                assert_relative_eq!(out.data[row * genes.len() + col], want, epsilon = 1e-6);
            }
        }
    }

    /// The raw layer must be readable, and it must differ from the normalised
    /// one, so the parameter is not silently ignored.
    #[test]
    fn test_streaming_raw_layer() {
        let (n_genes, n_cells) = (6usize, 12usize);
        let store = TempStore::new("rawlayer");
        let dense = synthetic_counts(n_genes, n_cells);
        write_store(store.path(), &dense, n_cells);
        let reader = ParallelSparseReader::new(store.path()).expect("reader opens");

        let op = path_operator(n_cells);
        let genes: Vec<usize> = (0..n_genes).collect();
        let base = MagicParams {
            n_steps: 0,
            clip_threshold: 0.0,
            ..Default::default()
        };

        let raw = magic_impute_genes(
            &reader,
            &op,
            &genes,
            Some(MagicParams {
                layer: MagicLayer::Raw,
                ..base
            }),
            0,
        )
        .unwrap();
        let norm = magic_impute_genes(&reader, &op, &genes, Some(base), 0).unwrap();

        for (row, cell) in (0..n_cells).enumerate() {
            for (col, gene) in (0..n_genes).enumerate() {
                assert_relative_eq!(
                    raw.data[row * n_genes + col],
                    dense[gene][cell] as f32,
                    epsilon = 1e-6
                );
            }
        }
        assert!(raw.data != norm.data, "the layer parameter was ignored");
    }

    /// The streaming clip must fire across block boundaries, and it must be the
    /// only difference between a clipped and an unclipped run.
    ///
    /// Every other streaming test pins `clip_threshold` at zero, so this is the
    /// only coverage of the live threshold and of the multi-block placement loop
    /// running with one.
    #[test]
    fn test_streaming_clip_across_blocks() {
        let (n_genes, n_cells) = (9usize, 18usize);
        let store = TempStore::new("clip");
        let dense = synthetic_counts(n_genes, n_cells);
        write_store(store.path(), &dense, n_cells);
        let reader = ParallelSparseReader::new(store.path()).expect("reader opens");

        let op = path_operator(n_cells);
        let genes: Vec<usize> = (0..n_genes).collect();
        let base = MagicParams {
            n_steps: 3,
            clip_threshold: 0.0,
            gene_batch_size: 2, // several blocks, so the placement loop repeats
            ..Default::default()
        };

        let unclipped = magic_impute_genes(&reader, &op, &genes, Some(base), 0).unwrap();
        let threshold = 0.5_f32;
        let clipped = magic_impute_genes(
            &reader,
            &op,
            &genes,
            Some(MagicParams {
                clip_threshold: threshold,
                ..base
            }),
            0,
        )
        .unwrap();

        // The clip must have bitten somewhere, or the fixture proves nothing.
        assert!(
            unclipped.data.iter().any(|&v| v > 0.0 && v < threshold),
            "fixture has no values in the clipping range"
        );

        for (raw, clip) in unclipped.data.iter().zip(clipped.data.iter()) {
            let want = if *raw < threshold { 0.0 } else { *raw };
            assert_relative_eq!(clip, &want, epsilon = 1e-6);
        }
    }

    /// An operator built against a different store must be named, not panic.
    #[test]
    fn test_streaming_rejects_store_size_mismatch() {
        let (n_genes, n_cells) = (4usize, 16usize);
        let store = TempStore::new("sizemismatch");
        let dense = synthetic_counts(n_genes, n_cells);
        write_store(store.path(), &dense, n_cells);
        let reader = ParallelSparseReader::new(store.path()).expect("reader opens");

        // The operator is told the store holds 8 cells when it holds 16. Every
        // selected index is below 8, so `from_knn` is happy; the scatter would
        // then index `lookup[12]` on a lookup of length 8.
        let selection: Vec<usize> = (0..8).collect();
        let (idx, dist) = path_knn(selection.len());
        let op = MagicOperator::from_knn(&idx, &dist, false, &selection, 8).unwrap();

        match magic_impute_genes(&reader, &op, &[0, 1], None, 0) {
            Err(BixverseErrors::MagicStoreSizeMismatch {
                operator_cells,
                store_cells,
            }) => {
                assert_eq!(operator_cells, 8);
                assert_eq!(store_cells, 16);
            }
            other => panic!("expected MagicStoreSizeMismatch, got {:?}", other.err()),
        }
    }

    /// A cell-based store must be rejected rather than silently misread.
    #[test]
    fn test_streaming_rejects_cell_based_store() {
        use crate::single_cell::sc_data::data_io::CsrCellChunk;

        let store = TempStore::new("cellbased");
        let mut writer =
            CellGeneSparseWriter::new(store.path(), true, 3, 2, 1e4).expect("writer opens");
        for cell in 0..3usize {
            writer
                .write_cell_chunk(CsrCellChunk::from_data(
                    &[1u32, 2],
                    &[0u32, 1],
                    cell,
                    1e4,
                    true,
                ))
                .expect("write cell chunk");
        }
        writer.finalise().expect("finalise");

        let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
        let op = path_operator(3);

        assert!(matches!(
            magic_impute_genes(&reader, &op, &[0, 1], None, 0),
            Err(BixverseErrors::ReaderModeMismatch { .. })
        ));
    }
}
