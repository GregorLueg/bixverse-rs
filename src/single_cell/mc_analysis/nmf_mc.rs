//! Meta cells version of NMF. Leverages the sparse implementation of the
//! HALS version. Single runs, multiple restarts, consensus NMF at a fixed
//! k and the k sweep that picks that k.
//!
//! Every entry point transposes CSR input to CSC first. The sparse
//! pre-processing indexes `indptr` per column, so a CSR matrix that slips
//! through scales the wrong axis and returns quiet nonsense rather than an
//! error.

use std::time::Instant;

use crate::methods::nmf_hals::consensus::*;
use crate::methods::nmf_hals::*;
use crate::prelude::*;
use crate::single_cell::sc_analysis::nmf_sc::{
    nmf_consensus_run_sparse, nmf_k_sweep_run_sparse, nmf_multiple_run_sparse,
    nmf_single_run_sparse,
};

/////////////
// Helpers //
/////////////

/// Transpose meta cell counts to CSC if they arrived as CSR
///
/// The sparse pre-processing indexes `indptr` per column, so CSR input scales
/// the wrong axis and returns quiet nonsense rather than an error. The R side
/// always hands over CSR, so this is the common path, not the exception.
///
/// ### Params
///
/// * `data` - The meta cell counts, in either orientation.
/// * `verbosity` - Resolved verbosity, used only for the transpose notice.
///
/// ### Returns
///
/// The same data in CSC orientation.
fn ensure_csc(
    data: CompressedSparseData2<f32>,
    verbosity: Verbosity,
) -> CompressedSparseData2<f32> {
    if data.cs_type.is_csr() {
        if verbosity.detailed_verbosity() {
            println!("NMF: meta cell data was provided as CSR. Transposing to CSC.")
        }
        data.transform()
    } else {
        data
    }
}

////////////////
// Single run //
////////////////

/// Run NMF on meta cells
///
/// This function runs NMF on the chosen the provided meta cells x genes with
/// the given `k`. To note, during the fitting of the HALS-based NMF two copies
/// of the data layer are held in memory for speedy matrix operations. This
/// doubles the memory of that layer.
///
/// ### Params
///
/// * `data` - The [CompressedSparseData2] on which to run the NMF, meta cells x
///   genes. Transposed to CSC first if supplied as CSR.
/// * `k` - Number of latent variables to return
/// * `preprocessing` - String to forward to
///   [crate::methods::nmf_hals::nmf_preprocessing::parse_nmf_processing].
/// * `use_second_layer` - Shall the second data layer (normalised counts) be
///   used.
/// * `nmf_hals_params` - Optional parameters for the [HalsOpts].
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// [NmfResult] or respective errors if something went wrong.
#[allow(clippy::too_many_arguments)]
pub fn nmf_single_run_mc(
    data: CompressedSparseData2<f32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    verbose: usize,
) -> Result<NmfResult<f32>, BixverseErrors> {
    let start_total = Instant::now();
    let verbosity = parse_verbosity_level(verbose);

    let data = ensure_csc(data, verbosity);

    if verbosity.normal_verbosity() {
        println!("NMF: Running NMF for metacells cells ...")
    }
    let res = nmf_single_run_sparse(
        data,
        k,
        preprocessing,
        use_second_layer,
        nmf_hals_params,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start_total.elapsed())
    }

    Ok(res)
}

///////////////////
// Multiple runs //
///////////////////

/// Run multiple rounds of NMF on a subset of cells and genes loaded from disk
///
/// This function runs multiple rounds of NMF with different random
/// initilisations on the provided meta cells x genes with the given `k`.
/// To note, during the fitting of the HALS-based NMF two copies
/// of the data layer are held in memory for speedy matrix operations. This
/// doubles the memory of that layer.
///
/// ### Params
///
/// * `data` - The [CompressedSparseData2] on which to run the NMF, meta cells x
///   genes. Transposed to CSC first if supplied as CSR.
/// * `k` - Number of latent variables to return
/// * `preprocessing` - String to forward to
///   [crate::methods::nmf_hals::nmf_preprocessing::parse_nmf_processing].
/// * `use_second_layer` - Shall the second data layer (normalised counts) be
///   used.
/// * `nmf_hals_params` - Optional parameters for the [HalsOpts].
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// [NmfResult] or respective errors if something went wrong.
#[allow(clippy::too_many_arguments)]
pub fn nmf_multiple_run_mc(
    data: CompressedSparseData2<f32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<StabilisedNmfResult<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start_total = Instant::now();

    let data = ensure_csc(data, verbosity);

    if verbosity.normal_verbosity() {
        println!("NMF: Running multiple NMF runs for meta cells ...")
    }
    let res = nmf_multiple_run_sparse(
        data,
        k,
        preprocessing,
        use_second_layer,
        nmf_hals_params,
        n_runs,
        base_seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start_total.elapsed())
    }

    Ok(res)
}

///////////////
// Consensus //
///////////////

/// Run consensus NMF at a single k on meta cells
///
/// Runs `n_runs` random restarts, clusters the pooled components and refits
/// the partner factor against the consensus one. See [nmf_consensus] for the
/// scale convention of the returned factors, which depends on the
/// [ConsensusTarget]. Two copies of the data layer are held in memory during
/// the fit, on top of the restart factors, so `n_runs` is the knob that
/// dominates the memory bill here.
///
/// ### Params
///
/// * `data` - The [CompressedSparseData2] to factorise, meta cells x genes.
///   Transposed to CSC first if supplied as CSR.
/// * `k` - Number of components. Must be at least 2.
/// * `preprocessing` - String to forward to
///   [crate::methods::nmf_hals::nmf_preprocessing::parse_nmf_processing].
/// * `use_second_layer` - Shall the second data layer (normalised counts) be
///   used.
/// * `nmf_hals_params` - Optional parameters for the [HalsOpts].
/// * `consensus_params` - Optional parameters for the [ConsensusParams].
/// * `n_runs` - Number of restarts. Must be at least 2.
/// * `base_seed` - The base seed. Restart `i` uses `base_seed + i`.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// [ConsensusNmfResult] or respective errors if something went wrong.
///
/// ### References
///
/// Kotliar et al., eLife, 2019
#[allow(clippy::too_many_arguments)]
pub fn nmf_consensus_run_mc(
    data: CompressedSparseData2<f32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    consensus_params: Option<ConsensusParams<f32>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<ConsensusNmfResult<f32>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start_total = Instant::now();

    let data = ensure_csc(data, verbosity);

    if verbosity.normal_verbosity() {
        println!("NMF: Running consensus NMF for meta cells ...")
    }
    let res = nmf_consensus_run_sparse(
        data,
        k,
        preprocessing,
        use_second_layer,
        nmf_hals_params,
        consensus_params,
        n_runs,
        base_seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start_total.elapsed())
    }

    Ok(res)
}

/////////////
// K sweep //
/////////////

/// Sweep k and report consensus stability against reconstruction error
///
/// Returns diagnostics only, no factors, so a wide `k_range` stays cheap in
/// memory. Pick the k where stability is high and the error curve has not yet
/// flattened, then call [nmf_consensus_run_mc] there. A k whose density filter
/// leaves fewer than `k` components is reported through `consensus_failed`
/// rather than aborting the whole sweep.
///
/// ### Params
///
/// * `data` - The [CompressedSparseData2] to factorise, meta cells x genes.
///   Transposed to CSC first if supplied as CSR.
/// * `k_range` - Ranks to evaluate. Must be non-empty, every entry at least 2.
/// * `preprocessing` - String to forward to
///   [crate::methods::nmf_hals::nmf_preprocessing::parse_nmf_processing].
/// * `use_second_layer` - Shall the second data layer (normalised counts) be
///   used.
/// * `nmf_hals_params` - Optional parameters for the [HalsOpts].
/// * `consensus_params` - Optional parameters for the [ConsensusParams].
/// * `n_runs` - Number of restarts per k. Must be at least 2.
/// * `base_seed` - The base seed. The i-th k uses `base_seed + i * n_runs`.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// One [KSweepEntry] per entry of `k_range`, or respective errors if something
/// went wrong.
///
/// ### References
///
/// Kotliar et al., eLife, 2019
#[allow(clippy::too_many_arguments)]
pub fn nmf_k_sweep_run_mc(
    data: CompressedSparseData2<f32>,
    k_range: &[usize],
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    consensus_params: Option<ConsensusParams<f32>>,
    n_runs: usize,
    base_seed: usize,
    verbose: usize,
) -> Result<Vec<KSweepEntry<f32>>, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start_total = Instant::now();

    let data = ensure_csc(data, verbosity);

    if verbosity.normal_verbosity() {
        println!(
            "NMF: Sweeping {} values of k for meta cells ...",
            k_range.len()
        )
    }
    let res = nmf_k_sweep_run_sparse(
        data,
        k_range,
        preprocessing,
        use_second_layer,
        nmf_hals_params,
        consensus_params,
        n_runs,
        base_seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start_total.elapsed())
    }

    Ok(res)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Block-structured meta cells x genes counts as CSR.
    ///
    /// Two clean gene programmes, so a k of 2 has an unambiguous answer and the
    /// restarts land in the same place regardless of initialisation.
    fn fixture_csr(n_cells: usize, n_genes: usize) -> CompressedSparseData2<f32> {
        let half_genes = n_genes / 2;
        let mut data: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = vec![0];

        for cell in 0..n_cells {
            let block = if cell < n_cells / 2 {
                0..half_genes
            } else {
                half_genes..n_genes
            };
            for gene in block {
                // A little per-cell and per-gene variation so the columns do
                // not all share one standard deviation.
                data.push(4.0 + (cell % 3) as f32 + (gene % 2) as f32 * 0.5);
                indices.push(gene as u32);
            }
            indptr.push(data.len() as u32);
        }

        CompressedSparseData2::<f32>::new_csr(&data, &indices, &indptr, None, (n_cells, n_genes))
    }

    /// Filter off, so the tests exercise the clustering rather than the density
    /// heuristic, which needs more restarts than a unit test should pay for.
    fn test_consensus_params() -> ConsensusParams<f32> {
        ConsensusParams::new(ConsensusTarget::HRows, None, None, 100, 3, 42)
    }

    fn test_hals_opts() -> HalsOpts<f32> {
        HalsOpts::<f32>::new(200, 1e-6, 1e-10, 10, NmfInit::Nndsvd)
    }

    /// CSR input must be transposed to CSC before pre-processing.
    ///
    /// This is the test that actually guards [ensure_csc]. With `sd` scaling the
    /// pre-processing walks `indptr` per column, so a CSR matrix that slips
    /// through scales the wrong axis and returns quiet nonsense. Drop the guard
    /// and the two paths diverge.
    #[test]
    fn csr_and_csc_inputs_agree_under_scaling() {
        let (k, n_runs) = (2, 3);
        let csr = fixture_csr(20, 10);
        let csc = csr.transform();
        assert!(csr.cs_type.is_csr());
        assert!(!csc.cs_type.is_csr());

        let from_csr = nmf_consensus_run_mc(
            csr,
            k,
            "sd",
            false,
            Some(test_hals_opts()),
            Some(test_consensus_params()),
            n_runs,
            42,
            0,
        )
        .unwrap();
        let from_csc = nmf_consensus_run_mc(
            csc,
            k,
            "sd",
            false,
            Some(test_hals_opts()),
            Some(test_consensus_params()),
            n_runs,
            42,
            0,
        )
        .unwrap();

        assert_eq!(from_csr.w.shape(), from_csc.w.shape());
        assert_eq!(from_csr.h.shape(), from_csc.h.shape());
        for i in 0..from_csr.h.nrows() {
            for j in 0..from_csr.h.ncols() {
                assert_relative_eq!(from_csr.h[(i, j)], from_csc.h[(i, j)], epsilon = 1e-5);
            }
        }
        assert_relative_eq!(from_csr.error, from_csc.error, epsilon = 1e-5);
    }

    /// The factors come back meta cells x k and k x genes, non-negative, with
    /// one pooled label per restart component.
    #[test]
    fn consensus_shapes_and_diagnostics() {
        let (n_cells, n_genes, k, n_runs) = (20, 10, 2, 3);
        let res = nmf_consensus_run_mc(
            fixture_csr(n_cells, n_genes),
            k,
            "none",
            false,
            Some(test_hals_opts()),
            Some(test_consensus_params()),
            n_runs,
            42,
            0,
        )
        .unwrap();

        assert_eq!(res.w.shape(), (n_cells, k));
        assert_eq!(res.h.shape(), (k, n_genes));
        assert_eq!(res.clusters.labels.len(), k * n_runs);
        assert_eq!(res.clusters.local_density.len(), k * n_runs);
        assert_eq!(res.clusters.sizes.len(), k);
        assert_eq!(res.run_errors.len(), n_runs);
        assert!(res.error.is_finite() && res.error >= 0.0);
        assert!(res.clusters.stability.is_finite());

        for i in 0..res.w.nrows() {
            for j in 0..res.w.ncols() {
                assert!(res.w[(i, j)] >= 0.0);
            }
        }
    }

    /// A k below 2 is refused before any restart runs.
    #[test]
    fn consensus_rejects_k_below_two() {
        let err = nmf_consensus_run_mc(
            fixture_csr(20, 10),
            1,
            "none",
            false,
            Some(test_hals_opts()),
            Some(test_consensus_params()),
            3,
            42,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, BixverseErrors::NmfConsensusInvalidK { k: 1 }));
    }

    /// One entry per requested k, in the order asked for.
    #[test]
    fn k_sweep_returns_one_entry_per_k() {
        let k_range = [2_usize, 3, 4];
        let res = nmf_k_sweep_run_mc(
            fixture_csr(24, 12),
            &k_range,
            "none",
            false,
            Some(test_hals_opts()),
            Some(test_consensus_params()),
            3,
            42,
            0,
        )
        .unwrap();

        assert_eq!(res.len(), k_range.len());
        for (entry, &k) in res.iter().zip(k_range.iter()) {
            assert_eq!(entry.k, k);
            assert!(entry.best_error.is_finite());
            assert!(entry.median_error.is_finite());
            assert!(entry.n_converged <= 3);
        }
    }

    /// The sweep takes CSR the same way the consensus entry point does.
    #[test]
    fn k_sweep_csr_and_csc_inputs_agree_under_scaling() {
        let k_range = [2_usize, 3];
        let csr = fixture_csr(20, 10);
        let csc = csr.transform();

        let from_csr = nmf_k_sweep_run_mc(
            csr,
            &k_range,
            "sd",
            false,
            Some(test_hals_opts()),
            Some(test_consensus_params()),
            3,
            42,
            0,
        )
        .unwrap();
        let from_csc = nmf_k_sweep_run_mc(
            csc,
            &k_range,
            "sd",
            false,
            Some(test_hals_opts()),
            Some(test_consensus_params()),
            3,
            42,
            0,
        )
        .unwrap();

        for (a, b) in from_csr.iter().zip(from_csc.iter()) {
            assert_eq!(a.k, b.k);
            assert_relative_eq!(a.best_error, b.best_error, epsilon = 1e-5);
        }
    }

    /// An empty k range is an error, not an empty result the caller has to
    /// notice.
    #[test]
    fn k_sweep_rejects_empty_range() {
        let err = nmf_k_sweep_run_mc(
            fixture_csr(20, 10),
            &[],
            "none",
            false,
            Some(test_hals_opts()),
            Some(test_consensus_params()),
            3,
            42,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, BixverseErrors::NmfKSweepEmptyRange));
    }
}
