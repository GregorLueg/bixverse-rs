//! Consensus NMF and k selection on top of the GPU solver, plus the top-level
//! entry points.
//!
//! Only the solver moves to the device. The consensus machinery itself, i.e.
//! pooling, the local-density filter, the k-means clustering, the silhouette and
//! the per-coordinate median, runs on
//! [`crate::methods::nmf_hals::consensus::consensus_from_restarts`] unchanged.
//! That work is over a `(k * n_runs) x dim` matrix with `k * n_runs` in the
//! hundreds, so it is nowhere near the solve in cost and there is nothing to
//! gain from porting it.
//!
//! What the GPU path does change is that `V` is uploaded once and reused across
//! every restart and every rank in a sweep. On the CPU each of the
//! `k_range.len() * n_runs` solves pays full memory traffic over `V` again.
//!
//! ### Precision
//!
//! These entry points are f32 only, matching the single-cell and metacell CPU
//! paths and the device. Callers holding f64 data cast before entering.

use cubecl::prelude::*;
use faer::MatRef;

use crate::gpu::methods_gpu::nmf_gpu::{
    GpuDenseNmfInput, GpuNmfData, GpuSparseNmfInput, NmfGpuScratch, nmf_hals_gpu, nmf_refit_h_gpu,
    nmf_refit_w_gpu, stabilised_nmf_gpu,
};
use crate::methods::nmf_hals::consensus::{
    ConsensusNmfResult, ConsensusParams, ConsensusTarget, KSweepEntry, consensus_from_restarts,
    median,
};
use crate::methods::nmf_hals::nmf_preprocessing::{
    NmfPreprocessing, ensure_csc, parse_nmf_processing,
};
use crate::methods::nmf_hals::sparse::SparseInput;
use crate::methods::nmf_hals::{
    HalsOpts, NmfResult, StabilisedNmfResult, nmf_preprocessing::nmf_process_dense,
    nmf_preprocessing::nmf_process_sparse,
};
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Parse the pre-processing string, warning and defaulting on unknown input.
///
/// Mirrors the CPU entry points so a string that works on one path works on the
/// other.
///
/// ### Params
///
/// * `preprocessing` - String to parse, see [`parse_nmf_processing`]
///
/// ### Returns
///
/// The [`NmfPreprocessing`] to apply.
fn resolve_preprocessing(preprocessing: &str) -> NmfPreprocessing {
    parse_nmf_processing(preprocessing).unwrap_or_else(|| {
        println!(
            "Unknown string provided: {:?}. Using the default (No additional pre-processing)",
            preprocessing
        );
        NmfPreprocessing::default()
    })
}

/// The relative-error denominator, floored at one.
///
/// ### Params
///
/// * `sq_frob` - `||V||_F^2`
///
/// ### Returns
///
/// `max(sq_frob, 1)`.
fn error_denominator(sq_frob: f32) -> f32 {
    if sq_frob > 0.0 { sq_frob } else { 1.0 }
}

/////////////////////////
// Consensus and sweep //
/////////////////////////

/// Consensus NMF on the GPU.
///
/// Runs `n_runs` restarts through [`stabilised_nmf_gpu`], clusters their
/// components with the shared host-side
/// [`consensus_from_restarts`], then refits the partner factor against the
/// consensus. Identical in structure and in reported quantities to
/// [`crate::methods::nmf_hals::consensus::nmf_consensus`].
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `k` - Number of components, at least two
/// * `n_runs` - Number of restarts, at least two
/// * `base_seed` - Seed offset; run `i` uses `base_seed + i`
/// * `opts` - Solver options; the `init` field is ignored by the restarts
/// * `params` - Optional [`ConsensusParams`], defaulted if `None`
/// * `scratch` - Reusable device buffers, sized for at least `k`
/// * `client` - CubeCL compute client
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A [`ConsensusNmfResult`]. `error` and `run_errors` are relative to
/// `||V||_F^2`, unlike [`NmfResult::final_loss`] which is absolute.
///
/// ### Errors
///
/// * `NmfConsensusInvalidK` if `k < 2`, `NmfConsensusTooFewRuns` if
///   `n_runs < 2`.
/// * `NmfConsensusEmptyCluster` if a cluster came out empty, since the consensus
///   factor would carry a zero row and the refit would be degenerate.
/// * Anything the solve or the consensus step raises.
#[allow(clippy::too_many_arguments)]
pub fn nmf_consensus_gpu<R, In>(
    v: &In,
    k: usize,
    n_runs: usize,
    base_seed: u64,
    opts: &HalsOpts<f32>,
    params: Option<ConsensusParams<f32>>,
    scratch: &NmfGpuScratch<R>,
    client: &ComputeClient<R>,
    verbose: usize,
) -> Result<ConsensusNmfResult<f32>, BixverseErrors>
where
    R: Runtime,
    In: GpuNmfData<R>,
{
    let params = params.unwrap_or_default();
    if k < 2 {
        return Err(BixverseErrors::NmfConsensusInvalidK { k });
    }
    if n_runs < 2 {
        return Err(BixverseErrors::NmfConsensusTooFewRuns { n_runs });
    }

    let restarts = stabilised_nmf_gpu(v, k, n_runs, base_seed, opts, scratch, client, verbose)?;
    let clusters = consensus_from_restarts(&restarts, k, &params, verbose)?;

    if let Some(cluster) = clusters.sizes.iter().position(|&s| s == 0) {
        return Err(BixverseErrors::NmfConsensusEmptyCluster { cluster, k });
    }

    let sq_frob = v.sq_frob();
    let (w, h, loss) = match params.target {
        ConsensusTarget::HRows => {
            // The consensus is already k x n, so it is H directly.
            let h = clusters.consensus.clone();
            let (w, loss) = nmf_refit_w_gpu(v, h.as_ref(), opts, scratch, client)?;
            (w, h, loss)
        }
        ConsensusTarget::WColumns => {
            // The consensus is k x m, so transpose it into an m x k W.
            let w = clusters.consensus.transpose().to_owned();
            let (h, loss) = nmf_refit_h_gpu(v, w.as_ref(), opts, scratch, client)?;
            (w, h, loss)
        }
    };

    let denom = error_denominator(sq_frob);
    let run_errors: Vec<f32> = restarts.losses.iter().map(|&l| l / denom).collect();

    Ok(ConsensusNmfResult {
        w,
        h,
        error: loss / denom,
        clusters,
        run_errors,
    })
}

/// Sweep k on the GPU and report stability against error.
///
/// One row per rank, no refit and no factors retained. `V` is uploaded once for
/// the whole sweep, which is where most of the GPU advantage sits: the CPU path
/// re-reads `V` for all `k_range.len() * n_runs` solves.
///
/// Ranks are swept sequentially. There is one device, so nothing to interleave.
///
/// Two conditions that are properties of a particular rank rather than of the
/// input are recorded rather than raised, matching the CPU path: an empty cluster
/// (see `n_empty_clusters`) and a density filter that left fewer than `k`
/// components (see `consensus_failed`, with `stability` set to `NaN`). Anything
/// else aborts the sweep.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `k_range` - Ranks to evaluate, non-empty, every entry at least two
/// * `n_runs` - Restarts per rank, at least two
/// * `base_seed` - Seed offset; each rank gets a disjoint block of seeds
/// * `opts` - Solver options; the `init` field is ignored by the restarts
/// * `params` - Optional [`ConsensusParams`], defaulted if `None`
/// * `scratch` - Reusable device buffers, sized for at least `max(k_range)`
/// * `client` - CubeCL compute client
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// One [`KSweepEntry`] per entry of `k_range`, in the order given.
///
/// ### Errors
///
/// * `NmfKSweepEmptyRange` if `k_range` is empty.
/// * `NmfConsensusTooFewRuns` if `n_runs < 2`, `NmfConsensusInvalidK` if any
///   rank is below two.
#[allow(clippy::too_many_arguments)]
pub fn nmf_k_sweep_gpu<R, In>(
    v: &In,
    k_range: &[usize],
    n_runs: usize,
    base_seed: u64,
    opts: &HalsOpts<f32>,
    params: Option<ConsensusParams<f32>>,
    scratch: &NmfGpuScratch<R>,
    client: &ComputeClient<R>,
    verbose: usize,
) -> Result<Vec<KSweepEntry<f32>>, BixverseErrors>
where
    R: Runtime,
    In: GpuNmfData<R>,
{
    if k_range.is_empty() {
        return Err(BixverseErrors::NmfKSweepEmptyRange);
    }
    if n_runs < 2 {
        return Err(BixverseErrors::NmfConsensusTooFewRuns { n_runs });
    }
    if let Some(&k) = k_range.iter().find(|&&k| k < 2) {
        return Err(BixverseErrors::NmfConsensusInvalidK { k });
    }

    let params = params.unwrap_or_default();
    let verbosity = parse_verbosity_level(verbose);
    let denom = error_denominator(v.sq_frob());

    let mut out = Vec::with_capacity(k_range.len());
    for (i, &k) in k_range.iter().enumerate() {
        if verbosity.normal_verbosity() {
            println!(
                "Running GPU NMF k sweep: k = {} ({} of {})",
                k,
                i + 1,
                k_range.len()
            );
        }

        let seed = base_seed + (i * n_runs) as u64;
        let restarts = stabilised_nmf_gpu(
            v,
            k,
            n_runs,
            seed,
            opts,
            scratch,
            client,
            verbose.saturating_sub(1),
        )?;

        let clusters = match consensus_from_restarts(
            &restarts,
            k,
            &params,
            verbose.saturating_sub(1),
        ) {
            Ok(clusters) => Some(clusters),
            Err(BixverseErrors::NmfConsensusTooFewComponents { n_surviving, .. }) => {
                if verbosity.normal_verbosity() {
                    println!(
                        "  k = {}: only {} components survived the density filter (need {}); stability not scored",
                        k, n_surviving, k
                    );
                }
                None
            }
            Err(e) => return Err(e),
        };

        out.push(KSweepEntry {
            k,
            stability: clusters.as_ref().map(|c| c.stability).unwrap_or(f32::NAN),
            best_error: restarts.losses[restarts.best_idx] / denom,
            median_error: median(&restarts.losses) / denom,
            consensus_failed: clusters.is_none(),
            n_dropped: clusters.as_ref().map(|c| c.n_dropped).unwrap_or(k * n_runs),
            n_empty_clusters: clusters.as_ref().map(|c| c.n_empty_clusters).unwrap_or(0),
            n_converged: restarts.converged.iter().filter(|&&c| c).count(),
        });
    }

    Ok(out)
}

////////////////////////
// Dense entry points //
////////////////////////

/// Run GPU NMF on a dense matrix (single run).
///
/// Assumes samples x features, like the CPU bulk entry points.
///
/// ### Params
///
/// * `data` - The matrix to factorise, samples x features
/// * `k` - Number of components
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `nmf_hals_params` - Optional [`HalsOpts`]
/// * `device` - CubeCL device to run on
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// An [`NmfResult`], or errors from validation or the device path.
pub fn nmf_single_run_gpu<R: Runtime>(
    data: MatRef<f32>,
    k: usize,
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<f32>>,
    device: R::Device,
    verbose: usize,
) -> Result<NmfResult<f32>, BixverseErrors> {
    let opts = nmf_hals_params.unwrap_or_default();
    let client = R::client(&device);

    let processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));
    let input = GpuDenseNmfInput::<R>::new(processed.as_ref(), &client)?;
    let (m, n) = GpuNmfData::shape(&input);
    let scratch = NmfGpuScratch::<R>::new(m, n, k, opts.eps, &client)?;

    nmf_hals_gpu(&input, k, &opts, &scratch, &client, verbose)
}

/// Run GPU NMF on a dense matrix across random restarts.
///
/// `V` is uploaded once and reused for every restart.
///
/// ### Params
///
/// * `data` - The matrix to factorise, samples x features
/// * `k` - Number of components per run
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `nmf_hals_params` - Optional [`HalsOpts`]; the `init` field is ignored
/// * `n_runs` - Number of restarts
/// * `base_seed` - Seed offset
/// * `device` - CubeCL device to run on
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A [`StabilisedNmfResult`], or errors from validation or the device path.
#[allow(clippy::too_many_arguments)]
pub fn nmf_multiple_run_gpu<R: Runtime>(
    data: MatRef<f32>,
    k: usize,
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<f32>>,
    n_runs: usize,
    base_seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<StabilisedNmfResult<f32>, BixverseErrors> {
    let opts = nmf_hals_params.unwrap_or_default();
    let client = R::client(&device);

    let processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));
    let input = GpuDenseNmfInput::<R>::new(processed.as_ref(), &client)?;
    let (m, n) = GpuNmfData::shape(&input);
    let scratch = NmfGpuScratch::<R>::new(m, n, k, opts.eps, &client)?;

    stabilised_nmf_gpu(
        &input,
        k,
        n_runs,
        base_seed as u64,
        &opts,
        &scratch,
        &client,
        verbose,
    )
}

/// Run GPU consensus NMF on a dense matrix.
///
/// ### Params
///
/// * `data` - The matrix to factorise, samples x features
/// * `k` - Number of components, at least two
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `nmf_hals_params` - Optional [`HalsOpts`]
/// * `consensus_params` - Optional [`ConsensusParams`]
/// * `n_runs` - Number of restarts, at least two
/// * `base_seed` - Seed offset
/// * `device` - CubeCL device to run on
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A [`ConsensusNmfResult`], or errors from validation or the device path.
#[allow(clippy::too_many_arguments)]
pub fn nmf_consensus_run_gpu<R: Runtime>(
    data: MatRef<f32>,
    k: usize,
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<f32>>,
    consensus_params: Option<ConsensusParams<f32>>,
    n_runs: usize,
    base_seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<ConsensusNmfResult<f32>, BixverseErrors> {
    let opts = nmf_hals_params.unwrap_or_default();
    let client = R::client(&device);

    let processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));
    let input = GpuDenseNmfInput::<R>::new(processed.as_ref(), &client)?;
    let (m, n) = GpuNmfData::shape(&input);
    let scratch = NmfGpuScratch::<R>::new(m, n, k, opts.eps, &client)?;

    nmf_consensus_gpu(
        &input,
        k,
        n_runs,
        base_seed as u64,
        &opts,
        consensus_params,
        &scratch,
        &client,
        verbose,
    )
}

/// Sweep k with GPU consensus NMF on a dense matrix.
///
/// The scratch is sized once at the largest rank in `k_range`, and `V` is
/// uploaded once for the whole sweep.
///
/// ### Params
///
/// * `data` - The matrix to factorise, samples x features
/// * `k_range` - Ranks to evaluate, non-empty, every entry at least two
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `nmf_hals_params` - Optional [`HalsOpts`]
/// * `consensus_params` - Optional [`ConsensusParams`]
/// * `n_runs` - Restarts per rank, at least two
/// * `base_seed` - Seed offset
/// * `device` - CubeCL device to run on
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// One [`KSweepEntry`] per rank, or errors from validation or the device path.
#[allow(clippy::too_many_arguments)]
pub fn nmf_k_sweep_run_gpu<R: Runtime>(
    data: MatRef<f32>,
    k_range: &[usize],
    preprocessing: &str,
    nmf_hals_params: Option<HalsOpts<f32>>,
    consensus_params: Option<ConsensusParams<f32>>,
    n_runs: usize,
    base_seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<Vec<KSweepEntry<f32>>, BixverseErrors> {
    if k_range.is_empty() {
        return Err(BixverseErrors::NmfKSweepEmptyRange);
    }
    let opts = nmf_hals_params.unwrap_or_default();
    let client = R::client(&device);

    let processed = nmf_process_dense(data, &resolve_preprocessing(preprocessing));
    let input = GpuDenseNmfInput::<R>::new(processed.as_ref(), &client)?;
    let (m, n) = GpuNmfData::shape(&input);
    let k_max = k_range.iter().copied().max().unwrap_or(0);
    let scratch = NmfGpuScratch::<R>::new(m, n, k_max, opts.eps, &client)?;

    nmf_k_sweep_gpu(
        &input,
        k_range,
        n_runs,
        base_seed as u64,
        &opts,
        consensus_params,
        &scratch,
        &client,
        verbose,
    )
}

/////////////////////////
// Sparse entry points //
/////////////////////////

/// Build the device-resident sparse input from a host matrix.
///
/// Pre-processing needs CSC, so the caller's matrix is converted before scaling
/// exactly as the CPU sparse path does.
///
/// ### Params
///
/// * `data` - Sparse `V`, samples x features
/// * `preprocessing` - Column-scaling mode
/// * `use_second_layer` - Use `data_2` (normalised counts) instead of `data`
/// * `client` - CubeCL compute client
/// * `verbose` - Verbosity level, used only for the transpose notice
///
/// ### Returns
///
/// The uploaded [`GpuSparseNmfInput`].
fn build_sparse_input<R: Runtime>(
    data: CompressedSparseData2<f32>,
    preprocessing: &str,
    use_second_layer: bool,
    client: &ComputeClient<R>,
    verbose: usize,
) -> Result<GpuSparseNmfInput<R>, BixverseErrors> {
    let csc = ensure_csc(data, parse_verbosity_level(verbose));
    let processed = nmf_process_sparse(
        &csc,
        &resolve_preprocessing(preprocessing),
        use_second_layer,
    )?;

    let host = if use_second_layer {
        SparseInput::<f32, f32>::from_secondary(&processed)?
    } else {
        SparseInput::<f32, f32>::from_primary(&processed)?
    };

    GpuSparseNmfInput::<R>::new(host, client)
}

/// Run GPU NMF on a sparse matrix (single run).
///
/// ### Params
///
/// * `data` - Sparse `V`, samples x features
/// * `k` - Number of components
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `use_second_layer` - Use `data_2` (normalised counts) instead of `data`
/// * `nmf_hals_params` - Optional [`HalsOpts`]
/// * `device` - CubeCL device to run on
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// An [`NmfResult`], or errors from validation or the device path.
#[allow(clippy::too_many_arguments)]
pub fn nmf_single_run_sparse_gpu<R: Runtime>(
    data: CompressedSparseData2<f32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    device: R::Device,
    verbose: usize,
) -> Result<NmfResult<f32>, BixverseErrors> {
    let opts = nmf_hals_params.unwrap_or_default();
    let client = R::client(&device);

    let input = build_sparse_input::<R>(data, preprocessing, use_second_layer, &client, verbose)?;
    let (m, n) = GpuNmfData::shape(&input);
    let scratch = NmfGpuScratch::<R>::new(m, n, k, opts.eps, &client)?;

    nmf_hals_gpu(&input, k, &opts, &scratch, &client, verbose)
}

/// Run GPU NMF on a sparse matrix across random restarts.
///
/// ### Params
///
/// * `data` - Sparse `V`, samples x features
/// * `k` - Number of components per run
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `use_second_layer` - Use `data_2` (normalised counts) instead of `data`
/// * `nmf_hals_params` - Optional [`HalsOpts`]; the `init` field is ignored
/// * `n_runs` - Number of restarts
/// * `base_seed` - Seed offset
/// * `device` - CubeCL device to run on
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A [`StabilisedNmfResult`], or errors from validation or the device path.
#[allow(clippy::too_many_arguments)]
pub fn nmf_multiple_run_sparse_gpu<R: Runtime>(
    data: CompressedSparseData2<f32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    n_runs: usize,
    base_seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<StabilisedNmfResult<f32>, BixverseErrors> {
    let opts = nmf_hals_params.unwrap_or_default();
    let client = R::client(&device);

    let input = build_sparse_input::<R>(data, preprocessing, use_second_layer, &client, verbose)?;
    let (m, n) = GpuNmfData::shape(&input);
    let scratch = NmfGpuScratch::<R>::new(m, n, k, opts.eps, &client)?;

    stabilised_nmf_gpu(
        &input,
        k,
        n_runs,
        base_seed as u64,
        &opts,
        &scratch,
        &client,
        verbose,
    )
}

/// Run GPU consensus NMF on a sparse matrix.
///
/// ### Params
///
/// * `data` - Sparse `V`, samples x features
/// * `k` - Number of components, at least two
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `use_second_layer` - Use `data_2` (normalised counts) instead of `data`
/// * `nmf_hals_params` - Optional [`HalsOpts`]
/// * `consensus_params` - Optional [`ConsensusParams`]
/// * `n_runs` - Number of restarts, at least two
/// * `base_seed` - Seed offset
/// * `device` - CubeCL device to run on
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A [`ConsensusNmfResult`], or errors from validation or the device path.
#[allow(clippy::too_many_arguments)]
pub fn nmf_consensus_run_sparse_gpu<R: Runtime>(
    data: CompressedSparseData2<f32>,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    consensus_params: Option<ConsensusParams<f32>>,
    n_runs: usize,
    base_seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<ConsensusNmfResult<f32>, BixverseErrors> {
    let opts = nmf_hals_params.unwrap_or_default();
    let client = R::client(&device);

    let input = build_sparse_input::<R>(data, preprocessing, use_second_layer, &client, verbose)?;
    let (m, n) = GpuNmfData::shape(&input);
    let scratch = NmfGpuScratch::<R>::new(m, n, k, opts.eps, &client)?;

    nmf_consensus_gpu(
        &input,
        k,
        n_runs,
        base_seed as u64,
        &opts,
        consensus_params,
        &scratch,
        &client,
        verbose,
    )
}

/// Sweep k with GPU consensus NMF on a sparse matrix.
///
/// This is the shape the GPU path is really for: `V` uploads once and serves all
/// `k_range.len() * n_runs` solves.
///
/// ### Params
///
/// * `data` - Sparse `V`, samples x features
/// * `k_range` - Ranks to evaluate, non-empty, every entry at least two
/// * `preprocessing` - Column-scaling mode, see [`parse_nmf_processing`]
/// * `use_second_layer` - Use `data_2` (normalised counts) instead of `data`
/// * `nmf_hals_params` - Optional [`HalsOpts`]
/// * `consensus_params` - Optional [`ConsensusParams`]
/// * `n_runs` - Restarts per rank, at least two
/// * `base_seed` - Seed offset
/// * `device` - CubeCL device to run on
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// One [`KSweepEntry`] per rank, or errors from validation or the device path.
#[allow(clippy::too_many_arguments)]
pub fn nmf_k_sweep_run_sparse_gpu<R: Runtime>(
    data: CompressedSparseData2<f32>,
    k_range: &[usize],
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: Option<HalsOpts<f32>>,
    consensus_params: Option<ConsensusParams<f32>>,
    n_runs: usize,
    base_seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<Vec<KSweepEntry<f32>>, BixverseErrors> {
    if k_range.is_empty() {
        return Err(BixverseErrors::NmfKSweepEmptyRange);
    }
    let opts = nmf_hals_params.unwrap_or_default();
    let client = R::client(&device);

    let input = build_sparse_input::<R>(data, preprocessing, use_second_layer, &client, verbose)?;
    let (m, n) = GpuNmfData::shape(&input);
    let k_max = k_range.iter().copied().max().unwrap_or(0);
    let scratch = NmfGpuScratch::<R>::new(m, n, k_max, opts.eps, &client)?;

    nmf_k_sweep_gpu(
        &input,
        k_range,
        n_runs,
        base_seed as u64,
        &opts,
        consensus_params,
        &scratch,
        &client,
        verbose,
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use faer::Mat;

    use crate::methods::nmf_hals::NmfInit;

    /// The default device, or `None` when the machine has no usable GPU, in
    /// which case every test here returns early rather than failing.
    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Exact rank-k non-negative product with clearly separated components, so
    /// the consensus clustering has something to find rather than a set of
    /// near-collinear directions.
    fn rank_k_input(m: usize, n: usize, k: usize) -> Mat<f32> {
        let w = Mat::<f32>::from_fn(m, k, |i, j| if i % k == j { 5.0 } else { 0.3 });
        let h = Mat::<f32>::from_fn(k, n, |i, j| if j % k == i { 4.0 } else { 0.2 });
        w * h
    }

    fn test_opts() -> HalsOpts<f32> {
        HalsOpts::<f32>::new(200, 1e-7, 1e-10, 10, NmfInit::Nndsvd)
    }

    // The whole consensus path end to end on device: restarts, the host-side
    // clustering, and the refit of the partner factor. The bookkeeping assertions
    // are the same ones the CPU consensus tests make.
    #[test]
    fn test_consensus_returns_usable_factors() {
        if try_device().is_none() {
            return;
        }
        let (m, n, k, n_runs) = (25usize, 18usize, 2usize, 4usize);
        let v = rank_k_input(m, n, k);

        let res = nmf_consensus_run_gpu::<WgpuRuntime>(
            v.as_ref(),
            k,
            "none",
            Some(test_opts()),
            None,
            n_runs,
            0,
            WgpuDevice::DefaultDevice,
            0,
        )
        .unwrap();

        assert_eq!(res.w.nrows(), m);
        assert_eq!(res.w.ncols(), k);
        assert_eq!(res.h.nrows(), k);
        assert_eq!(res.h.ncols(), n);
        assert_eq!(res.run_errors.len(), n_runs);
        assert_eq!(res.clusters.labels.len(), k * n_runs);

        // A loose bound here is worthless: refitting against a random H lands
        // around 0.5, so only a tight one says the consensus factor is real.
        assert!(
            res.error < 1e-4,
            "consensus error {} is too high",
            res.error
        );
        assert!(
            res.clusters.stability > 0.9,
            "stability {} is too low",
            res.clusters.stability
        );

        // The consensus rows carry the unit-norm invariant the clustering assumes.
        for r in 0..k {
            let sq: f32 = (0..res.clusters.consensus.ncols())
                .map(|j| res.clusters.consensus[(r, j)] * res.clusters.consensus[(r, j)])
                .sum();
            assert!(
                (sq.sqrt() - 1.0).abs() < 1e-4,
                "consensus row {r} is not unit norm"
            );
        }
    }

    // The W-target arm takes the other branch of the refit, so it needs its own
    // check that the returned orientation is the documented one.
    #[test]
    fn test_consensus_w_target_returns_the_documented_orientation() {
        if try_device().is_none() {
            return;
        }
        let (m, n, k, n_runs) = (25usize, 18usize, 2usize, 4usize);
        let v = rank_k_input(m, n, k);
        let params = ConsensusParams::<f32>::new(ConsensusTarget::WColumns, None, None, 100, 3, 42);

        let res = nmf_consensus_run_gpu::<WgpuRuntime>(
            v.as_ref(),
            k,
            "none",
            Some(test_opts()),
            Some(params),
            n_runs,
            0,
            WgpuDevice::DefaultDevice,
            0,
        )
        .unwrap();

        assert_eq!(res.w.nrows(), m);
        assert_eq!(res.w.ncols(), k);
        assert_eq!(res.h.nrows(), k);
        assert_eq!(res.h.ncols(), n);
        // The consensus is k x m for this target.
        assert_eq!(res.clusters.consensus.nrows(), k);
        assert_eq!(res.clusters.consensus.ncols(), m);
    }

    #[test]
    fn test_k_sweep_rows_are_ordered_and_scored() {
        if try_device().is_none() {
            return;
        }
        let (m, n) = (30usize, 20usize);
        let v = rank_k_input(m, n, 3);
        let k_range = [2usize, 3, 4];

        let rows = nmf_k_sweep_run_gpu::<WgpuRuntime>(
            v.as_ref(),
            &k_range,
            "none",
            Some(test_opts()),
            None,
            3,
            0,
            WgpuDevice::DefaultDevice,
            0,
        )
        .unwrap();

        assert_eq!(rows.len(), k_range.len());
        for (row, &k) in rows.iter().zip(k_range.iter()) {
            assert_eq!(row.k, k);
            assert!(row.best_error <= row.median_error + 1e-6);
            assert!(row.n_converged <= 3);
            if !row.consensus_failed {
                assert!(row.stability.is_finite());
            }
        }
    }

    #[test]
    fn test_consensus_and_sweep_argument_guards() {
        if try_device().is_none() {
            return;
        }
        let v = rank_k_input(20, 14, 2);
        let dev = || WgpuDevice::DefaultDevice;

        assert!(matches!(
            nmf_consensus_run_gpu::<WgpuRuntime>(
                v.as_ref(),
                1,
                "none",
                Some(test_opts()),
                None,
                4,
                0,
                dev(),
                0
            ),
            Err(BixverseErrors::NmfConsensusInvalidK { k: 1 })
        ));
        assert!(matches!(
            nmf_consensus_run_gpu::<WgpuRuntime>(
                v.as_ref(),
                2,
                "none",
                Some(test_opts()),
                None,
                1,
                0,
                dev(),
                0
            ),
            Err(BixverseErrors::NmfConsensusTooFewRuns { n_runs: 1 })
        ));
        assert!(matches!(
            nmf_k_sweep_run_gpu::<WgpuRuntime>(
                v.as_ref(),
                &[],
                "none",
                Some(test_opts()),
                None,
                3,
                0,
                dev(),
                0
            ),
            Err(BixverseErrors::NmfKSweepEmptyRange)
        ));
        assert!(matches!(
            nmf_k_sweep_run_gpu::<WgpuRuntime>(
                v.as_ref(),
                &[2, 1],
                "none",
                Some(test_opts()),
                None,
                3,
                0,
                dev(),
                0
            ),
            Err(BixverseErrors::NmfConsensusInvalidK { k: 1 })
        ));
    }

    // The sparse entry point has to reach the same answer as the dense one on
    // the same matrix, which also covers the CSR-to-CSC guard: the R side always
    // hands over CSR, and scaling the wrong axis would return quiet nonsense.
    #[test]
    fn test_sparse_entry_point_matches_dense() {
        if try_device().is_none() {
            return;
        }
        let (m, n, k) = (24usize, 16usize, 2usize);
        let v = rank_k_input(m, n, k);
        let csr = CompressedSparseData2::<f32, f32>::from_dense_matrix(
            v.as_ref(),
            CompressedSparseFormat::Csr,
        );

        let opts = HalsOpts::<f32>::new(200, 1e-7, 1e-10, 10, NmfInit::Random { seed: 5 });

        let dense = nmf_single_run_gpu::<WgpuRuntime>(
            v.as_ref(),
            k,
            "none",
            Some(opts.clone()),
            WgpuDevice::DefaultDevice,
            0,
        )
        .unwrap();
        let sparse = nmf_single_run_sparse_gpu::<WgpuRuntime>(
            csr,
            k,
            "none",
            false,
            Some(opts),
            WgpuDevice::DefaultDevice,
            0,
        )
        .unwrap();

        let sq_frob: f32 = (0..m)
            .map(|i| (0..n).map(|j| v[(i, j)] * v[(i, j)]).sum::<f32>())
            .sum();
        let rel_dense = dense.final_loss / sq_frob;
        let rel_sparse = sparse.final_loss / sq_frob;
        assert!(
            (rel_dense - rel_sparse).abs() < 1e-4,
            "dense {rel_dense} and sparse {rel_sparse} entry points disagree"
        );
    }
}
