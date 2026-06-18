//! Symphony reference building and query mapping.
//!
//! Reference: Kang et al., Nat Methods, 2021. The reference is a Harmony-
//! corrected embedding compressed into per-cluster summary terms (Nr, C);
//! query mapping projects new cells with the reference's PCA loadings and
//! corrects them via a closed-form mixture-of-experts ridge regression that
//! depends only on the cached terms, not on the reference cells.

use ann_search_rs::*;
use faer::linalg::solvers::{DenseSolveCore, PartialPivLu};
use faer::{Mat, MatRef};
use indexmap::IndexSet;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_batch_correction::batch_utils::cosine_normalise;
use crate::single_cell::sc_batch_correction::harmony::{
    BatchInfo, HarmonyParams, HarmonyResult, compute_cosine_distances, create_batch_infos,
    harmony_with_state, initialise_r_from_dist,
};
use crate::single_cell::sc_batch_correction::harmony_v2::{HarmonyParamsV2, harmony_v2_with_state};
use crate::single_cell::sc_processing::pca::{SingleCellPcaParams, pca_on_sc_sparse_stats};

////////////
// Consts //
////////////

/// Symphony default for z-score clipping.
const SCALE_CLIP: f32 = 10.0;

////////////
// Params //
////////////

/// Harmony backend to use when constructing the reference.
pub enum HarmonyBackend {
    /// Version 1 from Harmony, see Korsunsky, et al., Nat Methods, 2019.
    V1(HarmonyParams),
    /// Version 2 from Harmony, see Patikas, et al., bioRxiv, 2026.
    V2(HarmonyParamsV2),
}

/// Symphony query-mapping parameters.
pub struct SymphonyMapParams {
    /// Soft-clustering fuzziness for query -> reference centroid assignment.
    /// Smaller means closer to hard clustering. Symphony R default is 0.1.
    pub sigma: f32,
    /// Ridge penalty on batch coefficients. Intercept is never penalised.
    /// Symphony R hardcodes 1.0.
    pub lambda: f32,
}

/// Default implementation for [SymphonyMapParams].
impl Default for SymphonyMapParams {
    fn default() -> Self {
        Self {
            sigma: 0.1,
            lambda: 1.0,
        }
    }
}

/////////////
// Results //
/////////////

/// Symphony reference.
pub struct SymphonyReference {
    /// HVG indices into the original gene universe (same order as `gene_means`
    /// / `gene_sds` / loadings rows).
    pub gene_indices: Vec<usize>,
    /// Per-HVG mean of the normalised reference data.
    pub gene_means: Vec<f32>,
    /// Per-HVG standard deviation of the normalised reference data.
    pub gene_sds: Vec<f32>,
    /// PCA gene loadings (n_hvgs x d).
    pub loadings: Mat<f32>,
    /// Pre-Harmony PCA scores (N x d).
    pub z_orig: Mat<f32>,
    /// Post-Harmony corrected embedding (N x d).
    pub z_corr: Mat<f32>,
    /// Soft cluster assignments (K x N).
    pub r: Mat<f32>,
    /// Cosine-normalised reference centroids (K x d). Used for query soft
    /// clustering.
    pub centroids: Mat<f32>,
    /// Cluster sizes: row-sums of `r` (length K).
    pub nr: Vec<f32>,
    /// Cached `R * Z_corr` (K x d).
    pub c: Mat<f32>,
}

/// Symphony query mapping result.
pub struct SymphonyQuery {
    /// Query cells projected into reference PC space (N_q x d).
    pub z_pca: Mat<f32>,
    /// Query cells after MoE batch correction (N_q x d).
    pub z_corr: Mat<f32>,
    /// Query soft cluster assignments onto reference centroids (K x N_q).
    pub r: Mat<f32>,
}

/////////////
// Helpers //
/////////////

/// Symmetric clamp of a scalar to `[-t, t]`.
///
/// ### Params
///
/// * `x` - The value to clamp.
/// * `t` - The (non-negative) symmetric bound.
///
/// ### Returns
///
/// `x` clamped to `[-t, t]`.
#[inline]
fn clamp_sym(x: f32, t: f32) -> f32 {
    if x > t {
        t
    } else if x < -t {
        -t
    } else {
        x
    }
}

/// Construct the N_q x n_hvgs scaled dense matrix needed for projection.
///
/// Mirrors Symphony's R `mapQuery` gene-synchronisation step: for each
/// reference HVG, scales the query column with the *reference's* mean and
/// SD, clips at +/-10, fills implicit-zero entries with the appropriately
/// scaled value. HVGs absent from the query become all-zero columns.
///
/// ### Params
///
/// * `f_path_query` - Path to the gene-based binary file.
/// * `cell_indices_query` - The cell indices to keep.
/// * `ref_to_query_gene_map` - The reference to query gene map with `None`
///   for HVGs from the reference that are not represented in this data set.
/// * `ref_means` - The feature means from the reference.
/// * `ref_sds` - The standard deviations from the reference.
///
/// ### Returns
///
/// An N_q x n_hvgs dense matrix of clipped z-scores against the reference
/// statistics. Columns corresponding to absent HVGs are all zero.
fn build_scaled_query_matrix(
    f_path_query: &str,
    cell_indices_query: &[usize],
    ref_to_query_gene_map: &[Option<usize>],
    ref_means: &[f32],
    ref_sds: &[f32],
) -> Result<Mat<f32>, BixverseErrors> {
    let n_q = cell_indices_query.len();
    let n_hvgs = ref_to_query_gene_map.len();

    // gene indices of query genes we actually need to load.
    let present_query_indices: Vec<usize> =
        ref_to_query_gene_map.iter().filter_map(|x| *x).collect();

    // reverse map: query-file gene index -> reference HVG slot.
    let query_to_slot: FxHashMap<usize, usize> = ref_to_query_gene_map
        .iter()
        .enumerate()
        .filter_map(|(slot, opt)| opt.map(|q_idx| (q_idx, slot)))
        .collect();

    let cell_set: IndexSet<u32> = cell_indices_query.iter().map(|&x| x as u32).collect();
    let reader = ParallelSparseReader::new(f_path_query)?;
    let gene_chunks: Vec<CscGeneChunk> =
        reader.read_gene_parallel_filtered(&present_query_indices, &cell_set)?;

    // slot-indexed chunk lookup. slots with no chunk stay as all-zero
    // columns (HVGs absent from the query).
    let mut chunks_by_slot: Vec<Option<&CscGeneChunk>> = vec![None; n_hvgs];
    for chunk in &gene_chunks {
        if let Some(&slot) = query_to_slot.get(&chunk.original_index) {
            chunks_by_slot[slot] = Some(chunk);
        }
    }

    // column-major flat buffer: slot s occupies out[s * n_q .. (s+1) * n_q].
    let mut out = vec![0.0f32; n_hvgs * n_q];
    out.par_chunks_exact_mut(n_q)
        .zip(chunks_by_slot.par_iter())
        .enumerate()
        .for_each(|(slot, (col, maybe_chunk))| {
            let Some(chunk) = maybe_chunk else {
                return; // absent HVG -> zero column
            };
            let mean = ref_means[slot];
            let sd = ref_sds[slot];
            let inv_sd = if sd > 1e-8 { 1.0 / sd } else { 0.0 };
            let baseline = clamp_sym(-mean * inv_sd, SCALE_CLIP);

            col.fill(baseline);
            for (i, &pos) in chunk.indices.iter().enumerate() {
                let val = chunk.data_norm[i].to_f32();
                col[pos as usize] = clamp_sym((val - mean) * inv_sd, SCALE_CLIP);
            }
        });

    // Column-major flat -> N_q x n_hvgs Mat.
    Ok(Mat::<f32>::from_fn(n_q, n_hvgs, |i, j| out[j * n_q + i]))
}

/// Symphony MoE correction with cached reference compression terms.
///
/// Per-cluster ridge solve: design block (B+1) x (B+1) with `Nr[k]` added
/// to the intercept, `C[k,:]` added to the intercept RHS, identity ridge
/// on batch terms only. Intercept coefficient is zeroed before subtraction
/// (handled implicitly by skipping column 0 in the apply step).
///
/// ### Params
///
/// * `z_pca` - Query PCA scores (N_q x d).
/// * `r` - Query soft cluster assignments (K x N_q).
/// * `batch_infos` - Query batch infos.
/// * `nr` - Reference cluster sizes (length K).
/// * `c` - Reference compression term (K x d).
/// * `lambda` - Ridge penalty on batch terms.
///
/// ### Returns
///
/// The batch-corrected query embedding (N_q x d).
fn moe_correct_query(
    z_pca: MatRef<f32>,
    r: MatRef<f32>,
    batch_infos: &[BatchInfo],
    nr: &[f32],
    c: MatRef<f32>,
    lambda: f32,
) -> Mat<f32> {
    let n_q = z_pca.nrows();
    let d = z_pca.ncols();
    let k = r.nrows();
    let n_vars = batch_infos.len();

    assert_eq!(r.ncols(), n_q);
    assert_eq!(nr.len(), k);
    assert_eq!(c.nrows(), k);
    assert_eq!(c.ncols(), d);

    // Design column layout: intercept (col 0) + one-hot per variable.
    let mut offsets = Vec::with_capacity(n_vars);
    let mut col = 1usize;
    for info in batch_infos {
        offsets.push(col);
        col += info.n_levels;
    }
    let p = col;

    // Phase 1 (parallel over clusters): solve each cluster's regression.
    let weights: Vec<Mat<f32>> = (0..k)
        .into_par_iter()
        .map(|cluster| {
            let mut design_cov = Mat::<f64>::zeros(p, p);
            let mut phi_z = Mat::<f64>::zeros(p, d);
            let mut active: Vec<usize> = Vec::with_capacity(1 + n_vars);

            for cell in 0..n_q {
                let r_val = r[(cluster, cell)] as f64;

                active.clear();
                active.push(0);
                for var_idx in 0..n_vars {
                    let level = batch_infos[var_idx].cell_to_level[cell];
                    active.push(offsets[var_idx] + level);
                }

                for &ci in &active {
                    for feat in 0..d {
                        phi_z[(ci, feat)] += r_val * z_pca[(cell, feat)] as f64;
                    }
                }
                for (i, &ci) in active.iter().enumerate() {
                    for &cj in &active[i..] {
                        design_cov[(ci, cj)] += r_val;
                        if ci != cj {
                            design_cov[(cj, ci)] += r_val;
                        }
                    }
                }
            }

            // reference compression injections.
            design_cov[(0, 0)] += nr[cluster] as f64;
            for feat in 0..d {
                phi_z[(0, feat)] += c[(cluster, feat)] as f64;
            }

            // ridge: identity on batch terms, intercept unpenalised.
            for i in 1..p {
                design_cov[(i, i)] += lambda as f64;
            }

            let lu: PartialPivLu<f64> = design_cov.partial_piv_lu();
            let inv_cov = lu.inverse();
            let w_f64 = &inv_cov * &phi_z;
            Mat::<f32>::from_fn(p, d, |i, j| w_f64[(i, j)] as f32)
        })
        .collect();

    // Phase 2: subtract each cluster's batch correction.
    let mut out = vec![0.0f32; n_q * d];
    out.par_chunks_mut(d).enumerate().for_each(|(cell, row)| {
        for feat in 0..d {
            row[feat] = z_pca[(cell, feat)];
        }
        for cluster in 0..k {
            let w = &weights[cluster];
            let r_val = r[(cluster, cell)];
            for var_idx in 0..n_vars {
                let level = batch_infos[var_idx].cell_to_level[cell];
                let c_idx = offsets[var_idx] + level;
                for feat in 0..d {
                    row[feat] -= r_val * w[(c_idx, feat)];
                }
            }
        }
    });

    Mat::from_fn(n_q, d, |i, j| out[i * d + j])
}

///////////////
// Reference //
///////////////

/// Build a Symphony reference.
///
/// Runs sparse PCA over the HVGs (capturing per-gene means and SDs), runs
/// Harmony for batch correction, and compresses the result into the cached
/// terms used at query time.
///
/// ### Params
///
/// * `f_path` - Path to the gene-based sparse binary file.
/// * `cell_indices` - Cell indices to include in the reference.
/// * `hvg_indices` - HVG gene indices, already selected upstream.
/// * `batch_labels` - One label slice per batch variable, each with length
///   = `cell_indices.len()`. At least one variable required.
/// * `pca_params` - PCA parameters. `mean_center` and `normalise_variance`
///   should both be true: the stored means/SDs *are* the transformation
///   used to scale the query at mapping time.
/// * `no_pcs` - Number of principal components.
/// * `harmony_backend` - Which Harmony variant and its parameters.
/// * `clr_offsets` - Optional CLR offsets passed through to the PCA call.
/// * `seed` - Seed.
/// * `verbose` - Verbosity (0 silent, 1 normal, 2 detailed).
///
/// ### Returns
///
/// A [SymphonyReference] holding the PCA loadings, reference means/SDs,
/// pre- and post-Harmony embeddings, soft cluster assignments, cosine-
/// normalised centroids and the compression terms (`nr`, `c`) needed for
/// query mapping.
#[allow(clippy::too_many_arguments)]
pub fn build_symphony_reference(
    f_path: &str,
    cell_indices: &[usize],
    hvg_indices: &[usize],
    batch_labels: &[Vec<usize>],
    pca_params: &SingleCellPcaParams,
    no_pcs: usize,
    harmony_backend: HarmonyBackend,
    clr_offsets: Option<&[f64]>,
    seed: usize,
    verbose: usize,
) -> Result<SymphonyReference, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start = Instant::now();

    assert!(
        !batch_labels.is_empty(),
        "Need at least one batch variable for reference Harmony"
    );
    for (i, labels) in batch_labels.iter().enumerate() {
        assert_eq!(
            labels.len(),
            cell_indices.len(),
            "batch_labels[{}] length must equal cell_indices.len()",
            i
        );
    }

    if verbosity.normal_verbosity() {
        println!("Symphony: building reference");
    }

    let (scores, loadings, _, gene_means, gene_sds) = pca_on_sc_sparse_stats(
        f_path,
        cell_indices,
        hvg_indices,
        no_pcs,
        pca_params,
        clr_offsets,
        seed,
        verbose,
    )?;

    let z_orig = scores.clone();

    let harmony_result: HarmonyResult = match harmony_backend {
        HarmonyBackend::V1(ref params) => {
            harmony_with_state(scores.as_ref(), batch_labels, params, seed, verbose)?
        }
        HarmonyBackend::V2(ref params) => {
            harmony_v2_with_state(scores.as_ref(), batch_labels, params, seed, verbose)?
        }
    };

    let z_corr = harmony_result.z_corr;
    let r = harmony_result.r;

    let k = r.nrows();
    let n = r.ncols();
    let d = z_corr.ncols();

    // Nr = row sums of R; accumulate in f64 to guard against drift over
    // many cells before storing back as f32.
    let nr: Vec<f32> = (0..k)
        .into_par_iter()
        .map(|cluster| (0..n).map(|cell| r[(cluster, cell)] as f64).sum::<f64>() as f32)
        .collect();

    // C = R * Z_corr, K x d. Manual parallel matmul with f64 inner products:
    // faer's f32 GEMM is not Kahan-summed, so the dot product over n cells
    // drifts visibly at large n. Result stored back as f32.
    let c: Mat<f32> = {
        let r_ref = r.as_ref();
        let z_ref = z_corr.as_ref();
        let buf: Vec<f32> = (0..(k * d))
            .into_par_iter()
            .map(|idx| {
                let i = idx / d;
                let j = idx % d;
                let mut acc = 0.0f64;
                for cell in 0..n {
                    acc += r_ref[(i, cell)] as f64 * z_ref[(cell, j)] as f64;
                }
                acc as f32
            })
            .collect();
        Mat::<f32>::from_fn(k, d, |i, j| buf[i * d + j])
    };
    assert_eq!(c.nrows(), k);
    assert_eq!(c.ncols(), d);

    // Centroids = row-normalised C
    let centroids = cosine_normalise(&c);

    if verbosity.normal_verbosity() {
        println!("Symphony reference built in {:.2?}", start.elapsed());
    }

    Ok(SymphonyReference {
        gene_indices: hvg_indices.to_vec(),
        gene_means,
        gene_sds,
        loadings,
        z_orig,
        z_corr,
        r,
        centroids,
        nr,
        c,
    })
}

///////////////////
// Query mapping //
///////////////////

/// Map a query onto a Symphony reference.
///
/// ### Params
///
/// * `reference` - The Symphony reference.
/// * `f_path_query` - Path to the query gene-based sparse binary file.
/// * `cell_indices_query` - Query cell indices.
/// * `ref_to_query_gene_map` - For each reference HVG slot (in
///   `reference.gene_indices` order), the corresponding gene index in the
///   query file, or `None` if the HVG is absent from the query (a zero
///   column is filled in for that slot). Caller resolves gene identities
///   upstream.
/// * `batch_labels_query` - Per-batch-variable label slices for the query.
///   Pass an empty slice to skip batch correction (z_corr = z_pca).
/// * `params` - Mapping parameters (sigma, lambda).
/// * `verbose` - Verbosity (0 silent, 1 normal, 2 detailed).
///
/// ### Returns
///
/// A [SymphonyQuery] with the projected scores (`z_pca`), the MoE-corrected
/// embedding (`z_corr`), and the K x N_q soft assignment matrix (`r`)
/// against the reference centroids.
pub fn symphony_map_query(
    reference: &SymphonyReference,
    f_path_query: &str,
    cell_indices_query: &[usize],
    ref_to_query_gene_map: &[Option<usize>],
    batch_labels_query: &[Vec<usize>],
    params: &SymphonyMapParams,
    verbose: usize,
) -> Result<SymphonyQuery, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start = Instant::now();

    let n_hvgs = reference.gene_indices.len();
    assert_eq!(
        ref_to_query_gene_map.len(),
        n_hvgs,
        "Gene map length must equal number of reference HVGs"
    );
    for (i, labels) in batch_labels_query.iter().enumerate() {
        assert_eq!(
            labels.len(),
            cell_indices_query.len(),
            "batch_labels_query[{}] length must equal cell_indices_query.len()",
            i
        );
    }

    let n_q = cell_indices_query.len();
    let k = reference.centroids.nrows();

    if verbosity.normal_verbosity() {
        let n_present = ref_to_query_gene_map.iter().filter(|x| x.is_some()).count();
        println!(
            "Symphony query mapping: {} cells, {}/{} HVGs present in query",
            n_q, n_present, n_hvgs
        );
    }

    // 1. Build N_q x n_hvgs scaled, dense matrix using reference mean/SD.
    let scaled = build_scaled_query_matrix(
        f_path_query,
        cell_indices_query,
        ref_to_query_gene_map,
        &reference.gene_means,
        &reference.gene_sds,
    )?;

    // 2. Project: Z_pca = scaled * loadings  (N_q x d)
    let z_pca: Mat<f32> = scaled.as_ref() * reference.loadings.as_ref();

    // 3. Soft cluster against reference centroids in cosine space.
    let z_pca_cos = cosine_normalise(&z_pca);
    let dist = compute_cosine_distances(reference.centroids.as_ref(), z_pca_cos.as_ref());
    let sigma_vec = vec![params.sigma; k];
    let r_query = initialise_r_from_dist(dist.as_ref(), &sigma_vec)?;

    // 4. MoE correction with cached reference terms.
    let z_corr = if batch_labels_query.is_empty() {
        z_pca.clone()
    } else {
        let batch_infos = create_batch_infos(batch_labels_query, n_q)?;
        moe_correct_query(
            z_pca.as_ref(),
            r_query.as_ref(),
            &batch_infos,
            &reference.nr,
            reference.c.as_ref(),
            params.lambda,
        )
    };

    if verbosity.normal_verbosity() {
        println!("Symphony query mapped in {:.2?}", start.elapsed());
    }

    Ok(SymphonyQuery {
        z_pca,
        z_corr,
        r: r_query,
    })
}

/// As [symphony_map_query], but takes pre-extracted reference fields
/// instead of a full [SymphonyReference]. Useful for callers that hold
/// only a slim subset of the reference.
///
/// ### Params
///
/// * `f_path_query` - Path to the query gene-based sparse binary file.
/// * `cell_indices_query` - Query cell indices.
/// * `ref_gene_means` - Per-HVG mean of the normalised reference data.
/// * `ref_gene_sds` - Per-HVG standard deviation of the normalised reference data.
/// * `ref_loadings` - PCA gene loadings from the reference (n_hvgs x d).
/// * `ref_centroids` - Cosine-normalised reference centroids (K x d).
/// * `ref_nr` - Reference cluster sizes, i.e. row-sums of the reference `r` (length K).
/// * `ref_c` - Cached `R * Z_corr` compression term from the reference (K x d).
/// * `ref_to_query_gene_map` - For each reference HVG slot, the corresponding
///   gene index in the query file, or `None` if the HVG is absent from the
///   query (a zero column is filled in for that slot). Caller resolves gene
///   identities upstream.
/// * `batch_labels_query` - Per-batch-variable label slices for the query.
///   Pass an empty slice to skip batch correction (z_corr = z_pca).
/// * `params` - Mapping parameters (sigma, lambda).
/// * `verbose` - Verbosity (0 silent, 1 normal, 2 detailed).
///
/// ### Returns
///
/// A [SymphonyQuery] with the projected scores (`z_pca`), the MoE-corrected
/// embedding (`z_corr`), and the K x N_q soft assignment matrix (`r`)
/// against the reference centroids.
#[allow(clippy::too_many_arguments)]
pub fn symphony_map_query_parts(
    f_path_query: &str,
    cell_indices_query: &[usize],
    ref_gene_means: &[f32],
    ref_gene_sds: &[f32],
    ref_loadings: MatRef<f32>,
    ref_centroids: MatRef<f32>,
    ref_nr: &[f32],
    ref_c: MatRef<f32>,
    ref_to_query_gene_map: &[Option<usize>],
    batch_labels_query: &[Vec<usize>],
    params: &SymphonyMapParams,
    verbose: usize,
) -> Result<SymphonyQuery, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start = Instant::now();

    let n_hvgs = ref_gene_means.len();
    assert_eq!(ref_to_query_gene_map.len(), n_hvgs);
    assert_eq!(ref_gene_sds.len(), n_hvgs);
    assert_eq!(ref_loadings.nrows(), n_hvgs);
    for (i, labels) in batch_labels_query.iter().enumerate() {
        assert_eq!(
            labels.len(),
            cell_indices_query.len(),
            "batch_labels_query[{}] length must equal cell_indices_query.len()",
            i
        );
    }

    let n_q = cell_indices_query.len();
    let k = ref_centroids.nrows();

    if verbosity.normal_verbosity() {
        let n_present = ref_to_query_gene_map.iter().filter(|x| x.is_some()).count();
        println!(
            "Symphony query mapping: {} cells, {}/{} HVGs present in query",
            n_q, n_present, n_hvgs
        );
    }

    let scaled = build_scaled_query_matrix(
        f_path_query,
        cell_indices_query,
        ref_to_query_gene_map,
        ref_gene_means,
        ref_gene_sds,
    )?;

    let z_pca: Mat<f32> = scaled.as_ref() * ref_loadings;

    let z_pca_cos = cosine_normalise(&z_pca);
    let dist = compute_cosine_distances(ref_centroids, z_pca_cos.as_ref());
    let sigma_vec = vec![params.sigma; k];
    let r_query = initialise_r_from_dist(dist.as_ref(), &sigma_vec)?;

    let z_corr = if batch_labels_query.is_empty() {
        z_pca.clone()
    } else {
        let batch_infos = create_batch_infos(batch_labels_query, n_q)?;
        moe_correct_query(
            z_pca.as_ref(),
            r_query.as_ref(),
            &batch_infos,
            ref_nr,
            ref_c,
            params.lambda,
        )
    };

    if verbosity.normal_verbosity() {
        println!("Symphony query mapped in {:.2?}", start.elapsed());
    }

    Ok(SymphonyQuery {
        z_pca,
        z_corr,
        r: r_query,
    })
}

///////////////////////
// Label propagation //
///////////////////////

/// Cross-query version of [generate_knn_with_dist]: builds the index on
/// `reference`, queries `query` against it. Does not strip self-matches.
///
/// Used for reference -> query label transfer.
///
/// ### Params
///
/// * `reference` - The reference embedding to index (N_ref x d).
/// * `query` - The query embedding to search against the index (N_q x d).
/// * `knn_params` - KNN parameters (method, k, and method-specific settings),
///   see [KnnParams].
/// * `seed` - Seed.
/// * `verbose` - Verbosity.
///
/// ### Returns
///
/// A [ScKnnResults] holding, for each query cell, the indices of its k
/// nearest reference neighbours and their distances.
pub fn generate_knn_cross_with_dist(
    reference: MatRef<f32>,
    query: MatRef<f32>,
    knn_params: &KnnParams,
    seed: usize,
    verbose: bool,
) -> ScKnnResults {
    fn timed<T>(name: &str, verbose: bool, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        if verbose {
            println!("{}: {:.2?}", name, start.elapsed());
        }
        result
    }

    let knn_method = parse_knn_method(&knn_params.knn_method).unwrap_or_default();
    let k = knn_params.k;

    let (indices, distances) = match knn_method {
        KnnSearch::Annoy => {
            let index = timed("Generated Annoy index", verbose, || {
                build_annoy_index(reference, &knn_params.ann_dist, knn_params.n_tree, seed)
            })?;
            timed("Queried Annoy index", verbose, || {
                query_annoy_index(query, &index, k, knn_params.search_budget, true, verbose)
            })?
        }
        KnnSearch::Hnsw => {
            let index = timed("Generated HNSW index", verbose, || {
                build_hnsw_index(
                    reference,
                    knn_params.m,
                    knn_params.ef_construction,
                    &knn_params.ann_dist,
                    seed,
                    verbose,
                )
            });
            timed("Queried HNSW index", verbose, || {
                query_hnsw_index(query, &index, k, knn_params.ef_search, true, verbose)
            })?
        }
        KnnSearch::NNDescent => {
            let index = timed("Generated NNDescent index", verbose, || {
                build_nndescent_index(
                    reference,
                    &knn_params.ann_dist,
                    knn_params.delta,
                    knn_params.diversify_prob,
                    None,
                    None,
                    None,
                    None,
                    seed,
                    verbose,
                )
            })?;
            timed("Queried NNDescent index", verbose, || {
                query_nndescent_index(query, &index, k, knn_params.ef_budget, true, verbose)
            })?
        }
        KnnSearch::Exhaustive => {
            let index = timed("Generated Exhaustive index", verbose, || {
                build_exhaustive_index(reference, &knn_params.ann_dist)
            });
            timed("Queried Exhaustive index", verbose, || {
                query_exhaustive_index(query, &index, k, true, verbose)
            })?
        }
        KnnSearch::KmKnn => {
            let index = timed("Generated KmKnn index", verbose, || {
                build_kmknn_index(
                    reference,
                    &knn_params.ann_dist,
                    knn_params.n_list,
                    None,
                    seed,
                    verbose,
                )
            })?;
            timed("Queried KmKnn index", verbose, || {
                query_kmknn_index(query, &index, k, true, verbose)
            })?
        }
        KnnSearch::Ivf => {
            let index = timed("Generated IVF index", verbose, || {
                build_ivf_index(
                    reference,
                    knn_params.n_list,
                    None,
                    &knn_params.ann_dist,
                    seed,
                    verbose,
                )
            })?;
            timed("Queried IVF index", verbose, || {
                query_ivf_index(query, &index, k, knn_params.n_probe, true, verbose)
            })?
        }
    };

    // we passed return_dist = true above so distances are always Some
    Ok((indices, distances.expect("requested distances")))
}

/// Majority-vote label transfer from a kNN graph. Ties broken by lowest
/// label index. Confidence is the winning label's vote share.
///
/// ### Params
///
/// * `knn_indices` - For each query cell, the indices of its k reference
///   neighbours.
/// * `reference_labels` - Integer-encoded labels per reference cell.
/// * `n_labels` - Total number of distinct labels.
///
/// ### Returns
///
/// Tuple of `(predicted_label, confidence)` per query cell.
pub fn knn_majority_vote(
    knn_indices: &[Vec<usize>],
    reference_labels: &[usize],
    n_labels: usize,
) -> (Vec<usize>, Vec<f32>) {
    knn_indices
        .par_iter()
        .map(|neighbours| {
            let mut counts = vec![0u32; n_labels];
            for &nb in neighbours {
                counts[reference_labels[nb]] += 1;
            }
            let (best, &count) = counts.iter().enumerate().max_by_key(|(_, c)| **c).unwrap();
            (best, count as f32 / neighbours.len() as f32)
        })
        .unzip()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_batch_correction::harmony::create_batch_info;
    use approx::assert_relative_eq;
    use faer::mat;

    /////////////
    // Helpers //
    /////////////

    /// Small synthetic scenario: 6 cells, 2 features, 2 soft clusters,
    /// 1 batch variable with 2 levels (3 cells each). Feature 0 carries a
    /// clear +/-1 batch effect; feature 1 has only within-batch noise.
    fn make_simple_scenario() -> (Mat<f32>, Mat<f32>, Vec<BatchInfo>) {
        let z_pca = mat![
            [1.0_f32, 0.1],
            [1.0, 0.2],
            [1.0, 0.3],
            [-1.0, 0.1],
            [-1.0, 0.2],
            [-1.0, 0.3]
        ];
        let r = mat![
            [0.9_f32, 0.9, 0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let batch_infos = vec![create_batch_info(&labels, 6).unwrap()];
        (z_pca, r, batch_infos)
    }

    #[test]
    fn test_clamp_sym_within_range() {
        assert_eq!(clamp_sym(0.0, 10.0), 0.0);
        assert_eq!(clamp_sym(7.5, 10.0), 7.5);
        assert_eq!(clamp_sym(-7.5, 10.0), -7.5);
    }

    #[test]
    fn test_clamp_sym_at_boundary() {
        assert_eq!(clamp_sym(10.0, 10.0), 10.0);
        assert_eq!(clamp_sym(-10.0, 10.0), -10.0);
    }

    #[test]
    fn test_clamp_sym_outside_range() {
        assert_eq!(clamp_sym(15.0, 10.0), 10.0);
        assert_eq!(clamp_sym(-15.0, 10.0), -10.0);
        assert_eq!(clamp_sym(1e9, 10.0), 10.0);
        assert_eq!(clamp_sym(-1e9, 10.0), -10.0);
    }

    #[test]
    fn test_moe_correct_query_preserves_dimensions() {
        let (z_pca, r, batch_infos) = make_simple_scenario();
        let nr = vec![3.0_f32, 3.0];
        let c = mat![[0.0_f32, 0.0], [0.0, 0.0]];

        let result = moe_correct_query(
            z_pca.as_ref(),
            r.as_ref(),
            &batch_infos,
            &nr,
            c.as_ref(),
            1.0,
        );
        assert_eq!(result.nrows(), 6);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_moe_correct_query_reduces_batch_effect() {
        let (z_pca, r, batch_infos) = make_simple_scenario();
        // Reference centroids are at the origin in feature 0 (no shift).
        // Both clusters share the same feature-1 mean.
        let nr = vec![10.0_f32, 10.0];
        let c = mat![[0.0_f32, 2.0], [0.0, 2.0]];

        let result = moe_correct_query(
            z_pca.as_ref(),
            r.as_ref(),
            &batch_infos,
            &nr,
            c.as_ref(),
            1.0,
        );

        // Pre-correction batches differ by 2.0 in feature 0.
        let mean_b0_before = (z_pca[(0, 0)] + z_pca[(1, 0)] + z_pca[(2, 0)]) / 3.0;
        let mean_b1_before = (z_pca[(3, 0)] + z_pca[(4, 0)] + z_pca[(5, 0)]) / 3.0;
        let diff_before = (mean_b0_before - mean_b1_before).abs();

        let mean_b0_after = (result[(0, 0)] + result[(1, 0)] + result[(2, 0)]) / 3.0;
        let mean_b1_after = (result[(3, 0)] + result[(4, 0)] + result[(5, 0)]) / 3.0;
        let diff_after = (mean_b0_after - mean_b1_after).abs();

        assert!(
            diff_after < diff_before,
            "expected batch effect to shrink, before={diff_before}, after={diff_after}"
        );
    }

    #[test]
    fn test_moe_correct_query_no_batch_effect_stable() {
        // Query with no batch effect: all four cells at (0.1, 0.2), one
        // cluster (R all ones), reference Nr and C consistent with the
        // query distribution. Expected: output ~= input.
        let z_pca = mat![[0.1_f32, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]];
        let r = mat![[1.0_f32, 1.0, 1.0, 1.0]];
        let labels = vec![0, 0, 1, 1];
        let batch_infos = vec![create_batch_info(&labels, 4).unwrap()];
        let nr = vec![4.0_f32];
        // C = sum_i R[k,i] * z[i,:] under the matching distribution.
        let c = mat![[0.4_f32, 0.8]];

        let result = moe_correct_query(
            z_pca.as_ref(),
            r.as_ref(),
            &batch_infos,
            &nr,
            c.as_ref(),
            1.0,
        );

        for i in 0..4 {
            for j in 0..2 {
                assert_relative_eq!(result[(i, j)], z_pca[(i, j)], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_moe_correct_query_two_variables() {
        // Two batch variables, two levels each, all four combinations
        // represented. Tests that the design matrix layout handles
        // multiple variables correctly.
        let z_pca = mat![[1.0_f32, 0.5], [1.0, -0.5], [-1.0, 0.5], [-1.0, -0.5]];
        let r = mat![[1.0_f32, 1.0, 1.0, 1.0]];
        let labels_v1 = vec![0, 0, 1, 1]; // feature 0 batch effect
        let labels_v2 = vec![0, 1, 0, 1]; // feature 1 batch effect
        let batch_infos = vec![
            create_batch_info(&labels_v1, 4).unwrap(),
            create_batch_info(&labels_v2, 4).unwrap(),
        ];
        let nr = vec![10.0_f32];
        let c = mat![[0.0_f32, 0.0]];

        let result = moe_correct_query(
            z_pca.as_ref(),
            r.as_ref(),
            &batch_infos,
            &nr,
            c.as_ref(),
            1.0,
        );

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);

        // Both variables should be reduced post correction.
        let mean_v1b0_f0_before = (z_pca[(0, 0)] + z_pca[(1, 0)]) / 2.0;
        let mean_v1b1_f0_before = (z_pca[(2, 0)] + z_pca[(3, 0)]) / 2.0;
        let mean_v1b0_f0_after = (result[(0, 0)] + result[(1, 0)]) / 2.0;
        let mean_v1b1_f0_after = (result[(2, 0)] + result[(3, 0)]) / 2.0;
        assert!(
            (mean_v1b0_f0_after - mean_v1b1_f0_after).abs()
                < (mean_v1b0_f0_before - mean_v1b1_f0_before).abs()
        );

        let mean_v2b0_f1_before = (z_pca[(0, 1)] + z_pca[(2, 1)]) / 2.0;
        let mean_v2b1_f1_before = (z_pca[(1, 1)] + z_pca[(3, 1)]) / 2.0;
        let mean_v2b0_f1_after = (result[(0, 1)] + result[(2, 1)]) / 2.0;
        let mean_v2b1_f1_after = (result[(1, 1)] + result[(3, 1)]) / 2.0;
        assert!(
            (mean_v2b0_f1_after - mean_v2b1_f1_after).abs()
                < (mean_v2b0_f1_before - mean_v2b1_f1_before).abs()
        );
    }

    #[test]
    fn test_moe_correct_query_responds_to_nr() {
        // Same query, two reference Nr regimes. Different Nr should give
        // different corrections, confirming the cache injection is wired.
        let (z_pca, r, batch_infos) = make_simple_scenario();
        let c = mat![[0.0_f32, 0.0], [0.0, 0.0]];

        let small_nr = moe_correct_query(
            z_pca.as_ref(),
            r.as_ref(),
            &batch_infos,
            &[1.0, 1.0],
            c.as_ref(),
            1.0,
        );
        let large_nr = moe_correct_query(
            z_pca.as_ref(),
            r.as_ref(),
            &batch_infos,
            &[100.0, 100.0],
            c.as_ref(),
            1.0,
        );

        let mut max_diff = 0.0_f32;
        for i in 0..6 {
            for j in 0..2 {
                let d = (small_nr[(i, j)] - large_nr[(i, j)]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        assert!(
            max_diff > 1e-3,
            "Output should depend on Nr; max diff = {max_diff}"
        );
    }

    #[test]
    fn test_moe_correct_query_lambda_zero_is_finite() {
        // With lambda = 0 the batch terms are unpenalised. The design block
        // should still be invertible because Nr > 0 anchors the intercept
        // (otherwise the system can be singular under hard assignments).
        let (z_pca, r, batch_infos) = make_simple_scenario();
        let nr = vec![3.0_f32, 3.0];
        let c = mat![[0.0_f32, 0.0], [0.0, 0.0]];

        let result = moe_correct_query(
            z_pca.as_ref(),
            r.as_ref(),
            &batch_infos,
            &nr,
            c.as_ref(),
            0.0,
        );

        for i in 0..6 {
            for j in 0..2 {
                assert!(result[(i, j)].is_finite());
            }
        }
    }

    #[test]
    fn test_compression_cache_formula() {
        // Nr = row sums of R; C = R * Z_corr. This is the cache that the
        // query side consumes; the formula is inlined in
        // build_symphony_reference, so we re-derive it here as a guard.
        let r = mat![[0.7_f32, 0.3, 0.0], [0.3, 0.7, 1.0]];
        let z_corr = mat![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let k = r.nrows();
        let n = r.ncols();
        let nr: Vec<f32> = (0..k)
            .map(|cluster| (0..n).map(|cell| r[(cluster, cell)]).sum::<f32>())
            .collect();
        let c: Mat<f32> = r.as_ref() * z_corr.as_ref();

        let nr_expected = [1.0_f32, 2.0];
        let c_expected = mat![[1.6_f32, 2.6], [7.4, 9.4]];

        for i in 0..k {
            assert_relative_eq!(nr[i], nr_expected[i], epsilon = 1e-5);
        }
        for i in 0..k {
            for j in 0..z_corr.ncols() {
                assert_relative_eq!(c[(i, j)], c_expected[(i, j)], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_centroids_are_row_normalised() {
        // After cosine_normalise on the K x d compression matrix, each row
        // should have unit L2 norm.
        let c = mat![[3.0_f32, 4.0], [0.0, 5.0], [1.0, 0.0]];
        let centroids = cosine_normalise(&c);

        for i in 0..centroids.nrows() {
            let norm: f32 = (0..centroids.ncols())
                .map(|j| centroids[(i, j)] * centroids[(i, j)])
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }
}
