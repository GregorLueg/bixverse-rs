//! GPU-accelerated version of k-means clustering. Borrows heavily ideas from
//! the Flash-KMeans approach from Yang et al., arXive and ports over what can
//! be ported over to wgpu and cubecl.
//!
//! Re-uses the GPU infrastructure from `ann-search-rs` with `"gpu"` feature
//! enabled to avoid code duplication.

#![allow(missing_docs)]

use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::*;
use ann_search_rs::prelude::*;
use ann_search_rs::utils::dist::Dist;
use ann_search_rs::utils::{k_means_utils::*, matrix_to_flat};
use cubecl::prelude::*;
use faer::{Mat, MatRef};
use std::iter::Sum;
use std::time::Instant;

use crate::errors::BixverseErrors;
use crate::prelude::*;

////////////
// Params //
////////////

/// GPU k-means parameters. Mirrors [KMeansParamsWrappers]
#[derive(Clone, Copy, Debug)]
pub struct KMeansGpuParams {
    /// Maximum number of Lloyd's iterations.
    pub iters: usize,
    /// Optional initialisation strategy. `None` picks based on `n_centroids`.
    pub init: Option<KMeansInit>,
    /// Fixed number of iterations
    pub fixed: bool,
}

impl KMeansGpuParams {
    /// New params.
    ///
    /// ### Params
    ///
    /// * `iters` - Number of iterations
    /// * `init` - Optional initialisation
    /// * `fixed` - Shall the algorithm be run for a fixed set of iterations
    ///   or shall convergence be checked.
    pub fn new(iters: usize, init: Option<KMeansInit>, fixed: bool) -> Self {
        Self { iters, init, fixed }
    }
}

/// Default implementation for [KMeansGpuParams]
impl Default for KMeansGpuParams {
    fn default() -> Self {
        Self::new(50, None, true)
    }
}

/////////////
// Kernels //
/////////////

/// One-point-per-workgroup Euclidean argmin kernel.
///
/// Each workgroup cooperatively loads one point into shared memory; threads
/// then stride across centroids with each thread owning every `wg`-th centroid.
/// Thread 0 performs a serial argmin over the per-thread candidates stored in
/// shared memory and writes the winning centroid index. No N x K
/// materialisation.
///
/// ### Params
///
/// * `data` - Data points `[n_samples, dim / N]` as `Vector<F, N>`
/// * `centroids` - Centroid vectors `[k, dim / N]` as `Vector<F, N>`
/// * `assignments` - Output assignment indices `[n_samples]`
/// * `n_samples` - Total number of data points
/// * `k` - Number of centroids
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> point index; one workgroup
///   per point
#[cube(launch_unchecked)]
pub fn flash_assign_euclidean_smem<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    centroids: &Tensor<Vector<F, N>>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;

    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    if point_idx >= n_samples as usize {
        terminate!();
    }

    // Point cached in SMEM (cooperative load by all wg threads).
    let mut s_point = SharedMemory::<F>::new(dim_scalars);
    let p_base = point_idx * dim_lines;
    let mut load_idx: usize = tx;
    while load_idx < dim_scalars {
        let line_idx = load_idx / lanes;
        let lane = load_idx % lanes;
        let pl = data[p_base + line_idx];
        s_point[load_idx] = pl[lane];
        load_idx += wg;
    }
    sync_cube();

    // Stride centroids; one scalar argmin per thread.
    let mut best_dist = F::new(f32::MAX);
    let mut best_idx = 0u32;
    let mut c = tx as u32;
    while c < k {
        let mut sum = F::new(0.0);
        for i in 0..dim_lines {
            let c_line = centroids[c as usize * dim_lines + i];
            let s_off = i * lanes;
            #[unroll]
            for lane in 0..lanes {
                let diff = s_point[s_off + lane] - c_line[lane];
                sum += diff * diff;
            }
        }
        if sum < best_dist {
            best_dist = sum;
            best_idx = c;
        }
        c += wg as u32;
    }

    // Cross-thread argmin: thread 0 picks the winner.
    let mut s_dist = SharedMemory::<F>::new(32usize);
    let mut s_idx = SharedMemory::<u32>::new(32usize);
    s_dist[tx] = best_dist;
    s_idx[tx] = best_idx;
    sync_cube();

    if tx == 0usize {
        let mut bd = s_dist[0];
        let mut bi = s_idx[0];
        let mut t = 1u32;
        while t < wg as u32 {
            let d = s_dist[t as usize];
            if d < bd {
                bd = d;
                bi = s_idx[t as usize];
            }
            t += 1u32;
        }
        assignments[point_idx] = bi;
    }
}

/// One-point-per-workgroup cosine argmin kernel.
///
/// Each workgroup cooperatively loads one point into shared memory; threads
/// then stride across centroids with each thread owning every `wg`-th centroid.
/// Thread 0 performs a serial argmin over the per-thread candidates stored in
/// shared memory and writes the winning centroid index. Minimises
/// `1 - dot(x, c) / (||x|| * ||c||)` using precomputed norms.
///
/// ### Params
///
/// * `data` - Data points `[n_samples, dim / N]` as `Vector<F, N>`
/// * `centroids` - Centroid vectors `[k, dim / N]` as `Vector<F, N>`
/// * `point_norms` - Pre-computed L2 norms `[n_samples]`
/// * `centroid_norms` - Pre-computed L2 norms `[k]`
/// * `assignments` - Output assignment indices `[n_samples]`
/// * `n_samples` - Total number of data points
/// * `k` - Number of centroids
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> point index; one workgroup
///   per point
#[cube(launch_unchecked)]
pub fn flash_assign_cosine_smem<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    centroids: &Tensor<Vector<F, N>>,
    point_norms: &Tensor<F>,
    centroid_norms: &Tensor<F>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;

    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    if point_idx >= n_samples as usize {
        terminate!();
    }

    let mut s_point = SharedMemory::<F>::new(dim_scalars);
    let p_base = point_idx * dim_lines;
    let mut load_idx = tx;
    while load_idx < dim_scalars {
        let line_idx = load_idx / lanes;
        let lane = load_idx % lanes;
        let pl = data[p_base + line_idx];
        s_point[load_idx] = pl[lane];
        load_idx += wg;
    }
    sync_cube();

    let pnorm = point_norms[point_idx];

    let mut best_dist = F::new(f32::MAX);
    let mut best_idx = 0u32;
    let mut c = tx as u32;
    while c < k {
        let mut dot = F::new(0.0);
        for i in 0..dim_lines {
            let c_line = centroids[c as usize * dim_lines + i];
            let s_off = i * lanes;
            #[unroll]
            for lane in 0..lanes {
                dot += s_point[s_off + lane] * c_line[lane];
            }
        }
        let cnorm = centroid_norms[c as usize];
        let dist = F::new(1.0) - dot / (pnorm * cnorm);
        if dist < best_dist {
            best_dist = dist;
            best_idx = c;
        }
        c += wg as u32;
    }

    let mut s_dist = SharedMemory::<F>::new(32usize);
    let mut s_idx = SharedMemory::<u32>::new(32usize);
    s_dist[tx] = best_dist;
    s_idx[tx] = best_idx;
    sync_cube();

    if tx == 0usize {
        let mut bd = s_dist[0];
        let mut bi = s_idx[0];
        let mut t = 1u32;
        while t < wg as u32 {
            let d = s_dist[t as usize];
            if d < bd {
                bd = d;
                bi = s_idx[t as usize];
            }
            t += 1u32;
        }
        assignments[point_idx] = bi;
    }
}

/// Dispatch the assignment kernel and write results into a device-resident
/// buffer.
///
/// Internal to the driver loop. Dispatches either `flash_assign_euclidean_smem`
/// or `flash_assign_cosine_smem` depending on `metric`. Assumes `dim` is
/// already padded to a multiple of `LINE_SIZE` and all tensors reside on
/// `client`'s device.
///
/// ### Params
///
/// * `client` - CubeCL compute client for the target device
/// * `data_gpu` - Data points already on device `[n, dim]`
/// * `cent_gpu` - Centroid vectors `[k, dim]`
/// * `pnorm_gpu` - Pre-computed point L2 norms `[n]`; ignored for Euclidean
/// * `cnorm_gpu` - Pre-computed centroid L2 norms `[k]`; ignored for Euclidean
/// * `assign_gpu` - Output assignment indices `[n]`, overwritten in place
/// * `n` - Number of data points
/// * `k` - Number of centroids
/// * `dim` - Embedding dimensionality (must be a multiple of `LINE_SIZE`)
/// * `metric` - Distance metric (`SquaredEuclidean` or `Cosine`)
#[allow(clippy::too_many_arguments)]
fn flash_assign_device<T, R>(
    client: &ComputeClient<R>,
    data_gpu: &GpuTensor<R, T>,
    cent_gpu: &GpuTensor<R, T>,
    pnorm_gpu: &GpuTensor<R, T>,
    cnorm_gpu: &GpuTensor<R, T>,
    assign_gpu: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
    dim: usize,
    metric: &Dist,
) where
    R: Runtime,
    T: Float + Sum + cubecl::CubeElement + num_traits::Float + num_traits::FromPrimitive,
{
    let vec_size = LINE_SIZE;
    let dim_lines = dim / vec_size;
    let (gx, gy) = grid_2d(n as u32);
    let count = CubeCount::Static(gx, gy, 1);
    let cdim = CubeDim::new_1d(WORKGROUP_SIZE_X);

    match *metric {
        Dist::SquaredEuclidean => unsafe {
            flash_assign_euclidean_smem::launch_unchecked::<T, R>(
                client,
                count,
                cdim,
                vec_size,
                data_gpu.clone().into_tensor_arg(),
                cent_gpu.clone().into_tensor_arg(),
                assign_gpu.clone().into_tensor_arg(),
                n as u32,
                k as u32,
                dim_lines,
            );
        },
        Dist::Cosine => unsafe {
            flash_assign_cosine_smem::launch_unchecked::<T, R>(
                client,
                count,
                cdim,
                vec_size,
                data_gpu.clone().into_tensor_arg(),
                cent_gpu.clone().into_tensor_arg(),
                pnorm_gpu.clone().into_tensor_arg(),
                cnorm_gpu.clone().into_tensor_arg(),
                assign_gpu.clone().into_tensor_arg(),
                n as u32,
                k as u32,
                dim_lines,
            );
        },
        Dist::Manhattan => unreachable!(),
    }
}

/// Privatised histogram of cluster sizes.
///
/// Each workgroup maintains its own row in `privatised_counts [cube_count, k]`,
/// first zeroing it cooperatively and then counting the points from its stripe.
/// This avoids inter-workgroup atomic contention on a shared row. The
/// privatised rows are reduced by a subsequent `scan_columns_and_sum` pass.
///
/// ### Params
///
/// * `assignments` - Hard assignment indices `[n]`
/// * `privatised_counts` - Atomic per-workgroup cluster counters
///   `[cube_count * k]`, zeroed and filled in place
/// * `n` - Total number of data points
/// * `k` - Number of clusters
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> workgroup (row) index
#[cube(launch_unchecked)]
pub fn histogram_clusters_privatised(
    assignments: &Tensor<u32>,
    privatised_counts: &mut Tensor<Atomic<u32>>,
    n: u32,
    k: u32,
) {
    let r = CUBE_POS_X;
    let tx = UNIT_POS_X;
    let wg = WORKGROUP_SIZE_X;

    // 1. Cooperative Inline Zeroing:
    // Threads in this workgroup clear their designated chunk of the shared row
    let mut c = tx;
    while c < k {
        let idx = r * k + c;
        Atomic::fetch_and(&privatised_counts[idx as usize], 0u32);
        c += wg;
    }
    sync_cube(); // Enforce a barrier so no thread counts until everything is zeroed

    // 2. Standard privatised Counting Pass
    let total_threads = wg * CUBE_COUNT_X;
    let mut i = r * wg + tx;
    while i < n {
        let chunk_c = assignments[i as usize];
        let idx = r * k + chunk_c;
        Atomic::fetch_add(&privatised_counts[idx as usize], 1u32);
        i += total_threads;
    }
}

/// Column-wise prefix scan of the privatised count matrix.
///
/// For each cluster column `c`, serialises over all workgroup rows: replaces
/// each `privatised_counts[r, c]` with the running exclusive prefix sum and
/// writes the column total into `counts[c]`. This transforms the privatised
/// rows into per-workgroup exclusive write offsets for use by
/// `scatter_csr_privatised`.
///
/// ### Params
///
/// * `privatised_counts` - Privatised count matrix `[cube_count * k]`;
///   overwritten in place with exclusive row-prefix sums
/// * `counts` - Output total cluster sizes `[k]`
/// * `k` - Number of clusters
/// * `cube_count` - Number of workgroup rows
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> cluster column index
#[cube(launch_unchecked)]
pub fn scan_columns_and_sum(
    privatised_counts: &mut Tensor<u32>,
    counts: &mut Tensor<u32>,
    k: u32,
    cube_count: u32,
) {
    let c = ABSOLUTE_POS_X;
    if c >= k {
        terminate!();
    }

    let mut acc = 0u32;
    let mut r = 0u32;
    while r < cube_count {
        let idx = r * k + c;
        let val = privatised_counts[idx as usize];
        privatised_counts[idx as usize] = acc; // Overwrite with local block prefix sum
        acc += val;
        r += 1u32;
    }
    counts[c as usize] = acc;
}

/// Exclusive prefix sum of cluster counts into a `k + 1` offset array.
///
/// Single-thread serial scan. Writes global segment start positions;
/// `offsets[k]` equals the total number of points.
///
/// ### Params
///
/// * `counts` - Cluster sizes `[k]`
/// * `offsets` - Output exclusive prefix sums `[k + 1]`
/// * `k` - Number of clusters
///
/// ### Grid mapping
///
/// * Single cube, single thread (`UNIT_POS_X == 0`)
#[cube(launch_unchecked)]
pub fn exclusive_scan_offsets_2(counts: &Tensor<u32>, offsets: &mut Tensor<u32>, k: u32) {
    if UNIT_POS_X == 0u32 {
        offsets[0] = 0u32;
        let mut acc = 0u32;
        let mut c = 0u32;
        while c < k {
            acc += counts[c as usize];
            offsets[(c + 1u32) as usize] = acc;
            c += 1u32;
        }
    }
}

/// Add global segment offsets into the privatised prefix-sum rows.
///
/// Each element `privatised_counts[r * k + c]` holds a workgroup-local
/// exclusive offset after `scan_columns_and_sum`; adding `offsets[c]`
/// converts it to an absolute write position into the CSR index array, ready
/// for `scatter_csr_privatised`.
///
/// ### Params
///
/// * `privatised_counts` - Privatised prefix-sum matrix `[cube_count * k]`;
///   updated in place to global write cursors
/// * `offsets` - Global exclusive prefix sums `[k + 1]` from
///   `exclusive_scan_offsets_2`
/// * `k` - Number of clusters
/// * `cube_count` - Number of workgroup rows
#[cube(launch_unchecked)]
pub fn merge_offsets_to_cursors(
    privatised_counts: &mut Tensor<u32>,
    offsets: &Tensor<u32>,
    k: u32,
    cube_count: u32,
) {
    let idx = ABSOLUTE_POS_X as usize;
    let total_elements = (cube_count * k) as usize;
    if idx < total_elements {
        let c = (idx as u32) % k;
        privatised_counts[idx] += offsets[c as usize];
    }
}

/// Contention-free scatter of point indices into CSR order.
///
/// Each workgroup handles the same stripe of points it counted in
/// `histogram_clusters_privatised`, atomically advancing its own row's cursor
/// for each point. Cursors from different workgroups target non-overlapping
/// slots, so atomic contention is negligible compared with a naive single-row
/// approach. Within-segment order is non-deterministic, which is acceptable
/// because the downstream centroid summation is order-independent.
///
/// ### Params
///
/// * `assignments` - Hard assignment indices `[n]`
/// * `privatised_cursors` - Atomic per-workgroup write cursors `[cube_count * k]`,
///   seeded by `merge_offsets_to_cursors` and atomically advanced in place
/// * `all_indices` - Output point indices in CSR order `[n]`
/// * `n` - Total number of data points
/// * `k` - Number of clusters
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> workgroup (row) index
#[cube(launch_unchecked)]
pub fn scatter_csr_privatised(
    assignments: &Tensor<u32>,
    privatised_cursors: &mut Tensor<Atomic<u32>>,
    all_indices: &mut Tensor<u32>,
    n: u32,
    k: u32,
) {
    let r = CUBE_POS_X;
    let tx = UNIT_POS_X;
    let wg = WORKGROUP_SIZE_X;

    let total_threads = wg * CUBE_COUNT_X;
    let mut i = r * wg + tx;

    while i < n {
        let c = assignments[i as usize];
        let idx = r * k + c;
        let pos = Atomic::fetch_add(&privatised_cursors[idx as usize], 1u32);
        all_indices[pos as usize] = i;
        i += total_threads;
    }
}

/// Build a CSR layout from hard assignments using the privatised histogram
/// pipeline.
///
/// Runs five kernels in sequence: privatised histogram, column scan, global
/// prefix scan, cursor merging, and contention-free scatter. No host readback
/// occurs between stages. Pre-allocated scratch buffers are passed in so that
/// the driver loop can reuse them across iterations without extra allocation.
///
/// ### Params
///
/// * `assignments` - Hard assignment indices `[n]` already on device
/// * `n` - Number of data points
/// * `k` - Number of clusters
/// * `cube_count` - Number of workgroups used for the privatised histogram
/// * `privatised_counts` - Pre-allocated scratch buffer `[cube_count * k]`
/// * `counts` - Pre-allocated cluster size buffer `[k]`
/// * `offsets` - Pre-allocated offset buffer `[k + 1]`
/// * `all_indices` - Pre-allocated output index buffer `[n]`; cluster `c`
///   occupies `all_indices[offsets[c]..offsets[c+1]]` on exit
/// * `client` - CubeCL compute client for the target device
#[allow(clippy::too_many_arguments)]
pub fn build_csr_gpu_privatised<R>(
    assignments: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
    cube_count: usize,
    privatised_counts: &GpuTensor<R, u32>,
    counts: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    all_indices: &GpuTensor<R, u32>,
    client: &ComputeClient<R>,
) where
    R: Runtime,
{
    unsafe {
        // Step 1: privatised Histogram (handles its own zeroing)
        histogram_clusters_privatised::launch_unchecked::<R>(
            client,
            CubeCount::Static(cube_count as u32, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            assignments.clone().into_tensor_arg(),
            privatised_counts.clone().into_tensor_arg(),
            n as u32,
            k as u32,
        );

        // Step 2: Parallel Column Scan
        scan_columns_and_sum::launch_unchecked::<R>(
            client,
            CubeCount::Static(k.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            privatised_counts.clone().into_tensor_arg(),
            counts.clone().into_tensor_arg(),
            k as u32,
            cube_count as u32,
        );

        // Step 3: Global Offset Prefix Scan
        exclusive_scan_offsets_2::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            counts.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            k as u32,
        );

        // Step 4: Transform Local Counts to Global Cursors
        let total_elements = cube_count * k;
        merge_offsets_to_cursors::launch_unchecked::<R>(
            client,
            CubeCount::Static(total_elements.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            privatised_counts.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            k as u32,
            cube_count as u32,
        );

        // Step 5: Contention-Free Index Scatter
        scatter_csr_privatised::launch_unchecked::<R>(
            client,
            CubeCount::Static(cube_count as u32, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            assignments.clone().into_tensor_arg(),
            privatised_counts.clone().into_tensor_arg(),
            all_indices.clone().into_tensor_arg(),
            n as u32,
            k as u32,
        );
    }
}

/// Recompute centroids as the mean of assigned points via segmented reduction
/// over the CSR layout.
///
/// One workgroup per cluster. Thread `tx` owns output dimensions
/// `tx, tx + wg, ...` and accumulates them over the cluster's segment.
/// Atomic-free.
///
/// ### Params
///
/// * `data` - Flattened data points `[n, dim]`
/// * `all_indices` - Point indices in CSR order `[n]` from
///   `build_csr_gpu_privatised`
/// * `offsets` - Exclusive prefix sums `[k + 1]` from
///   `build_csr_gpu_privatised`; cluster `c` occupies
///   `all_indices[offsets[c]..offsets[c+1]]`
/// * `centroids` - Centroid vectors `[k, dim]`, updated in place; empty
///   clusters retain their prior value
/// * `k` - Number of clusters
/// * `dim` - Embedding dimensionality (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> cluster index
/// * `UNIT_POS_X` -> dimension stride offset within the cluster's centroid row
#[cube(launch_unchecked)]
pub fn segmented_centroid_update<F: Float>(
    data: &Tensor<F>,
    all_indices: &Tensor<u32>,
    offsets: &Tensor<u32>,
    centroids: &mut Tensor<F>,
    k: u32,
    #[comptime] dim: usize,
) {
    let cluster = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if cluster >= k {
        terminate!();
    }

    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;

    let seg_start = offsets[cluster as usize];
    let seg_end = offsets[(cluster + 1u32) as usize];
    let count = seg_end - seg_start;

    if count == 0u32 {
        terminate!();
    }

    let inv_count = F::new(1.0) / F::cast_from(count);
    let cent_base = cluster as usize * dim;

    let mut e = tx;
    while e < dim {
        let mut acc = F::new(0.0);
        let mut p = 0u32;
        while p < count {
            let global = all_indices[(seg_start + p) as usize];
            acc += data[global as usize * dim + e];
            p += 1u32;
        }
        centroids[cent_base + e] = acc * inv_count;
        e += wg;
    }
}

/// Recompute centroids in place from a CSR layout.
///
/// Wraps `segmented_centroid_update`. Empty clusters retain their current
/// value.
///
/// ### Params
///
/// * `data` - Data points already on device `[n, dim]`
/// * `all_indices` - Point indices in CSR order `[n]` from
///   `build_csr_gpu_privatised`
/// * `offsets` - Exclusive prefix sums `[k + 1]` from
///   `build_csr_gpu_privatised`
/// * `centroids` - Centroid vectors `[k, dim]`, updated in place
/// * `k` - Number of clusters
/// * `dim` - Embedding dimensionality
/// * `client` - CubeCL compute client for the target device
pub fn segmented_update<R, T>(
    data: &GpuTensor<R, T>,
    all_indices: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    centroids: &GpuTensor<R, T>,
    k: usize,
    dim: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    T: Float + cubecl::CubeElement + num_traits::Float,
{
    let (gx, gy) = grid_2d(k as u32);
    unsafe {
        segmented_centroid_update::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            data.clone().into_tensor_arg(),
            all_indices.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            centroids.clone().into_tensor_arg(),
            k as u32,
            dim,
        );
    }
}

/// Count how many points changed cluster between two assignment snapshots.
///
/// One thread per point; atomically increments `changed[0]` when a point's
/// current assignment differs from its previous one. `changed` must be
/// zero-initialised before launch. This is the convergence signal for the
/// driver: once the partition stops changing the centroids are stable, and
/// unlike centroid drift it reaches zero exactly (the cosine path's drift
/// never does, due to fp32 renormalisation noise).
///
/// ### Params
///
/// * `curr` - Current assignment indices `[n]`
/// * `prev` - Previous assignment indices `[n]`
/// * `changed` - Single-element atomic counter `[1]`, incremented in place
/// * `n` - Total number of data points
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> point index
#[cube(launch_unchecked)]
pub fn count_changed(
    curr: &Tensor<u32>,
    prev: &Tensor<u32>,
    changed: &mut Tensor<Atomic<u32>>,
    n: u32,
) {
    let i = ABSOLUTE_POS_X;
    if i >= n {
        terminate!();
    }
    if curr[i as usize] != prev[i as usize] {
        Atomic::fetch_add(&changed[0], 1u32);
    }
}

/// Per-centroid L2 norm of the centroid matrix.
///
/// One workgroup per centroid; thread 0 accumulates the sum of squares across
/// the full row and writes its square root. Kept device-side so that cosine
/// assignment never requires a host readback of the centroid matrix between
/// iterations.
///
/// ### Params
///
/// * `centroids` - Centroid vectors `[k, dim]`
/// * `norms` - Output L2 norms `[k]`, written by thread 0 of each workgroup
/// * `k` - Number of centroids
/// * `dim` - Embedding dimensionality (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> centroid index
/// * Only `UNIT_POS_X == 0` writes output
#[cube(launch_unchecked)]
pub fn centroid_norms_l2<F: Float>(
    centroids: &Tensor<F>,
    norms: &mut Tensor<F>,
    k: u32,
    #[comptime] dim: usize,
) {
    let c = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if c >= k {
        terminate!();
    }
    if UNIT_POS_X == 0u32 {
        let base = c as usize * dim;
        let mut acc = F::new(0.0);
        for e in 0..dim {
            let v = centroids[base + e];
            acc += v * v;
        }
        norms[c as usize] = F::sqrt(acc);
    }
}

/// One Lloyd's iteration entirely on device: assign, rebuild CSR, update
/// centroids.
///
/// Runs `flash_assign_device` to compute hard assignments,
/// `build_csr_gpu_privatised` to sort point indices into cluster segments, and
/// `segmented_update` to recompute each centroid as the mean of its assigned
/// points. For cosine distance, centroid norms are refreshed in place via
/// `centroid_norms_l2` so that `cnorm_gpu` is consistent with `cent_gpu` on
/// exit. Pre-allocated scratch buffers are passed in to avoid per-iteration
/// allocation.
///
/// ### Params
///
/// * `client` - CubeCL compute client for the target device
/// * `data_gpu` - Data points already on device `[n, dim]`
/// * `cent_gpu` - Centroid vectors `[k, dim]`, updated in place
/// * `pnorm_gpu` - Pre-computed point L2 norms `[n]`; ignored for Euclidean
/// * `cnorm_gpu` - Pre-computed centroid L2 norms `[k]`, refreshed on exit for
///   cosine; ignored for Euclidean
/// * `assign_gpu` - Output assignment indices `[n]`, overwritten each call
/// * `n` - Number of data points
/// * `k` - Number of centroids
/// * `dim` - Embedding dimensionality (must be a multiple of `LINE_SIZE`)
/// * `metric` - Distance metric (`SquaredEuclidean` or `Cosine`)
/// * `cube_count` - Number of workgroups used for the privatised histogram
/// * `privatised_counts` - Pre-allocated scratch buffer `[cube_count * k]`
/// * `counts` - Pre-allocated cluster size buffer `[k]`
/// * `offsets` - Pre-allocated offset buffer `[k + 1]`
/// * `all_indices` - Pre-allocated CSR index buffer `[n]`
/// * `fence_scratch` - Single-element buffer used as a GPU fence; see
///   `gpu_fence`
#[allow(clippy::too_many_arguments)]
fn lloyd_step<T, R>(
    client: &ComputeClient<R>,
    data_gpu: &GpuTensor<R, T>,
    cent_gpu: &GpuTensor<R, T>,
    pnorm_gpu: &GpuTensor<R, T>,
    cnorm_gpu: &GpuTensor<R, T>,
    assign_gpu: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
    dim: usize,
    metric: &Dist,
    cube_count: usize,
    privatised_counts: &GpuTensor<R, u32>,
    counts: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    all_indices: &GpuTensor<R, u32>,
) where
    R: Runtime,
    T: Float + Sum + cubecl::CubeElement + num_traits::Float + num_traits::FromPrimitive,
{
    flash_assign_device(
        client, data_gpu, cent_gpu, pnorm_gpu, cnorm_gpu, assign_gpu, n, k, dim, metric,
    );

    build_csr_gpu_privatised::<R>(
        assign_gpu,
        n,
        k,
        cube_count,
        privatised_counts,
        counts,
        offsets,
        all_indices,
        client,
    );

    segmented_update::<R, T>(data_gpu, all_indices, offsets, cent_gpu, k, dim, client);

    if *metric == Dist::Cosine {
        let (gx, gy) = grid_2d(k as u32);
        unsafe {
            centroid_norms_l2::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_1d(1),
                cent_gpu.clone().into_tensor_arg(),
                cnorm_gpu.clone().into_tensor_arg(),
                k as u32,
                dim,
            );
        }
    }
}

//////////
// Main //
//////////

/// Generate k-means clusters on the GPU.
///
/// Device-resident Lloyd's loop: FlashAssign for assignment, privatised
/// counting-sort CSR plus segmented reduction for the update. Initialisation
/// runs on the host (reusing `fast_random_init` / `kmeans_parallel_init`) and
/// is uploaded once. Convergence is detected via assignment stability: the loop
/// stops once the number of points changing cluster between iterations drops to
/// a small floor. This terminates the cosine path too, where fp32
/// renormalisation noise keeps per-centroid drift pinned at a tiny non-zero
/// value indefinitely. A small non-zero floor absorbs near-equidistant points
/// that flip between equally good centroids without stalling termination.
///
/// ### Params
///
/// * `data` - The data to cluster, samples x features
/// * `dist` - Distance metric, `"euclidean"` or `"cosine"`; unknown strings
///   default to squared Euclidean
/// * `n_centroids` - Number of centroids
/// * `kmeans_params` - Optional [KMeansGpuParams]
/// * `seed` - Seed for reproducible initialisation
/// * `device` - CubeCL runtime device
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(centroid matrix of shape n_centroids x dim, assignments)`
#[allow(clippy::too_many_arguments)]
pub fn k_means_clusters_gpu<T, R>(
    data: MatRef<T>,
    dist: &str,
    n_centroids: usize,
    kmeans_params: Option<KMeansGpuParams>,
    seed: usize,
    device: R::Device,
    verbose: bool,
) -> Result<(Mat<T>, Vec<usize>), BixverseErrors>
where
    R: Runtime,
    T: AnnSearchGpuFloat + AnnSearchFloat + BixverseFloat + Sum + cubecl::CubeElement,
{
    let start = Instant::now();

    let params = kmeans_params.unwrap_or_default();
    let dist = parse_ann_dist(dist).unwrap_or_else(|| {
        println!(
            "Unknown string provided ({:?}). Defaulting to Squared Euclidean",
            dist
        );
        Dist::default()
    });
    if dist == Dist::Manhattan {
        return Err(BixverseErrors::DistanceNotSupported(dist));
    }

    let (data_flat, n, dim) = matrix_to_flat(data);
    let dim_padded = dim.next_multiple_of(LINE_SIZE);
    let data_padded = if dim_padded != dim {
        pad_vectors(&data_flat, n, dim, dim_padded)
    } else {
        data_flat.clone()
    };

    let init_method = params.init.unwrap_or(if n_centroids > 200 {
        KMeansInit::Random
    } else {
        KMeansInit::KMeansParallel
    });
    let centroids = match init_method {
        KMeansInit::Random => {
            if verbose {
                println!("  Initialising centroids via fast random selection");
            }
            fast_random_init(&data_padded, dim_padded, n, n_centroids, seed)
        }
        KMeansInit::KMeansParallel => {
            if verbose {
                println!("  Initialising centroids via k-means||");
            }
            let init_norms: Vec<T> = (0..n)
                .map(|i| T::calculate_l2_norm(&data_padded[i * dim_padded..(i + 1) * dim_padded]))
                .collect();
            kmeans_parallel_init(
                &data_padded,
                &init_norms,
                dim_padded,
                n,
                n_centroids,
                &dist,
                seed,
            )
        }
    };

    if verbose {
        println!("  Moving data to GPU.");
    }

    let client = R::client(&device);
    let data_gpu = GpuTensor::<R, T>::from_slice(&data_padded, vec![n, dim_padded], &client);
    let cent_gpu =
        GpuTensor::<R, T>::from_slice(&centroids, vec![n_centroids, dim_padded], &client);
    let assign_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);

    if verbose {
        println!("  ... moved data to GPU: {:.2?}", start.elapsed());
    }

    let pnorm_gpu = if dist == Dist::Cosine {
        let pnorms: Vec<T> = (0..n)
            .map(|i| T::calculate_l2_norm(&data_padded[i * dim_padded..(i + 1) * dim_padded]))
            .collect();
        GpuTensor::<R, T>::from_slice(&pnorms, vec![n], &client)
    } else {
        GpuTensor::<R, T>::from_slice(&[T::one()], vec![1], &client)
    };

    let cnorm_gpu = if dist == Dist::Cosine {
        let cnorms: Vec<T> = (0..n_centroids)
            .map(|c| T::calculate_l2_norm(&centroids[c * dim_padded..(c + 1) * dim_padded]))
            .collect();
        GpuTensor::<R, T>::from_slice(&cnorms, vec![n_centroids], &client)
    } else {
        GpuTensor::<R, T>::from_slice(&[T::one()], vec![1], &client)
    };

    if verbose {
        println!(
            "  Running Lloyd's iterations (GPU, {})",
            if params.fixed {
                "fixed"
            } else {
                "assignment-checked"
            }
        );
    }

    let cube_count = 512usize;

    let privatised_counts = GpuTensor::<R, u32>::empty(vec![cube_count, n_centroids], &client);
    let counts = GpuTensor::<R, u32>::empty(vec![n_centroids], &client);
    let offsets = GpuTensor::<R, u32>::empty(vec![n_centroids + 1], &client);
    let all_indices = GpuTensor::<R, u32>::empty(vec![n], &client);

    if params.fixed {
        // No convergence check, no per-iteration readback: submit every
        // iteration back-to-back.
        if verbose {
            println!(
                "    Dispatching the {:?} iters to the GPU kernel.",
                params.iters
            )
        }
        for _ in 0..params.iters {
            lloyd_step(
                &client,
                &data_gpu,
                &cent_gpu,
                &pnorm_gpu,
                &cnorm_gpu,
                &assign_gpu,
                n,
                n_centroids,
                dim_padded,
                &dist,
                cube_count,
                &privatised_counts,
                &counts,
                &offsets,
                &all_indices,
            );
        }
    } else {
        // A change floor to absorb near-equidistant points that flip between
        // equally good centroids without stalling termination.
        let change_floor = (n / 10_000).max(1) as u32;

        let assign_alt_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);
        let (cnt_gx, cnt_gy) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));

        for iter in 0..params.iters {
            // `cur` receives this iteration's assignments; `prev` holds last.
            let (cur, prev) = if iter % 2 == 0 {
                (&assign_gpu, &assign_alt_gpu)
            } else {
                (&assign_alt_gpu, &assign_gpu)
            };

            lloyd_step(
                &client,
                &data_gpu,
                &cent_gpu,
                &pnorm_gpu,
                &cnorm_gpu,
                cur,
                n,
                n_centroids,
                dim_padded,
                &dist,
                cube_count,
                &privatised_counts,
                &counts,
                &offsets,
                &all_indices,
            );

            // Zeroed single-element atomic counter, recreated per iter.
            let changed_gpu = GpuTensor::<R, u32>::from_slice(&[0u32], vec![1], &client);
            unsafe {
                count_changed::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(cnt_gx, cnt_gy, 1),
                    CubeDim::new_1d(WORKGROUP_SIZE_X),
                    cur.clone().into_tensor_arg(),
                    prev.clone().into_tensor_arg(),
                    changed_gpu.clone().into_tensor_arg(),
                    n as u32,
                );
            }
            let changed = changed_gpu.read(&client)?[0];

            // Iteration 0 has no meaningful `prev` (uninitialised buffer), so
            // only test convergence from the second iteration onward.
            if iter > 0 && changed <= change_floor {
                if verbose {
                    println!(
                        "    Converged at iteration {} ({} assignments changed)",
                        iter + 1,
                        changed
                    );
                }
                break;
            }
            if verbose && (iter + 1) % 10 == 0 {
                println!(
                    "    Iteration {} complete ({} assignments changed)",
                    iter + 1,
                    changed
                );
            }
        }
    }

    // Final assignment against the converged centroids. cnorm_gpu is already
    // consistent with cent_gpu (refreshed at the end of the last lloyd_step).
    flash_assign_device(
        &client,
        &data_gpu,
        &cent_gpu,
        &pnorm_gpu,
        &cnorm_gpu,
        &assign_gpu,
        n,
        n_centroids,
        dim_padded,
        &dist,
    );
    let assignments: Vec<usize> = assign_gpu
        .read(&client)?
        .into_iter()
        .map(|x| x as usize)
        .collect();

    let final_cents = cent_gpu.clone().read(&client)?;
    let centroid_mat = Mat::from_fn(n_centroids, dim, |i, j| final_cents[i * dim_padded + j]);

    if verbose {
        println!(
            "Finished GPU-accelerated k-means in {:.2?}.",
            start.elapsed()
        );
    }

    Ok((centroid_mat, assignments))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use ann_search_rs::utils::dist::compute_l2_norm;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    // Wraps flash_assign_device into a Vec<usize> result for test convenience.
    fn run_assign(
        data: &[f32],
        dim: usize,
        n: usize,
        cents: &[f32],
        k: usize,
        metric: &Dist,
        device: &WgpuDevice,
    ) -> Vec<usize> {
        let client = WgpuRuntime::client(device);
        let data_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(data, vec![n, dim], &client);
        let cent_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(cents, vec![k, dim], &client);
        let assign_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n], &client);

        let pnorm_gpu = if *metric == Dist::Cosine {
            let v: Vec<f32> = (0..n)
                .map(|i| compute_l2_norm(&data[i * dim..(i + 1) * dim]))
                .collect();
            GpuTensor::<WgpuRuntime, f32>::from_slice(&v, vec![n], &client)
        } else {
            GpuTensor::<WgpuRuntime, f32>::from_slice(&[1.0f32], vec![1], &client)
        };
        let cnorm_gpu = if *metric == Dist::Cosine {
            let v: Vec<f32> = (0..k)
                .map(|c| compute_l2_norm(&cents[c * dim..(c + 1) * dim]))
                .collect();
            GpuTensor::<WgpuRuntime, f32>::from_slice(&v, vec![k], &client)
        } else {
            GpuTensor::<WgpuRuntime, f32>::from_slice(&[1.0f32], vec![1], &client)
        };

        flash_assign_device(
            &client,
            &data_gpu,
            &cent_gpu,
            &pnorm_gpu,
            &cnorm_gpu,
            &assign_gpu,
            n,
            k,
            dim,
            metric,
        );

        assign_gpu
            .read(&client)
            .unwrap()
            .into_iter()
            .map(|x| x as usize)
            .collect()
    }

    // Wraps build_csr_gpu_privatised + segmented_update.
    fn run_update(
        data: &[f32],
        assignments: &[u32],
        init_cents: &[f32],
        n: usize,
        k: usize,
        dim: usize,
        device: &WgpuDevice,
    ) -> Vec<f32> {
        let client = WgpuRuntime::client(device);
        let cube_count = 64usize;

        let data_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(data, vec![n, dim], &client);
        let assign_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(assignments, vec![n], &client);
        let cent_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(init_cents, vec![k, dim], &client);
        let privatised_counts = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0u32; cube_count * k],
            vec![cube_count * k],
            &client,
        );
        let counts = GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; k], vec![k], &client);
        let offsets =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; k + 1], vec![k + 1], &client);
        let all_indices = GpuTensor::<WgpuRuntime, u32>::empty(vec![n], &client);

        build_csr_gpu_privatised::<WgpuRuntime>(
            &assign_gpu,
            n,
            k,
            cube_count,
            &privatised_counts,
            &counts,
            &offsets,
            &all_indices,
            &client,
        );
        segmented_update::<WgpuRuntime, f32>(
            &data_gpu,
            &all_indices,
            &offsets,
            &cent_gpu,
            k,
            dim,
            &client,
        );

        cent_gpu.read(&client).unwrap()
    }

    /////////////
    // Helpers //
    /////////////

    fn cpu_assign_euclidean(
        data: &[f32],
        cents: &[f32],
        n: usize,
        k: usize,
        dim: usize,
    ) -> Vec<usize> {
        (0..n)
            .map(|i| {
                (0..k)
                    .map(|c| {
                        (0..dim)
                            .map(|j| (data[i * dim + j] - cents[c * dim + j]).powi(2))
                            .sum::<f32>()
                    })
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect()
    }

    fn cpu_assign_cosine(
        data: &[f32],
        cents: &[f32],
        n: usize,
        k: usize,
        dim: usize,
    ) -> Vec<usize> {
        (0..n)
            .map(|i| {
                let pnorm = compute_l2_norm(&data[i * dim..(i + 1) * dim]);
                (0..k)
                    .map(|c| {
                        let cnorm = compute_l2_norm(&cents[c * dim..(c + 1) * dim]);
                        let dot: f32 = (0..dim)
                            .map(|j| data[i * dim + j] * cents[c * dim + j])
                            .sum();
                        1.0 - dot / (pnorm * cnorm)
                    })
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect()
    }

    fn cpu_centroid_means(
        data: &[f32],
        assignments: &[usize],
        init: &[f32],
        n: usize,
        k: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..dim {
                sums[c * dim + j] += data[i * dim + j];
            }
        }
        let mut out = init.to_vec();
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    out[c * dim + j] = sums[c * dim + j] / counts[c] as f32;
                }
            }
        }
        out
    }

    //////////
    // Tests //
    //////////

    #[test]
    fn test_assign_euclidean() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (200, 16, 32);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.2)
            .collect();
        let cents: Vec<f32> = (0..k * dim)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.2)
            .collect();

        let got = run_assign(&data, dim, n, &cents, k, &Dist::SquaredEuclidean, &device);
        assert_eq!(got, cpu_assign_euclidean(&data, &cents, n, k, dim));
    }

    #[test]
    fn test_assign_cosine() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (120, 8, 32);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7 + 1) % 11) as f32 + 0.5)
            .collect();
        let cents: Vec<f32> = (0..k * dim)
            .map(|i| ((i * 13 + 3) % 17) as f32 + 0.5)
            .collect();

        let got = run_assign(&data, dim, n, &cents, k, &Dist::Cosine, &device);
        assert_eq!(got, cpu_assign_cosine(&data, &cents, n, k, dim));
    }

    // n not a multiple of the workgroup size to exercise the inactive-thread guard.
    #[test]
    fn test_assign_ragged_n() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (137, 5, 8);
        let data: Vec<f32> = (0..n * dim).map(|i| ((i * 5 + 2) % 13) as f32).collect();
        let cents: Vec<f32> = (0..k * dim).map(|i| ((i * 9 + 4) % 17) as f32).collect();

        let got = run_assign(&data, dim, n, &cents, k, &Dist::SquaredEuclidean, &device);
        assert_eq!(got.len(), n);
        assert_eq!(got, cpu_assign_euclidean(&data, &cents, n, k, dim));
    }

    #[test]
    fn test_csr_privatised_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);
        let assignments = vec![0u32, 1, 0, 2, 1, 0, 2, 2];
        let (n, k, cube_count) = (assignments.len(), 3usize, 64usize);

        let assign_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&assignments, vec![n], &client);
        let privatised_counts = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0u32; cube_count * k],
            vec![cube_count * k],
            &client,
        );
        let counts = GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; k], vec![k], &client);
        let offsets =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; k + 1], vec![k + 1], &client);
        let all_indices = GpuTensor::<WgpuRuntime, u32>::empty(vec![n], &client);

        build_csr_gpu_privatised::<WgpuRuntime>(
            &assign_gpu,
            n,
            k,
            cube_count,
            &privatised_counts,
            &counts,
            &offsets,
            &all_indices,
            &client,
        );

        let idx = all_indices.read(&client).unwrap();
        let off = offsets.read(&client).unwrap();

        let (cpu_idx, cpu_off) = ann_search_rs::utils::k_means_utils::build_csr_layout(
            assignments.iter().map(|&c| c as usize).collect(),
            n,
            k,
        );

        assert_eq!(off.iter().map(|&o| o as usize).collect::<Vec<_>>(), cpu_off);

        for c in 0..k {
            let s = off[c] as usize;
            let e = off[c + 1] as usize;
            let got: std::collections::HashSet<u32> = idx[s..e].iter().copied().collect();
            let want: std::collections::HashSet<u32> = cpu_idx[cpu_off[c]..cpu_off[c + 1]]
                .iter()
                .map(|&i| i as u32)
                .collect();
            assert_eq!(got, want, "cluster {} membership mismatch", c);
        }
    }

    #[test]
    fn test_segmented_update() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (300, 8, 32);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 11 + 5) % 23) as f32 * 0.3)
            .collect();
        let assignments: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
        let init = vec![-1.0f32; k * dim];

        let got = run_update(&data, &assignments, &init, n, k, dim, &device);
        let want = cpu_centroid_means(
            &data,
            &assignments.iter().map(|&c| c as usize).collect::<Vec<_>>(),
            &init,
            n,
            k,
            dim,
        );

        for j in 0..k * dim {
            assert!(
                (got[j] - want[j]).abs() < 1e-3,
                "elem {}: {} != {}",
                j,
                got[j],
                want[j]
            );
        }
    }

    // Cluster 1 gets no points; its centroid must survive untouched.
    #[test]
    fn test_segmented_update_empty_cluster() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (6, 3, 4);
        let data: Vec<f32> = (0..n * dim).map(|i| (i + 1) as f32).collect();
        let assignments = vec![0u32, 0, 2, 2, 0, 2];
        let init = vec![7.0f32; k * dim];

        let got = run_update(&data, &assignments, &init, n, k, dim, &device);

        for j in 0..dim {
            assert!(
                (got[dim + j] - 7.0).abs() < 1e-6,
                "empty cluster overwritten at dim {}",
                j
            );
        }
    }
}
