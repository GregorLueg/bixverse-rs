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
    /// * `fixed - Shall the algorithm be run for a fixed set of iterations
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
// Helpers //
/////////////

/// Pick the assign-kernel tile parameters from the (padded) dimensionality.
///
/// Returns `(rn, bk)`:
///
/// * `rn` - points each thread owns (register-blocking factor). Higher `rn`
///   means more arithmetic per shared-memory load (good), but each thread
///   holds `rn * dim_scalars` scalars in registers, so we taper it down as
///   `dim_scalars` grows to avoid register spills on modest GPUs. These
///   cut-offs are deliberately conservative starting points -- profile and
///   retune for your hardware.
/// * `bk` - centroids cached in shared memory per tile. The SMEM footprint is
///   `bk * dim_scalars * 4` bytes (f32). We target roughly 32 KiB and never go
///   below the previous fixed value of 16, so this is a strict improvement on
///   small/medium `dim` and a no-op on large `dim`.
///
/// `dim_scalars` here equals the padded `dim` (it is `dim_lines * LINE_SIZE`).
fn assign_tile_params(dim_scalars: usize) -> (usize, usize) {
    let d = dim_scalars.max(1);
    let rn = if d <= 64 {
        4
    } else if d <= 256 {
        2
    } else {
        1
    };
    // 8192 f32 == 32 KiB. clamp(16, 64) keeps us within a conservative SMEM
    // budget while never regressing below the original bk = 16.
    let bk = (8192 / d).clamp(16, 64);
    (rn, bk)
}

/// Online-argmin Euclidean assignment.
///
/// Each thread owns `rn` consecutive points (register blocking); centroids are
/// streamed through shared memory in tiles of `bk`. No N x K materialisation.
///
/// ### Params
///
/// * `data` - Data points `[n_samples, dim / N]` as `Vector<F, N>`
/// * `centroids` - Centroid vectors `[k, dim / N]` as `Vector<F, N>`
/// * `assignments` - Output assignment indices `[n_samples]`
/// * `n_samples` - Total number of data points
/// * `k` - Number of centroids
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
/// * `bk` - Number of centroids to cache per shared-memory tile (comptime)
/// * `rn` - Number of points each thread processes (comptime)
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X`
///   -> thread index; thread owns points `[thread_idx * rn .. + rn)`
#[cube(launch_unchecked)]
pub fn flash_assign_euclidean<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    centroids: &Tensor<Vector<F, N>>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] bk: usize,
    #[comptime] rn: usize,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;

    let thread_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    let p0 = thread_idx as usize * rn;

    let mut p = Array::<F>::new(rn * dim_scalars);
    for r in 0..rn {
        let pid = p0 + r;
        let p_base = if (pid as u32) < n_samples {
            pid * dim_lines
        } else {
            #[allow(clippy::useless_conversion)] // cubecl needs this
            0usize.into()
        };
        for i in 0..dim_lines {
            let pl = data[p_base + i];
            #[unroll]
            for lane in 0..lanes {
                p[r * dim_scalars + i * lanes + lane] = pl[lane];
            }
        }
    }

    let mut s_cent = SharedMemory::<F>::new(bk * dim_scalars);

    let mut best_dist = Array::<F>::new(rn);
    let mut best_idx = Array::<u32>::new(rn);
    for r in 0..rn {
        best_dist[r] = F::new(f32::MAX);
        best_idx[r] = 0u32;
    }

    #[allow(clippy::manual_div_ceil)]
    let n_tiles = (k + bk as u32 - 1u32) / bk as u32;
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_c0 = tile * bk as u32;

        let total_elems = bk * dim_scalars;
        let mut load_idx = tx;
        while load_idx < total_elems {
            let c_local = load_idx / dim_scalars;
            let elem = load_idx % dim_scalars;
            let c_global = tile_c0 + c_local as u32;
            if c_global < k {
                let line_idx = elem / lanes;
                let lane = elem % lanes;
                let cl = centroids[c_global as usize * dim_lines + line_idx];
                s_cent[load_idx] = cl[lane];
            } else {
                s_cent[load_idx] = F::new(0.0);
            }
            load_idx += wg;
        }
        sync_cube();

        let mut c_local = 0u32;
        while c_local < bk as u32 {
            let c_global = tile_c0 + c_local;
            if c_global < k {
                let cbase = c_local as usize * dim_scalars;
                let mut sum = Array::<F>::new(rn);
                for r in 0..rn {
                    sum[r] = F::new(0.0);
                }

                for e in 0..dim_scalars {
                    let cval = s_cent[cbase + e];
                    for r in 0..rn {
                        let diff = p[r * dim_scalars + e] - cval;
                        let acc = sum[r];
                        sum[r] = acc + diff * diff;
                    }
                }
                for r in 0..rn {
                    let s = sum[r];
                    if s < best_dist[r] {
                        best_dist[r] = s;
                        best_idx[r] = c_global;
                    }
                }
            }
            c_local += 1u32;
        }
        sync_cube();

        tile += 1u32;
    }

    for r in 0..rn {
        let pid = p0 + r;
        if (pid as u32) < n_samples {
            assignments[pid] = best_idx[r];
        }
    }
}

/// Online-argmin cosine assignment.
///
/// Minimises `1 - dot(x, c) / (||x|| * ||c||)` using precomputed norms.
/// Centroids are streamed through shared memory in tiles of `bk`.
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
/// * `bk` - Number of centroids to cache per shared-memory tile (comptime)
/// * `rn` - Number of points each thread processes (comptime)
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X`
///   -> thread index; thread owns points `[thread_idx * rn .. + rn)`
#[cube(launch_unchecked)]
pub fn flash_assign_cosine<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    centroids: &Tensor<Vector<F, N>>,
    point_norms: &Tensor<F>,
    centroid_norms: &Tensor<F>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] bk: usize,
    #[comptime] rn: usize,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;

    let thread_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    let p0 = thread_idx as usize * rn;

    let mut p = Array::<F>::new(rn * dim_scalars);
    let mut pnorm = Array::<F>::new(rn);
    for r in 0..rn {
        let pid = p0 + r;
        let safe = if (pid as u32) < n_samples {
            pid
        } else {
            #[allow(clippy::useless_conversion)] // cubecl needs this
            0usize.into()
        };
        let p_base = safe * dim_lines;
        for i in 0..dim_lines {
            let pl = data[p_base + i];
            #[unroll]
            for lane in 0..lanes {
                p[r * dim_scalars + i * lanes + lane] = pl[lane];
            }
        }
        pnorm[r] = point_norms[safe];
    }

    let mut s_cent = SharedMemory::<F>::new(bk * dim_scalars);

    let mut best_dist = Array::<F>::new(rn);
    let mut best_idx = Array::<u32>::new(rn);
    for r in 0..rn {
        best_dist[r] = F::new(f32::MAX);
        best_idx[r] = 0u32;
    }

    #[allow(clippy::manual_div_ceil)]
    let n_tiles = (k + bk as u32 - 1u32) / bk as u32;
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_c0 = tile * bk as u32;

        let total_elems = bk * dim_scalars;
        let mut load_idx = tx;
        while load_idx < total_elems {
            let c_local = load_idx / dim_scalars;
            let elem = load_idx % dim_scalars;
            let c_global = tile_c0 + c_local as u32;
            if c_global < k {
                let line_idx = elem / lanes;
                let lane = elem % lanes;
                let cl = centroids[c_global as usize * dim_lines + line_idx];
                s_cent[load_idx] = cl[lane];
            } else {
                s_cent[load_idx] = F::new(0.0);
            }
            load_idx += wg;
        }
        sync_cube();

        let mut c_local = 0u32;
        while c_local < bk as u32 {
            let c_global = tile_c0 + c_local;
            if c_global < k {
                let cbase = c_local as usize * dim_scalars;
                let mut dot = Array::<F>::new(rn);
                for r in 0..rn {
                    dot[r] = F::new(0.0);
                }
                for e in 0..dim_scalars {
                    let cval = s_cent[cbase + e];
                    for r in 0..rn {
                        let acc = dot[r];
                        dot[r] = acc + p[r * dim_scalars + e] * cval;
                    }
                }
                let cnorm = centroid_norms[c_global as usize];
                for r in 0..rn {
                    let dist = F::new(1.0) - dot[r] / (pnorm[r] * cnorm);
                    if dist < best_dist[r] {
                        best_dist[r] = dist;
                        best_idx[r] = c_global;
                    }
                }
            }
            c_local += 1u32;
        }
        sync_cube();

        tile += 1u32;
    }

    for r in 0..rn {
        let pid = p0 + r;
        if (pid as u32) < n_samples {
            assignments[pid] = best_idx[r];
        }
    }
}

/// Launch FlashAssign and return hard assignments.
///
/// `data` and `centroids` must have `dim` already padded to a multiple of
/// `LINE_SIZE`. Norms for cosine distance are computed on the host before
/// upload.
///
/// ### Params
///
/// * `data` - Flattened data points `[n * dim]`
/// * `dim` - Embedding dimensionality (must be divisible by `LINE_SIZE`)
/// * `n` - Number of data points
/// * `centroids` - Flattened centroid vectors `[k * dim]`
/// * `k` - Number of centroids
/// * `metric` - Distance metric (`SquaredEuclidean` or `Cosine`)
/// * `device` - CubeCL runtime device
///
/// ### Returns
///
/// Hard assignment index into `centroids` for each data point
pub fn flash_assign<T, R>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &[T],
    k: usize,
    metric: &Dist,
    device: R::Device,
) -> Result<Vec<usize>, BixverseErrors>
where
    R: Runtime,
    T: Float + Sum + cubecl::CubeElement + num_traits::Float + num_traits::FromPrimitive,
{
    let client = R::client(&device);
    let vec_size = LINE_SIZE;
    let dim_lines = dim / vec_size;
    // `dim` is the padded dim here, so dim == dim_scalars.
    let (rn, bk) = assign_tile_params(dim);

    let data_gpu = GpuTensor::<R, T>::from_slice(data, vec![n, dim], &client);
    let cent_gpu = GpuTensor::<R, T>::from_slice(centroids, vec![k, dim], &client);
    let assign_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);

    // Each thread now owns `rn` points, so we launch ceil(n / rn) threads.
    let n_threads = n.div_ceil(rn);
    let (gx, gy) = grid_2d((n_threads as u32).div_ceil(WORKGROUP_SIZE_X));
    let count = CubeCount::Static(gx, gy, 1);
    let cdim = CubeDim::new_2d(WORKGROUP_SIZE_X, 1);

    match *metric {
        Dist::SquaredEuclidean => unsafe {
            flash_assign_euclidean::launch_unchecked::<T, R>(
                &client,
                count,
                cdim,
                vec_size,
                data_gpu.into_tensor_arg(),
                cent_gpu.into_tensor_arg(),
                assign_gpu.clone().into_tensor_arg(),
                n as u32,
                k as u32,
                dim_lines,
                bk,
                rn,
            );
        },
        Dist::Cosine => {
            let pnorms: Vec<T> = (0..n)
                .map(|i| {
                    let s = &data[i * dim..(i + 1) * dim];
                    <T as num_traits::Float>::sqrt(
                        s.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b),
                    )
                })
                .collect();
            let cnorms: Vec<T> = (0..k)
                .map(|c| {
                    let s = &centroids[c * dim..(c + 1) * dim];
                    <T as num_traits::Float>::sqrt(
                        s.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b),
                    )
                })
                .collect();
            let pnorm_gpu = GpuTensor::<R, T>::from_slice(&pnorms, vec![n], &client);
            let cnorm_gpu = GpuTensor::<R, T>::from_slice(&cnorms, vec![k], &client);
            unsafe {
                flash_assign_cosine::launch_unchecked::<T, R>(
                    &client,
                    count,
                    cdim,
                    vec_size,
                    data_gpu.into_tensor_arg(),
                    cent_gpu.into_tensor_arg(),
                    pnorm_gpu.into_tensor_arg(),
                    cnorm_gpu.into_tensor_arg(),
                    assign_gpu.clone().into_tensor_arg(),
                    n as u32,
                    k as u32,
                    dim_lines,
                    bk,
                    rn,
                );
            }
        }
        Dist::Manhattan => unreachable!(),
    }

    let res = assign_gpu.read(&client)?;

    Ok(res.into_iter().map(|v| v as usize).collect())
}

/// Histogram of cluster sizes via atomic increments.
///
/// One thread per point. `counts` must be zero-initialised before launch.
///
/// ### Params
///
/// * `assignments` - Hard assignment indices `[n]`, one per point
/// * `counts` - Atomic cluster size counters `[k]`, incremented in place
/// * `n` - Total number of data points
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> point index
#[cube(launch_unchecked)]
pub fn histogram_clusters(assignments: &Tensor<u32>, counts: &mut Tensor<Atomic<u32>>, n: u32) {
    let i = ABSOLUTE_POS_X;
    if i >= n {
        terminate!();
    }
    let c = assignments[i as usize];
    Atomic::fetch_add(&counts[c as usize], 1u32);
}

/// Exclusive prefix sum of cluster counts into a `k+1` offset array.
///
/// Single-thread serial scan launched with one cube of one thread. Also
/// writes the per-cluster running write cursors used by `scatter_csr`.
/// Serial cost is negligible because k is small relative to n.
///
/// ### Params
///
/// * `counts` - Cluster sizes `[k]` produced by `histogram_clusters`
/// * `offsets` - Output exclusive prefix sums `[k + 1]`; `offsets[k]` equals
///   the total number of points
/// * `cursor` - Output per-cluster write cursors `[k]`, seeded to
///   `offsets[0..k]` for use by `scatter_csr`
/// * `k` - Number of clusters
///
/// ### Grid mapping
///
/// * Single cube, single thread (`UNIT_POS_X == 0`)
#[cube(launch_unchecked)]
pub fn exclusive_scan_offsets(
    counts: &Tensor<u32>,
    offsets: &mut Tensor<u32>,
    cursor: &mut Tensor<u32>,
    k: u32,
) {
    if UNIT_POS_X == 0u32 {
        offsets[0] = 0u32;
        let mut acc = 0u32;
        let mut c = 0u32;
        while c < k {
            cursor[c as usize] = acc;
            acc += counts[c as usize];
            offsets[(c + 1u32) as usize] = acc;
            c += 1u32;
        }
    }
}

/// Scatter point indices into CSR order via atomic slot claims.
///
/// One thread per point. Each thread atomically increments its cluster's
/// cursor to claim the next write slot, then stores its point index there.
/// Within-segment order is non-deterministic, which is acceptable because
/// the downstream centroid summation is order-independent.
///
/// ### Params
///
/// * `assignments` - Hard assignment indices `[n]`, one per point
/// * `cursor` - Per-cluster running write positions `[k]`, seeded by
///   `exclusive_scan_offsets` and atomically advanced in place
/// * `all_indices` - Output point indices in CSR order `[n]`
/// * `n` - Total number of data points
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> point index
#[cube(launch_unchecked)]
pub fn scatter_csr(
    assignments: &Tensor<u32>,
    cursor: &mut Tensor<Atomic<u32>>,
    all_indices: &mut Tensor<u32>,
    n: u32,
) {
    let i = ABSOLUTE_POS_X;
    if i >= n {
        terminate!();
    }
    let c = assignments[i as usize];
    let pos = Atomic::fetch_add(&cursor[c as usize], 1u32);
    all_indices[pos as usize] = i;
}

/// Build a CSR layout from hard assignments entirely on device.
///
/// Runs three kernels in sequence: `histogram_clusters` to count cluster sizes,
/// `exclusive_scan_offsets` to compute the offset array and seed write cursors,
/// and `scatter_csr` to place point indices into their cluster segments. No
/// host readback occurs between stages.
///
/// ### Params
///
/// * `assignments` - Hard assignment indices `[n]` already on device
/// * `n` - Number of data points
/// * `k` - Number of clusters
/// * `client` - CubeCL compute client for the target device
///
/// ### Returns
///
/// Tuple of `(all_indices, offsets)` where `all_indices` is `[n]` containing
/// point indices in CSR order and `offsets` is `[k + 1]` with exclusive
/// prefix sums; cluster `c` occupies `all_indices[offsets[c]..offsets[c+1]]`
pub fn build_csr_gpu<R>(
    assignments: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> (GpuTensor<R, u32>, GpuTensor<R, u32>)
where
    R: Runtime,
{
    let counts = GpuTensor::<R, u32>::from_slice(&vec![0u32; k], vec![k], client);
    let offsets = GpuTensor::<R, u32>::from_slice(&vec![0u32; k + 1], vec![k + 1], client);
    let cursor = GpuTensor::<R, u32>::from_slice(&vec![0u32; k], vec![k], client);
    let all_indices = GpuTensor::<R, u32>::empty(vec![n], client);

    let (gx, gy) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));

    let start = Instant::now();

    unsafe {
        histogram_clusters::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            assignments.clone().into_tensor_arg(),
            counts.clone().into_tensor_arg(),
            n as u32,
        );
    }

    client.sync();
    println!("  histogram: {:.2?}", start.elapsed());

    unsafe {
        exclusive_scan_offsets::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            counts.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            cursor.clone().into_tensor_arg(),
            k as u32,
        );
    }

    client.sync();
    println!("  scan: {:.2?}", start.elapsed());

    unsafe {
        scatter_csr::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            assignments.clone().into_tensor_arg(),
            cursor.clone().into_tensor_arg(),
            all_indices.clone().into_tensor_arg(),
            n as u32,
        );
    }

    client.sync();
    println!("  scatter: {:.2?}", start.elapsed());

    (all_indices, offsets)
}

#[cube(launch_unchecked)]
pub fn filter_centroid_update<F: Float>(
    data: &Tensor<F>,
    assignments: &Tensor<u32>,
    centroids: &mut Tensor<F>,
    n: u32,
    k: u32,
    #[comptime] dim: usize,
    #[comptime] dim_per_thread: usize,
) {
    let cluster = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if cluster >= k {
        terminate!();
    }

    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;
    let cent_base = cluster as usize * dim;

    let mut acc = Array::<F>::new(dim_per_thread);
    for r in 0..dim_per_thread {
        acc[r] = F::new(0.0);
    }

    let mut count: u32 = 0u32;
    let mut i = 0u32;
    while i < n {
        if assignments[i as usize] == cluster {
            count += 1u32;
            let row = i as usize * dim;
            for r in 0..dim_per_thread {
                let e = tx + r * wg;
                if e < dim {
                    acc[r] += data[row + e];
                }
            }
        }
        i += 1u32;
    }

    if count > 0u32 {
        let inv_count = F::new(1.0) / F::cast_from(count);
        for r in 0..dim_per_thread {
            let e = tx + r * wg;
            if e < dim {
                centroids[cent_base + e] = acc[r] * inv_count;
            }
        }
    }
}

pub fn filter_update<R, T>(
    data: &GpuTensor<R, T>,
    assignments: &GpuTensor<R, u32>,
    centroids: &GpuTensor<R, T>,
    n: usize,
    k: usize,
    dim: usize,
    client: &ComputeClient<R>,
) where
    R: Runtime,
    T: Float + cubecl::CubeElement + num_traits::Float,
{
    let dim_per_thread = dim.div_ceil(WORKGROUP_SIZE_X as usize);
    let (gx, gy) = grid_2d(k as u32);
    unsafe {
        filter_centroid_update::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            data.clone().into_tensor_arg(),
            assignments.clone().into_tensor_arg(),
            centroids.clone().into_tensor_arg(),
            n as u32,
            k as u32,
            dim,
            dim_per_thread,
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
/// * `all_indices` - Point indices in CSR order `[n]` from `build_csr_gpu`
/// * `offsets` - Exclusive prefix sums `[k + 1]` from `build_csr_gpu`;
///   cluster `c` occupies `all_indices[offsets[c]..offsets[c+1]]`
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

/// Force host to wait for all GPU work submitted so far. cubecl's `sync` only
/// flushes; a buffer readback is the only thing that guarantees a real fence.
fn gpu_fence<R: Runtime>(client: &ComputeClient<R>, scratch: &GpuTensor<R, u32>) {
    let _ = scratch.clone().read(client).unwrap();
}

/// Recompute centroids in place from a CSR layout.
///
/// Wraps `segmented_centroid_update`. Empty clusters retain their current
/// value.
///
/// ### Params
///
/// * `data` - Data points already on device `[n, dim]`
/// * `all_indices` - Point indices in CSR order `[n]` from `build_csr_gpu`
/// * `offsets` - Exclusive prefix sums `[k + 1]` from `build_csr_gpu`
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

/// Launch FlashAssign into a device-resident assignment buffer.
///
/// Internal to the driver loop. Dispatches either `flash_assign_euclidean` or
/// `flash_assign_cosine` depending on `metric`. Assumes `dim` is already
/// padded to a multiple of `LINE_SIZE` and all tensors reside on `client`'s
/// device. `bk` and `rn` must be derived from the same `dim` via
/// `assign_tile_params` to keep tile and register budgets consistent.
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
/// * `bk` - Number of centroids to cache per shared-memory tile in FlashAssign
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
    bk: usize,
) where
    R: Runtime,
    T: Float + Sum + cubecl::CubeElement + num_traits::Float + num_traits::FromPrimitive,
{
    let vec_size = LINE_SIZE;
    let dim_lines = dim / vec_size;
    // `bk` is supplied by the caller (computed once via assign_tile_params);
    // rn is derived here from the same dim so the two always agree.
    let (rn, _bk_unused) = assign_tile_params(dim);
    let n_threads = n.div_ceil(rn);
    let (gx, gy) = grid_2d((n_threads as u32).div_ceil(WORKGROUP_SIZE_X));
    let count = CubeCount::Static(gx, gy, 1);
    let cdim = CubeDim::new_1d(WORKGROUP_SIZE_X);

    match *metric {
        Dist::SquaredEuclidean => unsafe {
            flash_assign_euclidean::launch_unchecked::<T, R>(
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
                bk,
                rn,
            );
        },
        Dist::Cosine => unsafe {
            flash_assign_cosine::launch_unchecked::<T, R>(
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
                bk,
                rn,
            );
        },
        Dist::Manhattan => unreachable!(),
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

#[cube(launch_unchecked)]
pub fn histogram_clusters_privatized(
    assignments: &Tensor<u32>,
    privatized_counts: &mut Tensor<Atomic<u32>>,
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
        Atomic::fetch_and(&privatized_counts[idx as usize], 0u32);
        c += wg;
    }
    sync_cube(); // Enforce a barrier so no thread counts until everything is zeroed

    // 2. Standard Privatized Counting Pass
    let total_threads = wg * CUBE_COUNT_X;
    let mut i = r * wg + tx;
    while i < n {
        let chunk_c = assignments[i as usize];
        let idx = r * k + chunk_c;
        Atomic::fetch_add(&privatized_counts[idx as usize], 1u32);
        i += total_threads;
    }
}

#[cube(launch_unchecked)]
pub fn scan_columns_and_sum(
    privatized_counts: &mut Tensor<u32>,
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
        let val = privatized_counts[idx as usize];
        privatized_counts[idx as usize] = acc; // Overwrite with local block prefix sum
        acc += val;
        r += 1u32;
    }
    counts[c as usize] = acc;
}

#[cube(launch_unchecked)]
pub fn merge_offsets_to_cursors(
    privatized_counts: &mut Tensor<u32>,
    offsets: &Tensor<u32>,
    k: u32,
    cube_count: u32,
) {
    let idx = ABSOLUTE_POS_X as usize;
    let total_elements = (cube_count * k) as usize;
    if idx < total_elements {
        let c = (idx as u32) % k;
        privatized_counts[idx] += offsets[c as usize];
    }
}

#[cube(launch_unchecked)]
pub fn scatter_csr_privatized(
    assignments: &Tensor<u32>,
    privatized_cursors: &mut Tensor<Atomic<u32>>,
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
        let pos = Atomic::fetch_add(&privatized_cursors[idx as usize], 1u32);
        all_indices[pos as usize] = i;
        i += total_threads;
    }
}

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

pub fn build_csr_gpu_privatized<R>(
    assignments: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
    cube_count: usize,
    privatized_counts: &GpuTensor<R, u32>,
    counts: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    all_indices: &GpuTensor<R, u32>,
    client: &ComputeClient<R>,
) where
    R: Runtime,
{
    unsafe {
        // Step 1: Privatized Histogram (Now handles its own zeroing)
        histogram_clusters_privatized::launch_unchecked::<R>(
            client,
            CubeCount::Static(cube_count as u32, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            assignments.clone().into_tensor_arg(),
            privatized_counts.clone().into_tensor_arg(),
            n as u32,
            k as u32,
        );

        // Step 2: Parallel Column Scan
        scan_columns_and_sum::launch_unchecked::<R>(
            client,
            CubeCount::Static(k.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            privatized_counts.clone().into_tensor_arg(),
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
            privatized_counts.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            k as u32,
            cube_count as u32,
        );

        // Step 5: Contention-Free Index Scatter
        scatter_csr_privatized::launch_unchecked::<R>(
            client,
            CubeCount::Static(cube_count as u32, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            assignments.clone().into_tensor_arg(),
            privatized_counts.clone().into_tensor_arg(),
            all_indices.clone().into_tensor_arg(),
            n as u32,
            k as u32,
        );
    }
}

/// One Lloyd's iteration entirely on device: assign, rebuild CSR, update
/// centroids.
///
/// Runs `flash_assign_device` to compute hard assignments, `build_csr_gpu` to
/// sort point indices into cluster segments, and `segmented_update` to recompute
/// each centroid as the mean of its assigned points. For cosine distance,
/// centroid norms are refreshed in place via `centroid_norms_l2` so that
/// `cnorm_gpu` is consistent with `cent_gpu` on exit.
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
/// * `bk` - Number of centroids to cache per shared-memory tile in FlashAssign
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
    bk: usize,
    cube_count: usize,
    privatized_counts: &GpuTensor<R, u32>,
    counts: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    all_indices: &GpuTensor<R, u32>,
    fence_scratch: &GpuTensor<R, u32>,
) where
    R: Runtime,
    T: Float + Sum + cubecl::CubeElement + num_traits::Float + num_traits::FromPrimitive,
{
    let start = Instant::now();

    flash_assign_device(
        client, data_gpu, cent_gpu, pnorm_gpu, cnorm_gpu, assign_gpu, n, k, dim, metric, bk,
    );

    gpu_fence(client, fence_scratch);

    println!("Flash assign took {:.2?}", start.elapsed());

    build_csr_gpu_privatized::<R>(
        assign_gpu,
        n,
        k,
        cube_count,
        privatized_counts,
        counts,
        offsets,
        all_indices,
        client,
    );

    gpu_fence(client, fence_scratch);
    println!("CSR GPU privatised {:.2?}", start.elapsed());

    segmented_update::<R, T>(data_gpu, &all_indices, &offsets, cent_gpu, k, dim, client);

    gpu_fence(client, fence_scratch);
    println!("Segmented update in {:.2?}", start.elapsed());

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

/// Generate k-means clusters on the GPU
///
/// Device-resident Lloyd's loop: FlashAssign for assignment, counting-sort CSR
/// plus segmented reduction for the update. Initialisation runs on the host
/// (reusing `fast_random_init` / `kmeans_parallel_init`) and is uploaded once.
/// Convergence is detected via assignment stability: the loop stops once the
/// number of points changing cluster between iterations drops to a small floor.
/// This terminates the cosine path too, where fp32 renormalisation noise keeps
/// per-centroid drift pinned at a tiny non-zero value indefinitely. A small
/// non-zero floor absorbs near-equidistant points that flip between equally
/// good centroids without stalling termination.
///
/// ### Params
///
/// * `data` - The data to cluster. Samples x features
/// * `dist` - Distance metric, `"euclidean"` or `"cosine"`. Unknown strings
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

    let fence_scratch = GpuTensor::<R, u32>::from_slice(&[0u32], vec![1], &client);

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

    // bk (SMEM centroid tile) is chosen from the padded dim; rn (register
    // blocking) is derived from the same dim inside flash_assign_device, so
    // the two stay consistent. See assign_tile_params for the budget logic.
    let (_rn, bk) = assign_tile_params(dim_padded);

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

    let privatized_counts = GpuTensor::<R, u32>::empty(vec![cube_count, n_centroids], &client);
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
                bk,
                cube_count,
                &privatized_counts,
                &counts,
                &offsets,
                &all_indices,
                &fence_scratch,
            );
        }
    } else {
        // a change flor
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
                bk,
                cube_count,
                &privatized_counts,
                &counts,
                &offsets,
                &all_indices,
                &fence_scratch,
            );

            // Zeroed single-element atomic counter, recreated per iter (same
            // pattern as the CSR counters in build_csr_gpu).
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
        bk,
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
    use cubecl::wgpu::WgpuDevice;
    use cubecl::wgpu::WgpuRuntime;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::wgpu::WgpuRuntime::client(&device);
        }));
        result.ok().map(|_| device)
    }

    /////////////
    // Helpers //
    /////////////

    /// CPU reference: hard argmin under squared Euclidean, lowest index on ties.
    fn cpu_assign_euclidean(
        data: &[f32],
        cents: &[f32],
        n: usize,
        k: usize,
        dim: usize,
    ) -> Vec<usize> {
        (0..n)
            .map(|i| {
                let mut best = 0usize;
                let mut best_d = f32::MAX;
                for c in 0..k {
                    let mut sum = 0.0f32;
                    for j in 0..dim {
                        let diff = data[i * dim + j] - cents[c * dim + j];
                        sum += diff * diff;
                    }
                    if sum < best_d {
                        best_d = sum;
                        best = c;
                    }
                }
                best
            })
            .collect()
    }

    /// CPU reference: hard argmin under cosine distance, lowest index on ties.
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
                let mut best = 0usize;
                let mut best_d = f32::MAX;
                for c in 0..k {
                    let cnorm = compute_l2_norm(&cents[c * dim..(c + 1) * dim]);
                    let mut dot = 0.0f32;
                    for j in 0..dim {
                        dot += data[i * dim + j] * cents[c * dim + j];
                    }
                    let d = 1.0 - dot / (pnorm * cnorm);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                best
            })
            .collect()
    }

    fn assert_valid_euclidean_assignment(
        data: &[f32],
        cents: &[f32],
        got: &[usize],
        n: usize,
        k: usize,
        dim: usize,
        tol: f32,
    ) {
        for i in 0..n {
            let mut min_d = f32::MAX;
            for c in 0..k {
                let mut sum = 0.0f32;
                for j in 0..dim {
                    let diff = data[i * dim + j] - cents[c * dim + j];
                    sum += diff * diff;
                }
                if sum < min_d {
                    min_d = sum;
                }
            }
            let chosen = got[i];
            let mut chosen_d = 0.0f32;
            for j in 0..dim {
                let diff = data[i * dim + j] - cents[chosen * dim + j];
                chosen_d += diff * diff;
            }
            assert!(
                (chosen_d - min_d).abs() <= tol * (1.0 + min_d.abs()),
                "point {} chose cluster {} with dist {}, true min {}",
                i,
                chosen,
                chosen_d,
                min_d
            );
        }
    }

    /// CPU reference: per-cluster mean, empty clusters keep their init value.
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
        let data_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(data, vec![n, dim], &client);
        let assign_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(assignments, vec![n], &client);
        let cent_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(init_cents, vec![k, dim], &client);

        let (idx_gpu, off_gpu) = build_csr_gpu::<WgpuRuntime>(&assign_gpu, n, k, &client);
        segmented_update::<WgpuRuntime, f32>(
            &data_gpu, &idx_gpu, &off_gpu, &cent_gpu, k, dim, &client,
        );

        cent_gpu.read(&client).unwrap()
    }

    //////////////////
    // Actual tests //
    //////////////////

    #[test]
    fn test_flash_assign_euclidean_dim8() {
        let Some(device) = try_device() else { return };
        let n = 50usize;
        let k = 6usize;
        let dim = 8usize;

        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1)
            .collect();
        let cents: Vec<f32> = (0..k * dim)
            .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
            .collect();

        let got = flash_assign::<f32, WgpuRuntime>(
            &data,
            dim,
            n,
            &cents,
            k,
            &Dist::SquaredEuclidean,
            device,
        )
        .unwrap();

        assert_valid_euclidean_assignment(&data, &cents, &got, n, k, dim, 1e-4)
    }

    #[test]
    fn test_flash_assign_euclidean_dim32() {
        let Some(device) = try_device() else { return };
        let n = 200usize;
        let k = 16usize;
        let dim = 32usize;

        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.2)
            .collect();
        let cents: Vec<f32> = (0..k * dim)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.2)
            .collect();

        let got = flash_assign::<f32, WgpuRuntime>(
            &data,
            dim,
            n,
            &cents,
            k,
            &Dist::SquaredEuclidean,
            device,
        )
        .unwrap();
        let want = cpu_assign_euclidean(&data, &cents, n, k, dim);

        assert_eq!(got, want);
    }

    #[test]
    fn test_flash_assign_cosine_dim32() {
        let Some(device) = try_device() else { return };
        let n = 120usize;
        let k = 8usize;
        let dim = 32usize;

        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7 + 1) % 11) as f32 + 0.5)
            .collect();
        let cents: Vec<f32> = (0..k * dim)
            .map(|i| ((i * 13 + 3) % 17) as f32 + 0.5)
            .collect();

        let got = flash_assign::<f32, WgpuRuntime>(&data, dim, n, &cents, k, &Dist::Cosine, device)
            .unwrap();
        let want = cpu_assign_cosine(&data, &cents, n, k, dim);

        assert_eq!(got, want);
    }

    /// n deliberately not a multiple of WORKGROUP_SIZE_X to exercise the
    /// inactive-thread guard.
    #[test]
    fn test_flash_assign_ragged_n() {
        let Some(device) = try_device() else { return };
        let n = 137usize;
        let k = 5usize;
        let dim = 8usize;

        let data: Vec<f32> = (0..n * dim).map(|i| ((i * 5 + 2) % 13) as f32).collect();
        let cents: Vec<f32> = (0..k * dim).map(|i| ((i * 9 + 4) % 17) as f32).collect();

        let got = flash_assign::<f32, WgpuRuntime>(
            &data,
            dim,
            n,
            &cents,
            k,
            &Dist::SquaredEuclidean,
            device,
        )
        .unwrap();
        let want = cpu_assign_euclidean(&data, &cents, n, k, dim);

        assert_eq!(got.len(), n);
        assert_eq!(got, want);
    }

    /// Equidistant centroids: the kernel's strict `<` must pick the lower index,
    /// matching the CPU reference.
    #[test]
    fn test_flash_assign_tie_breaks_low() {
        let Some(device) = try_device() else { return };
        let dim = 4usize;
        // Point at [1,0,0,0]; c0=[0,0,0,0], c1 also =[0,0,0,0] -> tie -> c0.
        let data = vec![1.0f32, 0.0, 0.0, 0.0];
        let cents = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let got = flash_assign::<f32, WgpuRuntime>(
            &data,
            dim,
            1,
            &cents,
            2,
            &Dist::SquaredEuclidean,
            device,
        )
        .unwrap();

        assert_eq!(got, vec![0]);
    }

    /// Stresses the register-blocking path directly: n is deliberately not a
    /// multiple of the rn factor (so the per-thread tail guard fires) and k
    /// spans several bk shared-memory tiles. Must still match the CPU argmin.
    #[test]
    fn test_flash_assign_register_blocking_ragged() {
        let Some(device) = try_device() else { return };
        let n = 1023usize; // coprime-ish with any rn in {1,2,4}
        let k = 200usize; // spans multiple bk tiles
        let dim = 32usize; // rn = 4 for this dim

        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 31 + 7) % 97) as f32 * 0.05)
            .collect();
        let cents: Vec<f32> = (0..k * dim)
            .map(|i| ((i * 19 + 11) % 89) as f32 * 0.05)
            .collect();

        let got = flash_assign::<f32, WgpuRuntime>(
            &data,
            dim,
            n,
            &cents,
            k,
            &Dist::SquaredEuclidean,
            device,
        )
        .unwrap();
        let want = cpu_assign_euclidean(&data, &cents, n, k, dim);

        assert_eq!(got.len(), n);
        assert_eq!(got, want);
    }

    #[test]
    fn test_build_csr_gpu_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let assignments = vec![0u32, 1, 0, 2, 1, 0, 2, 2];
        let n = assignments.len();
        let k = 3usize;

        let assign_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&assignments, vec![n], &client);
        let (idx_gpu, off_gpu) = build_csr_gpu::<WgpuRuntime>(&assign_gpu, n, k, &client);

        let idx = idx_gpu.read(&client).unwrap();
        let off = off_gpu.read(&client).unwrap();

        // CPU ground truth via the existing layout builder from
        // `ann-search-rs`
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
    fn test_segmented_update_dim4() {
        let Some(device) = try_device() else { return };
        let n = 12usize;
        let k = 3usize;
        let dim = 4usize;

        let data: Vec<f32> = (0..n * dim).map(|i| ((i * 7 + 1) % 13) as f32).collect();
        let assignments = vec![0u32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let init = vec![99.0f32; k * dim];

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
                (got[j] - want[j]).abs() < 1e-4,
                "elem {}: {} != {}",
                j,
                got[j],
                want[j]
            );
        }
    }

    #[test]
    fn test_segmented_update_dim32() {
        let Some(device) = try_device() else { return };
        let n = 300usize;
        let k = 8usize;
        let dim = 32usize;

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

    /// Cluster 1 gets no points; its centroid must survive untouched.
    #[test]
    fn test_segmented_update_empty_cluster() {
        let Some(device) = try_device() else { return };
        let n = 6usize;
        let k = 3usize;
        let dim = 4usize;

        let data: Vec<f32> = (0..n * dim).map(|i| (i + 1) as f32).collect();
        let assignments = vec![0u32, 0, 2, 2, 0, 2]; // nothing in cluster 1
        let init = vec![7.0f32; k * dim];

        let got = run_update(&data, &assignments, &init, n, k, dim, &device);

        for j in 0..dim {
            assert!(
                (got[dim + j] - 7.0).abs() < 1e-6,
                "empty cluster overwritten at {}",
                j
            );
        }
    }
}
