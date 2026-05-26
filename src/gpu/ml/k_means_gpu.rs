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
    /// Convergence threshold on maximum per-centroid drift (L2).
    pub tol: f64,
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
    /// * `tol` - Tolerance parameter
    /// * `fixed - Shall the algorithm be run for a fixed set of iterations
    ///   or shall convergence be checked.
    pub fn new(iters: usize, init: Option<KMeansInit>, tol: f64, fixed: bool) -> Self {
        Self {
            iters,
            init,
            tol,
            fixed,
        }
    }
}

/// Default implementation for [KMeansGpuParams]
impl Default for KMeansGpuParams {
    fn default() -> Self {
        Self::new(50, None, 1e-5, true)
    }
}

/////////////
// Helpers //
/////////////

/// Online-argmin Euclidean assignment.
///
/// One thread per point; centroids are streamed through shared memory in
/// tiles of `bk`. No N x K materialisation.
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
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X`
///   -> point index
#[cube(launch_unchecked)]
pub fn flash_assign_euclidean<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    centroids: &Tensor<Vector<F, N>>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] bk: usize,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;

    // Read this thread's point into registers once. Inactive threads read
    // point 0 (harmless) and simply never write a result.
    let p_base: usize = if point_idx < n_samples {
        point_idx as usize * dim_lines
    } else {
        #[allow(clippy::useless_conversion)]
        0_usize.into()
    };
    let mut p = Array::<F>::new(dim_scalars);
    for i in 0..dim_lines {
        let pl = data[p_base + i];
        #[unroll]
        for lane in 0..lanes {
            p[i * lanes + lane] = pl[lane];
        }
    }

    let mut s_cent = SharedMemory::<F>::new(bk * dim_scalars);

    let mut best_dist = F::new(f32::MAX);
    let mut best_idx = 0u32;

    #[allow(clippy::manual_div_ceil)]
    let n_tiles = (k + bk as u32 - 1u32) / bk as u32;
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_c0 = tile * bk as u32;

        // Cooperative load of up to `bk` centroids into scalar shared memory.
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

        // Scan the cached tile, online argmin.
        let mut c_local = 0u32;
        while c_local < bk as u32 {
            let c_global = tile_c0 + c_local;
            if c_global < k {
                let cbase = c_local as usize * dim_scalars;
                let mut sum = F::new(0.0);
                for e in 0..dim_scalars {
                    let diff = p[e] - s_cent[cbase + e];
                    sum += diff * diff;
                }
                if sum < best_dist {
                    best_dist = sum;
                    best_idx = c_global;
                }
            }
            c_local += 1u32;
        }
        sync_cube();

        tile += 1u32;
    }

    if point_idx < n_samples {
        assignments[point_idx as usize] = best_idx;
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
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X`
///   -> point index
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
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;

    let safe_idx = if point_idx < n_samples {
        point_idx as usize
    } else {
        #[allow(clippy::useless_conversion)]
        0usize.into()
    };
    let p_base = safe_idx * dim_lines;
    let mut p = Array::<F>::new(dim_scalars);
    for i in 0..dim_lines {
        let pl = data[p_base + i];
        #[unroll]
        for lane in 0..lanes {
            p[i * lanes + lane] = pl[lane];
        }
    }
    let pnorm = point_norms[safe_idx];

    let mut s_cent = SharedMemory::<F>::new(bk * dim_scalars);

    let mut best_dist = F::new(f32::MAX);
    let mut best_idx = 0u32;

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
                let mut dot = F::new(0.0);
                for e in 0..dim_scalars {
                    dot += p[e] * s_cent[cbase + e];
                }
                let cnorm = centroid_norms[c_global as usize];
                let dist = F::new(1.0) - dot / (pnorm * cnorm);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = c_global;
                }
            }
            c_local += 1u32;
        }
        sync_cube();

        tile += 1u32;
    }

    if point_idx < n_samples {
        assignments[point_idx as usize] = best_idx;
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
    let bk = 16usize;

    let data_gpu = GpuTensor::<R, T>::from_slice(data, vec![n, dim], &client);
    let cent_gpu = GpuTensor::<R, T>::from_slice(centroids, vec![k, dim], &client);
    let assign_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);

    let (gx, gy) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));
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

    unsafe {
        histogram_clusters::launch_unchecked::<R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            assignments.clone().into_tensor_arg(),
            counts.clone().into_tensor_arg(),
            n as u32,
        );

        exclusive_scan_offsets::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            counts.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            cursor.clone().into_tensor_arg(),
            k as u32,
        );

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

    (all_indices, offsets)
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

/// Launch FlashAssign into a device-resident assignment buffer. Internal to the
/// driver loop; assumes `dim` is already padded to a multiple of LINE_SIZE and
/// all tensors live on `client`'s device.
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
    let (gx, gy) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));
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
            );
        },
        Dist::Manhattan => unreachable!(),
    }
}

/// Per-centroid squared L2 drift between consecutive centroid snapshots.
///
/// One workgroup per centroid; thread 0 accumulates the squared element-wise
/// differences across the full row. Used to detect convergence without reading
/// back the full centroid matrix.
///
/// ### Params
///
/// * `old_cents` - Centroid vectors before the update `[k, dim]`
/// * `new_cents` - Centroid vectors after the update `[k, dim]`
/// * `drift_sq` - Output per-centroid squared drift `[k]`, written by thread 0
/// * `k` - Number of centroids
/// * `dim` - Embedding dimensionality (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> centroid index
/// * Only `UNIT_POS_X == 0` writes output
#[cube(launch_unchecked)]
pub fn centroid_drift_sq<F: Float>(
    old_cents: &Tensor<F>,
    new_cents: &Tensor<F>,
    drift_sq: &mut Tensor<F>,
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
            let d = new_cents[base + e] - old_cents[base + e];
            acc += d * d;
        }
        drift_sq[c as usize] = acc;
    }
}

/// Reduce a vector of per-centroid squared drifts to its maximum.
///
/// Single-thread serial scan over `k` values. The result is written to
/// `out[0]` and read back to the host as a single scalar; taking `sqrt` on
/// the host converts squared drift to L2 drift. Serial cost is negligible
/// because `k` is small relative to `n`.
///
/// ### Params
///
/// * `values` - Per-centroid squared drifts `[k]` produced by `centroid_drift_sq`
/// * `out` - Single-element output buffer `[1]` receiving the maximum value
/// * `k` - Number of centroids
///
/// ### Grid mapping
///
/// * Single cube, single thread (`UNIT_POS_X == 0`)
#[cube(launch_unchecked)]
pub fn max_reduce<F: Float>(values: &Tensor<F>, out: &mut Tensor<F>, k: u32) {
    if UNIT_POS_X == 0u32 {
        let mut m = F::new(0.0);
        let mut i = 0u32;
        while i < k {
            let v = values[i as usize];
            if v > m {
                m = v;
            }
            i += 1u32;
        }
        out[0] = m;
    }
}

/// Element-wise copy of a flat float buffer.
///
/// One thread per element. Used to snapshot the centroid matrix before an
/// in-place update so that drift can be computed device-side without a host
/// readback.
///
/// ### Params
///
/// * `src` - Source buffer `[n_elems]`
/// * `dst` - Destination buffer `[n_elems]`, written in place
/// * `n_elems` - Total number of elements to copy
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> element index
#[cube(launch_unchecked)]
pub fn copy_f<F: Float>(src: &Tensor<F>, dst: &mut Tensor<F>, n_elems: u32) {
    let i = ABSOLUTE_POS_X;
    if i >= n_elems {
        terminate!();
    }
    dst[i as usize] = src[i as usize];
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

/// Per-centroid squared L2 drift between consecutive centroid snapshots.
///
/// One workgroup per centroid; thread 0 accumulates the squared element-wise
/// differences across the full row. Used to detect convergence without reading
/// back the full centroid matrix.
///
/// ### Params
///
/// * `old_cents` - Centroid vectors before the update `[k, dim]`
/// * `new_cents` - Centroid vectors after the update `[k, dim]`
/// * `drift_sq` - Output per-centroid squared drift `[k]`, written by thread 0
/// * `k` - Number of centroids
/// * `dim` - Embedding dimensionality (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> centroid index
/// * Only `UNIT_POS_X == 0` writes output

/// Reduce a vector of per-centroid squared drifts to its maximum.
///
/// Single-thread serial scan over `k` values. The result is written to
/// `out[0]` and read back to the host as a single scalar; taking `sqrt` on
/// the host converts squared drift to L2 drift. Serial cost is negligible
/// because `k` is small relative to `n`.
///
/// ### Params
///
/// * `values` - Per-centroid squared drifts `[k]` produced by `centroid_drift_sq`
/// * `out` - Single-element output buffer `[1]` receiving the maximum value
/// * `k` - Number of centroids
///
/// ### Grid mapping
///
/// * Single cube, single thread (`UNIT_POS_X == 0`)

/// Element-wise copy of a flat float buffer.
///
/// One thread per element. Used to snapshot the centroid matrix before an
/// in-place update so that drift can be computed device-side without a host
/// readback.
///
/// ### Params
///
/// * `src` - Source buffer `[n_elems]`
/// * `dst` - Destination buffer `[n_elems]`, written in place
/// * `n_elems` - Total number of elements to copy
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> element index

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

/// One Lloyd's iteration entirely on device: assign, rebuild CSR, update centroids.
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
) where
    R: Runtime,
    T: Float + Sum + cubecl::CubeElement + num_traits::Float + num_traits::FromPrimitive,
{
    flash_assign_device(
        client, data_gpu, cent_gpu, pnorm_gpu, cnorm_gpu, assign_gpu, n, k, dim, metric, bk,
    );

    let (idx_gpu, off_gpu) = build_csr_gpu::<R>(assign_gpu, n, k, client);
    segmented_update::<R, T>(data_gpu, &idx_gpu, &off_gpu, cent_gpu, k, dim, client);

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
/// Convergence is detected via maximum per-centroid drift, not assignment
/// stability, so near-equidistant points that flip between equal centroids do
/// not stall termination.
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

    let client = R::client(&device);
    let data_gpu = GpuTensor::<R, T>::from_slice(&data_padded, vec![n, dim_padded], &client);
    let cent_gpu =
        GpuTensor::<R, T>::from_slice(&centroids, vec![n_centroids, dim_padded], &client);
    let assign_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);

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

    let bk = 16usize;

    if verbose {
        println!(
            "  Running Lloyd's iterations (GPU, {})",
            if params.fixed {
                "fixed"
            } else {
                "drift-checked"
            }
        );
    }

    if params.fixed {
        // No convergence check, no per-iteration readback: submit every
        // iteration back-to-back.
        for iter in 0..params.iters {
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
            );
            if verbose && (iter + 1) % 10 == 0 {
                println!("    Iteration {} complete", iter + 1);
            }
        }
    } else {
        // Device-side drift; only a single f32 is read back per iteration.
        let old_cent_gpu = GpuTensor::<R, T>::empty(vec![n_centroids, dim_padded], &client);
        let drift_sq_gpu = GpuTensor::<R, T>::empty(vec![n_centroids], &client);
        let drift_max_gpu = GpuTensor::<R, T>::empty(vec![1], &client);

        let n_cent_elems = (n_centroids * dim_padded) as u32;
        let (copy_gx, copy_gy) = grid_2d(n_cent_elems.div_ceil(WORKGROUP_SIZE_X));
        let (drift_gx, drift_gy) = grid_2d(n_centroids as u32);

        for iter in 0..params.iters {
            // Snapshot pre-update centroids for the drift comparison.
            unsafe {
                copy_f::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(copy_gx, copy_gy, 1),
                    CubeDim::new_1d(WORKGROUP_SIZE_X),
                    cent_gpu.clone().into_tensor_arg(),
                    old_cent_gpu.clone().into_tensor_arg(),
                    n_cent_elems,
                );
            }

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
            );

            unsafe {
                centroid_drift_sq::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(drift_gx, drift_gy, 1),
                    CubeDim::new_1d(1),
                    old_cent_gpu.clone().into_tensor_arg(),
                    cent_gpu.clone().into_tensor_arg(),
                    drift_sq_gpu.clone().into_tensor_arg(),
                    n_centroids as u32,
                    dim_padded,
                );
                max_reduce::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    drift_sq_gpu.clone().into_tensor_arg(),
                    drift_max_gpu.clone().into_tensor_arg(),
                    n_centroids as u32,
                );
            }

            // sqrt(max squared drift) == max L2 drift; sqrt is monotonic.
            let max_drift = drift_max_gpu.clone().read(&client)?[0]
                .to_f64()
                .unwrap()
                .sqrt();

            if max_drift <= params.tol {
                if verbose {
                    println!("    Converged at iteration {}", iter + 1);
                }
                break;
            }
            if verbose && (iter + 1) % 10 == 0 {
                println!(
                    "    Iteration {} complete (max drift {:.6})",
                    iter + 1,
                    max_drift
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
