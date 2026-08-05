//! GPU benchmarks for k-means: assignment-kernel variants plus the full
//! device-resident Lloyd's loop.
//!
//! ### Assignment variants
//!
//! Each is self-contained (kernel plus launcher) and consumes the same
//! prepared GpuTensor inputs, so the comparison is fair.
//!
//! * `v1_baseline` - WG=32, SMEM-tiled centroids, scalar inner loop.
//! * `v2_wide_wg` - WG=128, otherwise identical to V1. This is what
//!   production dispatches (`flash_assign_device` uses `WORKGROUP_128`).
//! * `v3_rn` - WG=32, SMEM-tiled, register-blocked (rn points per
//!   thread, dim-aware via `rn_for`).
//! * `v4_vec` - WG=128, NO SMEM, vectorised inner arithmetic via
//!   Vector<F, N>. Conflates two changes (drop SMEM, vectorise); see comments
//!   on the kernel.
//!
//! ### Full loop
//!
//! `run_loop_suite` calls `k_means_clusters_gpu` end to end at the shapes
//! production actually runs (Harmony: large n, small dim, k 100-200) and at
//! the large-k / high-dim shapes the variants above were written for. Host
//! initialisation is timed separately because `kmeans_parallel_init` runs on
//! the CPU and is invisible in a single end-to-end number.
//!
//! Run with: cargo bench --bench gpu_k_means_bench --features gpu

#![allow(missing_docs)]

use std::time::{Duration, Instant};

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl::prelude::*;
use faer::Mat;

use ann_search_rs::utils::dist::{Dist, compute_l2_norm};
use ann_search_rs::utils::k_means_utils::{KMeansInit, fast_random_init, kmeans_parallel_init};
use cubecl_utils_rs::prelude::*;

use bixverse_rs::gpu::WORKGROUP_32;

use bixverse_rs::gpu::ml::k_means_gpu::{KMeansGpuParams, k_means_clusters_gpu};

////////////
// Shapes //
////////////

#[derive(Clone, Copy, Debug)]
struct AssignShape {
    n: usize,
    k: usize,
    dim: usize,
}

const SHAPES: &[AssignShape] = &[
    AssignShape {
        n: 10_000,
        k: 400,
        dim: 128,
    },
    AssignShape {
        n: 10_000,
        k: 400,
        dim: 512,
    },
];

/// Mirrors the production `assign_k_tile` heuristic. Note that production
/// dispatches at `WORKGROUP_128`, i.e. V2's launch config, not V1's.
fn k_tile_for(dim: usize) -> usize {
    match dim {
        0..=64 => 32,
        65..=256 => 16,
        257..=512 => 8,
        513..=1024 => 4,
        _ => 2,
    }
}

/// Dim-aware register-blocking factor. Keeps per-thread private memory
/// (`rn * dim * 4 B`) below the register-spill threshold on Apple GPUs.
/// At dim>256 collapses to rn=1, which makes V3 equivalent to V1 - informative
/// only at low/mid dim.
fn rn_for(dim: usize) -> usize {
    match dim {
        0..=128 => 4,
        129..=256 => 2,
        _ => 1,
    }
}

/////////////
// Kernels //
/////////////

// V1: baseline - one thread per point, WG=32, SMEM tile of k_tile centroids,
// scalar inner loop.
#[cube(launch_unchecked)]
fn assign_v1_baseline<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    centroids: &Tensor<Vector<F, N>>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] k_tile: usize,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let tx = UNIT_POS_X;
    let wg = WORKGROUP_32;

    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg + tx;
    let active = point_idx < n_samples;
    let p_idx_safe = if active {
        point_idx
    } else {
        #[allow(clippy::useless_conversion)]
        0u32.into()
    };
    let p_base = p_idx_safe as usize * dim_lines;

    let mut p = Array::<F>::new(dim_scalars);
    for i in 0..dim_lines {
        let pl = data[p_base + i];
        #[unroll]
        for lane in 0..lanes {
            p[i * lanes + lane] = pl[lane];
        }
    }

    let mut s_cents = SharedMemory::<F>::new(k_tile * dim_scalars);

    let mut best_dist = F::new(f32::MAX);
    let mut best_idx = 0u32;

    let kt = k_tile as u32;
    let n_tiles = k.div_ceil(kt);
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_c0 = tile * kt;

        let total_elems = k_tile * dim_scalars;
        let mut load_idx = tx as usize;
        while load_idx < total_elems {
            let c_local = load_idx / dim_scalars;
            let elem = load_idx % dim_scalars;
            let c_global = tile_c0 + c_local as u32;
            if c_global < k {
                let line_idx = elem / lanes;
                let lane = elem % lanes;
                let cl = centroids[c_global as usize * dim_lines + line_idx];
                s_cents[load_idx] = cl[lane];
            } else {
                s_cents[load_idx] = F::new(0.0);
            }
            load_idx += wg as usize;
        }
        sync_cube();

        let mut c_local = 0u32;
        while c_local < kt {
            let c_global = tile_c0 + c_local;
            if c_global < k {
                let cbase = c_local as usize * dim_scalars;
                let mut sum = F::new(0.0);
                for e in 0..dim_scalars {
                    let diff = p[e] - s_cents[cbase + e];
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

    if active {
        assignments[point_idx as usize] = best_idx;
    }
}

// V2: same as V1 but with comptime wg_size in place of WORKGROUP_32.
// Strictly the same arithmetic as V1 - only the workgroup width changes.
#[cube(launch_unchecked)]
fn assign_v2_wide_wg<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    centroids: &Tensor<Vector<F, N>>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] k_tile: usize,
    #[comptime] wg_size: u32,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let tx = UNIT_POS_X;

    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + tx;
    let active = point_idx < n_samples;
    let p_idx_safe = if active {
        point_idx
    } else {
        #[allow(clippy::useless_conversion)]
        0u32.into()
    };
    let p_base = p_idx_safe as usize * dim_lines;

    let mut p = Array::<F>::new(dim_scalars);
    for i in 0..dim_lines {
        let pl = data[p_base + i];
        #[unroll]
        for lane in 0..lanes {
            p[i * lanes + lane] = pl[lane];
        }
    }

    let mut s_cents = SharedMemory::<F>::new(k_tile * dim_scalars);

    let mut best_dist = F::new(f32::MAX);
    let mut best_idx = 0u32;

    let kt = k_tile as u32;
    let n_tiles = k.div_ceil(kt);
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_c0 = tile * kt;

        let total_elems = k_tile * dim_scalars;
        let mut load_idx = tx as usize;
        while load_idx < total_elems {
            let c_local = load_idx / dim_scalars;
            let elem = load_idx % dim_scalars;
            let c_global = tile_c0 + c_local as u32;
            if c_global < k {
                let line_idx = elem / lanes;
                let lane = elem % lanes;
                let cl = centroids[c_global as usize * dim_lines + line_idx];
                s_cents[load_idx] = cl[lane];
            } else {
                s_cents[load_idx] = F::new(0.0);
            }
            load_idx += wg_size as usize;
        }
        sync_cube();

        let mut c_local = 0u32;
        while c_local < kt {
            let c_global = tile_c0 + c_local;
            if c_global < k {
                let cbase = c_local as usize * dim_scalars;
                let mut sum = F::new(0.0);
                for e in 0..dim_scalars {
                    let diff = p[e] - s_cents[cbase + e];
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

    if active {
        assignments[point_idx as usize] = best_idx;
    }
}

// V3: register-blocked. Each thread owns rn points; one centroid scalar
// pulled from SMEM feeds rn independent accumulators. SMEM tiling and WG=32
// as V1. `rn` is comptime so the launcher monomorphises per rn value.
#[cube(launch_unchecked)]
fn assign_v3_rn<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    centroids: &Tensor<Vector<F, N>>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] k_tile: usize,
    #[comptime] rn: usize,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let tx = UNIT_POS_X;
    let wg = WORKGROUP_32;

    let thread_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg + tx;
    let p0 = thread_idx as usize * rn;

    let mut p = Array::<F>::new(rn * dim_scalars);
    for r in 0..rn {
        let pid = p0 + r;
        let safe = if (pid as u32) < n_samples {
            pid
        } else {
            #[allow(clippy::useless_conversion)]
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
    }

    let mut s_cents = SharedMemory::<F>::new(k_tile * dim_scalars);

    let mut best_dist = Array::<F>::new(rn);
    let mut best_idx = Array::<u32>::new(rn);
    for r in 0..rn {
        best_dist[r] = F::new(f32::MAX);
        best_idx[r] = 0u32;
    }

    let kt = k_tile as u32;
    let n_tiles = k.div_ceil(kt);
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_c0 = tile * kt;

        let total_elems = k_tile * dim_scalars;
        let mut load_idx = tx as usize;
        while load_idx < total_elems {
            let c_local = load_idx / dim_scalars;
            let elem = load_idx % dim_scalars;
            let c_global = tile_c0 + c_local as u32;
            if c_global < k {
                let line_idx = elem / lanes;
                let lane = elem % lanes;
                let cl = centroids[c_global as usize * dim_lines + line_idx];
                s_cents[load_idx] = cl[lane];
            } else {
                s_cents[load_idx] = F::new(0.0);
            }
            load_idx += wg as usize;
        }
        sync_cube();

        let mut c_local = 0u32;
        while c_local < kt {
            let c_global = tile_c0 + c_local;
            if c_global < k {
                let cbase = c_local as usize * dim_scalars;
                let mut sum = Array::<F>::new(rn);
                for r in 0..rn {
                    sum[r] = F::new(0.0);
                }
                for e in 0..dim_scalars {
                    let cval = s_cents[cbase + e];
                    for r in 0..rn {
                        let diff = p[r * dim_scalars + e] - cval;
                        let acc = sum[r];
                        sum[r] = acc + diff * diff;
                    }
                }
                for r in 0..rn {
                    if sum[r] < best_dist[r] {
                        best_dist[r] = sum[r];
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

// V4: vectorised inner arithmetic. WG=128, NO SMEM tile, centroids read
// directly from global memory as Vector<F, N> per inner step.
//
// The conflated knobs: drops SMEM AND vectorises arithmetic. Done together
// because:
//   (a) SharedMemory<Vector<F, N>> silently broadcasts lane 0 in cubecl on
//       wgpu, per the comment in dist_gpu.rs - so vectorised SMEM is broken.
//   (b) The Vector source doesn't expose a "construct from N scalars" API,
//       so we can't read 4 SMEM scalars and re-pack into a Vector for the
//       inner FMA.
//
// If V4 wins, a follow-up experiment is needed to disambiguate which knob
// did the work. The L2 caches centroids at dim=128 (635 KiB) cleanly; at
// dim=512 (2.5 MiB) it depends on Apple's per-shader-core L2 partition.
//
// SPECULATION FLAG: not yet compiled. Relies on:
//   * Vector<F, N> - Vector<F, N> producing a vector subtract
//   * Vector<F, N> * Vector<F, N> producing a vector multiply
//   * Array<Vector<F, N>>::new(dim_lines) being valid
// If any of these fail at compile time, the fix is local to this kernel.
#[cube(launch_unchecked)]
fn assign_v4_vec<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    centroids: &Tensor<Vector<F, N>>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] wg_size: u32,
) {
    let tx = UNIT_POS_X;
    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + tx;
    if point_idx >= n_samples {
        terminate!();
    }
    let p_base = point_idx as usize * dim_lines;

    // Point held in private memory as Vector<F, N>; no scalar explode.
    let mut p = Array::<Vector<F, N>>::new(dim_lines);
    for i in 0..dim_lines {
        p[i] = data[p_base + i];
    }

    let mut best_dist = F::new(f32::MAX);
    let mut best_idx = 0u32;

    let mut c = 0u32;
    while c < k {
        let cbase = c as usize * dim_lines;
        let mut sum = F::new(0.0);

        // Inner loop: one Vector subtract and one Vector multiply per
        // dim_line; horizontal reduce via lane extraction at the end.
        for i in 0..dim_lines {
            let c_vec = centroids[cbase + i];
            let diff = p[i] - c_vec;
            let sq = diff * diff;
            #[unroll]
            for lane in 0..LINE_SIZE {
                sum += sq[lane];
            }
        }

        if sum < best_dist {
            best_dist = sum;
            best_idx = c;
        }
        c += 1u32;
    }

    assignments[point_idx as usize] = best_idx;
}

///////////////////////
// Bench input/setup //
///////////////////////

#[derive(Clone)]
struct AssignInput<R: Runtime> {
    data: GpuTensor<R, f32>,
    cents: GpuTensor<R, f32>,
    assignments: GpuTensor<R, u32>,
}

fn make_input<R: Runtime>(shape: AssignShape, client: &ComputeClient<R>) -> AssignInput<R> {
    let data_host: Vec<f32> = (0..shape.n * shape.dim)
        .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1)
        .collect();
    let cent_host: Vec<f32> = (0..shape.k * shape.dim)
        .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
        .collect();

    AssignInput {
        data: GpuTensor::<R, f32>::from_slice(&data_host, vec![shape.n, shape.dim], client)
            .unwrap(),
        cents: GpuTensor::<R, f32>::from_slice(&cent_host, vec![shape.k, shape.dim], client)
            .unwrap(),
        assignments: GpuTensor::<R, u32>::empty(vec![shape.n], client).unwrap(),
    }
}

/////////////////////
// Bench wrappers  //
/////////////////////

const V2_WG: u32 = 128;
const V4_WG: u32 = 128;

///////////////
// Version 1 //
///////////////

struct V1Bench<R: Runtime> {
    shape: AssignShape,
    client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for V1Bench<R> {
    type Input = AssignInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        make_input(self.shape, &self.client)
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let dim_lines = self.shape.dim / LINE_SIZE;
        let k_tile = k_tile_for(self.shape.dim);
        let n_workgroups = (self.shape.n as u32).div_ceil(WORKGROUP_32);
        let (gx, gy) = grid_2d(n_workgroups, &GpuLimits::from_client(&self.client)).unwrap();

        unsafe {
            assign_v1_baseline::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_1d(WORKGROUP_32),
                LINE_SIZE,
                input.data.into_tensor_arg(),
                input.cents.into_tensor_arg(),
                input.assignments.into_tensor_arg(),
                self.shape.n as u32,
                self.shape.k as u32,
                dim_lines,
                k_tile,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "v1_baseline_wg32_n{}_k{}_d{}",
            self.shape.n, self.shape.k, self.shape.dim
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

///////////////
// Version 2 //
///////////////

struct V2Bench<R: Runtime> {
    shape: AssignShape,
    client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for V2Bench<R> {
    type Input = AssignInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        make_input(self.shape, &self.client)
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let dim_lines = self.shape.dim / LINE_SIZE;
        let k_tile = k_tile_for(self.shape.dim);
        let n_workgroups = (self.shape.n as u32).div_ceil(V2_WG);
        let (gx, gy) = grid_2d(n_workgroups, &GpuLimits::from_client(&self.client)).unwrap();

        unsafe {
            assign_v2_wide_wg::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_1d(V2_WG),
                LINE_SIZE,
                input.data.into_tensor_arg(),
                input.cents.into_tensor_arg(),
                input.assignments.into_tensor_arg(),
                self.shape.n as u32,
                self.shape.k as u32,
                dim_lines,
                k_tile,
                V2_WG,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "v2_wide_wg{}_n{}_k{}_d{}",
            V2_WG, self.shape.n, self.shape.k, self.shape.dim
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

///////////////
// Version 3 //
///////////////

struct V3Bench<R: Runtime> {
    shape: AssignShape,
    client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for V3Bench<R> {
    type Input = AssignInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        make_input(self.shape, &self.client)
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let dim_lines = self.shape.dim / LINE_SIZE;
        let k_tile = k_tile_for(self.shape.dim);
        let rn = rn_for(self.shape.dim);
        let n_threads = self.shape.n.div_ceil(rn) as u32;
        let n_workgroups = n_threads.div_ceil(WORKGROUP_32);
        let (gx, gy) = grid_2d(n_workgroups, &GpuLimits::from_client(&self.client)).unwrap();
        let count = CubeCount::Static(gx, gy, 1);
        let cdim = CubeDim::new_1d(WORKGROUP_32);

        // `rn` is comptime in the kernel; the launcher dispatches per rn value
        // so each gets its own monomorphisation.
        match rn {
            4 => unsafe {
                assign_v3_rn::launch_unchecked::<f32, R>(
                    &self.client,
                    count,
                    cdim,
                    LINE_SIZE,
                    input.data.into_tensor_arg(),
                    input.cents.into_tensor_arg(),
                    input.assignments.into_tensor_arg(),
                    self.shape.n as u32,
                    self.shape.k as u32,
                    dim_lines,
                    k_tile,
                    4,
                );
            },
            2 => unsafe {
                assign_v3_rn::launch_unchecked::<f32, R>(
                    &self.client,
                    count,
                    cdim,
                    LINE_SIZE,
                    input.data.into_tensor_arg(),
                    input.cents.into_tensor_arg(),
                    input.assignments.into_tensor_arg(),
                    self.shape.n as u32,
                    self.shape.k as u32,
                    dim_lines,
                    k_tile,
                    2,
                );
            },
            1 => unsafe {
                assign_v3_rn::launch_unchecked::<f32, R>(
                    &self.client,
                    count,
                    cdim,
                    LINE_SIZE,
                    input.data.into_tensor_arg(),
                    input.cents.into_tensor_arg(),
                    input.assignments.into_tensor_arg(),
                    self.shape.n as u32,
                    self.shape.k as u32,
                    dim_lines,
                    k_tile,
                    1,
                );
            },
            _ => unreachable!("rn_for returned unsupported value {}", rn),
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "v3_rn{}_wg32_n{}_k{}_d{}",
            rn_for(self.shape.dim),
            self.shape.n,
            self.shape.k,
            self.shape.dim
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

///////////////
// Version 4 //
///////////////

struct V4Bench<R: Runtime> {
    shape: AssignShape,
    client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for V4Bench<R> {
    type Input = AssignInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        make_input(self.shape, &self.client)
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let dim_lines = self.shape.dim / LINE_SIZE;
        let n_workgroups = (self.shape.n as u32).div_ceil(V4_WG);
        let (gx, gy) = grid_2d(n_workgroups, &GpuLimits::from_client(&self.client)).unwrap();

        unsafe {
            assign_v4_vec::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_1d(V4_WG),
                LINE_SIZE,
                input.data.into_tensor_arg(),
                input.cents.into_tensor_arg(),
                input.assignments.into_tensor_arg(),
                self.shape.n as u32,
                self.shape.k as u32,
                dim_lines,
                V4_WG,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "v4_vec_wg{}_n{}_k{}_d{}",
            V4_WG, self.shape.n, self.shape.k, self.shape.dim
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

/////////////////////
// Full-loop bench //
/////////////////////

/// One end-to-end k-means configuration.
#[derive(Clone, Copy, Debug)]
struct LoopShape {
    /// Human-readable label used in the output.
    label: &'static str,
    /// Number of points.
    n: usize,
    /// Embedding dimensionality. All entries are multiples of `LINE_SIZE`, so
    /// `k_means_clusters_gpu` takes its no-padding path.
    dim: usize,
    /// Number of centroids.
    k: usize,
    /// Distance metric string, as `k_means_clusters_gpu` parses it.
    metric: &'static str,
    /// Whether to run both initialisation strategies. Only meaningful below
    /// the `n_centroids > 200` threshold, where the default picks k-means||.
    sweep_init: bool,
}

/// The three Harmony rows are the production regime: `harmony_v2_gpu` is the
/// only caller and runs at large `n`, `dim` 20-50 and `k` 100-200 with cosine.
/// The last two are the shapes the assignment-kernel variants were written
/// for, kept so a change cannot silently regress them.
const LOOP_SHAPES: &[LoopShape] = &[
    LoopShape {
        label: "harmony-small",
        n: 100_000,
        dim: 32,
        k: 100,
        metric: "cosine",
        sweep_init: true,
    },
    LoopShape {
        label: "harmony-large",
        n: 1_000_000,
        dim: 48,
        k: 100,
        metric: "cosine",
        sweep_init: true,
    },
    LoopShape {
        label: "harmony-wide",
        n: 1_000_000,
        dim: 48,
        k: 200,
        metric: "cosine",
        sweep_init: true,
    },
    LoopShape {
        label: "large-k",
        n: 10_000,
        dim: 128,
        k: 400,
        metric: "euclidean",
        sweep_init: false,
    },
    LoopShape {
        label: "high-dim",
        n: 10_000,
        dim: 512,
        k: 400,
        metric: "euclidean",
        sweep_init: false,
    },
];

/// Lloyd's iterations per run. Matches `KMeansGpuParams::default`, which is
/// what Harmony gets.
const LOOP_ITERS: usize = 50;

/// Measured repetitions per configuration, after one discarded warm-up. Two
/// is enough to spot a wildly unstable reading without making a 1M-point
/// sweep take all afternoon.
const LOOP_REPS: usize = 2;

/// Deterministic synthetic data, flat row-major.
///
/// ### Params
///
/// * `n` - Number of points
/// * `dim` - Embedding dimensionality
///
/// ### Returns
///
/// `n * dim` values in row-major order.
fn make_loop_data(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim)
        .map(|i| {
            // A coarse cluster signal on top of a deterministic ramp, so the
            // partition is non-degenerate and empty clusters stay rare.
            let cluster = (i / dim) % 64;
            ((i * 13 + 7) % 29) as f32 * 0.1 + cluster as f32 * 0.05
        })
        .collect()
}

/// Time the CPU reference initialisation, for comparison only.
///
/// `k_means_clusters_gpu` now runs k-means|| on the device, so this is the
/// cost that path replaced rather than a component of the measured total. It
/// stays in the bench because it is the number that justified porting the
/// initialisation in the first place: 1.9 s at n = 1e6, k = 100 and 5.4 s at
/// k = 200, against a GPU loop of well under a second.
///
/// ### Params
///
/// * `data` - Flat row-major data `[n, dim]`
/// * `shape` - The configuration being measured
/// * `init` - Initialisation strategy
/// * `metric` - Parsed distance metric
///
/// ### Returns
///
/// `(elapsed, centroids)`
fn time_init(
    data: &[f32],
    shape: LoopShape,
    init: KMeansInit,
    metric: &Dist,
) -> (Duration, Vec<f32>) {
    let start = Instant::now();
    let cents = match init {
        KMeansInit::Random => fast_random_init(data, shape.dim, shape.n, shape.k, 42),
        KMeansInit::KMeansParallel => {
            let norms: Vec<f32> = (0..shape.n)
                .map(|i| compute_l2_norm(&data[i * shape.dim..(i + 1) * shape.dim]))
                .collect();
            kmeans_parallel_init(data, &norms, shape.dim, shape.n, shape.k, metric, 42)
        }
    };
    (start.elapsed(), cents)
}

/// Reject a run that produced no work.
///
/// `k_means_clusters_gpu` dispatches with `launch_unchecked`, which does
/// nothing and reports no error when a device limit is busted; the assignment
/// buffer is `GpuTensor::empty`, so a dead run returns uninitialised VRAM.
/// Out-of-range indices catch that, and the non-empty cluster count catches a
/// run that technically wrote but collapsed.
///
/// ### Params
///
/// * `assignments` - Hard assignments returned by the driver
/// * `shape` - The configuration being checked
fn guard_assignments(assignments: &[usize], shape: LoopShape) {
    assert_eq!(assignments.len(), shape.n, "wrong assignment count");
    assert!(
        assignments.iter().all(|&a| a < shape.k),
        "assignment out of range: the GPU almost certainly did no work"
    );
    let mut seen = vec![false; shape.k];
    for &a in assignments {
        seen[a] = true;
    }
    let occupied = seen.iter().filter(|&&s| s).count();
    assert!(
        occupied * 2 > shape.k,
        "only {}/{} clusters occupied, partition collapsed",
        occupied,
        shape.k
    );
}

/// Run one shape under one initialisation strategy and print the split
/// between host init and device loop.
fn run_loop_shape<R: Runtime>(shape: LoopShape, init: KMeansInit, device: &R::Device)
where
    R::Device: Clone,
{
    let flat = make_loop_data(shape.n, shape.dim);
    let mat = Mat::<f32>::from_fn(shape.n, shape.dim, |i, j| flat[i * shape.dim + j]);
    let metric = match shape.metric {
        "cosine" => Dist::Cosine,
        _ => Dist::SquaredEuclidean,
    };

    let params = KMeansGpuParams::new(LOOP_ITERS, Some(init), true, false);

    // Warm-up: compiles the shaders and faults in the buffers, both of which
    // would otherwise land entirely in the first measured rep.
    let (_, warm) = k_means_clusters_gpu::<f32, R>(
        mat.as_ref(),
        shape.metric,
        shape.k,
        Some(params),
        42,
        device.clone(),
        false,
    )
    .expect("warm-up run failed");
    guard_assignments(&warm, shape);

    let (init_time, _) = time_init(&flat, shape, init, &metric);

    for rep in 0..LOOP_REPS {
        let start = Instant::now();
        let (_, assignments) = k_means_clusters_gpu::<f32, R>(
            mat.as_ref(),
            shape.metric,
            shape.k,
            Some(params),
            42,
            device.clone(),
            false,
        )
        .expect("k-means run failed");
        let total = start.elapsed();
        guard_assignments(&assignments, shape);

        // `init_time` is the CPU reference cost. `k_means_clusters_gpu` runs
        // k-means|| on the device, so this is what the GPU path replaced, not
        // a component of `total`.
        println!(
            "  {:<14} init={:?}  rep {}: total {:>10.2?} | cpu-ref init {:>10.2?}",
            shape.label, init, rep, total, init_time,
        );
    }
}

/// End-to-end Lloyd's loop across every shape in [`LOOP_SHAPES`].
fn run_loop_suite<R: Runtime>(device: &R::Device)
where
    R::Device: Clone,
{
    println!(
        "\n====== k-means full-loop bench ({} iters) ======\n",
        LOOP_ITERS
    );

    // Under the profiler the whole sweep takes long enough to be unusable, so
    // allow narrowing to one row by label.
    let only = std::env::var("BIXVERSE_BENCH_SHAPE").ok();

    for &shape in LOOP_SHAPES {
        if let Some(want) = &only
            && want != shape.label
        {
            continue;
        }
        println!(
            "--- {}: n={}, k={}, dim={}, {} ---",
            shape.label, shape.n, shape.k, shape.dim, shape.metric
        );

        // `k_means_clusters_gpu` picks Random above k=200 and k-means|| at or
        // below it, so the production range straddles a cliff. Measure both
        // sides of it rather than trusting the default.
        if shape.sweep_init {
            run_loop_shape::<R>(shape, KMeansInit::Random, device);
            run_loop_shape::<R>(shape, KMeansInit::KMeansParallel, device);
        } else {
            run_loop_shape::<R>(shape, KMeansInit::Random, device);
        }
        println!();
    }
}

////////////
// Runner //
////////////

fn run_suite<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    println!("====== k-means assign kernel microbench ======");
    println!("Timing method: System (wall-clock)\n");

    for &shape in SHAPES {
        println!(
            "\n--- shape: n={}, k={}, dim={} (k*dim*4 = {} KiB) ---\n",
            shape.n,
            shape.k,
            shape.dim,
            (shape.k * shape.dim * 4) / 1024
        );

        let v1 = V1Bench::<R> {
            shape,
            client: client.clone(),
        };
        let v2 = V2Bench::<R> {
            shape,
            client: client.clone(),
        };
        let v3 = V3Bench::<R> {
            shape,
            client: client.clone(),
        };
        let v4 = V4Bench::<R> {
            shape,
            client: client.clone(),
        };

        println!("{}", v1.name());
        println!("{:?}\n", v1.run(TimingMethod::System));

        println!("{}", v2.name());
        println!("{:?}\n", v2.run(TimingMethod::System));

        println!("{}", v3.name());
        println!("{:?}\n", v3.run(TimingMethod::System));

        println!("{}", v4.name());
        println!("{:?}\n", v4.run(TimingMethod::System));
    }
}

fn main() {
    let device: <cubecl::wgpu::WgpuRuntime as Runtime>::Device = Default::default();

    // `BIXVERSE_BENCH_ONLY=micro|loop` runs a single group. Useful under
    // `CUBECL_DEBUG_OPTION=profile-medium`, where the profiler's syncs make
    // running both groups take far longer than it is worth.
    let only = std::env::var("BIXVERSE_BENCH_ONLY").unwrap_or_default();

    if only != "loop" {
        run_suite::<cubecl::wgpu::WgpuRuntime>(&device);
    }
    if only != "micro" {
        run_loop_suite::<cubecl::wgpu::WgpuRuntime>(&device);
    }
}
