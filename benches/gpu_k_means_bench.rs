//! GPU kernel microbenchmarks for the k-means assignment step.
//!
//! Each variant is self-contained: kernel function plus launcher. They all
//! consume the same prepared GpuTensor inputs so the comparison is fair.
//!
//! Variants:
//!
//! * `v1_baseline` - WG=32, SMEM-tiled centroids, scalar inner loop.
//! * `v2_wide_wg` - WG=128, otherwise identical to V1.
//! * `v3_rn` - WG=32, SMEM-tiled, register-blocked (rn points per
//!   thread, dim-aware via `rn_for`).
//! * `v4_vec` - WG=128, NO SMEM, vectorised inner arithmetic via
//!   Vector<F, N>. Conflates two changes (drop SMEM, vectorise); see comments
//!   on the kernel.
//!
//! Run with: cargo bench --bench gpu_kmeans_assign_kernels --features gpu

#![allow(dead_code, missing_docs)]

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl::prelude::*;

use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::*;

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

/// Mirrors the production `assign_k_tile` heuristic so V1 matches the real
/// kernel's launch config.
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
    let wg = WORKGROUP_SIZE_X;

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

// V2: same as V1 but with comptime wg_size in place of WORKGROUP_SIZE_X.
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
    let wg = WORKGROUP_SIZE_X;

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
        data: GpuTensor::<R, f32>::from_slice(&data_host, vec![shape.n, shape.dim], client),
        cents: GpuTensor::<R, f32>::from_slice(&cent_host, vec![shape.k, shape.dim], client),
        assignments: GpuTensor::<R, u32>::empty(vec![shape.n], client),
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
        let n_workgroups = (self.shape.n as u32).div_ceil(WORKGROUP_SIZE_X);
        let (gx, gy) = grid_2d(n_workgroups);

        unsafe {
            assign_v1_baseline::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_1d(WORKGROUP_SIZE_X),
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
        let (gx, gy) = grid_2d(n_workgroups);

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
        let n_workgroups = n_threads.div_ceil(WORKGROUP_SIZE_X);
        let (gx, gy) = grid_2d(n_workgroups);
        let count = CubeCount::Static(gx, gy, 1);
        let cdim = CubeDim::new_1d(WORKGROUP_SIZE_X);

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
        let (gx, gy) = grid_2d(n_workgroups);

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
    run_suite::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}
