//! Kernels for GPU HALS NMF.
//!
//! All dense factors and products are row-major `[rows, k]`, i.e. component
//! minor. `W` is `[m, k]`, and `H` is held transposed as `[n, k]`, so the two
//! Gram products, the two data products, the two sweeps and the normalisation
//! step all see the same layout. See the module doc of
//! [`crate::gpu::methods_gpu::nmf_gpu`] for why.
//!
//! ### The sweep
//!
//! HALS updates component `r` from components already updated in the same
//! sweep, so the `r` loop is strictly sequential. But `H[r, j]` reads only
//! `B[r, :]`, `A[r, j]` and `H[:, j]`, so *columns of H are independent of each
//! other*. One thread per column runs the whole k-step sweep, which makes the
//! entire sweep a single launch with no barriers rather than `k` launches.
//!
//! The same holds for `W` with one thread per row. And because `W^T W` and
//! `H H^T` are both symmetric, the row sweep's `B[r, s]` and the column
//! sweep's `D[s, c]` are the same walk, so [`fn@hals_sweep_gpu`] serves both.
//!
//! ### Where the working set lives
//!
//! Each thread needs its whole k-run addressable by a runtime index, which
//! rules out a register `Array` (this crate avoids dynamic indexing into one;
//! see the slot-by-comparison comment in `seacells_kernels.rs`), and unrolling
//! over a comptime bound would emit `k_cap^2` statements. So the run is staged
//! in shared memory, laid out component-major (`s * wg_size + tx`) so that
//! adjacent threads hit adjacent banks and no padding is needed. No thread ever
//! reads another thread's lane, so the kernel needs no `sync_cube`.
//!
//! The staged block is pinned to [`NMF_SWEEP_SMEM_ELEMS`] elements and the
//! workgroup width falls out of the rank tier, so the footprint is constant at
//! 16 KiB and two workgroups stay resident on a 32 KiB device regardless of `k`.
//!
//! `B` and `D` are workgroup-uniform, so they stay in global memory. Staging a
//! workgroup-uniform value buys nothing and spends the occupancy the staged run
//! needs.

#![allow(missing_docs)]

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;

use crate::gpu::{WORKGROUP_32, WORKGROUP_64, WORKGROUP_128, WORKGROUP_256};
use crate::prelude::*;

////////////
// Consts //
////////////

/// Elements of shared memory the sweep stages per workgroup.
///
/// The footprint is `NMF_SWEEP_SMEM_ELEMS * size_of::<F>()`, held constant
/// across rank tiers by trading workgroup width against rank capacity. At 4096
/// f32 that is 16 KiB, so two workgroups stay resident on the 32 KiB floor that
/// Apple Silicon reports, which is the step that matters: residency past two
/// buys little, and one resident workgroup roughly halves throughput.
pub const NMF_SWEEP_SMEM_ELEMS: u32 = 4096;

/// Largest rank the sweep is compiled for.
///
/// Bounded by the smallest workgroup worth dispatching. At `k_cap = 128` the
/// width is already down to 32, which is one SIMD group on Apple Silicon;
/// halving it again would leave a plane partly idle for no gain.
pub const NMF_MAX_RANK: usize = 128;

/// Workgroup width for the flat elementwise and per-row passes.
const NMF_ELEMENTWISE_WG: u32 = WORKGROUP_256;

/////////////
// Helpers //
/////////////

/// Rank capacity and workgroup width for a sweep at rank `k`.
///
/// The product is [`NMF_SWEEP_SMEM_ELEMS`] in every arm, so the shared-memory
/// footprint does not depend on `k`. Rounding the capacity up to a power of two
/// wastes at most half the staged block; the alternative is a shader per rank.
///
/// ### Params
///
/// * `k` - Number of components
///
/// ### Returns
///
/// `Some((k_cap, wg_size))`, or `None` if `k` exceeds [`NMF_MAX_RANK`].
pub fn sweep_tier(k: usize) -> Option<(u32, u32)> {
    match k {
        0 => None,
        1..=16 => Some((16, WORKGROUP_256)),
        17..=32 => Some((32, WORKGROUP_128)),
        33..=64 => Some((64, WORKGROUP_64)),
        65..=NMF_MAX_RANK => Some((128, WORKGROUP_32)),
        _ => None,
    }
}

/////////////
// Kernels //
/////////////

/// One HALS sweep over all `k` components of a `[rows, k]` factor.
///
/// For each component `r` in order, every row `i` is updated as
///
/// ```text
/// X[i, r] = max(eps, X[i, r] + (P[i, r] - sum_s G[r, s] X[i, s]) / G[r, r])
/// ```
///
/// which is the H row sweep with `X = H^T`, `G = W^T W`, `P = (W^T V)^T`, and
/// the W column sweep with `X = W`, `G = H H^T`, `P = V H^T`. The two coincide
/// because `G` is symmetric, so `G[r, :]` and `G[:, r]` are the same vector.
///
/// A component whose Gram diagonal is not positive is skipped rather than
/// floored, matching the CPU path. That is only reachable from a fully
/// collapsed component.
///
/// ### Params
///
/// * `x` - Factor `[rows, k]` row-major, updated in place
/// * `g` - Symmetric Gram `[k, k]` row-major
/// * `p` - Data product `[rows, k]` row-major
/// * `n_rows` - Number of rows of `x`
/// * `k` - Number of components, must be `<= k_cap`
/// * `eps` - Non-negativity floor as a one-element tensor. A buffer rather than
///   a scalar argument because a runtime float scalar would need a
///   `ScalarArgSettings` bound this module otherwise has no use for; the caller
///   uploads it once and reuses it across every launch.
/// * `k_cap` - Comptime rank capacity of the staged block
/// * `wg_size` - Comptime workgroup width, `k_cap * wg_size` elements staged
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> block of `wg_size` rows
/// * `UNIT_POS_X` -> row within the block
#[cube(launch_unchecked)]
pub fn hals_sweep_gpu<F: Float>(
    x: &mut Tensor<F>,
    g: &Tensor<F>,
    p: &Tensor<F>,
    eps: &Tensor<F>,
    n_rows: u32,
    k: u32,
    #[comptime] k_cap: u32,
    #[comptime] wg_size: u32,
) {
    // Declared at kernel scope before the bounds check. Safe to leave the tail
    // threads early because no thread reads another thread's lane, so there is
    // no barrier for them to skip.
    let mut stage = SharedMemory::<F>::new((wg_size * k_cap) as usize);

    let tx = UNIT_POS_X;
    let row = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + tx;
    if row >= n_rows {
        terminate!();
    }

    let base = row as usize * k as usize;

    // Load this thread's k-run. Adjacent threads read `k` apart, so the global
    // side is not coalesced, but the whole workgroup's run is contiguous and
    // every fetched line is fully consumed across the lanes.
    let mut s = 0u32;
    while s < k {
        stage[(s * wg_size + tx) as usize] = x[base + s as usize];
        s += 1u32;
    }

    let zero = F::new(0.0);
    let floor = eps[0];

    let mut r = 0u32;
    while r < k {
        let g_row = r * k;
        let grr = g[(g_row + r) as usize];
        if grr > zero {
            let mut dot = zero;
            let mut t = 0u32;
            while t < k {
                dot += g[(g_row + t) as usize] * stage[(t * wg_size + tx) as usize];
                t += 1u32;
            }
            let slot = (r * wg_size + tx) as usize;
            let mut updated = stage[slot] + (p[base + r as usize] - dot) / grr;
            if updated < floor {
                updated = floor;
            }
            stage[slot] = updated;
        }
        r += 1u32;
    }

    let mut w = 0u32;
    while w < k {
        x[base + w as usize] = stage[(w * wg_size + tx) as usize];
        w += 1u32;
    }
}

/// Turn per-column sums of squares into scale factors and their inverses.
///
/// One thread per component. A column whose norm is not positive gets a factor
/// of one in both directions, so the pair of [`fn@scale_columns_gpu`] calls
/// leaves it untouched. That reproduces the CPU path, which skips such a column
/// rather than dividing by zero.
///
/// ### Params
///
/// * `sq` - Per-column sums of squares `[k]`
/// * `norm` - Output column L2 norms `[k]`, one where the norm collapsed
/// * `inv_norm` - Output reciprocals `[k]`, one where the norm collapsed
/// * `k` - Number of components
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> component index
#[cube(launch_unchecked)]
pub fn hals_norm_factors_gpu<F: Float>(
    sq: &Tensor<F>,
    norm: &mut Tensor<F>,
    inv_norm: &mut Tensor<F>,
    k: u32,
) {
    let c = ABSOLUTE_POS_X;
    if c >= k {
        terminate!();
    }

    let zero = F::new(0.0);
    let one = F::new(1.0);

    let value = sq[c as usize];
    if value > zero {
        let n = F::sqrt(value);
        norm[c as usize] = n;
        inv_norm[c as usize] = one / n;
    } else {
        norm[c as usize] = one;
        inv_norm[c as usize] = one;
    }
}

/// Scale each column of a `[rows, k]` buffer by a per-column factor.
///
/// Flat grid-stride over `rows * k`. Used twice per iteration: `W`'s columns by
/// the reciprocal norms and `H^T`'s columns by the norms, which keeps `W H`
/// invariant while making `W`'s columns unit length. Because `H` is held
/// transposed, rescaling a row of `H` and a column of `W` are the same
/// operation, so one kernel covers both.
///
/// ### Params
///
/// * `x` - Buffer `[rows, k]` row-major, scaled in place
/// * `factors` - Per-column factors `[k]`
/// * `total` - `rows * k`
/// * `k` - Number of components
/// * `wg_size` - Comptime workgroup width
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> element
#[cube(launch_unchecked)]
pub fn scale_columns_gpu<F: Float>(
    x: &mut Tensor<F>,
    factors: &Tensor<F>,
    total: u32,
    k: u32,
    #[comptime] wg_size: u32,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if idx >= total {
        terminate!();
    }

    let c = idx % k;
    x[idx as usize] *= factors[c as usize];
}

/// Per-row dot products of two `[rows, k]` buffers.
///
/// `out[i] = sum_s X[i, s] * Y[i, s]`, so summing `out` gives the Frobenius
/// inner product. One partial per row rather than a device-wide reduction: it
/// keeps each f32 accumulation to `k` terms and lets the host finish in f64,
/// which is what the CPU objective does and what the cancellation in
/// `||V||^2 - 2<A, H> + <B, D>` needs.
///
/// ### Params
///
/// * `x` - First buffer `[rows, k]` row-major
/// * `y` - Second buffer `[rows, k]` row-major
/// * `out` - Per-row partials `[rows]`
/// * `n_rows` - Number of rows
/// * `k` - Number of components
/// * `wg_size` - Comptime workgroup width
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> row
#[cube(launch_unchecked)]
pub fn row_dot_partials_gpu<F: Float>(
    x: &Tensor<F>,
    y: &Tensor<F>,
    out: &mut Tensor<F>,
    n_rows: u32,
    k: u32,
    #[comptime] wg_size: u32,
) {
    let row = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if row >= n_rows {
        terminate!();
    }

    let base = row as usize * k as usize;
    let mut acc = F::new(0.0);
    let mut s = 0u32;
    while s < k {
        acc += x[base + s as usize] * y[base + s as usize];
        s += 1u32;
    }
    out[row as usize] = acc;
}

/// Fill a buffer with a constant.
///
/// Used to seed the refit paths, which start their free factor at `eps` rather
/// than from an initialisation.
///
/// ### Params
///
/// * `x` - Buffer to fill
/// * `value` - Value to write, as a one-element tensor. See the `eps` note on
///   [`fn@hals_sweep_gpu`] for why it is a buffer.
/// * `total` - Number of elements
/// * `wg_size` - Comptime workgroup width
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> element
#[cube(launch_unchecked)]
pub fn fill_constant_gpu<F: Float>(
    x: &mut Tensor<F>,
    value: &Tensor<F>,
    total: u32,
    #[comptime] wg_size: u32,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if idx >= total {
        terminate!();
    }
    x[idx as usize] = value[0];
}

///////////////
// Launchers //
///////////////

/// Dispatch [`fn@hals_sweep_gpu`] at the rank tier for `k`.
///
/// ### Params
///
/// * `x` - Factor `[n_rows, k]`, updated in place
/// * `g` - Symmetric Gram `[k, k]`
/// * `p` - Data product `[n_rows, k]`
/// * `n_rows` - Rows of `x`
/// * `k` - Number of components
/// * `eps` - Non-negativity floor, a one-element tensor
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, with `x` swept in place.
///
/// ### Errors
///
/// * `GpuNmfRankTooLarge` if `k` is above [`NMF_MAX_RANK`].
/// * `CubeclUtils` if the staged block does not fit the device's shared memory
///   or the grid busts the cube-count limit.
pub fn launch_hals_sweep<R, F>(
    x: &GpuTensor<R, F>,
    g: &GpuTensor<R, F>,
    p: &GpuTensor<R, F>,
    eps: &GpuTensor<R, F>,
    n_rows: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let (k_cap, wg) = sweep_tier(k).ok_or(BixverseErrors::GpuNmfRankTooLarge {
        k,
        max: NMF_MAX_RANK,
    })?;

    let limits = GpuLimits::from_client(client);
    fits_shared_memory(
        "hals_sweep_gpu",
        (k_cap * wg) as usize * size_of::<F>(),
        &limits,
    )?;

    let blocks = (n_rows as u32).div_ceil(wg);
    let (gx, gy) = grid_2d(blocks, &limits)?;
    let count = checked_cube_count("hals_sweep_gpu", gx, gy, 1, &limits)?;

    macro_rules! dispatch {
        ($cap:expr, $wg:expr) => {
            unsafe {
                hals_sweep_gpu::launch_unchecked::<F, R>(
                    client,
                    count,
                    CubeDim::new_1d($wg),
                    x.clone().into_tensor_arg(),
                    g.clone().into_tensor_arg(),
                    p.clone().into_tensor_arg(),
                    eps.clone().into_tensor_arg(),
                    n_rows as u32,
                    k as u32,
                    $cap,
                    $wg,
                );
            }
        };
    }

    match k_cap {
        16 => dispatch!(16, WORKGROUP_256),
        32 => dispatch!(32, WORKGROUP_128),
        64 => dispatch!(64, WORKGROUP_64),
        _ => dispatch!(128, WORKGROUP_32),
    }

    Ok(())
}

/// Dispatch [`fn@hals_norm_factors_gpu`]. One thread per component.
///
/// ### Params
///
/// * `sq` - Per-column sums of squares `[k]`
/// * `norm` - Output norms `[k]`
/// * `inv_norm` - Output reciprocals `[k]`
/// * `k` - Number of components
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CubeclUtils` if the grid busts the cube-count limit.
pub fn launch_hals_norm_factors<R, F>(
    sq: &GpuTensor<R, F>,
    norm: &GpuTensor<R, F>,
    inv_norm: &GpuTensor<R, F>,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let blocks = (k as u32).div_ceil(NMF_ELEMENTWISE_WG);
    let count = checked_cube_count("hals_norm_factors_gpu", blocks, 1, 1, &limits)?;

    unsafe {
        hals_norm_factors_gpu::launch_unchecked::<F, R>(
            client,
            count,
            CubeDim::new_1d(NMF_ELEMENTWISE_WG),
            sq.clone().into_tensor_arg(),
            norm.clone().into_tensor_arg(),
            inv_norm.clone().into_tensor_arg(),
            k as u32,
        );
    }

    Ok(())
}

/// Dispatch [`fn@scale_columns_gpu`] over a `[n_rows, k]` buffer.
///
/// ### Params
///
/// * `x` - Buffer `[n_rows, k]`, scaled in place
/// * `factors` - Per-column factors `[k]`
/// * `n_rows` - Rows of `x`
/// * `k` - Number of components
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CubeclUtils` if the grid busts the cube-count limit.
pub fn launch_scale_columns<R, F>(
    x: &GpuTensor<R, F>,
    factors: &GpuTensor<R, F>,
    n_rows: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let total = (n_rows * k) as u32;
    let (gx, gy) = grid_2d(total.div_ceil(NMF_ELEMENTWISE_WG), &limits)?;
    let count = checked_cube_count("scale_columns_gpu", gx, gy, 1, &limits)?;

    unsafe {
        scale_columns_gpu::launch_unchecked::<F, R>(
            client,
            count,
            CubeDim::new_1d(NMF_ELEMENTWISE_WG),
            x.clone().into_tensor_arg(),
            factors.clone().into_tensor_arg(),
            total,
            k as u32,
            NMF_ELEMENTWISE_WG,
        );
    }

    Ok(())
}

/// Dispatch [`fn@row_dot_partials_gpu`]. One thread per row.
///
/// ### Params
///
/// * `x` - First buffer `[n_rows, k]`
/// * `y` - Second buffer `[n_rows, k]`
/// * `out` - Per-row partials `[n_rows]`
/// * `n_rows` - Number of rows
/// * `k` - Number of components
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CubeclUtils` if the grid busts the cube-count limit.
pub fn launch_row_dot_partials<R, F>(
    x: &GpuTensor<R, F>,
    y: &GpuTensor<R, F>,
    out: &GpuTensor<R, F>,
    n_rows: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d((n_rows as u32).div_ceil(NMF_ELEMENTWISE_WG), &limits)?;
    let count = checked_cube_count("row_dot_partials_gpu", gx, gy, 1, &limits)?;

    unsafe {
        row_dot_partials_gpu::launch_unchecked::<F, R>(
            client,
            count,
            CubeDim::new_1d(NMF_ELEMENTWISE_WG),
            x.clone().into_tensor_arg(),
            y.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            n_rows as u32,
            k as u32,
            NMF_ELEMENTWISE_WG,
        );
    }

    Ok(())
}

/// Dispatch [`fn@fill_constant_gpu`].
///
/// ### Params
///
/// * `x` - Buffer to fill
/// * `value` - Value to write, a one-element tensor
/// * `total` - Number of elements
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CubeclUtils` if the grid busts the cube-count limit.
pub fn launch_fill_constant<R, F>(
    x: &GpuTensor<R, F>,
    value: &GpuTensor<R, F>,
    total: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d((total as u32).div_ceil(NMF_ELEMENTWISE_WG), &limits)?;
    let count = checked_cube_count("fill_constant_gpu", gx, gy, 1, &limits)?;

    unsafe {
        fill_constant_gpu::launch_unchecked::<F, R>(
            client,
            count,
            CubeDim::new_1d(NMF_ELEMENTWISE_WG),
            x.clone().into_tensor_arg(),
            value.clone().into_tensor_arg(),
            total as u32,
            NMF_ELEMENTWISE_WG,
        );
    }

    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use faer::{Mat, MatRef};

    use crate::gpu::linalg::spmm::launch_dense_column_sq_norm;
    use crate::methods::nmf_hals::{
        gram_h_ht, gram_wt_w, hals_sweep_cols, hals_sweep_rows, normalise_w_cols,
    };

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

    /////////////
    // Helpers //
    /////////////

    /// Deterministic non-negative matrix, `rows x cols`, faer column-major.
    fn build_mat(rows: usize, cols: usize, salt: usize) -> Mat<f32> {
        Mat::from_fn(rows, cols, |i, j| {
            (((i * 37 + j * 17 + salt * 7) % 23) as f32) * 0.1 + 0.05
        })
    }

    /// Flatten a faer `rows x k` matrix into row-major `[rows, k]`.
    fn to_row_major(x: MatRef<f32>) -> Vec<f32> {
        let (rows, k) = (x.nrows(), x.ncols());
        let mut out = vec![0f32; rows * k];
        for i in 0..rows {
            for c in 0..k {
                out[i * k + c] = x[(i, c)];
            }
        }
        out
    }

    /// Compare against a reference to a relative tolerance floored at one, so
    /// near-zero entries do not demand impossible precision. The all-zeros
    /// guard is the one that matters: a rejected dispatch leaves the output
    /// untouched and reports nothing, so without it a dead kernel passes.
    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        assert!(
            got.iter().any(|v| v.abs() > 1e-6),
            "output is all zeros, the GPU did no work"
        );
        for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() <= tol * b.abs().max(1.0),
                "index {i}: got {a} want {b}"
            );
        }
    }

    /// Run one GPU sweep over a `[rows, k]` factor and read the result back.
    fn run_sweep(x: &[f32], g: &[f32], p: &[f32], rows: usize, k: usize, eps: f32) -> Vec<f32> {
        let device = try_device().expect("no device");
        let client = WgpuRuntime::client(&device);

        let x_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(x, vec![rows, k], &client).unwrap();
        let g_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(g, vec![k, k], &client).unwrap();
        let p_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(p, vec![rows, k], &client).unwrap();
        let eps_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[eps], vec![1], &client).unwrap();

        launch_hals_sweep::<WgpuRuntime, f32>(&x_gpu, &g_gpu, &p_gpu, &eps_gpu, rows, k, &client)
            .unwrap();

        x_gpu.read(&client).unwrap()
    }

    //////////////////
    // Sweep parity //
    //////////////////

    // The H sweep. `H` is `k x n` on the CPU and `[n, k]` on device, so the
    // reference is transposed before comparing.
    #[test]
    fn test_sweep_matches_cpu_row_sweep() {
        if try_device().is_none() {
            return;
        }
        let (n, k) = (61usize, 7usize);
        let eps = 1e-10f32;

        // B has to be a plausible symmetric Gram with a positive diagonal.
        let w = build_mat(29, k, 1);
        let mut b = Mat::<f32>::zeros(k, k);
        gram_wt_w(w.as_ref(), &mut b);

        let mut h = build_mat(k, n, 2);
        let a = build_mat(k, n, 3);

        let got = run_sweep(
            &to_row_major(h.transpose()),
            &to_row_major(b.as_ref()),
            &to_row_major(a.transpose()),
            n,
            k,
            eps,
        );

        hals_sweep_rows(&mut h, b.as_ref(), a.as_ref(), eps);
        assert_close(&got, &to_row_major(h.transpose()), 1e-5);
    }

    // The W sweep, through the same kernel. This pins the claim in the module
    // doc: the row and column sweeps differ only in which way the symmetric
    // Gram is walked, so one kernel serves both.
    #[test]
    fn test_sweep_matches_cpu_col_sweep() {
        if try_device().is_none() {
            return;
        }
        let (m, k) = (53usize, 7usize);
        let eps = 1e-10f32;

        let h = build_mat(k, 31, 4);
        let mut d = Mat::<f32>::zeros(k, k);
        gram_h_ht(h.as_ref(), &mut d);

        let mut w = build_mat(m, k, 5);
        let c = build_mat(m, k, 6);

        let w_flat = to_row_major(w.as_ref());
        let got = run_sweep(
            &w_flat,
            &to_row_major(d.as_ref()),
            &to_row_major(c.as_ref()),
            m,
            k,
            eps,
        );

        hals_sweep_cols(&mut w, d.as_ref(), c.as_ref(), eps);
        assert_close(&got, &to_row_major(w.as_ref()), 1e-5);
    }

    // A collapsed component leaves its Gram diagonal at zero, which both paths
    // skip rather than divide by.
    #[test]
    fn test_sweep_skips_zero_gram_diagonal() {
        if try_device().is_none() {
            return;
        }
        let (m, k) = (40usize, 5usize);
        let eps = 1e-10f32;

        let mut d = Mat::<f32>::identity(k, k);
        d[(2, 2)] = 0.0;

        let mut w = build_mat(m, k, 8);
        let c = build_mat(m, k, 9);

        let w_flat = to_row_major(w.as_ref());
        let got = run_sweep(
            &w_flat,
            &to_row_major(d.as_ref()),
            &to_row_major(c.as_ref()),
            m,
            k,
            eps,
        );

        hals_sweep_cols(&mut w, d.as_ref(), c.as_ref(), eps);
        assert_close(&got, &to_row_major(w.as_ref()), 1e-5);

        // Component 2 stays exactly what it was, rather than being floored.
        for i in 0..m {
            assert_eq!(got[i * k + 2], w_flat[i * k + 2]);
        }
    }

    // The floor is at eps, not at zero. Driving the update strongly negative
    // has to land on eps exactly.
    #[test]
    fn test_sweep_floors_at_eps() {
        if try_device().is_none() {
            return;
        }
        let (m, k) = (16usize, 3usize);
        let eps = 1e-3f32;

        let d = Mat::<f32>::identity(k, k);
        let w = Mat::<f32>::full(m, k, 0.5f32);
        let c = Mat::<f32>::full(m, k, -100.0f32);

        let got = run_sweep(
            &to_row_major(w.as_ref()),
            &to_row_major(d.as_ref()),
            &to_row_major(c.as_ref()),
            m,
            k,
            eps,
        );

        for (i, &v) in got.iter().enumerate() {
            assert_eq!(v, eps, "index {i} was not floored to eps");
        }
    }

    // Every rank tier compiles and agrees with the CPU. The ranks straddle each
    // tier boundary, so the dispatch lands on a different shader each time.
    #[test]
    fn test_sweep_every_rank_tier() {
        if try_device().is_none() {
            return;
        }
        let eps = 1e-10f32;
        for k in [16usize, 17, 32, 33, 64, 65, 128] {
            let m = 3 * k + 5;
            let h = build_mat(k, 2 * k + 3, k);
            let mut d = Mat::<f32>::zeros(k, k);
            gram_h_ht(h.as_ref(), &mut d);

            let mut w = build_mat(m, k, k + 1);
            let c = build_mat(m, k, k + 2);

            let got = run_sweep(
                &to_row_major(w.as_ref()),
                &to_row_major(d.as_ref()),
                &to_row_major(c.as_ref()),
                m,
                k,
                eps,
            );

            hals_sweep_cols(&mut w, d.as_ref(), c.as_ref(), eps);
            assert_close(&got, &to_row_major(w.as_ref()), 1e-4);
        }
    }

    #[test]
    fn test_sweep_rejects_rank_above_the_largest_tier() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);
        let k = NMF_MAX_RANK + 1;
        let x = GpuTensor::<WgpuRuntime, f32>::empty(vec![4, k], &client).unwrap();
        let g = GpuTensor::<WgpuRuntime, f32>::empty(vec![k, k], &client).unwrap();
        let eps = GpuTensor::<WgpuRuntime, f32>::from_slice(&[1e-10], vec![1], &client).unwrap();

        let err = launch_hals_sweep::<WgpuRuntime, f32>(&x, &g, &x, &eps, 4, k, &client)
            .expect_err("rank above the largest tier must error");
        assert!(matches!(
            err,
            BixverseErrors::GpuNmfRankTooLarge { k: got, .. } if got == k
        ));
    }

    //////////////////////////
    // Normalisation parity //
    //////////////////////////

    #[test]
    fn test_normalisation_matches_cpu() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (37usize, 23usize, 6usize);
        let mut w = build_mat(m, k, 11);
        let mut h = build_mat(k, n, 12);

        let w_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &to_row_major(w.as_ref()),
            vec![m, k],
            &client,
        )
        .unwrap();
        let h_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &to_row_major(h.transpose()),
            vec![n, k],
            &client,
        )
        .unwrap();
        let sq = GpuTensor::<WgpuRuntime, f32>::empty(vec![k], &client).unwrap();
        let norm = GpuTensor::<WgpuRuntime, f32>::empty(vec![k], &client).unwrap();
        let inv = GpuTensor::<WgpuRuntime, f32>::empty(vec![k], &client).unwrap();

        launch_dense_column_sq_norm::<WgpuRuntime, f32>(&w_gpu, &sq, m, k, &client).unwrap();
        launch_hals_norm_factors::<WgpuRuntime, f32>(&sq, &norm, &inv, k, &client).unwrap();
        launch_scale_columns::<WgpuRuntime, f32>(&w_gpu, &inv, m, k, &client).unwrap();
        launch_scale_columns::<WgpuRuntime, f32>(&h_gpu, &norm, n, k, &client).unwrap();

        let w_got = w_gpu.read(&client).unwrap();
        let h_got = h_gpu.read(&client).unwrap();

        normalise_w_cols(&mut w, &mut h);

        assert_close(&w_got, &to_row_major(w.as_ref()), 1e-5);
        assert_close(&h_got, &to_row_major(h.transpose()), 1e-5);

        // The point of the step: unit column norms on W.
        for c in 0..k {
            let sq: f32 = (0..m).map(|i| w_got[i * k + c] * w_got[i * k + c]).sum();
            assert!(
                (sq.sqrt() - 1.0).abs() < 1e-4,
                "column {c} is not unit norm"
            );
        }
    }

    // A collapsed column is left alone rather than divided by zero.
    #[test]
    fn test_norm_factors_leave_a_collapsed_column_alone() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let k = 4usize;
        let sq_host = [4.0f32, 0.0, 9.0, 0.0];
        let sq = GpuTensor::<WgpuRuntime, f32>::from_slice(&sq_host, vec![k], &client).unwrap();
        let norm = GpuTensor::<WgpuRuntime, f32>::empty(vec![k], &client).unwrap();
        let inv = GpuTensor::<WgpuRuntime, f32>::empty(vec![k], &client).unwrap();

        launch_hals_norm_factors::<WgpuRuntime, f32>(&sq, &norm, &inv, k, &client).unwrap();

        assert_close(&norm.read(&client).unwrap(), &[2.0, 1.0, 3.0, 1.0], 1e-6);
        assert_close(
            &inv.read(&client).unwrap(),
            &[0.5, 1.0, 1.0 / 3.0, 1.0],
            1e-6,
        );
    }

    ////////////////////////
    // Objective partials //
    ////////////////////////

    #[test]
    fn test_row_dot_partials_sum_to_the_frobenius_inner_product() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (rows, k) = (301usize, 9usize);
        let x_flat = to_row_major(build_mat(rows, k, 13).as_ref());
        let y_flat = to_row_major(build_mat(rows, k, 14).as_ref());

        let x_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&x_flat, vec![rows, k], &client).unwrap();
        let y_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&y_flat, vec![rows, k], &client).unwrap();
        let out = GpuTensor::<WgpuRuntime, f32>::empty(vec![rows], &client).unwrap();

        launch_row_dot_partials::<WgpuRuntime, f32>(&x_gpu, &y_gpu, &out, rows, k, &client)
            .unwrap();

        let got = out.read(&client).unwrap();
        assert!(
            got.iter().any(|v| v.abs() > 1e-6),
            "output is all zeros, the GPU did no work"
        );

        let total: f64 = got.iter().map(|&v| v as f64).sum();
        let want: f64 = x_flat
            .iter()
            .zip(y_flat.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        assert!(
            (total - want).abs() <= 1e-4 * want.abs().max(1.0),
            "got {total} want {want}"
        );
    }

    #[test]
    fn test_fill_constant() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let total = 1000usize;
        let x = GpuTensor::<WgpuRuntime, f32>::empty(vec![total], &client).unwrap();
        let v = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.25f32], vec![1], &client).unwrap();
        launch_fill_constant::<WgpuRuntime, f32>(&x, &v, total, &client).unwrap();
        assert!(
            x.read(&client).unwrap().iter().all(|&v| v == 0.25),
            "buffer was not filled"
        );
    }

    ////////////////
    // Structural //
    ////////////////

    // The staged block is the same size at every rank tier, and two workgroups
    // stay resident on the 32 KiB floor. Fitting and being fast are different
    // questions: a footprint that fits perfectly while leaving only one resident
    // workgroup roughly halves throughput.
    #[test]
    fn test_sweep_shared_memory_budget_and_residency() {
        for k in [1usize, 16, 17, 32, 33, 64, 65, 128] {
            let (k_cap, wg) = sweep_tier(k).expect("tier must exist");
            assert_eq!(
                k_cap * wg,
                NMF_SWEEP_SMEM_ELEMS,
                "tier for k = {k} does not hold the staged block constant"
            );
            assert!(wg >= WORKGROUP_32, "tier for k = {k} is below one plane");
            assert!(k as u32 <= k_cap, "tier for k = {k} does not fit the rank");

            let footprint = (k_cap * wg) as usize * size_of::<f32>();
            assert_eq!(footprint, 16 * 1024);
            assert!(
                32 * 1024 / footprint >= 2,
                "only one workgroup would be resident at k = {k}"
            );
        }
        assert!(sweep_tier(0).is_none());
        assert!(sweep_tier(NMF_MAX_RANK + 1).is_none());
    }

    // Host-only, no device needed. The sweep grid is proportional to a data
    // dimension, which is exactly the case that busts the 65535-per-dimension
    // dispatch limit in production while passing every benchmarked shape.
    #[test]
    fn test_sweep_grid_within_dispatch_limit() {
        let (max_x, max_y, _) = WgpuRuntime::max_cube_count();
        let cap = max_x.min(max_y);

        // The threshold is real: a flat grid busts the limit at the widest
        // workgroup from 16.8M rows, and at the narrowest already from 2.1M.
        assert!((16_800_000u32).div_ceil(sweep_tier(16).unwrap().1) > max_x);
        assert!((2_100_000u32).div_ceil(sweep_tier(128).unwrap().1) > max_x);

        for rows in [1_000_000u32, 4_000_000, 16_800_000] {
            for k in [16usize, 128] {
                let (_, wg) = sweep_tier(k).unwrap();
                let blocks = rows.div_ceil(wg);
                let (gx, gy) = grid_2d_limited(blocks, cap).unwrap();
                assert!(gx <= max_x && gy <= max_y);
                assert!(gx as u64 * gy as u64 >= blocks as u64, "grid misses work");
            }
        }
    }

    // The elementwise and per-row passes have the same exposure.
    #[test]
    fn test_elementwise_grid_within_dispatch_limit() {
        let (max_x, max_y, _) = WgpuRuntime::max_cube_count();
        let cap = max_x.min(max_y);
        for total in [1_000_000u32, 100_000_000, 1_000_000_000] {
            let blocks = total.div_ceil(NMF_ELEMENTWISE_WG);
            let (gx, gy) = grid_2d_limited(blocks, cap).unwrap();
            assert!(gx <= max_x && gy <= max_y);
            assert!(gx as u64 * gy as u64 >= blocks as u64, "grid misses work");
        }
    }
}
