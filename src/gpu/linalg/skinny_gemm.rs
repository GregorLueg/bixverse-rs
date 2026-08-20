//! Register-tiled GEMM for a tall-skinny output: `C[rows, k] = A * B[len, k]`
//! with `k` small.
//!
//! This is the shape both data products of a HALS NMF iteration reduce to. `V` is
//! `m x n` uploaded as `V^T`, and one buffer serves both directions:
//!
//! * `(W^T V)^T = V^T W`, so `rows = n`, `len = m`, `A = V^T` read as stored
//! * `V H^T`, so `rows = m`, `len = n`, `A = V` read with its indices swapped
//!
//! The `a_transposed` flag picks between those two, which is why one kernel
//! covers both and why the staging decode differs between them: whichever axis
//! is contiguous in the stored buffer has to be the one adjacent threads walk.
//!
//! ### Why not cubek
//!
//! Measured on an M1 Max, 60 HALS iterations per shape, whole-solve wall clock
//! against the CPU solver in the same pass:
//!
//! | shape | cubek `SimpleUnit` | this kernel |
//! |---|---|---|
//! | 500 x 20000, k = 10 | 0.18x, 12.8 GFLOP/s (0.1% peak) | **2.04x**, 210 GFLOP/s |
//! | 5000 x 3000, k = 30 | does not launch | **2.56x**, 462 GFLOP/s |
//!
//! `Strategy::Auto` and a pinned `Strategy::SimpleUnit` both ask for 40960 bytes
//! of shared memory against the 32768 this device offers, so the second shape is
//! rejected outright and the CubeCL server thread dies with it. On the one shape
//! where cubek does run it managed 0.1% of peak FLOPs and 0.6% of bandwidth,
//! which put the whole GPU solve five and a half times *slower* than the CPU it
//! was supposed to accelerate.
//!
//! That is the same wrong-algorithm-for-the-shape problem recorded in
//! [`crate::gpu::linalg::gram`] and on `gram_partial` in
//! [`crate::gpu::linalg::cholesky_gpu`], for the same reason. One thread per
//! output element with a serial reduction has nothing to work with when the
//! output is `500 x 10` and the reduction is 20000 long.
//!
//! What is left is not a compute or a bandwidth wall: at 2.0% and 4.4% of peak
//! FLOPs against 10.5% and 7.7% of bandwidth, the kernel is bound by neither.
//! That is the issue-bound regime, where the lever is the register tile rather
//! than traffic, and the current 4 x 4 puts the inner loop at 0.5 memory
//! operations per FMA. Widening it further has not been swept.
//!
//! ### Structure
//!
//! Standard register-tiled GEMM. A workgroup owns a `SG_BM x SG_BN` block of the
//! output and one chunk of the reduction, stages `SG_BK` steps of both operands
//! in shared memory, and each thread accumulates an `SG_RT_M x SG_RT_N` register
//! tile. That puts the inner loop at `(RT_M + RT_N) / (RT_M * RT_N)` = 0.5 memory
//! operations per fused multiply-add, against 1.0 or worse for one-thread-per-
//! output.
//!
//! The reduction splits over chunks when there are too few output tiles to
//! saturate the device, which is the common case here rather than the exception:
//! at `rows = 500`, `k = 10` the output is four tiles. Partials are
//! `[chunks, rows, k]` and [`crate::gpu::linalg::gram::gram_reduce()`] sums them,
//! since that is exactly a `[chunks, X] -> [X]` reduction and it already carries
//! the two-dimensional grid decode that a large `X` needs.

#![allow(missing_docs)]

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;

use crate::gpu::linalg::gram::gram_reduce;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Output rows per workgroup.
const SG_BM: u32 = 128;

/// Output columns per workgroup, i.e. the slice of the `k` axis it owns.
///
/// Fixed rather than tiered to the rank so there is one shader instead of four.
/// At `k = 10` that wastes six of sixteen columns; at `k = 30` it takes two
/// column blocks and wastes two. Both are cheaper than a shader per rank, and
/// the waste is in the narrow axis, so it never touches the `A` traffic that
/// dominates.
const SG_BN: u32 = 16;

/// Reduction steps staged per shared-memory round.
///
/// Kept at 8 rather than widened for coalescing. The staging footprint is
/// `SG_BK * (SG_BM + SG_BN) * 4`, so 8 leaves seven workgroups resident on a
/// 32 KiB device and 32 would leave one. Occupancy beats coalescing here by a
/// wide margin; the same sweep on the Gram kernel measured 19.8 / 31.5 / 29.8 ms
/// for widths 8 / 16 / 32.
const SG_BK: u32 = 8;

/// Register tile height per thread.
const SG_RT_M: u32 = 4;

/// Register tile width per thread.
const SG_RT_N: u32 = 4;

/// Thread columns per workgroup.
const SG_TN: u32 = SG_BN / SG_RT_N;

/// Thread rows per workgroup.
const SG_TM: u32 = SG_BM / SG_RT_M;

/// Workgroup width. 128 threads, i.e. four planes on Apple Silicon.
pub const SG_THREADS: u32 = SG_TM * SG_TN;

/// Output tiles the split-K heuristic aims to dispatch before it stops splitting.
const SG_TARGET_CUBES: usize = 2048;

/// Hard cap on reduction chunks.
const SG_MAX_CHUNKS: usize = 256;

/// Shortest reduction worth giving its own chunk.
const SG_MIN_LEN_PER_CHUNK: usize = 512;

/// Ceiling on the partials buffer, in elements, implied by the other two bounds.
///
/// There is deliberately no separate memory bound in [`skinny_chunks`], because
/// one could never fire. The split only happens when `tiles < SG_TARGET_CUBES`,
/// and `rows * k <= tiles * SG_BM * SG_BN`, so
///
/// ```text
/// chunks * rows * k  <=  ceil(TARGET / tiles) * tiles * BM * BN
///                    <=  TARGET * BM * BN  +  rows * k
///                    <=  2 * TARGET * BM * BN
/// ```
///
/// where the middle step is the slack the ceiling in `wanted` adds. At f32 that
/// is 33.6 MB, whatever the shape. The heuristic test sweeps shapes against this
/// bound, which is how the dead memory guard the heuristic originally carried was
/// found.
///
/// Only the test that enforces it reads it, hence the gate: an `allow(dead_code)`
/// in `src/` would hide the next constant that really is orphaned.
#[cfg(test)]
const SG_MAX_PARTIAL_ELEMS: usize = 2 * SG_TARGET_CUBES * (SG_BM * SG_BN) as usize;

/// Workgroup width of [`gram_reduce`], which sums the split-K partials.
const SG_REDUCE_WG: u32 = 256;

/////////////
// Helpers //
/////////////

/// Split-K chunk count for one product.
///
/// Two bounds, and each binds for some reachable shape. `wanted` is how many
/// chunks it takes to fill the device given how few output tiles there are, and
/// `by_len` stops a short reduction being cut into slivers.
///
/// There is no third, memory bound. See `SG_MAX_PARTIAL_ELEMS`: `wanted`
/// already caps the partials buffer, so a memory ceiling could never fire.
///
/// ### Params
///
/// * `rows` - Output rows
/// * `k` - Output columns
/// * `len` - Reduction length
///
/// ### Returns
///
/// Number of reduction chunks, at least one and at most `SG_MAX_CHUNKS`.
pub fn skinny_chunks(rows: usize, k: usize, len: usize) -> usize {
    let tiles = rows.div_ceil(SG_BM as usize) * k.div_ceil(SG_BN as usize);
    if tiles >= SG_TARGET_CUBES {
        return 1;
    }
    let wanted = SG_TARGET_CUBES.div_ceil(tiles.max(1));
    let by_len = (len / SG_MIN_LEN_PER_CHUNK).max(1);
    wanted.min(by_len).clamp(1, SG_MAX_CHUNKS)
}

/// Elements a caller-owned partials buffer needs for one product.
///
/// ### Params
///
/// * `rows` - Output rows
/// * `k` - Output columns
/// * `len` - Reduction length
///
/// ### Returns
///
/// `chunks * rows * k`, or zero when the product runs in a single chunk and
/// writes straight into its output.
pub fn skinny_partial_elems(rows: usize, k: usize, len: usize) -> usize {
    let chunks = skinny_chunks(rows, k, len);
    if chunks > 1 { chunks * rows * k } else { 0 }
}

/////////////
// Kernels //
/////////////

/// Partial `C[rows, k] = A * B[len, k]` over one chunk of the reduction.
///
/// `A` is read out of a single `[len, rows]` or `[rows, len]` row-major buffer
/// according to `a_transposed`:
///
/// ```text
/// a_transposed = false   A[row, t] = a[row * len + t]
/// a_transposed = true    A[row, t] = a[t * rows + row]
/// ```
///
/// The staging decode follows the flag, so adjacent threads always walk whichever
/// axis is contiguous in the stored buffer. Getting that backwards costs more
/// than the register tile saves: the buffer layout decides which axis goes on the
/// thread index, and that has to be settled before anything else.
///
/// ### Params
///
/// * `a` - The `A` operand's backing buffer, row-major
/// * `b` - Right operand `[len, k]`, row-major
/// * `out` - Output `[chunks, rows, k]`, or `[rows, k]` when there is one chunk
/// * `rows` - Output rows
/// * `k` - Output columns
/// * `len` - Reduction length
/// * `len_per_chunk` - Reduction steps per chunk; the last chunk is short
/// * `n_col_blocks` - `ceil(k / SG_BN)`, used to decode the packed z axis
/// * `a_transposed` - Which way to index `a` (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> row block
/// * `CUBE_POS_Z % n_col_blocks` -> column block,
///   `CUBE_POS_Z / n_col_blocks` -> reduction chunk
///
/// Row blocks are flattened over x and y because that count is proportional to a
/// data dimension. Column blocks and chunks are both bounded small, so packing
/// them together onto z cannot approach the per-dimension dispatch limit.
#[cube(launch_unchecked)]
pub fn skinny_gemm_partial<F: Float>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    out: &mut Tensor<F>,
    rows: u32,
    k: u32,
    len: u32,
    len_per_chunk: u32,
    n_col_blocks: u32,
    #[comptime] a_transposed: bool,
) {
    let row_block = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let col_block = CUBE_POS_Z % n_col_blocks;
    let chunk = CUBE_POS_Z / n_col_blocks;

    let i0 = row_block * SG_BM;
    let j0 = col_block * SG_BN;

    let mut sa = SharedMemory::<F>::new((SG_BK * SG_BM) as usize);
    let mut sb = SharedMemory::<F>::new((SG_BK * SG_BN) as usize);

    if i0 >= rows {
        terminate!();
    }

    let t_start = chunk * len_per_chunk;
    let mut t_end = t_start + len_per_chunk;
    if t_end > len {
        t_end = len;
    }

    let tid = UNIT_POS_X;
    let tm = tid / SG_TN;
    let tn = tid % SG_TN;

    let zero = F::new(0.0);

    let mut acc = Array::<F>::new((SG_RT_M * SG_RT_N) as usize);
    #[unroll]
    for p in 0..SG_RT_M {
        #[unroll]
        for q in 0..SG_RT_N {
            acc[(p * SG_RT_N + q) as usize] = zero;
        }
    }

    let mut t = t_start;
    while t < t_end {
        // Stage A. Out-of-range entries are zero-filled so the inner loop needs
        // no guard. The decode differs by orientation: adjacent threads must walk
        // the axis that is contiguous in `a`, or the global reads scatter.
        let mut li = tid;
        while li < SG_BK * SG_BM {
            let kk = if a_transposed { li / SG_BM } else { li % SG_BK };
            let mm = if a_transposed { li % SG_BM } else { li / SG_BK };
            let tt = t + kk;
            let rr = i0 + mm;
            let mut v = zero;
            if tt < t_end && rr < rows {
                if a_transposed {
                    v = a[(tt * rows + rr) as usize];
                } else {
                    v = a[(rr * len + tt) as usize];
                }
            }
            sa[(kk * SG_BM + mm) as usize] = v;
            li += SG_THREADS;
        }

        // Stage B. `b` is `[len, k]` row-major, so adjacent threads take adjacent
        // columns.
        let mut lj = tid;
        while lj < SG_BK * SG_BN {
            let kk = lj / SG_BN;
            let nn = lj % SG_BN;
            let tt = t + kk;
            let cc = j0 + nn;
            let mut v = zero;
            if tt < t_end && cc < k {
                v = b[(tt * k + cc) as usize];
            }
            sb[lj as usize] = v;
            lj += SG_THREADS;
        }
        sync_cube();

        #[unroll]
        for kk in 0..SG_BK {
            let arow = kk * SG_BM + tm * SG_RT_M;
            let brow = kk * SG_BN + tn * SG_RT_N;
            let mut av = Array::<F>::new(SG_RT_M as usize);
            let mut bv = Array::<F>::new(SG_RT_N as usize);
            #[unroll]
            for p in 0..SG_RT_M {
                av[p as usize] = sa[(arow + p) as usize];
            }
            #[unroll]
            for q in 0..SG_RT_N {
                bv[q as usize] = sb[(brow + q) as usize];
            }
            #[unroll]
            for p in 0..SG_RT_M {
                #[unroll]
                for q in 0..SG_RT_N {
                    acc[(p * SG_RT_N + q) as usize] += av[p as usize] * bv[q as usize];
                }
            }
        }
        sync_cube();

        t += SG_BK;
    }

    let base = chunk * rows * k;
    #[unroll]
    for p in 0..SG_RT_M {
        let rr = i0 + tm * SG_RT_M + p;
        if rr < rows {
            #[unroll]
            for q in 0..SG_RT_N {
                let cc = j0 + tn * SG_RT_N + q;
                if cc < k {
                    out[(base + rr * k + cc) as usize] = acc[(p * SG_RT_N + q) as usize];
                }
            }
        }
    }
}

////////////////
// Dispatcher //
////////////////

/// Compute `C[rows, k] = A * B[len, k]`.
///
/// Selects split-K via [`skinny_chunks`]; when it returns one, the kernel
/// accumulates the whole reduction and writes straight into `c` so no reduce is
/// launched and `partials` goes untouched.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `a` - Backing buffer for the `A` operand, `[rows, len]` row-major when
///   `a_transposed` is false and `[len, rows]` when it is true
/// * `b` - Right operand `[len, k]` row-major
/// * `c` - Output `[rows, k]` row-major
/// * `partials` - Caller-owned scratch of at least
///   [`skinny_partial_elems`] elements; unused when the reduction is not split
/// * `rows` - Output rows
/// * `k` - Output columns
/// * `len` - Reduction length
/// * `a_transposed` - Which way to index `a`
///
/// ### Returns
///
/// `Ok(())`, with `c` holding the product.
///
/// ### Errors
///
/// * `GpuBindingTooLarge` if `partials` is smaller than the split needs.
/// * `CubeclUtils` if either grid is over the device's cube-count limit or the
///   staged block does not fit its shared memory.
#[allow(clippy::too_many_arguments)]
pub fn skinny_gemm<R, F>(
    client: &ComputeClient<R>,
    a: &GpuTensor<R, F>,
    b: &GpuTensor<R, F>,
    c: &GpuTensor<R, F>,
    partials: &GpuTensor<R, F>,
    rows: usize,
    k: usize,
    len: usize,
    a_transposed: bool,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    fits_shared_memory(
        "skinny_gemm_partial",
        (SG_BK * (SG_BM + SG_BN)) as usize * size_of::<F>(),
        &limits,
    )?;

    let n_chunks = skinny_chunks(rows, k, len);
    let len_per_chunk = len.div_ceil(n_chunks) as u32;
    let n_col_blocks = (k as u32).div_ceil(SG_BN);
    let row_blocks = (rows as u32).div_ceil(SG_BM);

    let split = n_chunks > 1;
    if split {
        let need = n_chunks * rows * k;
        if partials.len() < need {
            return Err(BixverseErrors::GpuBindingTooLarge {
                buffer: "skinny_gemm partials",
                bytes: need * size_of::<F>(),
                limit: partials.len() * size_of::<F>(),
            });
        }
    }
    let target = if split { partials } else { c };

    let (gx, gy) = grid_2d(row_blocks, &limits)?;
    let count = checked_cube_count(
        "skinny_gemm_partial",
        gx,
        gy,
        n_col_blocks * n_chunks as u32,
        &limits,
    )?;

    macro_rules! dispatch {
        ($transposed:expr) => {
            unsafe {
                skinny_gemm_partial::launch_unchecked::<F, R>(
                    client,
                    count,
                    CubeDim::new_1d(SG_THREADS),
                    a.clone().into_tensor_arg(),
                    b.clone().into_tensor_arg(),
                    target.clone().into_tensor_arg(),
                    rows as u32,
                    k as u32,
                    len as u32,
                    len_per_chunk,
                    n_col_blocks,
                    $transposed,
                );
            }
        };
    }

    if a_transposed {
        dispatch!(true);
    } else {
        dispatch!(false);
    }

    if split {
        let total = (rows * k) as u32;
        let (rx, ry) = grid_2d(total.div_ceil(SG_REDUCE_WG), &limits)?;
        let reduce_count = checked_cube_count("gram_reduce", rx, ry, 1, &limits)?;
        unsafe {
            gram_reduce::launch_unchecked::<F, R>(
                client,
                reduce_count,
                CubeDim::new_1d(SG_REDUCE_WG),
                partials.clone().into_tensor_arg(),
                c.clone().into_tensor_arg(),
                total,
                n_chunks as u32,
            );
        }
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

    /// Deterministic values, signed so a dropped term cannot cancel out of sight.
    fn build(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|t| (((t * 31 + salt * 17) % 23) as f32) * 0.1 - 1.1)
            .collect()
    }

    /// `C = A * B` on the host, row-major, with the same orientation flag.
    fn host_gemm(
        a: &[f32],
        b: &[f32],
        rows: usize,
        k: usize,
        len: usize,
        a_transposed: bool,
    ) -> Vec<f32> {
        let mut out = vec![0f32; rows * k];
        for r in 0..rows {
            for c in 0..k {
                let mut acc = 0f32;
                for t in 0..len {
                    let av = if a_transposed {
                        a[t * rows + r]
                    } else {
                        a[r * len + t]
                    };
                    acc += av * b[t * k + c];
                }
                out[r * k + c] = acc;
            }
        }
        out
    }

    /// Compare to a relative tolerance floored at one, plus the all-zeros guard.
    /// The guard is the one that matters: a rejected dispatch leaves the output
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

    /// Run the dispatcher and read the result back.
    fn run(rows: usize, k: usize, len: usize, a_transposed: bool) -> (Vec<f32>, Vec<f32>) {
        let device = try_device().expect("no device");
        let client = WgpuRuntime::client(&device);

        let a_host = build(rows * len, 1);
        let b_host = build(len * k, 2);

        let a_shape = if a_transposed {
            vec![len, rows]
        } else {
            vec![rows, len]
        };
        let a = GpuTensor::<WgpuRuntime, f32>::from_slice(&a_host, a_shape, &client).unwrap();
        let b = GpuTensor::<WgpuRuntime, f32>::from_slice(&b_host, vec![len, k], &client).unwrap();
        let c = GpuTensor::<WgpuRuntime, f32>::empty(vec![rows, k], &client).unwrap();

        let elems = skinny_partial_elems(rows, k, len).max(1);
        let partials = GpuTensor::<WgpuRuntime, f32>::empty(vec![elems], &client).unwrap();

        skinny_gemm::<WgpuRuntime, f32>(&client, &a, &b, &c, &partials, rows, k, len, a_transposed)
            .unwrap();

        let got = c.read(&client).unwrap();
        (got, host_gemm(&a_host, &b_host, rows, k, len, a_transposed))
    }

    // Ragged in every axis: rows is not a multiple of SG_BM, k not of SG_BN, len
    // not of SG_BK, so every edge tile has a tail and the last staged step is
    // short. Both orientations, because they take different staging decodes.
    #[test]
    fn test_skinny_gemm_ragged_tails() {
        if try_device().is_none() {
            return;
        }
        for transposed in [false, true] {
            let (got, want) = run(200, 22, 37, transposed);
            assert_close(&got, &want, 1e-4);
        }
    }

    // A single output tile with a long reduction, which is the shape the split-K
    // arm exists for. Assert the arm is actually taken before trusting the result.
    #[test]
    fn test_skinny_gemm_split_k_path() {
        if try_device().is_none() {
            return;
        }
        let (rows, k, len) = (64usize, 8usize, 4_096usize);
        assert!(
            skinny_chunks(rows, k, len) > 1,
            "this shape was supposed to take the split-K arm"
        );
        for transposed in [false, true] {
            let (got, want) = run(rows, k, len, transposed);
            assert_close(&got, &want, 1e-3);
        }
    }

    // The single-chunk arm, where the kernel writes straight into the output and
    // no reduce is launched. Enough output tiles that the heuristic stops
    // splitting.
    #[test]
    fn test_skinny_gemm_single_chunk_path() {
        if try_device().is_none() {
            return;
        }
        let (rows, k, len) = (4_096usize, 128usize, 64usize);
        assert_eq!(
            skinny_chunks(rows, k, len),
            1,
            "this shape was supposed to skip the split-K arm"
        );
        let (got, want) = run(rows, k, len, false);
        assert_close(&got, &want, 1e-4);
    }

    // A rank of one exercises the narrowest possible k axis, where fifteen of
    // sixteen staged columns are padding.
    #[test]
    fn test_skinny_gemm_rank_one() {
        if try_device().is_none() {
            return;
        }
        let (got, want) = run(300, 1, 71, true);
        assert_close(&got, &want, 1e-4);
    }

    #[test]
    fn test_skinny_gemm_rejects_a_short_partials_buffer() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (rows, k, len) = (64usize, 8usize, 4_096usize);
        assert!(skinny_chunks(rows, k, len) > 1);

        let a = GpuTensor::<WgpuRuntime, f32>::empty(vec![rows, len], &client).unwrap();
        let b = GpuTensor::<WgpuRuntime, f32>::empty(vec![len, k], &client).unwrap();
        let c = GpuTensor::<WgpuRuntime, f32>::empty(vec![rows, k], &client).unwrap();
        let partials = GpuTensor::<WgpuRuntime, f32>::empty(vec![1], &client).unwrap();

        assert!(matches!(
            skinny_gemm::<WgpuRuntime, f32>(&client, &a, &b, &c, &partials, rows, k, len, false),
            Err(BixverseErrors::GpuBindingTooLarge { .. })
        ));
    }

    ////////////////
    // Structural //
    ////////////////

    // Two workgroups have to stay resident, not merely fit. A footprint that fits
    // perfectly while leaving only one resident roughly halves throughput, so
    // fitting and being fast are different questions.
    #[test]
    fn test_skinny_gemm_shared_memory_within_budget() {
        let footprint = (SG_BK * (SG_BM + SG_BN)) as usize * size_of::<f32>();
        assert!(footprint <= 32 * 1024, "over the shared-memory floor");
        assert!(
            32 * 1024 / footprint >= 2,
            "only one workgroup would be resident"
        );
        assert_eq!(SG_THREADS, 128);
        assert_eq!(SG_BM % SG_RT_M, 0);
        assert_eq!(SG_BN % SG_RT_N, 0);
    }

    // Host-only, no device needed. The row-block count is proportional to a data
    // dimension, and the z axis carries column blocks times chunks. Both are
    // checked past the point where a flat grid would bust the per-dimension
    // dispatch limit, because that failure kills the CubeCL server thread and
    // surfaces as an unrelated error from a later call.
    #[test]
    fn test_skinny_gemm_grid_within_dispatch_limit() {
        let (max_x, max_y, max_z) = WgpuRuntime::max_cube_count();
        let cap = max_x.min(max_y);

        // The threshold is real: a flat row-block grid busts the limit here.
        assert!((16_000_000u32).div_ceil(SG_BM) > max_x);

        for rows in [1_000_000usize, 16_000_000, 100_000_000] {
            let blocks = (rows as u32).div_ceil(SG_BM);
            let (gx, gy) = grid_2d_limited(blocks, cap).unwrap();
            assert!(gx <= max_x && gy <= max_y);
            assert!(gx as u64 * gy as u64 >= blocks as u64, "grid misses work");
        }

        // The z axis packs column blocks and chunks together. Both are bounded
        // small by construction, so their product cannot approach the limit.
        let max_col_blocks = 128u32.div_ceil(SG_BN);
        assert!(max_col_blocks * SG_MAX_CHUNKS as u32 <= max_z);
    }

    // Every bound in the split-K heuristic has to bind for some reachable shape,
    // or it is dead code pretending to be a guard.
    #[test]
    fn test_skinny_chunks_each_bound_binds() {
        // Enough output tiles: no split at all.
        assert_eq!(skinny_chunks(1_000_000, 128, 4_096), 1);

        // `wanted` binds: few tiles, long reduction, small output.
        assert!(skinny_chunks(128, 16, 1_000_000) > 1);

        // `by_len` binds: few tiles but a reduction too short to cut up.
        assert_eq!(skinny_chunks(128, 16, 600), 1);

        // The cap binds.
        assert_eq!(skinny_chunks(128, 16, 1_000_000_000), SG_MAX_CHUNKS);

        // And the reason there is no memory bound: over a wide sweep of shapes
        // the partials buffer never approaches the ceiling the split implies, so
        // a memory guard could not fire. This is what killed the third bound the
        // heuristic originally carried.
        for &(rows, k) in &[
            (1usize, 1usize),
            (128, 16),
            (500, 10),
            (3_000, 30),
            (50_000, 30),
            (200_000, 128),
            (1_000_000, 64),
        ] {
            for &len in &[600usize, 4_096, 1_000_000, 1_000_000_000] {
                let elems = skinny_partial_elems(rows, k, len);
                assert!(
                    elems <= SG_MAX_PARTIAL_ELEMS,
                    "{rows}x{k} over {len}: partials {elems} above the implied ceiling"
                );
            }
        }
    }

    #[test]
    fn test_skinny_partial_elems_is_zero_when_unsplit() {
        assert_eq!(skinny_partial_elems(1_000_000, 128, 4_096), 0);
        let (rows, k, len) = (128usize, 16usize, 1_000_000usize);
        assert_eq!(
            skinny_partial_elems(rows, k, len),
            skinny_chunks(rows, k, len) * rows * k
        );
    }
}
