//! Symmetric Gram product `G = A A^T` for a feature-major dense matrix.
//!
//! Given `A` of shape `[d, n]` row-major, i.e. feature `i`'s samples contiguous
//! at `A[i * n ..]`, computes the full `[d, d]` symmetric matrix
//!
//! ```text
//! G[i, j] = sum_r A[i, r] * A[j, r]
//! ```
//!
//! This is what pairwise correlation and covariance reduce to once the columns
//! are centred and scaled. [`crate::gpu::linalg::corr`] is the caller, where the
//! same `[d, n]` buffer goes by `[n_cols, n_rows]`. It is a much narrower
//! problem than a general dense GEMM: one input stream rather than two, a
//! symmetric output, and a fixed layout.
//!
//! cubek has no strategy that exploits any of that. Measured on an
//! M1 Max, its `Strategy::Auto` pick (a `DoubleBufferingMatmulFamily` over
//! `UnitMatmulFamily`, one thread per output element with a serial reduction)
//! runs this at 5-7% of peak on ordinary shapes and **0.13%** at `d = 100,
//! n = 1e6`, where 100 by 100 outputs cannot fill the device at all.
//!
//! ### Structure
//!
//! Standard register-tiled GEMM, specialised three ways:
//!
//! * Both staged operands come from the same buffer.
//! * Only the upper-triangular tiles run; each writes its block and the
//!   transposed mirror, so half the work disappears.
//! * The reduction splits over row chunks when there are too few output tiles
//!   to saturate the device, which is the only thing that varies across the
//!   fat / tall / square / narrow regimes.

// The `#[cube]` macro generates undocumented launcher structs and functions.
#![allow(missing_docs)]

use ann_search_rs::gpu::grid_2d;
use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::prelude::*;

use crate::gpu::checked_cube_count;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Output block edge. Each workgroup owns a `GRAM_BM x GRAM_BM` block of `G`.
///
/// Swept against 32 and 128 on an M1 Max over the four shape regimes, in
/// milliseconds for the product alone:
///
/// | BM / BK / RT | fat | tall | square | narrow |
/// |---|---|---|---|---|
/// | 32 / 8 / 2  | 22.5 | 427.4 | 6.2 | 37.6 |
/// | **64 / 8 / 4**  | **19.8** | **243.1** | **6.2** | **17.4** |
/// | 128 / 8 / 8 | 24.3 | 230.4 | 9.4 | 19.0 |
///
/// 128 looked better on the long-reduction shape until the confound came out:
/// at `d = 2000` it leaves only 136 tiles, which trips [`gram_chunks`] into
/// splitting the reduction, and 64 with the same split applied closes the gap
/// to 1.06x while staying 1.51x ahead on square. 32 wastes the register tile.
const GRAM_BM: u32 = 64;

/// Rows of `A` staged per shared-memory step.
///
/// Sets the coalescing of the staging reads: the global buffer is contiguous
/// along `r`, so consecutive lanes read `GRAM_BK` consecutive floats, i.e. runs
/// of `GRAM_BK * 4` bytes. 8 gives 32-byte runs, 32 gives a full 128-byte cache
/// line.
///
/// Wider is worse, which is the opposite of what the coalescing argument
/// predicts. At `GRAM_BM = 64`, moving from 8 to 16 to 32 measured 19.8 /
/// 31.5 / 29.8 ms on the fat shape and 243 / 374 / 314 ms on the tall one. The
/// staging footprint is `2 * GRAM_BK * GRAM_SMEM_STRIDE * 4` bytes, so those
/// three sit at 7, 3 and 1 resident workgroups in a 32 KiB budget: occupancy
/// beats coalescing here by a wide margin.
const GRAM_BK: u32 = 8;

/// Register tile edge. Each thread accumulates a `GRAM_RT x GRAM_RT` block.
///
/// This is the whole point of the kernel. One output per thread costs 2 shared
/// reads per fused multiply-add; a 4x4 tile costs 8 shared reads per 16 FMAs,
/// i.e. 0.5, which is a 4x cut in memory-operation issue count for the same
/// arithmetic. Dropping to 2x2 costs 1.76x on the tall shape.
const GRAM_RT: u32 = 4;

/// Threads along one edge of the workgroup.
const GRAM_TN: u32 = GRAM_BM / GRAM_RT;

/// Total threads per workgroup.
const GRAM_THREADS: u32 = GRAM_TN * GRAM_TN;

/// Row stride of the shared staging arrays, padded by one float.
///
/// Staging reads coalesced from a feature-major buffer means consecutive lanes
/// write shared addresses `GRAM_BM` apart, which collides on every bank. The
/// classic `+1` makes the stride coprime with the bank count, so the writes go
/// out conflict-free for one extra float per staged row.
const GRAM_SMEM_STRIDE: u32 = GRAM_BM + 1;

/// Workgroups below which the device is considered under-occupied, so the
/// reduction is worth splitting over row chunks.
///
/// Deliberately well above what "fill the cores" suggests. An M1 Max has 32
/// cores holding several workgroups each, so a few hundred ought to be plenty,
/// but the tall shape keeps improving past that: at `d = 2000, n = 50000` this
/// measured 302 / 259 / 250 / 243 ms at 512 / 2048 / 8192 / 32768. The other
/// three regimes do not move, because `GRAM_MIN_ROWS_PER_CHUNK` and
/// [`GRAM_MAX_PARTIAL_BYTES`] hold them at one chunk regardless.
///
/// 2048 rather than the fastest value: past it the gain is 6% and the partials
/// buffer grows 4x, which is a poor trade for a scratch allocation that is
/// already 4x the output it feeds.
const GRAM_TARGET_CUBES: usize = 2048;

/// Upper bound on split-K chunks, independent of memory. Only ever reached at
/// very small `d`, where the partials are tiny.
const GRAM_MAX_CHUNKS: usize = 256;

/// Minimum rows per chunk. Below this the per-chunk fixed cost and the reduce
/// pass outweigh the extra parallelism.
const GRAM_MIN_ROWS_PER_CHUNK: usize = 1024;

/// Ceiling on the split-K partials buffer, `chunks * d * d * size_of::<F>()`.
///
/// This is the term that keeps split-K away from large `d`: the partials grow
/// quadratically while the benefit does not. At `d = 2000` it caps the split at
/// 4 chunks, and by `d = 4000` it forbids splitting altogether, which is also
/// the point where there are enough tiles that splitting stops helping.
const GRAM_MAX_PARTIAL_BYTES: usize = 64 * 1024 * 1024;

/// Threads per workgroup in [`gram_reduce`].
const GRAM_REDUCE_WG: u32 = 256;

/////////////
// Helpers //
/////////////

/// Split-K chunk count for the Gram reduction.
///
/// Splitting only helps when there are too few output tiles to fill the device,
/// which happens for small `d` regardless of how long the reduction is. For
/// everything else this returns 1 and the kernel writes straight into `G`.
///
/// ### Params
///
/// * `n` - Reduction length, i.e. rows of the original matrix
/// * `d` - Output edge, i.e. feature count
/// * `elem_bytes` - Size of one element of the partials buffer
///
/// ### Returns
///
/// Number of row chunks, at least 1 and at most [`GRAM_MAX_CHUNKS`].
fn gram_chunks(n: usize, d: usize, elem_bytes: usize) -> usize {
    let tiles = d.div_ceil(GRAM_BM as usize);
    let triangular = tiles * (tiles + 1) / 2;
    if triangular >= GRAM_TARGET_CUBES {
        return 1;
    }

    let wanted = GRAM_TARGET_CUBES.div_ceil(triangular.max(1));
    let by_rows = (n / GRAM_MIN_ROWS_PER_CHUNK).max(1);
    let by_memory = (GRAM_MAX_PARTIAL_BYTES / (d * d * elem_bytes).max(1)).max(1);

    wanted.min(by_rows).min(by_memory).clamp(1, GRAM_MAX_CHUNKS)
}

/////////////
// Kernels //
/////////////

/// Upper-triangular tile of `G = A A^T` over one chunk of rows.
///
/// Each workgroup owns a `GRAM_BM x GRAM_BM` block of `G` and one row chunk. It
/// stages `GRAM_BK` rows of the two feature slices into shared memory and
/// accumulates their outer products into a per-thread `GRAM_RT x GRAM_RT`
/// register tile, so every staged element feeds `GRAM_RT` fused multiply-adds
/// rather than one.
///
/// Workgroups below the diagonal terminate immediately: the tile above the
/// diagonal writes both its own block and the transposed mirror, so the lower
/// half is redundant. That is exactly half the arithmetic, and an
/// immediately-exiting workgroup is far cheaper than a triangular index decode.
/// Diagonal tiles cover both triangles of their own block and so write once.
///
/// Out-of-range rows and features are zero-filled during staging, so the
/// accumulation loop needs no bounds test.
///
/// ### Params
///
/// * `a` - Input `[d, n]` row-major, feature-major so feature `i` is contiguous
/// * `out` - Output `[n_chunks, d, d]` row-major; pass `G` itself when
///   `n_chunks` is 1
/// * `n` - Reduction length
/// * `d` - Output edge
/// * `rows_per_chunk` - Rows handled by each chunk; the last chunk is short
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> output tile row, `CUBE_POS_Y` -> output tile column
/// * `CUBE_POS_Z` -> row chunk
/// * `UNIT_POS_X` -> flattened `(row, column)` within the thread grid
#[cube(launch_unchecked)]
pub fn gram_symmetric<F: Float>(
    a: &Tensor<F>,
    out: &mut Tensor<F>,
    n: u32,
    d: u32,
    rows_per_chunk: u32,
) {
    // Lower triangle is covered by the mirror write of its transpose.
    if CUBE_POS_Y < CUBE_POS_X {
        terminate!();
    }

    let tid = UNIT_POS_X;
    let tm = tid / GRAM_TN;
    let tn = tid % GRAM_TN;

    let i0 = CUBE_POS_X * GRAM_BM;
    let j0 = CUBE_POS_Y * GRAM_BM;

    let r_start = CUBE_POS_Z * rows_per_chunk;
    let mut r_end = r_start + rows_per_chunk;
    if r_end > n {
        r_end = n;
    }

    let mut sa = SharedMemory::<F>::new((GRAM_BK * GRAM_SMEM_STRIDE) as usize);
    let mut sb = SharedMemory::<F>::new((GRAM_BK * GRAM_SMEM_STRIDE) as usize);

    let mut acc = Array::<F>::new((GRAM_RT * GRAM_RT) as usize);
    #[unroll]
    for t in 0..GRAM_RT * GRAM_RT {
        acc[t as usize] = F::new(0.0);
    }

    let stage = GRAM_BK * GRAM_BM;

    let mut r = r_start;
    while r < r_end {
        // Cooperative stage. `m` on the slow axis and `k` on the fast one, so
        // consecutive lanes read consecutive samples of one feature and the
        // global reads coalesce; the shared writes stride by the padded
        // GRAM_SMEM_STRIDE and so stay conflict-free.
        let mut li = tid;
        while li < stage {
            let m = li / GRAM_BK;
            let k = li % GRAM_BK;
            let row = r + k;
            let in_row = row < r_end;
            let dst = k * GRAM_SMEM_STRIDE + m;

            if in_row && i0 + m < d {
                sa[dst as usize] = a[((i0 + m) * n + row) as usize];
            } else {
                sa[dst as usize] = F::new(0.0);
            }
            if in_row && j0 + m < d {
                sb[dst as usize] = a[((j0 + m) * n + row) as usize];
            } else {
                sb[dst as usize] = F::new(0.0);
            }
            li += GRAM_THREADS;
        }
        sync_cube();

        let abase = tm * GRAM_RT;
        let bbase = tn * GRAM_RT;

        #[unroll]
        for k in 0..GRAM_BK {
            let koff = k * GRAM_SMEM_STRIDE;

            // Hoist both operand strips into registers before the outer
            // product, so the 16 FMAs cost 8 shared reads rather than 32.
            let mut av = Array::<F>::new(GRAM_RT as usize);
            let mut bv = Array::<F>::new(GRAM_RT as usize);
            #[unroll]
            for t in 0..GRAM_RT {
                av[t as usize] = sa[(koff + abase + t) as usize];
                bv[t as usize] = sb[(koff + bbase + t) as usize];
            }

            #[unroll]
            for p in 0..GRAM_RT {
                #[unroll]
                for q in 0..GRAM_RT {
                    acc[(p * GRAM_RT + q) as usize] += av[p as usize] * bv[q as usize];
                }
            }
        }
        sync_cube();

        r += GRAM_BK;
    }

    let base = CUBE_POS_Z * d * d;
    let on_diagonal = CUBE_POS_X == CUBE_POS_Y;

    #[unroll]
    for p in 0..GRAM_RT {
        #[unroll]
        for q in 0..GRAM_RT {
            let gi = i0 + tm * GRAM_RT + p;
            let gj = j0 + tn * GRAM_RT + q;
            if gi < d && gj < d {
                let v = acc[(p * GRAM_RT + q) as usize];
                out[(base + gi * d + gj) as usize] = v;
                // A diagonal tile already covers both triangles of its own
                // block, so mirroring it would be a duplicate write.
                if !on_diagonal {
                    out[(base + gj * d + gi) as usize] = v;
                }
            }
        }
    }
}

/// Sum the per-chunk contributions into `G`.
///
/// ### Params
///
/// * `partials` - Per-chunk contributions `[n_chunks, d, d]`
/// * `g` - Output `[d, d]`, row-major
/// * `total` - `d * d`
/// * `n_chunks` - Number of row chunks
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> workgroup index, flattened
///   over two dimensions because `d * d / GRAM_REDUCE_WG` is past the
///   65535-per-dimension dispatch limit from `d = 4096` upward
/// * `UNIT_POS_X` -> element within the workgroup block
#[cube(launch_unchecked)]
pub fn gram_reduce<F: Float>(partials: &Tensor<F>, g: &mut Tensor<F>, total: u32, n_chunks: u32) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * GRAM_REDUCE_WG + UNIT_POS_X;
    if idx >= total {
        terminate!();
    }

    let mut acc = F::new(0.0);
    let mut c = 0u32;
    while c < n_chunks {
        acc += partials[(c * total + idx) as usize];
        c += 1u32;
    }
    g[idx as usize] = acc;
}

////////////////
// Dispatcher //
////////////////

/// Compute `G = A A^T` for a feature-major `A`.
///
/// Selects split-K via `gram_chunks`; when it returns 1, [`gram_symmetric()`]
/// writes straight into `g` and no reduce is launched.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `a` - Input `[d, n]` row-major, feature-major
/// * `g` - Output `[d, d]` row-major
/// * `n` - Reduction length
/// * `d` - Output edge
///
/// ### Returns
///
/// `Ok(())` on success; `g` holds `A A^T`.
///
/// ### Errors
///
/// * `GpuCubeCountExceeded` if either grid is over the device limit.
pub fn gram_aat<R, F>(
    client: &ComputeClient<R>,
    a: &GpuTensor<R, F>,
    g: &GpuTensor<R, F>,
    n: usize,
    d: usize,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    let n_chunks = gram_chunks(n, d, size_of::<F>());
    let rows_per_chunk = n.div_ceil(n_chunks) as u32;
    let tiles = (d as u32).div_ceil(GRAM_BM);

    // With one chunk the kernel accumulates the whole reduction and can write
    // the output in place, which skips both the partials allocation and the
    // reduce launch.
    let partials = (n_chunks > 1).then(|| GpuTensor::<R, F>::empty(vec![n_chunks, d, d], client));
    let target = partials.as_ref().unwrap_or(g);

    let count = checked_cube_count::<R>("gram_symmetric", tiles, tiles, n_chunks as u32)?;
    unsafe {
        gram_symmetric::launch_unchecked::<F, R>(
            client,
            count,
            CubeDim::new_1d(GRAM_THREADS),
            a.clone().into_tensor_arg(),
            target.clone().into_tensor_arg(),
            n as u32,
            d as u32,
            rows_per_chunk,
        );
    }

    if let Some(partials) = partials {
        let total = (d * d) as u32;
        let (gx, gy) = grid_2d(total.div_ceil(GRAM_REDUCE_WG));
        let count = checked_cube_count::<R>("gram_reduce", gx, gy, 1)?;
        unsafe {
            gram_reduce::launch_unchecked::<F, R>(
                client,
                count,
                CubeDim::new_1d(GRAM_REDUCE_WG),
                partials.clone().into_tensor_arg(),
                g.clone().into_tensor_arg(),
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

    /////////////
    // Helpers //
    /////////////

    /// Deterministic feature-major `[d, n]` input.
    fn build_a(n: usize, d: usize) -> Vec<f32> {
        (0..n * d)
            .map(|t| {
                let i = t / n;
                let r = t % n;
                ((i * 31 + r * 17) % 23) as f32 * 0.1 - 1.1
            })
            .collect()
    }

    /// `A A^T` on the host, row-major `[d, d]`.
    fn gram_host(a: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0f32;
                for r in 0..n {
                    acc += a[i * n + r] * a[j * n + r];
                }
                out[i * d + j] = acc;
            }
        }
        out
    }

    /// Run `gram_aat` and read the result back.
    fn run(n: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
        let device = try_device().expect("no device");
        let client = WgpuRuntime::client(&device);

        let a_host = build_a(n, d);
        let a = GpuTensor::<WgpuRuntime, f32>::from_slice(&a_host, vec![d, n], &client);
        let g = GpuTensor::<WgpuRuntime, f32>::empty(vec![d, d], &client);

        gram_aat::<WgpuRuntime, f32>(&client, &a, &g, n, d).unwrap();

        let got = g.read(&client).unwrap();
        (got, gram_host(&a_host, n, d))
    }

    /// Compare `[d, d]` row-major buffers to a relative tolerance, floored at 1
    /// so near-zero entries do not demand impossible precision. The all-zeros
    /// guard is the one that matters: a rejected dispatch leaves the output
    /// untouched and reports nothing, so without it a dead kernel passes.
    fn assert_close(got: &[f32], want: &[f32], d: usize, tol: f32) {
        assert!(
            got.iter().any(|v| v.abs() > 1e-6),
            "output is all zeros, the GPU did no work"
        );
        for i in 0..d {
            for j in 0..d {
                let (a, b) = (got[i * d + j], want[i * d + j]);
                assert!(
                    (a - b).abs() <= tol * b.abs().max(1.0),
                    "({i},{j}): got {a} want {b}"
                );
            }
        }
    }

    ///////////
    // Tests //
    ///////////

    // Exactly one tile, no tails, single chunk.
    #[test]
    fn test_gram_aat_single_tile() {
        if try_device().is_none() {
            return;
        }
        let (n, d) = (200, 64);
        let (got, want) = run(n, d);
        assert_close(&got, &want, d, 1e-4);
    }

    // d is not a multiple of GRAM_BM, so every edge tile has a tail, and n is
    // not a multiple of GRAM_BK so the last staged step is short.
    #[test]
    fn test_gram_aat_ragged_tails() {
        if try_device().is_none() {
            return;
        }
        let (n, d) = (137, 150);
        let (got, want) = run(n, d);
        assert_close(&got, &want, d, 1e-4);
    }

    // The mirror write must reproduce the primary exactly: both come from the
    // same accumulator, so this is bitwise, not approximate.
    #[test]
    fn test_gram_aat_symmetric_bitwise() {
        if try_device().is_none() {
            return;
        }
        let d = 150;
        let (got, _) = run(137, d);
        for i in 0..d {
            for j in 0..d {
                assert_eq!(
                    got[i * d + j].to_bits(),
                    got[j * d + i].to_bits(),
                    "asymmetric at ({i},{j})"
                );
            }
        }
    }

    // Small d and long n forces the split-K arm, so the partials buffer and
    // the reduce kernel are both exercised.
    #[test]
    // Heavy: n = 40000 against a host reference that is a 92 MFLOP triple loop.
    #[cfg(feature = "large_scale_diagnostics")]
    fn test_gram_aat_split_k_path() {
        if try_device().is_none() {
            return;
        }
        let (n, d) = (40_000, 48);
        assert!(
            gram_chunks(n, d, size_of::<f32>()) > 1,
            "this shape was supposed to take the split-K arm"
        );
        let (got, want) = run(n, d);
        assert_close(&got, &want, d, 1e-3);
    }

    // Each of the three terms in the heuristic must be the one that binds for
    // some reachable shape, otherwise it is dead weight.
    #[test]
    fn test_gram_chunks_each_bound_binds() {
        let bytes = size_of::<f32>();

        // Tile count binds: enough output tiles to fill the device already.
        assert_eq!(gram_chunks(500, 8_000, bytes), 1);
        assert_eq!(gram_chunks(1_000_000, 8_000, bytes), 1);

        // Row count binds: too few rows to give every chunk real work.
        assert_eq!(gram_chunks(500, 64, bytes), 1);
        assert_eq!(gram_chunks(2_000, 2_000, bytes), 1);

        // Partials memory binds: d is large enough that chunk copies of the
        // output dominate, even though the tile count would allow a split.
        assert_eq!(gram_chunks(1_000_000, 4_000, bytes), 1);

        // Long reduction at moderate d: this is the case split-K exists for,
        // worth 1.16x on the tall shape.
        assert_eq!(gram_chunks(50_000, 2_000, bytes), 4);

        // Tiny d: three tiles cannot fill anything, so split hard.
        assert_eq!(gram_chunks(1_000_000, 100, bytes), GRAM_MAX_CHUNKS);
    }

    // The reduce grid must stay inside the per-dimension dispatch limit. Put
    // `d * d / GRAM_REDUCE_WG` straight on x, as cholesky_gpu's version does,
    // and it is over the limit from d = 4096 upward: the launch is rejected on
    // the cubecl server thread, that thread dies, and the next call on the
    // client surfaces an unrelated CallError.
    #[test]
    fn test_gram_reduce_grid_within_dispatch_limit() {
        let (max_x, max_y, _) = WgpuRuntime::max_cube_count();

        // The threshold is real, not hypothetical.
        assert!((20_000u32 * 20_000).div_ceil(GRAM_REDUCE_WG) > max_x);

        for d in [4_096u32, 20_000, 32_768] {
            let blocks = (d * d).div_ceil(GRAM_REDUCE_WG);
            let (gx, gy) = grid_2d(blocks);
            assert!(gx <= max_x && gy <= max_y, "d = {d}");
            assert!(gx as u64 * gy as u64 >= blocks as u64, "d = {d} uncovered");
        }
    }

    // The staging arrays must fit the device's threadgroup budget, and fitting
    // is not the same question as being fast: residency is
    // max_shared_memory_size / footprint, and the step from one resident
    // workgroup to two is worth more than anything else at this size.
    #[test]
    fn test_gram_shared_memory_within_budget() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);
        let budget = client.properties().hardware.max_shared_memory_size;

        let footprint = 2 * GRAM_BK as usize * GRAM_SMEM_STRIDE as usize * size_of::<f32>();
        assert!(
            footprint <= budget,
            "staging needs {footprint} B, device allows {budget} B"
        );
        assert!(
            budget / footprint >= 2,
            "only {} resident workgroup(s) at {footprint} B against a {budget} B budget",
            budget / footprint
        );
    }
}
