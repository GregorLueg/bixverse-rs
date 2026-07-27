//! Tall-skinny QR factorisation via CholeskyQR2 on the GPU.
//!
//! Given a tall-skinny matrix `Y` of shape `[n, s]` with `n >> s`, computes
//! an orthonormal `Q` of the same shape such that `Q^T Q == I` to fp32 tol.
//! Used as the QR step in randomised SVD where the iteration matrix needs
//! re-orthonormalisation between power iterations.
//!
//! ### Algorithm
//!
//! One CholeskyQR pass computes:
//!
//! 1. `G = Y^T Y` on the GPU. Shape `[s, s]`, K-reduction over `n`.
//! 2. Round-trip `G` to the host.
//! 3. Cholesky factor `G = L L^T` on the host via faer.
//! 4. Compute `R^{-1} = (L^T)^{-1}` on the host (small `s x s` triangular
//!    inverse).
//! 5. Upload `R^{-1}` to the GPU.
//! 6. `Q = Y R^{-1}` on the GPU.
//!
//! A single pass loses ~half the available precision because the condition
//! number of `G = Y^T Y` is the square of that of `Y`. CholeskyQR2 runs the
//! pass twice: the second pass starts from a near-orthonormal `Q1`, so its
//! `G2 = Q1^T Q1` is close to identity and the residual orthogonality error
//! drops back to fp32 machine precision. Standard reference: Yamamoto,
//! Nakatsukasa, Yanagisawa, Fukaya, "Roundoff error analysis of the
//! CholeskyQR2 algorithm" (2014).
//!
//! ### Memory
//!
//! One `[n, s]` scratch tensor is allocated for the intermediate `Q1`.
//! For the single-cell case (`n = 1M`, `s = 130`) that's ~520 MB in fp32.
//! The two `[s, s]` matrices and the host-side round-trip of `G` and
//! `R^{-1}` are tiny by comparison (~67 KB each).

// The `#[cube]` macro generates undocumented launcher structs and functions.
#![allow(missing_docs)]

use ann_search_rs::gpu::grid_2d;
use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
use cubek::matmul::definition::{MatmulElems, MatmulPrecision};
use cubek::matmul::launch::{Strategy, launch_ref};
use cubek::std::InputBinding;
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Mat, Side};

use crate::gpu::checked_cube_count;
use crate::prelude::*;
use crate::utils::faer_parallelism;

////////////
// Consts //
////////////

/// Output tile edge for the Gram kernel. One thread per `G[i, j]` in a
/// `GRAM_TILE x GRAM_TILE` block, so the workgroup is `GRAM_TILE^2` threads.
const GRAM_TILE: u32 = 16;

/// Rows of `Y` staged per shared-memory step in the Gram kernel. Each step
/// costs `2 * GRAM_ROWS_STEP * GRAM_TILE` loads and yields
/// `GRAM_ROWS_STEP * GRAM_TILE^2` fused multiply-adds, i.e. `GRAM_ROWS_STEP`
/// per loaded element.
const GRAM_ROWS_STEP: u32 = 8;

/// Upper bound on Gram split-K chunks. Partials cost `chunks * s * s * 4 B`,
/// which at `s = 130` is ~68 KB per chunk.
const GRAM_MAX_CHUNKS: usize = 64;

/// Minimum rows per Gram chunk. Below this the per-chunk fixed cost and the
/// reduction outweigh the extra parallelism.
const GRAM_MIN_ROWS_PER_CHUNK: usize = 1024;

/////////////
// Helpers //
/////////////

/// Split-K chunk count for the Gram kernel.
///
/// ### Params
///
/// * `n` - Row count of `Y`
///
/// ### Returns
///
/// Number of row chunks, at least 1 and at most [`GRAM_MAX_CHUNKS`].
fn gram_chunks(n: usize) -> usize {
    (n / GRAM_MIN_ROWS_PER_CHUNK).clamp(1, GRAM_MAX_CHUNKS)
}

/// Wrap a raw `Handle` as a `TensorHandle` with row-major strides for the given
/// 2D shape. If `transposed` is true the last two strides are swapped so the
/// matmul reads the underlying buffer as the transpose.
///
/// ### Params
///
/// * `handle` - Raw buffer handle
/// * `shape` - Logical 2D shape `[rows, cols]`
/// * `transposed` - If true, swap the last two strides
/// * `dtype` - Storage type for the tensor
///
/// ### Returns
///
/// A `TensorHandle` over the provided handle.
fn wrap_handle<R: Runtime>(
    handle: &Handle,
    shape: [usize; 2],
    transposed: bool,
    dtype: cubecl::ir::StorageType,
) -> TensorHandle<R> {
    let strides = if transposed {
        // shape is post-transpose; storage is row-major [shape[1], shape[0]]
        // with row stride shape[0], so the transposed view has strides
        // [1, shape[0]].
        vec![1usize, shape[0]]
    } else {
        vec![shape[1], 1usize]
    };

    TensorHandle::new(handle.clone(), shape.to_vec(), strides, dtype)
}

/// Dispatch a dense GEMM `C = A * B` through cubek's `launch_ref`.
///
/// `a_transposed` swaps the strides of A so the matmul interprets it as
/// `A^T` while reading the same underlying handle. Used for the
/// `G = Y^T Y` step where both operands share the Y buffer.
///
/// ### Params
///
/// * `a_handle` - Raw handle for A
/// * `a_logical_shape` - Shape of A as seen by the matmul (post-transpose
///   if `a_transposed` is true)
/// * `a_transposed` - If true, swap A's last two strides
/// * `b_handle` - Raw handle for B
/// * `b_logical_shape` - Shape of B (B is never transposed)
/// * `c_handle` - Raw handle for C
/// * `c_shape` - Shape of C
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())` on success.
///
/// ### Errors
///
/// * `GpuMatmul` if cubek's `launch_ref` fails.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm<R, MP>(
    a_handle: &Handle,
    a_logical_shape: [usize; 2],
    a_transposed: bool,
    b_handle: &Handle,
    b_logical_shape: [usize; 2],
    b_tranposed: bool,
    c_handle: &Handle,
    c_shape: [usize; 2],
    strategy: Option<Strategy>,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    MP: MatmulPrecision,
{
    let mut dtypes = MatmulElems::new_deprecated::<MP>();
    let strategy = strategy.unwrap_or(Strategy::Auto);

    let a_tensor = wrap_handle::<R>(a_handle, a_logical_shape, a_transposed, dtypes.lhs_global);
    let b_tensor = wrap_handle::<R>(b_handle, b_logical_shape, b_tranposed, dtypes.rhs_global);
    let c_tensor = wrap_handle::<R>(c_handle, c_shape, false, dtypes.acc_global);

    launch_ref(
        &strategy,
        client,
        InputBinding::Normal(a_tensor.binding(), dtypes.lhs_global),
        InputBinding::Normal(b_tensor.binding(), dtypes.rhs_global),
        c_tensor.binding(),
        &mut dtypes,
    )
    .map_err(|e| BixverseErrors::GpuMatmul(format!("{e:?}")))?;

    Ok(())
}

/// Cholesky factor an `s x s` SPD matrix `G` on the host via faer and return
/// `R^{-1}` (where `R = L^T`) in row-major layout, ready for upload back to the
/// GPU.
///
/// ### Params
///
/// * `g_row_major` - Row-major flat buffer of `G`, length `s * s`
/// * `s` - Side length of `G`
///
/// ### Returns
///
/// Row-major `R^{-1}` of shape `[s, s]` as a flat `Vec<T>` of length
/// `s * s`. `R^{-1}` is upper triangular.
///
/// ### Errors
///
/// * `FaerCholeskyError` if `G` is not SPD.
fn r_inverse_from_gram<T>(g_row_major: &[T], s: usize) -> Result<Vec<T>, BixverseErrors>
where
    T: BixverseFloat,
{
    let g = Mat::<T>::from_fn(s, s, |i, j| g_row_major[i * s + j]);

    // Cholesky: G = L L^T with L lower triangular -> use .llt() from faer here
    let chol = g.llt(Side::Lower)?;
    let l = chol.L();

    // R = L^T (upper triangular). Compute R^{-1} by solving R * X = I for X.
    let r = l.transpose().to_owned();
    let mut r_inv = Mat::<T>::identity(s, s);
    solve_upper_triangular_in_place(r.as_ref(), r_inv.as_mut(), faer_parallelism());

    // Flatten into row-major for upload.
    let mut out = vec![T::zero(); s * s];
    for i in 0..s {
        for j in 0..s {
            out[i * s + j] = r_inv[(i, j)];
        }
    }
    Ok(out)
}

//////////
// Gram //
//////////

/// Partial `G = Y^T Y` over one chunk of rows.
///
/// cubek has no split-K strategy, and `Strategy::Auto` picks `SimpleUnit` for
/// this shape: one unit per output element. At `s = 130` that is 16 900
/// threads each running a serial reduction over all `n` rows, which measured
/// 223 ms per call and 0.3% of peak on an M1 Max. Splitting the reduction
/// across row chunks and giving each workgroup an output tile fixes both the
/// parallelism and the reuse.
///
/// Each workgroup owns a `GRAM_TILE x GRAM_TILE` block of `G` and one chunk of
/// rows. It stages `GRAM_ROWS_STEP` rows of the two relevant column slices in
/// shared memory and accumulates their outer products, so every loaded element
/// feeds `GRAM_ROWS_STEP` fused multiply-adds. The full `[s, s]` block is
/// computed rather than just the lower triangle; the symmetry saving is not
/// worth the divergence.
///
/// ### Params
///
/// * `y` - Tall-skinny input `[n, s]`, row-major
/// * `partials` - Output `[n_chunks, s, s]`, one Gram contribution per chunk
/// * `n` - Row count
/// * `s` - Column count
/// * `rows_per_chunk` - Rows handled by each chunk; the last chunk is short
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> output tile row, `CUBE_POS_Y` -> output tile column
/// * `CUBE_POS_Z` -> row chunk
/// * `UNIT_POS_X`, `UNIT_POS_Y` -> position within the output tile
#[cube(launch_unchecked)]
pub fn gram_partial<F: Float>(
    y: &Tensor<F>,
    partials: &mut Tensor<F>,
    n: u32,
    s: u32,
    rows_per_chunk: u32,
) {
    let ti = UNIT_POS_X;
    let tj = UNIT_POS_Y;
    let i0 = CUBE_POS_X * GRAM_TILE;
    let j0 = CUBE_POS_Y * GRAM_TILE;
    let chunk = CUBE_POS_Z;

    let r_start = chunk * rows_per_chunk;
    let mut r_end = r_start + rows_per_chunk;
    if r_end > n {
        r_end = n;
    }

    let mut sa = SharedMemory::<F>::new((GRAM_ROWS_STEP * GRAM_TILE) as usize);
    let mut sb = SharedMemory::<F>::new((GRAM_ROWS_STEP * GRAM_TILE) as usize);

    let lid = tj * GRAM_TILE + ti;
    let n_threads = GRAM_TILE * GRAM_TILE;
    let stage = GRAM_ROWS_STEP * GRAM_TILE;

    let mut acc = F::new(0.0);

    let mut r = r_start;
    while r < r_end {
        // Cooperative stage of both column slices. Out-of-range rows and
        // columns are zero-filled so the accumulation below needs no guard.
        let mut li = lid;
        while li < stage {
            let rr = li / GRAM_TILE;
            let cc = li % GRAM_TILE;
            let row = r + rr;
            let in_row = row < r_end;

            if in_row && i0 + cc < s {
                sa[li as usize] = y[(row * s + i0 + cc) as usize];
            } else {
                sa[li as usize] = F::new(0.0);
            }
            if in_row && j0 + cc < s {
                sb[li as usize] = y[(row * s + j0 + cc) as usize];
            } else {
                sb[li as usize] = F::new(0.0);
            }
            li += n_threads;
        }
        sync_cube();

        #[unroll]
        for rr in 0..GRAM_ROWS_STEP {
            acc += sa[(rr * GRAM_TILE + ti) as usize] * sb[(rr * GRAM_TILE + tj) as usize];
        }
        sync_cube();

        r += GRAM_ROWS_STEP;
    }

    let gi = i0 + ti;
    let gj = j0 + tj;
    if gi < s && gj < s {
        partials[(chunk * s * s + gi * s + gj) as usize] = acc;
    }
}

/// Sum the per-chunk Gram contributions into `G`.
///
/// ### Params
///
/// * `partials` - Per-chunk contributions `[n_chunks, s, s]`
/// * `g` - Output Gram matrix `[s, s]`, row-major
/// * `s` - Column count of `Y`
/// * `n_chunks` - Number of row chunks
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> flat index into the `[s, s]` output
#[cube(launch_unchecked)]
pub fn gram_reduce<F: Float>(partials: &Tensor<F>, g: &mut Tensor<F>, s: u32, n_chunks: u32) {
    let idx = ABSOLUTE_POS_X;
    let total = s * s;
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

/// Compute `G = Y^T Y` into `g_scratch` via the split-K Gram kernels.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `y` - Tall-skinny input `[n, s]`
/// * `g_scratch` - Output `[s, s]`
/// * `gram_partials` - Scratch `[n_chunks, s, s]`, sized by [`gram_chunks`]
/// * `n` - Row count
/// * `s` - Column count
///
/// ### Returns
///
/// `Ok(())` on success; `g_scratch` holds `Y^T Y`.
///
/// ### Errors
///
/// * `GpuCubeCountExceeded` if either grid is over the device limit.
fn gram<R, T>(
    client: &ComputeClient<R>,
    y: &GpuTensor<R, T>,
    g_scratch: &GpuTensor<R, T>,
    gram_partials: &GpuTensor<R, T>,
    n: usize,
    s: usize,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement,
{
    let n_chunks = gram_chunks(n);
    let rows_per_chunk = n.div_ceil(n_chunks) as u32;
    let tiles = (s as u32).div_ceil(GRAM_TILE);
    let total = (s * s) as u32;

    let partial_count = checked_cube_count::<R>("gram_partial", tiles, tiles, n_chunks as u32)?;
    let reduce_count = checked_cube_count::<R>("gram_reduce", total.div_ceil(256), 1, 1)?;

    unsafe {
        gram_partial::launch_unchecked::<T, R>(
            client,
            partial_count,
            CubeDim::new_2d(GRAM_TILE, GRAM_TILE),
            y.clone().into_tensor_arg(),
            gram_partials.clone().into_tensor_arg(),
            n as u32,
            s as u32,
            rows_per_chunk,
        );

        gram_reduce::launch_unchecked::<T, R>(
            client,
            reduce_count,
            CubeDim::new_1d(256),
            gram_partials.clone().into_tensor_arg(),
            g_scratch.clone().into_tensor_arg(),
            s as u32,
            n_chunks as u32,
        );
    }

    Ok(())
}

//////////////////////
// Tall-skinny GEMM //
//////////////////////

/// Rows of `A` handled per workgroup in [`tall_skinny_mm`].
const TSMM_ROWS: u32 = 8;

/// Output columns handled per workgroup in [`tall_skinny_mm`].
const TSMM_COLS: u32 = 32;

/// Largest inner dimension the shared-memory staging supports.
///
/// Footprint is `TSMM_ROWS * TSMM_K_MAX * 4 B` = 16 KiB, which leaves room for
/// two resident workgroups inside a 32 KiB threadgroup budget. Above this the
/// caller falls back to the generic matmul.
const TSMM_K_MAX: u32 = 512;

/// Inner-loop unroll for [`tall_skinny_mm`], to keep several loads of `b` in
/// flight rather than one.
const TSMM_UNROLL: u32 = 4;

/// `C = A * B` for a tall-skinny `A` and a small `B`.
///
/// The same `SimpleUnit` problem as [`fn@gram_partial`], from the other side: for
/// `[n, k] * [k, s]` with `n` large and `k`, `s` small, cubek gives each unit
/// one output element and a serial `k`-loop that re-reads `A`'s row from
/// global for every output column. Measured 34 ms per call at
/// `[200000, 130] * [130, 130]`, i.e. 2% of peak FLOPs and 1.5% of bandwidth.
///
/// Here each workgroup owns `TSMM_ROWS` rows and `TSMM_COLS` output columns.
/// The rows of `A` are staged in shared memory once and reused across the
/// column tile; `B` is small enough to stay cache-resident and is read from
/// global, where consecutive threads hit consecutive columns and coalesce.
///
/// ### Params
///
/// * `a` - Left operand `[n, k_dim]`, row-major
/// * `b` - Right operand `[k_dim, s]`, row-major
/// * `c` - Output `[n, s]`, row-major
/// * `n` - Rows of `A`
/// * `k_dim` - Inner dimension, must be at most `TSMM_K_MAX`
/// * `s` - Output column count
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> row block. Flattened over two
///   dimensions because `n / TSMM_ROWS` is past the 65535-per-dimension
///   dispatch limit from ~524k rows upward.
/// * `CUBE_POS_Z` -> output column block
/// * `UNIT_POS_X` -> flattened `(row within block, column within block)`
#[cube(launch_unchecked)]
pub fn tall_skinny_mm<F: Float>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    c: &mut Tensor<F>,
    n: u32,
    k_dim: u32,
    s: u32,
) {
    let tid = UNIT_POS_X;
    let tr = tid / TSMM_COLS;
    let tc = tid % TSMM_COLS;
    let n_threads = TSMM_ROWS * TSMM_COLS;

    let blk = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let row0 = blk * TSMM_ROWS;
    let col0 = CUBE_POS_Z * TSMM_COLS;

    // Rows of A staged once and reused by every column in the tile. Stride is
    // the runtime `k_dim`, so the tail of the allocation goes unused when
    // `k_dim < TSMM_K_MAX`.
    let mut sa = SharedMemory::<F>::new((TSMM_ROWS * TSMM_K_MAX) as usize);

    let stage = TSMM_ROWS * k_dim;
    let mut li = tid;
    while li < stage {
        let rr = li / k_dim;
        let cc = li % k_dim;
        let row = row0 + rr;
        if row < n {
            sa[li as usize] = a[(row * k_dim + cc) as usize];
        } else {
            sa[li as usize] = F::new(0.0);
        }
        li += n_threads;
    }
    sync_cube();

    let row = row0 + tr;
    let col = col0 + tc;
    if row < n && col < s {
        let abase = tr * k_dim;
        let mut acc = F::new(0.0);

        let mut i = 0u32;
        while i + TSMM_UNROLL <= k_dim {
            #[unroll]
            for u in 0..TSMM_UNROLL {
                acc += sa[(abase + i + u) as usize] * b[((i + u) * s + col) as usize];
            }
            i += TSMM_UNROLL;
        }
        while i < k_dim {
            acc += sa[(abase + i) as usize] * b[(i * s + col) as usize];
            i += 1u32;
        }

        c[(row * s + col) as usize] = acc;
    }
}

///////////////
// QR passes //
///////////////

/// One CholeskyQR pass: `output = input * R^{-1}` where `R^T R = G =
/// input^T input`. Loses ~half the available precision; intended to be
/// invoked twice (see [`cholesky_qr2`]).
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `input` - Tall-skinny input `[n, s]`
/// * `output` - Tall-skinny output `[n, s]`, must be distinct from `input`
/// * `g_scratch` - Pre-allocated `[s, s]` scratch for the Gram matrix
/// * `n` - Row count
/// * `s` - Column count (rank + oversampling)
///
/// ### Returns
///
/// `Ok(())` on success; `output` holds the result.
///
/// ### Errors
///
/// * `GpuMatmul` if either GEMM dispatch fails.
/// * `FaerCholeskyError` if the Gram matrix is not SPD (e.g. `input` is
///   rank-deficient).
fn cholesky_qr_pass<R, T, MP>(
    client: &ComputeClient<R>,
    input: &GpuTensor<R, T>,
    output: &GpuTensor<R, T>,
    g_scratch: &GpuTensor<R, T>,
    gram_partials: &GpuTensor<R, T>,
    n: usize,
    s: usize,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement + BixverseFloat,
    MP: MatmulPrecision,
{
    // First step: G = input^T * input, via the split-K Gram kernels rather
    // than cubek. See [`gram_partial`] for why the generic matmul is a bad fit
    // for a `[s, s]` output with an `n`-long reduction.
    gram::<R, T>(client, input, g_scratch, gram_partials, n, s)?;

    // Second step: Read G back to the host. (Minor matrix, should be fine ...)
    let g_host = g_scratch.clone().read(client)?;

    // Third step: Cholesky + triangular inverse on the host via faer.
    let r_inv_host = r_inverse_from_gram(&g_host, s)?;

    // Fourth step: Upload R^{-1}. Fresh allocation per pass; trivially small.
    let r_inv_gpu = GpuTensor::<R, T>::from_slice(&r_inv_host, vec![s, s], client);

    // Last step output = input * R^{-1}. The dedicated kernel only covers
    // inner dimensions its shared-memory staging can hold; wider problems fall
    // back to the generic matmul, which is correct just slower.
    if (s as u32) <= TSMM_K_MAX {
        // Row blocks are flattened over (x, y): at 1M rows there are 125k of
        // them, roughly twice the per-dimension dispatch limit.
        let (gx, gy) = grid_2d((n as u32).div_ceil(TSMM_ROWS));
        let count =
            checked_cube_count::<R>("tall_skinny_mm", gx, gy, (s as u32).div_ceil(TSMM_COLS))?;

        unsafe {
            tall_skinny_mm::launch_unchecked::<T, R>(
                client,
                count,
                CubeDim::new_1d(TSMM_ROWS * TSMM_COLS),
                input.clone().into_tensor_arg(),
                r_inv_gpu.clone().into_tensor_arg(),
                output.clone().into_tensor_arg(),
                n as u32,
                s as u32,
                s as u32,
            );
        }
    } else {
        dense_gemm::<R, MP>(
            input.handle(),
            [n, s],
            false,
            r_inv_gpu.handle(),
            [s, s],
            false,
            output.handle(),
            [n, s],
            None,
            client,
        )?;
    }

    Ok(())
}

//////////
// Main //
//////////

/// Compute an orthonormal `Q` from a tall-skinny `Y` via CholeskyQR2.
///
/// Two passes of `cholesky_qr_pass`; the second cleans up the precision loss
/// from the first. On exit, `q` holds an orthonormal basis for the column space
/// of `y` to fp32 tolerance.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `y` - Tall-skinny input `[n, s]`, untouched on exit
/// * `q` - Tall-skinny output `[n, s]`, must be distinct from `y`
/// * `q1_scratch` - Pre-allocated `[n, s]` scratch, distinct from both `y` and
///   `q`; contents on entry are irrelevant and are overwritten
/// * `g_scratch` - Pre-allocated `[s, s]` scratch for the Gram matrix
/// * `n` - Row count (e.g. number of cells)
/// * `s` - Column count, rank plus oversampling
///
/// ### Returns
///
/// `Ok(())` on success; `q` holds an orthonormal basis for the column
/// space of `y`.
///
/// ### Errors
///
/// * `GpuMatmul` if any GEMM dispatch fails.
/// * `FaerCholeskyError` if either pass's Gram matrix is not SPD (e.g.
///   rank-deficient `y`).
///
/// ### Notes
///
/// Both scratch buffers are caller-owned so that a driver calling this in a
/// loop allocates them once. The `[n, s]` one is ~520 MB at 1M cells and
/// s = 130, and a fresh `client.empty()` pays page faults on its first kernel
/// write, so reallocating per call is not free even though allocation itself
/// returns quickly.
pub fn cholesky_qr2<R, T, MP>(
    client: &ComputeClient<R>,
    y: &GpuTensor<R, T>,
    q: &GpuTensor<R, T>,
    q1_scratch: &GpuTensor<R, T>,
    g_scratch: &GpuTensor<R, T>,
    n: usize,
    s: usize,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement + BixverseFloat,
    MP: MatmulPrecision,
{
    // Gram split-K partials. `gram_chunks(n) * s * s * 4 B` is ~4.3 MB at the
    // production shape, so allocating here rather than threading yet another
    // buffer through the caller costs nothing measurable.
    let gram_partials = GpuTensor::<R, T>::empty(vec![gram_chunks(n), s, s], client);

    // Pass 1: y -> q1 (approximately orthonormal, loses half the precision)
    cholesky_qr_pass::<R, T, MP>(client, y, q1_scratch, g_scratch, &gram_partials, n, s)?;

    // Pass 2: q1 -> q (recovers full fp32 precision)
    cholesky_qr_pass::<R, T, MP>(client, q1_scratch, q, g_scratch, &gram_partials, n, s)?;

    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

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

    fn matmul_row_major(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    /// Well-conditioned tall-skinny `[n, s]` input: densely sampled sinusoids
    /// of distinct frequency, which are near-orthogonal as columns.
    fn sinusoidal_tall_skinny(n: usize, s: usize) -> Vec<f32> {
        (0..n * s)
            .map(|i| {
                let row = i / s;
                let col = i % s;
                let x = (row as f32 + 0.5) / n as f32;
                (2.0 * std::f32::consts::PI * (col + 1) as f32 * x).sin()
            })
            .collect()
    }

    /// `Q^T Q` in row-major, for an `[n, s]` row-major `Q`.
    fn gram_host(q: &[f32], n: usize, s: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                let mut acc = 0.0f32;
                for k in 0..n {
                    acc += q[k * s + i] * q[k * s + j];
                }
                out[i * s + j] = acc;
            }
        }
        out
    }

    fn assert_close_to_identity(mat: &[f32], s: usize, tol: f32) {
        for i in 0..s {
            for j in 0..s {
                let want = if i == j { 1.0 } else { 0.0 };
                let got = mat[i * s + j];
                assert!(
                    (got - want).abs() < tol,
                    "({}, {}): {} != {}",
                    i,
                    j,
                    got,
                    want
                );
            }
        }
    }

    ///////////
    // Tests //
    ///////////

    // Build a known SPD G, factor and invert via r_inverse_from_gram, then
    // check G * (R_inv * R_inv^T) == I. Uses the identity
    // G^{-1} = (R^T R)^{-1} = R^{-1} R^{-T}.
    #[test]
    fn test_r_inverse_from_gram_round_trip() {
        let s = 4;
        // Diagonally dominant -> SPD.
        let g: Vec<f32> = vec![
            4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.5, 0.5, 0.0, 0.0, 0.5, 1.5,
        ];

        let r_inv = r_inverse_from_gram(&g, s).unwrap();

        // r_inv is upper triangular; spot-check.
        for i in 0..s {
            for j in 0..i {
                assert!(
                    r_inv[i * s + j].abs() < 1e-6,
                    "r_inv not upper triangular at ({}, {})",
                    i,
                    j,
                );
            }
        }

        // g_inv = r_inv * r_inv^T
        let mut r_inv_t = vec![0.0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                r_inv_t[i * s + j] = r_inv[j * s + i];
            }
        }
        let g_inv = matmul_row_major(&r_inv, &r_inv_t, s, s, s);
        let prod = matmul_row_major(&g, &g_inv, s, s, s);

        assert_close_to_identity(&prod, s, 1e-4);
    }

    // Q from CholeskyQR2 must satisfy Q^T Q == I to fp32 tolerance.
    #[test]
    fn test_cholesky_qr2_orthogonality() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, s) = (200, 8);
        let y_host = sinusoidal_tall_skinny(n, s);

        let y = GpuTensor::<WgpuRuntime, f32>::from_slice(&y_host, vec![n, s], &client);
        let q =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n * s], vec![n, s], &client);

        let q1_scratch = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, s], &client);
        let g_scratch = GpuTensor::<WgpuRuntime, f32>::empty(vec![s, s], &client);

        cholesky_qr2::<WgpuRuntime, f32, f32>(&client, &y, &q, &q1_scratch, &g_scratch, n, s)
            .unwrap();

        let q_host = q.read(&client).unwrap();
        let qtq = gram_host(&q_host, n, s);

        assert_close_to_identity(&qtq, s, 1e-4);
    }

    // The row-block axis of tall_skinny_mm must stay inside the per-dimension
    // dispatch limit. Putting `n / TSMM_ROWS` straight on x, as the kernel
    // originally did, is over the limit from 65535 * TSMM_ROWS = 524_280 rows
    // upward: the launch is rejected on the cubecl server thread, that thread
    // dies, and the next call on the client surfaces an unrelated CallError.
    #[test]
    fn test_tall_skinny_grid_within_dispatch_limit() {
        let (max_x, max_y, max_z) = WgpuRuntime::max_cube_count();
        let gz = 130u32.div_ceil(TSMM_COLS);

        // The threshold is real, not hypothetical.
        assert!((900_000u32).div_ceil(TSMM_ROWS) > max_x);

        for n in [524_281u32, 1_000_000, 5_000_000] {
            let blocks = n.div_ceil(TSMM_ROWS);
            let (gx, gy) = grid_2d(blocks);
            assert!(gx <= max_x && gy <= max_y && gz <= max_z, "n = {n}");
            assert!(gx as u64 * gy as u64 >= blocks as u64, "n = {n} uncovered");
        }
    }

    // End to end past that threshold. Cheap at s = 4: three [n, s] buffers of
    // 9.6 MB each.
    #[test]
    fn test_cholesky_qr2_above_dispatch_limit() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let (n, s) = (600_000, 4);
        let y_host = sinusoidal_tall_skinny(n, s);

        let y = GpuTensor::<WgpuRuntime, f32>::from_slice(&y_host, vec![n, s], &client);
        let q =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n * s], vec![n, s], &client);

        let q1_scratch = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, s], &client);
        let g_scratch = GpuTensor::<WgpuRuntime, f32>::empty(vec![s, s], &client);

        cholesky_qr2::<WgpuRuntime, f32, f32>(&client, &y, &q, &q1_scratch, &g_scratch, n, s)
            .unwrap();

        let q_host = q.read(&client).unwrap();

        // A rejected dispatch leaves the output untouched, so check for the
        // all-zero signature before the orthogonality assertion.
        assert!(
            q_host.iter().any(|v| v.abs() > 1e-6),
            "Q is all zeros, the GPU did no work"
        );
        assert_close_to_identity(&gram_host(&q_host, n, s), s, 1e-4);
    }
}
