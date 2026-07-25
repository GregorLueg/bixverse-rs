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

use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
use cubek::matmul::definition::{MatmulElems, MatmulPrecision};
use cubek::matmul::launch::{Strategy, launch_ref};
use cubek::std::InputBinding;
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Mat, Side};

use crate::prelude::*;
use crate::utils::faer_parallelism;

/////////////
// Helpers //
/////////////

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
    n: usize,
    s: usize,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement + BixverseFloat,
    MP: MatmulPrecision,
{
    // First step: G = input^T * input on the GPU. Both operands share the input
    // handle; the lhs has its strides swapped to express the transpose.
    dense_gemm::<R, MP>(
        input.handle(),
        [s, n],
        true, // transpose to get input^T as lhs
        input.handle(),
        [n, s],
        false,
        g_scratch.handle(),
        [s, s],
        None,
        client,
    )?;

    // Second step: Read G back to the host. (Minor matrix, should be fine ...)
    let g_host = g_scratch.clone().read(client)?;

    // Third step: Cholesky + triangular inverse on the host via faer.
    let r_inv_host = r_inverse_from_gram(&g_host, s)?;

    // Fourth step: Upload R^{-1}. Fresh allocation per pass; trivially small.
    let r_inv_gpu = GpuTensor::<R, T>::from_slice(&r_inv_host, vec![s, s], client);

    // Last step output = input * R^{-1}.
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

    Ok(())
}

//////////
// Main //
//////////

/// Compute an orthonormal `Q` from a tall-skinny `Y` via CholeskyQR2.
///
/// Two passes of [`cholesky_qr_pass`]; the second cleans up the precision loss
/// from the first. On exit, `q` holds an orthonormal basis for the column space
/// of `y` to fp32 tolerance.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `y` - Tall-skinny input `[n, s]`, untouched on exit
/// * `q` - Tall-skinny output `[n, s]`, must be distinct from `y`
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
pub fn cholesky_qr2<R, T, MP>(
    client: &ComputeClient<R>,
    y: &GpuTensor<R, T>,
    q: &GpuTensor<R, T>,
    n: usize,
    s: usize,
) -> Result<(), BixverseErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement + BixverseFloat,
    MP: MatmulPrecision,
{
    let g_scratch = GpuTensor::<R, T>::empty(vec![s, s], client);
    let q1_scratch = GpuTensor::<R, T>::empty(vec![n, s], client);

    // Pass 1: y -> q1 (approximately orthonormal, loses half the precision)
    cholesky_qr_pass::<R, T, MP>(client, y, &q1_scratch, &g_scratch, n, s)?;

    // Pass 2: q1 -> q (recovers full fp32 precision)
    cholesky_qr_pass::<R, T, MP>(client, &q1_scratch, q, &g_scratch, n, s)?;

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
        // Well-conditioned tall-skinny input.
        let y_host: Vec<f32> = (0..n * s)
            .map(|i| {
                let row = i / s;
                let col = i % s;
                let x = (row as f32 + 0.5) / n as f32;
                (2.0 * std::f32::consts::PI * (col + 1) as f32 * x).sin()
            })
            .collect();

        let y = GpuTensor::<WgpuRuntime, f32>::from_slice(&y_host, vec![n, s], &client);
        let q =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; n * s], vec![n, s], &client);

        cholesky_qr2::<WgpuRuntime, f32, f32>(&client, &y, &q, n, s).unwrap();

        let q_host = q.read(&client).unwrap();

        // Q^T Q in row-major.
        let mut qtq = vec![0.0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                let mut acc = 0.0f32;
                for k in 0..n {
                    acc += q_host[k * s + i] * q_host[k * s + j];
                }
                qtq[i * s + j] = acc;
            }
        }

        assert_close_to_identity(&qtq, s, 1e-4);
    }
}
