//! Randomised SVD for sparse matrices on the GPU.
//!
//! Driver that orchestrates the SpMM kernels in `spmm.rs`, the CholeskyQR2
//! step in `cholesky_qr.rs`, and a small host-side faer thin SVD to produce
//! the truncated SVD of a sparse matrix `A` of shape `(n, m)` with mean
//! centering and column scaling implicitly applied.
//!
//! ### Pipeline
//!
//! 1. Build a CSR view of A from the input CSC (host, O(nnz)).
//! 2. Upload both layouts, plus `mu` and `sigma`. Generate `Omega` on the
//!    host pre-scaled by `1/sigma`, upload.
//! 3. `Y = A * Omega_scaled - 1 * c^T`, with `c = mu^T * Omega_scaled`
//!    precomputed.
//! 4. CholeskyQR2 -> `Q` orthonormal.
//! 5. Power iterations: `Z = (A^T Q - mu * q_sum^T) / sigma`, then
//!    `Y = A * Z - 1 * c^T`, then CholeskyQR2. Two iterations is the
//!    default; the gain past that is small.
//! 6. Final `Z = (A^T Q - mu * q_sum^T) / sigma`, which equals `B^T` for
//!    the standard randomised SVD with `B = Q^T * X` where `X` is the
//!    centred-scaled A.
//! 7. Round-trip `Z` to host (~1 MB). Run faer's thin SVD on `Z`. The
//!    result decomposes as `Z = U_Z * diag(S) * V_Z^T`, with `U_Z` being
//!    the V of the underlying SVD (gene loadings) and `V_Z` being the
//!    small mixing matrix.
//! 8. Upload `V_Z[:, :rank]`, compute `U = Q * V_Z[:, :rank]` on the GPU.
//! 9. Download U, build the `SvdResults` and return.
//!
//! ### Memory
//!
//! For 1M cells, 2000 genes, ~10% density, rank 30 + oversampling 100:
//! ~2.4 GB sparse data (both layouts), three `[n, s]` buffers at ~520 MB
//! each (Y/Q workspace plus CholeskyQR2 scratch inside), small `[m, s]`
//! and `[s, s]` matrices that are negligible. Total ~3.9 GB. Fits on most
//! discrete GPUs and on Apple Silicon with unified memory.

use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::Runtime;
use cubek::matmul::definition::MatmulPrecision;
use faer::Mat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use crate::core::math::pca_svd::SvdResults;
use crate::core::math::sparse::transpose_sparse_single_layer;
use crate::core::math::*;
use crate::gpu::linalg::cholesky_gpu::{cholesky_qr2, dense_gemm};
use crate::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use crate::gpu::linalg::spmm::{
    launch_dense_column_sum, launch_dense_column_weighted_sum, launch_spmm_csc_transpose,
    launch_spmm_csr_forward,
};
use crate::prelude::*;
use crate::single_cell::sc_processing::pca::SingleCellPcaParams;

////////////
// Params //
////////////

/// Parameters controlling randomised SVD behaviour on the GPU.
#[derive(Clone, Copy, Debug)]
pub struct RandSvdGpuParams {
    /// Number of power iterations. Two is the sweet spot for single-cell
    /// data; one is too few, three or more rarely helps.
    pub n_power_iters: usize,
    /// Extra columns sampled on top of `n_components` to stabilise the
    /// estimate of the leading singular subspace.
    pub oversampling: usize,
}

impl RandSvdGpuParams {
    /// New params.
    pub fn new(n_power_iters: usize, oversampling: usize) -> Self {
        Self {
            n_power_iters,
            oversampling,
        }
    }
}

impl Default for RandSvdGpuParams {
    fn default() -> Self {
        Self {
            n_power_iters: DEFAULT_N_POWER_ITERS_RAND_SVD,
            oversampling: DEFAULT_OVERSAMPLING_RAND_SVD,
        }
    }
}

//////////
// Main //
//////////

/// Randomised SVD of a sparse matrix on the GPU with implicit mean
/// centering and column scaling.
///
/// `data` must be a CSC of shape `(n, m)` with the `data_2` layer
/// populated (the normalised counts in the single-cell pipeline).
/// `col_means` and `col_stds` must be precomputed against `data_2`,
/// length `m`. `col_stds` must be strictly positive (floor on the host
/// upstream).
///
/// ### Params
///
/// * `data` - CSC of A `(n, m)`, with `data_2` containing the values to
///   factorise
/// * `col_means` - Column means `[m]`
/// * `col_stds` - Column standard deviations `[m]`, floored > 0
/// * `n_components` - Number of leading singular triples to return
/// * `params` - Optional [`RandSvdGpuParams`]; defaults to 2 power iters,
///   10 oversampling
/// * `seed` - Seed for the random projection
/// * `device` - CubeCL runtime device
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `SvdResults<T>` with `u` of shape `[n, n_components]`, `s` of length
/// `n_components`, and `v` of shape `[m, n_components]`.
///
/// ### Notes
///
/// The current implementation will always run PCA on the data_2 layer, i.e.,
/// the normalised counts
#[allow(clippy::too_many_arguments)]
pub fn randomised_sparse_svd_gpu<R, T, MP>(
    data: CompressedSparseData2<T, T>,
    col_means: &[T],
    col_stds: &[T],
    row_offsets: Option<&[T]>,
    n_components: usize,
    params: Option<RandSvdGpuParams>,
    seed: u64,
    device: R::Device,
    verbose: usize,
) -> Result<SvdResults<T>, BixverseErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement + BixverseFloat,
    MP: MatmulPrecision,
{
    let verbosity = parse_verbosity_level(verbose);
    let start = Instant::now();
    let params = params.unwrap_or_default();

    if !data.cs_type.is_csc() {
        return Err(BixverseErrors::SparseLayoutMismatch {
            expected: CompressedSparseFormat::Csc,
            got: data.cs_type,
        });
    }
    let (n, m) = data.shape;
    if col_means.len() != m || col_stds.len() != m {
        return Err(BixverseErrors::DimensionMisMatchSparse {
            indices_len: col_means.len(),
            data_len: m,
        });
    }
    if let Some(off) = row_offsets {
        if off.len() != n {
            return Err(BixverseErrors::DimensionMisMatchSparse {
                indices_len: off.len(),
                data_len: n,
            });
        }
    }

    let use_clr = row_offsets.is_some();
    let s = n_components + params.oversampling;

    // -- Host prep --
    let csr_host = transpose_sparse_single_layer(&data, true)?;

    // -- GPU upload --
    let client = R::client(&device);
    let csr_gpu =
        GpuCompressedSparseData::<R, T>::from_compressed_sparse_data_2(&csr_host, true, &client)?;
    let csc_gpu =
        GpuCompressedSparseData::<R, T>::from_compressed_sparse_data_2(&data, true, &client)?;
    let mu_gpu = GpuTensor::<R, T>::from_slice(col_means, vec![m], &client);
    let sigma_gpu = GpuTensor::<R, T>::from_slice(col_stds, vec![m], &client);

    let zero_n = vec![T::from(0.0).unwrap(); n];
    let row_offsets_gpu = match row_offsets {
        Some(off) => GpuTensor::<R, T>::from_slice(off, vec![n], &client),
        None => GpuTensor::<R, T>::from_slice(&zero_n, vec![n], &client),
    };

    drop(csr_host);
    drop(data);

    let mut rng = StdRng::seed_from_u64(seed);
    let omega_scaled: Vec<T> = (0..m * s)
        .map(|i| {
            let j = i / s;
            let u = rng.random::<f64>();
            T::from(u).unwrap() / col_stds[j]
        })
        .collect();
    let omega_gpu = GpuTensor::<R, T>::from_slice(&omega_scaled, vec![m, s], &client);

    // Workspace buffers. x_sum and m_dot_q are zero-initialised so they
    // contribute no correction when CLR is off.
    let y_buf = GpuTensor::<R, T>::empty(vec![n, s], &client);
    let q_buf = GpuTensor::<R, T>::empty(vec![n, s], &client);
    let z_buf = GpuTensor::<R, T>::empty(vec![m, s], &client);
    let c_buf = GpuTensor::<R, T>::empty(vec![s], &client);
    let q_sum_buf = GpuTensor::<R, T>::empty(vec![s], &client);
    let zero_s = vec![T::from(0.0).unwrap(); s];
    let x_sum_buf = GpuTensor::<R, T>::from_slice(&zero_s, vec![s], &client);
    let m_dot_q_buf = GpuTensor::<R, T>::from_slice(&zero_s, vec![s], &client);

    // -- Initial sketch --
    launch_dense_column_weighted_sum::<R, T>(&mu_gpu, &omega_gpu, &c_buf, m, s, &client);
    if use_clr {
        launch_dense_column_sum::<R, T>(&omega_gpu, &x_sum_buf, m, s, &client);
    }
    launch_spmm_csr_forward::<R, T, T>(
        &csr_gpu,
        &omega_gpu,
        &c_buf,
        &row_offsets_gpu,
        &x_sum_buf,
        &y_buf,
        s,
        &client,
    )?;

    cholesky_qr2::<R, T, MP>(&client, &y_buf, &q_buf, n, s)?;

    // -- Power iters --
    for iter in 0..params.n_power_iters {
        launch_dense_column_sum::<R, T>(&q_buf, &q_sum_buf, n, s, &client);
        if use_clr {
            launch_dense_column_weighted_sum::<R, T>(
                &row_offsets_gpu,
                &q_buf,
                &m_dot_q_buf,
                n,
                s,
                &client,
            );
        }
        launch_spmm_csc_transpose::<R, T, T>(
            &csc_gpu,
            &q_buf,
            &q_sum_buf,
            &mu_gpu,
            &sigma_gpu,
            &m_dot_q_buf,
            &z_buf,
            s,
            &client,
        )?;

        launch_dense_column_weighted_sum::<R, T>(&mu_gpu, &z_buf, &c_buf, m, s, &client);
        if use_clr {
            launch_dense_column_sum::<R, T>(&z_buf, &x_sum_buf, m, s, &client);
        }
        launch_spmm_csr_forward::<R, T, T>(
            &csr_gpu,
            &z_buf,
            &c_buf,
            &row_offsets_gpu,
            &x_sum_buf,
            &y_buf,
            s,
            &client,
        )?;

        cholesky_qr2::<R, T, MP>(&client, &y_buf, &q_buf, n, s)?;

        if verbosity.detailed_verbosity() {
            println!(
                "    Power iter {}/{} done: {:.2?}",
                iter + 1,
                params.n_power_iters,
                start.elapsed()
            );
        }
    }

    // -- Final Z --
    launch_dense_column_sum::<R, T>(&q_buf, &q_sum_buf, n, s, &client);
    if use_clr {
        launch_dense_column_weighted_sum::<R, T>(
            &row_offsets_gpu,
            &q_buf,
            &m_dot_q_buf,
            n,
            s,
            &client,
        );
    }
    launch_spmm_csc_transpose::<R, T, T>(
        &csc_gpu,
        &q_buf,
        &q_sum_buf,
        &mu_gpu,
        &sigma_gpu,
        &m_dot_q_buf,
        &z_buf,
        s,
        &client,
    )?;

    // -- final svd --

    // size is tiny, so this shoud work nicely in faer

    let z_host = z_buf.read(&client)?;
    let z_mat = Mat::<T>::from_fn(m, s, |i, j| z_host[i * s + j]);
    let svd = z_mat
        .thin_svd()
        .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;

    let v_z = svd.V();
    let v_z_trunc: Vec<T> = (0..s)
        .flat_map(|i| (0..n_components).map(move |j| v_z[(i, j)]))
        .collect();
    let v_z_gpu = GpuTensor::<R, T>::from_slice(&v_z_trunc, vec![s, n_components], &client);

    let u_gpu = GpuTensor::<R, T>::empty(vec![n, n_components], &client);
    dense_gemm::<R, MP>(
        q_buf.handle(),
        [n, s],
        false,
        v_z_gpu.handle(),
        [s, n_components],
        u_gpu.handle(),
        [n, n_components],
        None,
        &client,
    )?;
    let u_host = u_gpu.read(&client)?;

    if verbosity.detailed_verbosity() {
        println!(
            "    Host SVD + U reconstruction done: {:.2?}",
            start.elapsed()
        );
    }

    let u_mat = Mat::<T>::from_fn(n, n_components, |i, j| u_host[i * n_components + j]);
    let u_z = svd.U();
    let v_mat = Mat::<T>::from_fn(m, n_components, |i, j| u_z[(i, j)]);
    let s_vec: Vec<T> = (0..n_components).map(|i| svd.S()[i]).collect();

    if verbosity.normal_verbosity() {
        println!(
            "  Finished randomised sparse SVD (GPU) in {:.2?}",
            start.elapsed()
        );
    }

    Ok(SvdResults {
        u: u_mat,
        s: s_vec,
        v: v_mat,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use faer::Mat;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    // Same MP_TYPE as in the cholesky tests; replace with the concrete
    // MatmulPrecision impl.

    /////////////
    // Helpers //
    /////////////

    fn dense_to_csc(a: &[f32], n: usize, m: usize) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for j in 0..m {
            for i in 0..n {
                let v = a[i * m + j];
                if v != 0.0 {
                    values.push(v);
                    indices.push(i as u32);
                }
            }
            indptr.push(values.len() as u32);
        }
        (values, indices, indptr)
    }

    fn build_dense(n: usize, m: usize) -> Vec<f32> {
        // Deterministic, structurally sparse but full column rank.
        (0..n * m)
            .map(|k| {
                let i = k / m;
                let j = k % m;
                if (i + 2 * j).is_multiple_of(5) {
                    ((i * 7 + j * 13 + 3) % 19) as f32 * 0.3 + 0.5
                } else {
                    0.0
                }
            })
            .collect()
    }

    ///////////
    // Tests //
    ///////////

    // GPU randomised SVD top singular values must match the dense ground
    // truth from faer to randomised-SVD tolerance.
    #[test]
    fn test_randomised_sparse_svd_gpu_singular_values() {
        let Some(device) = try_device() else { return };

        let (n, m, n_components) = (60usize, 20usize, 3usize);
        let a_dense = build_dense(n, m);

        let (values, indices, indptr) = dense_to_csc(&a_dense, n, m);
        let csc = CompressedSparseData2::<f32, f32>::new_csc(
            &values,
            &indices,
            &indptr,
            Some(&values),
            (n, m),
        );

        let mu = vec![0.0f32; m];
        let sigma = vec![1.0f32; m];

        let svd = randomised_sparse_svd_gpu::<WgpuRuntime, f32, f32>(
            csc,
            &mu,
            &sigma,
            None,
            n_components,
            Some(RandSvdGpuParams::new(2, 8)),
            42,
            device,
            0,
        )
        .unwrap();

        let a_mat = Mat::<f32>::from_fn(n, m, |i, j| a_dense[i * m + j]);
        let true_svd = a_mat.thin_svd().unwrap();
        let true_s = true_svd.S();

        for i in 0..n_components {
            let want = true_s[i];
            let got = svd.s[i];
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel < 1e-3,
                "singular value {}: got {}, want {}, rel err {}",
                i,
                got,
                want,
                rel
            );
        }

        // Singular values must be non-negative and weakly decreasing.
        for i in 0..n_components {
            assert!(svd.s[i] >= 0.0, "negative singular value at {}", i);
            if i > 0 {
                assert!(svd.s[i - 1] >= svd.s[i], "not sorted at {}", i);
            }
        }
    }

    #[test]
    fn test_randomised_sparse_svd_gpu_csr_input_rejected() {
        let Some(device) = try_device() else { return };

        let (n, m) = (10usize, 4usize);
        let a_dense = build_dense(n, m);

        // Build CSR instead of CSC.
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for i in 0..n {
            for j in 0..m {
                let v = a_dense[i * m + j];
                if v != 0.0 {
                    values.push(v);
                    indices.push(j as u32);
                }
            }
            indptr.push(values.len() as u32);
        }
        let csr = CompressedSparseData2::<f32, f32>::new_csr(
            &values,
            &indices,
            &indptr,
            Some(&values),
            (n, m),
        );

        let mu = vec![0.0f32; m];
        let sigma = vec![1.0f32; m];

        let res = randomised_sparse_svd_gpu::<WgpuRuntime, f32, f32>(
            csr, &mu, &sigma, None, 2, None, 42, device, 0,
        );
        assert!(matches!(
            res,
            Err(BixverseErrors::SparseLayoutMismatch { .. })
        ));
    }

    #[test]
    fn test_randomised_sparse_svd_gpu_dim_mismatch_rejected() {
        let Some(device) = try_device() else { return };

        let (n, m) = (10usize, 4usize);
        let a_dense = build_dense(n, m);
        let (values, indices, indptr) = dense_to_csc(&a_dense, n, m);
        let csc = CompressedSparseData2::<f32, f32>::new_csc(
            &values,
            &indices,
            &indptr,
            Some(&values),
            (n, m),
        );

        // Wrong length for col_means / col_stds.
        let mu = vec![0.0f32; m + 1];
        let sigma = vec![1.0f32; m + 1];

        let res = randomised_sparse_svd_gpu::<WgpuRuntime, f32, f32>(
            csc, &mu, &sigma, None, 2, None, 42, device, 0,
        );
        assert!(matches!(
            res,
            Err(BixverseErrors::DimensionMisMatchSparse { .. })
        ));
    }
}
