//! Exact truncated SVD of a sparse matrix on the GPU via block Lanczos.
//!
//! The operator is the gene-side Gram `D^-1 Xc^T Xc D^-1` (`m x m`), with
//! `Xc` the column-centred A (minus the row offsets under CLR) and `D` the
//! column standard deviations, all applied implicitly. The GPU does the two
//! SpMMs per block step; the Krylov basis (`m x` a few hundred) and
//! everything done to it stay on the host in f64 through
//! [`block_lanczos_thick_restart`], which Metal could not do anyway.
//!
//! ### Per block step
//!
//! 1. Host: `X = V / sigma`, upload (`m x b`).
//! 2. `Y = A X - 1 (mu^T X) - r (1^T X)` (`n x b`), the forward SpMM.
//! 3. `Z = (A^T Y - mu (1^T Y) - 1 (r^T Y)) / sigma` (`m x b`), the transpose
//!    SpMM.
//! 4. Read `Z` back.
//!
//! The finish is one more forward SpMM at width `n_components`:
//! `U = Xc D^-1 V / S`.

use cubecl::Device;
use cubecl::prelude::Client;
use cubecl_utils_rs::prelude::*;
use faer::{Mat, MatRef};
use std::time::Instant;

use crate::linalg::sparse_gpu::{GpuCompressedSparseData, csc_to_csr_gpu};
use crate::linalg::spmm::{
    launch_dense_column_sum, launch_dense_column_weighted_sum, launch_spmm_csc_transpose,
    launch_spmm_csr_forward,
};
use crate::prelude::*;
use bixverse_core::core::math::pca_svd::SvdResults;
use bixverse_core::core::math::sparse::{BlockLanczosParams, block_lanczos_thick_restart};

//////////////
// Operator //
//////////////

/// Device-resident pieces of the centred, scaled operator.
struct GramOperator<T: cubecl::prelude::Float + cubecl::CubeElement> {
    /// CSR of A, for the forward SpMM.
    csr: GpuCompressedSparseData<T>,
    /// CSC of A, for the transpose SpMM.
    csc: GpuCompressedSparseData<T>,
    /// Column means `[m]`.
    mu: GpuTensor<T>,
    /// Column standard deviations `[m]`.
    sigma: GpuTensor<T>,
    /// Row offsets `[n]`, zeros without CLR.
    offsets: GpuTensor<T>,
    /// Whether the row offsets are in play.
    use_clr: bool,
    /// Rows of A.
    n: usize,
}

impl<T> GramOperator<T>
where
    T: cubecl::prelude::Float + cubecl::CubeElement + BixverseFloat,
{
    /// Forward product `Y = A X - 1 (mu^T X) - r (1^T X)`.
    ///
    /// ### Params
    ///
    /// * `x` - Dense `[m, width]`, already divided by sigma
    /// * `width` - Columns of `x`
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// `Y`, `[n, width]` on the device.
    fn forward(
        &self,
        x: &GpuTensor<T>,
        width: usize,
        client: &Client,
    ) -> Result<GpuTensor<T>, GpuErrors> {
        let m = self.csr.shape.1;
        let c = GpuTensor::<T>::empty(vec![width], client)?;
        let x_sum = zeros::<T>(width, client)?;
        launch_dense_column_weighted_sum::<T>(&self.mu, x, &c, m, width, client)?;
        if self.use_clr {
            launch_dense_column_sum::<T>(x, &x_sum, m, width, client)?;
        }
        let y = GpuTensor::<T>::empty(vec![self.n, width], client)?;
        launch_spmm_csr_forward::<T, T>(
            &self.csr,
            x,
            &c,
            &self.offsets,
            &x_sum,
            &y,
            width,
            client,
        )?;
        Ok(y)
    }

    /// Transpose product `Z = (A^T Y - mu (1^T Y) - 1 (r^T Y)) / sigma`.
    ///
    /// ### Params
    ///
    /// * `y` - Dense `[n, width]`
    /// * `width` - Columns of `y`
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// `Z`, `[m, width]` on the device.
    fn transpose(
        &self,
        y: &GpuTensor<T>,
        width: usize,
        client: &Client,
    ) -> Result<GpuTensor<T>, GpuErrors> {
        let m = self.csc.shape.1;
        let y_sum = GpuTensor::<T>::empty(vec![width], client)?;
        let r_dot_y = zeros::<T>(width, client)?;
        launch_dense_column_sum::<T>(y, &y_sum, self.n, width, client)?;
        if self.use_clr {
            launch_dense_column_weighted_sum::<T>(
                &self.offsets,
                y,
                &r_dot_y,
                self.n,
                width,
                client,
            )?;
        }
        let z = GpuTensor::<T>::empty(vec![m, width], client)?;
        launch_spmm_csc_transpose::<T, T>(
            &self.csc,
            y,
            &y_sum,
            &self.mu,
            &self.sigma,
            &r_dot_y,
            &z,
            width,
            client,
        )?;
        Ok(z)
    }
}

/// Zero-filled device vector.
///
/// ### Params
///
/// * `len` - Length
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The tensor, `[len]`.
fn zeros<T>(len: usize, client: &Client) -> Result<GpuTensor<T>, GpuErrors>
where
    T: cubecl::prelude::Float + cubecl::CubeElement + BixverseFloat,
{
    Ok(GpuTensor::<T>::from_slice(
        &vec![T::zero(); len],
        vec![len],
        client,
    )?)
}

/// Upload `V / sigma` row-major.
///
/// ### Params
///
/// * `v` - Host block `m x width`
/// * `sigma` - Column standard deviations `[m]`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The scaled block, `[m, width]` on the device.
fn upload_scaled<T>(
    v: MatRef<f64>,
    sigma: &[f64],
    client: &Client,
) -> Result<GpuTensor<T>, GpuErrors>
where
    T: cubecl::prelude::Float + cubecl::CubeElement + BixverseFloat,
{
    let (m, width) = v.shape();
    let host: Vec<T> = (0..m * width)
        .map(|k| {
            let (j, c) = (k / width, k % width);
            T::from_f64(v[(j, c)] / sigma[j]).expect("f64 casts to the device float")
        })
        .collect();
    Ok(GpuTensor::<T>::from_slice(&host, vec![m, width], client)?)
}

//////////
// Main //
//////////

/// Exact truncated SVD of a sparse matrix on the GPU, with implicit mean
/// centring, column scaling and optional row offsets.
///
/// Block Lanczos on the gene-side Gram, see the module docs. The device float
/// sets the operator's precision; the solver itself runs in f64 on the host.
///
/// ### Params
///
/// * `data` - CSC of A `(n, m)`, with `data_2` containing the values to
///   factorise
/// * `col_means` - Column means `[m]`
/// * `col_stds` - Column standard deviations `[m]`, floored > 0
/// * `row_offsets` - Optional per-row offsets `[n]` (CLR)
/// * `n_components` - Number of leading singular triples to return
/// * `params` - Optional [`BlockLanczosParams`], defaulted when `None`
/// * `seed` - Seed for the starting block
/// * `device` - CubeCL runtime device
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `SvdResults<T>` with `u` `[n, k]`, `s` of length `k` and `v` `[m, k]`,
/// where `k <= n_components` (fewer only if the Krylov space closed early or
/// trailing eigenvalues of the Gram came out non-positive).
///
/// ### Errors
///
/// * `SparseLayoutMismatch` if `data` is not CSC.
/// * `DimensionMisMatchSparse` if a vector has the wrong length.
/// * `CubeclUtils` if a dispatch grid or a buffer busts a device limit.
#[allow(clippy::too_many_arguments)]
pub fn sparse_svd_lanczos_gpu<T>(
    data: CompressedSparseData2<T, T>,
    col_means: &[T],
    col_stds: &[T],
    row_offsets: Option<&[T]>,
    n_components: usize,
    params: Option<BlockLanczosParams>,
    seed: u64,
    device: Device,
    verbose: usize,
) -> Result<SvdResults<T>, GpuErrors>
where
    T: cubecl::prelude::Float + cubecl::CubeElement + BixverseFloat,
{
    let verbosity = parse_verbosity_level(verbose);
    let start = Instant::now();
    let params = params.unwrap_or_default();

    if !data.cs_type.is_csc() {
        return Err(CoreErrors::SparseLayoutMismatch {
            expected: CompressedSparseFormat::Csc,
            got: data.cs_type,
        }
        .into());
    }
    let (n, m) = data.shape;
    if col_means.len() != m || col_stds.len() != m {
        return Err(CoreErrors::DimensionMisMatchSparse {
            indices_len: col_means.len(),
            data_len: m,
        }
        .into());
    }
    if let Some(off) = row_offsets
        && off.len() != n
    {
        return Err(CoreErrors::DimensionMisMatchSparse {
            indices_len: off.len(),
            data_len: n,
        }
        .into());
    }

    let client = device.client();
    let csc = GpuCompressedSparseData::<T>::from_compressed_sparse_data_2(&data, true, &client)?;
    let csr = csc_to_csr_gpu(&csc, &data.indptr, &data.indices, &client)?;
    drop(data);
    let op = GramOperator {
        csr,
        csc,
        mu: GpuTensor::<T>::from_slice(col_means, vec![m], &client)?,
        sigma: GpuTensor::<T>::from_slice(col_stds, vec![m], &client)?,
        offsets: match row_offsets {
            Some(off) => GpuTensor::<T>::from_slice(off, vec![n], &client)?,
            None => zeros::<T>(n, &client)?,
        },
        use_clr: row_offsets.is_some(),
        n,
    };
    let sigma: Vec<f64> = col_stds
        .iter()
        .map(|x| x.to_f64().expect("float casts to f64"))
        .collect();

    let mut n_calls = 0usize;
    let ritz = block_lanczos_thick_restart::<GpuErrors, _>(m, n_components, seed, params, |v| {
        n_calls += 1;
        let width = v.ncols();
        let x = upload_scaled::<T>(v, &sigma, &client)?;
        let y = op.forward(&x, width, &client)?;
        let z = op.transpose(&y, width, &client)?.read(&client)?;
        Ok(Mat::from_fn(m, width, |j, c| {
            z[j * width + c].to_f64().expect("float casts to f64")
        }))
    })?;

    if verbosity.detailed_verbosity() {
        println!(
            "    Block Lanczos: {} restarts, {} block steps, residual {:.3e} of ||A|| {:.3e} (converged: {}): {:.2?}",
            ritz.restarts,
            n_calls,
            ritz.residual,
            ritz.norm_estimate,
            ritz.converged,
            start.elapsed()
        );
    }

    // A Gram eigenvalue at or below zero is round-off on a rank-deficient
    // matrix, not a singular value.
    let k = ritz.values.iter().take_while(|&&x| x > 0.0).count();
    let s: Vec<f64> = ritz.values[..k].iter().map(|x| x.sqrt()).collect();
    let v = MatRef::from_column_major_slice(&ritz.vectors[..k * m], m, k);

    let x = upload_scaled::<T>(v, &sigma, &client)?;
    let u_host = op.forward(&x, k, &client)?.read(&client)?;

    let from = |x: f64| T::from_f64(x).expect("f64 casts to the device float");
    let res = SvdResults {
        u: Mat::from_fn(n, k, |i, c| {
            from(u_host[i * k + c].to_f64().expect("float casts to f64") / s[c])
        }),
        s: s.iter().map(|&x| from(x)).collect(),
        v: Mat::from_fn(m, k, |j, c| from(v[(j, c)])),
    };

    if verbosity.normal_verbosity() {
        println!(
            "  Finished Lanczos sparse SVD (GPU) in {:.2?}",
            start.elapsed()
        );
    }

    Ok(res)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use bixverse_core::core::math::sparse::sparse_svd_lanczos;
    use cubecl::wgpu::WgpuDevice;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::default();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::Device::from(device.clone()).client();
        }))
        .ok()
        .map(|_| device)
    }

    /// Deterministic sparse `n x m` matrix with a decaying column scale, so the
    /// spectrum is spread and the leading components are well separated.
    fn build_csc(n: usize, m: usize) -> CompressedSparseData2<f32, f32> {
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        let mut state = 12345u64;
        for j in 0..m {
            for i in 0..n {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 33) as f32 / (1u64 << 31) as f32;
                if u < 0.15 {
                    values.push((1.0 + 3.0 * u) * (1.0 + (j % 7) as f32) + (i % 5) as f32);
                    indices.push(i as u32);
                }
            }
            indptr.push(values.len() as u32);
        }
        CompressedSparseData2::<f32, f32>::new_csc(
            &values,
            &indices,
            &indptr,
            Some(&values),
            (n, m),
        )
    }

    /// Column means and standard deviations of the `data_2` layer.
    fn col_stats(csc: &CompressedSparseData2<f32, f32>) -> (Vec<f32>, Vec<f32>) {
        let (n, m) = csc.shape;
        let vals = csc.data_2.as_ref().unwrap();
        (0..m)
            .map(|j| {
                let seg = &vals[csc.indptr[j] as usize..csc.indptr[j + 1] as usize];
                let mean = seg.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
                let sq = seg.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>()
                    + (n - seg.len()) as f64 * mean * mean;
                (mean as f32, ((sq / (n - 1) as f64).sqrt() as f32).max(1e-6))
            })
            .unzip()
    }

    /// Smallest `|cos|` between matching columns of two loading matrices.
    fn worst_alignment(a: &Mat<f32>, b: &Mat<f64>, k: usize) -> f64 {
        (0..k)
            .map(|c| {
                let dot: f64 = (0..a.nrows()).map(|j| a[(j, c)] as f64 * b[(j, c)]).sum();
                let na: f64 = (0..a.nrows())
                    .map(|j| (a[(j, c)] as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let nb: f64 = (0..b.nrows())
                    .map(|j| b[(j, c)].powi(2))
                    .sum::<f64>()
                    .sqrt();
                (dot / (na * nb)).abs()
            })
            .fold(1.0, f64::min)
    }

    /// The GPU solve matches the CPU f64 Lanczos, with and without row offsets.
    #[test]
    fn test_sparse_svd_lanczos_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            return;
        };
        let (n, m, k) = (600usize, 80usize, 6usize);
        let csc = build_csc(n, m);
        let (mu, sd) = col_stats(&csc);
        let offsets: Vec<f32> = (0..n).map(|i| 0.01 * (i % 13) as f32).collect();

        for off in [None, Some(offsets.as_slice())] {
            let mu64: Vec<f64> = mu.iter().map(|&x| x as f64).collect();
            let sd64: Vec<f64> = sd.iter().map(|&x| x as f64).collect();
            let off64: Option<Vec<f64>> = off.map(|o| o.iter().map(|&x| x as f64).collect());
            let want = sparse_svd_lanczos::<f32, f32, f64>(
                &csc,
                k,
                7,
                true,
                Some(&mu64),
                Some(&sd64),
                off64.as_deref(),
            )
            .unwrap();

            let params = BlockLanczosParams {
                block_size: 4,
                ..BlockLanczosParams::new()
            };
            let got = sparse_svd_lanczos_gpu::<f32>(
                csc.clone(),
                &mu,
                &sd,
                off,
                k,
                Some(params),
                7,
                device.clone().into(),
                0,
            )
            .unwrap();

            assert_eq!(got.s.len(), k);
            for c in 0..k {
                let rel = (got.s[c] as f64 - want.s[c]).abs() / want.s[c];
                assert!(
                    rel < 1e-4,
                    "offsets {}: s[{c}] off by {rel:e}",
                    off.is_some()
                );
            }
            let align = worst_alignment(&got.v, &want.v, k);
            assert!(
                align > 0.999,
                "offsets {}: worst |cos| {align}",
                off.is_some()
            );

            // U = Xc D^-1 V / S has unit columns when V does.
            for c in 0..k {
                let norm: f64 = (0..n)
                    .map(|i| (got.u[(i, c)] as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!((norm - 1.0).abs() < 1e-3, "u column {c} has norm {norm}");
            }
        }
    }

    /// A CSR input is rejected before anything reaches the device.
    #[test]
    fn test_sparse_svd_lanczos_gpu_csr_input_rejected() {
        let Some(device) = try_device() else {
            return;
        };
        let csr = build_csc(20, 6).transform();
        let res = sparse_svd_lanczos_gpu::<f32>(
            csr,
            &[0.0; 6],
            &[1.0; 6],
            None,
            2,
            None,
            1,
            device.into(),
            0,
        );
        assert!(matches!(
            res,
            Err(GpuErrors::Core(CoreErrors::SparseLayoutMismatch { .. }))
        ));
    }
}
