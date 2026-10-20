//! Exact truncated SVD of a sparse matrix on MLX via block Lanczos.
//!
//! The MLX twin of the cubecl `sparse_svd_lanczos_gpu`: the device applies
//! the gene-side Gram `D^-1 Xc^T Xc D^-1` a block at a time through the
//! [`Operator`] the randomised SVD uses, and the host runs
//! [`block_lanczos_thick_restart`] in f64. Finish is one forward SpMM at width
//! `n_components`, `U = Xc D^-1 V / S`.

use std::time::Instant;

use faer::{Mat, MatRef};

use crate::ffi::{Array, Stream, clear_cache, eval_all, install_error_handler};
use crate::linalg::rand_svd::Operator;
use crate::linalg::spmm::{MlxCsr, spmm_kernel};
use crate::prelude::*;
use bixverse_core::core::math::pca_svd::SvdResults;
use bixverse_core::core::math::sparse::{BlockLanczosParams, block_lanczos_thick_restart};

/// `V / sigma` as a row-major f32 device array.
///
/// ### Params
///
/// * `v` - Host block `m x width`
/// * `sigma` - Column standard deviations `[m]`
///
/// ### Returns
///
/// The scaled block, `[m, width]`.
fn scaled_block(v: MatRef<f64>, sigma: &[f64]) -> Array {
    let (m, width) = v.shape();
    let host: Vec<f32> = (0..m * width)
        .map(|k| (v[(k / width, k % width)] / sigma[k / width]) as f32)
        .collect();
    Array::from_vec_f32(host, &[m as i32, width as i32])
}

/// Exact truncated SVD of a sparse matrix on MLX, with implicit mean
/// centring, column scaling and optional row offsets.
///
/// ### Params
///
/// * `data` - CSC of A `(n, m)`, with `data_2` populated
/// * `col_means` - Column means `[m]`
/// * `col_stds` - Column standard deviations `[m]`, floored > 0
/// * `row_offsets` - Optional per-row offsets `[n]` (CLR)
/// * `n_components` - Number of leading singular triples to return
/// * `params` - Optional [`BlockLanczosParams`], defaulted when `None`
/// * `seed` - Seed for the starting block
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
/// * `SparseLayoutMismatch` if `data` is not CSC
/// * `DimensionMisMatchSparse` if a vector has the wrong length
/// * `Data2NotAvailable` if `data_2` is missing
/// * `MlxError` if mlx-c reports a failure
#[allow(clippy::too_many_arguments)]
pub fn sparse_svd_lanczos_mlx<T: BixverseFloat>(
    data: CompressedSparseData2<T, T>,
    col_means: &[T],
    col_stds: &[T],
    row_offsets: Option<&[T]>,
    n_components: usize,
    params: Option<BlockLanczosParams>,
    seed: u64,
    verbose: usize,
) -> Result<SvdResults<T>, MlxErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let start = Instant::now();
    let params = params.unwrap_or_default();

    if !data.cs_type.is_csc() {
        return Err(MlxErrors::Core(CoreErrors::SparseLayoutMismatch {
            expected: CompressedSparseFormat::Csc,
            got: data.cs_type,
        }));
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
    let to_f32 = |x: &T| x.to_f32().expect("finite float casts to f32");
    let to_f64 = |x: &T| x.to_f64().expect("float casts to f64");

    install_error_handler();
    let st = Stream::default_gpu();

    let values: Vec<f32> = data
        .data_2
        .as_ref()
        .ok_or(CoreErrors::Data2NotAvailable)?
        .iter()
        .map(to_f32)
        .collect();
    let csc = CompressedSparseData2::<f32, f32>::from_parts(
        Vec::new(),
        data.indices,
        data.indptr,
        Some(values),
        CompressedSparseFormat::Csc,
        (n, m),
    );
    let csr = csc.transform_single_layer(true)?;
    let mu: Vec<f32> = col_means.iter().map(to_f32).collect();
    let sigma: Vec<f64> = col_stds.iter().map(to_f64).collect();
    let sigma_f32: Vec<f32> = col_stds.iter().map(to_f32).collect();
    let offsets: Vec<f32> = match row_offsets {
        Some(off) => off.iter().map(to_f32).collect(),
        None => vec![0.0; n],
    };

    let op = Operator {
        kernel: spmm_kernel(),
        csr: MlxCsr::from_vecs(csr.indptr, csr.indices, csr.data_2.unwrap_or_default(), n)?,
        csc: MlxCsr::from_vecs(csc.indptr, csc.indices, csc.data_2.unwrap_or_default(), m)?,
        mu: Array::from_f32(&mu, &[m as i32, 1]),
        offsets: Array::from_f32(&offsets, &[n as i32, 1]),
        ones_n: Array::from_f32(&vec![1.0; n], &[n as i32]),
        use_clr: row_offsets.is_some(),
    };
    let sigma_dev = Array::from_f32(&sigma_f32, &[m as i32]);

    let mut n_calls = 0usize;
    let ritz = block_lanczos_thick_restart::<MlxErrors, _>(m, n_components, seed, params, |v| {
        n_calls += 1;
        let width = v.ncols();
        let x = scaled_block(v, &sigma);
        let y = op.forward(&x, width, &st)?;
        let z = op.transpose(&y, &sigma_dev, width, &st)?;
        eval_all(&[&z], false)?;
        let z_host = z.as_f32()?;
        Ok(Mat::from_fn(m, width, |j, c| z_host[j * width + c] as f64))
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

    let u = op.forward(&scaled_block(v, &sigma), k, &st)?;
    eval_all(&[&u], false)?;
    let u_host = u.as_f32()?;

    let from = |x: f64| T::from_f64(x).expect("f64 casts to the crate float");
    let res = SvdResults {
        u: Mat::from_fn(n, k, |i, c| from(u_host[i * k + c] as f64 / s[c])),
        s: s.iter().map(|&x| from(x)).collect(),
        v: Mat::from_fn(m, k, |j, c| from(v[(j, c)])),
    };
    drop((u, sigma_dev, op));
    clear_cache()?;

    if verbosity.normal_verbosity() {
        println!(
            "  Finished Lanczos sparse SVD (MLX) in {:.2?}",
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

    /// Deterministic sparse `n x m` matrix with a spread spectrum.
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

    /// The MLX solve matches the CPU f64 Lanczos, with and without row offsets.
    #[test]
    fn test_mlx_sparse_svd_lanczos_matches_cpu() {
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
            let got =
                sparse_svd_lanczos_mlx::<f32>(csc.clone(), &mu, &sd, off, k, Some(params), 7, 0)
                    .unwrap();

            assert_eq!(got.s.len(), k);
            for c in 0..k {
                let rel = (got.s[c] as f64 - want.s[c]).abs() / want.s[c];
                assert!(
                    rel < 1e-4,
                    "offsets {}: s[{c}] off by {rel:e}",
                    off.is_some()
                );

                let dot: f64 = (0..m).map(|j| got.v[(j, c)] as f64 * want.v[(j, c)]).sum();
                assert!(
                    dot.abs() > 0.999,
                    "offsets {}: |cos| {dot} at {c}",
                    off.is_some()
                );

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
    fn test_mlx_sparse_svd_lanczos_csr_input_rejected() {
        let csr = build_csc(20, 6).transform();
        let res = sparse_svd_lanczos_mlx::<f32>(csr, &[0.0; 6], &[1.0; 6], None, 2, None, 1, 0);
        assert!(matches!(
            res,
            Err(MlxErrors::Core(CoreErrors::SparseLayoutMismatch { .. }))
        ));
    }
}
