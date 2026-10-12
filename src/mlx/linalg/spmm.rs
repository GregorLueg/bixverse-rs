//! Sparse-by-dense products on MLX, for the randomised sparse SVD and the
//! sparse HALS NMF.
//!
//! The affine kernel serves both directions of the SVD operator. It computes
//! `Y = (A X - 1 a^T - w b^T) / d` row by row over a CSR of `A`, with `a`, `b`
//! per output column and `w`, `d` per output row:
//!
//! * forward, CSR of A: `a = mu^T X`, `w` the row offsets, `b = 1^T X`,
//!   `d = 1`
//! * transpose, the CSC of A read as the CSR of `A^T`: `a = offsets^T Q`,
//!   `w = mu`, `b = 1^T Q`, `d = sigma` (or `sigma^2` inside the power loop)
//!
//! Both orientations are uploaded once, so `A^T Q` needs neither atomics nor
//! the split-K partials of the cubecl transpose.
//!
//! The plain kernel shares the accumulation and stores `Y = A X` unchanged,
//! which is all NMF needs for `V H^T` and `V^T W`.
//!
//! ### Threading
//!
//! One SIMD group per CSR row. The lanes load 32 consecutive non-zeros of the
//! row at a time (coalesced), then walk them in turn, each `(j, v)` handed to
//! every lane by `simd_shuffle`. Lane `l` owns output columns `l, l + 32, ...`,
//! so the read of the dense row `X[j, :]` is coalesced as well and the
//! accumulators stay in registers (`ceil(s / 32)` per lane).

use crate::mlx::ffi::{Array, MLX_FLOAT32, MetalKernel, OutputSpec, Stream};
use crate::prelude::*;

///////////////
// Constants //
///////////////

/// SIMD-group width on Apple GPUs; one SIMD group owns one CSR row.
const SIMD_WIDTH: usize = 32;

/// Threads per threadgroup, i.e. eight rows per threadgroup.
const SPMM_THREADGROUP: i32 = 256;

/// Metal accumulation shared by both SpMM kernels, see the module doc.
/// Inputs: `indptr` `[rows + 1]` u32, `indices` `[nnz]` u32, `values` `[nnz]`
/// f32, `x` `[k, S]` f32, `n_rows` `[1]` u32. Leaves the row of `A X` in
/// `acc`. Template ints: `S` (output width) and `CPL` (columns per lane,
/// `ceil(S / 32)`).
const SPMM_ACCUMULATE: &str = r#"
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x / SIMD_WIDTH;
    if (row >= n_rows[0]) {
        return;
    }
    uint lo = indptr[row];
    uint hi = indptr[row + 1];
    float acc[CPL];
    for (uint k = 0; k < uint(CPL); ++k) {
        acc[k] = 0.0f;
    }
    for (uint base = lo; base < hi; base += SIMD_WIDTH) {
        uint e = base + lane;
        uint j = 0;
        float v = 0.0f;
        if (e < hi) {
            j = indices[e];
            v = values[e];
        }
        uint cnt = min(SIMD_WIDTH, hi - base);
        for (uint t = 0; t < cnt; ++t) {
            uint jt = simd_shuffle(j, ushort(t));
            float vt = simd_shuffle(v, ushort(t));
            size_t xb = size_t(jt) * uint(S);
            for (uint k = 0; k < uint(CPL); ++k) {
                uint c = lane + k * SIMD_WIDTH;
                if (c < uint(S)) {
                    acc[k] += vt * x[xb + c];
                }
            }
        }
    }
    size_t yb = size_t(row) * uint(S);
"#;

/// Metal store of the affine SpMM. Extra inputs: `col_a` `[S]`, `row_w`
/// `[rows]`, `col_b` `[S]`, `row_div` `[rows]`. Output: `y` `[rows, S]`.
const SPMM_AFFINE_STORE: &str = r#"
    float rw = row_w[row];
    float rd = row_div[row];
    for (uint k = 0; k < uint(CPL); ++k) {
        uint c = lane + k * SIMD_WIDTH;
        if (c < uint(S)) {
            y[yb + c] = (acc[k] - col_a[c] - rw * col_b[c]) / rd;
        }
    }
"#;

/// Metal store of the plain SpMM. Output: `y` `[rows, S]`.
const SPMM_PLAIN_STORE: &str = r#"
    for (uint k = 0; k < uint(CPL); ++k) {
        uint c = lane + k * SIMD_WIDTH;
        if (c < uint(S)) {
            y[yb + c] = acc[k];
        }
    }
"#;

/////////////
// Structs //
/////////////

/// A CSR matrix resident on the device, f32 values.
pub(crate) struct MlxCsr {
    /// Row pointers `[n_rows + 1]`
    indptr: Array,
    /// Column indices `[nnz]`
    indices: Array,
    /// Values `[nnz]`
    values: Array,
    /// Number of rows
    n_rows: usize,
}

impl MlxCsr {
    /// Upload a CSR.
    ///
    /// ### Params
    ///
    /// * `indptr` - Row pointers, length `n_rows + 1`
    /// * `indices` - Column indices, length `nnz`
    /// * `values` - Values, length `nnz`
    /// * `n_rows` - Number of rows
    ///
    /// ### Returns
    ///
    /// The device copy, or `InvalidArgument` if the buffers disagree with
    /// `n_rows` or a size exceeds what MLX can index.
    pub(crate) fn new(
        indptr: &[u32],
        indices: &[u32],
        values: &[f32],
        n_rows: usize,
    ) -> Result<Self, BixverseErrors> {
        let nnz = indices.len();
        if indptr.len() != n_rows + 1
            || values.len() != nnz
            || indptr.last().map(|&x| x as usize) != Some(nnz)
        {
            return Err(BixverseErrors::InvalidArgument(format!(
                "MLX CSR: indptr {} / indices {} / values {} inconsistent with {} rows",
                indptr.len(),
                nnz,
                values.len(),
                n_rows
            )));
        }
        // the grid is `n_rows * SIMD_WIDTH` threads in an i32
        if n_rows > i32::MAX as usize / SIMD_WIDTH || nnz > i32::MAX as usize {
            return Err(BixverseErrors::InvalidArgument(format!(
                "MLX CSR: {n_rows} rows / {nnz} non-zeros exceed the MLX index range"
            )));
        }
        Ok(Self {
            indptr: Array::from_u32(indptr, &[indptr.len() as i32]),
            indices: Array::from_u32(indices, &[nnz as i32]),
            values: Array::from_f32(values, &[nnz as i32]),
            n_rows,
        })
    }
}

////////////
// Kernel //
////////////

/// Metal header defining the SIMD width both kernels index by.
///
/// ### Returns
///
/// The header source.
fn simd_width_header() -> String {
    format!("constant constexpr uint SIMD_WIDTH = {SIMD_WIDTH};\n")
}

/// Register the affine SpMM kernel, see [`SPMM_ACCUMULATE`] and [`SPMM_AFFINE_STORE`].
///
/// ### Returns
///
/// The kernel handle
pub(crate) fn spmm_kernel() -> MetalKernel {
    MetalKernel::new(
        "bixverse_spmm_csr_affine",
        &[
            "indptr", "indices", "values", "x", "col_a", "row_w", "col_b", "row_div", "n_rows",
        ],
        &["y"],
        &simd_width_header(),
        &format!("{SPMM_ACCUMULATE}{SPMM_AFFINE_STORE}"),
    )
}

/// Queue `Y = (A X - 1 col_a^T - row_w col_b^T) / row_div`.
///
/// ### Params
///
/// * `kernel` - Kernel from [`spmm_kernel`]
/// * `csr` - CSR of `A`, `[n_rows, n_cols]`
/// * `x` - Dense right operand `[n_cols, s_width]`, row-major
/// * `col_a` - Per-column subtrahend `[s_width]`
/// * `row_w` - Per-row weight on `col_b` `[n_rows]`
/// * `col_b` - Per-column vector scaled by `row_w` `[s_width]`
/// * `row_div` - Per-row divisor `[n_rows]`, non-zero
/// * `s_width` - Output width
/// * `s` - Stream to run on
///
/// ### Returns
///
/// The (lazy) output `[n_rows, s_width]`
#[allow(clippy::too_many_arguments)]
pub(crate) fn spmm_affine(
    kernel: &MetalKernel,
    csr: &MlxCsr,
    x: &Array,
    col_a: &Array,
    row_w: &Array,
    col_b: &Array,
    row_div: &Array,
    s_width: usize,
    s: &Stream,
) -> Result<Array, BixverseErrors> {
    let rows = csr.n_rows as i32;
    let n_arr = Array::from_u32(&[csr.n_rows as u32], &[1]);
    let mut out = kernel.apply(
        &[
            &csr.indptr,
            &csr.indices,
            &csr.values,
            x,
            col_a,
            row_w,
            col_b,
            row_div,
            &n_arr,
        ],
        &[OutputSpec {
            shape: &[rows, s_width as i32],
            dtype: MLX_FLOAT32,
        }],
        [rows * SIMD_WIDTH as i32, 1, 1],
        [SPMM_THREADGROUP, 1, 1],
        &[
            ("S", s_width as i32),
            ("CPL", s_width.div_ceil(SIMD_WIDTH) as i32),
        ],
        s,
    )?;
    Ok(out.remove(0))
}

/// Register the plain SpMM kernel, see [`SPMM_ACCUMULATE`] and
/// [`SPMM_PLAIN_STORE`].
///
/// ### Returns
///
/// The kernel handle
pub(crate) fn spmm_plain_kernel() -> MetalKernel {
    MetalKernel::new(
        "bixverse_spmm_csr_plain",
        &["indptr", "indices", "values", "x", "n_rows"],
        &["y"],
        &simd_width_header(),
        &format!("{SPMM_ACCUMULATE}{SPMM_PLAIN_STORE}"),
    )
}

/// Queue `Y = A X`.
///
/// ### Params
///
/// * `kernel` - Kernel from [`spmm_plain_kernel`]
/// * `csr` - CSR of `A`, `[n_rows, n_cols]`
/// * `x` - Dense right operand `[n_cols, s_width]`, row-major
/// * `s_width` - Output width
/// * `s` - Stream to run on
///
/// ### Returns
///
/// The (lazy) output `[n_rows, s_width]`; an empty row of `A` gives zeros.
pub(crate) fn spmm_plain(
    kernel: &MetalKernel,
    csr: &MlxCsr,
    x: &Array,
    s_width: usize,
    s: &Stream,
) -> Result<Array, BixverseErrors> {
    let rows = csr.n_rows as i32;
    let n_arr = Array::from_u32(&[csr.n_rows as u32], &[1]);
    let mut out = kernel.apply(
        &[&csr.indptr, &csr.indices, &csr.values, x, &n_arr],
        &[OutputSpec {
            shape: &[rows, s_width as i32],
            dtype: MLX_FLOAT32,
        }],
        [rows * SIMD_WIDTH as i32, 1, 1],
        [SPMM_THREADGROUP, 1, 1],
        &[
            ("S", s_width as i32),
            ("CPL", s_width.div_ceil(SIMD_WIDTH) as i32),
        ],
        s,
    )?;
    Ok(out.remove(0))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx::ffi::{eval_all, install_error_handler};

    /// Dense row-major to CSR.
    fn dense_to_csr(a: &[f32], n: usize, m: usize) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for i in 0..n {
            for j in 0..m {
                let v = a[i * m + j];
                if v != 0.0 {
                    values.push(v);
                    indices.push(j as u32);
                }
            }
            indptr.push(values.len() as u32);
        }
        (values, indices, indptr)
    }

    /// Dense row-major to CSC, i.e. the CSR of the transpose.
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

    /// Host forward reference `A X - 1 c^T - r x_sum^T`.
    #[allow(clippy::too_many_arguments)]
    fn cpu_forward(
        a: &[f32],
        x: &[f32],
        correction: &[f32],
        row_offsets: &[f32],
        x_sum: &[f32],
        n: usize,
        m: usize,
        s: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; n * s];
        for i in 0..n {
            for col in 0..s {
                let mut acc = 0.0f32;
                for j in 0..m {
                    acc += a[i * m + j] * x[j * s + col];
                }
                y[i * s + col] = acc - correction[col] - row_offsets[i] * x_sum[col];
            }
        }
        y
    }

    /// Host transpose reference `(A^T Q - mu q_sum^T - 1 m_dot_q^T) / sigma`.
    #[allow(clippy::too_many_arguments)]
    fn cpu_transpose(
        a: &[f32],
        q: &[f32],
        q_sum: &[f32],
        mu: &[f32],
        sigma: &[f32],
        m_dot_q: &[f32],
        n: usize,
        m: usize,
        s: usize,
    ) -> Vec<f32> {
        let mut z = vec![0.0f32; m * s];
        for j in 0..m {
            for col in 0..s {
                let mut acc = 0.0f32;
                for i in 0..n {
                    acc += a[i * m + j] * q[i * s + col];
                }
                z[j * s + col] = (acc - mu[j] * q_sum[col] - m_dot_q[col]) / sigma[j];
            }
        }
        z
    }

    /// Run the kernel on a host CSR.
    #[allow(clippy::too_many_arguments)]
    fn run(
        csr: (Vec<f32>, Vec<u32>, Vec<u32>),
        shape: (usize, usize),
        x: &[f32],
        col_a: &[f32],
        row_w: &[f32],
        col_b: &[f32],
        row_div: &[f32],
        s_width: usize,
    ) -> Vec<f32> {
        install_error_handler();
        let st = Stream::default_gpu();
        let (values, indices, indptr) = csr;
        let dev = MlxCsr::new(&indptr, &indices, &values, shape.0).unwrap();
        let (r, k) = (shape.0 as i32, shape.1 as i32);
        let s_i = s_width as i32;
        let y = spmm_affine(
            &spmm_kernel(),
            &dev,
            &Array::from_f32(x, &[k, s_i]),
            &Array::from_f32(col_a, &[s_i]),
            &Array::from_f32(row_w, &[r]),
            &Array::from_f32(col_b, &[s_i]),
            &Array::from_f32(row_div, &[r]),
            s_width,
            &st,
        )
        .unwrap();
        eval_all(&[&y], false).unwrap();
        y.as_f32().unwrap().to_vec()
    }

    /// Max abs difference, asserted below `tol`.
    fn assert_vec_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len());
        let diff = got
            .iter()
            .zip(want)
            .fold(0.0f32, |d, (g, w)| d.max((g - w).abs()));
        eprintln!("{label}: max abs diff {diff:.3e}");
        assert!(diff < tol, "{label}: max abs diff {diff:.3e} >= {tol:.1e}");
    }

    /// Sparse test matrix: every fifth entry non-zero.
    fn sparse_dense(n: usize, m: usize) -> Vec<f32> {
        (0..n * m)
            .map(|i| {
                if i % 5 == 0 {
                    ((i * 7 + 3) % 23) as f32 * 0.2
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Fused forward product against the host, every correction term live,
    /// at widths below, on and off a multiple of the SIMD width.
    #[test]
    fn test_mlx_spmm_forward() {
        let (n, m) = (100, 60);
        let a = sparse_dense(n, m);
        for s in [16, 32, 40] {
            let x: Vec<f32> = (0..m * s)
                .map(|i| ((i * 11 + 5) % 19) as f32 * 0.1)
                .collect();
            let correction: Vec<f32> = (0..s).map(|i| i as f32 * 0.05).collect();
            let row_offsets: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1).collect();
            let x_sum: Vec<f32> = (0..s).map(|i| (i % 3) as f32 * 0.2).collect();
            let got = run(
                dense_to_csr(&a, n, m),
                (n, m),
                &x,
                &correction,
                &row_offsets,
                &x_sum,
                &vec![1.0; n],
                s,
            );
            let want = cpu_forward(&a, &x, &correction, &row_offsets, &x_sum, n, m, s);
            assert_vec_close(&got, &want, 1e-3, &format!("forward s={s}"));
        }
    }

    /// Transpose product over the CSC read as a CSR of `A^T`, non-trivial mu,
    /// sigma and offset term. The CSC rows run to 24 non-zeros.
    #[test]
    fn test_mlx_spmm_transpose() {
        let (n, m, s) = (120, 50, 16);
        let a = sparse_dense(n, m);
        let q: Vec<f32> = (0..n * s)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.1)
            .collect();
        let q_sum: Vec<f32> = (0..s).map(|c| (0..n).map(|i| q[i * s + c]).sum()).collect();
        let mu: Vec<f32> = (0..m).map(|j| ((j * 13 + 1) % 11) as f32 * 0.1).collect();
        let sigma: Vec<f32> = (0..m)
            .map(|j| 0.5 + ((j * 5 + 2) % 7) as f32 * 0.1)
            .collect();
        let m_dot_q: Vec<f32> = (0..s).map(|c| c as f32 * 0.05).collect();

        let got = run(
            dense_to_csc(&a, n, m),
            (m, n),
            &q,
            &m_dot_q,
            &mu,
            &q_sum,
            &sigma,
            s,
        );
        let want = cpu_transpose(&a, &q, &q_sum, &mu, &sigma, &m_dot_q, n, m, s);
        assert_vec_close(&got, &want, 1e-3, "transpose");
    }

    /// Rows far longer than one SIMD group, with a ragged final batch of
    /// non-zeros (100 = 3 * 32 + 4), in both directions.
    #[test]
    fn test_mlx_spmm_long_rows() {
        let (n, m, s) = (70, 100, 24);
        let a: Vec<f32> = (0..n * m)
            .map(|i| ((i * 13 + 7) % 29) as f32 * 0.05 - 0.6)
            .collect();
        let x: Vec<f32> = (0..m * s)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.1 - 0.9)
            .collect();
        let zeros_s = vec![0.0f32; s];
        let got = run(
            dense_to_csr(&a, n, m),
            (n, m),
            &x,
            &zeros_s,
            &vec![0.0; n],
            &zeros_s,
            &vec![1.0; n],
            s,
        );
        let want = cpu_forward(&a, &x, &zeros_s, &vec![0.0; n], &zeros_s, n, m, s);
        assert_vec_close(&got, &want, 1e-3, "long rows forward");

        let q: Vec<f32> = (0..n * s)
            .map(|i| ((i * 7 + 2) % 17) as f32 * 0.1 - 0.8)
            .collect();
        let got = run(
            dense_to_csc(&a, n, m),
            (m, n),
            &q,
            &zeros_s,
            &vec![0.0; m],
            &zeros_s,
            &vec![1.0; m],
            s,
        );
        let ones = vec![1.0f32; m];
        let want = cpu_transpose(&a, &q, &zeros_s, &vec![0.0; m], &ones, &zeros_s, n, m, s);
        assert_vec_close(&got, &want, 1e-3, "long rows transpose");
    }

    /// Empty rows must emit `-correction[col]` verbatim.
    #[test]
    fn test_mlx_spmm_forward_empty_row() {
        let (n, m, s) = (8, 4, 4);
        let mut a = vec![0.0f32; n * m];
        a[1] = 1.5;
        a[7 * m + 3] = -2.0;
        let x: Vec<f32> = (0..m * s).map(|i| (i + 1) as f32 * 0.1).collect();
        let correction: Vec<f32> = (0..s).map(|i| (i + 1) as f32 * 0.25).collect();
        let zeros_s = vec![0.0f32; s];
        let zeros_n = vec![0.0f32; n];

        let got = run(
            dense_to_csr(&a, n, m),
            (n, m),
            &x,
            &correction,
            &zeros_n,
            &zeros_s,
            &vec![1.0; n],
            s,
        );
        let want = cpu_forward(&a, &x, &correction, &zeros_n, &zeros_s, n, m, s);
        assert_vec_close(&got, &want, 1e-6, "empty rows");
        for i in 1..7 {
            for col in 0..s {
                assert!(
                    (got[i * s + col] + correction[col]).abs() < 1e-6,
                    "empty row {i} col {col} not equal to -correction"
                );
            }
        }
    }

    /// Buffers inconsistent with the shape must error before any upload.
    #[test]
    fn test_mlx_csr_shape_mismatch_rejected() {
        let res = MlxCsr::new(&[0, 1, 2], &[0, 1], &[1.0, 2.0], 3);
        assert!(matches!(res, Err(BixverseErrors::InvalidArgument(_))));
        let res = MlxCsr::new(&[0, 1, 3], &[0, 1], &[1.0, 2.0], 2);
        assert!(matches!(res, Err(BixverseErrors::InvalidArgument(_))));
    }
}
