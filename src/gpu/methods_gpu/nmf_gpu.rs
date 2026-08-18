//! GPU HALS NMF: solver, backends and restarts.
//!
//! Mirrors [`crate::methods::nmf_hals`] iteration for iteration, but keeps `V`,
//! `W`, `H` and every intermediate on the device for the whole solve. Only the
//! objective comes back, once per convergence check, as a small partials vector
//! the host finishes in f64.
//!
//! ### Layout
//!
//! `V` is `m x n`, samples by features. Every device buffer is row-major f32:
//!
//! | buffer | shape | note |
//! |---|---|---|
//! | `v_t` (dense) | `[n, m]` | faer `V` reinterpreted |
//! | `v_csr` / `v_csc` (sparse) | `(m, n)` | one upload each |
//! | `w` | `[m, k]` | |
//! | `h` | `[n, k]`, i.e. `H^T` | faer `H` reinterpreted |
//! | `a` = `(W^T V)^T` | `[n, k]` | matches `h` |
//! | `c` = `V H^T` | `[m, k]` | matches `w` |
//! | `b` = `W^T W`, `d` = `H H^T` | `[k, k]` | |
//!
//! Holding `H` transposed is what makes everything line up. Both Grams become
//! `Y^T Y` over a `[rows, k]` buffer, which is
//! [`crate::gpu::linalg::cholesky_gpu::gram`] unchanged. Both sparse products
//! become the plain SpMM pair in [`crate::gpu::linalg::spmm`] unchanged. Both
//! dense products become one GEMM each off the same `v_t`. Both sweeps become
//! one launch each over a contiguous k-run per thread. And rescaling a row of
//! `H` is the same operation as rescaling a column of `W`, so the normalisation
//! step needs one kernel rather than two.
//!
//! Only `W` needs a transpose at the host boundary. faer's `H` is `k x n`
//! column-major, whose flat buffer already *is* `[n, k]` row-major, and `V` is
//! `m x n` column-major, whose flat buffer already *is* `[n, m]`.
//!
//! That reinterpretation is a borrow only when the matrix has no padding between
//! columns, which is not the general case: faer rounds the column stride up to a
//! 16-element boundary, so a `k x n` matrix is contiguous only when `k` is a
//! multiple of 16. For `V` that is nearly always satisfied, because `m` is the
//! sample count, and it is the upload worth caring about. For `H` it usually is
//! not, so a `k * n` copy runs instead, which is nothing next to one iteration's
//! data product. Both paths are covered by tests; the shape decides which fires.
//!
//! ### Precision
//!
//! f32 on device. The objective is the one place that matters, because
//! `||V||^2 - 2<A, H> + <B, D>` cancels three large terms into a small one, so
//! the two inner products are reduced in f64 on the host exactly as the CPU
//! path does. That leaves the loss resolvable down to roughly `1e-6 * ||V||^2`;
//! the default `tol` of `1e-4` sits comfortably above it, but tolerances below
//! about `1e-6` are not meaningful here.
//!
//! `||V||_F^2` is accumulated in f64 and cast, unlike the CPU backends which
//! accumulate in `F`. Over a few hundred million entries that difference is
//! real, so GPU relative losses are slightly different from CPU ones by
//! construction rather than by accident.

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::core::math::pca_svd::SvdResults;
use crate::gpu::linalg::cholesky_gpu::{gram, gram_chunks};
use crate::gpu::linalg::corr::contiguous_col_major;
use crate::gpu::linalg::skinny_gemm::{skinny_gemm, skinny_partial_elems};
use crate::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use crate::gpu::linalg::spmm::{
    launch_dense_column_sq_norm, launch_spmm_csc_transpose_plain, launch_spmm_csr_plain,
};
use crate::gpu::methods_gpu::nmf_kernels::{
    NMF_MAX_RANK, launch_fill_constant, launch_hals_norm_factors, launch_hals_sweep,
    launch_row_dot_partials, launch_scale_columns, sweep_tier,
};
use crate::methods::nmf_hals::dense::DenseInput;
use crate::methods::nmf_hals::sparse::SparseInput;
use crate::methods::nmf_hals::{
    HalsOpts, NmfInit, NmfResult, StabilisedNmfResult, nndsvd_from_svd, random_init,
};
use crate::prelude::*;

///////////////////
// Layout helpers //
///////////////////

/// The device-resident factor pair, `W` as `[m, k]` and `H^T` as `[n, k]`.
type GpuFactors<R> = (GpuTensor<R, f32>, GpuTensor<R, f32>);

/// Flatten a faer `m x k` matrix into a row-major `[m, k]` buffer.
///
/// faer is column-major, so this is a genuine transpose. It runs once per solve
/// on upload and once on readback, against `m * k` elements, which is small
/// next to a single iteration's data product.
///
/// ### Params
///
/// * `w` - Left factor `m x k`
///
/// ### Returns
///
/// `m * k` values, row `i` contiguous at `i * k`.
fn w_to_row_major(w: MatRef<f32>) -> Vec<f32> {
    let (m, k) = (w.nrows(), w.ncols());
    let mut out = vec![0f32; m * k];
    out.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = w[(i, c)];
        }
    });
    out
}

/// Rebuild a faer `m x k` matrix from a row-major `[m, k]` buffer.
///
/// ### Params
///
/// * `buf` - Row-major `[m, k]` values
/// * `m` - Row count
/// * `k` - Column count
///
/// ### Returns
///
/// The `m x k` matrix.
fn w_from_row_major(buf: &[f32], m: usize, k: usize) -> Mat<f32> {
    Mat::from_fn(m, k, |i, c| buf[i * k + c])
}

/// Flatten a faer `k x n` matrix into a row-major `[n, k]` buffer, i.e. `H^T`.
///
/// A `k x n` column-major allocation with no inter-column padding already is
/// this buffer, so the contiguous case borrows and only a strided or padded
/// matrix pays a copy.
///
/// ### Params
///
/// * `h` - Right factor `k x n`
///
/// ### Returns
///
/// Either a borrow of `h`'s allocation or an owned copy, as `[n, k]` row-major.
fn h_to_row_major<'a>(h: MatRef<'a, f32>) -> std::borrow::Cow<'a, [f32]> {
    match contiguous_col_major(h) {
        Some(slice) => std::borrow::Cow::Borrowed(slice),
        None => {
            let (k, n) = (h.nrows(), h.ncols());
            let mut out = Vec::with_capacity(k * n);
            for j in 0..n {
                for r in 0..k {
                    out.push(h[(r, j)]);
                }
            }
            std::borrow::Cow::Owned(out)
        }
    }
}

/// Rebuild a faer `k x n` matrix from a row-major `[n, k]` buffer.
///
/// ### Params
///
/// * `buf` - Row-major `[n, k]` values, i.e. `H^T`
/// * `k` - Number of components
/// * `n` - Number of features
///
/// ### Returns
///
/// The `k x n` matrix.
fn h_from_row_major(buf: &[f32], k: usize, n: usize) -> Mat<f32> {
    Mat::from_fn(k, n, |r, j| buf[j * k + r])
}

//////////////
// Backends //
//////////////

/// Device-resident view of `V` for GPU HALS.
///
/// The two implementors differ only in how the data products are dispatched;
/// the solver is written against this trait exactly as the CPU solver is
/// written against [`crate::methods::nmf_hals::NmfInput`].
///
/// `top_k_svd` stays on the host. NNDSVD initialisation runs once per solve
/// against a device loop of hundreds of iterations, and reusing the CPU
/// decomposition keeps GPU and CPU runs starting from bit-identical factors,
/// which is what makes the parity tests meaningful.
pub trait GpuNmfData<R: Runtime> {
    /// Shape of `V`.
    ///
    /// ### Returns
    ///
    /// `(m, n)`, samples by features.
    fn shape(&self) -> (usize, usize);

    /// The squared Frobenius norm `||V||_F^2`, computed once at construction.
    ///
    /// ### Returns
    ///
    /// The squared Frobenius norm.
    fn sq_frob(&self) -> f32;

    /// Compute `(W^T V)^T` into `a`.
    ///
    /// ### Params
    ///
    /// * `w` - Left factor `[m, k]` row-major
    /// * `a` - Output `[n, k]` row-major, overwritten
    /// * `k` - Number of components
    /// * `client` - CubeCL compute client
    /// * `partials` - Split-K scratch for the dense backend, ignored by the
    ///   sparse one
    ///
    /// ### Returns
    ///
    /// `Ok(())`, with `a` holding `(W^T V)^T`.
    fn wt_v_gpu(
        &self,
        w: &GpuTensor<R, f32>,
        a: &GpuTensor<R, f32>,
        k: usize,
        client: &ComputeClient<R>,
        partials: &GpuTensor<R, f32>,
    ) -> Result<(), BixverseErrors>;

    /// Compute `V H^T` into `c`.
    ///
    /// ### Params
    ///
    /// * `h` - Right factor held transposed, `[n, k]` row-major
    /// * `c` - Output `[m, k]` row-major, overwritten
    /// * `k` - Number of components
    /// * `client` - CubeCL compute client
    /// * `partials` - Split-K scratch for the dense backend, ignored by the
    ///   sparse one
    ///
    /// ### Returns
    ///
    /// `Ok(())`, with `c` holding `V H^T`.
    fn v_ht_gpu(
        &self,
        h: &GpuTensor<R, f32>,
        c: &GpuTensor<R, f32>,
        k: usize,
        client: &ComputeClient<R>,
        partials: &GpuTensor<R, f32>,
    ) -> Result<(), BixverseErrors>;

    /// Top-k truncated SVD of `V`, on the host, for NNDSVD initialisation.
    ///
    /// ### Params
    ///
    /// * `k` - Number of singular triplets
    ///
    /// ### Returns
    ///
    /// The top-k [`SvdResults`], or a [`BixverseErrors`] if the decomposition
    /// fails.
    fn top_k_svd(&self, k: usize) -> Result<SvdResults<f32>, BixverseErrors>;
}

/// Dense `V` on device, uploaded once as `V^T`.
///
/// Holds the host matrix alongside the device copy so NNDSVD initialisation can
/// reuse the CPU randomised SVD.
pub struct GpuDenseNmfInput<'a, R: Runtime> {
    /// `V^T` as `[n, m]` row-major on device.
    v_t: GpuTensor<R, f32>,
    /// Host-side view, for the initialisation SVD only.
    host: DenseInput<'a, f32>,
    /// Logical shape `(m, n)` of `V`.
    shape: (usize, usize),
    /// Cached `||V||_F^2`, accumulated in f64.
    sq_frob: f32,
}

impl<'a, R: Runtime> GpuDenseNmfInput<'a, R> {
    /// Validate `V`, upload it, and cache its Frobenius norm.
    ///
    /// Non-negativity and finiteness are checked by [`DenseInput::new`]. The
    /// upload is zero-copy when `v` owns a contiguous column-major allocation,
    /// because a `m x n` column-major buffer already is `[n, m]` row-major.
    ///
    /// ### Params
    ///
    /// * `v` - The matrix to factorise, `m x n`, samples by features
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// A [`GpuDenseNmfInput`], or `NmfNonFinite` / `NmfNonNegativeViolated` if
    /// validation fails, or `CubeclUtils` if `V` busts the device's per-binding
    /// size limit.
    pub fn new(v: MatRef<'a, f32>, client: &ComputeClient<R>) -> Result<Self, BixverseErrors> {
        let host = DenseInput::new(v)?;
        let (m, n) = (v.nrows(), v.ncols());

        let owned;
        let flat: &[f32] = match contiguous_col_major(v) {
            Some(slice) => slice,
            None => {
                let mut buf = Vec::with_capacity(m * n);
                for j in 0..n {
                    for i in 0..m {
                        buf.push(v[(i, j)]);
                    }
                }
                owned = buf;
                &owned
            }
        };

        let sq_frob = flat.iter().map(|&x| x as f64 * x as f64).sum::<f64>() as f32;
        let v_t = GpuTensor::<R, f32>::from_slice(flat, vec![n, m], client)?;

        Ok(Self {
            v_t,
            host,
            shape: (m, n),
            sq_frob,
        })
    }
}

impl<R: Runtime> GpuNmfData<R> for GpuDenseNmfInput<'_, R> {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn sq_frob(&self) -> f32 {
        self.sq_frob
    }

    fn wt_v_gpu(
        &self,
        w: &GpuTensor<R, f32>,
        a: &GpuTensor<R, f32>,
        k: usize,
        client: &ComputeClient<R>,
        partials: &GpuTensor<R, f32>,
    ) -> Result<(), BixverseErrors> {
        let (m, n) = self.shape;
        // a[n, k] = V^T[n, m] * W[m, k], reducing over m. `v_t` already is V^T,
        // so it is indexed as stored.
        skinny_gemm::<R, f32>(client, &self.v_t, w, a, partials, n, k, m, false)
    }

    fn v_ht_gpu(
        &self,
        h: &GpuTensor<R, f32>,
        c: &GpuTensor<R, f32>,
        k: usize,
        client: &ComputeClient<R>,
        partials: &GpuTensor<R, f32>,
    ) -> Result<(), BixverseErrors> {
        let (m, n) = self.shape;
        // c[m, k] = V[m, n] * H^T[n, k], reducing over n. V is `v_t` with its two
        // indices swapped, which is what the transposed flag selects.
        skinny_gemm::<R, f32>(client, &self.v_t, h, c, partials, m, k, n, true)
    }

    fn top_k_svd(&self, k: usize) -> Result<SvdResults<f32>, BixverseErrors> {
        crate::methods::nmf_hals::NmfInput::top_k_svd(&self.host, k)
    }
}

/// Sparse `V` on device, uploaded once in both orientations.
///
/// `W^T V` wants the CSC and `V H^T` wants the CSR, so both are resident. The
/// host [`SparseInput`] is kept for the initialisation SVD, which means the
/// matched pair exists twice, once per side. That mirrors what the CPU path
/// already costs and is what lets NNDSVD reuse the validated Lanczos path.
///
/// The two directions are not symmetric in cost even though they do the same
/// amount of arithmetic. Profiled at 50000 x 3000, k = 30, 5% dense, the CSC
/// direction ran 9.0 ms per launch against the CSR direction's 4.1 ms. The
/// difference is the dense operand: `V H^T` gathers from `H^T`, which is
/// `[n, k]` and a few hundred kilobytes, so it stays cache-resident, while
/// `W^T V` gathers from `W`, which is `[m, k]` and megabytes at single-cell
/// scale, so every non-zero's gather goes further out. Nothing to do about it
/// short of blocking the cell axis, and the pair together already runs at 56% of
/// device bandwidth.
pub struct GpuSparseNmfInput<R: Runtime> {
    /// CSR of `V`, shape `(m, n)`.
    v_csr: GpuCompressedSparseData<R, f32>,
    /// CSC of `V`, shape `(m, n)`.
    v_csc: GpuCompressedSparseData<R, f32>,
    /// Host-side matched pair, for the initialisation SVD only.
    host: SparseInput<f32, f32>,
    /// Logical shape `(m, n)` of `V`.
    shape: (usize, usize),
    /// Cached `||V||_F^2`, accumulated in f64.
    sq_frob: f32,
}

impl<R: Runtime> GpuSparseNmfInput<R> {
    /// Build the matched pair on the host, validate it, and upload both layouts.
    ///
    /// ### Params
    ///
    /// * `host` - A validated [`SparseInput`] over `V`
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// A [`GpuSparseNmfInput`], or `CubeclUtils` if any of the six buffers busts
    /// the device's per-binding size limit.
    pub fn new(
        host: SparseInput<f32, f32>,
        client: &ComputeClient<R>,
    ) -> Result<Self, BixverseErrors> {
        let shape = crate::methods::nmf_hals::NmfInput::shape(&host);
        let sq_frob = host
            .csr()
            .data
            .iter()
            .map(|&x| x as f64 * x as f64)
            .sum::<f64>() as f32;

        let csr = host.csr();
        let csc = host.csc();

        let v_csr = GpuCompressedSparseData::<R, f32>::from_parts(
            &csr.data,
            &csr.indices,
            &csr.indptr,
            csr.cs_type,
            shape,
            client,
        )?;
        let v_csc = GpuCompressedSparseData::<R, f32>::from_parts(
            &csc.data,
            &csc.indices,
            &csc.indptr,
            csc.cs_type,
            shape,
            client,
        )?;

        Ok(Self {
            v_csr,
            v_csc,
            host,
            shape,
            sq_frob,
        })
    }
}

impl<R: Runtime> GpuNmfData<R> for GpuSparseNmfInput<R> {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn sq_frob(&self) -> f32 {
        self.sq_frob
    }

    fn wt_v_gpu(
        &self,
        w: &GpuTensor<R, f32>,
        a: &GpuTensor<R, f32>,
        k: usize,
        client: &ComputeClient<R>,
        _partials: &GpuTensor<R, f32>,
    ) -> Result<(), BixverseErrors> {
        // Z = A^T Q with A = V shaped (m, n): Q is W [m, k], Z is a [n, k]. The
        // SpMM accumulates per output row, so it needs no split-K scratch.
        launch_spmm_csc_transpose_plain::<R, f32, f32>(&self.v_csc, w, a, k, client)
    }

    fn v_ht_gpu(
        &self,
        h: &GpuTensor<R, f32>,
        c: &GpuTensor<R, f32>,
        k: usize,
        client: &ComputeClient<R>,
        _partials: &GpuTensor<R, f32>,
    ) -> Result<(), BixverseErrors> {
        // Y = A X with A = V shaped (m, n): X is H^T [n, k], Y is c [m, k]. The
        // SpMM accumulates per output row, so it needs no split-K scratch.
        launch_spmm_csr_plain::<R, f32, f32>(&self.v_csr, h, c, k, client)
    }

    fn top_k_svd(&self, k: usize) -> Result<SvdResults<f32>, BixverseErrors> {
        crate::methods::nmf_hals::NmfInput::top_k_svd(&self.host, k)
    }
}

/////////////
// Scratch //
/////////////

/// Every device buffer a GPU HALS solve reuses, allocated once.
///
/// Sized at the largest rank the caller will ask for, so one scratch serves an
/// entire k sweep. The buffers are used as tightly packed `[rows, k]` prefixes
/// at each rank, which is why the rank is a kernel argument rather than baked
/// into the allocation.
///
/// `W` and `H` are deliberately not held here: they are seeded from the host
/// once per restart, and CubeCL has no write-into-existing-buffer path, so they
/// are created per solve. The allocator pools same-sized requests, so the page
/// faults are paid on the first restart rather than on every one.
pub struct NmfGpuScratch<R: Runtime> {
    /// `(W^T V)^T`, `[n, k_max]`.
    a: GpuTensor<R, f32>,
    /// `V H^T`, `[m, k_max]`.
    c: GpuTensor<R, f32>,
    /// `W^T W`, `[k_max, k_max]`.
    b: GpuTensor<R, f32>,
    /// `H H^T`, `[k_max, k_max]`.
    d: GpuTensor<R, f32>,
    /// Split-K Gram partials, sized for the longer of the two reductions.
    gram_partials: GpuTensor<R, f32>,
    /// Split-K partials for the dense data products, sized for whichever of the
    /// two splits further. Allocated even for the sparse backend, where it is one
    /// element, so the trait signature does not have to be optional.
    gemm_partials: GpuTensor<R, f32>,
    /// Per-component sums of squares of `W`, `[k_max]`.
    sq: GpuTensor<R, f32>,
    /// Per-component L2 norms of `W`, `[k_max]`.
    norm: GpuTensor<R, f32>,
    /// Reciprocals of `norm`, `[k_max]`.
    inv_norm: GpuTensor<R, f32>,
    /// Per-row partials for the objective, `[max(m, n)]`.
    ///
    /// Sized for both axes because the two frozen-factor refits reduce over
    /// opposite ones: `H` frozen pairs `W` against `V H^T` over `m` rows, and `W`
    /// frozen pairs `H^T` against `(W^T V)^T` over `n`.
    obj_partials: GpuTensor<R, f32>,
    /// The non-negativity floor, as a one-element tensor.
    eps: GpuTensor<R, f32>,
    /// Largest rank this scratch covers.
    k_max: usize,
}

impl<R: Runtime> NmfGpuScratch<R> {
    /// Allocate the scratch for a problem of shape `(m, n)` up to rank `k_max`.
    ///
    /// ### Params
    ///
    /// * `m` - Number of samples
    /// * `n` - Number of features
    /// * `k_max` - Largest rank any solve using this scratch will request
    /// * `eps` - The non-negativity floor from [`HalsOpts`]
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// An [`NmfGpuScratch`], or `GpuNmfRankTooLarge` if `k_max` is above what
    /// the sweep is compiled for, or `CubeclUtils` if any buffer busts the
    /// device's per-binding size limit.
    pub fn new(
        m: usize,
        n: usize,
        k_max: usize,
        eps: f32,
        client: &ComputeClient<R>,
    ) -> Result<Self, BixverseErrors> {
        if sweep_tier(k_max).is_none() {
            return Err(BixverseErrors::GpuNmfRankTooLarge {
                k: k_max,
                max: NMF_MAX_RANK,
            });
        }

        // One Gram runs over m rows and the other over n, so size the partials
        // for whichever splits further.
        let chunks = gram_chunks(m).max(gram_chunks(n));

        // The dense products split independently: `(W^T V)^T` is `[n, k_max]`
        // reducing over m, and `V H^T` is `[m, k_max]` reducing over n. Size for
        // whichever needs more, and floor at one element so the sparse backend,
        // which never reads this, still gets a bindable buffer.
        let gemm_elems = skinny_partial_elems(n, k_max, m)
            .max(skinny_partial_elems(m, k_max, n))
            .max(1);

        Ok(Self {
            a: GpuTensor::<R, f32>::empty(vec![n, k_max], client)?,
            c: GpuTensor::<R, f32>::empty(vec![m, k_max], client)?,
            b: GpuTensor::<R, f32>::empty(vec![k_max, k_max], client)?,
            d: GpuTensor::<R, f32>::empty(vec![k_max, k_max], client)?,
            gram_partials: GpuTensor::<R, f32>::empty(vec![chunks, k_max, k_max], client)?,
            gemm_partials: GpuTensor::<R, f32>::empty(vec![gemm_elems], client)?,
            sq: GpuTensor::<R, f32>::empty(vec![k_max], client)?,
            norm: GpuTensor::<R, f32>::empty(vec![k_max], client)?,
            inv_norm: GpuTensor::<R, f32>::empty(vec![k_max], client)?,
            obj_partials: GpuTensor::<R, f32>::empty(vec![m.max(n)], client)?,
            eps: GpuTensor::<R, f32>::from_slice(&[eps], vec![1], client)?,
            k_max,
        })
    }

    /// Total VRAM footprint of the scratch buffers.
    ///
    /// ### Returns
    ///
    /// Bytes held on device, excluding `V` and the factors.
    pub fn vram_bytes(&self) -> usize {
        self.a.vram_bytes()
            + self.c.vram_bytes()
            + self.b.vram_bytes()
            + self.d.vram_bytes()
            + self.gram_partials.vram_bytes()
            + self.gemm_partials.vram_bytes()
            + self.sq.vram_bytes()
            + self.norm.vram_bytes()
            + self.inv_norm.vram_bytes()
            + self.obj_partials.vram_bytes()
            + self.eps.vram_bytes()
    }
}

/////////////
// Helpers //
/////////////

/// Frobenius inner product of two `[k, k]` row-major host buffers, in f64.
///
/// ### Params
///
/// * `x` - First buffer, `k * k` values
/// * `y` - Second buffer, `k * k` values
///
/// ### Returns
///
/// `sum_ij x[i, j] * y[i, j]`.
fn inner_kk(x: &[f32], y: &[f32]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum()
}

/// The relative-tolerance denominator, floored at one.
///
/// ### Params
///
/// * `sq_frob` - `||V||_F^2`
///
/// ### Returns
///
/// `max(sq_frob, 1)`.
fn relative_denominator(sq_frob: f32) -> f32 {
    if sq_frob > 1.0 { sq_frob } else { 1.0 }
}

/// Recompute both Grams and `W^T V`, then evaluate `||V - W H||_F^2`.
///
/// Uses the expansion `||V||^2 - 2<A, H> + <B, D>`, so `W H` is never
/// materialised. `<A, H>` comes back as one partial per feature and is summed
/// in f64; the `[k, k]` term is read back whole and reduced in f64 too. Both
/// match [`crate::methods::nmf_hals::compute_objective`].
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `w` - Left factor `[m, k]`
/// * `h` - Right factor transposed, `[n, k]`
/// * `scratch` - Reusable device buffers
/// * `k` - Number of components
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The reconstruction error, clamped at zero, with `b`, `d` and `a` left
/// holding the freshly computed products.
fn objective_gpu<R, In>(
    v: &In,
    w: &GpuTensor<R, f32>,
    h: &GpuTensor<R, f32>,
    scratch: &NmfGpuScratch<R>,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<f32, BixverseErrors>
where
    R: Runtime,
    In: GpuNmfData<R>,
{
    let (m, n) = v.shape();

    gram::<R, f32>(client, w, &scratch.b, &scratch.gram_partials, m, k)?;
    v.wt_v_gpu(w, &scratch.a, k, client, &scratch.gemm_partials)?;
    gram::<R, f32>(client, h, &scratch.d, &scratch.gram_partials, n, k)?;

    launch_row_dot_partials::<R, f32>(&scratch.a, h, &scratch.obj_partials, n, k, client)?;

    let partials = scratch.obj_partials.clone().read(client)?;
    let b_host = scratch.b.clone().read(client)?;
    let d_host = scratch.d.clone().read(client)?;

    let inner_ha: f64 = partials[..n].iter().map(|&x| x as f64).sum();
    // The `[k, k]` buffers are `k_max`-strided allocations used as a `k` prefix,
    // so only the leading `k * k` entries are live.
    let inner_bd = inner_kk(&b_host[..k * k], &d_host[..k * k]);

    let loss = v.sq_frob() as f64 - 2.0 * inner_ha + inner_bd;
    Ok(loss.max(0.0) as f32)
}

/// Normalise the columns of `W` to unit L2 and absorb the norms into `H`.
///
/// Because `H` is held transposed, "scale row `r` of `H`" and "scale column `r`
/// of `W`" are the same kernel over a `[rows, k]` buffer, so this is one norm
/// reduction, one factor pass and two scale passes. Collapsed columns get a
/// factor of one in both directions and so are left alone, as on the CPU.
///
/// ### Params
///
/// * `w` - Left factor `[m, k]`, normalised in place
/// * `h` - Right factor transposed `[n, k]`, rescaled in place
/// * `scratch` - Reusable device buffers
/// * `m` - Number of samples
/// * `n` - Number of features
/// * `k` - Number of components
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, with `W H` unchanged and `W`'s columns unit length.
fn normalise_gpu<R: Runtime>(
    w: &GpuTensor<R, f32>,
    h: &GpuTensor<R, f32>,
    scratch: &NmfGpuScratch<R>,
    m: usize,
    n: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<(), BixverseErrors> {
    launch_dense_column_sq_norm::<R, f32>(w, &scratch.sq, m, k, client)?;
    launch_hals_norm_factors::<R, f32>(&scratch.sq, &scratch.norm, &scratch.inv_norm, k, client)?;
    launch_scale_columns::<R, f32>(w, &scratch.inv_norm, m, k, client)?;
    launch_scale_columns::<R, f32>(h, &scratch.norm, n, k, client)?;
    Ok(())
}

/// Seed `W` and `H` on device from the host initialisation.
///
/// ### Params
///
/// * `w_host` - Left factor `m x k`
/// * `h_host` - Right factor `k x n`
/// * `k_max` - Rank the scratch buffers were sized at
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The device pair `(w, h)` as `[m, k]` and `[n, k]` row-major, see
/// [`GpuFactors`].
fn upload_factors<R: Runtime>(
    w_host: MatRef<f32>,
    h_host: MatRef<f32>,
    k_max: usize,
    client: &ComputeClient<R>,
) -> Result<GpuFactors<R>, BixverseErrors> {
    let (m, k) = (w_host.nrows(), w_host.ncols());
    let n = h_host.ncols();
    debug_assert!(k <= k_max, "rank above what the scratch was sized for");

    let w_flat = w_to_row_major(w_host);
    let h_flat = h_to_row_major(h_host);

    let w = GpuTensor::<R, f32>::from_slice(&w_flat, vec![m, k], client)?;
    let h = GpuTensor::<R, f32>::from_slice(&h_flat, vec![n, k], client)?;
    Ok((w, h))
}

//////////
// Main //
//////////

/// Frobenius HALS NMF on the GPU.
///
/// Same algorithm and same iteration order as
/// [`crate::methods::nmf_hals::nmf_hals`]: Gauss-Seidel, with the Gram and data
/// product recomputed between the `H` sweep and the `W` sweep so the `W` update
/// sees the already-updated `H`. `W` and `H` are initialised on the host and
/// stay on device until the solve finishes; the only readback inside the loop is
/// the objective, once per `check_every` iterations.
///
/// Results are not bit-identical to the CPU path. The factors agree to f32
/// GEMM-ordering differences, which for a converged solve is far below the
/// tolerance that stopped it.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `k` - Number of components
/// * `opts` - Solver options; see [`HalsOpts`]
/// * `scratch` - Reusable device buffers, sized for at least `k`
/// * `client` - CubeCL compute client
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// An [`NmfResult`] with `W` (`m x k`), `H` (`k x n`), the final reconstruction
/// loss, the iteration count and whether the tolerance was met.
///
/// ### Errors
///
/// * `GpuNmfRankTooLarge` if `k` is above what the sweep is compiled for.
/// * `NmfRankTooLarge` if NNDSVD is requested and `V` has fewer than `k`
///   singular values.
/// * `CubeclUtils` or `GpuMatmul` from the device path.
pub fn nmf_hals_gpu<R, In>(
    v: &In,
    k: usize,
    opts: &HalsOpts<f32>,
    scratch: &NmfGpuScratch<R>,
    client: &ComputeClient<R>,
    verbose: usize,
) -> Result<NmfResult<f32>, BixverseErrors>
where
    R: Runtime,
    In: GpuNmfData<R>,
{
    if sweep_tier(k).is_none() || k > scratch.k_max {
        return Err(BixverseErrors::GpuNmfRankTooLarge {
            k,
            max: NMF_MAX_RANK.min(scratch.k_max),
        });
    }

    let (m, n) = v.shape();
    let sq_frob = v.sq_frob();
    let verbosity = parse_verbosity_level(verbose);

    let (w_host, h_host) = match &opts.init {
        NmfInit::Nndsvd => {
            let svd = v.top_k_svd(k)?;
            nndsvd_from_svd(&svd, k, m, n)?
        }
        NmfInit::Random { seed } => random_init(m, n, k, sq_frob, *seed),
    };

    let (w, h) = upload_factors::<R>(w_host.as_ref(), h_host.as_ref(), scratch.k_max, client)?;

    let denom = relative_denominator(sq_frob);
    let mut final_loss = f32::INFINITY;
    let mut last_loss = f32::INFINITY;
    let mut converged = false;
    let mut n_iter = 0usize;

    for iter in 0..opts.max_iter {
        n_iter = iter + 1;

        gram::<R, f32>(client, &w, &scratch.b, &scratch.gram_partials, m, k)?;
        v.wt_v_gpu(&w, &scratch.a, k, client, &scratch.gemm_partials)?;
        launch_hals_sweep::<R, f32>(&h, &scratch.b, &scratch.a, &scratch.eps, n, k, client)?;

        gram::<R, f32>(client, &h, &scratch.d, &scratch.gram_partials, n, k)?;
        v.v_ht_gpu(&h, &scratch.c, k, client, &scratch.gemm_partials)?;
        launch_hals_sweep::<R, f32>(&w, &scratch.d, &scratch.c, &scratch.eps, m, k, client)?;

        normalise_gpu::<R>(&w, &h, scratch, m, n, k, client)?;

        if n_iter.is_multiple_of(opts.check_every) {
            let loss = objective_gpu(v, &w, &h, scratch, k, client)?;
            final_loss = loss;

            if n_iter > opts.check_every {
                if verbosity.normal_verbosity() {
                    println!(
                        "  NMF GPU: Iteration {} out of {} - current loss: {:.2?}",
                        iter + 1,
                        opts.max_iter,
                        loss
                    );
                }

                if (last_loss - loss).abs() / denom < opts.tol {
                    converged = true;
                    if verbosity.normal_verbosity() {
                        println!("  NMF GPU converged successfully after {} iters", iter + 1)
                    };
                    break;
                }
            }
            last_loss = loss;
        }
    }

    // `n_iter == 0` when `max_iter` is zero, and zero is a multiple of
    // everything, so without the first clause the loss stays at infinity.
    if n_iter == 0 || !n_iter.is_multiple_of(opts.check_every) {
        final_loss = objective_gpu(v, &w, &h, scratch, k, client)?;
    }

    let w_flat = w.read(client)?;
    let h_flat = h.read(client)?;

    Ok(NmfResult {
        w: w_from_row_major(&w_flat, m, k),
        h: h_from_row_major(&h_flat, k, n),
        final_loss,
        n_iter,
        converged,
    })
}

//////////////////////////
// Multiple run version //
//////////////////////////

/// Stabilised GPU NMF via random restarts.
///
/// Runs [`nmf_hals_gpu`] `n_runs` times with `base_seed + i`, reusing the
/// uploaded `V` and the whole scratch across every restart, and column-binds the
/// resulting `W` matrices for downstream consensus clustering. As on the CPU the
/// `init` field of `opts` is ignored and random initialisation is always used.
///
/// Restarts run one after another. There is one device, so there is nothing to
/// gain from interleaving them, and a serial order removes the CPU path's
/// dependence on the rayon pool while keeping the same seed-per-index
/// reproducibility.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `k` - Number of components per run
/// * `n_runs` - Number of restarts, at least one
/// * `base_seed` - Seed offset; run `i` uses `base_seed + i`
/// * `opts` - Solver options; `init` is ignored
/// * `scratch` - Reusable device buffers, sized for at least `k`
/// * `client` - CubeCL compute client
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// A [`StabilisedNmfResult`] with column-bound `W`, per-run `H`, per-run losses
/// and convergence flags, and the index of the lowest-loss run.
#[allow(clippy::too_many_arguments)]
pub fn stabilised_nmf_gpu<R, In>(
    v: &In,
    k: usize,
    n_runs: usize,
    base_seed: u64,
    opts: &HalsOpts<f32>,
    scratch: &NmfGpuScratch<R>,
    client: &ComputeClient<R>,
    verbose: usize,
) -> Result<StabilisedNmfResult<f32>, BixverseErrors>
where
    R: Runtime,
    In: GpuNmfData<R>,
{
    assert!(n_runs >= 1, "n_runs must be >= 1");

    let verbosity = parse_verbosity_level(verbose);
    let inner_verbose = verbose.saturating_sub(1);

    let mut runs: Vec<NmfResult<f32>> = Vec::with_capacity(n_runs);
    for i in 0..n_runs {
        let opts_i = HalsOpts {
            max_iter: opts.max_iter,
            tol: opts.tol,
            eps: opts.eps,
            check_every: opts.check_every,
            init: NmfInit::Random {
                seed: base_seed + i as u64,
            },
        };

        runs.push(nmf_hals_gpu(v, k, &opts_i, scratch, client, inner_verbose)?);

        if verbosity.normal_verbosity() {
            println!(" Finished stabilised GPU NMF run: {}", i + 1);
        }
    }

    let (m, _) = v.shape();

    let w_all = Mat::<f32>::from_fn(m, k * n_runs, |row, col| {
        let run_idx = col / k;
        let comp_idx = col % k;
        runs[run_idx].w[(row, comp_idx)]
    });

    let losses: Vec<f32> = runs.iter().map(|r| r.final_loss).collect();
    let converged: Vec<bool> = runs.iter().map(|r| r.converged).collect();
    let h_per_run: Vec<Mat<f32>> = runs.into_iter().map(|r| r.h).collect();

    let best_idx = losses
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(StabilisedNmfResult {
        w_all,
        h_per_run,
        losses,
        converged,
        best_idx,
    })
}

////////////
// Refits //
////////////

/// Refit `H` on the GPU against a frozen `W`.
///
/// `W^T W` and `W^T V` are loop invariants because `W` never changes, so `V` is
/// touched once rather than once per iteration and only `H H^T` is recomputed
/// when the objective is evaluated. `H` starts at `opts.eps` rather than from an
/// initialisation, and no normalisation is applied: `W` is frozen, so its scale
/// convention belongs to the caller.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `w_host` - Frozen left factor `m x k`, must match `v`'s row count
/// * `opts` - Solver options; `init` is ignored
/// * `scratch` - Reusable device buffers, sized for at least `k`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `(H of shape k x n, final ||V - W H||_F^2)`.
///
/// ### Errors
///
/// * `NmfDimensionMismatch` if `w_host` does not match `v`.
/// * `GpuNmfRankTooLarge` if the rank is above what the sweep is compiled for.
pub fn nmf_refit_h_gpu<R, In>(
    v: &In,
    w_host: MatRef<f32>,
    opts: &HalsOpts<f32>,
    scratch: &NmfGpuScratch<R>,
    client: &ComputeClient<R>,
) -> Result<(Mat<f32>, f32), BixverseErrors>
where
    R: Runtime,
    In: GpuNmfData<R>,
{
    let (m, n) = v.shape();
    if w_host.nrows() != m {
        return Err(BixverseErrors::NmfDimensionMismatch {
            expected: m,
            found: w_host.nrows(),
        });
    }
    let k = w_host.ncols();
    if sweep_tier(k).is_none() || k > scratch.k_max {
        return Err(BixverseErrors::GpuNmfRankTooLarge {
            k,
            max: NMF_MAX_RANK.min(scratch.k_max),
        });
    }

    let sq_frob = v.sq_frob();
    let denom = relative_denominator(sq_frob);

    let w_flat = w_to_row_major(w_host);
    let w = GpuTensor::<R, f32>::from_slice(&w_flat, vec![m, k], client)?;
    let h = GpuTensor::<R, f32>::empty(vec![n, k], client)?;
    launch_fill_constant::<R, f32>(&h, &scratch.eps, n * k, client)?;

    // Both loop invariants, because W never changes.
    gram::<R, f32>(client, &w, &scratch.b, &scratch.gram_partials, m, k)?;
    v.wt_v_gpu(&w, &scratch.a, k, client, &scratch.gemm_partials)?;

    let mut final_loss = f32::INFINITY;
    let mut last_loss = f32::INFINITY;
    let mut n_iter = 0usize;

    for iter in 0..opts.max_iter {
        n_iter = iter + 1;
        launch_hals_sweep::<R, f32>(&h, &scratch.b, &scratch.a, &scratch.eps, n, k, client)?;

        if n_iter.is_multiple_of(opts.check_every) {
            let loss = objective_fixed_w_gpu(v, &h, scratch, n, k, client)?;
            final_loss = loss;
            if n_iter > opts.check_every && (last_loss - loss).abs() / denom < opts.tol {
                break;
            }
            last_loss = loss;
        }
    }

    if n_iter == 0 || !n_iter.is_multiple_of(opts.check_every) {
        final_loss = objective_fixed_w_gpu(v, &h, scratch, n, k, client)?;
    }

    let h_flat = h.read(client)?;
    Ok((h_from_row_major(&h_flat, k, n), final_loss))
}

/// Refit `W` on the GPU against a frozen `H`.
///
/// Mirror of [`nmf_refit_h_gpu`]. `H H^T` and `V H^T` are the loop invariants,
/// and the objective uses `||V||^2 - 2<W, V H^T> + <W^T W, H H^T>` so a changing
/// `W` never forces another pass over `V`. No normalisation is applied.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `h_host` - Frozen right factor `k x n`, must match `v`'s column count
/// * `opts` - Solver options; `init` is ignored
/// * `scratch` - Reusable device buffers, sized for at least `k`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `(W of shape m x k, final ||V - W H||_F^2)`.
///
/// ### Errors
///
/// * `NmfDimensionMismatch` if `h_host` does not match `v`.
/// * `GpuNmfRankTooLarge` if the rank is above what the sweep is compiled for.
pub fn nmf_refit_w_gpu<R, In>(
    v: &In,
    h_host: MatRef<f32>,
    opts: &HalsOpts<f32>,
    scratch: &NmfGpuScratch<R>,
    client: &ComputeClient<R>,
) -> Result<(Mat<f32>, f32), BixverseErrors>
where
    R: Runtime,
    In: GpuNmfData<R>,
{
    let (m, n) = v.shape();
    if h_host.ncols() != n {
        return Err(BixverseErrors::NmfDimensionMismatch {
            expected: n,
            found: h_host.ncols(),
        });
    }
    let k = h_host.nrows();
    if sweep_tier(k).is_none() || k > scratch.k_max {
        return Err(BixverseErrors::GpuNmfRankTooLarge {
            k,
            max: NMF_MAX_RANK.min(scratch.k_max),
        });
    }

    let sq_frob = v.sq_frob();
    let denom = relative_denominator(sq_frob);

    let h_flat = h_to_row_major(h_host);
    let h = GpuTensor::<R, f32>::from_slice(&h_flat, vec![n, k], client)?;
    let w = GpuTensor::<R, f32>::empty(vec![m, k], client)?;
    launch_fill_constant::<R, f32>(&w, &scratch.eps, m * k, client)?;

    // Both loop invariants, because H never changes.
    gram::<R, f32>(client, &h, &scratch.d, &scratch.gram_partials, n, k)?;
    v.v_ht_gpu(&h, &scratch.c, k, client, &scratch.gemm_partials)?;

    let mut final_loss = f32::INFINITY;
    let mut last_loss = f32::INFINITY;
    let mut n_iter = 0usize;

    for iter in 0..opts.max_iter {
        n_iter = iter + 1;
        launch_hals_sweep::<R, f32>(&w, &scratch.d, &scratch.c, &scratch.eps, m, k, client)?;

        if n_iter.is_multiple_of(opts.check_every) {
            let loss = objective_fixed_h_gpu(v, &w, scratch, m, k, client)?;
            final_loss = loss;
            if n_iter > opts.check_every && (last_loss - loss).abs() / denom < opts.tol {
                break;
            }
            last_loss = loss;
        }
    }

    if n_iter == 0 || !n_iter.is_multiple_of(opts.check_every) {
        final_loss = objective_fixed_h_gpu(v, &w, scratch, m, k, client)?;
    }

    let w_flat = w.read(client)?;
    Ok((w_from_row_major(&w_flat, m, k), final_loss))
}

/// Objective with `W` frozen: `||V||^2 - 2<A, H> + <B, D>` with `A` and `B`
/// already resident.
///
/// Only `H H^T` is recomputed, which is the point of freezing a factor.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `h` - Right factor transposed, `[n, k]`
/// * `scratch` - Reusable device buffers, with `a` and `b` holding `(W^T V)^T`
///   and `W^T W`
/// * `n` - Number of features
/// * `k` - Number of components
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The reconstruction error, clamped at zero.
fn objective_fixed_w_gpu<R, In>(
    v: &In,
    h: &GpuTensor<R, f32>,
    scratch: &NmfGpuScratch<R>,
    n: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<f32, BixverseErrors>
where
    R: Runtime,
    In: GpuNmfData<R>,
{
    gram::<R, f32>(client, h, &scratch.d, &scratch.gram_partials, n, k)?;
    launch_row_dot_partials::<R, f32>(&scratch.a, h, &scratch.obj_partials, n, k, client)?;

    let partials = scratch.obj_partials.clone().read(client)?;
    let b_host = scratch.b.clone().read(client)?;
    let d_host = scratch.d.clone().read(client)?;

    let inner_ha: f64 = partials[..n].iter().map(|&x| x as f64).sum();
    let inner_bd = inner_kk(&b_host[..k * k], &d_host[..k * k]);

    let loss = v.sq_frob() as f64 - 2.0 * inner_ha + inner_bd;
    Ok(loss.max(0.0) as f32)
}

/// Objective with `H` frozen: `||V||^2 - 2<W, C> + <B, D>` with `C` and `D`
/// already resident.
///
/// Pairs `W` against `V H^T` instead of `H` against `W^T V`, so a changing `W`
/// never forces another pass over `V`. Only `W^T W` is recomputed.
///
/// ### Params
///
/// * `v` - Device-resident `V`
/// * `w` - Left factor `[m, k]`
/// * `scratch` - Reusable device buffers, with `c` and `d` holding `V H^T` and
///   `H H^T`
/// * `m` - Number of samples
/// * `k` - Number of components
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The reconstruction error, clamped at zero.
fn objective_fixed_h_gpu<R, In>(
    v: &In,
    w: &GpuTensor<R, f32>,
    scratch: &NmfGpuScratch<R>,
    m: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Result<f32, BixverseErrors>
where
    R: Runtime,
    In: GpuNmfData<R>,
{
    gram::<R, f32>(client, w, &scratch.b, &scratch.gram_partials, m, k)?;
    // `c` and `w` are both `[m, k]`, so this reduction runs over m rather than n.
    // `obj_partials` is sized for whichever axis is longer, so it serves both.
    launch_row_dot_partials::<R, f32>(w, &scratch.c, &scratch.obj_partials, m, k, client)?;

    let partials = scratch.obj_partials.clone().read(client)?;
    let b_host = scratch.b.clone().read(client)?;
    let d_host = scratch.d.clone().read(client)?;

    let inner_wc: f64 = partials[..m].iter().map(|&x| x as f64).sum();
    let inner_bd = inner_kk(&b_host[..k * k], &d_host[..k * k]);

    let loss = v.sq_frob() as f64 - 2.0 * inner_wc + inner_bd;
    Ok(loss.max(0.0) as f32)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    use crate::core::math::sparse::CompressedSparseData2;
    use crate::methods::nmf_hals::{NmfInput, compute_objective, gram_h_ht, gram_wt_w, nmf_hals};
    use crate::prelude::CompressedSparseFormat;

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

    /// Deterministic non-negative `m x n` matrix with an exact rank-`k` core, so
    /// a converged factorisation has somewhere to converge to.
    fn build_v(m: usize, n: usize, k: usize) -> Mat<f32> {
        let w = Mat::<f32>::from_fn(m, k, |i, c| (((i * 13 + c * 29) % 17) as f32) * 0.1 + 0.05);
        let h = Mat::<f32>::from_fn(k, n, |r, j| (((r * 31 + j * 11) % 19) as f32) * 0.1 + 0.05);
        w * h
    }

    /// A sparse `m x n` matrix with a fixed pattern, dense twin included so the
    /// sparse and dense backends can be checked against the same data.
    fn build_sparse_v(m: usize, n: usize) -> (CompressedSparseData2<f32>, Mat<f32>) {
        let dense = Mat::<f32>::from_fn(m, n, |i, j| {
            if (i * 7 + j * 3) % 5 == 0 {
                (((i * 19 + j * 23) % 13) as f32) * 0.25 + 0.5
            } else {
                0.0
            }
        });
        let csr = CompressedSparseData2::<f32, f32>::from_dense_matrix(
            dense.as_ref(),
            CompressedSparseFormat::Csr,
        );
        (csr, dense)
    }

    /// Flatten a faer `rows x cols` matrix into row-major `[rows, cols]`.
    fn to_row_major(x: MatRef<f32>) -> Vec<f32> {
        let (rows, cols) = (x.nrows(), x.ncols());
        let mut out = vec![0f32; rows * cols];
        for i in 0..rows {
            for c in 0..cols {
                out[i * cols + c] = x[(i, c)];
            }
        }
        out
    }

    /// True when every entry is non-negative.
    fn all_non_negative(x: MatRef<f32>) -> bool {
        (0..x.nrows()).all(|i| (0..x.ncols()).all(|j| x[(i, j)] >= 0.0))
    }

    /// `||A - B||_F^2` in f64.
    fn frob_sq_diff(a: MatRef<f32>, b: MatRef<f32>) -> f64 {
        (0..a.nrows())
            .map(|i| {
                (0..a.ncols())
                    .map(|j| {
                        let d = a[(i, j)] as f64 - b[(i, j)] as f64;
                        d * d
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    /// Relative comparison floored at one, plus the all-zeros guard: a rejected
    /// dispatch leaves the output untouched and reports success, so without the
    /// guard a dead kernel passes.
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

    ////////////////////
    // Layout helpers //
    ////////////////////

    // Both arms of the H reinterpretation, and which one fires. faer rounds the
    // column stride up to 16 elements, so `k = 16` is contiguous and borrows
    // while `k = 5` is padded and copies. Pinning that here is the point: every
    // other test in this module happens to use a padded shape, so without this
    // the borrow arm would run nowhere.
    #[test]
    fn test_h_row_major_round_trips_on_both_arms() {
        for (k, n, expect_borrow) in [(16usize, 11usize, true), (5, 11, false)] {
            let h = Mat::<f32>::from_fn(k, n, |r, j| (r * 100 + j) as f32);
            let flat = h_to_row_major(h.as_ref());
            assert_eq!(
                matches!(flat, std::borrow::Cow::Borrowed(_)),
                expect_borrow,
                "k = {k} took the wrong arm"
            );
            assert_eq!(flat.len(), k * n);
            for j in 0..n {
                for r in 0..k {
                    assert_eq!(flat[j * k + r], h[(r, j)]);
                }
            }
            let back = h_from_row_major(&flat, k, n);
            assert_eq!(to_row_major(back.as_ref()), to_row_major(h.as_ref()));
        }
    }

    #[test]
    fn test_w_row_major_round_trips() {
        let (m, k) = (13usize, 4usize);
        let w = Mat::<f32>::from_fn(m, k, |i, c| (i * 100 + c) as f32);
        let flat = w_to_row_major(w.as_ref());
        for i in 0..m {
            for c in 0..k {
                assert_eq!(flat[i * k + c], w[(i, c)]);
            }
        }
        let back = w_from_row_major(&flat, m, k);
        assert_eq!(to_row_major(back.as_ref()), to_row_major(w.as_ref()));
    }

    /////////////////////
    // Product parity  //
    /////////////////////

    // The two data products are where a layout mistake hides, because a wrong
    // transpose still produces plausible non-negative numbers. Both are checked
    // elementwise against the CPU backend on the same W and H.
    #[test]
    fn test_dense_products_match_cpu() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (43usize, 29usize, 6usize);
        let v = build_v(m, n, k);
        let cpu = DenseInput::new(v.as_ref()).unwrap();
        let gpu = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();

        assert_eq!(GpuNmfData::shape(&gpu), NmfInput::shape(&cpu));
        assert!(
            (GpuNmfData::sq_frob(&gpu) - NmfInput::sq_frob(&cpu)).abs()
                <= 1e-4 * NmfInput::sq_frob(&cpu),
            "Frobenius norms disagree"
        );

        let w = Mat::<f32>::from_fn(m, k, |i, c| (((i * 5 + c * 3) % 7) as f32) * 0.2 + 0.1);
        let h = Mat::<f32>::from_fn(k, n, |r, j| (((r * 7 + j * 5) % 9) as f32) * 0.2 + 0.1);

        let w_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &w_to_row_major(w.as_ref()),
            vec![m, k],
            &client,
        )
        .unwrap();
        let h_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &h_to_row_major(h.as_ref()),
            vec![n, k],
            &client,
        )
        .unwrap();
        let a = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client).unwrap();
        let c = GpuTensor::<WgpuRuntime, f32>::empty(vec![m, k], &client).unwrap();

        // Sized the way the solver sizes it, so the split-K arm is reached
        // exactly when it would be in a real solve.
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, 1e-10, &client).unwrap();
        gpu.wt_v_gpu(&w_gpu, &a, k, &client, &scratch.gemm_partials)
            .unwrap();
        gpu.v_ht_gpu(&h_gpu, &c, k, &client, &scratch.gemm_partials)
            .unwrap();

        let mut wtv = Mat::<f32>::zeros(k, n);
        let mut vht = Mat::<f32>::zeros(m, k);
        cpu.wt_v(w.as_ref(), &mut wtv);
        cpu.v_ht(h.as_ref(), &mut vht);

        // The device holds `(W^T V)^T`, so the reference is transposed.
        assert_close(
            &a.read(&client).unwrap(),
            &to_row_major(wtv.transpose()),
            1e-4,
        );
        assert_close(&c.read(&client).unwrap(), &to_row_major(vht.as_ref()), 1e-4);
    }

    // The same check at a shape where every faer matrix is genuinely contiguous,
    // so `V` uploads by borrow and `H` reinterprets by borrow. `m`, `n` and `k`
    // are all multiples of 16, which is faer's column-stride alignment. The
    // padded arm is what the test above exercises; this one covers the arm that
    // fires at production shapes, where `V` is the 400 MB upload.
    #[test]
    fn test_dense_products_match_cpu_on_the_contiguous_arm() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (48usize, 32usize, 16usize);
        let v = build_v(m, n, k);
        assert!(
            contiguous_col_major(v.as_ref()).is_some(),
            "this shape was supposed to take the borrow arm"
        );

        let cpu = DenseInput::new(v.as_ref()).unwrap();
        let gpu = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();

        let w = Mat::<f32>::from_fn(m, k, |i, c| (((i * 5 + c * 3) % 7) as f32) * 0.2 + 0.1);
        let h = Mat::<f32>::from_fn(k, n, |r, j| (((r * 7 + j * 5) % 9) as f32) * 0.2 + 0.1);
        assert!(
            matches!(h_to_row_major(h.as_ref()), std::borrow::Cow::Borrowed(_)),
            "this H was supposed to take the borrow arm"
        );

        let w_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &w_to_row_major(w.as_ref()),
            vec![m, k],
            &client,
        )
        .unwrap();
        let h_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &h_to_row_major(h.as_ref()),
            vec![n, k],
            &client,
        )
        .unwrap();
        let a = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client).unwrap();
        let c = GpuTensor::<WgpuRuntime, f32>::empty(vec![m, k], &client).unwrap();

        // Sized the way the solver sizes it, so the split-K arm is reached
        // exactly when it would be in a real solve.
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, 1e-10, &client).unwrap();
        gpu.wt_v_gpu(&w_gpu, &a, k, &client, &scratch.gemm_partials)
            .unwrap();
        gpu.v_ht_gpu(&h_gpu, &c, k, &client, &scratch.gemm_partials)
            .unwrap();

        let mut wtv = Mat::<f32>::zeros(k, n);
        let mut vht = Mat::<f32>::zeros(m, k);
        cpu.wt_v(w.as_ref(), &mut wtv);
        cpu.v_ht(h.as_ref(), &mut vht);

        assert_close(
            &a.read(&client).unwrap(),
            &to_row_major(wtv.transpose()),
            1e-4,
        );
        assert_close(&c.read(&client).unwrap(), &to_row_major(vht.as_ref()), 1e-4);
    }

    #[test]
    fn test_sparse_products_match_cpu() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (37usize, 23usize, 5usize);
        let (csr, dense) = build_sparse_v(m, n);
        let cpu = SparseInput::<f32, f32>::from_primary(&csr).unwrap();
        let gpu = GpuSparseNmfInput::<WgpuRuntime>::new(
            SparseInput::<f32, f32>::from_primary(&csr).unwrap(),
            &client,
        )
        .unwrap();

        assert_eq!(GpuNmfData::shape(&gpu), (m, n));
        let dense_cpu = DenseInput::new(dense.as_ref()).unwrap();
        assert!(
            (GpuNmfData::sq_frob(&gpu) - NmfInput::sq_frob(&dense_cpu)).abs()
                <= 1e-4 * NmfInput::sq_frob(&dense_cpu),
            "sparse and dense Frobenius norms disagree"
        );

        let w = Mat::<f32>::from_fn(m, k, |i, c| (((i * 5 + c * 3) % 7) as f32) * 0.2 + 0.1);
        let h = Mat::<f32>::from_fn(k, n, |r, j| (((r * 7 + j * 5) % 9) as f32) * 0.2 + 0.1);

        let w_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &w_to_row_major(w.as_ref()),
            vec![m, k],
            &client,
        )
        .unwrap();
        let h_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &h_to_row_major(h.as_ref()),
            vec![n, k],
            &client,
        )
        .unwrap();
        let a = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, k], &client).unwrap();
        let c = GpuTensor::<WgpuRuntime, f32>::empty(vec![m, k], &client).unwrap();

        // Sized the way the solver sizes it, so the split-K arm is reached
        // exactly when it would be in a real solve.
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, 1e-10, &client).unwrap();
        gpu.wt_v_gpu(&w_gpu, &a, k, &client, &scratch.gemm_partials)
            .unwrap();
        gpu.v_ht_gpu(&h_gpu, &c, k, &client, &scratch.gemm_partials)
            .unwrap();

        let mut wtv = Mat::<f32>::zeros(k, n);
        let mut vht = Mat::<f32>::zeros(m, k);
        cpu.wt_v(w.as_ref(), &mut wtv);
        cpu.v_ht(h.as_ref(), &mut vht);

        assert_close(
            &a.read(&client).unwrap(),
            &to_row_major(wtv.transpose()),
            1e-4,
        );
        assert_close(&c.read(&client).unwrap(), &to_row_major(vht.as_ref()), 1e-4);
    }

    // Both Grams come from `gram` unchanged, which only works because W is
    // `[m, k]` and H is held transposed as `[n, k]`.
    #[test]
    fn test_grams_match_cpu() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (97usize, 61usize, 8usize);
        let w = Mat::<f32>::from_fn(m, k, |i, c| (((i * 5 + c * 3) % 7) as f32) * 0.2 + 0.1);
        let h = Mat::<f32>::from_fn(k, n, |r, j| (((r * 7 + j * 5) % 9) as f32) * 0.2 + 0.1);

        let w_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &w_to_row_major(w.as_ref()),
            vec![m, k],
            &client,
        )
        .unwrap();
        let h_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &h_to_row_major(h.as_ref()),
            vec![n, k],
            &client,
        )
        .unwrap();
        let g = GpuTensor::<WgpuRuntime, f32>::empty(vec![k, k], &client).unwrap();
        let chunks = gram_chunks(m).max(gram_chunks(n));
        let partials = GpuTensor::<WgpuRuntime, f32>::empty(vec![chunks, k, k], &client).unwrap();

        gram::<WgpuRuntime, f32>(&client, &w_gpu, &g, &partials, m, k).unwrap();
        let mut wtw = Mat::<f32>::zeros(k, k);
        gram_wt_w(w.as_ref(), &mut wtw);
        assert_close(&g.read(&client).unwrap(), &to_row_major(wtw.as_ref()), 1e-4);

        let g2 = GpuTensor::<WgpuRuntime, f32>::empty(vec![k, k], &client).unwrap();
        gram::<WgpuRuntime, f32>(&client, &h_gpu, &g2, &partials, n, k).unwrap();
        let mut hht = Mat::<f32>::zeros(k, k);
        gram_h_ht(h.as_ref(), &mut hht);
        assert_close(
            &g2.read(&client).unwrap(),
            &to_row_major(hht.as_ref()),
            1e-4,
        );
    }

    #[test]
    fn test_objective_matches_cpu() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (53usize, 41usize, 6usize);
        let v = build_v(m, n, k);
        let cpu = DenseInput::new(v.as_ref()).unwrap();
        let gpu = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, 1e-10, &client).unwrap();

        let w = Mat::<f32>::from_fn(m, k, |i, c| (((i * 5 + c * 3) % 7) as f32) * 0.2 + 0.1);
        let h = Mat::<f32>::from_fn(k, n, |r, j| (((r * 7 + j * 5) % 9) as f32) * 0.2 + 0.1);

        let (w_gpu, h_gpu) =
            upload_factors::<WgpuRuntime>(w.as_ref(), h.as_ref(), k, &client).unwrap();
        let got = objective_gpu(&gpu, &w_gpu, &h_gpu, &scratch, k, &client).unwrap();

        let mut wtv = Mat::<f32>::zeros(k, n);
        let mut wtw = Mat::<f32>::zeros(k, k);
        let mut hht = Mat::<f32>::zeros(k, k);
        cpu.wt_v(w.as_ref(), &mut wtv);
        gram_wt_w(w.as_ref(), &mut wtw);
        gram_h_ht(h.as_ref(), &mut hht);
        let want = compute_objective(
            NmfInput::sq_frob(&cpu),
            h.as_ref(),
            wtv.as_ref(),
            wtw.as_ref(),
            hht.as_ref(),
        );

        assert!(got > 0.0, "objective is zero, the GPU did no work");
        assert!(
            (got - want).abs() <= 1e-3 * want.abs().max(1.0),
            "got {got} want {want}"
        );
    }

    /////////////////////
    // Solver parity   //
    /////////////////////

    /// Run one CPU and one GPU solve from the same random initialisation.
    fn solve_pair(
        m: usize,
        n: usize,
        k: usize,
        max_iter: usize,
    ) -> (NmfResult<f32>, NmfResult<f32>, f32) {
        let device = try_device().expect("no device");
        let client = WgpuRuntime::client(&device);

        let v = build_v(m, n, k);
        let opts = HalsOpts::<f32> {
            max_iter,
            tol: 1e-9,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Random { seed: 7 },
        };

        let cpu_in = DenseInput::new(v.as_ref()).unwrap();
        let cpu = nmf_hals(&cpu_in, k, &opts, 0).unwrap();

        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let gpu = nmf_hals_gpu(&gpu_in, k, &opts, &scratch, &client, 0).unwrap();

        (cpu, gpu, NmfInput::sq_frob(&cpu_in))
    }

    // One iteration from an identical initialisation. This is the tightest
    // agreement the two paths can be held to, and it pins the whole loop: Gram,
    // product, sweep, Gram, product, sweep, normalise, in that order. A
    // reordering that still converges would pass the recovery tests below and
    // fail here.
    #[test]
    fn test_one_iteration_matches_cpu_elementwise() {
        if try_device().is_none() {
            return;
        }
        let (m, n, k) = (47usize, 31usize, 5usize);
        let (cpu, gpu, _) = solve_pair(m, n, k, 1);

        assert_eq!(cpu.n_iter, 1);
        assert_eq!(gpu.n_iter, 1);
        assert_close(
            &to_row_major(gpu.w.as_ref()),
            &to_row_major(cpu.w.as_ref()),
            1e-4,
        );
        assert_close(
            &to_row_major(gpu.h.as_ref()),
            &to_row_major(cpu.h.as_ref()),
            1e-4,
        );
    }

    // To convergence the two paths are not expected to agree elementwise: f32
    // GEMM ordering differs, and HALS amplifies that over hundreds of
    // iterations. The contract is the reconstruction, so compare the losses.
    #[test]
    fn test_converged_loss_matches_cpu() {
        if try_device().is_none() {
            return;
        }
        let (m, n, k) = (47usize, 31usize, 5usize);
        let (cpu, gpu, sq_frob) = solve_pair(m, n, k, 300);

        let cpu_rel = cpu.final_loss / sq_frob;
        let gpu_rel = gpu.final_loss / sq_frob;
        assert!(
            (cpu_rel - gpu_rel).abs() < 1e-3,
            "relative losses disagree: cpu {cpu_rel} gpu {gpu_rel}"
        );
    }

    #[test]
    fn test_recovers_rank_k_dense() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (60usize, 40usize, 3usize);
        let v = build_v(m, n, k);
        let opts = HalsOpts::<f32> {
            max_iter: 400,
            tol: 1e-9,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Nndsvd,
        };

        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let res = nmf_hals_gpu(&gpu_in, k, &opts, &scratch, &client, 0).unwrap();

        let rel = res.final_loss / GpuNmfData::sq_frob(&gpu_in);
        assert!(
            rel < 1e-3,
            "relative loss {rel} is too high for an exact rank-k input"
        );
        assert!(all_non_negative(res.w.as_ref()), "W has a negative entry");
        assert!(all_non_negative(res.h.as_ref()), "H has a negative entry");
    }

    #[test]
    fn test_recovers_rank_k_sparse() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (48usize, 32usize, 3usize);
        let (csr, dense) = build_sparse_v(m, n);
        let opts = HalsOpts::<f32> {
            max_iter: 400,
            tol: 1e-9,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Random { seed: 3 },
        };

        let host = SparseInput::<f32, f32>::from_primary(&csr).unwrap();
        let gpu_in = GpuSparseNmfInput::<WgpuRuntime>::new(host, &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let sparse_res = nmf_hals_gpu(&gpu_in, k, &opts, &scratch, &client, 0).unwrap();

        // The sparse and dense backends see the same matrix, so they must reach
        // the same reconstruction from the same seed.
        let dense_in = GpuDenseNmfInput::<WgpuRuntime>::new(dense.as_ref(), &client).unwrap();
        let dense_res = nmf_hals_gpu(&dense_in, k, &opts, &scratch, &client, 0).unwrap();

        let sq_frob = GpuNmfData::sq_frob(&gpu_in);
        let rel_sparse = sparse_res.final_loss / sq_frob;
        let rel_dense = dense_res.final_loss / sq_frob;
        assert!(
            (rel_sparse - rel_dense).abs() < 1e-3,
            "sparse {rel_sparse} and dense {rel_dense} backends disagree"
        );
        assert!(all_non_negative(sparse_res.w.as_ref()));
    }

    // Same seed, same answer. Every kernel here has a fixed reduction order and
    // no atomics, so there is nothing left to vary between runs.
    #[test]
    fn test_deterministic_with_same_seed() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (40usize, 28usize, 4usize);
        let v = build_v(m, n, k);
        let opts = HalsOpts::<f32> {
            max_iter: 60,
            tol: 1e-9,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Random { seed: 11 },
        };

        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let first = nmf_hals_gpu(&gpu_in, k, &opts, &scratch, &client, 0).unwrap();
        let second = nmf_hals_gpu(&gpu_in, k, &opts, &scratch, &client, 0).unwrap();

        assert_eq!(first.final_loss, second.final_loss);
        assert_eq!(
            to_row_major(first.w.as_ref()),
            to_row_major(second.w.as_ref())
        );
        assert_eq!(
            to_row_major(first.h.as_ref()),
            to_row_major(second.h.as_ref())
        );
    }

    // A realistic single-cell shape, where the CPU reference is the expensive
    // half. Small shapes cannot catch a split-K arm that only opens at a long
    // reduction, nor a grid that only gets wide past a few thousand rows, so the
    // cheap tests above do not cover what this one does.
    #[test]
    // Heavy: 20000 x 2000 dense at k = 24 against a full CPU solve.
    #[cfg(feature = "large-test")]
    fn test_large_shape_matches_cpu() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (20_000usize, 2_000usize, 24usize);
        let v = build_v(m, n, k);
        let opts = HalsOpts::<f32> {
            max_iter: 30,
            tol: 0.0,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Random { seed: 13 },
        };

        let cpu_in = DenseInput::new(v.as_ref()).unwrap();
        let cpu = nmf_hals(&cpu_in, k, &opts, 0).unwrap();

        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let gpu = nmf_hals_gpu(&gpu_in, k, &opts, &scratch, &client, 0).unwrap();

        assert_eq!(cpu.n_iter, gpu.n_iter);
        assert!(gpu.final_loss.is_finite() && gpu.final_loss > 0.0);
        assert!(all_non_negative(gpu.w.as_ref()));
        assert!(all_non_negative(gpu.h.as_ref()));

        // The two do not agree elementwise here and are not expected to: f32
        // reduction orders differ and HALS amplifies that over 30 iterations. The
        // contract is the reconstruction, and the GPU's own f64 Frobenius norm is
        // the reference, because the CPU's f32 accumulation of `||V||^2` is off by
        // percent-level amounts at this size.
        let sq_frob = GpuNmfData::sq_frob(&gpu_in);
        let direct_gpu = frob_sq_diff((&gpu.w * &gpu.h).as_ref(), v.as_ref()) / sq_frob as f64;
        let direct_cpu = frob_sq_diff((&cpu.w * &cpu.h).as_ref(), v.as_ref()) / sq_frob as f64;
        assert!(
            (direct_gpu - direct_cpu).abs() < 5e-3,
            "reconstructions disagree: cpu {direct_cpu} gpu {direct_gpu}"
        );

        // The GPU's own reported loss has to match its own reconstruction.
        assert!(
            (gpu.final_loss as f64 / sq_frob as f64 - direct_gpu).abs() < 1e-4,
            "reported loss disagrees with the direct residual"
        );
    }

    #[test]
    fn test_rejects_rank_above_the_scratch() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n) = (20usize, 15usize);
        let v = build_v(m, n, 3);
        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, 4, 1e-10, &client).unwrap();
        let opts = HalsOpts::<f32>::default();

        assert!(matches!(
            nmf_hals_gpu(&gpu_in, 8, &opts, &scratch, &client, 0),
            Err(BixverseErrors::GpuNmfRankTooLarge { k: 8, .. })
        ));
        assert!(matches!(
            NmfGpuScratch::<WgpuRuntime>::new(m, n, NMF_MAX_RANK + 1, 1e-10, &client),
            Err(BixverseErrors::GpuNmfRankTooLarge { .. })
        ));
    }

    ///////////////////
    // Restarts      //
    ///////////////////

    #[test]
    fn test_stabilised_bookkeeping() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k, n_runs) = (40usize, 26usize, 4usize, 3usize);
        let v = build_v(m, n, k);
        let opts = HalsOpts::<f32> {
            max_iter: 40,
            tol: 1e-6,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Nndsvd,
        };

        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let res = stabilised_nmf_gpu(&gpu_in, k, n_runs, 100, &opts, &scratch, &client, 0).unwrap();

        assert_eq!(res.w_all.nrows(), m);
        assert_eq!(res.w_all.ncols(), k * n_runs);
        assert_eq!(res.h_per_run.len(), n_runs);
        assert_eq!(res.losses.len(), n_runs);
        assert_eq!(res.converged.len(), n_runs);

        let best = res
            .losses
            .iter()
            .cloned()
            .fold(f32::INFINITY, |a, b| a.min(b));
        assert_eq!(res.losses[res.best_idx], best);

        // `init` is ignored: every run is a distinct random restart, so no two
        // W blocks may agree.
        let diff: f32 = (0..m)
            .map(|i| (res.w_all[(i, 0)] - res.w_all[(i, k)]).abs())
            .sum();
        assert!(diff > 1e-6, "restarts 0 and 1 produced the same W");
    }

    #[test]
    fn test_stabilised_deterministic_with_same_seed() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (36usize, 24usize, 4usize);
        let v = build_v(m, n, k);
        let opts = HalsOpts::<f32> {
            max_iter: 30,
            tol: 1e-6,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Nndsvd,
        };

        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let a = stabilised_nmf_gpu(&gpu_in, k, 3, 42, &opts, &scratch, &client, 0).unwrap();
        let b = stabilised_nmf_gpu(&gpu_in, k, 3, 42, &opts, &scratch, &client, 0).unwrap();

        assert_eq!(a.losses, b.losses);
        assert_eq!(
            to_row_major(a.w_all.as_ref()),
            to_row_major(b.w_all.as_ref())
        );
    }

    ////////////
    // Refits //
    ////////////

    // A refit's contract is the reconstruction, not recovery of the factor that
    // generated the data: the frozen factor need not be full rank.
    #[test]
    fn test_refit_h_reconstructs_a_rank_k_input() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (50usize, 34usize, 3usize);
        let w_true =
            Mat::<f32>::from_fn(m, k, |i, c| (((i * 13 + c * 29) % 17) as f32) * 0.1 + 0.05);
        let h_true =
            Mat::<f32>::from_fn(k, n, |r, j| (((r * 31 + j * 11) % 19) as f32) * 0.1 + 0.05);
        let v = &w_true * &h_true;

        let opts = HalsOpts::<f32> {
            max_iter: 600,
            tol: 1e-12,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Nndsvd,
        };

        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let (h_fit, loss) =
            nmf_refit_h_gpu(&gpu_in, w_true.as_ref(), &opts, &scratch, &client).unwrap();

        assert_eq!(h_fit.nrows(), k);
        assert_eq!(h_fit.ncols(), n);

        // Check the reported loss against an independent reconstruction.
        let direct = frob_sq_diff((&w_true * &h_fit).as_ref(), v.as_ref());
        let sq_frob = GpuNmfData::sq_frob(&gpu_in) as f64;
        assert!(
            direct / sq_frob < 1e-6,
            "reconstruction is poor: {}",
            direct / sq_frob
        );
        assert!(
            (loss as f64 - direct).abs() <= 1e-3 * sq_frob,
            "reported loss {loss} disagrees with the direct residual {direct}"
        );
    }

    #[test]
    fn test_refit_w_reconstructs_a_rank_k_input() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (50usize, 34usize, 3usize);
        let w_true =
            Mat::<f32>::from_fn(m, k, |i, c| (((i * 13 + c * 29) % 17) as f32) * 0.1 + 0.05);
        let h_true =
            Mat::<f32>::from_fn(k, n, |r, j| (((r * 31 + j * 11) % 19) as f32) * 0.1 + 0.05);
        let v = &w_true * &h_true;

        let opts = HalsOpts::<f32> {
            max_iter: 600,
            tol: 1e-12,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Nndsvd,
        };

        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let (w_fit, loss) =
            nmf_refit_w_gpu(&gpu_in, h_true.as_ref(), &opts, &scratch, &client).unwrap();

        assert_eq!(w_fit.nrows(), m);
        assert_eq!(w_fit.ncols(), k);

        let direct = frob_sq_diff((&w_fit * &h_true).as_ref(), v.as_ref());
        let sq_frob = GpuNmfData::sq_frob(&gpu_in) as f64;
        assert!(
            direct / sq_frob < 1e-6,
            "reconstruction is poor: {}",
            direct / sq_frob
        );
        assert!(
            (loss as f64 - direct).abs() <= 1e-3 * sq_frob,
            "reported loss {loss} disagrees with the direct residual {direct}"
        );
    }

    #[test]
    fn test_refits_reject_a_dimension_mismatch() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (20usize, 14usize, 3usize);
        let v = build_v(m, n, k);
        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, 1e-10, &client).unwrap();
        let opts = HalsOpts::<f32>::default();

        let bad_w = Mat::<f32>::full(m + 1, k, 0.5);
        assert!(matches!(
            nmf_refit_h_gpu(&gpu_in, bad_w.as_ref(), &opts, &scratch, &client),
            Err(BixverseErrors::NmfDimensionMismatch { .. })
        ));

        let bad_h = Mat::<f32>::full(k, n + 1, 0.5);
        assert!(matches!(
            nmf_refit_w_gpu(&gpu_in, bad_h.as_ref(), &opts, &scratch, &client),
            Err(BixverseErrors::NmfDimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_zero_iterations_reports_a_finite_loss() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);

        let (m, n, k) = (24usize, 18usize, 3usize);
        let v = build_v(m, n, k);
        let opts = HalsOpts::<f32> {
            max_iter: 0,
            tol: 1e-6,
            eps: 1e-10,
            check_every: 10,
            init: NmfInit::Nndsvd,
        };

        let gpu_in = GpuDenseNmfInput::<WgpuRuntime>::new(v.as_ref(), &client).unwrap();
        let scratch = NmfGpuScratch::<WgpuRuntime>::new(m, n, k, opts.eps, &client).unwrap();
        let res = nmf_hals_gpu(&gpu_in, k, &opts, &scratch, &client, 0).unwrap();
        assert!(res.final_loss.is_finite(), "loss was left at infinity");
        assert_eq!(res.n_iter, 0);
        assert!(!res.converged);
    }
}
