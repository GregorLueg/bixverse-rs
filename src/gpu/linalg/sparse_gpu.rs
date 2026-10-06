//! GPU-resident compressed sparse matrix. Single-layout, layer-selected at
//! upload time. Hold two of these (one CSR, one CSC) when both directions
//! of SpMM are needed; [`csc_to_csr_gpu`] builds the second on the device
//! from the first, so only one layout crosses the bus.

// The `#[cube]` macro generates undocumented launcher structs and functions.
#![allow(missing_docs)]

use cubecl::Runtime;
use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use rayon::prelude::*;

use crate::gpu::WORKGROUP_256;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Columns covered by one word of the per-row column masks in
/// [`csc_to_csr_gpu`].
const MASK_BITS: usize = 32;

/// Device memory for one column group of [`csc_to_csr_gpu`]. The masks and the
/// slot bases are both `[words, n]` u32, so a group of `g` words costs
/// `8 g n` bytes. 256 MB takes 2000 HVGs at up to ~500k cells in one group.
const CSR_BUILD_BUDGET_BYTES: usize = 256 << 20;

///////////////////////
// GPU sparse format //
///////////////////////

/// GPU version of CompressedSparseData. Same general idea as
/// [CompressedSparseData2] with a single data layer.
pub struct GpuCompressedSparseData<R, F>
where
    R: Runtime,
    F: cubecl::CubeElement + cubecl::prelude::Numeric,
{
    /// Indptr of the data. Stored as `u32`
    pub indptr: GpuTensor<R, u32>,
    /// Indices of the data. Stored as `u32`
    pub indices: GpuTensor<R, u32>,
    /// Values of the data. Can be `u32` for raw counts or `fp32` for normalised
    /// counts.
    pub values: GpuTensor<R, F>,
    /// Enum defining if the data is stored in CSC or CSR, see
    /// [CompressedSparseFormat]
    pub cs_type: CompressedSparseFormat,
    /// The shape of the data, `(nrow, ncol)`
    pub shape: (usize, usize),
    /// Number of NNZ in the data
    pub nnz: usize,
}

/// Clone implementation for the [GpuCompressedSparseData]
impl<R, F> Clone for GpuCompressedSparseData<R, F>
where
    R: Runtime,
    F: cubecl::CubeElement + cubecl::prelude::Numeric,
{
    fn clone(&self) -> Self {
        Self {
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            values: self.values.clone(),
            cs_type: self.cs_type,
            shape: self.shape,
            nnz: self.nnz,
        }
    }
}

impl<R, F> GpuCompressedSparseData<R, F>
where
    R: Runtime,
    F: cubecl::CubeElement + cubecl::prelude::Numeric,
{
    /// Upload pre-built host parts. Use when values have already been
    /// converted to `F`.
    ///
    /// ### Params
    ///
    /// * `values` - The data values
    /// * `indices` - The data indices
    /// * `indptr` - The index pointers
    /// * `cs_type` - Enum defining the storage format (assuming samples x
    ///   features).
    /// * `shape` - Enum of `(nrow, ncol)`
    /// * `client` - The client on which to host the data
    ///
    /// ### Returns
    ///
    /// Self, or `CubeclUtils` if any of the three buffers busts the device's
    /// per-binding size limit.
    pub fn from_parts(
        values: &[F],
        indices: &[u32],
        indptr: &[u32],
        cs_type: CompressedSparseFormat,
        shape: (usize, usize),
        client: &ComputeClient<R>,
    ) -> Result<Self, BixverseErrors> {
        let nnz = values.len();
        Ok(Self {
            indptr: GpuTensor::from_slice(indptr, vec![indptr.len()], client)?,
            indices: GpuTensor::from_slice(indices, vec![nnz], client)?,
            values: GpuTensor::from_slice(values, vec![nnz], client)?,
            cs_type,
            shape,
            nnz,
        })
    }

    /// Upload a layer of a host-side `CompressedSparseData2`.
    ///
    /// ### Params
    ///
    /// * `src` - The host-side compressed sparse matrix.
    /// * `use_second_layer` - If `true`, uploads `data_2` (errors if
    ///   missing). If `false`, uploads `data`.
    /// * `client` - CubeCL compute client.
    ///
    /// ### Returns
    ///
    /// Self, `Data2NotAvailable` if `use_second_layer` is set and `src` has no
    /// second layer, or `CubeclUtils` if any of the three buffers busts the
    /// device's per-binding size limit.
    pub fn from_compressed_sparse_data_2<T, U>(
        src: &CompressedSparseData2<T, U>,
        use_second_layer: bool,
        client: &ComputeClient<R>,
    ) -> Result<Self, BixverseErrors>
    where
        T: BixverseNumeric + Into<F>,
        U: BixverseNumeric + Into<F>,
    {
        let values: Vec<F> = if use_second_layer {
            let layer = src
                .data_2
                .as_ref()
                .ok_or(BixverseErrors::Data2NotAvailable)?;
            layer.iter().copied().map(Into::into).collect()
        } else {
            src.data.iter().copied().map(Into::into).collect()
        };

        Self::from_parts(
            &values,
            &src.indices,
            &src.indptr,
            src.cs_type,
            src.shape,
            client,
        )
    }

    /// Total VRAM footprint of the three buffers.
    ///
    /// ### Returns
    ///
    /// The size on the GPU
    pub fn vram_bytes(&self) -> usize {
        self.indptr.vram_bytes() + self.indices.vram_bytes() + self.values.vram_bytes()
    }
}

//////////////////////////
// CSC to CSR on device //
//////////////////////////

/// Column of CSC position `p` within `[col_lo, col_hi)`.
///
/// Largest `c` with `col_indptr[c] <= p`, so empty columns are skipped.
///
/// ### Params
///
/// * `col_indptr` - CSC column pointers `[m + 1]`
/// * `p` - Position into the CSC row indices and values
/// * `col_lo` - First column of the group
/// * `col_hi` - One past the last column of the group
///
/// ### Returns
///
/// The column holding position `p`.
#[cube]
fn column_of_position(col_indptr: &Tensor<u32>, p: u32, col_lo: u32, col_hi: u32) -> u32 {
    let mut lo = col_lo;
    let mut hi = col_hi;
    while hi - lo > 1u32 {
        let mid = (lo + hi) / 2u32;
        if col_indptr[mid as usize] <= p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Zero a `u32` buffer used through atomics.
///
/// ### Params
///
/// * `buf` - Buffer to clear
/// * `len` - Elements to clear
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X` ->
///   element
#[cube(launch_unchecked)]
pub fn csr_zero_masks(buf: &mut Tensor<Atomic<u32>>, len: u32) {
    let t = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X;
    if t >= len {
        terminate!();
    }
    Atomic::store(&buf[t as usize], 0u32);
}

/// Mark which columns of a group each row holds.
///
/// Sets bit `(c - col_lo) % 32` of `mask[(c - col_lo) / 32, row]` for every
/// non-zero of the group. OR is order-independent, so the masks are
/// deterministic.
///
/// ### Params
///
/// * `col_indptr` - CSC column pointers `[m + 1]`
/// * `row_idx` - CSC row indices `[nnz]`
/// * `mask` - Column masks `[words, n_rows]`, zeroed
/// * `col_lo` - First column of the group, a multiple of 32
/// * `col_hi` - One past the last column of the group
/// * `nnz_lo` - First CSC position of the group
/// * `nnz_len` - Non-zeros in the group
/// * `n_rows` - Rows of the matrix
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X` ->
///   offset into the group's non-zeros
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn csr_mark_columns(
    col_indptr: &Tensor<u32>,
    row_idx: &Tensor<u32>,
    mask: &mut Tensor<Atomic<u32>>,
    col_lo: u32,
    col_hi: u32,
    nnz_lo: u32,
    nnz_len: u32,
    n_rows: u32,
) {
    let t = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X;
    if t >= nnz_len {
        terminate!();
    }
    let p = nnz_lo + t;
    let local = column_of_position(col_indptr, p, col_lo, col_hi) - col_lo;
    let row = row_idx[p as usize];
    let slot = (local / 32u32) * n_rows + row;
    Atomic::fetch_or(&mask[slot as usize], 1u32 << (local % 32u32));
}

/// Turn a row's column masks into CSR slots.
///
/// `base[w, row]` is the slot of the row's first non-zero in mask word `w`.
/// `cursor` carries each row's next free slot across column groups.
///
/// ### Params
///
/// * `mask` - Column masks `[words, n_rows]`
/// * `base` - Output slot bases `[words, n_rows]`
/// * `cursor` - Next free CSR slot per row `[n_rows]`, advanced in place
/// * `n_rows` - Rows of the matrix
/// * `words` - Mask words in this group
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X` ->
///   row
#[cube(launch_unchecked)]
pub fn csr_rank_rows(
    mask: &Tensor<Atomic<u32>>,
    base: &mut Tensor<u32>,
    cursor: &mut Tensor<u32>,
    n_rows: u32,
    words: u32,
) {
    let row = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X;
    if row >= n_rows {
        terminate!();
    }
    let mut next = cursor[row as usize];
    for w in 0..words {
        let slot = (w * n_rows + row) as usize;
        base[slot] = next;
        next += u32::count_ones(Atomic::load(&mask[slot]));
    }
    cursor[row as usize] = next;
}

/// Scatter a column group's non-zeros into the CSR.
///
/// The slot of a non-zero is its word's base plus the number of the row's
/// columns below it in the same word, so rows come out in column order, as a
/// host transpose would produce them.
///
/// ### Params
///
/// * `col_indptr` - CSC column pointers `[m + 1]`
/// * `row_idx` - CSC row indices `[nnz]`
/// * `values` - CSC values `[nnz]`
/// * `mask` - Column masks `[words, n_rows]`
/// * `base` - Slot bases `[words, n_rows]`
/// * `out_indices` - CSR column indices `[nnz]`
/// * `out_values` - CSR values `[nnz]`
/// * `col_lo` - First column of the group, a multiple of 32
/// * `col_hi` - One past the last column of the group
/// * `nnz_lo` - First CSC position of the group
/// * `nnz_len` - Non-zeros in the group
/// * `n_rows` - Rows of the matrix
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X` ->
///   offset into the group's non-zeros
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn csr_scatter_group<F: Numeric>(
    col_indptr: &Tensor<u32>,
    row_idx: &Tensor<u32>,
    values: &Tensor<F>,
    mask: &Tensor<Atomic<u32>>,
    base: &Tensor<u32>,
    out_indices: &mut Tensor<u32>,
    out_values: &mut Tensor<F>,
    col_lo: u32,
    col_hi: u32,
    nnz_lo: u32,
    nnz_len: u32,
    n_rows: u32,
) {
    let t = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X;
    if t >= nnz_len {
        terminate!();
    }
    let p = nnz_lo + t;
    let col = column_of_position(col_indptr, p, col_lo, col_hi);
    let local = col - col_lo;
    let row = row_idx[p as usize];
    let w_slot = ((local / 32u32) * n_rows + row) as usize;
    let below = Atomic::load(&mask[w_slot]) & ((1u32 << (local % 32u32)) - 1u32);
    let slot = base[w_slot] + u32::count_ones(below);
    out_indices[slot as usize] = col;
    out_values[slot as usize] = values[p as usize];
}

/// CSR row pointers of a CSC matrix, from its row indices.
///
/// Per-thread row histograms over contiguous slices of `row_idx`, summed and
/// scanned on the host: `O(nnz)` reads and an `n`-long scan.
///
/// ### Params
///
/// * `row_idx` - CSC row indices `[nnz]`
/// * `n_rows` - Number of rows
///
/// ### Returns
///
/// CSR row pointers `[n_rows + 1]`.
fn csr_indptr_from_csc_rows(row_idx: &[u32], n_rows: usize) -> Vec<u32> {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = row_idx.len().div_ceil(n_threads).max(1);
    let counts = row_idx
        .par_chunks(chunk)
        .map(|slice| {
            let mut c = vec![0u32; n_rows];
            for &r in slice {
                c[r as usize] += 1;
            }
            c
        })
        .reduce(
            || vec![0u32; n_rows],
            |mut a, b| {
                a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += y);
                a
            },
        );
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut acc = 0u32;
    indptr.push(0);
    for c in counts {
        acc += c;
        indptr.push(acc);
    }
    indptr
}

/// Build the CSR of a matrix on the device from its uploaded CSC.
///
/// Saves the host transpose and the second upload. The row pointers come from
/// the host (cheap, see [`csr_indptr_from_csc_rows`]). Indices and values are
/// placed by per-row column bitmasks, one word per 32 columns: mark, rank the
/// rows, scatter. Four launches per column group whatever the column count,
/// and the result matches `transpose_sparse_single_layer` exactly, order
/// within each row included.
///
/// ### Params
///
/// * `csc` - Uploaded CSC of A, shape `(n, m)`
/// * `host_col_indptr` - The same CSC's column pointers on the host `[m + 1]`,
///   used to bound each column group
/// * `host_row_idx` - The same CSC's row indices on the host `[nnz]`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The CSR of A on the device.
///
/// ### Errors
///
/// * `SparseLayoutMismatch` if `csc` is not CSC.
/// * `CubeclUtils` if a buffer busts the per-binding size limit or a grid is
///   over the device limit.
pub fn csc_to_csr_gpu<R, F>(
    csc: &GpuCompressedSparseData<R, F>,
    host_col_indptr: &[u32],
    host_row_idx: &[u32],
    client: &ComputeClient<R>,
) -> Result<GpuCompressedSparseData<R, F>, BixverseErrors>
where
    R: Runtime,
    F: cubecl::CubeElement + Numeric,
{
    if !csc.cs_type.is_csc() {
        return Err(BixverseErrors::SparseLayoutMismatch {
            expected: CompressedSparseFormat::Csc,
            got: csc.cs_type,
        });
    }
    csc_to_csr_gpu_grouped(
        csc,
        host_col_indptr,
        host_row_idx,
        CSR_BUILD_BUDGET_BYTES,
        client,
    )
}

/// [`csc_to_csr_gpu`] with an explicit memory budget per column group.
///
/// ### Params
///
/// * `csc` - Uploaded CSC of A, shape `(n, m)`
/// * `host_col_indptr` - The same CSC's column pointers on the host `[m + 1]`
/// * `host_row_idx` - The same CSC's row indices on the host `[nnz]`
/// * `budget_bytes` - Device memory for the masks and slot bases of one
///   column group; at least one 32-column word is always processed
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The CSR of A on the device.
fn csc_to_csr_gpu_grouped<R, F>(
    csc: &GpuCompressedSparseData<R, F>,
    host_col_indptr: &[u32],
    host_row_idx: &[u32],
    budget_bytes: usize,
    client: &ComputeClient<R>,
) -> Result<GpuCompressedSparseData<R, F>, BixverseErrors>
where
    R: Runtime,
    F: cubecl::CubeElement + Numeric,
{
    let (n, m) = csc.shape;
    let limits = GpuLimits::from_client(client);

    let row_ptr = csr_indptr_from_csc_rows(host_row_idx, n);
    let indptr = GpuTensor::<R, u32>::from_slice(&row_ptr, vec![n + 1], client)?;
    let cursor = GpuTensor::<R, u32>::from_slice(&row_ptr[..n], vec![n], client)?;
    let indices = GpuTensor::<R, u32>::empty(vec![csc.nnz], client)?;
    let values = GpuTensor::<R, F>::empty(vec![csc.nnz], client)?;

    let total_words = m.div_ceil(MASK_BITS);
    let group_words = (budget_bytes / (8 * n.max(1))).clamp(1, total_words.max(1));
    let mask = GpuTensor::<R, u32>::empty(vec![group_words * n.max(1)], client)?;
    let base = GpuTensor::<R, u32>::empty(vec![group_words * n.max(1)], client)?;
    let wg = CubeDim::new_1d(WORKGROUP_256);

    for first_word in (0..total_words).step_by(group_words) {
        let col_lo = first_word * MASK_BITS;
        let col_hi = ((first_word + group_words) * MASK_BITS).min(m);
        let words = (col_hi - col_lo).div_ceil(MASK_BITS);
        let nnz_lo = host_col_indptr[col_lo];
        let nnz_len = host_col_indptr[col_hi] - nnz_lo;
        if nnz_len == 0 {
            continue;
        }
        let mask_len = (words * n) as u32;

        let (zx, zy) = grid_2d(mask_len.div_ceil(WORKGROUP_256), &limits)?;
        let zero_count = checked_cube_count("csr_zero_masks", zx, zy, 1, &limits)?;
        let (gx, gy) = grid_2d(nnz_len.div_ceil(WORKGROUP_256), &limits)?;
        let nnz_count = checked_cube_count("csr_scatter_group", gx, gy, 1, &limits)?;
        let (rx, ry) = grid_2d((n as u32).div_ceil(WORKGROUP_256), &limits)?;
        let row_count = checked_cube_count("csr_rank_rows", rx, ry, 1, &limits)?;

        unsafe {
            csr_zero_masks::launch_unchecked::<R>(
                client,
                zero_count,
                wg,
                mask.clone().into_tensor_arg(),
                mask_len,
            );
            csr_mark_columns::launch_unchecked::<R>(
                client,
                nnz_count.clone(),
                wg,
                csc.indptr.clone().into_tensor_arg(),
                csc.indices.clone().into_tensor_arg(),
                mask.clone().into_tensor_arg(),
                col_lo as u32,
                col_hi as u32,
                nnz_lo,
                nnz_len,
                n as u32,
            );
            csr_rank_rows::launch_unchecked::<R>(
                client,
                row_count,
                wg,
                mask.clone().into_tensor_arg(),
                base.clone().into_tensor_arg(),
                cursor.clone().into_tensor_arg(),
                n as u32,
                words as u32,
            );
            csr_scatter_group::launch_unchecked::<F, R>(
                client,
                nnz_count,
                wg,
                csc.indptr.clone().into_tensor_arg(),
                csc.indices.clone().into_tensor_arg(),
                csc.values.clone().into_tensor_arg(),
                mask.clone().into_tensor_arg(),
                base.clone().into_tensor_arg(),
                indices.clone().into_tensor_arg(),
                values.clone().into_tensor_arg(),
                col_lo as u32,
                col_hi as u32,
                nnz_lo,
                nnz_len,
                n as u32,
            );
        }
    }

    Ok(GpuCompressedSparseData {
        indptr,
        indices,
        values,
        cs_type: CompressedSparseFormat::Csr,
        shape: (n, m),
        nnz: csc.nnz,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::sparse::transpose_sparse_single_layer;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Build a CSC with empty rows and an empty column, run the device build
    /// under `budget_bytes`, and compare it with the host transpose.
    fn check_csc_to_csr(n: usize, m: usize, budget_bytes: usize) {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);

        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for j in 0..m {
            for i in 0..n {
                // column 7 and every row divisible by 50 stay empty
                if j != 7 && i % 50 != 0 && (i * 3 + j * 7) % 5 == 0 {
                    values.push((i * 31 + j) as f32 * 0.01);
                    indices.push(i as u32);
                }
            }
            indptr.push(values.len() as u32);
        }
        let csc = CompressedSparseData2::<f32, f32>::new_csc(
            &values,
            &indices,
            &indptr,
            Some(&values),
            (n, m),
        );
        let want = transpose_sparse_single_layer(&csc, true).unwrap();

        let csc_gpu = GpuCompressedSparseData::<WgpuRuntime, f32>::from_parts(
            &values,
            &indices,
            &indptr,
            CompressedSparseFormat::Csc,
            (n, m),
            &client,
        )
        .unwrap();
        let got =
            csc_to_csr_gpu_grouped(&csc_gpu, &indptr, &indices, budget_bytes, &client).unwrap();

        assert!(got.cs_type.is_csr());
        assert_eq!(got.indptr.read(&client).unwrap(), want.indptr);
        assert_eq!(got.indices.read(&client).unwrap(), want.indices);
        assert_eq!(
            got.values.read(&client).unwrap(),
            *want.data_2.as_ref().unwrap()
        );
    }

    /// The device build is bitwise identical to the host transpose, including
    /// the order within each row, with empty rows and empty columns present.
    #[test]
    fn test_csc_to_csr_gpu_matches_host() {
        check_csc_to_csr(300, 40, CSR_BUILD_BUDGET_BYTES);
    }

    /// A budget of one mask word per group splits 100 columns into four
    /// groups, the last one partial, and the cursors carry across them.
    #[test]
    fn test_csc_to_csr_gpu_column_groups() {
        let n = 300;
        check_csc_to_csr(n, 100, 8 * n);
    }
}
