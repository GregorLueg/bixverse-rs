//! GPU-resident compressed sparse matrix. Single-layout, layer-selected at
//! upload time. Hold two of these (one CSR, one CSC) when both directions
//! of SpMM are needed.

use cubecl::Runtime;
use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;

use crate::prelude::*;

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
