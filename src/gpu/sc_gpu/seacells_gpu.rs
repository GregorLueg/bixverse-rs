//! GPU-accelerated SEACells fit.
//!
//! The per-archetype Frank-Wolfe gradient argmin dominates the runtime and
//! every other step of the B update is a few percent, so that argmin is the
//! only thing this module moves to the device. [GpuFwArgminB] implements the
//! [FwArgminB] seam with `fw_argmin_b`, and [seacells_fit_gpu] drives the same
//! [SEACells::fit_with] loop the CPU path uses.
//!
//! Nothing here holds an `n × k` dense buffer. `K²B`, `K²Aᵀ` and `B` stay
//! sparse; only `t1 = A Aᵀ` is dense, at `4k²` bytes.

use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::prelude::*;

use crate::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use crate::gpu::sc_gpu::kernels::seacells_kernels::{B_ARGMIN_BLOCKS, launch_fw_argmin_b};
use crate::prelude::*;
use crate::single_cell::mc_generation::seacells::{
    FwArgminB, SEACells, SEACellsParams, assignments_to_metacells,
};

///////////
// Types //
///////////

/// Output of a SEACells fit: hard assignment per cell, the cells grouped by
/// SEACell, and the RSS at each iteration.
pub type SeacellsFitResult = (Vec<usize>, Vec<Vec<usize>>, Vec<f32>);

/////////////////
// Back end    //
/////////////////

/// Device-resident B-argmin back end.
///
/// Scratch is allocated once in [GpuFwArgminB::new] and reused across every
/// Frank-Wolfe iteration and every outer iteration: `client.empty()` returns
/// quickly but the device pages are not backed until something writes them, so
/// allocating per call would pay that fault repeatedly.
///
/// `t1` and `K²Aᵀ` are uploaded once per B update via [FwArgminB::begin]; `K²B`
/// and `B` change every Frank-Wolfe iteration and are re-uploaded there. Their
/// non-zero counts drift between iterations, so those two are the only buffers
/// not pooled.
pub struct GpuFwArgminB<R: Runtime> {
    /// Compute client
    client: ComputeClient<R>,
    /// Number of cells
    n: usize,
    /// Number of archetypes
    k: usize,
    /// Workgroups the kernel dispatches, and therefore the partial-buffer depth
    blocks: usize,
    /// `A Aᵀ` dense `[k, k]`, rebound per B update
    t1: Option<GpuTensor<R, f32>>,
    /// `K² Aᵀ` as CSR `n × k`, rebound per B update
    t2: Option<GpuCompressedSparseData<R, f32>>,
    /// Per-block partial minima `[blocks, k]`
    part_val: GpuTensor<R, f32>,
    /// Per-block partial argmins `[blocks, k]`
    part_idx: GpuTensor<R, u32>,
    /// Per-block partial `sum(B * G)` `[blocks]`
    gap_partial: GpuTensor<R, f32>,
    /// Reduced minima `[k]`
    out_val: GpuTensor<R, f32>,
    /// Reduced argmins `[k]`
    out_idx: GpuTensor<R, u32>,
}

impl<R: Runtime> GpuFwArgminB<R> {
    /// Allocate the device scratch for an `n × k` problem.
    ///
    /// ### Params
    ///
    /// * `n` - Number of cells
    /// * `k` - Number of archetypes
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// Self, with all pooled buffers allocated.
    pub fn new(n: usize, k: usize, client: ComputeClient<R>) -> Self {
        let blocks = B_ARGMIN_BLOCKS.min(n.max(1) as u32) as usize;

        let part_val = GpuTensor::<R, f32>::empty(vec![blocks * k], &client);
        let part_idx = GpuTensor::<R, u32>::empty(vec![blocks * k], &client);
        let gap_partial = GpuTensor::<R, f32>::empty(vec![blocks], &client);
        let out_val = GpuTensor::<R, f32>::empty(vec![k], &client);
        let out_idx = GpuTensor::<R, u32>::empty(vec![k], &client);

        Self {
            client,
            n,
            k,
            blocks,
            t1: None,
            t2: None,
            part_val,
            part_idx,
            gap_partial,
            out_val,
            out_idx,
        }
    }

    /// VRAM held by the pooled scratch.
    ///
    /// ### Returns
    ///
    /// Total bytes across the partial and output buffers.
    pub fn vram_bytes(&self) -> usize {
        self.part_val.vram_bytes()
            + self.part_idx.vram_bytes()
            + self.gap_partial.vram_bytes()
            + self.out_val.vram_bytes()
            + self.out_idx.vram_bytes()
    }
}

impl<R: Runtime> FwArgminB for GpuFwArgminB<R> {
    fn begin(
        &mut self,
        t1: &CompressedSparseData2<f32>,
        t2: &CompressedSparseData2<f32>,
    ) -> Result<(), BixverseErrors> {
        if t1.shape != (self.k, self.k) {
            return Err(BixverseErrors::ShapeMismatch {
                expected: (self.k, self.k),
                got: t1.shape,
            });
        }
        if t2.shape != (self.n, self.k) {
            return Err(BixverseErrors::ShapeMismatch {
                expected: (self.n, self.k),
                got: t2.shape,
            });
        }

        // t1 runs close to dense in practice, so densifying costs little and
        // removes an indirection from the innermost loop.
        let mut dense = vec![0.0f32; self.k * self.k];
        for row in 0..self.k {
            for idx in t1.indptr[row] as usize..t1.indptr[row + 1] as usize {
                dense[row * self.k + t1.indices[idx] as usize] += t1.data[idx];
            }
        }

        self.t1 = Some(GpuTensor::<R, f32>::from_slice(
            &dense,
            vec![self.k * self.k],
            &self.client,
        ));
        self.t2 = Some(
            GpuCompressedSparseData::<R, f32>::from_compressed_sparse_data_2(
                t2,
                false,
                &self.client,
            )?,
        );

        Ok(())
    }

    fn argmins(
        &mut self,
        k2_b: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
    ) -> Result<(Vec<usize>, f32), BixverseErrors> {
        // The kernel launches unchecked and indexes `indptr` up to `n`, so a
        // wrong shape here is an out-of-bounds device read rather than an
        // error.
        for mat in [k2_b, b] {
            if mat.shape != (self.n, self.k) {
                return Err(BixverseErrors::ShapeMismatch {
                    expected: (self.n, self.k),
                    got: mat.shape,
                });
            }
        }

        let t1 = self
            .t1
            .as_ref()
            .ok_or(BixverseErrors::SEACellsModelNotFitted)?;
        let t2 = self
            .t2
            .as_ref()
            .ok_or(BixverseErrors::SEACellsModelNotFitted)?;

        let k2b_gpu = GpuCompressedSparseData::<R, f32>::from_compressed_sparse_data_2(
            k2_b,
            false,
            &self.client,
        )?;
        let b_gpu = GpuCompressedSparseData::<R, f32>::from_compressed_sparse_data_2(
            b,
            false,
            &self.client,
        )?;

        launch_fw_argmin_b(
            &k2b_gpu,
            t1,
            t2,
            &b_gpu,
            &self.part_val,
            &self.part_idx,
            &self.gap_partial,
            &self.out_val,
            &self.out_idx,
            self.n,
            self.k,
            &self.client,
        )?;

        let min_vals = self
            .out_val
            .clone()
            .read(&self.client)
            .map_err(|e| BixverseErrors::GpuMatmul(e.to_string()))?;
        let argmins = self
            .out_idx
            .clone()
            .read(&self.client)
            .map_err(|e| BixverseErrors::GpuMatmul(e.to_string()))?;
        let gaps = self
            .gap_partial
            .clone()
            .read(&self.client)
            .map_err(|e| BixverseErrors::GpuMatmul(e.to_string()))?;

        // gap = |sum(B * G) - sum_c min_c G[:, c]|, matching the CPU scan.
        let g_dot_b: f32 = gaps.iter().take(self.blocks).sum();
        let g_dot_e: f32 = min_vals.iter().sum();

        Ok((
            argmins.into_iter().map(|i| i as usize).collect(),
            (g_dot_b - g_dot_e).abs(),
        ))
    }
}

//////////
// Main //
//////////

/// Run the SEACells algorithm with the Frank-Wolfe B-gradient argmin on the GPU.
///
/// Kernel construction, archetype initialisation, the A update and the RSS all
/// run on the CPU exactly as in the pure-Rust path; only the gradient argmin
/// that dominates the runtime moves to the device.
///
/// ### Params
///
/// * `pca` - PCA/SVD embedding (n_cells × n_components)
/// * `knn_indices` - kNN indices for each cell
/// * `knn_distances` - kNN distances for each cell
/// * `squared_dist` - Are the distances squared (squared Euclidean for example)
/// * `params` - Algorithm parameters
/// * `seed` - Random seed for reproducibility
/// * `device` - Device to run the argmin on
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `(hard assignments per cell, metacell groupings, RSS history)`. The groupings
/// are one entry per archetype the initialisation actually selected, which is
/// `params.n_sea_cells` unless deduplication came back short.
///
/// ### References
///
/// Persad, et al., Nat. Biotechnol., 2023
#[allow(clippy::too_many_arguments)]
pub fn seacells_fit_gpu<R: Runtime>(
    pca: faer::MatRef<f32>,
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    squared_dist: bool,
    params: &SEACellsParams,
    seed: usize,
    device: R::Device,
    verbose: usize,
) -> Result<SeacellsFitResult, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let client = R::client(&device);

    let n = pca.nrows();

    let mut model = SEACells::new(n, params);
    model.construct_kernel_mat(pca, knn_indices, knn_distances, verbose);

    match params.n_landmarks {
        Some(n_landmarks) => model.initialise_archetypes_landmark(
            pca,
            knn_indices,
            knn_distances,
            squared_dist,
            n_landmarks,
            verbose,
            seed as u64,
        )?,
        None => model.initialise_archetypes(
            knn_indices,
            knn_distances,
            verbose,
            squared_dist,
            seed as u64,
        )?,
    }

    // Archetype initialisation dedups and can come back with fewer than
    // `n_sea_cells`, and the model sizes A and B from what it actually got. The
    // device scratch has to follow that rather than the requested count.
    let k = model.get_archetypes()?.len();

    let mut backend = GpuFwArgminB::<R>::new(n, k, client);

    if verbosity.detailed_verbosity() {
        println!(
            "GPU B-argmin scratch: {:.1} MB",
            backend.vram_bytes() as f64 / (1024.0 * 1024.0)
        );
    }

    model.fit_with(seed, verbose, &mut backend)?;

    let assignments = model.get_hard_assignments()?;
    let metacells = assignments_to_metacells(&assignments, k);
    let rss_history = model.get_rss_history().to_vec();

    Ok((assignments, metacells, rss_history))
}
