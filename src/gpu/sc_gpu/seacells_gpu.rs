//! GPU-accelerated SEACells fit.
//!
//! [GpuFwArgminB] implements the [FwArgminB] seam with `fw_argmin_b` for the B
//! update and `fw_columns_a_gpu` for the A update, and [seacells_fit_gpu] drives
//! the same [SEACells::fit_with] loop the CPU path uses.
//!
//! Neither kernel dominates the fit. Each beats the CPU scan it replaces by
//! several fold in isolation, but the whole fit moves by ~2.6x at 50k cells and
//! 666 archetypes, which is Amdahl rather than anything wrong with them: the
//! blocking readbacks around the kernels were about a fifth of wall-clock when
//! last attributed, and the kernel construction, archetype initialisation, `K²B`
//! maintenance and RSS all stay on the host.
//!
//! Nothing here holds an `n × k` dense buffer. `K²B`, `K²Aᵀ` and `B` stay
//! sparse, and the A update's atom scratch is `[n, cap]` where `cap` is the
//! widest `A` column plus `max_fw_iters`.
//!
//! `t1` stays sparse on both solves, paired with a `[k, wg + 1]` per-thread
//! segment table so a thread reaches its own contiguous column block without
//! searching. Nothing on the path is quadratic in `k`; it used to be densified
//! to `4k²` bytes on both, which is 381 MB at k = 10 000.

use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::prelude::*;

use crate::gpu::linalg::sparse_gpu::GpuCompressedSparseData;
use crate::gpu::sc_gpu::kernels::seacells_kernels::{
    A_COLUMNS_MAX_SLOTS, A_COLUMNS_WG_TIERS, B_ARGMIN_BLOCKS, B_ARGMIN_WG_TIERS,
    a_columns_capacity, a_columns_segments, a_columns_workgroup, b_argmin_workgroup,
    launch_fw_argmin_b, launch_fw_columns_a,
};
use crate::prelude::*;
use crate::single_cell::mc_generation::seacells::{
    CpuFwArgminB, FwArgminB, FwAtoms, SEACells, SEACellsParams, assignments_to_metacells,
    fw_columns_a,
};

///////////
// Types //
///////////

/// Output of a SEACells fit: hard assignment per cell, the cells grouped by
/// SEACell, the cell index of each archetype, and the RSS at each iteration.
pub type SeacellsFitResult = (Vec<usize>, Vec<Vec<usize>>, Vec<usize>, Vec<f32>);

//////////////
// Back end //
//////////////

/// Device-resident back end for both Frank-Wolfe solves.
///
/// Scratch is allocated once in [GpuFwArgminB::new] and reused across every
/// Frank-Wolfe iteration and every outer iteration: `client.empty()` returns
/// quickly but the device pages are not backed until something writes them, so
/// allocating per call would pay that fault repeatedly.
///
/// `t1`, its segment table and `K²Aᵀ` are uploaded once per B update via
/// [FwArgminB::begin]; `K²B` and `B` change every Frank-Wolfe iteration and are
/// re-uploaded there. Their non-zero counts drift between iterations, so those
/// two are the only buffers not pooled.
///
/// Either solve independently falls back to the CPU when no workgroup tier
/// covers `k`, so a shape can run one on the device and the other on the host.
pub struct GpuFwArgminB<R: Runtime> {
    /// Compute client
    client: ComputeClient<R>,
    /// Number of cells
    n: usize,
    /// Number of archetypes
    k: usize,
    /// Workgroups the kernel dispatches, and therefore the partial-buffer depth
    blocks: usize,
    /// `A Aᵀ` as CSR `k × k`, rebound per B update
    t1: Option<GpuCompressedSparseData<R, f32>>,
    /// Per-thread column segments of `t1`, `[k, b_wg + 1]`, rebound with it
    t1_seg: Option<GpuTensor<R, u32>>,
    /// Workgroup width the B argmin was last bound for
    b_wg: u32,
    /// CPU back end for the B argmin, used when no workgroup tier covers `k`
    b_cpu: Option<CpuFwArgminB>,
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
    /// Atom capacity the A-update scratch below is sized for, `0` when unset
    a_cap: usize,
    /// Cell count the A-update scratch is sized for, `0` when unset
    a_n: usize,
    /// A-update atom indices `[n, a_cap]`
    a_atom_idx: Option<GpuTensor<R, u32>>,
    /// A-update atom weights `[n, a_cap]`
    a_atom_val: Option<GpuTensor<R, f32>>,
    /// A-update atom counts `[n]`
    a_atom_cnt: Option<GpuTensor<R, u32>>,
    /// Verbosity, so a CPU fallback can announce itself
    verbose: usize,
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
        Self::with_verbosity(n, k, client, 0)
    }

    /// As [GpuFwArgminB::new], but reporting CPU fallbacks at detailed
    /// verbosity.
    ///
    /// ### Params
    ///
    /// * `n` - Number of cells
    /// * `k` - Number of archetypes
    /// * `client` - CubeCL compute client
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// Self, with all pooled buffers allocated.
    pub fn with_verbosity(n: usize, k: usize, client: ComputeClient<R>, verbose: usize) -> Self {
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
            t1_seg: None,
            b_wg: 0,
            b_cpu: None,
            t2: None,
            part_val,
            part_idx,
            gap_partial,
            out_val,
            out_idx,
            a_cap: 0,
            a_n: 0,
            a_atom_idx: None,
            a_atom_val: None,
            a_atom_cnt: None,
            verbose,
        }
    }

    /// VRAM held by the pooled scratch.
    ///
    /// ### Returns
    ///
    /// Total bytes across every pooled buffer, including the A-update scratch,
    /// which is allocated lazily and dominates once it exists.
    pub fn vram_bytes(&self) -> usize {
        let a_update: usize = [
            self.a_atom_idx.as_ref().map(|t| t.vram_bytes()),
            self.a_atom_val.as_ref().map(|t| t.vram_bytes()),
            self.a_atom_cnt.as_ref().map(|t| t.vram_bytes()),
        ]
        .into_iter()
        .flatten()
        .sum();

        self.part_val.vram_bytes()
            + self.part_idx.vram_bytes()
            + self.gap_partial.vram_bytes()
            + self.out_val.vram_bytes()
            + self.out_idx.vram_bytes()
            + a_update
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

        // `t1` stays sparse, paired with a per-thread segment table so each
        // thread reaches its own contiguous column block in one row without
        // searching. It used to be densified to `k²` floats, which is 381 MB at
        // k = 10 000 and grows quadratically; the inner loop then read a full
        // `k`-float row for every non-zero of the `K²B` row.
        // No tier means the register arrays would not fit and the dispatch
        // would be rejected silently, so this solve runs on the CPU instead.
        // The A solve still runs on the device; they are independent.
        let Some(wg) = b_argmin_workgroup(self.k) else {
            if parse_verbosity_level(self.verbose).detailed_verbosity() {
                println!(
                    "  B-gradient argmin on CPU: k {} fits no tier {:?}",
                    self.k, B_ARGMIN_WG_TIERS
                );
            }
            let mut cpu = CpuFwArgminB::default();
            cpu.begin(t1, t2)?;
            self.b_cpu = Some(cpu);
            return Ok(());
        };
        self.b_cpu = None;
        let seg_host = a_columns_segments(t1, wg, self.k.div_ceil(wg as usize));
        self.t1_seg = Some(GpuTensor::<R, u32>::from_slice(
            &seg_host,
            vec![seg_host.len()],
            &self.client,
        ));
        drop(seg_host);
        self.b_wg = wg;

        self.t1 = Some(
            GpuCompressedSparseData::<R, f32>::from_compressed_sparse_data_2(
                t1,
                false,
                &self.client,
            )?,
        );
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
        if let Some(cpu) = self.b_cpu.as_mut() {
            return cpu.argmins(k2_b, b);
        }

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
        let t1_seg = self
            .t1_seg
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
            t1_seg,
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

    fn columns_a(
        &mut self,
        t1: &CompressedSparseData2<f32>,
        a_prev_t: &CompressedSparseData2<f32>,
        k2_b: &CompressedSparseData2<f32>,
        k: usize,
        n: usize,
        n_iters: usize,
        pruning: Option<f32>,
    ) -> Result<Vec<FwAtoms>, BixverseErrors> {
        if t1.shape != (k, k) {
            return Err(BixverseErrors::ShapeMismatch {
                expected: (k, k),
                got: t1.shape,
            });
        }
        for mat in [a_prev_t, k2_b] {
            if mat.shape != (n, k) {
                return Err(BixverseErrors::ShapeMismatch {
                    expected: (n, k),
                    got: mat.shape,
                });
            }
        }

        let cap = a_columns_capacity(a_prev_t, n_iters);
        let Some(wg) = a_columns_workgroup(k, cap) else {
            if parse_verbosity_level(self.verbose).detailed_verbosity() {
                println!(
                    "  A-column solve on CPU: k {} at cap {} fits no tier {:?} within {} slots",
                    k, cap, A_COLUMNS_WG_TIERS, A_COLUMNS_MAX_SLOTS
                );
            }
            return Ok(fw_columns_a(t1, a_prev_t, k2_b, k, n, n_iters, pruning));
        };

        let t1_gpu = GpuCompressedSparseData::<R, f32>::from_compressed_sparse_data_2(
            t1,
            false,
            &self.client,
        )?;
        let seg_host = a_columns_segments(t1, wg, k.div_ceil(wg as usize));
        let t1_seg = GpuTensor::<R, u32>::from_slice(&seg_host, vec![seg_host.len()], &self.client);
        drop(seg_host);

        let a_prev_gpu = GpuCompressedSparseData::<R, f32>::from_compressed_sparse_data_2(
            a_prev_t,
            false,
            &self.client,
        )?;
        let k2b_gpu = GpuCompressedSparseData::<R, f32>::from_compressed_sparse_data_2(
            k2_b,
            false,
            &self.client,
        )?;

        let required = n * cap;
        if self.a_atom_idx.is_none() || self.a_n * self.a_cap < required || self.a_n < n {
            self.a_cap = cap;
            self.a_n = n;
            self.a_atom_idx = Some(GpuTensor::<R, u32>::empty(vec![required], &self.client));
            self.a_atom_val = Some(GpuTensor::<R, f32>::empty(vec![required], &self.client));
            self.a_atom_cnt = Some(GpuTensor::<R, u32>::empty(vec![n], &self.client));
        }
        let stride = self.a_cap;
        let atom_idx = self.a_atom_idx.as_ref().expect("just allocated").clone();
        let atom_val = self.a_atom_val.as_ref().expect("just allocated").clone();
        let atom_cnt = self.a_atom_cnt.as_ref().expect("just allocated").clone();

        let threshold =
            GpuTensor::<R, f32>::from_slice(&[pruning.unwrap_or(0.0)], vec![1], &self.client);

        launch_fw_columns_a(
            &t1_gpu,
            &t1_seg,
            &a_prev_gpu,
            &k2b_gpu,
            &atom_idx,
            &atom_val,
            &atom_cnt,
            &threshold,
            n,
            k,
            n_iters,
            stride as u32,
            pruning,
            None,
            Some(wg),
            &self.client,
        )?;

        let idx_host = atom_idx
            .clone()
            .read(&self.client)
            .map_err(|e| BixverseErrors::GpuMatmul(e.to_string()))?;
        let val_host = atom_val
            .clone()
            .read(&self.client)
            .map_err(|e| BixverseErrors::GpuMatmul(e.to_string()))?;
        let cnt_host = atom_cnt
            .clone()
            .read(&self.client)
            .map_err(|e| BixverseErrors::GpuMatmul(e.to_string()))?;

        if n_iters > 0 && cnt_host.contains(&0) {
            let dead = cnt_host.iter().filter(|c| **c == 0).count();
            println!(
                "  A-column solve fell back to CPU: the kernel wrote no atoms for {} of {} cells \
                 at k = {}, workgroup {}, {} slots. This is a rejected dispatch, not a result. \
                 Please report the device and shape.",
                dead,
                n,
                k,
                wg,
                k.div_ceil(wg as usize)
            );
            return Ok(fw_columns_a(t1, a_prev_t, k2_b, k, n, n_iters, pruning));
        }

        let drop_zeros = pruning.is_some();
        let mut columns = Vec::with_capacity(n);
        let mut keep_idx: Vec<u32> = Vec::with_capacity(stride);
        let mut keep_val: Vec<f32> = Vec::with_capacity(stride);
        for cell in 0..n {
            let count = cnt_host[cell] as usize;
            keep_idx.clear();
            keep_val.clear();
            for slot in 0..count {
                let weight = val_host[cell * stride + slot];
                if !drop_zeros || weight != 0.0 {
                    keep_idx.push(idx_host[cell * stride + slot]);
                    keep_val.push(weight);
                }
            }
            let mut atoms = FwAtoms::with_capacity(keep_idx.len());
            atoms.reset_from(&keep_idx, &keep_val);
            columns.push(atoms);
        }

        Ok(columns)
    }
}

//////////
// Main //
//////////

/// Run the SEACells algorithm with the Frank-Wolfe B-gradient argmin on the GPU.
///
/// Kernel construction, archetype initialisation and the RSS run on the CPU
/// exactly as in the pure-Rust path. Both Frank-Wolfe inner solves move to the
/// device: the B-gradient argmin via [FwArgminB::argmins] and the A-column
/// solve via [FwArgminB::columns_a].
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
/// `(hard assignments per cell, metacell groupings, archetype cell indices, RSS
/// history)`. The groupings and the archetypes are one entry per archetype the
/// initialisation actually selected, which is `params.n_sea_cells` unless
/// deduplication came back short.
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
    let archetypes = model.get_archetypes()?;
    let k = archetypes.len();

    let mut backend = GpuFwArgminB::<R>::with_verbosity(n, k, client, verbose);

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

    Ok((assignments, metacells, archetypes, rss_history))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::sc_gpu::kernels::seacells_kernels::A_COLUMNS_BLOCKS;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use rand::prelude::*;
    use rand::rngs::StdRng;

    use crate::gpu::sc_gpu::kernels::seacells_kernels::A_COLUMNS_WG;

    /// Skip rather than fail where no GPU is available.
    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Build a CSR with `per_row` positive entries per row, columns distinct.
    fn random_csr(
        n_rows: usize,
        n_cols: usize,
        per_row: usize,
        normalise: bool,
        coarse: bool,
        rng: &mut StdRng,
    ) -> CompressedSparseData2<f32> {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];

        for _ in 0..n_rows {
            let mut cols: Vec<u32> = (0..n_cols as u32).collect();
            cols.shuffle(rng);
            cols.truncate(per_row.min(n_cols));
            cols.sort_unstable();

            let mut vals: Vec<f32> = (0..cols.len())
                .map(|_| {
                    if coarse {
                        // Few distinct magnitudes, so exact ties in the gradient
                        // are frequent. Continuous values almost never tie, and
                        // a tie-break that reduces over the wrong axis passes
                        // cleanly against them.
                        (rng.random_range(1..4) as f32) * 0.25
                    } else {
                        rng.random::<f32>() + 0.05
                    }
                })
                .collect();
            if normalise {
                let sum: f32 = vals.iter().sum();
                for v in vals.iter_mut() {
                    *v /= sum;
                }
            }
            indices.extend_from_slice(&cols);
            data.extend_from_slice(&vals);
            indptr.push(indices.len() as u32);
        }

        CompressedSparseData2 {
            data,
            indices,
            indptr,
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (n_rows, n_cols),
        }
    }

    /// The A-column kernel must reproduce `fw_columns_a` atom for atom.
    ///
    /// This is the gate that makes the kernel believable. `launch_unchecked`
    /// fails silently when it busts a device limit, so the first assertions are
    /// that every column carries mass at all: a kernel that did nothing returns
    /// empty columns, which would otherwise read as a very fast pass.
    ///
    /// Both pruning settings are swept, since the pruning path is where the
    /// kernel diverges most from the CPU structure: it marks weights zero
    /// instead of compacting the atom list.
    #[test]
    fn test_fw_columns_a_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("no GPU available, skipping");
            return;
        };
        let client = WgpuRuntime::client(&device);

        // Both axes have to be crossed at once.
        //
        // `k` must exceed the workgroup width so each thread owns more than one
        // column: at `slots == 1` the lane index and the column index coincide
        // and a reduction that mixes the two axes up still gives right answers.
        //
        // `n` must exceed `A_COLUMNS_BLOCKS` so a workgroup grid-strides over
        // several cells. Below it every workgroup runs the loop body once and
        // the per-cell reset, the trailing barrier and any carry-over of shared
        // state between cells are all dead code.
        let n = 2_500usize;
        let k = 300usize;
        let n_iters = 15usize;
        assert!(n > A_COLUMNS_BLOCKS as usize, "grid stride not exercised");
        assert!(k > A_COLUMNS_WG as usize, "slots == 1, axes not separated");

        // `coarse` collapses the value range so the gradient ties constantly,
        // which is what real data does once the first Frank-Wolfe step zeroes
        // the seed and leaves two sparse vectors behind.
        for coarse in [false, true] {
            let mut rng = StdRng::seed_from_u64(11);
            let t1 = random_csr(k, k, 12, false, coarse, &mut rng);
            let a_prev_t = random_csr(n, k, 6, true, coarse, &mut rng);
            let k2_b = random_csr(n, k, 8, false, coarse, &mut rng);

            // Deliberately one back end for all three settings, and the
            // thresholds ascend. A back end that uploaded the threshold once and
            // pooled it would pass every call after the first with the wrong
            // value, and a fresh back end per iteration would hide that.
            let mut backend = GpuFwArgminB::<WgpuRuntime>::new(n, k, client.clone());

            for pruning in [None, Some(1e-7f32), Some(5e-2f32)] {
                let expected = fw_columns_a(&t1, &a_prev_t, &k2_b, k, n, n_iters, pruning);

                let actual = backend
                    .columns_a(&t1, &a_prev_t, &k2_b, k, n, n_iters, pruning)
                    .expect("GPU column solve failed");

                assert_eq!(actual.len(), n);

                // Silent-failure guard before any agreement claim.
                let total_mass: f64 = actual
                    .iter()
                    .map(|atoms| atoms.atoms().1.iter().map(|&w| w as f64).sum::<f64>())
                    .sum();
                assert!(
                    (total_mass - n as f64).abs() < 0.01 * n as f64,
                    "columns do not sum to one: total mass {} over {} cells (pruning {:?}, \
                 coarse {:?}), the kernel probably did no work",
                    total_mass,
                    n,
                    pruning,
                    coarse
                );

                // Compared as a set, not positionally. Pruning removes an atom on
                // the CPU so a later re-selection of that index pushes a fresh atom
                // at the end, whereas the kernel finds the index in its zeroed slot
                // and reuses it in place. The weights are identical either way and
                // `fw_atoms_to_csr` goes through `coo_to_csr`, which sorts, so the
                // position an atom occupies cannot reach the output.
                let sorted = |atoms: &FwAtoms| {
                    let (idx, val) = atoms.atoms();
                    let mut pairs: Vec<(u32, f32)> =
                        idx.iter().copied().zip(val.iter().copied()).collect();
                    pairs.sort_unstable_by_key(|&(i, _)| i);
                    pairs
                };

                let mut differing = 0usize;
                for (cell, (want, got)) in expected.iter().zip(actual.iter()).enumerate() {
                    let want_pairs = sorted(want);
                    let got_pairs = sorted(got);
                    let (want_idx, want_val): (Vec<u32>, Vec<f32>) =
                        want_pairs.iter().copied().unzip();
                    let (got_idx, got_val): (Vec<u32>, Vec<f32>) =
                        got_pairs.iter().copied().unzip();

                    let same = want_idx == got_idx
                        && want_val
                            .iter()
                            .zip(got_val.iter())
                            .all(|(a, b)| (a - b).abs() <= 1e-4 * a.abs().max(1e-3));
                    if !same {
                        if differing == 0 {
                            eprintln!(
                                "first mismatch at cell {} (pruning {:?}):\n  cpu {:?}\n  gpu {:?}",
                                cell, pruning, want_pairs, got_pairs
                            );
                        }
                        differing += 1;
                    }
                }

                // Near-ties in the argmin can resolve differently between a scalar
                // scan and a tree reduction, and one flip changes the rest of that
                // column's trajectory. Anything past a couple of percent is a real
                // disagreement rather than a tie-break.
                assert!(
                    differing * 50 <= n,
                    "{} / {} columns differ (pruning {:?}, coarse {:?})",
                    differing,
                    n,
                    pruning,
                    coarse
                );
            }
        }
    }
    /// Parity above the old `k <= 2048` ceiling, across every workgroup tier.
    ///
    /// Each `k` here selects a different tier and a different slot count, and
    /// none is a multiple of `wg * slots`, so the partial block at the end of
    /// the column range is exercised on every one. That tail is where blocked
    /// ownership can go wrong without the totals noticing: a thread whose block
    /// runs past `k` must contribute no argmin candidate.
    #[test]
    fn test_fw_columns_a_large_k_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("no GPU available, skipping");
            return;
        };
        let client = WgpuRuntime::client(&device);

        let n = 1_500usize;
        let n_iters = 12usize;

        // 3000 -> wg 128, 4500 -> wg 256, 9000 -> wg 512. All above the ceiling
        // the kernel used to fall back at, and none a power of two.
        for k in [3_000usize, 4_500, 9_000] {
            let wg = a_columns_workgroup(k, n_iters + 6).expect("no tier for k");
            let slots = k.div_ceil(wg as usize);
            assert!(
                wg as usize * slots > k,
                "k = {} is an exact multiple of wg * slots, tail not exercised",
                k
            );

            let mut rng = StdRng::seed_from_u64(k as u64);
            let t1 = random_csr(k, k, 20, false, false, &mut rng);
            let a_prev_t = random_csr(n, k, 6, true, false, &mut rng);
            let k2_b = random_csr(n, k, 10, false, false, &mut rng);

            let mut backend = GpuFwArgminB::<WgpuRuntime>::new(n, k, client.clone());

            for pruning in [None, Some(1e-7f32)] {
                let expected = fw_columns_a(&t1, &a_prev_t, &k2_b, k, n, n_iters, pruning);
                let actual = backend
                    .columns_a(&t1, &a_prev_t, &k2_b, k, n, n_iters, pruning)
                    .expect("GPU column solve failed");

                let total_mass: f64 = actual
                    .iter()
                    .map(|atoms| atoms.atoms().1.iter().map(|&w| w as f64).sum::<f64>())
                    .sum();
                assert!(
                    (total_mass - n as f64).abs() < 0.01 * n as f64,
                    "k = {} (wg {}, {} slots): total mass {} over {} cells, the kernel probably \
                     did no work",
                    k,
                    wg,
                    slots,
                    total_mass,
                    n
                );

                let mut differing = 0usize;
                for (want, got) in expected.iter().zip(actual.iter()) {
                    let key = |atoms: &FwAtoms| {
                        let (idx, val) = atoms.atoms();
                        let mut pairs: Vec<(u32, f32)> =
                            idx.iter().copied().zip(val.iter().copied()).collect();
                        pairs.sort_unstable_by_key(|&(i, _)| i);
                        pairs
                    };
                    let (want_pairs, got_pairs) = (key(want), key(got));
                    let same = want_pairs.len() == got_pairs.len()
                        && want_pairs.iter().zip(got_pairs.iter()).all(|(a, b)| {
                            a.0 == b.0 && (a.1 - b.1).abs() <= 1e-4 * a.1.abs().max(1e-3)
                        });
                    if !same {
                        differing += 1;
                    }
                }
                assert!(
                    differing * 50 <= n,
                    "k = {} (wg {}, {} slots): {} / {} columns differ (pruning {:?})",
                    k,
                    wg,
                    slots,
                    differing,
                    n,
                    pruning
                );
            }
        }
    }

    /// The segment table must bracket exactly the entries a thread owns.
    ///
    /// Checked against the definition rather than against the kernel, including
    /// the two cases the kernel relies on and a walk would not reveal: a block
    /// holding no entries at all, and a row whose entries all fall in one block.
    #[test]
    fn test_a_columns_segments_bracket_owned_columns() {
        let k = 300usize;
        let wg = 128u32;
        let slots = k.div_ceil(wg as usize);

        let mut rng = StdRng::seed_from_u64(7);
        let t1 = random_csr(k, k, 9, false, false, &mut rng);
        let seg = a_columns_segments(&t1, wg, slots);

        let stride = wg as usize + 1;
        assert_eq!(seg.len(), k * stride);

        let mut saw_empty = false;
        for row in 0..k {
            let (start, end) = (t1.indptr[row], t1.indptr[row + 1]);
            assert_eq!(seg[row * stride], start, "row {} start", row);
            assert_eq!(seg[row * stride + wg as usize], end, "row {} end", row);

            for t in 0..wg as usize {
                let (lo, hi) = (seg[row * stride + t], seg[row * stride + t + 1]);
                assert!(lo <= hi, "row {} thread {}: bounds inverted", row, t);
                if lo == hi {
                    saw_empty = true;
                }
                let base = (t * slots) as u32;
                for p in lo..hi {
                    let col = t1.indices[p as usize];
                    assert!(
                        col >= base && col < base + slots as u32,
                        "row {} thread {}: column {} outside [{}, {})",
                        row,
                        t,
                        col,
                        base,
                        base + slots as u32
                    );
                }
            }
        }
        assert!(
            saw_empty,
            "no empty block seen, the sparse case is untested"
        );
    }

    /// Drive [launch_fw_columns_a] directly so the reduction arm can be forced.
    ///
    /// ### Params
    ///
    /// * `t1` - `Bᵀ K² B` as CSR `k × k`
    /// * `a_prev_t` - `A_prevᵀ` as CSR `n × k`
    /// * `k2_b` - `K² B` as CSR `n × k`
    /// * `n` - Cells
    /// * `k` - Archetypes
    /// * `n_iters` - Frank-Wolfe iterations
    /// * `pruning` - Pruning threshold, or `None`
    /// * `use_plane` - Force the plane or shared-memory reduction
    /// * `wg` - Force a workgroup width, or `None` for the selected tier
    /// * `client` - Compute client
    ///
    /// ### Returns
    ///
    /// `(atom indices, atom weights, atom counts, stride)`.
    #[allow(clippy::too_many_arguments)]
    fn run_columns_a_raw(
        t1: &CompressedSparseData2<f32>,
        a_prev_t: &CompressedSparseData2<f32>,
        k2_b: &CompressedSparseData2<f32>,
        n: usize,
        k: usize,
        n_iters: usize,
        pruning: Option<f32>,
        use_plane: bool,
        wg: Option<u32>,
        client: &ComputeClient<WgpuRuntime>,
    ) -> (Vec<u32>, Vec<f32>, Vec<u32>, usize) {
        let cap = a_columns_capacity(a_prev_t, n_iters);

        let t1_gpu = GpuCompressedSparseData::<WgpuRuntime, f32>::from_compressed_sparse_data_2(
            t1, false, client,
        )
        .expect("upload failed");
        let width = wg.unwrap_or(A_COLUMNS_WG);
        let seg_host = a_columns_segments(t1, width, k.div_ceil(width as usize));
        let t1_seg =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&seg_host, vec![seg_host.len()], client);
        let a_gpu = GpuCompressedSparseData::<WgpuRuntime, f32>::from_compressed_sparse_data_2(
            a_prev_t, false, client,
        )
        .expect("upload failed");
        let k2b_gpu = GpuCompressedSparseData::<WgpuRuntime, f32>::from_compressed_sparse_data_2(
            k2_b, false, client,
        )
        .expect("upload failed");

        let atom_idx = GpuTensor::<WgpuRuntime, u32>::empty(vec![n * cap], client);
        let atom_val = GpuTensor::<WgpuRuntime, f32>::empty(vec![n * cap], client);
        let atom_cnt = GpuTensor::<WgpuRuntime, u32>::empty(vec![n], client);
        let thr =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&[pruning.unwrap_or(0.0)], vec![1], client);

        launch_fw_columns_a(
            &t1_gpu,
            &t1_seg,
            &a_gpu,
            &k2b_gpu,
            &atom_idx,
            &atom_val,
            &atom_cnt,
            &thr,
            n,
            k,
            n_iters,
            cap as u32,
            pruning,
            Some(use_plane),
            wg,
            client,
        )
        .expect("launch failed");

        (
            atom_idx.clone().read(client).expect("read failed"),
            atom_val.clone().read(client).expect("read failed"),
            atom_cnt.clone().read(client).expect("read failed"),
            cap,
        )
    }

    /// The shared-memory tree arm must agree with the plane arm.
    ///
    /// `plane_reduce_viable` is true on Apple Silicon, so every other test in
    /// this file takes the plane path and the `else` branches, which carry their
    /// own tie-break and barrier structure, would otherwise never execute
    /// anywhere.
    #[test]
    fn test_fw_columns_a_reduction_arms_agree() {
        let Some(device) = try_device() else {
            eprintln!("no GPU available, skipping");
            return;
        };
        let client = WgpuRuntime::client(&device);

        let n = 1_500usize;
        let k = 300usize;
        let n_iters = 12usize;

        // Coarse values so the tie-break is exercised: that is where the two
        // arms are most likely to diverge.
        let mut rng = StdRng::seed_from_u64(5);
        let t1 = random_csr(k, k, 12, false, true, &mut rng);
        let a_prev_t = random_csr(n, k, 6, true, true, &mut rng);
        let k2_b = random_csr(n, k, 8, false, true, &mut rng);

        for pruning in [None, Some(1e-7f32)] {
            let (plane_idx, plane_val, plane_cnt, cap) = run_columns_a_raw(
                &t1, &a_prev_t, &k2_b, n, k, n_iters, pruning, true, None, &client,
            );
            let (tree_idx, tree_val, tree_cnt, tree_cap) = run_columns_a_raw(
                &t1, &a_prev_t, &k2_b, n, k, n_iters, pruning, false, None, &client,
            );

            assert_eq!(cap, tree_cap);
            assert_eq!(
                plane_cnt, tree_cnt,
                "atom counts differ (pruning {:?})",
                pruning
            );
            assert_eq!(
                plane_idx, tree_idx,
                "atom indices differ (pruning {:?})",
                pruning
            );
            for (cell, (a, b)) in plane_val.iter().zip(tree_val.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5 * a.abs().max(1e-3),
                    "weight {} differs at slot {} (pruning {:?}): {} vs {}",
                    cell,
                    cell % cap,
                    pruning,
                    a,
                    b
                );
            }
        }
    }

    /// The atom capacity ceiling must be exact on both sides.
    ///
    /// `cap = widest column of A_prev + n_iters`, and thread `i` owns atom slot
    /// `i`, so `cap` equal to the chosen workgroup width is the last
    /// configuration the kernel can represent and the last append writes the
    /// highest in-bounds slot. One more has to reach a wider tier, or the CPU
    /// when none is wide enough. Pinned to the narrowest tier here, which is the
    /// width this `k` selects.
    #[test]
    fn test_fw_columns_a_capacity_boundary() {
        let Some(device) = try_device() else {
            eprintln!("no GPU available, skipping");
            return;
        };
        let client = WgpuRuntime::client(&device);

        let n = 300usize;
        let k = 300usize;
        let n_iters = 15usize;
        let wg = A_COLUMNS_WG as usize;

        for widest in [wg - n_iters - 1, wg - n_iters, wg - n_iters + 1] {
            let mut rng = StdRng::seed_from_u64(3);
            let t1 = random_csr(k, k, 12, false, false, &mut rng);
            let a_prev_t = random_csr(n, k, widest, true, false, &mut rng);
            let k2_b = random_csr(n, k, 8, false, false, &mut rng);

            let cap = a_columns_capacity(&a_prev_t, n_iters);
            assert_eq!(cap, widest + n_iters);

            let expected = fw_columns_a(&t1, &a_prev_t, &k2_b, k, n, n_iters, Some(1e-7));
            let mut backend = GpuFwArgminB::<WgpuRuntime>::new(n, k, client.clone());
            let actual = backend
                .columns_a(&t1, &a_prev_t, &k2_b, k, n, n_iters, Some(1e-7))
                .expect("column solve failed");

            assert_eq!(actual.len(), n);
            let mass: f64 = actual
                .iter()
                .map(|atoms| atoms.atoms().1.iter().map(|&w| w as f64).sum::<f64>())
                .sum();
            assert!(
                (mass - n as f64).abs() < 0.01 * n as f64,
                "columns do not sum to one at cap {}: mass {}",
                cap,
                mass
            );

            let differing = expected
                .iter()
                .zip(actual.iter())
                .filter(|(want, got)| {
                    let mut w: Vec<_> = want.atoms().0.to_vec();
                    let mut g: Vec<_> = got.atoms().0.to_vec();
                    w.sort_unstable();
                    g.sort_unstable();
                    w != g
                })
                .count();
            assert!(
                differing * 100 <= n,
                "{} / {} columns differ at cap {} (ceiling {})",
                differing,
                n,
                cap,
                wg
            );
        }
    }
}
