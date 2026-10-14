//! SEACells on MLX: the Frank-Wolfe B-gradient argmin as custom Metal kernels.
//!
//! [MlxFwArgminB] implements the [FwArgminB] seam and [seacells_fit_mlx]
//! drives the same [SEACells::fit_with] loop as the CPU and cubecl paths. The A
//! update stays on the CPU (`FwArgminB::columns_a` default).
//!
//! ### The argmin kernel
//!
//! `G[i, c] = sum_m K²B[i, m] t1[m, c] - K²Aᵀ[i, c]` is never stored. One SIMD
//! group owns one row and one tile of `T` columns at a time, and builds that
//! tile of the gradient row in threadgroup memory: the lanes load 32 non-zeros
//! of the `K²B` row, broadcast them one by one with `simd_shuffle`, and stride
//! the matching tile segment of the sparse `t1` row, scatter-adding into the
//! buffer. Within one `t1` row the columns are distinct, so no two lanes hit
//! the same address in a step. `K²Aᵀ` is scattered the same way. Lane `l` then
//! folds columns `l, l + 32, ...` of the tile into a running `(min, row)` in
//! registers. A `[k, tiles + 1]` segment table for `t1` and an `[n, tiles + 1]`
//! one for `K²Aᵀ` are built once per B update, so no lane ever searches.
//!
//! SIMD groups grid-stride over rows in ascending order with a strict `<`, so
//! the lowest row wins a tie within a group; a second kernel reduces the
//! per-group partials and breaks ties on the row index, as the CPU scan does.
//!
//! The summation order per entry is the CPU's: products over ascending `m`,
//! then `K²Aᵀ` subtracted.

use faer::MatRef;

use crate::mlx::ffi::{
    Array, MLX_FLOAT32, MLX_UINT32, MetalKernel, OutputSpec, Stream, eval_all,
    install_error_handler,
};
use crate::prelude::*;
use crate::single_cell::mc_generation::seacells::{
    FwArgminB, SEACells, SEACellsParams, SeacellsFitResult, assignments_to_metacells,
    csr_column_segments,
};

////////////
// Consts //
////////////

/// Lanes per SIMD group on Apple GPUs.
const SIMD_WIDTH: usize = 32;

/// Columns of the gradient one SIMD group holds at a time. Sets the
/// threadgroup buffer (`4 T` bytes per SIMD group) and the running minima in
/// registers (`2 T / 32` per lane). Swept 256 / 512 / 1024 on captured fit
/// operands (M1 Max, per call): 4.8 / 6.4 / 6.9 ms at 50k x 666 and 152 /
/// 159 / 220 ms at 200k x 2666.
const B_TILE: usize = 256;

/// SIMD groups per threadgroup of the argmin kernel. 4 against 8 was within
/// noise at both shapes.
const B_TG_SIMDGROUPS: usize = 4;

/// SIMD groups grid-striding over the rows, per column tile. Fixed rather than
/// derived from `n`, so the partials stay `[B_GROUPS, k]`. 512 / 1024 / 2048
/// were within 10% of each other at both shapes.
const B_GROUPS: usize = 1024;

/// Threads per threadgroup of the partial reduction, one thread per column.
const REDUCE_TG: usize = 256;

////////////////////
// Kernel sources //
////////////////////

/// Fused gradient and per-group column argmin, see the module doc.
///
/// Inputs: `k2b_indptr` `[n + 1]`, `k2b_indices`, `k2b_values`, `t1_indices`,
/// `t1_values`, `t1_seg` `[k, NT + 1]`, `t2_indices`, `t2_values`, `t2_seg`
/// `[n, NT + 1]`, `b_indptr` `[n + 1]`, `b_indices`, `b_values`, `dims` `[1]`
/// (`n`). Outputs: `part_val`, `part_idx` `[groups, K]`, `gap_part`
/// `[NT, groups]`. Template ints: `K`, `T` (tile), `NT` (tiles), `SG` (SIMD
/// groups per threadgroup).
const ARGMIN_SOURCE: &str = r#"
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint grp = threadgroup_position_in_grid.x * uint(SG) + sg;
    uint n_grp = threadgroups_per_grid.x * uint(SG);
    uint tile = threadgroup_position_in_grid.y;
    uint c0 = tile * uint(T);
    uint width = min(uint(T), uint(K) - c0);
    uint n = dims[0];
    const uint J = uint(T) / SIMD_W;
    const uint SEGS = uint(NT) + 1;

    threadgroup float buf_all[SG * T];
    threadgroup float* buf = buf_all + sg * uint(T);

    float run_val[J];
    uint run_idx[J];
    for (uint j = 0; j < J; ++j) {
        run_val[j] = FLT_MAX;
        run_idx[j] = 0;
    }
    float gap = 0.0f;

    for (uint row = grp; row < n; row += n_grp) {
        for (uint j = 0; j < J; ++j) {
            buf[lane + j * SIMD_W] = 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint lo = k2b_indptr[row];
        uint hi = k2b_indptr[row + 1];
        for (uint base = lo; base < hi; base += SIMD_W) {
            uint e = base + lane;
            uint m = 0;
            float v = 0.0f;
            if (e < hi) {
                m = k2b_indices[e];
                v = k2b_values[e];
            }
            uint cnt = min(SIMD_W, hi - base);
            for (uint t = 0; t < cnt; ++t) {
                uint mt = simd_shuffle(m, ushort(t));
                float vt = simd_shuffle(v, ushort(t));
                uint s1 = t1_seg[mt * SEGS + tile + 1];
                for (uint p = t1_seg[mt * SEGS + tile] + lane; p < s1; p += SIMD_W) {
                    buf[t1_indices[p] - c0] += vt * t1_values[p];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        uint q1 = t2_seg[row * SEGS + tile + 1];
        for (uint p = t2_seg[row * SEGS + tile] + lane; p < q1; p += SIMD_W) {
            buf[t2_indices[p] - c0] -= t2_values[p];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint b1 = b_indptr[row + 1];
        for (uint p = b_indptr[row] + lane; p < b1; p += SIMD_W) {
            uint col = b_indices[p];
            if (col >= c0 && col < c0 + width) {
                gap += b_values[p] * buf[col - c0];
            }
        }
        for (uint j = 0; j < J; ++j) {
            uint col = lane + j * SIMD_W;
            float g = buf[col];
            if (col < width && g < run_val[j]) {
                run_val[j] = g;
                run_idx[j] = row;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    size_t ob = size_t(grp) * uint(K) + c0;
    for (uint j = 0; j < J; ++j) {
        uint col = lane + j * SIMD_W;
        if (col < width) {
            part_val[ob + col] = run_val[j];
            part_idx[ob + col] = run_idx[j];
        }
    }
    gap = simd_sum(gap);
    if (lane == 0) {
        gap_part[tile * n_grp + grp] = gap;
    }
"#;

/// Reduce the per-group partials to one `(min, row)` per column, ties to the
/// lowest row.
///
/// Inputs: `part_val`, `part_idx` `[groups, K]`, `dims` `[1]` (`groups`).
/// Outputs: `out_val`, `out_idx` `[K]`. Template int: `K`.
const REDUCE_SOURCE: &str = r#"
    uint c = thread_position_in_grid.x;
    if (c >= uint(K)) {
        return;
    }
    float best = FLT_MAX;
    uint best_row = UINT_MAX;
    for (uint g = 0; g < dims[0]; ++g) {
        size_t o = size_t(g) * uint(K) + c;
        float v = part_val[o];
        uint i = part_idx[o];
        if (v < best || (v == best && i < best_row)) {
            best = v;
            best_row = i;
        }
    }
    out_val[c] = best;
    out_idx[c] = best_row;
"#;

/////////////
// Helpers //
/////////////

/// Narrow a size to the `i32` MLX shapes and grids take.
///
/// ### Params
///
/// * `x` - Size
///
/// ### Returns
///
/// `x` as `i32`, or `InvalidArgument` if it does not fit.
fn to_i32(x: usize) -> Result<i32, BixverseErrors> {
    i32::try_from(x).map_err(|_| {
        BixverseErrors::InvalidArgument(format!("MLX SEACells: size {x} exceeds the i32 range"))
    })
}

/// Upload a CSR's three buffers.
///
/// ### Params
///
/// * `csr` - CSR matrix
///
/// ### Returns
///
/// `(indptr, indices, values)` on the device, or `InvalidArgument` if a length
/// exceeds the MLX index range.
fn upload_csr(csr: &CompressedSparseData2<f32>) -> Result<(Array, Array, Array), BixverseErrors> {
    let nnz = to_i32(csr.indices.len())?;
    Ok((
        Array::from_u32(&csr.indptr, &[to_i32(csr.indptr.len())?]),
        Array::from_u32(&csr.indices, &[nnz]),
        Array::from_f32(&csr.data, &[nnz]),
    ))
}

/// A CSR with a per-row column-tile segment table in place of `indptr`.
struct TiledCsr {
    /// Column indices `[nnz]`
    indices: Array,
    /// Values `[nnz]`
    values: Array,
    /// Segment table `[rows, n_tiles + 1]`, see [csr_column_segments]
    seg: Array,
}

impl TiledCsr {
    /// Upload a CSR with its tile segments.
    ///
    /// ### Params
    ///
    /// * `csr` - CSR matrix, column indices ascending per row
    /// * `n_tiles` - Column tiles
    /// * `tile` - Columns per tile
    ///
    /// ### Returns
    ///
    /// The device copy, or `InvalidArgument` if a length exceeds the MLX index
    /// range.
    fn new(
        csr: &CompressedSparseData2<f32>,
        n_tiles: usize,
        tile: usize,
    ) -> Result<Self, BixverseErrors> {
        let nnz = to_i32(csr.indices.len())?;
        let seg = csr_column_segments(csr, n_tiles, tile);
        let seg_len = to_i32(seg.len())?;
        Ok(Self {
            indices: Array::from_u32(&csr.indices, &[nnz]),
            values: Array::from_f32(&csr.data, &[nnz]),
            seg: Array::from_vec_u32(seg, &[seg_len]),
        })
    }
}

//////////////
// Back end //
//////////////

/// MLX back end for the B-gradient argmin.
///
/// `t1` and `K²Aᵀ` are uploaded with their tile segment tables once per B
/// update in [FwArgminB::begin]; `K²B` and `B` are uploaded per Frank-Wolfe
/// iteration. No buffer here is `n × k` dense and the per-lane register state
/// is fixed by [B_TILE], so there is no ceiling on `k`.
pub struct MlxFwArgminB {
    /// Stream all kernels run on
    s: Stream,
    /// Gradient and per-group argmin, see [ARGMIN_SOURCE]
    argmin: MetalKernel,
    /// Partial reduction, see [REDUCE_SOURCE]
    reduce: MetalKernel,
    /// Number of cells
    n: usize,
    /// Number of archetypes
    k: usize,
    /// Columns per tile, `min(k, B_TILE)` rounded up to whole SIMD groups
    tile: usize,
    /// Column tiles, `ceil(k / tile)`
    n_tiles: usize,
    /// SIMD groups per tile, a multiple of [B_TG_SIMDGROUPS]
    groups: usize,
    /// `[n]` as a one-element u32 array for the argmin kernel
    dims_n: Array,
    /// `[groups]` as a one-element u32 array for the reduction
    dims_groups: Array,
    /// `A Aᵀ` with its tile segments, rebound per B update
    t1: Option<TiledCsr>,
    /// `K² Aᵀ` with its tile segments, rebound per B update
    t2: Option<TiledCsr>,
}

impl MlxFwArgminB {
    /// Compile the kernels and size the launch for an `n × k` problem.
    ///
    /// ### Params
    ///
    /// * `n` - Number of cells
    /// * `k` - Number of archetypes
    ///
    /// ### Returns
    ///
    /// Self, or `InvalidArgument` if `n` or `k` is zero or a launch dimension
    /// exceeds the MLX range.
    pub fn new(n: usize, k: usize) -> Result<Self, BixverseErrors> {
        if n == 0 || k == 0 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "MlxFwArgminB: empty problem, n = {n}, k = {k}"
            )));
        }
        install_error_handler();
        let header = format!("constant constexpr uint SIMD_W = {SIMD_WIDTH};\n");
        let argmin = MetalKernel::new(
            "bixverse_seacells_fw_argmin_b",
            &[
                "k2b_indptr",
                "k2b_indices",
                "k2b_values",
                "t1_indices",
                "t1_values",
                "t1_seg",
                "t2_indices",
                "t2_values",
                "t2_seg",
                "b_indptr",
                "b_indices",
                "b_values",
                "dims",
            ],
            &["part_val", "part_idx", "gap_part"],
            &header,
            ARGMIN_SOURCE,
        );
        let reduce = MetalKernel::new(
            "bixverse_seacells_reduce_argmin",
            &["part_val", "part_idx", "dims"],
            &["out_val", "out_idx"],
            "",
            REDUCE_SOURCE,
        );

        let tile = k.min(B_TILE).next_multiple_of(SIMD_WIDTH);
        let n_tiles = k.div_ceil(tile);
        let groups = B_GROUPS.min(n.next_multiple_of(B_TG_SIMDGROUPS));
        to_i32(groups * k)?;
        to_i32(groups * SIMD_WIDTH)?;

        Ok(Self {
            s: Stream::default_gpu(),
            argmin,
            reduce,
            n,
            k,
            tile,
            n_tiles,
            groups,
            dims_n: Array::from_u32(&[n as u32], &[1]),
            dims_groups: Array::from_u32(&[groups as u32], &[1]),
            t1: None,
            t2: None,
        })
    }
}

impl FwArgminB for MlxFwArgminB {
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
        self.t1 = Some(TiledCsr::new(t1, self.n_tiles, self.tile)?);
        self.t2 = Some(TiledCsr::new(t2, self.n_tiles, self.tile)?);
        Ok(())
    }

    fn argmins(
        &mut self,
        k2_b: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
    ) -> Result<(Vec<usize>, f32), BixverseErrors> {
        // The kernel indexes `indptr` up to `n` unchecked.
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

        let (k2b_ptr, k2b_idx, k2b_val) = upload_csr(k2_b)?;
        let (b_ptr, b_idx, b_val) = upload_csr(b)?;

        let (k, groups) = (to_i32(self.k)?, to_i32(self.groups)?);
        let parts = self.argmin.apply(
            &[
                &k2b_ptr,
                &k2b_idx,
                &k2b_val,
                &t1.indices,
                &t1.values,
                &t1.seg,
                &t2.indices,
                &t2.values,
                &t2.seg,
                &b_ptr,
                &b_idx,
                &b_val,
                &self.dims_n,
            ],
            &[
                OutputSpec {
                    shape: &[groups, k],
                    dtype: MLX_FLOAT32,
                },
                OutputSpec {
                    shape: &[groups, k],
                    dtype: MLX_UINT32,
                },
                OutputSpec {
                    shape: &[to_i32(self.n_tiles)?, groups],
                    dtype: MLX_FLOAT32,
                },
            ],
            [groups * SIMD_WIDTH as i32, to_i32(self.n_tiles)?, 1],
            [to_i32(B_TG_SIMDGROUPS * SIMD_WIDTH)?, 1, 1],
            &[
                ("K", k),
                ("T", to_i32(self.tile)?),
                ("NT", to_i32(self.n_tiles)?),
                ("SG", B_TG_SIMDGROUPS as i32),
            ],
            &self.s,
        )?;
        let out = self.reduce.apply(
            &[&parts[0], &parts[1], &self.dims_groups],
            &[
                OutputSpec {
                    shape: &[k],
                    dtype: MLX_FLOAT32,
                },
                OutputSpec {
                    shape: &[k],
                    dtype: MLX_UINT32,
                },
            ],
            [to_i32(self.k.next_multiple_of(REDUCE_TG))?, 1, 1],
            [REDUCE_TG as i32, 1, 1],
            &[("K", k)],
            &self.s,
        )?;
        eval_all(&[&out[0], &out[1], &parts[2]], false)?;

        // gap = |sum(B * G) - sum_c min_c G[:, c]|, matching the CPU scan.
        let g_dot_b: f32 = parts[2].as_f32()?.iter().sum();
        let g_dot_e: f32 = out[0].as_f32()?.iter().sum();
        let argmins = out[1].as_u32()?.iter().map(|&i| i as usize).collect();

        Ok((argmins, (g_dot_b - g_dot_e).abs()))
    }
}

//////////
// Main //
//////////

/// Run SEACells with the Frank-Wolfe B-gradient argmin on MLX.
///
/// Kernel construction, archetype initialisation, the A update and the RSS run
/// on the CPU exactly as in the pure-Rust path.
///
/// ### Params
///
/// * `pca` - PCA/SVD embedding (n_cells × n_components)
/// * `knn_indices` - kNN indices for each cell
/// * `knn_distances` - kNN distances for each cell
/// * `params` - Algorithm parameters
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `(hard assignments per cell, metacell groupings, archetype cell indices, RSS
/// history)`, one grouping and one archetype per archetype the initialisation
/// actually selected.
///
/// ### References
///
/// Persad, et al., Nat. Biotechnol., 2023
pub fn seacells_fit_mlx(
    pca: MatRef<f32>,
    knn_indices: &[Vec<usize>],
    knn_distances: &[Vec<f32>],
    params: &SEACellsParams,
    seed: usize,
    verbose: usize,
) -> Result<SeacellsFitResult, BixverseErrors> {
    let n = pca.nrows();

    let mut model = SEACells::new(n, params);
    model.construct_kernel_mat(pca, knn_indices, knn_distances, verbose);

    match params.n_landmarks {
        Some(n_landmarks) => model.initialise_archetypes_landmark(
            pca,
            knn_indices,
            knn_distances,
            n_landmarks,
            verbose,
            seed as u64,
        )?,
        None => model.initialise_archetypes(knn_indices, knn_distances, verbose, seed as u64)?,
    }

    // Initialisation dedups and can come back short of `n_sea_cells`.
    let archetypes = model.get_archetypes()?;
    let k = archetypes.len();

    let mut backend = MlxFwArgminB::new(n, k)?;
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
    use crate::single_cell::mc_generation::seacells::CpuFwArgminB;
    use approx::assert_relative_eq;

    /// Dense row-major to CSR, dropping exact zeros.
    ///
    /// ### Params
    ///
    /// * `dense` - Row-major `[rows, cols]`
    /// * `rows` - Rows
    /// * `cols` - Columns
    ///
    /// ### Returns
    ///
    /// The CSR.
    fn to_csr(dense: &[f32], rows: usize, cols: usize) -> CompressedSparseData2<f32> {
        let (mut data, mut indices, mut indptr) = (Vec::new(), Vec::new(), vec![0u32]);
        for r in 0..rows {
            for c in 0..cols {
                let v = dense[r * cols + c];
                if v != 0.0 {
                    data.push(v);
                    indices.push(c as u32);
                }
            }
            indptr.push(indices.len() as u32);
        }
        CompressedSparseData2::new_csr(&data, &indices, &indptr, None, (rows, cols))
    }

    /// Deterministic pseudo-random dense fixture with a controlled zero rate,
    /// the same generator as the cubecl kernel tests.
    ///
    /// ### Params
    ///
    /// * `len` - Number of values
    /// * `seed` - Offsets the sequence
    /// * `zero_every` - Every n-th value is zeroed
    ///
    /// ### Returns
    ///
    /// The values.
    fn fixture(len: usize, seed: usize, zero_every: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                if (i + seed).is_multiple_of(zero_every) {
                    0.0
                } else {
                    (((i * 37 + seed * 17) % 23) as f32 - 11.0) * 0.1
                }
            })
            .collect()
    }

    /// Symmetric `[k, k]` fixture, standing in for `A Aᵀ`. The CPU scan reads
    /// row `c` of `t1` as its column `c`, so an asymmetric `t1` is not a valid
    /// input.
    ///
    /// ### Params
    ///
    /// * `k` - Side
    /// * `seed` - Offsets the sequence
    ///
    /// ### Returns
    ///
    /// The values, row-major.
    fn sym_fixture(k: usize, seed: usize) -> Vec<f32> {
        let raw = fixture(k * k, seed, 4);
        (0..k * k)
            .map(|x| {
                let (i, j) = (x / k, x % k);
                if i <= j {
                    raw[i * k + j]
                } else {
                    raw[j * k + i]
                }
            })
            .collect()
    }

    /// Run the CPU and MLX back ends on the same dense operands and assert
    /// they agree: argmins exactly where the minimum is unambiguous, and
    /// otherwise a picked row that really is a minimum.
    ///
    /// ### Params
    ///
    /// * `k2b` - Dense `[n, k]`
    /// * `t1` - Dense `[k, k]`
    /// * `t2` - Dense `[n, k]`
    /// * `b` - Dense `[n, k]`
    /// * `n` - Rows
    /// * `k` - Columns
    ///
    /// ### Returns
    ///
    /// The MLX argmins.
    fn check_against_cpu(
        k2b: &[f32],
        t1: &[f32],
        t2: &[f32],
        b: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<usize> {
        let (k2b_s, t1_s, t2_s, b_s) = (
            to_csr(k2b, n, k),
            to_csr(t1, k, k),
            to_csr(t2, n, k),
            to_csr(b, n, k),
        );
        let mut cpu = CpuFwArgminB::default();
        cpu.begin(&t1_s, &t2_s).unwrap();
        let (want_idx, want_gap) = cpu.argmins(&k2b_s, &b_s).unwrap();

        let mut mlx = MlxFwArgminB::new(n, k).unwrap();
        mlx.begin(&t1_s, &t2_s).unwrap();
        let (got_idx, got_gap) = mlx.argmins(&k2b_s, &b_s).unwrap();

        let grad = |i: usize, c: usize| -> f32 {
            let mut g = 0.0f32;
            for m in 0..k {
                g += k2b[i * k + m] * t1[m * k + c];
            }
            g - t2[i * k + c]
        };
        let mut ties = 0usize;
        for c in 0..k {
            if got_idx[c] != want_idx[c] {
                ties += 1;
                assert_relative_eq!(
                    grad(got_idx[c], c),
                    grad(want_idx[c], c),
                    max_relative = 1e-5,
                    epsilon = 1e-6
                );
            }
        }
        assert!(ties * 50 <= k, "{ties} / {k} columns disagree at n = {n}");
        assert_relative_eq!(got_gap, want_gap, max_relative = 1e-3, epsilon = 1e-3);
        got_idx
    }

    /// Parity with the CPU scan. `(3000, 1100)` covers several column tiles
    /// with a partial last one and several rows per SIMD group.
    #[test]
    fn test_mlx_fw_argmin_b_matches_cpu() {
        for (n, k) in [(40usize, 7usize), (257, 33), (1000, 130), (3000, 1100)] {
            let k2b = fixture(n * k, 1, 3);
            let t1 = sym_fixture(k, 5);
            let t2 = fixture(n * k, 9, 5);
            let b = fixture(n * k, 13, 11);
            check_against_cpu(&k2b, &t1, &t2, &b, n, k);
        }
    }

    /// Rows with no non-zeros in `K²B` still take part: their gradient is
    /// `-K²Aᵀ[i, :]`.
    #[test]
    fn test_mlx_fw_argmin_b_handles_empty_rows() {
        let (n, k) = (64usize, 9usize);
        let mut k2b = fixture(n * k, 1, 3);
        for i in (0..n).step_by(3) {
            k2b[i * k..(i + 1) * k].fill(0.0);
        }
        let t1 = sym_fixture(k, 5);
        let t2 = fixture(n * k, 9, 5);
        let b = fixture(n * k, 13, 11);
        check_against_cpu(&k2b, &t1, &t2, &b, n, k);
    }

    /// Ties go to the lowest row, across SIMD groups and column tiles.
    #[test]
    fn test_mlx_fw_argmin_b_tie_breaks_on_lowest_row() {
        let (n, k) = (4096usize, 600usize);
        let mut t1 = vec![0.0f32; k * k];
        for c in 0..k {
            t1[c * k + c] = 1.0;
        }
        let t2 = vec![0.0f32; n * k];
        let b = vec![0.0f32; n * k];
        let k2b: Vec<f32> = (0..n * k).map(|x| -1.0 - (x % k) as f32).collect();

        let got = check_against_cpu(&k2b, &t1, &t2, &b, n, k);
        assert!(got.iter().all(|&i| i == 0), "ties did not resolve to row 0");
    }
}
