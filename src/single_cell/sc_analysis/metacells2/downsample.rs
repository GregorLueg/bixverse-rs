//! Per-row binomial downsampling using a cumulative segment tree.
//!
//! Each cell's gene counts are downsampled to a common UMI total by sampling
//! without replacement from the multinomial defined by the row's non-zero
//! entries. Each draw is `O(log nnz)` via the tree; total cost per row is
//! `O(nnz + samples · log nnz)`.
//!
//! The tree is built only over the row's non-zero entries (not over the full
//! gene axis), so sparse rows pay sparse cost. Padded zero leaves bring the
//! tree size up to the next power of two; their zero counts mean the random
//! walk can never route to them.
//!
//! Per-row seeding follows upstream MC2: `row_seed = seed + row_index * 997`
//! when `seed != 0`, else `row_seed = 0`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::prelude::*;

use super::params::SelectParams;
use super::pile::Pile;

/// Slice seed multiplier
const SLICE_SEED_MULT: u64 = 997;

/// Downsample each cell of `pile.raw` to a common UMI target.
///
/// The target is computed from the cell library size distribution as
/// `min(max(min_samples, q(min_q)), q(max_q))` where `q(p)` is the linearly
/// interpolated `p`-th quantile of `pile.umis_per_cell`. Cells already at or
/// below the target are copied through unchanged; others are subsampled to
/// exactly the target.
///
/// Populates `pile.downsampled` with the same sparsity pattern as
/// `pile.raw`. Explicit zeros may appear where a non-zero entry was sampled
/// to zero — downstream stages tolerate this.
///
/// ### Params
///
/// * `pile` - The pile to downsample.
/// * `params` - Selection parameters; only the three `downsample_*` fields
///   are read.
/// * `rng_seed` - Pile-level seed. Per-row seed is derived as
///   `rng_seed + row_index * 997` (matching upstream MC2).
pub fn downsample_pile(pile: &mut Pile, params: &SelectParams, rng_seed: u64) {
    let raw = &pile.raw;
    let n_cells = raw.shape.0;

    let mut out_data = vec![0u32; raw.data.len()];

    if n_cells == 0 {
        pile.downsampled = Some(CompressedSparseData2 {
            data: out_data,
            indices: raw.indices.clone(),
            indptr: raw.indptr.clone(),
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: raw.shape,
        });
        return;
    }

    let target = compute_downsample_target(
        &pile.umis_per_cell,
        params.downsample_min_samples,
        params.downsample_min_cell_quantile,
        params.downsample_max_cell_quantile,
    );

    let out_ptr_addr = out_data.as_mut_ptr() as usize;
    let indptr = &raw.indptr;
    let in_data = &raw.data;

    (0..n_cells).into_par_iter().for_each(|row_index| {
        let start = indptr[row_index];
        let end = indptr[row_index + 1];
        if start == end {
            return;
        }

        let row_in = &in_data[start..end];
        let row_seed = if rng_seed == 0 {
            0
        } else {
            rng_seed.wrapping_add((row_index as u64).wrapping_mul(SLICE_SEED_MULT))
        };

        // SAFETY: `indptr` defines disjoint `[start, end)` ranges per row;
        // `into_par_iter` over `0..n_cells` schedules each `row_index` to
        // exactly one task, so writes through these slices never alias.
        // Same pattern as `compute_grad_a_colmajor` in the SEACells module.
        let row_out = unsafe {
            std::slice::from_raw_parts_mut((out_ptr_addr as *mut u32).add(start), end - start)
        };

        downsample_row(row_in, row_out, target, row_seed);
    });

    pile.downsampled = Some(CompressedSparseData2 {
        data: out_data,
        indices: raw.indices.clone(),
        indptr: raw.indptr.clone(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: raw.shape,
    });
}

/// Downsample a single row's non-zero counts to at most `samples` total.
///
/// If the row total is already `<= samples`, the input is copied through
/// unchanged. Otherwise exactly `samples` items are drawn without replacement
/// via the cumulative segment tree.
///
/// ### Params
///
/// * `input` - Raw counts for the row's non-zero entries.
/// * `output` - Destination slice of the same length as `input`.
/// * `samples` - Target UMI count to downsample to.
/// * `seed` - RNG seed for this row; `0` means unseeded.
///
/// ### Returns
///
/// Nothing; results are written into `output`.
fn downsample_row(input: &[u32], output: &mut [u32], samples: u32, seed: u64) {
    debug_assert_eq!(input.len(), output.len());

    if input.is_empty() {
        return;
    }
    if input.len() == 1 {
        output[0] = input[0].min(samples);
        return;
    }

    let mut tree = vec![0u64; tree_size(input.len())];
    initialise_tree(input, &mut tree);

    let total = *tree.last().expect("tree non-empty for input.len() >= 2");
    if total <= samples as u64 {
        output.copy_from_slice(input);
        return;
    }

    for o in output.iter_mut() {
        *o = 0;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..samples {
        let r = rng.random_range(0..total);
        let idx = random_sample(&mut tree, r);
        debug_assert!(idx < output.len(), "padded leaf reached — tree malformed");
        output[idx] += 1;
    }
}

/// Linearly interpolated quantile (matching numpy's default method).
///
/// ### Params
///
/// * `values` - Unsorted slice of values to compute the quantile over.
/// * `q` - Quantile in `[0.0, 1.0]`.
///
/// ### Returns
///
/// The interpolated quantile value, or `0.0` if `values` is empty.
fn quantile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute the per-pile UMI target for downsampling.
///
/// The target is `min(max(min_samples, q(min_q)), q(max_q))`, rounded to the
/// nearest `u32`.
///
/// ### Params
///
/// * `umis_per_cell` - Per-cell UMI totals.
/// * `min_samples` - Hard lower bound on the target.
/// * `min_cell_quantile` - Lower quantile; the target is at least this value.
/// * `max_cell_quantile` - Upper quantile; the target is capped at this value.
///
/// ### Returns
///
/// The clamped, rounded UMI target.
fn compute_downsample_target(
    umis_per_cell: &[f32],
    min_samples: u32,
    min_cell_quantile: f32,
    max_cell_quantile: f32,
) -> u32 {
    let lo = quantile(umis_per_cell, min_cell_quantile);
    let hi = quantile(umis_per_cell, max_cell_quantile);
    let target = lo.max(min_samples as f32).min(hi);
    target.round().max(0.0) as u32
}

/// Smallest power of two greater than or equal to `n`.
///
/// Returns `1` for `n <= 1`.
///
/// ### Params
///
/// * `n` - Input value.
///
/// ### Returns
///
/// The smallest power of two `>= n`.
#[inline]
fn ceil_power_of_two(n: usize) -> usize {
    if n <= 1 { 1 } else { n.next_power_of_two() }
}

/// Total length of the segment-tree buffer required for an input of size `n`.
///
/// ### Params
///
/// * `n` - Number of input elements.
///
/// ### Returns
///
/// Buffer length `2 * ceil_pow2(n) - 1`, or `0` for `n <= 1`.
#[inline]
fn tree_size(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        2 * ceil_power_of_two(n) - 1
    }
}

/// Build a cumulative segment tree over `input` in bottom-up order.
///
/// Leaves occupy the first `ceil_pow2(n)` slots, padded with zeros. Each
/// successive level holds pair-wise sums of the previous one, ending with a
/// single root at `tree[tree.len() - 1]` equal to the total of `input`.
///
/// ### Params
///
/// * `input` - Raw counts; must have length `>= 2`.
/// * `tree`  - Buffer of length `tree_size(input.len())`; overwritten in place.
///
/// ### Returns
///
/// Nothing; `tree` is populated in place.
fn initialise_tree(input: &[u32], tree: &mut [u64]) {
    let n = input.len();
    debug_assert!(n >= 2);

    let leaf_count = ceil_power_of_two(n);
    debug_assert_eq!(tree.len(), 2 * leaf_count - 1);

    for (slot, &v) in tree.iter_mut().zip(input.iter()) {
        *slot = v as u64;
    }
    for slot in tree[n..leaf_count].iter_mut() {
        *slot = 0;
    }

    let mut level_size = leaf_count;
    let mut base = 0usize;
    while level_size > 1 {
        let next_base = base + level_size;
        let half = level_size / 2;
        for i in 0..half {
            tree[next_base + i] = tree[base + 2 * i] + tree[base + 2 * i + 1];
        }
        base = next_base;
        level_size = half;
    }
}

/// Sample one item from the segment tree, decrementing counts along the path.
///
/// Traverses from root to leaf guided by `random`, subtracting `1` at every
/// node visited so that subsequent calls draw without replacement.
///
/// ### Params
///
/// * `tree`   - Mutable segment tree built by `initialise_tree`; mutated in place.
/// * `random` - Uniform draw in `[0, root_count)`; consumed by the walk.
///
/// ### Returns
///
/// The leaf index in `[0, n)` corresponding to the sampled entry.
fn random_sample(tree: &mut [u64], mut random: u64) -> usize {
    let mut size_of_level: usize = 1;
    let mut base_of_level: isize = (tree.len() as isize) - 1;
    let mut index_in_level: usize = 0;

    loop {
        let idx = (base_of_level as usize) + index_in_level;
        debug_assert!(tree[idx] > random);
        tree[idx] -= 1;

        size_of_level *= 2;
        base_of_level -= size_of_level as isize;

        if base_of_level < 0 {
            return index_in_level;
        }

        index_in_level *= 2;
        let left_idx = (base_of_level as usize) + index_in_level;
        let left_val = tree[left_idx];
        if random >= left_val {
            random -= left_val;
            index_in_level += 1;
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_power_of_two_values() {
        assert_eq!(ceil_power_of_two(0), 1);
        assert_eq!(ceil_power_of_two(1), 1);
        assert_eq!(ceil_power_of_two(2), 2);
        assert_eq!(ceil_power_of_two(3), 4);
        assert_eq!(ceil_power_of_two(4), 4);
        assert_eq!(ceil_power_of_two(5), 8);
        assert_eq!(ceil_power_of_two(8), 8);
        assert_eq!(ceil_power_of_two(9), 16);
    }

    #[test]
    fn tree_size_values() {
        assert_eq!(tree_size(0), 0);
        assert_eq!(tree_size(1), 0);
        assert_eq!(tree_size(2), 3); // 2*2 - 1
        assert_eq!(tree_size(3), 7); // 2*4 - 1
        assert_eq!(tree_size(4), 7);
        assert_eq!(tree_size(5), 15);
    }

    #[test]
    fn quantile_known_values() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&v, 0.0) - 1.0).abs() < 1e-6);
        assert!((quantile(&v, 1.0) - 5.0).abs() < 1e-6);
        assert!((quantile(&v, 0.5) - 3.0).abs() < 1e-6);
        // 0.25 of 5 elements: pos = 0.25 * 4 = 1.0 -> exactly index 1 -> 2.0
        assert!((quantile(&v, 0.25) - 2.0).abs() < 1e-6);
        // empty input falls back to 0.0
        assert_eq!(quantile(&[], 0.5), 0.0);
    }

    #[test]
    fn compute_downsample_target_clamps() {
        let umis = vec![100.0f32, 200.0, 300.0, 400.0, 500.0];
        // min_samples below quantile floor: result clamped up to floor.
        assert_eq!(compute_downsample_target(&umis, 50, 0.0, 1.0), 100);
        // min_samples within band: result is min_samples.
        assert_eq!(compute_downsample_target(&umis, 250, 0.0, 1.0), 250);
        // min_samples above quantile ceiling: result clamped down to ceiling.
        assert_eq!(compute_downsample_target(&umis, 600, 0.0, 1.0), 500);
    }

    #[test]
    fn initialise_tree_root_is_total() {
        let input = vec![1u32, 2, 3, 4, 5];
        let mut tree = vec![0u64; tree_size(input.len())];
        initialise_tree(&input, &mut tree);
        // Root holds the sum of all leaves.
        assert_eq!(*tree.last().unwrap(), 15);
        // Leaves preserved at the start.
        for (i, &v) in input.iter().enumerate() {
            assert_eq!(tree[i], v as u64);
        }
    }

    #[test]
    fn downsample_row_passthrough_when_total_le_samples() {
        let input = vec![1u32, 2, 3];
        let mut output = vec![0u32; 3];
        downsample_row(&input, &mut output, 10, 42);
        assert_eq!(output, vec![1, 2, 3]);
    }

    #[test]
    fn downsample_row_subsample_sum_matches_target() {
        let input = vec![20u32, 30, 50];
        let mut output = vec![0u32; 3];
        downsample_row(&input, &mut output, 30, 42);
        assert_eq!(output.iter().sum::<u32>(), 30);
        // No bin exceeds its input (without-replacement guarantee).
        for (o, i) in output.iter().zip(input.iter()) {
            assert!(o <= i);
        }
    }

    #[test]
    fn downsample_row_deterministic_under_same_seed() {
        let input = vec![10u32, 20, 30, 40];
        let mut a = vec![0u32; 4];
        let mut b = vec![0u32; 4];
        downsample_row(&input, &mut a, 25, 12345);
        downsample_row(&input, &mut b, 25, 12345);
        assert_eq!(a, b);
    }

    #[test]
    fn downsample_row_single_element_caps() {
        let input = vec![100u32];
        let mut output = vec![0u32; 1];
        downsample_row(&input, &mut output, 30, 1);
        assert_eq!(output, vec![30]);

        let mut output2 = vec![0u32; 1];
        downsample_row(&input, &mut output2, 200, 1);
        assert_eq!(output2, vec![100]); // capped at the available count
    }

    #[test]
    fn downsample_row_empty_is_noop() {
        let input: Vec<u32> = vec![];
        let mut output: Vec<u32> = vec![];
        downsample_row(&input, &mut output, 100, 1);
        assert!(output.is_empty());
    }

    #[test]
    fn downsample_pile_preserves_sparsity_and_is_deterministic() {
        // Two cells, three genes. Cell 0: [10, 20, 30]; cell 1: [40, 50, 60].
        let raw = CompressedSparseData2 {
            data: vec![10u32, 20, 30, 40, 50, 60],
            indices: vec![0, 1, 2, 0, 1, 2],
            indptr: vec![0, 3, 6],
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (2, 3),
        };
        let umis = vec![60.0f32, 150.0];

        let make_pile = || Pile {
            cell_indices: vec![0, 1],
            raw: raw.clone(),
            umis_per_cell: umis.clone(),
            n_genes: 3,
            downsampled: None,
            selected_gene_indices: None,
            selected_dense: None,
        };

        let mut pa = make_pile();
        let mut pb = make_pile();
        let params = SelectParams::default();
        downsample_pile(&mut pa, &params, 99);
        downsample_pile(&mut pb, &params, 99);

        let a = pa.downsampled.unwrap();
        let b = pb.downsampled.unwrap();
        // Same seed -> same data byte-for-byte (per-row seeding is deterministic).
        assert_eq!(a.data, b.data);
        // Sparsity pattern preserved: indices and indptr untouched.
        assert_eq!(a.indices, raw.indices);
        assert_eq!(a.indptr, raw.indptr);
        assert_eq!(a.shape, raw.shape);
    }
}
