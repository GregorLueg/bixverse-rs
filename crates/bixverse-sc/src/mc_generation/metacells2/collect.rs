//! Metacell profiles for the divide-and-conquer groups phase, upstream's
//! `collect_metacells(..., _metacell_groups=True)` (`metacells/pipeline/collect.py`)
//! with its default geometric mean.
//!
//! For metacell `m` with member cells `i` (library `u_i`, weight
//! `w_i = log2(u_i)`, regularisation `r_i = reg / u_i`), the fraction of gene
//! `j` is the weighted geometric mean of `u_ij / u_i + r_i` minus that of
//! `r_i`, clipped at zero and normalised to sum to one, then scaled by the
//! metacell's total UMIs. The logs are accumulated sparsely: every member
//! contributes `w_i ln(r_i)` and only stored counts add a correction.
//!
//! Upstream keeps the scaled fractions as floats and runs the next level on
//! them. This port floors them to counts, since the pile machinery is count
//! based; the groups level only decides which metacells share a final pile.

use rayon::prelude::*;

use crate::prelude::*;

/// Upstream's `metacell_umis_regularization`.
pub const METACELL_UMIS_REGULARISATION: f64 = 1.0 / 16.0;

/// Profile of one metacell from its members' rows.
///
/// ### Params
///
/// * `rows` - Members' raw counts, cell-major.
///
/// ### Returns
///
/// `(gene, count)` pairs, genes ascending, counts floored and positive.
fn metacell_profile(rows: &CompressedSparseData2<u32, f32>) -> Vec<(u32, u32)> {
    let n_cells = rows.shape.0;
    let umis: Vec<f64> = (0..n_cells)
        .map(|i| {
            rows.data[rows.indptr[i] as usize..rows.indptr[i + 1] as usize]
                .iter()
                .map(|&v| v as f64)
                .sum()
        })
        .collect();
    let total_umis: u64 = umis.iter().map(|&u| u as u64).sum();

    let mut entries: Vec<(u32, f64, u32)> = Vec::new();
    let mut weight_sum = 0.0f64;
    let mut base = 0.0f64;
    for i in 0..n_cells {
        if umis[i] <= 0.0 {
            continue;
        }
        let w = umis[i].log2();
        let r = METACELL_UMIS_REGULARISATION / umis[i];
        weight_sum += w;
        base += w * r.ln();
        for idx in rows.indptr[i] as usize..rows.indptr[i + 1] as usize {
            let u = rows.data[idx];
            if u > 0 {
                let correction = w * ((u as f64 / umis[i] + r).ln() - r.ln());
                entries.push((rows.indices[idx], correction, u));
            }
        }
    }
    entries.sort_unstable_by_key(|e| e.0);

    // per gene: (gene, geomean excess over the regularisation)
    let mut fractions: Vec<(u32, f64)> = Vec::new();
    for group in entries.chunk_by(|a, b| a.0 == b.0) {
        let correction: f64 = group.iter().map(|e| e.1).sum();
        let fraction = if weight_sum > 0.0 {
            ((base + correction) / weight_sum).exp() - (base / weight_sum).exp()
        } else {
            // every member has a single UMI: no log weight, so plain sums
            group.iter().map(|e| e.2 as f64).sum::<f64>()
        };
        if fraction > 0.0 {
            fractions.push((group[0].0, fraction));
        }
    }
    let norm: f64 = fractions.iter().map(|f| f.1).sum();
    if norm <= 0.0 {
        return Vec::new();
    }
    fractions
        .into_iter()
        .filter_map(|(g, f)| {
            // upstream rounds the fraction to f32 before scaling
            let count = ((f / norm) as f32 * total_umis as f32).floor();
            (count >= 1.0).then_some((g, count as u32))
        })
        .collect()
}

/// Build the metacell × gene profile matrix of the groups phase.
///
/// ### Params
///
/// * `load_rows` - Loads the raw rows of a set of elements, in order.
/// * `members_of_metacell` - Element indices per metacell.
/// * `n_genes` - Gene count.
/// * `max_cells_per_load` - Upper bound on cells read at once.
///
/// ### Returns
///
/// `(profiles, sizes)`: the CSR profiles, one row per metacell, and the
/// member count per metacell (the next level's mass).
pub(super) fn collect_metacell_profiles<F>(
    load_rows: F,
    members_of_metacell: &[Vec<usize>],
    n_genes: usize,
    max_cells_per_load: usize,
) -> Result<(CompressedSparseData2<u32, f32>, Vec<f32>), ScErrors>
where
    F: Fn(&[usize]) -> Result<CompressedSparseData2<u32, f32>, ScErrors>,
{
    let mut profiles: Vec<Vec<(u32, u32)>> = Vec::with_capacity(members_of_metacell.len());
    let mut start = 0;
    while start < members_of_metacell.len() {
        // batch whole metacells up to the load budget
        let mut end = start;
        let mut cells = 0;
        while end < members_of_metacell.len()
            && (end == start || cells + members_of_metacell[end].len() <= max_cells_per_load)
        {
            cells += members_of_metacell[end].len();
            end += 1;
        }
        let batch = &members_of_metacell[start..end];
        let elements: Vec<usize> = batch.iter().flatten().copied().collect();
        let rows = load_rows(&elements)?;

        let mut offsets = Vec::with_capacity(batch.len() + 1);
        offsets.push(0usize);
        for members in batch {
            offsets.push(offsets.last().unwrap() + members.len());
        }
        let batch_profiles: Vec<Vec<(u32, u32)>> = (0..batch.len())
            .into_par_iter()
            .map(|m| {
                let local: Vec<usize> = (offsets[m]..offsets[m + 1]).collect();
                rows.slice_rows(&local).map(|r| metacell_profile(&r))
            })
            .collect::<Result<_, _>>()?;
        profiles.extend(batch_profiles);
        start = end;
    }

    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0u32];
    for profile in profiles {
        for (g, c) in profile {
            indices.push(g);
            data.push(c);
        }
        indptr.push(data.len() as u32);
    }
    let sizes = members_of_metacell.iter().map(|m| m.len() as f32).collect();
    Ok((
        CompressedSparseData2 {
            data,
            indices,
            indptr,
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (members_of_metacell.len(), n_genes),
        },
        sizes,
    ))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(rows: &[&[(u32, u32)]], n_genes: usize) -> CompressedSparseData2<u32, f32> {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for row in rows {
            for &(g, v) in *row {
                indices.push(g);
                data.push(v);
            }
            indptr.push(data.len() as u32);
        }
        CompressedSparseData2 {
            data,
            indices,
            indptr,
            cs_type: CompressedSparseFormat::Csr,
            data_2: None,
            shape: (rows.len(), n_genes),
        }
    }

    /// Identical members reproduce their own profile, up to flooring.
    #[test]
    fn test_metacell_profile_identical_cells() {
        let row: &[(u32, u32)] = &[(0, 60), (2, 30), (3, 10)];
        let rows = csr(&[row, row, row], 4);
        let profile = metacell_profile(&rows);
        let genes: Vec<u32> = profile.iter().map(|p| p.0).collect();
        assert_eq!(genes, vec![0, 2, 3]);
        let total: u32 = profile.iter().map(|p| p.1).sum();
        // floors lose at most one count per gene
        assert!((297..=300).contains(&total), "total {total}");
        assert!(profile[0].1 > profile[1].1 && profile[1].1 > profile[2].1);
    }

    /// Profiles come out one row per metacell regardless of batching.
    #[test]
    fn test_collect_batches_metacells() {
        let rows = csr(
            &[
                &[(0, 5)],
                &[(0, 5)],
                &[(1, 7)],
                &[(1, 7)],
                &[(0, 3), (1, 3)],
            ],
            2,
        );
        let members = vec![vec![0, 1], vec![2, 3], vec![4]];
        let load = |e: &[usize]| Ok(rows.slice_rows(e)?);
        let (a, sizes) = collect_metacell_profiles(load, &members, 2, 2).unwrap();
        let (b, _) = collect_metacell_profiles(load, &members, 2, 100).unwrap();
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.data, b.data);
        assert_eq!(a.shape, (3, 2));
        assert_eq!(sizes, vec![2.0, 2.0, 1.0]);
    }
}
