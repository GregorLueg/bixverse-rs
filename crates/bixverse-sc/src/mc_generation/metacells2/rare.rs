//! Rare gene modules, upstream's `find_rare_gene_modules`
//! (`metacells/tools/rare.py`).
//!
//! Genes expressed in very few cells but strongly correlated with each other
//! mark rare cell states that a global pass would discount. The steps follow
//! upstream: pick sparse candidates, correlate them (by default correlations of
//! correlations), cluster them hierarchically, keep the tree nodes whose mean
//! similarity is high enough, extend each module with related genes, then give
//! each cell the module it expresses most strongly.
//!
//! Cells are streamed in batches of [`CELL_BATCH_SIZE`]: three passes over all
//! cells (per-gene statistics, candidate entries, module totals) plus a read of
//! each module's expressing cells, so memory is bounded by the batch, the
//! candidate entries (sparse by construction) and a few per-cell vectors.
//!
//! Similarities are rounded to `f32` after each correlation, as upstream's
//! correlation kernels return `float32`.

use rayon::prelude::*;

use bixverse_core::ml::clustering::hierarchical::{condensed_euclidean, linkage};

use crate::prelude::*;

use super::params::{RareParams, RareSimilarityMethod};

///////////
// Types //
///////////

/// Output of [`find_rare_gene_modules`], upstream's `rare_gene_module`,
/// `rare_gene` and `cells_rare_gene_module` annotations.
#[derive(Clone, Debug)]
pub struct RareGeneModules {
    /// Module per cell, `-1` for cells in none.
    pub module_of_cell: Vec<i32>,
    /// Module per gene, `-1` for genes in none. A related gene listed in two
    /// modules takes the later one, as upstream.
    pub module_of_gene: Vec<i32>,
    /// True for genes in any module.
    pub rare_gene_mask: Vec<bool>,
    /// Number of modules; IDs are dense in `[0, n_modules)`.
    pub n_modules: usize,
}

/// Per-gene statistics over all cells.
struct GeneStats {
    /// Cells with a non-zero count.
    nnz: Vec<u64>,
    /// Largest count.
    max: Vec<f32>,
    /// Total count.
    total: Vec<f64>,
    /// Cells, zeros included, with a count of at least `min_cell_module_total`.
    at_least: Vec<u64>,
}

/// Candidate genes and their non-zero entries per cell.
struct Candidates {
    /// Gene index per candidate, ascending.
    genes: Vec<usize>,
    /// Row pointers into `index` and `value`, one row per cell.
    indptr: Vec<usize>,
    /// Candidate index of each entry.
    index: Vec<u32>,
    /// Count of each entry.
    value: Vec<f32>,
}

/////////////
// Helpers //
/////////////

/// Read all cells in batches of [`CELL_BATCH_SIZE`] and hand each batch over.
///
/// ### Params
///
/// * `rows` - Raw cell-major rows of the given cells.
/// * `n_cells` - Cells in the source.
/// * `f` - Called with the first cell of the batch and its rows.
///
/// ### Returns
///
/// `Ok` once every batch is processed.
fn for_each_batch<R, F>(rows: &R, n_cells: usize, mut f: F) -> Result<(), ScErrors>
where
    R: Fn(&[usize]) -> Result<CompressedSparseData2<u32, f32>, ScErrors>,
    F: FnMut(usize, &CompressedSparseData2<u32, f32>) -> Result<(), ScErrors>,
{
    for start in (0..n_cells).step_by(CELL_BATCH_SIZE) {
        let cells: Vec<usize> = (start..(start + CELL_BATCH_SIZE).min(n_cells)).collect();
        f(start, &rows(&cells)?)?;
    }
    Ok(())
}

/// Entries of one CSR row.
///
/// ### Params
///
/// * `csr` - The matrix.
/// * `row` - Row index.
///
/// ### Returns
///
/// The column indices and values of the row.
#[inline]
fn row_of(csr: &CompressedSparseData2<u32, f32>, row: usize) -> (&[u32], &[u32]) {
    let (a, b) = (csr.indptr[row] as usize, csr.indptr[row + 1] as usize);
    (&csr.indices[a..b], &csr.data[a..b])
}

/// Per-gene statistics in one streaming pass.
///
/// ### Params
///
/// * `rows` - Raw cell-major rows of the given cells.
/// * `n_cells` - Cells in the source.
/// * `n_genes` - Genes in the source.
/// * `threshold` - `min_cell_module_total`.
///
/// ### Returns
///
/// The [`GeneStats`].
fn gene_stats<R>(
    rows: &R,
    n_cells: usize,
    n_genes: usize,
    threshold: f32,
) -> Result<GeneStats, ScErrors>
where
    R: Fn(&[usize]) -> Result<CompressedSparseData2<u32, f32>, ScErrors>,
{
    let new = || GeneStats {
        nnz: vec![0; n_genes],
        max: vec![0.0; n_genes],
        total: vec![0.0; n_genes],
        at_least: vec![0; n_genes],
    };
    let merge = |mut a: GeneStats, b: GeneStats| {
        for g in 0..n_genes {
            a.nnz[g] += b.nnz[g];
            a.max[g] = a.max[g].max(b.max[g]);
            a.total[g] += b.total[g];
            a.at_least[g] += b.at_least[g];
        }
        a
    };

    let mut stats = new();
    for_each_batch(rows, n_cells, |_, csr| {
        let n_rows = csr.shape.0;
        let chunk = n_rows.div_ceil(rayon::current_num_threads()).max(1);
        let batch = (0..n_rows)
            .into_par_iter()
            .step_by(chunk)
            .map(|start| {
                let mut s = new();
                for row in start..(start + chunk).min(n_rows) {
                    let (indices, data) = row_of(csr, row);
                    for (&g, &v) in indices.iter().zip(data) {
                        if v == 0 {
                            continue;
                        }
                        let (g, v) = (g as usize, v as f32);
                        s.nnz[g] += 1;
                        s.max[g] = s.max[g].max(v);
                        s.total[g] += v as f64;
                        if v >= threshold {
                            s.at_least[g] += 1;
                        }
                    }
                }
                s
            })
            .reduce(new, merge);
        stats = merge(std::mem::replace(&mut stats, new()), batch);
        Ok(())
    })?;

    // zeros count towards `at_least` when the threshold admits them
    if threshold <= 0.0 {
        for g in 0..n_genes {
            stats.at_least[g] += n_cells as u64 - stats.nnz[g];
        }
    }
    Ok(stats)
}

/// Upstream's `_pick_candidates`: sparse genes with a high enough maximum.
///
/// Beyond `max_genes`, the genes in the fewest cells are kept. Upstream takes
/// them with `argpartition`, whose choice among genes tied at the cut is
/// arbitrary; here the lower gene index wins.
///
/// ### Params
///
/// * `stats` - Per-gene statistics.
/// * `n_cells` - Cells in the source.
/// * `allowed` - Genes that may be picked.
/// * `params` - Rare module parameters.
///
/// ### Returns
///
/// Candidate genes, ascending.
fn pick_candidates(
    stats: &GeneStats,
    n_cells: usize,
    allowed: &[bool],
    params: &RareParams,
) -> Vec<usize> {
    let fraction = |g: usize| stats.nnz[g] as f64 / n_cells as f64;
    let mut genes: Vec<usize> = (0..stats.nnz.len())
        .filter(|&g| {
            allowed[g]
                && fraction(g) <= params.max_gene_cell_fraction
                && stats.max[g] >= params.min_gene_maximum as f32
        })
        .collect();
    if genes.len() > params.max_genes {
        genes.sort_by_key(|&g| (stats.nnz[g], g));
        genes.truncate(params.max_genes);
        genes.sort_unstable();
    }
    genes
}

/// Stream the candidates' non-zero entries of every cell.
///
/// ### Params
///
/// * `rows` - Raw cell-major rows of the given cells.
/// * `n_cells` - Cells in the source.
/// * `n_genes` - Genes in the source.
/// * `genes` - Candidate genes, ascending.
///
/// ### Returns
///
/// The [`Candidates`].
fn collect_candidates<R>(
    rows: &R,
    n_cells: usize,
    n_genes: usize,
    genes: Vec<usize>,
) -> Result<Candidates, ScErrors>
where
    R: Fn(&[usize]) -> Result<CompressedSparseData2<u32, f32>, ScErrors>,
{
    let mut slot = vec![u32::MAX; n_genes];
    for (c, &g) in genes.iter().enumerate() {
        slot[g] = c as u32;
    }
    let mut out = Candidates {
        genes,
        indptr: Vec::with_capacity(n_cells + 1),
        index: Vec::new(),
        value: Vec::new(),
    };
    out.indptr.push(0);
    for_each_batch(rows, n_cells, |_, csr| {
        let per_row: Vec<Vec<(u32, f32)>> = (0..csr.shape.0)
            .into_par_iter()
            .map(|row| {
                let (indices, data) = row_of(csr, row);
                indices
                    .iter()
                    .zip(data)
                    .filter(|&(&g, &v)| v > 0 && slot[g as usize] != u32::MAX)
                    .map(|(&g, &v)| (slot[g as usize], v as f32))
                    .collect()
            })
            .collect();
        for entries in per_row {
            for (c, v) in entries {
                out.index.push(c);
                out.value.push(v);
            }
            out.indptr.push(out.index.len());
        }
        Ok(())
    })?;
    Ok(out)
}

/// Pearson correlation from sums, as upstream's reproducible kernel
/// (`correlate.cpp`): zero when either side has no variance, clipped to
/// `[-1, 1]` and rounded to `f32`.
///
/// ### Params
///
/// * `n` - Elements per vector.
/// * `xy` - Sum of the products.
/// * `x` - Sum of the first vector.
/// * `xx` - Sum of its squares.
/// * `y` - Sum of the second vector.
/// * `yy` - Sum of its squares.
///
/// ### Returns
///
/// The correlation.
#[inline]
fn correlation_from_sums(n: f64, xy: f64, x: f64, xx: f64, y: f64, yy: f64) -> f32 {
    let numerator = n * xy - x * y;
    let factors = ((n * xx - x * x) * (n * yy - y * y)).sqrt();
    if factors != 0.0 {
        ((numerator / factors) as f32).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Correlation between candidate genes over all cells, from the sparse
/// entries: the sums are exact for integer counts, so no dense matrix is
/// needed.
///
/// ### Params
///
/// * `cand` - Candidate entries.
/// * `n_cells` - Cells in the source.
///
/// ### Returns
///
/// The `c × c` correlation matrix, row-major, diagonal `1`.
fn candidate_correlation(cand: &Candidates, n_cells: usize) -> Vec<f32> {
    let c = cand.genes.len();
    let mut sum = vec![0.0f64; c];
    let mut sum_sq = vec![0.0f64; c];
    for (&i, &v) in cand.index.iter().zip(&cand.value) {
        let v = v as f64;
        sum[i as usize] += v;
        sum_sq[i as usize] += v * v;
    }

    let chunk = n_cells.div_ceil(rayon::current_num_threads()).max(1);
    let cross = (0..n_cells)
        .into_par_iter()
        .step_by(chunk)
        .map(|start| {
            let mut m = vec![0.0f64; c * c];
            for cell in start..(start + chunk).min(n_cells) {
                let (a, b) = (cand.indptr[cell], cand.indptr[cell + 1]);
                for p in a..b {
                    let (i, vi) = (cand.index[p] as usize, cand.value[p] as f64);
                    for q in p + 1..b {
                        let (j, vj) = (cand.index[q] as usize, cand.value[q] as f64);
                        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                        m[lo * c + hi] += vi * vj;
                    }
                }
            }
            m
        })
        .reduce(
            || vec![0.0f64; c * c],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );

    let n = n_cells as f64;
    let mut out = vec![0.0f32; c * c];
    for i in 0..c {
        out[i * c + i] = 1.0;
        for j in i + 1..c {
            let r =
                correlation_from_sums(n, cross[i * c + j], sum[i], sum_sq[i], sum[j], sum_sq[j]);
            out[i * c + j] = r;
            out[j * c + i] = r;
        }
    }
    out
}

/// Correlation between the rows of a dense square matrix, upstream's
/// `corrcoef(similarity, per=None)` for the repeated methods.
///
/// ### Params
///
/// * `s` - Row-major `c × c` matrix.
/// * `c` - Its order.
///
/// ### Returns
///
/// The `c × c` correlation of its rows, diagonal `1`.
fn row_correlation(s: &[f32], c: usize) -> Vec<f32> {
    let sums: Vec<(f64, f64)> = s
        .par_chunks_exact(c)
        .map(|row| {
            row.iter().fold((0.0f64, 0.0f64), |(a, b), &v| {
                let v = v as f64;
                (a + v, v.mul_add(v, b))
            })
        })
        .collect();
    let n = c as f64;
    let upper: Vec<Vec<f32>> = (0..c)
        .into_par_iter()
        .map(|i| {
            let ri = &s[i * c..(i + 1) * c];
            (i + 1..c)
                .map(|j| {
                    let rj = &s[j * c..(j + 1) * c];
                    let xy = ri
                        .iter()
                        .zip(rj)
                        .fold(0.0f64, |acc, (&a, &b)| (a as f64).mul_add(b as f64, acc));
                    correlation_from_sums(n, xy, sums[i].0, sums[i].1, sums[j].0, sums[j].1)
                })
                .collect()
        })
        .collect();
    let mut out = vec![0.0f32; c * c];
    for (i, row) in upper.iter().enumerate() {
        out[i * c + i] = 1.0;
        for (k, &r) in row.iter().enumerate() {
            let j = i + 1 + k;
            out[i * c + j] = r;
            out[j * c + i] = r;
        }
    }
    out
}

/// Upstream's `_genes_similarity` via `compute_var_var_similarity`.
///
/// ### Params
///
/// * `cand` - Candidate entries.
/// * `n_cells` - Cells in the source.
/// * `method` - Similarity method.
///
/// ### Returns
///
/// The `c × c` similarity, row-major.
fn genes_similarity(cand: &Candidates, n_cells: usize, method: RareSimilarityMethod) -> Vec<f32> {
    let c = cand.genes.len();
    let abs = |m: &mut Vec<f32>| m.iter_mut().for_each(|v| *v = v.abs());
    let mut s = candidate_correlation(cand, n_cells);
    match method {
        RareSimilarityMethod::Pearson => {}
        RareSimilarityMethod::AbsPearson => abs(&mut s),
        RareSimilarityMethod::RepeatedPearson => s = row_correlation(&s, c),
        RareSimilarityMethod::RepeatedAbsPearson => {
            abs(&mut s);
            s = row_correlation(&s, c);
            abs(&mut s);
        }
    }
    s
}

/// Upstream's `_cluster_genes` and `_identify_genes`: walk the merges in
/// order, accepting a merge of two live nodes when the mean off-diagonal
/// similarity of their union reaches `min_module_correlation`.
///
/// ### Params
///
/// * `similarity` - The `c × c` similarity, row-major.
/// * `genes` - Candidate genes.
/// * `params` - Rare module parameters.
///
/// ### Returns
///
/// The surviving nodes, singletons first then merges in tree order, as gene
/// indices ascending within each node.
///
/// ### Errors
///
/// Linkage errors.
fn identify_genes(
    similarity: &[f32],
    genes: &[usize],
    params: &RareParams,
) -> Result<Vec<Vec<usize>>, ScErrors> {
    let c = genes.len();
    let mut nodes: Vec<Option<Vec<usize>>> = (0..c).map(|i| Some(vec![i])).collect();
    if c >= 2 {
        let as_f64: Vec<f64> = similarity.iter().map(|&v| v as f64).collect();
        let steps = linkage(&condensed_euclidean(&as_f64, c), c, params.cluster_method)?;
        nodes.resize(2 * c - 1, None);
        for (k, step) in steps.iter().enumerate() {
            let (Some(left), Some(right)) = (&nodes[step.left], &nodes[step.right]) else {
                continue;
            };
            let mut merged: Vec<usize> = left.iter().chain(right).copied().collect();
            merged.sort_unstable();
            let m = merged.len();
            let total: f64 = merged
                .iter()
                .map(|&i| {
                    merged
                        .iter()
                        .filter(|&&j| j != i)
                        .map(|&j| similarity[i * c + j] as f64)
                        .sum::<f64>()
                })
                .sum();
            if total / ((m * m - m) as f64) < params.min_module_correlation {
                continue;
            }
            nodes[c + k] = Some(merged);
            nodes[step.left] = None;
            nodes[step.right] = None;
        }
    }
    Ok(nodes
        .into_iter()
        .flatten()
        .map(|node| node.into_iter().map(|i| genes[i]).collect())
        .collect())
}

/// Module UMI total per cell from the candidate entries.
///
/// ### Params
///
/// * `cand` - Candidate entries.
/// * `in_module` - Per candidate, whether it is in the module.
///
/// ### Returns
///
/// The total per cell.
fn module_totals(cand: &Candidates, in_module: &[bool]) -> Vec<f32> {
    (0..cand.indptr.len() - 1)
        .into_par_iter()
        .map(|cell| {
            (cand.indptr[cell]..cand.indptr[cell + 1])
                .filter(|&p| in_module[cand.index[p] as usize])
                .map(|p| cand.value[p])
                .sum()
        })
        .collect()
}

/// Upstream's `_related_genes`: drop modules with too few or too many
/// expressing cells and extend the others with related genes.
///
/// ### Params
///
/// * `rows` - Raw cell-major rows of the given cells.
/// * `stats` - Per-gene statistics.
/// * `cand` - Candidate entries.
/// * `modules` - Gene lists from [`identify_genes`].
/// * `allowed` - Genes that may be related.
/// * `max_cells` - Upstream's `max_cells_of_modules`.
/// * `params` - Rare module parameters.
///
/// ### Returns
///
/// The surviving modules with their related genes appended, ascending.
fn related_genes<R>(
    rows: &R,
    stats: &GeneStats,
    cand: &Candidates,
    modules: &[Vec<usize>],
    allowed: &[bool],
    max_cells: usize,
    params: &RareParams,
) -> Result<Vec<Vec<usize>>, ScErrors>
where
    R: Fn(&[usize]) -> Result<CompressedSparseData2<u32, f32>, ScErrors>,
{
    let n_genes = stats.nnz.len();
    let threshold = params.min_cell_module_total as f32;
    let fold = 2f64.powf(params.min_related_gene_fold_factor);
    let mut slot = vec![usize::MAX; n_genes];
    for (c, &g) in cand.genes.iter().enumerate() {
        slot[g] = c;
    }
    let mut in_any = vec![false; n_genes];
    for module in modules
        .iter()
        .filter(|m| m.len() >= params.min_genes_of_modules)
    {
        module.iter().for_each(|&g| in_any[g] = true);
    }

    let mut out = Vec::new();
    for module in modules
        .iter()
        .filter(|m| m.len() >= params.min_genes_of_modules)
    {
        let mut in_module = vec![false; cand.genes.len()];
        module.iter().for_each(|&g| in_module[slot[g]] = true);
        let totals = module_totals(cand, &in_module);
        let expressed: Vec<usize> = (0..totals.len()).filter(|&i| totals[i] > 0.0).collect();
        if expressed.len() > max_cells || expressed.len() < params.min_cells_of_modules {
            continue;
        }

        let csr = rows(&expressed)?;
        let mut max_expressed = vec![0.0f32; n_genes];
        let mut total_expressed = vec![0.0f64; n_genes];
        for row in 0..csr.shape.0 {
            let (indices, data) = row_of(&csr, row);
            for (&g, &v) in indices.iter().zip(data) {
                let g = g as usize;
                max_expressed[g] = max_expressed[g].max(v as f32);
                total_expressed[g] += v as f64;
            }
        }
        let sum_expressed: f64 = total_expressed.iter().sum();
        let sum_background: f64 = (0..n_genes)
            .map(|g| stats.total[g] - total_expressed[g])
            .sum();
        let is_related = |g: usize| {
            let expressed_fraction = total_expressed[g] / sum_expressed;
            let background_fraction = (stats.total[g] - total_expressed[g]) / sum_background;
            allowed[g]
                && max_expressed[g] >= params.min_gene_maximum as f32
                && expressed_fraction >= background_fraction * fold
        };

        let strong_base = totals.iter().filter(|&&t| t >= threshold).count() as f64;
        let mut extended = module.clone();
        let mut in_this = vec![false; n_genes];
        module.iter().for_each(|&g| in_this[g] = true);
        let candidates: Vec<usize> = (0..n_genes)
            .filter(|&g| !in_this[g] && !in_any[g] && is_related(g))
            .collect();
        for g in candidates {
            // over all cells: cells outside `expressed` have a module total
            // of zero, so only the expressed ones change against `at_least`
            let mut values = vec![0.0f32; expressed.len()];
            for (pos, value) in values.iter_mut().enumerate() {
                let (indices, data) = row_of(&csr, pos);
                if let Some(k) = indices.iter().position(|&i| i as usize == g) {
                    *value = data[k] as f32;
                }
            }
            let mut strong = stats.at_least[g] as i64;
            for (pos, &v) in values.iter().enumerate() {
                strong -= i64::from(v >= threshold);
                strong += i64::from(v + totals[expressed[pos]] >= threshold);
            }
            if strong as f64 > params.max_related_gene_increase_factor * strong_base {
                continue;
            }
            extended.push(g);
        }
        out.push(extended);
    }
    Ok(out)
}

/// Upstream's `_identify_cells` and `_compress_modules`: assign each cell the
/// module it expresses most strongly relative to the module's median, then
/// drop modules with too few or too many cells and renumber the rest.
///
/// ### Params
///
/// * `rows` - Raw cell-major rows of the given cells.
/// * `n_cells` - Cells in the source.
/// * `n_genes` - Genes in the source.
/// * `modules` - Modules from [`related_genes`].
/// * `max_cells` - Upstream's `max_cells_of_modules`.
/// * `params` - Rare module parameters.
///
/// ### Returns
///
/// The module per cell and the kept modules' genes.
fn identify_cells<R>(
    rows: &R,
    n_cells: usize,
    n_genes: usize,
    modules: &[Vec<usize>],
    max_cells: usize,
    params: &RareParams,
) -> Result<(Vec<i32>, Vec<Vec<usize>>), ScErrors>
where
    R: Fn(&[usize]) -> Result<CompressedSparseData2<u32, f32>, ScErrors>,
{
    let n_modules = modules.len();
    let threshold = params.min_cell_module_total as f32;
    let mut modules_of_gene: Vec<Vec<usize>> = vec![Vec::new(); n_genes];
    for (m, genes) in modules.iter().enumerate() {
        genes.iter().for_each(|&g| modules_of_gene[g].push(m));
    }

    // totals per cell, row-major `n_cells × n_modules`
    let mut totals = vec![0.0f32; n_cells * n_modules];
    for_each_batch(rows, n_cells, |start, csr| {
        let n_rows = csr.shape.0;
        totals[start * n_modules..(start + n_rows) * n_modules]
            .par_chunks_exact_mut(n_modules)
            .enumerate()
            .for_each(|(row, out)| {
                let (indices, data) = row_of(csr, row);
                for (&g, &v) in indices.iter().zip(data) {
                    for &m in &modules_of_gene[g as usize] {
                        out[m] += v as f32;
                    }
                }
            });
        Ok(())
    })?;

    let mut module_of_cell = vec![-1i32; n_cells];
    let mut max_strength = vec![0.0f32; n_cells];
    let mut kept = vec![true; n_modules];
    for m in 0..n_modules {
        let total = |cell: usize| totals[cell * n_modules + m];
        let strong: Vec<usize> = (0..n_cells).filter(|&c| total(c) >= threshold).collect();
        if strong.len() > max_cells || strong.len() < params.min_cells_of_modules {
            kept[m] = false;
            continue;
        }
        let mut sorted: Vec<f32> = strong.iter().map(|&c| total(c)).collect();
        sorted.sort_unstable_by(f32::total_cmp);
        let half = sorted.len() / 2;
        let median = if sorted.len() % 2 == 1 {
            sorted[half]
        } else {
            (sorted[half - 1] + sorted[half]) / 2.0
        };
        for &c in &strong {
            let strength = total(c) / median;
            if strength >= max_strength[c] {
                max_strength[c] = strength;
                module_of_cell[c] = m as i32;
            }
        }
    }

    let mut cells_of = vec![0usize; n_modules];
    for &m in module_of_cell.iter().filter(|&&m| m >= 0) {
        cells_of[m as usize] += 1;
    }
    let mut renumber = vec![-1i32; n_modules];
    let mut genes = Vec::new();
    for m in (0..n_modules).filter(|&m| kept[m]) {
        if cells_of[m] < params.min_cells_of_modules || cells_of[m] > max_cells {
            continue;
        }
        renumber[m] = genes.len() as i32;
        genes.push(modules[m].clone());
    }
    for m in module_of_cell.iter_mut().filter(|m| **m >= 0) {
        *m = renumber[*m as usize];
    }
    Ok((module_of_cell, genes))
}

//////////
// Main //
//////////

/// Detect rare gene modules and the cells expressing them, upstream's
/// `find_rare_gene_modules`.
///
/// ### Params
///
/// * `rows` - Raw cell-major counts of the given cells, `0..n_cells` in the
///   caller's cell space. Called once per batch of [`CELL_BATCH_SIZE`] cells
///   per pass and once per module for its expressing cells.
/// * `n_cells` - Cells in the source.
/// * `n_genes` - Genes in the source.
/// * `lateral_gene_mask` - Optional genes never picked as candidates or
///   related genes.
/// * `target_metacell_size` - Target cells per metacell.
/// * `target_pile_size` - Target cells per pile; with the previous argument
///   it bounds how many cells a module may have.
/// * `params` - Rare module parameters.
/// * `verbose` - Print the modules found.
///
/// ### Returns
///
/// The [`RareGeneModules`].
///
/// ### Errors
///
/// `Mc2InvalidParams` for zero minimum module sizes or a zero target pile
/// size, `Mc2LengthMismatch` for a lateral mask of the wrong length, and
/// errors from `rows`.
#[allow(clippy::too_many_arguments)]
pub fn find_rare_gene_modules<R>(
    rows: R,
    n_cells: usize,
    n_genes: usize,
    lateral_gene_mask: Option<&[bool]>,
    target_metacell_size: usize,
    target_pile_size: usize,
    params: &RareParams,
    verbose: bool,
) -> Result<RareGeneModules, ScErrors>
where
    R: Fn(&[usize]) -> Result<CompressedSparseData2<u32, f32>, ScErrors>,
{
    if params.min_cells_of_modules == 0 || params.min_genes_of_modules == 0 {
        return Err(ScErrors::Mc2InvalidParams {
            reason: "rare min_cells_of_modules and min_genes_of_modules must be positive".into(),
        });
    }
    if target_pile_size == 0 {
        return Err(ScErrors::Mc2InvalidParams {
            reason: "target_pile_size must be positive".into(),
        });
    }
    let allowed: Vec<bool> = match lateral_gene_mask {
        Some(m) if m.len() != n_genes => {
            return Err(ScErrors::Mc2LengthMismatch {
                what: "lateral_gene_mask",
                expected: n_genes,
                got: m.len(),
            });
        }
        Some(m) => m.iter().map(|&l| !l).collect(),
        None => vec![true; n_genes],
    };

    let mut result = RareGeneModules {
        module_of_cell: vec![-1; n_cells],
        module_of_gene: vec![-1; n_genes],
        rare_gene_mask: vec![false; n_genes],
        n_modules: 0,
    };

    let stats = gene_stats(&rows, n_cells, n_genes, params.min_cell_module_total as f32)?;
    let genes = pick_candidates(&stats, n_cells, &allowed, params);
    if genes.len() < params.min_genes_of_modules {
        return Ok(result);
    }
    let cand = collect_candidates(&rows, n_cells, n_genes, genes)?;
    let similarity = genes_similarity(&cand, n_cells, params.similarity);
    let modules = identify_genes(&similarity, &cand.genes, params)?;

    let max_cells_of_random_pile =
        target_metacell_size as f64 * params.max_cells_factor_of_random_pile;
    let max_cells = (max_cells_of_random_pile * n_cells as f64 / target_pile_size as f64) as usize;
    let modules = related_genes(&rows, &stats, &cand, &modules, &allowed, max_cells, params)?;
    let (module_of_cell, modules) =
        identify_cells(&rows, n_cells, n_genes, &modules, max_cells, params)?;

    for (m, genes) in modules.iter().enumerate() {
        for &g in genes {
            result.module_of_gene[g] = m as i32;
            result.rare_gene_mask[g] = true;
        }
    }
    result.module_of_cell = module_of_cell;
    result.n_modules = modules.len();
    if verbose {
        println!(
            "Rare gene modules: {} candidates, {} modules, {} cells",
            cand.genes.len(),
            result.n_modules,
            result.module_of_cell.iter().filter(|&&m| m >= 0).count()
        );
    }
    Ok(result)
}
