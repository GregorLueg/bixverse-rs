//! DIALOGUE stage three: meta-analysis across partners, and the refit.
//!
//! Stage two tested each gene once per partner cell type. This stage combines
//! those verdicts, then rebuilds each programme's score from the genes that
//! survived rather than from the canonical weights, by non-negative least
//! squares staged over how broadly a gene is supported.
//!
//! ### Nothing dense of size cells by genes is built
//!
//! Upstream materialises the gene-z-scored expression as a cells by genes
//! matrix and hands it to `nnls`. That is the largest allocation in the whole
//! method and it is avoidable: the staged fit only ever needs `Z'Z`, `Z'y` and
//! `y'y`. Passing the residual between strata is
//! `Z'(y - Z_S b) = Z'y - (Z'Z_S) b`, and the early stop falls out too, because
//! z-scoring leaves every column summing to zero and so
//! `cor(y, fitted) = b'Z'y / sqrt(b'Z'Z b * var(y))`.
//!
//! The one thing that does not survive the reduction is upstream's
//! `length(unique(fitted)) > 10` guard, which exists only to keep `cor` from
//! being called on a constant vector. `b'Z'Z b > 0` says the same thing.
//!
//! `Z'Z` itself is built from the raw sparse values rather than the z-scored
//! dense ones. Standardising is affine, `Z = X S + 1 o'`, so
//! `Z'Z = S(X'X - n m m')S` up to the sign flips, and `X'X` can be accumulated
//! as one rank-one update per cell over that cell's non-zero genes. On droplet
//! data that is far cheaper than the dense product, and it never allocates the
//! dense block at all.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::math::linear_algebra::{nnls_gram, ols_residualise};
use crate::core::math::stats::{calc_fdr, fisher_combine, p_adjust_holm};
use crate::core::math::vector_helpers::pearson_correlation;
use crate::prelude::*;
use crate::single_cell::sc_analysis::dialogue::hlm::{DialogueStep2Result, p_from_signed_log};
use crate::single_cell::sc_analysis::dialogue::params::DialogueParams;
use crate::single_cell::sc_analysis::dialogue::pmd::{
    CellTypeView, DialogueStep1Result, ProgrammeSignature,
};

use faer::Mat;

//////////////////
// Result types //
//////////////////

/// What the meta-analysis concluded about one gene in one programme.
#[derive(Clone, Debug)]
pub struct GeneVerdict {
    /// Cell type the gene belongs to.
    pub cell_type: usize,
    /// Programme index.
    pub programme: usize,
    /// Gene index.
    pub gene: usize,
    /// Whether the gene entered from the up or the down side.
    pub up: bool,
    /// Partners supporting the gene at [crate::single_cell::sc_analysis::dialogue::params::RefineParams::support_p].
    pub n_supporting: usize,
    /// Fraction of partners supporting it.
    pub support_fraction: f64,
    /// Fisher-combined p across partners on the up side, or 1.0 for a down gene.
    pub p_up: f64,
    /// The same on the down side.
    pub p_down: f64,
    /// Non-negative coefficient from the staged refit. Zero means the refit
    /// dropped the gene.
    pub coefficient: f64,
}

/// The finished DIALOGUE output.
#[derive(Clone, Debug)]
pub struct DialogueResult {
    /// Sample codes shared by every cell type.
    pub shared_samples: Vec<usize>,
    /// Sparse canonical weights per cell type from stage one.
    pub ws: Vec<Mat<f64>>,
    /// Feature columns that survived the ANOVA filter, per cell type.
    pub kept_features: Vec<Vec<usize>>,
    /// Empirical p per programme and cell-type pair, `k x n_pairs`.
    pub emp_p: Mat<f64>,
    /// Canonical correlation per programme and pair on the real fit.
    pub pair_cor: Mat<f64>,
    /// Which cell types each programme spans.
    pub mcp_cell_types: Vec<Vec<usize>>,
    /// Final programme scores per cell type, `n_cells_in_type x k`,
    /// residualised on the quality covariate.
    pub scores: Vec<Mat<f64>>,
    /// Stage one's canonical scores, kept for comparison against the refit.
    pub cca_scores: Vec<Mat<f64>>,
    /// Correlation between the two, per cell type and programme. A low value
    /// means the gene-level refit drifted away from the programme the
    /// decomposition found.
    pub refit_fidelity: Mat<f64>,
    /// Every gene the meta-analysis reached a verdict on.
    pub verdicts: Vec<GeneVerdict>,
    /// Permissive gene lists, indexed `[cell_type][programme]`.
    pub permissive: Vec<Vec<ProgrammeSignature>>,
    /// Strict gene lists, indexed `[cell_type][programme]`.
    pub strict: Vec<Vec<ProgrammeSignature>>,
}

///////////////////
// Meta-analysis //
///////////////////

/// One gene's row in the per-cell-type meta-analysis.
#[derive(Clone, Debug)]
struct MetaRow {
    programme: usize,
    gene: usize,
    up: bool,
    /// Signed `-log10 p` per partner, `NaN` where that partner had no verdict.
    z: Vec<f64>,
}

/// Combines stage two's per-partner verdicts for one cell type.
///
/// Benjamini-Hochberg runs within each `programme x direction` stratum and
/// separately per partner, then Fisher combines across partners. Both details
/// are upstream's and both change which genes survive.
///
/// ### Params
///
/// * `rows` - One row per gene and programme
/// * `partners` - Partner cell types, in column order
/// * `cell_type` - The cell type these genes belong to, stamped onto each
///   verdict
/// * `support_p` - Adjusted p below which a partner counts as supporting
///
/// ### Returns
///
/// A verdict per row, in the same order.
fn meta_analyse(
    rows: &[MetaRow],
    partners: &[usize],
    cell_type: usize,
    support_p: f64,
) -> Vec<GeneVerdict> {
    let n_partners = partners.len();
    let n_rows = rows.len();
    let mut p_up = vec![f64::NAN; n_rows * n_partners];
    let mut p_down = vec![f64::NAN; n_rows * n_partners];

    // Strata are (programme, direction), matching upstream's `programF`.
    let mut strata: FxHashMap<(usize, bool), Vec<usize>> = FxHashMap::default();
    for (i, row) in rows.iter().enumerate() {
        strata.entry((row.programme, row.up)).or_default().push(i);
    }

    for members in strata.values() {
        for col in 0..n_partners {
            let raw_up: Vec<f64> = members
                .iter()
                .map(|&i| p_from_signed_log(rows[i].z[col], true))
                .collect();
            let raw_down: Vec<f64> = members
                .iter()
                .map(|&i| p_from_signed_log(rows[i].z[col], false))
                .collect();
            // calc_fdr has no NaN contract, so the missing verdicts are held
            // out of the adjustment and put back afterwards.
            let keep: Vec<usize> = (0..members.len())
                .filter(|&j| raw_up[j].is_finite())
                .collect();
            if keep.is_empty() {
                continue;
            }
            let up_in: Vec<f64> = keep.iter().map(|&j| raw_up[j]).collect();
            let down_in: Vec<f64> = keep.iter().map(|&j| raw_down[j]).collect();
            // Upstream's `p.adjust.mat.per.label` branches on the column
            // count: with one partner it calls bare `p.adjust`, whose default
            // method is Holm, and only with two or more does it reach the
            // Benjamini-Hochberg path. Two cell types is the commonest
            // configuration, so this branch is not an edge case.
            let (up_adj, down_adj) = if n_partners < 2 {
                (p_adjust_holm(&up_in), p_adjust_holm(&down_in))
            } else {
                (calc_fdr(&up_in), calc_fdr(&down_in))
            };
            for (slot, &j) in keep.iter().enumerate() {
                p_up[members[j] * n_partners + col] = up_adj[slot];
                p_down[members[j] * n_partners + col] = down_adj[slot];
            }
        }
    }

    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            let up_slice = &p_up[i * n_partners..(i + 1) * n_partners];
            let down_slice = &p_down[i * n_partners..(i + 1) * n_partners];
            let observed = up_slice.iter().filter(|p| p.is_finite()).count();
            let side = if row.up { up_slice } else { down_slice };
            let n_supporting = side
                .iter()
                .filter(|p| p.is_finite() && **p < support_p)
                .count();
            let support_fraction = if observed == 0 {
                0.0
            } else {
                n_supporting as f64 / observed as f64
            };
            // Upstream zeroes the irrelevant side to one so a later `p < cut`
            // can only fire on the direction the gene came in on.
            let p_up_combined = if row.up {
                fisher_combine(up_slice).unwrap_or(f64::NAN)
            } else {
                1.0
            };
            let p_down_combined = if row.up {
                1.0
            } else {
                fisher_combine(down_slice).unwrap_or(f64::NAN)
            };
            GeneVerdict {
                cell_type,
                programme: row.programme,
                gene: row.gene,
                up: row.up,
                n_supporting,
                support_fraction,
                p_up: p_up_combined,
                p_down: p_down_combined,
                coefficient: 0.0,
            }
        })
        .collect()
}

///////////////////////////
// The Gram accumulation //
///////////////////////////

/// The reduced form of one programme's gene block.
struct GeneGram {
    /// `Z'Z`, genes by genes.
    ztz: Mat<f64>,
    /// `Z'y` against the canonical score.
    zty: Vec<f64>,
    /// `y'y - (sum y)^2 / n`, the centred sum of squares of the response.
    y_centred_ss: f64,
    /// Per-gene scale, carrying the sign flip for down genes.
    scale: Vec<f64>,
    /// Per-gene offset.
    offset: Vec<f64>,
    /// Cell-major non-zeros, `(gene_slot, raw_value)` per cell.
    by_cell: Vec<Vec<(u32, f32)>>,
}

/// Reduces one programme's gene block to the quantities the refit needs.
///
/// See the module documentation for why `Z'Z` is built from the raw sparse
/// values and corrected afterwards rather than from the dense z-scored block.
///
/// ### Params
///
/// * `reader` - Gene-major reader
/// * `genes` - The programme's genes
/// * `flip` - Whether each gene enters negated, parallel to `genes`
/// * `view` - The cell type
/// * `response` - The canonical score being refitted, length `n_cells_in_type`
///
/// ### Returns
///
/// The [GeneGram], or a read error.
fn build_gram<S: SingleCellReading>(
    reader: &S,
    genes: &[usize],
    flip: &[bool],
    view: &CellTypeView,
    response: &[f64],
) -> Result<GeneGram, BixverseErrors> {
    let n_in_type = view.cells.len();
    let n_genes = genes.len();
    let total_cells = reader.get_header().total_cells;

    let position = view.position_table(total_cells);

    let chunks = reader.read_gene_parallel(genes)?;
    let mut by_cell: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_in_type];
    let mut sum_x = vec![0.0_f64; n_genes];
    let mut sum_xx = vec![0.0_f64; n_genes];
    let mut sum_xy = vec![0.0_f64; n_genes];

    for (slot, chunk) in chunks.iter().enumerate() {
        for (&cell, value) in chunk.indices.iter().zip(chunk.data_norm.iter()) {
            let row = position[cell as usize];
            if row == u32::MAX {
                continue;
            }
            let v = value.to_f32();
            let vd = v as f64;
            sum_x[slot] += vd;
            sum_xx[slot] += vd * vd;
            sum_xy[slot] += vd * response[row as usize];
            by_cell[row as usize].push((slot as u32, v));
        }
    }

    let n = n_in_type as f64;
    let mut scale = vec![0.0_f64; n_genes];
    let mut offset = vec![0.0_f64; n_genes];
    for g in 0..n_genes {
        let mean = sum_x[g] / n;
        // R's `sd`, so an n - 1 denominator.
        let var = (sum_xx[g] - n * mean * mean) / (n - 1.0);
        let sd = var.sqrt();
        let sd = if sd > 0.0 && sd.is_finite() { sd } else { 1.0 };
        let sign = if flip[g] { -1.0 } else { 1.0 };
        scale[g] = sign / sd;
        offset[g] = -sign * mean / sd;
    }

    // Debugs
    debug_assert!(
        by_cell
            .iter()
            .all(|e| e.windows(2).all(|w| w[0].0 < w[1].0)),
        "by_cell must be sorted by gene slot"
    );

    let raw = by_cell
        .par_iter()
        .fold(
            || Mat::<f64>::zeros(n_genes, n_genes),
            |mut acc, entries| {
                for (a, (ga, va)) in entries.iter().enumerate() {
                    let va = *va as f64;
                    for (gb, vb) in entries.iter().skip(a) {
                        acc[(*ga as usize, *gb as usize)] += va * (*vb as f64);
                    }
                }
                acc
            },
        )
        .reduce(
            || Mat::<f64>::zeros(n_genes, n_genes),
            |mut a, b| {
                for i in 0..n_genes {
                    for j in i..n_genes {
                        a[(i, j)] += b[(i, j)];
                    }
                }
                a
            },
        );

    let sum_y: f64 = response.iter().sum();
    let sum_yy: f64 = response.iter().map(|v| v * v).sum();

    let mut ztz = Mat::<f64>::zeros(n_genes, n_genes);
    for i in 0..n_genes {
        for j in i..n_genes {
            let cross = raw[(i, j)];
            let value = scale[i] * scale[j] * cross
                + scale[i] * offset[j] * sum_x[i]
                + scale[j] * offset[i] * sum_x[j]
                + n * offset[i] * offset[j];
            ztz[(i, j)] = value;
            ztz[(j, i)] = value;
        }
    }
    let zty: Vec<f64> = (0..n_genes)
        .map(|g| scale[g] * sum_xy[g] + offset[g] * sum_y)
        .collect();

    Ok(GeneGram {
        ztz,
        zty,
        y_centred_ss: sum_yy - sum_y * sum_y / n,
        scale,
        offset,
        by_cell,
    })
}

/// The staged non-negative refit.
///
/// Genes are grouped by how broadly they are supported and fitted in descending
/// order of support, each stage against the previous one's residual, stopping
/// once the accumulated fit tracks the original score closely enough. A final
/// catch-all stage picks up whatever the support threshold excluded.
///
/// ### Params
///
/// * `gram` - The reduced gene block
/// * `support` - Support fraction per gene
/// * `params` - Stage three parameters
///
/// ### Returns
///
/// The non-negative coefficient per gene.
fn staged_nnls(gram: &GeneGram, support: &[f64], params: &DialogueParams) -> Vec<f64> {
    let refine = &params.refine;
    let n_genes = support.len();
    let mut coef = vec![0.0_f64; n_genes];
    let mut residual = gram.zty.clone();

    // Distinct support levels, most-supported first.
    let mut levels: Vec<f64> = support.iter().copied().filter(|v| v.is_finite()).collect();
    levels.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    levels.dedup();

    // The catch-all boundary is the last level the loop *reached*, not the last
    // one it fitted. A stratum skipped for being too small still moves the
    // boundary past itself, so its genes are dropped rather than swept up at
    // the end. That is upstream's behaviour and it is load-bearing: without it
    // a thin stratum would be refitted alongside the poorly supported tail.
    let mut boundary = f64::INFINITY;
    for level in levels.iter().copied() {
        if level < refine.min_support_fraction {
            break;
        }
        boundary = level;
        let members: Vec<usize> = (0..n_genes).filter(|&g| support[g] == level).collect();
        if members.len() < refine.min_stratum {
            continue;
        }
        apply_stage(gram, &members, &mut coef, &mut residual);
        if tracks_well(gram, &coef, refine.early_stop_cor) {
            return coef;
        }
    }

    // Catch-all over the tail below the boundary. Upstream reads an undefined
    // variable here when no level clears the support threshold at all; the
    // boundary stays at infinity in that case, so everything is swept up.
    let rest: Vec<usize> = (0..n_genes)
        .filter(|&g| support[g].is_finite() && support[g] < boundary && coef[g] == 0.0)
        .collect();
    if rest.len() >= refine.min_stratum {
        apply_stage(gram, &rest, &mut coef, &mut residual);
    }
    coef
}

/// Fits one stratum against the running residual and folds it in.
///
/// ### Params
///
/// * `gram` - The reduced gene block
/// * `members` - Gene slots in this stratum
/// * `coef` - Accumulated coefficients, updated in place
/// * `residual` - `Z'` of the running residual, updated in place
fn apply_stage(gram: &GeneGram, members: &[usize], coef: &mut [f64], residual: &mut [f64]) {
    let m = members.len();
    let sub = Mat::<f64>::from_fn(m, m, |i, j| gram.ztz[(members[i], members[j])]);
    let rhs: Vec<f64> = members.iter().map(|&g| residual[g]).collect();
    let Ok(solution) = nnls_gram(sub.as_ref(), &rhs, None) else {
        return;
    };
    for (i, &g) in members.iter().enumerate() {
        coef[g] += solution[i];
    }
    // Z'(y - Z_S b) = Z'y - (Z'Z_S) b.
    for (i, r) in residual.iter_mut().enumerate() {
        let mut delta = 0.0;
        for (j, &g) in members.iter().enumerate() {
            delta += gram.ztz[(i, g)] * solution[j];
        }
        *r -= delta;
    }
}

/// Whether the accumulated fit already tracks the original score.
///
/// `cor(y, fitted) = b'Z'y / sqrt(b'Z'Z b * var(y))`, which holds because every
/// column of `Z` is z-scored and therefore sums to zero. The positivity of
/// `b'Z'Z b` also stands in for upstream's guard against correlating against a
/// constant vector.
///
/// ### Params
///
/// * `gram` - The reduced gene block
/// * `coef` - Accumulated coefficients
/// * `threshold` - Correlation at which to stop
///
/// ### Returns
///
/// `true` once the fit is close enough.
fn tracks_well(gram: &GeneGram, coef: &[f64], threshold: f64) -> bool {
    let mut quad = 0.0;
    for (i, ci) in coef.iter().enumerate() {
        if *ci == 0.0 {
            continue;
        }
        for (j, cj) in coef.iter().enumerate() {
            if *cj == 0.0 {
                continue;
            }
            quad += ci * gram.ztz[(i, j)] * cj;
        }
    }
    if !(quad.is_finite() && quad > 0.0) || gram.y_centred_ss <= 0.0 {
        return false;
    }
    let cross: f64 = coef.iter().zip(gram.zty.iter()).map(|(c, z)| c * z).sum();
    let r = cross / (quad * gram.y_centred_ss).sqrt();
    r.is_finite() && r > threshold
}

/// Projects the fitted coefficients back to per-cell scores.
///
/// ### Params
///
/// * `gram` - The reduced gene block, holding the cell-major non-zeros
/// * `coef` - The fitted coefficients
///
/// ### Returns
///
/// One score per cell, in the cell type's row order.
fn project(gram: &GeneGram, coef: &[f64]) -> Vec<f64> {
    // The offsets contribute the same constant to every cell.
    let base: f64 = coef
        .iter()
        .zip(gram.offset.iter())
        .map(|(c, o)| c * o)
        .sum();
    gram.by_cell
        .par_iter()
        .map(|entries| {
            let mut acc = base;
            for (slot, value) in entries.iter() {
                let g = *slot as usize;
                acc += coef[g] * gram.scale[g] * (*value as f64);
            }
            acc
        })
        .collect()
}

///////////////
// The stage //
///////////////

/// Runs DIALOGUE stage three.
///
/// ### Params
///
/// * `reader` - Gene-major reader
/// * `views` - One resolved view per cell type
/// * `step1` - Stage one output
/// * `step2` - Stage two output
/// * `params` - The full parameter set
/// * `verbose` - `0` silent, non-zero prints progress
///
/// ### Returns
///
/// The finished [DialogueResult], or the first error encountered.
pub(crate) fn run_step3<S: SingleCellReading>(
    reader: &S,
    views: &[CellTypeView],
    step1: &DialogueStep1Result,
    step2: &DialogueStep2Result,
    params: &DialogueParams,
    verbose: usize,
) -> Result<DialogueResult, BixverseErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let n_types = views.len();
    let k = params.pmd.k;
    let refine = &params.refine;

    let mut verdicts: Vec<GeneVerdict> = Vec::new();
    let mut scores: Vec<Mat<f64>> = Vec::with_capacity(n_types);
    let mut permissive: Vec<Vec<ProgrammeSignature>> = Vec::with_capacity(n_types);
    let mut strict: Vec<Vec<ProgrammeSignature>> = Vec::with_capacity(n_types);
    let mut refit_fidelity = Mat::<f64>::zeros(n_types, k);

    for cell_type in 0..n_types {
        // Gather this cell type's genes and their per-partner verdicts.
        let mine: Vec<_> = step2
            .associations
            .iter()
            .filter(|a| a.cell_type == cell_type)
            .collect();
        let mut partners: Vec<usize> = mine.iter().map(|a| a.partner).collect();
        partners.sort_unstable();
        partners.dedup();

        let mut index: FxHashMap<(usize, usize, bool), usize> = FxHashMap::default();
        let mut rows: Vec<MetaRow> = Vec::new();
        for a in mine.iter() {
            let key = (a.programme, a.gene, a.up);
            let slot = *index.entry(key).or_insert_with(|| {
                rows.push(MetaRow {
                    programme: a.programme,
                    gene: a.gene,
                    up: a.up,
                    z: vec![f64::NAN; partners.len()],
                });
                rows.len() - 1
            });
            let col = partners
                .binary_search(&a.partner)
                .expect("partner collected");
            rows[slot].z[col] = a.z;
        }

        let mut cell_verdicts = meta_analyse(&rows, &partners, cell_type, refine.support_p);

        // Refit, one programme at a time.
        let n_in_type = views[cell_type].cells.len();
        let mut refitted = Mat::<f64>::zeros(n_in_type, k);
        let mut permissive_here = vec![ProgrammeSignature::default(); k];
        let mut strict_here = vec![ProgrammeSignature::default(); k];

        for programme in 0..k {
            let slots: Vec<usize> = (0..cell_verdicts.len())
                .filter(|&i| cell_verdicts[i].programme == programme)
                .collect();
            // The refit target is the *unresidualised* projection, matching
            // upstream: the quality covariate is regressed out of the refit's
            // output further down, not out of its input. Using the residualised
            // score here fits a different problem and picks different genes.
            let canonical: Vec<f64> = (0..n_in_type)
                .map(|i| step1.raw_scores[cell_type][(i, programme)])
                .collect();

            if slots.is_empty() {
                // No genes reached a verdict, so the canonical score stands.
                for i in 0..n_in_type {
                    refitted[(i, programme)] = canonical[i];
                }
                continue;
            }

            let genes: Vec<usize> = slots.iter().map(|&i| cell_verdicts[i].gene).collect();
            let flip: Vec<bool> = slots.iter().map(|&i| !cell_verdicts[i].up).collect();
            let support: Vec<f64> = slots
                .iter()
                .map(|&i| cell_verdicts[i].support_fraction)
                .collect();

            let gram = build_gram(reader, &genes, &flip, &views[cell_type], &canonical)?;
            let coef = staged_nnls(&gram, &support, params);
            for (slot, &i) in slots.iter().enumerate() {
                cell_verdicts[i].coefficient = coef[slot];
            }

            let projected = project(&gram, &coef);
            let all_zero = coef.iter().all(|c| *c == 0.0);
            for i in 0..n_in_type {
                refitted[(i, programme)] = if all_zero { canonical[i] } else { projected[i] };
            }

            // The two gene lists. `lb` is half the cell types the programme
            // spans, rounded up, which is upstream's majority rule.
            let spanned = step1.mcp_cell_types[programme].len().max(1);
            let lb = spanned.div_ceil(2);
            for (slot, &i) in slots.iter().enumerate() {
                let v = &cell_verdicts[i];
                let majority = v.n_supporting >= lb
                    && (if v.up { v.p_up } else { v.p_down }) < refine.permissive_p;
                if coef[slot] > 0.0 || majority {
                    let list = if v.up {
                        &mut permissive_here[programme].up
                    } else {
                        &mut permissive_here[programme].down
                    };
                    list.push(v.gene);
                }
                let unanimous = (v.support_fraction - 1.0).abs() < f64::EPSILON
                    && (v.p_up < refine.strict_p || v.p_down < refine.strict_p);
                if unanimous {
                    let list = if v.up {
                        &mut strict_here[programme].up
                    } else {
                        &mut strict_here[programme].down
                    };
                    list.push(v.gene);
                }
            }
        }

        for sig in permissive_here.iter_mut().chain(strict_here.iter_mut()) {
            sig.up.sort_unstable();
            sig.up.dedup();
            sig.down.sort_unstable();
            sig.down.dedup();
        }

        // Same confounder, same treatment as the canonical scores.
        let quality = Mat::<f64>::from_fn(n_in_type, 1, |r, _| views[cell_type].quality[r]);
        let residualised = ols_residualise(refitted.as_ref(), quality.as_ref())?;

        // Upstream's `cca.fit`: the canonical score against the refitted one,
        // both residualised. Comparing the pre-residualisation pair would
        // measure something slightly different.
        for programme in 0..k {
            let before: Vec<f64> = (0..n_in_type)
                .map(|i| step1.scores[cell_type][(i, programme)])
                .collect();
            let after: Vec<f64> = (0..n_in_type)
                .map(|i| residualised[(i, programme)])
                .collect();
            refit_fidelity[(cell_type, programme)] =
                pearson_correlation(&before, &after).unwrap_or(f64::NAN);
        }

        if verbosity.normal_verbosity() {
            let sizes: Vec<String> = permissive_here
                .iter()
                .map(|s| format!("{}/{}", s.up.len(), s.down.len()))
                .collect();
            println!(
                "Cell type {cell_type}: refined signature sizes up/down {}",
                sizes.join(", ")
            );
        }

        scores.push(residualised);
        permissive.push(permissive_here);
        strict.push(strict_here);
        verdicts.extend(cell_verdicts);
    }

    Ok(DialogueResult {
        shared_samples: step1.shared_samples.clone(),
        ws: step1.ws.clone(),
        kept_features: step1.kept_features.clone(),
        emp_p: step1.emp_p.clone(),
        pair_cor: step1.pair_cor.clone(),
        mcp_cell_types: step1.mcp_cell_types.clone(),
        scores,
        cca_scores: step1.scores.clone(),
        refit_fidelity,
        verdicts,
        permissive,
        strict,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_data::in_memory_io::InMemorySparseReader;
    use approx::assert_relative_eq;

    /// A small gene-major store: 12 cells, 5 genes, mixed sparsity.
    fn tiny_store() -> CompressedSparseData2<u32, f32> {
        let n_cells = 12usize;
        let n_genes = 5usize;
        let mut data: Vec<u32> = Vec::new();
        let mut norm: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = vec![0];
        for g in 0..n_genes {
            for c in 0..n_cells {
                // Deterministic, and gene 3 is deliberately sparse.
                let v = if g == 3 && c % 4 != 0 {
                    0.0
                } else {
                    (((c * 7 + g * 13) % 9) as f32) * 0.25
                };
                if v > 0.0 {
                    data.push((v * 4.0) as u32);
                    norm.push(v);
                    indices.push(c as u32);
                }
            }
            indptr.push(indices.len() as u32);
        }
        CompressedSparseData2::new_csc(&data, &indices, &indptr, Some(&norm), (n_cells, n_genes))
    }

    /// Densifies what `build_gram` reduces, so the reduction can be checked
    /// against it directly.
    fn dense_z(
        matrix: &CompressedSparseData2<u32, f32>,
        genes: &[usize],
        flip: &[bool],
        cells: &[usize],
    ) -> Mat<f64> {
        let n = cells.len();
        let mut z = Mat::<f64>::zeros(n, genes.len());
        let norm = matrix.data_2.as_ref().unwrap();
        for (slot, &g) in genes.iter().enumerate() {
            let lo = matrix.indptr[g] as usize;
            let hi = matrix.indptr[g + 1] as usize;
            let mut column = vec![0.0_f64; n];
            for k in lo..hi {
                let cell = matrix.indices[k] as usize;
                if let Some(row) = cells.iter().position(|&c| c == cell) {
                    column[row] = norm[k] as f64;
                }
            }
            let mean = column.iter().sum::<f64>() / n as f64;
            let var = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            let sd = var.sqrt();
            let sd = if sd > 0.0 { sd } else { 1.0 };
            let sign = if flip[slot] { -1.0 } else { 1.0 };
            for row in 0..n {
                z[(row, slot)] = sign * (column[row] - mean) / sd;
            }
        }
        z
    }

    /// The stage-three counterpart of `test_assembled_statistics_match_the_explicit_design`.
    ///
    /// `build_gram` reconstructs `Z'Z` from raw sparse values plus an affine
    /// correction rather than forming the dense z-scored block. That is the
    /// same class of reduction as stage two's, with the same failure mode: if
    /// the four-term expansion or the scale/offset drift, every refit is
    /// quietly wrong and nothing downstream would notice.
    #[test]
    fn test_build_gram_matches_the_dense_reduction() {
        let matrix = tiny_store();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let cells: Vec<usize> = (0..12).collect();
        let sample_ids = vec![0usize; 12];
        let quality = vec![0.0_f64; 12];
        let view = CellTypeView::new(&cells, &sample_ids, &quality).unwrap();

        let genes = vec![0usize, 1, 3, 4];
        let flip = vec![false, true, false, true];
        let response: Vec<f64> = (0..12).map(|i| ((i * 5 % 7) as f64) - 3.0).collect();

        let gram = build_gram(&reader, &genes, &flip, &view, &response).unwrap();
        let z = dense_z(&matrix, &genes, &flip, &cells);

        for a in 0..genes.len() {
            // Every column is z-scored, so it sums to zero. The early-stop
            // identity in `tracks_well` depends on this.
            let column_sum: f64 = (0..12).map(|i| z[(i, a)]).sum();
            assert!(column_sum.abs() < 1e-9, "column {a} sum was {column_sum}");

            let expected_zty: f64 = (0..12).map(|i| z[(i, a)] * response[i]).sum();
            assert_relative_eq!(gram.zty[a], expected_zty, epsilon = 1e-9);

            for b in 0..genes.len() {
                let expected: f64 = (0..12).map(|i| z[(i, a)] * z[(i, b)]).sum();
                assert_relative_eq!(gram.ztz[(a, b)], expected, epsilon = 1e-9);
            }
        }

        let mean = response.iter().sum::<f64>() / 12.0;
        let expected_ss: f64 = response.iter().map(|v| (v - mean).powi(2)).sum();
        assert_relative_eq!(gram.y_centred_ss, expected_ss, max_relative = 1e-12);
    }

    /// `project` must reproduce `Z b` for the same coefficients.
    #[test]
    fn test_project_matches_the_dense_product() {
        let matrix = tiny_store();
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let cells: Vec<usize> = (0..12).collect();
        let view = CellTypeView::new(&cells, &[0usize; 12], &[0.0_f64; 12]).unwrap();

        let genes = vec![0usize, 3, 4];
        let flip = vec![false, true, false];
        let response: Vec<f64> = (0..12).map(|i| i as f64 * 0.3).collect();
        let gram = build_gram(&reader, &genes, &flip, &view, &response).unwrap();
        let z = dense_z(&matrix, &genes, &flip, &cells);

        let coef = vec![0.7, 0.0, 1.4];
        let got = project(&gram, &coef);
        for i in 0..12 {
            let expected: f64 = (0..genes.len()).map(|g| z[(i, g)] * coef[g]).sum();
            assert_relative_eq!(got[i], expected, epsilon = 1e-9);
        }
    }
}
